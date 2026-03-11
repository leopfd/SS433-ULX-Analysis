import os
import glob
import time
import math
import csv
import multiprocess
import signal
import shutil
from queue import Empty
from functools import partial
from tqdm import tqdm
from PIL import Image

import config
import lib.sherpa_fit as sherpa_fit
from lib.arguments import get_pipeline_args

def _fixed_bar_format():
    cols = shutil.get_terminal_size(fallback=(100, 24)).columns
    # Reserve space for description + stats; clamp to a reasonable width.
    width = max(24, min(60, cols - 40))
    return f"{{l_bar}}\033[1m{{bar:{width}}}\033[0m{{r_bar}}"

def compile_pngs_to_pdf(pbar, png_files, pdf_filename):
    if not png_files: return
    if not os.path.exists(png_files[0]):
        print(f"error: cannot find file {png_files[0]} to start pdf.")
        return
    def _open_rgb(path):
        img = Image.open(path)
        if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
        else:
            img = img.convert("RGB")
        return img
    # Prefer streaming PDF assembly if img2pdf is available to reduce memory usage.
    try:
        import img2pdf  # type: ignore
    except Exception:
        img2pdf = None

    existing_files = [f for f in png_files if os.path.exists(f)]
    if not existing_files:
        print("error: no existing png files found to compile.")
        return

    if img2pdf is not None:
        try:
            import warnings
            import contextlib
            import io
            stderr_buf = io.StringIO()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Image contains an alpha channel.*")
                with contextlib.redirect_stderr(stderr_buf):
                    pdf_bytes = img2pdf.convert(existing_files)
            # Surface any non-alpha warnings from stderr
            for line in stderr_buf.getvalue().splitlines():
                if "Image contains an alpha channel" in line:
                    continue
                if line.strip():
                    print(line)
            with open(pdf_filename, "wb") as f:
                f.write(pdf_bytes)
            pbar.update(len(png_files))
            return
        except Exception as e:
            print(f"warning: img2pdf failed ({e}); falling back to PIL.")

    images = []

    # Open the first image to establish the base for the PDF file
    img1 = _open_rgb(existing_files[0])
    pbar.update(1)

    # Iterate through the rest of the file list and append them
    for png_file in existing_files[1:]:
        try:
            images.append(_open_rgb(png_file))
        except Exception:
            print(f"warning: could not open file {png_file}, skipping.")
        pbar.update(1)

    # Save the accumulated images as a single PDF document
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Image contains an alpha channel.*")
        img1.save(pdf_filename, "PDF", resolution=400.0, save_all=True, append_images=images)

def run_pipeline():
    # Disable the safety limit for image size to handle large scientific plots
    Image.MAX_IMAGE_PIXELS = None
    
    # Retrieve configuration variables for the fitting process
    multi_n_components = config.NUM_COMPS
    n_components_by_obs = dict(getattr(config, "NUM_COMPS_BY_OBS", {}) or {})
    run_mcmc = config.RUN_MCMC
    recalculate_chains = config.RECALC_CHAINS
    mcmc_iterations = config.MCMC_ITER

    # Worker computes walkers from the chosen per-observation component count.
    mcmc_n_walkers = None
    mcmc_ball_size = config.MCMC_BALL

    os.chdir(config.BASE_DIR)
    
    # Locate all spline-corrected event files in the directory structure
    event_files = sorted(glob.glob('*/repro/*splinecorr.fits'))[:]
    
    # Filter the file list if a specific observation set was requested
    if config.OBS_SELECTION:
        allowed_obs = set()
        for part in config.OBS_SELECTION.split(','):
            if '-' in part:
                start, end = part.split('-')
                allowed_obs.update(str(i) for i in range(int(start), int(end) + 1))
            else:
                allowed_obs.add(part.strip())
        
        event_files = [f for f in event_files if f.split(os.sep)[0] in allowed_obs]
        
        if not event_files:
            print(f"warning: no files matched observation selection: {config.OBS_SELECTION}")
            return False

    if n_components_by_obs:
        available_obs = {f.split(os.sep)[0] for f in event_files}
        ignored = sorted(
            [obs for obs in n_components_by_obs.keys() if obs not in available_obs],
            key=lambda s: int(s),
        )
        if ignored:
            print(
                "warning: --comps-per-obs contains observations with no matching files: "
                + ", ".join(ignored)
            )

    def _chain_file_for_obs(obsid):
        n_components = int(n_components_by_obs.get(obsid, multi_n_components))
        walkers = 4 * (n_components * 3 + 2)
        ball_str = str(mcmc_ball_size).replace('.', 'p')
        base_name = (
            f"mcmc-chain-{n_components}comp-"
            f"{walkers}walkers-"
            f"{mcmc_iterations}steps-"
            f"{ball_str}ball"
        )

        folder_parts = [base_name]
        signifiers = list(getattr(config, "CHAIN_SIGNIFIERS", []) or [])
        if signifiers:
            step_str_simple = str(mcmc_iterations)
            step_str_k = (
                f"{int(mcmc_iterations/1000)}k"
                if mcmc_iterations >= 1000 and mcmc_iterations % 1000 == 0
                else ""
            )
            for s in signifiers:
                if s == "mcmc":
                    continue
                if s == step_str_simple:
                    continue
                if s == step_str_k:
                    continue
                folder_parts.append(s)

        chain_bin_size = 0.25
        folder_parts.append(f"bin{str(chain_bin_size).replace('.', 'p')}")
        chain_dir = os.path.join(config.DIR_CHAINS, "-".join(folder_parts))
        return os.path.join(chain_dir, f"{obsid}_chain.h5")

    allow_chain_changes = False
    if run_mcmc and event_files:
        obs_ids_in_run = sorted({f.split(os.sep)[0] for f in event_files}, key=lambda s: int(s))
        existing_chain_files = {}
        for obs in obs_ids_in_run:
            chain_path = _chain_file_for_obs(obs)
            if os.path.exists(chain_path):
                existing_chain_files[obs] = chain_path

        def _expected_geometry(obsid):
            n_components = int(n_components_by_obs.get(obsid, multi_n_components))
            expected_ndim = n_components * 3 + 2
            walkers_requested = (
                4 * expected_ndim if mcmc_n_walkers is None else int(mcmc_n_walkers)
            )
            expected_nwalkers = (
                walkers_requested
                if walkers_requested >= 2 * expected_ndim
                else 2 * expected_ndim + 2
            )
            return expected_nwalkers, expected_ndim

        def _chain_needs_destructive_change(obsid, chain_path, backend_cls):
            if recalculate_chains:
                return True, "--recalc requested"

            expected_nwalkers, expected_ndim = _expected_geometry(obsid)

            try:
                backend = backend_cls(chain_path, read_only=True)
            except Exception as e:
                return True, f"cannot open chain ({e})"

            try:
                current_steps = int(backend.iteration)
            except Exception as e:
                return True, f"cannot read iteration ({e})"

            if current_steps <= 0:
                return True, "existing zero-step chain file"

            try:
                b_nwalkers, b_ndim = backend.shape
            except Exception as e:
                return True, f"shape check failed ({e})"

            if (b_nwalkers, b_ndim) != (expected_nwalkers, expected_ndim):
                return (
                    True,
                    f"shape mismatch ({b_nwalkers},{b_ndim}) != ({expected_nwalkers},{expected_ndim})",
                )

            return False, ""

        destructive_candidates = []
        if existing_chain_files:
            try:
                from emcee.backends import HDFBackend  # type: ignore
            except Exception as e:
                for obsid, chain_path in existing_chain_files.items():
                    destructive_candidates.append((obsid, chain_path, f"cannot inspect chain ({e})"))
            else:
                for obsid, chain_path in existing_chain_files.items():
                    needs_change, reason = _chain_needs_destructive_change(obsid, chain_path, HDFBackend)
                    if needs_change:
                        destructive_candidates.append((obsid, chain_path, reason))

        if destructive_candidates:
            print(
                f"found {len(destructive_candidates)} chain(s) that likely require reset/recalc before use:"
            )
            preview = destructive_candidates[:6]
            for obsid, _, reason in preview:
                print(f"  {obsid}: {reason}")
            if len(destructive_candidates) > len(preview):
                print("  ...")
            reply = input("allow destructive chain changes for this run? [y/n]: ").strip().lower()
            allow_chain_changes = reply in ("y", "yes")
            if not allow_chain_changes:
                print("destructive chain changes were not approved; those observations will fail if reset is required.")
    
    pdf_out_filename = config.FIT_PLOT_PDF
    multi_pdf_out_filename = config.MULTI_FIT_PDF
    
    results_filename = config.FULL_LOG_TXT
    multi_results_filename = config.MULTI_LOG_TXT
    
    # Determine progress bar ticks per file to synchronize with MCMC steps
    progress_step = None
    if run_mcmc:
        target_updates = max(1, int(config.MCMC_PROGRESS_TARGET_UPDATES))
        scaled_step = math.ceil(mcmc_iterations / target_updates)
        progress_step = max(1, min(int(config.MCMC_PROGRESS_STEP), int(scaled_step)))
        ticks_per_file = math.ceil(mcmc_iterations / progress_step)
    else:
        ticks_per_file = 0
    total_steps = len(event_files) * ticks_per_file

    # Use spawn context for compatibility across different OS multiprocessing implementations
    ctx = multiprocess.get_context('spawn')

    def _init_worker():
        # Ignore SIGINT in workers; parent handles Ctrl-C and signals stop_event.
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # Create a manager queue to handle progress updates from child processes (spawn-safe)
    manager = ctx.Manager()
    progress_queue = manager.Queue()
    stop_event = manager.Event()
    interrupt_requested = False

    def _handle_sigint(signum, frame):
        nonlocal interrupt_requested
        interrupt_requested = True
        stop_event.set()
    
    # Freeze constant arguments into a partial function to pass to the worker pool
    worker_func = partial(sherpa_fit.process_observation, 
                          progress_queue=progress_queue,
                          stop_event=stop_event,
                          obsid_coords=config.OBSID_COORDS, 
                          mcmc_scale_factors={}, 
                          emp_psf_file=config.EMP_PSF_FILE,
                          n_components_multi=multi_n_components,
                          n_components_by_obs=n_components_by_obs,
                          run_mcmc_multi=run_mcmc,
                          mcmc_iter_multi=mcmc_iterations,
                          mcmc_n_walkers=mcmc_n_walkers,  
                          mcmc_ball_size=mcmc_ball_size,
                          allow_chain_changes=allow_chain_changes,
                          auto_stop=config.AUTO_STOP,
                          sigma_val=config.SIGMA_VAL,
                          progress_step=progress_step if run_mcmc else None,
                          recalc=recalculate_chains,
                          chain_base_dir=config.DIR_CHAINS,
                          signifiers=config.CHAIN_SIGNIFIERS,
                          ephemeris=config.EPHEMERIS,
                         )

    if run_mcmc:
        auto_stop_label = "on" if config.AUTO_STOP else "off"
        check_interval = max(
            sherpa_fit.AUTO_STOP_CHECK_INTERVAL,
            int(mcmc_iterations / sherpa_fit.AUTO_STOP_TARGET_CHECKS),
        )
        tau_factor = f"{sherpa_fit.AUTO_STOP_TAU_FACTOR}×τ"
        print(f"Auto-stop is {auto_stop_label}. Chain will stop when steps are greater than {tau_factor}...")

    num_processes = os.cpu_count()
    print(f"starting parallel processing on {num_processes} cores...\n")
    start_total_time = time.time()
    # Execute the worker function across all event files using a process pool
    with tqdm(
        total=total_steps,
        desc="processing observations",
        bar_format=_fixed_bar_format(),
    ) as pbar:
        prev_handler = signal.signal(signal.SIGINT, _handle_sigint)
        pool = ctx.Pool(
            processes=num_processes,
            maxtasksperchild=1,
            initializer=_init_worker,
        )
        try:
            async_result = pool.map_async(worker_func, event_files)
            
            # Poll the worker pool and update the progress bar from the queue until all tasks are done
            def _drain_progress_queue():
                while True:
                    try:
                        msg = progress_queue.get_nowait()
                    except Empty:
                        break
                    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "adjust_total":
                        delta = int(msg[1])
                        if delta > 0:
                            pbar.total = max(pbar.n, pbar.total - delta)
                            pbar.refresh()
                    elif isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "log":
                        pbar.write(msg[1])
                    else:
                        pbar.update(int(msg))

            while not async_result.ready():
                _drain_progress_queue()
                if interrupt_requested:
                    raise KeyboardInterrupt()
                time.sleep(0.1) 
            
            # Ensure any remaining progress updates are processed after the pool finishes
            _drain_progress_queue()
            results = async_result.get()
        except KeyboardInterrupt:
            stop_event.set()
            if pool is not None:
                pool.close()
                grace_seconds = getattr(config, "STOP_GRACE_SECONDS", 10)
                deadline = time.time() + max(0, int(grace_seconds))
                terminated = False
                # Wait briefly for workers to flush/close HDF5 before forcing termination.
                while time.time() < deadline:
                    try:
                        alive = [p for p in pool._pool if p.is_alive()]
                    except Exception:
                        alive = []
                    if not alive:
                        break
                    time.sleep(0.1)
                else:
                    terminated = True
                if terminated:
                    pool.terminate()
                pool.join()
                pool = None
            if 'terminated' in locals() and terminated:
                print("\n\033[1m[pipeline]\033[0m Interrupt received. Workers terminated after grace period; chains may be partial.\n")
            else:
                print("\n\033[1m[pipeline]\033[0m Interrupt received. Workers stopped cleanly; partial chains preserved.\n")
            return False
        finally:
            signal.signal(signal.SIGINT, prev_handler)
            if pool is not None:
                pool.close()
                pool.join()

    print()
    
    end_total_time = time.time()
    print(f"\n--- parallel processing complete in {(end_total_time - start_total_time) / 60.0:.2f} minutes ---\n")

    # Sort results to ensure the output text logs are ordered by Observation ID
    results.sort(key=lambda x: x[0])
    all_pdf_out_files = []
    all_multi_pdf_out_files = []
    srcflux_rows = []

    # Write the consolidated results to text files
    with open(results_filename, 'w') as results_file, open(multi_results_filename, 'w') as multi_results_file:
        for res in results:
            (obsid, header_text, centroid_fit_summary, src_fit_summary,
             srcflux_summary, srcflux_record, multi_fit_summary, multi_results_text,
             pdf_out_files_worker, multi_pdf_out_files_worker) = res
            
            results_file.write(header_text)
            results_file.write("\nCENTROID FIT SUMMARY:\n\n")
            results_file.write(centroid_fit_summary)
            results_file.write("SOURCE FIT SUMMARY:\n\n")
            results_file.write(src_fit_summary)
            results_file.write("SRCFLUX SUMMARY:\n\n")
            results_file.write(srcflux_summary)
            results_file.write("MULTI-COMPONENT FIT SUMMARY:\n\n")
            results_file.write(multi_fit_summary)

            if srcflux_record is not None:
                srcflux_rows.append(srcflux_record)
            
            multi_results_file.write(multi_results_text)
            
            all_pdf_out_files.extend(pdf_out_files_worker)
            all_multi_pdf_out_files.extend(multi_pdf_out_files_worker)

    if srcflux_rows:
        fieldnames = [
            "obs_id",
            "status",
            "flux_table",
            "rate_nominal",
            "rate_minus",
            "rate_plus",
            "flux_nominal",
            "flux_minus",
            "flux_plus",
            "srcreg",
            "bkgreg",
            "psffile",
        ]
        with open(config.SRCFLUX_TABLE_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in srcflux_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Text logs written to:\n  {config.get_rel_path(results_filename)}\n  {config.get_rel_path(multi_results_filename)}")
    if srcflux_rows:
        print(f"  {config.get_rel_path(config.SRCFLUX_TABLE_CSV)}")
    print('\ncompiling pdfs...\n')

    # Compile individual plot images into a single PDF report
    total_plots_to_compile = len(all_pdf_out_files) + len(all_multi_pdf_out_files)
    with tqdm(
        total=total_plots_to_compile,
        desc="compiling pdf plots",
        bar_format=_fixed_bar_format(),
    ) as pbar:
        try:
            compile_pngs_to_pdf(pbar, all_pdf_out_files, pdf_out_filename)
        except Exception as e:
            print(f"\nerror: could not compile {pdf_out_filename}: {e}")

        try:
            compile_pngs_to_pdf(pbar, all_multi_pdf_out_files, multi_pdf_out_filename)
        except Exception as e:
            print(f"\nerror: could not compile {multi_pdf_out_filename}: {e}")

    # Remove temporary PNG files to keep the directory clean
    print("\ncleaning up temporary png files...")
    temp_files_to_clean = glob.glob("2Dfits/temp_*.png")
    for f in temp_files_to_clean:
        try:
            os.remove(f)
        except Exception as e:
            print(f"\nwarning: could not remove {f}: {e}")

    print('\nprocess complete')
    return True

if __name__ == '__main__':
    args = get_pipeline_args()
    config.update_config_from_args(args)
    run_pipeline()
