import os
import glob
import time
import math
import multiprocess
from functools import partial
from tqdm import tqdm
from PIL import Image

import config
import lib.sherpa_fit as sherpa_fit
from lib.arguments import get_pipeline_args

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
            with open(pdf_filename, "wb") as f:
                f.write(img2pdf.convert(existing_files))
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
    run_mcmc = config.RUN_MCMC
    recalculate_chains = config.RECALC_CHAINS
    mcmc_iterations = config.MCMC_ITER
    
    # Calculate number of walkers based on the number of free parameters per component
    mcmc_n_walkers = 4 * (multi_n_components * 3 + 2)       
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
            return
    
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
    
    # Create a manager queue to handle progress updates from child processes (spawn-safe)
    manager = ctx.Manager()
    progress_queue = manager.Queue()
    
    # Freeze constant arguments into a partial function to pass to the worker pool
    worker_func = partial(sherpa_fit.process_observation, 
                          progress_queue=progress_queue,
                          obsid_coords=config.OBSID_COORDS, 
                          mcmc_scale_factors={}, 
                          emp_psf_file=config.EMP_PSF_FILE,
                          n_components_multi=multi_n_components,
                          run_mcmc_multi=run_mcmc,
                          mcmc_iter_multi=mcmc_iterations,
                          mcmc_n_walkers=mcmc_n_walkers,  
                          mcmc_ball_size=mcmc_ball_size,
                          auto_stop=config.AUTO_STOP,
                          sigma_val=config.SIGMA_VAL,
                          progress_step=progress_step if run_mcmc else None,
                          recalc=recalculate_chains,
                          chain_base_dir=config.DIR_CHAINS,
                          signifiers=config.SIGNIFIERS,
                          ephemeris=config.EPHEMERIS
                         )

    if run_mcmc:
        auto_stop_label = "on" if config.AUTO_STOP else "off"
        check_interval = max(
            sherpa_fit.AUTO_STOP_CHECK_INTERVAL,
            int(mcmc_iterations / sherpa_fit.AUTO_STOP_TARGET_CHECKS),
        )
        print(
            f"auto-stop: {auto_stop_label} "
            f"(check every {check_interval} steps; "
            f"stop at > {sherpa_fit.AUTO_STOP_TAU_FACTOR}*tau)"
        )

    num_processes = os.cpu_count()
    print(f"starting parallel processing on {num_processes} cores...\n")
    start_total_time = time.time()
    
    # Execute the worker function across all event files using a process pool
    with tqdm(total=total_steps, desc="processing observations", bar_format="{l_bar}{r_bar}") as pbar:
        with ctx.Pool(processes=num_processes) as pool:
            async_result = pool.map_async(worker_func, event_files)
            
            # Poll the worker pool and update the progress bar from the queue until all tasks are done
            def _drain_progress_queue():
                while not progress_queue.empty():
                    msg = progress_queue.get()
                    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "adjust_total":
                        delta = int(msg[1])
                        if delta > 0:
                            pbar.total = max(pbar.n, pbar.total - delta)
                            pbar.refresh()
                    else:
                        pbar.update(int(msg))

            while not async_result.ready():
                _drain_progress_queue()
                time.sleep(0.1) 
            
            # Ensure any remaining progress updates are processed after the pool finishes
            _drain_progress_queue()
            results = async_result.get()

    print()
    
    end_total_time = time.time()
    print(f"\n--- parallel processing complete in {(end_total_time - start_total_time) / 60.0:.2f} minutes ---\n")

    # Sort results to ensure the output text logs are ordered by Observation ID
    results.sort(key=lambda x: x[0])
    all_pdf_out_files = []
    all_multi_pdf_out_files = []

    # Write the consolidated results to text files
    with open(results_filename, 'w') as results_file, open(multi_results_filename, 'w') as multi_results_file:
        for res in results:
            (obsid, header_text, centroid_fit_summary, src_fit_summary, 
             multi_fit_summary, multi_results_text, 
             pdf_out_files_worker, multi_pdf_out_files_worker) = res
            
            results_file.write(header_text)
            results_file.write("\nCENTROID FIT SUMMARY:\n\n")
            results_file.write(centroid_fit_summary)
            results_file.write("SOURCE FIT SUMMARY:\n\n")
            results_file.write(src_fit_summary)
            results_file.write("MULTI-COMPONENT FIT SUMMARY:\n\n")
            results_file.write(multi_fit_summary)
            
            multi_results_file.write(multi_results_text)
            
            all_pdf_out_files.extend(pdf_out_files_worker)
            all_multi_pdf_out_files.extend(multi_pdf_out_files_worker)

    print(f"Text logs written to:\n  {config.get_rel_path(results_filename)}\n  {config.get_rel_path(multi_results_filename)}")
    print('\ncompiling pdfs...\n')

    # Compile individual plot images into a single PDF report
    total_plots_to_compile = len(all_pdf_out_files) + len(all_multi_pdf_out_files)
    with tqdm(total=total_plots_to_compile, desc="compiling pdf plots", bar_format="{l_bar}{r_bar}") as pbar:
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

if __name__ == '__main__':
    args = get_pipeline_args()
    config.update_config_from_args(args)
    run_pipeline()
