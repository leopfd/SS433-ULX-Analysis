import os
import glob
import time
import multiprocess
from functools import partial
from tqdm import tqdm
from PIL import Image

import config
from lib.sherpa_fit import process_observation
from lib.arguments import get_pipeline_args

def compile_pngs_to_pdf(pbar, png_files, pdf_filename):
    if not png_files: return
    if not os.path.exists(png_files[0]):
        print(f"error: cannot find file {png_files[0]} to start pdf.")
        return
    images = []
    
    # Open the first image to establish the base for the PDF file
    img1 = Image.open(png_files[0]).convert('RGB')
    pbar.update(1) 
    
    # Iterate through the rest of the file list and append them
    for png_file in png_files[1:]:
        if os.path.exists(png_file):
            images.append(Image.open(png_file).convert('RGB'))
        else:
            print(f"warning: missing file {png_file}, skipping.")
        pbar.update(1) 
    
    # Save the accumulated images as a single PDF document
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
    target_update_count = 200
    if run_mcmc:
        worker_interval = max(1, int(mcmc_iterations / target_update_count))
        ticks_per_file = mcmc_iterations // worker_interval
    else:
        ticks_per_file = 0
    total_steps = len(event_files) * ticks_per_file

    # Use spawn context for compatibility across different OS multiprocessing implementations
    ctx = multiprocess.get_context('spawn')
    
    # Create a manager queue to handle progress updates from child processes
    manager = ctx.Manager()
    progress_queue = manager.Queue()
    
    # Freeze constant arguments into a partial function to pass to the worker pool
    worker_func = partial(process_observation, 
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
                          progress_chunks=target_update_count,
                          recalc=recalculate_chains,
                          chain_base_dir=config.DIR_CHAINS,
                          signifiers=config.SIGNIFIERS,
                          ephemeris=config.EPHEMERIS
                         )

    num_processes = os.cpu_count()
    print(f"starting parallel processing on {num_processes} cores...\n")
    start_total_time = time.time()
    
    # Execute the worker function across all event files using a process pool
    with tqdm(total=total_steps, desc="processing observations", bar_format="{l_bar}{r_bar}") as pbar:
        with ctx.Pool(processes=num_processes) as pool:
            async_result = pool.map_async(worker_func, event_files)
            
            # Poll the worker pool and update the progress bar from the queue until all tasks are done
            while not async_result.ready():
                while not progress_queue.empty():
                    pbar.update(progress_queue.get())
                time.sleep(0.1) 
            
            # Ensure any remaining progress updates are processed after the pool finishes
            while not progress_queue.empty():
                pbar.update(progress_queue.get())
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