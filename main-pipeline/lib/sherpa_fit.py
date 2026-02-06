import os
import time
import logging
import numpy as np

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from astropy.io import fits
from scipy.ndimage import rotate, gaussian_filter
import emcee
from emcee.backends import HDFBackend
import corner

from astropy.time import Time
from lib.physics import ss433_phases

from lib.image_utils import data_extract_quickpos_iter, write_pixelscale, compute_split_rhat

AUTO_STOP_CHECK_INTERVAL = 200
AUTO_STOP_TARGET_CHECKS = 1000
AUTO_STOP_MIN_STEPS = 1000
AUTO_STOP_THIN_TARGET = 2000
AUTO_STOP_TAU_FACTOR = 100

from sherpa.astro.ui import (
    load_data, image_close, image_data, load_psf, set_psf, image_psf,
    gauss2d, const2d, set_source, freeze, link, show_model, set_stat,
    set_method, set_method_opt, fit, get_fit_results, thaw, set_sampler,
    set_sampler_opt, covar, get_covar_results, get_draws, get_model_component, 
    get_model_component_image, get_data_image, get_model_image, get_data,
    clean, calc_stat
)
from ciao_contrib.runtool import dmcopy, reproject_image

# Suppress standard Sherpa information messages to keep logs clean
logging.getLogger("sherpa").setLevel(logging.WARNING)

def src_psf_images(obsid, infile, x0, y0, diameter, wcs_ra, wcs_dec, binsize=0.25, shape='square', psfimg=True, showimg=False, empirical_psf=None):
    """
    Extracts a region around the source and optionally prepares an empirical PSF
    It handles rotation and reprojection of the PSF to match the observation roll angle
    """
    # Define the spatial region string based on the requested shape
    if shape.lower() == 'circle':
        region_str = f"circle({x0},{y0},{diameter/2})"
    elif shape.lower() == 'square':
        region_str = f"box({x0},{y0},{diameter},{diameter},0)"
        img_region_str = f"box(256.5,256.5,{diameter*4},{diameter*4},0)"
    else:
        region_str = shape.lower()
        
    # Calculate dimensions in logical pixels to name files consistently
    logical_width = diameter/binsize
    imagefile=f'{obsid}/src_image_{shape}_{int(logical_width)}pixel.fits'
    psf_rotated = f'{obsid}/psf_rotated.fits'
    psf_rotated_cut = f'{obsid}/psf_rotated_cut.fits'
    emp_psf_imagefile = f'{obsid}/psf_image_{shape}_empirical_{int(logical_width)}pixel.fits'
    
    # Reset CIAO tools to default state to avoid parameter pollution
    dmcopy.punlearn()
    dmcopy.clobber = 'yes'
    reproject_image.punlearn()
    reproject_image.clobber = 'yes'

    # Extract the source image from the event file using the defined region and binning
    dmcopy.infile = f'{infile}[sky={region_str}][bin x=::{binsize},y=::{binsize}]'
    dmcopy.outfile = imagefile
    dmcopy()
    load_data(imagefile)

    # Process Empirical PSF if provided
    # The PSF must be rotated to match the roll angle of the specific observation
    if empirical_psf is not None:
        try:
            with fits.open(infile) as hdu_match:
                # Attempt to find the roll angle in primary or secondary headers
                if 'ROLL_NOM' in hdu_match[0].header:
                    roll_nom = hdu_match[0].header['ROLL_NOM']
                elif hdu_match[1].header and 'ROLL_NOM' in hdu_match[1].header:
                    roll_nom = hdu_match[1].header['ROLL_NOM']
                elif hdu_match[1].header and 'ROLL_PNT' in hdu_match[1].header:
                    roll_nom = hdu_match[1].header['ROLL_PNT']
                else:
                    print(f"  error: could not find 'ROLL_NOM' or 'ROLL_PNT' in {infile}")
                    return
        except Exception as e:
            print(f"  error: could not read match file header: {e}")
            return
        
        # Calculate rotation angle relative to the PSF default orientation
        angle_to_rotate = roll_nom - 45.0
        
        try:
            with fits.open(empirical_psf) as hdu_psf:
                if hdu_psf[0].data is None:
                    psf_data = hdu_psf[1].data
                    psf_header = hdu_psf[1].header
                else:
                    psf_data = hdu_psf[0].data
                    psf_header = hdu_psf[0].header
        except Exception as e:
            print(f"  error: could not read psf file data/header: {e}")
            return
        
        # Rotate the PSF image data using cubic interpolation
        rotated_psf_data = rotate(
            psf_data, angle_to_rotate, reshape=False, cval=0.0, order=3
        )
        psf_header.add_history(f"rotated by {angle_to_rotate:.4f} deg")
        hdu_out = fits.PrimaryHDU(data=rotated_psf_data, header=psf_header)
        try:
            hdu_out.writeto(psf_rotated, overwrite=True)
        except Exception as e:
            print(f"  error: could not write output file: {psf_rotated}")
        
        # Update WCS in the rotated PSF to ensure correct alignment during reprojection
        try:
            with fits.open(psf_rotated) as hdu_rot:
                nx = hdu_rot[0].header['NAXIS1']
                ny = hdu_rot[0].header['NAXIS2']
            write_pixelscale(file=psf_rotated, nx=nx, ny=ny, ra=str(wcs_ra), dec=str(wcs_dec))
        except Exception as e:
            print(f"!!! error (obsid {obsid}): wcs stamping failed: {e}")

        # Cutout the center of the rotated PSF and reproject it to match the source image grid
        dmcopy.infile = f'{psf_rotated}[{img_region_str}][bin x=::{binsize*4},y=::{binsize*4}]'
        dmcopy.outfile = psf_rotated_cut
        dmcopy()
        reproject_image.infile = psf_rotated_cut
        reproject_image.matchfile = imagefile
        reproject_image.outfile = emp_psf_imagefile
        reproject_image.method = 'sum'
        reproject_image()
        
        # Load the prepared PSF into Sherpa
        load_psf(f'centr_psf{obsid}', emp_psf_imagefile)
        set_psf(f'centr_psf{obsid}')

    return binsize

def gaussian_image_fit(observation, n_components, position, ampl, fwhm,
                       background=0, pos_min=(0, 0), pos_max=None, exptime=None, lock_fwhm=True,
                       freeze_components=None, use_mcmc=True, mcmc_iter=5000, mcmc_burn_in_frac=0.2,
                       n_walkers=32, ball_size=1e-4, auto_stop=False, sigma_val=1, 
                       prefix="g", confirm=True, imgfit=False, progress_chunks=50, progress_step=None, progress_queue=None,
                       chain_base_dir=None, recalc=False, bin_size=None, signifiers=None, ephemeris=None, date_obs=None):
    """
    Main fitting driver using Sherpa
    Sets up a 2D Gaussian model (single or multi component), fits using optimization algorithms,
    and optionally explores the parameter space using MCMC (emcee)
    """
    
    # Helper functions to expand single value inputs into lists for multiple components
    def process_numeric_param(param, name):
        if isinstance(param, (int, float)): return [param] * n_components
        elif isinstance(param, list):
            if len(param) != n_components: raise ValueError(f"list of {name} must have length {n_components}.")
            return param
        else: raise ValueError(f"{name} must be a number or a list.")

    def process_tuple_param(param, name):
        if isinstance(param, (tuple, list)) and len(param) == 2 and all(isinstance(x, (int, float)) for x in param):
            return [param] * n_components
        elif isinstance(param, list):
            if len(param) != n_components: raise ValueError(f"list of {name} must have length {n_components}.")
            return param
        else: raise ValueError(f"{name} must be a tuple (x, y) or a list.")

    # Validate and process all input parameters into lists matching component count
    positions = process_tuple_param(position, "position")
    ampls = process_numeric_param(ampl, "ampl")
    fwhms = process_numeric_param(fwhm, "fwhm")
    pos_mins = process_tuple_param(pos_min, "pos_min")
    pos_maxs = [None] * n_components if pos_max is None else process_tuple_param(pos_max, "pos_max")

    # Build the Sherpa model expression
    comp_names = []
    gaussian_components = []
    model_components = []
    
    # Create Gaussian components
    for i in range(1, n_components + 1):
        comp_name = f"{prefix}{i}"
        comp_names.append(comp_name)
        comp = gauss2d(comp_name)
        gaussian_components.append(comp)
        model_components.append(comp)
    
    # Create Background component if requested
    bkg_comp = None
    if background > 0:
        bkg_comp = const2d("c1")
        model_components.append(bkg_comp)
    
    if model_components: set_source(sum(model_components))
    else: raise ValueError("model expression is empty.")

    # Initialize component parameters with guesses and bounds
    freeze_list = (freeze_components if isinstance(freeze_components, list) else ([freeze_components] if freeze_components is not None else []))
    for i, comp in enumerate(gaussian_components):
        comp_number = i + 1
        comp.xpos = positions[i][0]; comp.ypos = positions[i][1]
        comp.ampl = ampls[i]; comp.fwhm = fwhms[i]
        
        # Apply strict bounds to prevent components drifting off image
        if hasattr(comp.xpos, 'min'): comp.xpos.min = pos_mins[i][0]
        if hasattr(comp.ypos, 'min'): comp.ypos.min = pos_mins[i][1]
        if pos_maxs[i] is not None:
            if hasattr(comp.xpos, 'max'): comp.xpos.max = pos_maxs[i][0]
            if hasattr(comp.ypos, 'max'): comp.ypos.max = pos_maxs[i][1]
        if hasattr(comp.ampl, 'min'): comp.ampl.min = 0
        if comp_number in freeze_list: freeze(comp)

    # Link FWHM parameters if requested so all components share the same width
    # This is often useful when assuming the PSF width is constant across the field
    central_component = 1
    if lock_fwhm:
        master = gaussian_components[central_component-1].fwhm
        for idx, comp in enumerate(gaussian_components):
            if idx != (central_component-1): link(comp.fwhm, master)

    if bkg_comp is not None:
        bkg_comp.c0 = background
        if hasattr(bkg_comp.c0, 'min'): bkg_comp.c0.min = 0

    if confirm:
        show_model()
        if input(f"  (obsid {observation}) proceed with fit? (y/n): ").lower() != "y": return None, None, None, None, None

    # Perform initial optimization using Sherpa built in methods
    # We use C STAT (Cash statistic) suitable for Poisson data 
    # Two stage optimization: MonCar (Monte Carlo) to find global basin, then Simplex for local refinement
    set_stat('cstat')
    set_method('moncar'); set_method_opt('numcores', 1)
    set_method_opt('population_size', 10 * 16 * (n_components * 3 + 1)); set_method_opt('xprob', 0.5); set_method_opt('weighting_factor', 0.5)
    fit()
    set_method('simplex'); fit()
    fit_results = get_fit_results()

    # Identify parameters to sample in MCMC (thawed parameters only)
    thawed_pars = []
    thawed_par_names = []
    for i, comp in enumerate(gaussian_components):
        comp_number = i + 1
        if comp_number not in freeze_list:
            thaw(comp.ampl); thawed_pars.append(comp.ampl); thawed_par_names.append(comp.ampl.fullname)
            if not (lock_fwhm and comp_number != central_component):
                 thaw(comp.fwhm); thawed_pars.append(comp.fwhm); thawed_par_names.append(comp.fwhm.fullname)
            thaw(comp.xpos, comp.ypos)
            thawed_pars.append(comp.xpos); thawed_par_names.append(comp.xpos.fullname)
            thawed_pars.append(comp.ypos); thawed_par_names.append(comp.ypos.fullname)
    if bkg_comp is not None and not bkg_comp.c0.frozen:
        thawed_pars.append(bkg_comp.c0); thawed_par_names.append(bkg_comp.c0.fullname)

    best_fit_values = [p.val for p in thawed_pars]
    best_fit_stat = fit_results.statval

    # Configure label independent geometry priors for Observation 26575
    # This observation requires specific constraints to handle complex morphology where standard labeling fails
    # Instead of assuming "g1 is East", we analyze the geometric distribution of points
    geom_prior_cfg = None
    obs_str = str(observation)

    if obs_str == "26575" and n_components == 4:
        # Define image center in logical pixels which aligns with model coordinates
        data_img = get_data_image()
        ny, nx = data_img.y.shape
        x_ctr = nx / 2.0 + 0.5
        y_ctr = ny / 2.0 + 0.5

        def idx(fullname):
            return thawed_par_names.index(fullname) if fullname in thawed_par_names else None

        # Retrieve indices for the positional parameters of components g1 through g4
        xy_idx = []
        ok = True
        for i in range(1, 5):
            xi = idx(f"g{i}.xpos")
            yi = idx(f"g{i}.ypos")
            if xi is None or yi is None:
                ok = False
                break
            xy_idx.append((xi, yi))

        if ok:
            geom_prior_cfg = {
                "x_ctr": x_ctr,
                "y_ctr": y_ctr,
                "xy_idx": xy_idx,
                "r_min": 4.0,   # Minimum distance jets must maintain from the core
                "d_min": 5.0,   # Minimum separation required between any two jets
                "x_gap": 2.0,   # Exclusion zone around the core in X to prevent East West crossing
                "dx_min": 4.0,  # Minimum ordering separation along X within a single side
            }

    mcmc_results = None
    walker_map_fig = None
    corner_fig = None
    mcmc_duration_str = ""
    flux_results = None
    
    # Start MCMC Sampling if enabled
    if use_mcmc:
        mcmc_start_time = time.time()
        ndim = len(thawed_pars)
        
        # Define the Log Probability function for MCMC
        # This includes standard parameter bounds and the custom geometric priors
        def log_probability(theta):
            # Check standard Sherpa parameter bounds (hard limits)
            for param, value in zip(thawed_pars, theta):
                if value < param.min or value > param.max:
                    return -np.inf

            # Apply label independent priors for 26575 4 component fit
            # This logic dynamically assigns "Core", "East", and "West" labels based on position
            # allowing the sampler to swap identities if needed to find the global optimum
            if geom_prior_cfg is not None:
                cfg = geom_prior_cfg

                # Extract all coordinate pairs from the current sample theta
                pts = [(theta[xi], theta[yi]) for (xi, yi) in cfg["xy_idx"]]

                # Identify the core as the component closest to the image center
                d2 = [(x - cfg["x_ctr"])**2 + (y - cfg["y_ctr"])**2 for (x, y) in pts]
                icore = int(np.argmin(d2))
                x_core, y_core = pts[icore]

                # Classify remaining components as East or West and enforce separation from the core
                east = []
                west = []
                for j, (x, y) in enumerate(pts):
                    if j == icore:
                        continue

                    # Ensure no jet overlaps the core position
                    if np.hypot(x - x_core, y - y_core) < cfg["r_min"]:
                        return -np.inf

                    # Classify side based on X position relative to core (HRC coordinates)
                    if x < x_core - cfg["x_gap"]:
                        east.append((x, y))
                    elif x > x_core + cfg["x_gap"]:
                        west.append((x, y))
                    else:
                        return -np.inf

                # Enforce component split constraints
                # We expect either 2 East 1 West or 1 East 2 West configuration
                if not ((len(east) == 2 and len(west) == 1) or (len(east) == 1 and len(west) == 2)):
                    return -np.inf

                # Check for non overlap among all jets using pairwise separation
                jets = east + west
                for a in range(len(jets)):
                    xa, ya = jets[a]
                    for b in range(a + 1, len(jets)):
                        xb, yb = jets[b]
                        if np.hypot(xa - xb, ya - yb) < cfg["d_min"]:
                            return -np.inf

                # Enforce spatial ordering within the East side
                # Inner component is closer to core so it has a larger X because East is left
                east_sorted_x = sorted([x for (x, _) in east], reverse=True)
                for k in range(len(east_sorted_x) - 1):
                    if (east_sorted_x[k] - east_sorted_x[k + 1]) < cfg["dx_min"]:
                        return -np.inf

                # Enforce spatial ordering within the West side
                # Inner component is closer to core so it has a smaller X
                west_sorted_x = sorted([x for (x, _) in west])
                for k in range(len(west_sorted_x) - 1):
                    if (west_sorted_x[k + 1] - west_sorted_x[k]) < cfg["dx_min"]:
                        return -np.inf

            # If all checks pass set Sherpa parameters and compute the C statistic
            # Return -0.5 * CSTAT as the log likelihood
            for param, value in zip(thawed_pars, theta):
                param.val = value
            return -0.5 * calc_stat()

        current_n_walkers = n_walkers if n_walkers >= 2 * ndim else 2 * ndim + 2
        
        # Prepare backend storage for the MCMC chain
        ball_str = str(ball_size).replace('.', 'p')
        base_name = (f"mcmc-chain-{n_components}comp-"
                     f"{current_n_walkers}walkers-"
                     f"{mcmc_iter}steps-"
                     f"{ball_str}ball")
        
        folder_parts = [base_name]

        # Construct unique folder name based on configuration signifiers
        if signifiers:
            step_str_simple = str(mcmc_iter)
            step_str_k = f"{int(mcmc_iter/1000)}k" if mcmc_iter >= 1000 and mcmc_iter % 1000 == 0 else ""
            
            for s in signifiers:
                if s == 'mcmc': continue
                if s == step_str_simple: continue
                if s == step_str_k: continue
                folder_parts.append(s)

        if bin_size is not None:
            bin_str = str(bin_size).replace('.', 'p')
            folder_parts.append(f"bin{bin_str}")

        param_folder_name = "-".join(folder_parts)
        
        if chain_base_dir is None:
            chain_base_dir = os.path.join(os.getcwd(), "2Dfits", "emcee_chains")
                
        chain_dir = os.path.join(chain_base_dir, param_folder_name)
        os.makedirs(chain_dir, exist_ok=True)
        chain_filename = os.path.join(chain_dir, f"{observation}_chain.h5")
        
        backend = HDFBackend(chain_filename, compression="gzip", compression_opts=4)

        # Check for existing chain to resume
        current_steps = 0
        try:
            current_steps = backend.iteration
        except:
            pass

        run_sampler = True
        p0 = None
        check_interval = max(AUTO_STOP_CHECK_INTERVAL, int(mcmc_iter / AUTO_STOP_TARGET_CHECKS))
        min_steps_before_check = max(AUTO_STOP_MIN_STEPS, check_interval)
        
        if not recalc and current_steps > 0:
            if current_steps >= mcmc_iter:
                print(f"[{observation}] Found complete chain ({current_steps} steps). Skipping fit.")
                run_sampler = False
            else:
                print(f"[{observation}] Found partial chain ({current_steps}/{mcmc_iter} steps). Resuming...")
                p0 = None # Resumes automatically from backend state
        
        elif recalc and current_steps > 0:
            backend.reset(current_n_walkers, ndim)
        
        else:
            backend.reset(current_n_walkers, ndim)

        # Generate initial walker positions ball around the best fit found by Simplex
        if run_sampler and backend.iteration == 0:
            best_fit_pos = np.array(best_fit_values)
            p0 = best_fit_pos + ball_size * np.random.randn(current_n_walkers, ndim)
            for i, param in enumerate(thawed_pars):
                p0[:, i] = np.clip(p0[:, i], param.min + 1e-6, param.max - 1e-6)

        try:
            sampler = emcee.EnsembleSampler(current_n_walkers, ndim, log_probability, backend=backend)
            
            if run_sampler:
                try:
                    current_step = backend.iteration
                    remaining_steps = mcmc_iter - current_step
                    
                    if remaining_steps > 0:
                        if progress_step is not None:
                            update_interval = max(1, int(progress_step))
                        else:
                            update_interval = max(1, int(mcmc_iter / progress_chunks))
                        if remaining_steps < update_interval:
                            update_interval = remaining_steps
                        # Iterate step by step to allow interruption and progress reporting
                        for i, sample in enumerate(sampler.sample(p0, iterations=remaining_steps, progress=False)):
                            
                            # Update global progress bar via queue
                            if progress_queue and (i + 1) % update_interval == 0: 
                                progress_queue.put(1)
                            
                            # Auto stop logic based on autocorrelation time
                            if auto_stop and sampler.iteration >= min_steps_before_check and (sampler.iteration % check_interval == 0):
                                try:
                                    # Set tol to 0 to prevent crash on short chains
                                    thin = max(1, int(sampler.iteration / AUTO_STOP_THIN_TARGET))
                                    try:
                                        tau = sampler.get_autocorr_time(tol=0, thin=thin)
                                    except TypeError:
                                        tau = sampler.get_autocorr_time(tol=0)
                                    
                                    if np.all(np.isfinite(tau)):
                                        limit = AUTO_STOP_TAU_FACTOR * np.max(tau)
                                        
                                        if sampler.iteration > limit:
                                            print(f"\n[{observation}] Converged at step {sampler.iteration} (tau={np.max(tau):.1f}). Stopping early.")
                                            break
                                except Exception:
                                    pass
                        
                except Exception as e:
                    print(f"  error (obsid {observation}) sampler crashed: {e}")
            
            # Post Processing: Calculate burn in and statistics
            try:
                # Calculate Tau on the full chain first
                tau_est = sampler.get_autocorr_time(tol=0)
                max_tau = np.max(tau_est)
                
                if np.isfinite(max_tau) and max_tau > 0:
                    # Use 2 times tau as the burn in period
                    discard = int(2.0 * max_tau)
                else:
                    raise ValueError("Tau infinite or invalid")

            except Exception:
                # Fallback to using the fixed fraction default 0.2
                discard = int(mcmc_iter * mcmc_burn_in_frac)

            # Safety check to ensure we don't discard the entire chain
            if discard >= sampler.iteration:
                discard = int(sampler.iteration * 0.5)

            # Extract flattened chains for parameter estimation
            flat_samples = sampler.get_chain(discard=discard, flat=True)
            raw_chain = sampler.get_chain(discard=discard)
            
            # Compute convergence diagnostics
            try:
                tau = sampler.get_autocorr_time(tol=0) 
                tau_max = np.max(tau)
                ess = (raw_chain.shape[0] * raw_chain.shape[1]) / tau_max
            except Exception:
                tau = [np.nan] * ndim
                tau_max = np.nan
                ess = 0

            try:
                rhat_vals = compute_split_rhat(raw_chain)
                rhat_max = np.max(rhat_vals)
            except Exception:
                rhat_vals = [np.nan] * ndim
                rhat_max = np.nan

            auto_stop_label = "on" if auto_stop else "off"
            conv_str = (
                f"convergence stats:\n"
                f"  max autocorr time (tau): {tau_max:.1f} steps\n"
                f"  max split-rhat:          {rhat_max:.4f} (goal < 1.1)\n"
                f"  effective samples (ess): {int(ess)}\n"
                f"  chain length / tau:      {raw_chain.shape[0] / tau_max:.1f} (goal > 50)\n"
                f"  auto-stop:               {auto_stop_label} (check every {check_interval} steps; stop at > {AUTO_STOP_TAU_FACTOR}*tau)\n\n"
            )

            # Determine quantiles for error reporting based on requested sigma
            if sigma_val == 1:
                q_low, q_mid, q_high = 15.865, 50.0, 84.135
            elif sigma_val == 2:
                q_low, q_mid, q_high = 2.275, 50.0, 97.725
            elif sigma_val == 3:
                q_low, q_mid, q_high = 0.135, 50.0, 99.865
            else:
                q_low, q_mid, q_high = 15.865, 50.0, 84.135

            # Calculate parameter statistics
            mcmc_results_data = {'parnames': [], 'parvals': [], 'parmins': [], 'parmaxes': []}
            for i, name in enumerate(thawed_par_names):
                mcmc_vals = flat_samples[:, i]
                p_low, p_mid, p_high = np.percentile(mcmc_vals, [q_low, q_mid, q_high])
                mcmc_results_data['parnames'].append(name)
                mcmc_results_data['parvals'].append(p_mid)
                mcmc_results_data['parmins'].append(p_low)
                mcmc_results_data['parmaxes'].append(p_high)
            mcmc_results = mcmc_results_data

            # Check if MCMC found a better likelihood than the initial fit
            log_probs = sampler.get_log_prob(discard=discard, flat=True)
            max_idx = np.argmax(log_probs)
            best_mcmc_stat = -2.0 * log_probs[max_idx]
            
            if best_mcmc_stat < best_fit_stat:
                best_fit_values = list(flat_samples[max_idx])
                best_fit_stat = best_mcmc_stat

            # Calculate flux/rate statistics derived from Amplitude and FWHM
            flux_results = {}
            if exptime is not None:
                fwhm_master_name = gaussian_components[central_component - 1].fwhm.fullname
                if fwhm_master_name in thawed_par_names:
                    f_idx_master = thawed_par_names.index(fwhm_master_name)
                    F_chain = flat_samples[:, f_idx_master]
                    for comp in gaussian_components:
                        amp_name = comp.ampl.fullname
                        if amp_name in thawed_par_names:
                            a_idx = thawed_par_names.index(amp_name)
                            A_chain = flat_samples[:, a_idx]
                            # Flux is proportional to Amplitude * FWHM^2
                            flux_chain = A_chain * (F_chain**2)
                            F_low, F_mid, F_high = np.percentile(flux_chain, [q_low, q_mid, q_high])
                            flux_results[comp.name] = (F_low, F_mid, F_high)

            # Update Sherpa model parameters to the best values found
            for param, val in zip(thawed_pars, best_fit_values): param.val = val

            # Generate Walker Density Map Plot
            walker_map_fig, ax = plt.subplots(1, 1, figsize=(19, 19))
            data_img = get_data_image(); data_vals = data_img.y
            ny, nx = data_vals.shape
            plot_extent = [0.5, nx + 0.5, 0.5, ny + 0.5]
            min_pos = np.min(data_vals[data_vals > 0]) if np.any(data_vals > 0) else 1e-9
            display_floor = min_pos / 10.0
            data_masked = np.maximum(data_vals, display_floor)
            log_norm = mcolors.LogNorm(vmin=display_floor, vmax=np.max(data_vals))
            
            im_data = ax.imshow(data_masked, origin='lower', cmap='gray_r', norm=log_norm, 
                                interpolation='nearest', extent=plot_extent)
            
            colors = ['cyan', 'lime', 'magenta', 'orange', 'yellow']
            dark_colors = ['navy', 'darkgreen', 'indigo', 'xkcd:burgundy', 'xkcd:shit']
            
            # Overlay component position distributions
            best_label_used = False
            median_label_used = False
            for i, comp_name in enumerate(comp_names):
                x_name = f"{comp_name}.xpos"; y_name = f"{comp_name}.ypos"
                if x_name in thawed_par_names and y_name in thawed_par_names:
                    x_idx = thawed_par_names.index(x_name)
                    y_idx = thawed_par_names.index(y_name)
                    x_pts = raw_chain[:, :, x_idx].flatten()
                    y_pts = raw_chain[:, :, y_idx].flatten()
                    
                    # Create 2D histogram of walker positions
                    H, xedges, yedges = np.histogram2d(
                        y_pts, x_pts, bins=[ny, nx], 
                        range=[[0.5, ny + 0.5], [0.5, nx + 0.5]]
                    )
                    if np.sum(H) > 0:
                        comp_color = colors[i % len(colors)]
                        dark_c = dark_colors[i % len(dark_colors)]
                        base_rgb = mcolors.to_rgb(comp_color)
                        smooth_H = gaussian_filter(H, sigma=1.0)
                        peak = smooth_H.max()
                        levels = [peak * 0.1, peak * 0.3, peak * 0.5, peak * 0.7, peak * 0.9]
                        fill_colors = [(*base_rgb, 0.1), (*base_rgb, 0.3), (*base_rgb, 0.5), (*base_rgb, 0.7), (*base_rgb, 0.9)]
                        ax.contourf(smooth_H, levels=levels, colors=fill_colors, extend='max', extent=plot_extent)
                        ax.contour(smooth_H, levels=levels, colors=[comp_color], linewidths=1.0, alpha=0.9, extent=plot_extent)

                        # Optional Overlay of Kinematic Model for context
                        if ephemeris is not None and date_obs is not None:
                            try:
                                # Identify the Core (Closest to Center)
                                img_cy, img_cx = data_vals.shape
                                center_x_pix = img_cx / 2.0 + 0.5
                                center_y_pix = img_cy / 2.0 + 0.5
                                
                                min_dist = 1e9
                                core_x, core_y = center_x_pix, center_y_pix
                                
                                for comp in gaussian_components:
                                    # Use current best fit values
                                    cx = comp.xpos.val
                                    cy = comp.ypos.val
                                    dist = np.hypot(cx - center_x_pix, cy - center_y_pix)
                                    if dist < min_dist:
                                        min_dist = dist
                                        core_x, core_y = cx, cy

                                # Calculate Jet Trajectory
                                try:
                                    # Try standard string format YYYY MM DD
                                    t_obs = Time(date_obs, format='isot').mjd
                                except (TypeError, ValueError):
                                    # If that fails assume it is already a float MJD
                                    t_obs = Time(date_obs, format='mjd').mjd
                                
                                # Create a lookback time array
                                days_back = np.linspace(0, 300, 100)
                                jd_ej = (t_obs + 2400000.5) - days_back
                                
                                # Get RA and Dec offsets in radians
                                mu_b_ra, mu_b_dec, mu_r_ra, mu_r_dec, _, _ = ss433_phases(jd_ej, ephemeris)
                                
                                # Convert to Pixels
                                D_pc = 5500.0
                                rad_to_arcsec = 206265.0
                                hrc_pix_scale = 0.13175 * (bin_size if bin_size else 1.0)
                                
                                c_pc_day = (299792.458 * 86400) / (3.08567758 * 10**13)
                                factor = (c_pc_day * days_back / D_pc) * rad_to_arcsec
                                
                                # Blue Jet
                                x_off_b = -1 * (mu_b_ra * factor) / hrc_pix_scale
                                y_off_b = (mu_b_dec * factor) / hrc_pix_scale
                                
                                # Red Jet
                                x_off_r = -1 * (mu_r_ra * factor) / hrc_pix_scale
                                y_off_r = (mu_r_dec * factor) / hrc_pix_scale

                                # Plot
                                jet_color_b = "#2a9df4"
                                jet_color_r = "#d62728"
                                ax.plot(core_x + x_off_b, core_y + y_off_b, '--', color=jet_color_b, linewidth=3.2, alpha=0.9, label='Jet Model (Approaching)')
                                ax.plot(core_x + x_off_r, core_y + y_off_r, '--', color=jet_color_r, linewidth=3.2, alpha=0.9, label='Jet Model (Receding)')
                                ax.scatter([core_x], [core_y], marker='+', color='white', s=100, zorder=30, label=None)

                            except Exception as e:
                                print(f"Warning: Could not overlay jet model: {e}")

                        bf_x = best_fit_values[x_idx]; bf_y = best_fit_values[y_idx]
                        bf_label = "Best Fit" if not best_label_used else "_nolegend_"
                        ax.scatter(bf_x, bf_y, marker='o', color=dark_c, s=160, zorder=20, edgecolors='white', label=bf_label)
                        best_label_used = True
                        if mcmc_results is not None:
                            med_x = mcmc_results['parvals'][x_idx]
                            med_y = mcmc_results['parvals'][y_idx]
                            med_label = "Median" if not median_label_used else "_nolegend_"
                            ax.scatter(med_x, med_y, marker='x', color=dark_c, s=320, linewidth=3, zorder=19, label=med_label)
                            median_label_used = True

            ax.set_title(f"Walker Density Map - Obsid {observation}", fontsize=32, pad=22)
            ax.set_xlabel("X Pixel", fontsize=28); ax.set_ylabel("Y Pixel", fontsize=28)
            ax.tick_params(labelsize=26)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend = ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=26)
            legend.get_frame().set_alpha(0.9)
            cbar = walker_map_fig.colorbar(im_data, ax=ax, shrink=0.8, pad=0.03)
            cbar.set_label("Counts", fontsize=28, labelpad=-2)
            cbar.ax.tick_params(labelsize=26)
            walker_map_fig.tight_layout()

            # Generate Corner Plot
            total_samples = flat_samples.shape[0]
            threshold = 1000000 
            if total_samples > threshold:
                stride = int(total_samples / threshold)
                plot_samples = flat_samples[::stride]
                title_suffix = f"(downsampled {stride}x)"
            else:
                plot_samples = flat_samples
                title_suffix = "(full chain)"

            corner_fig = corner.corner(
                plot_samples, labels=thawed_par_names, quantiles=[0.16, 0.5, 0.84],
                show_titles=True, title_fmt=".3f", truths=best_fit_values, truth_color='red', quiet=True
            )
            corner_fig.suptitle(f"corner plot {title_suffix} - obsid {observation}", y=1.02)
            mcmc_end_time = time.time()
            mcmc_duration_min = (mcmc_end_time - mcmc_start_time) / 60.0
            mcmc_duration_str = (f"emcee execution time = {mcmc_duration_min:.2f} minutes\n"
                                 f"mean acceptance fraction = {np.mean(sampler.acceptance_fraction):.3f}\n"
                                 f"{conv_str}")

        except Exception as e:
            mcmc_results = None
            mcmc_duration_str = f"emcee failed: {e}\n\n"

    # Compile Final Text Summary
    fit_summary = (
        f"method = {fit_results.methodname}\nstatistic = {fit_results.statname}\n"
        f"final c-stat = {best_fit_stat:.2f} (simplex+mcmc)\n" 
        f"reduced statistic = {fit_results.rstat:.5f}\n\n"
    )
    
    def fmt_val(val): return "------" if val is None else f"{val:>10.3f}"
        
    if mcmc_results is not None:
        param_table = [
            f"emcee results (red line = best fit, black = median):",
            f"{'param':<12} {'best fit':>10} {'median':>10} {'-error':>10} {'+error':>10}",
            f"{'-'*5:<12} {'-'*8:>10} {'-'*8:>10} {'-'*8:>10} {'-'*8:>10}"
        ]
        for name, median, low, high, best in zip(mcmc_results['parnames'], 
                                                 mcmc_results['parvals'], 
                                                 mcmc_results['parmins'], 
                                                 mcmc_results['parmaxes'],
                                                 best_fit_values):
            err_minus = median - low
            err_plus = high - median
            display_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
            param_table.append(f"{display_name:<12} {fmt_val(best)} {fmt_val(median)} {fmt_val(err_minus)} {fmt_val(err_plus)}")
        param_table = "\n".join(param_table)
    else:
        param_table = "best-fit values (no mcmc):\n" + "\n".join([f"{n:<12} {fmt_val(v)}" for n, v in zip(fit_results.parnames, fit_results.parvals)])
            
    summary_output = fit_summary + mcmc_duration_str + param_table + '\n'

    if exptime and use_mcmc and mcmc_results is not None:
        rate_block_rows = ["component count rates (counts/s):"]
        for comp in gaussian_components:
            short = comp.name.split('.')[-1]
            if flux_results is not None and comp.name in flux_results:
                F_low, F_mid, F_high = flux_results[comp.name]
                rate_mid = F_mid / exptime
                rate_minus = (F_mid - F_low) / exptime
                rate_plus = (F_high - F_mid) / exptime
                rate_block_rows.append(f"  {short:<6}: {rate_mid:7.4f}  (-{rate_minus:6.4f}/+{rate_plus:6.4f})")
            else:
                comp_img = get_model_component_image(comp.name)
                total_cts = comp_img.y.sum()
                rate = total_cts / exptime
                rate_block_rows.append(f"  {short:<6}: {rate:7.4f}  (no mcmc rate errors)")
        summary_output += "\n" + "\n".join(rate_block_rows) + "\n"
    else:
        summary_output = fit_summary + param_table + '\n\n\n\n'

    # Create Diagnostic Figure (Data, Model, Residuals)
    fig, axs = plt.subplots(1, 3, figsize=(30, 15))
    data_img = get_data_image(); data_vals = data_img.y
    min_pos = np.min(data_vals[data_vals > 0]) if np.any(data_vals > 0) else 1e-9
    display_floor = min_pos / 10.0
    data_masked = np.maximum(data_vals, display_floor)
    model_img = get_model_image(); model_vals = model_img.y
    model_masked = np.maximum(model_vals, display_floor)
    
    # Calculate Poisson Deviance for residuals
    D = 2.0 * (data_masked * np.log(data_masked / model_masked) - (data_masked - model_masked))
    resid_dev = np.sign(data_vals - model_vals) * np.sqrt(np.abs(D))
    log_norm = mcolors.LogNorm(vmin=display_floor, vmax=np.max(data_vals))
    
    ax = axs[0]
    im = ax.imshow(data_masked, origin='lower', cmap='gnuplot2', norm=log_norm, interpolation='nearest')
    base_colors = ['white', 'cyan', 'lime', 'xkcd:periwinkle']
    legend_elements = []
    for i, comp_name in enumerate(comp_names):
        comp_vals = get_model_component_image(comp_name).y
        if not np.any(comp_vals > 0): continue
        if comp_name == 'g1':
            color = 'white'
        elif comp_name == 'g4':
            color = 'xkcd:light lavender'
        else:
            color = base_colors[i % len(base_colors)]
        peak_val = float(np.nanmax(comp_vals))
        if peak_val <= 0:
            continue
        # Contours at fixed fractions of the peak: 95%, 80%, 60%, 20%
        contour_levels = peak_val * np.array([0.95, 0.80, 0.60, 0.20], dtype=float)
        contour_levels = np.unique(np.clip(contour_levels, np.finfo(float).eps, None))
        ax.contour(
            comp_vals,
            levels=contour_levels,
            colors=[color],
            linestyles=['--'],
            linewidths=2.0,
            alpha=0.9,
        )
        legend_elements.append(Line2D([0], [0], lw=2, linestyle='--', color=color, label=f"{comp_name}"))
    if legend_elements: ax.legend(handles=legend_elements, loc='upper right', fontsize=20)
    ax.set_title(f"{observation} Data and Best Fit Overlay", fontsize=26, pad=16)
    ax.set_xlabel("X Pixel", fontsize=24)
    ax.set_ylabel("Y Pixel", fontsize=24)
    cbar = fig.colorbar(im, ax=ax, label="Counts", shrink=0.53, pad=0.006)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Counts", fontsize=22, labelpad=-4)

    ax = axs[1]
    im = ax.imshow(model_masked, origin='lower', cmap='gnuplot2', norm=log_norm, interpolation='nearest')
    ax.set_title("Best Fit Model", fontsize=26, pad=16)
    ax.set_xlabel("X Pixel", fontsize=24)
    ax.set_ylabel("Y Pixel", fontsize=24)
    cbar = fig.colorbar(im, ax=ax, label="Model Counts", shrink=0.53, pad=0.006)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Model Counts", fontsize=22, labelpad=-4)

    ax = axs[2]
    im = ax.imshow(np.abs(resid_dev), origin='lower', cmap='gnuplot2', norm=mcolors.Normalize(vmin=0, vmax=5), interpolation='nearest')
    ax.set_title("Poisson Deviance Residuals", fontsize=26, pad=16)
    ax.set_xlabel("X Pixel", fontsize=24)
    ax.set_ylabel("Y Pixel", fontsize=24)
    cbar = fig.colorbar(im, ax=ax, label="Poisson deviance |√D|", shrink=0.53, pad=0.006)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Poisson deviance |√D|", fontsize=22, labelpad=0)
        
    for ax in axs:
        ax.tick_params(labelsize=18)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return summary_output, fig, corner_fig, walker_map_fig

def process_observation(infile, progress_queue, obsid_coords, mcmc_scale_factors, emp_psf_file,
                        n_components_multi, run_mcmc_multi, mcmc_iter_multi,
                        mcmc_n_walkers, mcmc_ball_size, auto_stop=False, sigma_val=1, progress_step=None, progress_chunks=50,
                        recalc=False, chain_base_dir=None, signifiers=None, ephemeris=None):
    """
    Worker function to process a single observation end to end
    Orchestrates the Centroid Fit -> Source Fit -> Multi Component Fit pipeline
    """
    
    pdf_out_files = []
    multi_pdf_out_files = []
    
    obsid = os.path.dirname(os.path.dirname(infile))
    # Seed random generator deterministically using the Observation ID
    try:
        np.random.seed(int(obsid))
    except ValueError:
        np.random.seed(hash(obsid) % (2**32 - 1))

    if obsid not in obsid_coords:
        return (obsid, "", "", "", "", "", [], [])
    current_ra, current_dec = obsid_coords[obsid]
    
    # Extract data and get initial quick centroid
    date, exptime, pixel_x0_best, pixel_y0_best, cnt, qp_figs = data_extract_quickpos_iter(infile)
    
    header_text = f"observation: {obsid}\ninfile: {infile}\ndate: {date}, exptime: {exptime}\n"

    # Stage 1 Centroid Fit
    # Fit a single Gaussian to a large region to robustly find the global center
    img_width = 40
    cent_binsize = 1.0
    src_psf_images(obsid, infile, pixel_x0_best, pixel_y0_best, img_width, wcs_ra=current_ra, wcs_dec=current_dec, binsize=cent_binsize, psfimg=False, empirical_psf=None)
    logical_width = img_width / cent_binsize
    img_center = logical_width / 2.0 + 0.5
    
    centroid_fit_summary, centroid_fit_fig, _, _ = gaussian_image_fit(
        obsid, 1, (img_center, img_center), cnt, (1.0 / cent_binsize),
        prefix="centrg", background=0.1, pos_max=(logical_width, logical_width),
        use_mcmc=False, confirm=False, sigma_val=1
    )
    
    if centroid_fit_summary is None:
        clean()
        return (obsid, "", "", "", "", "", [], [])

    temp_cent_fit_png = f"2Dfits/temp_{obsid}_cent_fit.png"
    centroid_fit_fig.savefig(temp_cent_fit_png, dpi=400)
    plt.close(centroid_fit_fig)
    pdf_out_files.append(temp_cent_fit_png)

    # Calculate physical coordinates of the best fit centroid
    d = get_data()
    crval_x, crval_y = d.sky.crval
    crpix_x, crpix_y = d.sky.crpix
    cdelt_x, cdelt_y = d.sky.cdelt
    comp_x = get_model_component('centrg1').xpos.val
    comp_y = get_model_component('centrg1').ypos.val
    xphys_best = crval_x + (comp_x - crpix_x) * cdelt_x
    yphys_best = crval_y + (comp_y - crpix_y) * cdelt_y

    # Stage 2 Single Component Source Fit
    # Fit the source again at higher resolution using the empirical PSF
    img_width = 10
    src_binsize = 0.25
    src_psf_images(obsid, infile, xphys_best, yphys_best, img_width, wcs_ra=current_ra, wcs_dec=current_dec, binsize=src_binsize, psfimg=True, empirical_psf=emp_psf_file)
    logical_width = img_width / src_binsize 
    img_center = logical_width / 2.0 + 0.5 
    pixel_scale_guess = 1.0 / src_binsize 
    scaled_cnt_guess = cnt / (pixel_scale_guess**2)
    scaled_fwhm_guess = 1.0 * pixel_scale_guess 
    
    src_fit_summary, src_fit_fig, _, _ = gaussian_image_fit(
        obsid, 1, (img_center, img_center), scaled_cnt_guess, scaled_fwhm_guess,
        prefix="srcg", pos_max=(logical_width, logical_width),
        use_mcmc=False, confirm=False, sigma_val=1
    )
    
    if src_fit_summary is None:
        clean()
        return (obsid, header_text, centroid_fit_summary, "", "", "", pdf_out_files, [])

    temp_src_fit_png = f"2Dfits/temp_{obsid}_src_fit.png"
    src_fit_fig.savefig(temp_src_fit_png, dpi=400)
    plt.close(src_fit_fig)
    pdf_out_files.append(temp_src_fit_png)

    # Stage 3 Multi Component Fit
    # Use the single component fit as a template to seed the multi component model
    src_comp = get_model_component('srcg1')
    srcfit_off_x = src_comp.xpos.val - img_center 
    srcfit_off_y = src_comp.ypos.val - img_center 
    src_ampl = src_comp.ampl.val
    src_fwhm = src_comp.fwhm.val
    
    # Switch to a wider field of view for the multi-component fit
    img_width = 40 
    multi_binsize = 0.25
    src_psf_images(obsid, infile, xphys_best, yphys_best, img_width, wcs_ra=current_ra, wcs_dec=current_dec, binsize=multi_binsize, empirical_psf=emp_psf_file)
    
    logical_width = img_width / multi_binsize 
    img_center = logical_width / 2.0 + 0.5   
    pixel_scale = src_binsize / multi_binsize 
    new_xpos = img_center + (srcfit_off_x * pixel_scale)
    new_ypos = img_center + (srcfit_off_y * pixel_scale)
    scaled_src_fwhm = src_fwhm * pixel_scale
    scaled_src_ampl = src_ampl / (pixel_scale**2)
    
    pixel_scale_guess = 1.0 / multi_binsize 
    scaled_cnt_ampl = cnt / (pixel_scale_guess**2)
    scaled_default_fwhm = 1.0 * pixel_scale_guess 

    # Initialize all components at the center and let the optimizer spread them out
    n_components = n_components_multi 
    positions = [(new_xpos, new_ypos)] + [(img_center, img_center)] * (n_components - 1)
    amplitudes = [scaled_src_ampl] + [scaled_cnt_ampl] * (n_components - 1)
    fwhms = [scaled_src_fwhm] + [scaled_default_fwhm] * (n_components - 1)

    multi_fit_summary, multi_fit_fig, multi_corner_fig, multi_walker_fig = gaussian_image_fit(
        obsid, n_components, positions, amplitudes, fwhms,
        prefix="g", background=0.1, pos_max=(logical_width, logical_width),
        pos_min=(0, 0), exptime=exptime, confirm=False, 
        use_mcmc=run_mcmc_multi, mcmc_iter=mcmc_iter_multi,
        n_walkers=mcmc_n_walkers, ball_size=mcmc_ball_size,
        auto_stop=auto_stop,
        sigma_val=sigma_val,
        progress_chunks=progress_chunks, progress_step=progress_step, progress_queue=progress_queue,
        chain_base_dir=chain_base_dir,
        recalc=recalc,
        bin_size=multi_binsize,
        signifiers=signifiers,
        ephemeris=ephemeris,
        date_obs=date
    )

    if multi_fit_summary is None:
        clean()
        return (obsid, header_text, centroid_fit_summary, src_fit_summary, "", "", pdf_out_files, [])

    temp_multi_fit_png = f"2Dfits/temp_{obsid}_multi_fit.png"
    multi_fit_fig.savefig(temp_multi_fit_png, dpi=400)
    plt.close(multi_fit_fig)
    pdf_out_files.append(temp_multi_fit_png)
    multi_pdf_out_files.append(temp_multi_fit_png)

    if multi_walker_fig is not None:
        temp_walker_png = f"2Dfits/temp_{obsid}_walker_map.png"
        multi_walker_fig.savefig(temp_walker_png, dpi=400)
        plt.close(multi_walker_fig)
        pdf_out_files.append(temp_walker_png)

    if multi_corner_fig is not None:
        temp_corner_png = f"2Dfits/temp_{obsid}_corner.png"
        multi_corner_fig.savefig(temp_corner_png, dpi=400)
        plt.close(multi_corner_fig)
        pdf_out_files.append(temp_corner_png)

    multi_results_text = f"observation: {obsid}\ninfile: {infile}\ndate: {date}, exptime: {exptime}\n{multi_fit_summary}\n\n"
    clean()

    return (obsid, header_text, centroid_fit_summary, src_fit_summary, multi_fit_summary, multi_results_text, pdf_out_files, multi_pdf_out_files)
