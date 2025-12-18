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
from scipy.ndimage import rotate
import emcee
from emcee.backends import HDFBackend
import corner

from lib.image_utils import data_extract_quickpos_iter, write_pixelscale, compute_split_rhat

from sherpa.astro.ui import (
    load_data, image_close, image_data, load_psf, set_psf, image_psf,
    gauss2d, const2d, set_source, freeze, link, show_model, set_stat,
    set_method, set_method_opt, fit, get_fit_results, thaw, set_sampler,
    set_sampler_opt, covar, get_covar_results, get_draws, get_model_component, 
    get_model_component_image, get_data_image, get_model_image, get_data,
    clean, calc_stat
)
from ciao_contrib.runtool import dmcopy, reproject_image

# suppress sherpa info messages
logging.getLogger("sherpa").setLevel(logging.WARNING)

def src_psf_images(obsid, infile, x0, y0, diameter, wcs_ra, wcs_dec, binsize=0.25, shape='square', psfimg=True, showimg=False, empirical_psf=None):
    if shape.lower() == 'circle':
        region_str = f"circle({x0},{y0},{diameter/2})"
    elif shape.lower() == 'square':
        region_str = f"box({x0},{y0},{diameter},{diameter},0)"
        img_region_str = f"box(256.5,256.5,{diameter*4},{diameter*4},0)"
    else:
        region_str = shape.lower()
        
    logical_width = diameter/binsize
    imagefile=f'{obsid}/src_image_{shape}_{int(logical_width)}pixel.fits'
    psf_rotated = f'{obsid}/psf_rotated.fits'
    psf_rotated_cut = f'{obsid}/psf_rotated_cut.fits'
    emp_psf_imagefile = f'{obsid}/psf_image_{shape}_empirical_{int(logical_width)}pixel.fits'
    
    dmcopy.punlearn()
    dmcopy.clobber = 'yes'
    reproject_image.punlearn()
    reproject_image.clobber = 'yes'

    dmcopy.infile = f'{infile}[sky={region_str}][bin x=::{binsize},y=::{binsize}]'
    dmcopy.outfile = imagefile
    dmcopy()
    load_data(imagefile)

    if empirical_psf is not None:
        try:
            with fits.open(infile) as hdu_match:
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
        
        rotated_psf_data = rotate(
            psf_data, angle_to_rotate, reshape=False, cval=0.0, order=3
        )
        psf_header.add_history(f"rotated by {angle_to_rotate:.4f} deg")
        hdu_out = fits.PrimaryHDU(data=rotated_psf_data, header=psf_header)
        try:
            hdu_out.writeto(psf_rotated, overwrite=True)
        except Exception as e:
            print(f"  error: could not write output file: {psf_rotated}")
        
        try:
            with fits.open(psf_rotated) as hdu_rot:
                nx = hdu_rot[0].header['NAXIS1']
                ny = hdu_rot[0].header['NAXIS2']
            write_pixelscale(file=psf_rotated, nx=nx, ny=ny, ra=str(wcs_ra), dec=str(wcs_dec))
        except Exception as e:
            print(f"!!! error (obsid {obsid}): wcs stamping failed: {e}")

        dmcopy.infile = f'{psf_rotated}[{img_region_str}][bin x=::{binsize*4},y=::{binsize*4}]'
        dmcopy.outfile = psf_rotated_cut
        dmcopy()
        reproject_image.infile = psf_rotated_cut
        reproject_image.matchfile = imagefile
        reproject_image.outfile = emp_psf_imagefile
        reproject_image.method = 'sum'
        reproject_image()
        load_psf(f'centr_psf{obsid}', emp_psf_imagefile)
        set_psf(f'centr_psf{obsid}')

    return binsize

def gaussian_image_fit(observation, n_components, position, ampl, fwhm,
                       background=0, pos_min=(0, 0), pos_max=None, exptime=None, lock_fwhm=True,
                       freeze_components=None, use_mcmc=True, mcmc_iter=5000, mcmc_burn_in_frac=0.2,
                       n_walkers=32, ball_size=1e-4, sigma_val=1, 
                       prefix="g", confirm=True, imgfit=False, progress_chunks=50, progress_queue=None,
                       chain_base_dir=None, recalc=False, bin_size=None):
    
    # helper to expand single value inputs
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

    positions = process_tuple_param(position, "position")
    ampls = process_numeric_param(ampl, "ampl")
    fwhms = process_numeric_param(fwhm, "fwhm")
    pos_mins = process_tuple_param(pos_min, "pos_min")
    pos_maxs = [None] * n_components if pos_max is None else process_tuple_param(pos_max, "pos_max")

    comp_names = []
    gaussian_components = []
    model_components = []
    for i in range(1, n_components + 1):
        comp_name = f"{prefix}{i}"
        comp_names.append(comp_name)
        comp = gauss2d(comp_name)
        gaussian_components.append(comp)
        model_components.append(comp)
    
    bkg_comp = None
    if background > 0:
        bkg_comp = const2d("c1")
        model_components.append(bkg_comp)
    
    if model_components: set_source(sum(model_components))
    else: raise ValueError("model expression is empty.")

    freeze_list = (freeze_components if isinstance(freeze_components, list) else ([freeze_components] if freeze_components is not None else []))
    for i, comp in enumerate(gaussian_components):
        comp_number = i + 1
        comp.xpos = positions[i][0]; comp.ypos = positions[i][1]
        comp.ampl = ampls[i]; comp.fwhm = fwhms[i]
        if hasattr(comp.xpos, 'min'): comp.xpos.min = pos_mins[i][0]
        if hasattr(comp.ypos, 'min'): comp.ypos.min = pos_mins[i][1]
        if pos_maxs[i] is not None:
            if hasattr(comp.xpos, 'max'): comp.xpos.max = pos_maxs[i][0]
            if hasattr(comp.ypos, 'max'): comp.ypos.max = pos_maxs[i][1]
        if hasattr(comp.ampl, 'min'): comp.ampl.min = 0
        if comp_number in freeze_list: freeze(comp)

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

    set_stat('cstat')
    set_method('moncar'); set_method_opt('numcores', 1)
    set_method_opt('population_size', 10 * 16 * (n_components * 3 + 1)); set_method_opt('xprob', 0.5); set_method_opt('weighting_factor', 0.5)
    fit()
    set_method('simplex'); fit()
    fit_results = get_fit_results()

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

    mcmc_results = None
    walker_map_fig = None
    corner_fig = None
    mcmc_duration_str = ""
    flux_results = None
    
    if use_mcmc:
        mcmc_start_time = time.time()
        ndim = len(thawed_pars)
        
        def log_probability(theta):
            for param, value in zip(thawed_pars, theta):
                if value < param.min or value > param.max: return -np.inf
            for param, value in zip(thawed_pars, theta): param.val = value
            return -0.5 * calc_stat()

        current_n_walkers = n_walkers if n_walkers >= 2 * ndim else 2 * ndim + 2
        
        ball_str = str(ball_size).replace('.', 'p')
        param_folder_name = (f"mcmc-chain-{n_components}comp-"
                             f"{current_n_walkers}walkers-"
                             f"{mcmc_iter}steps-"
                             f"{ball_str}ball")
        
        if bin_size is not None:
            bin_str = str(bin_size).replace('.', 'p')
            param_folder_name += f"-bin{bin_str}"
        
        if chain_base_dir is None:
            chain_base_dir = os.path.join(os.getcwd(), "2Dfits", "emcee_chains")
                
        chain_dir = os.path.join(chain_base_dir, param_folder_name)
        os.makedirs(chain_dir, exist_ok=True)
        chain_filename = os.path.join(chain_dir, f"{observation}_chain.h5")
        
        backend = HDFBackend(chain_filename, compression="gzip", compression_opts=4)

        current_steps = 0
        try:
            current_steps = backend.iteration
        except:
            pass

        run_sampler = True
        p0 = None
        
        if not recalc and current_steps > 0:
            if current_steps >= mcmc_iter:
                print(f"[{observation}] Found complete chain ({current_steps} steps). Skipping fit.")
                run_sampler = False
            else:
                print(f"[{observation}] Found partial chain ({current_steps}/{mcmc_iter} steps). Resuming...")
                p0 = None # Resumes automatically
        
        elif recalc and current_steps > 0:
            print(f"[{observation}] Recalc requested. Overwriting existing chain at {chain_filename}")
            backend.reset(current_n_walkers, ndim)
        
        else:
            print(f"[{observation}] No valid chain found. Starting fresh at {chain_filename}")
            backend.reset(current_n_walkers, ndim)

        # Generate initial position if starting fresh or resetting
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
                        update_interval = max(1, int(mcmc_iter / progress_chunks))
                        for i, sample in enumerate(sampler.sample(p0, iterations=remaining_steps, progress=False)):
                            if progress_queue and (i + 1) % update_interval == 0: 
                                progress_queue.put(1)
                        
                except Exception as e:
                    print(f"  error (obsid {observation}) sampler crashed: {e}")
            
            discard = int(mcmc_iter * mcmc_burn_in_frac)
            if sampler.iteration < discard: discard = 0
                 
            flat_samples = sampler.get_chain(discard=discard, flat=True)
            raw_chain = sampler.get_chain(discard=discard) 
            
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

            conv_str = (
                f"convergence stats:\n"
                f"  max autocorr time (tau): {tau_max:.1f} steps\n"
                f"  max split-rhat:          {rhat_max:.4f} (goal < 1.1)\n"
                f"  effective samples (ess): {int(ess)}\n"
                f"  chain length / tau:      {raw_chain.shape[0] / tau_max:.1f} (goal > 50)\n\n"
            )

            if sigma_val == 1:
                q_low, q_mid, q_high = 15.865, 50.0, 84.135
            elif sigma_val == 2:
                q_low, q_mid, q_high = 2.275, 50.0, 97.725
            elif sigma_val == 3:
                q_low, q_mid, q_high = 0.135, 50.0, 99.865
            else:
                q_low, q_mid, q_high = 15.865, 50.0, 84.135

            mcmc_results_data = {'parnames': [], 'parvals': [], 'parmins': [], 'parmaxes': []}
            for i, name in enumerate(thawed_par_names):
                mcmc_vals = flat_samples[:, i]
                p_low, p_mid, p_high = np.percentile(mcmc_vals, [q_low, q_mid, q_high])
                mcmc_results_data['parnames'].append(name)
                mcmc_results_data['parvals'].append(p_mid)
                mcmc_results_data['parmins'].append(p_low)
                mcmc_results_data['parmaxes'].append(p_high)
            mcmc_results = mcmc_results_data

            log_probs = sampler.get_log_prob(discard=discard, flat=True)
            max_idx = np.argmax(log_probs)
            best_mcmc_stat = -2.0 * log_probs[max_idx]
            
            if best_mcmc_stat < best_fit_stat:
                best_fit_values = list(flat_samples[max_idx])
                best_fit_stat = best_mcmc_stat

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
                            flux_chain = A_chain * (F_chain**2)
                            F_low, F_mid, F_high = np.percentile(flux_chain, [q_low, q_mid, q_high])
                            flux_results[comp.name] = (F_low, F_mid, F_high)

            for param, val in zip(thawed_pars, best_fit_values): param.val = val

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
            
            for i, comp_name in enumerate(comp_names):
                x_name = f"{comp_name}.xpos"; y_name = f"{comp_name}.ypos"
                if x_name in thawed_par_names and y_name in thawed_par_names:
                    x_idx = thawed_par_names.index(x_name)
                    y_idx = thawed_par_names.index(y_name)
                    x_pts = raw_chain[:, :, x_idx].flatten()
                    y_pts = raw_chain[:, :, y_idx].flatten()
                    
                    H, xedges, yedges = np.histogram2d(
                        y_pts, x_pts, bins=[ny, nx], 
                        range=[[0.5, ny + 0.5], [0.5, nx + 0.5]]
                    )
                    if np.sum(H) > 0:
                        comp_color = colors[i % len(colors)]
                        dark_c = dark_colors[i % len(dark_colors)]
                        base_rgb = mcolors.to_rgb(comp_color)
                        peak = H.max()
                        levels = [peak * 0.1, peak * 0.3, peak * 0.5, peak * 0.7, peak * 0.9]
                        fill_colors = [(*base_rgb, 0.1), (*base_rgb, 0.3), (*base_rgb, 0.5), (*base_rgb, 0.7), (*base_rgb, 0.9)]
                        ax.contourf(H, levels=levels, colors=fill_colors, extend='max', extent=plot_extent)
                        ax.contour(H, levels=levels, colors=[comp_color], linewidths=1.0, alpha=0.9, extent=plot_extent)
                        bf_x = best_fit_values[x_idx]; bf_y = best_fit_values[y_idx]
                        ax.scatter(bf_x, bf_y, marker='o', color=dark_c, s=100, zorder=20, edgecolors='white', label=f"{comp_name} best fit")
                        if mcmc_results is not None:
                            med_x = mcmc_results['parvals'][x_idx]
                            med_y = mcmc_results['parvals'][y_idx]
                            ax.scatter(med_x, med_y, marker='x', color=dark_c, s=200, linewidth=3, zorder=19, label=f"{comp_name} median")

            ax.set_title(f"walker density map - obsid {observation}"); ax.set_xlabel("x pixel"); ax.set_ylabel("y pixel")
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            walker_map_fig.colorbar(im_data, ax=ax, label="counts", shrink=0.8); walker_map_fig.tight_layout()

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

    fig, axs = plt.subplots(1, 3, figsize=(30, 15))
    data_img = get_data_image(); data_vals = data_img.y
    min_pos = np.min(data_vals[data_vals > 0]) if np.any(data_vals > 0) else 1e-9
    display_floor = min_pos / 10.0
    data_masked = np.maximum(data_vals, display_floor)
    model_img = get_model_image(); model_vals = model_img.y
    model_masked = np.maximum(model_vals, display_floor)
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
        color = base_colors[i % len(base_colors)]
        ax.contour(comp_vals, levels=[0.2 * np.max(comp_vals)], colors=[color], linestyles=['--'], linewidths=2)
        legend_elements.append(Line2D([0], [0], lw=2, linestyle='--', color=color, label=f"{comp_name}"))
    if legend_elements: ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(f"{observation} data + best fit overlay"); fig.colorbar(im, ax=ax, label="counts", shrink=0.53)

    ax = axs[1]
    im = ax.imshow(model_masked, origin='lower', cmap='gnuplot2', norm=log_norm, interpolation='nearest')
    ax.set_title("best fit model"); fig.colorbar(im, ax=ax, label="model counts", shrink=0.53)

    ax = axs[2]
    im = ax.imshow(np.abs(resid_dev), origin='lower', cmap='gnuplot2', norm=mcolors.Normalize(vmin=0, vmax=5), interpolation='nearest')
    ax.set_title("poisson deviance (best fit)"); fig.colorbar(im, ax=ax, label="|residuals|", shrink=0.53)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return summary_output, fig, corner_fig, walker_map_fig

def process_observation(infile, progress_queue, obsid_coords, mcmc_scale_factors, emp_psf_file,
                        n_components_multi, run_mcmc_multi, mcmc_iter_multi,
                        mcmc_n_walkers, mcmc_ball_size, sigma_val, progress_chunks=50, recalc=False,
                        chain_base_dir=None):
    
    pdf_out_files = []
    multi_pdf_out_files = []
    
    obsid = os.path.dirname(os.path.dirname(infile))
    try:
        np.random.seed(int(obsid))
    except ValueError:
        np.random.seed(hash(obsid) % (2**32 - 1))

    if obsid not in obsid_coords:
        return (obsid, "", "", "", "", "", [], [])
    current_ra, current_dec = obsid_coords[obsid]
    
    date, exptime, pixel_x0_best, pixel_y0_best, cnt, qp_figs = data_extract_quickpos_iter(infile)
    
    header_text = f"observation: {obsid}\ninfile: {infile}\ndate: {date}, exptime: {exptime}\n"

    # stage 1: centroid fit
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
    centroid_fit_fig.savefig(temp_cent_fit_png)
    plt.close(centroid_fit_fig)
    pdf_out_files.append(temp_cent_fit_png)

    d = get_data()
    crval_x, crval_y = d.sky.crval
    crpix_x, crpix_y = d.sky.crpix
    cdelt_x, cdelt_y = d.sky.cdelt
    comp_x = get_model_component('centrg1').xpos.val
    comp_y = get_model_component('centrg1').ypos.val
    xphys_best = crval_x + (comp_x - crpix_x) * cdelt_x
    yphys_best = crval_y + (comp_y - crpix_y) * cdelt_y

    # stage 2: single component source fit
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
    src_fit_fig.savefig(temp_src_fit_png)
    plt.close(src_fit_fig)
    pdf_out_files.append(temp_src_fit_png)

    # stage 3: multi component fit
    src_comp = get_model_component('srcg1')
    srcfit_off_x = src_comp.xpos.val - img_center 
    srcfit_off_y = src_comp.ypos.val - img_center 
    src_ampl = src_comp.ampl.val
    src_fwhm = src_comp.fwhm.val
    
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
        sigma_val=sigma_val,
        progress_chunks=progress_chunks, progress_queue=progress_queue,
        chain_base_dir=chain_base_dir,
        recalc=recalc,
        bin_size=multi_binsize
    )

    if multi_fit_summary is None:
        clean()
        return (obsid, header_text, centroid_fit_summary, src_fit_summary, "", "", pdf_out_files, [])

    temp_multi_fit_png = f"2Dfits/temp_{obsid}_multi_fit.png"
    multi_fit_fig.savefig(temp_multi_fit_png)
    plt.close(multi_fit_fig)
    pdf_out_files.append(temp_multi_fit_png)
    multi_pdf_out_files.append(temp_multi_fit_png)

    if multi_walker_fig is not None:
        temp_walker_png = f"2Dfits/temp_{obsid}_walker_map.png"
        multi_walker_fig.savefig(temp_walker_png)
        plt.close(multi_walker_fig)
        pdf_out_files.append(temp_walker_png)

    if multi_corner_fig is not None:
        temp_corner_png = f"2Dfits/temp_{obsid}_corner.png"
        multi_corner_fig.savefig(temp_corner_png)
        plt.close(multi_corner_fig)
        pdf_out_files.append(temp_corner_png)

    multi_results_text = f"observation: {obsid}\ninfile: {infile}\ndate: {date}, exptime: {exptime}\n{multi_fit_summary}\n\n"
    clean()

    return (obsid, header_text, centroid_fit_summary, src_fit_summary, multi_fit_summary, multi_results_text, pdf_out_files, multi_pdf_out_files)