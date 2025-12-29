import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import config
from lib.physics import ss433_phases, get_precession_limits, _get_closest_geometric_point

def _plot_jet_trajectories_on_ax(ax, mjd_obs, params, mappable, r_max, betas=None):
    """
    Helper function to plot the theoretical jet trajectories on a polar axis
    Calculates the path for both East and West jets over a range of travel times
    """
    jd_obs = mjd_obs + 2400000.5
    
    # Use specific fitted betas if provided otherwise default to the canonical value
    beta_e = betas['east'] if betas else params['beta']
    beta_w = betas['west'] if betas else params['beta']
    
    # Generate time grid for the trajectory
    t_travel = np.linspace(1.0, 325, 1000)
    jd_ej = jd_obs - t_travel
    
    # Calculate the kinematic positions for the entire time grid
    mu_e_ra, mu_e_dec, mu_w_ra, mu_w_dec, pa_e, pa_w = ss433_phases(jd_ej, params, beta_east=beta_e, beta_west=beta_w)
    
    # Convert proper motion to radial distance in arcseconds
    rad_e = (np.sqrt(mu_e_ra**2 + mu_e_dec**2) * config.C_PC_PER_DAY * t_travel / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
    rad_w = (np.sqrt(mu_w_ra**2 + mu_w_dec**2) * config.C_PC_PER_DAY * t_travel / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
    
    cmap, norm = mappable.get_cmap(), mappable.norm
    
    # Scatter plot the points colored by their age
    ax.scatter(np.deg2rad(pa_e), rad_e, c=t_travel, cmap=cmap, norm=norm, s=15, zorder=1)
    ax.scatter(np.deg2rad(pa_w), rad_w, c=t_travel, cmap=cmap, norm=norm, s=15, zorder=1)
    
    # Calculate and plot the precession cone limits
    limits = get_precession_limits(params)
    cone_style = dict(ls='--', lw=1, zorder=0.1)
    
    # Plot Cones where East is Blue and West is Red
    ax.plot([0, np.deg2rad(limits['east_min'])], [0, r_max+1], color='deepskyblue', **cone_style)
    ax.plot([0, np.deg2rad(limits['east_max'])], [0, r_max+1], color='deepskyblue', **cone_style)
    ax.plot([0, np.deg2rad(limits['west_min'])], [0, r_max+1], color='lightcoral', **cone_style)
    ax.plot([0, np.deg2rad(limits['west_max'])], [0, r_max+1], color='lightcoral', **cone_style)
    
    # Mark the central source location
    ax.plot(0, 0, '*', color='gold', markersize=15, zorder=10)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.grid(True, linestyle=':', alpha=0.7)

def _plot_jet_error_region_on_ax(ax, jet_side, fit_results, params, mappable):
    """
    Visualizes the uncertainty region for the jet trajectory based on the fitted beta error bounds
    Fills the area between the upper and lower beta trajectories
    """
    try:
        jet_entries = fit_results['jets'][jet_side]
        if not jet_entries: return
        lower_beta = jet_entries[0].get('beta_lower_bound')
        upper_beta = jet_entries[0].get('beta_upper_bound')
        if pd.isna(lower_beta) or pd.isna(upper_beta): return
    except (KeyError, IndexError): 
        return

    mjd_obs = fit_results['mjd_obs']
    jd_obs = mjd_obs + 2400000.5
    t_travel = np.linspace(0.0, 325, 1000)
    jd_ej = jd_obs - t_travel
    
    # Set beta bounds for the specific side being plotted
    beta_e_low = lower_beta if jet_side == 'east' else params['beta']
    beta_w_low = lower_beta if jet_side == 'west' else params['beta']
    beta_e_upp = upper_beta if jet_side == 'east' else params['beta']
    beta_w_upp = upper_beta if jet_side == 'west' else params['beta']

    # Calculate trajectories for both lower and upper beta bounds
    mu_e_l, mu_e_d_l, mu_w_l, mu_w_d_l, pa_e_l, pa_w_l = ss433_phases(jd_ej, params, beta_east=beta_e_low, beta_west=beta_w_low)
    mu_e_u, mu_e_d_u, mu_w_u, mu_w_d_u, pa_e_u, pa_w_u = ss433_phases(jd_ej, params, beta_east=beta_e_upp, beta_west=beta_w_upp)
    
    def get_coords(mu_ra, mu_dec, pa):
        rad = (np.sqrt(mu_ra**2 + mu_dec**2) * config.C_PC_PER_DAY * t_travel / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
        return rad, np.deg2rad(pa)

    if jet_side == 'east':
        rad_low, pa_low = get_coords(mu_e_l, mu_e_d_l, pa_e_l)
        rad_upp, pa_upp = get_coords(mu_e_u, mu_e_d_u, pa_e_u)
    else:
        rad_low, pa_low = get_coords(mu_w_l, mu_w_d_l, pa_w_l)
        rad_upp, pa_upp = get_coords(mu_w_u, mu_w_d_u, pa_w_u)

    # Fill the polygon defined by the two trajectories
    for i in range(len(t_travel) - 1):
        thetas = [pa_low[i], pa_upp[i], pa_upp[i+1], pa_low[i+1]]
        rads = [rad_low[i], rad_upp[i], rad_upp[i+1], rad_low[i+1]]
        avg_time = (t_travel[i] + t_travel[i+1]) / 2.0
        color = mappable.to_rgba(avg_time)
        ax.fill(thetas, rads, color=color, edgecolor='none', alpha=0.3, zorder=0.5)

def plot_fit_and_calc_results(obs_id, blob_data_list, fit_results, params, pdf_object=None):
    """
    Main plotting function that creates a side by side comparison
    Left panel shows the default model
    Right panel shows the best fit model with uncertainty regions and observed data
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), subplot_kw={'projection': 'polar'})
    fig.suptitle(f"observation id: {obs_id}", fontsize=20, y=0.97)
    mjd_obs = fit_results['mjd_obs']

    east_blobs = [b for b in blob_data_list if b['comp'].startswith('east')]
    west_blobs = [b for b in blob_data_list if b['comp'].startswith('west')]

    # Determine plot limits based on the furthest observed component
    all_rads = [b['rad_obs'] + b.get('rad_err_U', 0) for b in blob_data_list]
    max_rad = max(all_rads) if all_rads else 1.0
    common_rmax = max(1.3, max_rad * 1.2)
    
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=mcolors.Normalize(vmin=1.0, vmax=225))
    
    # Left Panel Default Model
    _plot_jet_trajectories_on_ax(ax1, mjd_obs, params, mappable, common_rmax)
    ax1.set_title(f"model at mjd {mjd_obs:.1f}\n(default beta = {params['beta']:.4f})", pad=20, fontsize=14)

    # Right Panel Fit Results
    _plot_jet_trajectories_on_ax(ax2, mjd_obs, params, mappable, common_rmax, betas=fit_results.get('fitted_betas'))
    _plot_jet_error_region_on_ax(ax2, 'east', fit_results, params, mappable)
    _plot_jet_error_region_on_ax(ax2, 'west', fit_results, params, mappable)
    ax2.set_title(f"fit/calculation results at mjd {mjd_obs:.1f}", pad=20, fontsize=14)

    def plot_blobs(blobs, color, label_prefix):
        for i, b in enumerate(blobs):
            label = f"{label_prefix} {b['comp']}" if i == 0 else None
            for ax in [ax1, ax2]:
                ax.errorbar(
                    np.deg2rad(b['pa_obs']), b['rad_obs'], 
                    yerr=[[abs(b['rad_err_L'])], [abs(b['rad_err_U'])]], 
                    xerr=[[np.deg2rad(abs(b['pa_err_L']))], [np.deg2rad(abs(b['pa_err_U']))]], 
                    fmt='o', color=color, ecolor='gray', capsize=3, zorder=3, label=label
                )

    plot_blobs(east_blobs, 'blue', "obs")
    plot_blobs(west_blobs, 'red', "obs")

    # Overlay connecting lines between observations and their model counterparts
    for jet_side, blob_list, color_char in [('east', east_blobs, 'b'), ('west', west_blobs, 'r')]:
        if not blob_list: continue
        results_list = fit_results['jets'][jet_side]
        
        for i, blob in enumerate(blob_list):
            try: res_entry = results_list[i]
            except IndexError: continue

            if res_entry['method'].startswith('fit'):
                # Draw a dashed line from the observation to the closest point on the model curve
                point = _get_closest_geometric_point(blob, jet_side, fit_results, params)
                ax2.plot([np.deg2rad(blob['pa_obs']), np.deg2rad(point['model_pa'])], 
                         [blob['rad_obs'], point['model_rad']], ls='--', c=color_char, alpha=0.6)
                
                label = None
                if i == 0:
                    beta_val = res_entry['fitted_beta']
                    if pd.notna(res_entry.get('beta_upper_bound')):
                        err_pos = res_entry['beta_upper_bound'] - beta_val
                        err_neg = beta_val - res_entry['beta_lower_bound']
                        label = f"fit {jet_side} (beta={beta_val:.4f} +{err_pos:.4f}/-{err_neg:.4f})"
                    else:
                        label = f"fit {jet_side} (beta={beta_val:.4f})"
                ax2.plot(np.deg2rad(point['model_pa']), point['model_rad'], 'x', c='w', ms=5, mec=color_char, zorder=4, label=label)
            else:
                # If no fit was found draw a dotted line to the origin as a fallback
                ax2.plot([0, np.deg2rad(blob['pa_obs'])], [0, blob['rad_obs']], ls=':', c=color_char, alpha=0.5)
                offset = 0.95 if jet_side == 'east' else 1.05
                ax2.text(offset*np.deg2rad(blob['pa_obs']), offset*blob['rad_obs'], f" {blob['comp']}", color=color_char, fontsize=8)

    ax1.set_rmax(common_rmax); ax1.set_rmin(0)
    ax2.set_rmax(common_rmax); ax2.set_rmin(0)
    
    # Configure Legend and Layout
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(-0.265, 1.05), frameon=False, fontsize=11)
    
    fig.subplots_adjust(left=0.05, right=0.88, top=0.85, bottom=0.1, wspace=0)
    
    # Add shared colorbar for time/age
    cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('travel time / age (days)', fontsize=12)
    
    if pdf_object: pdf_object.savefig(fig); plt.close(fig)
    else: plt.show()