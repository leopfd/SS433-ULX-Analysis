import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import config
from lib.physics import ss433_phases, get_precession_limits, _get_closest_geometric_point

def _plot_jet_trajectories_on_ax(ax, mjd_obs, params, mappable, r_max, betas=None):
    jd_obs = mjd_obs + 2400000.5
    beta_b = betas['blue'] if betas else params['beta']
    beta_r = betas['red'] if betas else params['beta']
    t_travel = np.linspace(1.0, 325, 1000)
    jd_ej = jd_obs - t_travel
    mu_b_ra, mu_b_dec, mu_r_ra, mu_r_dec, pa_b, pa_r = ss433_phases(jd_ej, params, beta_blue=beta_b, beta_red=beta_r)
    rad_b = (np.sqrt(mu_b_ra**2 + mu_b_dec**2) * config.C_PC_PER_DAY * t_travel / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
    rad_r = (np.sqrt(mu_r_ra**2 + mu_r_dec**2) * config.C_PC_PER_DAY * t_travel / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
    cmap, norm = mappable.get_cmap(), mappable.norm
    ax.scatter(np.deg2rad(pa_b), rad_b, c=t_travel, cmap=cmap, norm=norm, s=15, zorder=1)
    ax.scatter(np.deg2rad(pa_r), rad_r, c=t_travel, cmap=cmap, norm=norm, s=15, zorder=1)
    limits = get_precession_limits(params)
    ax.plot([0, np.deg2rad(limits['blue_min'])], [0, r_max+1], color='deepskyblue', ls='--', lw=1, zorder=0.1)
    ax.plot([0, np.deg2rad(limits['blue_max'])], [0, r_max+1], color='deepskyblue', ls='--', lw=1, zorder=0.1)
    ax.plot([0, np.deg2rad(limits['red_min'])], [0, r_max+1], color='lightcoral', ls='--', lw=1, zorder=0.1)
    ax.plot([0, np.deg2rad(limits['red_max'])], [0, r_max+1], color='lightcoral', ls='--', lw=1, zorder=0.1)
    ax.plot(0, 0, '*', color='gold', markersize=15, zorder=10)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(1); ax.grid(True, linestyle=':', alpha=0.7)

def _plot_jet_error_region_on_ax(ax, jet_type, fit_results, params, mappable):
    try:
        if jet_type == 'blue':
            lower_beta = fit_results['jets']['blue']['beta_lower_bound']
            upper_beta = fit_results['jets']['blue']['beta_upper_bound']
        else:
            lower_beta = fit_results['jets']['red']['beta_lower_bound']
            upper_beta = fit_results['jets']['red']['beta_upper_bound']
    except KeyError: return
    if pd.isna(lower_beta) or pd.isna(upper_beta): return
    mjd_obs = fit_results['mjd_obs']
    jd_obs = mjd_obs + 2400000.5
    t_travel = np.linspace(0.0, 325, 1000)
    jd_ej = jd_obs - t_travel
    beta_b_low = lower_beta if jet_type == 'blue' else params['beta']
    beta_r_low = lower_beta if jet_type == 'red' else params['beta']
    mu_b_ra_l, mu_b_dec_l, mu_r_ra_l, mu_r_dec_l, pa_b_l, pa_r_l = ss433_phases(jd_ej, params, beta_blue=beta_b_low, beta_red=beta_r_low)
    beta_b_upp = upper_beta if jet_type == 'blue' else params['beta']
    beta_r_upp = upper_beta if jet_type == 'red' else params['beta']
    mu_b_ra_u, mu_b_dec_u, mu_r_ra_u, mu_r_dec_u, pa_b_u, pa_r_u = ss433_phases(jd_ej, params, beta_blue=beta_b_upp, beta_red=beta_r_upp)
    
    if jet_type == 'blue':
        rad_low = (np.sqrt(mu_b_ra_l**2 + mu_b_dec_l**2) * config.C_PC_PER_DAY * t_travel / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
        pa_low_rad = np.deg2rad(pa_b_l)
        rad_upp = (np.sqrt(mu_b_ra_u**2 + mu_b_dec_u**2) * config.C_PC_PER_DAY * t_travel / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
        pa_upp_rad = np.deg2rad(pa_b_u)
    else:
        rad_low = (np.sqrt(mu_r_ra_l**2 + mu_r_dec_l**2) * config.C_PC_PER_DAY * t_travel / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
        pa_low_rad = np.deg2rad(pa_r_l)
        rad_upp = (np.sqrt(mu_r_ra_u**2 + mu_r_dec_u**2) * config.C_PC_PER_DAY * t_travel / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
        pa_upp_rad = np.deg2rad(pa_r_u)

    for i in range(len(t_travel) - 1):
        thetas = [pa_low_rad[i], pa_upp_rad[i], pa_upp_rad[i+1], pa_low_rad[i+1]]
        rads = [rad_low[i], rad_upp[i], rad_upp[i+1], rad_low[i+1]]
        avg_time = (t_travel[i] + t_travel[i+1]) / 2.0
        color = mappable.to_rgba(avg_time)
        ax.fill(thetas, rads, color=color, edgecolor='none', alpha=0.5, zorder=0.5)

def plot_fit_and_calc_results(obs_id, blob_data_list, fit_results, params, pdf_object=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), subplot_kw={'projection': 'polar'})
    fig.suptitle(f"observation id: {obs_id}", fontsize=20, y=0.97)
    mjd_obs = fit_results['mjd_obs']
    blob_blue = next(b for b in blob_data_list if b['comp'] == 'east')
    blob_red = next(b for b in blob_data_list if b['comp'] == 'west')
    max_rad = max(b['rad_obs'] + b['rad_err_U'] for b in blob_data_list)
    common_rmax = max(1.3, max_rad * 1.2)
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=mcolors.Normalize(vmin=1.0, vmax=225))
    
    _plot_jet_trajectories_on_ax(ax1, mjd_obs, params, mappable, common_rmax)
    ax1.set_title(f"model at mjd {mjd_obs:.1f}\n(default beta = {params['beta']:.4f})", pad=20, fontsize=14)

    _plot_jet_trajectories_on_ax(ax2, mjd_obs, params, mappable, common_rmax, betas=fit_results['fitted_betas'])
    _plot_jet_error_region_on_ax(ax2, 'blue', fit_results, params, mappable)
    _plot_jet_error_region_on_ax(ax2, 'red', fit_results, params, mappable)
    ax2.set_title(f"fit/calculation results at mjd {mjd_obs:.1f}", pad=20, fontsize=14)

    for ax in [ax1, ax2]:
        ax.errorbar(np.deg2rad(blob_blue['pa_obs']), blob_blue['rad_obs'], yerr=[[abs(blob_blue['rad_err_L'])], [abs(blob_blue['rad_err_U'])]], xerr=[[np.deg2rad(abs(blob_blue['pa_err_L']))], [np.deg2rad(abs(blob_blue['pa_err_U']))]], fmt='o', color='blue', ecolor='gray', capsize=3, zorder=3, label="observed blue jet")
        ax.errorbar(np.deg2rad(blob_red['pa_obs']), blob_red['rad_obs'], yerr=[[abs(blob_red['rad_err_L'])], [abs(blob_red['rad_err_U'])]], xerr=[[np.deg2rad(abs(blob_red['pa_err_L']))], [np.deg2rad(abs(blob_red['pa_err_U']))]], fmt='o', color='red', ecolor='gray', capsize=3, zorder=3, label="observed red jet")

    if fit_results['jets']['blue']['method'] == 'fit':
        point = _get_closest_geometric_point(blob_blue, 'blue', fit_results, params)
        ax2.plot([np.deg2rad(blob_blue['pa_obs']), np.deg2rad(point['model_pa'])], [blob_blue['rad_obs'], point['model_rad']], ls='--', c='blue')
        beta_val = fit_results['jets']['blue']['fitted_beta']
        err_pos = fit_results['jets']['blue']['beta_upper_bound'] - beta_val
        err_neg = beta_val - fit_results['jets']['blue']['beta_lower_bound']
        ax2.plot(np.deg2rad(point['model_pa']), point['model_rad'], 'x', c='w', ms=5, mec='b', zorder=4, label=f"fitted blue jet (beta={beta_val:.4f} +{err_pos:.4f} / -{err_neg:.4f}c)")
    else:
        ax2.plot([0, np.deg2rad(blob_blue['pa_obs'])], [0, blob_blue['rad_obs']], ls=':', c='blue')
        ax2.text(0.95*np.deg2rad(blob_blue['pa_obs']), 0.95*blob_blue['rad_obs'], f" t={fit_results['jets']['blue']['travel_time']:.0f}d", color='blue', ha='left', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

    if fit_results['jets']['red']['method'] == 'fit':
        point = _get_closest_geometric_point(blob_red, 'red', fit_results, params)
        ax2.plot([np.deg2rad(blob_red['pa_obs']), np.deg2rad(point['model_pa'])], [blob_red['rad_obs'], point['model_rad']], ls='--', c='red')
        beta_val = fit_results['jets']['red']['fitted_beta']
        err_pos = fit_results['jets']['red']['beta_upper_bound'] - beta_val
        err_neg = beta_val - fit_results['jets']['red']['beta_lower_bound']
        ax2.plot(np.deg2rad(point['model_pa']), point['model_rad'], 'x', c='w', ms=5, mec='r', zorder=4, label=f"fitted red jet (beta={beta_val:.4f} +{err_pos:.4f} / -{err_neg:.4f}c)")
    else:
        ax2.plot([0, np.deg2rad(blob_red['pa_obs'])], [0, blob_red['rad_obs']], ls=':', c='red')
        ax2.text(1.05*np.deg2rad(blob_red['pa_obs']), 1.05*blob_red['rad_obs'], f" t={fit_results['jets']['red']['travel_time']:.0f}d", color='red', ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
    
    ax1.set_rmax(common_rmax); ax1.set_rmin(0)
    ax2.set_rmax(common_rmax); ax2.set_rmin(0)
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(-0.265, 1.05), frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.85, bottom=0.1, wspace=0)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('travel time / age (days)', fontsize=12)
    if pdf_object: pdf_object.savefig(fig); plt.close(fig)
    else: plt.show()