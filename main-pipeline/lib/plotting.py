import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
import config
from lib.physics import (
    ss433_phases,
    get_precession_limits,
    _get_closest_geometric_point,
    ss433_mu_from_config_ephemeris,
    tau_core_to_knot_days_from_projected,
    _age_curve_one_cycle,
)

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
    t_travel = _age_curve_one_cycle(params)
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
    t_travel = _age_curve_one_cycle(params)
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
    Main plotting function for jet kinematics
    Shows the best fit model with uncertainty regions and observed data
    """
    fig, ax_fit = plt.subplots(1, 1, figsize=(12, 9), subplot_kw={'projection': 'polar'})
    betas = fit_results.get('fitted_betas', {})

    def side_has_fit(side):
        entries = fit_results.get('jets', {}).get(side, [])
        for e in entries:
            if str(e.get('method', '')).startswith('fit') and not pd.isna(e.get('fitted_beta')):
                return True
        return False

    def get_beta(side):
        val = betas.get(side) if betas else None
        if val is None:
            entries = fit_results.get('jets', {}).get(side, [])
            for e in entries:
                if str(e.get('method', '')).startswith('fit') and not pd.isna(e.get('fitted_beta')):
                    return e.get('fitted_beta')
        return val

    beta_e = get_beta('east') if side_has_fit('east') else None
    beta_w = get_beta('west') if side_has_fit('west') else None
    beta_e_str = f"{beta_e:.4f}" if beta_e is not None else "n/a"
    beta_w_str = f"{beta_w:.4f}" if beta_w is not None else "n/a"
    beta_parts = [f"MJD={fit_results['mjd_obs']:.1f}"]
    if beta_e is not None:
        beta_parts.append(f"β$_\\leftarrow$={beta_e_str}")
    if beta_w is not None:
        beta_parts.append(f"β$_\\rightarrow$={beta_w_str}")
    beta_suffix = ", ".join(beta_parts)
    fig.suptitle(f"Jet Model Fit ObsID {obs_id} ({beta_suffix})", fontsize=20, y=0.97)
    mjd_obs = fit_results['mjd_obs']

    east_blobs = [b for b in blob_data_list if b['comp'].startswith('east')]
    west_blobs = [b for b in blob_data_list if b['comp'].startswith('west')]

    # Determine plot limits based on the furthest observed component
    all_rads = [b['rad_obs'] + b.get('rad_err_U', 0) for b in blob_data_list]
    max_rad = max(all_rads) if all_rads else 1.0
    common_rmax = max(1.3, max_rad * 1.2)
    
    mappable = plt.cm.ScalarMappable(
        cmap=plt.cm.rainbow,
        norm=mcolors.Normalize(vmin=0, vmax=225))
    
    # Fit Results Panel
    _plot_jet_trajectories_on_ax(ax_fit, mjd_obs, params, mappable, common_rmax, betas=fit_results.get('fitted_betas'))
    _plot_jet_error_region_on_ax(ax_fit, 'east', fit_results, params, mappable)
    _plot_jet_error_region_on_ax(ax_fit, 'west', fit_results, params, mappable)
    ax_fit.set_ylabel(None)
    ax_fit.set_thetagrids(range(0, 360, 45), labels=[f"{deg}°" for deg in range(0, 360, 45)], fontsize=14)
    ax_fit.set_rlabel_position(135)
    ax_fit.tick_params(labelsize=14)
    # Give only 3-digit degree labels a little extra radial offset to avoid clipping
    for lbl in ax_fit.xaxis.get_majorticklabels():
        text = lbl.get_text().replace("°", "").strip()
        try:
            angle_val = float(text)
        except ValueError:
            continue
        if abs(angle_val) >= 100:
            x_pos, y_pos = lbl.get_position()
            lbl.set_position((x_pos, y_pos - 0.03))
    label_theta = np.deg2rad(145.5)  # place label along a consistent angle
    label_r = 0.92 * common_rmax  # push start of text outward so the whole phrase stays inside the field
    ax_fit.text(
        label_theta, label_r, "Projected separation (arcsec)",
        rotation=48.5, rotation_mode="anchor",
        ha="left", va="center", color="#000", fontsize=14,
        zorder=5,
    )

    def plot_blobs(blobs, color):
        for i, b in enumerate(blobs):
            ax_fit.errorbar(
                np.deg2rad(b['pa_obs']), b['rad_obs'], 
                yerr=[[abs(b['rad_err_L'])], [abs(b['rad_err_U'])]], 
                xerr=[[np.deg2rad(abs(b['pa_err_L']))], [np.deg2rad(abs(b['pa_err_U']))]], 
                fmt='o', color=color, ecolor='gray', capsize=3, zorder=3, label=None
            )

    plot_blobs(east_blobs, 'blue')
    plot_blobs(west_blobs, 'red')

    def _add_tau_label(ax, th, rr, text, color_char):
        r_text = 0.58 * float(rr)  # slightly closer to center for readability

        p0 = ax.transData.transform((th, 0.0))
        p1 = ax.transData.transform((th, float(rr)))
        dx, dy = (p1 - p0)

        ang = np.degrees(np.arctan2(dy, dx))

        # Keep text upright
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        norm = np.hypot(dx, dy)
        if norm < 1e-9:
            ax.text(th, r_text, text, fontsize=10, color=color_char,
                    ha="center", va="center", alpha=0.9)
            return

        nx, ny = -dy / norm, dx / norm  # perpendicular in screen coords

        ax.annotate(
            text,
            xy=(th, r_text),
            xytext=(8 * nx, 8 * ny),      # 8 px offset "above" the arrow
            textcoords="offset pixels",
            fontsize=10,
            color=color_char,
            ha="center",
            va="center",
            alpha=0.9,
            rotation=ang,
            rotation_mode="anchor",
        )

    # Overlay connecting lines between observations and their model counterparts
    for jet_side, blob_list, color_char in [('east', east_blobs, 'b'), ('west', west_blobs, 'r')]:
        if not blob_list: continue
        results_map = {e["blob_id"]: e for e in fit_results["jets"][jet_side]}

        for i, blob in enumerate(blob_list):
            res_entry = results_map.get(blob["comp"])
            if res_entry is None:
                continue

            th = np.deg2rad(blob['pa_obs'])
            rr = blob['rad_obs']

            # draw arrow (core -> knot)
            point = None
            if res_entry["method"].startswith("fit"):
                point = _get_closest_geometric_point(blob, jet_side, fit_results, params)

            if point is not None:
                th_end = np.deg2rad(point["model_pa"])
                rr_end = float(point["model_rad"])
            else:
                th_end = th
                rr_end = rr

            ax_fit.annotate(
                "",
                xy=(th_end, rr_end),
                xytext=(th_end, 0.0),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=color_char, alpha=0.6),
                zorder=2,
            )

            # compute tau(core->knot) for label
            tau_txt = None
            try:
                point_tau = _get_closest_geometric_point(blob, jet_side, fit_results, params)
                jd_ej = point_tau.get("jd_ej", None)

                if jd_ej is not None:
                    mu_e, mu_w = ss433_mu_from_config_ephemeris(jd_ej)
                    mu = mu_e if jet_side == "east" else mu_w

                    rad_for_tau = float(point_tau.get("model_rad", rr))
                    tau_days = tau_core_to_knot_days_from_projected(rad_for_tau, mu)
                    tau_txt = f"τ={tau_days:.1f} d"
            except Exception:
                pass

            if tau_txt:
                _add_tau_label(ax_fit, th_end, rr_end, tau_txt, color_char)
    

            if res_entry['method'].startswith('fit'):
                # Draw a dashed line from the observation to the closest point on the model curve
                point = _get_closest_geometric_point(blob, jet_side, fit_results, params)
                ax_fit.plot([np.deg2rad(blob['pa_obs']), np.deg2rad(point['model_pa'])], 
                            [blob['rad_obs'], point['model_rad']], ls='--', c=color_char, alpha=0.6)
                
                ax_fit.plot(
                    np.deg2rad(point["model_pa"]), point["model_rad"],
                    marker="X",
                    markersize=5,
                    markerfacecolor="white",
                    markeredgecolor=color_char,
                    markeredgewidth=1,
                    linestyle="None",
                    zorder=4,
                    label=None,
                )
            else:
                # No intersection: assume plane of sky (mu=0) and canonical beta (for bookkeeping)
                mu_plane = 0.0
                tau_days = tau_core_to_knot_days_from_projected(float(rr), mu_plane)
                _add_tau_label(ax_fit, th, rr, f"τ={tau_days:.1f} d", color_char)

                ax_fit.plot(
                    th, rr,
                    marker="X",
                    ms=5,
                    mfc="w",
                    mec=color_char,
                    mew=1.0,
                    zorder=4,
                    label=None,
                )

    ax_fit.set_rmax(common_rmax); ax_fit.set_rmin(0)
    # Set radial grid every 0.2 arcsec and label every other tick starting with the second
    r_ticks = np.arange(0, common_rmax + 0.0001, 0.2)
    ax_fit.yaxis.set_major_locator(FixedLocator(r_ticks))
    r_tick_labels = [f"{r:.1f}" if (i % 2 == 0 and i > 0) else "" for i, r in enumerate(r_ticks)]
    ax_fit.set_yticklabels(r_tick_labels, fontsize=14)
    
    # Configure Legend and Layout
    handles = [
        Line2D([0], [0], marker='X', linestyle='None', markersize=7,
               markerfacecolor='white', markeredgecolor='#000', markeredgewidth=1.3, color='#000',
               label='Best-fit knot position'),
        Line2D([0], [0], marker='o', linestyle='None', markersize=6,
               markerfacecolor='#000', markeredgecolor='#000', color='#000',
               label='Observed knot with positional errors'),
        Line2D([0], [0], linestyle='-', color='#000', lw=1.8,
               marker=r'$\rightarrow$', markersize=12, markerfacecolor='white', markeredgecolor='#000',
               label='X-ray travel time (τ) arrow'),
    ]
    legend = ax_fit.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.88),
        frameon=True,
        facecolor='white',
        edgecolor='#999',
        framealpha=0.9,
        fontsize=14,
        labelcolor="#000",
    )
    
    # Leave a narrow but safe margin on the right so the colorbar is very close yet not clipped
    # Slightly lower the axes to add space between the title and the top tick labels
    fig.subplots_adjust(left=0.1, right=0.9, top=0.86, bottom=0.12)
    
    # Add shared colorbar for time/age
    cbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.7])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('Jet Travel Time Since Ejection (Days)', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=14)
    
    if pdf_object: pdf_object.savefig(fig); plt.close(fig)
    else: plt.show()
