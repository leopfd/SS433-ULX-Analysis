#!/usr/bin/env python
"""
3D interactive visualization of SS433 jet structure.
Converts the 2D polar plot (ax2) from plotting.py into an interactive 3D plot.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_PIPELINE = ROOT / "main-pipeline"
sys.path.insert(0, str(MAIN_PIPELINE))
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib import animation
import matplotlib.image as mpimg
from astropy.io import fits

import config
from lib.physics import (
    ss433_phases,
    _age_curve_one_cycle,
    ss433_mu_at_jd,
    fit_and_calculate_jets,
    _get_closest_geometric_point,
)


def _overlay_path_for_obs(obs_id: int):
    return Path(config.BASE_DIR) / f"{obs_id}" / "src_image_square_160pixel.fits"


def _load_overlay_image(path):
    path = Path(path)
    if path.suffix.lower() in {".fits", ".fit"}:
        data = fits.getdata(path, squeeze=True)
        img = np.asarray(data, dtype=float)
    else:
        img = mpimg.imread(path)
        if img.ndim == 3:
            img = img[..., :3]
            img = np.mean(img, axis=2)
    img = np.asarray(img, dtype=float)
    if img.max() > 0:
        img = img / img.max()
    return img


def load_observation(obs_id: int, ephemeris=None):
    """Load blob data from the tracker table CSV."""
    params = dict(config.EPHEMERIS if ephemeris is None else ephemeris)
    params.setdefault("age_window_days", 17.0)
    params.setdefault("age_grid_n", 1200)

    tracker_table_csv = '/Users/leodrake/Documents/ss433/HRC_2024/2Dfits/comp tracker tables/comp-tracker-table-99comp-1sigma-mcmc-200k-bin0p25.csv'
    df = pd.read_csv(tracker_table_csv)
    df_obs = df[df["obs_id"] == obs_id].copy()

    if df_obs.empty:
        raise ValueError(f"No rows found for obs_id={obs_id}")

    mjd_obs = float(df_obs["mjd"].iloc[0])

    # Get core position for reference
    core_row = df_obs[df_obs["component"] == "core"]
    if not core_row.empty:
        core_x = float(core_row["xpos"].iloc[0])
        core_y = float(core_row["ypos"].iloc[0])
    else:
        core_x = core_y = config.CENTER_PIXEL if config.CENTER_PIXEL else 0

    # Convert pixel coordinates to arcsec
    # HRC pixel scale is 0.1317" per pixel, binned to 0.25 scale: 0.13175 * 0.25
    plate_scale = 0.13175 * 0.25  # arcsec per binned pixel
    
    blob_data_list = []
    for _, row in df_obs.iterrows():
        comp = str(row["component"])
        if comp in ("core", "bkg"):
            continue
        
        # Measure offset from core position
        dx_px = row['xpos'] - core_x
        dy_px = row['ypos'] - core_y
        
        # Convert to arcsec
        dx_arcsec = dx_px * plate_scale
        dy_arcsec = dy_px * plate_scale
        
        # Use same PA convention as track_components.py: arctan2(-x, y)
        radius = np.sqrt(dx_arcsec**2 + dy_arcsec**2)
        pa = np.rad2deg(np.arctan2(-dx_arcsec, dy_arcsec)) % 360
        
        # Errors
        radius_err_plus = row.get('plus_err', 0) * plate_scale
        radius_err_minus = row.get('minus_err', 0) * plate_scale
        
        # Estimate PA errors from radius errors
        if radius > 1e-6:
            pa_err = np.rad2deg(radius_err_plus / radius)  # Small angle approximation
        else:
            pa_err = 1.0
        
        blob_data_list.append({
            "comp": comp,
            "mjd_obs": mjd_obs,
            "pa_obs": pa,
            "rad_obs": radius,
            "rad_err_L": -radius_err_minus,
            "rad_err_U": radius_err_plus,
            "pa_err_L": -pa_err,
            "pa_err_U": pa_err,
        })

    return blob_data_list, params, mjd_obs


def _rad_from_mu(age_days, mu_ra, mu_dec):
    """Convert proper motion and age to projected radius in arcseconds."""
    return (
        np.sqrt(mu_ra**2 + mu_dec**2)
        * config.C_PC_PER_DAY
        * age_days
        / config.D_SS433_PC
    ) * config.ARCSEC_PER_RADIAN


def _xyz_from_polar_and_mu(pa_deg, r_proj, mu_los):
    """
    Convert polar coordinates (PA, radius) and line-of-sight mu to 3D Cartesian.
    
    pa_deg: position angle in degrees (using arctan2(-x, y) convention)
    r_proj: projected radius (arcsec)
    mu_los: cos(theta_LOS) - direction cosine along LOS
    
    Returns (x, y, z) in arcsec
    """
    pa_rad = np.deg2rad(pa_deg)
    
    # Sky plane coordinates (matching arctan2(-x, y) convention: x = -r*sin(PA), y = r*cos(PA))
    x_old = -r_proj * np.sin(pa_rad)
    y_old = r_proj * np.cos(pa_rad)
    
    # Line-of-sight depth scaled consistently
    sin_thlos = np.sqrt(np.maximum(1e-12, 1.0 - mu_los**2))
    z_old = -r_proj * mu_los / sin_thlos

    # Rotate coordinates so LOS is +y: (x, z_old, y_old) and mirror sky vertical so negative z becomes positive
    x = x_old
    y = z_old         # LOS depth now on +y
    z = y_old         # sky vertical axis mirrored
    
    return x, y, z


def create_3d_scene(
    blob_data_list,
    params,
    mjd_obs,
    fit_results=None,
    betas=None,
    age_frac_period=2.0,
    use_fitted_betas=True,
):
    """Build 3D jet trajectories and blob positions."""
    jd_obs = mjd_obs + 2400000.5
    
    # Get age grid
    base_n = int(params.get("age_grid_n", 1200))
    n_grid = int(max(10, base_n * (age_frac_period / 2.0)))
    age = _age_curve_one_cycle(params, n=n_grid, frac_period=age_frac_period)
    jd_ej = jd_obs - age
    
    # Use fitted betas when available, otherwise fall back to config default
    if betas is None and fit_results and use_fitted_betas:
        betas = fit_results.get("fitted_betas")
    if betas:
        beta_e = float(betas.get('east', params['beta']))
        beta_w = float(betas.get('west', params['beta']))
    else:
        beta_e = beta_w = float(params['beta'])
    
    # Get kinematics for all ages
    mu_e_ra, mu_e_dec, mu_w_ra, mu_w_dec, pa_e, pa_w = ss433_phases(
        jd_ej, params, beta_east=beta_e, beta_west=beta_w
    )
    
    # Convert to projected radius
    r_e = _rad_from_mu(age, mu_e_ra, mu_e_dec)
    r_w = _rad_from_mu(age, mu_w_ra, mu_w_dec)
    
    # Get LOS mu from ephemeris (honor full/complex params)
    mu_e_los, mu_w_los = ss433_mu_at_jd(jd_ej, params)
    
    # Convert to 3D
    xe, ye, ze = _xyz_from_polar_and_mu(pa_e, r_e, mu_e_los)
    xw, yw, zw = _xyz_from_polar_and_mu(pa_w, r_w, mu_w_los)
    
    # Position blobs on their intersection with the 3D jet trajectory
    blobs_xyz = []
    blobs_colors = []
    
    for blob in blob_data_list:
        is_east = blob['comp'].startswith('east')
        color = 'blue' if is_east else 'red'
        jet_side = 'east' if is_east else 'west'
        
        # Get the jet trajectory for this side
        if is_east:
            jet_pa = pa_e
            jet_r = r_e
            mu_los_curve = mu_e_los
        else:
            jet_pa = pa_w
            jet_r = r_w
            mu_los_curve = mu_w_los
        
        # Find where blob's observed PA and radius intersect the jet trajectory
        # by finding the age that best matches the observed position
        pa_obs = blob['pa_obs']
        r_obs = blob['rad_obs']
        
        # Calculate angular distance between observed and model PA at each age
        dpa = np.abs((jet_pa - pa_obs + 180) % 360 - 180)  # Wrapped angular difference
        dr = np.abs(jet_r - r_obs)
        
        # Find age that best matches observed position (minimize combined distance)
        # Weight PA and radius equally
        distance = np.sqrt((dpa**2) + (dr**2))
        closest_age_idx = np.argmin(distance)
        
        # Get 3D position at this age
        pa_at_age = jet_pa[closest_age_idx]
        r_at_age = jet_r[closest_age_idx]
        mu_los_at_age = mu_los_curve[closest_age_idx]
        
        # Convert to 3D
        xb, yb, zb = _xyz_from_polar_and_mu(pa_at_age, r_at_age, mu_los_at_age)
        blobs_xyz.append([xb, yb, zb])
        blobs_colors.append(color)
        
        if False:  # Fallback code (disabled)
            pass  # Old fallback approach not needed
    
    blobs_xyz = np.array(blobs_xyz) if blobs_xyz else np.zeros((0, 3))
    
    return {
        'age': age,
        'east_xyz': (xe, ye, ze),
        'west_xyz': (xw, yw, zw),
        'blobs_xyz': blobs_xyz,
        'blobs_colors': blobs_colors,
        'mjd_obs': mjd_obs,
        'betas': (beta_e, beta_w),
    }


def plot_3d_scene(
    scene,
    interactive=True,
    show_blobs=True,
    show_light_arrows=True,
    max_age_index=None,
    return_artists=False,
    axis_limit_override=None,
    invert_xaxis=True,
    invert_zaxis=False,
    clip_radius=None,
    overlay_image=None,
    overlay_scale_arcsec=None,
    overlay_alpha=0.4,
):
    """Create interactive 3D plot matching ax2 style."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    age = scene['age']
    xe, ye, ze = scene['east_xyz']
    xw, yw, zw = scene['west_xyz']
    blobs_xyz = scene['blobs_xyz']
    blobs_colors = scene['blobs_colors']
    mjd_obs = scene['mjd_obs']
    beta_e, beta_w = scene['betas']

    # Limit how many time steps to show (used for time-lapse animation)
    limit = None if max_age_index is None else max(1, min(len(age), max_age_index))
    age_plot = age if limit is None else age[:limit]
    xe_plot, ye_plot, ze_plot = [
        arr if limit is None else arr[:limit] for arr in (xe, ye, ze)
    ]
    xw_plot, yw_plot, zw_plot = [
        arr if limit is None else arr[:limit] for arr in (xw, yw, zw)
    ]
    
    # Color map for age (matching ax2)
    cmap = plt.cm.rainbow
    norm = mcolors.Normalize(vmin=0, vmax=max(225, np.max(age)))

    # Equal aspect ratio precompute
    all_coords = np.concatenate([xe, ye, ze, xw, yw, zw])
    if show_blobs and len(blobs_xyz) > 0:
        all_coords = np.concatenate([all_coords, blobs_xyz.flatten()])
    r_max = np.max(np.abs(all_coords))
    r_max = max(r_max, 1.0)
    padding = 0.2 * r_max
    axis_limit = r_max + padding
    if axis_limit_override is not None:
        axis_limit = float(axis_limit_override)

    # Offset jets so they emerge from a small sphere
    sphere_r = min(0.2, 0.05 * axis_limit)

    def _offset_from_sphere(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        u = np.zeros((3,) + r.shape, dtype=float)
        mask = r > 1e-8
        u[0][mask] = x[mask] / r[mask]
        u[1][mask] = y[mask] / r[mask]
        u[2][mask] = z[mask] / r[mask]
        new_r = sphere_r + r  # always start at sphere surface
        return u[0] * new_r, u[1] * new_r, u[2] * new_r

    xe_plot, ye_plot, ze_plot = _offset_from_sphere(xe_plot, ye_plot, ze_plot)
    xw_plot, yw_plot, zw_plot = _offset_from_sphere(xw_plot, yw_plot, zw_plot)
    blobs_xyz_offset = blobs_xyz.copy()
    if blobs_xyz_offset.size > 0:
        bx, by, bz = blobs_xyz_offset[:, 0], blobs_xyz_offset[:, 1], blobs_xyz_offset[:, 2]
        bx, by, bz = _offset_from_sphere(bx, by, bz)
        blobs_xyz_offset[:, 0] = bx
        blobs_xyz_offset[:, 1] = by
        blobs_xyz_offset[:, 2] = bz

    def _clip_xyz_c(x, y, z, c):
        if clip_radius is None:
            return x, y, z, c
        r = np.sqrt(x**2 + y**2 + z**2)
        mask = r <= clip_radius
        return x[mask], y[mask], z[mask], c[mask]

    xe_plot, ye_plot, ze_plot, age_e = _clip_xyz_c(xe_plot, ye_plot, ze_plot, age_plot)
    xw_plot, yw_plot, zw_plot, age_w = _clip_xyz_c(xw_plot, yw_plot, zw_plot, age_plot)

    # Plot jet trajectories colored by age
    scatter_e = ax.scatter(xe_plot, ye_plot, ze_plot, c=age_e, cmap=cmap, norm=norm, s=10, alpha=0.8, zorder=1)
    scatter_w = ax.scatter(xw_plot, yw_plot, zw_plot, c=age_w, cmap=cmap, norm=norm, s=10, alpha=0.8, zorder=1)
    
    # Add light travel time arrows from source (core) to each blob
    if show_light_arrows:
        for xyz, color in zip(blobs_xyz_offset, blobs_colors):
            arrow_color = 'blue' if color == 'blue' else 'red'
            ax.quiver(0, 0, 0, xyz[0], xyz[1], xyz[2], 
                     color=arrow_color, alpha=0.5, arrow_length_ratio=0.2, linewidth=2)
    
    # Plot observed blobs (higher zorder to appear on top of jets)
    if show_blobs and len(blobs_xyz_offset) > 0:
        for i, (xyz, color) in enumerate(zip(blobs_xyz_offset, blobs_colors)):
            marker_color = 'blue' if color == 'blue' else 'red'
            ax.scatter(*xyz, c=marker_color, s=100, marker='o', edgecolors='black', 
                      linewidths=2, zorder=10, label='Jet knot' if i == 0 else '')
    
    # Labels and title
    ax.set_xlabel('x (arcsec)', fontsize=11)
    ax.set_ylabel('y (arcsec)', fontsize=11)
    ax.set_zlabel('z (arcsec)', fontsize=11)
    ax.set_title(f'3D Jet Structure (mjd {mjd_obs:.1f} | β_E={beta_e:.4f} β_W={beta_w:.4f})',
                fontsize=12, pad=20)

    # Hide box panes; keep axes lines
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
    ax.grid(False)
    
    # Equal aspect ratio
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    # Mirror axes to match desired handedness
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_zaxis:
        ax.invert_zaxis()

    # Plot central source as a small sphere
    sphere_r = min(0.2, 0.05 * axis_limit)
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    xs = sphere_r * np.outer(np.cos(u), np.sin(v))
    ys = sphere_r * np.outer(np.sin(u), np.sin(v))
    zs = sphere_r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='black', edgecolor='none', alpha=1.0, zorder=9)

    # Optional overlay image on -y plane
    if overlay_image is not None and overlay_scale_arcsec:
        try:
            img = np.asarray(overlay_image, dtype=float)
            if img.ndim == 3:
                img = img[..., :3]
                img = np.mean(img, axis=2)
            img_min = img.min()
            img = img - img_min
            img = np.log10(img + 1e-6)
            img = img - img.min()
            finite = img[np.isfinite(img)]
            if finite.size > 0:
                lo, hi = np.percentile(finite, [2, 98])
                scale = max(hi - lo, 1e-6)
                img = (img - lo) / scale
                img = np.clip(img, 0.0, 1.0)
            else:
                if img.max() > 0:
                    img = img / img.max()
            img = 1.0 - img  # invert so high counts are dark
            alpha_map = overlay_alpha * (img < 0.999)
            h, w = img.shape
            half_w = 0.5 * w * overlay_scale_arcsec
            half_h = 0.5 * h * overlay_scale_arcsec
            x_extent = np.linspace(-half_w, half_w, w)
            z_extent = np.linspace(-half_h, half_h, h)
            Xg, Zg = np.meshgrid(x_extent, z_extent)
            Yg = np.full_like(Xg, -axis_limit)
            rgba = plt.cm.gray(img)
            rgba[..., -1] = alpha_map
            ax.plot_surface(
                Xg,
                Yg,
                Zg,
                rstride=1,
                cstride=1,
                facecolors=rgba,
                shade=False,
                linewidth=0,
                antialiased=False,
                zorder=0,
            )
        except Exception:
            pass

    # Observer on +y side at z=0 (center of z-plane)
    observer_x, observer_y, observer_z = 0.0, axis_limit, 0.0
    observer_handle = ax.scatter(
        [observer_x],
        [observer_y],
        [observer_z],
        c='black',
        marker='^',
        s=120,
        edgecolors='white',
        linewidths=0.8,
        zorder=12,
        label='',
    )

    # Add arrows from blobs to observer (after limits/observer placement)
    observer_arrow_handles = []
    if show_light_arrows and len(blobs_xyz_offset) > 0:
        obs_vec = np.array([observer_x, observer_y, observer_z])
        for xyz, color in zip(blobs_xyz_offset, blobs_colors):
            vec = obs_vec - np.array(xyz)
            arrow_color = 'blue' if color == 'blue' else 'red'
            h = ax.quiver(
                xyz[0], xyz[1], xyz[2],
                vec[0], vec[1], vec[2],
                color=arrow_color,
                alpha=0.35,
                arrow_length_ratio=0.1,
                linewidth=1.5,
            )
            observer_arrow_handles.append(h)
    
    # Colorbar
    cbar = plt.colorbar(scatter_e, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Age (days)', fontsize=10)
    
    # Legend
    handles, labels = [], []
    # Source
    handles.append(Line2D([0], [0], marker='*', color='gold', markeredgecolor='black', markeredgewidth=0.5, markersize=12, linestyle='None'))
    labels.append('Source')
    # Jet knot
    if show_blobs and len(blobs_xyz) > 0:
        handles.append(Line2D([0], [0], marker='o', color='gray', markeredgecolor='black', markeredgewidth=1.5, markersize=8, linestyle='None'))
        labels.append('Jet knot')
    # Light path
    if show_light_arrows and len(blobs_xyz) > 0:
        arrow_handle = Line2D([0], [0], color='gray', lw=2, marker='>', markersize=8)
        handles.append(arrow_handle)
        labels.append('Light travel path')
    # Observer
    handles.append(Line2D([0], [0], marker='^', color='black', markeredgecolor='white', markeredgewidth=0.8, markersize=10, linestyle='None'))
    labels.append('Observer')
    ax.legend(handles, labels, loc='upper right', fontsize=10)
    
    # Set view: looking from +y toward origin at z=0
    ax.view_init(elev=0, azim=90)
    
    if return_artists:
        fig.tight_layout()
        return fig, ax, scatter_e, scatter_w
    if interactive:
        plt.tight_layout()
        plt.show()
    else:
        fig.tight_layout()
        return fig, ax


def save_scene(scene, out_path, seconds, fps, mode="rotate", overlay_image=None, overlay_scale_arcsec=None, overlay_alpha=0.4):
    """
    Save either a static image or an animation depending on extension and mode.
    mode: 'rotate' spins the camera; 'time' grows the jets over time.
    """
    out_path = Path(out_path).expanduser()
    ext = out_path.suffix.lower()
    is_anim = ext in {".mp4", ".gif"}
    mode = mode.lower()
    if mode not in {"rotate", "time"}:
        raise ValueError("mode must be 'rotate' or 'time'")

    show_blobs = show_light_arrows = mode != "time"

    if not is_anim:
        fig, ax = plot_3d_scene(
            scene,
            interactive=False,
            show_blobs=show_blobs,
            show_light_arrows=show_light_arrows,
            overlay_image=overlay_image,
            overlay_scale_arcsec=overlay_scale_arcsec,
            overlay_alpha=overlay_alpha,
        )
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        return out_path

    if mode == "rotate":
        # Build a slightly tilted view for 3D rotation
        fig, ax = plot_3d_scene(
            scene,
            interactive=False,
            show_blobs=show_blobs,
            show_light_arrows=show_light_arrows,
            invert_zaxis=False,
            overlay_image=overlay_image,
            overlay_scale_arcsec=overlay_scale_arcsec,
            overlay_alpha=overlay_alpha,
        )
        start_elev, start_azim = 0, 90  # view from +y toward origin
        base_elev, base_azim = 35, 70  # slightly higher and diagonal
        ax.view_init(elev=start_elev, azim=start_azim)

        frames = max(2, int(seconds * fps))
        progress_step = max(1, frames // 20)
        hold_frames = min(frames - 1, int(fps * 5))  # ~5s hold on initial view
        move_frames = max(2, frames - hold_frames)

        def update(i):
            if i % progress_step == 0 or i == frames - 1:
                print(f"Rendering frame {i+1}/{frames}", end="\r")
            if i < hold_frames:
                elev = start_elev
                azim = start_azim
            else:
                idx = i - hold_frames
                phase = idx / (move_frames - 1)  # 0 -> 1 after hold
                ramp_in = 0.08
                ramp_out = 0.08
                spin_frac = max(1e-6, 1.0 - ramp_in - ramp_out)
                spin_phase = np.clip((phase - ramp_in) / spin_frac, 0.0, 1.0)
                spin_deg = 540.0

                if phase < ramp_in:
                    t = phase / max(ramp_in, 1e-6)
                    elev = start_elev + (base_elev - start_elev) * t
                    azim = start_azim + (base_azim - start_azim) * t
                elif phase > 1.0 - ramp_out:
                    t = (phase - (1.0 - ramp_out)) / max(ramp_out, 1e-6)
                    azim_spin = base_azim + spin_deg
                    elev_spin = base_elev  # wobble ends at zero amplitude at spin_phase=1
                    elev = elev_spin + (start_elev - elev_spin) * t
                    azim = azim_spin + (start_azim - azim_spin) * t
                else:
                    azim = base_azim + spin_deg * spin_phase + 15.0 * np.sin(2 * np.pi * spin_phase)
                    elev = base_elev + 12.0 * np.sin(4 * np.pi * spin_phase)
            ax.view_init(elev=elev, azim=azim)
            return ax,
    else:
        # Animate jets growing with time; hide observation markers and arrows.
        fig, ax, scatter_e, scatter_w = plot_3d_scene(
            scene,
            interactive=False,
            show_blobs=False,
            show_light_arrows=False,
            return_artists=True,
            axis_limit_override=5.0,
            clip_radius=5.0,
            invert_zaxis=False,
            overlay_image=overlay_image,
            overlay_scale_arcsec=overlay_scale_arcsec,
            overlay_alpha=overlay_alpha,
        )
        base_elev, base_azim = 30, 90
        spin_deg = 360.0  # full rotation over the animation
        wobble_deg = 5.0
        travel_overshoot = 1  # let knots run past the precomputed length
        ax.view_init(elev=base_elev, azim=base_azim)

        age = scene["age"]
        xe, ye, ze = scene["east_xyz"]
        xw, yw, zw = scene["west_xyz"]
        sphere_r_time = min(0.2, 0.05 * 5.0)

        def _offset_from_sphere_arr(x, y, z, r):
            rad = np.sqrt(x**2 + y**2 + z**2)
            rad = np.nan_to_num(rad, nan=0.0, posinf=0.0, neginf=0.0)
            u = np.zeros((3,) + rad.shape, dtype=float)
            mask = rad > 1e-8
            u[0][mask] = x[mask] / rad[mask]
            u[1][mask] = y[mask] / rad[mask]
            u[2][mask] = z[mask] / rad[mask]
            new_r = r + rad
            return u[0] * new_r, u[1] * new_r, u[2] * new_r

        xe, ye, ze = _offset_from_sphere_arr(xe, ye, ze, sphere_r_time)
        xw, yw, zw = _offset_from_sphere_arr(xw, yw, zw, sphere_r_time)
        frames = max(2, int(seconds * fps))
        progress_step = max(1, frames // 20)
        max_age = float(np.max(age))
        # Ejection timeline: earliest knot (largest age) starts at t=0; youngest at t=max_age.
        eject_time = max_age - age

        # Jet direction arrows (at core, rotate with precession)
        arrow_len = 0.7

        def _unit_vec_at(idx, x, y, z):
            vec = np.array([x[idx], y[idx], z[idx]], dtype=float)
            n = np.linalg.norm(vec)
            if n < 1e-6 and idx + 1 < len(x):
                vec = np.array([x[idx + 1], y[idx + 1], z[idx + 1]], dtype=float)
                n = np.linalg.norm(vec)
            if n < 1e-6:
                return np.array([1.0, 0.0, 0.0])
            return vec / n

        dir_e0 = _unit_vec_at(1, xe, ye, ze)
        dir_w0 = _unit_vec_at(1, xw, yw, zw)
        line_e = ax.quiver(
            0,
            0,
            0,
            dir_e0[0] * arrow_len,
            dir_e0[1] * arrow_len,
            dir_e0[2] * arrow_len,
            color='blue',
            alpha=0.7,
            linewidth=1.2,
            arrow_length_ratio=0.2,
            zorder=5,
        )
        line_w = ax.quiver(
            0,
            0,
            0,
            dir_w0[0] * arrow_len,
            dir_w0[1] * arrow_len,
            dir_w0[2] * arrow_len,
            color='red',
            alpha=0.7,
            linewidth=1.2,
            arrow_length_ratio=0.2,
            zorder=5,
        )

        def _positions_at(t_now, x, y, z):
            active = t_now >= eject_time
            if not np.any(active):
                return np.array([]), np.array([]), np.array([]), np.array([])
            age_active = age[active]
            frac = (t_now - eject_time[active]) / np.maximum(age_active, 1e-6)
            frac = np.clip(frac, 0.0, travel_overshoot)
            colors = np.minimum(frac, 1.0) * age_active  # cap color at age grid
            return x[active] * frac, y[active] * frac, z[active] * frac, colors

        # Start animation in a ballistic state instead of the static full swirl.
        xe0, ye0, ze0, ce0 = _positions_at(0.0, xe, ye, ze)
        xw0, yw0, zw0, cw0 = _positions_at(0.0, xw, yw, zw)
        scatter_e._offsets3d = (xe0, ye0, ze0)
        scatter_w._offsets3d = (xw0, yw0, zw0)
        scatter_e.set_array(ce0)
        scatter_w.set_array(cw0)

        def update(i):
            if i % progress_step == 0 or i == frames - 1:
                print(f"Rendering frame {i+1}/{frames}", end="\r")
            t_now = (i / (frames - 1)) * max_age * travel_overshoot
            xe_i, ye_i, ze_i, c_e = _positions_at(t_now, xe, ye, ze)
            xw_i, yw_i, zw_i, c_w = _positions_at(t_now, xw, yw, zw)
            scatter_e._offsets3d = (xe_i, ye_i, ze_i)
            scatter_w._offsets3d = (xw_i, yw_i, zw_i)
            scatter_e.set_array(c_e)
            scatter_w.set_array(c_w)
            spin_phase = i / max(frames - 1, 1)
            azim = base_azim + spin_deg * spin_phase
            elev = base_elev + wobble_deg * np.sin(2 * np.pi * spin_phase)
            ax.view_init(elev=elev, azim=azim)
            # Update jet direction arrows to current precession phase
            idx_dir = min(int((t_now / max(max_age, 1e-6)) * (len(age) - 1)), len(age) - 1)
            dir_e = _unit_vec_at(idx_dir, xe, ye, ze)
            dir_w = _unit_vec_at(idx_dir, xw, yw, zw)
            nonlocal line_e, line_w
            line_e.remove()
            line_w.remove()
            line_e = ax.quiver(
                0, 0, 0,
                dir_e[0] * arrow_len,
                dir_e[1] * arrow_len,
                dir_e[2] * arrow_len,
                color='blue',
                alpha=0.7,
                linewidth=1.2,
                arrow_length_ratio=0.2,
                zorder=5,
            )
            line_w = ax.quiver(
                0, 0, 0,
                dir_w[0] * arrow_len,
                dir_w[1] * arrow_len,
                dir_w[2] * arrow_len,
                color='red',
                alpha=0.7,
                linewidth=1.2,
                arrow_length_ratio=0.2,
                zorder=5,
            )
            return scatter_e, scatter_w, line_e, line_w

    writer = None
    target_path = out_path
    if ext == ".mp4":
        if animation.writers.is_available("ffmpeg"):
            writer = animation.FFMpegWriter(fps=fps)
        elif animation.writers.is_available("pillow"):
            # Fallback: write GIF instead of failing outright
            target_path = out_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=fps)
            print(f"ffmpeg not available; saving GIF instead: {target_path}")
        else:
            raise RuntimeError("ffmpeg is required for mp4 animations (or Pillow for gif fallback).")
    else:
        if not animation.writers.is_available("pillow"):
            raise RuntimeError("pillow writer is required to write gif animations.")
        writer = animation.PillowWriter(fps=fps)

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(target_path, writer=writer, dpi=150)
    return target_path


def main():
    ap = argparse.ArgumentParser(description='3D interactive jet visualization')
    ap.add_argument('--obs-id', required=True, type=int, help='Observation ID')
    ap.add_argument('--out', default=None, help='Output file (png, gif, mp4)')
    ap.add_argument('--seconds', type=float, default=8, help='Animation duration')
    ap.add_argument('--fps', type=int, default=30, help='Animation FPS')
    ap.add_argument(
        '--mode',
        choices=['rotate', 'time'],
        default='rotate',
        help="Animation style: 'rotate' spins the camera, 'time' shows ballistic knots building the swirl (no obs/arrows).",
    )
    args = ap.parse_args()
    
    # Load data
    ephem_for_mode = (
        dict(config.EPHEM_FULL)
        if args.mode == "time"
        else dict(config.EPHEM_SIMPLE if config.EPHEMERIS == {} else config.EPHEMERIS)
    )
    # Keep global in sync for any downstream calls that read config.EPHEMERIS
    config.EPHEMERIS = ephem_for_mode
    blob_data_list, params, mjd_obs = load_observation(args.obs_id, ephemeris=ephem_for_mode)

    # Load overlay image from fixed path per obs (bin 0.25)
    overlay_image = None
    overlay_scale = 0.13175 * 0.25
    overlay_alpha = 0.4
    if args.mode == "rotate":
        overlay_path = _overlay_path_for_obs(args.obs_id)
        try:
            if overlay_path.exists():
                overlay_image = _load_overlay_image(overlay_path)
            else:
                print(f"Warning: overlay image not found at {overlay_path}")
        except Exception as e:
            print(f"Warning: could not load overlay image at {overlay_path}: {e}")
    
    # Fit and calculate jets
    fit_results = fit_and_calculate_jets(blob_data_list, params)
    
    # Create scene using fitted results
    age_frac_period = 10.0 if args.mode == "time" else 2.0
    scene = create_3d_scene(
        blob_data_list,
        params,
        mjd_obs,
        fit_results=fit_results,
        age_frac_period=age_frac_period,
        use_fitted_betas=(args.mode != "time"),
    )
    
    # Plot
    if args.out:
        saved_to = save_scene(
            scene,
            args.out,
            args.seconds,
            args.fps,
            mode=args.mode,
            overlay_image=overlay_image if args.mode == "rotate" else None,
            overlay_scale_arcsec=overlay_scale,
            overlay_alpha=overlay_alpha,
        )
        print(f"Saved to {saved_to}")
    else:
        plot_3d_scene(
            scene,
            interactive=True,
            overlay_image=overlay_image if args.mode == "rotate" else None,
            overlay_scale_arcsec=overlay_scale,
            overlay_alpha=overlay_alpha,
        )


if __name__ == '__main__':
    main()
