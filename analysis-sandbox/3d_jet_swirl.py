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

import config
from lib.physics import (
    ss433_phases,
    _age_curve_one_cycle,
    ss433_mu_from_config_ephemeris,
    fit_and_calculate_jets,
    _get_closest_geometric_point,
)


def load_observation(obs_id: int):
    """Load blob data from the tracker table CSV."""
    params = dict(config.EPHEMERIS)
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
    x = -r_proj * np.sin(pa_rad)
    y = r_proj * np.cos(pa_rad)
    
    # Line-of-sight depth scaled consistently
    sin_thlos = np.sqrt(np.maximum(1e-12, 1.0 - mu_los**2))
    z = r_proj * mu_los / sin_thlos
    
    return x, y, z


def create_3d_scene(blob_data_list, params, mjd_obs, fit_results=None, betas=None):
    """Build 3D jet trajectories and blob positions."""
    jd_obs = mjd_obs + 2400000.5
    
    # Get age grid
    age = _age_curve_one_cycle(params)
    jd_ej = jd_obs - age
    
    # Use fitted betas when available, otherwise fall back to config default
    if betas is None and fit_results:
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
    
    # Get LOS mu from ephemeris
    mu_e_los, mu_w_los = ss433_mu_from_config_ephemeris(jd_ej)
    
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


def plot_3d_scene(scene, interactive=True):
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
    
    # Color map for age (matching ax2)
    cmap = plt.cm.rainbow
    norm = mcolors.Normalize(vmin=0, vmax=225)
    
    # Plot jet trajectories colored by age
    scatter_e = ax.scatter(xe, ye, ze, c=age, cmap=cmap, norm=norm, s=15, alpha=0.8, zorder=1)
    scatter_w = ax.scatter(xw, yw, zw, c=age, cmap=cmap, norm=norm, s=15, alpha=0.8, zorder=1)
    
    # Add light travel time arrows from source (core) to each blob
    for xyz, color in zip(blobs_xyz, blobs_colors):
        arrow_color = 'blue' if color == 'blue' else 'red'
        ax.quiver(0, 0, 0, xyz[0], xyz[1], xyz[2], 
                 color=arrow_color, alpha=0.5, arrow_length_ratio=0.2, linewidth=2)
    
    # Plot observed blobs (higher zorder to appear on top of jets)
    if len(blobs_xyz) > 0:
        for i, (xyz, color) in enumerate(zip(blobs_xyz, blobs_colors)):
            marker_color = 'blue' if color == 'blue' else 'red'
            ax.scatter(*xyz, c=marker_color, s=100, marker='o', edgecolors='black', 
                      linewidths=2, zorder=10, label='Obs' if i == 0 else '')
    
    # Plot central source
    ax.scatter([0], [0], [0], c='gold', marker='*', s=300, edgecolors='black', 
              linewidths=0.5, zorder=10, label='Source')
    
    # Labels and title
    ax.set_xlabel('x (arcsec)', fontsize=11)
    ax.set_ylabel('y (arcsec)', fontsize=11)
    ax.set_zlabel('z (arcsec)', fontsize=11)
    ax.set_title(f'3D Jet Structure (mjd {mjd_obs:.1f} | β_E={beta_e:.4f} β_W={beta_w:.4f})',
                fontsize=12, pad=20)
    
    # Equal aspect ratio
    all_coords = np.concatenate([xe, ye, ze, xw, yw, zw])
    if len(blobs_xyz) > 0:
        all_coords = np.concatenate([all_coords, blobs_xyz.flatten()])
    
    r_max = np.max(np.abs(all_coords))
    r_max = max(r_max, 1.0)
    padding = 0.2 * r_max
    axis_limit = r_max + padding
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    
    # Colorbar
    cbar = plt.colorbar(scatter_e, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Age (days)', fontsize=10)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    arrow_handle = Line2D([0], [0], color='gray', lw=2, marker='>', markersize=8)
    handles.append(arrow_handle)
    labels.append('Light-travel arrow')
    ax.legend(handles, labels, loc='upper right', fontsize=10)
    
    # Set view: looking straight down at the sky plane (as in polar plot)
    ax.view_init(elev=90, azim=270)
    
    if interactive:
        plt.tight_layout()
        plt.show()
    else:
        fig.tight_layout()
        return fig, ax


def save_scene(scene, out_path, seconds, fps):
    """
    Save either a static image or a rotating animation depending on extension.
    """
    out_path = Path(out_path).expanduser()
    ext = out_path.suffix.lower()
    is_anim = ext in {".mp4", ".gif"}

    if not is_anim:
        fig, ax = plot_3d_scene(scene, interactive=False)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        return out_path

    # Build a slightly tilted view for 3D rotation
    fig, ax = plot_3d_scene(scene, interactive=False)
    base_elev, base_azim = 30, 270
    ax.view_init(elev=base_elev, azim=base_azim)

    frames = max(1, int(seconds * fps))

    def update(i):
        phase = i / frames
        azim = base_azim + 540.0 * phase + 15.0 * np.sin(2 * np.pi * phase)
        elev = base_elev + 12.0 * np.sin(4 * np.pi * phase)
        ax.view_init(elev=elev, azim=azim)
        return ax,

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
    args = ap.parse_args()
    
    # Load data
    blob_data_list, params, mjd_obs = load_observation(args.obs_id)
    
    # Fit and calculate jets
    fit_results = fit_and_calculate_jets(blob_data_list, params)
    
    # Create scene using fitted results
    scene = create_3d_scene(blob_data_list, params, mjd_obs, fit_results=fit_results)
    
    # Plot
    if args.out:
        saved_to = save_scene(scene, args.out, args.seconds, args.fps)
        print(f"Saved to {saved_to}")
    else:
        plot_3d_scene(scene, interactive=True)


if __name__ == '__main__':
    main()
