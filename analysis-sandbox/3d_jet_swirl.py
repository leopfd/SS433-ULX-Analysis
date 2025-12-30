# analysis_sandbox/ax2_flyout.py
"""
Animate an existing Matplotlib *polar* axis (ax2) so it starts as a flat 2D plot
and then rotates into 3D.

Two modes are supported:

1) mode="texture"  (recommended)
   - Renders ax2 to an RGBA image and maps it onto a 3D plane.
   - This preserves *everything* drawn on ax2: scatter, fills, arrows, text, etc.
   - No need to "extract" individual artist data.

2) mode="data"
   - Extracts only what can be cleanly recovered as (theta, r) from common artists
     (Line2D and PathCollection offsets).
   - Good if you want true 3D geometry, but it will NOT preserve every detail.

Usage:
    from analysis_sandbox.ax2_flyout import flyout_3d

    # ax2 is the polar axes object that already has your plot on it
    anim = flyout_3d(ax2, mode="texture")   # shows animation

Optional saving:
    flyout_3d(ax2, mode="texture", out="ax2_flyout.mp4", fps=30)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def _render_ax_to_rgba(ax, dpi: int = 200):
    """
    Render a single axes region to an RGBA image array.

    This is the cleanest way to "get everything from ax2":
    you let Matplotlib draw it, then you capture the pixels.
    """
    fig = ax.figure
    fig.canvas.draw()

    # Tight bounding box around *this axes* in display pixels
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    w_px = int(np.ceil(bbox.width * dpi))
    h_px = int(np.ceil(bbox.height * dpi))

    # Create a temp figure sized exactly to the axes bbox
    tmp = plt.figure(figsize=(bbox.width, bbox.height), dpi=dpi)
    tmp_ax = tmp.add_axes([0, 0, 1, 1])

    # Copy the *visual* content by drawing the original fig and cropping:
    # easiest + most faithful is to render the full original fig then crop.
    fig.canvas.draw()
    rgba_full = np.asarray(fig.canvas.buffer_rgba())

    # Convert bbox (in inches at fig.dpi) to pixel coordinates in the full canvas
    fig_w, fig_h = fig.canvas.get_width_height()
    bbox = ax.get_window_extent()

    x0 = int(np.floor(bbox.x0))
    x1 = int(np.ceil(bbox.x1))
    y0 = int(np.floor(bbox.y0))
    y1 = int(np.ceil(bbox.y1))

    # Convert from display coords (origin bottom-left)
    # to image coords (origin top-left)
    fig_w, fig_h = fig.canvas.get_width_height()
    y0_img = fig_h - y1
    y1_img = fig_h - y0

    rgba = rgba_full[y0_img:y1_img, x0:x1, :].copy()

    plt.close(tmp)
    return rgba


def _extract_theta_r_from_ax(ax):
    """
    Best-effort extraction of (theta, r) data from common artists.

    This will typically recover:
      - Line2D: ax.lines (get_data -> theta, r)
      - Scatter: PathCollection with get_offsets -> theta, r

    It will NOT reliably recover:
      - filled polygons with transforms
      - arrows/annotations
      - text
      - color-mapped fills

    If you want everything, use mode="texture".
    """
    items = []

    # Lines
    for ln in ax.lines:
        try:
            th, rr = ln.get_data()
            th = np.asarray(th, dtype=float)
            rr = np.asarray(rr, dtype=float)
            if th.size and rr.size:
                items.append(("line", th, rr, ln))
        except Exception:
            continue

    # Collections (scatter, etc.)
    for coll in ax.collections:
        # PathCollection (scatter) usually provides offsets in data coords
        if hasattr(coll, "get_offsets"):
            off = coll.get_offsets()
            if off is None:
                continue
            off = np.asarray(off)
            if off.ndim == 2 and off.shape[1] >= 2 and off.shape[0] > 0:
                th = off[:, 0].astype(float)
                rr = off[:, 1].astype(float)
                items.append(("scatter", th, rr, coll))

    return items


def flyout_3d(
    ax2,
    mode: str = "texture",
    out: str | None = None,
    fps: int = 30,
    frames: int = 180,
    elev_start: float = 90.0,
    elev_end: float = 20.0,
    azim_start: float = -60.0,
    azim_spin: float = 180.0,
    plane_scale: float = 1.0,
):
    """
    Create and optionally save a 3D flyout animation from an existing polar axes (ax2).

    Parameters
    ----------
    ax2 : matplotlib.axes._axes.Axes
        A POLAR axes that already has your plot drawn on it.

    mode : {"texture", "data"}
        - "texture": map a rendered snapshot of ax2 onto a 3D plane (faithful).
        - "data": re-plot extracted theta,r data into 3D (partial).

    out : str or None
        If provided, saves animation to this path (.mp4 recommended).
        Requires ffmpeg for mp4. If missing, try .gif.

    frames : int
        Number of animation frames.

    elev_start/elev_end : float
        Camera elevation in degrees. 90 means looking straight down at the plane.

    azim_start : float
        Starting camera azimuth.

    azim_spin : float
        Total degrees to spin azimuth over the animation.

    plane_scale : float
        Scale factor for the texture plane size.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection="3d")

    # Make 3D axes visually clean
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.set_zlabel("")
    ax3.set_box_aspect((1, 1, 0.6))

    if mode == "texture":
        rgba = _render_ax_to_rgba(ax2, dpi=200)

        # Build a plane in 3D with the image as facecolors
        h, w = rgba.shape[0], rgba.shape[1]
        # plane coordinates (normalized to [-1, 1])
        xs = np.linspace(-1, 1, w) * plane_scale
        ys = np.linspace(-1, 1, h) * plane_scale
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)

        # Matplotlib expects facecolors as (M, N, 4) float in [0,1]
        face = rgba.astype(np.float32) / 255.0

        surf = ax3.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            facecolors=face,
            shade=False,
            linewidth=0,
            antialiased=False,
        )

        # Set bounds
        lim = 1.05 * plane_scale
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_zlim(-lim * 0.6, lim * 0.6)

    elif mode == "data":
        items = _extract_theta_r_from_ax(ax2)

        # Convert polar to Cartesian in the "sky plane"
        # x = r*sin(theta), y = r*cos(theta) matches your repo convention
        for kind, th, rr, artist in items:
            x = rr * np.sin(th)
            y = rr * np.cos(th)
            z = np.zeros_like(x)

            if kind == "scatter":
                ax3.scatter(x, y, z, s=6, depthshade=False)
            else:
                ax3.plot(x, y, z, linewidth=1)

        # Auto bounds
        all_rr = np.concatenate([it[2] for it in items], axis=0) if items else np.array([1.0])
        rmax = float(np.nanmax(all_rr)) if all_rr.size else 1.0
        lim = max(1.0, rmax) * 1.05
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_zlim(-lim * 0.6, lim * 0.6)

    else:
        raise ValueError("mode must be 'texture' or 'data'")

    def _frame(i):
        t = i / max(1, (frames - 1))
        elev = elev_start + (elev_end - elev_start) * t
        azim = azim_start + azim_spin * t
        ax3.view_init(elev=elev, azim=azim)
        return ()

    anim = FuncAnimation(fig3, _frame, frames=frames, interval=1000 / fps, blit=False)

    if out:
        # mp4 requires ffmpeg; gif usually works if pillow is installed
        try:
            anim.save(out, fps=fps)
        except Exception as e:
            raise RuntimeError(
                f"Saving failed ({e}). If you're trying .mp4, ensure ffmpeg is installed. "
                f"Otherwise try out='something.gif'."
            )

    plt.show()
    return anim