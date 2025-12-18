import numpy as np
import os
from astropy.io import fits
from scipy.optimize import curve_fit
from ciao_contrib.runtool import dmhedit
from coords.format import ra2deg, dec2deg

def quickpos(x, y, x0, y0, iterations=1, size_list=None, binsize_list=None, doplot=False):
    # iteratively refines the centroid position using 1d histogram fitting
    if size_list is None:
        size_list = [np.max(x) - np.min(x)] * iterations
    if binsize_list is None:
        binsize_list = [1.0] * iterations

    fig_list = []
    current_x0, current_y0 = x0, y0
    cnt = None
    best_x0, best_y0 = current_x0, current_y0 

    for i in range(iterations):
        size = size_list[i]
        binsize = binsize_list[i]

        ob = np.where((np.abs(x - current_x0) < size) & (np.abs(y - current_y0) < size))
        
        if len(ob[0]) == 0:
            continue 

        xbins = np.arange(current_x0 - size, current_x0 + size + binsize, binsize)
        ybins = np.arange(current_y0 - size, current_y0 + size + binsize, binsize)

        xhist, xedges = np.histogram(x[ob], bins=xbins)
        yhist, yedges = np.histogram(y[ob], bins=ybins)

        xval = 0.5 * (xedges[:-1] + xedges[1:])
        yval = 0.5 * (yedges[:-1] + yedges[1:])

        def gaussian(x, a, mu, sigma, offset):
            return a * np.exp(-((x - mu)**2) / (2 * sigma**2)) + offset

        xcnt = 0
        ycnt = 0
        
        try:
            xmax = np.max(xhist)
            x0_new = xval[np.argmax(xhist)]
            xestpar = [xmax, x0_new, 2 * binsize, 0]
            xpar, _ = curve_fit(gaussian, xval, xhist, p0=xestpar)
            best_x0 = xpar[1] 
            xcnt = xpar[0] * xpar[2] * np.sqrt(2 * np.pi)
        except Exception:
            xcnt = 0

        try:
            ymax = np.max(yhist)
            y0_new = yval[np.argmax(yhist)]
            yestpar = [ymax, y0_new, 2 * binsize, 0]
            ypar, _ = curve_fit(gaussian, yval, yhist, p0=yestpar)
            best_y0 = ypar[1] 
            ycnt = ypar[0] * ypar[2] * np.sqrt(2 * np.pi)
        except Exception:
            ycnt = 0

        cnt = 0.5 * (xcnt + ycnt)
        current_x0 = best_x0
        current_y0 = best_y0

    return best_x0, best_y0, cnt, fig_list

def data_extract_quickpos_iter(infile, iters=3, sizes=[10, 5, 1.5], binsizes=[0.1, 0.1, 0.05]):
    # extracts data from a fits file and runs quickpos to get an initial centroid
    with fits.open(infile) as obs:
        hdr = obs[1].header
        data = obs[1].data
        
        scale = hdr['tcdlt20']
        xc = hdr['tcrpx20']
        exptime = hdr['exposure']
            
        mjd_start = hdr['mjd-obs']
        half_expos = 0.5 * (hdr['tstop']-hdr['tstart'])
        date = mjd_start + half_expos / 86400
        
        x = (data['x'] - xc) * scale * 3600
        y = (data['y'] - xc) * scale * 3600
        
        rr = np.sqrt(x**2 + y**2)
        ok = np.where(rr < 20)
        
        x0_est = np.average(x[ok])
        y0_est = np.average(y[ok])

    x0_best, y0_best, cnt, qp_figs = quickpos(x[ok], y[ok], x0_est, y0_est, iters, sizes, binsizes)
    
    pixel_x0_best = x0_best / (scale * 3600) + xc
    pixel_y0_best = y0_best / (scale * 3600) + xc

    return date, exptime, pixel_x0_best, pixel_y0_best, cnt, qp_figs

def write_pixelscale(file: str, nx: int, ny: int, ra: str, dec: str, hrc_pscale_arcsec: float = 0.13175):
    # adds wcs header information using dmhedit
    x_pix_ctr = (nx / 2.0) + 0.5
    y_pix_ctr = (ny / 2.0) + 0.5
    hrc_pscale_deg = hrc_pscale_arcsec / 3600.
    x_platescale = -abs(hrc_pscale_deg / 4.)
    y_platescale = abs(hrc_pscale_deg / 4.)
    ra_deg = ra2deg(ra)
    dec_deg = dec2deg(dec)
    wcs_params = [
        ("WCSAXES", 2, "short", None), ("CRPIX1", x_pix_ctr, "float", None),
        ("CRPIX2", y_pix_ctr, "float", None), ("CDELT1", x_platescale, "float", "deg"),
        ("CDELT2", y_platescale, "float", "deg"), ("CUNIT1", "deg", "string", None),
        ("CUNIT2", "deg", "string", None), ("CTYPE1", "RA---TAN", "string", None),
        ("CTYPE2", "DEC--TAN", "string", None), ("CRVAL1", ra_deg, "float", "deg"),
        ("CRVAL2", dec_deg, "float", "deg"), ("LONPOLE", 180.0, "float", "deg"),
        ("LATPOLE", 0, "float", "deg"), ("RADESYS", "ICRS", "string", None),
    ]
    try:
        for key, value, dtype, unit in wcs_params:
            dmhedit(infile=file, op="add", key=key, value=value, datatype=dtype, unit=unit)
    except Exception as e:
        obsid = os.path.basename(os.path.dirname(file))
        print(f"  error (obsid {obsid}): dmhedit failed: {e}")

def compute_split_rhat(chain):
    # calculates the split-rhat statistic for convergence
    n_steps, n_walkers, n_params = chain.shape
    half = n_steps // 2
    split_chain = np.concatenate((chain[:half], chain[half:]), axis=1)
    N = half
    M = n_walkers * 2
    var_within = np.var(split_chain, axis=0, ddof=1)
    W = np.mean(var_within, axis=0)
    mean_chains = np.mean(split_chain, axis=0)
    B = N * np.var(mean_chains, axis=0, ddof=1)
    var_plus = ((N - 1) / N) * W + (1 / N) * B
    rhat = np.sqrt(var_plus / W)
    return rhat