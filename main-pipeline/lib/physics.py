import numpy as np
import pandas as pd
from scipy.optimize import minimize  # kept for compatibility
import config


def _age_curve_one_cycle(params, n=None, min_age=0.0, frac_period=2):
    """
    Construct an age grid limited to a fraction of one precession cycle.
    All fitting/interval/closest-point logic must use the same grid to be consistent.
    """
    P = float(params["precession_period"])
    max_age = frac_period * P
    if max_age <= min_age:
        raise ValueError(f"Bad age grid: max_age={max_age} <= min_age={min_age}")

    if n is None:
        n = int(params.get("age_grid_n", 1200))
    return np.linspace(min_age, max_age, int(n))


def _wrap_deg(angle_deg):
    """
    Wrap degrees to be within [-180, +180).
    """
    return (angle_deg + 180.0) % 360.0 - 180.0


def _sigma_r_pa(blob_data):
    """
    Symmetric 1-sigma uncertainties in (r, PA) from existing +/- errors.
    """
    sig_r = (float(blob_data["rad_err_U"]) + abs(float(blob_data["rad_err_L"]))) / 2.0
    sig_pa = (float(blob_data["pa_err_U"]) + abs(float(blob_data["pa_err_L"]))) / 2.0
    sig_r = max(sig_r, 1e-6)
    sig_pa = max(sig_pa, 1e-3)
    return sig_r, sig_pa


def _polar_inside_mask(rad_curve, pa_deg_curve, blob):
    """
    Inside the 1-sigma polar error box:
      |dr| <= sigma_r AND |dPA| <= sigma_PA.
    """
    r0 = float(blob["rad_obs"])
    pa0 = float(blob["pa_obs"])
    sig_r, sig_pa = _sigma_r_pa(blob)

    dr = np.abs(rad_curve - r0)
    dpa = np.abs(_wrap_deg(pa_deg_curve - pa0))
    return (dr <= sig_r) & (dpa <= sig_pa)


def _polar_misfit(rad_curve, pa_deg_curve, blob):
    """
    Dimensionless misfit used for closest-point selection:
      (dr/sig_r)^2 + (dPA/sig_PA)^2.
    """
    r0 = float(blob["rad_obs"])
    pa0 = float(blob["pa_obs"])
    sig_r, sig_pa = _sigma_r_pa(blob)

    dr = (rad_curve - r0) / sig_r
    dpa = _wrap_deg(pa_deg_curve - pa0) / sig_pa
    return dr * dr + dpa * dpa


def _curve_rad_pa_deg(age_curve, jd_ej_curve, params, jet_side, beta_e, beta_w):
    """
    Compute model (rad, PA) arrays on the provided grid for a given jet side and betas.
    """
    mu_e_ra, mu_e_dec, mu_w_ra, mu_w_dec, pa_e, pa_w = ss433_phases(
        jd_ej_curve, params, beta_east=beta_e, beta_west=beta_w
    )

    if jet_side == "east":
        mu_ra, mu_dec = mu_e_ra, mu_e_dec
        pa_deg_curve = pa_e
    else:
        mu_ra, mu_dec = mu_w_ra, mu_w_dec
        pa_deg_curve = pa_w

    rad_curve = (
        np.sqrt(mu_ra**2 + mu_dec**2)
        * config.C_PC_PER_DAY
        * age_curve
        / config.D_SS433_PC
    ) * config.ARCSEC_PER_RADIAN

    return rad_curve, pa_deg_curve

def _blob_intersects_at_beta(blob, params, mjd_obs, jet_side, beta_e, beta_w):
    age_curve = _age_curve_one_cycle(params)
    jd_ej_curve = (mjd_obs + 2400000.5) - age_curve

    win = _canonical_age_window_mask(age_curve, jd_ej_curve, params, jet_side, blob)
    rad_curve, pa_curve = _curve_rad_pa_deg(age_curve, jd_ej_curve, params, jet_side, beta_e, beta_w)

    if np.any(win):
        return bool(np.any(_polar_inside_mask(rad_curve[win], pa_curve[win], blob)))
    return bool(np.any(_polar_inside_mask(rad_curve, pa_curve, blob)))


def _canonical_age_window_mask(age_curve, jd_ej_curve, params, jet_side, blob):
    """
    Define an age window centered on the closest point found on the canonical-beta curve.
    This prevents matching a knot to a different precession-phase branch.
    """
    age_window_days = float(params.get("age_window_days", 17.0))
    beta_ref = float(params["beta"])

    rad_ref, pa_ref = _curve_rad_pa_deg(
        age_curve, jd_ej_curve, params, jet_side, beta_ref, beta_ref
    )

    mis = _polar_misfit(rad_ref, pa_ref, blob)
    idx0 = int(np.argmin(mis))
    t0 = float(age_curve[idx0])

    return (age_curve >= t0 - age_window_days) & (age_curve <= t0 + age_window_days)


def _calculate_cartesian_obs_and_errors(blob_data):
    """
    Convert polar observation to cartesian with propagated errors.
    Kept for compatibility with plotting/diagnostics that assume cartesian sigmas.
    """
    pa_rad = np.deg2rad(blob_data["pa_obs"])

    x_obs = blob_data["rad_obs"] * np.sin(pa_rad)
    y_obs = blob_data["rad_obs"] * np.cos(pa_rad)

    sig_r = (blob_data["rad_err_U"] + abs(blob_data["rad_err_L"])) / 2.0
    sig_pa_deg = (blob_data["pa_err_U"] + abs(blob_data["pa_err_L"])) / 2.0
    sig_pa_rad = np.deg2rad(sig_pa_deg)

    sig_x_sq = (np.sin(pa_rad) * sig_r) ** 2 + (blob_data["rad_obs"] * np.cos(pa_rad) * sig_pa_rad) ** 2
    sig_y_sq = (np.cos(pa_rad) * sig_r) ** 2 + (blob_data["rad_obs"] * -np.sin(pa_rad) * sig_pa_rad) ** 2

    sig_x_sq = max(sig_x_sq, 1e-9)
    sig_y_sq = max(sig_y_sq, 1e-9)

    return x_obs, y_obs, sig_x_sq, sig_y_sq


def get_precession_limits(params):
    """
    Determine min/max PA over one full cycle for east/west jets.
    """
    num_points = 720
    jd_cycle = params["jd0_precession"] + np.linspace(0, params["precession_period"], num_points)
    *_, pa_east_cycle, pa_west_cycle = ss433_phases(jd_cycle, params)
    return {
        "east_min": np.min(pa_east_cycle),
        "east_max": np.max(pa_east_cycle),
        "west_min": np.min(pa_west_cycle),
        "west_max": np.max(pa_west_cycle),
    }


def ss433_phases(jd_obs, params, beta_east=None, beta_west=None):
    """
    Kinematic model for SS433:
      returns (mu_e_ra, mu_e_dec, mu_w_ra, mu_w_dec, pa_east, pa_west).
    """
    inc, chi = params["inclination"], params["prec_pa"]
    base_beta_e = beta_east if beta_east is not None else params["beta"]
    base_beta_w = beta_west if beta_west is not None else params["beta"]

    effective_theta = params["theta"]
    effective_beta_e = base_beta_e
    effective_beta_w = base_beta_w

    prec_phase = ((jd_obs - params["jd0_precession"]) / params["precession_period"]) % 1.0

    if params.get("model_type") == "full":
        phi = params["phi0"] - 2 * np.pi * prec_phase
    else:
        phi = -2 * np.pi * prec_phase

    if params.get("model_type") == "full":
        orb_phase = ((jd_obs - params["jd0_orb"]) / params["orbital_period"]) % 1.0
        nut_phase = ((jd_obs - params["jd0_nut"]) / params["nut_period"]) % 1.0
        effective_theta += params["nut_ampl"] * np.cos(2 * np.pi * nut_phase)
        orb_velocity_term = params["beta_orb_ampl"] * np.sin(2 * np.pi * orb_phase + params["beta_orb_phase0"])
        effective_beta_e += orb_velocity_term
        effective_beta_w += orb_velocity_term

    sin_theta, cos_theta = np.sin(effective_theta), np.cos(effective_theta)
    sin_inc, cos_inc = np.sin(inc), np.cos(inc)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_chi, cos_chi = np.sin(chi), np.cos(chi)

    mu = sin_theta * sin_inc * cos_phi + cos_theta * cos_inc

    v_ra = (
        sin_chi * sin_theta * sin_phi
        + cos_chi * sin_inc * cos_theta
        - cos_chi * cos_inc * sin_theta * cos_phi
    )
    v_dec = (
        cos_chi * sin_theta * sin_phi
        - sin_chi * sin_inc * cos_theta
        + sin_chi * cos_inc * sin_theta * cos_phi
    )

    mu_west_ra = -effective_beta_w * v_ra / (1 + effective_beta_w * mu)
    mu_west_dec = -effective_beta_w * v_dec / (1 + effective_beta_w * mu)

    mu_east_ra = effective_beta_e * v_ra / (1 - effective_beta_e * mu)
    mu_east_dec = effective_beta_e * v_dec / (1 - effective_beta_e * mu)

    pa_east = np.degrees(np.arctan2(mu_east_ra, mu_east_dec))
    pa_west = np.degrees(np.arctan2(mu_west_ra, mu_west_dec))

    return mu_east_ra, mu_east_dec, mu_west_ra, mu_west_dec, pa_east, pa_west


def ss433_mu_from_config_ephemeris(jd):
    """
    Returns (mu_east, mu_west) where mu = cos(theta_LOS), using config.EPHEMERIS.
    """
    p = config.EPHEMERIS

    inc = p["inclination"]
    theta = p["theta"]

    prec_phase = ((jd - p["jd0_precession"]) / p["precession_period"]) % 1.0
    phi = -2.0 * np.pi * prec_phase

    mu_east = np.sin(theta) * np.sin(inc) * np.cos(phi) + np.cos(theta) * np.cos(inc)
    mu_west = -mu_east
    return mu_east, mu_west


def tau_core_to_knot_days_from_projected(rad_arcsec, mu):
    """
    Core → knot light travel time tau = r/c using projected separation and mu = cos(theta_LOS).
    """
    sin_th = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    Rproj_pc = (rad_arcsec / config.ARCSEC_PER_RADIAN) * config.D_SS433_PC
    return (Rproj_pc / config.C_PC_PER_DAY) / sin_th


def get_valid_beta_interval(blob_data, params, mjd_obs, jet_side):
    """
    Return (beta_min, beta_max) where the model intersects the blob's polar 1-sigma box
    within the blob's canonical-beta age window.
    """
    age_curve = _age_curve_one_cycle(params)
    jd_ej_curve = (mjd_obs + 2400000.5) - age_curve

    win = _canonical_age_window_mask(age_curve, jd_ej_curve, params, jet_side, blob_data)

    def check_beta(b):
        b = float(b)
        beta0 = float(params["beta"])
        beta_e = b if jet_side == "east" else beta0
        beta_w = b if jet_side == "west" else beta0

        rad_curve, pa_curve = _curve_rad_pa_deg(age_curve, jd_ej_curve, params, jet_side, beta_e, beta_w)

        if np.any(win):
            return bool(np.any(_polar_inside_mask(rad_curve[win], pa_curve[win], blob_data)))

        return bool(np.any(_polar_inside_mask(rad_curve, pa_curve, blob_data)))

    betas = np.linspace(0.20, 0.30, 201)
    valid = [b for b in betas if check_beta(b)]
    if not valid:
        return None
    return (float(min(valid)), float(max(valid)))


def _fit_single_side(blobs, params, mjd_obs, jet_side, regularization_strength=0.0):
    """
    Choose beta by scanning a grid.

    Primary objective: minimize number of blobs that never intersect the polar 1-sigma box
    within their canonical-beta age window. If any beta yields zero misses, only those
    candidates are eligible.

    Secondary objective: minimize summed minimum polar misfit.

    Tertiary objective: regularize toward params["beta"] with sigma=regularization_strength.
    """
    if not blobs:
        return float(params["beta"]), np.nan, np.nan, "no_data"

    age_curve = _age_curve_one_cycle(params)
    jd_ej_curve = (mjd_obs + 2400000.5) - age_curve

    beta0 = float(params["beta"])
    test_betas = np.linspace(0.20, 0.30, int(params.get("beta_grid_n", 500)))

    blob_wins = [_canonical_age_window_mask(age_curve, jd_ej_curve, params, jet_side, b) for b in blobs]

    best_any = (np.inf, np.inf, np.inf)
    best_any_beta = beta0

    best_zero = (np.inf, np.inf, np.inf)
    best_zero_beta = beta0
    found_zero = False

    for b in test_betas:
        b = float(b)
        if jet_side == "east":
            rad_curve, pa_curve = _curve_rad_pa_deg(age_curve, jd_ej_curve, params, jet_side, b, beta0)
        else:
            rad_curve, pa_curve = _curve_rad_pa_deg(age_curve, jd_ej_curve, params, jet_side, beta0, b)

        n_miss = 0
        sum_mis = 0.0

        for win, blob in zip(blob_wins, blobs):
            if np.any(win):
                r = rad_curve[win]
                pa = pa_curve[win]
            else:
                r = rad_curve
                pa = pa_curve

            if not np.any(_polar_inside_mask(r, pa, blob)):
                n_miss += 1

            sum_mis += float(np.min(_polar_misfit(r, pa, blob)))

        reg_pen = 0.0
        if regularization_strength and regularization_strength > 0:
            sigma_b = float(regularization_strength)
            reg_pen = ((b - beta0) / sigma_b) ** 2

        score = (int(n_miss), float(sum_mis), float(reg_pen))

        if score < best_any:
            best_any = score
            best_any_beta = b

        if n_miss == 0:
            found_zero = True
            if score < best_zero:
                best_zero = score
                best_zero_beta = b

    best_beta = best_zero_beta if found_zero else best_any_beta
    best_score = best_zero if found_zero else best_any

    intervals = []
    for blob in blobs:
        ival = get_valid_beta_interval(blob, params, mjd_obs, jet_side)
        if ival:
            intervals.append(ival)
        else:
            intervals = None
            break

    lower, upper = np.nan, np.nan
    method = "fit"

    if intervals:
        global_min = max(i[0] for i in intervals)
        global_max = min(i[1] for i in intervals)
        if global_min <= global_max:
            lower, upper = float(global_min), float(global_max)
            best_beta = float(np.clip(best_beta, lower, upper))
        else:
            method = "fit (inconsistent)"
    else:
        method = "fit (outliers)"

    if int(best_score[0]) > 0 and method == "fit":
        method = f"fit (missed {int(best_score[0])} ellipse{'s' if int(best_score[0]) != 1 else ''})"

    return float(best_beta), lower, upper, method


def fit_and_calculate_jets(blob_data_list, params, regularization_strength=0.0):
    """
    Fit east and west sides independently and return a dict matching downstream expectations.
    """
    if not blob_data_list:
        return {"success": False, "message": "no blob data"}

    mjd_obs = blob_data_list[0]["mjd_obs"]

    east_blobs = [b for b in blob_data_list if b["comp"].startswith("east")]
    west_blobs = [b for b in blob_data_list if b["comp"].startswith("west")]

    results = {
        "mjd_obs": mjd_obs,
        "success": True,
        "jets": {},
        "east_candidates": east_blobs,
        "west_candidates": west_blobs,
    }

    beta_e, low_e, up_e, meth_e = _fit_single_side(
        east_blobs, params, mjd_obs, "east", regularization_strength=regularization_strength
    )
    beta_w, low_w, up_w, meth_w = _fit_single_side(
        west_blobs, params, mjd_obs, "west", regularization_strength=regularization_strength
    )

    results["fitted_betas"] = {"east": beta_e, "west": beta_w}
    results["jets"]["east"] = []
    results["jets"]["west"] = []

    def make_entry(blob, beta, low, up, meth, jet_side):
        beta0 = float(params["beta"])

        beta_e = float(beta) if jet_side == "east" else beta0
        beta_w = float(beta) if jet_side == "west" else beta0

        ok = _blob_intersects_at_beta(blob, params, mjd_obs, jet_side, beta_e, beta_w)

        method = meth if ok else "plane_sky"

        return {
            "blob_id": blob["comp"],
            "method": method,
            "fitted_beta": float(beta),
            "beta_lower_bound": low,
            "beta_upper_bound": up,
        }

    for blob in east_blobs:
        results["jets"]["east"].append(make_entry(blob, beta_e, low_e, up_e, meth_e, "east"))

    for blob in west_blobs:
        results["jets"]["west"].append(make_entry(blob, beta_w, low_w, up_w, meth_w, "west"))

    return results


def _get_closest_geometric_point(blob_data, jet_side, fit_results, params):
    """
    Closest model point using polar misfit, restricted to the canonical-beta age window.
    """

    if str(fit_results.get("jets", {}).get(jet_side, [{}])[0].get("method", "")).startswith("plane"):
        return {
            "model_pa": float(blob_data["pa_obs"]),
            "model_rad": float(blob_data["rad_obs"]),
            "model_age": np.nan,
            "jd_ej": None,
        }

    beta_e = float(fit_results["fitted_betas"]["east"])
    beta_w = float(fit_results["fitted_betas"]["west"])

    age_curve = _age_curve_one_cycle(params)
    jd_ej_curve = (fit_results["mjd_obs"] + 2400000.5) - age_curve

    win = _canonical_age_window_mask(age_curve, jd_ej_curve, params, jet_side, blob_data)

    rad_curve, pa_curve = _curve_rad_pa_deg(
        age_curve, jd_ej_curve, params, jet_side, beta_e, beta_w
    )

    if np.any(win):
        mis = _polar_misfit(rad_curve[win], pa_curve[win], blob_data)
        idx_local = int(np.argmin(mis))
        idx = int(np.flatnonzero(win)[idx_local])
    else:
        mis = _polar_misfit(rad_curve, pa_curve, blob_data)
        idx = int(np.argmin(mis))

    return {
        "model_pa": float(pa_curve[idx]),
        "model_rad": float(rad_curve[idx]),
        "model_age": float(age_curve[idx]),
        "jd_ej": float(jd_ej_curve[idx]),
    }