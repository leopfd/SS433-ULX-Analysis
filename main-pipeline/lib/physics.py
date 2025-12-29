import numpy as np
import pandas as pd
from scipy.optimize import minimize  # kept for compatibility
import config


# Utilities for grid generation and coordinate transformation
def _age_curve_one_cycle(params, n, min_age=0.0, frac_period=0.99):
    """
    Construct an age grid limited to a fraction of one precession cycle
    The fraction is kept below 1 to avoid ambiguity between phase 0 and phase 1
    which represent the same direction in the cycle
    """
    P = float(params["precession_period"])
    max_age = frac_period * P
    
    # Validation check to prevent generating an inverted or empty time grid
    if max_age <= min_age:
        raise ValueError(f"Bad age grid: max_age={max_age} <= min_age={min_age}")
    return np.linspace(min_age, max_age, n)


def _wrap_deg(angle_deg):
    """
    Wrap degrees to be within the range of negative 180 to positive 180
    """
    return (angle_deg + 180.0) % 360.0 - 180.0


def _calculate_cartesian_obs_and_errors(blob_data):
    """
    Converts polar observation data into Cartesian coordinates and 
    propagates the measurement errors
    """
    pa_rad = np.deg2rad(blob_data["pa_obs"])
    
    # Convert polar (radius, angle) to Cartesian (x, y)
    x_obs = blob_data["rad_obs"] * np.sin(pa_rad)
    y_obs = blob_data["rad_obs"] * np.cos(pa_rad)

    # Average the upper and lower error bounds to get a single sigma value
    sig_r = (blob_data["rad_err_U"] + abs(blob_data["rad_err_L"])) / 2.0
    sig_pa_deg = (blob_data["pa_err_U"] + abs(blob_data["pa_err_L"])) / 2.0
    sig_pa_rad = np.deg2rad(sig_pa_deg)

    # Propagate the radial and angular errors into Cartesian x and y errors
    # using the standard Jacobian approximation for polar-to-cartesian transformation
    sig_x_sq = (np.sin(pa_rad) * sig_r) ** 2 + (blob_data["rad_obs"] * np.cos(pa_rad) * sig_pa_rad) ** 2
    sig_y_sq = (np.cos(pa_rad) * sig_r) ** 2 + (blob_data["rad_obs"] * -np.sin(pa_rad) * sig_pa_rad) ** 2

    # Floor errors to a small positive value (epsilon) to prevent divide by zero exceptions
    # during later Chi-squared calculations
    sig_x_sq = max(sig_x_sq, 1e-9)
    sig_y_sq = max(sig_y_sq, 1e-9)

    return x_obs, y_obs, sig_x_sq, sig_y_sq


# Model Definitions
def get_precession_limits(params):
    """
    Determine the bounding box of the jet path on the sky over one full cycle
    Useful for setting plot limits or filtering outliers
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
    Calculate the kinematic model for SS433 at specific observation dates
    This includes precession and optionally nutation and orbital modulation
    Returns proper motion vectors and position angles
    """
    inc, chi = params["inclination"], params["prec_pa"]
    base_beta_e = beta_east if beta_east is not None else params["beta"]
    base_beta_w = beta_west if beta_west is not None else params["beta"]

    effective_theta = params["theta"]
    effective_beta_e = base_beta_e
    effective_beta_w = base_beta_w

    # Calculate the precession phase (0 to 1) based on the reference Julian Date
    prec_phase = ((jd_obs - params["jd0_precession"]) / params["precession_period"]) % 1.0
    
    # Adjust phase definition based on whether we are using the simple or full ephemeris model
    if params.get("model_type") == "full":
        phi = params["phi0"] - 2 * np.pi * prec_phase
    else:
        phi = -2 * np.pi * prec_phase

    # Apply perturbations if the full ephemeris model is selected
    # This adds nutation to the cone opening angle (theta) and orbital variation to velocity (beta)
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

    # Calculate projection unit vectors for the jet orientation
    # mu is the component along the line of sight
    mu = sin_theta * sin_inc * cos_phi + cos_theta * cos_inc
    
    # v_ra and v_dec represent the vector components in the plane of the sky
    v_ra = (sin_chi * sin_theta * sin_phi
            + cos_chi * sin_inc * cos_theta
            - cos_chi * cos_inc * sin_theta * cos_phi)
    v_dec = (cos_chi * sin_theta * sin_phi
             - sin_chi * sin_inc * cos_theta
             + sin_chi * cos_inc * sin_theta * cos_phi)

    # Compute proper motions for West (receding) and East (approaching) jets
    # The denominator (1 +/- beta * mu) accounts for relativistic time dilation / light travel time effects
    mu_west_ra = -effective_beta_w * v_ra / (1 + effective_beta_w * mu)
    mu_west_dec = -effective_beta_w * v_dec / (1 + effective_beta_w * mu)

    mu_east_ra = effective_beta_e * v_ra / (1 - effective_beta_e * mu)
    mu_east_dec = effective_beta_e * v_dec / (1 - effective_beta_e * mu)

    # Convert the proper motion vectors into position angles
    pa_east = np.degrees(np.arctan2(mu_east_ra, mu_east_dec))
    pa_west = np.degrees(np.arctan2(mu_west_ra, mu_west_dec))

    return mu_east_ra, mu_east_dec, mu_west_ra, mu_west_dec, pa_east, pa_west


# Interval Helper Functions
def get_valid_beta_interval(blob_data, params, mjd_obs, jet_side):
    """
    Finds the range of beta values where the model trajectory intersects the error ellipse of the blob
    Used to calculate error bars for the velocity
    """
    age_curve = _age_curve_one_cycle(params, n=1000, min_age=0.0, frac_period=0.99)
    jd_ej_curve = (mjd_obs + 2400000.5) - age_curve

    x_obs, y_obs, sig_x_sq, sig_y_sq = _calculate_cartesian_obs_and_errors(blob_data)

    def check_beta(b):
        beta_e = b if jet_side == "east" else params["beta"]
        beta_w = b if jet_side == "west" else params["beta"]

        # Generate the kinematic model for this specific beta trial
        mu_e_ra, mu_e_dec, mu_w_ra, mu_w_dec, _, _ = ss433_phases(
            jd_ej_curve, params, beta_east=beta_e, beta_west=beta_w
        )

        mu_ra = mu_e_ra if jet_side == "east" else mu_w_ra
        mu_dec = mu_e_dec if jet_side == "east" else mu_w_dec

        # Convert proper motion rates to on sky separation in arcseconds
        rad_curve = (
            np.sqrt(mu_ra**2 + mu_dec**2)
            * config.C_PC_PER_DAY
            * age_curve
            / config.D_SS433_PC
        ) * config.ARCSEC_PER_RADIAN

        pa_rad_curve = np.arctan2(mu_ra, mu_dec)
        x_mod = rad_curve * np.sin(pa_rad_curve)
        y_mod = rad_curve * np.cos(pa_rad_curve)

        # Calculate Chi squared distance between model curve and the observation point
        chi_sq_curve = ((x_mod - x_obs) ** 2 / sig_x_sq) + ((y_mod - y_obs) ** 2 / sig_y_sq)
        
        # Valid if the curve passes within 1 sigma of the observation
        return np.min(chi_sq_curve) <= 1.0

    betas = np.linspace(0.20, 0.30, 201)
    valid_betas = [b for b in betas if check_beta(b)]
    if not valid_betas:
        return None
    return (min(valid_betas), max(valid_betas))


# Core Fitting Logic
def _fit_single_side(blobs, params, mjd_obs, jet_side, regularization_strength=0.0):
    """
    Fit beta for one side by brute force scan
    The objective prioritizes crossing the error ellipses first
    Prefer betas that enter the 1 sigma error ellipses of as many blobs as possible
    Among betas with the same number of crossed blobs minimize total chi squared
    Apply PA penalty and beta regularization as tie breakers
    """
    if not blobs:
        return params["beta"], np.nan, np.nan, "no_data"

    # Define the search grid for velocity (beta)
    # Using a dense grid ensures we don't miss narrow solutions in the complex parameter space
    test_betas = np.linspace(0.20, 0.30, 500)
    # test_betas = np.arange(0.20, 0.3000001, 0.0001)

    # Pre calculate the age curve for a full cycle to compare against all beta trials
    age_curve = _age_curve_one_cycle(params, n=1200, min_age=0.0, frac_period=0.99)
    jd_ej_curve = (mjd_obs + 2400000.5) - age_curve

    best_beta = params["beta"]
    best_score = (np.inf, np.inf, np.inf, np.inf)  # n_outside, sum_chi, sum_pa_pen, reg_pen

    for b in test_betas:
        # Generate the model track for this specific beta
        if jet_side == "east":
            mu_ra, mu_dec, _, _, pa_e, _ = ss433_phases(
                jd_ej_curve, params, beta_east=b, beta_west=params["beta"]
            )
            pa_deg_curve = pa_e
        else:
            _, _, mu_ra, mu_dec, _, pa_w = ss433_phases(
                jd_ej_curve, params, beta_east=params["beta"], beta_west=b
            )
            pa_deg_curve = pa_w

        # Convert angular rates to physical sky positions
        rad_curve = (
            np.sqrt(mu_ra**2 + mu_dec**2)
            * config.C_PC_PER_DAY
            * age_curve
            / config.D_SS433_PC
        ) * config.ARCSEC_PER_RADIAN

        pa_rad_curve = np.arctan2(mu_ra, mu_dec)
        xm, ym = rad_curve * np.sin(pa_rad_curve), rad_curve * np.cos(pa_rad_curve)

        n_outside = 0
        sum_chi = 0.0
        sum_pa_pen = 0.0

        # Evaluate this beta against every observed blob
        for blob in blobs:
            xo, yo, sx, sy = _calculate_cartesian_obs_and_errors(blob)
            
            # Find the minimum distance from this blob to the model curve
            dist_sq = ((xm - xo) ** 2 / sx) + ((ym - yo) ** 2 / sy)

            m = float(np.min(dist_sq))          # best chi^2 along curve for this blob
            idx = int(np.argmin(dist_sq))       # where that best match occurs

            # Primary criterion checks if the curve enters the 1 sigma ellipse
            # If the closest point is > 1 sigma away it counts as a miss
            if m > 1.0:
                n_outside += 1

            # Secondary criterion uses total closeness (sum of Chi squared)
            sum_chi += m

            # Tertiary criterion adds penalty if the Position Angle is misaligned
            # This helps break ties where distance is similar but orientation is wrong
            pa_obs = float(blob["pa_obs"])
            sig_pa = (float(blob["pa_err_U"]) + abs(float(blob["pa_err_L"]))) / 2.0
            sig_pa = max(sig_pa, 0.5)  # floor in degrees
            dpa = _wrap_deg(pa_deg_curve[idx] - pa_obs)
            sum_pa_pen += (dpa / sig_pa) ** 2

        # Quaternary criterion penalizes deviation from the canonical beta value
        # This keeps the solution physical if the data is ambiguous
        reg_pen = 0.0
        if regularization_strength and regularization_strength > 0:
            sigma_b = float(regularization_strength)
            reg_pen = ((b - float(params["beta"])) / sigma_b) ** 2

        score = (n_outside, sum_chi, sum_pa_pen, reg_pen)

        # Update best fit if this score is lower (better)
        # Tuple comparison automatically prioritizes n_outside then sum_chi etc
        if score < best_score:
            best_score = score
            best_beta = b

    # Determine confidence intervals logic
    # We find the min/max beta that still intersects the error ellipses
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
            lower, upper = global_min, global_max
            best_beta = np.clip(best_beta, lower, upper)
        else:
            method = "fit (inconsistent)"
    else:
        method = "fit (outliers)"

    # Add a hint if the best solution still missed some ellipses
    if best_score[0] > 0 and method == "fit":
        method = f"fit (missed {int(best_score[0])} ellipse{'s' if best_score[0] != 1 else ''})"

    return best_beta, lower, upper, method


def fit_and_calculate_jets(blob_data_list, params, regularization_strength=0.0):
    """
    Main driver function to fit jet kinematics
    Separates data into East/West components and fits them independently
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

    # Perform independent fits for East and West jets
    # We allow regularization to be passed down to control beta deviation
    beta_e, low_e, up_e, meth_e = _fit_single_side(
        east_blobs, params, mjd_obs, "east", regularization_strength=regularization_strength
    )
    beta_w, low_w, up_w, meth_w = _fit_single_side(
        west_blobs, params, mjd_obs, "west", regularization_strength=regularization_strength
    )

    results["fitted_betas"] = {"east": beta_e, "west": beta_w}
    results["jets"]["east"] = []
    results["jets"]["west"] = []

    def make_entry(blob, beta, low, up, meth):
        return {
            "blob_id": blob["comp"],
            "method": meth,
            "fitted_beta": beta,
            "beta_lower_bound": low,
            "beta_upper_bound": up,
        }

    for blob in east_blobs:
        results["jets"]["east"].append(make_entry(blob, beta_e, low_e, up_e, meth_e))

    for blob in west_blobs:
        results["jets"]["west"].append(make_entry(blob, beta_w, low_w, up_w, meth_w))

    return results


def _get_closest_geometric_point(blob_data, jet_side, fit_results, params):
    """
    Given a fitted beta, find the exact point on the model curve closest to the observation
    Returns the model age, radius, and PA at that point
    """
    beta_e = fit_results["fitted_betas"]["east"]
    beta_w = fit_results["fitted_betas"]["west"]

    x_obs, y_obs, sig_x_sq, sig_y_sq = _calculate_cartesian_obs_and_errors(blob_data)

    # Generate a high resolution curve to find the precise nearest neighbor
    age_curve = _age_curve_one_cycle(params, n=2000, min_age=0.0, frac_period=0.99)
    jd_ej_curve = (fit_results["mjd_obs"] + 2400000.5) - age_curve

    mu_e_ra, mu_e_dec, mu_w_ra, mu_w_dec, pa_e, pa_w = ss433_phases(
        jd_ej_curve, params, beta_east=beta_e, beta_west=beta_w
    )

    if jet_side == "east":
        rad_curve = (
            np.sqrt(mu_e_ra**2 + mu_e_dec**2)
            * config.C_PC_PER_DAY
            * age_curve
            / config.D_SS433_PC
        ) * config.ARCSEC_PER_RADIAN
        pa_curve = pa_e
    else:
        rad_curve = (
            np.sqrt(mu_w_ra**2 + mu_w_dec**2)
            * config.C_PC_PER_DAY
            * age_curve
            / config.D_SS433_PC
        ) * config.ARCSEC_PER_RADIAN
        pa_curve = pa_w

    x_model = rad_curve * np.sin(np.deg2rad(pa_curve))
    y_model = rad_curve * np.cos(np.deg2rad(pa_curve))

    dist_sq = ((x_model - x_obs) ** 2 / sig_x_sq) + ((y_model - y_obs) ** 2 / sig_y_sq)
    idx = int(np.argmin(dist_sq))

    return {"model_pa": float(pa_curve[idx]), "model_rad": float(rad_curve[idx]), "model_age": float(age_curve[idx])}