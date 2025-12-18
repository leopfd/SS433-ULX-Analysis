import numpy as np
import pandas as pd
from scipy.optimize import minimize
import config

def get_precession_limits(params):
    num_points = 720
    jd_cycle = params['jd0_precession'] + np.linspace(0, params['precession_period'], num_points)
    *_, pa_blue_cycle, pa_red_cycle = ss433_phases(jd_cycle, params)
    return {
        'blue_min': np.min(pa_blue_cycle), 'blue_max': np.max(pa_blue_cycle),
        'red_min': np.min(pa_red_cycle),   'red_max': np.max(pa_red_cycle)
    }

def ss433_phases(jd_obs, params, beta_blue=None, beta_red=None):
    inc, chi = params['inclination'], params['prec_pa']
    base_beta_b = beta_blue if beta_blue is not None else params['beta']
    base_beta_r = beta_red if beta_red is not None else params['beta']

    effective_theta = params['theta']
    effective_beta_b = base_beta_b
    effective_beta_r = base_beta_r

    prec_phase = ((jd_obs - params['jd0_precession']) / params['precession_period']) % 1.0
    if params.get('model_type') == 'full':
        phi = params['phi0'] - 2 * np.pi * prec_phase
    else: 
        phi = -2 * np.pi * prec_phase

    if params.get('model_type') == 'full':
        orb_phase = ((jd_obs - params['jd0_orb']) / params['orbital_period']) % 1.0
        nut_phase = ((jd_obs - params['jd0_nut']) / params['nut_period']) % 1.0
        effective_theta += params['nut_ampl'] * np.cos(2 * np.pi * nut_phase)
        orb_velocity_term = params['beta_orb_ampl'] * np.sin(2 * np.pi * orb_phase + params['beta_orb_phase0'])
        effective_beta_b += orb_velocity_term
        effective_beta_r += orb_velocity_term

    sin_theta, cos_theta = np.sin(effective_theta), np.cos(effective_theta)
    sin_inc, cos_inc = np.sin(inc), np.cos(inc)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_chi, cos_chi = np.sin(chi), np.cos(chi)

    mu = (sin_theta * sin_inc * cos_phi + cos_theta * cos_inc)
    v_ra = (sin_chi * sin_theta * sin_phi + cos_chi * sin_inc * cos_theta - cos_chi * cos_inc * sin_theta * cos_phi)
    v_dec = (cos_chi * sin_theta * sin_phi - sin_chi * sin_inc * cos_theta + sin_chi * cos_inc * sin_theta * cos_phi)

    mu_red_ra = -effective_beta_r * v_ra / (1 + effective_beta_r * mu)
    mu_red_dec = -effective_beta_r * v_dec / (1 + effective_beta_r * mu)
    mu_blue_ra = effective_beta_b * v_ra / (1 - effective_beta_b * mu)
    mu_blue_dec = effective_beta_b * v_dec / (1 - effective_beta_b * mu)

    pa_blue = np.degrees(np.arctan2(mu_blue_ra, mu_blue_dec))
    pa_red = np.degrees(np.arctan2(mu_red_ra, mu_red_dec))

    return mu_blue_ra, mu_blue_dec, mu_red_ra, mu_red_dec, pa_blue, pa_red

def _calculate_cartesian_obs_and_errors(blob_data):
    pa_rad = np.deg2rad(blob_data['pa_obs'])
    x_obs = blob_data['rad_obs'] * np.sin(pa_rad)
    y_obs = blob_data['rad_obs'] * np.cos(pa_rad)
    
    sig_r = (blob_data['rad_err_U'] + abs(blob_data['rad_err_L'])) / 2.0
    sig_pa_rad = np.deg2rad((blob_data['pa_err_U'] + abs(blob_data['pa_err_L'])) / 2.0)
    
    sig_x_sq = max((np.sin(pa_rad) * sig_r)**2 + (blob_data['rad_obs'] * np.cos(pa_rad) * sig_pa_rad)**2, 1e-9)
    sig_y_sq = max((np.cos(pa_rad) * sig_r)**2 + (blob_data['rad_obs'] * -np.sin(pa_rad) * sig_pa_rad)**2, 1e-9)
    
    if sig_x_sq == 0: sig_x_sq = 1e-9
    if sig_y_sq == 0: sig_y_sq = 1e-9

    return x_obs, y_obs, sig_x_sq, sig_y_sq

def calculate_outlier_travel_time(blob_data, jet_type, limits, params):
    rad_obs = blob_data['rad_obs']
    v_ang_c = params['beta']
    v_ang_arcsec_per_day = (v_ang_c * config.C_PC_PER_DAY / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
    if v_ang_arcsec_per_day < 1e-9:  
        return {'travel_time': np.inf}
    travel_time = rad_obs / v_ang_arcsec_per_day
    return {'travel_time': travel_time, 'limit_pa': None}

def calculate_beta_error_bounds(blob_data, best_fit_beta, params, mjd_obs, beta_bounds_tuple):
    def check_intersection(beta_test):
        age_curve = np.linspace(10, 400, 1000)
        jd_ej_curve = (mjd_obs + 2400000.5) - age_curve
        is_blue = blob_data['comp'] == 'east' 
        beta_b = beta_test if is_blue else params['beta']
        beta_r = beta_test if not is_blue else params['beta']
        
        mu_b_ra, mu_b_dec, mu_r_ra, mu_r_dec, _, _ = ss433_phases(jd_ej_curve, params, beta_blue=beta_b, beta_red=beta_r)
        x_obs, y_obs, sig_x_sq, sig_y_sq = _calculate_cartesian_obs_and_errors(blob_data)
        
        mu_ra = mu_b_ra if is_blue else mu_r_ra
        mu_dec = mu_b_dec if is_blue else mu_r_dec
        
        rad_curve = (np.sqrt(mu_ra**2 + mu_dec**2) * config.C_PC_PER_DAY * age_curve / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
        pa_rad_curve = np.arctan2(mu_ra, mu_dec)
        x_model, y_model = rad_curve * np.sin(pa_rad_curve), rad_curve * np.cos(pa_rad_curve)
        
        chi_sq_curve = ((x_model - x_obs)**2 / sig_x_sq) + ((y_model - y_obs)**2 / sig_y_sq)
        return np.min(chi_sq_curve) <= 1.0

    beta_min, beta_max = beta_bounds_tuple
    beta_scan_range = np.linspace(beta_min, beta_max, 241)
    intersecting_betas = [b for b in beta_scan_range if check_intersection(b)]

    if not intersecting_betas:
        return np.nan, np.nan
        
    return min(intersecting_betas), max(intersecting_betas)

def fit_and_calculate_jets(blob_data_list, params, regularization_strength=0):
    try:
        blob_blue = next(b for b in blob_data_list if b['comp'] == 'east')
        blob_red = next(b for b in blob_data_list if b['comp'] == 'west')
    except (StopIteration, IndexError):
        return {'success': False, 'message': "could not identify jet pair."}

    mjd_obs = blob_data_list[0]['mjd_obs']
    limits = get_precession_limits(params)
    leeway = 10.0
    
    results = {'mjd_obs': mjd_obs, 'success': True, 'jets': {}}
    jets_to_fit = []

    pa_blue_obs = blob_blue['pa_obs']
    if limits['blue_min'] - leeway <= pa_blue_obs <= limits['blue_max'] + leeway:
        results['jets']['blue'] = {'method': 'fit'}
        jets_to_fit.append('blue')
    else:
        calc = calculate_outlier_travel_time(blob_blue, 'blue', limits, params)
        results['jets']['blue'] = {'method': 'calculate', **calc}

    pa_red_obs = blob_red['pa_obs']
    if limits['red_min'] - leeway <= pa_red_obs <= limits['red_max'] + leeway:
        results['jets']['red'] = {'method': 'fit'}
        jets_to_fit.append('red')
    else:
        calc = calculate_outlier_travel_time(blob_red, 'red', limits, params)
        results['jets']['red'] = {'method': 'calculate', **calc}

    if jets_to_fit:
        for jet_name in jets_to_fit:
            def get_single_jet_error(beta_to_fit):
                beta_val = beta_to_fit[0]
                beta_b = beta_val if jet_name == 'blue' else params['beta']
                beta_r = beta_val if jet_name == 'red' else params['beta']
                blob_data = blob_blue if jet_name == 'blue' else blob_red

                age_curve = np.linspace(10, 400, 1000)
                jd_ej_curve = (mjd_obs + 2400000.5) - age_curve
                mu_b_ra,mu_b_dec,mu_r_ra,mu_r_dec,_,_ = ss433_phases(jd_ej_curve,params,beta_b,beta_r)
                
                x_obs, y_obs, sig_x_sq, sig_y_sq = _calculate_cartesian_obs_and_errors(blob_data)
                
                mu_ra = mu_b_ra if jet_name == 'blue' else mu_r_ra
                mu_dec = mu_b_dec if jet_name == 'blue' else mu_r_dec
                
                rad_curve = (np.sqrt(mu_ra**2 + mu_dec**2)*config.C_PC_PER_DAY*age_curve/config.D_SS433_PC)*config.ARCSEC_PER_RADIAN
                pa_curve = np.arctan2(mu_ra, mu_dec)
                x_model, y_model = rad_curve * np.sin(pa_curve), rad_curve * np.cos(pa_curve)
                
                chi_sq = np.min(((x_model-x_obs)**2/sig_x_sq) + ((y_model-y_obs)**2/sig_y_sq))
                penalty = (beta_val - params['beta'])**2
                return chi_sq + regularization_strength * penalty

            initial_guess = [params['beta']]
            bounds = [(0.2, 0.32)]
            res = minimize(get_single_jet_error, initial_guess, bounds=bounds, method='L-BFGS-B')

            if res.success:
                best_beta = res.x[0]
                blob_to_test = blob_blue if jet_name == 'blue' else blob_red
                lower, upper = calculate_beta_error_bounds(blob_to_test, best_beta, params, mjd_obs, bounds[0])
                results['jets'][jet_name]['fitted_beta'] = best_beta
                results['jets'][jet_name]['beta_lower_bound'] = lower
                results['jets'][jet_name]['beta_upper_bound'] = upper
            else:
                results['jets'][jet_name]['fitted_beta'] = np.nan
                results['jets'][jet_name]['beta_lower_bound'] = np.nan
                results['jets'][jet_name]['beta_upper_bound'] = np.nan

    for jet_name in ['blue', 'red']:
        if results['jets'][jet_name]['method'] == 'fit':
            if pd.isna(results['jets'][jet_name].get('beta_lower_bound')):
                blob_data = blob_blue if jet_name == 'blue' else blob_red
                calc_results = calculate_outlier_travel_time(blob_data, jet_name, limits, params)
                results['jets'][jet_name] = {'method': 'calculate', **calc_results}

    final_fitted_betas = {}
    for jet_name in ['blue', 'red']:
        if results['jets'][jet_name]['method'] == 'fit':
            final_fitted_betas[jet_name] = results['jets'][jet_name]['fitted_beta']
        else:
            final_fitted_betas[jet_name] = params['beta']
    results['fitted_betas'] = final_fitted_betas
    return results

def _get_closest_geometric_point(blob_data, jet_type, fit_results, params):
    x_obs, y_obs, sig_x_sq, sig_y_sq = _calculate_cartesian_obs_and_errors(blob_data)
    age_curve = np.linspace(10, 400, 1000)
    jd_ej_curve = (fit_results['mjd_obs'] + 2400000.5) - age_curve
    beta_b = fit_results['fitted_betas']['blue']
    beta_r = fit_results['fitted_betas']['red']
    mu_b_ra, mu_b_dec, mu_r_ra, mu_r_dec, pa_b, pa_r = ss433_phases(jd_ej_curve, params, beta_blue=beta_b, beta_red=beta_r)
    
    if jet_type == 'blue':
        rad_curve = (np.sqrt(mu_b_ra**2 + mu_b_dec**2) * config.C_PC_PER_DAY * age_curve / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
        pa_curve = pa_b 
    else: 
        rad_curve = (np.sqrt(mu_r_ra**2 + mu_r_dec**2) * config.C_PC_PER_DAY * age_curve / config.D_SS433_PC) * config.ARCSEC_PER_RADIAN
        pa_curve = pa_r 

    x_model = rad_curve * np.sin(np.deg2rad(pa_curve))
    y_model = rad_curve * np.cos(np.deg2rad(pa_curve))
    chi_sq_dist_curve = ((x_model - x_obs)**2 / sig_x_sq) + ((y_model - y_obs)**2 / sig_y_sq)
    idx = np.argmin(chi_sq_dist_curve)
    return {'model_pa': pa_curve[idx], 'model_rad': rad_curve[idx], 'model_age': age_curve[idx]}