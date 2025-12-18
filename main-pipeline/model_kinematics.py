import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import config
from lib.physics import fit_and_calculate_jets, _get_closest_geometric_point
from lib.plotting import plot_fit_and_calc_results
from lib.arguments import get_pipeline_args

import track_components

def run_kinematic_analysis(input_df):
    ss433_params = config.EPHEMERIS
    
    pdf_output_path = config.PLOT_OUTPUT_PDF
    
    pdf_pages = PdfPages(pdf_output_path)
    all_results_data = []

    df = input_df.copy()

    is_first_iteration = True
    
    for obs_id, group in df.groupby('obs_id'):
        blue_data = group[group['component'] == 'east']
        if group.empty or blue_data.empty: continue
        red_data = group[group['component'] == 'west']

        if len(blue_data) == 1 and len(red_data) == 1:
            blue_row = blue_data.iloc[0]
            red_row = red_data.iloc[0]
            
            blob_pair_data = [
                {'mjd_obs': blue_row['mjd'], 'comp': 'east', 'pa_obs': blue_row['PA'], 'rad_obs': blue_row['radius'], 'rad_err_U': blue_row['radius_plus_err'], 'rad_err_L': blue_row['radius_minus_err'], 'pa_err_U': blue_row['PA_err_plus'], 'pa_err_L': blue_row['PA_err_minus']},
                {'mjd_obs': red_row['mjd'], 'comp': 'west', 'pa_obs': red_row['PA'], 'rad_obs': red_row['radius'], 'rad_err_U': red_row['radius_plus_err'], 'rad_err_L': red_row['radius_minus_err'], 'pa_err_U': red_row['PA_err_plus'], 'pa_err_L': red_row['PA_err_minus']}
            ]
            
            reg_strength = 0 if not is_first_iteration else 0
            is_first_iteration = False

            analysis_results = fit_and_calculate_jets(blob_pair_data, ss433_params, regularization_strength=reg_strength)

            if analysis_results['success']:
                plot_fit_and_calc_results(obs_id, blob_pair_data, analysis_results, ss433_params, pdf_object=pdf_pages)
                
                for jet_color in ['blue', 'red']:
                    jet_info = analysis_results['jets'][jet_color]
                    blob_data = blob_pair_data[0] if jet_color == 'blue' else blob_pair_data[1]
                    row = {
                        'obs_id': obs_id,
                        'mjd': analysis_results['mjd_obs'],
                        'jet_color': jet_color,
                        'component_name': blob_data['comp'],
                        'method': jet_info['method'],
                        'beta': np.nan,
                        'beta_err_pos': np.nan,
                        'beta_err_neg': np.nan,
                        'travel_time_days': np.nan
                    }

                    if jet_info['method'] == 'fit':
                        beta_val = jet_info.get('fitted_beta')
                        lower = jet_info.get('beta_lower_bound')
                        upper = jet_info.get('beta_upper_bound')
                        row['beta'] = beta_val
                        if pd.notna(lower) and pd.notna(upper):
                            row['beta_err_pos'] = upper - beta_val
                            row['beta_err_neg'] = beta_val - lower
                        fitted_point = _get_closest_geometric_point(blob_data, jet_color, analysis_results, ss433_params)
                        row['travel_time_days'] = fitted_point.get('model_age')
                    else:
                        row['beta'] = ss433_params['beta'] 
                        row['travel_time_days'] = jet_info.get('travel_time')
                    
                    all_results_data.append(row)
            else:
                print(f"--> processing failed for {obs_id}: {analysis_results.get('message', 'n/a')}")
        else:
            print(f"--> skipping {obs_id}: did not find a valid blue/red jet pair.")

    pdf_pages.close()
    print(f"\nall plots saved to '{pdf_output_path}'")
    
    return pd.DataFrame(all_results_data)

if __name__ == "__main__":
    args = get_pipeline_args()
    config.update_config_from_args(args)

    print("Running standalone: fetching data from tracker...")
    tracker_df = track_components.run_tracker_analysis()
    
    if tracker_df is not None and not tracker_df.empty:
        run_kinematic_analysis(tracker_df)
    else:
        print("Error: Tracker returned no data.")