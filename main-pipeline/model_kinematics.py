import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import config
from lib.physics import fit_and_calculate_jets, _get_closest_geometric_point
from lib.plotting import plot_fit_and_calc_results
from lib.arguments import get_pipeline_args
import track_components

def run_kinematic_analysis(input_df):
    # Initialize configuration parameters and output containers
    ss433_params = config.EPHEMERIS
    pdf_output_path = config.PLOT_OUTPUT_PDF
    pdf_pages = PdfPages(pdf_output_path)
    all_results_data = []

    df = input_df.copy()

    # Iterate through each observation group independently to model kinematics per epoch
    for obs_id, group in df.groupby('obs_id'):
        # Separate data into east and west components based on naming convention
        east_data = group[group['component'].str.startswith('east', na=False)]
        west_data = group[group['component'].str.startswith('west', na=False)]
        
        # Skip observation if no valid jet components are found
        if east_data.empty and west_data.empty: 
            continue

        # Convert DataFrame rows into a standardized dictionary format for the physics module
        blob_data_list = []
        def row_to_blob(r):
            return {
                'mjd_obs': r['mjd'], 'comp': r['component'], 
                'pa_obs': r['PA'], 'rad_obs': r['radius'], 
                'rad_err_U': r['radius_plus_err'], 'rad_err_L': r['radius_minus_err'], 
                'pa_err_U': r['PA_err_plus'], 'pa_err_L': r['PA_err_minus']
            }

        for _, row in east_data.iterrows(): blob_data_list.append(row_to_blob(row))
        for _, row in west_data.iterrows(): blob_data_list.append(row_to_blob(row))

        # Perform jet physics fitting and kinematic calculations
        analysis_results = fit_and_calculate_jets(blob_data_list, ss433_params)

        if analysis_results['success']:
            # Generate diagnostic plots if the fit was successful
            plot_fit_and_calc_results(obs_id, blob_data_list, analysis_results, ss433_params, pdf_object=pdf_pages)
            
            # Extract results for both jet sides and format them for the final table
            for jet_side in ['east', 'west']:
                jet_entries = analysis_results['jets'][jet_side] 
                candidates = analysis_results[f'{jet_side}_candidates']
                
                for i, result_info in enumerate(jet_entries):
                    blob = candidates[i]
                    
                    row = {
                        'obs_id': obs_id,
                        'mjd': analysis_results['mjd_obs'],
                        'jet_side': jet_side,
                        'component_name': blob['comp'],
                        'method': result_info['method'],
                        'beta': np.nan,
                        'beta_err_pos': np.nan,
                        'beta_err_neg': np.nan,
                        'travel_time_days': np.nan
                    }

                    # Populate beta velocity values
                    beta_val = result_info.get('fitted_beta')
                    row['beta'] = beta_val
                    
                    # Calculate asymmetric errors for beta if bounds are available
                    lower = result_info.get('beta_lower_bound')
                    upper = result_info.get('beta_upper_bound')
                    if pd.notna(lower) and pd.notna(upper):
                        row['beta_err_neg'] = beta_val - lower
                        row['beta_err_pos'] = upper - beta_val
                        
                    # Calculate the theoretical travel time based on the geometric model
                    fitted_point = _get_closest_geometric_point(blob, jet_side, analysis_results, ss433_params)
                    row['travel_time_days'] = fitted_point.get('model_age')
                    
                    all_results_data.append(row)
        else:
            print(f"--> processing failed for {obs_id}: {analysis_results.get('message', 'n/a')}")

    # Finalize PDF output and report completion
    pdf_pages.close()
    print(f"\nall plots saved to '{config.get_rel_path(pdf_output_path)}'")
    
    return pd.DataFrame(all_results_data)

if __name__ == "__main__":
    # Load command line arguments and update global configuration
    args = get_pipeline_args()
    config.update_config_from_args(args)
    
    # Run tracker analysis first to get component positions
    tracker_df = track_components.run_tracker_analysis()
    
    # Proceed to kinematic analysis only if valid tracker data exists
    if tracker_df is not None and not tracker_df.empty:
        run_kinematic_analysis(tracker_df)