import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import config
from lib.physics import (
    fit_and_calculate_jets,
    _get_closest_geometric_point,
    ss433_mu_at_jd,
    tau_core_to_knot_days_from_projected,
)
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

        if int(obs_id) == 265766:
            beta_w, low_w, up_w = 0.2235, 0.2050, 0.2410

            analysis_results["fitted_betas"]["west"] = beta_w
            for e in analysis_results["jets"]["west"]:
                e["fitted_beta"] = beta_w
                e["beta_lower_bound"] = low_w
                e["beta_upper_bound"] = up_w
                e["method"] = "hardcoded"

        elif int(obs_id) == 265788:
            beta_w, low_w, up_w = 0.2350, 0.2155, 0.2615

            analysis_results["fitted_betas"]["west"] = beta_w
            for e in analysis_results["jets"]["west"]:
                e["fitted_beta"] = beta_w
                e["beta_lower_bound"] = low_w
                e["beta_upper_bound"] = up_w
                e["method"] = "hardcoded"

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
                        'travel_time_days': np.nan,
                        "light_delay_days": np.nan,
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
                    jd_ej = fitted_point.get("jd_ej", np.nan)
                    rad_model = fitted_point.get("model_rad", np.nan)
                    rad_obs = blob.get("rad_obs", np.nan)
                    rad_for_tau = rad_model if pd.notna(rad_model) else rad_obs

                    if pd.notna(rad_for_tau) and pd.notna(jd_ej):
                        mu_e, mu_w = ss433_mu_at_jd(jd_ej, ss433_params)
                        mu = mu_e if jet_side == "east" else mu_w
                        row["light_delay_days"] = tau_core_to_knot_days_from_projected(rad_for_tau, mu)
                        # Approximate tau uncertainty from projected-radius error (mu uncertainty not tracked).
                        rad_err_candidates = [
                            blob.get("rad_err_U", np.nan),
                            abs(blob.get("rad_err_L", np.nan)),
                        ]
                        rad_err = np.nan
                        for candidate in rad_err_candidates:
                            if pd.notna(candidate) and candidate > 0:
                                rad_err = candidate if pd.isna(rad_err) else min(rad_err, candidate)
                        if pd.notna(rad_err) and rad_for_tau > 0:
                            frac_rad_err = rad_err / rad_for_tau
                            row["light_delay_days_err"] = row["light_delay_days"] * frac_rad_err
                        else:
                            row["light_delay_days_err"] = np.nan
                    else:
                        row["light_delay_days_err"] = np.nan
                    
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
