import time
import sys
import os
import config

import fit_images
import track_components
import model_kinematics
import lib.swift_compare

def run():
    print(f"starting ss433 analysis pipeline...")
    print(f"id: {config.FILE_ID}\n")

    start_time = time.time()

    # stage 1: fit images
    print("\n=== stage 1: fitting images ===\n")
    fit_images.run_pipeline()
    
    # stage 2: track components
    print("\n=== stage 2: tracking components ===\n")
    # capture the dataframe returned by the tracker
    tracker_df = track_components.run_tracker_analysis()

    if tracker_df is None or tracker_df.empty:
        print("critical error: stage 2 returned no data. aborting.")
        sys.exit(1)

    # stage 3: kinematic fitting
    print("\n=== stage 3: kinematic fitting ===\n")
    # pass tracker_df into the kinematic fitter
    ejection_df = model_kinematics.run_kinematic_analysis(tracker_df)
    
    if ejection_df is None or ejection_df.empty:
        print("critical error: stage 3 returned no results. aborting.")
        sys.exit(1)

    # stage 4: swift comparison
    print("\n=== stage 4: swift comparison ===\n")
    lib.swift_compare.plot_swift_comparison(tracker_df, ejection_df)

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\npipeline finished in {total_time:.2f} minutes.")
    print(f"\n=== process complete ===")

if __name__ == "__main__":
    from lib.arguments import get_pipeline_args
    args = get_pipeline_args()
    config.update_config_from_args(args)
    
    run()