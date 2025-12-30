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

    # Stage 1 Image Fitting
    # Processes raw observation files to fit spatial models and extract component centroids
    print("\n=== stage 1: fitting images ===\n")
    fit_images.run_pipeline()
    
    # Stage 2 Component Tracking
    # Aggregates individual fit results to track component movement over time
    print("\n=== stage 2: tracking components ===\n")
    tracker_df = track_components.run_tracker_analysis()

    if tracker_df is None or tracker_df.empty:
        print("critical error: stage 2 returned no data. aborting.")
        sys.exit(1)

    # Stage 3 Kinematic Fitting
    # Uses the tracked component trajectories to model jet kinematics and ejection dates
    print("\n=== stage 3: kinematic fitting ===\n")
    ejection_df = model_kinematics.run_kinematic_analysis(tracker_df)
    
    if ejection_df is None or ejection_df.empty:
        print("critical error: stage 3 returned no results. aborting.")
        sys.exit(1)

    # Stage 4 External Data Comparison
    # Compares the calculated kinematic ejection times with external Swift X ray data
    print("\n=== stage 4: swift comparison ===\n")
    lib.swift_compare.plot_swift_comparison(tracker_df, ejection_df)

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\npipeline finished in {total_time:.2f} minutes.")
    print(f"\n=== process complete ===\n")

if __name__ == "__main__":
    from lib.arguments import get_pipeline_args
    args = get_pipeline_args()
    config.update_config_from_args(args)
    
    run()