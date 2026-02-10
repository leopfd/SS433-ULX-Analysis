import time
import sys
import os
import config

import fit_images
import track_components
import model_kinematics
import lib.swift_compare


def _normalize_skip_stages(args):
    """Return a set of canonical stage names to skip."""
    if not args or not getattr(args, "skip_stages", None):
        return set()

    aliases = {
        "1": "fit",
        "fit": "fit",
        "fit-images": "fit",
        "images": "fit",
        "fit_images": "fit",
        "stage1": "fit",
        "2": "track",
        "track": "track",
        "tracking": "track",
        "stage2": "track",
        "3": "kinematics",
        "kin": "kinematics",
        "kinematics": "kinematics",
        "stage3": "kinematics",
        "4": "swift",
        "swift": "swift",
        "swift-compare": "swift",
        "compare": "swift",
        "stage4": "swift",
    }

    skip = set()
    for raw in args.skip_stages:
        key = aliases.get(str(raw).lower())
        if not key:
            print(f"\033[1m[pipeline]\033[0m Warning: unknown stage '{raw}' in --skip-stages; ignoring.")
            continue
        skip.add(key)
    return skip


def run(args=None):
    skip_stages = _normalize_skip_stages(args)

    print("\033[1mstarting ss433 analysis pipeline...\033[0m")
    if skip_stages:
        print(f"skipping stages: {', '.join(sorted(skip_stages))}")
    print(f"id: {config.FILE_ID}\n")

    start_time = time.time()

    # Stage 1 Image Fitting
    # Processes raw observation files to fit spatial models and extract component centroids
    if "fit" in skip_stages:
        print("\n\033[1m=== stage 1: fitting images (skipped) ===\033[0m\n")
    else:
        print("\n\033[1m=== stage 1: fitting images ===\033[0m\n")
        fit_images.run_pipeline()
    
    # Stage 2 Component Tracking
    # Aggregates individual fit results to track component movement over time
    if "track" in skip_stages:
        print("\n\033[1m=== stage 2: tracking components (skipped) ===\033[0m\n")
        tracker_df = None
    else:
        print("\n\033[1m=== stage 2: tracking components ===\033[0m\n")
        tracker_df = track_components.run_tracker_analysis()

    needs_tracker = "kinematics" not in skip_stages
    if needs_tracker and (tracker_df is None or tracker_df.empty):
        print("critical error: stage 2 returned no data. aborting.")
        sys.exit(1)

    # Stage 3 Kinematic Fitting
    # Uses the tracked component trajectories to model jet kinematics and ejection dates
    if "kinematics" in skip_stages:
        print("\n\033[1m=== stage 3: kinematic fitting (skipped) ===\033[0m\n")
        ejection_df = None
    else:
        print("\n\033[1m=== stage 3: kinematic fitting ===\033[0m\n")
        ejection_df = model_kinematics.run_kinematic_analysis(tracker_df)
        
        if ejection_df is None or ejection_df.empty:
            print("critical error: stage 3 returned no results. aborting.")
            sys.exit(1)

    # Stage 4 External Data Comparison
    # Compares the calculated kinematic ejection times with external Swift X ray data
    if "swift" in skip_stages:
        print("\n\033[1m=== stage 4: swift comparison (skipped) ===\033[0m\n")
    else:
        if ejection_df is None:
            print("critical error: stage 3 was skipped or failed; cannot run swift comparison.")
            sys.exit(1)
        print("\n\033[1m=== stage 4: swift comparison ===\033[0m\n")
        lib.swift_compare.plot_swift_comparison(tracker_df, ejection_df)

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\n\033[1mpipeline finished in {total_time:.2f} minutes.\033[0m")
    print(f"\n\033[1m=== process complete ===\033[0m\n")

if __name__ == "__main__":
    from lib.arguments import get_pipeline_args
    args = get_pipeline_args()
    config.update_config_from_args(args)
    
    run(args)
