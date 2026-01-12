import argparse
import sys

def get_pipeline_args():
    # Initialize the argument parser with a description of the tool
    parser = argparse.ArgumentParser(description="SS433 Analysis Pipeline")

    # Command line arguments for selecting specific data ranges
    parser.add_argument("--obs", type=str, help="observation ids to process (e.g. 26569, 26569-26572)")

    # Arguments defining the core fitting model parameters and image resolution
    parser.add_argument("--comps", type=int, default=3, help="number of components for multi-fit (default: 3)")
    parser.add_argument("--sigma", type=int, default=1, choices=[1, 2, 3], help="sigma confidence interval (1, 2, or 3) (default: 1)")
    parser.add_argument("--bin", type=float, default=0.25, help="bin size (default: 0.25)")
    
    # Flags and settings to control the MCMC sampling behavior and performance
    parser.add_argument("--no-mcmc", action="store_true", help="disable mcmc (run fast fit only)")
    parser.add_argument("--recalc", action="store_true", help="force recalculation of mcmc chains")
    parser.add_argument("--steps", type=int, default=500, help="mcmc iterations (default: 500)")
    parser.add_argument("--ball", type=float, default=1e-4, help="mcmc ball size (default: 1e-4)")
    parser.add_argument("--auto-stop", action="store_true", help="stop mcmc automatically when convergence is reached")

    # Configuration for physical models and output file labeling conventions
    parser.add_argument("--ephem", type=str, default="simple", choices=["simple", "full"], help="ephemeris model: 'simple' or 'full' (default: simple)")
    parser.add_argument("--sigs", nargs="+", default=['mcmc'], 
                        help="list of signifiers for filenames (default: 'mcmc' - steps added auto)")

    parser.add_argument(
        "--skip-stages",
        "--skip",
        nargs="+",
        default=[],
        dest="skip_stages",
        metavar="STAGE",
        help="Stages to skip (accepts 1/fit, 2/track, 3/kinematics, 4/swift).",
    )

    # System configuration overrides
    parser.add_argument("--base-dir", type=str, help="override base directory")
    
    return parser.parse_args()
