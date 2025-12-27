import argparse
import sys

def get_pipeline_args():
    parser = argparse.ArgumentParser(description="SS433 Analysis Pipeline")

    # data selection
    parser.add_argument("--obs", type=str, help="observation ids to process (e.g. 26569, 26569-26572)")

    # fitting parameters
    parser.add_argument("--comps", type=int, default=3, help="number of components for multi-fit (default: 3)")
    parser.add_argument("--sigma", type=int, default=1, choices=[1, 2, 3], help="sigma confidence interval (1, 2, or 3) (default: 1)")
    parser.add_argument("--bin", type=float, default=0.25, help="bin size (default: 0.25)")
    
    # mcmc control
    parser.add_argument("--no-mcmc", action="store_true", help="disable mcmc (run fast fit only)")
    parser.add_argument("--recalc", action="store_true", help="force recalculation of mcmc chains")
    parser.add_argument("--steps", type=int, default=500, help="mcmc iterations (default: 500)")
    parser.add_argument("--ball", type=float, default=1e-4, help="mcmc ball size (default: 1e-4)")
    parser.add_argument("--auto-stop", action="store_true", help="stop mcmc automatically when convergence is reached")

    # ephemeris and labels
    parser.add_argument("--ephem", type=str, default="simple", choices=["simple", "full"], help="ephemeris model: 'simple' or 'full' (default: simple)")
    parser.add_argument("--sigs", nargs="+", default=['mcmc'], 
                        help="list of signifiers for filenames (default: 'mcmc' - steps added auto)")

    # file paths
    parser.add_argument("--base-dir", type=str, help="override base directory")
    
    return parser.parse_args()