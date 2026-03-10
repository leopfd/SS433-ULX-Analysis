import os
import numpy as np

# Default configuration values used as fallbacks if no command line arguments are provided
BASE_DIR = '/Users/leodrake/Documents/ss433/HRC_2024'
NUM_COMPS = 4
SIGMA_VAL = 1
BIN_SIZE = 0.25
RUN_MCMC = True
RECALC_CHAINS = False 
MCMC_ITER = 500
MCMC_BALL = 1e-4
AUTO_STOP = True
MCMC_PROGRESS_STEP = 1000
MCMC_PROGRESS_TARGET_UPDATES = 200
SIGNIFIERS = ['mcmc'] 
CHAIN_SIGNIFIERS = ['mcmc']
EPHEM_CHOICE = 'simple'
OBS_SELECTION = None
NUM_COMPS_BY_OBS = {}

# Global variables for file paths and IDs that will be dynamically generated later
FILE_ID = ""
FITS_DIR = ""

# Subdirectory paths to be created within the main FITS directory
DIR_LOGS_FULL = ""
DIR_LOGS_MULTI = ""
DIR_PLOTS_FULL = ""   
DIR_PLOTS_MULTI = ""  
DIR_CHAINS = ""
DIR_TRACKER = ""
DIR_DATA = ""
DIR_TRACKER_PLOTS = ""
DIR_JET_PLOTS = ""

# Specific file paths for logging and output
FULL_LOG_TXT = ""       
MULTI_LOG_TXT = ""      
FIT_PLOT_PDF = ""       
MULTI_FIT_PDF = ""      
TRACKER_TABLE_CSV = ""
PLOT_OUTPUT_PDF = ""

# Swift and External Data configuration
SWIFT_FILE = "/Users/leodrake/Documents/ss433/swift-hrc-data.txt"
HRC_SCALE_FACTOR = 50.0

# Physics constants and unit conversions
CENTER_PIXEL = None 
G1_COMPONENT = 'core'
D_SS433_PC = 5500.0
C_PC_PER_DAY = (299792.458 * 86400) / (3.08567758 * 10**13)
ARCSEC_PER_RADIAN = (180.0 / np.pi) * 3600.0
EMP_PSF_FILE = os.path.join(BASE_DIR, 'empPSF_iARLac_v2025_2017-2025.fits')

# Mapping of Observation IDs to their Right Ascension and Declination strings
OBSID_COORDS = {
    "26568": ("287.9565362", "4.9826061"),
    "26569": ("287.9563218", "4.9827745"),
    "26570": ("287.9563754", "4.9825322"),
    "26571": ("287.9561693", "4.9827006"),
    "26572": ("287.9565032", "4.9826636"),
    "26573": ("287.9565444", "4.9826390"),
    "26574": ("287.9562518", "4.9825651"),
    "26575": ("287.9566969", "4.9828114"),
    "26576": ("287.9566351", "4.9826718"),
    "26577": ("287.9565238", "4.9826020"),
    "26578": ("287.9566021", "4.9826800"),
    "26579": ("287.9565733", "4.9825774")
}

# Ephemeris model definitions
EPHEMERIS = {} 
EPHEM_SIMPLE = {
    'model_type': 'simple',
    # Precession terms adopted in Yusuke et al. (2026), Table 1:
    # basis from Gies et al. (2002) with updated t0,prec from this work.
    'jd0_precession': 2400000.5 + 51467.67,
    'precession_period': 162.15,
    'beta': 0.2602,
    'theta': np.radians(19.85),
    'inclination': np.radians(78.83),
    # Not specified in Yusuke et al. (2026) Table 1; kept as pipeline convention.
    'prec_pa': np.radians(10.0)
}

# Full ephemeris model including nutation and orbital parameters
EPHEM_FULL = {
    **EPHEM_SIMPLE,
    'model_type': 'full',  # override simple
    # Use t0,prec as phase reference for the full model.
    'phi0': 0.0,
    # Nutation terms from Davydov et al. (2008), with updated t0,nut and A_nut
    # reported in Yusuke et al. (2026), Table 1.
    'jd0_nut': 2400000.5 + 43032.423,
    'nut_period': 6.287599,
    'nut_ampl': 0.00689349,
    # Orbital terms from Cherepashchuk et al. (2023), as adopted in Yusuke et al. (2026).
    'jd0_orb': 2400000.5 + 51737.04,
    'orbital_period': 13.08250,
    'orbital_period_dot_ss': 1.14e-7,
    # Not tabulated by Yusuke et al. (2026); retained for this model formulation.
    'beta_orb_ampl': 0.004,
    'beta_orb_phase0': np.pi 
}


def parse_components_by_obs(spec):
    """
    Parse per observation component overrides.

    Expected format:
      OBS:COMPS[,OBS:COMPS...]

    OBS can be a single integer (e.g. 26569) or an inclusive range
    (e.g. 26569-26572).
    """
    mapping = {}
    if spec is None:
        return mapping

    entries = [part.strip() for part in str(spec).split(",") if part.strip()]
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                f"invalid --comps-per-obs entry '{entry}': expected OBS:COMPS "
                "(example: 26569:4 or 26569-26572:5)"
            )

        obs_part, comp_part = [x.strip() for x in entry.split(":", 1)]
        if not obs_part:
            raise ValueError(f"invalid --comps-per-obs entry '{entry}': missing observation selector")

        try:
            n_comp = int(comp_part)
        except ValueError as exc:
            raise ValueError(
                f"invalid component count '{comp_part}' in --comps-per-obs entry '{entry}'"
            ) from exc
        if n_comp < 1:
            raise ValueError(
                f"invalid component count '{comp_part}' in --comps-per-obs entry '{entry}': must be >= 1"
            )

        if "-" in obs_part:
            bounds = [x.strip() for x in obs_part.split("-", 1)]
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(
                    f"invalid observation range '{obs_part}' in --comps-per-obs entry '{entry}'"
                )
            try:
                start = int(bounds[0])
                end = int(bounds[1])
            except ValueError as exc:
                raise ValueError(
                    f"invalid observation range '{obs_part}' in --comps-per-obs entry '{entry}'"
                ) from exc
            if end < start:
                raise ValueError(
                    f"invalid observation range '{obs_part}' in --comps-per-obs entry '{entry}': end < start"
                )
            obs_ids = [str(obs) for obs in range(start, end + 1)]
        else:
            try:
                obs_ids = [str(int(obs_part))]
            except ValueError as exc:
                raise ValueError(
                    f"invalid observation id '{obs_part}' in --comps-per-obs entry '{entry}'"
                ) from exc

        for obs_id in obs_ids:
            mapping[obs_id] = n_comp

    return mapping

def format_components_by_obs_for_id(mapping):
    """
    Compact per-observation component overrides for FILE_ID.

    Example:
      26568:3,26569:2,26570:2,26571:2,26573:4 -> 2c_697071-3c_68-4c_73
    """
    if not mapping:
        return ""

    grouped = {}
    for obs, comp in mapping.items():
        comp_int = int(comp)
        obs_int = int(obs)
        grouped.setdefault(comp_int, []).append(obs_int)

    segments = []
    for comp_int in sorted(grouped.keys()):
        obs_suffixes = "".join(f"{obs_int % 100:02d}" for obs_int in sorted(grouped[comp_int]))
        segments.append(f"{comp_int}c_{obs_suffixes}")

    return "-".join(segments)

def update_config_from_args(args=None):
    # Bring in global variables to update them based on arguments or derived calculations
    global NUM_COMPS, SIGMA_VAL, BIN_SIZE, RUN_MCMC, RECALC_CHAINS
    global MCMC_ITER, MCMC_BALL, AUTO_STOP, SIGNIFIERS, CHAIN_SIGNIFIERS, EPHEM_CHOICE, BASE_DIR
    global FILE_ID, FITS_DIR, CENTER_PIXEL
    global DIR_LOGS_FULL, DIR_LOGS_MULTI, DIR_PLOTS_FULL, DIR_PLOTS_MULTI, DIR_CHAINS, DIR_TRACKER, DIR_DATA, DIR_TRACKER_PLOTS, DIR_JET_PLOTS
    global FULL_LOG_TXT, MULTI_LOG_TXT, FIT_PLOT_PDF, MULTI_FIT_PDF
    global TRACKER_TABLE_CSV, PLOT_OUTPUT_PDF
    global EMP_PSF_FILE, EPHEMERIS
    global SWIFT_FILE, HRC_SCALE_FACTOR
    global OBS_SELECTION, NUM_COMPS_BY_OBS

    # Update configuration if command line arguments are present
    if args:
        if args.base_dir: BASE_DIR = args.base_dir
        NUM_COMPS = args.comps
        NUM_COMPS_BY_OBS = parse_components_by_obs(getattr(args, "comps_per_obs", None))
        SIGMA_VAL = args.sigma
        BIN_SIZE = args.bin
        RUN_MCMC = not args.no_mcmc
        RECALC_CHAINS = args.recalc
        MCMC_ITER = args.steps
        MCMC_BALL = args.ball
        AUTO_STOP = args.auto_stop
        SIGNIFIERS = args.sigs.copy()
        CHAIN_SIGNIFIERS = args.sigs.copy()
        EPHEM_CHOICE = args.ephem
        OBS_SELECTION = args.obs
    else:
        NUM_COMPS_BY_OBS = {}
        CHAIN_SIGNIFIERS = ['mcmc']

    if NUM_COMPS < 1:
        raise ValueError(f"--comps must be >= 1 (got {NUM_COMPS})")

    # Format the step count string for filename inclusion using k notation for large numbers
    if MCMC_ITER > 9999:
        step_str = f"{int(MCMC_ITER/1000)}k"
    else:
        step_str = str(MCMC_ITER)

    if step_str not in SIGNIFIERS:
        SIGNIFIERS.append(step_str)

    # Select the appropriate ephemeris dictionary based on user choice
    if EPHEM_CHOICE == 'full':
        EPHEMERIS = EPHEM_FULL
    else:
        EPHEMERIS = EPHEM_SIMPLE

    # Parse observation selection string to create a compact identifier for the filename
    if OBS_SELECTION:
        parts = OBS_SELECTION.split(',')
        short_parts = []
        for p in parts:
            p = p.strip()
            if '-' in p:
                start, end = p.split('-')
                short_parts.append(f"{start.strip()[-2:]}--{end.strip()[-2:]}")
            else:
                short_parts.append(p[-2:])
        
        # Add to signifiers list so it becomes part of the hyphenated ID
        SIGNIFIERS.append("_".join(short_parts))

    # Define directory structure relative to the base directory
    FITS_DIR = os.path.join(BASE_DIR, '2Dfits')
    DIR_LOGS_FULL = os.path.join(FITS_DIR, 'fit results')
    DIR_LOGS_MULTI = os.path.join(FITS_DIR, 'multi comp fit results')
    DIR_PLOTS_FULL = os.path.join(FITS_DIR, 'fit plots') 
    DIR_PLOTS_MULTI = os.path.join(FITS_DIR, 'multi comp fit plots')
    DIR_CHAINS = os.path.join(FITS_DIR, 'emcee_chains')
    DIR_TRACKER = os.path.join(FITS_DIR, 'comp tracker tables')
    DIR_DATA = os.path.join(FITS_DIR, 'data tables')
    DIR_TRACKER_PLOTS = os.path.join(FITS_DIR, 'comp tracker plots')
    DIR_JET_PLOTS = os.path.join(FITS_DIR, 'jet plots')

    # Ensure all directories exist
    for d in [FITS_DIR, DIR_LOGS_FULL, DIR_LOGS_MULTI, DIR_PLOTS_FULL, DIR_PLOTS_MULTI, DIR_CHAINS, DIR_TRACKER, DIR_DATA, DIR_TRACKER_PLOTS, DIR_JET_PLOTS]:
        os.makedirs(d, exist_ok=True)

    # Calculate center pixel coordinate based on logical width and bin size
    logical_width = 40.0 / BIN_SIZE 
    CENTER_PIXEL = (logical_width / 2.0) + 0.5

    # Generate the unique File ID string used for naming all output files
    bin_str = str(BIN_SIZE).replace('.', 'p')
    sigma_str = str(SIGMA_VAL) 
    signifiers_str = "-".join(SIGNIFIERS)
    
    if NUM_COMPS_BY_OBS:
        comp_spec = format_components_by_obs_for_id(NUM_COMPS_BY_OBS)
        comps_id = f"mixedcomp-{comp_spec}"
    else:
        comps_id = f"{NUM_COMPS}comp"

    FILE_ID = f"{comps_id}-{sigma_str}sigma-{signifiers_str}-bin{bin_str}"

    # Construct full file paths using the generated File ID
    FULL_LOG_TXT = os.path.join(DIR_LOGS_FULL, f'fit-results-{FILE_ID}.txt')
    MULTI_LOG_TXT = os.path.join(DIR_LOGS_MULTI, f'multi-comp-fit-results-{FILE_ID}.txt')
    FIT_PLOT_PDF = os.path.join(DIR_PLOTS_FULL, f'fit-plots-{FILE_ID}.pdf')
    MULTI_FIT_PDF = os.path.join(DIR_PLOTS_MULTI, f'multi-comp-plots-{FILE_ID}.pdf')
    
    TRACKER_TABLE_CSV = os.path.join(DIR_TRACKER, f'comp-tracker-table-{FILE_ID}.csv')
        
    PLOT_OUTPUT_PDF = os.path.join(DIR_JET_PLOTS, f'ss433-jet-fit-results-{FILE_ID}.pdf')

    if args:
        print(f"\nCONFIGURATION UPDATE")
        if NUM_COMPS_BY_OBS:
            items = sorted(NUM_COMPS_BY_OBS.items(), key=lambda kv: int(kv[0]))
            preview_items = items[:8]
            preview = ", ".join(f"{obs}:{n}" for obs, n in preview_items)
            if len(items) > len(preview_items):
                preview += ", ..."
            print(
                f"  Components: default={NUM_COMPS} with per-obs overrides "
                f"({len(NUM_COMPS_BY_OBS)} obs: {preview})"
            )
        else:
            print(f"  Components: {NUM_COMPS}")
        print(f"  Sigma: {SIGMA_VAL} | Bin: {BIN_SIZE} (Center: {CENTER_PIXEL})")
        print(f"  MCMC: {RUN_MCMC} (Steps: {MCMC_ITER})")
        print(f"  ID String: {FILE_ID}")
        print(f"\n")

def get_rel_path(path):
    # Returns the path relative to BASE_DIR for cleaner printing
    try:
        return os.path.relpath(path, BASE_DIR)
    except ValueError:
        return path

update_config_from_args(None)
