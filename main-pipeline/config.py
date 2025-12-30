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
SIGNIFIERS = ['mcmc'] 
EPHEM_CHOICE = 'simple'
OBS_SELECTION = None

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
    'jd0_precession': 2400000.5 + 59898.78, 
    'jd0_precession_err': 0.4, 
    'precession_period': 160.14,
    'precession_period_err': 0.17,
    'beta': 0.2591, 
    'beta_err': 0.0007,
    'theta': np.radians(19.64), 
    'theta_err': np.radians(0.10),
    'inclination': np.radians(78.92), 
    'inclination_err': np.radians(0.06),
    'prec_pa': np.radians(10.0), 
    'prec_pa_err': 0.0 
}

# Full ephemeris model including nutation and orbital parameters
EPHEM_FULL = {
    'model_type': 'full',
    **EPHEM_SIMPLE,
    'phi0': np.radians(-49.0), 
    'jd0_nut': 2400000.5 + 59797.68,
    'jd0_nut_err': 0.09,
    'nut_period': 6.28802,
    'nut_period_err': 0.00005,
    'nut_ampl': np.radians(0.0063 * (180/np.pi)), 
    'nut_ampl_err': np.radians(0.0003 * (180/np.pi)), 
    'jd0_orb': 2460503.14,
    'orbital_period': 13.082989,
    'beta_orb_ampl': 0.004,
    'beta_orb_phase0': np.pi 
}

def update_config_from_args(args=None):
    # Bring in global variables to update them based on arguments or derived calculations
    global NUM_COMPS, SIGMA_VAL, BIN_SIZE, RUN_MCMC, RECALC_CHAINS
    global MCMC_ITER, MCMC_BALL, AUTO_STOP, SIGNIFIERS, EPHEM_CHOICE, BASE_DIR
    global FILE_ID, FITS_DIR, CENTER_PIXEL
    global DIR_LOGS_FULL, DIR_LOGS_MULTI, DIR_PLOTS_FULL, DIR_PLOTS_MULTI, DIR_CHAINS, DIR_TRACKER, DIR_DATA, DIR_TRACKER_PLOTS, DIR_JET_PLOTS
    global FULL_LOG_TXT, MULTI_LOG_TXT, FIT_PLOT_PDF, MULTI_FIT_PDF
    global TRACKER_TABLE_CSV, PLOT_OUTPUT_PDF
    global EMP_PSF_FILE, EPHEMERIS
    global SWIFT_FILE, HRC_SCALE_FACTOR
    global OBS_SELECTION

    # Update configuration if command line arguments are present
    if args:
        if args.base_dir: BASE_DIR = args.base_dir
        NUM_COMPS = args.comps
        SIGMA_VAL = args.sigma
        BIN_SIZE = args.bin
        RUN_MCMC = not args.no_mcmc
        RECALC_CHAINS = args.recalc
        MCMC_ITER = args.steps
        MCMC_BALL = args.ball
        AUTO_STOP = args.auto_stop
        SIGNIFIERS = args.sigs.copy() 
        EPHEM_CHOICE = args.ephem
        OBS_SELECTION = args.obs

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
    
    FILE_ID = f"{NUM_COMPS}comp-{sigma_str}sigma-{signifiers_str}-bin{bin_str}"

    # Construct full file paths using the generated File ID
    FULL_LOG_TXT = os.path.join(DIR_LOGS_FULL, f'fit-results-{FILE_ID}.txt')
    MULTI_LOG_TXT = os.path.join(DIR_LOGS_MULTI, f'multi-comp-fit-results-{FILE_ID}.txt')
    FIT_PLOT_PDF = os.path.join(DIR_PLOTS_FULL, f'fit-plots-{FILE_ID}.pdf')
    MULTI_FIT_PDF = os.path.join(DIR_PLOTS_MULTI, f'multi-comp-plots-{FILE_ID}.pdf')
    
    TRACKER_TABLE_CSV = os.path.join(DIR_TRACKER, f'comp-tracker-table-{FILE_ID}.csv')
        
    PLOT_OUTPUT_PDF = os.path.join(DIR_JET_PLOTS, f'ss433-jet-fit-results-{FILE_ID}.pdf')

    if args:
        print(f"\nCONFIGURATION UPDATE")
        print(f"  Components: {NUM_COMPS} | Sigma: {SIGMA_VAL} | Bin: {BIN_SIZE} (Center: {CENTER_PIXEL})")
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