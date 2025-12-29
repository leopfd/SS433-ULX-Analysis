import re
import pandas as pd
import numpy as np
from collections import defaultdict

def load_sherpa_log_to_dataframe(filename):
    # Compile regex patterns to extract observation metadata and fit statistics from the log file
    obs_id_re = re.compile(r"Observation:\s*(\d+)", re.IGNORECASE)
    date_re = re.compile(r"Date:\s*([\d\.]+).*?Exptime:\s*([\d\.]+)", re.IGNORECASE)
    conf_re = re.compile(r"^\s*(g\d+|c\d+)\.(?P<param>[a-zA-Z0-9]+)\s+(?P<val>[-\d.eE]+)\s+(?P<low>[-\d.eE]+|-------)\s+(?P<up>[-\d.eE]+|-------)", re.M)
    mcmc_re = re.compile(r"^\s*(g\d+|c\d+)\.(?P<param>[a-zA-Z0-9]+)\s+(?P<val>[-\d.eE]+)\s+(?:[-\d.eE]+)\s+(?P<low>[-\d.eE]+)\s+(?P<up>[-\d.eE]+)", re.M)
    rate_line_re = re.compile(r"^\s*(g\d+|c\d+)\s*:\s*(?P<val>[-\d.eE]+)\s*\((?P<low>[-\d.eE+]+)\s*/\s*(?P<up>[-\d.eE+]+)\)", re.M)

    try:
        with open(filename, 'r') as f:
            raw_text = f.read() 
    except FileNotFoundError:
        print(f"error: file not found at {filename}")
        return pd.DataFrame()

    # Split the raw log text into blocks corresponding to individual observations
    obs_blocks = re.split(r'(?i)(?=Observation:)', raw_text)
    rows = []
    
    for block in obs_blocks:
        if not block.strip(): continue

        obs_match = obs_id_re.search(block)
        if not obs_match: continue
        obs_id = int(obs_match.group(1))
        
        date_match = date_re.search(block)
        if date_match:
            mjd, exptime = float(date_match.group(1)), float(date_match.group(2))
        else:
            mjd, exptime = (np.nan, np.nan)

        comps = defaultdict(dict)
        # Determine which regex to use based on whether MCMC results are present in the block
        target_re = mcmc_re if ("emcee Results" in block or "emcee results" in block) else conf_re
            
        for match in target_re.finditer(block):
            comps[match.group(1)][match.group('param')] = (float(match.group('val')), match.group('low'), match.group('up'))

        rates = {}
        for match in rate_line_re.finditer(block):
            rates[match.group(1)] = (float(match.group('val')), float(match.group('low')), float(match.group('up')))

        g_ids = [k for k in comps.keys() if k.startswith('g')]
        if not g_ids: continue 

        def get_val(cid, p): return comps[cid].get(p, (0,0,0))[0]

        # Identify the core component by finding the Gaussian with the maximum amplitude
        core_id = max(g_ids, key=lambda c: get_val(c, 'ampl'))
        core_x = get_val(core_id, 'xpos')
        core_y = get_val(core_id, 'ypos')
        
        mapping = {core_id: 'core'}
        
        # Classify components as East or West based on their position angle relative to the core
        east_cands = []
        west_cands = []
        others = []
        
        EAST_MIN, EAST_MAX = 68, 128
        WEST_MIN, WEST_MAX = 248, 308
        
        for cid in g_ids:
            if cid == core_id: continue
            
            x = get_val(cid, 'xpos')
            y = get_val(cid, 'ypos')
            
            # Calculate position angle accounting for Right Ascension increasing to the left
            dx = x - core_x
            dy = y - core_y
            pa_deg = np.degrees(np.arctan2(-dx, dy))
            
            dist = np.hypot(dx, dy)
            
            # Normalize angle to 0 to 360 for easier comparison
            pa_norm = pa_deg % 360
            
            is_east = (EAST_MIN <= pa_norm <= EAST_MAX)
            is_west = (WEST_MIN <= pa_norm <= WEST_MAX)
            
            if is_east:
                east_cands.append((cid, dist))
            elif is_west:
                west_cands.append((cid, dist))
            else:
                others.append((cid, dist))

        # Sort candidate components by their distance from the core so index 1 is closest
        east_cands.sort(key=lambda x: x[1])
        west_cands.sort(key=lambda x: x[1])
        others.sort(key=lambda x: x[1])
        
        for i, (cid, _) in enumerate(east_cands, 1):
            mapping[cid] = f'east_{i}'
        for i, (cid, _) in enumerate(west_cands, 1):
            mapping[cid] = f'west_{i}'
        for i, (cid, _) in enumerate(others, 1):
            mapping[cid] = f'other_{i}'

        for c_id in comps:
            if c_id.startswith('c'): mapping[c_id] = 'bkg'

        # Iterate through the mapped components to construct the final data rows
        for old_id, new_name in mapping.items():
            if old_id not in comps: continue
            
            row = {'obs_id': obs_id, 'mjd': mjd, 'exptime': exptime, 'component': new_name}
            
            for param, (val, low_s, up_s) in comps[old_id].items():
                row[param] = val
                row[f'{param}_minus'] = low_s
                row[f'{param}_plus'] = up_s
            
            if old_id in rates:
                r_val, low_raw, up_raw = rates[old_id]
                # Normalize error bars to ensure they are positive absolute differences
                row['nominal'] = r_val
                row['minus_err'] = abs(low_raw) if low_raw < 0 else abs(r_val - low_raw)
                row['plus_err'] = abs(up_raw) if up_raw > 0 else abs(up_raw - r_val)
            else:
                row['nominal'] = np.nan; row['minus_err'] = np.nan; row['plus_err'] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty: return df

    err_cols = [c for c in df.columns if c.endswith('_minus') or c.endswith('_plus')]
    for col in err_cols: df[col] = pd.to_numeric(df[col], errors='coerce').abs()
    
    if df['mjd'].isna().all() and 'obs_id' in df.columns:
        df['mjd'] = df['obs_id'].astype('category').cat.codes

    return df