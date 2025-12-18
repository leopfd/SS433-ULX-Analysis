import re
import pandas as pd
import numpy as np
from collections import defaultdict

def load_sherpa_log_to_dataframe(filename):
    # FIX: Added re.IGNORECASE to handle 'observation:' vs 'Observation:'
    obs_id_re = re.compile(r"Observation:\s*(\d+)", re.IGNORECASE)
    date_re = re.compile(r"Date:\s*([\d\.]+).*?Exptime:\s*([\d\.]+)", re.IGNORECASE)
    
    # Standard confidence bounds
    conf_re = re.compile(r"^\s*(g\d+|c\d+)\.(?P<param>[a-zA-Z0-9]+)\s+(?P<val>[-\d.eE]+)\s+(?P<low>[-\d.eE]+|-------)\s+(?P<up>[-\d.eE]+|-------)", re.M)
    
    # MCMC results regex
    mcmc_re = re.compile(r"^\s*(g\d+|c\d+)\.(?P<param>[a-zA-Z0-9]+)\s+(?P<val>[-\d.eE]+)\s+(?:[-\d.eE]+)\s+(?P<low>[-\d.eE]+)\s+(?P<up>[-\d.eE]+)", re.M)
    
    # Rate line regex
    rate_line_re = re.compile(r"^\s*(g\d+|c\d+)\s*:\s*(?P<val>[-\d.eE]+)\s*\((?P<low>[-\d.eE+]+)\s*/\s*(?P<up>[-\d.eE+]+)\)", re.M)

    try:
        with open(filename, 'r') as f:
            raw_text = f.read() 
    except FileNotFoundError:
        print(f"error: file not found at {filename}")
        return pd.DataFrame()

    # FIX: Added (?i) flag to make the split case-insensitive
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
        
        if "emcee Results" in block or "emcee results" in block:
            target_re = mcmc_re
        else:
            target_re = conf_re
            
        for match in target_re.finditer(block):
            c_id = match.group(1)
            param = match.group('param')
            val = float(match.group('val'))
            low_str = match.group('low')
            up_str = match.group('up')
            comps[c_id][param] = (val, low_str, up_str)

        rates = {}
        for match in rate_line_re.finditer(block):
            c_id = match.group(1)
            val = float(match.group('val'))
            low_val = float(match.group('low'))
            up_val = float(match.group('up'))
            
            if low_val < 0:
                minus_err = abs(low_val)
                plus_err = abs(up_val)
            else:
                minus_err = abs(up_val)
                plus_err = abs(low_val)
            rates[c_id] = (val, minus_err, plus_err)

        g_ids = [k for k in comps.keys() if k.startswith('g')]
        if not g_ids: continue 

        def get_val(cid, p): return comps[cid].get(p, (0,0,0))[0]

        # Logic to identify components (Core vs Jets)
        # Assumes the component with max Amplitude is the Core
        core_id = max(g_ids, key=lambda c: get_val(c, 'ampl'))
        core_x = get_val(core_id, 'xpos')
        
        mapping = {core_id: 'core'}
        
        remaining = [c for c in g_ids if c != core_id]
        left_cands = [c for c in remaining if get_val(c, 'xpos') < core_x]
        right_cands = [c for c in remaining if get_val(c, 'xpos') >= core_x]
        
        extras = []

        if left_cands:
            left_sorted = sorted(left_cands, key=lambda c: get_val(c, 'xpos'), reverse=True)
            mapping[left_sorted[0]] = 'east'
            if len(left_sorted) > 1:
                extras.extend(left_sorted[1:])
        
        if right_cands:
            right_sorted = sorted(right_cands, key=lambda c: get_val(c, 'xpos'))
            mapping[right_sorted[0]] = 'west'
            if len(right_sorted) > 1:
                extras.extend(right_sorted[1:])
                
        extras_sorted = sorted(extras, key=lambda c: get_val(c, 'xpos'))
        for i, eid in enumerate(extras_sorted, start=1):
            mapping[eid] = f'extra_{i}'
        
        for c_id in comps:
            if c_id.startswith('c'):
                mapping[c_id] = 'bkg'

        for old_id, new_name in mapping.items():
            if old_id not in comps: continue
            
            row = {
                'obs_id': obs_id,
                'mjd': mjd,
                'exptime': exptime,
                'component': new_name
            }
            
            for param, (val, low_s, up_s) in comps[old_id].items():
                row[param] = val
                row[f'{param}_minus'] = low_s
                row[f'{param}_plus'] = up_s
            
            if old_id in rates:
                r_val, r_min, r_plus = rates[old_id]
                row['nominal'] = r_val
                row['minus_err'] = r_min
                row['plus_err'] = r_plus
            else:
                row['nominal'] = np.nan
                row['minus_err'] = np.nan
                row['plus_err'] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    
    if df.empty:
        return df

    err_cols = [c for c in df.columns if c.endswith('_minus') or c.endswith('_plus')]
    for col in err_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').abs()
    
    if df['mjd'].isna().all() and 'obs_id' in df.columns:
        df['mjd'] = df['obs_id'].astype('category').cat.codes

    return df