import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

def plot_swift_comparison(tracker_df, ejection_df):
    """
    Plots Swift vs HRC comparison using in-memory dataframes.
    Call this AFTER running your kinematic fitting stage (Stage 3).
    """
    
    print("\n[Swift Plot] Generating Comparison Plot...")
    
    if not os.path.exists(config.SWIFT_FILE):
        print(f"Error: Swift data file not found at {config.SWIFT_FILE}")
        return

    # 1. Prepare Ejection Data 
    if 'ejection_mjd' not in ejection_df.columns:
        if 'travel_time_days' in ejection_df.columns:
            ejection_df['ejection_mjd'] = ejection_df['mjd'] - ejection_df['travel_time_days']
        else:
            print("[Swift Plot] Error: 'travel_time_days' missing from ejection dataframe.")
            return

    # Calculate errors if not present
    if 'ejection_mjd_err_pos' not in ejection_df.columns:
        safe_beta = np.where(ejection_df['beta'] != 0, ejection_df['beta'], np.nan)
        ejection_df['ejection_mjd_err_pos'] = (ejection_df['travel_time_days'] * ejection_df['beta_err_pos']) / safe_beta
        ejection_df['ejection_mjd_err_neg'] = (ejection_df['travel_time_days'] * ejection_df['beta_err_neg']) / safe_beta
        ejection_df['ejection_mjd_err_pos'].fillna(0, inplace=True)
        ejection_df['ejection_mjd_err_neg'].fillna(0, inplace=True)

    # 2. Merge Dataframes
    hrc_cols = ['obs_id', 'component', 'nominal', 'minus_err', 'plus_err']
    hrc_data = tracker_df[[c for c in hrc_cols if c in tracker_df.columns]].copy()
    
    ejection_cols = ['obs_id', 'component_name', 'ejection_mjd', 'method', 'ejection_mjd_err_pos', 'ejection_mjd_err_neg']
    if 'component_name' not in ejection_df.columns and 'component' in ejection_df.columns:
         ejection_df['component_name'] = ejection_df['component']
         
    ej_data = ejection_df[[c for c in ejection_cols if c in ejection_df.columns]].copy()

    plot_df = pd.merge(
        hrc_data,
        ej_data,
        left_on=['obs_id', 'component'],
        right_on=['obs_id', 'component_name']
    )

    if plot_df.empty:
        print("[Swift Plot] Warning: Merge resulted in empty dataframe.")
        return

    # 3. Load Swift Data
    swift_list = []
    try:
        with open(config.SWIFT_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("!", "NO", "#")): continue
                tok = line.split()
                if len(tok) < 6: continue
                try:
                    mjd_val, rate, err_low, err_high = map(float, [tok[0], tok[3], tok[4], tok[5]])
                    swift_list.append((mjd_val, rate, 0.5 * (abs(err_low) + abs(err_high))))
                except: continue
    except: pass

    if not swift_list:
        print("[Swift Plot] Error: No valid Swift data loaded.")
        return

    swift_arr = np.array(swift_list)
    order = np.argsort(swift_arr[:, 0])
    swift_dates, swift_rate, swift_err = swift_arr[order].T

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.errorbar(swift_dates, swift_rate, yerr=swift_err, fmt='.', color='gray', ecolor='lightgray', 
                label='Swift/XRT (WT+PC)', zorder=1)

    g2_label = 'g2' if 'g2' in plot_df['component'].values else 'east'
    g3_label = 'g3' if 'g3' in plot_df['component'].values else 'west'

    g2_data = plot_df[plot_df['component'] == g2_label]
    g3_data = plot_df[plot_df['component'] == g3_label]

    suspect_obs_ids = []
    if not g2_data.empty:
        suspect_obs_ids = g2_data.sort_values('ejection_mjd').head(3)['obs_id'].tolist()

    def plot_subset(data, method, is_suspect, color, marker, label_base):
        mask = (data['method'] == method) if method == 'fit' else (data['method'] != 'fit')
        if is_suspect:
            mask = mask & data['obs_id'].isin(suspect_obs_ids)
            face = 'yellow'
        else:
            mask = mask & ~data['obs_id'].isin(suspect_obs_ids)
            face = color
        
        subset = data[mask]
        if subset.empty: return

        lbl = f"{label_base} {'Suspect' if is_suspect else ''} {method.capitalize()} (x{int(config.HRC_SCALE_FACTOR)})"
        if is_suspect and method != 'fit': lbl = "_nolegend_"

        ax.errorbar(
            x=subset['ejection_mjd'], 
            y=subset['nominal'] * config.HRC_SCALE_FACTOR,
            xerr=[np.abs(subset['ejection_mjd_err_neg']), np.abs(subset['ejection_mjd_err_pos'])],
            yerr=[subset['minus_err'] * config.HRC_SCALE_FACTOR, subset['plus_err'] * config.HRC_SCALE_FACTOR],
            fmt=marker, linestyle='', markerfacecolor=face, markeredgecolor='black', 
            color=color, ecolor=color, capsize=3, alpha=0.9, zorder=10, label=lbl
        )

    plot_subset(g2_data, 'fit', False, 'deepskyblue', 'o', "East Jet")
    plot_subset(g2_data, 'calc', False, 'deepskyblue', '<', "East Jet")
    plot_subset(g2_data, 'fit', True, 'deepskyblue', 'o', "East Jet")
    plot_subset(g2_data, 'calc', True, 'deepskyblue', '<', "East Jet")

    plot_subset(g3_data, 'fit', False, 'lightcoral', 'o', "West Jet")
    plot_subset(g3_data, 'calc', False, 'lightcoral', '<', "West Jet")

    ax.set_xlabel("MJD (days)")
    ax.set_ylabel("Count Rate (cts/s)")
    ax.set_title("Swift Count Rates vs. HRC Count Rates at Component Ejection Time")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    out_file = os.path.join(config.DIR_JET_PLOTS, f'swift-comparison-{config.FILE_ID}.pdf')
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Swift comparison plot saved to: {out_file}")
    plt.close(fig)