import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config


def plot_swift_comparison(tracker_df, ejection_df):
    """
    Plots Swift vs HRC comparison using in memory dataframes
    Call this AFTER running your kinematic fitting stage Stage 3

    This function supports component names like east 1 or east 2 and west 1 or west 2
    It treats any method starting with fit as a fit
    It keeps the original Swift text parsing to avoid breaking existing file formats
    """

    print("[Swift Plot] Generating Comparison Plot...")

    if not os.path.exists(config.SWIFT_FILE):
        print(f"Error: Swift data file not found at {config.get_rel_path(config.SWIFT_FILE)}")
        return

    # Prepare Ejection Data ensuring robustness against missing columns
    if 'ejection_mjd' not in ejection_df.columns:
        if 'travel_time_days' in ejection_df.columns:
            ejection_df = ejection_df.copy()
            ejection_df['ejection_mjd'] = ejection_df['mjd'] - ejection_df['travel_time_days']
        else:
            print("[Swift Plot] Error: 'travel_time_days' missing from ejection dataframe.")
            return

    # Calculate errors if not present
    if 'ejection_mjd_err_pos' not in ejection_df.columns or 'ejection_mjd_err_neg' not in ejection_df.columns:
        # Requires beta and beta error columns to exist otherwise default to 0 error
        ejection_df = ejection_df.copy()

        if all(c in ejection_df.columns for c in ['beta', 'beta_err_pos', 'beta_err_neg', 'travel_time_days']):
            safe_beta = np.where(ejection_df['beta'] != 0, ejection_df['beta'], np.nan)
            ejection_df['ejection_mjd_err_pos'] = (ejection_df['travel_time_days'] * ejection_df['beta_err_pos']) / safe_beta
            ejection_df['ejection_mjd_err_neg'] = (ejection_df['travel_time_days'] * ejection_df['beta_err_neg']) / safe_beta
            ejection_df['ejection_mjd_err_pos'] = ejection_df['ejection_mjd_err_pos'].fillna(0)
            ejection_df['ejection_mjd_err_neg'] = ejection_df['ejection_mjd_err_neg'].fillna(0)
        else:
            ejection_df['ejection_mjd_err_pos'] = 0.0
            ejection_df['ejection_mjd_err_neg'] = 0.0

    # Merge Dataframes matching observation IDs and component names
    hrc_cols = ['obs_id', 'component', 'nominal', 'minus_err', 'plus_err']
    hrc_data = tracker_df[[c for c in hrc_cols if c in tracker_df.columns]].copy()

    ejection_cols = ['obs_id', 'component_name', 'ejection_mjd', 'method', 'ejection_mjd_err_pos', 'ejection_mjd_err_neg']
    if 'component_name' not in ejection_df.columns and 'component' in ejection_df.columns:
        ejection_df = ejection_df.copy()
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

    # Load Swift Data using the original text parsing logic
    swift_list = []
    try:
        with open(config.SWIFT_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("!", "NO", "#")):
                    continue
                tok = line.split()
                if len(tok) < 6:
                    continue
                try:
                    mjd_val, rate, err_low, err_high = map(float, [tok[0], tok[3], tok[4], tok[5]])
                    swift_list.append((mjd_val, rate, 0.5 * (abs(err_low) + abs(err_high))))
                except Exception:
                    continue
    except Exception:
        pass

    if not swift_list:
        print("[Swift Plot] Error: No valid Swift data loaded.")
        return

    swift_arr = np.array(swift_list)
    order = np.argsort(swift_arr[:, 0])
    swift_dates, swift_rate, swift_err = swift_arr[order].T

    # Plotting setup
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.errorbar(
        swift_dates, swift_rate, yerr=swift_err,
        fmt='.', color='gray', ecolor='lightgray',
        label='Swift/XRT (WT+PC)', zorder=1
    )

    comp_series = plot_df['component'].astype(str)

    east_mask = comp_series.str.startswith(('east', 'g2'))
    west_mask = comp_series.str.startswith(('west', 'g3'))

    g2_data_all = plot_df[east_mask].copy()
    g3_data_all = plot_df[west_mask].copy()

    if g2_data_all.empty and g3_data_all.empty:
        print("[Swift Plot] Warning: No east/west components found in merged table.")
        print("[Swift Plot] Components present:", sorted(plot_df['component'].astype(str).unique()))
        return

    # Keep suspect logic which picks earliest 3 ejection times from EAST side if available else from WEST
    suspect_obs_ids = []
    if not g2_data_all.empty:
        suspect_obs_ids = g2_data_all.sort_values('ejection_mjd').head(3)['obs_id'].tolist()
    elif not g3_data_all.empty:
        suspect_obs_ids = g3_data_all.sort_values('ejection_mjd').head(3)['obs_id'].tolist()

    # Method classification where anything starting with fit counts as fit
    def is_fit_method(series):
        return series.astype(str).str.lower().str.startswith('fit')

    # Updated plot subset function that keeps styling but uses is fit method check
    def plot_subset(data, method, is_suspect, color, marker, label_base):
        if data.empty:
            return

        fit_mask = is_fit_method(data['method']) if 'method' in data.columns else pd.Series(False, index=data.index)
        mask = fit_mask if method == 'fit' else ~fit_mask

        if is_suspect:
            mask = mask & data['obs_id'].isin(suspect_obs_ids)
            face = 'yellow'
        else:
            mask = mask & ~data['obs_id'].isin(suspect_obs_ids)
            face = color

        subset = data[mask]
        if subset.empty:
            return

        lbl = f"{label_base} {'Suspect' if is_suspect else ''} {method.capitalize()} (x{int(config.HRC_SCALE_FACTOR)})"
        if is_suspect and method != 'fit':
            lbl = "_nolegend_"

        ax.errorbar(
            x=subset['ejection_mjd'].values,
            y=(subset['nominal'] * config.HRC_SCALE_FACTOR).values,
            xerr=[np.abs(subset['ejection_mjd_err_neg'].values), np.abs(subset['ejection_mjd_err_pos'].values)],
            yerr=[(subset['minus_err'] * config.HRC_SCALE_FACTOR).values, (subset['plus_err'] * config.HRC_SCALE_FACTOR).values],
            fmt=marker, linestyle='',
            markerfacecolor=face, markeredgecolor='black',
            color=color, ecolor=color, capsize=3, alpha=0.9, zorder=10, label=lbl
        )

    # Plot East side components
    plot_subset(g2_data_all, 'fit', False, 'deepskyblue', 'o', "East Jet")
    plot_subset(g2_data_all, 'calc', False, 'deepskyblue', '<', "East Jet")
    plot_subset(g2_data_all, 'fit', True,  'deepskyblue', 'o', "East Jet")
    plot_subset(g2_data_all, 'calc', True, 'deepskyblue', '<', "East Jet")

    # Plot West side components
    plot_subset(g3_data_all, 'fit', False, 'lightcoral', 'o', "West Jet")
    plot_subset(g3_data_all, 'calc', False, 'lightcoral', '<', "West Jet")

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
    print(f"———>Swift comparison plot saved to: {config.get_rel_path(out_file)}")
    plt.close(fig)