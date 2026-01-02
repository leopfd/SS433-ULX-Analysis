import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

    # Core emission time for reflected light: observed MJD minus core->knot light-travel time
    if "light_delay_days" in ejection_df.columns:
        ejection_df = ejection_df.copy()
        ejection_df["light_delay_days_err"] = ejection_df.get("light_delay_days_err", np.nan)
        ejection_df["ejection_mjd_lt"] = ejection_df["mjd"] - ejection_df["light_delay_days"]
        ejection_df["ejection_mjd_lt_err"] = ejection_df["light_delay_days_err"]

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
    hrc_cols = ['obs_id', 'component', 'mjd', 'nominal', 'minus_err', 'plus_err']
    hrc_data = tracker_df[[c for c in hrc_cols if c in tracker_df.columns]].copy()

    ejection_cols = [
        'obs_id',
        'component_name',
        'ejection_mjd',
        'ejection_mjd_lt',
        'ejection_mjd_lt_err',
        'method',
        'ejection_mjd_err_pos',
        'ejection_mjd_err_neg',
        'light_delay_days_err',
    ]
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

    if 'method' in plot_df.columns:
        plot_df = plot_df.copy()
        plot_df['method_norm'] = plot_df['method'].astype(str).str.strip().str.lower()
        # Collapse any non-fit into calc for plotting so plane/other fall into calc.
        plot_df['plot_method'] = np.where(
            plot_df['method_norm'].str.startswith('fit'),
            'fit',
            'calc'
        )

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

    out_file = os.path.join(config.DIR_JET_PLOTS, f'swift-comparison-{config.FILE_ID}.pdf')

    style_ctx = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    }

    with plt.style.context(style_ctx), PdfPages(out_file) as pdf:

        # original ejection ages
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

        obs_min = plot_df["mjd"].min() if "mjd" in plot_df.columns else plot_df["ejection_mjd"].min()
        obs_max = plot_df["mjd"].max() if "mjd" in plot_df.columns else plot_df["ejection_mjd"].max()
        def age_to_colors(values, vmin, vmax, cmap):
            if values.empty:
                return np.array([])
            span = vmax - vmin
            norm = (values - vmin) / span if span != 0 else np.zeros_like(values, dtype=float)
            return cmap(np.clip(norm, 0, 1))

        # Shared colormap for both panels
        cmap_common = plt.cm.rainbow_r

        def is_fit_method(series):
            return series.astype(str).str.startswith('fit')

        def is_calc_method(series):
            s = series.astype(str)
            return s.str.startswith('calc') | (~is_fit_method(series))

        def plot_subset(
            data,
            method,
            marker,
            labels_used,
            x_field="ejection_mjd",
            xerr_lower_col="ejection_mjd_err_neg",
            xerr_upper_col="ejection_mjd_err_pos",
            xerr_sym_col=None,
            color_min=None,
            color_max=None,
            color_cmap=None,
            color_field=None,
            marker_size=7,
        ):
            if data.empty:
                return

            method_col = (
                'plot_method'
                if 'plot_method' in data.columns
                else 'method_norm'
                if 'method_norm' in data.columns
                else ('method' if 'method' in data.columns else None)
            )

            if method_col is not None:
                fit_mask = is_fit_method(data[method_col])
                calc_mask = is_calc_method(data[method_col])
            else:
                fit_mask = pd.Series(False, index=data.index)
                calc_mask = ~fit_mask

            if method == 'fit':
                mask = fit_mask
            else:
                mask = calc_mask

            subset = data[mask]
            if subset.empty:
                return

            cf = color_field if color_field is not None else x_field
            if cf not in subset.columns:
                cf = x_field
            color_min = color_min if color_min is not None else subset[cf].min()
            color_max = color_max if color_max is not None else subset[cf].max()
            cmap = color_cmap if color_cmap is not None else cmap_common
            colors = age_to_colors(subset[cf], color_min, color_max, cmap)

            for row, color in zip(subset.itertuples(), colors):
                if method == "calc":
                    lbl = "_nolegend_"
                else:
                    lbl = f"Jet knot (x{int(config.HRC_SCALE_FACTOR)})" if not labels_used.get(method) else "_nolegend_"
                x_val = getattr(row, x_field)
                if xerr_sym_col:
                    sym = getattr(row, xerr_sym_col, np.nan)
                    xerr_val = [[abs(sym)], [abs(sym)]]
                else:
                    lower = getattr(row, xerr_lower_col, np.nan)
                    upper = getattr(row, xerr_upper_col, np.nan)
                    xerr_val = [[abs(lower)], [abs(upper)]]
                ax.errorbar(
                    x=x_val,
                    y=row.nominal * config.HRC_SCALE_FACTOR,
                    xerr=xerr_val,
                    yerr=[[row.minus_err * config.HRC_SCALE_FACTOR], [row.plus_err * config.HRC_SCALE_FACTOR]],
                    fmt=marker, linestyle='',
                    markersize=marker_size,
                    markerfacecolor=color, markeredgecolor='black',
                    color=color, ecolor=color, capsize=3, alpha=0.95, zorder=12, label=lbl
                )
                labels_used[method] = True

        label_tracker = {"fit": False, "calc": False}
        plot_subset(g2_data_all, 'fit', 'o', label_tracker, marker_size=8,
                    color_field="mjd", color_min=obs_min, color_max=obs_max)
        plot_subset(g2_data_all, 'calc', 'o', label_tracker, marker_size=9,
                    color_field="mjd", color_min=obs_min, color_max=obs_max)
        plot_subset(g3_data_all, 'fit', 'o', label_tracker, marker_size=8,
                    color_field="mjd", color_min=obs_min, color_max=obs_max)
        plot_subset(g3_data_all, 'calc', 'o', label_tracker, marker_size=9,
                    color_field="mjd", color_min=obs_min, color_max=obs_max)

        ax.set_xlabel("MJD (days)")
        ax.set_ylabel("Count Rate (cts/s)")
        ax.set_title("Swift vs. HRC Count Rates — Jet Knot Ejection Time")
        ax.grid(True, linestyle='--', alpha=0.6)

        sm = plt.cm.ScalarMappable(cmap=cmap_common, norm=plt.Normalize(vmin=obs_min, vmax=obs_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Observation MJD")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # core emission times (obs - tau_core->knot)
        if "ejection_mjd_lt" in plot_df.columns and plot_df["ejection_mjd_lt"].notna().any():

            fig, ax = plt.subplots(figsize=(15, 7))

            ax.errorbar(
                swift_dates, swift_rate, yerr=swift_err,
                fmt='.', color='gray', ecolor='lightgray',
                label='Swift/XRT (WT+PC)', zorder=1
            )

            plot_df_lt = plot_df.assign(ejection_mjd=plot_df["ejection_mjd_lt"])

            g2_data_all = plot_df_lt[east_mask].copy()
            g3_data_all = plot_df_lt[west_mask].copy()

            obs_min = plot_df_lt["mjd"].min() if "mjd" in plot_df_lt.columns else plot_df_lt["ejection_mjd"].min()
            obs_max = plot_df_lt["mjd"].max() if "mjd" in plot_df_lt.columns else plot_df_lt["ejection_mjd"].max()
            # Core-emission (t_obs - tau): shared colormap
            age_cmap_core = cmap_common

            label_tracker = {"fit": False, "calc": False}
            plot_subset(
                g2_data_all, 'fit', 'o', label_tracker,
                x_field="ejection_mjd_lt", xerr_sym_col="ejection_mjd_lt_err",
                color_field="mjd", color_min=obs_min, color_max=obs_max, color_cmap=age_cmap_core
            )
            plot_subset(
                g2_data_all, 'calc', 'o', label_tracker,
                x_field="ejection_mjd_lt", xerr_sym_col="ejection_mjd_lt_err",
                color_field="mjd", color_min=obs_min, color_max=obs_max, color_cmap=age_cmap_core
            )
            plot_subset(
                g3_data_all, 'fit', 'o', label_tracker,
                x_field="ejection_mjd_lt", xerr_sym_col="ejection_mjd_lt_err",
                color_field="mjd", color_min=obs_min, color_max=obs_max, color_cmap=age_cmap_core
            )
            plot_subset(
                g3_data_all, 'calc', 'o', label_tracker,
                x_field="ejection_mjd_lt", xerr_sym_col="ejection_mjd_lt_err",
                color_field="mjd", color_min=obs_min, color_max=obs_max, color_cmap=age_cmap_core
            )

            ax.set_xlabel("MJD (days)")
            ax.set_ylabel("Count Rate (cts/s)")
            ax.set_title("Swift vs. HRC Count Rates — Reflected Radiation Emission Time")
            ax.grid(True, linestyle='--', alpha=0.6)

            sm = plt.cm.ScalarMappable(cmap=age_cmap_core, norm=plt.Normalize(vmin=obs_min, vmax=obs_max))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Observation MJD")

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"———>Swift comparison plot saved to: {config.get_rel_path(out_file)}")
