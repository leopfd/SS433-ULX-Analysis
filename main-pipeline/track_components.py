import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import config
from lib.log_parser import load_sherpa_log_to_dataframe
from lib.arguments import get_pipeline_args

def run_tracker_analysis():
    df = None
    
    # Check if a tracker table already exists to avoid re-parsing logs
    if os.path.exists(config.TRACKER_TABLE_CSV):
        print(f"Loading existing tracker table: {config.get_rel_path(config.TRACKER_TABLE_CSV)}")
        try:
            df = pd.read_csv(config.TRACKER_TABLE_CSV)
        except Exception as e:
            print(f"\033[1m[ERROR]\033[0m Failed to read CSV: {e}")
            raise
            
    else:
        # Parse the multi-component log file if no existing table is found
        print("No existing tracker table found. Running Auto-Tracker...")
        print(f"Loading and processing logs: {config.get_rel_path(config.MULTI_LOG_TXT)}")
        df = load_sherpa_log_to_dataframe(config.MULTI_LOG_TXT)

        if df.empty:
            raise ValueError("Dataframe is empty. Check if the log file exists and has correct format.")

        df.to_csv(config.TRACKER_TABLE_CSV, index=False)
        print(f"Tracker table saved to: {config.TRACKER_TABLE_CSV}")

    # Isolate the core component to serve as the reference point for relative positioning
    ref_df = df[df['component'] == config.G1_COMPONENT][['obs_id', 'xpos', 'ypos']]
    ref_df = ref_df.rename(columns={'xpos': 'ref_x', 'ypos': 'ref_y'})

    df = df.merge(ref_df, on='obs_id', how='left')
    
    # Fill missing reference coordinates with the center pixel for observations where the core fit failed
    df['ref_x'] = df['ref_x'].fillna(config.CENTER_PIXEL)
    df['ref_y'] = df['ref_y'].fillna(config.CENTER_PIXEL)
    
    # Calculate the shift required to center the reference component
    df['dx'] = df['ref_x'] - config.CENTER_PIXEL
    df['dy'] = df['ref_y'] - config.CENTER_PIXEL
    
    # Apply the calculated shift to all component positions
    df['xpos'] -= df['dx']
    df['ypos'] -= df['dy']
    
    # Drop temporary columns used for calculation
    df.drop(columns=['dx', 'dy', 'ref_x', 'ref_y'], inplace=True)

    # Calculate offsets relative to the center pixel
    df['xoff'] = df['xpos'] - config.CENTER_PIXEL
    df['yoff'] = df['ypos'] - config.CENTER_PIXEL

    # Convert Cartesian offsets to Polar coordinates (Position Angle and Radius)
    pa_rad = np.arctan2(-df['xoff'], df['yoff'])
    df['PA'] = np.degrees(pa_rad)
    df['pa_rad'] = pa_rad 

    # Propagate errors for Position Angle calculation
    d2 = df['xoff']**2 + df['yoff']**2
    
    # Handle division by zero for components exactly at the center
    dpa_dx = np.divide(-df['yoff'], d2, out=np.full_like(d2, np.nan), where=d2 != 0)
    dpa_dy = np.divide(df['xoff'], d2, out=np.full_like(d2, np.nan), where=d2 != 0)

    df['PA_err_plus'] = np.degrees(np.sqrt((dpa_dx * df['xpos_plus'])**2 + (dpa_dy * df['ypos_plus'])**2))
    df['PA_err_minus'] = np.degrees(np.sqrt((dpa_dx * df['xpos_minus'])**2 + (dpa_dy * df['ypos_minus'])**2))
    
    # Propagate errors for Radius calculation
    pixscale_arcsec = 0.13175 * config.BIN_SIZE 
    df['radius'] = np.hypot(df['xoff'], df['yoff']) * pixscale_arcsec

    r_pix = df['radius'] / pixscale_arcsec
    is_zero = np.isclose(r_pix, 0)
    
    # Calculate radius errors handling the special case where radius is near zero
    df['radius_plus_err'] = np.where(
        is_zero, 
        np.hypot(df['xpos_plus'], df['ypos_plus']), 
        np.sqrt((df['xoff']*df['xpos_plus'])**2 + (df['yoff']*df['ypos_plus'])**2)/r_pix
    ) * pixscale_arcsec
    
    df['radius_minus_err'] = np.where(
        is_zero, 
        np.hypot(df['xpos_minus'], df['ypos_minus']), 
        np.sqrt((df['xoff']*df['xpos_minus'])**2 + (df['yoff']*df['ypos_minus'])**2)/r_pix
    ) * pixscale_arcsec

    # Sort data chronologically by Modified Julian Date (MJD)
    if 'mjd' in df.columns:
        df.sort_values('mjd', inplace=True)
    
    df['flag'] = 'clean'

    # Pivot data to organize flux/rate values by component for easier plotting
    pivoted = df.pivot_table(index='mjd', columns='component', values=['nominal', 'plus_err', 'minus_err'])
    
    try:
        df_nom, df_plus, df_minus = [pivoted[val].sort_index() for val in ['nominal', 'plus_err', 'minus_err']]
    except KeyError:
        print("Warning: Could not pivot data for plotting (possibly missing columns). Skipping plots.")
        return df

    grouped_by_comp = df.groupby('component')
    
    # Identify non-core components for analysis
    comps = [c for c in df_nom.columns if not (c == 'core' or c == 'bkg')]
    
    # Assign colormaps based on component direction or type
    comp_cmaps = {}
    
    for c in comps:
        if c.startswith('east'):
            comp_cmaps[c] = plt.cm.Blues
        elif c.startswith('west'):
            comp_cmaps[c] = plt.cm.Reds
        elif c.startswith('other'):
            comp_cmaps[c] = plt.cm.Greens 
        else:
            comp_cmaps[c] = plt.cm.Purples

    time_min, time_max = df['mjd'].min(), df['mjd'].max()
    time_norm = plt.Normalize(vmin=time_min, vmax=time_max)
        
    n = len(comps)
    if n == 0:
        print("Warning: No components found to plot.")
        return df

    # Define visual offsets for grouped plotting
    delta = 0.02
    offsets = {c: (i - (n - 1) / 2) * delta for i, c in enumerate(comps)}
    
    pdf_name = f'comp-tracker-plots-{config.FILE_ID}.pdf'
    pdf_filename = os.path.join(config.DIR_TRACKER_PLOTS, pdf_name)

    with PdfPages(pdf_filename) as pdf:
        discrete_colors = ['dodgerblue', 'mediumseagreen', 'mediumslateblue', 'lightcoral']
        comp_discrete_map = {comp: discrete_colors[i % len(discrete_colors)] for i, comp in enumerate(comps)}

        # Plot 1 Position Angle and Count Rates
        fig = plt.figure(figsize=(12, 6))
        gs  = GridSpec(n, 2, figure=fig, width_ratios=[1,1], hspace=0, wspace=0.3)

        # Left Panel Position Angle vs Time
        ax_pa = fig.add_subplot(gs[:,0])
        for comp, color in comp_discrete_map.items():
            if comp in grouped_by_comp.groups:
                grp = grouped_by_comp.get_group(comp)
                ax_pa.errorbar(grp['mjd'].values, grp['PA'].values, yerr=[grp['PA_err_minus'].values, grp['PA_err_plus'].values], marker='.', linestyle='-', capsize=3, color=color, label=comp)
                
        ax_pa.set_ylabel('position angle (°)')
        ax_pa.set_xlabel('mjd')
        ax_pa.set_title('position angle vs time')
        ax_pa.set_ylim(-180,180)
        ax_pa.grid(True)
        ax_pa.legend()

        # Right Panel Stacked Count Rates
        ax_bottom = None
        for i_comp, focus in reversed(list(enumerate(comps))):
            if i_comp == n - 1:
                ax = fig.add_subplot(gs[i_comp, 1])
                ax.set_xlabel('mjd')
                ax_bottom = ax
            else:
                ax = fig.add_subplot(gs[i_comp, 1], sharex=ax_bottom)
                ax.tick_params(labelbottom=False)
            ax.grid(True, zorder=0)
            
            for comp_name, current_color in comp_discrete_map.items():
                if comp_name not in df_nom.columns: continue
                x_val = df_nom.index + offsets[comp_name]
                y_val = df_nom[comp_name]
                y_err_val = [df_minus[comp_name], df_plus[comp_name]]
                alpha_val, line_style, label_text, z_order = (1.0, '-', comp_name, 10) if comp_name == focus else (0.3, '', None, 1)
                x_arr = x_val.values
                y_arr = y_val.values
                y_err_arr = [df_minus[comp_name].values, df_plus[comp_name].values]

                ax.errorbar(x_arr, y_arr, yerr=y_err_arr, color=current_color, marker='.', linestyle=line_style, capsize=3, alpha=alpha_val, label=label_text, zorder=z_order)

            ax.set_yticks([0.1,0.3])
            if ax.has_data(): ax.legend(loc='upper left')

        fig.text(0.73, 0.885, 'component count rates', ha='center', va='bottom', fontsize=17)
        fig.text(0.495, 0.5, 'count rate (counts/s)', va='center', rotation='vertical')
        pdf.savefig(fig)
        plt.close(fig) 

        # Plot 2 Polar Plot of Sky Positions
        fig_polar = plt.figure(figsize=(10, 8))
        ax_polar = fig_polar.add_subplot(111, projection='polar')
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(1)
        ax_polar.set_thetamin(-180)
        ax_polar.set_thetamax(180)
        ax_polar.set_rlabel_position(135)

        for comp in comps:
            if comp in grouped_by_comp.groups:
                grp = grouped_by_comp.get_group(comp)
                cmap = comp_cmaps.get(comp, plt.cm.Greys)
                for _, row in grp.iterrows():
                    normalized_t = time_norm(row['mjd'])
                    color_idx = 0.4 + 0.6 * normalized_t
                    t_color = cmap(color_idx)
                    p_rad = float(row['pa_rad'])
                    rad   = float(row['radius'])
                    pa_min = float(np.deg2rad(row['PA_err_minus']))
                    pa_pl  = float(np.deg2rad(row['PA_err_plus']))
                    r_min  = float(row['radius_minus_err'])
                    r_pl   = float(row['radius_plus_err'])

                    ax_polar.errorbar(p_rad, rad, xerr=[[pa_min], [pa_pl]], yerr=[[r_min], [r_pl]], marker='.', linestyle='', color=t_color, capsize=2, markersize=8)

        # Add colorbar for time evolution
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, norm=time_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_polar, pad=0.1)
        cbar.set_label('mjd (light=early, dark=late)')

        legend_elements = []
        for c in comps:
            cm = comp_cmaps.get(c, plt.cm.Greys)
            c_color = cm(0.7) 
            legend_elements.append(Line2D([0], [0], marker='.', color=c_color, label=c, markerfacecolor=c_color, markersize=8, linestyle='None'))
            
        ax_polar.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.1), title="components")

        angles = np.arange(-150, 180, 30)
        angles = np.append(angles, 180)
        ax_polar.set_rmin(0)
        ax_polar.set_thetagrids(angles, [f"{int(a)}°" for a in angles])
        ax_polar.set_title('on sky component positions (arcsec)')

        pdf.savefig(fig_polar)
        plt.close(fig_polar)
    
    print(f"\nplots saved to: {config.get_rel_path(pdf_filename)}")
    
    return df

if __name__ == "__main__":
    args = get_pipeline_args()
    config.update_config_from_args(args)
    run_tracker_analysis()
