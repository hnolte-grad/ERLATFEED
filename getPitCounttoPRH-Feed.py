import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def getPitstoPRHFeed():
    prh_path = 'C:\\_temp workspace\\1ER-LAT-FEED\\data\\feed\\prh-feedevents\\prh-feedevents_MASTER.csv'
    analyzed_path = 'C:\\_temp workspace\\1ER-LAT-FEED\\data\\feed\\analyzed-feedevents\\analyzed-feedevents_MASTER.csv'

    prh_df = pd.read_csv(prh_path)
    analyzed_df = pd.read_csv(analyzed_path)
        # Ensure feedID is treated consistently
    prh_df['feedID'] = prh_df['feedID'].astype(str)
    analyzed_df['feedID'] = analyzed_df['feedID'].astype(str)
        # Merge 'pitcount' into prh-feedevents based on 'feedID'
    merged_df = prh_df.merge(analyzed_df[['feedID', 'pitcount']], on='feedID', how='left', suffixes=('', '_analyzed'))

        # If desired, overwrite the old prh-feedevents file:
        # merged_df.to_csv(prh_path, index=False)

        # Or, just keep it in memory as the updated prh DataFrame
    prh_df = merged_df

        # Confirm it worked
    print(prh_df[['feedID', 'pitcount']].head())
    return prh_df

def makeprrrpplot1(dat, grouping_var=None, opacity_var=None):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    lw = 0.2  # Increased line width slightly for visibility

    # Set color mapping
    if grouping_var and grouping_var in dat.columns:
        group_vals = dat[grouping_var].dropna().unique()
        cmap = cm.get_cmap('tab10', len(group_vals)) if len(group_vals) <= 10 else cm.get_cmap('viridis', len(group_vals))
        color_map = {val: cmap(i) for i, val in enumerate(sorted(group_vals))}
    else:
        color_map = {}

    # Set opacity mapping
    if opacity_var and opacity_var in dat.columns:
        opacity_vals = dat[opacity_var].dropna().unique()
        norm = mcolors.Normalize(vmin=np.min(opacity_vals), vmax=np.max(opacity_vals))
        opacity_map = {val: norm(val) for val in sorted(opacity_vals)}
    else:
        opacity_map = {}

    legend_elements = []

    for i, (feedID, group) in enumerate(dat.groupby('feedID')):
        group = group.copy()
        group['time_seconds'] -= group['time_seconds'].iloc[0]
        group['pitch_deg'] = np.degrees(group['pitch'])  # Convert pitch to degrees

        # Color
        color = 'black'
        if grouping_var and grouping_var in group.columns:
            color_key = group[grouping_var].iloc[0]
            color = color_map.get(color_key, 'black')

        # Opacity
        alpha = 1.0
        if opacity_var and opacity_var in group.columns:
            op_key = group[opacity_var].iloc[0]
            alpha = opacity_map.get(op_key, 1.0)

        # Plot all signals
        axs[0].plot(group['time_seconds'], group['depth'], color=color, alpha=alpha, linewidth=lw)
        axs[1].plot(group['time_seconds'], group['roll'], color=color, alpha=alpha, linewidth=lw)
        #axs[2].plot(group['time_seconds'], group['rollrate'], color=color, alpha=alpha, linewidth=lw)
        axs[2].plot(group['time_seconds'], group['pitch_deg'], color=color, alpha=alpha, linewidth=lw)

    # Format each axis
    axs[0].set_ylabel('Depth (m)', color='black')
    axs[0].invert_yaxis()
    axs[0].grid(True, color='gray', linestyle='dotted', linewidth=0.5)
    axs[0].tick_params(labelbottom=False)  # ❌ remove x tick labels

    axs[1].set_ylabel('Roll (deg)', color='black')
    axs[1].grid(True, color='gray', linestyle='dotted', linewidth=0.5)
    axs[1].tick_params(labelbottom=False)

    # axs[2].set_ylabel('Roll rate (°/s)', color='black')
    # axs[2].set_ylim(-1, 360)
    # axs[2].grid(True, color='gray', linestyle='dotted', linewidth=0.5)

    axs[2].set_ylabel('Pitch (deg)', color='black')
    axs[2].set_xlabel('Time (s)', color='black')
    axs[2].grid(True, color='gray', linestyle='dotted', linewidth=0.5)

    # X axis formatting
    max_time = dat['time_seconds'].max()
    for ax in axs:
        ax.set_xlim([0, max_time])
    plt.xticks(range(0, int(max_time) + 1, 60))

    # Construct legend handles
    if grouping_var:
        for k, v in color_map.items():
            legend_elements.append(Line2D([0], [0], color=v, lw=2, label=f"{grouping_var}: {k}"))

    if opacity_var:
        # Sample a few distinct opacities for legend
        sampled_opacities = np.linspace(0.3, 1.0, num=4)
        for a in sampled_opacities:
            legend_elements.append(Line2D([0], [0], color='gray', alpha=a, lw=2, label=f"{opacity_var} α={a:.2f}"))

    if legend_elements:
        axs[0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True)


    plt.tight_layout()
    plt.show()

def makeprrrpplot(dat, grouping_var=None, opacity_var=None):
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    lw = 0.2  # Increased line width slightly for visibility

    # Set color mapping
    if grouping_var and grouping_var in dat.columns:
        group_vals = dat[grouping_var].dropna().unique()
        cmap = cm.get_cmap('tab10', len(group_vals)) if len(group_vals) <= 10 else cm.get_cmap('viridis', len(group_vals))
        color_map = {val: cmap(i) for i, val in enumerate(sorted(group_vals))}
    else:
        color_map = {}

    # Set opacity mapping
    if opacity_var and opacity_var in dat.columns:
        opacity_vals = dat[opacity_var].dropna().unique()
        norm = mcolors.Normalize(vmin=np.min(opacity_vals), vmax=np.max(opacity_vals))
        opacity_map = {val: norm(val) for val in sorted(opacity_vals)}
    else:
        opacity_map = {}

    legend_elements = []

    for i, (feedID, group) in enumerate(dat.groupby('feedID')):
        group = group.copy()
        group['time_seconds'] -= group['time_seconds'].iloc[0]
        group['pitch_deg'] = np.degrees(group['pitch'])  # Convert pitch to degrees

        # Color
        color = 'black'
        if grouping_var and grouping_var in group.columns:
            color_key = group[grouping_var].iloc[0]
            color = color_map.get(color_key, 'black')

        # Opacity
        alpha = 1.0
        if opacity_var and opacity_var in group.columns:
            op_key = group[opacity_var].iloc[0]
            alpha = opacity_map.get(op_key, 1.0)

        # Plot all signals
        axs[0].plot(group['time_seconds'], group['depth'], color=color, alpha=alpha, linewidth=lw)
        axs[1].plot(group['time_seconds'], group['roll'], color=color, alpha=alpha, linewidth=lw)
        axs[2].plot(group['time_seconds'], group['pitch_deg'], color=color, alpha=alpha, linewidth=lw)

    # Format axes
    axs[0].set_ylabel('Depth (m)', color='black')
    axs[0].invert_yaxis()
    axs[0].grid(True, color='gray', linestyle='dotted', linewidth=0.5)
    axs[0].tick_params(labelbottom=False)

    axs[1].set_ylabel('Roll (deg)', color='black')
    axs[1].grid(True, color='gray', linestyle='dotted', linewidth=0.5)
    axs[1].tick_params(labelbottom=False)

    axs[2].set_ylabel('Pitch (deg)', color='black')
    axs[2].set_xlabel('Time (s)', color='black')
    axs[2].grid(True, color='gray', linestyle='dotted', linewidth=0.5)

    # X axis formatting
    max_time = dat['time_seconds'].max()
    for ax in axs:
        ax.set_xlim([0, max_time])
    plt.xticks(range(0, int(max_time) + 1, 60))

    # Construct legend handles with boxes for color
    if grouping_var:
        for k, v in color_map.items():
            legend_elements.append(Patch(facecolor=v, edgecolor='black', label=f"{grouping_var}: {k}"))

    if opacity_var:
        sampled_opacities = np.linspace(0.3, 1.0, num=4)
        for a in sampled_opacities:
            legend_elements.append(Patch(facecolor='gray', alpha=a, edgecolor='black', label=f"{opacity_var} α={a:.2f}"))

    if legend_elements:
        axs[0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True)

    plt.tight_layout()
    plt.show()

dat = getPitstoPRHFeed()
#dat = dat[dat['time_seconds'] <= 10]  # Only first 10 seconds

makeprrrpplot(dat, 'pitcount', None)