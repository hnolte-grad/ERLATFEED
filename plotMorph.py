import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def plot_MinimalScatter(dat, xvar, yvar, zvar, zopt, ntx, l, h, reg):
    # AESTHETICS #
    plt.rcParams['font.family'] = 'Times New Roman'  # Set font to Times New Roman
    plt.rcParams['font.size'] = 12
    plt.rcParams['text.color'] = 'black'  # Set text color to black

    # DATA #
    x = dat[xvar]
    y = dat[yvar]

    # PLOT #
    fig, ax = plt.subplots(figsize=(l, h), facecolor='white', dpi=300)  # Set background to white
    ax.set_facecolor('white')  # Set the axis background to white
    if zvar is not None:
        z = dat[zvar].astype(str)
        unique_z_values = np.unique(z)

        legend_handles = []
        legend_labels = []

        if zopt == 1:
            # Use different markers
            markers = ['o', 's', '^', 'D', 'p', '*', 'H', '+', '1', 'x', 'v', '<', '>', '8', '|', '_']
            if len(unique_z_values) > len(markers):
                raise ValueError(f"Too many unique values in '{zvar}' for available markers ({len(markers)}).")
            marker_mapping = {val: markers[i % len(markers)] for i, val in enumerate(unique_z_values)}
            for zi in unique_z_values:
                subset = dat[z == zi]
                scatter = ax.scatter(subset[xvar], subset[yvar],
                                     color='black', s=40, edgecolors='black',
                                     zorder=1, marker=marker_mapping[zi], alpha=0.7)
                legend_handles.append(scatter)
                legend_labels.append(str(zi))

        elif zopt == 2:
            palette = sns.color_palette("hls", n_colors=len(unique_z_values))
            color_mapping = {val: palette[i % len(palette)] for i, val in enumerate(unique_z_values)}
            for zi in unique_z_values:
                subset = dat[z == zi]
                scatter = ax.scatter(subset[xvar], subset[yvar],
                                     color=color_mapping[zi], s=40, edgecolors='black',
                                     zorder=1, marker='o', alpha=0.7)
                legend_handles.append(scatter)
                legend_labels.append(str(zi))

        else:
            raise ValueError("zopt must be 1 (markers) or 2 (colors)")

        ax.legend(legend_handles, legend_labels, title=zvar, loc='upper left',
                  fontsize=10, facecolor='white', edgecolor='black', bbox_to_anchor=(1.05, 1))

    # MODIFY AXES, FRAME, ETC.
    x_margin = 0.02 * (max(x) - min(x))  # Small margin to prevent axis from touching the data
    y_margin = 0.02 * (max(y) - min(y))  # Small margin to prevent axis from touching the data

    # Set the axis limits with a slight margin
    #ax.set_xlim(min(x) - x_margin, max(x) + x_margin)
    #ax.set_ylim(min(y) - y_margin, max(y) + y_margin)

    # Move the spines (axis lines) inward a bit (adjusting from their normal position)
    #ax.spines['left'].set_position(('data', min(x) - x_margin))  # Move the left spine inward
    #ax.spines['bottom'].set_position(('data', min(y) - y_margin))  # Move the bottom spine inward

    # Set spine colors to black
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Modify the length of the spines (only within the data range)
    ax.spines['left'].set_bounds(min(y), max(y))  # Make the left spine only extend between min and max y
    ax.spines['bottom'].set_bounds(min(x), max(x))  # Make the bottom spine only extend between min and max x

    #Set number of ticks
    num_ticks_x = ntx  # number of ticks
    num_ticks_y = ntx
    x_ticks = np.linspace(min(x), max(x), num_ticks_x)  # space ticks evenly
    y_ticks = np.linspace(min(y), max(y), num_ticks_y)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Adjust the tick parameters, setting ticks and labels to black
    ax.tick_params(axis='both', which='major', labelsize=20, labelcolor='black',
                   width=1.5, direction='out', length=5, colors='black')  # tick parameters
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')  # bold the tick labels

    # IF YOU ASKED FOR A REGRESSION...
    if reg is not None:
        # Perform linear regression using numpy's polyfit function
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Create the regression line
        x_vals = np.linspace(min(x), max(x), 100)
        y_vals = slope * x_vals + intercept

        # Calculate R-squared value
        y_pred = slope * np.array(dat[xvar]) + intercept  # Predicted values
        residuals = np.array(y) - y_pred  # Residuals (observed - predicted)
        ss_residual = np.sum(residuals ** 2)  # Sum of squared residuals
        ss_total = np.sum((np.array(y) - np.mean(y)) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_residual / ss_total)  # R-squared value
        # Plot the regression line (black)
        ax.plot(x_vals, y_vals, color='black', linewidth=0.5, linestyle='--', zorder=3)

        # Add annotations
        ax.text(0.85, 0.9, f"slope: {slope:.2f}", transform=ax.transAxes, color='black', fontsize=20,
                ha='left')
        ax.text(0.85, 0.85, f"r²: {r_squared:.3f}", transform=ax.transAxes, color='black', fontsize=20,
                ha='left')
        ax.text(0.85, 0.8, f"p:  {p_value:.4f}", transform=ax.transAxes, color='black', fontsize=20,
                ha='left')

        # Create a dashed box around the annotations
        # bbox_xmin = 0.8  # X position of the box (start)
        # bbox_xmax = 1.0  # X position of the box (end)
        # bbox_ymin = 0.75  # Y position of the box (start)
        # bbox_ymax = 1.0  # Y position of the box (end)
        #
        # # Add the dashed box around the annotations
        # rect = Rectangle((bbox_xmin, bbox_ymin), bbox_xmax - bbox_xmin, bbox_ymax - bbox_ymin,
        #                  linewidth=1, edgecolor='black', linestyle='--', facecolor='none', zorder=2)
        # ax.add_patch(rect)

    ax.set_xlabel(xvar, fontsize=20, color='black')
    ax.set_ylabel(yvar, fontsize=20, color='black')

    # Adjust layout to prevent overlap with the legend
    plt.subplots_adjust(right=0.85)

    plt.show()

os.chdir('C:\\_temp workspace\\1ER-LAT-FEED\\data')
morphdat = pd.DataFrame(pd.read_csv(os.getcwd()+"\\morph\\morphdata_ERLATFEED.csv"))
plot_MinimalScatter(morphdat, xvar='Cu/Ch', yvar= 'CBL/BIZ', zvar='Common', zopt = 2, ntx= 4,l = 12, h =10, reg=None)
