"""
========================================================================================================================
Script Name: plotAuditExample.py
Author: Hannah Clayton
Created: 04.16.2025
Description:
    plots for ERLATFEED manuscript auditing example. generates two plots 1) for the whole deployment (depth, tide, roll)
    and 2) an example from a single feeding event (smoothed jerk, depth, rate of roll, and roll).
Usage:
Dependencies:
    - Python >= 3.12
Notes:
========================================================================================================================
"""
from workhorseFuncs import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_singlefeed(csv_path, ntx=3, l=8, h=10):
    # Load and preprocess
    df = pd.read_csv(csv_path)
    df['pitch_deg'] = np.degrees(df['pitch'])
    dt = 0.1
    df['droll'] = np.gradient(df['roll'], dt)  # still computed just in case
    print("\nFirst 10 rows after processing:\n")
    print(df[['sec', 'depth', 'roll', 'pitch_deg', 'sjerk']].head(10))

    # Set plot style
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.edgecolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'axes.labelcolor': 'white'
    })

    # Only 4 plots now
    fig, axs = plt.subplots(4, 1, figsize=(l, h), sharex=True, dpi=150)
    fig.patch.set_alpha(0.0)

    # Define plots
    plots = [
        ('sjerk', 'Smoothed Jerk'),
        ('roll', 'Roll (°)'),
        ('pitch_deg', 'Pitch (°)'),
        ('depth', 'Depth (m)'),
    ]

    for ax, (col, ylabel) in zip(axs, plots):
        x = df['sec']
        y = df[col]

        # Plot setup
        ax.set_facecolor('none')
        ax.plot(x, y, color='white', linewidth=0.8)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        if col == 'depth':
            ax.invert_yaxis()

        for spine in ax.spines.values():
            spine.set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set ticks
        x_ticks = np.linspace(x.min(), x.max(), ntx)
        y_ticks = [round(y.min(), 2), round(y.max(), 2)]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        ax.tick_params(
            axis='both', which='major',
            labelsize=12, labelcolor='white',
            width=0.7, direction='out', length=2.5, color='white'
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        ax.set_ylabel(ylabel, color='white')

    axs[-1].set_xlabel("Time (s)", color='white')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()


# SET UP WORKSPACE
ddir = 'C:\\_temp workspace\\1ER-LAT-FEED\\data\\figure data\\audit example\\'                                          # data directory
os.chdir(ddir)

plot_singlefeed(ddir+"prh-feedevents_er160406-21_er160406-21_TD1.0_1.0_FP1.0_B2_FE1.csv")








