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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt  # make sure scipy is installed

# SET UP WORKSPACE
ddir = 'C:\\_temp workspace\\1ER-LAT-FEED\\data\\figure data\\audit example\\'
os.chdir(ddir)

def lowpass_butter(data, cutoff_hz, fs, order=3):
    """
    Zero-phase Butterworth low-pass filter using filtfilt.
    data      : 1D array-like (e.g., roll in degrees)
    cutoff_hz : cutoff frequency in Hz
    fs        : sampling frequency in Hz
    order     : filter order (2–4 is usually plenty)
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff_hz / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def plot_singlefeed(csv_path, ntx=3, l=8, h=10,
                    fs=10.0, cutoff_hz=0.5, filter_order=3):
    # Load the data
    df = pd.read_csv(csv_path)

    # Sampling parameters
    dt = 1.0 / fs

    # Pitch in degrees (roll is already in degrees)
    df['pitch_deg'] = np.degrees(df['pitch'])

    # ----------------------------------------------------------
    # LOW-PASS FILTER ON ROLL (IN DEGREES) FOR droll ONLY
    # ----------------------------------------------------------
    # Use raw roll (deg) as input to the filter
    roll_raw = df['roll'].values.astype(float)

    # If there are any NaNs in roll, interpolate so filtfilt doesn't blow up
    if np.isnan(roll_raw).any():
        roll_series = pd.Series(roll_raw)
        roll_raw = roll_series.interpolate(limit_direction='both').values

    # Low-pass filtered roll (still in degrees)
    roll_filt = lowpass_butter(
        roll_raw,
        cutoff_hz=cutoff_hz,
        fs=fs,
        order=filter_order
    )
    df['roll_filt_deg'] = roll_filt

    # Rate of roll from filtered roll (deg)
    df['droll'] = np.gradient(df['roll_filt_deg'], dt)

    # Quick sanity check
    print("\nFirst 10 rows after processing:\n")
    print(df[['sec', 'depth', 'roll', 'roll_filt_deg', 'droll', 'pitch_deg', 'sjerk']].head(10))
    print("\nroll (raw) min/max:", np.nanmin(df['roll']), np.nanmax(df['roll']))
    print("roll_filt_deg min/max:", np.nanmin(df['roll_filt_deg']), np.nanmax(df['roll_filt_deg']))
    print("droll min/max:", np.nanmin(df['droll']), np.nanmax(df['droll']))

    # ----------------------------------------------------------
    # PLOTTING
    # ----------------------------------------------------------
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })

    fig, axs = plt.subplots(5, 1, figsize=(l, h), sharex=True, dpi=200)

    # NOTE: For roll, we plot the RAW roll column
    plots = [
        ('sjerk',     'Smoothed Jerk'),
        ('roll',      'Roll (°) — raw'),
        ('droll',     'Rate of Roll (°/s) — from low-pass roll'),
        ('pitch_deg', 'Pitch (°)'),
        ('depth',     'Depth (m)'),
    ]

    for ax, (col, ylabel) in zip(axs, plots):
        x = df['sec'].values
        y = df[col].values.astype(float)

        # Handle NaNs for axis scaling
        valid = ~np.isnan(y)
        if not np.any(valid):
            continue

        y_valid = y[valid]
        y_min, y_max = y_valid.min(), y_valid.max()

        # Plot line
        line_color = 'firebrick' if col == 'roll' else 'black'
        ax.plot(x, y, color=line_color, linewidth=0.5)

        # Axis limits
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y_min, y_max)

        # Invert and shade for depth
        if col == 'depth':
            ax.invert_yaxis()
            ax.fill_between(x, y_min, y, where=valid, color='lightblue', alpha=0.3)

        # Remove unnecessary spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Move left spine slightly inside
        ax.spines['left'].set_position(('data', x.min() - 0.1))

        # Only move bottom spine if this is NOT the depth plot
        if col != 'depth':
            ax.spines['bottom'].set_position(('data', y_min - 0.1))

        # Set custom ticks
        x_ticks = np.linspace(x.min(), x.max(), ntx)
        y_ticks = np.linspace(y_min, y_max, ntx)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Format ticks
        ax.tick_params(
            axis='both', which='major',
            labelsize=12, labelcolor='black',
            width=0.7, direction='out', length=2.5
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # Set y-label
        ax.set_ylabel(ylabel)

    axs[-1].set_xlabel("Time (s)")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


# Example call
plot_singlefeed(
    ddir + "prh-feedevents_er160406-21_er160406-21_TD1.0_1.0_FP1.0_B2_FE1.csv"
)
