import pandas as pd
import numpy as np
from scipy.stats import circmean, circstd

# ---------------------------------------
# CONFIG
# ---------------------------------------
xlsx_path = r"C:\\_temp workspace\\1ER-LAT-FEED\\data\\feed\\analyzed-feedevents\\analyzed-feedevents_MASTER.xlsx"
behaves_path = r"C:\\_temp workspace\\1ER-LAT-FEED\\data\\tags\\behaves\\behaves_MASTERFEED.xlsx"

# Columns + their units
cols_rad = ["meanPitchFP", "meanHead"]   # radians
cols_deg = ["meanRoll_deg"]              # degrees

# Map columns to conceptual roles
pitch_col = "meanPitchFP"
roll_col  = "meanRoll_deg"
head_col  = "meanHead"

# ---------------------------------------
# LOAD DATA
# ---------------------------------------
df = pd.read_excel(xlsx_path)
roll_deg_values = df[roll_col].dropna().values

beh = pd.read_excel(behaves_path)

# ---------------------------------------
# CONVERT DEGREE COLUMNS → RADIANS
# ---------------------------------------
df[cols_deg] = np.deg2rad(df[cols_deg])

# ---------------------------------------
# FUNCTION: Circular Range
# ---------------------------------------
def circular_range(data_rad):
    data = np.sort(data_rad)
    diffs = np.diff(np.concatenate([data, data[:1] + 2*np.pi]))
    return 2 * np.pi - np.max(diffs)

# ---------------------------------------
# MAIN XLSX CIRCULAR STATS
# ---------------------------------------
results = {}
all_cols = cols_rad + cols_deg

for col in all_cols:
    data_rad = df[col].dropna().values

    mu = circmean(data_rad, high=np.pi, low=-np.pi)
    sd = circstd(data_rad, high=np.pi, low=-np.pi)

    stats = {
        "circular_mean_rad": mu,
        "circular_mean_deg": np.degrees(mu),
        "circular_sd_rad": sd,
        "circular_sd_deg": np.degrees(sd),
    }

    # Linear ranges
    if col == pitch_col:
        pitch_min = np.nanmin(data_rad)
        pitch_max = np.nanmax(data_rad)
        r = pitch_max - pitch_min
        stats.update({
            "linear_min_rad": pitch_min,
            "linear_max_rad": pitch_max,
            "linear_range_rad": r,
            "linear_range_deg": np.degrees(r)
        })

    if col == roll_col:
        roll_min = np.nanmin(roll_deg_values)
        roll_max = np.nanmax(roll_deg_values)
        r_deg = roll_max - roll_min
        stats.update({
            "linear_min_deg": roll_min,
            "linear_max_deg": roll_max,
            "linear_range_deg": r_deg,
            "linear_range_rad": np.deg2rad(r_deg)
        })

    if col == head_col:
        cr = circular_range(data_rad)
        stats["circular_range_rad"] = cr
        stats["circular_range_deg"] = np.degrees(cr)

    results[col] = stats


# ---------------------------------------
# BEHAVES FILE: HEADING + STD
# ---------------------------------------
behaves_stats = {}

# meanHeading circular stats
if "meanHeading" in beh.columns:
    heading_rad = beh["meanHeading"].dropna().values
    beh_mu = circmean(heading_rad, high=np.pi, low=-np.pi)
    beh_sd = circstd(heading_rad, high=np.pi, low=-np.pi)

    behaves_stats.update({
        "circular_mean_rad": beh_mu,
        "circular_mean_deg": np.degrees(beh_mu),
        "circular_sd_rad": beh_sd,
        "circular_sd_deg": np.degrees(beh_sd),
    })

# stdHeading – linear stats (this is a spread, not circular)
if "stdHeading" in beh.columns:
    std_vals = beh["stdHeading"].dropna().values

    mean_rad = np.nanmean(std_vals)
    sd_rad   = np.nanstd(std_vals)

    behaves_stats.update({
        "stdHeading_mean_rad": mean_rad,
        "stdHeading_mean_deg": np.degrees(mean_rad),
        "stdHeading_sd_rad": sd_rad,
        "stdHeading_sd_deg": np.degrees(sd_rad),
    })

# ---------------------------------------
# PRINT MAIN RESULTS
# ---------------------------------------
for col, stats in results.items():
    print(f"\n===== {col.upper()} =====")
    print(f"Circular mean: {stats['circular_mean_rad']:.4f} rad  ({stats['circular_mean_deg']:.2f}°)")
    print(f"Circular SD:   {stats['circular_sd_rad']:.4f} rad  ({stats['circular_sd_deg']:.2f}°)")

    if col == pitch_col:
        print(f"Pitch min/max: {stats['linear_min_rad']:.4f}–{stats['linear_max_rad']:.4f} rad "
              f"({np.degrees(stats['linear_min_rad']):.2f}–{np.degrees(stats['linear_max_rad']):.2f}°)")
        print(f"Range:         {stats['linear_range_rad']:.4f} rad  ({stats['linear_range_deg']:.2f}°)")

    if col == roll_col:
        print(f"Roll min/max:  {stats['linear_min_deg']:.2f}–{stats['linear_max_deg']:.2f}°")
        print(f"Range:         {stats['linear_range_rad']:.4f} rad  ({stats['linear_range_deg']:.2f}°)")

    if col == head_col:
        print(f"Circular range (heading): {stats['circular_range_rad']:.4f} rad "
              f"({stats['circular_range_deg']:.2f}°)")


# ---------------------------------------
# PRINT BEHAVES RESULTS
# ---------------------------------------
print("\n===== BEHAVES: HEADING & SD =====")

if "meanHeading" in beh.columns:
    print(f"Circular mean (meanHeading): {behaves_stats['circular_mean_rad']:.4f} rad "
          f"({behaves_stats['circular_mean_deg']:.2f}°)")
    print(f"Circular SD   (meanHeading): {behaves_stats['circular_sd_rad']:.4f} rad "
          f"({behaves_stats['circular_sd_deg']:.2f}°)")

if "stdHeading" in beh.columns:
    print(f"\nMean of stdHeading: {behaves_stats['stdHeading_mean_rad']:.4f} rad  "
          f"({behaves_stats['stdHeading_mean_deg']:.2f}°)")
    print(f"SD of stdHeading:   {behaves_stats['stdHeading_sd_rad']:.4f} rad  "
          f"({behaves_stats['stdHeading_sd_deg']:.2f}°)")
