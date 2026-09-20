# ===================================================================
# INTERACTIVE ROLL-RATE ANALYSIS TOOL (MULTI-FILE + HISTOGRAM)
# Developer: Hannah Clayton
# Project: ERLATFEED ms
# Purpose:
#   - Load multiple PRHD/PRH CSVs from a directory
#   - Compute roll rate (deg/s) for each event
#   - Display overlay plot + histograms
#   - Provide interactive slider to adjust roll-rate threshold
# ===================================================================

import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------- USER CONFIG ----------------
input_dir = r"C:\_temp workspace\1ER-LAT-FEED\data\3d model\surface area\feed prhs"  # folder of PRH/PRHD CSVs
pattern = "*-prh*.csv"      # pattern for files to include
time_col = "sec"           # column for time (seconds or datetime)
roll_col = "roll"           # column for roll (radians)
fs = None                   # sampling rate (Hz) if no time column present
max_files = None            # limit number of files (None = all)

# ---------------- FUNCTION DEFINITIONS ----------------

def load_prhd(path):
    """Load roll (rad) and time (s) from a PRHD or PRH CSV."""
    df = pd.read_csv(path)
    if roll_col not in df.columns:
        raise KeyError(f"Missing '{roll_col}' column in {path}")

    if time_col in df.columns:
        t = df[time_col].values
        if np.issubdtype(df[time_col].dtype, np.datetime64):
            t = (t - t[0]) / np.timedelta64(1, "s")
    else:
        if fs is None:
            raise ValueError(f"No '{time_col}' column and no fs provided.")
        t = np.arange(len(df)) / fs

    roll_rad = df[roll_col].to_numpy(dtype=float)
    return t, roll_rad


def compute_roll_rate(t, roll_rad):
    """Compute roll rate in deg/s using gradient over time."""
    roll_deg = np.degrees(roll_rad)
    dt = np.gradient(t)
    roll_rate_deg_s = np.gradient(roll_deg) / dt
    roll_rate_deg_s = np.nan_to_num(roll_rate_deg_s, nan=0.0, posinf=0.0, neginf=0.0)
    return roll_deg, roll_rate_deg_s


# ---------------- MAIN ANALYSIS ----------------
def main():
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {input_dir} matching '{pattern}'")
    if max_files:
        files = files[:max_files]

    print(f"Found {len(files)} PRHD/PRH files.\n")

    all_rates = []
    event_names = []

    for path in files:
        name = os.path.basename(path)
        try:
            t, roll_rad = load_prhd(path)
            roll_deg, roll_rate = compute_roll_rate(t, roll_rad)
        except Exception as e:
            print(f"[SKIP] {name} → {e}")
            continue

        all_rates.append(np.abs(roll_rate))
        event_names.append(name)

        print(f"{name:<45} mean={np.mean(np.abs(roll_rate)):.2f}°/s  "
              f"p95={np.percentile(np.abs(roll_rate),95):.2f}°/s  "
              f"max={np.max(np.abs(roll_rate)):.2f}°/s")

    if not all_rates:
        raise RuntimeError("No valid roll data found.")

    # Flatten all roll-rate values
    all_rates_flat = np.concatenate(all_rates)

    # --- Interactive Plot: Histogram + Threshold Slider ---
    fig = go.Figure()

    # Histogram trace
    fig.add_trace(go.Histogram(
        x=all_rates_flat,
        nbinsx=120,
        marker_color="mediumvioletred",
        opacity=0.75,
        name="|Roll rate| (°/s)"
    ))

    # Initial threshold line (30°/s)
    threshold = 30
    fig.add_shape(
        type="line",
        x0=threshold, x1=threshold, y0=0, y1=1, xref="x", yref="paper",
        line=dict(color="gray", width=2, dash="dash"), name="threshold"
    )

    # Layout
    fig.update_layout(
        title="Distribution of Absolute Roll Rates Across All Events",
        xaxis_title="Roll Rate (°/s)",
        yaxis_title="Frequency",
        bargap=0.05,
        template="plotly_white",
        sliders=[{
            "pad": {"t": 50},
            "currentvalue": {"prefix": "Threshold: ", "suffix": " °/s", "font": {"size": 16}},
            "steps": [
                {
                    "method": "relayout",
                    "label": str(v),
                    "args": [{"shapes[0].x0": v, "shapes[0].x1": v}],
                }
                for v in range(10, 101, 5)
            ]
        }]
    )

    fig.show()

    # --- Per-event 95th percentile plot ---
    p95_values = [np.percentile(r, 95) for r in all_rates]
    df_summary = pd.DataFrame({
        "Event": event_names,
        "95th_percentile": p95_values
    })

    fig2 = px.histogram(
        df_summary,
        x="95th_percentile",
        nbins=20,
        color_discrete_sequence=["darkturquoise"],
        title="Distribution of Event 95th Percentile Roll Rates",
        labels={"95th_percentile": "Event 95th Percentile (°/s)"}
    )
    fig2.add_vline(x=30, line_dash="dash", line_color="gray", annotation_text="30°/s")
    fig2.update_layout(template="plotly_white")
    fig2.show()

    # --- Summary stats ---
    print("\n=== SUMMARY (abs roll rate °/s) ===")
    df_stats = pd.DataFrame({
        "mean": [np.mean(r) for r in all_rates],
        "p95": [np.percentile(r, 95) for r in all_rates],
        "max": [np.max(r) for r in all_rates],
    })
    print(df_stats.describe())
    print(f"\nSuggested threshold range: "
          f"{np.percentile(df_stats['p95'], [10,90])} °/s "
          f"(10th–90th percentile of event 95th percentiles)")

# ---------------- SCRIPT ENTRY ----------------
if __name__ == "__main__":
    main()
