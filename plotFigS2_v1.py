# prt_build_and_plot_v4.py
# ------------------------------------------------------------
# Layout: 2 rows (deployments) x 3 columns
# Columns:
#   1 = tidal height (feet)
#   2 = depth (m)
#   3 = roll (deg)
#
# Aesthetics:
# - tidal height line: black
# - shade where tidal height > 4.5 with light green
# - NO horizontal threshold line
# - depth inverted (y-axis)
# - roll converted radians -> degrees and colored red
# - DT formatted mm/dd/yy HH:mm and rotated
# - Deployments DO NOT share x-axis
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.io import loadmat


CSV_PATHS = [
    r"C:\_temp workspace\1ERLATFEED\data\fig s2\data\prh-tides_er150417-3.csv",
    r"C:\_temp workspace\1ERLATFEED\data\fig s2\data\prh-tides_er240409-71.csv",
]

DATA_DIR = Path(r"C:\_temp workspace\1ERLATFEED\data\fig s2\data")
PRED_SHADE_THRESH = 4.5


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def dep_id_from_csv(csv_path: str | Path) -> str:
    stem = Path(csv_path).stem
    parts = stem.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse depID from {csv_path}")
    return parts[1]


def _unwrap(x):
    while isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()
    return x


def extract_prh_fields(mat_path: str | Path):
    S = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    def from_struct(st, name):
        if hasattr(st, name):
            return _unwrap(getattr(st, name))
        return None

    p = S.get("p", None)
    roll = S.get("roll", None)

    if p is None or roll is None:
        for struct_name in ("d", "prh"):
            if struct_name in S:
                st = S[struct_name]
                if p is None:
                    p = from_struct(st, "p")
                if roll is None:
                    roll = from_struct(st, "roll")

    if p is None or roll is None:
        raise KeyError(f"Could not find p/roll in {mat_path}")

    return np.asarray(p).reshape(-1), np.asarray(roll).reshape(-1)


def build_dataset(csv_path: str | Path):
    csv_path = Path(csv_path)
    dep_id = dep_id_from_csv(csv_path)

    df = pd.read_csv(csv_path)
    df["DT"] = pd.to_datetime(df["DT"], errors="coerce")
    df["Prediction"] = pd.to_numeric(df["Prediction"], errors="coerce")

    mat_path = DATA_DIR / f"{dep_id} 10Hzprh.mat"
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    p, roll = extract_prh_fields(mat_path)

    if len(df) != len(p):
        raise ValueError(f"Length mismatch for {dep_id}")

    out = pd.DataFrame({
        "DT": df["DT"].values,
        "tidal_height_ft": df["Prediction"].values,
        "depth_m": p,
        "roll_deg": np.rad2deg(roll),
    })

    return dep_id, out


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_two_by_three(dep_data, out_png=None):

    n = len(dep_data)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=3,
        figsize=(18, 5 * n),
        sharex=False   # DO NOT share x between deployments
    )

    if n == 1:
        axes = np.array([axes])

    dt_fmt = mdates.DateFormatter("%m/%d/%y %H:%M")

    for i, (dep_id, df) in enumerate(dep_data):

        x = df["DT"]

        # --- tidal height
        ax = axes[i, 0]
        y = df["tidal_height_ft"].values
        ax.plot(x, y, color="black", linewidth=0.9)

        ax.fill_between(
            x,
            y,
            PRED_SHADE_THRESH,
            where=(y > PRED_SHADE_THRESH),
            interpolate=True,
            color="lightgreen",
            alpha=0.35
        )

        ax.set_title("tidal height (feet)")
        ax.set_ylabel(dep_id)
        ax.grid(True, alpha=0.3)

        # --- depth
        ax = axes[i, 1]
        ax.plot(x, df["depth_m"].values, linewidth=0.9)
        ax.set_title("depth (m)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        # --- roll
        ax = axes[i, 2]
        ax.plot(x, df["roll_deg"].values, color="red", linewidth=0.9)
        ax.set_title("roll (deg)")
        ax.grid(True, alpha=0.3)

        # --- DT formatting for all three in this row
        for j in range(3):
            axes[i, j].xaxis.set_major_formatter(dt_fmt)
            for tick in axes[i, j].get_xticklabels():
                tick.set_rotation(35)
                tick.set_horizontalalignment("right")

    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")

    plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    dep_data = []

    for csv_path in CSV_PATHS:
        dep_id, df = build_dataset(csv_path)
        dep_data.append((dep_id, df))

        out_csv = DATA_DIR / f"prt_{dep_id}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")

    out_png = DATA_DIR / "prt_2rows_3cols.png"
    plot_two_by_three(dep_data, out_png)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()