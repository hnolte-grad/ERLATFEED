# prt_build_and_plot_v7.py
# ------------------------------------------------------------
# Ensures er160406-22 is ALWAYS the middle subplot.
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
    r"C:\_temp workspace\1ERLATFEED\data\fig s2\data\prh-tides_er160406-22.csv",  # middle
    r"C:\_temp workspace\1ERLATFEED\data\fig s2\data\prh-tides_er240409-71.csv",
]

# Explicit plotting order
PLOT_ORDER = [
    "er150417-3",
    "er160406-22",  # middle subplot
    "er240409-71",
]

DATA_DIR = Path(r"C:\_temp workspace\1ERLATFEED\data\fig s2\data")
META_PATH = DATA_DIR / "metadata_ERLATFEED.csv"
PRED_SHADE_THRESH = 4.5


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def dep_id_from_csv(csv_path: str | Path) -> str:
    return Path(csv_path).stem.split("_", 1)[1]


def _unwrap(x):
    while isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()
    return x


def extract_roll(mat_path):
    S = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    def from_struct(st, name):
        if hasattr(st, name):
            return _unwrap(getattr(st, name))
        return None

    roll = S.get("roll", None)
    if roll is None:
        for sname in ("d", "prh"):
            if sname in S:
                roll = from_struct(S[sname], "roll")
                if roll is not None:
                    break

    if roll is None:
        raise KeyError(f"Could not find roll in {mat_path}")

    return np.asarray(roll).reshape(-1)


def load_metadata(meta_path):
    meta = pd.read_csv(meta_path)
    meta["tagon"] = pd.to_datetime(meta["tagon"], errors="coerce")
    meta["tagoff"] = pd.to_datetime(meta["tagoff"], errors="coerce")
    return meta


def get_tag_window(meta, dep_id):
    m = meta.loc[meta["depID"].astype(str) == str(dep_id)]
    if m.empty:
        raise KeyError(f"No metadata found for {dep_id}")
    tagon = m.iloc[0]["tagon"]
    tagoff = m.iloc[0]["tagoff"]
    return tagon, tagoff


def build_dataset(csv_path, meta):

    dep_id = dep_id_from_csv(csv_path)
    df = pd.read_csv(csv_path)

    df["DT"] = pd.to_datetime(df["DT"], errors="coerce")
    df["Prediction"] = pd.to_numeric(df["Prediction"], errors="coerce")

    mat_path = DATA_DIR / f"{dep_id} 10Hzprh.mat"
    roll_rad = extract_roll(mat_path)

    if len(df) != len(roll_rad):
        raise ValueError(f"Length mismatch for {dep_id}")

    out = pd.DataFrame({
        "DT": df["DT"],
        "tidal_height_ft": df["Prediction"],
        "roll_deg": np.rad2deg(roll_rad),
    })

    tagon, tagoff = get_tag_window(meta, dep_id)

    out = out.loc[(out["DT"] >= tagon) & (out["DT"] <= tagoff)].reset_index(drop=True)

    return dep_id, out, (tagon, tagoff)


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_roll_and_tide(dep_data, out_png=None):

    # Reorder according to PLOT_ORDER
    dep_dict = {dep_id: (dep_id, df, window) for dep_id, df, window in dep_data}
    ordered_data = [dep_dict[d] for d in PLOT_ORDER if d in dep_dict]

    n = len(ordered_data)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(18, 5 * n), sharex=False)

    if n == 1:
        axes = [axes]

    dt_fmt = mdates.DateFormatter("%m/%d/%y %H:%M")

    for i, (dep_id, df, (tagon, tagoff)) in enumerate(ordered_data):

        ax1 = axes[i]
        x = df["DT"]
        tide = df["tidal_height_ft"].values
        roll = df["roll_deg"].values

        # --- tidal height
        ax1.plot(x, tide, color="black", linewidth=0.9)
        ax1.fill_between(
            x,
            tide,
            PRED_SHADE_THRESH,
            where=(tide > PRED_SHADE_THRESH),
            interpolate=True,
            color="lightgreen",
            alpha=0.35
        )
        ax1.set_ylabel("tidal height (feet)")
        ax1.grid(True, alpha=0.3)

        # --- roll
        ax2 = ax1.twinx()
        ax2.plot(x, roll, color="red", linewidth=0.9, alpha = 0.65)
        ax2.set_ylabel("roll (deg)", color="red")

        # ---- Tick formatting ----
        ax1.xaxis.set_major_formatter(dt_fmt)

        # Set tick label font size = 18
        ax1.tick_params(axis="both", labelsize=12)
        ax2.tick_params(axis="both", labelsize=18)

    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")

    plt.show()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    meta = load_metadata(META_PATH)

    dep_data = []
    for csv_path in CSV_PATHS:
        dep_id, df, window = build_dataset(csv_path, meta)
        dep_data.append((dep_id, df, window))

        out_csv = DATA_DIR / f"prt_{dep_id}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved truncated CSV: {out_csv}")

    out_png = DATA_DIR / "prt_roll_and_tide_truncated.png"
    plot_roll_and_tide(dep_data, out_png)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()