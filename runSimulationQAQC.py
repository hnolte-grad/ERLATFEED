# ================================================================
# QA/QC VALIDATION PLOTTER for results_validation.csv
# ================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

path = r"C:\_temp workspace\1ER-LAT-FEED\data\3d model\surface area\total area\validation\results_validation.csv"
df = pd.read_csv(path)

# --- compute grouped stats (if multiple deps) ---
for depid, g in df.groupby("depID"):
    g = g.sort_values("resolution_px")

    # 1️⃣  Resolution convergence
    plt.figure(figsize=(6,4))
    plt.plot(g["resolution_px"], g["total_area_m2_raw"], "o-", label="Raw area")
    plt.plot(g["resolution_px"], g["total_area_m2_morph"], "o-", label="Morph area")
    plt.xscale("log", base=2)
    plt.xlabel("Resolution (px)")
    plt.ylabel("Total area (m²)")
    plt.title(f"{depid} – Resolution convergence")
    plt.grid(True, ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(path), f"{depid}_QC_convergence.png"), dpi=300)
    plt.close()

    # 2️⃣  Morphological gain stability
    plt.figure(figsize=(5,3))
    plt.plot(g["resolution_px"], g["pct_morph_diff"], "o-", color="red")
    plt.xscale("log", base=2)
    plt.xlabel("Resolution (px)")
    plt.ylabel("Δ Morph (%)")
    plt.title("Morphological correction vs. resolution")
    plt.grid(True, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(path), f"{depid}_QC_morphGain.png"), dpi=300)
    plt.close()

    # 3️⃣  Cluster collapse
    plt.figure(figsize=(5,3))
    plt.plot(g["resolution_px"], g["clusters_raw"], "x--", color="gray", label="Raw clusters")
    plt.plot(g["resolution_px"], g["clusters_corr"], "x--", color="green", label="Corrected clusters")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Resolution (px)")
    plt.ylabel("Cluster count (log scale)")
    plt.title("Connectivity collapse check")
    plt.legend()
    plt.grid(True, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(path), f"{depid}_QC_clusters.png"), dpi=300)
    plt.close()

    # 4️⃣  Residual slope (% change per doubling)
    g["area_slope_pct"] = g["total_area_m2_morph"].pct_change() * 100
    print(f"\n=== {depid} QC SUMMARY ===")
    print(f"  Final area: {g['total_area_m2_morph'].iloc[-1]:.4f} m²")
    print(f"  Mean Δ area per step: {abs(g['area_slope_pct']).mean():.2f}%")
    print(f"  Max clusters raw: {g['clusters_raw'].max()} → min corrected: {g['clusters_corr'].min()}")
