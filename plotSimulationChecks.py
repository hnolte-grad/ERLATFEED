# ================================================================
# ANALYZE VALIDATION RESULTS (Advanced)
# ================================================================
# Purpose:
#   Deeper diagnostic analysis of the Blender contact-area validation
#   results from results_validation.csv.
#
# Features:
#   • Convergence curves (area, % diff, compute time)
#   • Consistency check: n_contact_px * px_area_m2 ≈ total_area_m2
#   • Asymptotic (infinite-resolution) area estimate via 1/res fit
#   • Efficiency plot: calc_time / n_contact_px
#   • Summary table with reliability metrics
#
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- CONFIG ----------------
csv_path = r"C:\_temp workspace\1ER-LAT-FEED\data\3d model\checks\results_validation.csv"
dep_filter = None  # e.g. "er210415-58_1.0_2.0_7.0_B53_FE4" or leave None

# ---------------- LOAD ----------------
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}")

df = pd.read_csv(csv_path)
if dep_filter:
    df = df[df["depID"] == dep_filter]

df = df.sort_values("resolution_px").reset_index(drop=True)

# ---------------- BASIC SUMMARY ----------------
print("\n=== Validation Results ===")
print(df[["resolution_px","total_area_m2","pct_diff_vs_max","calc_time_s"]])
print("===================================\n")

# ---------------- CONSISTENCY CHECK ----------------
df["area_from_pixels"] = df["n_contact_px"] * df["px_area_m2"]
df["pct_error_area_calc"] = 100 * (df["total_area_m2"] - df["area_from_pixels"]) / df["total_area_m2"]

mean_err = df["pct_error_area_calc"].mean()
max_err  = df["pct_error_area_calc"].abs().max()
print(f"Mean internal area error: {mean_err:.4f}% | Max: {max_err:.4f}%")

# ---------------- ASYMPTOTIC FIT ----------------
df["inv_res"] = 1 / df["resolution_px"]
coeffs = np.polyfit(df["inv_res"], df["total_area_m2"], 1)
asymptotic_area = coeffs[-1]
print(f"Estimated asymptotic (∞-resolution) area ≈ {asymptotic_area:.6f} m²")

# ---------------- EFFICIENCY ----------------
df["sec_per_px"] = df["calc_time_s"] / df["n_contact_px"]

# ---------------- PLOTS ----------------
plt.figure(figsize=(10, 10))

# --- 1. Area convergence ---
plt.subplot(4, 1, 1)
plt.plot(df["resolution_px"], df["total_area_m2"], "o-", lw=2, label="Measured")
plt.plot(df["resolution_px"], np.polyval(coeffs, df["inv_res"]), "k--", label="∞-res fit")
plt.axhline(asymptotic_area, color="gray", ls=":", label=f"Asymptotic ≈ {asymptotic_area:.3f} m²")
plt.xlabel("Raster resolution (px width)")
plt.ylabel("Contact area (m²)")
plt.title("Convergence of Contact-Area Estimate")
plt.legend()
plt.grid(True, ls="--", alpha=0.6)

# --- 2. Percent difference vs max ---
plt.subplot(4, 1, 2)
plt.plot(df["resolution_px"], df["pct_diff_vs_max"], "s-", color="orange", lw=2)
plt.axhline(0, color="gray", lw=1)
plt.xlabel("Raster resolution (px width)")
plt.ylabel("Δ area vs max (%)")
plt.title("Percent Difference from Highest Resolution")
plt.grid(True, ls="--", alpha=0.6)

# --- 3. Computation time ---
plt.subplot(4, 1, 3)
plt.plot(df["resolution_px"], df["calc_time_s"], "^-", color="green", lw=2)
plt.xlabel("Raster resolution (px width)")
plt.ylabel("Computation time (s)")
plt.title("Processing Time vs Resolution")
plt.grid(True, ls="--", alpha=0.6)

# --- 4. Efficiency per contact pixel ---
plt.subplot(4, 1, 4)
plt.plot(df["resolution_px"], df["sec_per_px"]*1e6, "d-", color="purple", lw=2)
plt.xlabel("Raster resolution (px width)")
plt.ylabel("Time per contact pixel (µs)")
plt.title("Efficiency (µs per Contact Pixel)")
plt.grid(True, ls="--", alpha=0.6)

plt.tight_layout()
plt.show()

# ---------------- SUMMARY METRICS ----------------
std_area = df["total_area_m2"].std()
ref_area = df["total_area_m2"].iloc[-1]
cv = 100 * std_area / ref_area

print("\n=== Summary Metrics ===")
print(f"Asymptotic area (∞-res fit):  {asymptotic_area:.6f} m²")
print(f"Final measured area:          {ref_area:.6f} m²")
print(f"Std. deviation:               {std_area:.6f} m² ({cv:.3f}% of final)")
print(f"Mean area calc error:         {mean_err:.4f}%")
print(f"Mean seconds per pixel:       {df['sec_per_px'].mean():.2e} s")
print("===================================")
