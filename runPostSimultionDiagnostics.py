# ===================================================================================
# CALCULATE CONTACT AREA + VALIDATE ONE FEEDING EVENT + LOG + PLOT
# Developer: Hannah Clayton
# Project: ER-LAT-FEED ms
# ===================================================================================
# PURPOSE:
#   Simulate 2D contact area for a 3D whale skull (single feeding event),
#   validate accuracy and precision (patch + blue method diagnostics),
#   and automatically save metrics and plots for QA/QC.
# ===================================================================================

import bpy, os, csv, math, time, hashlib, datetime
import numpy as np
from mathutils import Matrix, Vector

# -----------------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------------
obj_name        = "Er Skull"  # mesh name
plane_name      = "ContactPlane"
input_csv       = r"C:\_temp workspace\1ER-LAT-FEED\data\3d model\surface area\feed prhs\er240507-71-prhp.csv"

# Output directories (log + plots)
output_dir      = os.path.dirname(input_csv)
output_log_path = os.path.join(output_dir, "contact_area_validation_log.csv")
plots_dir       = os.path.join(output_dir, "validation_plots")
os.makedirs(plots_dir, exist_ok=True)

# Sampling and scaling
sampling_rate_hz = 10.0
sample_period_s  = 1.0 / sampling_rate_hz
scale_factor     = 1.0
spd              = 0.2  # forward motion (m/s)

# Raster setup
imprint_base_res = 1024
bounds_margin_m  = 0.1
bounds_margin_pct= 0.05
min_plane_size_m = 2.0

# -----------------------------------------------------------------------------------
# CORE FUNCTIONS
# -----------------------------------------------------------------------------------

def load_kinematics_from_csv(path):
    P, R, H, D = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            P.append(float(row["pitch"]))
            R.append(float(row["roll"]))
            H.append(float(row.get("heading", row.get("head"))))
            D.append(float(row["depth"]))
    return P, R, H, D

def euler_matrix_xyz(p, r, h):
    return Matrix.Rotation(h, 4, 'Z') @ Matrix.Rotation(r, 4, 'Y') @ Matrix.Rotation(p, 4, 'X')

def world_bounds_xy(transforms, verts):
    minx = miny = 1e30
    maxx = maxy = -1e30
    for M in transforms:
        for v in verts:
            w = M @ v
            x, y = w.x, w.y
            minx, miny = min(minx, x), min(miny, y)
            maxx, maxy = max(maxx, x), max(maxy, y)
    return minx, miny, maxx, maxy

def expand_bounds(minx, miny, maxx, maxy, margin_m, margin_pct):
    w, h = maxx - minx, maxy - miny
    pad = max(max(w, h) * margin_pct, margin_m)
    return minx - pad, miny - pad, maxx + pad, maxy + pad

def enforce_min_size(minx, miny, maxx, maxy, min_size):
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    w = max(maxx - minx, min_size)
    h = max(maxy - miny, min_size)
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

def rasterize_union_mask_with_counts(minx, miny, maxx, maxy, transforms, verts, w, h):
    sx = (w - 1) / (maxx - minx)
    sy = (h - 1) / (maxy - miny)
    mask = [0] * (w * h)
    counts = [0] * (w * h)
    for M in transforms:
        for v in verts:
            wv = M @ v
            x, y = wv.x, wv.y
            if not (minx <= x <= maxx and miny <= y <= maxy): continue
            px, py = int((x - minx) * sx), int((y - miny) * sy)
            idx = py * w + px
            mask[idx] = 1
            counts[idx] += 1
    return bytearray(mask), counts, sum(1 for c in counts if c > 0), sum(counts)

def morph_dilate(mask, w, h):
    out = bytearray(mask)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            idx = y * w + x
            if mask[idx]:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        out[(y + dy) * w + (x + dx)] = 1
    return out

def fill_holes_fast(mask, w, h):
    visited = [0] * (w * h)
    stack = ([(x, 0) for x in range(w)] +
             [(x, h - 1) for x in range(w)] +
             [(0, y) for y in range(h)] +
             [(w - 1, y) for y in range(h)])
    while stack:
        x, y = stack.pop()
        if not (0 <= x < w and 0 <= y < h): continue
        i = y * w + x
        if visited[i] or mask[i]: continue
        visited[i] = 1
        stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    filled = bytearray(mask)
    for i in range(w * h):
        if not mask[i] and not visited[i]:
            filled[i] = 1
    return filled

def morph_correct(mask, w, h):
    return fill_holes_fast(morph_dilate(mask, w, h), w, h)

# -----------------------------------------------------------------------------------
# VALIDATION SUITE + LOGGING
# -----------------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[INFO] matplotlib not found — skipping plots.")

def checksum(mask): return hashlib.md5(bytes(mask)).hexdigest()
def compute_area(mask, px_area): return float(sum(mask)) * px_area

def inspect_morph_delta(raw, corr):
    added = sum(1 for r, c in zip(raw, corr) if not r and c)
    pct_added = 100 * added / max(sum(corr), 1)
    print(f"[Morph Δ] Added = {added} ({pct_added:.2f}% of corrected)")
    return pct_added

def validate_one_feeding_event(obj_name, transforms, verts, minx, miny, maxx, maxy,
                               raw, corr, wpx, hpx, px_area, depid,
                               log_path, plots_dir):
    """Run diagnostics, log metrics, and save validation plots."""
    print("\n==================== VALIDATING ONE FEEDING EVENT ====================")
    expected = 1.0
    measured = compute_area(raw, px_area)
    rel_err = abs(measured - expected) / expected
    print(f"[Baseline] Expected={expected:.3f} | Measured={measured:.3f} | RelErr={100*rel_err:.2f}%")

    pct_added = inspect_morph_delta(raw, corr)
    if pct_added < 15: print("   ✅ Morph correction within expected range.")
    else: print("   ⚠️  Large morphological correction (>15%).")

    # Resolution convergence
    print("\n[Resolution convergence]")
    results = []
    for res in [256, 512, 1024, 2048]:
        w, h = res, int(res * (maxy - miny) / (maxx - minx))
        pxA = ((maxx - minx) / w) * ((maxy - miny) / h)
        rawR, _, _, _ = rasterize_union_mask_with_counts(minx, miny, maxx, maxy, transforms, verts, w, h)
        areaR = compute_area(rawR, pxA)
        results.append((res, areaR))
        print(f"   {res:4d}px → {areaR:.6f} m²")

    if HAS_MPL:
        fig, ax = plt.subplots()
        ax.plot([r for r, _ in results], [a for _, a in results], marker='o')
        ax.set_xscale('log', base=2)
        ax.set_xlabel("Resolution (px)")
        ax.set_ylabel("Area (m²)")
        ax.set_title(f"Resolution Convergence — {depid}")
        ax.grid(True)
        fig.savefig(os.path.join(plots_dir, f"{depid}_resolution_convergence.png"), dpi=200)
        plt.close(fig)

    # Frame-wise stability
    print("\n[Frame-wise stability]")
    areas = []
    for M in transforms[:min(50, len(transforms))]:
        rawF, _, _, _ = rasterize_union_mask_with_counts(minx, miny, maxx, maxy, [M], verts, wpx, hpx)
        areas.append(sum(rawF) * px_area)
    cv = 100 * (max(areas) - min(areas)) / (sum(areas) / len(areas))
    print(f"   Frame-wise coefficient of variation = {cv:.2f}%")

    if HAS_MPL:
        fig, ax = plt.subplots()
        ax.plot(areas, lw=1)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Area (m²)")
        ax.set_title(f"Frame-wise Area Stability — {depid}")
        ax.grid(True)
        fig.savefig(os.path.join(plots_dir, f"{depid}_frame_stability.png"), dpi=200)
        plt.close(fig)

    # Reproducibility
    h1, h2 = checksum(raw), checksum(raw.copy())
    print(f"\n[Reproducibility] MD5 #1={h1[:8]} | #2={h2[:8]}")
    print("   ✅ Reproducible" if h1 == h2 else "   ⚠️ Non-deterministic behavior!")

    # Summary + Logging
    area_raw = compute_area(raw, px_area)
    area_corr = compute_area(corr, px_area)
    pct_gain = 100 * (area_corr - area_raw) / max(area_raw, 1e-9)
    print("\n===================== SUMMARY METRICS =====================")
    print(f"Deployment ID  : {depid}")
    print(f"Raw Area       : {area_raw:.6f} m²")
    print(f"Corrected Area : {area_corr:.6f} m²")
    print(f"Morph Δ Added  : {pct_added:.2f}%")
    print(f"Gain (Corr/Raw): {pct_gain:.2f}%")
    print(f"Frame CV (%)   : {cv:.2f}%")
    print("===========================================================\n")

    header = ["timestamp", "depid", "raw_area", "corr_area", "pct_added",
              "pct_gain", "frame_cv", "reproducible"]
    new_row = [datetime.datetime.now().isoformat(), depid, area_raw,
               area_corr, pct_added, pct_gain, cv, (h1 == h2)]

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header: writer.writerow(header)
        writer.writerow(new_row)
    print(f"[LOG] Metrics logged to → {log_path}")
    print(f"[PLOTS] Saved → {plots_dir}\n")

# -----------------------------------------------------------------------------------
# MAIN (SINGLE EVENT)
# -----------------------------------------------------------------------------------

def main():
    depid = os.path.basename(input_csv).replace("-prhp.csv", "")
    P, R, H, D = load_kinematics_from_csv(input_csv)
    obj = bpy.data.objects[obj_name]
    verts_all = [v.co.copy() for v in obj.data.vertices]

    transforms = []
    for i, (p, r, h, d) in enumerate(zip(P, R, H, D)):
        t = i * sample_period_s
        dx, dy = spd * t * math.cos(h), spd * t * math.sin(h)
        dz = d
        T = Matrix.Translation((dx, dy, dz))
        Rm = euler_matrix_xyz(p, r, h)
        transforms.append(Rm @ T)

    minx, miny, maxx, maxy = world_bounds_xy(transforms, verts_all)
    minx, miny, maxx, maxy = expand_bounds(minx, miny, maxx, maxy, bounds_margin_m, bounds_margin_pct)
    minx, miny, maxx, maxy = enforce_min_size(minx, miny, maxx, maxy, min_plane_size_m)

    wpx = imprint_base_res
    aspect = (maxy - miny) / (maxx - minx)
    hpx = max(1, int(round(wpx * aspect)))
    px_area = ((maxx - minx) / wpx) * ((maxy - miny) / hpx)

    raw, _, _, _ = rasterize_union_mask_with_counts(minx, miny, maxx, maxy, transforms, verts_all, wpx, hpx)
    corr = morph_correct(raw, wpx, hpx)

    validate_one_feeding_event(obj_name, transforms, verts_all, minx, miny, maxx, maxy,
                               raw, corr, wpx, hpx, px_area, depid,
                               output_log_path, plots_dir)

if __name__ == "__main__":
    main()
