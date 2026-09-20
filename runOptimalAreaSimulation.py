#!/usr/bin/env python3
# ==============================================================
# Standalone Contact-Area + Morphology + (Optional) Max-Area Planner
# - No Blender. Uses OBJ 'v' vertices and PRHD CSV (radians, meters).
# - Outputs: diagnostic PNG, a per-run CSV row, and console summary.
# ==============================================================
import os, csv, math, time, struct, zlib, argparse, random
import numpy as np
import pandas as pd

# -------------------- CONFIG DEFAULTS --------------------
IMPRINT_BASE_RES = 1024          # width in px; height from aspect
BOUNDS_MARGIN_M  = 0.10
BOUNDS_MARGIN_P  = 0.05
MIN_PLANE_SIZE_M = 2.0
VERTEX_STRIDE    = 1
TRANSFORM_STRIDE = 2
SAVE_OVERLAY     = True
COLOR_RAW        = (0,0,0,255)           # black
COLOR_ADDED      = (255,105,180,255)     # hot pink
COLOR_CLEAR      = (0,0,0,0)
SCALE_BAR_COLOR  = (0,0,0,255)
SCALE_BAR_LEN_M  = 1.0
SCALE_BAR_H_PX   = 6

# -------------------- IO HELPERS --------------------
def load_vertices_from_obj(path, stride=1):
    """Load only 'v' lines from OBJ; return (N,3) float32 array."""
    vs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "): continue
            _, xs, ys, zs, *rest = line.strip().split()
            vs.append((float(xs), float(ys), float(zs)))
    if not vs:
        raise RuntimeError(f"No vertices found in OBJ: {path}")
    V = np.asarray(vs, dtype=np.float32)
    return V[::max(1, stride)]

def load_prhd_csv(path, step=1):
    P,R,H,D = [],[],[],[]
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for i,row in enumerate(r):
            if i % max(1, step): continue
            P.append(float(row["pitch"]))
            R.append(float(row["roll"]))
            h = row.get("head", row.get("heading"))
            if h is None: raise KeyError("CSV missing 'head'/'heading'")
            H.append(float(h))
            D.append(float(row["depth"]))
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if P.size == 0: raise RuntimeError("No PRHD rows after downsample.")
    return P,R,H,D

# -------------------- MATH HELPERS --------------------
def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=np.float64)

def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=np.float64)

def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=np.float64)

def euler_matrix_xyz(p, r, h):
    # Rz(h) @ Ry(r) @ Rx(p)
    return rot_z(h) @ rot_y(r) @ rot_x(p)

def _wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def circ_var_rad(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 2: return 0.0
    C = np.mean(np.cos(a)); S = np.mean(np.sin(a))
    R = math.hypot(C,S)
    return float(max(0.0, min(1.0, 1.0 - R)))

def coverage_frac_rad(a, bins=36):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 2: return 0.0
    width = (2.0 * math.pi) / bins
    a01 = (a + math.pi) % (2.0 * math.pi)
    idx = np.floor(a01 / width).astype(int)
    idx = np.clip(idx, 0, bins-1)
    return float(np.unique(idx).size) / float(bins)

def turns_and_span_rad(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 2: return 0.0, 0.0
    unwrapped = [a[0]]
    for x in a[1:]:
        prev = unwrapped[-1]
        d = x - prev
        while d <= -math.pi: d += 2*math.pi
        while d >   math.pi: d -= 2*math.pi
        unwrapped.append(prev + d)
    u = np.asarray(unwrapped)
    total = np.sum(np.abs(np.diff(u)))
    return float(total / (2*math.pi)), float(np.degrees(u.max() - u.min()))

# -------------------- BOUNDS --------------------
def world_bounds_xy(transforms, V):
    """Transforms: list of (R,d) where position offset is (0,0,-depth) already applied to z before projection."""
    minx=miny= 1e30
    maxx=maxy=-1e30
    for (R, dz) in transforms:
        # world = R @ V.T ; then translate z by -depth if you need – but for XY, translation in z doesn't matter.
        W = (R @ V.T).T
        x = W[:,0]; y = W[:,1]
        minx = min(minx, float(x.min())); maxx = max(maxx, float(x.max()))
        miny = min(miny, float(y.min())); maxy = max(maxy, float(y.max()))
    return minx, miny, maxx, maxy

def expand_bounds(minx, miny, maxx, maxy, margin_m, margin_pct):
    w = maxx - minx; h = maxy - miny
    pad = max(max(w,h)*margin_pct, margin_m)
    return (minx-pad, miny-pad, maxx+pad, maxy+pad)

def enforce_min_size(minx, miny, maxx, maxy, min_size):
    cx, cy = 0.5*(minx+maxx), 0.5*(miny+maxy)
    w = max(maxx - minx, min_size)
    h = max(maxy - miny, min_size)
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)

# -------------------- RASTERIZATION --------------------
def rasterize_union_mask(minx, miny, maxx, maxy, transforms, V, w, h):
    """transforms: list of (R, depth) ; V: (N,3)"""
    mask = np.zeros((h, w), dtype=np.uint8)
    spanx = maxx - minx; spany = maxy - miny
    sx = (w - 1) / spanx; sy = (h - 1) / spany
    for (R, depth) in transforms:
        W = (R @ V.T).T     # (N,3)
        x = W[:,0]; y = W[:,1]
        # map to pixels
        px = ((x - minx) * sx).astype(np.int64)
        py = ((y - miny) * sy).astype(np.int64)
        # in-bounds
        m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        mask[py[m], px[m]] = 1
    return mask

# -------------------- MORPHOLOGY --------------------
def morph_dilate(mask):
    """3x3 dilation once (uint8)"""
    h, w = mask.shape
    out = mask.copy()
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx==0 and dy==0:
                continue
            out[1:h-1,1:w-1] |= mask[1+dy:h-1+dy,1+dx:w-1+dx]
    return out

def fill_holes_fast(mask):
    """Flood fill from edges on background; fill enclosed holes."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    stack = [(x,0) for x in range(w)] + [(x,h-1) for x in range(w)] + [(0,y) for y in range(h)] + [(w-1,y) for y in range(h)]
    while stack:
        x,y = stack.pop()
        if x<0 or x>=w or y<0 or y>=h: continue
        if visited[y,x] or mask[y,x]:   continue
        visited[y,x] = 1
        stack.extend(((x+1,y),(x-1,y),(x,y+1),(x,y-1)))
    filled = mask.copy()
    holes = (mask==0) & (visited==0)
    filled[holes] = 1
    return filled

def morph_correct(mask):
    return fill_holes_fast(morph_dilate(mask))

# -------------------- PNG --------------------
def draw_scale_bar(px, w, h, px_area, color, length_m, height_px):
    ppm = (1.0/px_area)**0.5
    bar_w = int(length_m * ppm)
    bar_w = max(1, min(bar_w, w//2))
    margin = int(h*0.05)
    y0, y1 = h - margin - height_px, h - margin
    x1, x0 = w - margin, w - margin - bar_w
    x0 = max(0,x0); x1=min(w,x1); y0=max(0,y0); y1=min(h,y1)
    r,g,b,a = color
    for y in range(y0,y1):
        o = (y*w + x0)*4
        px[o:o+4*(x1-x0)] = bytes([r,g,b,a]) * (x1-x0)

def save_overlay_png(path, raw, corr, px_area, color_raw, color_added, color_clear, scale_color):
    """raw/corr are (h,w) uint8 masks."""
    h, w = raw.shape
    buf = bytearray(4*w*h)
    for i in range(h*w):
        if corr.ravel()[i]:
            c = color_raw if raw.ravel()[i] else color_added
        else:
            c = color_clear
        off = i*4
        buf[off:off+4] = bytes(c)
    draw_scale_bar(buf, w, h, px_area, scale_color, SCALE_BAR_LEN_M, SCALE_BAR_H_PX)
    # write PNG
    rows = b"".join(b"\x00"+buf[y*w*4:(y+1)*w*4] for y in range(h))
    comp = zlib.compress(rows, 9)
    def chunk(tag, data):
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag+data) & 0xffffffff)
    with open(path,"wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack("!2I5B", w, h, 8, 6, 0, 0, 0)))
        f.write(chunk(b"IDAT", comp))
        f.write(chunk(b"IEND", b""))

# -------------------- MAX-AREA PLANNER (GREEDY) --------------------
def make_candidate_library(P_range, R_range, H_range, D_range, nP=5, nR=5, nH=24, nD=3):
    Ps = np.linspace(P_range[0], P_range[1], max(1,nP))
    Rs = np.linspace(R_range[0], R_range[1], max(1,nR))
    Hs = np.linspace(H_range[0], H_range[1], max(1,nH), endpoint=False)
    Ds = np.linspace(D_range[0], D_range[1], max(1,nD))
    lib = np.array([(p,r,h,d) for p in Ps for r in Rs for h in Hs for d in Ds], dtype=np.float64)
    return lib

def evaluate_pose_mask(pose, V, minx, miny, maxx, maxy, w, h):
    p,r,h,d = pose
    R = euler_matrix_xyz(p,r,h)
    # z-translation by depth does not affect XY projection
    spanx = maxx - minx; spany = maxy - miny
    sx = (w-1)/spanx; sy = (h-1)/spany
    W = (R @ V.T).T
    px = ((W[:,0] - minx)*sx).astype(np.int64)
    py = ((W[:,1] - miny)*sy).astype(np.int64)
    mask = np.zeros((h,w), dtype=np.uint8)
    m = (px>=0)&(px<w)&(py>=0)&(py<h)
    mask[py[m], px[m]] = 1
    return mask

def greedy_plan(V, bounds, grid, P_range, R_range, H_range, D_range,
                n_frames=32, nP=5, nR=5, nH=24, nD=3):
    minx,miny,maxx,maxy = bounds
    w,h,px_area = grid
    poses = make_candidate_library(P_range, R_range, H_range, D_range, nP,nR,nH,nD)
    masks = [evaluate_pose_mask(p, V, minx,miny,maxx,maxy, w,h) for p in poses]
    union = np.zeros((h,w), dtype=np.uint8)
    chosen = []
    for _ in range(n_frames):
        best_idx, best_gain = -1, 0
        for i, m in enumerate(masks):
            if i in chosen: continue
            gain = np.sum((m==1) & (union==0))
            if gain > best_gain:
                best_gain = gain; best_idx = i
        if best_idx < 0 or best_gain == 0: break
        union |= masks[best_idx]
        chosen.append(best_idx)
    area_m2 = float(np.sum(union) * px_area)
    planned = poses[chosen] if chosen else np.zeros((0,4))
    return planned, area_m2, union

# -------------------- PIPELINE --------------------
def run_standalone(mesh_obj_path, prhd_csv, out_dir,
                   vertex_stride=VERTEX_STRIDE, transform_stride=TRANSFORM_STRIDE,
                   base_res=IMPRINT_BASE_RES):
    os.makedirs(out_dir, exist_ok=True)
    depid = os.path.splitext(os.path.basename(prhd_csv))[0].replace("-prhp","")
    V = load_vertices_from_obj(mesh_obj_path, stride=vertex_stride)

    P,R,H,D = load_prhd_csv(prhd_csv, step=1)   # you can downsample here if desired
    transforms_all = [(euler_matrix_xyz(p,r,h), d) for p,r,h,d in zip(P,R,H,D)]
    transforms = transforms_all[::max(1, transform_stride)]

    # Bounds
    minx,miny,maxx,maxy = world_bounds_xy(transforms, V)
    minx,miny,maxx,maxy = expand_bounds(minx,miny,maxx,maxy, BOUNDS_MARGIN_M, BOUNDS_MARGIN_P)
    minx,miny,maxx,maxy = enforce_min_size(minx,miny,maxx,maxy, MIN_PLANE_SIZE_M)

    # Raster grid
    width_m  = maxx - minx; height_m = maxy - miny
    aspect = height_m/width_m if width_m>0 else 1.0
    wpx = int(base_res); hpx = max(1, int(round(wpx * aspect)))
    px_area = (width_m / wpx) * (height_m / hpx)

    # Rasterize
    t0 = time.time()
    raw = rasterize_union_mask(minx,miny,maxx,maxy, transforms, V, wpx, hpx)
    corr = morph_correct(raw)
    calc_time = time.time() - t0

    area_raw  = float(raw.sum()  * px_area)
    area_corr = float(corr.sum() * px_area)
    pct_gain  = (area_corr/area_raw - 1.0)*100.0 if area_raw>0 else 0.0

    # Heading metrics (all frames)
    h_cvar = circ_var_rad(H)
    h_cov  = coverage_frac_rad(H, bins=36)
    h_turns, h_span_deg = turns_and_span_rad(H)

    # Save overlay
    if SAVE_OVERLAY:
        overlay_path = os.path.join(out_dir, f"{depid}_diag.png")
        save_overlay_png(overlay_path, raw, corr, px_area, COLOR_RAW, COLOR_ADDED, COLOR_CLEAR, SCALE_BAR_COLOR)

    # Optional: planner (compare max-area under same bounds/grid)
    P_range = (float(P.min()), float(P.max()))
    R_range = (float(R.min()), float(R.max()))
    H_range = (0.0, 2.0*math.pi)     # allow full circle
    D_range = (float(D.min()), float(D.max()))
    planned, max_area, union_pl = greedy_plan(
        V, (minx,miny,maxx,maxy), (wpx,hpx,px_area),
        P_range, R_range, H_range, D_range,
        n_frames=min(32, len(transforms)), nP=5, nR=5, nH=24, nD=3
    )

    # Results row
    row = {
        "depID": depid,
        "z_plane_m": -float(D.max()),
        "area_raw_m2": round(area_raw,6),
        "area_corr_m2": round(area_corr,6),
        "pct_morph_gain": round(pct_gain,3),
        "px_area_m2": px_area,
        "calc_time_s": round(calc_time,3),
        "heading_circ_var": round(h_cvar,4),
        "heading_coverage_frac": round(h_cov,4),
        "heading_turns": round(h_turns,2),
        "heading_span_deg": round(h_span_deg,1),
        "n_frames_total": int(len(P)),
        "n_frames_used": int(len(transforms)),
        "transform_stride": int(transform_stride),
        "planned_frames": int(planned.shape[0]),
        "planned_area_m2": round(max_area,6),
        "area_headroom_pct": round(100.0*(max_area - area_corr)/max(1e-9, area_corr), 2)
    }
    return row

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Standalone skull contact area (no Blender).")
    ap.add_argument("--mesh", required=True, help="Path to skull OBJ (uses only 'v' lines).")
    ap.add_argument("--prhd", required=True, help="Path to *-prhp.csv with pitch,roll,head/heading,depth (radians, meters).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--vstride", type=int, default=VERTEX_STRIDE)
    ap.add_argument("--tstride", type=int, default=TRANSFORM_STRIDE)
    ap.add_argument("--res", type=int, default=IMPRINT_BASE_RES)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    row = run_standalone(args.mesh, args.prhd, args.outdir, args.vstride, args.tstride, args.res)

    # append results.csv in outdir
    csv_path = os.path.join(args.outdir, "standalone_results.csv")
    exists = os.path.isfile(csv_path)
    cols = [
        "depID","z_plane_m","area_raw_m2","area_corr_m2","pct_morph_gain",
        "px_area_m2","calc_time_s",
        "heading_circ_var","heading_coverage_frac","heading_turns","heading_span_deg",
        "n_frames_total","n_frames_used","transform_stride",
        "planned_frames","planned_area_m2","area_headroom_pct"
    ]
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists: w.writeheader()
        w.writerow(row)
    print("\n== SUMMARY ==")
    for k,v in row.items():
        print(f"{k}: {v}")
    print(f"\nOverlay PNG (if enabled) and results saved to: {args.outdir}")

if __name__ == "__main__":
    main()
