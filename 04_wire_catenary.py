"""04 - Fit catenary curves to wires."""

# --- 04 - wire uplift + catenary fitting (draft) ---

# --- 1. Config ---
# %%
from pathlib import Path
import numpy as np

BASE   = Path(".").resolve()
DM     = BASE / "colmap_workspace_video_2" / "dense_merged"
PLY_GEOM   = DM / "fused.ply"
PLY_METRIC = DM / "fused_metric.ply"
SPARSE = DM / "sparse"
MASK_WIRE = BASE / "sam_output" / "masks" / "wire"
OUT = BASE / "uplift_output"; OUT.mkdir(exist_ok=True)

F, CX, CY, K1 = 1708.51, 1920.0, 1080.0, -0.00315
W4K, H4K = 3840, 2160
ZS = 0.25
TOL_REL, TOL_ABS = 0.03, 0.0

MIN_SEEN  = 2
MIN_VOTES = 2
WIRE_FRAC = 0.15

UP = np.array([0.066, 0.058, 0.996]); UP /= np.linalg.norm(UP)

DBS_EPS = 0.6
DBS_MIN = 8
SPAN_MIN_PTS = 25

SPAN_MIN_LEN   = 4.0
SPAN_LIN_MAX   = 0.15
SPAN_HORIZ_MAX = 0.45

POLE_EPS   = 1.0
POLE_MIN   = 30
POLE_TOP_FRAC = 0.10
ANCHOR_MAX_PERP = 2.0
ANCHOR_MARGIN   = 6.0
ANCHOR_W        = 30

CAT_SAMPLES = 60
CLEAR_STEP  = 0.10

CLEAR_THRESH = 3.0
CLEAR_MAX    = 10.0

COLOR = {0:(110,110,110), 1:(60,200,60), 2:(250,120,20), 3:(240,30,240)}
print("wire masks:", MASK_WIRE.is_dir(),
      "| n:", len(list(MASK_WIRE.glob('*.png'))) if MASK_WIRE.is_dir() else 0)

# --- 2. Load point cloud / poses / existing labels (ply helpers same as 02/03) ---
# %%
import pycolmap

PLY_DT = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                   ('nx','<f4'),('ny','<f4'),('nz','<f4'),
                   ('red','u1'),('green','u1'),('blue','u1')])

def read_ply(path):
    with open(path, 'rb') as f:
        n = None
        while True:
            line = f.readline()
            if line.startswith(b'element vertex'): n = int(line.split()[-1])
            if line.strip() == b'end_header': break
        return np.fromfile(f, dtype=PLY_DT, count=n)

def write_ply_rgb(path, xyz, rgb):
    n = len(xyz)
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode()
    dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    arr = np.empty(n, dt)
    arr['x'],arr['y'],arr['z'] = xyz[:,0],xyz[:,1],xyz[:,2]
    arr['red'],arr['green'],arr['blue'] = rgb[:,0],rgb[:,1],rgb[:,2]
    with open(path,'wb') as f:
        f.write(header); arr.tofile(f)

print("loading clouds ...")
g = read_ply(PLY_GEOM)
Xw = np.stack([g['x'], g['y'], g['z']], 1).astype(np.float64)
m = read_ply(PLY_METRIC)
Xm = np.stack([m['x'], m['y'], m['z']], 1).astype(np.float32)
Cm = np.stack([m['red'], m['green'], m['blue']], 1)
N = len(Xw); print("N points:", N)
assert (g['red'] == m['red']).all(), "point order mismatch!"

lp = OUT / "labels_clean.npy"
labels = np.load(lp if lp.exists() else OUT / "labels.npy").astype(np.int8)
print("base labels:", lp.name if lp.exists() else "labels.npy",
      "| wire so far:", int((labels == 3).sum()))

print("loading poses ...")
rec = pycolmap.Reconstruction(str(SPARSE))
POSES = {}
for im in rec.images.values():
    cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    POSES[Path(im.name).stem] = cfw.matrix().astype(np.float64)
print("poses:", len(POSES))

# --- 3. Projection + occlusion functions (same as 02) ---
# %%
from PIL import Image

GW, GH = int(round(W4K*ZS)), int(round(H4K*ZS))

def project(Rt):
    Xc = Xw @ Rt[:, :3].T + Rt[:, 3]
    z = Xc[:, 2]
    front = z > 1e-6
    zz = np.where(front, z, 1.0)
    a = Xc[:, 0] / zz; b = Xc[:, 1] / zz
    r2 = a*a + b*b
    d = 1.0 + K1*r2
    u = F*a*d + CX
    v = F*b*d + CY
    return u, v, z, front

def visible_mask(u, v, z, front):
    inb = front & (u >= 0) & (u < W4K) & (v >= 0) & (v < H4K)
    idx = np.where(inb)[0]
    ui = u[idx].astype(np.int32); vi = v[idx].astype(np.int32); zc = z[idx]
    gflat = (vi*ZS).astype(np.int32)*GW + (ui*ZS).astype(np.int32)
    zbuf = np.full(GW*GH, np.inf)
    order = np.argsort(zc)[::-1]
    zbuf[gflat[order]] = zc[order]
    vis = zc <= zbuf[gflat]*(1.0+TOL_REL) + TOL_ABS
    return idx[vis], ui[vis], vi[vis]

def load_mask(folder, stem):
    p = folder / f"{stem}.png"
    if not p.exists(): return None
    return np.asarray(Image.open(p)) > 0

# --- 4. Single-frame sanity check (wire-point projection alignment + wire mask hits) ---
# %%
import matplotlib.pyplot as plt

stems_all = [s for s in sorted(POSES) if (MASK_WIRE/f"{s}.png").exists()]
assert stems_all, "no wire mask found - run 01b first"

GOOD_HITS = 150

def _wire_hits(stem):
    u, v, z, front = project(POSES[stem])
    vidx, vu, vv = visible_mask(u, v, z, front)
    wm = load_mask(MASK_WIRE, stem)
    if wm is None or len(vidx) == 0:
        return 0, None
    is_w = wm[vv, vu]
    return int(is_w.sum()), (vu, vv, is_w)

best_n, best_stem, best_packed, n_scanned = -1, None, None, 0
for s in stems_all:
    n_scanned += 1
    n, packed = _wire_hits(s)
    if n > best_n:
        best_n, best_stem, best_packed = n, s, packed
    if best_n >= GOOD_HITS:
        break
print(f"scanned {n_scanned}/{len(stems_all)} frames | best: {best_stem} hits {best_n}")
assert best_n > 0, ("no 3D points hit the wire mask in any frame - go back to 01b to tune the wire mask, "
                    "or wire MVS produced no points (check the wire point count in 03)")

stem = best_stem
vu, vv, is_w = best_packed
print("sample:", stem, "| wire hits:", best_n)

img = np.asarray(Image.open(DM / "images" / f"{stem}.jpg").convert("RGB"))
sx, sy = img.shape[1]/W4K, img.shape[0]/H4K
plt.figure(figsize=(15, 8)); plt.imshow(img)
plt.scatter(vu[is_w]*sx, vv[is_w]*sy, s=3, c='magenta', alpha=0.6, label='wire pts')
plt.legend(); plt.axis('off'); plt.title(f"{stem}  (wire hits {int(is_w.sum())})")
plt.show()

# --- Check: magenta points should land on the wires; 0 hits or all on trees -> retune wire mask in 01b ---

# --- 5. Full voting -> uplift wire points ---
# %%
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

seen       = np.zeros(N, np.int32)
wire_votes = np.zeros(N, np.int32)

print("processing", len(stems_all), "frames")
for stem in tqdm(stems_all):
    u, v, z, front = project(POSES[stem])
    vidx, vu, vv = visible_mask(u, v, z, front)
    if len(vidx) == 0:
        continue
    seen[vidx] += 1
    wm = load_mask(MASK_WIRE, stem)
    if wm is not None:
        w = wm[vv, vu]; wire_votes[vidx[w]] += 1

ok = seen >= MIN_SEEN
wf = np.where(ok, wire_votes/np.maximum(seen, 1), 0.0)
is_wire = ok & (wf >= WIRE_FRAC) & (wire_votes >= MIN_VOTES)
print("uplift wire pts:", int(is_wire.sum()))

labels_wire = labels.copy()
labels_wire[is_wire] = 3
np.save(OUT / "labels_wire.npy", labels_wire)
for k, nm in [(0,'bg'),(1,'tree'),(2,'pole'),(3,'wire')]:
    print(f"  {nm:4s}: {(labels_wire==k).sum():>10,d}")

wsel = np.where(labels_wire == 3)[0]
write_ply_rgb(OUT / "wire_metric_uplift.ply", Xm[wsel], np.tile(COLOR[3], (len(wsel),1)).astype(np.uint8))
print("wrote wire_metric_uplift.ply:", len(wsel), "pts")

# --- 6. Cluster into individual spans ---
# %%
from sklearn.cluster import DBSCAN

Pw = Xm[wsel].astype(np.float64)
db = DBSCAN(eps=DBS_EPS, min_samples=DBS_MIN, n_jobs=-1).fit(Pw)
lab = db.labels_
raw_ids = [c for c in sorted(set(lab)) if c != -1 and (lab == c).sum() >= SPAN_MIN_PTS]

def _span_geom(P):
    c = P.mean(0); Q = P - c
    l, V = np.linalg.eigh(Q.T @ Q / len(P))
    l = l[::-1]; e0 = V[:, ::-1][:, 0]
    lin = float(l[1] / max(l[0], 1e-9))
    horiz = float(abs(e0 @ UP))
    t = Q @ e0
    return float(t.max() - t.min()), lin, horiz

spans, span_ids, dropped = [], [], []
for c in raw_ids:
    P = Pw[lab == c]
    length, lin, horiz = _span_geom(P)
    if length >= SPAN_MIN_LEN and lin <= SPAN_LIN_MAX and horiz <= SPAN_HORIZ_MAX:
        spans.append(P); span_ids.append(c)
    else:
        dropped.append((len(P), length, lin, horiz))
noise = int((lab == -1).sum())
print(f"wire pts {len(Pw)} -> {len(raw_ids)} raw -> {len(spans)} spans kept after geometry filter "
      f"(dropped {len(dropped)}), noise {noise}")
for i, P in enumerate(spans):
    length, lin, horiz = _span_geom(P)
    print(f"  span {i}: {len(P):>6d} pts  len={length:5.1f}m  lin={lin:.3f}  horiz={horiz:.3f}")
if dropped:
    print("\ntop-8 dropped clusters by point count (expected tree/ground/pole false spans):")
    for n, length, lin, horiz in sorted(dropped, reverse=True)[:8]:
        why = []
        if length < SPAN_MIN_LEN:   why.append(f"short({length:.1f}m)")
        if lin > SPAN_LIN_MAX:      why.append(f"thick(lin{lin:.2f})")
        if horiz > SPAN_HORIZ_MAX:  why.append(f"vertical(horiz{horiz:.2f})")
        print(f"  n={n:>5d}  len={length:5.1f}m  lin={lin:.3f}  horiz={horiz:.3f}  <- {'/'.join(why)}")

# --- 7. Extract pole tops (attachment points) ---
# %%
from sklearn.cluster import DBSCAN as _DBSCAN

_tmp = np.array([1.0, 0.0, 0.0])
E_A = _tmp - (_tmp @ UP)*UP; E_A /= np.linalg.norm(E_A)
E_B = np.cross(UP, E_A)

psel = np.where(labels_wire == 2)[0]
Pp = Xm[psel].astype(np.float64)
XYp = np.stack([Pp @ E_A, Pp @ E_B], 1)

GRID = POLE_EPS * 0.5
cell = np.floor(XYp / GRID).astype(np.int64)
uniq, inv, cnt = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()
cell_xy = (uniq + 0.5) * GRID
print(f"pole pts {len(Pp):,} -> {len(uniq)} occupied cells (densest cell {cnt.max():,} pts)")

clab_cells = _DBSCAN(eps=POLE_EPS, min_samples=POLE_MIN).fit(
    cell_xy, sample_weight=cnt.astype(float)).labels_
plab = clab_cells[inv]

pole_tops = []
for c in sorted(set(plab)):
    if c == -1:
        continue
    sel = plab == c
    if sel.sum() < POLE_MIN:
        continue
    pts = Pp[sel]
    zc = pts @ UP
    thr = np.quantile(zc, 1.0 - POLE_TOP_FRAC)
    pole_tops.append(pts[zc >= thr].mean(0))
pole_tops = np.array(pole_tops) if pole_tops else np.empty((0, 3))
print("poles:", len(pole_tops), "attachment points")
if len(pole_tops):
    write_ply_rgb(OUT / "pole_tops.ply", pole_tops.astype(np.float32),
                  np.tile(COLOR[2], (len(pole_tops),1)).astype(np.uint8))
    print("wrote pole_tops.ply")

# --- 8. Fit catenary per span (anchor endpoints at pole tops) ---
# %%
from scipy.optimize import curve_fit

def _catenary(t, a, b, c):
    return c + a*(np.cosh((t - b)/a) - 1.0)

def _pick_anchors(t, e1, hmean, anchors3d):
    if anchors3d is None or len(anchors3d) == 0:
        return np.empty(0), np.empty(0)
    za = anchors3d @ UP
    Qa = (anchors3d - np.outer(za, UP)) - hmean
    ta = Qa @ e1
    perp = np.linalg.norm(Qa - np.outer(ta, e1), axis=1)
    keep = (perp <= ANCHOR_MAX_PERP) & (ta >= t.min()-ANCHOR_MARGIN) & (ta <= t.max()+ANCHOR_MARGIN)
    ta, za = ta[keep], za[keep]
    if len(ta) == 0:
        return np.empty(0), np.empty(0)
    idx = sorted({int(np.argmin(ta)), int(np.argmax(ta))})
    return ta[idx], za[idx]

def fit_span(P, anchors3d=None):
    z = P @ UP
    Ph = P - np.outer(z, UP)
    hmean = Ph.mean(0)
    Q = Ph - hmean
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    e1 = Vt[0]
    t = Q @ e1

    at, az = _pick_anchors(t, e1, hmean, anchors3d)
    n_anchor = int(len(at))
    tt = np.concatenate([t, np.repeat(at, ANCHOR_W)]) if n_anchor else t
    zz = np.concatenate([z, np.repeat(az, ANCHOR_W)]) if n_anchor else z
    lo = min(t.min(), at.min()) if n_anchor else t.min()
    hi = max(t.max(), at.max()) if n_anchor else t.max()
    span_len = float(hi - lo)

    model = "catenary"
    try:
        p, _ = curve_fit(_catenary, tt, zz,
                         p0=[max(span_len, 1.0)*2.0, float(tt.mean()), float(zz.min())],
                         maxfev=20000)
        if not np.isfinite(_catenary(tt, *p)).all():
            raise RuntimeError("nan")
    except Exception:
        model = "quad"
        p = np.polyfit(tt, zz, 2)
    zf = _catenary(t, *p) if model == "catenary" else np.polyval(p, t)
    rms = float(np.sqrt(np.mean((z - zf)**2)))

    def sample(n):
        ts = np.linspace(lo, hi, n)
        zs = _catenary(ts, *p) if model == "catenary" else np.polyval(p, ts)
        return (hmean[None, :] + ts[:, None]*e1[None, :] + zs[:, None]*UP[None, :]).astype(np.float32)

    poly = sample(CAT_SAMPLES)
    n_fine = max(50, int(span_len / max(CLEAR_STEP, 1e-3)))
    poly_fine = sample(n_fine)
    return dict(model=model, rms=rms, span_len=span_len, n=int(len(P)),
                n_anchor=n_anchor, anchored=bool(n_anchor >= 2),
                poly=poly, poly_fine=poly_fine,
                params=[float(x) for x in np.ravel(p)],
                ends=poly[[0, -1]].astype(float).tolist())

results = [fit_span(P, pole_tops) for P in spans]
for i, r in enumerate(results):
    tag = f"anchored({r['n_anchor']})" if r['anchored'] else f"free({r['n_anchor']})"
    print(f"span {i}: n={r['n']:>5d}  len={r['span_len']:6.1f}m  {r['model']:8s}  "
          f"rms={r['rms']:.3f}m  {tag}")

# --- 9. Write: fitted polyline + parameters ---
# %%
import json

if results:
    poly_all = np.concatenate([r["poly"] for r in results], 0)
    write_ply_rgb(OUT / "catenary_spans.ply", poly_all,
                  np.tile(COLOR[3], (len(poly_all),1)).astype(np.uint8))
    print("wrote catenary_spans.ply:", len(poly_all), "sampled pts,", len(results), "spans")
else:
    print("no fittable spans - check wire uplift point count / DBS_EPS")

params = [{k: r[k] for k in ("model","rms","span_len","n","n_anchor","anchored","params","ends")}
          for r in results]
(OUT / "catenary_params.json").write_text(json.dumps(params, indent=2))
print("wrote catenary_params.json")
print("\nopen fused_metric_labeled_clean.ply + catenary_spans.ply + pole_tops.ply together in CloudCompare")

# --- 10. Tree-wire clearance analysis ---
# %%
from scipy.spatial import cKDTree
import json

assert results, "no spans - get the upstream steps working first"
wire_curve = np.concatenate([r["poly_fine"] for r in results], 0).astype(np.float64)

tsel = np.where(labels_wire == 1)[0]
Xt = Xm[tsel].astype(np.float64)
kdt = cKDTree(wire_curve)
dist, _ = kdt.query(Xt, workers=-1)
viol = dist < CLEAR_THRESH
print(f"tree pts {len(Xt):,} | intrusions(<{CLEAR_THRESH}m): {int(viol.sum()):,} "
      f"({100*viol.mean():.2f}%) | min clearance {dist.min():.2f}m")

r01 = np.clip(dist / CLEAR_MAX, 0, 1)
rgb_t = np.empty((len(Xt), 3), np.uint8)
rgb_t[:, 0] = (255 * np.clip(2*(1-r01), 0, 1)).astype(np.uint8)   # R
rgb_t[:, 1] = (255 * np.clip(2*r01, 0, 1)).astype(np.uint8)       # G
rgb_t[:, 2] = 40
rgb_t[viol] = (255, 0, 0)

xyz_out = np.concatenate([Xt, wire_curve], 0).astype(np.float32)
rgb_out = np.concatenate([rgb_t, np.tile(COLOR[3], (len(wire_curve),1))], 0).astype(np.uint8)
write_ply_rgb(OUT / "clearance_tree.ply", xyz_out, rgb_out)
write_ply_rgb(OUT / "clearance_violations.ply", Xt[viol].astype(np.float32),
              np.tile((255,0,0), (int(viol.sum()),1)).astype(np.uint8))

tree_kdt = cKDTree(Xt)
per_span = []
for i, r in enumerate(results):
    d_i, _ = tree_kdt.query(r["poly_fine"].astype(np.float64), workers=-1)
    within = tree_kdt.query_ball_point(r["poly_fine"].astype(np.float64), CLEAR_THRESH)
    n_within = len({j for sub in within for j in sub})
    per_span.append({"span": i, "len_m": round(r["span_len"], 2),
                     "min_tree_clearance_m": round(float(d_i.min()), 3),
                     "n_tree_within_thresh": int(n_within),
                     "anchored": r["anchored"]})
    print(f"  span {i}: nearest tree {d_i.min():5.2f}m, tree pts within threshold {n_within}")

report = {"clear_thresh_m": CLEAR_THRESH, "n_tree_pts": int(len(Xt)),
          "n_violation": int(viol.sum()), "frac_violation": float(viol.mean()),
          "min_clearance_m": float(dist.min()), "per_span": per_span}
(OUT / "clearance_report.json").write_text(json.dumps(report, indent=2))
print("\nwrote clearance_tree.ply / clearance_violations.ply / clearance_report.json ->", OUT)

# %%
plt.figure(figsize=(9,4))
plt.hist(np.clip(dist, 0, CLEAR_MAX*1.5), bins=60)
plt.axvline(CLEAR_THRESH, color='r', ls='--', label=f'thresh {CLEAR_THRESH}m')
plt.xlabel('tree->wire clearance (m)'); plt.ylabel('# tree pts'); plt.legend()
plt.title('vegetation-to-conductor clearance'); plt.show()

# --- 11. (optional) back-projection self-check ---
# %%
stem = stems_all[len(stems_all)//2]
u, v, z, front = project(POSES[stem])
vidx, vu, vv = visible_mask(u, v, z, front)
Lw = labels_wire[vidx] == 3
img = np.asarray(Image.open(DM / "images" / f"{stem}.jpg").convert("RGB"))
sx, sy = img.shape[1]/W4K, img.shape[0]/H4K
plt.figure(figsize=(15, 8)); plt.imshow(img)
plt.scatter(vu[Lw]*sx, vv[Lw]*sy, s=3, c='magenta', alpha=0.6, label='wire (uplift)')
plt.legend(); plt.axis('off'); plt.title(stem + "  wire uplift reproject")
plt.show()
