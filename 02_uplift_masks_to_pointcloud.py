"""02 - Uplift 2D masks to 3D point cloud."""

# --- Uplift: project 2D masks onto the point cloud, label each point (tree / pole) ---

# --- 1. Config ---
# %%
from pathlib import Path
import numpy as np

BASE   = Path(".").resolve()
DM     = BASE / "colmap_workspace_video_2" / "dense_merged"
PLY_GEOM   = DM / "fused.ply"
PLY_METRIC = DM / "fused_metric.ply"
SPARSE = DM / "sparse"
MASK_TREE = BASE / "sam_output" / "masks" / "tree"
MASK_POLE = BASE / "sam_output" / "masks" / "pole"
OUT = BASE / "uplift_output"; OUT.mkdir(exist_ok=True)

F, CX, CY, K1 = 1708.51, 1920.0, 1080.0, -0.00315
W4K, H4K = 3840, 2160

ZS = 0.25
TOL_REL, TOL_ABS = 0.03, 0.0

MIN_SEEN  = 3
MIN_VOTES = 2
TREE_FRAC = 0.35
POLE_FRAC = 0.25

COLOR = {0: (110, 110, 110), 1: (60, 200, 60), 2: (250, 120, 20)}  # bg / tree / pole
print("ply:", PLY_GEOM.name, "exists", PLY_GEOM.exists(), "| metric", PLY_METRIC.exists())

# --- 2. Load point cloud / poses (with ply read/write helpers) ---
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

print("loading poses ...")
rec = pycolmap.Reconstruction(str(SPARSE))
POSES = {}
for im in rec.images.values():
    cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    POSES[Path(im.name).stem] = cfw.matrix().astype(np.float64)   # 3x4 world->cam
print("poses:", len(POSES))

# --- 3. Projection + occlusion + voting functions ---
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

# --- 4. Single-frame sanity check (confirm projection + distortion sampling align) ---
# %%
import matplotlib.pyplot as plt

stem = sorted(POSES)[len(POSES)//2]
print("sample:", stem)
u, v, z, front = project(POSES[stem])
vidx, vu, vv = visible_mask(u, v, z, front)
tm = load_mask(MASK_TREE, stem); pm = load_mask(MASK_POLE, stem)
is_t = tm[vv, vu]; is_p = pm[vv, vu]

img = np.asarray(Image.open(DM / "images" / f"{stem}.jpg").convert("RGB"))
sx, sy = img.shape[1]/W4K, img.shape[0]/H4K
plt.figure(figsize=(15, 8)); plt.imshow(img)
plt.scatter(vu[is_t]*sx, vv[is_t]*sy, s=1, c='lime', alpha=0.3, label='tree pts')
plt.scatter(vu[is_p]*sx, vv[is_p]*sy, s=2, c='red', alpha=0.5, label='pole pts')
plt.legend(); plt.axis('off'); plt.title(f"{stem}  (tree {is_t.sum()}, pole {is_p.sum()})")
plt.show()

# --- Check: green points on canopy, red points on poles; a global shift means projection is off ---

# --- 5. Full voting ---
# %%
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

seen       = np.zeros(N, np.int32)
tree_votes = np.zeros(N, np.int32)
pole_votes = np.zeros(N, np.int32)

stems = [s for s in sorted(POSES) if (MASK_TREE/f"{s}.png").exists()]
print("processing", len(stems), "frames")
for stem in tqdm(stems):
    u, v, z, front = project(POSES[stem])
    vidx, vu, vv = visible_mask(u, v, z, front)
    if len(vidx) == 0:
        continue
    seen[vidx] += 1
    tm = load_mask(MASK_TREE, stem)
    if tm is not None:
        t = tm[vv, vu]; tree_votes[vidx[t]] += 1
    pm = load_mask(MASK_POLE, stem)
    if pm is not None:
        p = pm[vv, vu]; pole_votes[vidx[p]] += 1
print("done. seen>0:", int((seen > 0).sum()))

# --- 6. Assign labels + stats ---
# %%
ok = seen >= MIN_SEEN
tf = np.where(ok, tree_votes/np.maximum(seen, 1), 0.0)
pf = np.where(ok, pole_votes/np.maximum(seen, 1), 0.0)

is_tree = ok & (tf >= TREE_FRAC) & (tree_votes >= MIN_VOTES)
is_pole = ok & (pf >= POLE_FRAC) & (pole_votes >= MIN_VOTES)

label = np.zeros(N, np.int8)
label[is_tree] = 1
label[is_pole & (pf >= tf)] = 2
label[is_pole & ~is_tree] = 2

for k, name in [(0,'bg'), (1,'tree'), (2,'pole')]:
    print(f"  {name:4s}: {(label==k).sum():>10,d}  ({100*(label==k).mean():.1f}%)")

# --- 7. Write results ---
# %%
np.save(OUT / "labels.npy", label)

rgb = Cm.copy()
for k in (0, 1, 2):
    if k == 0:
        sel = label == 0
        rgb[sel] = (0.4*Cm[sel] + 0.6*np.array(COLOR[0])).astype(np.uint8)
    else:
        rgb[label == k] = COLOR[k]
write_ply_rgb(OUT / "fused_metric_labeled.ply", Xm, rgb)

for k, name in [(1,'tree'), (2,'pole')]:
    sel = label == k
    write_ply_rgb(OUT / f"{name}_metric.ply", Xm[sel], Cm[sel])
    print(f"  {name}: {int(sel.sum()):,} points -> {name}_metric.ply")

print("\noutput dir:", OUT.resolve())
print("open fused_metric_labeled.ply in CloudCompare / MeshLab to check")

# --- 8. (optional) back-projection self-check ---
# %%
def reproject_iou(stem, cls_label, mask_folder):
    u, v, z, front = project(POSES[stem])
    vidx, vu, vv = visible_mask(u, v, z, front)
    gt = load_mask(mask_folder, stem)
    pred_pts = label[vidx] == cls_label
    gt_pts = gt[vv, vu]
    inter = (pred_pts & gt_pts).sum()
    union = (pred_pts | gt_pts).sum()
    return inter/union if union else float('nan'), int(pred_pts.sum()), int(gt_pts.sum())

import numpy as np
sample_stems = [sorted(POSES)[i] for i in (50, 150, 250, 350, 450) if i < len(POSES)]
for s in sample_stems:
    if not (MASK_TREE/f"{s}.png").exists(): continue
    ti = reproject_iou(s, 1, MASK_TREE)
    pi = reproject_iou(s, 2, MASK_POLE)
    print(f"{s}: tree IoU {ti[0]:.2f} (pred {ti[1]}, gt {ti[2]}) | pole IoU {pi[0]:.2f} (pred {pi[1]}, gt {pi[2]})")
