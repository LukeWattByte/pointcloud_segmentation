"""05 - Collision detection MVP."""

# --- 05 - tree-wire collision MVP (collision engine + visualization) ---

# --- 1. Config ---
# %%
from pathlib import Path
import numpy as np

BASE = Path(".").resolve()
DM   = BASE / "colmap_workspace_video_2" / "dense_merged"
UPL  = BASE / "uplift_output"
OUT  = BASE / "collision_output"; OUT.mkdir(exist_ok=True)

PLY_METRIC = DM / "fused_metric.ply"
WIRE_PLY   = UPL / "catenary_spans.ply"
POLE_PLY   = UPL / "pole_tops.ply"
LABELS     = UPL / "labels_wire.npy"

UP = np.array([0.066, 0.058, 0.996]); UP /= np.linalg.norm(UP)

ENVELOPE_R = 2.0

GROW_H     = 1.5
GROW_RATE  = 0.5
BLOWOUT    = 1.0

HOT_EPS = 1.5        # m
HOT_MIN = 20

PLOT_TREE_SUB = 40000
COLOR = {"tree":(120,120,120), "wire":(240,30,240),
         "pole":(250,120,20), "hit":(255,0,0)}
print("inputs exist:",
      "wire", WIRE_PLY.exists(), "| labels", LABELS.exists(),
      "| metric", PLY_METRIC.exists(), "| poles", POLE_PLY.exists())

# --- 2. Load (conductors + tree points + attachment points from 04) ---
# %%
_PLY_T = {'float':'<f4','float32':'<f4','double':'<f8','uchar':'u1','uint8':'u1',
          'int':'<i4','uint':'<u4','short':'<i2','ushort':'<u2','char':'i1'}

def read_ply_auto(path):
    fields = []; n = None
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'ply', "not a ply"
        while True:
            parts = f.readline().split()
            if not parts:
                continue
            k = parts[0]
            if k == b'element' and parts[1] == b'vertex':
                n = int(parts[2])
            elif k == b'property' and parts[1] != b'list':
                fields.append((parts[2].decode(), _PLY_T[parts[1].decode()]))
            elif k == b'end_header':
                break
        return np.fromfile(f, dtype=np.dtype(fields), count=n)

def xyz_of(a):
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)

wire_curve = xyz_of(read_ply_auto(WIRE_PLY))
print("wire samples:", len(wire_curve))

labels = np.load(LABELS).astype(np.int8)
m = read_ply_auto(PLY_METRIC)
Xall = xyz_of(m)
assert len(Xall) == len(labels), "point count mismatch between cloud and labels"
Xt = Xall[labels == 1]
print("tree points:", len(Xt))

poles = xyz_of(read_ply_auto(POLE_PLY)) if POLE_PLY.exists() else np.empty((0,3))
print("pole tops:", len(poles))

_t = np.array([1.0, 0.0, 0.0]); E_A = _t - (_t@UP)*UP; E_A /= np.linalg.norm(E_A)
E_B = np.cross(UP, E_A)
def to2d(P):  return np.stack([P @ E_A, P @ E_B], 1)

# --- 3. Collision kernel (point-to-conductor min distance + envelope test) ---
# %%
from scipy.spatial import cKDTree

kdt = cKDTree(wire_curve)

def clearance(pts):
    d, _ = kdt.query(pts, workers=-1)
    return d

dist_base = clearance(Xt)
print(f"baseline clearance: min {dist_base.min():.2f}m  "
      f"median {np.median(dist_base):.2f}m  "
      f"collisions(<{ENVELOPE_R}m) {int((dist_base<ENVELOPE_R).sum()):,}")

# --- 4. Scenario simulation (growth / wind sway / combined) ---
# %%
dist_grow = clearance(Xt + GROW_H*UP)

SCEN = [
    ("baseline",             dist_base, ENVELOPE_R),
    (f"+growth {GROW_H}m",   dist_grow, ENVELOPE_R),
    (f"+blowout {BLOWOUT}m", dist_base, ENVELOPE_R + BLOWOUT),
    ("+both",                dist_grow, ENVELOPE_R + BLOWOUT),
]
scen_masks = {}
print(f"{'scenario':18s} {'collisions':>12s} {'% trees':>9s}")
for name, d, R in SCEN:
    mk = d < R
    scen_masks[name] = mk
    print(f"{name:18s} {int(mk.sum()):>12,d} {100*mk.mean():>8.2f}%")

worst = scen_masks["+both"]
print("\nworst-case colliding tree points:", int(worst.sum()))

# %%
years = np.clip((dist_base - ENVELOPE_R) / max(GROW_RATE, 1e-6), 0, None)
near = dist_base < (ENVELOPE_R + GROW_RATE*10)
if near.any():
    yy = years[near]
    print(f"approaching tree points {int(near.sum()):,} | years to collision: "
          f"soonest {yy.min():.1f}yr  median {np.median(yy):.1f}yr  "
          f"(points <5yr {int((yy<5).sum()):,})")
else:
    print("no tree points approaching within ~10 years")

# --- 5. Collision hotspot clustering ---
# %%
from sklearn.cluster import DBSCAN
import json

Xh = Xt[worst]
hotspots = []
if len(Xh) >= HOT_MIN:
    hl = DBSCAN(eps=HOT_EPS, min_samples=HOT_MIN//2, n_jobs=-1).fit(Xh).labels_
    for c in sorted(set(hl)):
        if c == -1:
            continue
        pts = Xh[hl == c]
        if len(pts) < HOT_MIN:
            continue
        ctr = pts.mean(0)
        dmin = float(clearance(ctr[None, :])[0])
        rad = float(np.linalg.norm(to2d(pts) - to2d(ctr[None,:]), axis=1).max())
        hotspots.append({"n_pts": int(len(pts)),
                         "center_xyz": [round(float(v),2) for v in ctr],
                         "min_clearance_m": round(dmin,2),
                         "radius_m": round(rad,2)})
hotspots.sort(key=lambda h: h["min_clearance_m"])
print(f"collision hotspots: {len(hotspots)}")
for i, h in enumerate(hotspots):
    print(f"  #{i}: {h['n_pts']:>5d} pts, min clearance {h['min_clearance_m']:.2f}m, "
          f"radius {h['radius_m']:.1f}m, center {h['center_xyz']}")

# --- 6. Visualization ---
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

rng = np.random.default_rng(0)
sub = rng.choice(len(Xt), min(PLOT_TREE_SUB, len(Xt)), replace=False)
Xt_s = Xt[sub]; hit_s = worst[sub]

def _set_equal_3d(ax, P):
    mn, mx = P.min(0), P.max(0); c = (mn+mx)/2; r = (mx-mn).max()/2
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)

fig = plt.figure(figsize=(16, 7))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(*Xt_s[~hit_s].T, s=1, c='#bbbbbb', alpha=0.25, label='tree')
ax.scatter(*wire_curve.T,   s=2, c='magenta', label='conductor')
if len(poles):
    ax.scatter(*poles.T, s=40, c='orange', marker='^', label='pole top')
ax.scatter(*Xt_s[hit_s].T,  s=6, c='red', label='collision')
_set_equal_3d(ax, np.concatenate([Xt_s, wire_curve], 0))
ax.set_title(f"collision scene  (worst-case hits {int(worst.sum()):,})")
ax.legend(loc='upper right', fontsize=8)

ax2 = fig.add_subplot(1, 2, 2)
T2 = to2d(Xt_s); W2 = to2d(wire_curve)
ax2.scatter(T2[~hit_s,0], T2[~hit_s,1], s=1, c='#cccccc', alpha=0.4)
ax2.scatter(W2[:,0], W2[:,1], s=2, c='magenta', label='conductor')
ax2.scatter(T2[hit_s,0], T2[hit_s,1], s=4, c='red', label='collision')
for h in hotspots:
    c2 = to2d(np.array(h["center_xyz"])[None,:])[0]
    ax2.add_patch(plt.Circle(c2, max(h["radius_m"],1.0), fill=False,
                             ec='red', lw=1.5))
    ax2.annotate(f"{h['min_clearance_m']:.1f}m", c2, fontsize=8, color='darkred')
ax2.set_aspect('equal'); ax2.set_title(f"top-down hotspots ({len(hotspots)})")
ax2.legend(loc='upper right', fontsize=8)
plt.tight_layout(); plt.show()

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4))
names = [s[0] for s in SCEN]; counts = [int(scen_masks[n].sum()) for n in names]
bars = a1.bar(names, counts, color=['#4c9','#fb3','#e77','#c33'])
a1.set_ylabel('# collision tree pts'); a1.set_title('collisions by scenario')
for b, c in zip(bars, counts):
    a1.text(b.get_x()+b.get_width()/2, c, f"{c:,}", ha='center', va='bottom', fontsize=9)
a1.tick_params(axis='x', rotation=15)

a2.hist(np.clip(dist_base, 0, ENVELOPE_R*5), bins=60, color='#69a')
a2.axvline(ENVELOPE_R, color='r', ls='--', label=f'envelope {ENVELOPE_R}m')
a2.axvline(ENVELOPE_R+BLOWOUT, color='orange', ls=':', label=f'+blowout {ENVELOPE_R+BLOWOUT}m')
a2.set_xlabel('baseline clearance (m)'); a2.set_ylabel('# tree pts')
a2.set_title('tree-to-conductor clearance'); a2.legend()
plt.tight_layout(); plt.show()

# --- 7. Export (CloudCompare / report) ---
# %%
import json

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

tree_rgb = np.tile(COLOR["tree"], (len(Xt),1)).astype(np.uint8)
tree_rgb[worst] = COLOR["hit"]
parts_xyz = [Xt, wire_curve]
parts_rgb = [tree_rgb, np.tile(COLOR["wire"], (len(wire_curve),1)).astype(np.uint8)]
if len(poles):
    parts_xyz.append(poles); parts_rgb.append(np.tile(COLOR["pole"], (len(poles),1)).astype(np.uint8))
write_ply_rgb(OUT / "collision_worstcase.ply",
              np.concatenate(parts_xyz,0).astype(np.float32),
              np.concatenate(parts_rgb,0).astype(np.uint8))

write_ply_rgb(OUT / "hotspots.ply", Xt[worst].astype(np.float32),
              np.tile(COLOR["hit"], (int(worst.sum()),1)).astype(np.uint8))

report = {
    "envelope_r_m": ENVELOPE_R, "grow_h_m": GROW_H, "blowout_m": BLOWOUT,
    "grow_rate_m_per_yr": GROW_RATE,
    "n_tree_pts": int(len(Xt)),
    "scenarios": {n: int(scen_masks[n].sum()) for n in names},
    "min_clearance_m": float(dist_base.min()),
    "soonest_years_to_collision": float(years[near].min()) if near.any() else None,
    "n_hotspots": len(hotspots), "hotspots": hotspots,
}
(OUT / "collision_report.json").write_text(json.dumps(report, indent=2))
print("wrote -> collision_worstcase.ply / hotspots.ply / collision_report.json ->", OUT)
print("open collision_worstcase.ply in CloudCompare: colliding tree pts red, conductors magenta, poles orange")
