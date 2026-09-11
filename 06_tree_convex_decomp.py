"""06 - Tree convex decomposition."""

# --- 06 - Tree -> convex decomposition physical boundary (CoACD / V-HACD approach) ---

# --- 0. Dependencies ---
# %%
# import sys; !{sys.executable} -m pip install coacd

# %%
from pathlib import Path
import numpy as np, time, json

BASE = Path(".").resolve()
DM   = BASE / "colmap_workspace_video_2" / "dense_merged"   # fused_metric.ply
OUT  = BASE / "uplift_output"
TCV  = OUT / "tree_convex"; TCV.mkdir(exist_ok=True)
UP   = np.array([0.066, 0.058, 0.996]); UP /= np.linalg.norm(UP)

TREE_VOX     = 0.20
EPS_TREE     = 0.8
MIN_TREE     = 50
TREE_MINPTS  = 1500
MAX_TREES    = None

STRIP_TRUNK  = True
TREE_VERT_MAX, TREE_LIN_MIN, TREE_THICK_MAX = 0.85, 0.85, 0.20

COACD_THRESHOLD = 0.05
COACD_MAX_HULLS = 32
ALPHA_MESH      = 1.0

HAVE = {}
for m in ["scipy", "sklearn", "trimesh", "coacd", "matplotlib"]:
    try: __import__(m); HAVE[m] = True
    except Exception: HAVE[m] = False
USE_COACD = HAVE["coacd"]
print("deps:", HAVE)
print("convex decomposition engine:", "CoACD" if USE_COACD else "KMeans+convex hull (fallback; install coacd and re-run to upgrade)")

# %%
PLY_DT = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('nx','<f4'),('ny','<f4'),('nz','<f4'),
                   ('red','u1'),('green','u1'),('blue','u1')])
def read_ply(p):
    with open(p,'rb') as f:
        n=None
        while True:
            line=f.readline()
            if line.startswith(b'element vertex'): n=int(line.split()[-1])
            if line.strip()==b'end_header': break
        return np.fromfile(f,dtype=PLY_DT,count=n)
def write_ply(p,xyz,rgb):
    n=len(xyz); h=("ply\nformat binary_little_endian 1.0\nelement vertex "+str(n)+"\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()
    dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    a=np.empty(n,dt); a['x'],a['y'],a['z']=xyz[:,0],xyz[:,1],xyz[:,2]
    a['red'],a['green'],a['blue']=rgb[:,0],rgb[:,1],rgb[:,2]
    with open(p,'wb') as f: f.write(h); a.tofile(f)
def voxel_ds(P, v):
    key=np.floor(P/v).astype(np.int64)
    _,idx=np.unique(key, axis=0, return_index=True)
    return P[idx]

m=read_ply(DM/"fused_metric.ply")
X=np.stack([m['x'],m['y'],m['z']],1).astype(np.float64)
lab=np.load(OUT/"labels_clean.npy").astype(np.int8)
assert len(X)==len(lab), (len(X), len(lab))
Xt = X[lab==1]
print(f"tree points: {len(Xt):,}")

# --- 1. Split into individual trees + (optional) remove trunk line-like points ---
# %%
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def pole_like(P, k=12):
    nn=NearestNeighbors(n_neighbors=min(k,len(P))).fit(P)
    _,ind=nn.kneighbors(P)
    vert=np.empty(len(P)); lin=np.empty(len(P)); thick=np.empty(len(P))
    for s in range(0,len(P),100000):
        sl=slice(s,min(s+100000,len(P)))
        Q=P[ind[sl]]-P[ind[sl]].mean(1,keepdims=True)
        C=np.einsum('nki,nkj->nij',Q,Q)/Q.shape[1]
        w,V=np.linalg.eigh(C)
        ax=V[...,2]
        vert[sl]=np.abs(np.einsum('ni,i->n',ax,UP))
        l2=np.clip(w[:,2],1e-12,None); l1=np.clip(w[:,1],0,None); l0=np.clip(w[:,0],0,None)
        lin[sl]=(w[:,2]-w[:,1])/l2
        thick[sl]=np.sqrt(l1)
    return (vert>TREE_VERT_MAX)&(lin>TREE_LIN_MIN)&(thick<TREE_THICK_MAX)

Tds = voxel_ds(Xt, TREE_VOX)
print(f"{len(Xt):,} -> {len(Tds):,} voxels @ {TREE_VOX} m")
if STRIP_TRUNK:
    drop = pole_like(Tds)
    print(f"removed trunk/pole-like points {int(drop.sum()):,}  remaining {int((~drop).sum()):,}")
    Tds = Tds[~drop]

lab_t = DBSCAN(eps=EPS_TREE, min_samples=MIN_TREE, n_jobs=-1).fit_predict(Tds)
inst=[]
for cid in np.unique(lab_t):
    if cid<0: continue
    P=Tds[lab_t==cid]
    if len(P) < TREE_MINPTS*len(Tds)/max(len(Xt),1):
        continue
    inst.append(P)
inst.sort(key=len, reverse=True)
if MAX_TREES: inst=inst[:MAX_TREES]
print(f"{len(inst)} tree instances; point counts {[len(P) for P in inst]}")

# --- 2. Convex decomposition core ---
# %%
from scipy.spatial import Delaunay, ConvexHull

def alpha_mesh(P, alpha):
    if len(P) < 4: return P, np.empty((0,3), int)
    tri = Delaunay(P)
    T = tri.simplices
    A=P[T[:,0]]; B=P[T[:,1]]; C=P[T[:,2]]; D=P[T[:,3]]
    def sq(v): return (v*v).sum(-1)
    a2,b2,c2,d2 = sq(A),sq(B),sq(C),sq(D)
    M   = np.stack([B-A, C-A, D-A], axis=1)
    rhs = 0.5*np.stack([b2-a2, c2-a2, d2-a2], axis=-1)     # (nT,3)
    det = np.linalg.det(M)
    good = np.abs(det) > 1e-12
    Ms  = np.where(good[:,None,None], M, np.eye(3))
    cen = np.linalg.solve(Ms, rhs[...,None])[...,0]
    R   = np.linalg.norm(cen - A, axis=1)
    R[~good] = np.inf
    keep = T[np.isfinite(R) & (R < alpha)]
    if len(keep)==0: return P, np.empty((0,3), int)
    faces = np.concatenate([keep[:,[0,1,2]],keep[:,[0,1,3]],keep[:,[0,2,3]],keep[:,[1,2,3]]])
    faces = np.sort(faces, axis=1)
    uniq, cnt = np.unique(faces, axis=0, return_counts=True)
    return P, uniq[cnt==1]

def decompose_coacd(P):
    import coacd
    try: coacd.set_log_level("error")
    except Exception: pass
    V, F = alpha_mesh(P, ALPHA_MESH)
    if len(F) < 4:
        h=ConvexHull(P); return [(P[np.unique(h.simplices)], None)]
    mesh = coacd.Mesh(np.asarray(V,dtype=np.float64), np.asarray(F,dtype=np.int32))
    parts = coacd.run_coacd(mesh, threshold=COACD_THRESHOLD,
                            max_convex_hull=COACD_MAX_HULLS, preprocess_mode="auto")
    return [(np.asarray(v,float), np.asarray(f,int)) for v,f in parts]

def decompose_kmeans(P, part_pts=1200):
    from sklearn.cluster import KMeans
    k = int(np.clip(round(len(P)/part_pts), 1, COACD_MAX_HULLS))
    if k<=1 or len(P)<8:
        h=ConvexHull(P); return [(P, h.simplices)]
    lbl = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(P)
    out=[]
    for c in range(k):
        Q=P[lbl==c]
        if len(Q)<4: continue
        try: h=ConvexHull(Q); out.append((Q, h.simplices))
        except Exception: pass
    return out

def decompose_tree(P):
    return decompose_coacd(P) if USE_COACD else decompose_kmeans(P)

# %%
trees=[]
t0=time.time()
for i,P in enumerate(inst):
    ts=time.time()
    try:
        parts=decompose_tree(P)
    except Exception as e:
        print(f"  tree {i}: failed {type(e).__name__}: {e} -> single-hull fallback")
        h=ConvexHull(P); parts=[(P[np.unique(h.simplices)], None)]
    trees.append(dict(pts=P, parts=parts, c=P.mean(0)))
    print(f"  tree {i:2d}: {len(P):6d} pts -> {len(parts):2d} convex parts  ({time.time()-ts:.1f}s)")
print(f"{len(trees)} trees total, {sum(len(t['parts']) for t in trees)} convex parts, {time.time()-t0:.1f}s")

# --- 3. Export ---
# %%
def write_obj(path, parts):
    with open(path,"w") as f:
        voff=0
        for k,(V,F) in enumerate(parts):
            h=ConvexHull(V); Vh=V; Fh=h.simplices
            f.write(f"o part_{k}\n")
            for p in Vh: f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
            for tri in Fh: f.write(f"f {tri[0]+1+voff} {tri[1]+1+voff} {tri[2]+1+voff}\n")
            voff+=len(Vh)

rng=np.random.default_rng(0)
all_xyz=[]; all_rgb=[]; hull_store=[]
for i,t in enumerate(trees):
    write_obj(TCV/f"tree_{i}_parts.obj", t["parts"])
    tree_eqs=[]
    for (V,F) in t["parts"]:
        h=ConvexHull(V); eqs=h.equations                  # (m,4): n(3)+offset
        tree_eqs.append(eqs.astype(np.float32))
        cen=V[h.simplices].mean(1)
        pts=np.vstack([V[h.vertices], cen])
        col=rng.integers(40,235,3)
        all_xyz.append(pts); all_rgb.append(np.tile(col,(len(pts),1)))
    hull_store.append(tree_eqs)

if all_xyz:
    XYZ=np.vstack(all_xyz).astype(np.float32)
    RGB=np.vstack(all_rgb).astype(np.uint8)
    write_ply(TCV/"tree_convex_all.ply", XYZ, RGB)
    flat_eqs=[]; flat_tid=[]
    for ti,tree_eqs in enumerate(hull_store):
        for eqs in tree_eqs:
            flat_eqs.append(np.asarray(eqs,dtype=np.float32)); flat_tid.append(ti)
    obj=np.empty(len(flat_eqs),dtype=object)
    for j,e in enumerate(flat_eqs): obj[j]=e
    np.savez(TCV/"tree_convex_hulls.npz",
             hulls=obj, hull_tree=np.array(flat_tid,np.int32),
             centers=np.array([t["c"] for t in trees]))
    print("wrote:", TCV/"tree_convex_all.ply", f"({len(XYZ):,} pts)")
    print("wrote:", TCV/"tree_convex_hulls.npz", f"({len(trees)} trees)")
    print("wrote:", f"{len(trees)} tree_<id>_parts.obj  ->", TCV)

# --- 4. Analytic clearance (point -> distance to convex-part union) ---
# %%
def dist_to_hull(pts, eqs):
    d = pts @ eqs[:,:3].T + eqs[:,3]
    dmax = d.max(1)
    return np.clip(dmax, 0, None)

def clearance_to_trees(pts, hull_store):
    best=np.full(len(pts), np.inf)
    for tree_eqs in hull_store:
        for eqs in tree_eqs:
            best=np.minimum(best, dist_to_hull(pts, np.asarray(eqs,float)))
    return best

if trees:
    probe = trees[0]["pts"][:2000]
    d = clearance_to_trees(probe, [hull_store[0]])
    print(f"self-test tree0: clearance median {np.median(d):.3f} m (should be ~0), max {d.max():.3f}")

# W=np.stack([w['x'],w['y'],w['z']],1).astype(float)
# dw=clearance_to_trees(W, hull_store)

# --- 5. Visualization (matplotlib, one sample tree) ---
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SHOW=0
if trees:
    t=trees[SHOW]; P=t["pts"]
    fig=plt.figure(figsize=(8,8)); ax=fig.add_subplot(111,projection='3d')
    sub=P[np.random.default_rng(0).choice(len(P),min(4000,len(P)),replace=False)]
    ax.scatter(*sub.T, s=2, c='#2c8', alpha=0.35)
    rng=np.random.default_rng(1)
    for (V,F) in t["parts"]:
        h=ConvexHull(V)
        polys=[V[s] for s in h.simplices]
        pc=Poly3DCollection(polys, alpha=0.25, facecolor=rng.random(3), edgecolor='k', linewidths=0.2)
        ax.add_collection3d(pc)
    ax.set_title(f"tree {SHOW}: {len(t['parts'])} convex parts ({'CoACD' if USE_COACD else 'KMeans'})")
    try: ax.set_box_aspect(np.ptp(P,0))
    except Exception: pass
    plt.tight_layout(); plt.show()
