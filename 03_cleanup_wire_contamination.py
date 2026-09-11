"""03 - Clean wire contamination from the point cloud."""

# --- Clean wire contamination (tree/pole -> wire) ---

# --- 1. Config ---
# %%
from pathlib import Path
import numpy as np

BASE = Path(".").resolve()
DM   = BASE / "colmap_workspace_video_2" / "dense_merged"
OUT  = BASE / "uplift_output"

UP = np.array([0.066, 0.058, 0.996]); UP /= np.linalg.norm(UP)
K  = 100
LIN_MIN  = 0.90
VERT_MAX = 0.45
THICK_MAX = 0.15
COLOR = {0:(110,110,110), 1:(60,200,60), 2:(250,120,20), 3:(240,30,240)}  # bg/tree/pole/wire

# --- 2. Read/write helpers + load ---
# %%
PLY_DT=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('nx','<f4'),('ny','<f4'),('nz','<f4'),
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
    n=len(xyz); h=(f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()
    dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    a=np.empty(n,dt); a['x'],a['y'],a['z']=xyz[:,0],xyz[:,1],xyz[:,2]
    a['red'],a['green'],a['blue']=rgb[:,0],rgb[:,1],rgb[:,2]
    with open(p,'wb') as f: f.write(h); a.tofile(f)

m=read_ply(DM/"fused_metric.ply")
X=np.stack([m['x'],m['y'],m['z']],1).astype(np.float64)
C=np.stack([m['red'],m['green'],m['blue']],1)
labels=np.load(OUT/"labels.npy").astype(np.int8)
print("points", len(X))

# --- 3. Per-point local PCA features + wire tagging ---
# %%
from sklearn.neighbors import NearestNeighbors

def local_pca(P, k=K, chunk=200000):
    nn=NearestNeighbors(n_neighbors=k,n_jobs=-1).fit(P)
    lin=np.empty(len(P)); vert=np.empty(len(P)); thick=np.empty(len(P))
    for i in range(0,len(P),chunk):
        sl=slice(i,min(i+chunk,len(P)))
        _,idx=nn.kneighbors(P[sl]); nb=P[idx]
        c=nb-nb.mean(1,keepdims=True); cov=np.einsum('bki,bkj->bij',c,c)/k
        w,V=np.linalg.eigh(cov); l0,l1=w[:,2],w[:,1]
        lin[sl]=(l0-l1)/(l0+1e-12); vert[sl]=np.abs(V[:,:,2]@UP)
        thick[sl]=np.sqrt(np.clip(l1,0,None))
    return lin,vert,thick

new=labels.copy()
for name,lid in [("pole",2),("tree",1)]:
    sel=np.where(labels==lid)[0]
    lin,vert,thick=local_pca(X[sel])
    wire=(lin>=LIN_MIN)&(vert<=VERT_MAX)&(thick<=THICK_MAX)
    new[sel[wire]]=3
    print(f"[{name}] {len(sel):>8,d} -> wire {int(wire.sum()):>6,d} ({100*wire.mean():.2f}%)")
for k,nm in [(0,'bg'),(1,'tree'),(2,'pole'),(3,'wire')]:
    print(f"  {nm:4s}: {(new==k).sum():>10,d}")

# --- 4. Write cleaned result ---
# %%
rgb=C.copy()
for k in (0,1,2,3):
    s=new==k
    rgb[s]=(0.4*C[s]+0.6*np.array(COLOR[0])).astype(np.uint8) if k==0 else COLOR[k]
write_ply(OUT/"fused_metric_labeled_clean.ply", X.astype(np.float32), rgb)
for k,nm in [(1,'tree'),(2,'pole'),(3,'wire')]:
    s=new==k; write_ply(OUT/f"{nm}_metric_clean.ply", X[s].astype(np.float32), C[s])
np.save(OUT/"labels_clean.npy", new)
print("wrote *_clean.ply + labels_clean.npy ->", OUT)
print("open fused_metric_labeled_clean.ply in CloudCompare: tree=green/pole=orange/wire=magenta/bg=gray")

# --- 5. (optional) back-projection check ---
# %%
import pycolmap
from PIL import Image
F,CX,CY,K1=1708.51,1920.0,1080.0,-0.00315; W4K,H4K=3840,2160
ZS=0.25; GW,GH=int(W4K*ZS),int(H4K*ZS); TOL_REL=0.03
COL={1:(60,220,60),2:(255,120,20),3:(255,0,255)}

g=read_ply(DM/"fused.ply"); Xw=np.stack([g['x'],g['y'],g['z']],1).astype(np.float64)
rec=pycolmap.Reconstruction(str(DM/"sparse")); POSES={}
for im in rec.images.values():
    cfw=im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    POSES[Path(im.name).stem]=cfw.matrix().astype(np.float64)

def proj(Rt):
    Xc=Xw@Rt[:,:3].T+Rt[:,3]; z=Xc[:,2]; fr=z>1e-6; zz=np.where(fr,z,1.0)
    a=Xc[:,0]/zz; b=Xc[:,1]/zz; d=1.0+K1*(a*a+b*b)
    return F*a*d+CX, F*b*d+CY, z, fr
def vis(u,v,z,fr):
    inb=fr&(u>=0)&(u<W4K)&(v>=0)&(v<H4K); idx=np.where(inb)[0]
    ui=u[idx].astype(np.int32); vi=v[idx].astype(np.int32); zc=z[idx]
    gf=(vi*ZS).astype(np.int32)*GW+(ui*ZS).astype(np.int32)
    zb=np.full(GW*GH,np.inf); o=np.argsort(zc)[::-1]; zb[gf[o]]=zc[o]
    mk=zc<=zb[gf]*(1+TOL_REL); return idx[mk],ui[mk],vi[mk]

import matplotlib.pyplot as plt
stem="v1_000000" if "v1_000000" in POSES else sorted(POSES)[0]
u,v,z,fr=proj(POSES[stem]); idx,ui,vi=vis(u,v,z,fr); L=new[idx]
img=np.asarray(Image.open(DM/"images"/f"{stem}.jpg").convert("RGB")).copy()
sx,sy=img.shape[1]/W4K, img.shape[0]/H4K
plt.figure(figsize=(15,8)); plt.imshow(img)
for k,c in [(1,'lime'),(2,'orange'),(3,'magenta')]:
    mk=L==k; plt.scatter(ui[mk]*sx,vi[mk]*sy,s=(1 if k==1 else 4),c=c,alpha=.5,
                         label=['','tree','pole','wire'][k])
plt.legend(); plt.axis('off'); plt.title(stem); plt.show() 
