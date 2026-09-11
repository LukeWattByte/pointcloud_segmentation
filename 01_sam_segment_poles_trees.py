"""01 - SAM segmentation of poles and trees."""

# --- Keyframe 2D segmentation: poles + trees (Grounded-SAM) ---

# --- SAM Segmentation ---

# --- Configuration ---
# %%
from pathlib import Path

# paths
FRAMES_DIR = Path("video_frames_2")
OUT_DIR    = Path("sam_output")

# models
GDINO_ID = "IDEA-Research/grounding-dino-base"   # faster: grounding-dino-tiny
SAM_ID   = "facebook/sam-vit-base"               # slower: facebook/sam-vit-huge

# text prompts & thresholds for each class
CLASSES = {
    "tree": {
        "prompt": "tree.",
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "color": (60, 200, 60),
    },
    "pole": {
        "prompt": "utility pole. power pole. electricity pole. telephone pole.",
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "color": (250, 120, 20),
    },
}

# batch & performance settings
MAX_FRAMES   = None
SKIP_EXISTING = True
SAM_BOX_CHUNK = 48
SAVE_OVERLAY_EVERY = 10
EMPTY_CACHE_EVERY  = 20

print("frames dir exists:", FRAMES_DIR.is_dir())
print("n frames:", len(list(FRAMES_DIR.glob("*.jpg"))))

# --- load model ---
# %%
import torch
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    SamProcessor, SamModel,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

print("loading GroundingDINO ...")
gdino_proc = AutoProcessor.from_pretrained(GDINO_ID)
gdino = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(DEVICE).eval()

print("loading SAM ...")
sam_proc = SamProcessor.from_pretrained(SAM_ID)
sam = SamModel.from_pretrained(SAM_ID).to(DEVICE).eval()
print("done.")

# --- Core Function ---
# %%
import numpy as np
from PIL import Image

def _post_process_gdino(outputs, input_ids, box_thr, text_thr, target_size):
    try:
        return gdino_proc.post_process_grounded_object_detection(
            outputs, input_ids, threshold=box_thr,
            text_threshold=text_thr, target_sizes=[target_size])
    except TypeError:
        return gdino_proc.post_process_grounded_object_detection(
            outputs, input_ids, box_threshold=box_thr,
            text_threshold=text_thr, target_sizes=[target_size])

def _post_process_masks(pred_masks, original_sizes, reshaped_sizes):
    fn = getattr(sam_proc, "post_process_masks", None) or sam_proc.image_processor.post_process_masks
    return fn(pred_masks, original_sizes, reshaped_sizes)

@torch.no_grad()
def detect_boxes(image, prompt, box_thr, text_thr):
    inputs = gdino_proc(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    outputs = gdino(**inputs)
    res = _post_process_gdino(outputs, inputs.input_ids, box_thr, text_thr, image.size[::-1])[0]
    boxes = res["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = res["scores"].detach().cpu().numpy().astype(np.float32)
    return boxes, scores

@torch.no_grad()
def segment_boxes(image, boxes):
    out = []
    if len(boxes) == 0:
        return out
    for i in range(0, len(boxes), SAM_BOX_CHUNK):
        chunk = boxes[i:i + SAM_BOX_CHUNK].tolist()
        inputs = sam_proc(image, input_boxes=[chunk], return_tensors="pt").to(DEVICE)
        outputs = sam(**inputs, multimask_output=False)
        masks = _post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        for m in masks:
            out.append(m[0].numpy().astype(bool))
        del inputs, outputs, masks
    return out

def build_masks(image, cls_cfg):
    # sem uint8 0/255, inst uint16, boxes, scores
    W, H = image.size
    boxes, scores = detect_boxes(image, cls_cfg["prompt"],
                                 cls_cfg["box_threshold"], cls_cfg["text_threshold"])
    # confidence desc order
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    masks = segment_boxes(image, boxes)

    sem = np.zeros((H, W), np.uint8)
    inst = np.zeros((H, W), np.uint16)
    for idx, m in enumerate(masks, start=1):
        sem[m] = 255
        inst[m] = idx
    return sem, inst, boxes, scores

def overlay_image(image, per_class):
    # per_class: dict cls -> (sem, color)
    base = np.asarray(image).astype(np.float32)
    for cls, (sem, color) in per_class.items():
        m = sem > 0
        base[m] = 0.5 * base[m] + 0.5 * np.array(color, np.float32)
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))

# --- Single frame sanity check ---
# %%
import matplotlib.pyplot as plt

frames = sorted(FRAMES_DIR.glob("*.jpg"))
sample = frames[len(frames) // 2]
print("sample:", sample.name)

img = Image.open(sample).convert("RGB")
per_class = {}
for cls, cfg in CLASSES.items():
    sem, inst, boxes, scores = build_masks(img, cfg)
    per_class[cls] = (sem, cfg["color"])
    print(f"  {cls:5s}: {len(boxes)} instances, scores={np.round(scores,2)[:8]}")

vis = overlay_image(img, per_class)
plt.figure(figsize=(16, 9))
plt.imshow(vis); plt.axis("off"); plt.title(sample.name)
plt.show()

# --- All frames processing ---
# %%
import json, time

for cls in CLASSES:
    (OUT_DIR / "masks" / cls).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "inst" / cls).mkdir(parents=True, exist_ok=True)
(OUT_DIR / "meta").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "overlay").mkdir(parents=True, exist_ok=True)

frames = sorted(FRAMES_DIR.glob("*.jpg"))
if MAX_FRAMES:
    frames = frames[:MAX_FRAMES]

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

t0 = time.time()
n_done, n_fail = 0, 0
for i, path in enumerate(tqdm(frames, desc="frames")):
    stem = path.stem
    meta_path = OUT_DIR / "meta" / f"{stem}.json"
    if SKIP_EXISTING and meta_path.exists():
        continue
    try:
        img = Image.open(path).convert("RGB")
        W, H = img.size
        rec = {"frame": stem, "width": W, "height": H, "classes": {}}
        per_class = {}
        for cls, cfg in CLASSES.items():
            sem, inst, boxes, scores = build_masks(img, cfg)
            Image.fromarray(sem).save(OUT_DIR / "masks" / cls / f"{stem}.png")
            Image.fromarray(inst).save(OUT_DIR / "inst" / cls / f"{stem}.png")
            per_class[cls] = (sem, cfg["color"])
            rec["classes"][cls] = {
                "n_instances": int(len(boxes)),
                "boxes": np.round(boxes, 1).tolist(),
                "scores": [round(float(s), 4) for s in scores],
            }
        meta_path.write_text(json.dumps(rec))
        if SAVE_OVERLAY_EVERY and (i % SAVE_OVERLAY_EVERY == 0):
            overlay_image(img, per_class).save(
                OUT_DIR / "overlay" / f"{stem}.jpg", quality=85)
        n_done += 1
    except Exception as e:
        n_fail += 1
        print(f"[FAIL] {stem}: {e}")
    if DEVICE == "cuda" and EMPTY_CACHE_EVERY and (i % EMPTY_CACHE_EVERY == 0):
        torch.cuda.empty_cache()

dt = time.time() - t0
print(f"\nCompletion: {n_done} frames, failed {n_fail} frames, time {dt/60:.1f} min "
      f"({dt/max(n_done,1):.1f} s/frame)")

# --- Summary ---
# %%
import json
metas = sorted((OUT_DIR / "meta").glob("*.json"))
tot = {cls: 0 for cls in CLASSES}
frames_with = {cls: 0 for cls in CLASSES}
for mp in metas:
    rec = json.loads(mp.read_text())
    for cls, d in rec["classes"].items():
        tot[cls] += d["n_instances"]
        frames_with[cls] += int(d["n_instances"] > 0)

print(f"Processed frames: {len(metas)}")
for cls in CLASSES:
    print(f"  {cls:5s}: Total {tot[cls]} instances, appearing in {frames_with[cls]}/{len(metas)} frames")
print("\nOutput directory:", OUT_DIR.resolve())

# --- Uplift Masks to Pointcloud ---

# --- Configuration ---
# %%
from pathlib import Path
import numpy as np

BASE   = Path(".").resolve()
DM     = BASE / "colmap_workspace_video_2" / "dense_merged"
PLY_GEOM   = DM / "fused.ply"           # projection (same coord sys as poses)
PLY_METRIC = DM / "fused_metric.ply"    # output (metric frames, same point order)
SPARSE = DM / "sparse"
MASK_TREE = BASE / "sam_output" / "masks" / "tree"
MASK_POLE = BASE / "sam_output" / "masks" / "pole"
OUT = BASE / "uplift_output"; OUT.mkdir(exist_ok=True)

# original camera (SIMPLE_RADIAL, 4K)
F, CX, CY, K1 = 1708.51, 1920.0, 1080.0, -0.00315
W4K, H4K = 3840, 2160

# z-buffer occlusion: build on downsampled grid (fill holes + speed up), visibility criterion z <= nearest surface*(1+TOL_REL)+TOL_ABS
ZS = 0.25                # grid scaling -> 960x540
TOL_REL, TOL_ABS = 0.03, 0.0

# voting -> labels
MIN_SEEN  = 3           # points must be seen in at least this many frames to participate in decision making
MIN_VOTES = 2           # at least this many votes for a class
TREE_FRAC = 0.35        # tree votes / visible count threshold
POLE_FRAC = 0.25        # pole votes / visible count threshold (poles are rarer, so lower threshold)

COLOR = {0: (110, 110, 110), 1: (60, 200, 60), 2: (250, 120, 20)}  # bg / tree / pole
print("ply:", PLY_GEOM.name, "exists", PLY_GEOM.exists(), "| metric", PLY_METRIC.exists())

# --- Read Pointcloud and Position ---
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
Xw = np.stack([g['x'], g['y'], g['z']], 1).astype(np.float64)   # projection coord sys
m = read_ply(PLY_METRIC)
Xm = np.stack([m['x'], m['y'], m['z']], 1).astype(np.float32)   # output coord sys (metric frames, same point order)    
Cm = np.stack([m['red'], m['green'], m['blue']], 1)            # original colors
N = len(Xw); print("N points:", N)
assert (g['red'] == m['red']).all(), "point order mismatch!"

print("loading poses ...")
rec = pycolmap.Reconstruction(str(SPARSE))
POSES = {}
for im in rec.images.values():
    cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    POSES[Path(im.name).stem] = cfw.matrix().astype(np.float64)   # 3x4 world->cam
print("poses:", len(POSES))

# --- Projection and Vote function ---
# %%
from PIL import Image

GW, GH = int(round(W4K*ZS)), int(round(H4K*ZS))

def project(Rt):
    # return 4K pixel u,v, camera depth z, whether in front
    Xc = Xw @ Rt[:, :3].T + Rt[:, 3]
    z = Xc[:, 2]
    front = z > 1e-6
    zz = np.where(front, z, 1.0)
    a = Xc[:, 0] / zz; b = Xc[:, 1] / zz
    r2 = a*a + b*b
    d = 1.0 + K1*r2                  # SIMPLE_RADIAL
    u = F*a*d + CX
    v = F*b*d + CY
    return u, v, z, front

def visible_mask(u, v, z, front):
    # in the image + pass z-buffer
    inb = front & (u >= 0) & (u < W4K) & (v >= 0) & (v < H4K)
    idx = np.where(inb)[0]
    ui = u[idx].astype(np.int32); vi = v[idx].astype(np.int32); zc = z[idx]
    gflat = (vi*ZS).astype(np.int32)*GW + (ui*ZS).astype(np.int32)
    zbuf = np.full(GW*GH, np.inf)
    order = np.argsort(zc)[::-1]                 # far to near for z-buffer (keep nearest)
    zbuf[gflat[order]] = zc[order]
    vis = zc <= zbuf[gflat]*(1.0+TOL_REL) + TOL_ABS
    return idx[vis], ui[vis], vi[vis]

def load_mask(folder, stem):
    p = folder / f"{stem}.png"
    if not p.exists(): return None
    return np.asarray(Image.open(p)) > 0

# --- Single frame sanity check ---
# %%
import matplotlib.pyplot as plt

stem = sorted(POSES)[len(POSES)//2]
print("sample:", stem)
u, v, z, front = project(POSES[stem])
vidx, vu, vv = visible_mask(u, v, z, front)
tm = load_mask(MASK_TREE, stem); pm = load_mask(MASK_POLE, stem)
is_t = tm[vv, vu]; is_p = pm[vv, vu]

img = np.asarray(Image.open(DM / "images" / f"{stem}.jpg").convert("RGB"))
# dense results are 2000x1121, visualize by scaling 4K projected coordinates to that size
sx, sy = img.shape[1]/W4K, img.shape[0]/H4K
plt.figure(figsize=(15, 8)); plt.imshow(img)
plt.scatter(vu[is_t]*sx, vv[is_t]*sy, s=1, c='lime', alpha=0.3, label='tree pts')
plt.scatter(vu[is_p]*sx, vv[is_p]*sy, s=2, c='red', alpha=0.5, label='pole pts')
plt.legend(); plt.axis('off'); plt.title(f"{stem}  (tree {is_t.sum()}, pole {is_p.sum()})")
plt.show()

# --- All votes ---
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

# --- Labeling and Summary ---
