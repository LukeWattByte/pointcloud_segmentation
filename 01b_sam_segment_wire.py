"""01b - SAM segmentation of wires."""

# --- Keyframe 2D segmentation: wires (Grounded-SAM + thin-structure filter) ---

# --- 1. Dependencies ---
# %%
# %pip install -U "transformers>=4.51" timm accelerate scipy
print("same deps as 01; scipy used for thin-structure filtering")

# --- 2. Config ---
# %%
from pathlib import Path

FRAMES_DIR = Path("video_frames_2")
OUT_DIR    = Path("sam_output")

GDINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID   = "facebook/sam-vit-base"

CLASSES = {
    "wire": {
        "prompt": "power line. power cable. electric wire. overhead line. transmission line.",
        "box_threshold": 0.18,
        "text_threshold": 0.12,
        "color": (240, 30, 240),
        "thin_open": 15,
        "min_len":   15,
    },
}

MAX_FRAMES   = None
SKIP_EXISTING = True
SAM_BOX_CHUNK = 48
SAVE_OVERLAY_EVERY = 10
EMPTY_CACHE_EVERY  = 20

print("frames dir exists:", FRAMES_DIR.is_dir())
print("n frames:", len(list(FRAMES_DIR.glob("*.jpg"))))

# --- 3. Load models ---
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

# --- 4. Core functions (with thin-structure filter) ---
# %%
import numpy as np
from PIL import Image

try:
    from scipy.ndimage import binary_opening, label as cc_label, find_objects
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    print("warning: scipy not installed, thin-structure filtering skipped")

def _disk(r):
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y) <= r*r

def thin_filter(mask_bool, open_r, min_len):
    if not _HAS_SCIPY or not open_r:
        return mask_bool
    thick = binary_opening(mask_bool, structure=_disk(open_r))
    thin = mask_bool & ~thick
    if min_len:
        lab, n = cc_label(thin)
        keep_ids = [i for i, sl in enumerate(find_objects(lab), start=1)
                    if sl is not None and
                    ((sl[0].stop - sl[0].start) ** 2 +
                     (sl[1].stop - sl[1].start) ** 2) ** 0.5 >= min_len]
        thin = np.isin(lab, keep_ids)
    return thin

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
        for mm in masks:
            out.append(mm[0].numpy().astype(bool))
        del inputs, outputs, masks
    return out

def build_masks(image, cls_cfg):
    W, H = image.size
    boxes, scores = detect_boxes(image, cls_cfg["prompt"],
                                 cls_cfg["box_threshold"], cls_cfg["text_threshold"])
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    masks = segment_boxes(image, boxes)

    sem_bool = np.zeros((H, W), bool)
    inst = np.zeros((H, W), np.uint16)
    for idx, m in enumerate(masks, start=1):
        sem_bool |= m
        inst[m] = idx

    thin = thin_filter(sem_bool, cls_cfg.get("thin_open", 0), cls_cfg.get("min_len", 0))
    inst[~thin] = 0
    sem = (thin.astype(np.uint8)) * 255
    return sem, inst, boxes, scores

def overlay_image(image, per_class):
    base = np.asarray(image).astype(np.float32)
    for cls, (sem, color) in per_class.items():
        m = sem > 0
        base[m] = 0.35 * base[m] + 0.65 * np.array(color, np.float32)
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))

# --- 5. Single-frame sanity check (review before running the full batch) ---
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
    print(f"  {cls:5s}: {len(boxes)} boxes, mask px={int((sem>0).sum())}, "
          f"scores={np.round(scores,2)[:8]}")

vis = overlay_image(img, per_class)
plt.figure(figsize=(16, 9))
plt.imshow(vis); plt.axis("off"); plt.title(sample.name + "  (wire = magenta)")
plt.show() 

# --- Check: the magenta mask should hug the wires as thin lines ---

# --- 5b. (fallback, optional) classic line detection LSD ---
# %%
def lsd_wire_mask(image, min_len=200, max_vert_cos=0.9, thick=3):
    import cv2
    g = np.asarray(image.convert("L"))
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(g)[0]
    mask = np.zeros(g.shape, np.uint8)
    if lines is None:
        return mask
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        L = (dx*dx + dy*dy) ** 0.5
        if L < min_len:
            continue
        if abs(dy) / (L + 1e-6) > max_vert_cos:
            continue
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thick)
    return mask

m = lsd_wire_mask(img); plt.figure(figsize=(16,9)); plt.imshow(m, cmap='gray'); plt.show()
print("LSD fallback defined; uncomment above to try it when needed")

# --- 6. Batch-process all frames ---
# %%
import json, time

for cls in CLASSES:
    (OUT_DIR / "masks" / cls).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "inst" / cls).mkdir(parents=True, exist_ok=True)
(OUT_DIR / "meta_wire").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "overlay_wire").mkdir(parents=True, exist_ok=True)

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
    meta_path = OUT_DIR / "meta_wire" / f"{stem}.json"
    if SKIP_EXISTING and meta_path.exists():
        continue
    try:
        img = Image.open(path).convert("RGB")
        W, H = img.size
        rec = {"frame": stem, "width": W, "height": H, "classes": {}}
        per_class = {}
        for cls, cfg in CLASSES.items():
            sem, inst, boxes, scores = build_masks(img, cfg)
            # sem, inst, boxes, scores = lsd_wire_mask(img, cfg)
            Image.fromarray(sem).save(OUT_DIR / "masks" / cls / f"{stem}.png")
            Image.fromarray(inst).save(OUT_DIR / "inst" / cls / f"{stem}.png")
            per_class[cls] = (sem, cfg["color"])
            rec["classes"][cls] = {
                "n_instances": int(len(boxes)),
                "mask_px": int((sem > 0).sum()),
                "boxes": np.round(boxes, 1).tolist(),
                "scores": [round(float(s), 4) for s in scores],
            }
        meta_path.write_text(json.dumps(rec))
        if SAVE_OVERLAY_EVERY and (i % SAVE_OVERLAY_EVERY == 0):
            overlay_image(img, per_class).save(
                OUT_DIR / "overlay_wire" / f"{stem}.jpg", quality=85)
        n_done += 1
    except Exception as e:
        n_fail += 1
        print(f"[FAIL] {stem}: {e}")
    if DEVICE == "cuda" and EMPTY_CACHE_EVERY and (i % EMPTY_CACHE_EVERY == 0):
        torch.cuda.empty_cache()

dt = time.time() - t0
print(f"\ndone: {n_done} frames, {n_fail} failed, {dt/60:.1f} min "
      f"({dt/max(n_done,1):.1f} s/frame)")

# --- 7. Summary stats ---
# %%
import json
metas = sorted((OUT_DIR / "meta_wire").glob("*.json"))
tot = 0
frames_with = 0
px_tot = 0
for mp in metas:
    rec = json.loads(mp.read_text())
    d = rec["classes"]["wire"]
    tot += d["n_instances"]
    px_tot += d.get("mask_px", 0)
    frames_with += int(d.get("mask_px", 0) > 0)

print(f"frames processed: {len(metas)}")
print(f"  wire: {tot} detection boxes total, frames with wire pixels {frames_with}/{len(metas)}, "
      f"avg {px_tot/max(len(metas),1):.0f} px/frame")
print("\noutput dir:", (OUT_DIR / 'masks' / 'wire').resolve())
print("next -> 04_wire_catenary: uplift wire masks to 3D and fit catenaries")
