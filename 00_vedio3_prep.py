"""00 - video-3 preprocessing: extract keyframes, telemetry, masks for COLMAP."""

# --- 00 - vedio-3 preprocessing & camera capture test ---

# --- 0. Config ---
# %%
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE  = Path(".").resolve()
PREP  = BASE / "vedio3_prep"
VIDEO = BASE / "vedio-3" / "DJI_20260723124901_0001_D.MP4"
OUT   = BASE / "vedio3_frames"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(PREP))
import dji_telemetry as tel
import check_horizon_lock as hz
import extract_keyframes as ek

FLOW_THRESH  = 100.0
PROFILE_STEP = 5

# COLMAP
COLMAP = Path(r"C:\Users\yhw15\Tools\COLMAP\bin\colmap.exe")
VOCAB  = COLMAP.parent.parent / "vocab_tree_flickr100K_words256K.bin"
WS     = BASE / "colmap_workspace_video_3"

CAM_PARAMS = "1031.0,1034.6,960.0,540.0,-0.0052,0.0024,0.00011,-0.00126"

def largest_model(sparse_root):
    cand = [d for d in sorted(Path(sparse_root).glob("*")) if (d / "images.bin").exists()
            or (d / "images.txt").exists()]
    if not cand:
        return None
    import pycolmap
    best = max(cand, key=lambda d: len(pycolmap.Reconstruction(str(d)).images))
    for d in cand:
        n = len(pycolmap.Reconstruction(str(d)).images)
        print(f"  {d.name}: {n} images{'   <- pick' if d == best else ''}")
    return best

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("video :", VIDEO.name, "exists", VIDEO.exists())
print("prep  :", PREP.name, "exists", PREP.exists())
print("colmap:", COLMAP, "exists", COLMAP.exists())
print("vocab :", VOCAB.name, "exists", VOCAB.exists())

# --- 1. Video spec probe ---
# %%
cap = cv2.VideoCapture(str(VIDEO))
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)
N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin1")
cap.release()

print(f"{W}x{H} @ {FPS:.3f} fps   {N} frames = {N/FPS:.1f} s   codec {fourcc}")
print(f"file {VIDEO.stat().st_size/1e9:.2f} GB  ->  {VIDEO.stat().st_size*8/(N/FPS)/1e6:.1f} Mbps")
assert (W, H) == (1920, 1080), "resolution differs from record; re-measure constants below"

# --- 2. Per-frame telemetry (djmd track) ---
# %%
t = tel.read_telemetry(VIDEO)
yaw, pitch, roll_body = tel.attitude(t["quat"])
resid = tel.gravity_residual(t["quat"], t["accel"])
t_s = (t["t_us"] - t["t_us"][0]) / 1e6

print(f"samples {len(yaw)} (video {N} frames)  duration {t_s[-1]:.1f}s  "
      f"dt median {np.median(np.diff(t['t_us'])):.0f}us")
print(f"quat norm {np.linalg.norm(t['quat'], axis=1).mean():.6f}   "
      f"gravity residual mean {resid.mean():.2f}deg  p95 {np.percentile(resid,95):.2f}deg")
print(f"pitch {pitch.mean():+.1f} ± {pitch.std():.1f} deg   "
      f"body roll {roll_body.mean():+.1f} ± {roll_body.std():.1f} deg   "
      f"yaw span {yaw.max()-yaw.min():.0f} deg")

np.savez_compressed(PREP / "telemetry.npz", frame=np.arange(len(yaw)), t_s=t_s,
                    quat=t["quat"], accel=t["accel"], yaw=yaw, pitch=pitch,
                    roll_body=roll_body)
print("wrote", PREP / "telemetry.npz")

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
ax[0].plot(t_s, pitch, lw=.6, label="pitch")
ax[0].plot(t_s, roll_body, lw=.6, label="roll (body)")
ax[0].set_xlabel("s"); ax[0].set_ylabel("deg"); ax[0].legend(); ax[0].set_title("attitude")
ax[1].plot(t_s, yaw, lw=.8)
ax[1].set_xlabel("s"); ax[1].set_ylabel("deg"); ax[1].set_title("yaw (unwrap) - steps = 90 deg block turns")
ax[2].hist(resid, bins=60)
ax[2].set_xlabel("deg"); ax[2].set_title("gravity residual (quat vs accel)")
plt.tight_layout(); plt.show()

# --- 3. Horizon-lock detection - run this first on every new clip ---
# %%
roll_img_ref = hz.image_roll_reference(t["quat"])
cap = cv2.VideoCapture(str(VIDEO))
ra = hz.test_frame_rotation(cap, roll_img_ref)      # (d_imu_roll, image_rot, inlier)
rb = hz.test_vertical_lines(cap, roll_img_ref)      # (imu_roll, line_tilt, n_lines)
cap.release()

slope_a, _, corr_a = hz._fit(ra[:, 0], ra[:, 1])
slope_b, _, corr_b = hz._fit(rb[:, 0], rb[:, 1])
print(f"[A] frame-to-frame rotation  n={len(ra):3d}  slope {slope_a:+.3f}  corr {corr_a:+.3f}")
print(f"[B] vertical-line tilt       n={len(rb):3d}  slope {slope_b:+.3f}  corr {corr_b:+.3f}")

mean_slope = (slope_a + slope_b) / 2
HORIZON_LOCKED = mean_slope < 0.3
print(f"\n=== mean slope {mean_slope:+.3f} -> "
      f"horizon lock {'ON (frame locked level)' if HORIZON_LOCKED else 'OFF'} ===")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a_, (d, lab) in zip(ax, [(ra, "A frame rotation"), (rb, "B vertical-line tilt")]):
    a_.scatter(d[:, 0], d[:, 1], s=14, alpha=.7)
    lim = np.abs(d[:, 0]).max() * 1.1
    x = np.linspace(-lim, lim, 10)
    a_.plot(x, x, "r--", lw=1, label="slope +1 (no lock)")
    a_.axhline(0, color="g", ls="--", lw=1, label="slope 0 (lock on)")
    a_.set_xlabel("IMU body roll (deg)"); a_.set_ylabel("image roll (deg)")
    a_.set_title(lab); a_.legend(fontsize=7)
plt.tight_layout(); plt.show()

# --- Three consequences of the lock being on (each used below) ---

# --- 4. Extract keyframes + hood mask ---
# %%
cache = OUT / "motion_profile.npz"
if cache.exists() and int(np.load(cache)["step"]) == PROFILE_STEP:
    z = np.load(cache); prof, flowmap = z["profile"], z["flowmap"]
    print(f"reuse cache {cache.name} ({len(prof)} samples)")
else:
    print("pass 1a: scan motion profile (full decode, ~40 s)...")
    prof, _ = ek.build_profile(VIDEO, PROFILE_STEP)
    print("pass 1b: dense optical flow to find hood ...")
    flowmap, n_dense = ek.build_hood_map(VIDEO, prof)
    print(f"  used {n_dense} high-speed frame pairs")
    np.savez_compressed(cache, profile=prof, flowmap=flowmap, step=PROFILE_STEP)

mask = ek.hood_mask(flowmap, (H, W))
HOOD_TOP = int(np.argmax((mask == 0).any(1)))

# %%
cap = cv2.VideoCapture(str(VIDEO)); cap.set(cv2.CAP_PROP_POS_FRAMES, 5259)
ok, frame = cap.read(); cap.release()
overlay = frame.copy(); overlay[mask == 0] = (0, 0, 255)
overlay = cv2.addWeighted(frame, .55, overlay, .45, 0)

fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
im = ax[0].imshow(flowmap, cmap="inferno")
ax[0].set_title("dense flow magnitude median (dark = static = hood)"); plt.colorbar(im, ax=ax[0], shrink=.8)
ax[0].axhline(HOOD_TOP * flowmap.shape[0] / H, color="cyan", lw=1)
ax[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
ax[1].set_title(f"mask preview: below row {HOOD_TOP}, {(mask==0).mean()*100:.1f}% of frame")
for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
plt.tight_layout(); plt.show()

# --- Check the right image: the red band must cover the hood; if not, set HOOD_TOP manually ---
# %%
sel, flow = ek.select_frames(prof, FLOW_THRESH, W / ek.PROF_W)
gaps = np.diff(sel)
print(f"motion profile: median {np.median(flow):.1f}px / {PROFILE_STEP} frames")
print(f"selected keyframes {len(sel)} (threshold {FLOW_THRESH}px)")
print(f"  frame gap median {np.median(gaps):.0f} frames ≈ {np.median(gaps)/FPS:.2f}s, "
      f"max {gaps.max()} frames ≈ {gaps.max()/FPS:.1f}s")
print(f"  -> largest gap is the 35 s stop: cumulative-flow sampling auto-skips static segments")

# %%
tp = prof[:, 0] / FPS
d  = prof[:, 1] * W / ek.PROF_W
yw = yaw[np.clip(prof[:, 0].astype(int), 0, len(yaw) - 1)]
dt = np.diff(tp, prepend=tp[0] - PROFILE_STEP / FPS)
x = np.cumsum(d * np.cos(np.radians(yw)) * dt)
y = np.cumsum(d * np.sin(np.radians(yw)) * dt)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].plot(tp, d, lw=.6)
ax[0].axhline(4 * PROFILE_STEP / 15, color="r", ls="--", lw=.8, label="static threshold")
ax[0].scatter(sel / FPS, np.zeros(len(sel)) - 5, s=1, c="g", label="selected")
ax[0].set_xlabel("s"); ax[0].set_ylabel(f"px / {PROFILE_STEP} frames")
ax[0].set_title("apparent motion profile"); ax[0].legend(fontsize=8)
ax[1].plot(x, y, lw=1); ax[1].plot(x[0], y[0], "go", label="start"); ax[1].plot(x[-1], y[-1], "ro", label="end")
ax[1].set_aspect("equal"); ax[1].legend(fontsize=8)
ax[1].set_title("dead-reckoned loop shape (yaw x flow proxy) - self-crossing = T2 chance")
plt.tight_layout(); plt.show()

# --- Write frames to disk (~860 jpg + masks, takes tens of seconds) ---
# %%
WRITE_FRAMES = True

if WRITE_FRAMES:
    (OUT / "images").mkdir(exist_ok=True); (OUT / "masks").mkdir(exist_ok=True)
    want = set(int(v) for v in sel)
    cap = cv2.VideoCapture(str(VIDEO))
    rows, i, last = [], 0, int(sel.max())
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            if ok:
                name = f"v3_{i:06d}.jpg"
                cv2.imwrite(str(OUT / "images" / name), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(OUT / "masks" / f"{name}.png"), mask)
                rows.append([name, i, i / FPS, yaw[i], pitch[i], roll_body[i]])
        i += 1
    cap.release()

    with open(OUT / "manifest.csv", "w", encoding="utf-8") as f:
        f.write("image,frame,t_s,yaw_deg,pitch_deg,roll_body_deg\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.3f}\n")
    print(f"wrote {len(rows)} images -> {OUT/'images'}")
    print(f"per-image masks (COLMAP convention <image_name>.png, 0=ignore) -> {OUT/'masks'}")
    print(f"manifest -> {OUT/'manifest.csv'}")
else:
    print("skipped (WRITE_FRAMES=False)")

cv2.imwrite(str(OUT / "hood_mask.png"), mask)
print("single-file mask ->", OUT / "hood_mask.png")

# --- 5. COLMAP ---
# %%
assert COLMAP.exists(), f"colmap.exe not found: {COLMAP}"
assert (OUT / "hood_mask.png").exists(), "run §4 first to generate hood_mask.png"

ps = rf'''# vedio-3 -> COLMAP    (run in demo/:  .\run_colmap_video3.ps1 )
# Generated by 00_vedio3_prep §5 - do not edit by hand; edit the source and re-run this cell
$ErrorActionPreference = "Stop"
$colmap = "{COLMAP}"
$ws     = "{WS.name}"
$images = "{OUT.name}/images"
$mask   = "{OUT.name}/hood_mask.png"

New-Item -ItemType Directory -Force $ws | Out-Null

# ---------- 1. Feature extraction ----------
# camera_params calibrated from a 90-image trial run, avoids converging from a poor 2304px prior
& $colmap feature_extractor `
  --database_path $ws/database.db `
  --image_path $images `
  --ImageReader.camera_mask_path $mask `
  --ImageReader.camera_model OPENCV `
  --ImageReader.single_camera 1 `
  --ImageReader.camera_params "{CAM_PARAMS}" `
  --FeatureExtraction.max_image_size 1920
if ($LASTEXITCODE -ne 0) {{ throw "feature_extractor failed" }}

# ---------- 2. Matching (exhaustive) ----------
# 862 images = 370k pairs, ~20-30 min on GPU; more thorough loop closure than vocab-tree retrieval
# (exhaustive finds every revisit, retrieval only top-k).
& $colmap exhaustive_matcher --database_path $ws/database.db
if ($LASTEXITCODE -ne 0) {{ throw "exhaustive_matcher failed" }}

# --- Alternative: sequential + vocab-tree loop detection (faster, but not usable on this machine) ---
# The existing vocab_tree_flickr100K_words256K.bin is the old FLANN format;
# COLMAP since 2025-05 uses FAISS and loading it errors with
#   "Failed to read faiss index ... legacy flann-based index"
# To use it, download a FAISS-format tree from the COLMAP site. Revisit when exhaustive gets too slow.
# & $colmap sequential_matcher `
#   --database_path $ws/database.db `
#   --SequentialMatching.overlap 10 `
#   --SequentialMatching.quadratic_overlap 1 `
#   --SequentialMatching.loop_detection 1 `
#   --SequentialMatching.vocab_tree_path "{VOCAB}"

# ---------- 3. Sparse reconstruction ----------
# Note: mapper may emit several sub-models sparse/0, sparse/1 ...; use largest_model() to pick
New-Item -ItemType Directory -Force $ws/sparse | Out-Null
& $colmap mapper `
  --database_path $ws/database.db `
  --image_path $images `
  --output_path $ws/sparse
if ($LASTEXITCODE -ne 0) {{ throw "mapper failed" }}

Write-Host ""
Write-Host "Sparse reconstruction done. Sub-models:"
Get-ChildItem $ws/sparse -Directory | ForEach-Object {{ Write-Host "  $($_.Name)" }}
Write-Host "-> back to 00_vedio3_prep §6 (auto-picks the largest sub-model)"
Write-Host "   dense reconstruction: run_colmap_video3_dense.ps1 (verify sparse first; slow)"
'''
(BASE / "run_colmap_video3.ps1").write_text(ps, encoding="utf-8-sig")
print("wrote run_colmap_video3.ps1")

ps_dense = rf'''# vedio-3 -> COLMAP dense reconstruction (run run_colmap_video3.ps1 first and verify sparse)
$ErrorActionPreference = "Stop"
$colmap = "{COLMAP}"
$ws     = "{WS.name}"
$images = "{OUT.name}/images"
$model  = "$ws/sparse/0"      # <- set to the sub-model picked in §6

& $colmap image_undistorter `
  --image_path $images `
  --input_path $model `
  --output_path $ws/dense `
  --output_type COLMAP
if ($LASTEXITCODE -ne 0) {{ throw "image_undistorter failed" }}

& $colmap patch_match_stereo --workspace_path $ws/dense
if ($LASTEXITCODE -ne 0) {{ throw "patch_match_stereo failed" }}

& $colmap stereo_fusion `
  --workspace_path $ws/dense `
  --output_path $ws/dense/fused.ply
if ($LASTEXITCODE -ne 0) {{ throw "stereo_fusion failed" }}

Write-Host "-> $ws/dense/fused.ply   back to §6 for gravity alignment"
'''
(BASE / "run_colmap_video3_dense.ps1").write_text(ps_dense, encoding="utf-8-sig")
print("wrote run_colmap_video3_dense.ps1")
print(ps)

# --- Pause here and run COLMAP ---

# --- 6. T1 - gravity axis (no IMU needed) ---
# %%
import gravity_from_horizon_lock as gv

print("mapper sub-models:")
SPARSE = largest_model(WS / "sparse")
assert SPARSE is not None, f"no model in {WS/'sparse'} - run run_colmap_video3.ps1 first"
print("using", SPARSE)

poses, centers, n_total = gv.load_poses(SPARSE)
n_input = len(list((OUT / "images").glob("*.jpg")))
print(f"{n_total} images in model / {n_input} fed in -> registration {n_total/n_input*100:.0f}%"
      f"   {'✅' if n_total/n_input > 0.7 else '⚠️ sampling too sparse or wrong camera model'}")
print(f"{len(poses)} with parsed frame numbers")

up, resid_h, eig, cond = gv.solve_up(poses)
print(f"\n[1] up = [{up[0]:+.4f} {up[1]:+.4f} {up[2]:+.4f}]")
print(f"    residual (camera x-axis off horizontal) median {np.median(resid_h):.2f}deg  "
      f"p90 {np.percentile(resid_h,90):.2f}deg      {'✅' if np.median(resid_h)<1 else '⚠️'}")
print(f"    eigenvalues {eig[0]:.4g} / {eig[1]:.4g} / {eig[2]:.4g}   "
      f"condition ratio {cond:.4f}   {'✅' if cond<0.1 else '⚠️ camera orientations not spread out'}")

# %%
man = np.genfromtxt(OUT / "manifest.csv", delimiter=",", names=True,
                    dtype=None, encoding="utf-8")
imu_pitch = {int(f): p for f, p in zip(man["frame"], man["pitch_deg"])}
pairs = np.array([(np.degrees(np.arcsin(np.clip(up @ R[2, :], -1, 1))), imu_pitch[k])
                  for k, R in poses.items() if k in imu_pitch])

gain, bias = np.linalg.lstsq(
    np.vstack([pairs[:, 1], np.ones(len(pairs))]).T, pairs[:, 0], rcond=None)[0]
corr = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
print(f"[2] pitch comparison n={len(pairs)}  regression COLMAP = {gain:.3f}xIMU {bias:+.2f}  corr {corr:+.3f}")
print(f"    image pitch std {pairs[:,0].std():.2f}deg  vs  body {pairs[:,1].std():.2f}deg")
if gain < 0.5:
    print(f"    -> gain {gain:.2f}: EIS removed ~{(1-gain)*100:.0f}% of pitch motion, as expected. "
          f"This must NOT be used as the up-axis criterion (trial: 0.185 / corr 0.40)")
else:
    print(f"    ⚠️ gain {gain:.2f} near 1 = EIS is not stabilizing pitch - revisit §3")

C_all = np.array([centers[k] for k in sorted(centers)])
c0 = C_all - C_all.mean(0)
_u, _s, Vt = np.linalg.svd(c0, full_matrices=False)
normal = Vt[2] * np.sign(Vt[2] @ up)
ang = np.degrees(np.arccos(np.clip(normal @ up, -1, 1)))
print(f"[3] trajectory-plane normal vs up: angle {ang:.2f}deg (planarity {_s[2]/_s[1]:.3f})")
print("    note: Golden is sloped; this is only a magnitude check, a few degrees off is normal")

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].hist(resid_h, bins=50); ax[0].axvline(np.median(resid_h), color="r", ls="--")
ax[0].set_xlabel("deg"); ax[0].set_title("[1] camera x-axis off horizontal")
ax[1].scatter(pairs[:, 1], pairs[:, 0], s=8, alpha=.6)
lim = np.array([pairs[:, 1].min(), pairs[:, 1].max()])
ax[1].plot(lim, lim + bias, "r--", lw=1, label="gain 1 (EIS not stabilizing pitch)")
ax[1].plot(lim, gain * lim + bias, "g-", lw=1.2, label=f"measured gain {gain:.2f}")
ax[1].set_xlabel("IMU body pitch (deg)"); ax[1].set_ylabel("COLMAP-derived pitch (deg)")
ax[1].set_title("[2] pitch comparison (look at gain, not correlation)"); ax[1].legend(fontsize=7)
ax[2].scatter(c0 @ np.cross(up, [1, 0, 0]), c0 @ np.cross(up, np.cross(up, [1, 0, 0])),
              s=3, c=np.arange(len(c0)), cmap="viridis")
ax[2].set_aspect("equal"); ax[2].set_title("[3] camera centers on horizontal plane (= true top-down trajectory)")
plt.tight_layout(); plt.show()

# --- Write the gravity alignment back to the point cloud ---
# %%
z_axis = np.array([0, 0, 1.0])
v = np.cross(up, z_axis); s_ = np.linalg.norm(v); c_ = up @ z_axis
if s_ < 1e-8:
    R_align = np.eye(3) if c_ > 0 else np.diag([1, -1, -1.0])
else:
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_align = np.eye(3) + vx + vx @ vx * ((1 - c_) / s_ ** 2)

import json
(PREP / "gravity_align.json").write_text(json.dumps(dict(
    up_world=up.tolist(), R_align=R_align.tolist(),
    resid_median_deg=float(np.median(resid_h)), condition_ratio=float(cond),
    pitch_gain=float(gain), pitch_corr=float(corr),
    traj_plane_angle_deg=float(ang), model=str(SPARSE)), indent=2), encoding="utf-8")
print("R_align =\n", np.round(R_align, 4))
print("\nwrote", PREP / "gravity_align.json")

FUSED = WS / "dense" / "fused.ply"
if FUSED.exists():
    gv._rotate_ply(FUSED, FUSED.with_suffix(".up.ply"), R_align)
    print("wrote", FUSED.with_suffix(".up.ply"))
else:
    print(f"({FUSED} not present yet - run dense first, then re-run this cell)")

# --- 7. T2 - repeatability error bars (the most valuable part of this clip) ---
# %%
frames = np.array(sorted(centers))
C = np.array([centers[k] for k in frames])
T = frames / FPS

extent = np.linalg.norm(C - C.mean(0), axis=1).max()
REVISIT_R = 0.02 * extent
MIN_DT    = 60.0

D  = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
DT = np.abs(T[:, None] - T[None, :])
revisit = (D < REVISIT_R) & (DT > MIN_DT)
n_rev = revisit.any(1).sum()
print(f"trajectory scale {extent:.2f} (COLMAP units)  revisit radius {REVISIT_R:.3f}")
print(f"frames with revisits: {n_rev} / {len(frames)}  ({n_rev/len(frames)*100:.0f}%)")
if n_rev == 0:
    print("⚠️ no revisits found - increase REVISIT_R or confirm the loop actually closed")

# %%
pairs_ab = [(i, j) for i in range(len(frames)) for j in np.where(revisit[i])[0]
            if T[i] < T[j]]
A_idx = sorted({i for i, _ in pairs_ab}); B_idx = sorted({j for _, j in pairs_ab})
print(f"pass A {len(A_idx)} frames   pass B {len(B_idx)} frames")

for tag, idx in [("A", A_idx), ("B", B_idx)]:
    p = OUT / f"pass_{tag}_images.txt"
    p.write_text("\n".join(f"v3_{frames[i]:06d}.jpg" for i in idx), encoding="utf-8")
    print("wrote", p)

e1 = np.cross(up, [1, 0, 0]); e1 /= np.linalg.norm(e1)
e2 = np.cross(up, e1)
P = np.stack([(C - C.mean(0)) @ e1, (C - C.mean(0)) @ e2], 1)
plt.figure(figsize=(6.5, 6.5))
plt.plot(P[:, 0], P[:, 1], lw=.8, color="0.7", label="full track")
plt.scatter(P[A_idx, 0], P[A_idx, 1], s=12, c="tab:blue", label=f"pass A ({len(A_idx)})")
plt.scatter(P[B_idx, 0], P[B_idx, 1], s=12, c="tab:red", label=f"pass B ({len(B_idx)})")
plt.gca().set_aspect("equal"); plt.legend(); plt.title("revisit segments = the two T2 passes")
plt.show()

# --- How to continue T2 ---

# --- 8. Next steps ---
