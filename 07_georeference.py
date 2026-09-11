"""07 - georeference fused.ply (video_3) using gravity alignment + manual control points.

Pipeline:
  1. load fused.ply (raw COLMAP frame, arbitrary scale/rotation/origin)
  2. apply R_align from vedio3_prep/gravity_align.json -> gravity "up" already solved via
     IMU horizon-lock (no GPS needed), so pitch/roll are already correct after this step.
  3. remaining unknowns: yaw (compass heading), scale (COLMAP units are NOT meters), and
     horizontal/vertical translation. These are solved from manually-picked control points:
     pick a point in fused.ply (CloudCompare "Point #..." readout = raw xyz, BEFORE gravity
     alignment), eyeball the same feature on satellite imagery (Google/Bing Maps) for its
     lat/lon, add a row to `control_points` below.
       - 1 point  -> can only pin that one spot. scale/yaw stay unresolved (assumed 1.0 / 0 deg)
                      -> NOT valid for anything beyond the anchor itself.
       - >=2 points -> full 2D similarity (scale + yaw + translation) solved by least squares.
                      -> use 2-3 points spread across the capture, not clustered together.
"""

# --- 0. Config ---
# %%
import json
from pathlib import Path
import numpy as np

BASE   = Path(".").resolve()
WS     = BASE / "colmap_workspace_video_3"
DENSE  = WS / "dense"
PLY_IN = DENSE / "fused.ply"
PREP   = BASE / "vedio3_prep"
GRAVITY_JSON = PREP / "gravity_align.json"
OUT    = BASE / "georef_output"; OUT.mkdir(exist_ok=True)

print("ply    :", PLY_IN, "exists", PLY_IN.exists())
print("gravity:", GRAVITY_JSON, "exists", GRAVITY_JSON.exists())

# --- 1. Generic PLY read/write (keeps whatever property list the file has) ---
# %%
_TMAP = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'uchar': 'u1',
         'uint8': 'u1', 'int': '<i4', 'short': '<i2', 'ushort': '<u2'}

def read_ply_generic(path):
    with open(path, 'rb') as f:
        header, props, n = [], [], None
        while True:
            line = f.readline()
            header.append(line)
            if line.startswith(b'element vertex'):
                n = int(line.split()[-1])
            elif line.startswith(b'property'):
                parts = line.split()
                props.append((parts[1].decode(), parts[2].decode()))
            elif line.strip() == b'end_header':
                break
        dt = np.dtype([(nm, _TMAP[ty]) for ty, nm in props])
        arr = np.fromfile(f, dtype=dt, count=n)
    return arr, header

def write_ply_generic(path, arr, header):
    with open(path, 'wb') as f:
        f.writelines(header)
        arr.tofile(f)

arr, header = read_ply_generic(PLY_IN)
print(f"{len(arr):,} points, fields: {arr.dtype.names}")

xyz_raw = np.stack([arr['x'], arr['y'], arr['z']], 1).astype(np.float64)
has_normals = all(k in arr.dtype.names for k in ('nx', 'ny', 'nz'))
nrm_raw = np.stack([arr['nx'], arr['ny'], arr['nz']], 1).astype(np.float64) if has_normals else None

# --- 2. Gravity ("up") alignment - already solved from IMU, no control points needed ---
# %%
align = json.loads(GRAVITY_JSON.read_text())
R_align = np.array(align['R_align'])          # rotates raw fused.ply frame -> Z-up frame
print("R_align loaded, resid_median_deg =", align.get('resid_median_deg'))

xyz_up = xyz_raw @ R_align.T                   # x,y = horizontal-ish, z = up (arbitrary scale/yaw/origin still)
nrm_up = (nrm_raw @ R_align.T) if has_normals else None

# --- 3. Control points: pointcloud xyz (RAW fused.ply, same as CloudCompare readout) -> lat/lon ---
# %%
# Add rows here. xyz_raw must be picked directly from fused.ply (pre-gravity-alignment),
# exactly like the "Point #3807787" CloudCompare readout.
control_points = [
    dict(name="pole_3807787", xyz_raw=(-0.032206, 0.030240, -0.685244),
         lat=39.75113026724487, lon=-105.2159056166409),
    # dict(name="...", xyz_raw=(x, y, z), lat=..., lon=...),
]

if len(control_points) < 2:
    print("!!! WARNING: only", len(control_points), "control point(s).")
    print("    Scale and heading are UNRESOLVED (assumed scale=1.0, yaw=0 deg as placeholders).")
    print("    Only the anchor point itself will land on its real-world spot; everything else")
    print("    in the cloud may be stretched/rotated arbitrarily relative to it.")
    print("    -> pick >=1 more point (ideally 2 total, well separated) before trusting this.")

# --- 4. lat/lon -> local flat-earth meters (equirectangular, fine at this spatial scale) ---
# %%
R_EARTH = 6378137.0

def latlon_to_local_m(lat, lon, lat0, lon0):
    north = np.radians(lat - lat0) * R_EARTH
    east  = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    return east, north

def local_m_to_latlon(east, north, lat0, lon0):
    lat = lat0 + np.degrees(north / R_EARTH)
    lon = lon0 + np.degrees(east / (R_EARTH * np.cos(np.radians(lat0))))
    return lat, lon

lat0 = np.mean([cp['lat'] for cp in control_points])
lon0 = np.mean([cp['lon'] for cp in control_points])
print(f"local origin: lat0={lat0:.8f} lon0={lon0:.8f}")

src_xy, dst_en = [], []
for cp in control_points:
    xyz_up_cp = np.array(cp['xyz_raw']) @ R_align.T
    src_xy.append(xyz_up_cp[:2])
    dst_en.append(latlon_to_local_m(cp['lat'], cp['lon'], lat0, lon0))
src_xy = np.array(src_xy)
dst_en = np.array(dst_en)

# --- 5. Solve horizontal similarity transform (scale + yaw + translation) ---
# %%
def fit_similarity_2d(src, dst):
    """Least-squares similarity: dst ~= scale * R @ src + t.  Needs >=2 points (Umeyama)."""
    n = len(src)
    if n < 2:
        return 1.0, np.eye(2), dst[0] - src[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / n
    scale = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - scale * R @ mu_s
    return scale, R, t

scale, R2, t2 = fit_similarity_2d(src_xy, dst_en)
yaw_deg = np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))
print(f"scale={scale:.6f}  yaw={yaw_deg:.2f} deg  t={t2}")

if len(control_points) >= 2:
    pred = (src_xy @ R2.T) * scale + t2
    err = np.linalg.norm(pred - dst_en, axis=1)
    for cp, e in zip(control_points, err):
        print(f"  residual @ {cp['name']}: {e:.2f} m")

# --- 6. Apply full transform to the whole cloud, write output ---
# %%
R3 = np.eye(3); R3[:2, :2] = R2                # yaw-only 3x3 (Z axis untouched)
east_north = (xyz_up[:, :2] @ R2.T) * scale + t2
z_geo = xyz_up[:, 2] * scale                    # NOT anchored to real elevation - see note below

xyz_geo = np.column_stack([east_north, z_geo])
arr_out = arr.copy()
arr_out['x'], arr_out['y'], arr_out['z'] = xyz_geo[:, 0], xyz_geo[:, 1], xyz_geo[:, 2]
if has_normals:
    nrm_geo = nrm_up @ R3.T                     # rotate only, never scale/translate normals
    arr_out['nx'], arr_out['ny'], arr_out['nz'] = nrm_geo[:, 0], nrm_geo[:, 1], nrm_geo[:, 2]

write_ply_generic(OUT / "fused_geo.ply", arr_out, header)
print("wrote", OUT / "fused_geo.ply")
print("coords = local ENU meters relative to lat0/lon0 (NOT lat/lon, NOT a real elevation datum)")

(OUT / "georef_transform.json").write_text(json.dumps(dict(
    R_align=R_align.tolist(), R_yaw=R2.tolist(), scale=float(scale), yaw_deg=float(yaw_deg),
    t_east_north=t2.tolist(), lat0=float(lat0), lon0=float(lon0),
    z_note="z is scale * gravity-up z, relative to an arbitrary local reference - "
           "not tied to a real geodetic elevation unless a control point supplies one",
    control_points=control_points,
), indent=2))
print("wrote", OUT / "georef_transform.json  (reuse this to transform pole/tree/wire clouds later)")

# --- 7. Sanity check: recover lat/lon at each control point ---
# %%
for cp in control_points:
    xyz_up_cp = np.array(cp['xyz_raw']) @ R_align.T
    en = xyz_up_cp[:2] @ R2.T * scale + t2
    lat_chk, lon_chk = local_m_to_latlon(en[0], en[1], lat0, lon0)
    print(f"{cp['name']}: recovered ({lat_chk:.8f}, {lon_chk:.8f})  vs input ({cp['lat']:.8f}, {cp['lon']:.8f})")
