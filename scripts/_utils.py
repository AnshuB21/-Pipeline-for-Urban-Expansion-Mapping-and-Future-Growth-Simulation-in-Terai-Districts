"""
Shared utilities used across all pipeline stages.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pathlib import Path
import logging
import json

log = logging.getLogger(__name__)


def load_params(pipeline_dir):
    """Load auto-computed parameters from Stage 1."""
    path = Path(pipeline_dir) / "pipeline_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"pipeline_params.json not found. Run Stage 1 first.")
    with open(path) as f:
        return json.load(f)


def compute_indices(bands, sensor):
    """Compute NDVI, NDBI, MNDWI for L5 or L8/9."""
    def ratio(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((a+b)!=0,(a-b)/(a+b),0.0).astype(np.float32)
    idx = {}
    if sensor == "L5":
        if "B4" in bands and "B3" in bands:
            idx["NDVI"]  = ratio(bands["B4"], bands["B3"])
        if "B5" in bands and "B4" in bands:
            idx["NDBI"]  = ratio(bands["B5"], bands["B4"])
        if "B2" in bands and "B5" in bands:
            idx["MNDWI"] = ratio(bands["B2"], bands["B5"])
    else:
        if "B5" in bands and "B4" in bands:
            idx["NDVI"]  = ratio(bands["B5"], bands["B4"])
        if "B6" in bands and "B5" in bands:
            idx["NDBI"]  = ratio(bands["B6"], bands["B5"])
        if "B3" in bands and "B6" in bands:
            idx["MNDWI"] = ratio(bands["B3"], bands["B6"])
    return idx


def align_features(X, cur_names, exp_names, n):
    """Reorder columns to match training feature order exactly."""
    out = np.zeros((n, len(exp_names)), dtype=np.float32)
    m   = {nm: i for i, nm in enumerate(cur_names)}
    for j, name in enumerate(exp_names):
        if name in m:
            out[:, j] = X[:, m[name]]
    return out


def reproject_to_crs(src_path, dst_path, target_crs, target_res=30):
    """Reproject raster to target CRS and resolution."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height,
            *src.bounds, resolution=target_res)
        profile = src.profile.copy()
        profile.update(crs=target_crs, transform=transform,
                       width=width, height=height, compress="lzw")
        with rasterio.open(dst_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                reproject(source=rasterio.band(src, b),
                          destination=rasterio.band(dst, b),
                          src_transform=src.transform,
                          src_crs=src.crs,
                          dst_transform=transform,
                          dst_crs=target_crs,
                          resampling=Resampling.bilinear)
    log.info(f"  Reprojected {src_path.name} → {dst_path.name}")


def align_to_ref(src_path, ref_path, dst_path,
                 resampling=Resampling.nearest, dtype=None):
    """Align src raster to exactly match ref grid."""
    with rasterio.open(ref_path) as ref:
        dst_t = ref.transform
        dst_c = ref.crs
        H, W  = ref.height, ref.width

    with rasterio.open(src_path) as src:
        out_dtype = dtype or src.dtypes[0]
        profile   = src.profile.copy()
        profile.update(crs=dst_c, transform=dst_t,
                       width=W, height=H,
                       dtype=out_dtype, compress="lzw")
        aligned = np.zeros((src.count, H, W), dtype=out_dtype)
        for b in range(1, src.count + 1):
            reproject(source=rasterio.band(src, b),
                      destination=aligned[b-1],
                      src_transform=src.transform,
                      src_crs=src.crs,
                      dst_transform=dst_t,
                      dst_crs=dst_c,
                      resampling=resampling)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(aligned)
    log.info(f"  Aligned {src_path.name} → {dst_path.name} ({W}×{H})")


def get_prob_map(img_path, band_names, sensor, clf,
                 feat_names, chunk=512):
    """Compute P(urban) for every pixel."""
    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        nd   = src.nodata if src.nodata is not None else 0
    probs = np.zeros(H * W, dtype=np.float32)
    with rasterio.open(img_path) as src:
        for row_off in range(0, H, chunk):
            row_end = min(row_off + chunk, H)
            win     = rasterio.windows.Window(
                0, row_off, W, row_end-row_off)
            c       = src.read(window=win).astype(np.float32)
            nb, h, w = c.shape
            valid   = np.all(c != nd, axis=0).ravel()
            bd      = {name: c[i] for i, name in
                       enumerate(band_names[:nb])}
            idx     = compute_indices(bd, sensor)
            fo      = list(bd.keys()) + list(idx.keys())
            Xr      = np.stack(list(bd.values())+list(idx.values()),
                               axis=-1).reshape(-1, len(fo))
            X       = align_features(Xr, fo, feat_names, h*w)
            X       = np.nan_to_num(X, nan=0.0,
                                    posinf=1.0, neginf=0.0)
            s, e    = row_off*W, row_off*W+h*w
            if valid.any():
                probs[s:e][valid] = clf.predict_proba(X[valid])[:, 1]
    return probs.reshape(H, W)


def find_threshold_for_target(prob_map, target_km2, res=30):
    """Binary search for threshold matching target km²."""
    target_px = int(target_km2 / (res/1000)**2)
    lo, hi    = 0.01, 0.99
    for _ in range(20):
        mid = (lo + hi) / 2
        if int((prob_map >= mid).sum()) > target_px:
            lo = mid
        else:
            hi = mid
    actual_km2 = float((prob_map >= mid).sum() * (res/1000)**2)
    log.info(f"  Threshold: {mid:.4f} → {actual_km2:.1f} km² "
             f"(target: {target_km2:.1f} km²)")
    return mid


def pixels_to_km2(n, res=30):
    return n * (res / 1000) ** 2
