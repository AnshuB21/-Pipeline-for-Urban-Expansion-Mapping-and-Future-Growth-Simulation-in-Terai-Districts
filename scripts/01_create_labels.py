"""
Stage 1 — Auto-Threshold Label Creation
=========================================
Automatically finds the optimal GHSL threshold for this AOI
by analysing the full GHSL time series (1975-2020).

"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import logging
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg

log = logging.getLogger(__name__)


def align_to_ref(src_path, dst_path, ref_path,
                 resampling=Resampling.average):
    with rasterio.open(ref_path) as ref:
        dst_t = ref.transform
        dst_c = ref.crs
        H, W  = ref.height, ref.width
    with rasterio.open(src_path) as src:
        aligned = np.zeros((H, W), dtype=np.float32)
        reproject(source=rasterio.band(src, 1),
                  destination=aligned,
                  src_transform=src.transform,
                  src_crs=src.crs,
                  dst_transform=dst_t,
                  dst_crs=dst_c,
                  resampling=resampling)
        profile = src.profile.copy()
    profile.update(crs=dst_c, transform=dst_t,
                   width=W, height=H, count=1,
                   dtype="float32", nodata=None, compress="lzw")
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(aligned[np.newaxis])
    return aligned


def get_urban_area_at_threshold(ghsl_aligned, threshold, res=30):
    """Return urban area in km² at a given threshold."""
    return float((ghsl_aligned > threshold).sum() * (res/1000)**2)


def find_optimal_threshold(ghsl_path, ref_path, target_pct,
                            label=""):
    """
    Find threshold where urban area = target_pct of total AOI area.
    target_pct estimated from landscape type (flat=higher, hilly=lower).

    Uses binary search over threshold values.
    """
    with rasterio.open(ref_path) as ref:
        H, W         = ref.height, ref.width
        total_px     = H * W
        total_km2    = total_px * (30/1000)**2

    # Align GHSL to ref
    tmp = cfg.LABELS_DIR / f"_tmp_threshold_{label}.tif"
    aligned = align_to_ref(ghsl_path, tmp, ref_path)
    tmp.unlink(missing_ok=True)

    target_km2 = total_km2 * target_pct
    target_px  = int(target_km2 / (30/1000)**2)

    # Binary search for threshold
    lo, hi = 1.0, 2000.0
    for _ in range(30):
        mid   = (lo + hi) / 2
        n_px  = int((aligned > mid).sum())
        if n_px > target_px:
            lo = mid
        else:
            hi = mid

    final_km2 = float((aligned > mid).sum() * (30/1000)**2)
    log.info(f"  Auto threshold ({label}): {mid:.1f} m² "
             f"→ {final_km2:.1f} km² "
             f"({final_km2/total_km2*100:.1f}% of AOI)")
    return mid, aligned, final_km2


def estimate_urban_fraction(slope_path, ref_path, ghsl_path=None):
    """
    Estimate optimal urban fraction using two signals:

    Signal 1 — Terrain slope:
      Flat terrain = more urban possible
      Hilly terrain = less urban

    Signal 2 — GHSL median built density:
      Dense built pixels = urban AOI = lower threshold needed
      Sparse built pixels = rural AOI = higher threshold needed

    Combining both gives a threshold that adapts to each AOI
    automatically — no hardcoded values needed.
    """
    from rasterio.warp import reproject, Resampling as RS

    with rasterio.open(ref_path) as ref:
        H, W         = ref.height, ref.width
        dst_t, dst_c = ref.transform, ref.crs
        total_px     = H * W
        total_km2    = total_px * (30/1000)**2

    # ── Signal 1: Terrain slope ───────────────────────────────
    if slope_path.exists():
        with rasterio.open(slope_path) as src:
            slope_data = np.zeros((H, W), dtype=np.float32)
            reproject(source=rasterio.band(src, 1),
                      destination=slope_data,
                      src_transform=src.transform,
                      src_crs=src.crs,
                      dst_transform=dst_t,
                      dst_crs=dst_c,
                      resampling=RS.bilinear)
        avg_slope = float(slope_data[slope_data > 0].mean())
    else:
        avg_slope = 5.0
        log.warning("  Slope not found — using default 5°")

    log.info(f"  Average slope: {avg_slope:.2f}°")

    # Slope-based terrain class
    if avg_slope < 2:
        terrain       = "Very flat Terai"
        slope_pct     = 0.22
    elif avg_slope < 5:
        terrain       = "Flat Terai"
        slope_pct     = 0.16
    elif avg_slope < 10:
        terrain       = "Undulating"
        slope_pct     = 0.10
    else:
        terrain       = "Hilly"
        slope_pct     = 0.05

    log.info(f"  Terrain type: {terrain}")

    # ── Signal 2: GHSL built density ─────────────────────────
    # How dense are the built pixels in this AOI?
    # Dense = urban district = can support more urban pixels
    # Sparse = rural district = fewer urban pixels
    if ghsl_path and Path(ghsl_path).exists():
        with rasterio.open(ghsl_path) as src:
            ghsl_data = np.zeros((H, W), dtype=np.float32)
            reproject(source=rasterio.band(src, 1),
                      destination=ghsl_data,
                      src_transform=src.transform,
                      src_crs=src.crs,
                      dst_transform=dst_t,
                      dst_crs=dst_c,
                      resampling=RS.average)

        built = ghsl_data[ghsl_data > 0]

        if built.size > 0:
            # Median built density of pixels that have any building
            median_density = float(np.median(built))
            pct_with_any   = float((ghsl_data > 50).sum()) / total_px

            log.info(f"  Median built density: {median_density:.1f} m²/pixel")
            log.info(f"  Pixels with any building: "
                     f"{pct_with_any*100:.1f}% of AOI")

            # Density adjustment factor
            # High median density = dense urban = allow more pixels
            # Low median density  = sparse rural = fewer pixels
            if median_density > 500:
                density_factor = 1.3    # very dense urban
            elif median_density > 300:
                density_factor = 1.1    # dense urban
            elif median_density > 150:
                density_factor = 1.0    # moderate
            elif median_density > 80:
                density_factor = 0.85   # sparse
            else:
                density_factor = 0.70   # very sparse rural

            log.info(f"  Density factor: {density_factor:.2f}  "
                     f"(1.0=neutral, >1=denser, <1=sparser)")
        else:
            density_factor = 1.0
            log.warning("  No built pixels found in GHSL — "
                        "using neutral density factor")
    else:
        density_factor = 1.0
        log.warning("  GHSL not available for density check")

    # ── Combine both signals ──────────────────────────────────
    max_urban_pct = slope_pct * density_factor

    # Hard limits — never go below 1% or above 35%
    max_urban_pct = float(np.clip(max_urban_pct, 0.01, 0.35))

    log.info(f"  Slope-based pct:  {slope_pct*100:.1f}%")
    log.info(f"  After density adj:{max_urban_pct*100:.1f}%  "
             f"(slope × {density_factor:.2f})")
    log.info(f"  Final urban target: {max_urban_pct*100:.1f}% of AOI "
             f"= {total_km2*max_urban_pct:.1f} km²")

    return max_urban_pct, avg_slope, terrain


def compute_growth_rate(ghsl_path_1985, ghsl_path_2020,
                        ref_path, threshold_1985, threshold_2020):
    """
    Compute annual urban growth rate from GHSL epochs.
    This replaces the hardcoded 4.75% rate.
    """
    tmp1 = cfg.LABELS_DIR / "_tmp_rate1985.tif"
    tmp2 = cfg.LABELS_DIR / "_tmp_rate2020.tif"

    aligned_1985 = align_to_ref(ghsl_path_1985, tmp1, ref_path)
    aligned_2020 = align_to_ref(ghsl_path_2020, tmp2, ref_path)
    tmp1.unlink(missing_ok=True)
    tmp2.unlink(missing_ok=True)

    area_1985 = float((aligned_1985 > threshold_1985).sum() * (30/1000)**2)
    area_2020 = float((aligned_2020 > threshold_2020).sum() * (30/1000)**2)

    years = 2020 - 1985
    if area_1985 > 0:
        annual_rate = (np.power(area_2020/area_1985, 1/years) - 1)
    else:
        annual_rate = 0.05   # fallback

    log.info(f"  GHSL 1985: {area_1985:.1f} km²")
    log.info(f"  GHSL 2020: {area_2020:.1f} km²")
    log.info(f"  Annual growth rate: {annual_rate*100:.2f}%/year")
    return annual_rate, area_1985, area_2020


def binarize_and_save(aligned, out_path, threshold, ref_path):
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    binary    = (aligned > threshold).astype(np.uint8)
    urban_km2 = float(binary.sum() * (30/1000)**2)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(binary[np.newaxis])
    log.info(f"  Saved: {out_path.name}  ({urban_km2:.1f} km²)")
    return urban_km2


def run():
    cfg.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MAPS_DIR.mkdir(parents=True, exist_ok=True)

    for p in [cfg.LANDSAT_1985_PATH, cfg.LANDSAT_2023_PATH,
              cfg.GHSL_PATH_1985, cfg.GHSL_PATH]:
        if not p.exists():
            log.error(f"Missing: {p.name} — run Stage 0 first")
            sys.exit(1)

    log.info("══════════════════════════════════════════════")
    log.info("STAGE 1 — AUTO-THRESHOLD LABEL CREATION")
    log.info(f"  AOI: {cfg.AOI_NAME}")
    log.info("══════════════════════════════════════════════")

    # Step 1 — Estimate urban fraction from terrain
    log.info("\n── Step 1: Terrain analysis ──────────────────")
    result = estimate_urban_fraction(cfg.SLOPE_PATH,
                                     cfg.LANDSAT_2023_PATH,
                                     ghsl_path=cfg.GHSL_PATH)
    if isinstance(result, tuple):
        max_urban_pct, avg_slope, terrain = result
    else:
        max_urban_pct = result
        avg_slope     = 5.0
        terrain       = "Unknown"

    # Step 2 — Find optimal threshold for 2020 labels
    log.info("\n── Step 2: Find 2023 threshold ───────────────")
    thresh_2020, aligned_2020, area_2020 = find_optimal_threshold(
        cfg.GHSL_PATH, cfg.LANDSAT_2023_PATH,
        target_pct=max_urban_pct, label="2023")

    # Step 3 — Find optimal threshold for 1985 labels
    # 1985 urban is typically 20-40% of 2020 urban
    # Use proportionally smaller target
    log.info("\n── Step 3: Find 1985 threshold ───────────────")
    target_1985_pct = max_urban_pct * 0.3   # ~30% of max
    thresh_1985, aligned_1985, area_1985 = find_optimal_threshold(
        cfg.GHSL_PATH_1985, cfg.LANDSAT_1985_PATH,
        target_pct=target_1985_pct, label="1985")

    # Step 4 — Compute growth rate
    log.info("\n── Step 4: Compute growth rate ───────────────")
    annual_rate, ghsl_1985_km2, ghsl_2020_km2 = compute_growth_rate(
        cfg.GHSL_PATH_1985, cfg.GHSL_PATH,
        cfg.LANDSAT_2023_PATH, thresh_1985, thresh_2020)

    # Step 5 — Save binary labels
    log.info("\n── Step 5: Save labels ───────────────────────")
    km2_1985 = binarize_and_save(
        aligned_1985,
        cfg.LABELS_DIR / "urban_labels_1985_aligned.tif",
        thresh_1985, cfg.LANDSAT_1985_PATH)

    km2_2023 = binarize_and_save(
        aligned_2020,
        cfg.LABELS_DIR / "urban_labels_2023_aligned.tif",
        thresh_2020, cfg.LANDSAT_2023_PATH)

    # Step 6 — Save computed parameters to JSON
    # These are loaded by later stages
    params = {
        "aoi_name":          cfg.AOI_NAME,
        "terrain":           terrain,
        "avg_slope":         avg_slope,
        "threshold_1985":    thresh_1985,
        "threshold_2023":    thresh_2020,
        "area_1985_km2":     km2_1985,
        "area_2023_km2":     km2_2023,
        "ghsl_1985_km2":     ghsl_1985_km2,
        "ghsl_2020_km2":     ghsl_2020_km2,
        "annual_rate":       annual_rate,
        "max_urban_pct":     max_urban_pct,
    }

    params_path = cfg.PIPELINE_DIR / "pipeline_params.json"
    with open(params_path, "w") as f:
        import json
        json.dump(params, f, indent=2)
    log.info(f"\n  Parameters saved → {params_path.name}")

    # GHSL timeline chart
    fig, ax = plt.subplots(figsize=(9, 5))
    years  = [1985, 2020]
    areas  = [ghsl_1985_km2, ghsl_2020_km2]
    ax.bar([str(y) for y in years], areas,
           color=["#f6b26b","#cc0000"], edgecolor="white")
    for i, (y, a) in enumerate(zip(years, areas)):
        ax.text(i, a + max(areas)*0.02,
                f"{a:.1f} km²", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Urban Area (km²)")
    ax.set_title(f"GHSL Urban Extent — {cfg.AOI_NAME}\n"
                 f"Annual growth: {annual_rate*100:.2f}%/year",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(cfg.MAPS_DIR / "ghsl_timeline.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    log.info("\n══════════════════════════════════════════════")
    log.info("STAGE 1 COMPLETE")
    log.info("══════════════════════════════════════════════")
    log.info(f"  Terrain:     {terrain}  (slope={avg_slope:.1f}°)")
    log.info(f"  1985 labels: {km2_1985:.1f} km²  "
             f"(threshold={thresh_1985:.0f} m²)")
    log.info(f"  2023 labels: {km2_2023:.1f} km²  "
             f"(threshold={thresh_2020:.0f} m²)")
    log.info(f"  Growth rate: {annual_rate*100:.2f}%/year  (auto-computed)")
    log.info("══════════════════════════════════════════════")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
