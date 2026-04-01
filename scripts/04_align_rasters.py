"""Stage 4 — Align All Rasters to Common Grid"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
import logging

import config as cfg
from scripts._utils import align_to_ref

log = logging.getLogger(__name__)


def gdal_align(src, ref, dst, r="near"):
    with rasterio.open(ref) as rf:
        b = rf.bounds
    cmd = ["gdalwarp",
           "-t_srs", cfg.TARGET_CRS,
           "-tr", str(cfg.TARGET_RES), str(cfg.TARGET_RES),
           "-te", str(b.left),str(b.bottom),str(b.right),str(b.top),
           "-te_srs", cfg.TARGET_CRS, "-tap",
           "-r", r, "-of", "GTiff", "-co", "COMPRESS=LZW",
           "-overwrite", str(src), str(dst)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def align_one(src, ref, dst, gr="near",
              rr=Resampling.nearest, dtype=None):
    log.info(f"  {Path(src).name} → {Path(dst).name}")
    if not gdal_align(src, ref, dst, gr):
        align_to_ref(src, ref, dst, rr, dtype)
    log.info(f"    ✅ {Path(dst).stat().st_size/1e6:.1f} MB")


def run():
    cfg.RASTERS_DIR.mkdir(parents=True, exist_ok=True)
    ref = cfg.RASTERS_DIR / f"urban_{cfg.YEAR_RECENT}_aligned.tif"
    if not ref.exists():
        log.error("urban_2023_aligned.tif not found — run Stage 3")
        sys.exit(1)

    log.info("══════════════════════════════════════════════")
    log.info(f"STAGE 4 — ALIGN RASTERS  [{cfg.AOI_NAME}]")
    log.info("══════════════════════════════════════════════")

    to_align = [
        (cfg.RASTERS_DIR/f"urban_{cfg.YEAR_HISTORICAL}.tif",
         cfg.RASTERS_DIR/f"urban_{cfg.YEAR_HISTORICAL}_aligned.tif",
         "near", Resampling.nearest, "uint8"),
        (cfg.ELEVATION_PATH,
         cfg.RASTERS_DIR/"elevation_aligned.tif",
         "bilinear", Resampling.bilinear, "float32"),
        (cfg.SLOPE_PATH,
         cfg.RASTERS_DIR/"slope_aligned.tif",
         "bilinear", Resampling.bilinear, "float32"),
        (cfg.DIST_WATER_PATH,
         cfg.RASTERS_DIR/"distance_to_water_aligned.tif",
         "bilinear", Resampling.bilinear, "float32"),
        (cfg.DIST_URBAN_PATH,
         cfg.RASTERS_DIR/"distance_to_urban_aligned.tif",
         "bilinear", Resampling.bilinear, "float32"),
    ]

    for src, dst, gr, rr, dt in to_align:
        if Path(src).exists():
            align_one(src, ref, dst, gr, rr, dt)
        else:
            log.warning(f"  Skipping missing: {Path(src).name}")

    log.info("✓ Stage 4 complete")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
