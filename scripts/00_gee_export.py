"""
Stage 0 — Automatic GEE Export
================================
Downloads all required GeoTiffs from Google Earth Engine
for the AOI defined in user_config.py.

What it exports:
  - Landsat 5 TM composite (historical)
  - Landsat 8/9 OLI composite (recent)
  - GHSL built-up surface (all epochs 1975-2020)
  - SRTM elevation
  - Slope derived from SRTM
  - Distance to water (JRC GSW)
  - Distance to existing urban (from GHSL 2020)

Requirements:
  pip install earthengine-api
  earthengine authenticate  (run once in terminal)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
import shutil

log = logging.getLogger(__name__)

from config import (
    AOI_NAME, AOI_COORDS,
    YEAR_HISTORICAL, YEAR_RECENT,
    DATE_HISTORICAL_START, DATE_HISTORICAL_END,
    DATE_RECENT_START, DATE_RECENT_END,
    DRIVE_FOLDER, DRIVE_ROOT, RAW_DIR,
    LANDSAT_1985_PATH, LANDSAT_2023_PATH,
    GHSL_PATH, GHSL_PATH_1985,
    ELEVATION_PATH, SLOPE_PATH,
    DIST_WATER_PATH, DIST_URBAN_PATH,
)

# Import user config directly for dates
import user_config as UC


def init_gee():
    """Initialise Google Earth Engine."""
    try:
        import ee
        try:
            ee.Initialize(project="ee-default")
        except Exception:
            ee.Authenticate()
            ee.Initialize(project="ee-default")
        log.info("  ✅ Google Earth Engine initialised")
        return ee
    except ImportError:
        log.error("earthengine-api not installed")
        log.error("Run: pip install earthengine-api")
        sys.exit(1)
    except Exception as e:
        log.error(f"GEE initialisation failed: {e}")
        log.error("Run: earthengine authenticate")
        sys.exit(1)


def wait_for_tasks(tasks, poll_interval=30):
    """Wait for all GEE export tasks to complete."""
    log.info(f"  Waiting for {len(tasks)} export tasks ...")
    import ee
    pending = list(tasks)
    while pending:
        still_running = []
        for task in pending:
            status = task.status()["state"]
            if status in ("COMPLETED",):
                log.info(f"    ✅ {task.status()['description']}")
            elif status in ("FAILED", "CANCELLED"):
                log.error(f"    ❌ {task.status()['description']}: "
                          f"{task.status().get('error_message','')}")
            else:
                still_running.append(task)
        pending = still_running
        if pending:
            log.info(f"    {len(pending)} tasks still running ...")
            time.sleep(poll_interval)
    log.info("  ✅ All export tasks complete")


def copy_from_drive():
    """Copy exported files from Google Drive to RAW_DIR."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    file_map = {
        f"{AOI_NAME}_landsat_{YEAR_HISTORICAL}.tif": LANDSAT_1985_PATH,
        f"{AOI_NAME}_landsat_{YEAR_RECENT}.tif":     LANDSAT_2023_PATH,
        f"{AOI_NAME}_GHSL_2020.tif":                 GHSL_PATH,
        f"{AOI_NAME}_GHSL_1985.tif":                 GHSL_PATH_1985,
        f"{AOI_NAME}_elevation.tif":                 ELEVATION_PATH,
        f"{AOI_NAME}_slope.tif":                     SLOPE_PATH,
        f"{AOI_NAME}_distance_to_water.tif":         DIST_WATER_PATH,
        f"{AOI_NAME}_distance_to_urban.tif":         DIST_URBAN_PATH,
    }

    log.info("  Copying from Google Drive ...")
    all_ok = True
    for fname, dst in file_map.items():
        src = DRIVE_ROOT / fname
        if src.exists():
            shutil.copy(src, dst)
            size = dst.stat().st_size / 1e6
            log.info(f"    ✅ {fname}  ({size:.1f} MB)")
        else:
            log.warning(f"    ⚠️  Not found yet: {fname}")
            all_ok = False
    return all_ok


def run():
    log.info("══════════════════════════════════════════════")
    log.info("STAGE 0 — AUTOMATIC GEE EXPORT")
    log.info(f"  AOI: {AOI_NAME}")
    log.info("══════════════════════════════════════════════")

    ee = init_gee()

    # ── Define AOI ────────────────────────────────────────────
    aoi = ee.Geometry.Polygon([AOI_COORDS])
    log.info(f"  AOI defined: {len(AOI_COORDS)} vertices")

    # ── Masking functions ─────────────────────────────────────
    def mask_scale_l5(image):
        qa   = image.select("QA_PIXEL")
        mask = (qa.bitwiseAnd(1 << 3).eq(0)
                .And(qa.bitwiseAnd(1 << 4).eq(0)))
        return (image.updateMask(mask)
                .select("SR_B.")
                .multiply(0.0000275).add(-0.2)
                .copyProperties(image, image.propertyNames()))

    def mask_scale_oli(image):
        qa   = image.select("QA_PIXEL")
        mask = (qa.bitwiseAnd(1 << 3).eq(0)
                .And(qa.bitwiseAnd(1 << 4).eq(0)))
        return (image.updateMask(mask)
                .select("SR_B.")
                .multiply(0.0000275).add(-0.2)
                .copyProperties(image, image.propertyNames()))

    # ── Landsat 5 TM — Historical ─────────────────────────────
    l5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
          .filterBounds(aoi)
          .filterDate(UC.DATE_HISTORICAL_START, UC.DATE_HISTORICAL_END)
          .filter(ee.Filter.calendarRange(10, 4, "month"))
          .filter(ee.Filter.lt("CLOUD_COVER_LAND", 20))
          .map(mask_scale_l5))

    n_l5 = l5.size().getInfo()
    log.info(f"  Landsat 5 scenes: {n_l5}")
    if n_l5 == 0:
        log.error("No Landsat 5 scenes found — check date range and AOI")
        sys.exit(1)

    landsat_1985 = (l5.median()
                    .select(["SR_B1","SR_B2","SR_B3",
                             "SR_B4","SR_B5","SR_B7"],
                            ["B1","B2","B3","B4","B5","B7"])
                    .clip(aoi).toFloat())

    # ── Landsat 8+9 OLI — Recent ──────────────────────────────
    def select_oli(ic):
        return (ic.select(["SR_B2","SR_B3","SR_B4",
                            "SR_B5","SR_B6","SR_B7"],
                           ["B2","B3","B4","B5","B6","B7"]))

    l8 = select_oli(
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(UC.DATE_RECENT_START, UC.DATE_RECENT_END)
        .filter(ee.Filter.calendarRange(10, 4, "month"))
        .filter(ee.Filter.lt("CLOUD_COVER_LAND", 20))
        .map(mask_scale_oli))

    l9 = select_oli(
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(UC.DATE_RECENT_START, UC.DATE_RECENT_END)
        .filter(ee.Filter.calendarRange(10, 4, "month"))
        .filter(ee.Filter.lt("CLOUD_COVER_LAND", 20))
        .map(mask_scale_oli))

    landsat_2023 = l8.merge(l9).median().clip(aoi).toFloat()
    n_recent = l8.merge(l9).size().getInfo()
    log.info(f"  Landsat 8+9 scenes: {n_recent}")

    # ── GHSL — all available epochs ───────────────────────────
    ghsl_2020 = (ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
                 .filterDate("2020-01-01","2021-01-01")
                 .first().select("built_surface").clip(aoi))

    ghsl_1985 = (ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
                 .filterDate("1985-01-01","1986-01-01")
                 .first().select("built_surface").clip(aoi))

    # ── Elevation + Slope ─────────────────────────────────────
    dem   = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    slope = ee.Terrain.slope(dem).clip(aoi)

    # ── Distance to water ─────────────────────────────────────
    water = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
             .select("occurrence").gt(50).clip(aoi))
    dist_water = (water.eq(0)
                  .fastDistanceTransform(512)
                  .sqrt().multiply(30)
                  .clip(aoi).toFloat())

    # ── Distance to existing urban ────────────────────────────
    urban_mask = ghsl_2020.gt(250)
    dist_urban = (urban_mask.eq(0)
                  .fastDistanceTransform(512)
                  .sqrt().multiply(100)
                  .clip(aoi).toFloat())

    # ── Export all images ─────────────────────────────────────
    export_params = {
        "folder":    DRIVE_FOLDER,
        "region":    aoi,
        "scale":     30,
        "maxPixels": 1e10,
        "fileFormat": "GeoTIFF",
        "formatOptions": {"cloudOptimized": True},
    }

    exports = [
        (landsat_1985,      f"{AOI_NAME}_landsat_{YEAR_HISTORICAL}"),
        (landsat_2023,      f"{AOI_NAME}_landsat_{YEAR_RECENT}"),
        (ghsl_2020.toFloat(),f"{AOI_NAME}_GHSL_2020"),
        (ghsl_1985.toFloat(),f"{AOI_NAME}_GHSL_1985"),
        (dem.toFloat(),     f"{AOI_NAME}_elevation"),
        (slope.toFloat(),   f"{AOI_NAME}_slope"),
        (dist_water,        f"{AOI_NAME}_distance_to_water"),
        (dist_urban,        f"{AOI_NAME}_distance_to_urban"),
    ]

    log.info(f"\n  Starting {len(exports)} export tasks ...")
    tasks = []
    for image, name in exports:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=name,
            fileNamePrefix=name,
            **export_params,
        )
        task.start()
        tasks.append(task)
        log.info(f"    Started: {name}")

    # ── Wait for completion ───────────────────────────────────
    wait_for_tasks(tasks)

    # ── Copy to RAW_DIR ───────────────────────────────────────
    log.info("\n  Copying files from Drive to pipeline ...")
    all_ok = copy_from_drive()

    if all_ok:
        log.info("\n✅ Stage 0 complete — all files downloaded")
        log.info(f"   Files in: {RAW_DIR}")
    else:
        log.warning("\n⚠️  Some files not yet available")
        log.warning("  Wait a few minutes and run Stage 0 again")
        log.warning("  Or manually copy files from Drive")

    log.info("══════════════════════════════════════════════")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
