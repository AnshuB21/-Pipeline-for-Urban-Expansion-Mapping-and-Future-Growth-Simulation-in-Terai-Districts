"""
Urban Expansion Pipeline — Master Runner
=========================================
Works for ANY AOI. Edit user_config.py then run this.

Stages
------
  0  Download all data from GEE automatically
  1  Auto-threshold label creation + growth rate calculation
  2  Train Random Forest classifiers (retrained per AOI)
  3  Apply classifiers (auto-calibrated thresholds)
  4  Align all rasters to common grid
  5  Change detection
  6  CA-ANN future growth simulation (rate-constrained)
  7  Internal spatial validation (70/30 holdout)

Usage
-----
  # Full pipeline (download + process)
  python run_pipeline.py

  # Skip GEE download (files already in RAW_DIR)
  python run_pipeline.py --from-stage 1

  # Run specific stages
  python run_pipeline.py --stages 1 2 3

  # Check inputs only
  python run_pipeline.py --check-inputs

  # See current AOI settings
  python run_pipeline.py --info
"""

import sys
import time
import argparse
import logging
import traceback
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Create output directories
from config import PIPELINE_DIR, AOI_NAME
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PIPELINE_DIR / "pipeline.log", mode="a"),
    ]
)
log = logging.getLogger(__name__)

STAGES = {
    0: ("Download from GEE",               "scripts.00_gee_export"),
    1: ("Auto-Threshold Labels",           "scripts.01_create_labels"),
    2: ("Train Classifiers",               "scripts.02_train_classifier"),
    3: ("Apply Classifiers",               "scripts.03_apply_classifier"),
    4: ("Align Rasters",                   "scripts.04_align_rasters"),
    5: ("Change Detection",                "scripts.05_change_detection"),
    6: ("CA-ANN Future Simulation",        "scripts.06_ca_ann_model"),
    7: ("Spatial Validation",              "scripts.07_validation"),
}


def run_stage(num):
    name, module_path = STAGES[num]
    log.info(f"\n{'═'*55}")
    log.info(f"  STAGE {num} — {name}")
    log.info(f"{'═'*55}\n")
    t0 = time.time()
    try:
        mod = importlib.import_module(module_path)
        importlib.reload(mod)
        mod.run()
        log.info(f"\n✅ Stage {num} complete  ({time.time()-t0:.1f}s)")
        return True
    except SystemExit as e:
        if e.code == 0:
            return True
        log.error(f"Stage {num} exited with code {e.code}")
        return False
    except Exception:
        log.error(f"\n❌ Stage {num} FAILED  ({time.time()-t0:.1f}s)")
        log.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description=f"Urban Expansion Pipeline — {AOI_NAME}")
    parser.add_argument("--stages",     nargs="+", type=int)
    parser.add_argument("--from-stage", type=int)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--info",         action="store_true")
    args = parser.parse_args()

    if args.info:
        from config import (TARGET_CRS, PIPELINE_DIR,
                            LANDSAT_1985_PATH, LANDSAT_2023_PATH)
        from user_config import AOI_COORDS, YEAR_HISTORICAL, YEAR_RECENT
        import numpy as np
        lons = [c[0] for c in AOI_COORDS]
        lats = [c[1] for c in AOI_COORDS]
        print(f"\nAOI:          {AOI_NAME}")
        print(f"Centroid:     {np.mean(lons):.4f}°E  "
              f"{np.mean(lats):.4f}°N")
        print(f"UTM zone:     {TARGET_CRS}  (auto-detected)")
        print(f"Years:        {YEAR_HISTORICAL} → {YEAR_RECENT}")
        print(f"Output dir:   {PIPELINE_DIR}")
        return

    if args.check_inputs:
        from config import check_inputs
        ok = check_inputs()
        sys.exit(0 if ok else 1)

    if args.stages:
        to_run = args.stages
    elif args.from_stage:
        to_run = list(range(args.from_stage, max(STAGES)+1))
    else:
        to_run = list(STAGES.keys())

    invalid = [s for s in to_run if s not in STAGES]
    if invalid:
        log.error(f"Invalid stage numbers: {invalid}")
        log.error(f"Valid: {list(STAGES.keys())}")
        sys.exit(1)

    log.info("══════════════════════════════════════════════")
    log.info("  URBAN EXPANSION PIPELINE")
    log.info(f"  AOI: {AOI_NAME}")
    log.info(f"  Stages: {to_run}")
    log.info("══════════════════════════════════════════════")

    t_total = time.time()
    failed  = []

    for num in to_run:
        ok = run_stage(num)
        if not ok:
            failed.append(num)
            log.error(f"\nStopped at Stage {num}")
            log.error(f"Fix the error then resume:")
            log.error(f"  python run_pipeline.py --from-stage {num}")
            break

    elapsed = time.time() - t_total
    log.info(f"\n{'═'*55}")
    if not failed:
        log.info(f"  ✅ ALL STAGES COMPLETE  ({elapsed:.0f}s)")
        log.info(f"  Results: {PIPELINE_DIR}/outputs/")
    else:
        log.info(f"  ❌ STOPPED at Stage {failed[0]}")
    log.info("═"*55)


if __name__ == "__main__":
    main()
