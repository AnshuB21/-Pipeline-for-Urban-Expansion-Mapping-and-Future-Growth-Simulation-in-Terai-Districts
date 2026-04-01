"""
Stage 3 — Apply Classifiers (Auto-Calibrated)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, rasterio, joblib, logging
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import config as cfg
from scripts._utils import (load_params, get_prob_map,
                             find_threshold_for_target,
                             reproject_to_crs, pixels_to_km2)

log = logging.getLogger(__name__)


def run():
    cfg.RASTERS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MAPS_DIR.mkdir(parents=True, exist_ok=True)

    # Load auto-computed thresholds from Stage 1
    params      = load_params(cfg.PIPELINE_DIR)
    thresh_1985 = params["threshold_1985"]
    thresh_2023 = params["threshold_2023"]
    area_1985   = params["area_1985_km2"]
    area_2023   = params["area_2023_km2"]

    log.info("══════════════════════════════════════════════")
    log.info(f"STAGE 3 — APPLY CLASSIFIERS  [{cfg.AOI_NAME}]")
    log.info("  Auto-calibrated to GHSL for this AOI")
    log.info("══════════════════════════════════════════════")

    for p in [cfg.MODELS_DIR/"classifier_1985.pkl",
              cfg.MODELS_DIR/"classifier_2023.pkl"]:
        if not p.exists():
            log.error(f"Missing: {p.name} — run Stage 2 first")
            sys.exit(1)

    b1985 = joblib.load(cfg.MODELS_DIR / "classifier_1985.pkl")
    b2023 = joblib.load(cfg.MODELS_DIR / "classifier_2023.pkl")

    # Align Landsat to project CRS
    al1985 = cfg.RASTERS_DIR / "landsat_1985_aligned.tif"
    al2023 = cfg.RASTERS_DIR / "landsat_2023_aligned.tif"
    if not al1985.exists():
        reproject_to_crs(cfg.LANDSAT_1985_PATH, al1985,
                         cfg.TARGET_CRS, cfg.TARGET_RES)
    if not al2023.exists():
        reproject_to_crs(cfg.LANDSAT_2023_PATH, al2023,
                         cfg.TARGET_CRS, cfg.TARGET_RES)

    def classify_epoch(img, bands, sensor, bundle,
                       target_km2, out_bin, out_prob, label):
        log.info(f"\n── {label} ──────────────────────────────────")
        log.info(f"  GHSL target: {target_km2:.1f} km²")
        prob_map = get_prob_map(img, bands, sensor,
                                bundle["model"],
                                bundle["feature_names"])
        threshold = find_threshold_for_target(prob_map, target_km2)

        with rasterio.open(img) as src:
            prof = src.profile.copy()
            nd   = src.nodata if src.nodata is not None else 0
            data = src.read(1)

        valid  = (data != nd)
        binary = np.where(valid,
                          (prob_map>=threshold).astype(np.uint8), 255)
        km2    = float((binary==1).sum() * (cfg.TARGET_RES/1000)**2)

        p = prof.copy()
        p.update(count=1, dtype="uint8", nodata=255, compress="lzw")
        with rasterio.open(out_bin, "w", **p) as dst:
            dst.write(binary[np.newaxis])

        p.update(dtype="float32", nodata=-1)
        with rasterio.open(out_prob, "w", **p) as dst:
            dst.write(np.where(valid, prob_map, -1)
                      .astype(np.float32)[np.newaxis])

        log.info(f"  ✓ {km2:.1f} km² urban  "
                 f"(threshold={threshold:.4f})")
        return {"urban_km2": km2, "threshold": threshold}

    s1 = classify_epoch(al1985, cfg.LANDSAT_BANDS_1985, "L5",
                        b1985, area_1985,
                        cfg.RASTERS_DIR/"urban_1985.tif",
                        cfg.RASTERS_DIR/"prob_1985.tif", "1985")

    s2 = classify_epoch(al2023, cfg.LANDSAT_BANDS_2023, "L8",
                        b2023, area_2023,
                        cfg.RASTERS_DIR/"urban_2023.tif",
                        cfg.RASTERS_DIR/"prob_2023.tif", "2023")

    growth = s2["urban_km2"] - s1["urban_km2"]
    pct    = 100*growth/s1["urban_km2"] if s1["urban_km2"]>0 else 0

    log.info("\n══════════════════════════════════════════════")
    log.info("CLASSIFICATION RESULTS")
    log.info(f"  1985: {s1['urban_km2']:.1f} km²")
    log.info(f"  2023: {s2['urban_km2']:.1f} km²")
    log.info(f"  Growth: +{growth:.1f} km²  (+{pct:.0f}%)")
    log.info("══════════════════════════════════════════════")

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = mcolors.ListedColormap(["#d9ead3","#cc0000"])
    for ax, path, yr, s in zip(
        axes,
        [cfg.RASTERS_DIR/"urban_1985.tif",
         cfg.RASTERS_DIR/"urban_2023.tif"],
        [cfg.YEAR_HISTORICAL, cfg.YEAR_RECENT],
        [s1, s2]
    ):
        with rasterio.open(path) as src:
            d = src.read(1).astype(float)
            d[d==255] = np.nan
        ax.imshow(d, cmap=cmap, vmin=0, vmax=1,
                  origin="upper", interpolation="none")
        ax.set_title(f"Urban {yr}\n{s['urban_km2']:.1f} km²",
                     fontsize=13, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"{cfg.AOI_NAME} — Urban Classification\n"
                 f"Growth: +{growth:.1f} km²  (+{pct:.0f}%)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.MAPS_DIR/"urban_maps_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("✓ Stage 3 complete")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
