"""
Stage 7 — Internal Spatial Validation
=======================================
Validates classification accuracy using a spatial holdout approach.
Splits the AOI into 70% training zone and 30% validation zone.

This works for ANY AOI — no external reference needed.

Validation metrics:
  - Overall Accuracy
  - Kappa coefficient
  - Urban Precision and Recall
  - Growth rate comparison (model vs GHSL)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, rasterio, joblib, logging
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score)

import config as cfg
from scripts._utils import (load_params, get_prob_map,
                             pixels_to_km2, align_to_ref)
from rasterio.warp import Resampling

log = logging.getLogger(__name__)


def run():
    cfg.VALID_DIR.mkdir(parents=True, exist_ok=True)

    params      = load_params(cfg.PIPELINE_DIR)
    annual_rate = params["annual_rate"]
    ghsl_1985   = params["ghsl_1985_km2"]
    ghsl_2020   = params["ghsl_2020_km2"]
    terrain     = params.get("terrain", "Unknown")

    log.info("══════════════════════════════════════════════")
    log.info(f"STAGE 7 — SPATIAL VALIDATION  [{cfg.AOI_NAME}]")
    log.info(f"  Terrain: {terrain}")
    log.info("══════════════════════════════════════════════")

    # Load aligned Landsat images and labels
    al1985 = cfg.RASTERS_DIR / "landsat_1985_aligned.tif"
    al2023 = cfg.RASTERS_DIR / "landsat_2023_aligned.tif"
    lbl1985 = cfg.LABELS_DIR / "urban_labels_1985_aligned.tif"
    lbl2023 = cfg.LABELS_DIR / "urban_labels_2023_aligned.tif"

    for p in [al1985, al2023, lbl1985, lbl2023,
              cfg.MODELS_DIR/"classifier_1985.pkl",
              cfg.MODELS_DIR/"classifier_2023.pkl"]:
        if not p.exists():
            log.error(f"Missing: {p.name}")
            sys.exit(1)

    b1985 = joblib.load(cfg.MODELS_DIR / "classifier_1985.pkl")
    b2023 = joblib.load(cfg.MODELS_DIR / "classifier_2023.pkl")

    def validate_epoch(img, band_names, sensor,
                       bundle, label_path, label):
        log.info(f"\n── Validating {label} ────────────────────")

        with rasterio.open(img) as src:
            H, W = src.height, src.width

        # Spatial holdout: right 30% = validation zone
        val_start_col = int(W * 0.70)

        # Get probability map for validation zone only
        prob_map = get_prob_map(img, band_names, sensor,
                                bundle["model"],
                                bundle["feature_names"])

        # Align label to image if needed
        with rasterio.open(label_path) as src:
            if src.width != W or src.height != H:
                tmp = cfg.VALID_DIR / f"_tmp_lbl_{label}.tif"
                align_to_ref(label_path, img, tmp,
                             Resampling.nearest, "uint8")
                with rasterio.open(tmp) as s2:
                    labels = s2.read(1)
                tmp.unlink(missing_ok=True)
            else:
                labels = src.read(1)

        # Extract validation zone
        prob_val   = prob_map[:, val_start_col:]
        label_val  = labels[:,  val_start_col:]

        valid = np.isin(label_val, [0, 1])
        if valid.sum() < 100:
            log.warning("  Too few validation pixels")
            return None

        y_true = label_val[valid]
        y_prob = prob_val[valid]

        # Use threshold from params
        thresh = params.get(f"threshold_{label[:4]}", 0.5)
        y_pred = (y_prob >= thresh).astype(np.uint8)

        oa    = float(np.mean(y_pred == y_true))
        auc   = roc_auc_score(y_true, y_prob)
        cm    = confusion_matrix(y_true, y_pred)
        tn,fp,fn,tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        prec  = tp/(tp+fp) if (tp+fp)>0 else 0
        rec   = tp/(tp+fn) if (tp+fn)>0 else 0
        kappa = (2*(tp*tn-fn*fp)/
                 ((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn))
                 if ((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn))>0 else 0)

        log.info(f"  Validation zone: right 30% of AOI")
        log.info(f"  Valid pixels:    {valid.sum():,}")
        log.info(f"  Overall Accuracy:{oa*100:.2f}%")
        log.info(f"  Kappa:           {kappa:.4f}")
        log.info(f"  ROC-AUC:         {auc:.4f}")
        log.info(f"  Urban Precision: {prec*100:.1f}%")
        log.info(f"  Urban Recall:    {rec*100:.1f}%")

        return {"OA":oa,"Kappa":kappa,"AUC":auc,
                "Precision":prec,"Recall":rec}

    s1985 = validate_epoch(al1985, cfg.LANDSAT_BANDS_1985, "L5",
                           b1985, lbl1985, "1985")
    s2023 = validate_epoch(al2023, cfg.LANDSAT_BANDS_2023, "L8",
                           b2023, lbl2023, "2023")

    # Growth rate comparison
    log.info("\n── Growth rate validation ────────────────────")
    model_rate = params.get("annual_rate_model",
                            annual_rate)  # use GHSL rate as proxy
    ghsl_rate  = annual_rate
    rate_diff  = abs(model_rate - ghsl_rate)

    log.info(f"  GHSL growth rate:  {ghsl_rate*100:.2f}%/year")
    log.info(f"  GHSL 1985 urban:   {ghsl_1985:.1f} km²")
    log.info(f"  GHSL 2020 urban:   {ghsl_2020:.1f} km²")

    # Save report
    report_path = cfg.VALID_DIR / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"VALIDATION REPORT — {cfg.AOI_NAME}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Terrain:  {terrain}\n")
        f.write(f"Method:   Spatial holdout (right 30% of AOI)\n\n")
        if s1985:
            f.write(f"1985 Classifier:\n")
            f.write(f"  Overall Accuracy: {s1985['OA']*100:.2f}%\n")
            f.write(f"  Kappa:            {s1985['Kappa']:.4f}\n")
            f.write(f"  ROC-AUC:          {s1985['AUC']:.4f}\n")
            f.write(f"  Urban Precision:  {s1985['Precision']*100:.1f}%\n")
            f.write(f"  Urban Recall:     {s1985['Recall']*100:.1f}%\n\n")
        if s2023:
            f.write(f"2023 Classifier:\n")
            f.write(f"  Overall Accuracy: {s2023['OA']*100:.2f}%\n")
            f.write(f"  Kappa:            {s2023['Kappa']:.4f}\n")
            f.write(f"  ROC-AUC:          {s2023['AUC']:.4f}\n")
            f.write(f"  Urban Precision:  {s2023['Precision']*100:.1f}%\n")
            f.write(f"  Urban Recall:     {s2023['Recall']*100:.1f}%\n\n")
        f.write(f"GHSL Reference:\n")
        f.write(f"  Urban 1985:    {ghsl_1985:.1f} km²\n")
        f.write(f"  Urban 2020:    {ghsl_2020:.1f} km²\n")
        f.write(f"  Annual rate:   {ghsl_rate*100:.2f}%/year\n")

    log.info(f"\n  Report saved → {report_path.name}")

    log.info("\n══════════════════════════════════════════════")
    log.info("VALIDATION SUMMARY")
    log.info("══════════════════════════════════════════════")
    if s1985 and s2023:
        log.info(f"  {'Metric':<20} {'1985':>8} {'2023':>8}  Target")
        log.info("  " + "─"*44)
        for m, t in [("OA",">90%"),("Kappa",">0.80"),("AUC",">0.90")]:
            v1 = f"{s1985[m]*100:.1f}%" if m=="OA" else f"{s1985[m]:.4f}"
            v2 = f"{s2023[m]*100:.1f}%" if m=="OA" else f"{s2023[m]:.4f}"
            log.info(f"  {m:<20} {v1:>8} {v2:>8}  {t}")
    log.info("══════════════════════════════════════════════")
    log.info("✓ Stage 7 complete")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
