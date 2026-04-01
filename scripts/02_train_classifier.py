"""
Stage 2 — Train Epoch-Matched Classifiers
==========================================
Trains classifiers specific to this AOI's spectral environment.
Loads thresholds computed automatically by Stage 1.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import joblib
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score)
import rasterio

import config as cfg
from scripts._utils import (load_params, compute_indices, pixels_to_km2)

log = logging.getLogger(__name__)


def load_training_data(img_path, label_path, band_names,
                       sensor, n_urban, n_nonurban):
    with rasterio.open(img_path) as src:
        data   = src.read().astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999

    with rasterio.open(label_path) as src:
        labels = src.read(1)

    if data.shape[1:] != labels.shape:
        log.error(f"Shape mismatch: image={data.shape[1:]} "
                  f"labels={labels.shape}")
        log.error("Re-run Stage 1 to regenerate aligned labels")
        sys.exit(1)

    bands      = {name: data[i] for i, name in
                  enumerate(band_names[:data.shape[0]])}
    idx        = compute_indices(bands, sensor)
    feat_names = list(bands.keys()) + list(idx.keys())
    stack      = np.stack(list(bands.values()) +
                          list(idx.values()), axis=-1)
    X_all      = stack.reshape(-1, len(feat_names))
    y_all      = labels.ravel()
    valid      = np.all(data != nodata, axis=0).ravel()
    keep       = valid & np.isin(y_all, [0, 1])
    X, y       = X_all[keep], y_all[keep]

    n_u  = int((y==1).sum())
    n_nu = int((y==0).sum())
    log.info(f"  Valid: {keep.sum():,}  "
             f"(urban={n_u:,}  non-urban={n_nu:,})")

    rng    = np.random.RandomState(cfg.RANDOM_STATE)
    n_u_s  = min(n_urban,    n_u)
    n_nu_s = min(n_nonurban, n_nu)
    sel    = np.concatenate([
        rng.choice(np.where(y==1)[0], n_u_s,  replace=False),
        rng.choice(np.where(y==0)[0], n_nu_s, replace=False),
    ])
    rng.shuffle(sel)
    log.info(f"  Sample: {len(sel):,}  "
             f"(urban={n_u_s:,}  non-urban={n_nu_s:,})")
    return X[sel], y[sel], feat_names


def train_save(X, y, feat_names, class_weight,
               model_name, epoch_label):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.TEST_SIZE,
        stratify=y, random_state=cfg.RANDOM_STATE)

    clf = RandomForestClassifier(
        n_estimators=cfg.N_ESTIMATORS,
        max_depth=cfg.MAX_DEPTH,
        n_jobs=-1,
        random_state=cfg.RANDOM_STATE,
        class_weight=class_weight,
    )
    clf.fit(X_tr, y_tr)

    cv     = cross_val_score(clf, X_tr, y_tr, cv=5,
                             scoring="f1", n_jobs=-1)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)
    oa     = np.mean(y_pred == y_te)

    log.info(f"  CV F1: {cv.mean():.4f} ± {cv.std():.4f}")
    log.info(f"  OA:    {oa*100:.2f}%   AUC: {auc:.4f}")
    log.info(f"\n{classification_report(y_te, y_pred, digits=4, target_names=['Non-urban','Urban'])}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_te, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Non-urban","Urban"]
                           ).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Confusion Matrix — {epoch_label}",
                      fontweight="bold")
    imp = dict(zip(feat_names, clf.feature_importances_))
    si  = sorted(imp.items(), key=lambda x: x[1])
    axes[1].barh([k for k,v in si], [v for k,v in si], color="steelblue")
    axes[1].set_title(f"Feature Importance — {epoch_label}",
                      fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.MAPS_DIR /
                f"training_{epoch_label.replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump({"model": clf, "feature_names": feat_names,
                 "oa": oa, "auc": auc},
                cfg.MODELS_DIR / f"{model_name}.pkl")
    log.info(f"  Saved → {model_name}.pkl")
    return {"OA": oa, "AUC": auc, "CV_F1": cv.mean()}


def run():
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MAPS_DIR.mkdir(parents=True, exist_ok=True)

    label_1985 = cfg.LABELS_DIR / "urban_labels_1985_aligned.tif"
    label_2023 = cfg.LABELS_DIR / "urban_labels_2023_aligned.tif"

    for p, name in [
        (cfg.LANDSAT_1985_PATH, cfg.LANDSAT_1985_PATH.name),
        (cfg.LANDSAT_2023_PATH, cfg.LANDSAT_2023_PATH.name),
        (label_1985,            label_1985.name),
        (label_2023,            label_2023.name),
    ]:
        if not p.exists():
            log.error(f"Missing: {name}")
            sys.exit(1)

    log.info("══════════════════════════════════════════════")
    log.info(f"STAGE 2 — TRAIN CLASSIFIERS  [{cfg.AOI_NAME}]")
    log.info("══════════════════════════════════════════════")

    log.info("\n── Classifier 1: 1985 ────────────────────────")
    X1, y1, f1 = load_training_data(
        cfg.LANDSAT_1985_PATH, label_1985,
        cfg.LANDSAT_BANDS_1985, "L5",
        cfg.N_URBAN_SAMPLES, cfg.N_NONURBAN_SAMPLES)
    s1 = train_save(X1, y1, f1, cfg.CLASS_WEIGHT_1985,
                    "classifier_1985", "1985 Epoch")

    log.info("\n── Classifier 2: 2023 ────────────────────────")
    X2, y2, f2 = load_training_data(
        cfg.LANDSAT_2023_PATH, label_2023,
        cfg.LANDSAT_BANDS_2023, "L8",
        cfg.N_URBAN_SAMPLES, cfg.N_NONURBAN_SAMPLES)
    s2 = train_save(X2, y2, f2, cfg.CLASS_WEIGHT_2023,
                    "classifier_2023", "2023 Epoch")

    import shutil
    shutil.copy(cfg.MODELS_DIR / "classifier_2023.pkl",
                cfg.MODELS_DIR / "classifier.pkl")

    log.info("\n══════════════════════════════════════════════")
    log.info("TRAINING COMPLETE")
    log.info(f"  1985  F1={s1['CV_F1']:.4f}  AUC={s1['AUC']:.4f}")
    log.info(f"  2023  F1={s2['CV_F1']:.4f}  AUC={s2['AUC']:.4f}")
    log.info("══════════════════════════════════════════════")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
