"""Stage 6 — CA-ANN Future Simulation (Rate-Constrained)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, rasterio, joblib, logging
from scipy.ndimage import uniform_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config as cfg
from scripts._utils import load_params, pixels_to_km2

log = logging.getLogger(__name__)


def load_r(p):
    with rasterio.open(p) as s:
        return s.read(1), s.profile.copy()


def norm(a, mask):
    o = a.astype(np.float32).copy()
    v = o[mask]
    if v.size == 0: return np.zeros_like(o)
    lo,hi = np.nanpercentile(v,2), np.nanpercentile(v,98)
    if np.isclose(lo,hi): return np.zeros_like(o)
    o = np.clip(o,lo,hi); o = (o-lo)/(hi-lo); o[~mask]=0.0
    return o.astype(np.float32)


def nbr(b, r):
    return uniform_filter(b.astype(np.float32),size=2*r+1,mode="nearest")


def feats(cur,du,dw,el,sl,mask):
    return np.stack([cur.astype(np.float32), nbr(cur,cfg.CA_NEIGHBORHOOD),
                     norm(du,mask), norm(dw,mask),
                     norm(el,mask), norm(sl,mask)], axis=-1)


def run():
    cfg.PREDS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load auto-computed growth rate from Stage 1
    params      = load_params(cfg.PIPELINE_DIR)
    annual_rate = params["annual_rate"]

    p85  = cfg.RASTERS_DIR/f"urban_{cfg.YEAR_HISTORICAL}_aligned.tif"
    p23  = cfg.RASTERS_DIR/f"urban_{cfg.YEAR_RECENT}_aligned.tif"
    p_du = cfg.RASTERS_DIR/"distance_to_urban_aligned.tif"
    p_dw = cfg.RASTERS_DIR/"distance_to_water_aligned.tif"
    p_el = cfg.RASTERS_DIR/"elevation_aligned.tif"
    p_sl = cfg.RASTERS_DIR/"slope_aligned.tif"

    log.info("══════════════════════════════════════════════")
    log.info(f"STAGE 6 — CA-ANN SIMULATION  [{cfg.AOI_NAME}]")
    log.info(f"  Growth rate: {annual_rate*100:.2f}%/year  "
             f"(auto-computed from GHSL)")
    log.info("══════════════════════════════════════════════")

    for p in [p85,p23,p_du,p_dw,p_el,p_sl]:
        if not p.exists():
            log.error(f"Missing: {p.name} — run Stage 4")
            sys.exit(1)

    u85,prof = load_r(p85)
    u23,_    = load_r(p23)
    du,_     = load_r(p_du)
    dw,_     = load_r(p_dw)
    el,_     = load_r(p_el)
    sl,_     = load_r(p_sl)

    u85  = np.where(u85==255,0,u85).astype(np.uint8)
    u23  = np.where(u23==255,0,u23).astype(np.uint8)
    mask = np.ones_like(u23, dtype=bool)

    F     = feats(u85,du,dw,el,sl,mask)
    trans = ((u85==0)&(u23==1)).astype(np.uint8)
    tm    = mask & ((trans==1)|((u85==0)&(u23==0)))
    X     = np.nan_to_num(F[tm], nan=0.0, posinf=1.0, neginf=0.0)
    y     = trans[tm]

    log.info(f"  Samples: {len(X):,}  "
             f"(grew={int(y.sum()):,}  stable={int((y==0).sum()):,})")

    Xtr,Xte,ytr,yte = train_test_split(
        X,y,test_size=0.2,stratify=y,random_state=cfg.RANDOM_STATE)

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(cfg.ANN_HIDDEN_LAYERS),
        learning_rate_init=cfg.ANN_LEARNING_RATE,
        batch_size=cfg.ANN_BATCH_SIZE,
        max_iter=cfg.ANN_EPOCHS,
        random_state=cfg.RANDOM_STATE,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10)

    log.info("  Training ANN ...")
    clf.fit(Xtr, ytr)
    yp  = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, yp)
    log.info(f"  ROC-AUC: {auc:.4f}")
    log.info("\n" + classification_report(
        yte,(yp>=cfg.CA_THRESHOLD).astype(np.uint8),
        target_names=["No transition","Transition"]))

    joblib.dump({"model":clf,"auc":auc,"annual_rate":annual_rate},
                cfg.MODELS_DIR/"ca_ann_model.pkl")

    # Rate-constrained simulation
    future_year = cfg.YEAR_RECENT + cfg.CA_ITERATIONS
    log.info(f"\n  Simulating {cfg.CA_ITERATIONS} years → {future_year}")
    log.info(f"  Rate: {annual_rate*100:.2f}%/year from GHSL")

    cur    = u23.copy()
    H,W    = cur.shape
    px_km2 = (cfg.TARGET_RES/1000)**2
    span   = cfg.YEAR_RECENT - cfg.YEAR_HISTORICAL
    combined = np.zeros((H,W), dtype=np.float32)

    for step in range(cfg.CA_ITERATIONS):
        F2       = feats(cur,du,dw,el,sl,mask)
        F2       = np.nan_to_num(F2,nan=0.0,posinf=1.0,neginf=0.0)
        pp       = clf.predict_proba(F2.reshape(-1,6))[:,1].reshape(H,W)
        pa       = 1.0 - np.power(np.clip(1.0-pp,0,1), 5.0/span)
        n        = nbr(cur,cfg.CA_NEIGHBORHOOD)
        combined = np.clip(pa*(0.6+0.8*n), 0, 1)

        # Rate constraint — use auto-computed GHSL rate
        cur_km2  = float((cur==1).sum() * px_km2)
        quota_px = int(cur_km2 * annual_rate / px_km2)
        cands    = mask & (cur==0)
        cprobs   = np.where(cands, combined, -1)

        if quota_px > 0 and cands.sum() > 0:
            flat     = cprobs.ravel()
            top_idx  = np.argpartition(flat, -quota_px)[-quota_px:]
            gf       = np.zeros(H*W, dtype=bool)
            gf[top_idx] = True
            grow     = gf.reshape(H,W) & cands
        else:
            grow = np.zeros((H,W), dtype=bool)

        cur[grow] = 1
        log.info(f"    Year {step+1}/{cfg.CA_ITERATIONS}  "
                 f"+{float(grow.sum()*px_km2):.1f} km²")

    a_now = float((u23==1).sum()*px_km2)
    a_fut = float((cur==1).sum()*px_km2)

    p2 = prof.copy()
    p2.update(count=1,dtype="uint8",compress="lzw",nodata=255)
    with rasterio.open(cfg.PREDS_DIR/f"urban_pred_{future_year}.tif",
                       "w",**p2) as dst:
        dst.write(cur.astype(np.uint8),1)

    p2.update(dtype="float32",nodata=-1.0)
    with rasterio.open(
        cfg.PREDS_DIR/f"transition_probability_{future_year}.tif",
        "w",**p2) as dst:
        dst.write(combined,1)

    # Figure
    fig,axes = plt.subplots(1,3,figsize=(18,6))
    cmap = mcolors.ListedColormap(["#d9ead3","#cc0000"])
    norm_b = mcolors.BoundaryNorm([-.5,.5,1.5],2)
    for ax,arr,title,km2 in [
        (axes[0],u23,f"Current {cfg.YEAR_RECENT}",a_now),
        (axes[1],cur,f"Predicted {future_year}",a_fut),
        (axes[2],combined,"Transition Probability",None),
    ]:
        if km2 is not None:
            vis = np.where(arr==255,np.nan,arr.astype(float))
            ax.imshow(vis,cmap=cmap,norm=norm_b,
                      origin="upper",interpolation="none")
            ax.set_title(f"{title}\n{km2:.1f} km²",
                         fontsize=11,fontweight="bold")
        else:
            im = ax.imshow(arr,cmap="YlOrRd",vmin=0,vmax=0.3,
                           origin="upper",interpolation="none")
            plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
            ax.set_title(title,fontsize=11,fontweight="bold")
        ax.axis("off")

    plt.suptitle(f"CA-ANN Urban Growth — {cfg.AOI_NAME}\n"
                 f"Growth {cfg.YEAR_RECENT}→{future_year}: "
                 f"+{a_fut-a_now:.1f} km²  "
                 f"(rate={annual_rate*100:.2f}%/yr)",
                 fontsize=13,fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.MAPS_DIR/"ca_ann_simulation.png",
                dpi=150,bbox_inches="tight")
    plt.close()

    log.info("\n══════════════════════════════════════════════")
    log.info("CA-ANN COMPLETE")
    log.info(f"  Urban {cfg.YEAR_RECENT}: {a_now:.1f} km²")
    log.info(f"  Urban {future_year}: {a_fut:.1f} km²")
    log.info(f"  Growth: +{a_fut-a_now:.1f} km²")
    log.info(f"  AUC: {auc:.4f}")
    log.info("══════════════════════════════════════════════")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
