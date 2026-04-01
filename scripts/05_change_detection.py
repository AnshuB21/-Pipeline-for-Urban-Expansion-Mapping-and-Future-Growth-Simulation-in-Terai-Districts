"""Stage 5 — Change Detection"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, rasterio, logging, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import config as cfg
from scripts._utils import load_params, pixels_to_km2

log = logging.getLogger(__name__)

CLASSES = {
    0: ("Stable Non-urban", "#d9ead3"),
    1: ("Stable Urban",     "#f6b26b"),
    2: ("New Urban Growth", "#cc0000"),
    3: ("Urban Loss",       "#6fa8dc"),
}


def run():
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MAPS_DIR.mkdir(parents=True, exist_ok=True)

    p1985 = cfg.RASTERS_DIR / f"urban_{cfg.YEAR_HISTORICAL}_aligned.tif"
    p2023 = cfg.RASTERS_DIR / f"urban_{cfg.YEAR_RECENT}_aligned.tif"
    for p in [p1985, p2023]:
        if not p.exists():
            log.error(f"Missing: {p.name} — run Stage 4")
            sys.exit(1)

    params      = load_params(cfg.PIPELINE_DIR)
    annual_rate = params["annual_rate"]

    log.info("══════════════════════════════════════════════")
    log.info(f"STAGE 5 — CHANGE DETECTION  [{cfg.AOI_NAME}]")
    log.info("══════════════════════════════════════════════")

    with rasterio.open(p1985) as s:
        old  = s.read(1);  profile = s.profile.copy()
        nd   = s.nodata if s.nodata is not None else 255
    with rasterio.open(p2023) as s:
        new  = s.read(1)

    valid  = (old!=nd) & (new!=nd)
    change = np.full_like(old, 255, dtype=np.uint8)
    change[valid&(old==0)&(new==0)] = 0
    change[valid&(old==1)&(new==1)] = 1
    change[valid&(old==0)&(new==1)] = 2
    change[valid&(old==1)&(new==0)] = 3

    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(cfg.OUTPUTS_DIR/"change_map.tif","w",**profile) as dst:
        dst.write(change[np.newaxis])

    stats   = {v: {"label":l, "km2": pixels_to_km2(int((change==v).sum()))}
               for v,(l,_) in CLASSES.items()}
    old_km2 = pixels_to_km2(int((old[valid]==1).sum()))
    new_km2 = pixels_to_km2(int((new[valid]==1).sum()))
    g_km2   = stats[2]["km2"]
    g_pct   = g_km2/old_km2*100 if old_km2>0 else 0
    a_pct   = g_pct/(cfg.YEAR_RECENT-cfg.YEAR_HISTORICAL)

    log.info(f"\n  Urban {cfg.YEAR_HISTORICAL}: {old_km2:.1f} km²")
    log.info(f"  Urban {cfg.YEAR_RECENT}: {new_km2:.1f} km²")
    log.info(f"  Growth: +{g_km2:.1f} km²  (+{g_pct:.1f}%)")
    log.info(f"  Rate:   {a_pct:.2f}%/year  "
             f"(GHSL: {annual_rate*100:.2f}%/year)")

    # CSV
    with open(cfg.OUTPUTS_DIR/"change_statistics.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Class","Label","Area_km2"])
        for v,s in stats.items():
            w.writerow([v, s["label"], f"{s['km2']:.2f}"])
        w.writerow([])
        w.writerow(["Metric","Value"])
        w.writerow([f"Urban_{cfg.YEAR_HISTORICAL}_km2", f"{old_km2:.2f}"])
        w.writerow([f"Urban_{cfg.YEAR_RECENT}_km2",     f"{new_km2:.2f}"])
        w.writerow(["Growth_km2",      f"{g_km2:.2f}"])
        w.writerow(["Growth_pct",      f"{g_pct:.2f}"])
        w.writerow(["Annual_rate_pct", f"{a_pct:.2f}"])
        w.writerow(["GHSL_rate_pct",   f"{annual_rate*100:.2f}"])

    # Change map figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cb = mcolors.ListedColormap(["#d9ead3","#cc0000"])
    nb = mcolors.BoundaryNorm([-.5,.5,1.5], 2)
    cc = mcolors.ListedColormap([c for _,(_,c) in CLASSES.items()])
    nc = mcolors.BoundaryNorm([-.5,.5,1.5,2.5,3.5], 4)

    for ax,arr,title in [
        (axes[0], np.where(old==255,np.nan,old.astype(float)),
         f"Urban {cfg.YEAR_HISTORICAL}  ({old_km2:.1f} km²)"),
        (axes[1], np.where(new==255,np.nan,new.astype(float)),
         f"Urban {cfg.YEAR_RECENT}  ({new_km2:.1f} km²)"),
    ]:
        ax.imshow(arr,cmap=cb,norm=nb,origin="upper",interpolation="none")
        ax.set_title(title,fontsize=12,fontweight="bold")
        ax.axis("off")

    axes[2].imshow(np.where(change==255,np.nan,change.astype(float)),
                   cmap=cc,norm=nc,origin="upper",interpolation="none")
    axes[2].set_title(f"Change {cfg.YEAR_HISTORICAL}→{cfg.YEAR_RECENT}",
                      fontsize=12,fontweight="bold")
    axes[2].axis("off")
    axes[2].legend(handles=[
        mpatches.Patch(color=c,
                       label=f"{l} ({stats[v]['km2']:.1f} km²)")
        for v,(l,c) in CLASSES.items()],
        loc="lower right", fontsize=8)

    fig.text(0.01,0.01,
             f"{cfg.AOI_NAME}\n"
             f"{cfg.YEAR_HISTORICAL}: {old_km2:.1f} km²  "
             f"{cfg.YEAR_RECENT}: {new_km2:.1f} km²\n"
             f"Growth: +{g_km2:.1f} km² (+{g_pct:.0f}%)\n"
             f"Rate: {a_pct:.2f}%/year",
             fontsize=9,
             bbox=dict(boxstyle="round",fc="lightyellow",alpha=0.9))
    plt.suptitle(f"Urban Expansion — {cfg.AOI_NAME} "
                 f"{cfg.YEAR_HISTORICAL}→{cfg.YEAR_RECENT}",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.MAPS_DIR/"urban_expansion_map.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Direction rose
    H,W  = change.shape
    cy,cx = H//2, W//2
    dirs = {"N":change[:cy,cx:cx+1],"NE":change[:cy,cx:],
            "E":change[cy:cy+1,cx:],"SE":change[cy:,cx:],
            "S":change[cy:,cx:cx+1],"SW":change[cy:,:cx],
            "W":change[cy:cy+1,:cx],"NW":change[:cy,:cx]}
    dkm2   = {d: pixels_to_km2(int((a==2).sum()))
               for d,a in dirs.items()}
    order  = ["N","NE","E","SE","S","SW","W","NW"]
    vals   = [dkm2[d] for d in order]
    angles = np.linspace(0,2*np.pi,8,endpoint=False)
    fig,ax = plt.subplots(subplot_kw={"projection":"polar"},figsize=(7,7))
    ax.bar(angles,vals,width=2*np.pi/8*0.8,bottom=0,
           color=["#cc0000" if v>0 else "#e0e0e0" for v in vals],
           alpha=0.85,edgecolor="white")
    ax.set_xticks(angles); ax.set_xticklabels(order,fontsize=11)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_title(f"Growth Direction — {cfg.AOI_NAME}\n"
                 f"{cfg.YEAR_HISTORICAL}→{cfg.YEAR_RECENT}",
                 fontsize=13,fontweight="bold",pad=20)
    for ang,val,lbl in zip(angles,vals,order):
        if val>0:
            ax.text(ang,val+max(vals)*0.05,f"{val:.0f}",
                    ha="center",va="bottom",fontsize=9,fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.MAPS_DIR/"growth_direction_rose.png",
                dpi=150,bbox_inches="tight")
    plt.close()

    log.info(f"\n  Dominant direction: "
             f"{max(dkm2,key=dkm2.get)}  "
             f"({max(dkm2.values()):.1f} km²)")
    log.info("✓ Stage 5 complete")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
