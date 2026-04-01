import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

RASTERS  = Path("/content/pipeline_outputs/Banke/rasters")
OUTPUTS  = Path("/content/pipeline_outputs/Banke/outputs")
MAPS     = Path("/content/pipeline_outputs/Banke/outputs/maps")
PREDS    = Path("/content/pipeline_outputs/Banke/outputs/predictions")
MAPS.mkdir(parents=True, exist_ok=True)

def load(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def heatmap(data, nodata=255):
    """Convert binary raster to kernel density heatmap."""
    binary = np.where(data == nodata, 0, data).astype(np.float32)
    # Apply Gaussian kernel — sigma controls smoothness
    density = gaussian_filter(binary, sigma=15)
    # Mask outside AOI
    density[data == nodata] = np.nan
    return density

# Load all rasters
u1985    = load(RASTERS / "urban_1985_aligned.tif")
u2023    = load(RASTERS / "urban_2023_aligned.tif")
change   = load(OUTPUTS / "change_map.tif")
try:
    future = load(PREDS / "urban_pred_2033.tif")
    has_future = True
except:
    has_future = False

# Compute heatmaps
h1985   = heatmap(u1985)
h2023   = heatmap(u2023)

# Growth hotspot — only new urban pixels
new_urban = np.where((u1985==0)&(u2023==1)&(change!=255),
                     1.0, 0.0)
h_growth  = gaussian_filter(new_urban, sigma=15)
h_growth[change==255] = np.nan

if has_future:
    h_future = heatmap(future)

# ── Plot ──────────────────────────────────────────────────────
n_panels = 4 if has_future else 3
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.ravel()

cmap = "RdYlGn_r"   # matches the image you shared — blue→green→yellow→red

panels = [
    (h1985,   "Urban Density 1985"),
    (h2023,   "Urban Density 2023"),
    (h_growth,"Growth Hotspots 1985→2023"),
]
if has_future:
    panels.append((h_future, "Predicted Urban 2033"))

for ax, (data, title) in zip(axes, panels):
    # Mask zeros for cleaner look
    masked = np.ma.masked_where(
        (data == 0) | np.isnan(data), data)
    im = ax.imshow(masked, cmap=cmap,
                   origin="upper", interpolation="bilinear",
                   vmin=np.nanpercentile(data[data>0], 5)
                       if data[data>0].size > 0 else 0,
                   vmax=np.nanpercentile(data[data>0], 99)
                       if data[data>0].size > 0 else 1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Density")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

# Hide unused panel if only 3
if not has_future:
    axes[3].axis("off")

plt.suptitle("Urban Expansion Heatmaps — Banke District, Nepal",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
out = MAPS / "urban_heatmaps.png"
plt.savefig(out, dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print(f" Saved → {out}")