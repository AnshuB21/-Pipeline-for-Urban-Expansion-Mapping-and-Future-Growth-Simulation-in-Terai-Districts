"""
USER CONFIGURATION — Edit this file only
=========================================
This is the ONLY file you need to edit.
The pipeline handles everything else automatically.

Steps:
  1. Fill in your AOI details below
  2. Run: python run_pipeline.py
  3. All outputs go to /content/pipeline_outputs/<AOI_NAME>/
"""

# ─── Study area ───────────────────────────────────────────────
AOI_NAME = "Banke"   # Name of your district/area

# Draw your AOI in GEE using the polygon tool
# Right-click polygon → Copy → paste coordinates here
AOI_COORDS = [
    [81.5804462723946,  28.242744340900337],
    [81.5639667802071,  28.189499791620374],
    [81.49530222942585, 28.141072631100005],
    [81.48431590130085, 28.08777745763861 ],
    [81.55710032512897, 28.051424661121995],
    [81.62713816692585, 28.010209962841863],
    [81.70541575481647, 27.9883840287223  ],
    [81.73150828411335, 28.008997526928592],
    [81.71090891887897, 28.075661225512068],
    [81.7012958817696,  28.095046540669294],
    [81.7122822098946,  28.177395056237327],
    [81.76584055950397, 28.220965688839495],
    [81.68344309856647, 28.271775627550884],
    [81.64499095012897, 28.281450965813857],
]

# ─── Time periods ─────────────────────────────────────────────
YEAR_HISTORICAL = 1985   # Historical epoch (Landsat 5 era)
YEAR_RECENT     = 2023   # Recent epoch (Landsat 8/9 era)

# Landsat date ranges for compositing
# Historical: use 1988-1992 (earliest Landsat 5 coverage)
# Recent: use 2020-2023 (Landsat 8+9 merged)
DATE_HISTORICAL_START = "1988-01-01"
DATE_HISTORICAL_END   = "1992-12-31"
DATE_RECENT_START     = "2020-01-01"
DATE_RECENT_END       = "2023-12-31"

# ─── Google Drive folder for GEE exports ──────────────────────
# GEE will export files to this folder in your Google Drive
DRIVE_FOLDER = "UrbanPipeline"

# ─── That is all you need to fill in ─────────────────────────
# Everything below is computed automatically by the pipeline

# UTM zone is calculated from AOI centroid longitude
# For Nepal: EPSG:32644 (UTM 44N) or EPSG:32645 (UTM 45N)
# Pipeline detects this automatically

# GHSL thresholds are optimised automatically from time series
# Growth rate is computed from GHSL automatically
# Classification thresholds are auto-calibrated per AOI
