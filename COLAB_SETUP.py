"""
COLAB_SETUP.py — Run this first in Google Colab
================================================
Sets up everything needed to run the pipeline.

Run this cell once at the start of every Colab session:
  exec(open('COLAB_SETUP.py').read())

Or in a notebook cell:
  !python COLAB_SETUP.py
"""

import subprocess
import sys
import os
from pathlib import Path

print("═" * 55)
print("  URBAN EXPANSION PIPELINE — COLAB SETUP")
print("═" * 55)

# ── Step 1: Install dependencies ──────────────────────────────
print("\n── Step 1: Installing dependencies ──────────────")
packages = [
    "earthengine-api",
    "rasterio",
    "geopandas",
    "scikit-learn",
    "scipy",
    "joblib",
    "matplotlib",
]

for pkg in packages:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        capture_output=True, text=True
    )
    print(f"  ✅ {pkg}")

# ── Step 2: Mount Google Drive ────────────────────────────────
print("\n── Step 2: Mount Google Drive ────────────────────")
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("  ✅ Google Drive mounted")
except Exception as e:
    print(f"  ⚠️  Drive mount: {e}")

# ── Step 3: Authenticate GEE ──────────────────────────────────
print("\n── Step 3: Google Earth Engine ───────────────────")
try:
    import ee
    try:
        ee.Initialize(project="ee-default")
        print("  ✅ GEE already authenticated")
    except Exception:
        print("  Running GEE authentication ...")
        ee.Authenticate()
        ee.Initialize(project="ee-default")
        print("  ✅ GEE authenticated")
except ImportError:
    print("  ❌ earthengine-api not installed")

# ── Step 4: Set working directory ────────────────────────────
print("\n── Step 4: Working directory ─────────────────────")
pipeline_dir = Path("/content/final_pipeline")
if pipeline_dir.exists():
    os.chdir(pipeline_dir)
    print(f"  ✅ Working directory: {pipeline_dir}")
else:
    print(f"  ⚠️  {pipeline_dir} not found")
    print("  Unzip the pipeline first:")
    print("  import zipfile")
    print("  with zipfile.ZipFile('final_pipeline.zip') as z:")
    print("      z.extractall('/content/')")

# ── Step 5: Show current AOI ──────────────────────────────────
print("\n── Step 5: Current configuration ────────────────")
try:
    sys.path.insert(0, str(pipeline_dir))
    from user_config import AOI_NAME, AOI_COORDS, YEAR_HISTORICAL, YEAR_RECENT
    import numpy as np
    lons = [c[0] for c in AOI_COORDS]
    lats = [c[1] for c in AOI_COORDS]
    print(f"  AOI:    {AOI_NAME}")
    print(f"  Centre: {np.mean(lons):.4f}°E  {np.mean(lats):.4f}°N")
    print(f"  Years:  {YEAR_HISTORICAL} → {YEAR_RECENT}")
    print()
    print("  To change AOI: edit user_config.py")
except Exception as e:
    print(f"  ⚠️  Could not read user_config.py: {e}")

print()
print("═" * 55)
print("  SETUP COMPLETE")
print("  Run the pipeline:")
print("  !python run_pipeline.py            # full pipeline")
print("  !python run_pipeline.py --from-stage 1  # skip GEE")
print("  !python run_pipeline.py --info     # show AOI info")
print("═" * 55)
