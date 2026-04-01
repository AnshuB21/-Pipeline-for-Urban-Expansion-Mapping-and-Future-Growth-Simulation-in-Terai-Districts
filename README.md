# Urban Expansion Analysis Pipeline (Banke District)

This repository contains a modular Python pipeline for processing satellite imagery, classifying urban land cover, and simulating future urban growth using a CA-ANN (Cellular Automata - Artificial Neural Network) model.

# Getting Started

1. Environment Setup
   Ensure your Google Drive is mounted and the required dependencies are installed:

Core Libraries: earthengine-api, rasterio, geopandas, scikit-learn, scipy.

Hardware: A GPU backend is recommended for Stage 02 (Model Training).

2. Data Preparation
   Run the File Setup cell in the notebook to copy your .tif files from GeoTiffs or MyDrive into the local /content/pipeline_outputs/Banke/raw directory.

# Operation Steps (The Pipeline)

You can run the entire workflow by calling !python run_pipeline.py. If you need to debug or run specific parts, follow this sequence:

Stage 1: Pre-processing & Labeling
Script: 01_create_labels.py

What it does: Uses GHSL (Global Human Settlement Layer) and terrain slope to automatically determine the urban threshold for your area of interest (AOI). It generates the ground-truth labels needed for training.

Stage 2: Classifier Training
Script: 02_train_classifier.py

What it does: Trains a Scikit-Learn classifier (Random Forest) on Landsat spectral bands to distinguish between "Urban" and "Non-Urban" pixels.

Output: Generates training_Epoch.png plots to show model convergence.

Stage 3: Land Cover Mapping
Script: 03_apply_classifier.py

What it does: Applies the trained model to the full Landsat scenes for 1985 and 2023.

Output: Two classified rasters representing urban extent at both time points.

Stage 4: Spatial Alignment & Change Detection
Scripts: 04_align_rasters.py & 05_change_detection.py

What it does: Ensures all rasters have the exact same dimensions/projection and calculates the "Change Map" (where growth occurred).

Stage 5: Future Simulation (CA-ANN)
Script: 06_ca_ann_model.py

What it does: Uses a Neural Network to learn the transition potential based on drivers (distance to water, distance to roads, slope, elevation). It then runs a Cellular Automata simulation to predict urban extent for 2033.

Stage 6: Validation
Script: 07_validation.py

What it does: Calculates the Accuracy and AUC (Area Under Curve) of the predictions against historical data.

# Visualizing Results

After running the pipeline, the following maps are generated in /outputs/maps:

urban_expansion_map.png: Shows 1985 vs. 2023 growth.

urban_heatmaps.png: Kernel density estimation of urban centers.

growth_direction_rose.png: A rose diagram showing the dominant geographic direction of expansion (e.g., SW dominant for Banke).

📂 Directory Structure
/scripts: The core Python modules.

/pipeline_outputs/[AOI]/raw: Your input .tif files.

/pipeline_outputs/[AOI]/rasters: Aligned intermediate rasters.

/pipeline_outputs/[AOI]/outputs: Final maps and prediction files.

# To get visual reports in your collab

from IPython.display import Image, display
from pathlib import Path

MAPS = Path("/content/pipeline_outputs/Banke/outputs/maps")
PREDS = Path("/content/pipeline_outputs/Banke/outputs/predictions")

# Find all PNG files

all_images = sorted(MAPS.glob("\*.png"))

print(f"Found {len(all_images)} images:\n")

for img_path in all_images:
print(f"── {img_path.name}")
display(Image(filename=str(img_path), width=800))
print()
