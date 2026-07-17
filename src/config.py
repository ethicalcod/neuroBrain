
from pathlib import Path

# Project Configuration

PROJECT_ROOT = Path("/content/neuroBrain")

DATASET_ROOT = PROJECT_ROOT / "Task01_BrainTumour"

IMAGES_PATH = DATASET_ROOT / "imagesTr"
LABELS_PATH = DATASET_ROOT / "labelsTr"

FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

print("Configuration loaded successfully.")
