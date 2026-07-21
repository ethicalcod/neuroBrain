
"""
NeuroBrain Project Configuration

This module centralizes project paths and global configuration.
"""

from pathlib import Path

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Dataset

DATASET_NAME = "Task01_BrainTumour"

DATASET_FILE_ID = "1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU"

ARCHIVE_NAME = "Task01_BrainTumour.tar"

DATASET_ROOT = PROJECT_ROOT / DATASET_NAME

IMAGES_PATH = DATASET_ROOT / "imagesTr"
LABELS_PATH = DATASET_ROOT / "labelsTr"
TEST_IMAGES_PATH = DATASET_ROOT / "imagesTs"

DATASET_JSON = DATASET_ROOT / "dataset.json"

RANDOM_SEED = 42

# Training Placeholders

BATCH_SIZE = 2
NUM_WORKERS = 2

# Visualization

FIGURE_DPI = 300
