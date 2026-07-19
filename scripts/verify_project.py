"""
Verify NeuroBrain project structure.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import *

checks = {
    "Project Root": PROJECT_ROOT,
    "Dataset": DATASET_ROOT,
    "imagesTr": IMAGES_PATH,
    "labelsTr": LABELS_PATH,
    "dataset.json": DATASET_JSON,
    "docs": DOCS_DIR,
    "figures": FIGURES_DIR,
    "models": MODELS_DIR,
    "notebooks": NOTEBOOKS_DIR,
    "results": RESULTS_DIR,
    "scripts": SCRIPTS_DIR,
    "src": SRC_DIR,
}

print("NeuroBrain Project Verification")

failed = False

for name, path in checks.items():
    if path.exists():
        print(f"{name} Exists!")
    else:
        print(f"{name} Doesn't exist!")
        failed = True

if failed:
    print("Verification failed.")
else:
    print(" Everything looks good!")
