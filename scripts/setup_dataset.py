"""
Download and prepare the MSD Brain Tumour dataset.
"""

import shutil
import subprocess
import tarfile

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (
    PROJECT_ROOT,
    DATASET_ROOT,
    DATASET_FILE_ID,
    ARCHIVE_NAME,
)

ARCHIVE = PROJECT_ROOT / ARCHIVE_NAME


def dataset_exists():
    """Return True if the dataset is already available."""
    return (
        DATASET_ROOT.exists()
        and (DATASET_ROOT / "imagesTr").exists()
        and (DATASET_ROOT / "labelsTr").exists()
        and (DATASET_ROOT / "dataset.json").exists()
    )


if dataset_exists():
    print(" Dataset already exists.")
    print(DATASET_ROOT)
    raise SystemExit

print("Downloading Brain Tumour Dataset")

# Install gdown only if missing
if shutil.which("gdown") is None:
    subprocess.run(["pip", "install", "-q", "gdown"], check=True)

# Download archive
subprocess.run(
    [
        "gdown",
        "--id",
        DATASET_FILE_ID,
        "-O",
        str(ARCHIVE),
    ],
    check=True,
)

print("Download complete.")

# Extract
print("Extracting archive...")

with tarfile.open(ARCHIVE) as tar:
    tar.extractall(PROJECT_ROOT)

print("Extraction complete.")

# Remove archive
ARCHIVE.unlink()

print("Archive removed.")

print("\n Dataset setup completed successfully.")
