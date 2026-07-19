"""
Complete project setup.
"""

import shutil
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import *


print("NeuroBrain Project Setup")

# Install tree (if needed)

if shutil.which("tree") is None:
    print("Installing tree...")
    subprocess.run(
        ["apt-get", "update"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["apt-get", "install", "-y", "tree"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

# Install Python requirements

requirements = Path("requirements.txt")

if requirements.exists():
    print("Installing Python packages...")
    subprocess.run(
        ["pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )

# Dataset setup

subprocess.run(
    ["python", "scripts/setup_dataset.py"],
    check=True,
)

# Verification

subprocess.run(
    ["python", "scripts/verify_project.py"],
    check=True,
)

# Show structure

print("\nProject structure:\n")

subprocess.run(
    ["tree", "-L", "2"],
    check=True,
)

print("\nSetup completed successfully.")
