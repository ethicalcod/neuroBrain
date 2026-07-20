
"""
Complete project setup.
"""

import shutil
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT


print("NeuroBrain Project Setup")

# Create required directories

required_dirs = [
    "docs",
    "figures",
    "models",
    "notebooks",
    "results",
    "scripts",
    "src",
]

print("Checking project structure...")

for folder in required_dirs:
    path = PROJECT_ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"{folder} Exists!")


# Install tree (if needed)

if shutil.which("tree") is None:
    print("\nInstalling tree...")
    subprocess.run(
        ["apt-get", "update", "-qq"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["apt-get", "install", "-y", "tree"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# Install Python requirements

requirements = PROJECT_ROOT / "requirements.txt"

if requirements.exists():
    print("Installing Python packages...")
    subprocess.run(
        ["pip", "install", "-q", "-r", str(requirements)],
        check=True,
    )

# Dataset setup

subprocess.run(
    [sys.executable, "scripts/setup_dataset.py"],
    check=True,
)

# Verification

subprocess.run(
    [sys.executable, "scripts/verify_project.py"],
    check=True,
)

# Show structure

print("\nProject structure:\n")

subprocess.run(
    ["tree", "-L", "2"],
    check=True,
)

print("\nSetup completed successfully.")
