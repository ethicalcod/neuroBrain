"""
Dataset utilities for NeuroBrain.

Provides helper functions for loading, validating,
and organizing the Medical Segmentation Decathlon
Brain Tumour dataset.
"""

from pathlib import Path

from src.config import IMAGES_PATH, LABELS_PATH


def _valid_nifti_files(directory: Path):
    """
    Return valid BraTS NIfTI files only.
    """

    files = []

    for file in sorted(directory.glob("*.nii.gz")):

        if file.name.startswith("."):
            continue

        if not file.name.startswith("BRATS_"):
            continue

        files.append(file)

    return files


def get_image_files():
    """Return all MRI image files."""
    return _valid_nifti_files(IMAGES_PATH)


def get_label_files():
    """Return all segmentation label files."""
    return _valid_nifti_files(LABELS_PATH)


def verify_dataset():
    """
    Verify dataset integrity.
    """

    images = get_image_files()
    labels = get_label_files()

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    assert len(images) == len(labels), \
        "Image/label counts do not match."

    for image, label in zip(images, labels):

        assert image.name == label.name, \
            f"Mismatch: {image.name} != {label.name}"

    print("Dataset verification passed.")


def get_patient_records():
    """
    Return one dictionary per patient.
    """

    verify_dataset()

    images = get_image_files()
    labels = get_label_files()

    patients = []

    for image, label in zip(images, labels):

        patients.append(
            {
                "patient_id": image.stem.replace(".nii", ""),
                "image": image,
                "label": label,
            }
        )

    return patients
