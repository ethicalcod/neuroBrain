"""
PyTorch Dataset for NeuroBrain.
"""

import nibabel as nib
import torch
from torch.utils.data import Dataset

from src.dataset import get_patient_records
from src.preprocessing import normalize_nonzero


class BrainTumourDataset(Dataset):
    """
    PyTorch Dataset for the Medical Segmentation Decathlon
    Brain Tumour dataset.
    """

    def __init__(self, transform=None):
        """
        Parameters
        ----------
        transform : callable, optional
            Optional transform to be applied on a sample.
        """

        self.records = get_patient_records()
        self.transform = transform

    def __len__(self):
        """Return total number of patients."""
        return len(self.records)

    def __getitem__(self, index):
        """
        Load one patient and return image, label and patient ID.
        """

        patient = self.records[index]

        # Load MRI image

        image = nib.load(patient["image"]).get_fdata().astype("float32")

        # Normalize each MRI modality independently

        for channel in range(image.shape[-1]):
            image[:, :, :, channel] = normalize_nonzero(
                image[:, :, :, channel]
            )

        # Load segmentation mask

        label = nib.load(patient["label"]).get_fdata().astype("uint8")

        # Convert image from
        # (H, W, D, C) -> (C, H, W, D)

        image = image.transpose(3, 0, 1, 2)

        # Convert NumPy arrays to PyTorch tensors

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        # Create sample

        sample = {
            "patient_id": patient["patient_id"],
            "image": image,
            "label": label,
        }

        # Apply transforms (if provided)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
