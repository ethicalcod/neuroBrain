
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

    The dataset is responsible only for:
    1. Loading the MRI volume.
    2. Normalizing each MRI modality.
    3. Loading the segmentation label.
    4. Converting arrays to PyTorch tensors.

    Spatial cropping and augmentation are handled by transforms.
    """

    def __init__(self, transform=None):
        """
        Parameters
        ----------
        transform : callable, optional
            Optional transform to be applied to the sample.
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

        # Expected raw shape:
        # (H, W, D, C)
        #
        # Example:
        # (240, 240, 155, 4)

        # Normalize each MRI modality independently.

        for channel in range(image.shape[-1]):
            image[:, :, :, channel] = normalize_nonzero(
                image[:, :, :, channel]
            )

    
        # Load segmentation mask

        label = nib.load(patient["label"]).get_fdata().astype("uint8")

        # Expected raw label shape:
        # (H, W, D)
        #
        # Example:
        # (240, 240, 155)

     
        # Convert image to channel-first format

        image = image.transpose(3, 0, 1, 2)

        # Image:
        # (H, W, D, C)
        #      ↓
        # (C, H, W, D)

        # Convert NumPy arrays to PyTorch tensors

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        # Create sample

        sample = {
            "patient_id": patient["patient_id"],
            "image": image,
            "label": label,
        }

  
        # Apply transforms

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
