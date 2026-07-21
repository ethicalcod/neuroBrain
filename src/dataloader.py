
"""
PyTorch Dataset for NeuroBrain.
"""

import random

import nibabel as nib
import torch
from torch.utils.data import Dataset

from src.config import PATCH_SIZE
from src.dataset import get_patient_records
from src.preprocessing import normalize_nonzero


class BrainTumourDataset(Dataset):
    """
    PyTorch Dataset for the Medical Segmentation Decathlon
    Brain Tumour dataset.
    """

    def __init__(self, transform=None):
        self.records = get_patient_records()
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        patient = self.records[index]

        # Load MRI

        image = nib.load(
            patient["image"]
        ).get_fdata().astype("float32")

        # Normalize every modality independently

        for channel in range(image.shape[-1]):
            image[..., channel] = normalize_nonzero(
                image[..., channel]
            )

        # Load segmentation mask

        label = nib.load(
            patient["label"]
        ).get_fdata().astype("uint8")

        # Convert image to channel-first
        # (H,W,D,C) -> (C,H,W,D)

        image = image.transpose(3, 0, 1, 2)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        # Manual random crop

        crop_h, crop_w, crop_d = PATCH_SIZE

        _, H, W, D = image.shape

        start_h = random.randint(0, H - crop_h)
        start_w = random.randint(0, W - crop_w)
        start_d = random.randint(0, D - crop_d)

        image = image[
            :,
            start_h:start_h + crop_h,
            start_w:start_w + crop_w,
            start_d:start_d + crop_d,
        ]

        label = label[
            start_h:start_h + crop_h,
            start_w:start_w + crop_w,
            start_d:start_d + crop_d,
        ]

        sample = {
            "patient_id": patient["patient_id"],
            "image": image,
            "label": label,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
