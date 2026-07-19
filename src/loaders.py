
"""
DataLoader utilities for NeuroBrain.
"""

import torch
from torch.utils.data import DataLoader, random_split

from src.config import RANDOM_SEED, BATCH_SIZE
from src.dataloader import BrainTumourDataset


def create_dataloaders():

    dataset = BrainTumourDataset()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader
