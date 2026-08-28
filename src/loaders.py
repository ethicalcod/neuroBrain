
"""
DataLoader utilities for NeuroBrain.
"""

import torch
from torch.utils.data import DataLoader, random_split

from src.config import RANDOM_SEED, BATCH_SIZE, NUM_WORKERS
from src.dataloader import BrainTumourDataset
from src.transforms import (
    get_train_transforms,
    get_val_transforms,
)


def create_dataloaders():
    """
    Create training and validation DataLoaders.

    The dataset is split deterministically using RANDOM_SEED.
    Training and validation datasets use separate transform pipelines.
    """

    # Create a base dataset without transforms.
    full_dataset = BrainTumourDataset(transform=None)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_indices, val_indices = random_split(
        range(len(full_dataset)),
        [train_size, val_size],
        generator=generator,
    )

    # Create separate datasets so that training and validation
    # can use different transforms.

    train_dataset = BrainTumourDataset(
        transform=get_train_transforms()
    )

    val_dataset = BrainTumourDataset(
        transform=get_val_transforms()
    )

    # Keep only the indices belonging to each split.
    train_dataset = torch.utils.data.Subset(
        train_dataset,
        train_indices.indices,
    )

    val_dataset = torch.utils.data.Subset(
        val_dataset,
        val_indices.indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader
