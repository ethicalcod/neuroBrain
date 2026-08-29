
"""
Training and validation utilities for NeuroBrain.
"""

import torch

from src.inference import predict_volume


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        3D U-Net segmentation model.

    loader : DataLoader
        Training DataLoader.

    criterion : callable
        Segmentation loss function.

    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.

    device : torch.device
        Device used for computation.

    Returns
    -------
    float
        Average training loss.
    """

    model.train()

    running_loss = 0.0
    num_batches = 0

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)

        loss = criterion(predictions, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0

    return running_loss / num_batches


def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    overlap=0.25,
):
    """
    Validate the model for one epoch using sliding-window inference.

    Full validation volumes are never passed directly through the
    3D U-Net. Instead, MONAI performs overlapping 96x96x96
    window inference and reconstructs the full-volume prediction.

    Parameters
    ----------
    model : torch.nn.Module
        3D U-Net segmentation model.

    loader : DataLoader
        Validation DataLoader.

    criterion : callable
        Segmentation loss function.

    device : torch.device
        Device used for computation.

    roi_size : tuple
        Spatial size of each sliding-window patch.

    sw_batch_size : int
        Number of sliding-window patches processed simultaneously.

    overlap : float
        Fraction of overlap between neighboring windows.

    Returns
    -------
    float
        Average validation loss.
    """

    model.eval()

    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Full-volume sliding-window inference.
            predictions = predict_volume(
                model=model,
                image=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
            )

            loss = criterion(predictions, labels)

            running_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        return 0.0

    return running_loss / num_batches
