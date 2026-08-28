
"""
Inference utilities for NeuroBrain.

Provides sliding-window inference for full 3D MRI volumes.
"""

import torch
from monai.inferers import sliding_window_inference


def predict_volume(
    model,
    image,
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    overlap=0.25,
):
    """
    Run sliding-window inference on a full MRI volume.

    Parameters
    ----------
    model : torch.nn.Module
        Trained 3D segmentation model.

    image : torch.Tensor
        Input MRI volume with shape:
        [B, C, H, W, D]

    roi_size : tuple
        Spatial size of each inference window.

    sw_batch_size : int
        Number of windows processed simultaneously.

    overlap : float
        Fraction of overlap between neighboring windows.

    Returns
    -------
    torch.Tensor
        Segmentation logits with shape:
        [B, num_classes, H, W, D]
    """

    model.eval()

    with torch.no_grad():

        prediction = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )

    return prediction
