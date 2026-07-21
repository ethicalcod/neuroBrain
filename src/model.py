
"""
3D U-Net model for NeuroBrain.
"""

import torch
import torch.nn as nn

from monai.networks.nets import UNet

from src.config import (
    IN_CHANNELS,
    OUT_CHANNELS,
    FEATURES,
    DROPOUT,
)


class NeuroBrainUNet(nn.Module):
    """
    3D U-Net for brain tumour segmentation.
    """

    def __init__(self):
        super().__init__()

        self.model = UNet(
            spatial_dims=3,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            channels=FEATURES,
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=DROPOUT,
        )

    def forward(self, x):
        return self.model(x)
