
"""
Loss functions for NeuroBrain.

This module provides the hybrid Dice + Cross Entropy loss
used for multi-class 3D brain tumour segmentation.
"""

import torch
from monai.losses import DiceCELoss


class DiceCrossEntropyLoss:
    """
    Hybrid Dice + Cross Entropy loss for multi-class 3D segmentation.

    Network output:
        (B, C, H, W, D)

    Ground-truth labels:
        (B, H, W, D)

    The integer class labels are converted to one-hot format
    internally for the Dice component.
    """

    def __init__(
        self,
        include_background: bool = True,
        softmax: bool = True,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        num_classes: int = 4,
    ):
        self.num_classes = num_classes

        self.loss = DiceCELoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=softmax,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

    def __call__(self, prediction, target):
        """
        Compute Dice + Cross Entropy loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Shape (B, C, H, W, D).

        target : torch.Tensor
            Shape (B, H, W, D) or (B, 1, H, W, D).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """

        if target.ndim == prediction.ndim - 1:
            target = target.unsqueeze(1)

        return self.loss(prediction, target)
