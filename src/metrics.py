
"""
Evaluation metrics for NeuroBrain.

This module provides Dice-based metrics for multi-class
brain tumour segmentation.

The implementation reports:

1. Per-class Dice scores:
   - Edema (ED)
   - Non-enhancing tumour
   - Enhancing tumour (ET)

2. BraTS-style composite-region Dice scores:
   - Whole Tumour (WT)
   - Tumour Core (TC)
   - Enhancing Tumour (ET)

Background is excluded from the primary mean Dice score.
"""

import torch


def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Dice similarity coefficient for binary masks.

    If both masks are empty, Dice is defined as 1.0 because
    the prediction perfectly agrees with the ground truth.
    """

    prediction = prediction.float()
    target = target.float()

    intersection = torch.sum(prediction * target)

    prediction_sum = torch.sum(prediction)
    target_sum = torch.sum(target)

    denominator = prediction_sum + target_sum

    # Both masks are empty.
    if denominator.item() == 0:
        return torch.tensor(
            1.0,
            device=prediction.device,
            dtype=torch.float32,
        )

    return (2.0 * intersection) / (denominator + epsilon)

def multiclass_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4,
) -> dict:
    """
    Compute Dice score independently for each segmentation class.

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted class labels with shape (B, H, W, D).

    target : torch.Tensor
        Ground-truth class labels with shape (B, H, W, D).

    num_classes : int, default=4
        Number of segmentation classes.

    Returns
    -------
    dict
        Dictionary containing Dice scores for each class.
    """

    scores = {}

    for class_id in range(num_classes):

        pred_mask = prediction == class_id
        target_mask = target == class_id

        scores[class_id] = dice_score(
            pred_mask,
            target_mask,
        )

    return scores


def tumor_region_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    Compute Dice scores for composite tumour regions.

    Regions
    -------
    WT : Whole Tumour
        Classes 1 + 2 + 3

    TC : Tumour Core
        Classes 2 + 3

    ET : Enhancing Tumour
        Class 3

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted class labels with shape (B, H, W, D).

    target : torch.Tensor
        Ground-truth class labels with shape (B, H, W, D).

    Returns
    -------
    dict
        Dictionary containing WT, TC and ET Dice scores.
    """

    prediction_wt = prediction > 0
    target_wt = target > 0

    prediction_tc = (prediction == 2) | (prediction == 3)
    target_tc = (target == 2) | (target == 3)

    prediction_et = prediction == 3
    target_et = target == 3

    return {
        "WT": dice_score(
            prediction_wt,
            target_wt,
        ),
        "TC": dice_score(
            prediction_tc,
            target_tc,
        ),
        "ET": dice_score(
            prediction_et,
            target_et,
        ),
    }
