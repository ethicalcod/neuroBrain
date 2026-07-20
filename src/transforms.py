
"""
MONAI transform pipelines for NeuroBrain.
"""

from monai.transforms import (
    Compose,
    EnsureTyped,
)


def get_train_transforms():
    """
    Training transform pipeline.
    """

    return Compose(
        [
            EnsureTyped(
                keys=["image", "label"],
            ),
        ]
    )


def get_val_transforms():
    """
    Validation transform pipeline.
    """

    return Compose(
        [
            EnsureTyped(
                keys=["image", "label"],
            ),
        ]
    )
