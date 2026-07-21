
"""
Training and validation transforms for NeuroBrain.
"""

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    EnsureTyped,
)


def get_train_transforms():

    return Compose(

        [

            NormalizeIntensityd(
                keys="image",
                nonzero=True,
                channel_wise=True,
            ),

            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=0,
            ),

            EnsureTyped(
                keys=["image", "label"],
            ),

        ]

    )


def get_val_transforms():

    return Compose(

        [

            NormalizeIntensityd(
                keys="image",
                nonzero=True,
                channel_wise=True,
            ),

            EnsureTyped(
                keys=["image", "label"],
            ),

        ]

    )
