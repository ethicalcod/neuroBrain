
"""
Deterministic transforms for visualization and debugging.

Unlike the training transforms, these transforms contain
no random augmentations. They are intended only for
verification and visualization.
"""

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    EnsureTyped,
)


def get_debug_transforms():
    """
    Deterministic transform pipeline.

    Used for:
    - visualization
    - debugging
    - sanity checks

    No random augmentations are applied.
    """

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
