
"""
Training and validation transforms for NeuroBrain.
"""

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    RandSpatialCropd,
)

from src.config import PATCH_SIZE


class RemoveLabelChannel(MapTransform):
    """
    Remove the artificial channel dimension from the segmentation label.

    MONAI spatial transforms expect channel-first data. The label is
    therefore temporarily represented as [1, H, W, D] and converted
    back to [H, W, D] after spatial transformations.
    """

    def __call__(self, data):
        d = dict(data)

        label = d["label"]

        if label.ndim == 4 and label.shape[0] == 1:
            d["label"] = label[0]

        return d


def get_train_transforms():
    """
    Training transform pipeline.

    Training uses:
    - channel-first formatting
    - modality-wise intensity normalization
    - random 96x96x96 spatial crops
    - random flipping
    """

    return Compose(
        [
            # Add a channel dimension to the label while keeping
            # the MRI image's existing channel dimension.
            EnsureChannelFirstd(
                keys=["label"],
                channel_dim="no_channel",
            ),

            # Normalize each MRI modality independently.
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
                channel_wise=True,
            ),

            # Extract a random training patch.
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=PATCH_SIZE,
                random_size=False,
            ),

            # Random spatial augmentation.
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=0,
            ),

            # Convert to tensors / MetaTensors.
            EnsureTyped(
                keys=["image", "label"],
            ),

            # Remove the temporary label channel.
            RemoveLabelChannel(
                keys=["label"],
            ),
        ]
    )


def get_val_transforms():
    """
    Validation transform pipeline.

    Validation does not perform random spatial cropping or
    random augmentation. The complete MRI volume is retained.
    """

    return Compose(
        [
            # Normalize each MRI modality independently.
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
                channel_wise=True,
            ),

            # Convert to tensors / MetaTensors.
            EnsureTyped(
                keys=["image", "label"],
            ),
        ]
    )
