
"""
Visualization utilities for NeuroBrain.
"""

import torch
import matplotlib.pyplot as plt


MODALITIES = {
    0: "FLAIR",
    1: "T1",
    2: "T1ce",
    3: "T2",
}


def compare_samples(
    original,
    augmented,
    modality=0,
    slice_index=None,
):
    """
    Compare original and augmented MRI with segmentation overlay.

    Parameters
    ----------
    original : dict
        Original sample.

    augmented : dict
        Augmented sample.

    modality : int
        MRI modality:
        0 = FLAIR
        1 = T1
        2 = T1ce
        3 = T2

    slice_index : int or None
        Slice to visualize.
        If None, automatically selects the slice
        with the largest tumour area.
    """
    # Automatically choose best tumour slice

    if slice_index is None:

        label = original["label"]

        tumour_area = (label > 0).sum(dim=(0, 1))

        slice_index = torch.argmax(tumour_area).item()

    modality_name = MODALITIES.get(modality, f"Channel {modality}")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 12),
    )

    # Original MRI

    axes[0, 0].imshow(
        original["image"][modality][:, :, slice_index],
        cmap="gray",
    )

    axes[0, 0].set_title(
        f"Original {modality_name}"
    )

    axes[0, 0].axis("off")

    # Original Overlay

    axes[0, 1].imshow(
        original["image"][modality][:, :, slice_index],
        cmap="gray",
    )

    axes[0, 1].imshow(
        original["label"][:, :, slice_index],
        cmap="jet",
        alpha=0.40,
    )

    axes[0, 1].set_title(
        f"Original Overlay ({modality_name})"
    )

    axes[0, 1].axis("off")

    # Augmented MRI

    axes[1, 0].imshow(
        augmented["image"][modality][:, :, slice_index],
        cmap="gray",
    )

    axes[1, 0].set_title(
        f"Augmented {modality_name}"
    )

    axes[1, 0].axis("off")

    # Augmented Overlay

    axes[1, 1].imshow(
        augmented["image"][modality][:, :, slice_index],
        cmap="gray",
    )

    axes[1, 1].imshow(
        augmented["label"][:, :, slice_index],
        cmap="jet",
        alpha=0.40,
    )

    axes[1, 1].set_title(
        f"Augmented Overlay ({modality_name})"
    )

    axes[1, 1].axis("off")

    plt.tight_layout()

    plt.show()

    tumour_voxels = int(
        (original["label"][:, :, slice_index] > 0).sum()
    )


    print(f"Patient          : {original['patient_id']}")
    print(f"MRI Modality     : {modality_name}")
    print(f"Visualized Slice : {slice_index}")
    print(f"Tumour Voxels    : {tumour_voxels}")
  
