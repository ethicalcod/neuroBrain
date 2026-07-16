
import numpy as np


def normalize_nonzero(volume):
    """
    Normalize a 3D MRI volume using only non-zero voxels.

    Parameters
    ----------
    volume : np.ndarray
        Input MRI volume.

    Returns
    -------
    np.ndarray
        Normalized MRI volume.
    """

    normalized = volume.copy().astype(np.float32)

    mask = volume > 0

    if np.any(mask):
        mean = volume[mask].mean()
        std = volume[mask].std()

        if std > 0:
            normalized[mask] = (volume[mask] - mean) / std

    return normalized
