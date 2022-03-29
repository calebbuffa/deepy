from __future__ import annotations

import numpy as np
from numpy import ndarray

def chunk_image(image: ndarray, mask: ndarray: size: int) -> list[ndarray, ndarray]:
    """
    Divide pointcloud into chunks of size meters.

    Parameters
    ----------
    image : ndarray
        2D image.
    mask : ndarray
        2D mask.
    size : float
        Size of chunks.

    Returns
    -------
    list[ndarray]
        List of arrays.
    """
    xy_min = array[:, :2].min(axis=0)
    xy_max = array[:, :2].max(axis=0)

    x_range = np.arange(xy_min[0], xy_max[0], meters)
    y_range = np.arange(xy_min[1], xy_max[1], meters)

    image_chunks = []
    mask_chunks = []
    for (x_block_min, y_block_min) in zip(x_range, y_range):
        x_block_max = x_block_min + meters
        y_block_max = y_block_min + meters

        block_min_idx = np.argmin(
            np.abs(array[:, :2] - np.array(x_block_min, y_block_min))
        )
        block_max_idx = np.argmin(
            np.abs(array[:, :2] - np.array(x_block_max, y_block_max))
        )

        image_chunk = array[block_min_idx:block_max_idx]
        mask_chunk = labels[block_min_idx:block_max_idx]
        if block_points.any():
            image_chunks.append(image_chunk)
            mask_chunks.append(mask_chunk)

    return image_chunks, mask_chunks
