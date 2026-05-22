"""
I/O and image normalization utilities.

Pure numpy/tifffile functions, except load_png which needs bpy.
"""

import numpy as np
import bpy


def load_png(image_path):
    """Load .png into numpy array."""
    image = bpy.data.images.load(image_path)
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape((height, width, -1))


def normalize_quantiles(image, quantiles=(0.01, 0.99), channel_axis=None, clip=False,
                        data_type=None):
    """
    Normalize a multi-dimensional image by setting given quantiles to 0 and 1.

    Parameters
    ----------
    image : np.array
        Multi-dimensional image.
    quantiles : tuple
        Image quantile to set to 0 and 1.
    channel_axis : int or None
        If None, the image is assumed to have only a single channel.
        If int, indicates the position of the channel axis.
        Each channel is normalized separately.
    clip : bool
        Whether to clip image to 0-1. Automatically enabled if converting to int dtype.
    data_type : None, np.uint8 or np.uint16
        If not None, image is converted to given data type.

    Returns
    -------
    image_normalized : np.array
        Normalized image, the same shape as input.
    """
    image = image.astype(float)  # avoid overflow during subtraction
    if channel_axis is None:
        image_normalized = image - np.nanquantile(image, quantiles[0])
        denom = np.nanquantile(image_normalized, quantiles[1])
        if denom != 0:
            image_normalized /= denom
        image_normalized = np.nan_to_num(image_normalized)
    else:
        image_normalized = np.moveaxis(image, channel_axis, 0)
        image_normalized = np.stack([ch - np.nanquantile(ch, quantiles[0]) for ch in image_normalized])
        denoms = np.array([np.nanquantile(ch, quantiles[1]) for ch in image_normalized])
        denoms[denoms == 0] = 1.0
        image_normalized = np.stack([ch / d for ch, d in zip(image_normalized, denoms)])
        image_normalized = np.moveaxis(np.nan_to_num(image_normalized), 0, channel_axis)
    if clip or (data_type is not None):
        image_normalized = np.clip(image_normalized, 0, 1)
    if data_type is np.uint8:
        image_normalized = np.round((2**8 - 1) * image_normalized).astype(np.uint8)
    if data_type is np.uint16:
        image_normalized = np.round((2**16 - 1) * image_normalized).astype(np.uint16)
    return image_normalized


def axis_order_to_transpose(axis_order_string):
    """Convert string describing axis order into tuple for use in np.transpose."""
    if ''.join(sorted(axis_order_string)) not in ('xyz', 'cxyz'):
        raise ValueError(
            f"Axis order must be 'xyz', 'cxyz', or a permutation; got {axis_order_string!r}"
        )
    if 'c' in axis_order_string:
        return [axis_order_string.index(k) for k in 'cxyz']
    return [axis_order_string.index(k) for k in 'xyz']
