"""Low-level image transformations for DXAS.

Provides elemental operations (rotate, shift, flip, filter) plus utilities
for multi-scan registration (thresholding, shift detection, stitching).
"""

import numpy as np
import scipy.ndimage as nd

__all__ = [
    "rotate_image",
    "shift_image",
    "flip_image_horizontal",
    "flip_image_vertical",
    "invert_image",
    "log_image",
    "threshold_image_min",
    "threshold_image_max",
    "median_filter_image",
    "gaussian_filter_image",
    "register_thresholding",
    "find_shifts",
    "stitch_scans",
]


def rotate_image(img, angle):
    return nd.rotate(img, -angle)


def shift_image(img, x_shift, y_shift):
    return nd.shift(img, (x_shift, y_shift))


def flip_image_horizontal(img):
    return np.fliplr(img)


def flip_image_vertical(img):
    return np.flipud(img)


def invert_image(img):
    return -img


def log_image(img):
    return np.log(img)


def threshold_image_min(img, min_val):
    img[img < min_val] = min_val
    return img


def threshold_image_max(img, max_val):
    img[img > max_val] = max_val
    return img


def median_filter_image(img, size):
    return nd.median_filter(img, size=size)


def gaussian_filter_image(img, sigma):
    return nd.gaussian_filter(img, sigma=sigma)


def register_thresholding(
    imgs: np.ndarray,
    binary_lower_lm: float,
    binary_upper_lm: float,
    footprint: np.ndarray = None,
    show: bool = True,
) -> np.ndarray:
    """Threshold and morphologically open a stack of images for registration.

    Pixels outside ``[binary_lower_lm, binary_upper_lm]`` are set to zero,
    then :func:`skimage.morphology.binary_opening` is applied to each frame.

    Parameters
    ----------
    imgs : ndarray, shape (N, H, W)
        Image stack.
    binary_lower_lm, binary_upper_lm : float
        Intensity thresholds.
    footprint : ndarray or None
        Structuring element for binary opening.  Defaults to a 5×5 array of
        ones.
    show : bool
        If ``True``, display the result via pyqtgraph.

    Returns
    -------
    ndarray, shape (N, H, W), dtype bool
        Thresholded and opened binary image stack.
    """
    import skimage.morphology as morph

    if footprint is None:
        footprint = np.ones((5, 5))

    binary = imgs.copy()
    binary[binary > binary_upper_lm] = 0
    binary[binary < binary_lower_lm] = 0
    binary = np.asarray(
        [morph.binary_opening(frame, footprint=footprint) for frame in binary]
    )

    if show:
        import pyqtgraph as pg
        pg.image(binary)

    return binary


def find_shifts(register_binary: np.ndarray) -> list:
    """Compute image shifts relative to the first frame.

    Uses :func:`skimage.registration.phase_cross_correlation` to measure the
    shift of each frame relative to ``register_binary[0]``.

    Parameters
    ----------
    register_binary : ndarray, shape (N, H, W)
        Binary or floating-point image stack (e.g. output of
        :func:`register_thresholding`).

    Returns
    -------
    list of ndarray
        Length N-1; each element is a 1-D array ``[row_shift, col_shift]``.
    """
    from skimage.registration import phase_cross_correlation

    ref = register_binary[0]
    return [
        phase_cross_correlation(ref, register_binary[i], normalization=None)[0]
        for i in range(1, register_binary.shape[0])
    ]


def stitch_scans(
    imgs: np.ndarray,
    masks: np.ndarray,
    show: bool = True,
) -> np.ndarray:
    """Average a scan stack within a boolean mask.

    Parameters
    ----------
    imgs : ndarray, shape (N, H, W)
        Image stack.
    masks : ndarray, shape (N, H, W)
        Boolean mask stack (``True`` = include pixel).
    show : bool
        If ``True``, display the result via pyqtgraph.

    Returns
    -------
    ndarray, shape (H, W)
        Mean image computed only over masked pixels.
    """
    masked = np.ma.array(imgs.copy(), mask=np.logical_not(masks))
    mean = masked.mean(axis=0)

    if show:
        import pyqtgraph as pg
        pg.image(mean.T)

    return mean
