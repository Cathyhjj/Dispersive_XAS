"""Image pre-processing pipeline for DXAS.

Converts raw detector images into absorption maps (μx) via Beer-Lambert law:

    μx = -log(I / I₀)

where *I* is the sample image and *I₀* is the flat-field (beam-only) image,
both dark-corrected and denoised beforehand.
"""

import os
from typing import Optional

import numpy as np
import scipy.ndimage as nd

from . import h5io
from .utils import date_today

__all__ = [
    "pre_process",
    "pre_process_scan",
    "ximea_correction",
]


def pre_process(
    data: np.ndarray,
    flat: np.ndarray,
    dark: Optional[np.ndarray] = None,
    denoise_size: int = 3,
    savedata: bool = True,
    prefix: str = "preproc",
) -> dict:
    """Process raw DXAS images into an absorption map.

    Steps:

    1. Dark-field subtraction (optional).
    2. Median-filter denoising.
    3. Clip non-positive pixels to a small positive value.
    4. Compute transmission ``T = data / flat``.
    5. Compute absorption ``μx = -log(T)``.

    Parameters
    ----------
    data : ndarray, shape (H, W)
        Sample image (averaged over frames if needed before calling).
    flat : ndarray, shape (H, W)
        Flat-field image (beam without sample).
    dark : ndarray or None
        Dark-field image.  If provided, subtracted from both *data* and *flat*.
    denoise_size : int
        Kernel size for the median denoising filter (default: 3).
    savedata : bool
        If ``True``, write results to HDF5 files under a dated directory.
    prefix : str
        Directory name suffix appended after the date (e.g. ``YYYYMMDD_preproc``).

    Returns
    -------
    dict
        ``{'data': ndarray, 'flat': ndarray, 'transmission': ndarray, 'mux': ndarray}``
    """
    if dark is not None:
        data_dc = data - dark
        flat_dc = flat - dark
    else:
        data_dc = data.copy()
        flat_dc = flat.copy()

    data_dc = nd.median_filter(data_dc, size=denoise_size)
    flat_dc = nd.median_filter(flat_dc, size=denoise_size)

    # Clip non-positive values to avoid log(0) and division by zero
    flat_dc[flat_dc <= 0] = 1e-4
    data_dc[data_dc <= 0] = 1e-4

    transmission = data_dc / flat_dc
    mux = -np.log(transmission)

    if savedata:
        directory = os.path.join(os.getcwd(), date_today() + "_" + prefix)
        os.makedirs(directory, exist_ok=True)
        h5io.h5write(os.path.join(directory, "00_data.h5"), var1=data_dc)
        h5io.h5write(os.path.join(directory, "01_flat.h5"), var1=flat_dc)
        h5io.h5write(os.path.join(directory, "02_transmission.h5"), var1=transmission)
        h5io.h5write(os.path.join(directory, "03_mux.h5"), var1=mux)

    return {
        "data": data_dc,
        "flat": flat_dc,
        "transmission": transmission,
        "mux": mux,
    }


def pre_process_scan(
    data_darkcorr: np.ndarray,
    flat_darkcorr: np.ndarray,
    denoise_size: int = 3,
    savedata: bool = True,
    prefix: str = "preproc",
) -> dict:
    """Process a stack of pre-dark-corrected scan images.

    Applies median denoising to the entire stack and computes transmission
    and absorption frame-by-frame.  Optionally saves one HDF5 file per frame
    into four subdirectories.

    Parameters
    ----------
    data_darkcorr : ndarray, shape (N, H, W)
        Dark-corrected sample image stack.
    flat_darkcorr : ndarray, shape (N, H, W)
        Dark-corrected flat-field image stack.
    denoise_size : int
        Kernel size for median denoising (default: 3).
    savedata : bool
        If ``True``, save per-frame HDF5 files.
    prefix : str
        Output directory name.

    Returns
    -------
    dict
        ``{'data': ndarray, 'flat': ndarray, 'transmission': ndarray, 'mux': ndarray}``
        Each array has shape (N, H, W).
    """
    data_dn = nd.median_filter(data_darkcorr, size=denoise_size)
    flat_dn = nd.median_filter(flat_darkcorr, size=denoise_size)

    flat_dn[flat_dn <= 0] = 1e-4
    data_dn[data_dn <= 0] = 1e-4

    transmission = data_dn / flat_dn
    mux = -np.log(transmission)

    if savedata:
        directory = os.path.join(os.getcwd(), prefix)
        subfolders = ["00_data", "01_flat", "02_transmission", "03_mux"]
        for sub in subfolders:
            os.makedirs(os.path.join(directory, sub), exist_ok=True)

        n_frames = data_darkcorr.shape[0]
        for i in range(n_frames):
            h5io.h5write(
                os.path.join(directory, "00_data", f"scan_{i:05d}_data.h5"),
                var1=data_dn[i],
            )
            h5io.h5write(
                os.path.join(directory, "01_flat", f"scan_{i:05d}_flat.h5"),
                var1=flat_dn[i],
            )
            h5io.h5write(
                os.path.join(directory, "02_transmission", f"scan_{i:05d}_transmission.h5"),
                var1=transmission[i],
            )
            h5io.h5write(
                os.path.join(directory, "03_mux", f"scan_{i:05d}_mux.h5"),
                var1=mux[i],
            )

    return {
        "data": data_dn,
        "flat": flat_dn,
        "transmission": transmission,
        "mux": mux,
    }


def ximea_correction(
    data: np.ndarray,
    crop_slice=None,
    show: bool = False,
):
    """Correct a Ximea detector image for column-wise offset non-uniformity.

    Parameters
    ----------
    data : ndarray, shape (H, W)
        Raw detector image.
    crop_slice : numpy slice or None
        Region to use for estimating the column offset.
        Defaults to ``np.s_[25:475, :]``.
    show : bool
        If ``True``, display diagnostic plots (requires pyqtgraph).

    Returns
    -------
    data_corr_minus : ndarray
        Offset-subtracted image.
    data_corr_divide : ndarray
        Offset-divided image.
    """
    if crop_slice is None:
        crop_slice = np.s_[25:475, : data.shape[1]]

    corr_mask = np.zeros_like(data) + 0.2
    corr_mask[crop_slice] = 1.0

    ximea_offset = np.average(nd.median_filter(data[crop_slice], size=3), axis=0)
    data_corr_minus = data - ximea_offset
    data_corr_divide = data / ximea_offset

    if show:
        import matplotlib.pyplot as plt
        import pyqtgraph as pg

        plt.imshow(data * corr_mask, vmin=400, vmax=1000)
        plt.figure()
        plt.plot(ximea_offset)
        plt.title("Ximea column offset")
        pg.image(data_corr_minus.T)
        pg.image(data_corr_divide.T)
        pg.image(nd.median_filter(data_corr_minus, size=3))
        pg.image(nd.median_filter(data_corr_divide, size=3))

    return data_corr_minus, data_corr_divide
