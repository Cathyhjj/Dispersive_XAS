"""Spectral generation and manipulation functions for DXAS.

Canonical array convention
--------------------------
All spectra inside this module use shape **(2, N)** where row 0 is the
energy/pixel axis and row 1 is the intensity axis.  The helper functions
:func:`spec_shaper` and :func:`spec_wrapper` transparently convert to/from
the transposed **(N, 2)** layout used by some loaders.
"""

import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si

__all__ = [
    "spec_shaper",
    "spec_wrapper",
    "spectrum_generate",
    "norm_spec",
    "interpt_spec",
    "spec_cropping",
    "spec_save",
    "peak_finder",
    "find_edge_jump",
    "find_edge_pnts",
    "intensity_at_energy",
    "atten_slope_corr",
]


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def spec_shaper(spectrum: np.ndarray) -> np.ndarray:
    """Return a spectrum with shape **(2, N)**.

    Accepts either **(2, N)** or **(N, 2)** input and returns **(2, N)**.
    """
    if spectrum.shape[-1] == 2:
        return spectrum.T
    return spectrum


def spec_wrapper(
    energy: np.ndarray,
    intensity: np.ndarray,
    output: tuple = (2, -1),
) -> np.ndarray:
    """Stack *energy* and *intensity* into a single spectrum array.

    Parameters
    ----------
    energy : ndarray, shape (N,)
    intensity : ndarray, shape (N,)
    output : tuple
        Target shape hint.  ``(2, -1)`` → shape (2, N);
        any other value → shape (N, 2).

    Returns
    -------
    ndarray
    """
    spectrum = np.vstack((energy, intensity))
    if output[0] == 2:
        return spectrum
    return spectrum.T


# ---------------------------------------------------------------------------
# Spectrum generation
# ---------------------------------------------------------------------------


def spectrum_generate(
    crop_mux: np.ndarray,
    mode: str = "average",
    title: str = "temp",
    show: bool = True,
    **kwargs,
) -> np.ndarray:
    """Collapse a 2-D masked absorption image into a 1-D spectrum.

    Each column corresponds to one energy/pixel channel; rows are spatial
    pixels.  The mask (if a masked array is supplied) suppresses bad pixels.

    Parameters
    ----------
    crop_mux : ndarray or MaskedArray, shape (rows, cols)
        2-D absorption image; columns → energy axis.
    mode : {'average', 'sum'}
        Reduction method along the spatial (row) axis.
    title : str
        Plot title (when *show* is ``True``).
    show : bool
        Plot the resulting spectrum.
    **kwargs
        Extra arguments forwarded to ``plt.plot``.

    Returns
    -------
    ndarray, shape (2, N)
        Row 0: pixel indices; row 1: intensity.
    """
    ener_pnt = np.arange(crop_mux.shape[1])
    if mode == "sum":
        intensity = np.sum(crop_mux, axis=0)
    else:
        intensity = np.average(crop_mux, axis=0)

    if show:
        plt.plot(ener_pnt, intensity, **kwargs)
        plt.title(title)

    return np.array([ener_pnt, intensity])


# ---------------------------------------------------------------------------
# Normalisation / interpolation
# ---------------------------------------------------------------------------


def norm_spec(
    spectrum: np.ndarray,
    x0: Optional[Union[float, list]] = None,
    x1: Optional[Union[float, list]] = None,
    show: bool = False,
    **kwargs,
) -> np.ndarray:
    """Min–max normalise a spectrum's intensity to [0, 1].

    Parameters
    ----------
    spectrum : ndarray, shape (2, N) or (N, 2)
    x0, x1 : float or list of float, optional
        Energy/pixel bounds that define the normalisation range.
        If ``None``, the global min and max are used.
        Pass lists to combine multiple intervals.
    show : bool
        Plot the normalised spectrum.
    **kwargs
        Extra arguments forwarded to ``plt.plot``.

    Returns
    -------
    ndarray
        Normalised spectrum in the **same shape** as *spectrum*.
    """
    spec = spec_shaper(spectrum)
    energy = spec[0]
    intensity = spec[1]

    if x1 is None:
        max_val = intensity.max()
        min_val = intensity.min()
    elif isinstance(x0, list):
        mask = np.ma.masked_inside(energy, x0[0], x1[0])
        for i in range(1, len(x0)):
            mask *= np.ma.masked_inside(energy, x0[i], x1[i])
        max_val = intensity[mask.mask].max()
        min_val = intensity[mask.mask].min()
    else:
        mask = np.ma.masked_inside(energy, x0, x1)
        max_val = intensity[mask.mask].max()
        min_val = intensity[mask.mask].min()

    intensity_norm = (intensity - min_val) / (max_val - min_val)

    if show:
        plt.plot(energy, intensity_norm, **kwargs)

    return spec_wrapper(energy, intensity_norm, output=spectrum.shape)


def interpt_spec(
    spec: np.ndarray,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    pnts: int = 3000,
) -> np.ndarray:
    """Interpolate a spectrum onto a uniform grid.

    Parameters
    ----------
    spec : ndarray, shape (2, N) or (N, 2)
    x_min, x_max : float, optional
        Range to interpolate over.  Defaults to the spectrum's full range.
    pnts : int
        Number of output points (default: 3000).

    Returns
    -------
    ndarray, shape (2, pnts)
    """
    spec = spec_shaper(spec)
    if x_min is None:
        x_min = spec[0].min()
        x_max = spec[0].max()
    new_x = np.linspace(x_min, x_max, pnts, endpoint=False)
    f = si.interp1d(spec[0], spec[1])
    return np.array([new_x, f(new_x)])


# ---------------------------------------------------------------------------
# Cropping / saving
# ---------------------------------------------------------------------------


def spec_cropping(
    spec: np.ndarray,
    crop_E1: float,
    crop_E2: float,
    show: bool = False,
) -> np.ndarray:
    """Crop a spectrum to a contiguous energy range.

    Only a *contiguous* block of indices within ``[crop_E1, crop_E2]`` is
    returned, so this safely removes isolated outlier regions.

    Parameters
    ----------
    spec : ndarray, shape (2, N) or (N, 2)
    crop_E1, crop_E2 : float
        Lower and upper energy/pixel bounds.
    show : bool
        Plot the cropped spectrum.

    Returns
    -------
    ndarray, shape (2, M)
    """
    spec = spec_shaper(spec)
    idx = np.where((spec[0] > crop_E1) & (spec[0] < crop_E2))[0]
    # Keep only the first contiguous block
    expected = np.arange(idx[0], idx[0] + len(idx))
    contiguous = idx[idx - expected == 0]
    cropped = spec[:, contiguous]
    if show:
        plt.plot(cropped[0], cropped[1])
    return cropped


def spec_save(
    spec: np.ndarray,
    crop_E1: Optional[float] = None,
    crop_E2: Optional[float] = None,
    save_name: str = "cropped_spec",
    show: bool = True,
) -> np.ndarray:
    """Save a spectrum to a plain-text file.

    Parameters
    ----------
    spec : ndarray
    crop_E1, crop_E2 : float, optional
        If provided, crop the spectrum before saving.
    save_name : str
        Base filename (without extension).  A date prefix is prepended.
    show : bool
        Plot the (possibly cropped) spectrum.

    Returns
    -------
    ndarray
        The saved spectrum.
    """
    from .utils import date_today

    spec = spec_shaper(spec)
    if crop_E1 is not None:
        spec = spec_cropping(spec, crop_E1, crop_E2, show=True)

    os.makedirs("saved_spec", exist_ok=True)
    file_name = os.path.join("saved_spec", f"{date_today()}_{save_name}.dat")
    print(f"Saving {file_name}")
    np.savetxt(file_name, spec.T)

    if show:
        plt.plot(spec[0], spec[1])
    return spec


# ---------------------------------------------------------------------------
# Peak finding / edge detection
# ---------------------------------------------------------------------------


def peak_finder(
    spec: np.ndarray,
    spec_min: Optional[float] = None,
    spec_max: Optional[float] = None,
    peak_n: Optional[int] = None,
    prominence: float = 0.01,
    filtering: bool = False,
    show: bool = True,
    **kwargs,
) -> tuple:
    """Find peaks and troughs in a spectrum.

    Parameters
    ----------
    spec : ndarray
    spec_min, spec_max : float, optional
        Energy interval to restrict peak search.
    peak_n : int, optional
        Return only the first *peak_n* peaks (sorted by position).
    prominence : float
        Minimum peak prominence (default: 0.01).
    filtering : bool
        Apply Savitzky–Golay smoothing before detection.
    show : bool
        Plot the original spectrum, the filtered/masked version, and
        the detected peaks.
    **kwargs
        Extra arguments for ``scipy.signal.savgol_filter``
        (e.g. ``window_length``, ``polyorder``).

    Returns
    -------
    (ndarray, ndarray)
        Peak indices and the corresponding energy/pixel values.
    """
    from scipy import signal

    spec = spec_shaper(spec)
    spec_y = spec[1].copy()

    if spec_min is not None:
        interval = np.ma.masked_inside(spec[0], spec_min, spec_max)
        spec_y = spec[1] * interval.mask

    if filtering:
        spec_y = signal.savgol_filter(spec_y, **kwargs)

    peaks_pos, _ = signal.find_peaks(spec_y, prominence=prominence)
    peaks_neg, _ = signal.find_peaks(-spec_y, prominence=prominence)
    peaks = np.sort(np.hstack([peaks_pos, peaks_neg]))

    if peak_n is not None:
        peaks = peaks[:peak_n]

    if show:
        plt.figure()
        plt.plot(spec[0], spec[1], label="original")
        plt.plot(spec[0], spec_y, label="filtered/masked")
        plt.plot(spec[0][peaks], spec[1][peaks], "o", color="red")
        plt.legend()

    return peaks, spec[0][peaks]


def find_edge_jump(
    spec: np.ndarray,
    show: bool = True,
    prominence: float = 0.005,
) -> int:
    """Locate the absorption edge midpoint by finding the steepest derivative.

    Parameters
    ----------
    spec : ndarray, shape (2, N)
    show : bool
        Overlay derivative on the spectrum plot.
    prominence : float
        Minimum prominence for derivative peaks.

    Returns
    -------
    int
        Index of the edge midpoint in the spectrum array.
    """
    import scipy as sp

    spec = spec_shaper(spec)
    x, y = spec[0], spec[1]

    spl = sp.interpolate.splrep(x, y)
    dy = sp.interpolate.splev(x, spl, der=1)
    dy_f = sp.signal.savgol_filter(dy, 11, 2)
    dy_f = dy_f / dy_f.max()

    peaks = sp.signal.find_peaks(dy_f, prominence=prominence)[0]
    idx0 = peaks[np.argsort(-dy[peaks])[0]]
    idx1 = peaks[np.argsort(-dy[peaks])[1]]
    mid = idx0 // 2 + idx1 // 2

    if show:
        plt.plot(x, y)
        plt.plot(x, dy_f * 0.5)
        plt.hlines(
            np.zeros(len(x)),
            xmin=x.min(),
            xmax=x.max(),
            linestyles=":",
            alpha=0.3,
        )
        plt.scatter(x[mid], y[mid], color="r")

    return mid


def find_edge_pnts(
    spec: np.ndarray,
    y_pnts: np.ndarray,
    edge_min: Optional[float] = None,
    edge_max: Optional[float] = None,
    show: bool = True,
) -> np.ndarray:
    """Interpolate the energy/pixel values at given intensity fractions.

    Useful for locating edge positions at specific normalised intensities
    (e.g. half-edge, pre-edge).

    Parameters
    ----------
    spec : ndarray, shape (2, N)
    y_pnts : ndarray
        Target intensity values.
    edge_min, edge_max : float, optional
        Restrict interpolation to this energy window.
    show : bool

    Returns
    -------
    ndarray
        Energy/pixel value at each requested intensity.
    """
    spec = spec_shaper(spec)
    if edge_min is not None:
        interval = np.ma.masked_inside(spec[0], edge_min, edge_max)
        x_pnts = np.interp(y_pnts, spec[1][interval.mask], spec[0][interval.mask])
    else:
        x_pnts = np.interp(y_pnts, spec[1], spec[0])

    if show:
        plt.plot(spec[0], spec[1])
        plt.plot(x_pnts, y_pnts, "o")

    return x_pnts


def intensity_at_energy(
    energy: np.ndarray,
    spec_1d: np.ndarray,
    E_eV: float,
) -> float:
    """Interpolate the spectrum intensity at a specific energy.

    Parameters
    ----------
    energy : ndarray
        Energy axis in eV.
    spec_1d : ndarray
        Intensity axis.
    E_eV : float
        Target energy in eV.

    Returns
    -------
    float
    """
    from scipy.interpolate import interp1d

    return float(interp1d(energy, spec_1d)(E_eV))


# ---------------------------------------------------------------------------
# Attenuation slope correction (optional: requires xraylib)
# ---------------------------------------------------------------------------


def atten_slope_corr(
    element: int,
    data_x: np.ndarray,
    E_threshold: float = 0,
    show: bool = True,
) -> np.ndarray:
    """Compute an attenuation-slope correction curve using tabulated cross-sections.

    Requires the optional dependency ``xraylib``.

    Parameters
    ----------
    element : int
        Atomic number Z.
    data_x : ndarray
        Energy array in eV for which the correction is evaluated.
    E_threshold : float
        Energies below this value are set to 1 (no correction).
    show : bool
        Plot the correction curve.

    Returns
    -------
    ndarray
        Normalised correction curve, same length as *data_x*.
    """
    import xraylib
    from scipy.interpolate import interp1d

    energy_keV = np.arange(23, 28, 0.005)
    density = xraylib.ElementDensity(element)
    cs = np.array([xraylib.CS_Total(element, E) for E in energy_keV]) * density
    cs_norm = (cs - cs.min()) / (cs.max() - cs.min())

    correction = interp1d(energy_keV, cs_norm)(data_x / 1000.0)
    correction[data_x < E_threshold] = 1.0

    if show:
        plt.figure()
        plt.plot(data_x, correction)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Correction factor")

    return correction
