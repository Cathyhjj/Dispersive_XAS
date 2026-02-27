"""Spectral generation and manipulation functions for DXAS.

Canonical array convention
--------------------------
All spectra inside this module use shape **(2, N)** where row 0 is the
energy/pixel axis and row 1 is the intensity axis.
"""

from __future__ import annotations

import os
from typing import Optional, Union

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


def _plot_lines(traces, title="", x_label="X", y_label="Y"):
    from ._display import show_lines

    show_lines(traces=traces, title=title, x_label=x_label, y_label=y_label, show=True)


def spec_shaper(spectrum: np.ndarray) -> np.ndarray:
    """Return a spectrum with shape **(2, N)**."""
    spec = np.asarray(spectrum)
    if spec.ndim != 2:
        raise ValueError(f"Expected 2-D spectrum, got shape {spec.shape}")
    if spec.shape[-1] == 2:
        return spec.T
    return spec


def spec_wrapper(
    energy: np.ndarray,
    intensity: np.ndarray,
    output: tuple = (2, -1),
) -> np.ndarray:
    """Stack *energy* and *intensity* into a single spectrum array."""
    spectrum = np.vstack((energy, intensity))
    if output[0] == 2:
        return spectrum
    return spectrum.T


def spectrum_generate(
    crop_mux: np.ndarray,
    mode: str = "average",
    title: str = "Spectrum",
    show: bool = True,
    **kwargs,
) -> np.ndarray:
    """Collapse a 2-D image into a 1-D spectrum."""
    data = np.ma.asarray(crop_mux)
    ener_pnt = np.arange(data.shape[1], dtype=float)

    if mode == "sum":
        intensity = np.ma.sum(data, axis=0)
    else:
        intensity = np.ma.mean(data, axis=0)

    if isinstance(intensity, np.ma.MaskedArray):
        intensity = intensity.filled(np.nan)
    intensity = np.asarray(intensity, dtype=float)

    if show:
        _plot_lines(
            traces=[{"x": ener_pnt, "y": intensity, "name": title, "style": kwargs}],
            title=title,
            x_label="Pixel",
            y_label="Intensity",
        )

    return np.array([ener_pnt, intensity], dtype=float)


def _normalization_mask(energy: np.ndarray, x0, x1) -> np.ndarray:
    if x1 is None:
        return np.isfinite(energy)
    if isinstance(x0, (list, tuple, np.ndarray)):
        if not isinstance(x1, (list, tuple, np.ndarray)):
            raise ValueError("x0 and x1 must both be lists/arrays when using multiple ranges.")
        if len(x0) != len(x1):
            raise ValueError("x0 and x1 must have the same length.")
        mask = np.zeros_like(energy, dtype=bool)
        for lo, hi in zip(x0, x1):
            lo, hi = sorted((float(lo), float(hi)))
            mask |= (energy >= lo) & (energy <= hi)
        return mask
    lo, hi = sorted((float(x0), float(x1)))
    return (energy >= lo) & (energy <= hi)


def norm_spec(
    spectrum: np.ndarray,
    x0: Optional[Union[float, list]] = None,
    x1: Optional[Union[float, list]] = None,
    show: bool = False,
    robust_percentile: Optional[tuple[float, float]] = None,
    **kwargs,
) -> np.ndarray:
    """Min-max normalise a spectrum's intensity to [0, 1]."""
    spec = spec_shaper(spectrum)
    energy = np.asarray(spec[0], dtype=float)
    intensity = np.asarray(spec[1], dtype=float)

    mask = _normalization_mask(energy, x0, x1)
    ref = intensity[mask & np.isfinite(intensity)]
    if ref.size < 2:
        ref = intensity[np.isfinite(intensity)]
    if ref.size < 2:
        raise ValueError("Not enough finite points for normalization.")

    if robust_percentile is not None:
        p0, p1 = robust_percentile
        min_val, max_val = np.percentile(ref, [p0, p1])
    else:
        min_val, max_val = np.min(ref), np.max(ref)

    denom = max_val - min_val
    if abs(denom) < 1e-12:
        intensity_norm = np.zeros_like(intensity)
    else:
        intensity_norm = (intensity - min_val) / denom
    intensity_norm = np.nan_to_num(intensity_norm, nan=0.0, posinf=1.0, neginf=0.0)

    if show:
        _plot_lines(
            traces=[{"x": energy, "y": intensity_norm, "name": "normalized", "style": kwargs}],
            title="Normalized spectrum",
            x_label="Energy / Pixel",
            y_label="Normalized intensity",
        )

    return spec_wrapper(energy, intensity_norm, output=spectrum.shape)


def interpt_spec(
    spec: np.ndarray,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    pnts: int = 3000,
) -> np.ndarray:
    """Interpolate a spectrum onto a uniform grid."""
    spec = spec_shaper(spec)
    x = np.asarray(spec[0], dtype=float)
    y = np.asarray(spec[1], dtype=float)

    # scipy interp1d requires monotonic x; duplicate x values are removed.
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    x_unique, uniq_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[uniq_idx]

    if x_min is None:
        x_min = float(x_unique.min())
    if x_max is None:
        x_max = float(x_unique.max())
    new_x = np.linspace(x_min, x_max, pnts, endpoint=False)
    f = si.interp1d(x_unique, y_unique, bounds_error=False, fill_value="extrapolate")
    return np.array([new_x, f(new_x)], dtype=float)


def spec_cropping(
    spec: np.ndarray,
    crop_E1: float,
    crop_E2: float,
    show: bool = False,
) -> np.ndarray:
    """Crop a spectrum to a contiguous energy range."""
    spec = spec_shaper(spec)
    lo, hi = sorted((crop_E1, crop_E2))
    idx = np.where((spec[0] >= lo) & (spec[0] <= hi))[0]
    if idx.size == 0:
        raise ValueError(f"No points found in crop range [{lo}, {hi}].")
    contiguous = idx[:1]
    for k in idx[1:]:
        if k == contiguous[-1] + 1:
            contiguous = np.append(contiguous, k)
        else:
            break
    cropped = spec[:, contiguous]
    if show:
        _plot_lines(
            traces=[{"x": cropped[0], "y": cropped[1], "name": "cropped"}],
            title=f"Cropped [{lo}, {hi}]",
            x_label="Energy / Pixel",
            y_label="Intensity",
        )
    return cropped


def spec_save(
    spec: np.ndarray,
    crop_E1: Optional[float] = None,
    crop_E2: Optional[float] = None,
    save_name: str = "cropped_spec",
    show: bool = True,
) -> np.ndarray:
    """Save a spectrum to a plain-text file."""
    from .utils import date_today

    spec = spec_shaper(spec)
    if crop_E1 is not None:
        spec = spec_cropping(spec, crop_E1, crop_E2, show=False)

    os.makedirs("saved_spec", exist_ok=True)
    file_name = os.path.join("saved_spec", f"{date_today()}_{save_name}.dat")
    print(f"Saving {file_name}")
    np.savetxt(file_name, spec.T)

    if show:
        _plot_lines(
            traces=[{"x": spec[0], "y": spec[1], "name": save_name}],
            title="Saved spectrum",
            x_label="Energy / Pixel",
            y_label="Intensity",
        )
    return spec


def _safe_savgol(y: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    from scipy.signal import savgol_filter

    w = max(int(window_length), int(polyorder) + 2)
    if w % 2 == 0:
        w += 1
    w = min(w, y.size - 1 if y.size % 2 == 0 else y.size)
    if w <= polyorder:
        w = polyorder + 3
        if w % 2 == 0:
            w += 1
    if w >= y.size:
        w = y.size - 1 if y.size % 2 == 0 else y.size
    if w < 3:
        return y
    return savgol_filter(y, window_length=w, polyorder=min(polyorder, w - 1))


def peak_finder(
    spec: np.ndarray,
    spec_min: Optional[float] = None,
    spec_max: Optional[float] = None,
    peak_n: Optional[int] = None,
    prominence: float = 0.01,
    filtering: bool = False,
    show: bool = True,
    include_troughs: bool = True,
    **kwargs,
) -> tuple:
    """Find major peaks (and optionally troughs) in a spectrum."""
    from scipy import signal

    spec = spec_shaper(spec)
    x = np.asarray(spec[0], dtype=float)
    y = np.asarray(spec[1], dtype=float)

    if spec_min is None:
        search_mask = np.ones_like(x, dtype=bool)
    else:
        lo, hi = sorted((float(spec_min), float(spec_max)))
        search_mask = (x >= lo) & (x <= hi)

    x_search = x[search_mask]
    y_search = y[search_mask]
    if x_search.size < 3:
        return np.array([], dtype=int), np.array([], dtype=float)

    if filtering:
        y_work = _safe_savgol(
            y_search,
            window_length=int(kwargs.pop("window_length", 9)),
            polyorder=int(kwargs.pop("polyorder", 2)),
        )
    else:
        y_work = y_search.copy()

    peaks_pos, props_pos = signal.find_peaks(y_work, prominence=prominence)
    all_idx = [peaks_pos]
    all_prom = [props_pos.get("prominences", np.zeros_like(peaks_pos, dtype=float))]

    if include_troughs:
        peaks_neg, props_neg = signal.find_peaks(-y_work, prominence=prominence)
        all_idx.append(peaks_neg)
        all_prom.append(props_neg.get("prominences", np.zeros_like(peaks_neg, dtype=float)))

    idx = np.hstack(all_idx).astype(int)
    prom = np.hstack(all_prom).astype(float)
    if idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if peak_n is not None and idx.size > peak_n:
        keep = np.argsort(prom)[-int(peak_n) :]
        idx = idx[keep]
        prom = prom[keep]

    idx = np.sort(idx)
    full_indices = np.where(search_mask)[0][idx]
    peak_x = x[full_indices]

    if show:
        traces = [
            {"x": x, "y": y, "name": "original"},
            {"x": x_search, "y": y_work, "name": "filtered / search"},
            {
                "x": peak_x,
                "y": y[full_indices],
                "name": "detected peaks",
                "mode": "markers",
                "marker": {"color": "red", "size": 9},
            },
        ]
        _plot_lines(traces, title="Peak finding", x_label="Energy / Pixel", y_label="Intensity")

    return full_indices, peak_x


def find_edge_jump(
    spec: np.ndarray,
    show: bool = True,
    prominence: float = 0.005,
) -> int:
    """Locate edge jump index by maximum first derivative."""
    from scipy.signal import find_peaks

    spec = spec_shaper(spec)
    x = np.asarray(spec[0], dtype=float)
    y = np.asarray(spec[1], dtype=float)
    if x.size < 5:
        return int(np.argmax(y))

    dy = np.gradient(y, x)
    dy_f = _safe_savgol(dy, window_length=11, polyorder=2)
    if np.nanmax(np.abs(dy_f)) > 0:
        dy_norm = dy_f / np.nanmax(np.abs(dy_f))
    else:
        dy_norm = dy_f

    peaks, props = find_peaks(dy_norm, prominence=prominence)
    if peaks.size > 0:
        best = peaks[np.argmax(props["prominences"])]
    else:
        best = int(np.nanargmax(dy_norm))

    if show:
        traces = [
            {"x": x, "y": y, "name": "spectrum"},
            {"x": x, "y": 0.5 * dy_norm, "name": "0.5 * normalized derivative"},
            {
                "x": np.array([x[best]]),
                "y": np.array([y[best]]),
                "name": "edge",
                "mode": "markers",
                "marker": {"color": "red", "size": 10},
            },
        ]
        _plot_lines(traces, title="Edge jump", x_label="Energy / Pixel", y_label="Signal")

    return int(best)


def find_edge_pnts(
    spec: np.ndarray,
    y_pnts: np.ndarray,
    edge_min: Optional[float] = None,
    edge_max: Optional[float] = None,
    show: bool = True,
) -> np.ndarray:
    """Interpolate energy/pixel values at requested intensity levels."""
    spec = spec_shaper(spec)
    x = np.asarray(spec[0], dtype=float)
    y = np.asarray(spec[1], dtype=float)

    if edge_min is not None:
        lo, hi = sorted((float(edge_min), float(edge_max)))
        m = (x >= lo) & (x <= hi)
        x = x[m]
        y = y[m]
    if x.size < 2:
        raise ValueError("Not enough points in selected edge range.")

    # Interpolate x(y); enforce monotonic y for stable interpolation.
    order = np.argsort(y)
    y_sorted = y[order]
    x_sorted = x[order]
    y_unique, idx = np.unique(y_sorted, return_index=True)
    x_unique = x_sorted[idx]
    x_pnts = np.interp(np.asarray(y_pnts, dtype=float), y_unique, x_unique)

    if show:
        traces = [
            {"x": spec[0], "y": spec[1], "name": "spectrum"},
            {
                "x": x_pnts,
                "y": y_pnts,
                "name": "edge points",
                "mode": "markers",
                "marker": {"color": "red", "size": 9},
            },
        ]
        _plot_lines(traces, title="Edge point interpolation", x_label="Energy / Pixel", y_label="Intensity")

    return x_pnts


def intensity_at_energy(
    energy: np.ndarray,
    spec_1d: np.ndarray,
    E_eV: float,
) -> float:
    """Interpolate spectrum intensity at a specific energy."""
    from scipy.interpolate import interp1d

    f = interp1d(np.asarray(energy, dtype=float), np.asarray(spec_1d, dtype=float), bounds_error=False, fill_value="extrapolate")
    return float(f(E_eV))


def atten_slope_corr(
    element: int,
    data_x: np.ndarray,
    E_threshold: float = 0,
    show: bool = True,
) -> np.ndarray:
    """Compute attenuation-slope correction using xraylib cross-sections."""
    import xraylib
    from scipy.interpolate import interp1d

    energy_keV = np.arange(23, 28, 0.005)
    density = xraylib.ElementDensity(element)
    cs = np.array([xraylib.CS_Total(element, E) for E in energy_keV]) * density
    cs_norm = (cs - cs.min()) / (cs.max() - cs.min())

    correction = interp1d(energy_keV, cs_norm, bounds_error=False, fill_value="extrapolate")(
        np.asarray(data_x, dtype=float) / 1000.0
    )
    correction = np.asarray(correction, dtype=float)
    correction[data_x < E_threshold] = 1.0

    if show:
        _plot_lines(
            traces=[{"x": data_x, "y": correction, "name": "correction"}],
            title="Attenuation slope correction",
            x_label="Energy (eV)",
            y_label="Correction factor",
        )

    return correction
