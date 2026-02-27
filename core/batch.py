"""Core batch helpers (no plotting/UI)."""

from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime
from typing import Optional, Sequence

import h5py
import numpy as np
import scipy.ndimage as nd

from .analysis import XAS_spec
from .calibration import EDXAS_Calibrate
from .data_io import load_nexus_entry
from .preprocessing import pre_process
from .spectrum import find_edge_pnts, norm_spec, peak_finder, spec_shaper

__all__ = [
    "norm_spec_preview",
    "find_h5_files",
    "find_nearest_flatfield",
    "calibrate_from_reference_foil",
    "save_calibration_model",
    "load_calibration_model",
    "apply_calibration_to_scan",
]

_DEFAULT_FACTOR = 200.0


def norm_spec_preview(
    spec: np.ndarray,
    x1: int,
    x2: int,
    factor: float = _DEFAULT_FACTOR,
) -> np.ndarray:
    """Normalize a 1-D spectrum by its value range within a pixel window."""
    spec = np.nan_to_num(
        spec,
        nan=float(np.nanmean(spec)),
        posinf=float(np.nanmax(spec)),
        neginf=float(np.nanmin(spec)),
    )
    ref = spec[x1:x2]
    smin, smax = float(np.min(ref)), float(np.max(ref))
    return ((spec - smin) / (smax - smin + 1e-12)) * factor


def _parse_timestamp_from_name(path: str) -> Optional[datetime]:
    m = re.search(r"(\d{8})_(\d{4})", os.path.basename(path))
    if m is None:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")


def find_h5_files(
    folder: str,
    include: Optional[Sequence[str] | str] = None,
    exclude: Optional[Sequence[str] | str] = None,
) -> list[str]:
    """List HDF5 files in *folder* filtered by include/exclude keywords."""
    files = sorted(glob.glob(os.path.join(folder, "*.h5")))
    if include is not None:
        include_list = [include] if isinstance(include, str) else list(include)
        include_list = [s.lower() for s in include_list]
        files = [
            f
            for f in files
            if all(k in os.path.basename(f).lower() for k in include_list)
        ]
    if exclude is not None:
        exclude_list = [exclude] if isinstance(exclude, str) else list(exclude)
        exclude_list = [s.lower() for s in exclude_list]
        files = [
            f
            for f in files
            if all(k not in os.path.basename(f).lower() for k in exclude_list)
        ]
    return files


def find_nearest_flatfield(
    reference_path: str,
    folder: Optional[str] = None,
    candidates: Optional[Sequence[str]] = None,
    flat_keywords: Sequence[str] = ("ff", "flatfield", "flat"),
) -> str:
    """Find the flat-field file closest in timestamp to a reference scan."""
    ref_ts = _parse_timestamp_from_name(reference_path)
    if ref_ts is None:
        raise ValueError(f"Could not parse timestamp from: {reference_path}")

    if candidates is None:
        if folder is None:
            folder = os.path.dirname(reference_path)
        candidates = find_h5_files(folder)

    ref_base = os.path.abspath(reference_path)
    keys = tuple(k.lower() for k in flat_keywords)
    scored = []
    for path in candidates:
        abs_path = os.path.abspath(path)
        if abs_path == ref_base:
            continue
        name = os.path.basename(abs_path).lower()
        if not any(k in name for k in keys):
            continue
        ts = _parse_timestamp_from_name(abs_path)
        if ts is None:
            continue
        dt = abs((ts - ref_ts).total_seconds())
        scored.append((dt, abs_path))
    if not scored:
        raise FileNotFoundError(
            f"No flatfield-like file found near {reference_path} in {folder or 'candidate list'}"
        )
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _build_row_mask(shape: tuple[int, int], row_range: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r0, r1 = sorted((int(row_range[0]), int(row_range[1])))
    r0 = int(np.clip(r0, 0, h))
    r1 = int(np.clip(r1, 0, h))
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, :] = True
    return m


def calibrate_from_reference_foil(
    foil_path: str,
    flat_path: str,
    standard_spec: np.ndarray,
    row_range: tuple[int, int] = (155, 235),
    denoise_size: int = 3,
    median_size: int = 3,
    gaussian_sigma: float = 0.2,
    norm_range_pixels: tuple[int, int] = (50, 130),
    interp_pts: int = 1000,
    y_points: Sequence[float] = (0.2, 0.7),
    exp_edge_range_pixels: tuple[float, float] = (0, 200),
    exp_peak_range_pixels: tuple[float, float] = (100, 700),
    peak_n: int = 8,
    peak_prominence: float = 0.01,
    standard_norm_range_eV: tuple[float, float] = (8950, 9050),
    standard_edge_range_eV: tuple[float, float] = (8900, 9050),
    standard_peak_range_eV: tuple[float, float] = (8950, 9300),
    poly_order: int = 2,
    show: bool = True,
    save_param: bool = False,
) -> tuple[EDXAS_Calibrate, dict]:
    """Fit a pixel->energy calibration from a Cu foil and a reference standard."""
    foil = load_nexus_entry(foil_path)["data"]
    flat = load_nexus_entry(flat_path)["data"]

    processed = pre_process(
        np.mean(foil, axis=0),
        np.mean(flat, axis=0),
        denoise_size=denoise_size,
        savedata=False,
        prefix="",
    )
    mux = np.asarray(processed["mux"], dtype=float)
    m = _build_row_mask(mux.shape, row_range=row_range)

    xas = XAS_spec(mux, m=m)
    if median_size > 1:
        xas.median_filtering(size=median_size)
    if gaussian_sigma > 0:
        xas.gaussian_filtering(sigma=gaussian_sigma)
    xas.spec_generate(show=show)
    xas.spec_normalize(
        norm_range_pixels[0],
        norm_range_pixels[1],
        interp_pts,
        show=show,
    )
    y_pts = np.asarray(y_points, dtype=float)
    xas.find_edge(y_pts, exp_edge_range_pixels[0], exp_edge_range_pixels[1], show=show)
    xas.find_peaks(
        exp_peak_range_pixels[0],
        exp_peak_range_pixels[1],
        peak_n=peak_n,
        show=show,
        filtering=True,
        window_length=25,
        polyorder=2,
        prominence=peak_prominence,
    )

    standard = spec_shaper(np.asarray(standard_spec, dtype=float))
    standard_n = norm_spec(
        standard,
        standard_norm_range_eV[0],
        standard_norm_range_eV[1],
        show=show,
    )
    std_peaks = peak_finder(
        standard_n,
        standard_peak_range_eV[0],
        standard_peak_range_eV[1],
        peak_n=peak_n,
        show=show,
        filtering=True,
        window_length=41,
        polyorder=2,
        prominence=peak_prominence,
    )
    std_edge = find_edge_pnts(
        standard_n,
        y_pts,
        standard_edge_range_eV[0],
        standard_edge_range_eV[1],
        show=show,
    )
    target = np.insert(std_peaks[1], 0, std_edge)
    train = np.asarray(xas.train[0], dtype=float)

    n = min(train.size, target.size)
    fit = EDXAS_Calibrate(
        xas.norm_spec[0],
        standard_n,
        train[:n],
        target[:n],
        order=poly_order,
        show=show,
        save_param=save_param,
    )
    meta = {
        "foil_path": os.path.abspath(foil_path),
        "flat_path": os.path.abspath(flat_path),
        "row_range": [int(row_range[0]), int(row_range[1])],
        "norm_range_pixels": [int(norm_range_pixels[0]), int(norm_range_pixels[1])],
        "poly_order": int(poly_order),
        "rmse": float(fit.rmse),
    }
    return fit, meta


def save_calibration_model(
    file_path: str,
    calibration: EDXAS_Calibrate | np.ndarray,
    metadata: Optional[dict] = None,
) -> str:
    """Save calibration coefficients and metadata to a JSON file."""
    if hasattr(calibration, "coef"):
        coef = np.asarray(calibration.coef, dtype=float)
        order = int(getattr(calibration, "order", len(coef) - 1))
        rmse = float(getattr(calibration, "rmse", np.nan))
    else:
        coef = np.asarray(calibration, dtype=float).reshape(-1)
        order = int(len(coef) - 1)
        rmse = float("nan")

    payload = {
        "coef": coef.tolist(),
        "order": order,
        "rmse": rmse,
        "metadata": metadata or {},
    }
    directory = os.path.dirname(os.path.abspath(file_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return os.path.abspath(file_path)


def load_calibration_model(file_path: str) -> dict:
    """Load a calibration model saved by :func:`save_calibration_model`."""
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["coef"] = np.asarray(payload["coef"], dtype=float)
    return payload


def _normalize_spectra_chunk(
    specs: np.ndarray,
    norm_range_pixels: Optional[tuple[int, int]],
) -> np.ndarray:
    if norm_range_pixels is None:
        return specs
    x1, x2 = sorted((int(norm_range_pixels[0]), int(norm_range_pixels[1])))
    x1 = max(0, x1)
    x2 = min(specs.shape[1], x2)
    ref = specs[:, x1:x2]
    smin = np.nanmin(ref, axis=1, keepdims=True)
    smax = np.nanmax(ref, axis=1, keepdims=True)
    return (specs - smin) / (smax - smin + 1e-12)


def apply_calibration_to_scan(
    data_path: str,
    flat_path: str,
    calibration: EDXAS_Calibrate | dict | np.ndarray,
    row_range: tuple[int, int] = (155, 235),
    norm_range_pixels: Optional[tuple[int, int]] = (50, 130),
    denoise_size: int = 3,
    median_size: int = 3,
    gaussian_sigma: float = 0.2,
    chunk_size: int = 500,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    output_h5: Optional[str] = None,
    output_dtype: str = "float32",
) -> dict:
    """Apply a saved calibration to all spectra in a large scan efficiently."""
    if isinstance(calibration, dict):
        coef = np.asarray(calibration["coef"], dtype=float).reshape(-1)
    elif hasattr(calibration, "coef"):
        coef = np.asarray(calibration.coef, dtype=float).reshape(-1)
    else:
        coef = np.asarray(calibration, dtype=float).reshape(-1)

    with h5py.File(data_path, "r") as f_data, h5py.File(flat_path, "r") as f_flat:
        dset = f_data["/entry/data/data"]
        fset = f_flat["/entry/data/data"]
        n_total, h, w = dset.shape
        if end_frame is None:
            end_frame = n_total
        start_frame = int(np.clip(start_frame, 0, n_total))
        end_frame = int(np.clip(end_frame, start_frame, n_total))
        n_frames = end_frame - start_frame

        r0, r1 = sorted((int(row_range[0]), int(row_range[1])))
        r0 = int(np.clip(r0, 0, h))
        r1 = int(np.clip(r1, r0 + 1, h))

        flat_avg = np.mean(fset[:, r0:r1, :], axis=0, dtype=np.float64)
        if denoise_size > 1:
            flat_avg = nd.median_filter(flat_avg, size=denoise_size)
        flat_avg = np.clip(flat_avg, 1e-6, None)

        pixels = np.arange(w, dtype=float)
        energy = np.polyval(coef, pixels)

        if output_h5 is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_h5)), exist_ok=True)
            fout = h5py.File(output_h5, "w")
            ds_spec = fout.create_dataset(
                "spectra",
                shape=(n_frames, w),
                dtype=output_dtype,
                compression="gzip",
                chunks=(min(chunk_size, max(1, n_frames)), w),
            )
            fout.create_dataset("energy", data=energy.astype(np.float64))
            fout.create_dataset("pixel", data=pixels.astype(np.float64))
            fout.attrs["source_data"] = os.path.abspath(data_path)
            fout.attrs["source_flat"] = os.path.abspath(flat_path)
            fout.attrs["start_frame"] = start_frame
            fout.attrs["end_frame"] = end_frame
        else:
            fout = None
            ds_spec = None
            chunks = []

        try:
            out_i = 0
            for s in range(start_frame, end_frame, chunk_size):
                e = min(s + chunk_size, end_frame)
                data_chunk = np.asarray(dset[s:e, r0:r1, :], dtype=np.float64)
                if denoise_size > 1:
                    data_chunk = nd.median_filter(
                        data_chunk, size=(1, denoise_size, denoise_size)
                    )
                data_chunk = np.clip(data_chunk, 1e-6, None)
                mux = np.log(flat_avg[None, :, :] / data_chunk)

                if median_size > 1:
                    mux = nd.median_filter(mux, size=(1, median_size, median_size))
                if gaussian_sigma > 0:
                    mux = nd.gaussian_filter(
                        mux, sigma=(0.0, float(gaussian_sigma), float(gaussian_sigma))
                    )

                specs = np.mean(mux, axis=1)
                specs = _normalize_spectra_chunk(specs, norm_range_pixels)
                specs = np.nan_to_num(specs, nan=0.0, posinf=1.0, neginf=0.0)

                if ds_spec is not None:
                    ds_spec[out_i : out_i + specs.shape[0]] = specs.astype(output_dtype)
                else:
                    chunks.append(specs.astype(np.float64))
                out_i += specs.shape[0]
                print(f"Processed frames {s}:{e}")
        finally:
            if fout is not None:
                fout.close()

    out = {
        "energy": energy,
        "coef": coef,
        "n_frames": n_frames,
        "row_range": (r0, r1),
    }
    if output_h5 is not None:
        out["output_h5"] = os.path.abspath(output_h5)
    else:
        out["spectra"] = np.vstack(chunks) if chunks else np.empty((0, w))
    return out
