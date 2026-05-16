"""ROI helpers for horizontal and tilted beam bands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import h5py
import numpy as np
import scipy.ndimage as nd

__all__ = [
    "build_roi_mask",
    "fit_tilted_band_roi",
    "infer_tilted_band_roi_from_paths",
    "load_roi_json",
    "make_tilted_band_roi",
    "normalize_roi_spec",
    "prepare_roi_weights",
    "roi_boundary_rows",
    "roi_row_bounds",
    "roi_weighted_column_mean",
    "save_roi_json",
    "tilted_band_controls_from_roi",
]


def _clip_row_range(shape: tuple[int, int], row_range: tuple[int, int]) -> tuple[int, int]:
    h, _w = shape
    r0, r1 = sorted((int(row_range[0]), int(row_range[1])))
    r0 = int(np.clip(r0, 0, h))
    r1 = int(np.clip(r1, r0 + 1, h))
    return r0, r1


def _tilted_row_bounds(shape: tuple[int, int], spec: Mapping[str, object]) -> tuple[int, int]:
    h, w = shape
    cols = np.arange(w, dtype=float)
    center0 = float(spec["center_row_at_col0"])
    slope = float(spec["slope_per_col"])
    half_width = max(0.5, float(spec["half_width"]))
    center = center0 + slope * cols
    top = np.floor(center - half_width)
    bottom = np.ceil(center + half_width + 1.0)
    r0 = int(np.clip(np.min(top), 0, h))
    r1 = int(np.clip(np.max(bottom), r0 + 1, h))
    return r0, r1


def make_tilted_band_roi(
    shape: tuple[int, int],
    left_center_row: float,
    right_center_row: float,
    half_width: float,
) -> dict[str, object]:
    """Build a normalized tilted-band ROI from left/right edge center rows."""
    h, w = shape
    max_row = max(0.0, float(h - 1))
    left = float(np.clip(left_center_row, 0.0, max_row))
    right = float(np.clip(right_center_row, 0.0, max_row))
    width = max(0.5, float(half_width))
    slope = 0.0 if w <= 1 else (right - left) / float(w - 1)
    return normalize_roi_spec(
        shape,
        roi={
            "kind": "tilted_band",
            "center_row_at_col0": left,
            "slope_per_col": slope,
            "half_width": width,
        },
    )


def tilted_band_controls_from_roi(
    shape: tuple[int, int],
    roi: Optional[Mapping[str, object]] = None,
    row_range: Optional[tuple[int, int]] = None,
) -> tuple[float, float, float]:
    """Return ``(left_center_row, right_center_row, half_width)`` controls."""
    spec = normalize_roi_spec(shape, row_range=row_range, roi=roi)
    h, w = shape
    max_row = max(0.0, float(h - 1))

    if spec["kind"] == "row_range":
        row_start = float(spec["row_start"])
        row_stop = float(spec["row_stop"])
        center = 0.5 * (row_start + max(row_start, row_stop - 1.0))
        half_width = max(0.5, 0.5 * max(1.0, row_stop - row_start))
        center = float(np.clip(center, 0.0, max_row))
        return center, center, half_width

    center0 = float(np.clip(spec["center_row_at_col0"], 0.0, max_row))
    center1 = center0 + float(spec["slope_per_col"]) * max(0, w - 1)
    center1 = float(np.clip(center1, 0.0, max_row))
    half_width = max(0.5, float(spec["half_width"]))
    return center0, center1, half_width


def normalize_roi_spec(
    shape: tuple[int, int],
    row_range: Optional[tuple[int, int]] = None,
    roi: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    """Return a validated ROI spec for *shape*.

    Parameters
    ----------
    shape:
        Image shape ``(rows, cols)``.
    row_range:
        Backward-compatible horizontal row band.
    roi:
        Optional ROI mapping. Supported kinds:
        - ``{"kind": "row_range", "row_start": ..., "row_stop": ...}``
        - ``{"kind": "tilted_band", "center_row_at_col0": ..., "slope_per_col": ..., "half_width": ...}``
    """
    if roi is None:
        if row_range is None:
            raise ValueError("Either row_range or roi must be provided.")
        r0, r1 = _clip_row_range(shape, row_range)
        return {
            "kind": "row_range",
            "row_start": int(r0),
            "row_stop": int(r1),
            "row_bounds": [int(r0), int(r1)],
        }

    spec = dict(roi)
    kind = str(spec.get("kind", "tilted_band"))

    if kind == "row_range":
        if "row_start" in spec and "row_stop" in spec:
            r0, r1 = _clip_row_range(shape, (int(spec["row_start"]), int(spec["row_stop"])))
        elif "row_bounds" in spec:
            bounds = spec["row_bounds"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError("row_bounds must be a length-2 sequence.")
            r0, r1 = _clip_row_range(shape, (int(bounds[0]), int(bounds[1])))
        elif "row_range" in spec:
            bounds = spec["row_range"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError("row_range must be a length-2 sequence.")
            r0, r1 = _clip_row_range(shape, (int(bounds[0]), int(bounds[1])))
        else:
            raise ValueError("row_range ROI requires row_start/row_stop or row_bounds.")
        return {
            "kind": "row_range",
            "row_start": int(r0),
            "row_stop": int(r1),
            "row_bounds": [int(r0), int(r1)],
        }

    if kind != "tilted_band":
        raise ValueError(f"Unsupported ROI kind: {kind!r}")

    missing = [k for k in ("center_row_at_col0", "slope_per_col", "half_width") if k not in spec]
    if missing:
        raise ValueError(f"Tilted ROI is missing required keys: {missing}")

    out = {
        "kind": "tilted_band",
        "center_row_at_col0": float(spec["center_row_at_col0"]),
        "slope_per_col": float(spec["slope_per_col"]),
        "half_width": max(0.5, float(spec["half_width"])),
    }
    r0, r1 = _tilted_row_bounds(shape, out)
    out["row_bounds"] = [int(r0), int(r1)]
    if "threshold_fraction" in spec:
        out["threshold_fraction"] = float(spec["threshold_fraction"])
    if "shrink_fraction" in spec:
        out["shrink_fraction"] = float(spec["shrink_fraction"])
    if "smooth_sigma_rows" in spec:
        out["smooth_sigma_rows"] = float(spec["smooth_sigma_rows"])
    if "smooth_sigma_cols" in spec:
        out["smooth_sigma_cols"] = float(spec["smooth_sigma_cols"])
    return out


def roi_row_bounds(
    shape: tuple[int, int],
    row_range: Optional[tuple[int, int]] = None,
    roi: Optional[Mapping[str, object]] = None,
) -> tuple[int, int]:
    spec = normalize_roi_spec(shape, row_range=row_range, roi=roi)
    bounds = spec["row_bounds"]
    return int(bounds[0]), int(bounds[1])


def roi_boundary_rows(
    shape: tuple[int, int],
    row_range: Optional[tuple[int, int]] = None,
    roi: Optional[Mapping[str, object]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cols, top_rows, bottom_rows)`` for plotting."""
    h, w = shape
    spec = normalize_roi_spec(shape, row_range=row_range, roi=roi)
    cols = np.arange(w, dtype=float)

    if spec["kind"] == "row_range":
        top = np.full(w, float(spec["row_start"]), dtype=float)
        bottom = np.full(w, float(spec["row_stop"]) - 1.0, dtype=float)
        return cols, top, bottom

    center = float(spec["center_row_at_col0"]) + float(spec["slope_per_col"]) * cols
    half_width = float(spec["half_width"])
    top = np.clip(center - half_width, 0.0, float(h - 1))
    bottom = np.clip(center + half_width, 0.0, float(h - 1))
    return cols, top, bottom


def build_roi_mask(
    shape: tuple[int, int],
    row_range: Optional[tuple[int, int]] = None,
    roi: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """Build a boolean ROI mask for an image ``(rows, cols)``."""
    h, w = shape
    spec = normalize_roi_spec(shape, row_range=row_range, roi=roi)
    rows = np.arange(h, dtype=float)[:, None]

    if spec["kind"] == "row_range":
        r0 = int(spec["row_start"])
        r1 = int(spec["row_stop"])
        mask = np.zeros((h, w), dtype=bool)
        mask[r0:r1, :] = True
        return mask

    cols = np.arange(w, dtype=float)[None, :]
    center = float(spec["center_row_at_col0"]) + float(spec["slope_per_col"]) * cols
    half_width = float(spec["half_width"])
    return (rows >= (center - half_width)) & (rows < (center + half_width + 1.0))


def prepare_roi_weights(
    shape: tuple[int, int],
    row_range: Optional[tuple[int, int]] = None,
    roi: Optional[Mapping[str, object]] = None,
    dtype: np.dtype | type = np.float32,
) -> tuple[dict[str, object], tuple[int, int], np.ndarray, np.ndarray]:
    """Return ``(roi_spec, row_bounds, row_weights, col_weight_sum)``."""
    roi_spec = normalize_roi_spec(shape, row_range=row_range, roi=roi)
    r0, r1 = roi_row_bounds(shape, roi=roi_spec)
    row_weights = build_roi_mask(shape, roi=roi_spec)[r0:r1, :].astype(dtype, copy=False)
    col_weight_sum = np.clip(row_weights.sum(axis=0, keepdims=True), 1.0, None).astype(dtype, copy=False)
    return roi_spec, (r0, r1), row_weights, col_weight_sum


def roi_weighted_column_mean(
    image: np.ndarray,
    row_range: Optional[tuple[int, int]] = None,
    roi: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """Average image rows inside the ROI for each detector column."""
    arr = np.asarray(image, dtype=np.float32)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None, :, :]
        squeeze = True
    if arr.ndim != 3:
        raise ValueError("image must be 2-D or 3-D with shape (..., rows, cols)")

    _roi_spec, (r0, r1), row_weights, col_weight_sum = prepare_roi_weights(
        arr.shape[-2:],
        row_range=row_range,
        roi=roi,
    )
    arr_roi = arr[:, r0:r1, :]
    out = (arr_roi * row_weights[None, :, :]).sum(axis=1) / col_weight_sum
    return out[0] if squeeze else out


def save_roi_json(
    path: str | Path,
    roi: Mapping[str, object],
    metadata: Optional[Mapping[str, object]] = None,
) -> str:
    """Save ROI spec to a JSON file for reuse across scans."""
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"roi": dict(roi)}
    if metadata:
        payload["metadata"] = dict(metadata)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out_path)


def load_roi_json(path: str | Path) -> dict[str, object]:
    """Load ROI spec from a JSON file saved by :func:`save_roi_json`."""
    in_path = Path(path).expanduser().resolve()
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    roi = payload.get("roi", payload) if isinstance(payload, dict) else payload
    if not isinstance(roi, dict):
        raise ValueError(f"ROI JSON does not contain a ROI mapping: {in_path}")
    return dict(roi)


def fit_tilted_band_roi(
    image: np.ndarray,
    threshold_fraction: float = 0.55,
    shrink_fraction: float = 0.92,
    smooth_sigma_rows: float = 2.0,
    smooth_sigma_cols: float = 6.0,
    min_width: int = 8,
    min_valid_columns: int = 32,
) -> dict[str, object]:
    """Fit a tilted beam band from a representative 2-D mux image."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("image must be 2-D")

    clean = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    smooth = nd.gaussian_filter(
        clean,
        sigma=(float(smooth_sigma_rows), float(smooth_sigma_cols)),
        mode="nearest",
    )

    cols_keep: list[float] = []
    centers: list[float] = []
    widths: list[float] = []
    weights: list[float] = []

    for col in range(smooth.shape[1]):
        profile = smooth[:, col]
        if not np.any(np.isfinite(profile)):
            continue
        hi = float(np.nanpercentile(profile, 98))
        lo = float(np.nanpercentile(profile, 20))
        if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
            continue

        peak = int(np.nanargmax(profile))
        threshold = lo + float(threshold_fraction) * (hi - lo)
        active = np.flatnonzero(profile >= threshold)
        if active.size == 0:
            continue
        splits = np.where(np.diff(active) != 1)[0] + 1
        groups = np.split(active, splits)
        group = next((g for g in groups if int(g[0]) <= peak <= int(g[-1])), None)
        if group is None:
            group = max(groups, key=len)
        width = int(group[-1] - group[0] + 1)
        if width < int(min_width):
            continue

        cols_keep.append(float(col))
        centers.append(0.5 * (float(group[0]) + float(group[-1])))
        widths.append(float(width))
        weights.append(max(hi - lo, 1e-6))

    if len(cols_keep) < int(min_valid_columns):
        raise RuntimeError(
            f"Could not fit a tilted ROI: only {len(cols_keep)} valid columns found."
        )

    cols_arr = np.asarray(cols_keep, dtype=float)
    centers_arr = np.asarray(centers, dtype=float)
    widths_arr = np.asarray(widths, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)

    center_med = float(np.median(centers_arr))
    center_mad = float(np.median(np.abs(centers_arr - center_med)))
    if center_mad <= 1e-6:
        center_mad = 1.0
    width_med = float(np.median(widths_arr))
    width_mad = float(np.median(np.abs(widths_arr - width_med)))
    if width_mad <= 1e-6:
        width_mad = 1.0

    keep = (
        (np.abs(centers_arr - center_med) <= max(4.0 * center_mad, 12.0))
        & (np.abs(widths_arr - width_med) <= max(4.0 * width_mad, 12.0))
    )
    if np.count_nonzero(keep) >= int(min_valid_columns):
        cols_arr = cols_arr[keep]
        centers_arr = centers_arr[keep]
        widths_arr = widths_arr[keep]
        weights_arr = weights_arr[keep]

    slope, intercept = np.polyfit(cols_arr, centers_arr, deg=1, w=weights_arr)
    half_width = max(1.0, 0.5 * float(np.median(widths_arr)) * float(shrink_fraction))

    roi = {
        "kind": "tilted_band",
        "center_row_at_col0": float(intercept),
        "slope_per_col": float(slope),
        "half_width": float(half_width),
        "threshold_fraction": float(threshold_fraction),
        "shrink_fraction": float(shrink_fraction),
        "smooth_sigma_rows": float(smooth_sigma_rows),
        "smooth_sigma_cols": float(smooth_sigma_cols),
    }
    roi["row_bounds"] = list(_tilted_row_bounds(img.shape, roi))
    return roi


def infer_tilted_band_roi_from_paths(
    data_path: str,
    flat_path: str,
    frame_index: int = 0,
    frame_average: int = 1,
    threshold_fraction: float = 0.55,
    shrink_fraction: float = 0.92,
    smooth_sigma_rows: float = 2.0,
    smooth_sigma_cols: float = 6.0,
    median_size: int = 0,
) -> dict[str, object]:
    """Build a representative mux image from HDF5 paths and fit a tilted ROI."""
    with h5py.File(flat_path, "r") as f_flat:
        flat_mean = np.asarray(f_flat["/entry/data/data"][:], dtype=np.float32).mean(axis=0)

    with h5py.File(data_path, "r") as f_data:
        dset = f_data["/entry/data/data"]
        n_frames = int(dset.shape[0])
        frame_index = int(np.clip(frame_index, 0, max(0, n_frames - 1)))
        frame_average = max(1, int(frame_average))
        start = int(np.clip(frame_index, 0, n_frames - 1))
        stop = int(np.clip(start + frame_average, start + 1, n_frames))
        data_mean = np.asarray(dset[start:stop], dtype=np.float32).mean(axis=0)

    if int(median_size) > 1:
        k = int(median_size)
        if k % 2 == 0:
            k += 1
        flat_mean = nd.median_filter(flat_mean, size=(k, k), mode="nearest")
        data_mean = nd.median_filter(data_mean, size=(k, k), mode="nearest")

    with np.errstate(divide="ignore", invalid="ignore"):
        mux = np.log(np.clip(flat_mean, 1e-6, None) / np.clip(data_mean, 1e-6, None))
    mux[~np.isfinite(mux)] = 0.0
    roi = fit_tilted_band_roi(
        mux,
        threshold_fraction=threshold_fraction,
        shrink_fraction=shrink_fraction,
        smooth_sigma_rows=smooth_sigma_rows,
        smooth_sigma_cols=smooth_sigma_cols,
    )
    roi["preview_frame_index"] = int(frame_index)
    roi["preview_frame_average"] = int(frame_average)
    return roi
