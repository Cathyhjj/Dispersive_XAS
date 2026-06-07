#!/usr/bin/env python3
"""PyQtGraph viewer for the June 5 fly-scanning mapping HDF file.

This script mirrors the analysis in 20260604_fly_scanning.ipynb while keeping
the notebook unchanged. It fills acquired frames in horizontal-fast order:

    frame 0 -> row 0, col 0
    frame 1 -> row 0, col 1
    frame 1001 -> row 1, col 0
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi


DEFAULT_HDF = "202606050048-mappingovernight_fly_scanning-grid_fly_scan-5be3b2c1.hdf"
DEFAULT_FLAT = "202606041316-flatfield_lambda-count_multiple-4ec39cb3.hdf"

DETECTOR_KEY = "lambda_250k"
HORIZONTAL_MOTOR = "aerotech-horizontal"
VERTICAL_MOTOR = "aerotech-vertical"
IT_KEY = "It-count"

FAST_AXIS = "horizontal"
COORDS = "motor"
REDUCTION = "mean"
FRAME_AGGREGATION = "first"

PREVIEW_ROI_VERTICAL = (120, 200)
SPECTRUM_DIFFERENCE_X1 = 121
SPECTRUM_DIFFERENCE_X2 = 133
MUX_DERIVATIVE_X_RANGE = (120, 140)
DEFAULT_SPECTRUM_ROI_JSON = "../saved_rois/Pt_L3_roi.json"

TOP_N_REGIONS = 30
TOP_N_SOURCE = "lambda"
SNAKE_AXES = False
BINARY_THRESHOLD = 18.0
SEGMENTATION_MIN_REGION_PIXELS = 1
SEGMENTATION_SMOOTH_SIGMA = 0.0
SEGMENTATION_CONNECTIVITY = 2
REGION_CENTER_METHOD = "weighted"
REGION_RANK_METRIC = "peak"
PREVIEW_SPECTRUM = ("p1", "p3")


@dataclass
class EmbeddedGridRun:
    path: Path
    entry: str
    detector_key: str
    detector_path: str
    metadata: dict
    data_values: dict
    nframes: int
    frame_shape: tuple[int, int]
    data_keys: list[str]


@dataclass
class PreviewRecord:
    region: dict
    position: dict
    image: np.ndarray
    profile: np.ndarray
    roi_start: int
    roi_stop: int


@dataclass
class DerivativeRecord:
    region: dict
    position: dict
    x_values: np.ndarray
    derivative: np.ndarray
    peak_x: float
    peak_value: float


@dataclass
class AnalysisResults:
    grid_run: EmbeddedGridRun
    flat_run: EmbeddedGridRun
    metadata_grid_shape: tuple[int, int]
    plot_grid_shape: tuple[int, int]
    x_axis: np.ndarray
    y_axis: np.ndarray
    x_label: str
    y_label: str
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    lambda_grid: np.ndarray
    it_grid: np.ndarray
    difference_grid: np.ndarray
    difference_label: str
    difference_roi_vertical: tuple[int, int]
    difference_x_indices: tuple[int, int]
    spectrum_roi: dict
    binary_mask: np.ndarray
    masked_difference_grid: np.ndarray
    bright_region_mask: np.ndarray
    bright_region_labels: np.ndarray
    top_regions: list[dict]
    top_regions_by_difference: list[dict]
    preview_records: list[PreviewRecord]
    mux_records: list[PreviewRecord]
    mux_derivative_records: list[DerivativeRecord]
    mux_derivative_peak_x_grid: np.ndarray
    mux_derivative_peak_value_grid: np.ndarray
    mux_derivative_x_indices: tuple[int, int]
    flatfield_average: np.ndarray
    it_available: bool
    args: argparse.Namespace


def resolve_existing_path(path) -> Path:
    path = Path(path).expanduser()
    if path.exists():
        return path

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        script_dir = Path(__file__).resolve().parent
        search_roots = [Path.cwd(), *Path.cwd().parents, script_dir, *script_dir.parents]
        candidates.extend(root / path for root in search_roots)
        if path.parts and path.parts[0] == "data":
            tail = Path(*path.parts[1:])
            candidates.extend(root / "data" / tail for root in search_roots)
        candidates.extend([Path("..") / "data" / path.name, Path("data") / path.name])

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    tried = "\n".join(f"  - {candidate}" for candidate in candidates[:16])
    raise FileNotFoundError(f"{path} was not found. Tried:\n{tried}")


def data_path(filename_or_path) -> Path:
    path = Path(filename_or_path)
    if path.parts and path.parts[0] == "data":
        return resolve_existing_path(path)
    return resolve_existing_path(Path("data") / path)


def decode_hdf_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_metadata_value(value):
    value = decode_hdf_scalar(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    return value
    return value


def metadata_field(run: EmbeddedGridRun, name: str, default=None):
    return run.metadata.get(name, default)


def read_embedded_grid_run(path: Path, detector_key: str = DETECTOR_KEY) -> EmbeddedGridRun:
    path = resolve_existing_path(path)

    with h5py.File(path, "r") as h5:
        entry = decode_hdf_scalar(h5.attrs.get("default"))
        if not entry:
            entries = [name for name, obj in h5.items() if isinstance(obj, h5py.Group)]
            if len(entries) != 1:
                raise ValueError(f"Could not determine the run entry in {path}")
            entry = entries[0]

        data_group_path = f"{entry}/data"
        if data_group_path not in h5:
            raise KeyError(f"{data_group_path!r} not found in {path}")
        data_group = h5[data_group_path]

        detector_path = f"{data_group_path}/{detector_key}"
        if detector_path not in h5:
            primary_detector_path = f"{entry}/instrument/bluesky/streams/primary/{detector_key}/value"
            if primary_detector_path not in h5:
                raise KeyError(f"Could not find detector data for {detector_key!r} in {path}")
            detector_path = primary_detector_path

        detector_dataset = h5[detector_path]
        if detector_dataset.ndim != 3:
            raise ValueError(
                f"Expected detector data with shape (nframes, ny, nx); got {detector_dataset.shape!r}"
            )
        detector_shape = tuple(int(value) for value in detector_dataset.shape)

        metadata = {}
        metadata_group_path = f"{entry}/instrument/bluesky/metadata"
        if metadata_group_path in h5:
            metadata_group = h5[metadata_group_path]
            metadata = {
                key: parse_metadata_value(dataset[()])
                for key, dataset in metadata_group.items()
                if isinstance(dataset, h5py.Dataset)
            }

        data_values = {}
        data_keys = sorted(data_group.keys())
        for key, dataset in data_group.items():
            if key == detector_key or not isinstance(dataset, h5py.Dataset):
                continue
            if dataset.ndim == 1:
                data_values[key] = np.asarray(dataset[:], dtype=float)

    return EmbeddedGridRun(
        path=path,
        entry=entry,
        detector_key=detector_key,
        detector_path=detector_path,
        metadata=metadata,
        data_values=data_values,
        nframes=detector_shape[0],
        frame_shape=(detector_shape[1], detector_shape[2]),
        data_keys=data_keys,
    )


def grid_shape_from_metadata(run: EmbeddedGridRun) -> tuple[int, int]:
    shape = metadata_field(run, "start.shape")
    if shape is None:
        raise ValueError("Grid scan metadata does not contain start.shape.")
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D grid shape, got {shape!r}")
    return int(shape[0]), int(shape[1])


def detector_frame_grid_shape(
    run: EmbeddedGridRun, metadata_shape: tuple[int, int], fast_axis=FAST_AXIS
) -> tuple[int, int]:
    nframes = int(run.nframes)
    metadata_rows, metadata_cols = (int(metadata_shape[0]), int(metadata_shape[1]))
    metadata_cells = metadata_rows * metadata_cols
    fast_axis = str(fast_axis).strip().lower()
    if nframes == metadata_cells:
        return metadata_rows, metadata_cols
    if fast_axis in {"horizontal", "x", "columns", "cols"} and metadata_cols > 0 and nframes % metadata_cols == 0:
        return nframes // metadata_cols, metadata_cols
    if fast_axis in {"vertical", "y", "rows"} and metadata_rows > 0 and nframes % metadata_rows == 0:
        return metadata_rows, nframes // metadata_rows
    if fast_axis not in {"horizontal", "x", "columns", "cols", "vertical", "y", "rows"}:
        raise ValueError(f"Unsupported fast axis: {fast_axis!r}.")
    if metadata_cols > 0 and nframes % metadata_cols == 0:
        return nframes // metadata_cols, metadata_cols
    if metadata_rows > 0 and nframes % metadata_rows == 0:
        return metadata_rows, nframes // metadata_rows
    return nframes, 1


def commanded_axes_from_metadata(
    run: EmbeddedGridRun, shape: tuple[int, int], metadata_shape=None
) -> tuple[list[str], np.ndarray, np.ndarray]:
    motors = metadata_field(run, "start.motors", [VERTICAL_MOTOR, HORIZONTAL_MOTOR])
    extents = metadata_field(run, "start.extents")
    if extents is None or len(extents) != 2:
        raise ValueError("Grid scan metadata does not contain two motor extents.")
    if len(motors) != 2:
        raise ValueError(f"Expected two motors in metadata, got {motors!r}")

    full_shape = tuple(int(value) for value in (metadata_shape or shape))
    axes_by_motor = {}
    for motor, extent, count in zip(motors, extents, full_shape):
        axes_by_motor[str(motor)] = np.linspace(float(extent[0]), float(extent[1]), int(count))

    if HORIZONTAL_MOTOR not in axes_by_motor or VERTICAL_MOTOR not in axes_by_motor:
        raise ValueError(f"Metadata motors {motors!r} do not include expected motors.")

    def acquired_axis(full_axis, count):
        full_axis = np.asarray(full_axis, dtype=float)
        count = int(count)
        if count <= full_axis.size:
            return full_axis[:count]
        return np.linspace(float(full_axis[0]), float(full_axis[-1]), count)

    y_axis = acquired_axis(axes_by_motor[VERTICAL_MOTOR], int(shape[0]))
    x_axis = acquired_axis(axes_by_motor[HORIZONTAL_MOTOR], int(shape[1]))
    return list(motors), y_axis, x_axis


def pad_and_reshape(values, shape, label, fast_axis=FAST_AXIS):
    values = np.asarray(values, dtype=float).ravel()
    rows, cols = (int(shape[0]), int(shape[1]))
    expected = rows * cols
    if values.size < expected:
        padded = np.full(expected, np.nan, dtype=float)
        padded[: values.size] = values
        values = padded
        print(f"{label}: padded to {expected} cells")
    elif values.size > expected:
        print(f"{label}: truncated {values.size} values to {expected} cells")
        values = values[:expected]

    fast_axis = str(fast_axis).strip().lower()
    if fast_axis in {"horizontal", "x", "columns", "cols"}:
        return values.reshape((rows, cols)), values.size
    if fast_axis in {"vertical", "y", "rows"}:
        return values.reshape((cols, rows)).T, values.size
    raise ValueError(f"Unsupported fast axis: {fast_axis!r}.")


def frame_index_from_grid_position(row, col, shape, fast_axis=FAST_AXIS):
    rows, cols = (int(shape[0]), int(shape[1]))
    row = int(row)
    col = int(col)
    fast_axis = str(fast_axis).strip().lower()
    if fast_axis in {"horizontal", "x", "columns", "cols"}:
        return row * cols + col
    if fast_axis in {"vertical", "y", "rows"}:
        return col * rows + row
    raise ValueError(f"Unsupported fast axis: {fast_axis!r}.")


def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("centers must be a non-empty 1D array")
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=float)
    deltas = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = centers[:-1] + deltas / 2.0
    edges[0] = centers[0] - deltas[0] / 2.0
    edges[-1] = centers[-1] + deltas[-1] / 2.0
    return edges


def plot_axes_from_mode(
    run: EmbeddedGridRun,
    shape: tuple[int, int],
    metadata_shape: tuple[int, int],
    coordinate_mode: str,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    coordinate_mode = str(coordinate_mode).strip().lower()
    if coordinate_mode == "index":
        rows, cols = (int(shape[0]), int(shape[1]))
        return (
            np.arange(1, cols + 1, dtype=float),
            np.arange(1, rows + 1, dtype=float),
            "horizontal scan index (1-based)",
            "vertical scan index (1-based)",
        )
    if coordinate_mode == "motor":
        _, y_axis, x_axis = commanded_axes_from_metadata(run, shape, metadata_shape)
        return x_axis, y_axis, HORIZONTAL_MOTOR, VERTICAL_MOTOR
    raise ValueError(f"Unsupported coordinate mode: {coordinate_mode!r}.")


def validate_vertical_roi(image_shape, roi):
    if roi is None or len(roi) != 2:
        raise ValueError("preview_roi_vertical must be a (start, stop) pair.")
    start, stop = sorted(int(value) for value in roi)
    height = int(image_shape[0])
    start = max(0, min(start, height))
    stop = max(0, min(stop, height))
    if stop <= start:
        raise ValueError(f"preview_roi_vertical={roi!r} does not define a non-empty vertical ROI.")
    return start, stop


def load_roi_json(path):
    path = resolve_existing_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    roi = payload.get("roi", payload) if isinstance(payload, dict) else payload
    if not isinstance(roi, dict):
        raise ValueError(f"ROI JSON does not contain a ROI mapping: {path}")
    return dict(roi), path


def tilted_roi_row_bounds(shape, spec):
    height, width = (int(shape[0]), int(shape[1]))
    cols = np.arange(width, dtype=float)
    center = float(spec["center_row_at_col0"]) + float(spec["slope_per_col"]) * cols
    half_width = max(0.5, float(spec["half_width"]))
    top = np.floor(center - half_width)
    bottom = np.ceil(center + half_width + 1.0)
    start = int(np.clip(np.min(top), 0, height))
    stop = int(np.clip(np.max(bottom), start + 1, height))
    return start, stop


def normalize_spectrum_roi(shape, row_range=None, roi=None):
    height, _width = (int(shape[0]), int(shape[1]))
    if roi is None:
        start, stop = validate_vertical_roi(shape, row_range or (0, height))
        return {
            "kind": "row_range",
            "row_start": int(start),
            "row_stop": int(stop),
            "row_bounds": [int(start), int(stop)],
        }

    spec = dict(roi)
    kind = str(spec.get("kind", "tilted_band"))
    if kind == "row_range":
        if "row_bounds" in spec:
            start, stop = validate_vertical_roi(shape, spec["row_bounds"])
        else:
            start, stop = validate_vertical_roi(shape, (spec["row_start"], spec["row_stop"]))
        return {
            "kind": "row_range",
            "row_start": int(start),
            "row_stop": int(stop),
            "row_bounds": [int(start), int(stop)],
        }

    if kind != "tilted_band":
        raise ValueError(f"Unsupported ROI kind: {kind!r}")
    missing = [key for key in ("center_row_at_col0", "slope_per_col", "half_width") if key not in spec]
    if missing:
        raise ValueError(f"Tilted-band ROI is missing required keys: {missing}")

    out = {
        "kind": "tilted_band",
        "center_row_at_col0": float(spec["center_row_at_col0"]),
        "slope_per_col": float(spec["slope_per_col"]),
        "half_width": max(0.5, float(spec["half_width"])),
    }
    start, stop = tilted_roi_row_bounds(shape, out)
    out["row_bounds"] = [int(start), int(stop)]
    return out


def spectrum_roi_description(roi):
    if not roi:
        return "none"
    if roi.get("kind") == "tilted_band":
        return (
            "tilted_band "
            f"center_row_at_col0={float(roi['center_row_at_col0']):.3f}, "
            f"slope_per_col={float(roi['slope_per_col']):.6f}, "
            f"half_width={float(roi['half_width']):.3f}, "
            f"row_bounds={tuple(roi['row_bounds'])}"
        )
    return f"row_range rows {int(roi['row_start'])}:{int(roi['row_stop'])}"


def spectrum_roi_boundary_rows(shape, roi):
    height, width = (int(shape[0]), int(shape[1]))
    roi = normalize_spectrum_roi(shape, roi=roi)
    cols = np.arange(width, dtype=float)
    if roi["kind"] == "row_range":
        top = np.full(width, float(roi["row_start"]), dtype=float)
        bottom = np.full(width, float(roi["row_stop"]) - 1.0, dtype=float)
        return cols, top, bottom

    center = float(roi["center_row_at_col0"]) + float(roi["slope_per_col"]) * cols
    half_width = float(roi["half_width"])
    top = np.clip(center - half_width, 0.0, float(height - 1))
    bottom = np.clip(center + half_width, 0.0, float(height - 1))
    return cols, top, bottom


def prepare_spectrum_roi_weights(shape, row_range=None, roi=None, dtype=np.float32):
    roi_spec = normalize_spectrum_roi(shape, row_range=row_range, roi=roi)
    row_start, row_stop = (int(roi_spec["row_bounds"][0]), int(roi_spec["row_bounds"][1]))
    height, width = (int(shape[0]), int(shape[1]))

    if roi_spec["kind"] == "row_range":
        weights = np.ones((row_stop - row_start, width), dtype=dtype)
    else:
        rows = np.arange(row_start, row_stop, dtype=float)[:, None]
        cols = np.arange(width, dtype=float)[None, :]
        center = float(roi_spec["center_row_at_col0"]) + float(roi_spec["slope_per_col"]) * cols
        half_width = float(roi_spec["half_width"])
        band_top = center - half_width
        band_bottom = center + half_width
        pixel_top = rows - 0.5
        pixel_bottom = rows + 0.5
        overlap = np.minimum(pixel_bottom, band_bottom) - np.maximum(pixel_top, band_top)
        weights = np.clip(overlap, 0.0, 1.0).astype(dtype, copy=False)

    eps = np.finfo(np.dtype(dtype)).eps
    col_weight_sum = np.clip(weights.sum(axis=0, keepdims=True), eps, None).astype(dtype, copy=False)
    return roi_spec, (row_start, row_stop), weights, col_weight_sum


def weighted_column_mean_from_roi_slice(values, weights, col_weight_sum):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    col_weight_sum = np.asarray(col_weight_sum, dtype=float)
    if values.ndim == 2:
        return (values * weights).sum(axis=0) / col_weight_sum[0]
    return (values * weights[None, :, :]).sum(axis=1) / col_weight_sum


def spectrum_roi_weighted_column_mean(image, roi):
    roi_spec, (row_start, row_stop), weights, col_weight_sum = prepare_spectrum_roi_weights(
        np.asarray(image).shape[-2:],
        roi=roi,
    )
    values = np.asarray(image, dtype=float)
    if values.ndim == 2:
        return weighted_column_mean_from_roi_slice(values[row_start:row_stop, :], weights, col_weight_sum), (
            row_start,
            row_stop,
        )
    return weighted_column_mean_from_roi_slice(values[:, row_start:row_stop, :], weights, col_weight_sum), (
        row_start,
        row_stop,
    )


def validate_profile_x_indices(profile_length, x1, x2):
    x1 = int(x1)
    x2 = int(x2)
    if not (0 <= x1 < int(profile_length)):
        raise ValueError(f"x1={x1} is outside the detector-x range.")
    if not (0 <= x2 < int(profile_length)):
        raise ValueError(f"x2={x2} is outside the detector-x range.")
    return x1, x2


def detector_scalar_and_roi_difference_series(
    run: EmbeddedGridRun,
    roi,
    x1,
    x2,
    reduction="mean",
    progress_every=250,
    chunk_size=64,
):
    roi_spec, (roi_start, roi_stop), roi_weights, col_weight_sum = prepare_spectrum_roi_weights(
        run.frame_shape,
        roi=roi,
    )
    x1, x2 = validate_profile_x_indices(run.frame_shape[1], x1, x2)
    scalar_values = np.empty(run.nframes, dtype=float)
    difference_values = np.empty(run.nframes, dtype=float)

    with h5py.File(run.path, "r") as h5:
        dataset = h5[run.detector_path]
        for start in range(0, run.nframes, int(chunk_size)):
            stop = min(start + int(chunk_size), run.nframes)
            frames = np.asarray(dataset[start:stop], dtype=float)
            if reduction == "mean":
                scalar_values[start:stop] = np.mean(frames, axis=(1, 2))
            elif reduction == "sum":
                scalar_values[start:stop] = np.sum(frames, axis=(1, 2), dtype=np.float64)
            else:
                raise ValueError(f"Unsupported reduction: {reduction!r}.")

            profiles = weighted_column_mean_from_roi_slice(
                frames[:, roi_start:roi_stop, :],
                roi_weights,
                col_weight_sum,
            )
            difference_values[start:stop] = profiles[:, x1] - profiles[:, x2]

            if progress_every and stop % int(progress_every) < int(chunk_size):
                print(f"processed {stop}/{run.nframes} detector frames")

    return scalar_values, difference_values, (roi_start, roi_stop), (x1, x2)


def load_detector_image_for_frame(run: EmbeddedGridRun, frame_index):
    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= run.nframes:
        raise IndexError(f"frame_index {frame_index} is outside {run.nframes} frames.")
    with h5py.File(run.path, "r") as h5:
        return np.asarray(h5[run.detector_path][frame_index], dtype=float)


def vertical_roi_average_profile(image, roi):
    return spectrum_roi_weighted_column_mean(image, roi)


def detector_image_display_limits(values, lower_percentile=1.0, upper_percentile=99.5):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    vmin, vmax = np.percentile(finite, [float(lower_percentile), float(upper_percentile)])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None, None
    return float(vmin), float(vmax)


def smooth_grid_for_segmentation(value_grid, sigma):
    filled = np.nan_to_num(value_grid, nan=0.0)
    if sigma is None or float(sigma) <= 0:
        return filled
    return ndi.gaussian_filter(filled, sigma=float(sigma))


def binary_threshold_mask(value_grid, binary_threshold, smooth_sigma):
    finite_values = value_grid[np.isfinite(value_grid)]
    if finite_values.size == 0:
        return np.zeros_like(value_grid, dtype=bool), np.nan
    threshold_value = float(binary_threshold)
    smoothed_grid = smooth_grid_for_segmentation(value_grid, smooth_sigma)
    binary_mask = np.isfinite(value_grid) & (smoothed_grid >= threshold_value)
    return binary_mask, threshold_value


def region_score_from_values(values, metric):
    metric_key = str(metric).strip().lower()
    if metric_key == "peak":
        return float(np.max(values))
    if metric_key == "sum":
        return float(np.sum(values))
    if metric_key == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unsupported region rank metric: {metric!r}.")


def region_center_from_values(region_mask, value_grid, x_coord_grid, y_coord_grid, method):
    method_key = str(method).strip().lower()
    if method_key == "geometric":
        weights = np.ones(np.count_nonzero(region_mask), dtype=float)
    elif method_key == "weighted":
        weights = np.clip(value_grid[region_mask], a_min=0.0, a_max=None)
        if not np.any(weights > 0):
            weights = np.ones(np.count_nonzero(region_mask), dtype=float)
    else:
        raise ValueError(f"Unsupported region center method: {method!r}.")

    x_coords = x_coord_grid[region_mask]
    y_coords = y_coord_grid[region_mask]
    return float(np.average(x_coords, weights=weights)), float(np.average(y_coords, weights=weights))


def masked_difference_selection_grid(binary_mask, difference_grid):
    selection_grid = np.full(np.asarray(difference_grid).shape, np.nan, dtype=float)
    valid_mask = np.asarray(binary_mask, dtype=bool) & np.isfinite(difference_grid)
    selection_grid[valid_mask] = difference_grid[valid_mask]
    return selection_grid, valid_mask


def rank_regions_from_binary_mask(
    binary_mask,
    score_grid,
    x_coord_grid,
    y_coord_grid,
    min_region_pixels,
    connectivity,
    center_method,
    rank_metric,
    threshold_value=np.nan,
):
    score_grid = np.asarray(score_grid, dtype=float)
    selection_mask = np.asarray(binary_mask, dtype=bool) & np.isfinite(score_grid)
    if not np.any(selection_mask):
        empty_labels = np.zeros_like(selection_mask, dtype=int)
        return [], selection_mask, selection_mask.copy(), empty_labels, threshold_value

    structure = ndi.generate_binary_structure(2, int(connectivity))
    labeled_grid, nlabels = ndi.label(selection_mask, structure=structure)

    regions = []
    filtered_mask = np.zeros_like(selection_mask, dtype=bool)
    filtered_labels = np.zeros_like(labeled_grid, dtype=int)
    next_label = 1

    for label_id in range(1, nlabels + 1):
        region_mask = labeled_grid == label_id
        region_size = int(np.count_nonzero(region_mask))
        if region_size < int(min_region_pixels):
            continue
        region_values = score_grid[region_mask]
        center_x, center_y = region_center_from_values(
            region_mask, score_grid, x_coord_grid, y_coord_grid, center_method
        )
        region_indices = np.argwhere(region_mask)
        peak_offset = int(np.argmax(region_values))
        peak_row, peak_col = region_indices[peak_offset]
        regions.append(
            {
                "label_id": next_label,
                "size": region_size,
                "peak_value": float(np.max(region_values)),
                "mean_value": float(np.mean(region_values)),
                "sum_value": float(np.sum(region_values)),
                "score": region_score_from_values(region_values, rank_metric),
                "center_x": center_x,
                "center_y": center_y,
                "peak_row": int(peak_row),
                "peak_col": int(peak_col),
                "peak_x": float(x_coord_grid[peak_row, peak_col]),
                "peak_y": float(y_coord_grid[peak_row, peak_col]),
            }
        )
        filtered_mask |= region_mask
        filtered_labels[region_mask] = next_label
        next_label += 1

    regions.sort(key=lambda item: item["score"], reverse=True)
    for rank, region in enumerate(regions, start=1):
        region["rank"] = rank
    return regions, selection_mask, filtered_mask, filtered_labels, threshold_value


def order_regions_for_scan_path(regions, y_coord_grid, snake_axes=False):
    if not regions:
        return []
    row_y_positions = {
        int(region["peak_row"]): float(np.nanmedian(y_coord_grid[int(region["peak_row"]), :]))
        for region in regions
    }
    ordered_rows = sorted(row_y_positions, key=lambda row: (row_y_positions[row], row))

    ordered_regions = []
    for row_offset, row in enumerate(ordered_rows):
        row_regions = [dict(region) for region in regions if int(region["peak_row"]) == row]
        row_regions.sort(key=lambda region: (region["peak_x"], region["center_x"], region["peak_y"]))
        if bool(snake_axes) and row_offset % 2 == 1:
            row_regions.reverse()
        ordered_regions.extend(row_regions)

    for scan_order, region in enumerate(ordered_regions, start=1):
        region["scan_order"] = scan_order
    return ordered_regions


def region_preview_position(region, region_labels, x_coord_grid, y_coord_grid, fast_axis=FAST_AXIS):
    region_mask = region_labels == region["label_id"]
    if np.any(region_mask):
        indices = np.argwhere(region_mask)
        x_values = np.asarray(x_coord_grid[region_mask], dtype=float)
        y_values = np.asarray(y_coord_grid[region_mask], dtype=float)
        deltas = (x_values - float(region["center_x"])) ** 2 + (y_values - float(region["center_y"])) ** 2
        row, col = indices[int(np.argmin(deltas))]
    else:
        row = int(region["peak_row"])
        col = int(region["peak_col"])
    return {
        "row": int(row),
        "col": int(col),
        "frame_index": int(frame_index_from_grid_position(row, col, x_coord_grid.shape, fast_axis)),
        "x": float(x_coord_grid[int(row), int(col)]),
        "y": float(y_coord_grid[int(row), int(col)]),
    }


def normalize_preview_selection(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"all", "*"}:
            return "all"
        if key.startswith("p"):
            return key
        return f"p{int(key)}"
    return f"p{int(value)}"


def resolve_preview_regions(preview_spectrum, top_regions):
    selected_by_scan_key = {region["scan_key"]: region for region in top_regions}
    preview_keys = [normalize_preview_selection(value) for value in preview_spectrum]
    if "all" in preview_keys:
        return preview_keys, list(top_regions), []
    preview_regions = [selected_by_scan_key[key] for key in preview_keys if key in selected_by_scan_key]
    missing_preview_keys = [key for key in preview_keys if key not in selected_by_scan_key]
    return preview_keys, preview_regions, missing_preview_keys


def average_detector_frames(run: EmbeddedGridRun):
    sum_image = np.zeros(run.frame_shape, dtype=float)
    with h5py.File(run.path, "r") as h5:
        dataset = h5[run.detector_path]
        for frame_index in range(run.nframes):
            sum_image += np.asarray(dataset[frame_index], dtype=float)
    return sum_image / float(run.nframes)


def mux_from_flatfield(flatfield, image):
    flatfield = np.asarray(flatfield, dtype=float)
    image = np.asarray(image, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((flatfield > 0) & (image > 0), np.log(flatfield / image), np.nan)


def validate_derivative_x_range(profile_length, x_range):
    if x_range is None or len(x_range) != 2:
        raise ValueError("derivative_x must be a (start, stop) pair.")
    start, stop = sorted(int(value) for value in x_range)
    if not (0 <= start < int(profile_length)):
        raise ValueError(f"derivative start x={start} is outside the detector-x range.")
    if not (0 <= stop < int(profile_length)):
        raise ValueError(f"derivative stop x={stop} is outside the detector-x range.")
    if stop <= start:
        raise ValueError(f"derivative_x={x_range!r} does not define a useful x range.")
    return start, stop


def derivative_peak_from_profile(profile, x_range):
    profile = np.asarray(profile, dtype=float)
    start, stop = validate_derivative_x_range(profile.size, x_range)
    derivative = np.gradient(profile)
    x_values = np.arange(start, stop + 1, dtype=float)
    selected = derivative[start : stop + 1]
    finite = np.isfinite(selected)
    if not np.any(finite):
        return x_values, selected, np.nan, np.nan
    filled = np.where(finite, selected, -np.inf)
    peak_offset = int(np.argmax(filled))
    return x_values, selected, float(start + peak_offset), float(selected[peak_offset])


def derivative_record_from_profile(region, position, profile, x_range):
    x_values, derivative, peak_x, peak_value = derivative_peak_from_profile(profile, x_range)
    return DerivativeRecord(region, position, x_values, derivative, peak_x, peak_value)


def mux_derivative_peak_series(
    run: EmbeddedGridRun,
    flatfield_average,
    roi,
    x_range,
    progress_every=250,
    chunk_size=64,
):
    _roi_spec, (roi_start, roi_stop), roi_weights, _col_weight_sum = prepare_spectrum_roi_weights(
        run.frame_shape,
        roi=roi,
    )
    x_start, x_stop = validate_derivative_x_range(run.frame_shape[1], x_range)
    flatfield_average = np.asarray(flatfield_average, dtype=float)
    flat_roi = flatfield_average[roi_start:roi_stop, :]
    peak_x_values = np.full(run.nframes, np.nan, dtype=float)
    peak_derivative_values = np.full(run.nframes, np.nan, dtype=float)

    with h5py.File(run.path, "r") as h5:
        dataset = h5[run.detector_path]
        for start in range(0, run.nframes, int(chunk_size)):
            stop = min(start + int(chunk_size), run.nframes)
            frames = np.asarray(dataset[start:stop, roi_start:roi_stop, :], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                mux_frames = np.where((flat_roi > 0) & (frames > 0), np.log(flat_roi / frames), np.nan)
            finite_mux = np.isfinite(mux_frames)
            valid_weights = np.where(finite_mux, roi_weights[None, :, :], 0.0)
            counts = np.sum(valid_weights, axis=1)
            sums = np.sum(np.where(finite_mux, mux_frames, 0.0) * roi_weights[None, :, :], axis=1)
            profiles = np.full_like(sums, np.nan, dtype=float)
            np.divide(sums, counts, out=profiles, where=counts > 0)
            derivatives = np.gradient(profiles, axis=1)
            selected = derivatives[:, x_start : x_stop + 1]
            finite = np.isfinite(selected)
            any_finite = np.any(finite, axis=1)
            filled = np.where(finite, selected, -np.inf)
            peak_offsets = np.argmax(filled, axis=1)
            rows = np.arange(stop - start)
            chunk_peak_x = (x_start + peak_offsets).astype(float)
            chunk_peak_values = selected[rows, peak_offsets]
            chunk_peak_x[~any_finite] = np.nan
            chunk_peak_values[~any_finite] = np.nan
            peak_x_values[start:stop] = chunk_peak_x
            peak_derivative_values[start:stop] = chunk_peak_values

            if progress_every and stop % int(progress_every) < int(chunk_size):
                print(f"processed {stop}/{run.nframes} mux derivative frames")

    return peak_x_values, peak_derivative_values, (x_start, x_stop)


def parse_preview_values(values):
    if values is None:
        return list(PREVIEW_SPECTRUM)
    if len(values) == 1 and "," in values[0]:
        return [item.strip() for item in values[0].split(",") if item.strip()]
    return list(values)


def build_analysis(args: argparse.Namespace) -> AnalysisResults:
    grid_run = read_embedded_grid_run(data_path(args.hdf))
    flat_run = read_embedded_grid_run(data_path(args.flat))

    loaded_roi = None
    if args.spectrum_roi_json:
        loaded_roi, loaded_roi_path = load_roi_json(args.spectrum_roi_json)
        print(f"Using spectrum ROI JSON: {loaded_roi_path}")
    spectrum_roi = normalize_spectrum_roi(grid_run.frame_shape, row_range=args.roi_y, roi=loaded_roi)
    print(f"  spectrum ROI: {spectrum_roi_description(spectrum_roi)}")

    metadata_grid_shape = grid_shape_from_metadata(grid_run)
    detector_fill_shape = detector_frame_grid_shape(grid_run, metadata_grid_shape, args.fast_axis)
    plot_grid_shape = metadata_grid_shape if args.plot_grid == "full" else detector_fill_shape
    expected_cells = int(np.prod(plot_grid_shape))
    print("Embedded mapping HDF")
    print(f"  path: {grid_run.path}")
    print(f"  detector shape: ({grid_run.nframes}, {grid_run.frame_shape[0]}, {grid_run.frame_shape[1]})")
    print(f"  metadata shape: {metadata_grid_shape} -> {int(np.prod(metadata_grid_shape))} commanded cells")
    print(f"  detector-frame fill shape: {detector_fill_shape} -> {int(np.prod(detector_fill_shape))} acquired cells")
    print(f"  plot map shape: {plot_grid_shape} -> {expected_cells} cells ({args.plot_grid})")
    print(f"  scan order: {args.fast_axis}-fast")
    print(f"  plot coordinates: {args.coords}")
    if args.plot_grid == "full" and grid_run.nframes < int(np.prod(metadata_grid_shape)):
        print("  note: full-grid plotting pads missing detector cells with NaN; acquired-grid plotting is usually clearer.")

    x_axis, y_axis, x_label, y_label = plot_axes_from_mode(
        grid_run, plot_grid_shape, metadata_grid_shape, args.coords
    )
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    x_edges = centers_to_edges(x_axis)
    y_edges = centers_to_edges(y_axis)

    it_values = grid_run.data_values.get(IT_KEY)
    it_available = it_values is not None and np.any(np.isfinite(it_values))
    if not it_available:
        print(f"{IT_KEY!r} was not found; continuing without an It plot.")
        it_values = np.full(grid_run.nframes, np.nan, dtype=float)
    it_grid, _ = pad_and_reshape(it_values, plot_grid_shape, IT_KEY, args.fast_axis)

    print("Computing lambda scalar and ROI-difference maps from detector frames...")
    scalar_values, difference_values, difference_roi_vertical, difference_x_indices = (
        detector_scalar_and_roi_difference_series(
            grid_run,
            spectrum_roi,
            args.x1,
            args.x2,
            reduction=args.reduction,
            progress_every=args.progress_every,
            chunk_size=args.chunk_size,
        )
    )
    lambda_grid, _ = pad_and_reshape(scalar_values, plot_grid_shape, "lambda scalar values", args.fast_axis)
    difference_grid, _ = pad_and_reshape(
        difference_values,
        plot_grid_shape,
        f"lambda ROI difference x={difference_x_indices[0]} - x={difference_x_indices[1]}",
        args.fast_axis,
    )
    difference_label = (
        f"profile[x={difference_x_indices[0]}] - profile[x={difference_x_indices[1]}] "
        f"using {spectrum_roi_description(spectrum_roi)}"
    )

    if args.top_source == "lambda":
        top_source_grid = lambda_grid
    elif args.top_source == "it":
        if not it_available:
            raise RuntimeError("TOP_N_SOURCE=it requires an It-count stream, but this HDF file has none.")
        top_source_grid = it_grid
    else:
        raise ValueError(f"Unsupported top source: {args.top_source!r}")

    binary_mask, threshold_value = binary_threshold_mask(
        top_source_grid, binary_threshold=args.binary_threshold, smooth_sigma=args.smooth_sigma
    )
    masked_difference_grid, difference_selection_mask = masked_difference_selection_grid(binary_mask, difference_grid)
    regions, _, bright_region_mask, bright_region_labels, _ = rank_regions_from_binary_mask(
        difference_selection_mask,
        masked_difference_grid,
        x_grid,
        y_grid,
        min_region_pixels=args.min_region_pixels,
        connectivity=args.connectivity,
        center_method=args.center_method,
        rank_metric=args.rank_metric,
        threshold_value=threshold_value,
    )

    top_regions = order_regions_for_scan_path(
        regions[: min(max(args.top_n_regions, 0), len(regions))],
        y_grid,
        snake_axes=args.snake_axes,
    )
    for scan_index, region in enumerate(top_regions, start=1):
        region["scan_key"] = f"p{scan_index}"
    top_regions_by_difference = [dict(region) for region in sorted(top_regions, key=lambda r: r["peak_value"], reverse=True)]
    for difference_order, region in enumerate(top_regions_by_difference, start=1):
        region["difference_order"] = difference_order

    print(f"Selected {len(top_regions)} bright-region centers.")
    if args.coords == "index":
        coordinate_dict = {
            r["scan_key"]: (int(round(r["center_x"])), int(round(r["center_y"]))) for r in top_regions
        }
    else:
        coordinate_dict = {
            r["scan_key"]: (round(float(r["center_x"]), 3), round(float(r["center_y"]), 3)) for r in top_regions
        }
    print_coordinate_dict(coordinate_dict, x_label, y_label)

    preview_keys, preview_regions, missing_preview_points = resolve_preview_regions(args.preview, top_regions)
    if missing_preview_points:
        print(f"Skipped preview labels not present in selected points: {missing_preview_points}")

    preview_records: list[PreviewRecord] = []
    for region in preview_regions:
        position = region_preview_position(region, bright_region_labels, x_grid, y_grid, args.fast_axis)
        image = load_detector_image_for_frame(grid_run, position["frame_index"])
        profile, (roi_start, roi_stop) = vertical_roi_average_profile(image, spectrum_roi)
        preview_records.append(PreviewRecord(region, position, image, profile, roi_start, roi_stop))

    flatfield_average = average_detector_frames(flat_run)
    if flatfield_average.shape != grid_run.frame_shape:
        raise ValueError(f"Flatfield shape {flatfield_average.shape!r} does not match sample {grid_run.frame_shape!r}.")

    print(
        f"Computing mux first-derivative peak maps over x={args.derivative_x[0]}:{args.derivative_x[1]}..."
    )
    mux_peak_x_values, mux_peak_derivative_values, mux_derivative_x_indices = mux_derivative_peak_series(
        grid_run,
        flatfield_average,
        spectrum_roi,
        args.derivative_x,
        progress_every=args.progress_every,
        chunk_size=args.chunk_size,
    )
    mux_derivative_peak_x_grid, _ = pad_and_reshape(
        mux_peak_x_values,
        plot_grid_shape,
        "mux derivative peak x values",
        args.fast_axis,
    )
    mux_derivative_peak_value_grid, _ = pad_and_reshape(
        mux_peak_derivative_values,
        plot_grid_shape,
        "mux derivative peak values",
        args.fast_axis,
    )

    mux_records: list[PreviewRecord] = []
    mux_derivative_records: list[DerivativeRecord] = []
    for region in preview_regions:
        position = region_preview_position(region, bright_region_labels, x_grid, y_grid, args.fast_axis)
        image = load_detector_image_for_frame(grid_run, position["frame_index"])
        mux_image = mux_from_flatfield(flatfield_average, image)
        mux_profile, (roi_start, roi_stop) = vertical_roi_average_profile(mux_image, spectrum_roi)
        mux_records.append(PreviewRecord(region, position, mux_image, mux_profile, roi_start, roi_stop))
        mux_derivative_records.append(
            derivative_record_from_profile(region, position, mux_profile, mux_derivative_x_indices)
        )

    return AnalysisResults(
        grid_run=grid_run,
        flat_run=flat_run,
        metadata_grid_shape=metadata_grid_shape,
        plot_grid_shape=plot_grid_shape,
        x_axis=x_axis,
        y_axis=y_axis,
        x_label=x_label,
        y_label=y_label,
        x_edges=x_edges,
        y_edges=y_edges,
        x_grid=x_grid,
        y_grid=y_grid,
        lambda_grid=lambda_grid,
        it_grid=it_grid,
        difference_grid=difference_grid,
        difference_label=difference_label,
        difference_roi_vertical=difference_roi_vertical,
        difference_x_indices=difference_x_indices,
        spectrum_roi=spectrum_roi,
        binary_mask=binary_mask,
        masked_difference_grid=masked_difference_grid,
        bright_region_mask=bright_region_mask,
        bright_region_labels=bright_region_labels,
        top_regions=top_regions,
        top_regions_by_difference=top_regions_by_difference,
        preview_records=preview_records,
        mux_records=mux_records,
        mux_derivative_records=mux_derivative_records,
        mux_derivative_peak_x_grid=mux_derivative_peak_x_grid,
        mux_derivative_peak_value_grid=mux_derivative_peak_value_grid,
        mux_derivative_x_indices=mux_derivative_x_indices,
        flatfield_average=flatfield_average,
        it_available=it_available,
        args=args,
    )


def print_coordinate_dict(coordinate_dict, x_label="x", y_label="y"):
    if not coordinate_dict:
        print("No coordinates selected.")
        return
    print(f"Coordinate dictionary ({x_label}, {y_label}):")
    print("{")
    for key, value in coordinate_dict.items():
        print(f"    {key!r}: {value},")
    print("}")


def safe_path_component(value, fallback="unknown_sample"):
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text or fallback


def sample_name_from_results(results: AnalysisResults) -> str:
    plan_args = metadata_field(results.grid_run, "start.plan_args", {})
    md = plan_args.get("md", {}) if isinstance(plan_args, dict) else {}
    return metadata_field(results.grid_run, "start.sample_name") or md.get("sample_name") or results.grid_run.path.stem


def output_folder_name_from_results(results: AnalysisResults) -> str:
    output_folder_name = getattr(results.args, "output_folder_name", None)
    if output_folder_name:
        return safe_path_component(output_folder_name)
    output_folder = getattr(results.args, "output_folder", "sample")
    if output_folder == "hdf":
        return safe_path_component(results.grid_run.path.stem)
    if output_folder == "sample":
        return safe_path_component(sample_name_from_results(results))
    raise ValueError(f"Unsupported output folder mode: {output_folder!r}")


def results_output_dir(results: AnalysisResults, root) -> Path:
    output_dir = Path(root).expanduser() / output_folder_name_from_results(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def file_prefix_from_results(results: AnalysisResults) -> str:
    sample_name = safe_path_component(sample_name_from_results(results))
    scan_id = metadata_field(results.grid_run, "start.scan_id")
    if scan_id is not None:
        return f"{sample_name}_scan{scan_id}"
    return f"{sample_name}_{results.grid_run.path.stem}"


def require_pyqtgraph():
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    except Exception as exc:
        raise SystemExit(
            "PyQtGraph and a Qt binding are required for GUI plotting.\n"
            "Install them in this Python environment, for example:\n"
            "  python3 -m pip install pyqtgraph PyQt6\n"
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc
    return pg, QtCore, QtGui, QtWidgets


def get_colormap(pg, name):
    try:
        return pg.colormap.get(name, source="matplotlib")
    except Exception:
        if name == "coolwarm":
            return pg.ColorMap(
                [0.0, 0.5, 1.0],
                [(59, 76, 192), (240, 240, 240), (180, 4, 38)],
            )
        if name in {"gray", "gray_r"}:
            colors = [(255, 255, 255), (0, 0, 0)] if name == "gray_r" else [(0, 0, 0), (255, 255, 255)]
            return pg.ColorMap([0.0, 1.0], colors)
        return pg.colormap.get("viridis")


def finite_levels(data, symmetric=False, lower=1.0, upper=99.5):
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    if symmetric:
        limit = float(np.nanmax(np.abs(finite)))
        return (-limit, limit) if limit > 0 else None
    vmin, vmax = np.percentile(finite, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    return float(vmin), float(vmax)


def finite_span(edges):
    edges = np.asarray(edges, dtype=float)
    finite = edges[np.isfinite(edges)]
    if finite.size < 2:
        return 1.0
    span = float(np.nanmax(finite) - np.nanmin(finite))
    return abs(span) if span != 0 else 1.0


def map_figure_size(x_edges, y_edges, base=6.6, min_size=5.2, max_size=12.5):
    aspect = finite_span(x_edges) / finite_span(y_edges)
    if not np.isfinite(aspect) or aspect <= 0:
        aspect = 1.0
    if aspect >= 1.0:
        return min(max_size, max(min_size, base * aspect)), base
    return base, min(max_size, max(min_size, base / aspect))


def make_mesh_item(pg, x_edges, y_edges, z, cmap_name, levels=None):
    cmap = get_colormap(pg, cmap_name)
    x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)
    kwargs = {"colorMap": cmap}
    if levels is not None:
        kwargs["levels"] = levels
    try:
        return pg.PColorMeshItem(x_mesh, y_mesh, np.asarray(z, dtype=float), **kwargs)
    except TypeError:
        item = pg.PColorMeshItem(x_mesh, y_mesh, np.asarray(z, dtype=float))
        if hasattr(item, "setColorMap"):
            item.setColorMap(cmap)
        if levels is not None and hasattr(item, "setLevels"):
            item.setLevels(levels)
        return item


def add_map(
    layout,
    pg,
    row,
    col,
    title,
    z,
    x_edges,
    y_edges,
    cmap_name,
    levels=None,
    unavailable_text=None,
    x_label="x",
    y_label="y",
):
    plot = layout.addPlot(row=row, col=col, title=title)
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", y_label)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setAspectLocked(True, ratio=1)
    if unavailable_text:
        text = pg.TextItem(unavailable_text, anchor=(0.5, 0.5), color=(220, 220, 220))
        text.setPos(float(np.mean(x_edges)), float(np.mean(y_edges)))
        plot.addItem(text)
        return plot
    mesh = make_mesh_item(pg, x_edges, y_edges, z, cmap_name, levels)
    plot.addItem(mesh)
    return plot


def add_region_markers(pg, plot, regions):
    if not regions:
        return
    spots = [
        {
            "pos": (float(region["center_x"]), float(region["center_y"])),
            "data": region,
            "brush": (255, 255, 255, 0),
            "pen": pg.mkPen("w", width=2),
            "size": 13,
        }
        for region in regions
    ]
    scatter = pg.ScatterPlotItem(spots=spots)
    plot.addItem(scatter)
    for region in regions:
        text = pg.TextItem(str(region["scan_order"]), color="k", anchor=(0.5, 0.5), fill=(255, 255, 255, 220))
        text.setPos(float(region["center_x"]), float(region["center_y"]))
        plot.addItem(text)


def detector_edges(shape):
    height, width = shape
    return np.arange(width + 1, dtype=float) - 0.5, np.arange(height + 1, dtype=float) - 0.5


def format_metadata_number(value, precision=6):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{precision}g}"


def format_metadata_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, indent=2)
    return str(value)


def scan_axis_summary(results: AnalysisResults) -> list[str]:
    motors = metadata_field(results.grid_run, "start.motors", [])
    extents = metadata_field(results.grid_run, "start.extents", [])
    shape = metadata_field(results.grid_run, "start.shape", results.metadata_grid_shape)
    snaking = metadata_field(results.grid_run, "start.snaking", [])

    rows = []
    for axis_index, motor in enumerate(motors):
        count = int(shape[axis_index]) if axis_index < len(shape) else None
        extent = extents[axis_index] if axis_index < len(extents) else None
        snake = snaking[axis_index] if axis_index < len(snaking) else False
        if extent is not None and len(extent) == 2 and count:
            start, stop = float(extent[0]), float(extent[1])
            step = (stop - start) / (count - 1) if count > 1 else np.nan
            rows.append(
                f"{motor}: {count} steps, {format_metadata_number(start)} to "
                f"{format_metadata_number(stop)}, step {format_metadata_number(step)}, snake={bool(snake)}"
            )
        else:
            rows.append(f"{motor}: {count or '?'} steps, extent unknown, snake={bool(snake)}")
    return rows


def scan_setup_lines(results: AnalysisResults, detailed=False) -> list[str]:
    metadata = results.grid_run.metadata
    plan_args = metadata_field(results.grid_run, "start.plan_args", {})
    md = plan_args.get("md", {}) if isinstance(plan_args, dict) else {}

    plan_name = metadata.get("start.plan_name", "unknown")
    scan_id = metadata.get("start.scan_id", "unknown")
    sample_name = metadata.get("start.sample_name") or md.get("sample_name", "unknown")
    num_points = metadata.get("start.num_points", int(np.prod(results.metadata_grid_shape)))

    lines = [
        f"Plan: {plan_name} | scan_id: {scan_id} | sample: {sample_name}",
        (
            f"Commanded grid: {results.metadata_grid_shape[0]} x {results.metadata_grid_shape[1]} "
            f"({num_points} points) | acquired detector frames: {results.grid_run.nframes} "
            f"-> map {results.plot_grid_shape[0]} x {results.plot_grid_shape[1]}"
        ),
    ]
    lines.extend(scan_axis_summary(results))

    if isinstance(plan_args, dict):
        extras = []
        for key in ("dwell_time", "trigger", "snake_axes"):
            if key in plan_args:
                extras.append(f"{key}: {plan_args[key]}")
        if extras:
            lines.append(" | ".join(extras))

    if detailed:
        lines.extend(
            [
                f"HDF: {results.grid_run.path.name}",
                f"Detector path: {results.grid_run.detector_path}",
                f"Detector shape: ({results.grid_run.nframes}, {results.grid_run.frame_shape[0]}, {results.grid_run.frame_shape[1]})",
                f"Spectrum ROI: {spectrum_roi_description(results.spectrum_roi)}",
                f"ROI-difference x: {results.difference_x_indices[0]} - {results.difference_x_indices[1]}",
                f"Mux derivative range: x {results.mux_derivative_x_indices[0]}:{results.mux_derivative_x_indices[1]}",
                f"Region selection: source={results.args.top_source}, threshold={results.args.binary_threshold:g}, "
                f"rank={results.args.rank_metric}, top_n={results.args.top_n_regions}",
            ]
        )
    return lines


def add_scan_setup_label(layout, pg, row, col, colspan, results: AnalysisResults):
    text = "<br>".join(html.escape(line) for line in scan_setup_lines(results))
    label = layout.addLabel(text, row=row, col=col, colspan=colspan, justify="left")
    try:
        label.setText(text, size="10pt")
    except TypeError:
        label.setText(text)
    return label


def build_scan_info_tab(results: AnalysisResults, QtWidgets):
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)

    summary = QtWidgets.QTextEdit()
    summary.setReadOnly(True)
    summary.setPlainText("\n".join(scan_setup_lines(results, detailed=True)))
    layout.addWidget(summary)

    metadata_table = QtWidgets.QTableWidget()
    visible_keys = [
        "start.plan_name",
        "start.plan_args",
        "start.shape",
        "start.extents",
        "start.motors",
        "start.num_points",
        "start.num_intervals",
        "start.snaking",
        "start.sample_name",
        "start.scan_id",
        "start.uid",
        "start.time",
        "stop.exit_status",
        "stop.num_events",
        "stop.reason",
    ]
    available_rows = [
        (key, results.grid_run.metadata[key])
        for key in visible_keys
        if key in results.grid_run.metadata
    ]
    metadata_table.setColumnCount(2)
    metadata_table.setHorizontalHeaderLabels(["Metadata key", "Value"])
    metadata_table.setRowCount(len(available_rows))
    for row, (key, value) in enumerate(available_rows):
        key_item = QtWidgets.QTableWidgetItem(key)
        value_item = QtWidgets.QTableWidgetItem(format_metadata_value(value))
        metadata_table.setItem(row, 0, key_item)
        metadata_table.setItem(row, 1, value_item)
    metadata_table.resizeColumnsToContents()
    metadata_table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(metadata_table, stretch=1)

    return widget


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to save mapping result images. "
            "Install it with: python3 -m pip install matplotlib"
        ) from exc
    return plt


def add_roi_overlay_matplotlib(ax, image_shape, roi):
    cols, top_rows, bottom_rows = spectrum_roi_boundary_rows(image_shape, roi)
    ax.plot(cols, top_rows, color="cyan", linewidth=1.2)
    ax.plot(cols, bottom_rows, color="cyan", linewidth=1.2)
    ax.fill_between(cols, top_rows, bottom_rows, color="cyan", alpha=0.10)


def add_roi_overlay_pyqt(pg, plot, image_shape, roi):
    cols, top_rows, bottom_rows = spectrum_roi_boundary_rows(image_shape, roi)
    plot.plot(cols, top_rows, pen=pg.mkPen("c", width=2))
    plot.plot(cols, bottom_rows, pen=pg.mkPen("c", width=2))


def save_grid_png(plt, path, z, x_edges, y_edges, title, x_label, y_label, cmap_name, levels=None, regions=None):
    fig, ax = plt.subplots(figsize=map_figure_size(x_edges, y_edges), dpi=180, constrained_layout=True)
    kwargs = {"cmap": cmap_name, "shading": "auto"}
    if levels is not None:
        kwargs["vmin"], kwargs["vmax"] = levels
    mesh = ax.pcolormesh(x_edges, y_edges, np.asarray(z, dtype=float), **kwargs)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.036, pad=0.025)
    colorbar.ax.tick_params(labelsize=8)
    ax.set_title(title, pad=10, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f6f7f9")
    ax.grid(True, color="white", linewidth=0.8, alpha=0.75)
    ax.tick_params(labelsize=8)

    if regions:
        for region in regions:
            x = float(region["center_x"])
            y = float(region["center_y"])
            label = str(region.get("scan_order", region.get("rank", "")))
            ax.scatter([x], [y], s=58, facecolors="white", edgecolors="black", linewidths=1.1, zorder=3)
            ax.text(x, y, label, ha="center", va="center", fontsize=7, color="black", zorder=4)

    fig.savefig(path)
    plt.close(fig)


def save_detector_preview_png(
    plt,
    path,
    record,
    title_prefix,
    profile_label,
    x1,
    x2,
    cmap_name,
    roi=None,
    derivative_record=None,
):
    fig, (image_ax, profile_ax) = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=180)
    levels = finite_levels(record.image, lower=1.0, upper=99.5)
    kwargs = {"origin": "lower", "cmap": cmap_name, "aspect": "auto"}
    if levels is not None:
        kwargs["vmin"], kwargs["vmax"] = levels
    image = image_ax.imshow(record.image, **kwargs)
    fig.colorbar(image, ax=image_ax, fraction=0.046, pad=0.04)
    add_roi_overlay_matplotlib(image_ax, record.image.shape, roi or {"kind": "row_range", "row_bounds": [record.roi_start, record.roi_stop]})
    image_ax.axvline(x1, color="#00e5ff", linewidth=1.2)
    image_ax.axvline(x2, color="#ffdf5d", linewidth=1.2)
    image_ax.set_title(
        f"{title_prefix} {record.region['scan_order']} | "
        f"center ({format_metadata_number(record.region['center_x'])}, "
        f"{format_metadata_number(record.region['center_y'])})"
    )
    image_ax.set_xlabel("detector x pixel")
    image_ax.set_ylabel("detector y pixel")

    profile_line = profile_ax.plot(np.arange(record.profile.size), record.profile, linewidth=1.6, label=profile_label)
    profile_ax.axvline(x1, color="#00e5ff", linewidth=1.2)
    profile_ax.axvline(x2, color="#ffdf5d", linewidth=1.2)
    profile_ax.set_title(profile_label)
    profile_ax.set_xlabel("detector x pixel")
    profile_ax.set_ylabel(profile_label)
    profile_ax.grid(True, alpha=0.25)

    if derivative_record is not None:
        derivative_ax = profile_ax.twinx()
        derivative_line = derivative_ax.plot(
            derivative_record.x_values,
            derivative_record.derivative,
            color="#d62728",
            linewidth=1.6,
            label="d(mux)/dx",
        )
        if np.isfinite(derivative_record.peak_x) and np.isfinite(derivative_record.peak_value):
            derivative_ax.scatter(
                [derivative_record.peak_x],
                [derivative_record.peak_value],
                s=48,
                color="black",
                zorder=3,
            )
            derivative_ax.axvline(derivative_record.peak_x, color="black", linestyle="--", linewidth=1.0)
        profile_ax.set_title(
            f"{profile_label}\nd(mux)/dx max at x={format_metadata_number(derivative_record.peak_x)}"
        )
        derivative_ax.set_ylabel("first derivative")
        lines = profile_line + derivative_line
        labels = [line.get_label() for line in lines]
        profile_ax.legend(lines, labels, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_region_coordinates_csv(path, results: AnalysisResults):
    fieldnames = [
        "scan_key",
        "scan_order",
        "rank",
        "center_x",
        "center_y",
        "peak_x",
        "peak_y",
        "peak_row",
        "peak_col",
        "size",
        "peak_value",
        "mean_value",
        "sum_value",
        "score",
        "mux_derivative_peak_x",
        "mux_derivative_peak_value",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for region in results.top_regions:
            row_data = {key: region.get(key, "") for key in fieldnames}
            peak_row = int(region.get("peak_row", -1))
            peak_col = int(region.get("peak_col", -1))
            if (
                0 <= peak_row < results.mux_derivative_peak_x_grid.shape[0]
                and 0 <= peak_col < results.mux_derivative_peak_x_grid.shape[1]
            ):
                row_data["mux_derivative_peak_x"] = results.mux_derivative_peak_x_grid[peak_row, peak_col]
                row_data["mux_derivative_peak_value"] = results.mux_derivative_peak_value_grid[peak_row, peak_col]
            writer.writerow(row_data)


def save_analysis_results(results: AnalysisResults, root) -> Path:
    plt = require_matplotlib()
    output_dir = results_output_dir(results, root)
    prefix = file_prefix_from_results(results)
    diff_levels = finite_levels(results.difference_grid, symmetric=True)

    save_grid_png(
        plt,
        output_dir / f"{prefix}_01_lambda_scalar.png",
        results.lambda_grid,
        results.x_edges,
        results.y_edges,
        f"Lambda scalar map ({results.args.reduction})",
        results.x_label,
        results.y_label,
        "viridis",
        levels=finite_levels(results.lambda_grid),
    )
    save_grid_png(
        plt,
        output_dir / f"{prefix}_02_lambda_roi_difference.png",
        results.difference_grid,
        results.x_edges,
        results.y_edges,
        f"Lambda ROI difference\n{results.difference_label}",
        results.x_label,
        results.y_label,
        "coolwarm",
        levels=diff_levels,
    )
    save_grid_png(
        plt,
        output_dir / f"{prefix}_03_binary_mask.png",
        results.binary_mask.astype(float),
        results.x_edges,
        results.y_edges,
        f"Binary {results.args.top_source} mask (>= {results.args.binary_threshold:g})",
        results.x_label,
        results.y_label,
        "gray_r",
        levels=(0, 1),
    )
    save_grid_png(
        plt,
        output_dir / f"{prefix}_04_masked_difference.png",
        results.masked_difference_grid,
        results.x_edges,
        results.y_edges,
        f"Masked lambda ROI difference ({results.args.top_source} mask)",
        results.x_label,
        results.y_label,
        "coolwarm",
        levels=diff_levels,
    )
    save_grid_png(
        plt,
        output_dir / f"{prefix}_05_top_region_centers.png",
        results.masked_difference_grid,
        results.x_edges,
        results.y_edges,
        f"Top {len(results.top_regions)} ranked bright-region centers",
        results.x_label,
        results.y_label,
        "coolwarm",
        levels=diff_levels,
        regions=results.top_regions,
    )
    save_grid_png(
        plt,
        output_dir / f"{prefix}_06_mux_derivative_peak_x.png",
        results.mux_derivative_peak_x_grid,
        results.x_edges,
        results.y_edges,
        (
            f"Mux first-derivative max x "
            f"(range {results.mux_derivative_x_indices[0]}:{results.mux_derivative_x_indices[1]})"
        ),
        results.x_label,
        results.y_label,
        "viridis",
        levels=finite_levels(results.mux_derivative_peak_x_grid),
        regions=results.top_regions,
    )
    save_grid_png(
        plt,
        output_dir / f"{prefix}_07_mux_derivative_peak_value.png",
        results.mux_derivative_peak_value_grid,
        results.x_edges,
        results.y_edges,
        (
            f"Mux first-derivative max value "
            f"(range {results.mux_derivative_x_indices[0]}:{results.mux_derivative_x_indices[1]})"
        ),
        results.x_label,
        results.y_label,
        "magma",
        levels=finite_levels(results.mux_derivative_peak_value_grid),
        regions=results.top_regions,
    )

    for record in results.preview_records:
        save_detector_preview_png(
            plt,
            output_dir / f"{prefix}_lambda_preview_p{record.region['scan_order']}.png",
            record,
            "Point",
            "ROI-weighted lambda spectrum",
            results.difference_x_indices[0],
            results.difference_x_indices[1],
            "magma",
            roi=results.spectrum_roi,
        )

    for record, derivative_record in zip(results.mux_records, results.mux_derivative_records):
        save_detector_preview_png(
            plt,
            output_dir / f"{prefix}_mux_preview_p{record.region['scan_order']}.png",
            record,
            "Mux point",
            "ROI-weighted mux spectrum",
            results.difference_x_indices[0],
            results.difference_x_indices[1],
            "magma",
            roi=results.spectrum_roi,
            derivative_record=derivative_record,
        )

    save_region_coordinates_csv(output_dir / f"{prefix}_top_region_coordinates.csv", results)
    (output_dir / f"{prefix}_scan_info.txt").write_text(
        "\n".join(scan_setup_lines(results, detailed=True)) + "\n",
        encoding="utf-8",
    )
    return output_dir


def add_detector_preview(
    layout,
    pg,
    row,
    record,
    title_prefix,
    profile_label,
    x1,
    x2,
    roi=None,
    derivative_record=None,
    adjustable_image_levels=False,
):
    x_edges, y_edges = detector_edges(record.image.shape)
    levels = finite_levels(record.image, lower=1.0, upper=99.5)
    image_title = (
        f"{title_prefix} {record.region['scan_order']} | "
        f"center ({int(round(record.region['center_x']))}, {int(round(record.region['center_y']))}) | "
        f"frame {record.position['frame_index']}"
    )
    if adjustable_image_levels:
        image_plot = layout.addPlot(row=row, col=0, title=image_title)
        image_plot.setLabel("bottom", "detector x pixel")
        image_plot.setLabel("left", "detector y pixel")
        image_plot.showGrid(x=True, y=True, alpha=0.2)
        image_item = pg.ImageItem(axisOrder="row-major")
        image_item.setImage(np.asarray(record.image, dtype=float), autoLevels=levels is None)
        if levels is not None:
            image_item.setLevels(levels)
        image_item.setRect(-0.5, -0.5, record.image.shape[1], record.image.shape[0])
        if hasattr(image_item, "setColorMap"):
            image_item.setColorMap(get_colormap(pg, "magma"))
        image_plot.addItem(image_item)
        histogram = pg.HistogramLUTItem(image=image_item)
        if hasattr(histogram.gradient, "setColorMap"):
            histogram.gradient.setColorMap(get_colormap(pg, "magma"))
        if levels is not None:
            histogram.setLevels(*levels)
        try:
            histogram.setMaximumWidth(115)
        except Exception:
            pass
        layout.addItem(histogram, row=row, col=1)
        profile_col = 2
    else:
        image_plot = add_map(
            layout,
            pg,
            row,
            0,
            image_title,
            record.image,
            x_edges,
            y_edges,
            "magma",
            levels=levels,
        )
        profile_col = 1
    image_plot.setLabel("bottom", "detector x pixel")
    image_plot.setLabel("left", "detector y pixel")
    add_roi_overlay_pyqt(
        pg,
        image_plot,
        record.image.shape,
        roi or {"kind": "row_range", "row_bounds": [record.roi_start, record.roi_stop]},
    )
    image_plot.addItem(pg.InfiniteLine(pos=x1, angle=90, pen=pg.mkPen((0, 229, 255), width=2)))
    image_plot.addItem(pg.InfiniteLine(pos=x2, angle=90, pen=pg.mkPen((255, 223, 93), width=2)))

    profile_plot = layout.addPlot(row=row, col=profile_col, title=f"{title_prefix} {record.region['scan_order']} | {profile_label}")
    profile_plot.plot(np.arange(record.profile.size), record.profile, pen=pg.mkPen((31, 119, 180), width=2))
    profile_plot.addItem(pg.InfiniteLine(pos=x1, angle=90, pen=pg.mkPen((0, 229, 255), width=2)))
    profile_plot.addItem(pg.InfiniteLine(pos=x2, angle=90, pen=pg.mkPen((255, 223, 93), width=2)))
    profile_plot.setLabel("bottom", "detector x pixel")
    profile_plot.setLabel("left", profile_label)
    profile_plot.showGrid(x=True, y=True, alpha=0.25)

    if derivative_record is not None:
        profile_plot.setTitle(
            f"{title_prefix} {record.region['scan_order']} | "
            f"{profile_label} + d(mux)/dx max x={format_metadata_number(derivative_record.peak_x)}"
        )
        derivative_view = pg.ViewBox()
        right_axis = profile_plot.getAxis("right")
        right_axis.show()
        right_axis.setLabel("first derivative")
        right_axis.linkToView(derivative_view)
        profile_plot.scene().addItem(derivative_view)
        derivative_view.setXLink(profile_plot)
        derivative_view.addItem(
            pg.PlotCurveItem(
                derivative_record.x_values,
                derivative_record.derivative,
                pen=pg.mkPen((214, 39, 40), width=2),
            )
        )
        if np.isfinite(derivative_record.peak_x) and np.isfinite(derivative_record.peak_value):
            derivative_view.addItem(
                pg.ScatterPlotItem(
                    spots=[
                        {
                            "pos": (derivative_record.peak_x, derivative_record.peak_value),
                            "brush": pg.mkBrush("k"),
                            "pen": pg.mkPen("w", width=1),
                            "size": 10,
                        }
                    ]
                )
            )
            derivative_view.addItem(
                pg.InfiniteLine(pos=derivative_record.peak_x, angle=90, pen=pg.mkPen("k", width=1))
            )

        def update_derivative_view():
            derivative_view.setGeometry(profile_plot.vb.sceneBoundingRect())
            derivative_view.linkedViewChanged(profile_plot.vb, derivative_view.XAxis)

        update_derivative_view()
        profile_plot.vb.sigResized.connect(update_derivative_view)
        profile_plot._derivative_overlay = (derivative_view, update_derivative_view)

        legend = profile_plot.addLegend(offset=(10, 10))
        legend.addItem(
            profile_plot.plot([], [], pen=pg.mkPen((31, 119, 180), width=2)),
            "mux",
        )
        legend.addItem(
            pg.PlotCurveItem(
                [],
                [],
                pen=pg.mkPen((214, 39, 40), width=2),
            ),
            "d(mux)/dx",
        )


def preview_combo_label(record, derivative_record=None):
    label = (
        f"p{record.region['scan_order']} | "
        f"center ({format_metadata_number(record.region['center_x'])}, "
        f"{format_metadata_number(record.region['center_y'])}) | "
        f"frame {record.position['frame_index']}"
    )
    if derivative_record is not None:
        label += f" | dmax x={format_metadata_number(derivative_record.peak_x)}"
    return label


def build_preview_selector_tab(
    pg,
    QtWidgets,
    records,
    title_prefix,
    profile_label_template,
    x1,
    x2,
    roi=None,
    derivative_records=None,
    adjustable_image_levels=False,
):
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)

    if not records:
        layout.addWidget(QtWidgets.QLabel("No preview_spectrum entries matched the selected points."))
        return widget

    combo = QtWidgets.QComboBox()
    stack = QtWidgets.QStackedWidget()
    for index, record in enumerate(records):
        derivative_record = derivative_records[index] if derivative_records is not None else None
        combo.addItem(preview_combo_label(record, derivative_record), index)
        graphics = pg.GraphicsLayoutWidget()
        add_detector_preview(
            graphics,
            pg,
            0,
            record,
            title_prefix,
            profile_label_template(record),
            x1,
            x2,
            roi=roi,
            derivative_record=derivative_record,
            adjustable_image_levels=adjustable_image_levels,
        )
        stack.addWidget(graphics)

    combo.currentIndexChanged.connect(stack.setCurrentIndex)
    layout.addWidget(combo)
    layout.addWidget(stack, stretch=1)
    widget._preview_controls = (combo, stack)
    return widget


def build_window(results: AnalysisResults, pg, QtWidgets):
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Fly-Scanning Mapping PyQtGraph Viewer")
    tabs = QtWidgets.QTabWidget()
    window.setCentralWidget(tabs)

    overview = pg.GraphicsLayoutWidget()
    tabs.addTab(overview, "Overview")
    diff_levels = finite_levels(results.difference_grid, symmetric=True)
    add_scan_setup_label(overview, pg, 0, 0, 2, results)
    add_map(
        overview,
        pg,
        1,
        0,
        f"Lambda scalar map ({results.args.reduction})",
        results.lambda_grid,
        results.x_edges,
        results.y_edges,
        "viridis",
        levels=finite_levels(results.lambda_grid),
        x_label=results.x_label,
        y_label=results.y_label,
    )
    add_map(
        overview,
        pg,
        1,
        1,
        f"Lambda ROI difference\n{results.difference_label}",
        results.difference_grid,
        results.x_edges,
        results.y_edges,
        "coolwarm",
        levels=diff_levels,
        x_label=results.x_label,
        y_label=results.y_label,
    )

    selection = pg.GraphicsLayoutWidget()
    tabs.addTab(selection, "Region Selection")
    add_scan_setup_label(selection, pg, 0, 0, 2, results)
    add_map(
        selection,
        pg,
        1,
        0,
        f"Lambda ROI spectral-difference map\n{results.difference_label}",
        results.difference_grid,
        results.x_edges,
        results.y_edges,
        "coolwarm",
        levels=diff_levels,
        x_label=results.x_label,
        y_label=results.y_label,
    )
    add_map(
        selection,
        pg,
        1,
        1,
        f"Binary {results.args.top_source} mask (>= {results.args.binary_threshold:g})",
        results.binary_mask.astype(float),
        results.x_edges,
        results.y_edges,
        "gray_r",
        levels=(0, 1),
        x_label=results.x_label,
        y_label=results.y_label,
    )
    add_map(
        selection,
        pg,
        2,
        0,
        f"Masked lambda ROI difference ({results.args.top_source} mask)\n{results.difference_label}",
        results.masked_difference_grid,
        results.x_edges,
        results.y_edges,
        "coolwarm",
        levels=diff_levels,
        x_label=results.x_label,
        y_label=results.y_label,
    )
    labeled_plot = add_map(
        selection,
        pg,
        2,
        1,
        f"Top {len(results.top_regions)} ranked bright-region centers",
        results.masked_difference_grid,
        results.x_edges,
        results.y_edges,
        "coolwarm",
        levels=diff_levels,
        x_label=results.x_label,
        y_label=results.y_label,
    )
    add_region_markers(pg, labeled_plot, results.top_regions)

    tabs.addTab(build_scan_info_tab(results, QtWidgets), "Scan Info")

    derivative_maps = pg.GraphicsLayoutWidget()
    tabs.addTab(derivative_maps, "Mux Derivative")
    add_scan_setup_label(derivative_maps, pg, 0, 0, 2, results)
    peak_x_plot = add_map(
        derivative_maps,
        pg,
        1,
        0,
        (
            f"Mux first-derivative max x\n"
            f"range {results.mux_derivative_x_indices[0]}:{results.mux_derivative_x_indices[1]}"
        ),
        results.mux_derivative_peak_x_grid,
        results.x_edges,
        results.y_edges,
        "viridis",
        levels=finite_levels(results.mux_derivative_peak_x_grid),
        x_label=results.x_label,
        y_label=results.y_label,
    )
    add_region_markers(pg, peak_x_plot, results.top_regions)
    peak_value_plot = add_map(
        derivative_maps,
        pg,
        1,
        1,
        (
            f"Mux first-derivative max value\n"
            f"range {results.mux_derivative_x_indices[0]}:{results.mux_derivative_x_indices[1]}"
        ),
        results.mux_derivative_peak_value_grid,
        results.x_edges,
        results.y_edges,
        "magma",
        levels=finite_levels(results.mux_derivative_peak_value_grid),
        x_label=results.x_label,
        y_label=results.y_label,
    )
    add_region_markers(pg, peak_value_plot, results.top_regions)

    tabs.addTab(
        build_preview_selector_tab(
            pg,
            QtWidgets,
            results.preview_records,
            "Point",
            lambda _record: "ROI-weighted lambda spectrum",
            results.difference_x_indices[0],
            results.difference_x_indices[1],
            roi=results.spectrum_roi,
        ),
        "Lambda Previews",
    )

    tabs.addTab(
        build_preview_selector_tab(
            pg,
            QtWidgets,
            results.mux_records,
            "Mux point",
            lambda _record: "ROI-weighted mux spectrum",
            results.difference_x_indices[0],
            results.difference_x_indices[1],
            roi=results.spectrum_roi,
            derivative_records=results.mux_derivative_records,
            adjustable_image_levels=True,
        ),
        "Mux Previews",
    )

    window.resize(1500, 950)
    return window


def run_gui(results: AnalysisResults):
    pg, _QtCore, _QtGui, QtWidgets = require_pyqtgraph()
    pg.setConfigOptions(background="w", foreground="k", antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = build_window(results, pg, QtWidgets)
    window.show()
    return app.exec()


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf", default=DEFAULT_HDF, help="Mapping HDF filename or path.")
    parser.add_argument("--flat", default=DEFAULT_FLAT, help="Flatfield HDF filename or path.")
    parser.add_argument("--fast-axis", default=FAST_AXIS, choices=["horizontal", "vertical"])
    parser.add_argument(
        "--plot-grid",
        default="acquired",
        choices=["acquired", "full"],
        help="Plot only acquired detector rows or the full commanded metadata grid padded with NaN.",
    )
    parser.add_argument(
        "--coords",
        default=COORDS,
        choices=["index", "motor"],
        help="Use aerotech motor coordinates for physically scaled maps; choose index for scan-grid indices.",
    )
    parser.add_argument("--reduction", default=REDUCTION, choices=["mean", "sum"])
    parser.add_argument("--roi-y", nargs=2, type=int, default=PREVIEW_ROI_VERTICAL, metavar=("START", "STOP"))
    parser.add_argument(
        "--spectrum-roi-json",
        default=DEFAULT_SPECTRUM_ROI_JSON,
        help="ROI JSON used for lambda/mux spectra. Set to '' to fall back to --roi-y.",
    )
    parser.add_argument("--x1", type=int, default=SPECTRUM_DIFFERENCE_X1)
    parser.add_argument("--x2", type=int, default=SPECTRUM_DIFFERENCE_X2)
    parser.add_argument(
        "--derivative-x",
        nargs=2,
        type=int,
        default=MUX_DERIVATIVE_X_RANGE,
        metavar=("START", "STOP"),
        help="Detector x range used to find the max first derivative in mux spectra.",
    )
    parser.add_argument("--top-n-regions", type=int, default=TOP_N_REGIONS)
    parser.add_argument("--top-source", default=TOP_N_SOURCE, choices=["lambda", "it"])
    parser.add_argument("--binary-threshold", type=float, default=BINARY_THRESHOLD)
    parser.add_argument("--min-region-pixels", type=int, default=SEGMENTATION_MIN_REGION_PIXELS)
    parser.add_argument("--smooth-sigma", type=float, default=SEGMENTATION_SMOOTH_SIGMA)
    parser.add_argument("--connectivity", type=int, default=SEGMENTATION_CONNECTIVITY, choices=[1, 2])
    parser.add_argument("--center-method", default=REGION_CENTER_METHOD, choices=["weighted", "geometric"])
    parser.add_argument("--rank-metric", default=REGION_RANK_METRIC, choices=["peak", "sum", "mean"])
    parser.add_argument("--snake-axes", action="store_true", default=SNAKE_AXES)
    parser.add_argument("--preview", nargs="*", default=list(PREVIEW_SPECTRUM), help="Preview labels, e.g. p1 p3, p1,p3, or all.")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--save-results",
        dest="save_results",
        action="store_true",
        default=True,
        help="Save static PNG maps/previews, coordinate CSV, and scan info text. Enabled by default.",
    )
    parser.add_argument(
        "--no-save-results",
        dest="save_results",
        action="store_false",
        help="Do not save static mapping result files.",
    )
    parser.add_argument(
        "--results-dir",
        default="mapping_results",
        help="Directory where result subfolders are written.",
    )
    parser.add_argument(
        "--output-folder",
        default="sample",
        choices=["sample", "hdf"],
        help="Name result subfolders by sample metadata or by HDF filename stem.",
    )
    parser.add_argument(
        "--output-folder-name",
        default=None,
        help="Override the generated result subfolder name.",
    )
    parser.add_argument("--compute-only", action="store_true", help="Run analysis and print coordinates without opening PyQtGraph.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.roi_y = tuple(args.roi_y)
    args.derivative_x = tuple(args.derivative_x)
    if args.spectrum_roi_json is not None and not str(args.spectrum_roi_json).strip():
        args.spectrum_roi_json = None
    args.preview = parse_preview_values(args.preview)
    if not args.compute_only:
        require_pyqtgraph()
    results = build_analysis(args)
    if args.compute_only:
        print("Computation completed. GUI was not opened because --compute-only was set.")
    if args.save_results:
        output_dir = save_analysis_results(results, args.results_dir)
        print(f"Saved mapping results to: {output_dir}")
    if args.compute_only:
        return 0
    return run_gui(results)


if __name__ == "__main__":
    raise SystemExit(main())
