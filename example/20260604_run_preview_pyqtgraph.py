#!/usr/bin/env python3
"""PyQtGraph batch preview viewer for 20260604_run_preview.ipynb.

This script keeps the notebook untouched. It reuses the notebook's HDF5
discovery, nearest-flatfield pairing, ROI, and spectrum-preview math, but
draws the preview figures in a PyQtGraph desktop window instead of writing
Plotly/HTML output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR_DEFAULT = SCRIPT_DIR / "data"
BAD_FILES_DEFAULT = SCRIPT_DIR / "bad_files.txt"
ROI_JSON_DEFAULT = SCRIPT_DIR / "saved_rois" / "Pt_L3_roi.json"
SAFE_PLUGIN_DIR = SCRIPT_DIR / ".hdf5_plugin"
EPICS_UNIX_OFFSET = 631_152_000


def configure_environment() -> None:
    SAFE_PLUGIN_DIR.mkdir(exist_ok=True)
    if not Path(os.environ.get("HDF5_PLUGIN_PATH", "")).exists():
        os.environ["HDF5_PLUGIN_PATH"] = str(SAFE_PLUGIN_DIR)
    try:
        import hdf5plugin  # noqa: F401
    except Exception:
        pass

    try:
        import skimage  # noqa: F401
    except ModuleNotFoundError:
        sys.modules.setdefault("skimage", types.ModuleType("skimage"))


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or SCRIPT_DIR).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "Python_codes" / "Dispersive_XAS").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root.")


configure_environment()
REPO_ROOT = find_repo_root()
PYTHON_CODES_DIR = REPO_ROOT / "Python_codes"
if str(PYTHON_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODES_DIR))

from Dispersive_XAS.core.roi import (  # noqa: E402
    infer_tilted_band_roi_from_paths,
    load_roi_json,
    prepare_roi_weights,
    roi_boundary_rows,
    roi_weighted_column_mean,
)
from Dispersive_XAS.web.batch import _compute_chunk_specs  # noqa: E402


@dataclass(frozen=True)
class ExportEntry:
    path: Path
    name: str
    uid: str
    scan_id: int
    scan_time: datetime | None
    sample_name: str
    plan_name: str
    shape: tuple[int, ...]

    @property
    def time_label(self) -> str:
        return self.scan_time.strftime("%Y-%m-%d:%H-%M") if self.scan_time else ""

    @property
    def nframes(self) -> int:
        return int(self.shape[0]) if len(self.shape) >= 3 else 1


@dataclass(frozen=True)
class BatchPair:
    data_entry: ExportEntry
    flat_entry: ExportEntry
    delta_minutes: float


@dataclass
class PreviewResult:
    pair: BatchPair
    selected_roi: dict[str, object] | None
    row_range: tuple[int, int] | None
    roi: dict[str, object] | None
    norm_range: tuple[int, int]
    start_frame: int
    end_frame: int
    per_frame_specs: np.ndarray
    avg_specs: np.ndarray
    avg_frame_indices: np.ndarray
    snapshot_frame: int
    flat_snapshot_frame: int
    snapshot_data: np.ndarray
    snapshot_flat: np.ndarray
    snapshot_mux: np.ndarray


def decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_jsonish(value):
    value = decode_text(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except Exception:
                return value
    return value


def sanitize_name(value) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")


def preview_stem(entry: ExportEntry) -> str:
    if entry.scan_time is None:
        return sanitize_name(entry.name)
    return f"{entry.scan_time.strftime('%Y%m%d_%H%M')}{sanitize_name(entry.sample_name)}"


def bad_file_filters(path: Path) -> tuple[set[Path], set[str]]:
    bad_paths: set[Path] = set()
    bad_names: set[str] = set()
    if not path.exists():
        return bad_paths, bad_names
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.split("#", 1)[0].strip()
        if not item:
            continue
        raw_path = Path(item).expanduser()
        resolved = raw_path if raw_path.is_absolute() else SCRIPT_DIR / raw_path
        bad_paths.add(resolved.resolve())
        bad_names.add(raw_path.name)
    return bad_paths, bad_names


def is_bad_h5_path(path: Path, bad_files_path: Path) -> bool:
    bad_paths, bad_names = bad_file_filters(bad_files_path)
    path = Path(path).expanduser()
    return path.resolve() in bad_paths or path.name in bad_names


def default_entry_key(h5_file: h5py.File) -> str | None:
    entry_key = decode_text(h5_file.attrs.get("default"))
    if entry_key and entry_key in h5_file:
        return str(entry_key)
    if "entry" in h5_file:
        return "entry"
    for key, value in h5_file.items():
        if isinstance(value, h5py.Group) and "instrument/bluesky/metadata" in value:
            return key
    return None


def h5_metadata_value(h5_file: h5py.File, key: str, default=None):
    entry_key = default_entry_key(h5_file)
    if not entry_key:
        return default
    metadata_path = f"{entry_key}/instrument/bluesky/metadata"
    if metadata_path not in h5_file:
        return default
    metadata = h5_file[metadata_path]
    if key not in metadata:
        return default
    return parse_jsonish(metadata[key][()])


def lambda_dataset_path(h5_file: h5py.File) -> str:
    for key in ("entry/data/data", "entry/instrument/detector/data"):
        if key in h5_file:
            return h5_file[key].name
    entry_key = default_entry_key(h5_file)
    if entry_key:
        for key in (
            f"{entry_key}/instrument/bluesky/streams/primary/lambda_250k/value",
            f"{entry_key}/instrument/bluesky/streams/primary/lambda/value",
        ):
            if key in h5_file:
                return h5_file[key].name
    raise KeyError("Could not find a lambda detector dataset in the HDF5 file.")


def detector_dataset(h5_file: h5py.File):
    return h5_file[lambda_dataset_path(h5_file)]


def scan_time_from_file(h5_file: h5py.File, path: Path) -> datetime | None:
    start_time_unix = h5_metadata_value(h5_file, "start.time")
    if start_time_unix is not None:
        return datetime.fromtimestamp(float(start_time_unix))
    match = re.match(r"(?P<stamp>\d{12})-", path.name)
    if match:
        return datetime.strptime(match.group("stamp"), "%Y%m%d%H%M")
    return None


def inspect_export(path: Path) -> ExportEntry:
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as h5_file:
        dataset = detector_dataset(h5_file)
        scan_time = scan_time_from_file(h5_file, path)
        uid = decode_text(h5_metadata_value(h5_file, "start.uid", path.stem))
        scan_id = int(h5_metadata_value(h5_file, "start.scan_id", -1))
        sample_name = decode_text(h5_metadata_value(h5_file, "start.sample_name", "")) or path.stem
        plan_name = decode_text(h5_metadata_value(h5_file, "start.plan_name", ""))
        shape = tuple(int(v) for v in dataset.shape)
    return ExportEntry(
        path=path,
        name=path.name,
        uid=str(uid),
        scan_id=scan_id,
        scan_time=scan_time,
        sample_name=str(sample_name),
        plan_name=str(plan_name),
        shape=shape,
    )


def normalize_selected_path(path_setting: str | Path, data_dir: Path) -> Path:
    path = Path(path_setting).expanduser()
    if not path.is_absolute():
        path = data_dir / path
    return path.resolve()


def discover_paths(args) -> tuple[list[Path], list[Path], list[dict[str, str]]]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    bad_files_path = Path(args.bad_files).expanduser().resolve()
    skipped: list[dict[str, str]] = []

    data_paths = [
        path
        for path in sorted(data_dir.glob(args.data_glob))
        if not is_bad_h5_path(path, bad_files_path)
        and not any(token.lower() in path.name.lower() for token in args.data_exclude_token)
    ]
    if args.only_data:
        selected = {normalize_selected_path(path, data_dir) for path in args.only_data}
        data_paths = [path for path in data_paths if path.resolve() in selected]

    if args.fixed_flatfield:
        flat_paths = [normalize_selected_path(args.fixed_flatfield, data_dir)]
    else:
        seen: set[Path] = set()
        flat_paths = []
        token = args.flatfield_token.lower()
        for pattern in ("*.hdf", "*.h5", "*.HDF", "*.H5"):
            for path in sorted(data_dir.glob(pattern)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                if token in path.name.lower() and not is_bad_h5_path(path, bad_files_path):
                    flat_paths.append(path)
    return data_paths, flat_paths, skipped


def nearest_entry(
    target_entry: ExportEntry,
    candidate_entries: Sequence[ExportEntry],
    prefer: str = "nearest",
) -> ExportEntry:
    if not candidate_entries:
        raise ValueError("No flatfield entries are available.")
    if target_entry.scan_time is None:
        raise ValueError(f"{target_entry.name} has no timestamp.")
    if prefer == "after":
        after = [
            entry for entry in candidate_entries
            if entry.scan_time is not None and entry.scan_time >= target_entry.scan_time
        ]
        if after:
            return min(after, key=lambda entry: entry.scan_time - target_entry.scan_time)
    elif prefer == "before":
        before = [
            entry for entry in candidate_entries
            if entry.scan_time is not None and entry.scan_time <= target_entry.scan_time
        ]
        if before:
            return min(before, key=lambda entry: target_entry.scan_time - entry.scan_time)
    elif prefer != "nearest":
        raise ValueError(f"Unsupported flatfield preference: {prefer!r}")

    return min(
        [entry for entry in candidate_entries if entry.scan_time is not None],
        key=lambda entry: abs((entry.scan_time - target_entry.scan_time).total_seconds()),
    )


def discover_batch_pairs(args) -> tuple[list[BatchPair], list[dict[str, str]]]:
    data_paths, flat_paths, skipped = discover_paths(args)

    flat_entries: list[ExportEntry] = []
    for path in flat_paths:
        try:
            flat_entries.append(inspect_export(path))
        except Exception as exc:
            skipped.append({
                "role": "flatfield",
                "name": path.name,
                "path": str(path),
                "reason": f"{type(exc).__name__}: {exc}",
            })
    if not flat_entries:
        raise RuntimeError(
            f"No usable flatfields matched {args.flatfield_token!r} in {Path(args.data_dir).resolve()}."
        )

    pairs: list[BatchPair] = []
    for path in data_paths:
        try:
            data_entry = inspect_export(path)
            flat_entry = flat_entries[0] if args.fixed_flatfield else nearest_entry(
                data_entry,
                flat_entries,
                prefer=args.flatfield_preference,
            )
            if data_entry.scan_time is None or flat_entry.scan_time is None:
                delta_minutes = float("nan")
            else:
                delta_minutes = abs((flat_entry.scan_time - data_entry.scan_time).total_seconds()) / 60.0
            pairs.append(BatchPair(data_entry=data_entry, flat_entry=flat_entry, delta_minutes=delta_minutes))
        except Exception as exc:
            skipped.append({
                "role": "data",
                "name": path.name,
                "path": str(path),
                "reason": f"{type(exc).__name__}: {exc}",
            })
    return pairs, skipped


def entry_summary(entry: ExportEntry) -> dict[str, object]:
    return {
        "name": entry.name,
        "sample_name": entry.sample_name,
        "scan_id": int(entry.scan_id),
        "time": entry.scan_time.isoformat() if entry.scan_time else None,
        "shape": list(entry.shape),
        "nframes": int(entry.nframes),
        "path": str(entry.path),
    }


def roi_json_path_for_entry(args, entry: ExportEntry) -> Path | None:
    if args.roi_json is None:
        return None
    raw = Path(args.roi_json).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (SCRIPT_DIR / raw).resolve()


def infer_tilted_roi_direct(
    data_path: Path,
    flat_path: Path,
    frame_index: int,
    frame_average: int,
    threshold_fraction: float,
    shrink_fraction: float,
    smooth_sigma_rows: float,
    smooth_sigma_cols: float,
    median_size: int,
) -> dict[str, object]:
    """Use the package ROI fitter via temporary legacy views when needed."""
    # The package helper expects /entry/data/data. The exported files used here
    # usually do not have that layout, so create lightweight external-link views.
    view_dir = SCRIPT_DIR / "derived_legacy_h5"
    view_dir.mkdir(exist_ok=True)
    data_view = ensure_legacy_lambda_view(data_path, view_dir / f"{data_path.stem}.h5")
    flat_view = ensure_legacy_lambda_view(flat_path, view_dir / f"{flat_path.stem}.h5")
    return infer_tilted_band_roi_from_paths(
        data_path=str(data_view),
        flat_path=str(flat_view),
        frame_index=frame_index,
        frame_average=frame_average,
        threshold_fraction=threshold_fraction,
        shrink_fraction=shrink_fraction,
        smooth_sigma_rows=smooth_sigma_rows,
        smooth_sigma_cols=smooth_sigma_cols,
        median_size=median_size,
    )


def ensure_legacy_lambda_view(source_path: Path, view_path: Path) -> Path:
    source_path = Path(source_path).resolve()
    view_path = Path(view_path).resolve()
    view_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source_file:
        target_dataset = lambda_dataset_path(source_file)
    if view_path.exists() or view_path.is_symlink():
        view_path.unlink()
    with h5py.File(view_path, "w") as view_file:
        data_group = view_file.create_group("entry").create_group("data")
        data_group["data"] = h5py.ExternalLink(str(source_path), target_dataset)
        view_file.attrs["source_file"] = str(source_path)
        view_file.attrs["source_dataset"] = target_dataset
    return view_path


def selected_roi_for_pair(args, pair: BatchPair) -> dict[str, object] | None:
    roi_json_path = roi_json_path_for_entry(args, pair.data_entry)
    if args.use_saved_roi and roi_json_path is not None and roi_json_path.exists():
        return load_roi_json(roi_json_path)
    if args.row_range is not None:
        return {
            "kind": "row_range",
            "row_start": int(args.row_range[0]),
            "row_stop": int(args.row_range[1]),
            "row_bounds": [int(args.row_range[0]), int(args.row_range[1])],
        }
    if args.use_tilted_roi:
        return infer_tilted_roi_direct(
            pair.data_entry.path,
            pair.flat_entry.path,
            frame_index=args.snapshot_frame,
            frame_average=args.tilted_roi_frame_average,
            threshold_fraction=args.tilted_roi_threshold_fraction,
            shrink_fraction=args.tilted_roi_shrink_fraction,
            smooth_sigma_rows=args.tilted_roi_smooth_sigma_rows,
            smooth_sigma_cols=args.tilted_roi_smooth_sigma_cols,
            median_size=args.median_size,
        )
    return None


def infer_norm_range(
    data_path: Path,
    flat_path: Path,
    row_range: tuple[int, int] | None,
    roi: Mapping[str, object] | None,
    window: int,
    sample_frames: int,
) -> tuple[int, int]:
    with h5py.File(flat_path, "r") as flat_file:
        flat_avg = np.asarray(detector_dataset(flat_file)[:], dtype=np.float32).mean(axis=0)

    with h5py.File(data_path, "r") as data_file:
        data_ds = detector_dataset(data_file)
        stop = min(int(sample_frames), int(data_ds.shape[0]))
        data = np.asarray(data_ds[:stop], dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        mux = np.log(np.clip(flat_avg[None, :, :], 1e-6, None) / np.clip(data, 1e-6, None))
    mux[~np.isfinite(mux)] = 0.0
    mean_spec = roi_weighted_column_mean(mux, row_range=row_range, roi=roi).mean(axis=0)

    width = int(mean_spec.shape[0])
    if width <= 0:
        raise RuntimeError("Could not infer a normalization window from an empty spectrum.")
    if int(window) >= width:
        return 0, width

    best_span = -np.inf
    best_start = 0
    for start in range(0, width - int(window) + 1):
        segment = mean_spec[start:start + int(window)]
        span = float(np.nanmax(segment) - np.nanmin(segment))
        if span > best_span:
            best_span = span
            best_start = start
    return int(best_start), int(best_start + int(window))


def compute_preview(pair: BatchPair, args) -> PreviewResult:
    selected_roi = selected_roi_for_pair(args, pair)
    row_range: tuple[int, int] | None = None
    roi: dict[str, object] | None = None
    if selected_roi is not None:
        row_range = tuple(int(v) for v in selected_roi["row_bounds"])
        if selected_roi.get("kind") == "tilted_band":
            roi = dict(selected_roi)
    elif args.row_range is not None:
        row_range = tuple(int(v) for v in args.row_range)

    norm_range = infer_norm_range(
        pair.data_entry.path,
        pair.flat_entry.path,
        row_range=row_range,
        roi=roi,
        window=args.norm_window,
        sample_frames=args.norm_sample_frames,
    )

    with h5py.File(pair.data_entry.path, "r") as data_file, h5py.File(pair.flat_entry.path, "r") as flat_file:
        data_ds = detector_dataset(data_file)
        flat_ds = detector_dataset(flat_file)
        nframes = int(data_ds.shape[0])
        start_frame = max(0, int(args.start_frame))
        end_frame = nframes if args.end_frame is None else min(nframes, int(args.end_frame))
        if end_frame <= start_frame:
            raise ValueError(f"Empty frame range {start_frame}:{end_frame} for {pair.data_entry.name}.")

        roi_spec, (fr0, fr1), row_weights, col_weight_sum = prepare_roi_weights(
            (int(data_ds.shape[1]), int(data_ds.shape[2])),
            row_range=row_range,
            roi=roi,
            dtype=np.float32,
        )
        row_range = tuple(int(v) for v in roi_spec["row_bounds"])
        if roi_spec.get("kind") == "tilted_band":
            roi = dict(roi_spec)

        flat_avg_roi = np.asarray(flat_ds[:, fr0:fr1, :], dtype=np.float32).mean(axis=0)
        flat_avg_full = np.asarray(flat_ds[:], dtype=np.float32).mean(axis=0)

        snapshot_frame = int(np.clip(args.snapshot_frame, 0, nframes - 1))
        flat_snapshot_frame = int(np.clip(args.flat_snapshot_frame, 0, int(flat_ds.shape[0]) - 1))
        snapshot_data = np.asarray(data_ds[snapshot_frame], dtype=np.float32)
        snapshot_flat = np.asarray(flat_ds[flat_snapshot_frame], dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            snapshot_mux = np.log(
                np.clip(flat_avg_full, 1e-6, None) / np.clip(snapshot_data, 1e-6, None)
            )
        snapshot_mux[~np.isfinite(snapshot_mux)] = 0.0

        all_per_frame: list[np.ndarray] = []
        all_avg: list[np.ndarray] = []
        avg_frame_indices: list[np.ndarray] = []
        for chunk_start in range(start_frame, end_frame, int(args.chunk_size)):
            chunk_end = min(chunk_start + int(args.chunk_size), end_frame)
            per_frame_specs, specs_avg, n_chunk, num_groups, _width = _compute_chunk_specs(
                data_ds,
                flat_avg_roi,
                fr0,
                fr1,
                row_weights,
                col_weight_sum,
                chunk_start,
                chunk_end,
                int(args.aver_n),
                norm_range[0],
                norm_range[1],
                float(args.factor),
                median_size=int(args.median_size),
            )
            if n_chunk > 0:
                all_per_frame.append(per_frame_specs)
            if num_groups > 0:
                all_avg.append(specs_avg)
                avg_frame_indices.append(chunk_start + np.arange(num_groups) * int(args.aver_n))

    per_frame = np.vstack(all_per_frame) if all_per_frame else np.empty((0, 0), dtype=np.float32)
    avg_specs = np.vstack(all_avg) if all_avg else np.empty((0, per_frame.shape[1]), dtype=np.float32)
    avg_frames = np.concatenate(avg_frame_indices) if avg_frame_indices else np.empty((0,), dtype=int)
    return PreviewResult(
        pair=pair,
        selected_roi=selected_roi,
        row_range=row_range,
        roi=roi,
        norm_range=norm_range,
        start_frame=start_frame,
        end_frame=end_frame,
        per_frame_specs=per_frame,
        avg_specs=avg_specs,
        avg_frame_indices=avg_frames,
        snapshot_frame=snapshot_frame,
        flat_snapshot_frame=flat_snapshot_frame,
        snapshot_data=snapshot_data,
        snapshot_flat=snapshot_flat,
        snapshot_mux=snapshot_mux,
    )


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


def get_colormap(pg, name: str):
    try:
        return pg.colormap.get(name, source="matplotlib")
    except Exception:
        if name in {"gray", "gray_r"}:
            colors = [(255, 255, 255), (0, 0, 0)] if name == "gray_r" else [(0, 0, 0), (255, 255, 255)]
            return pg.ColorMap([0.0, 1.0], colors)
        if name == "coolwarm":
            return pg.ColorMap([0.0, 0.5, 1.0], [(59, 76, 192), (240, 240, 240), (180, 4, 38)])
        return pg.colormap.get("viridis")


def finite_levels(data, symmetric: bool = False, lower: float = 1.0, upper: float = 99.5):
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    if symmetric:
        limit = float(np.nanmax(np.abs(finite)))
        return (-limit, limit) if limit > 0 else None
    vmin, vmax = np.percentile(finite, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    return float(vmin), float(vmax)


def make_mesh_item(pg, x_edges, y_edges, z, cmap_name: str, levels=None):
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


def add_map(layout, pg, row, col, title, z, x_edges, y_edges, cmap_name, levels=None, invert_y=False):
    plot = layout.addPlot(row=row, col=col, title=title)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.addItem(make_mesh_item(pg, x_edges, y_edges, z, cmap_name, levels))
    if invert_y:
        plot.invertY(True)
    return plot


def detector_edges(shape):
    rows, cols = shape
    return np.arange(cols + 1, dtype=float) - 0.5, np.arange(rows + 1, dtype=float) - 0.5


def add_roi_overlay(pg, plot, image_shape, row_range, roi):
    if row_range is None and roi is None:
        return
    cols, top_rows, bottom_rows = roi_boundary_rows(image_shape, row_range=row_range, roi=roi)
    pen = pg.mkPen("c", width=2)
    plot.plot(cols, top_rows, pen=pen)
    plot.plot(cols, bottom_rows, pen=pen)


def add_norm_lines(pg, plot, norm_range):
    for x in norm_range:
        plot.addItem(pg.InfiniteLine(pos=int(x), angle=90, pen=pg.mkPen((0, 229, 255), width=1)))


def add_snapshot_tab(tabs, pg, result: PreviewResult):
    widget = pg.GraphicsLayoutWidget()
    tabs.addTab(widget, "Snapshots")
    x_edges, y_edges = detector_edges(result.snapshot_data.shape)
    items = [
        (
            f"Data frame {result.snapshot_frame}\n{result.pair.data_entry.name}",
            result.snapshot_data,
            "magma",
            False,
        ),
        (
            f"Flat frame {result.flat_snapshot_frame}\n{result.pair.flat_entry.name}",
            result.snapshot_flat,
            "magma",
            False,
        ),
        (
            f"Mux frame {result.snapshot_frame}\nlog(mean flat / data)",
            result.snapshot_mux,
            "magma",
            False,
        ),
    ]
    for col, (title, image, cmap, symmetric) in enumerate(items):
        plot = add_map(
            widget,
            pg,
            0,
            col,
            title,
            image,
            x_edges,
            y_edges,
            cmap,
            levels=finite_levels(image, symmetric=symmetric),
            invert_y=True,
        )
        plot.setLabel("bottom", "detector x pixel")
        plot.setLabel("left", "detector y pixel")
        add_roi_overlay(pg, plot, image.shape, result.row_range, result.roi)


def add_heatmap_tab(tabs, pg, result: PreviewResult):
    widget = pg.GraphicsLayoutWidget()
    tabs.addTab(widget, "Spectrum Heatmap")
    if result.per_frame_specs.size == 0:
        widget.addLabel("No spectra were computed.", row=0, col=0)
        return
    nframes, width = result.per_frame_specs.shape
    x_edges = np.arange(width + 1, dtype=float) - 0.5
    y_edges = np.arange(result.start_frame, result.start_frame + nframes + 1, dtype=float) - 0.5
    plot = add_map(
        widget,
        pg,
        0,
        0,
        f"Normalized spectrum heatmap | frames {result.start_frame}:{result.end_frame}",
        result.per_frame_specs,
        x_edges,
        y_edges,
        "magma",
        levels=finite_levels(result.per_frame_specs, lower=1.0, upper=99.0),
    )
    plot.setLabel("bottom", "detector x pixel")
    plot.setLabel("left", "frame")
    add_norm_lines(pg, plot, result.norm_range)


def plot_spectra_lines(pg, plot, specs, x, frame_indices, max_lines, label_prefix):
    if specs.size == 0:
        return
    n = int(specs.shape[0])
    count = min(n, int(max_lines))
    indices = np.linspace(0, n - 1, count, dtype=int) if count < n else np.arange(n)
    for line_index, spec_index in enumerate(indices):
        color = pg.intColor(line_index, hues=max(1, len(indices)), values=1, maxValue=255)
        pen = pg.mkPen(color, width=1)
        name = f"{label_prefix} {int(frame_indices[spec_index])}" if len(frame_indices) else None
        plot.plot(x, specs[spec_index], pen=pen, name=name)


def add_lines_tab(tabs, pg, result: PreviewResult, averaged: bool, max_lines: int):
    title = "Averaged Spectra" if averaged else "Frame Spectra"
    widget = pg.GraphicsLayoutWidget()
    tabs.addTab(widget, title)
    specs = result.avg_specs if averaged else result.per_frame_specs
    if specs.size == 0:
        widget.addLabel("No spectra were computed.", row=0, col=0)
        return
    width = int(specs.shape[1])
    x = np.arange(width)
    frame_indices = result.avg_frame_indices if averaged else np.arange(result.start_frame, result.start_frame + specs.shape[0])
    plot = widget.addPlot(row=0, col=0, title=title)
    plot.setLabel("bottom", "detector x pixel")
    plot.setLabel("left", "normalized mux")
    plot.showGrid(x=True, y=True, alpha=0.25)
    add_norm_lines(pg, plot, result.norm_range)
    plot_spectra_lines(
        pg,
        plot,
        specs,
        x,
        frame_indices,
        max_lines=max_lines,
        label_prefix="avg frame" if averaged else "frame",
    )


def build_summary_text(pairs: Sequence[BatchPair], skipped: Sequence[Mapping[str, str]], result: PreviewResult | None = None) -> str:
    lines: list[str] = []
    lines.append(f"Batch pairs: {len(pairs)}")
    lines.append(f"Skipped files: {len(skipped)}")
    if result is not None:
        pair = result.pair
        lines.append("")
        lines.append("Current pair")
        lines.append(f"  Data:      {pair.data_entry.name}")
        lines.append(f"  Flatfield: {pair.flat_entry.name}")
        lines.append(f"  Delta:     {pair.delta_minutes:.2f} min")
        lines.append(f"  Frames:    {result.start_frame}:{result.end_frame}")
        lines.append(f"  ROI rows:  {result.row_range}")
        lines.append(f"  ROI kind:  {(result.roi or result.selected_roi or {}).get('kind', 'row_range/all')}")
        lines.append(f"  Norm x:    {result.norm_range[0]}:{result.norm_range[1]}")
    lines.append("")
    lines.append("Discovered pairs")
    for index, pair in enumerate(pairs):
        lines.append(
            f"  [{index:02d}] {pair.data_entry.name} -> {pair.flat_entry.name} "
            f"({pair.delta_minutes:.2f} min)"
        )
    if skipped:
        lines.append("")
        lines.append("Skipped")
        for item in skipped:
            lines.append(f"  {item.get('role')}: {item.get('name')} - {item.get('reason')}")
    return "\n".join(lines)


class PreviewWindow:
    def __init__(self, args, pairs: Sequence[BatchPair], skipped: Sequence[Mapping[str, str]], pg, QtWidgets):
        self.args = args
        self.pairs = list(pairs)
        self.skipped = list(skipped)
        self.pg = pg
        self.QtWidgets = QtWidgets

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("DXAS Batch Preview - PyQtGraph")

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        outer.addLayout(controls)
        self.combo = QtWidgets.QComboBox()
        for index, pair in enumerate(self.pairs):
            self.combo.addItem(
                f"{index:02d}: {pair.data_entry.sample_name} -> {pair.flat_entry.sample_name} "
                f"({pair.delta_minutes:.1f} min)",
                index,
            )
        controls.addWidget(self.combo, stretch=1)

        self.load_button = QtWidgets.QPushButton("Load Preview")
        self.load_button.clicked.connect(self.load_selected)
        controls.addWidget(self.load_button)

        self.status_label = QtWidgets.QLabel("Ready")
        controls.addWidget(self.status_label)

        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs, stretch=1)
        self.add_summary_tab()

    def add_summary_tab(self, result: PreviewResult | None = None):
        text = self.QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(build_summary_text(self.pairs, self.skipped, result=result))
        self.tabs.addTab(text, "Summary")

    def load_selected(self):
        index = int(self.combo.currentData())
        pair = self.pairs[index]
        self.status_label.setText(f"Computing {pair.data_entry.name} ...")
        self.load_button.setEnabled(False)
        self.QtWidgets.QApplication.processEvents()
        try:
            result = compute_preview(pair, self.args)
            self.tabs.clear()
            self.add_summary_tab(result=result)
            add_snapshot_tab(self.tabs, self.pg, result)
            add_heatmap_tab(self.tabs, self.pg, result)
            add_lines_tab(self.tabs, self.pg, result, averaged=True, max_lines=self.args.max_avg_lines)
            add_lines_tab(self.tabs, self.pg, result, averaged=False, max_lines=self.args.max_frame_lines)
            self.status_label.setText(f"Loaded {pair.data_entry.sample_name}")
        except Exception as exc:
            self.status_label.setText(f"Failed: {type(exc).__name__}")
            self.QtWidgets.QMessageBox.critical(
                self.window,
                "Preview failed",
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            self.load_button.setEnabled(True)

    def show(self):
        self.window.resize(1500, 950)
        self.window.show()


def run_gui(args, pairs: Sequence[BatchPair], skipped: Sequence[Mapping[str, str]]) -> int:
    pg, _QtCore, _QtGui, QtWidgets = require_pyqtgraph()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer = PreviewWindow(args, pairs, skipped, pg, QtWidgets)
    if args.pair_index is not None:
        viewer.combo.setCurrentIndex(int(args.pair_index))
        viewer.load_selected()
    viewer.show()
    return int(app.exec())


def positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR_DEFAULT))
    parser.add_argument("--data-glob", default="*UFIS*count_multiple*.hdf")
    parser.add_argument("--flatfield-token", default="flatfield")
    parser.add_argument("--fixed-flatfield", default=None)
    parser.add_argument("--flatfield-preference", default="nearest", choices=["nearest", "before", "after"])
    parser.add_argument("--data-exclude-token", nargs="*", default=["flatfield"])
    parser.add_argument("--only-data", nargs="*", default=None)
    parser.add_argument("--bad-files", default=str(BAD_FILES_DEFAULT))

    parser.add_argument("--roi-json", default=str(ROI_JSON_DEFAULT))
    parser.add_argument("--no-saved-roi", dest="use_saved_roi", action="store_false")
    parser.set_defaults(use_saved_roi=True)
    parser.add_argument("--row-range", nargs=2, type=int, default=None, metavar=("START", "STOP"))
    parser.add_argument("--use-tilted-roi", action="store_true", default=False)
    parser.add_argument("--tilted-roi-frame-average", type=positive_int, default=5)
    parser.add_argument("--tilted-roi-threshold-fraction", type=float, default=0.55)
    parser.add_argument("--tilted-roi-shrink-fraction", type=float, default=0.90)
    parser.add_argument("--tilted-roi-smooth-sigma-rows", type=float, default=2.0)
    parser.add_argument("--tilted-roi-smooth-sigma-cols", type=float, default=6.0)

    parser.add_argument("--aver-n", type=positive_int, default=5)
    parser.add_argument("--chunk-size", type=positive_int, default=250)
    parser.add_argument("--median-size", type=int, default=3)
    parser.add_argument("--norm-window", type=positive_int, default=80)
    parser.add_argument("--norm-sample-frames", type=positive_int, default=100)
    parser.add_argument("--factor", type=float, default=200.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--snapshot-frame", type=int, default=0)
    parser.add_argument("--flat-snapshot-frame", type=int, default=0)
    parser.add_argument("--max-frame-lines", type=positive_int, default=200)
    parser.add_argument("--max-avg-lines", type=positive_int, default=500)
    parser.add_argument("--pair-index", type=int, default=None)
    parser.add_argument("--compute-only", action="store_true")
    return parser


def print_compute_summary(result: PreviewResult, pairs: Sequence[BatchPair], skipped: Sequence[Mapping[str, str]]) -> None:
    print(build_summary_text(pairs, skipped, result=result))
    print("")
    print(f"per_frame_specs shape: {result.per_frame_specs.shape}")
    print(f"avg_specs shape:       {result.avg_specs.shape}")
    print(f"snapshot_data shape:   {result.snapshot_data.shape}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.data_dir = str(Path(args.data_dir).expanduser().resolve())
    args.bad_files = str(Path(args.bad_files).expanduser().resolve())

    pairs, skipped = discover_batch_pairs(args)
    if not pairs:
        raise SystemExit("No usable data/flatfield pairs were discovered.")
    if args.pair_index is not None and not (0 <= int(args.pair_index) < len(pairs)):
        raise SystemExit(f"--pair-index must be between 0 and {len(pairs) - 1}.")

    print(f"Discovered {len(pairs)} usable pairs and {len(skipped)} skipped files.")
    if args.compute_only:
        pair_index = int(args.pair_index or 0)
        result = compute_preview(pairs[pair_index], args)
        print_compute_summary(result, pairs, skipped)
        return 0
    return run_gui(args, pairs, skipped)


if __name__ == "__main__":
    raise SystemExit(main())
