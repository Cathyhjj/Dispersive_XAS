"""Batch preview processing for large time-resolved DXAS datasets.

Provides fast, memory-efficient frame-by-frame spectrum generation and
multi-format preview plots for quick quality assessment before full
calibrated analysis.

Key distinction
---------------
:func:`norm_spec_preview` normalises by **pixel index** (fast, no energy
calibration needed).  For calibrated, energy-axis normalisation use
:func:`~Dispersive_XAS.spectrum.norm_spec` instead.

Output format
-------------
``output_format='html'`` (default)
    Three interactive Plotly HTML files per chunk — open in any browser,
    supports free zoom, pan, and hover tooltips.

Inline display (Jupyter)
------------------------
Pass ``display_inline=True`` to render the Plotly figures directly inside
a Jupyter notebook cell (in addition to saving the HTML files).  This
enables instant interactive zoom/pan without leaving the notebook.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import scipy.ndimage as ndi

from ..progress import BatchProgressReporter, emit_progress
from ..core.batch import norm_spec_preview
from ..core.data_io import load_nexus_entry
from ..core.roi import prepare_roi_weights

__all__ = [
    "norm_spec_preview",
    "plot_spectra_in_chunks",
    "preview_spectra_html",
]
_DEFAULT_FACTOR = 200.0


def _clip_pixel_range(
    pixel_range: Tuple[int, int] | None,
    width: int,
    name: str,
) -> Tuple[int, int] | None:
    if pixel_range is None:
        return None
    if len(pixel_range) != 2:
        raise ValueError(f"{name} must be a 2-value pixel range, got {pixel_range!r}")
    x1, x2 = sorted((int(pixel_range[0]), int(pixel_range[1])))
    x1 = max(0, min(x1, int(width) - 1))
    x2 = max(x1 + 1, min(x2, int(width)))
    if x2 <= x1:
        raise ValueError(f"{name} is empty after clipping to detector width {width}: {pixel_range!r}")
    return x1, x2


def _compute_chunk_specs(
    data: np.ndarray,
    flat_avg: np.ndarray,
    row_start: int,
    row_stop: int,
    row_weights: np.ndarray,
    col_weight_sum: np.ndarray,
    chunk_start: int,
    chunk_end: int,
    aver_n: int,
    norm_x1: int,
    norm_x2: int,
    factor: float,
    median_size: int = 0,
    pre_edge_norm_range: Tuple[int, int] | None = None,
    post_edge_norm_range: Tuple[int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """Compute per-frame and grouped-average spectra for one chunk.

    Shared by :func:`plot_spectra_in_chunks` and :func:`preview_spectra_html`.

    Returns
    -------
    per_frame_specs : ndarray, shape (n_frames, W)
    specs_avg : ndarray, shape (num_groups, W)
    n_frames : int
    num_groups : int
    W : int  — detector width in pixels
    """
    def _normalize_specs_batch(specs: np.ndarray) -> np.ndarray:
        if specs.size == 0:
            return specs
        width = int(specs.shape[1])
        pre_range = _clip_pixel_range(pre_edge_norm_range, width, "pre_edge_norm_range")
        post_range = _clip_pixel_range(post_edge_norm_range, width, "post_edge_norm_range")
        if (pre_range is None) != (post_range is None):
            raise ValueError(
                "pre_edge_norm_range and post_edge_norm_range must be set together."
            )
        if pre_range is not None and post_range is not None:
            pre_ref = specs[:, pre_range[0]:pre_range[1]]
            post_ref = specs[:, post_range[0]:post_range[1]]
            pre_level = np.mean(pre_ref, axis=1, keepdims=True)
            post_level = np.mean(post_ref, axis=1, keepdims=True)
            out = (specs - pre_level) / (post_level - pre_level + 1e-12)
            out[~np.isfinite(out)] = 0.0
            return out
        x1 = max(0, min(int(norm_x1), specs.shape[1] - 1))
        x2 = max(x1 + 1, min(int(norm_x2), specs.shape[1]))
        ref = specs[:, x1:x2]
        smin = np.min(ref, axis=1, keepdims=True)
        smax = np.max(ref, axis=1, keepdims=True)
        out = ((specs - smin) / (smax - smin + 1e-12)) * float(factor)
        out[~np.isfinite(out)] = 0.0
        return out

    n_frames = int(chunk_end - chunk_start)
    if n_frames <= 0:
        return np.empty((0, 0)), np.empty((0, 0)), 0, 0, 0

    # Vectorized chunk processing for speed.
    chunk = np.asarray(data[chunk_start:chunk_end, row_start:row_stop, :], dtype=np.float32)
    if int(median_size) > 1:
        k = int(median_size)
        if k % 2 == 0:
            k += 1
        # Spatial median filter only (rows, columns); do not blur time axis.
        chunk = ndi.median_filter(chunk, size=(1, k, k), mode="nearest")

    flat2d = np.asarray(flat_avg, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        mux = np.log(flat2d[None, :, :] / chunk)
    mux[~np.isfinite(mux)] = 0.0
    per_frame_specs = _normalize_specs_batch(
        (mux * row_weights[None, :, :]).sum(axis=1) / col_weight_sum
    )

    W = int(per_frame_specs.shape[1])
    num_groups = int(n_frames // aver_n) if aver_n > 0 else 0
    if num_groups > 0:
        usable = chunk[: num_groups * aver_n]
        grouped = usable.reshape(num_groups, int(aver_n), row_stop - row_start, W).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            gmux = np.log(flat2d[None, :, :] / grouped)
        gmux[~np.isfinite(gmux)] = 0.0
        specs_avg = _normalize_specs_batch(
            (gmux * row_weights[None, :, :]).sum(axis=1) / col_weight_sum
        )
    else:
        specs_avg = np.empty((0, W), dtype=np.float32)

    return per_frame_specs, specs_avg, n_frames, num_groups, W


def plot_spectra_in_chunks(
    data_path: str,
    flat_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    aver_n: int = 1,
    flat_range: Tuple[int, int] = (180, 230),
    norm_x1: int = 180,
    norm_x2: int = 200,
    x1: int = 100,
    x2: int = 160,
    chunk_size: int = 1000,
    cmap_name: str = "magma",
    factor: float = _DEFAULT_FACTOR,
    output_format: str = "html",
    max_line_traces: int = 200,
    display_inline: bool = False,
    median_size: int = 0,
    roi: dict | None = None,
    output_dir: str | None = None,
    progress: BatchProgressReporter | None = None,
    progress_stage: str = "preview_chunk_export",
    pre_edge_norm_range: Tuple[int, int] | None = None,
    post_edge_norm_range: Tuple[int, int] | None = None,
) -> None:
    """Generate and save batch preview plots for a large DXAS scan.

    Processes the scan in chunks of *chunk_size* frames to keep memory
    use bounded.  Three plot types are saved per chunk:

    1. **Heatmap / Imshow** – 2-D colour map (frames × pixels).  In HTML
       mode this is fully zoomable and shows pixel/frame/intensity on hover.
    2. **Lines (averaged)** – spectra averaged in groups of *aver_n*, colour-
       coded from first (dark) to last (bright) frame in the chunk.
    3. **Lines (no averaging)** – one spectrum per frame (subsampled if needed).
       In HTML mode, frames are subsampled to at most *max_line_traces* traces
       when the chunk exceeds that limit (the heatmap covers the full detail).

    Output directory:
    ``<data_dir>/preliminary_results_<scan_basename>/``

    Parameters
    ----------
    data_path : str
        Path to the sample HDF5 file (NeXus ``/entry/data/data`` layout).
    flat_path : str
        Path to the flat-field HDF5 file.
    start_frame : int
        First frame to process (default: 0).
    end_frame : int or None
        Last frame (exclusive).  Defaults to the total number of frames.
    aver_n : int
        Frames averaged per group in plot 2 (default: 1, i.e. no averaging).
    flat_range : (int, int)
        Pixel row range ``(fr0, fr1)`` used to spatially average the
        flat-field (selects the beam footprint rows).
    norm_x1, norm_x2 : int
        Legacy one-window min/max normalization bounds. Used only when
        ``pre_edge_norm_range`` and ``post_edge_norm_range`` are not supplied.
    pre_edge_norm_range, post_edge_norm_range : (int, int) or None
        Manual edge-step normalization windows. When both are supplied, each
        spectrum is normalized as ``(spec - mean(pre)) / (mean(post) - mean(pre))``
        so the pre-edge window averages to 0 and the post-edge window averages
        to 1.
    x1, x2 : int
        Pixel column range for plot x-axis limits.
    chunk_size : int
        Frames per processing chunk (default: 1000).
    cmap_name : str
        Colormap name (default: ``'magma'``).
    factor : float
        Normalisation scale passed to :func:`norm_spec_preview`.
    output_format : {'html'}
        Interactive Plotly HTML output. Any non-``'html'`` value falls back
        to ``'html'`` in the web-first workflow.
    max_line_traces : int
        Maximum number of line traces in the no-averaging HTML plot.
        Frames are subsampled uniformly when the chunk exceeds this limit.
        Default: 200.
    display_inline : bool
        If ``True``, call ``fig.show()`` after saving each HTML file so that
        the interactive Plotly figures appear directly inside a Jupyter
        notebook cell.  Users can then zoom, pan, and hover without opening
        external files.  Default: ``False``.
    median_size : int
        Optional spatial median filter kernel size for hot-pixel suppression.
        ``0`` or ``1`` disables filtering; ``3`` is a common fast choice.
    roi : dict or None
        Optional ROI spec. When provided, it overrides ``flat_range`` and can
        represent a tilted beam band.
    output_dir : str or None
        Optional directory for the generated preview files. Defaults to
        ``<data_dir>/preliminary_results_<scan_basename>/``.
    """
    data = load_nexus_entry(data_path)["data"]   # (N, H, W)
    flat = load_nexus_entry(flat_path)["data"]    # (N, H, W)

    if end_frame is None:
        end_frame = data.shape[0]

    if output_dir is None:
        base_name = os.path.splitext(data_path)[0]
        save_dir = os.path.join(
            os.path.dirname(base_name),
            "preliminary_results_" + os.path.basename(base_name),
        )
    else:
        save_dir = os.fspath(output_dir)
    os.makedirs(save_dir, exist_ok=True)

    roi_spec, (fr0, fr1), row_weights, col_weight_sum = prepare_roi_weights(
        (int(data.shape[1]), int(data.shape[2])),
        row_range=flat_range,
        roi=roi,
        dtype=np.float32,
    )
    flat_avg = np.average(flat[:, fr0:fr1, :], axis=0)  # (fr1-fr0, W)

    if output_format != "html":
        print(
            f"output_format={output_format!r} is deprecated; using 'html' interactive output."
        )
    do_html = True
    do_png = False
    total_frames = max(0, end_frame - start_frame)
    total_chunks = max(1, (total_frames + chunk_size - 1) // chunk_size) if total_frames > 0 else 0
    use_edge_norm = pre_edge_norm_range is not None or post_edge_norm_range is not None
    if (pre_edge_norm_range is None) != (post_edge_norm_range is None):
        raise ValueError("pre_edge_norm_range and post_edge_norm_range must be set together.")
    norm_label = (
        f"edge-step pre={tuple(pre_edge_norm_range)} post={tuple(post_edge_norm_range)}"
        if use_edge_norm
        else f"min-max x={norm_x1}:{norm_x2}, factor={factor:g}"
    )
    heatmap_zmax = 1.0 if use_edge_norm else 1.5 * float(factor)

    emit_progress(
        progress,
        progress_stage,
        status="running",
        current=0,
        total=total_chunks,
        unit="chunks",
        message=f"Starting chunk preview export for {total_frames} frames",
        extra={
            "data_path": os.path.abspath(data_path),
            "save_dir": save_dir,
            "roi": roi_spec,
        },
    )

    for chunk_index, chunk_start in enumerate(range(start_frame, end_frame, chunk_size), start=1):
        chunk_end = min(chunk_start + chunk_size, end_frame)
        per_frame_specs, specs_avg, n_frames, num_groups, W = _compute_chunk_specs(
            data, flat_avg, fr0, fr1,
            row_weights,
            col_weight_sum,
            chunk_start, chunk_end,
            aver_n, norm_x1, norm_x2, factor,
            median_size=median_size,
            pre_edge_norm_range=pre_edge_norm_range,
            post_edge_norm_range=post_edge_norm_range,
        )
        if n_frames <= 0:
            continue

        col_end = min(x2 + 50, W)
        saved = []

        if do_html:
            saved += _save_html_chunk(
                per_frame_specs, specs_avg,
                chunk_start, chunk_end, n_frames, num_groups,
                x1, col_end, aver_n, cmap_name, factor,
                save_dir, data_path, max_line_traces,
                display_inline=display_inline,
                heatmap_zmin=0.0,
                heatmap_zmax=heatmap_zmax,
                norm_label=norm_label,
            )

        emit_progress(
            progress,
            progress_stage,
            status="running",
            current=chunk_index,
            total=total_chunks,
            unit="chunks",
            message=f"Saved preview chunk {chunk_start}:{chunk_end}",
            extra={
                "frame_start": int(chunk_start),
                "frame_end": int(chunk_end),
                "saved_files": saved,
            },
        )
        print("Saved:\n" + "\n".join(f"  {p}" for p in saved))

    emit_progress(
        progress,
        progress_stage,
        status="completed",
        current=total_chunks,
        total=total_chunks,
        unit="chunks",
        message=f"Chunk preview export finished ({total_chunks} filesets)",
        extra={
            "data_path": os.path.abspath(data_path),
            "save_dir": save_dir,
        },
    )


def preview_spectra_html(
    data_path: str,
    flat_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    aver_n: int = 1,
    flat_range: Tuple[int, int] = (180, 230),
    norm_x1: int = 180,
    norm_x2: int = 200,
    chunk_size: int = 1000,
    cmap_name: str = "magma",
    factor: float = _DEFAULT_FACTOR,
    max_line_traces: int = 200,
    display_inline: bool = True,
    median_size: int = 0,
    roi: dict | None = None,
    output_dir: str | None = None,
    progress: BatchProgressReporter | None = None,
    progress_stage: str = "preview_full_export",
    pre_edge_norm_range: Tuple[int, int] | None = None,
    post_edge_norm_range: Tuple[int, int] | None = None,
) -> None:
    """Generate **three** interactive HTML plots covering **all frames**.

    Unlike :func:`plot_spectra_in_chunks`, which saves a separate set of files
    for every chunk, this function collects computed spectra across the whole
    scan and writes exactly **one** heatmap, one averaged-lines, and one
    no-averaging-lines HTML file.  No ``x1``/``x2`` pixel cropping is applied
    — every detector column is shown — because the Plotly viewer lets users
    freely box-zoom, drag the range slider, and pan to any region of interest.

    Raw image data is still read in chunks of *chunk_size* frames so that the
    full detector array never has to sit in memory all at once; only the
    lightweight computed spectra (``float64``, shape ``(N_frames, W_pixels)``)
    are accumulated before plotting.

    Output directory:
    ``<data_dir>/preliminary_results_<scan_basename>/``

    Parameters
    ----------
    data_path : str
        Path to the sample HDF5 file (NeXus ``/entry/data/data`` layout).
    flat_path : str
        Path to the flat-field HDF5 file.
    start_frame : int
        First frame to process (default: 0).
    end_frame : int or None
        Last frame (exclusive).  Defaults to the total number of frames.
    aver_n : int
        Frames averaged per group in the lines-averaged plot (default: 1).
    flat_range : (int, int)
        Pixel row range ``(fr0, fr1)`` for spatial flat-field averaging.
    norm_x1, norm_x2 : int
        Legacy one-window min/max normalization bounds. Used only when
        ``pre_edge_norm_range`` and ``post_edge_norm_range`` are not supplied.
    chunk_size : int
        Frames read from disk at once (default: 1000).  Does **not** split
        the output — all frames always appear in a single set of HTML files.
    cmap_name : str
        Plotly/Matplotlib colormap name (default: ``'magma'``).
    factor : float
        Normalisation scale passed to the legacy min/max normalization.
    pre_edge_norm_range, post_edge_norm_range : (int, int) or None
        Manual edge-step normalization windows. When both are supplied, each
        spectrum is normalized as ``(spec - mean(pre)) / (mean(post) - mean(pre))``
        so the pre-edge window averages to 0 and the post-edge window averages
        to 1.
    max_line_traces : int
        Maximum line traces in the no-averaging plot; frames are subsampled
        uniformly beyond this limit (default: 200).
    display_inline : bool
        If ``True`` (default), render each figure directly inside the Jupyter
        notebook cell via ``fig.show()``.  Set to ``False`` to only save the
        HTML files without displaying them.
    median_size : int
        Optional spatial median filter kernel size for hot-pixel suppression.
        ``0`` or ``1`` disables filtering; ``3`` is a common fast choice.
    roi : dict or None
        Optional ROI spec. When provided, it overrides ``flat_range`` and can
        represent a tilted beam band.
    output_dir : str or None
        Optional directory for the generated preview files. Defaults to
        ``<data_dir>/preliminary_results_<scan_basename>/``.
    """
    data = load_nexus_entry(data_path)["data"]   # (N, H, W)
    flat = load_nexus_entry(flat_path)["data"]    # (N, H, W)

    if end_frame is None:
        end_frame = data.shape[0]

    if output_dir is None:
        base_name = os.path.splitext(data_path)[0]
        save_dir = os.path.join(
            os.path.dirname(base_name),
            "preliminary_results_" + os.path.basename(base_name),
        )
    else:
        save_dir = os.fspath(output_dir)
    os.makedirs(save_dir, exist_ok=True)

    roi_spec, (fr0, fr1), row_weights, col_weight_sum = prepare_roi_weights(
        (int(data.shape[1]), int(data.shape[2])),
        row_range=flat_range,
        roi=roi,
        dtype=np.float32,
    )
    flat_avg = np.average(flat[:, fr0:fr1, :], axis=0)  # (fr1-fr0, W)

    # Accumulate computed spectra across all chunks (raw images are NOT kept).
    all_per_frame: list = []
    all_avg: list = []
    total_frames = max(0, end_frame - start_frame)
    total_chunks = max(1, (total_frames + chunk_size - 1) // chunk_size) if total_frames > 0 else 0
    use_edge_norm = pre_edge_norm_range is not None or post_edge_norm_range is not None
    if (pre_edge_norm_range is None) != (post_edge_norm_range is None):
        raise ValueError("pre_edge_norm_range and post_edge_norm_range must be set together.")
    norm_label = (
        f"edge-step pre={tuple(pre_edge_norm_range)} post={tuple(post_edge_norm_range)}"
        if use_edge_norm
        else f"min-max x={norm_x1}:{norm_x2}, factor={factor:g}"
    )
    heatmap_zmax = 1.0 if use_edge_norm else 1.5 * float(factor)

    emit_progress(
        progress,
        progress_stage,
        status="running",
        current=0,
        total=total_chunks,
        unit="chunks",
        message=f"Starting full preview export for {total_frames} frames",
        extra={
            "data_path": os.path.abspath(data_path),
            "save_dir": save_dir,
            "roi": roi_spec,
        },
    )

    for chunk_index, chunk_start in enumerate(range(start_frame, end_frame, chunk_size), start=1):
        chunk_end = min(chunk_start + chunk_size, end_frame)
        per_frame_specs, specs_avg, n_frames, _num_groups, _W = _compute_chunk_specs(
            data, flat_avg, fr0, fr1,
            row_weights,
            col_weight_sum,
            chunk_start, chunk_end,
            aver_n, norm_x1, norm_x2, factor,
            median_size=median_size,
            pre_edge_norm_range=pre_edge_norm_range,
            post_edge_norm_range=post_edge_norm_range,
        )
        if n_frames <= 0:
            continue
        all_per_frame.append(per_frame_specs)
        if len(specs_avg):
            all_avg.append(specs_avg)
        emit_progress(
            progress,
            progress_stage,
            status="running",
            current=chunk_index,
            total=total_chunks,
            unit="chunks",
            message=f"Processed frames {chunk_start}-{chunk_end}",
            extra={
                "frame_start": int(chunk_start),
                "frame_end": int(chunk_end),
            },
        )

    if not all_per_frame:
        print("No frames processed.")
        emit_progress(
            progress,
            progress_stage,
            status="completed",
            current=0,
            total=total_chunks,
            unit="chunks",
            message="No frames processed",
            extra={
                "data_path": os.path.abspath(data_path),
                "save_dir": save_dir,
            },
        )
        return

    all_per_frame = np.vstack(all_per_frame)   # (total_frames, W)
    all_avg = np.vstack(all_avg) if all_avg else np.empty((0, all_per_frame.shape[1]))

    total_frames = all_per_frame.shape[0]
    total_groups = all_avg.shape[0]
    W = all_per_frame.shape[1]

    print(f"Saving HTML for all {total_frames} frames …")
    # x1=0, col_end=W → full pixel range in all three plots.
    saved = _save_html_chunk(
        all_per_frame, all_avg,
        start_frame, start_frame + total_frames, total_frames, total_groups,
        0, W, aver_n, cmap_name, factor,
        save_dir, data_path, max_line_traces,
        display_inline=display_inline,
        heatmap_zmin=0.0,
        heatmap_zmax=heatmap_zmax,
        norm_label=norm_label,
    )
    emit_progress(
        progress,
        progress_stage,
        status="completed",
        current=total_chunks,
        total=total_chunks,
        unit="chunks",
        message=f"Saved full preview HTML for {total_frames} frames",
        extra={
            "data_path": os.path.abspath(data_path),
            "save_dir": save_dir,
            "saved_files": saved,
        },
    )
    print("Saved:\n" + "\n".join(f"  {p}" for p in saved))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _save_html_chunk(
    per_frame_specs: np.ndarray,
    specs_avg: np.ndarray,
    chunk_start: int,
    chunk_end: int,
    n_frames: int,
    num_groups: int,
    x1: int,
    col_end: int,
    aver_n: int,
    cmap_name: str,
    factor: float,
    save_dir: str,
    data_path: str,
    max_line_traces: int,
    display_inline: bool = False,
    heatmap_zmin: float = 0.0,
    heatmap_zmax: float | None = None,
    norm_label: str | None = None,
) -> list:
    """Save ONE combined interactive HTML with three stacked subplots.

    The single output file ``preview_*.html`` contains:

    * **Row 1 — Heatmap**: full-resolution 2-D colour map (frames × pixels).
    * **Row 2 — Averaged lines**: spectra averaged in groups of *aver_n*.
    * **Row 3 — Individual frames**: one trace per frame, subsampled if needed.

    All three subplots share the x-axis (pixel column), so zooming or
    panning the x-axis in any subplot moves all three simultaneously.
    Each subplot has its own independent y-axis that can be zoomed by
    dragging on the y-axis tick labels.

    Each subplot is ~500 px tall.  Uses ``write_html(include_plotlyjs='cdn')``
    so the correct version-pinned Plotly JS URL is always embedded.

    Returns
    -------
    list of str
        Single-element list: path of the saved HTML file.
    """
    import plotly.colors
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    scan_name = os.path.basename(data_path)
    pixel_axis = list(range(x1, col_end))
    frame_axis = list(range(chunk_start, chunk_end))
    heatmap_zmax = 1.5 * float(factor) if heatmap_zmax is None else float(heatmap_zmax)
    norm_label = norm_label or f"min-max x-range, factor={factor:g}"

    # Subsampling for line plots so we keep responsiveness on very large scans.
    step = max(1, n_frames // max_line_traces)
    disp_idx = list(range(0, n_frames, step))
    n_disp = len(disp_idx)
    subsample_note = f"1 in {step} shown ({n_disp} traces)" if step > 1 else f"all {n_frames:,}"
    avg_step = max(1, num_groups // max_line_traces) if num_groups > 0 else 1
    avg_idx = list(range(0, num_groups, avg_step)) if num_groups > 0 else []
    n_avg_disp = len(avg_idx)
    avg_note = (
        f"every {aver_n} frame(s), 1 in {avg_step} shown ({n_avg_disp} traces)"
        if num_groups > 0 and avg_step > 1
        else f"every {aver_n} frame(s), {num_groups:,} traces"
    )

    # ---- Build combined 3-column figure (side by side) --------------------
    subplot_titles = [
        f"<b>Heatmap</b><br>{n_frames:,} frames",
        f"<b>Averaged lines</b><br>{avg_note}",
        f"<b>Individual frames</b><br>no averaging  [{subsample_note}]",
    ]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=subplot_titles,
        shared_xaxes=False,  # each panel has its own x-axis (independent zoom)
        shared_yaxes=False,  # each panel has its own y-axis (independent zoom)
        horizontal_spacing=0.12,
    )

    # ---- Col 1: Heatmap ----------------------------------------------------
    fig.add_trace(
        go.Heatmap(
            z=per_frame_specs[:, x1:col_end],
            x=pixel_axis,
            y=frame_axis,
            colorscale=cmap_name,
            zmin=float(heatmap_zmin),
            zmax=heatmap_zmax,
            colorbar=dict(
                title=dict(text="Intensity", side="right"),
                len=0.5, x=0.248, y=0.5, thickness=8,
            ),
            hovertemplate=(
                "Pixel: %{x}<br>Frame: %{y}<br>"
                "Intensity: %{z:.2f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    def _rough_edge_jump(y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        if y.size < 8:
            jump = float(np.nanpercentile(y, 90) - np.nanpercentile(y, 10))
            return jump if np.isfinite(jump) and jump > 1e-9 else 1.0
        n = max(2, y.size // 10)
        pre = float(np.nanmean(y[:n]))
        post = float(np.nanmean(y[-n:]))
        jump = abs(post - pre)
        if not np.isfinite(jump) or jump <= 1e-9:
            jump = float(np.nanpercentile(y, 90) - np.nanpercentile(y, 10))
        return jump if np.isfinite(jump) and jump > 1e-9 else 1.0

    def _default_offset(lines: list[np.ndarray]) -> float:
        if not lines:
            return 1.0
        # Fast rough default: about one-third of the edge jump.
        jump = _rough_edge_jump(lines[0])
        return max(jump / 3.0, 1e-6)

    def _stacked_range(lines: list[np.ndarray], offset: float, scale: float) -> list[float]:
        if not lines:
            return [0.0, 1.0]
        y0 = np.asarray(lines[0], dtype=float)
        y_last = np.asarray(lines[-1], dtype=float)
        ymin = float(np.nanmin(y0))
        n = len(lines)
        center = float(np.nanmean(y_last) + (offset * scale * (n - 1)))
        pad = max(_rough_edge_jump(y_last), offset * scale) * 1.2
        ymax = center + pad
        if not np.isfinite(ymin):
            ymin = 0.0
        if not np.isfinite(ymax) or ymax <= ymin:
            ymax = ymin + max(pad, 1.0)
        return [ymin, ymax]

    # ---- Col 2: Averaged lines (stacked with vertical offset) ---------------
    avg_raw: list[np.ndarray] = []
    avg_trace_indices: list[int] = []
    avg_frame_labels: list[int] = []
    avg_offset0 = 1.0
    if n_avg_disp > 0:
        colors = plotly.colors.sample_colorscale(cmap_name, np.linspace(0, 1, n_avg_disp))
        px_arr = np.arange(x1, col_end)
        avg_raw = [np.asarray(specs_avg[gi, x1:col_end], dtype=float) for gi in avg_idx]
        avg_offset0 = _default_offset(avg_raw)
        for rank, gi in enumerate(avg_idx):
            frame_label = chunk_start + gi * aver_n
            avg_frame_labels.append(int(frame_label))
            fig.add_trace(
                go.Scattergl(
                    x=px_arr,
                    y=avg_raw[rank] + avg_offset0 * rank,
                    mode="lines",
                    name=f"Avg@{frame_label}",
                    line=dict(width=1, color=colors[rank]),
                    opacity=0.7,
                    hovertemplate=(
                        "Avg start frame " + str(frame_label) + "<br>"
                        "Pixel: %{x}<br>"
                        "Intensity: %{y:.3f}<extra></extra>"
                    ),
                    customdata=avg_raw[rank],
                    showlegend=False,
                ),
                row=1, col=2,
            )
            avg_trace_indices.append(len(fig.data) - 1)

    # ---- Col 3: Individual frames (stacked with vertical offset) ------------
    frame_raw: list[np.ndarray] = []
    frame_trace_indices: list[int] = []
    frame_trace_labels: list[int] = []
    frame_offset0 = 1.0
    if n_disp > 0:
        colors_noavg = plotly.colors.sample_colorscale(cmap_name, np.linspace(0, 1, n_disp))
        px_arr = np.arange(x1, col_end)
        frame_raw = [np.asarray(per_frame_specs[i, x1:col_end], dtype=float) for i in disp_idx]
        frame_offset0 = _default_offset(frame_raw)
        for rank, i in enumerate(disp_idx):
            frame_label = chunk_start + i
            frame_trace_labels.append(int(frame_label))
            fig.add_trace(
                go.Scattergl(
                    x=px_arr,
                    y=frame_raw[rank] + frame_offset0 * rank,
                    mode="lines",
                    name=f"Frame {frame_label}",
                    line=dict(width=0.8, color=colors_noavg[rank]),
                    opacity=0.45,
                    hovertemplate=(
                        "Frame " + str(frame_label) + "<br>"
                        "Pixel: %{x}<br>"
                        "Intensity: %{y:.3f}<extra></extra>"
                    ),
                    customdata=frame_raw[rank],
                    showlegend=False,
                ),
                row=1, col=3,
            )
            frame_trace_indices.append(len(fig.data) - 1)

    # Interactive offset scaling for both stacked line panels.
    # 1x is default; other levels help with crowded/sparse regions.
    scale_vals = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    if avg_trace_indices or frame_trace_indices:
        def _stack_with_scale(lines: list[np.ndarray], base: float, scale: float) -> list[list[float]]:
            return [(y + (base * scale * k)).tolist() for k, y in enumerate(lines)]

        steps = []
        all_indices = avg_trace_indices + frame_trace_indices
        for sc in scale_vals:
            ys = _stack_with_scale(avg_raw, avg_offset0, sc) + _stack_with_scale(frame_raw, frame_offset0, sc)
            r2 = _stacked_range(avg_raw, avg_offset0, sc)
            r3 = _stacked_range(frame_raw, frame_offset0, sc)
            steps.append(
                dict(
                    label=f"{sc:g}x",
                    method="update",
                    args=[
                        {"y": ys},
                        {"yaxis2.range": r2, "yaxis3.range": r3},
                        all_indices,
                    ],
                )
            )
        fig.update_layout(
            sliders=[
                dict(
                    active=3,  # default 1.0x
                    x=0.18,
                    y=-0.12,
                    len=0.62,
                    xanchor="left",
                    currentvalue=dict(prefix="Line offset: "),
                    pad=dict(t=0, b=0),
                    steps=steps,
                )
            ]
        )
        fig.update_yaxes(range=_stacked_range(avg_raw, avg_offset0, 1.0), row=1, col=2)
        fig.update_yaxes(range=_stacked_range(frame_raw, frame_offset0, 1.0), row=1, col=3)

    # ---- Global layout -----------------------------------------------------
    fig.update_layout(
        title=dict(
            text=(
                f"DXAS Batch Preview \u2014 {scan_name}"
                f"<br><sup>frames {chunk_start:,}\u2013{chunk_end:,}"
                f" &nbsp;|&nbsp; N={n_frames:,}"
                f" &nbsp;|&nbsp; averaged every {aver_n} frame(s)"
                f" &nbsp;|&nbsp; colour: {cmap_name}"
                f" &nbsp;|&nbsp; norm: {norm_label}</sup>"
                "<br><sup style='color:#666'>"
                "Zoom: click+drag on plot (both axes) &nbsp;\u2022&nbsp;"
                "y-axis only: drag on y-axis labels &nbsp;\u2022&nbsp;"
                "x-axis only: drag on x-axis labels &nbsp;\u2022&nbsp;"
                "pan: Shift+drag &nbsp;\u2022&nbsp; reset: double-click"
                "</sup>"
            ),
            font=dict(size=13),
            y=0.99, yanchor="top",
        ),
        height=1000,
        autosize=True,
        font=dict(size=11),
        showlegend=False,
        dragmode="zoom",
        hoverlabel=dict(bgcolor="white", font_size=11),
        margin=dict(t=150, b=130, l=70, r=70),
    )

    # Axis labels — x and y on every panel
    for col in (1, 2, 3):
        fig.update_xaxes(title_text="Pixel column", row=1, col=col)
    fig.update_yaxes(title_text="Frame index", row=1, col=1)
    fig.update_yaxes(title_text="Offset normalized intensity (a.u.)", row=1, col=2)
    fig.update_yaxes(title_text="Offset normalized intensity (a.u.)", row=1, col=3)

    # ---- Save --------------------------------------------------------------
    path_preview = os.path.join(
        save_dir,
        f"preview_{chunk_start:05d}-{chunk_end:05d}_N{n_frames}.html",
    )
    plot_div_id = "dxas_preview_plot"
    fig_json = fig.to_json()
    x_lo_default = int(x1)
    x_hi_default = int(max(x1, col_end - 1))
    f_lo_default = int(chunk_start)
    f_hi_default = int(max(chunk_start, chunk_end - 1))
    c_lo_default = float(heatmap_zmin)
    c_hi_default = float(heatmap_zmax)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>DXAS Preview</title>
  <script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; }}
    #wrap {{ padding: 10px 12px 12px 12px; }}
    #controls {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 10px;
    }}
    .ctrl {{
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 10px;
      background: #fafbfc;
    }}
    .ctrl h4 {{ margin: 0 0 8px 0; font-size: 14px; }}
    .row {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
    label {{ font-size: 12px; color: #444; }}
    input {{ width: 92px; padding: 4px 6px; font-size: 12px; }}
    button {{ padding: 4px 8px; font-size: 12px; cursor: pointer; }}
    .hint {{ font-size: 11px; color: #666; margin-top: 6px; }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="{plot_div_id}" style="width:100%;"></div>
    <div id="controls">
      <div class="ctrl">
        <h4>Heatmap color scale</h4>
        <div class="row">
          <label for="c_min">Vmin</label><input id="c_min" type="number" step="any">
          <label for="c_max">Vmax</label><input id="c_max" type="number" step="any">
          <button id="apply_c">Apply</button>
          <button id="reset_c">Reset</button>
        </div>
        <div class="hint">Applies only to the first heatmap panel.</div>
      </div>
      <div class="ctrl">
        <h4>X ranges (all 3 plots)</h4>
        <div class="row">
          <label for="x_min">X min</label><input id="x_min" type="number" step="1">
          <label for="x_max">X max</label><input id="x_max" type="number" step="1">
          <button id="apply_x">Apply</button>
          <button id="reset_x">Reset</button>
        </div>
      </div>
      <div class="ctrl">
        <h4>Frame ranges</h4>
        <div class="row">
          <label for="f_min">Frame min</label><input id="f_min" type="number" step="1">
          <label for="f_max">Frame max</label><input id="f_max" type="number" step="1">
          <button id="apply_f">Apply</button>
          <button id="reset_f">Reset</button>
        </div>
        <div class="hint">Heatmap Y-range is zoomed; line traces outside range are hidden.</div>
      </div>
    </div>
  </div>
  <script>
    const fig = {fig_json};
    const avgTraceIdx = {json.dumps(avg_trace_indices)};
    const avgFrameLabels = {json.dumps(avg_frame_labels)};
    const frameTraceIdx = {json.dumps(frame_trace_indices)};
    const frameLabels = {json.dumps(frame_trace_labels)};
    const xDefault = [{x_lo_default}, {x_hi_default}];
    const fDefault = [{f_lo_default}, {f_hi_default}];
    const cDefault = [{c_lo_default:.12g}, {c_hi_default:.12g}];

    const gd = document.getElementById("{plot_div_id}");
    const cMinEl = document.getElementById("c_min");
    const cMaxEl = document.getElementById("c_max");
    const xMinEl = document.getElementById("x_min");
    const xMaxEl = document.getElementById("x_max");
    const fMinEl = document.getElementById("f_min");
    const fMaxEl = document.getElementById("f_max");

    function setDefaults() {{
      cMinEl.value = cDefault[0];
      cMaxEl.value = cDefault[1];
      xMinEl.value = xDefault[0];
      xMaxEl.value = xDefault[1];
      fMinEl.value = fDefault[0];
      fMaxEl.value = fDefault[1];
    }}

    function applyColorRange() {{
      const a = Number(cMinEl.value);
      const b = Number(cMaxEl.value);
      if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return;
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      Plotly.restyle(gd, {{"zmin": [lo], "zmax": [hi], "zauto": [false]}}, [0]);
    }}

    function applyXRange() {{
      const a = Number(xMinEl.value);
      const b = Number(xMaxEl.value);
      if (!Number.isFinite(a) || !Number.isFinite(b)) return;
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      Plotly.relayout(gd, {{
        "xaxis.range": [lo, hi],
        "xaxis2.range": [lo, hi],
        "xaxis3.range": [lo, hi]
      }});
    }}

    function applyFrameRange() {{
      const a = Number(fMinEl.value);
      const b = Number(fMaxEl.value);
      if (!Number.isFinite(a) || !Number.isFinite(b)) return;
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);

      // Heatmap frame zoom.
      Plotly.relayout(gd, {{"yaxis.range": [lo, hi]}});

      // Hide traces outside selected frame window.
      let avgVis = [];
      let frameVis = [];
      const ops = [];
      if (avgTraceIdx.length > 0) {{
        avgVis = avgFrameLabels.map((f) => (f >= lo && f <= hi));
        ops.push(Plotly.restyle(gd, {{"visible": avgVis}}, avgTraceIdx));
      }}
      if (frameTraceIdx.length > 0) {{
        frameVis = frameLabels.map((f) => (f >= lo && f <= hi));
        ops.push(Plotly.restyle(gd, {{"visible": frameVis}}, frameTraceIdx));
      }}

      // After visibility updates, autoscale y for line panels so selected
      // spectra fill each panel vertically.
      Promise.all(ops).then(() => {{
        const relayout = {{}};
        if (avgVis.some(Boolean)) relayout["yaxis2.autorange"] = true;
        if (frameVis.some(Boolean)) relayout["yaxis3.autorange"] = true;
        if (Object.keys(relayout).length > 0) {{
          Plotly.relayout(gd, relayout);
        }}
      }});
    }}

    Plotly.newPlot(gd, fig.data, fig.layout, {{responsive: true, displaylogo: false}}).then(() => {{
      setDefaults();
      applyColorRange();
      applyFrameRange();
      document.getElementById("apply_c").addEventListener("click", applyColorRange);
      document.getElementById("reset_c").addEventListener("click", () => {{
        cMinEl.value = cDefault[0];
        cMaxEl.value = cDefault[1];
        applyColorRange();
      }});
      document.getElementById("apply_x").addEventListener("click", applyXRange);
      document.getElementById("reset_x").addEventListener("click", () => {{
        xMinEl.value = xDefault[0];
        xMaxEl.value = xDefault[1];
        applyXRange();
      }});
      document.getElementById("apply_f").addEventListener("click", applyFrameRange);
      document.getElementById("reset_f").addEventListener("click", () => {{
        fMinEl.value = fDefault[0];
        fMaxEl.value = fDefault[1];
        applyFrameRange();
      }});
    }});
  </script>
</body>
</html>
"""
    with open(path_preview, "w", encoding="utf-8") as f:
        f.write(page)
    saved = [path_preview]
    if display_inline:
        fig.show()

    return saved


def _save_png_chunk(
    per_frame_specs: np.ndarray,
    specs_avg: np.ndarray,
    chunk_start: int,
    chunk_end: int,
    n_frames: int,
    num_groups: int,
    x1: int,
    x2: int,
    col_end: int,
    aver_n: int,
    cmap_name: str,
    factor: float,
    save_dir: str,
    data_path: str,
) -> list:
    """Deprecated static-PNG path retained for compatibility."""
    print("PNG preview output is deprecated in the web-first refactor. No files written.")
    return []
