"""Batch preview processing for large time-resolved DXAS datasets.

Provides fast, memory-efficient frame-by-frame spectrum generation and
multi-format preview plots for quick quality assessment before full
calibrated analysis.

Key distinction
---------------
:func:`norm_spec_preview` normalises by **pixel index** (fast, no energy
calibration needed).  For calibrated, energy-axis normalisation use
:func:`~Dispersive_XAS.spectrum.norm_spec` instead.

Output formats
--------------
``output_format='html'`` (default)
    Three interactive Plotly HTML files per chunk — open in any browser,
    supports free zoom, pan, and hover tooltips.

``output_format='png'``
    Three static Matplotlib PNG files per chunk (original behaviour).

``output_format='both'``
    All six files per chunk.

Inline display (Jupyter)
------------------------
Pass ``display_inline=True`` to render the Plotly figures directly inside
a Jupyter notebook cell (in addition to saving the HTML files).  This
enables instant interactive zoom/pan without leaving the notebook.
"""

import os
from typing import Optional, Tuple

import numpy as np

from .io import load_nexus_entry

__all__ = [
    "norm_spec_preview",
    "plot_spectra_in_chunks",
    "preview_spectra_html",
]

_DEFAULT_FACTOR = 200.0


def norm_spec_preview(
    spec: np.ndarray,
    x1: int,
    x2: int,
    factor: float = _DEFAULT_FACTOR,
) -> np.ndarray:
    """Normalise a 1-D spectrum by its value range within a pixel window.

    Non-finite values are replaced with in-range statistics before
    normalisation so the function is safe for raw (uncalibrated) spectra.

    .. note::
        This is a **pixel-index** normalisation intended for quick batch
        previews.  It returns values in ``[0, factor]``, not ``[0, 1]``.
        For scientific energy-axis normalisation use
        :func:`~Dispersive_XAS.spectrum.norm_spec`.

    Parameters
    ----------
    spec : ndarray, shape (W,)
        1-D spectrum (column-averaged intensity across spatial rows).
    x1, x2 : int
        Pixel index bounds of the reference region used for normalisation.
    factor : float
        Output scale (default: 200).

    Returns
    -------
    ndarray, shape (W,)
        Normalised spectrum in ``[0, factor]``.
    """
    spec = np.nan_to_num(
        spec,
        nan=float(np.nanmean(spec)),
        posinf=float(np.nanmax(spec)),
        neginf=float(np.nanmin(spec)),
    )
    ref = spec[x1:x2]
    smin, smax = float(np.min(ref)), float(np.max(ref))
    return ((spec - smin) / (smax - smin + 1e-12)) * factor


def _compute_chunk_specs(
    data: np.ndarray,
    flat_avg: np.ndarray,
    fr0: int,
    fr1: int,
    chunk_start: int,
    chunk_end: int,
    aver_n: int,
    norm_x1: int,
    norm_x2: int,
    factor: float,
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
    n_frames = chunk_end - chunk_start

    per_frame_specs = []
    for fi in range(chunk_start, chunk_end):
        data_row = np.average(data[fi : fi + 1, fr0:fr1, :], axis=0)
        mux = np.log(flat_avg / data_row)
        mux[~np.isfinite(mux)] = 0.0
        spec = np.average(mux, axis=0)
        per_frame_specs.append(norm_spec_preview(spec, norm_x1, norm_x2, factor))
    per_frame_specs = np.asarray(per_frame_specs)  # (n_frames, W)

    W = per_frame_specs.shape[1]
    num_groups = n_frames // aver_n if aver_n > 0 else 0

    specs_avg = []
    for gi in range(num_groups):
        s = chunk_start + gi * aver_n
        e = s + aver_n
        data_row = np.average(data[s:e, fr0:fr1, :], axis=0)
        mux = np.log(flat_avg / data_row)
        mux[~np.isfinite(mux)] = 0.0
        sp = np.average(mux, axis=0)
        specs_avg.append(norm_spec_preview(sp, norm_x1, norm_x2, factor))
    specs_avg = np.asarray(specs_avg) if specs_avg else np.empty((0, W))

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
) -> None:
    """Generate and save batch preview plots for a large DXAS scan.

    Processes the scan in chunks of *chunk_size* frames to keep memory
    use bounded.  Three plot types are saved per chunk:

    1. **Heatmap / Imshow** – 2-D colour map (frames × pixels).  In HTML
       mode this is fully zoomable and shows pixel/frame/intensity on hover.
    2. **Lines (averaged)** – spectra averaged in groups of *aver_n*, colour-
       coded from first (dark) to last (bright) frame in the chunk.
    3. **Lines (no averaging)** – one spectrum per frame, vertically offset.
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
        Pixel index bounds for :func:`norm_spec_preview`.
    x1, x2 : int
        Pixel column range for plot x-axis limits.
    chunk_size : int
        Frames per processing chunk (default: 1000).
    cmap_name : str
        Colormap name (default: ``'magma'``).  Accepted by both Matplotlib
        and Plotly.
    factor : float
        Normalisation scale passed to :func:`norm_spec_preview`.
    output_format : {'html', 'png', 'both'}
        * ``'html'`` – interactive Plotly HTML (default; open in any browser).
        * ``'png'``  – static Matplotlib PNG (original behaviour).
        * ``'both'`` – save all six files per chunk.
    max_line_traces : int
        Maximum number of line traces in the no-averaging HTML plot.
        Frames are subsampled uniformly when the chunk exceeds this limit.
        Default: 200.  (Ignored for PNG output.)
    display_inline : bool
        If ``True``, call ``fig.show()`` after saving each HTML file so that
        the interactive Plotly figures appear directly inside a Jupyter
        notebook cell.  Users can then zoom, pan, and hover without opening
        external files.  Default: ``False``.  (Ignored for PNG output.)
    """
    data = load_nexus_entry(data_path)["data"]   # (N, H, W)
    flat = load_nexus_entry(flat_path)["data"]    # (N, H, W)

    if end_frame is None:
        end_frame = data.shape[0]

    base_name = os.path.splitext(data_path)[0]
    save_dir = os.path.join(
        os.path.dirname(base_name),
        "preliminary_results_" + os.path.basename(base_name),
    )
    os.makedirs(save_dir, exist_ok=True)

    fr0, fr1 = flat_range
    flat_avg = np.average(flat[:, fr0:fr1, :], axis=0)  # (fr1-fr0, W)

    do_html = output_format in ("html", "both")
    do_png  = output_format in ("png",  "both")

    for chunk_start in range(start_frame, end_frame, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_frame)
        per_frame_specs, specs_avg, n_frames, num_groups, W = _compute_chunk_specs(
            data, flat_avg, fr0, fr1,
            chunk_start, chunk_end,
            aver_n, norm_x1, norm_x2, factor,
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
            )

        if do_png:
            saved += _save_png_chunk(
                per_frame_specs, specs_avg,
                chunk_start, chunk_end, n_frames, num_groups,
                x1, x2, col_end, aver_n, cmap_name, factor,
                save_dir, data_path,
            )

        print("Saved:\n" + "\n".join(f"  {p}" for p in saved))


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
        Pixel index bounds for :func:`norm_spec_preview`.
    chunk_size : int
        Frames read from disk at once (default: 1000).  Does **not** split
        the output — all frames always appear in a single set of HTML files.
    cmap_name : str
        Plotly/Matplotlib colormap name (default: ``'magma'``).
    factor : float
        Normalisation scale passed to :func:`norm_spec_preview`.
    max_line_traces : int
        Maximum line traces in the no-averaging plot; frames are subsampled
        uniformly beyond this limit (default: 200).
    display_inline : bool
        If ``True`` (default), render each figure directly inside the Jupyter
        notebook cell via ``fig.show()``.  Set to ``False`` to only save the
        HTML files without displaying them.
    """
    data = load_nexus_entry(data_path)["data"]   # (N, H, W)
    flat = load_nexus_entry(flat_path)["data"]    # (N, H, W)

    if end_frame is None:
        end_frame = data.shape[0]

    base_name = os.path.splitext(data_path)[0]
    save_dir = os.path.join(
        os.path.dirname(base_name),
        "preliminary_results_" + os.path.basename(base_name),
    )
    os.makedirs(save_dir, exist_ok=True)

    fr0, fr1 = flat_range
    flat_avg = np.average(flat[:, fr0:fr1, :], axis=0)  # (fr1-fr0, W)

    # Accumulate computed spectra across all chunks (raw images are NOT kept).
    all_per_frame: list = []
    all_avg: list = []

    for chunk_start in range(start_frame, end_frame, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_frame)
        per_frame_specs, specs_avg, n_frames, _num_groups, _W = _compute_chunk_specs(
            data, flat_avg, fr0, fr1,
            chunk_start, chunk_end,
            aver_n, norm_x1, norm_x2, factor,
        )
        if n_frames <= 0:
            continue
        all_per_frame.append(per_frame_specs)
        if len(specs_avg):
            all_avg.append(specs_avg)
        print(f"  processed frames {chunk_start}–{chunk_end}")

    if not all_per_frame:
        print("No frames processed.")
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

    # Subsampling for the no-avg line plot
    step = max(1, n_frames // max_line_traces)
    disp_idx = list(range(0, n_frames, step))
    n_disp = len(disp_idx)
    subsample_note = (
        f"1 in {step} shown ({n_disp} traces)" if step > 1
        else f"all {n_frames:,}"
    )

    # ---- Build combined 3-column figure (side by side) --------------------
    subplot_titles = [
        f"<b>Heatmap</b><br>{n_frames:,} frames",
        f"<b>Averaged lines</b><br>every {aver_n} frame(s)  [{num_groups:,} traces]",
        f"<b>Individual frames</b><br>no averaging  [{subsample_note}]",
    ]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=subplot_titles,
        shared_xaxes=False,  # each panel has its own x-axis (independent zoom)
        shared_yaxes=False,  # each panel has its own y-axis (independent zoom)
        horizontal_spacing=0.10,
    )

    # ---- Col 1: Heatmap ----------------------------------------------------
    fig.add_trace(
        go.Heatmap(
            z=per_frame_specs[:, x1:col_end],
            x=pixel_axis,
            y=frame_axis,
            colorscale=cmap_name,
            zmin=0,
            zmax=1.5 * factor,
            colorbar=dict(
                title=dict(text="Intensity", side="right"),
                len=0.5, x=0.29, y=0.5, thickness=10,
            ),
            hovertemplate=(
                "Pixel: %{x}<br>Frame: %{y}<br>"
                "Intensity: %{z:.2f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # ---- Col 2: Averaged lines ---------------------------------------------
    if num_groups > 0:
        colors = plotly.colors.sample_colorscale(
            cmap_name, np.linspace(0, 1, num_groups)
        )
        px_arr = np.arange(x1, col_end)
        step_ticks = max(1, num_groups // 10)
        for gi in range(num_groups):
            frame_label = chunk_start + gi * aver_n
            fig.add_trace(
                go.Scattergl(
                    x=px_arr,
                    y=specs_avg[gi, x1:col_end] + gi,
                    mode="lines",
                    name=f"Frame {frame_label}",
                    line=dict(width=1, color=colors[gi]),
                    hovertemplate=(
                        "Frame " + str(frame_label) + "<br>"
                        "Pixel: %{x}<br>"
                        "Intensity: %{customdata:.3f}<extra></extra>"
                    ),
                    customdata=specs_avg[gi, x1:col_end],
                    showlegend=False,
                ),
                row=1, col=2,
            )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(0, num_groups, step_ticks)),
            ticktext=[
                str(chunk_start + gi * aver_n)
                for gi in range(0, num_groups, step_ticks)
            ],
            row=1, col=2,
        )

    # ---- Col 3: Individual frames ------------------------------------------
    if n_disp > 0:
        colors_noavg = plotly.colors.sample_colorscale(
            cmap_name, np.linspace(0, 1, n_disp)
        )
        px_arr = np.arange(x1, col_end)
        step_ticks_noavg = max(1, n_frames // 10)
        for rank, i in enumerate(disp_idx):
            frame_label = chunk_start + i
            fig.add_trace(
                go.Scattergl(
                    x=px_arr,
                    y=per_frame_specs[i, x1:col_end] + i,
                    mode="lines",
                    name=f"Frame {frame_label}",
                    line=dict(width=0.8, color=colors_noavg[rank]),
                    hovertemplate=(
                        "Frame " + str(frame_label) + "<br>"
                        "Pixel: %{x}<br>"
                        "Intensity: %{customdata:.3f}<extra></extra>"
                    ),
                    customdata=per_frame_specs[i, x1:col_end],
                    showlegend=False,
                ),
                row=1, col=3,
            )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(0, n_frames, step_ticks_noavg)),
            ticktext=[
                str(chunk_start + i)
                for i in range(0, n_frames, step_ticks_noavg)
            ],
            row=1, col=3,
        )

    # ---- Global layout -----------------------------------------------------
    fig.update_layout(
        title=dict(
            text=(
                f"DXAS Batch Preview \u2014 {scan_name}"
                f"<br><sup>frames {chunk_start:,}\u2013{chunk_end:,}"
                f" &nbsp;|&nbsp; N={n_frames:,}"
                f" &nbsp;|&nbsp; averaged every {aver_n} frame(s)"
                f" &nbsp;|&nbsp; colour: {cmap_name}</sup>"
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
        width=1150,
        font=dict(size=11),
        showlegend=False,
        dragmode="zoom",
        hoverlabel=dict(bgcolor="white", font_size=11),
        margin=dict(t=150, b=60, l=70, r=70),
    )

    # Axis labels — x and y on every panel
    for col in (1, 2, 3):
        fig.update_xaxes(title_text="Pixel column", row=1, col=col)
    fig.update_yaxes(title_text="Frame index", row=1, col=1)
    fig.update_yaxes(title_text="Frame index", row=1, col=2)
    fig.update_yaxes(title_text="Frame index", row=1, col=3)

    # ---- Save --------------------------------------------------------------
    path_preview = os.path.join(
        save_dir,
        f"preview_{chunk_start:05d}-{chunk_end:05d}_N{n_frames}.html",
    )
    fig.write_html(path_preview, include_plotlyjs="cdn")
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
    """Save three static Matplotlib PNG plots for one chunk.

    Returns
    -------
    list of str
        Paths of the three saved PNG files.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    cmap = colormaps.get_cmap(cmap_name)
    scan_name = os.path.basename(data_path)
    W = per_frame_specs.shape[1]
    saved = []

    # ---- 1) Lines (no averaging) ------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 8))
    for i, sp in enumerate(per_frame_specs):
        ax.plot(sp + i)
    ax.set_xlim(x1, x2 + 50)
    ax.set_xlabel(scan_name)
    ax.set_ylabel("Frame index")
    ax.set_title(f"Lines (no avg) frames {chunk_start}–{chunk_end}  [N={n_frames}]")
    y_pos = np.arange(0, n_frames, max(1, n_frames // 10))
    ax.set_yticks(y_pos, [str(chunk_start + y) for y in y_pos])
    ax.set_ylim(
        float(np.min(per_frame_specs[0][x1:x2])),
        float(np.max(per_frame_specs[-1][x1:x2])) + (n_frames - 1),
    )
    path_noavg = os.path.join(
        save_dir,
        f"lines_noavg_{chunk_start:05d}-{chunk_end:05d}_N{n_frames}.png",
    )
    fig.savefig(path_noavg, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(path_noavg)

    # ---- 2) Lines (averaged) ----------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 8))
    if num_groups > 0:
        colors = cmap(np.linspace(0, 1, num_groups))
        for gi, color in enumerate(colors):
            ax.plot(specs_avg[gi] + gi, color=color, linewidth=1)
        y_pos = np.arange(0, num_groups, max(1, num_groups // 10))
        ax.set_yticks(y_pos, [str(chunk_start + y * aver_n) for y in y_pos])
        ax.set_ylim(
            float(np.min(specs_avg[0][x1:x2])),
            float(np.max(specs_avg[-1][x1:x2])) + (num_groups - 1),
        )
    ax.set_xlim(x1, x2 + 50)
    ax.set_xlabel(scan_name)
    ax.set_ylabel("Averaged frame index")
    ax.set_title(
        f"Lines (avg {aver_n}) frames {chunk_start}–{chunk_end}  [N={num_groups}]"
    )
    path_avg = os.path.join(
        save_dir,
        f"lines_avg{aver_n}_{chunk_start:05d}-{chunk_end:05d}_N{num_groups}.png",
    )
    fig.savefig(path_avg, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(path_avg)

    # ---- 3) Imshow --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(
        per_frame_specs,
        aspect="auto",
        cmap=cmap_name,
        vmin=0,
        vmax=1.5 * factor,
        extent=[0, W, chunk_start, chunk_end],
        origin="lower",
    )
    ax.set_xlim(x1, x2 + 50)
    ax.set_xlabel(scan_name)
    ax.set_ylabel("Frame index")
    ax.set_title(
        f"Imshow (no avg) frames {chunk_start}–{chunk_end}  [N={n_frames}]"
    )
    ax.set_yticks(np.arange(chunk_start, chunk_end + 1, 100))
    ax.set_yticks(np.arange(chunk_start, chunk_end + 1, 10), minor=True)
    fig.colorbar(im, ax=ax, label="Normalised intensity")
    path_im = os.path.join(
        save_dir,
        f"imshow_{chunk_start:05d}-{chunk_end:05d}_N{n_frames}.png",
    )
    fig.savefig(path_im, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(path_im)

    return saved
