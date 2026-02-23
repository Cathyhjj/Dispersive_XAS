"""Batch preview processing for large time-resolved DXAS datasets.

Provides fast, memory-efficient frame-by-frame spectrum generation and
multi-format preview plots for quick quality assessment before full
calibrated analysis.

Key distinction
---------------
:func:`norm_spec_preview` normalises by **pixel index** (fast, no energy
calibration needed).  For calibrated, energy-axis normalisation use
:func:`~Dispersive_XAS.spectrum.norm_spec` instead.
"""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .io import load_nexus_entry

__all__ = [
    "norm_spec_preview",
    "plot_spectra_in_chunks",
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
) -> None:
    """Generate and save batch preview plots for a large DXAS scan.

    Processes the scan in chunks of *chunk_size* frames to keep memory
    use bounded.  Three image files are saved per chunk:

    1. **Lines (no averaging)** – one spectrum per frame, vertically offset.
    2. **Lines (averaged)** – spectra averaged in groups of *aver_n*.
    3. **Imshow** – 2-D colour map (frames × pixels).

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
        Matplotlib colormap name for the imshow plot (default: ``'magma'``).
    factor : float
        Normalisation scale passed to :func:`norm_spec_preview`.
    """
    from matplotlib import colormaps

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

    cmap = colormaps.get_cmap(cmap_name)
    fr0, fr1 = flat_range
    flat_avg = np.average(flat[:, fr0:fr1, :], axis=0)  # (fr1-fr0, W)

    for chunk_start in range(start_frame, end_frame, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_frame)
        n_frames = chunk_end - chunk_start
        if n_frames <= 0:
            continue

        # ---- Per-frame spectra ----
        per_frame_specs = []
        for fi in range(chunk_start, chunk_end):
            data_row = np.average(data[fi : fi + 1, fr0:fr1, :], axis=0)
            mux = np.log(flat_avg / data_row)
            mux[~np.isfinite(mux)] = 0.0
            spec = np.average(mux, axis=0)
            per_frame_specs.append(norm_spec_preview(spec, norm_x1, norm_x2, factor))
        per_frame_specs = np.asarray(per_frame_specs)  # (n_frames, W)

        # ---- 1) Line plot – no averaging ----
        fig, ax = plt.subplots(figsize=(5, 8))
        for i, sp in enumerate(per_frame_specs):
            ax.plot(sp + i)
        ax.set_xlim(x1, x2 + 50)
        ax.set_xlabel(os.path.basename(data_path))
        ax.set_ylabel("Frame index")
        ax.set_title(
            f"Lines (no avg) frames {chunk_start}–{chunk_end}  [N={n_frames}]"
        )
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

        # ---- 2) Line plot – averaged ----
        num_groups = n_frames // aver_n if aver_n > 0 else 0
        fig, ax = plt.subplots(figsize=(5, 8))
        specs_avg = []
        if num_groups > 0:
            colors = cmap(np.linspace(0, 1, num_groups))
            for gi, color in zip(range(num_groups), colors):
                s = chunk_start + gi * aver_n
                e = s + aver_n
                data_row = np.average(data[s:e, fr0:fr1, :], axis=0)
                mux = np.log(flat_avg / data_row)
                mux[~np.isfinite(mux)] = 0.0
                sp = np.average(mux, axis=0)
                normed = norm_spec_preview(sp, norm_x1, norm_x2, factor)
                specs_avg.append(normed)
                ax.plot(normed + gi, color=color, linewidth=1)
        ax.set_xlim(x1, x2 + 50)
        ax.set_xlabel(os.path.basename(data_path))
        ax.set_ylabel("Averaged frame index")
        ax.set_title(
            f"Lines (avg {aver_n}) frames {chunk_start}–{chunk_end}  [N={num_groups}]"
        )
        if num_groups > 0:
            y_pos = np.arange(0, num_groups, max(1, num_groups // 10))
            ax.set_yticks(y_pos, [str(chunk_start + y * aver_n) for y in y_pos])
            ax.set_ylim(
                float(np.min(specs_avg[0][x1:x2])),
                float(np.max(specs_avg[-1][x1:x2])) + (num_groups - 1),
            )
        path_avg = os.path.join(
            save_dir,
            f"lines_avg{aver_n}_{chunk_start:05d}-{chunk_end:05d}_N{num_groups}.png",
        )
        fig.savefig(path_avg, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ---- 3) Imshow ----
        fig, ax = plt.subplots(figsize=(8, 10))
        im = ax.imshow(
            per_frame_specs,
            aspect="auto",
            cmap=cmap_name,
            vmin=0,
            vmax=1.5 * factor,
            extent=[0, per_frame_specs.shape[1], chunk_start, chunk_end],
            origin="lower",
        )
        ax.set_xlim(x1, x2 + 50)
        ax.set_xlabel(os.path.basename(data_path))
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

        print(f"Saved:\n  {path_noavg}\n  {path_avg}\n  {path_im}")
