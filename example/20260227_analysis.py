"""Large-quantity DXAS analysis workflow (calibration + transition analytics).

This script is designed for thousands of frames:
1) Optional fast HTML previews.
2) Fit one pixel->energy calibration from Cu foil + Cu standard.
3) Save/reload calibration model.
4) Apply calibration to all scan spectra in chunks.
5) Analyze transition dynamics:
   - smoothed spectra
   - Peak A / Peak B intensity ratio vs time
   - two-state residual (intermediate-phase indicator)
   - difference-spectrum evolution map
6) Export interactive HTML figures with free zoom/pan.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import hdf5plugin  # noqa: F401  # required before h5py for compressed datasets
import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, savgol_filter

# Make `import Dispersive_XAS` work when running this script directly.
PYTHON_CODES_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODES_DIR))

import Dispersive_XAS as dxas


@dataclass
class AnalysisConfig:
    data_dir: Path
    scan_file: str = "20251011_1915_SHUANG_CuO_start_Al100um_100ms_001.h5"
    reverse_scan_file: str = "20251011_2019_SHUANG_CuO_stop_Al100um_100ms_001.h5"
    use_reverse_scan: bool = False
    foil_file: str = "20251011_1857_Cu_foil_Al100um_Ag25um_100ms_002.h5"
    cu_standard: Path = (
        Path(__file__).resolve().parents[1]
        / "01_analysis_for_one_spectra"
        / "20250717_Ni_foil_and_sample"
        / "standard_XAS"
        / "APS"
        / "CuFoil_new.0001.nor"
    )
    row_range: tuple[int, int] = (155, 235)
    norm_range_pixels: tuple[int, int] = (50, 130)
    chunk_size: int = 500
    preview_chunk_size: int = 1000
    preview_median_size: int = 3
    analysis_dirname: str = "analysis_20260227"
    overwrite: bool = False
    # Smoothing and peak analysis
    spectral_savgol_window: int = 11
    spectral_savgol_poly: int = 2
    temporal_smooth_window: int = 5
    peak_a_ev: float | None = None
    peak_b_ev: float | None = None
    peak_search_range: tuple[float, float] = (8970.0, 9040.0)
    peak_halfwidth_eV: float = 1.5
    ref_average_frames: int = 200
    max_heatmap_frames: int = 2500


def _resolve_scan(path_or_name: str, data_dir: Path, include_kw: str) -> Path:
    p = Path(path_or_name)
    if p.is_file():
        return p.resolve()
    cand = data_dir / path_or_name
    if cand.is_file():
        return cand.resolve()
    scans = dxas.find_h5_files(
        str(data_dir),
        include=include_kw,
        exclude=("ff", "flatfield", "flat"),
    )
    if not scans:
        raise FileNotFoundError(
            f"Could not find scan '{path_or_name}' and no candidates with keyword '{include_kw}'."
        )
    return Path(scans[0]).resolve()


def _resolve_foil(path_or_name: str, data_dir: Path) -> Path:
    p = Path(path_or_name)
    if p.is_file():
        return p.resolve()
    cand = data_dir / path_or_name
    if cand.is_file():
        return cand.resolve()

    foils = dxas.find_h5_files(str(data_dir), include="Cu_foil")
    if not foils:
        raise FileNotFoundError("No Cu_foil file found in data directory.")
    preferred = [f for f in foils if f.endswith("_002.h5")]
    return Path(preferred[0] if preferred else foils[0]).resolve()


def _preview_output_dir(scan_path: Path) -> Path:
    base = scan_path.with_suffix("")
    return base.parent / f"preliminary_results_{base.name}"


def _stage_preview_html(scan_path: Path, analysis_dir: Path, tag: str) -> dict:
    src_dir = _preview_output_dir(scan_path)
    if not src_dir.exists():
        return {"files": [], "primary": None}

    staged: list[Path] = []
    for src in sorted(src_dir.glob("preview_*.html")):
        dst = analysis_dir / f"{tag}_{src.name}"
        shutil.copy2(src, dst)
        staged.append(dst)

    if not staged:
        return {"files": [], "primary": None}

    def _score(p: Path) -> tuple[int, int, float]:
        stem = p.stem  # e.g., forward_preview_00000-10000_N10000
        n_frames = 0
        span = 0
        try:
            n_frames = int(stem.split("_N")[-1])
        except Exception:  # noqa: BLE001
            n_frames = 0
        try:
            rng = stem.split("preview_")[-1].split("_N")[0]
            s0, s1 = rng.split("-")
            span = int(s1) - int(s0)
        except Exception:  # noqa: BLE001
            span = 0
        return (n_frames, span, p.stat().st_mtime)

    primary = max(staged, key=_score)
    return {"files": [str(p) for p in staged], "primary": str(primary)}


def _safe_window(window: int, n: int, poly: int) -> int:
    w = max(int(window), int(poly) + 3)
    if w % 2 == 0:
        w += 1
    max_w = n if (n % 2 == 1) else n - 1
    w = min(w, max_w)
    if w <= poly:
        w = poly + 3
        if w % 2 == 0:
            w += 1
    return max(3, min(w, max_w))


def _smooth_spectra(
    spectra: np.ndarray,
    spectral_window: int,
    spectral_poly: int,
    temporal_window: int,
) -> np.ndarray:
    arr = np.asarray(spectra, dtype=float)
    out = arr.copy()
    if spectral_window > 1 and out.shape[1] >= 5:
        w = _safe_window(spectral_window, out.shape[1], spectral_poly)
        out = savgol_filter(out, window_length=w, polyorder=min(spectral_poly, w - 1), axis=1)
    if temporal_window > 1 and out.shape[0] > 2:
        out = uniform_filter1d(out, size=int(temporal_window), axis=0, mode="nearest")
    return out


def _auto_pick_peak_centers(
    energy: np.ndarray,
    mean_spec: np.ndarray,
    search_range: tuple[float, float],
) -> tuple[float, float]:
    lo, hi = sorted((float(search_range[0]), float(search_range[1])))
    m = (energy >= lo) & (energy <= hi)
    x = np.asarray(energy[m], dtype=float)
    y = np.asarray(mean_spec[m], dtype=float)
    if x.size < 5:
        return float(np.median(energy) - 5.0), float(np.median(energy) + 5.0)

    amp = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
    prominence = max(0.002, amp * 0.05)
    idx, props = find_peaks(y, prominence=prominence, distance=max(3, x.size // 50))
    if idx.size >= 2:
        top2 = np.argsort(props["prominences"])[-2:]
        centers = np.sort(x[idx[top2]])
        return float(centers[0]), float(centers[1])

    # fallback: split window into two halves and take local maxima
    mid = x.size // 2
    a = x[np.argmax(y[:mid])] if mid > 1 else x[np.argmax(y)]
    b = x[mid + np.argmax(y[mid:])] if x.size - mid > 1 else x[np.argmax(y)]
    return float(min(a, b)), float(max(a, b))


def _peak_intensity(spectra: np.ndarray, energy: np.ndarray, center_eV: float, halfwidth_eV: float) -> np.ndarray:
    m = (energy >= center_eV - halfwidth_eV) & (energy <= center_eV + halfwidth_eV)
    if np.count_nonzero(m) < 2:
        idx = int(np.argmin(np.abs(energy - center_eV)))
        return np.asarray(spectra[:, idx], dtype=float)
    return np.nanmax(np.asarray(spectra[:, m], dtype=float), axis=1)


def _two_state_metrics(
    spectra: np.ndarray,
    ref_avg_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = spectra.shape[0]
    k = int(np.clip(ref_avg_n, 5, max(5, n // 2)))
    s0 = np.mean(spectra[:k], axis=0)
    s1 = np.mean(spectra[-k:], axis=0)
    v = s0 - s1
    denom = float(np.dot(v, v)) + 1e-12

    alpha = np.dot(spectra - s1[None, :], v) / denom
    alpha = np.clip(alpha, 0.0, 1.0)
    fit = alpha[:, None] * s0[None, :] + (1.0 - alpha[:, None]) * s1[None, :]
    resid = np.sqrt(np.mean((spectra - fit) ** 2, axis=1))
    return alpha, resid, s0, s1


def _downsample_rows_for_heatmap(
    data: np.ndarray,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = data.shape[0]
    if n <= max_rows:
        frame_axis = np.arange(n, dtype=float)
        return data, frame_axis
    edges = np.linspace(0, n, max_rows + 1, dtype=int)
    ds = np.vstack([np.mean(data[edges[i] : edges[i + 1]], axis=0) for i in range(max_rows)])
    frame_axis = 0.5 * (edges[:-1] + edges[1:] - 1)
    return ds, frame_axis


def _write_interactive_plots(
    energy: np.ndarray,
    spectra_smooth: np.ndarray,
    ratio: np.ndarray,
    alpha: np.ndarray,
    residual: np.ndarray,
    diff: np.ndarray,
    peak_a: float,
    peak_b: float,
    out_dir: Path,
    tag: str,
    max_heatmap_frames: int,
) -> dict:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_dir.mkdir(parents=True, exist_ok=True)

    diff_ds, frame_ds = _downsample_rows_for_heatmap(diff, max_rows=max_heatmap_frames)
    fig_map = go.Figure(
        data=[
            go.Heatmap(
                z=diff_ds,
                x=energy,
                y=frame_ds,
                colorscale="RdBu",
                zmid=0.0,
                colorbar=dict(title="Δμ"),
            )
        ]
    )
    fig_map.add_vline(x=peak_a, line=dict(color="black", width=2, dash="dot"))
    fig_map.add_vline(x=peak_b, line=dict(color="black", width=2, dash="dot"))
    fig_map.update_layout(
        template="plotly_white",
        title=f"{tag}: Difference spectra map (zoomable)",
        xaxis_title="Energy (eV)",
        yaxis_title="Frame index (downsampled if needed)",
        height=620,
    )
    map_html = out_dir / f"{tag}_difference_map.html"
    p1, p99 = np.nanpercentile(diff_ds, [1.0, 99.0])
    if not np.isfinite(p1):
        p1 = float(np.nanmin(diff_ds))
    if not np.isfinite(p99):
        p99 = float(np.nanmax(diff_ds))
    if not np.isfinite(p1) or not np.isfinite(p99) or p1 == p99:
        p1, p99 = -1.0, 1.0
    sym = float(max(abs(p1), abs(p99)))
    if sym <= 0:
        sym = 1.0

    map_html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{tag} difference map</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 12px; }}
    #ctrls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 8px; }}
    label {{ font-size: 13px; color: #374151; }}
    input, select, button {{ font-size: 13px; padding: 3px 6px; }}
    #plot {{ width: 100%; height: 88vh; min-height: 620px; }}
  </style>
</head>
<body>
  <div id="ctrls">
    <label>Cmap
      <select id="cmap">
        <option value="RdBu" selected>RdBu</option>
        <option value="RdBu_r">RdBu_r</option>
        <option value="Viridis">Viridis</option>
        <option value="Magma">Magma</option>
        <option value="Cividis">Cividis</option>
        <option value="Turbo">Turbo</option>
      </select>
    </label>
    <label>vmin <input id="vmin" type="number" step="any" /></label>
    <label>vmax <input id="vmax" type="number" step="any" /></label>
    <button id="apply">Apply</button>
    <button id="auto">Auto (P1-P99)</button>
    <button id="sym">Symmetric</button>
  </div>
  <div id="plot"></div>
  <script>
    const fig = {fig_map.to_json()};
    const plot = document.getElementById("plot");
    Plotly.newPlot(plot, fig.data, fig.layout, {{responsive: true, displaylogo: false}});

    const p1 = {float(p1)};
    const p99 = {float(p99)};
    const sym = {sym};
    const vminEl = document.getElementById("vmin");
    const vmaxEl = document.getElementById("vmax");
    const cmapEl = document.getElementById("cmap");
    vminEl.value = (-sym).toFixed(6);
    vmaxEl.value = (sym).toFixed(6);

    function applyRange() {{
      const vmin = Number(vminEl.value);
      const vmax = Number(vmaxEl.value);
      if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmin >= vmax) return;
      Plotly.restyle(plot, {{zmin: vmin, zmax: vmax}}, [0]);
    }}
    function autoRange() {{
      vminEl.value = p1.toFixed(6);
      vmaxEl.value = p99.toFixed(6);
      applyRange();
    }}
    function symmetricRange() {{
      const a = Math.max(Math.abs(Number(vminEl.value) || 0), Math.abs(Number(vmaxEl.value) || 0), sym);
      vminEl.value = (-a).toFixed(6);
      vmaxEl.value = a.toFixed(6);
      applyRange();
    }}
    document.getElementById("apply").addEventListener("click", applyRange);
    document.getElementById("auto").addEventListener("click", autoRange);
    document.getElementById("sym").addEventListener("click", symmetricRange);
    cmapEl.addEventListener("change", () => {{
      Plotly.restyle(plot, {{colorscale: cmapEl.value}}, [0]);
    }});
    applyRange();
  </script>
</body>
</html>
"""
    map_html.write_text(map_html_text, encoding="utf-8")

    n = spectra_smooth.shape[0]
    idx = np.unique(np.linspace(0, n - 1, 8, dtype=int))
    fig_summary = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.08,
        subplot_titles=(
            "Peak A / Peak B intensity ratio",
            "Two-state fraction (alpha) and residual",
            "Smoothed spectra (sampled frames)",
        ),
    )
    fig_summary.add_trace(
        go.Scatter(x=np.arange(n), y=ratio, mode="lines", name="A/B ratio", line=dict(color="#1f77b4")),
        row=1,
        col=1,
    )
    fig_summary.add_trace(
        go.Scatter(x=np.arange(n), y=alpha, mode="lines", name="alpha", line=dict(color="#2ca02c")),
        row=2,
        col=1,
    )
    fig_summary.add_trace(
        go.Scatter(x=np.arange(n), y=residual, mode="lines", name="residual RMS", line=dict(color="#d62728")),
        row=2,
        col=1,
    )
    for i in idx:
        fig_summary.add_trace(
            go.Scatter(
                x=energy,
                y=spectra_smooth[i],
                mode="lines",
                name=f"frame_{i}",
                line=dict(width=1),
                opacity=0.55,
            ),
            row=3,
            col=1,
        )
    fig_summary.add_vline(x=peak_a, row=3, col=1, line=dict(color="black", width=2, dash="dot"))
    fig_summary.add_vline(x=peak_b, row=3, col=1, line=dict(color="black", width=2, dash="dot"))
    fig_summary.update_layout(
        template="plotly_white",
        title=f"{tag}: Ratio + phase metrics + spectra",
        height=980,
    )
    fig_summary.update_xaxes(title_text="Frame index", row=1, col=1)
    fig_summary.update_xaxes(title_text="Frame index", row=2, col=1)
    fig_summary.update_xaxes(title_text="Energy (eV)", row=3, col=1)
    fig_summary.update_yaxes(title_text="I(A)/I(B)", row=1, col=1)
    fig_summary.update_yaxes(title_text="alpha / residual", row=2, col=1)
    fig_summary.update_yaxes(title_text="Normalized μ", row=3, col=1)
    summary_html = out_dir / f"{tag}_ratio_phase_summary.html"
    fig_summary.write_html(str(summary_html), include_plotlyjs="cdn")

    return {
        "difference_map_html": str(map_html),
        "summary_html": str(summary_html),
    }


def _analyze_calibrated_file(
    calibrated_h5: Path,
    out_dir: Path,
    tag: str,
    spectral_window: int,
    spectral_poly: int,
    temporal_window: int,
    peak_a_ev: float | None,
    peak_b_ev: float | None,
    peak_search_range: tuple[float, float],
    peak_halfwidth_eV: float,
    ref_average_frames: int,
    max_heatmap_frames: int,
) -> dict:
    with h5py.File(calibrated_h5, "r") as f:
        energy = np.asarray(f["energy"][:], dtype=float)
        spectra = np.asarray(f["spectra"][:], dtype=float)

    spectra_s = _smooth_spectra(
        spectra,
        spectral_window=spectral_window,
        spectral_poly=spectral_poly,
        temporal_window=temporal_window,
    )

    mean_spec = np.mean(spectra_s, axis=0)
    if peak_a_ev is None or peak_b_ev is None:
        auto_a, auto_b = _auto_pick_peak_centers(energy, mean_spec, search_range=peak_search_range)
        if peak_a_ev is None:
            peak_a_ev = auto_a
        if peak_b_ev is None:
            peak_b_ev = auto_b
    peak_a_ev = float(peak_a_ev)
    peak_b_ev = float(peak_b_ev)
    if peak_a_ev > peak_b_ev:
        peak_a_ev, peak_b_ev = peak_b_ev, peak_a_ev

    i_a = _peak_intensity(spectra_s, energy, center_eV=peak_a_ev, halfwidth_eV=peak_halfwidth_eV)
    i_b = _peak_intensity(spectra_s, energy, center_eV=peak_b_ev, halfwidth_eV=peak_halfwidth_eV)
    ratio = i_a / (i_b + 1e-12)

    alpha, residual, s0, s1 = _two_state_metrics(spectra_s, ref_avg_n=ref_average_frames)
    diff = spectra_s - s0[None, :]

    html = _write_interactive_plots(
        energy=energy,
        spectra_smooth=spectra_s,
        ratio=ratio,
        alpha=alpha,
        residual=residual,
        diff=diff,
        peak_a=peak_a_ev,
        peak_b=peak_b_ev,
        out_dir=out_dir,
        tag=tag,
        max_heatmap_frames=max_heatmap_frames,
    )

    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med))) + 1e-12
    z = (residual - med) / (1.4826 * mad)
    intermediate_idx = np.where(z > 3.0)[0]

    result = {
        "calibrated_h5": str(calibrated_h5),
        "n_frames": int(spectra_s.shape[0]),
        "peak_a_eV": peak_a_ev,
        "peak_b_eV": peak_b_ev,
        "spectral_window": int(spectral_window),
        "spectral_poly": int(spectral_poly),
        "temporal_window": int(temporal_window),
        "peak_halfwidth_eV": float(peak_halfwidth_eV),
        "ref_average_frames": int(ref_average_frames),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
        "residual_median": med,
        "residual_mad": mad,
        "intermediate_candidate_frames": intermediate_idx.tolist(),
        **html,
    }
    series_path = out_dir / f"{tag}_timeseries.csv"
    np.savetxt(
        series_path,
        np.column_stack((np.arange(spectra_s.shape[0]), ratio, alpha, residual)),
        delimiter=",",
        header="frame,ratio,alpha,residual_rms",
        comments="",
    )
    summary_path = out_dir / f"{tag}_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["summary_json"] = str(summary_path)
    result["timeseries_csv"] = str(series_path)
    return result


def _write_forward_reverse_compare(forward: dict, reverse: dict, out_dir: Path) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Read ratio/residual back from the HTML-input summaries by recomputing compact arrays.
    # We keep this lightweight and robust by reloading from calibrated files.
    def _ratio_resid(
        path: str,
        peak_a: float,
        peak_b: float,
        spectral_window: int,
        spectral_poly: int,
        temporal_window: int,
        peak_halfwidth_eV: float,
        ref_average_frames: int,
    ):
        with h5py.File(path, "r") as f:
            e = np.asarray(f["energy"][:], dtype=float)
            s = np.asarray(f["spectra"][:], dtype=float)
        s = _smooth_spectra(s, spectral_window, spectral_poly, temporal_window)
        ia = _peak_intensity(s, e, peak_a, peak_halfwidth_eV)
        ib = _peak_intensity(s, e, peak_b, peak_halfwidth_eV)
        r = ia / (ib + 1e-12)
        _, resid, _, _ = _two_state_metrics(s, ref_avg_n=ref_average_frames)
        return r, resid

    r_fw, resid_fw = _ratio_resid(
        forward["calibrated_h5"],
        forward["peak_a_eV"],
        forward["peak_b_eV"],
        int(forward["spectral_window"]),
        int(forward["spectral_poly"]),
        int(forward["temporal_window"]),
        float(forward["peak_halfwidth_eV"]),
        int(forward["ref_average_frames"]),
    )
    r_rv, resid_rv = _ratio_resid(
        reverse["calibrated_h5"],
        reverse["peak_a_eV"],
        reverse["peak_b_eV"],
        int(reverse["spectral_window"]),
        int(reverse["spectral_poly"]),
        int(reverse["temporal_window"]),
        float(reverse["peak_halfwidth_eV"]),
        int(reverse["ref_average_frames"]),
    )

    x_fw = np.linspace(0, 1, len(r_fw))
    x_rv = np.linspace(0, 1, len(r_rv))

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.11,
        subplot_titles=("Peak A/B ratio (normalized progression)", "Two-state residual (normalized progression)"),
    )
    fig.add_trace(go.Scatter(x=x_fw, y=r_fw, name="forward ratio", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_rv, y=r_rv, name="reverse ratio", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_fw, y=resid_fw, name="forward residual", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_rv, y=resid_rv, name="reverse residual", mode="lines"), row=2, col=1)
    fig.update_layout(template="plotly_white", title="Forward/Reverse comparison", height=760)
    fig.update_xaxes(title_text="Normalized progression (0->1)", row=1, col=1)
    fig.update_xaxes(title_text="Normalized progression (0->1)", row=2, col=1)
    fig.update_yaxes(title_text="I(A)/I(B)", row=1, col=1)
    fig.update_yaxes(title_text="Residual RMS", row=2, col=1)

    out_path = out_dir / "forward_reverse_comparison.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return str(out_path)


def _write_summary_dashboard(
    out_dir: Path,
    forward: dict,
    reverse: dict | None,
    compare_html: str | None,
    main_preview_html: str | None,
    manifest: dict,
) -> str:
    def _name(p: str | None) -> str:
        return Path(p).name if p else ""

    def _fmt(x: float | None) -> str:
        return "N/A" if x is None else f"{x:.6f}"

    forward_metrics = f"""
    <tr><td>Peak A (eV)</td><td>{_fmt(float(forward.get("peak_a_eV", np.nan)))}</td></tr>
    <tr><td>Peak B (eV)</td><td>{_fmt(float(forward.get("peak_b_eV", np.nan)))}</td></tr>
    <tr><td>Ratio Min</td><td>{_fmt(float(forward.get("ratio_min", np.nan)))}</td></tr>
    <tr><td>Ratio Max</td><td>{_fmt(float(forward.get("ratio_max", np.nan)))}</td></tr>
    <tr><td>Residual Median</td><td>{_fmt(float(forward.get("residual_median", np.nan)))}</td></tr>
    """

    reverse_metrics = ""
    reverse_blocks = ""
    preview_blocks = ""
    preview_links = ""
    if main_preview_html:
        preview_links = f'<a href="{_name(main_preview_html)}" target="_blank">10k spectra viewer</a>'
        preview_blocks = f"""
        <h2>10,000-Spectra Viewer (Main Scan)</h2>
        <iframe src="{_name(main_preview_html)}" loading="lazy"></iframe>
        """

    if reverse is not None:
        reverse_metrics = f"""
        <tr><td>Reverse Peak A (eV)</td><td>{_fmt(float(reverse.get("peak_a_eV", np.nan)))}</td></tr>
        <tr><td>Reverse Peak B (eV)</td><td>{_fmt(float(reverse.get("peak_b_eV", np.nan)))}</td></tr>
        <tr><td>Reverse Ratio Min</td><td>{_fmt(float(reverse.get("ratio_min", np.nan)))}</td></tr>
        <tr><td>Reverse Ratio Max</td><td>{_fmt(float(reverse.get("ratio_max", np.nan)))}</td></tr>
        <tr><td>Reverse Residual Median</td><td>{_fmt(float(reverse.get("residual_median", np.nan)))}</td></tr>
        """
        reverse_blocks = f"""
        <h2>Reverse: Difference Map</h2>
        <iframe src="{_name(reverse.get("difference_map_html"))}" loading="lazy"></iframe>
        <h2>Reverse: Ratio + Phase Summary</h2>
        <iframe src="{_name(reverse.get("summary_html"))}" loading="lazy"></iframe>
        """
        if compare_html:
            reverse_blocks += f"""
            <h2>Forward vs Reverse Comparison (if enabled)</h2>
            <iframe src="{_name(compare_html)}" loading="lazy"></iframe>
            """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>DXAS Analysis Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #1f2937; }}
    h1, h2 {{ margin: 10px 0; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; background: #fafafa; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 4px; font-size: 14px; }}
    td:first-child {{ color: #4b5563; width: 56%; }}
    iframe {{ width: 100%; min-height: 780px; border: 1px solid #d1d5db; border-radius: 8px; }}
    .small iframe {{ min-height: 650px; }}
    .links a {{ margin-right: 14px; }}
  </style>
</head>
<body>
  <h1>DXAS Large-Quantity Analysis Summary</h1>
  <div class="meta">
    <div class="card">
      <h2>Run Info</h2>
      <table>
        <tr><td>Scan</td><td>{Path(manifest["scan_path"]).name}</td></tr>
        <tr><td>Reverse Scan (optional)</td><td>{Path(manifest["reverse_scan_path"]).name if manifest["reverse_scan_path"] else "N/A"}</td></tr>
        <tr><td>Foil</td><td>{Path(manifest["foil_path"]).name}</td></tr>
        <tr><td>Calibration RMSE (eV)</td><td>{_fmt(float(manifest["rmse"]))}</td></tr>
      </table>
    </div>
    <div class="card">
      <h2>Metrics</h2>
      <table>
        {forward_metrics}
        {reverse_metrics}
      </table>
    </div>
  </div>

	  <div class="card links">
	    <strong>Data Exports:</strong>
	    <a href="{_name(forward.get("timeseries_csv"))}" target="_blank">main_timeseries.csv</a>
	    {f'<a href="{_name(reverse.get("timeseries_csv"))}" target="_blank">reverse_timeseries.csv</a>' if reverse else ""}
	    {preview_links}
	    <a href="analysis_manifest.json" target="_blank">analysis_manifest.json</a>
	  </div>

      {preview_blocks}

	  <h2>Main Scan: Difference Map</h2>
	  <iframe src="{_name(forward.get("difference_map_html"))}" loading="lazy"></iframe>
  <h2>Main Scan: Ratio + Phase Summary</h2>
  <iframe src="{_name(forward.get("summary_html"))}" loading="lazy"></iframe>

  <div class="small">
    {reverse_blocks}
  </div>
</body>
</html>
"""

    legacy = out_dir / "analysis_summary_all.html"
    if legacy.exists():
        legacy.unlink()
    out_path = out_dir / "00_analysis_summary_all.html"
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


def run_analysis(cfg: AnalysisConfig, make_previews: bool = True) -> dict:
    data_dir = cfg.data_dir.resolve()
    scan_path = _resolve_scan(cfg.scan_file, data_dir, include_kw="SHUANG_CuO_start")
    analysis_dir = data_dir / f"{cfg.analysis_dirname}_{scan_path.stem}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    reverse_path = None
    if cfg.use_reverse_scan:
        try:
            reverse_path = _resolve_scan(cfg.reverse_scan_file, data_dir, include_kw="SHUANG_CuO_stop")
        except Exception as exc:  # noqa: BLE001
            print(f"[warning] Reverse scan not found, skip reverse analysis: {exc}")
            reverse_path = None

    foil_path = _resolve_foil(cfg.foil_file, data_dir)
    scan_flat = Path(dxas.find_nearest_flatfield(str(scan_path), folder=str(data_dir)))
    foil_flat = Path(dxas.find_nearest_flatfield(str(foil_path), folder=str(data_dir)))
    reverse_flat = (
        Path(dxas.find_nearest_flatfield(str(reverse_path), folder=str(data_dir)))
        if reverse_path is not None
        else None
    )

    print(f"Scan      : {scan_path.name}")
    print(f"Scan flat : {scan_flat.name}")
    if reverse_path is not None:
        print(f"Reverse   : {reverse_path.name}")
        print(f"Rev flat  : {reverse_flat.name}")
    print(f"Cu foil   : {foil_path.name}")
    print(f"Foil flat : {foil_flat.name}")

    main_preview = {"files": [], "primary": None}
    reverse_preview = {"files": [], "primary": None}

    if make_previews:
        dxas.plot_spectra_in_chunks(
            data_path=str(scan_path),
            flat_path=str(scan_flat),
            aver_n=5,
            flat_range=cfg.row_range,
            norm_x1=cfg.norm_range_pixels[0],
            norm_x2=cfg.norm_range_pixels[1],
            x1=cfg.norm_range_pixels[0],
            x2=400,
            chunk_size=cfg.preview_chunk_size,
            cmap_name="magma",
            output_format="html",
            display_inline=False,
            median_size=cfg.preview_median_size,
        )
        dxas.preview_spectra_html(
            data_path=str(scan_path),
            flat_path=str(scan_flat),
            aver_n=5,
            flat_range=cfg.row_range,
            norm_x1=cfg.norm_range_pixels[0],
            norm_x2=cfg.norm_range_pixels[1],
            chunk_size=cfg.preview_chunk_size,
            cmap_name="magma",
            display_inline=False,
            median_size=cfg.preview_median_size,
        )
        if reverse_path is not None:
            dxas.preview_spectra_html(
                data_path=str(reverse_path),
                flat_path=str(reverse_flat),
                aver_n=5,
                flat_range=cfg.row_range,
                norm_x1=cfg.norm_range_pixels[0],
                norm_x2=cfg.norm_range_pixels[1],
                chunk_size=cfg.preview_chunk_size,
                cmap_name="magma",
                display_inline=False,
                median_size=cfg.preview_median_size,
            )

    # Stage the large-scan HTML viewer(s) into the analysis folder for
    # one-stop browsing. This also picks a "primary" 10k-style viewer.
    main_preview = _stage_preview_html(scan_path, analysis_dir, "forward")
    if reverse_path is not None:
        reverse_preview = _stage_preview_html(reverse_path, analysis_dir, "reverse")
    if not main_preview["primary"]:
        # Guarantee one full evolution viewer in summary even when preview files
        # were not previously generated.
        dxas.preview_spectra_html(
            data_path=str(scan_path),
            flat_path=str(scan_flat),
            aver_n=5,
            flat_range=cfg.row_range,
            norm_x1=cfg.norm_range_pixels[0],
            norm_x2=cfg.norm_range_pixels[1],
            chunk_size=cfg.preview_chunk_size,
            cmap_name="magma",
            display_inline=False,
            median_size=cfg.preview_median_size,
        )
        main_preview = _stage_preview_html(scan_path, analysis_dir, "forward")
    if main_preview["primary"]:
        print(f"[main] 10k viewer        : {main_preview['primary']}")
    if reverse_preview["primary"]:
        print(f"[reverse] 10k viewer     : {reverse_preview['primary']}")

    if not cfg.cu_standard.exists():
        raise FileNotFoundError(f"Cu standard not found: {cfg.cu_standard}")
    cu_standard = dxas.spec_shaper(np.loadtxt(cfg.cu_standard, usecols=(0, 1)))

    fit, meta = dxas.calibrate_from_reference_foil(
        foil_path=str(foil_path),
        flat_path=str(foil_flat),
        standard_spec=cu_standard,
        row_range=cfg.row_range,
        denoise_size=3,
        median_size=3,
        gaussian_sigma=0.2,
        norm_range_pixels=cfg.norm_range_pixels,
        interp_pts=1000,
        y_points=(0.2, 0.7),
        exp_edge_range_pixels=(0, 220),
        exp_peak_range_pixels=(100, 700),
        peak_n=8,
        peak_prominence=0.01,
        standard_norm_range_eV=(8950, 9050),
        standard_edge_range_eV=(8920, 9020),
        standard_peak_range_eV=(8950, 9300),
        poly_order=2,
        show=False,
        save_param=False,
    )

    model_name = f"calibration_{foil_path.stem}.json"
    model_path = data_dir / model_name
    if cfg.overwrite or (not model_path.exists()):
        dxas.save_calibration_model(str(model_path), fit, metadata=meta)
    model = dxas.load_calibration_model(str(model_path))
    print(f"Calibration RMSE (eV): {fit.rmse:.6f}")
    print(f"Calibration model     : {model_path}")

    def _apply_one(tag: str, data_path: Path, flat_path: Path) -> Path:
        out_name = f"calibrated_{data_path.stem}.h5"
        out_path = data_dir / out_name
        if cfg.overwrite or (not out_path.exists()):
            batch = dxas.apply_calibration_to_scan(
                data_path=str(data_path),
                flat_path=str(flat_path),
                calibration=model,
                row_range=cfg.row_range,
                norm_range_pixels=cfg.norm_range_pixels,
                denoise_size=3,
                median_size=3,
                gaussian_sigma=0.2,
                chunk_size=cfg.chunk_size,
                output_h5=str(out_path),
            )
            print(f"[{tag}] Calibrated spectra H5 : {batch['output_h5']}")
            print(f"[{tag}] Frames processed      : {batch['n_frames']}")
        else:
            print(f"[{tag}] Reusing existing calibrated file: {out_path}")
        return out_path

    forward_h5 = _apply_one("forward", scan_path, scan_flat)
    reverse_h5 = _apply_one("reverse", reverse_path, reverse_flat) if reverse_path is not None else None

    forward_result = _analyze_calibrated_file(
        calibrated_h5=forward_h5,
        out_dir=analysis_dir,
        tag="forward",
        spectral_window=cfg.spectral_savgol_window,
        spectral_poly=cfg.spectral_savgol_poly,
        temporal_window=cfg.temporal_smooth_window,
        peak_a_ev=cfg.peak_a_ev,
        peak_b_ev=cfg.peak_b_ev,
        peak_search_range=cfg.peak_search_range,
        peak_halfwidth_eV=cfg.peak_halfwidth_eV,
        ref_average_frames=cfg.ref_average_frames,
        max_heatmap_frames=cfg.max_heatmap_frames,
    )
    print(f"[forward] Peak A/B centers: {forward_result['peak_a_eV']:.3f}, {forward_result['peak_b_eV']:.3f} eV")
    print(f"[forward] Difference map   : {forward_result['difference_map_html']}")
    print(f"[forward] Summary plot    : {forward_result['summary_html']}")

    reverse_result = None
    compare_html = None
    if reverse_h5 is not None:
        # Keep peak definitions consistent for forward/reverse comparison.
        rev_peak_a = cfg.peak_a_ev if cfg.peak_a_ev is not None else float(forward_result["peak_a_eV"])
        rev_peak_b = cfg.peak_b_ev if cfg.peak_b_ev is not None else float(forward_result["peak_b_eV"])
        reverse_result = _analyze_calibrated_file(
            calibrated_h5=reverse_h5,
            out_dir=analysis_dir,
            tag="reverse",
            spectral_window=cfg.spectral_savgol_window,
            spectral_poly=cfg.spectral_savgol_poly,
            temporal_window=cfg.temporal_smooth_window,
            peak_a_ev=rev_peak_a,
            peak_b_ev=rev_peak_b,
            peak_search_range=cfg.peak_search_range,
            peak_halfwidth_eV=cfg.peak_halfwidth_eV,
            ref_average_frames=cfg.ref_average_frames,
            max_heatmap_frames=cfg.max_heatmap_frames,
        )
        compare_html = _write_forward_reverse_compare(forward_result, reverse_result, out_dir=analysis_dir)
        print(f"[reverse] Peak A/B centers: {reverse_result['peak_a_eV']:.3f}, {reverse_result['peak_b_eV']:.3f} eV")
        print(f"[compare] Forward/Reverse : {compare_html}")

    result = {
        "scan_path": str(scan_path),
        "scan_flat": str(scan_flat),
        "reverse_scan_path": str(reverse_path) if reverse_path is not None else None,
        "reverse_scan_flat": str(reverse_flat) if reverse_flat is not None else None,
        "foil_path": str(foil_path),
        "foil_flat": str(foil_flat),
        "calibration_model": str(model_path),
        "calibrated_forward_h5": str(forward_h5),
        "calibrated_reverse_h5": str(reverse_h5) if reverse_h5 is not None else None,
        "rmse": float(fit.rmse),
        "analysis_dir": str(analysis_dir),
        "forward_analysis": forward_result,
        "reverse_analysis": reverse_result,
        "forward_reverse_compare_html": compare_html,
        "forward_preview_html_files": main_preview["files"],
        "reverse_preview_html_files": reverse_preview["files"],
        "forward_preview_primary_html": main_preview["primary"],
        "reverse_preview_primary_html": reverse_preview["primary"],
        "main_preview_primary_html": main_preview["primary"],
    }
    summary_dashboard = _write_summary_dashboard(
        out_dir=analysis_dir,
        forward=forward_result,
        reverse=reverse_result,
        compare_html=compare_html,
        main_preview_html=main_preview["primary"],
        manifest=result,
    )
    result["summary_dashboard_html"] = summary_dashboard
    with open(analysis_dir / "analysis_manifest.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nInteractive exploration tips:")
    print(f"- Open one file to see everything: {summary_dashboard}")
    if main_preview["primary"]:
        print(f"- 10k spectra viewer (main): {main_preview['primary']}")
    if reverse_preview["primary"]:
        print(f"- 10k spectra viewer (reverse): {reverse_preview['primary']}")
    print("- Box-select to zoom, double-click to reset, and use pan mode for navigation.")
    print("- Use the Plotly toolbar to autoscale or download snapshots.")
    print("- Quantitative traces are saved as *_timeseries.csv for your own custom analysis.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DXAS large-quantity analysis pipeline.")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent / "data" / "Jiantao"),
        help="Directory containing scan/foil/flatfield H5 files.",
    )
    parser.add_argument(
        "--scan-file",
        default="20251011_1915_SHUANG_CuO_start_Al100um_100ms_001.h5",
        help="Target scan file (name or full path).",
    )
    parser.add_argument(
        "--reverse-scan-file",
        default="20251011_2019_SHUANG_CuO_stop_Al100um_100ms_001.h5",
        help="Reverse-process scan file (name or full path).",
    )
    parser.add_argument(
        "--with-reverse-scan",
        action="store_true",
        help="Enable reverse-process analysis/comparison.",
    )
    parser.add_argument(
        "--foil-file",
        default="20251011_1857_Cu_foil_Al100um_Ag25um_100ms_002.h5",
        help="Cu foil file for calibration (name or full path).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip HTML preview generation.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for calibrated batch processing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing calibrated/model outputs.",
    )
    parser.add_argument(
        "--peak-a-ev",
        type=float,
        default=None,
        help="Optional manual Peak A center (eV). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--peak-b-ev",
        type=float,
        default=None,
        help="Optional manual Peak B center (eV). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--spectral-window",
        type=int,
        default=11,
        help="Savitzky-Golay window for spectral smoothing (set 1 to disable).",
    )
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=5,
        help="Moving-average window across frames (set 1 to disable).",
    )
    parser.add_argument(
        "--preview-median-size",
        type=int,
        default=3,
        help="Median kernel for raw 10k viewer hot-pixel suppression (0 to disable).",
    )
    parser.add_argument(
        "--peak-halfwidth-ev",
        type=float,
        default=1.5,
        help="Half-width around each peak center (eV) for peak intensity extraction.",
    )
    parser.add_argument(
        "--peak-search-min",
        type=float,
        default=8970.0,
        help="Auto-peak search lower bound (eV).",
    )
    parser.add_argument(
        "--peak-search-max",
        type=float,
        default=9040.0,
        help="Auto-peak search upper bound (eV).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = AnalysisConfig(
        data_dir=Path(args.data_dir),
        scan_file=args.scan_file,
        reverse_scan_file=args.reverse_scan_file,
        use_reverse_scan=bool(args.with_reverse_scan),
        foil_file=args.foil_file,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        peak_a_ev=args.peak_a_ev,
        peak_b_ev=args.peak_b_ev,
        spectral_savgol_window=args.spectral_window,
        temporal_smooth_window=args.temporal_window,
        preview_median_size=args.preview_median_size,
        peak_halfwidth_eV=args.peak_halfwidth_ev,
        peak_search_range=(args.peak_search_min, args.peak_search_max),
    )
    run_analysis(config, make_previews=not args.no_preview)
