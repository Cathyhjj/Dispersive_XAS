"""Configuration models for large-quantity DXAS batch analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_DEFAULT_PYTHON_CODES_DIR = Path(__file__).resolve().parents[2]


@dataclass
class BatchAnalysisConfig:
    """Runtime settings for the large-quantity DXAS pipeline."""

    data_dir: Path = _DEFAULT_PYTHON_CODES_DIR / "02_analysis_for_large_quantity" / "data" / "Jiantao"
    scan_file: str = "20251011_1915_SHUANG_CuO_start_Al100um_100ms_001.h5"
    reverse_scan_file: str = "20251011_2019_SHUANG_CuO_stop_Al100um_100ms_001.h5"
    use_reverse_scan: bool = False
    foil_file: str = "20251011_1857_Cu_foil_Al100um_Ag25um_100ms_002.h5"
    cu_standard: Path = (
        _DEFAULT_PYTHON_CODES_DIR
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


# Backward-compatible alias (matches previous script naming).
AnalysisConfig = BatchAnalysisConfig


__all__ = ["BatchAnalysisConfig", "AnalysisConfig"]
