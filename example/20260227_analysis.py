"""Large-quantity DXAS analysis runner.

This script intentionally keeps only user-editable parameters and delegates
all workflow logic to ``Dispersive_XAS.batch``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import Dispersive_XAS` work when running this script directly.
PYTHON_CODES_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODES_DIR))

from Dispersive_XAS.batch import BatchAnalysisConfig, run_large_quantity_analysis


# -----------------------------------------------------------------------------
# User parameters
# -----------------------------------------------------------------------------
CONFIG = BatchAnalysisConfig(
    data_dir=Path(__file__).resolve().parent / "data" / "Jiantao",
    scan_file="20251011_1915_SHUANG_CuO_start_Al100um_100ms_001.h5",
    reverse_scan_file="20251011_2019_SHUANG_CuO_stop_Al100um_100ms_001.h5",
    use_reverse_scan=False,
    foil_file="20251011_1857_Cu_foil_Al100um_Ag25um_100ms_002.h5",
    row_range=(155, 235),
    norm_range_pixels=(50, 130),
    chunk_size=500,
    preview_chunk_size=1000,
    preview_median_size=3,
    analysis_dirname="analysis_20260227",
    overwrite=False,
    spectral_savgol_window=11,
    spectral_savgol_poly=2,
    temporal_smooth_window=5,
    peak_a_ev=None,
    peak_b_ev=None,
    peak_search_range=(8970.0, 9040.0),
    peak_halfwidth_eV=1.5,
    ref_average_frames=200,
    max_heatmap_frames=2500,
)

MAKE_PREVIEWS = True


if __name__ == "__main__":
    run_large_quantity_analysis(CONFIG, make_previews=MAKE_PREVIEWS)
