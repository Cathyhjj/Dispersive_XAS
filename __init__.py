"""Dispersive_XAS package.

Architecture:
- `Dispersive_XAS.core`: numerical/analysis code
- `Dispersive_XAS.web`: GUI/web plotting and ROI tools
"""

from . import core
from . import batch
from . import web
from . import visualization
from .batch import AnalysisConfig, BatchAnalysisConfig, run_analysis, run_large_quantity_analysis
from .core import (
    EDXAS_Calibrate,
    XAS_spec,
    apply_calibration_to_scan,
    atten_slope_corr,
    calibrate_from_reference_foil,
    calibrate_regression,
    find_h5_files,
    find_edge_jump,
    find_edge_pnts,
    find_nearest_flatfield,
    find_shifts,
    intensity_at_energy,
    interpt_spec,
    list_standards,
    load_bluesky_h5,
    load_mask_h5,
    load_nexus_entry,
    load_processed,
    load_processed_scans,
    norm_spec,
    norm_spec_preview,
    peak_finder,
    pre_process,
    pre_process_scan,
    raw_loading,
    register_thresholding,
    saveh5,
    save_calibration_model,
    save_mask_h5,
    spec_average,
    spec_cropping,
    spec_save,
    spec_shaper,
    spec_wrapper,
    spectrum_generate,
    standard_spec,
    stitch_scans,
    ximea_correction,
    load_calibration_model,
)
from .core import calibration as calibration
from .core import analysis as analysis
from .core import batch as core_batch
from .core import crystal as crystal
from .core import data_io as data_io
from .core import h5io as h5io
from .core import image_processing as image_processing
from .core import preprocessing as preprocessing
from .core import spectrum as spectrum
from .core import standards as standards
from .core import utils as utils
from .core.utils import (
    binning,
    change_font_size,
    color_gradient,
    date_today,
    make_gif,
    make_video,
    time_now,
    timestamp_convert,
)
from .visualization import PgSpec, show_roi
from .web import (
    batch as web_batch,
    plot_spectra_in_chunks,
    preview_spectra_html,
    select_rect_roi,
    show_image,
    show_image_stack,
    show_line,
    show_lines,
    show_mask_overlay,
)

# Backward-compat alias for previous `Dispersive_XAS.io` API shape.
io = data_io

__version__ = "0.3.0"

__all__ = [
    "core",
    "batch",
    "web",
    "web_batch",
    "visualization",
    "crystal",
    "analysis",
    "core_batch",
    "h5io",
    "image_processing",
    "preprocessing",
    "spectrum",
    "standards",
    "utils",
    "data_io",
    "io",
    "BatchAnalysisConfig",
    "AnalysisConfig",
    "run_large_quantity_analysis",
    "run_analysis",
    "XAS_spec",
    "spec_average",
    "norm_spec_preview",
    "find_h5_files",
    "find_nearest_flatfield",
    "calibrate_from_reference_foil",
    "save_calibration_model",
    "load_calibration_model",
    "apply_calibration_to_scan",
    "plot_spectra_in_chunks",
    "preview_spectra_html",
    "EDXAS_Calibrate",
    "calibrate_regression",
    "find_shifts",
    "register_thresholding",
    "stitch_scans",
    "load_bluesky_h5",
    "load_nexus_entry",
    "load_mask_h5",
    "load_processed",
    "load_processed_scans",
    "raw_loading",
    "saveh5",
    "save_mask_h5",
    "pre_process",
    "pre_process_scan",
    "ximea_correction",
    "atten_slope_corr",
    "find_edge_jump",
    "find_edge_pnts",
    "intensity_at_energy",
    "interpt_spec",
    "norm_spec",
    "peak_finder",
    "spec_cropping",
    "spec_save",
    "spec_shaper",
    "spec_wrapper",
    "spectrum_generate",
    "list_standards",
    "standard_spec",
    "binning",
    "change_font_size",
    "color_gradient",
    "date_today",
    "make_gif",
    "make_video",
    "time_now",
    "timestamp_convert",
    "PgSpec",
    "select_rect_roi",
    "show_roi",
    "show_line",
    "show_lines",
    "show_image",
    "show_image_stack",
    "show_mask_overlay",
]
