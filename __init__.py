"""Dispersive_XAS package.

Architecture:
- `Dispersive_XAS.core`: numerical/analysis code
- `Dispersive_XAS.web`: GUI/web plotting and ROI tools
"""

from . import core
from . import batch
from . import progress
from . import web
from . import visualization
from .batch import AnalysisConfig, BatchAnalysisConfig, run_analysis, run_large_quantity_analysis
from .progress import BatchProgressReporter
from .core import (
    EDXAS_Calibrate,
    XAS_spec,
    apply_calibration_to_scan,
    atten_slope_corr,
    build_roi_mask,
    calibrate_from_reference_foil,
    calibrate_regression,
    find_h5_files,
    find_edge_jump,
    find_edge_pnts,
    find_nearest_flatfield,
    fit_tilted_band_roi,
    find_shifts,
    infer_tilted_band_roi_from_paths,
    intensity_at_energy,
    interpt_spec,
    list_standards,
    load_roi_json,
    load_bluesky_h5,
    load_mask_h5,
    load_nexus_entry,
    load_processed,
    load_processed_scans,
    make_tilted_band_roi,
    normalize_roi_spec,
    norm_spec,
    norm_spec_preview,
    peak_finder,
    prepare_roi_weights,
    pre_process,
    pre_process_scan,
    raw_loading,
    register_thresholding,
    roi_boundary_rows,
    roi_row_bounds,
    roi_weighted_column_mean,
    save_roi_json,
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
    tilted_band_controls_from_roi,
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
    select_tilted_band_roi,
    show_image,
    show_image_stack,
    show_line,
    show_lines,
    show_mask_overlay,
    TiltedBandROIEditor,
)

# Backward-compat alias for previous `Dispersive_XAS.io` API shape.
io = data_io

__version__ = "0.3.0"

__all__ = [
    "core",
    "batch",
    "progress",
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
    "BatchProgressReporter",
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
    "normalize_roi_spec",
    "build_roi_mask",
    "prepare_roi_weights",
    "roi_row_bounds",
    "roi_boundary_rows",
    "roi_weighted_column_mean",
    "fit_tilted_band_roi",
    "infer_tilted_band_roi_from_paths",
    "make_tilted_band_roi",
    "tilted_band_controls_from_roi",
    "save_roi_json",
    "load_roi_json",
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
    "TiltedBandROIEditor",
    "select_rect_roi",
    "select_tilted_band_roi",
    "show_roi",
    "show_line",
    "show_lines",
    "show_image",
    "show_image_stack",
    "show_mask_overlay",
]
