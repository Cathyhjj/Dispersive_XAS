"""Dispersive_XAS package.

Architecture:
- `Dispersive_XAS.core`: numerical/analysis code
- `Dispersive_XAS.web`: GUI/web plotting and ROI tools
"""

from . import core
from . import web
from . import visualization
from .core import (
    EDXAS_Calibrate,
    XAS_spec,
    atten_slope_corr,
    calibrate_regression,
    find_edge_jump,
    find_edge_pnts,
    find_shifts,
    intensity_at_energy,
    interpt_spec,
    list_standards,
    load_bluesky_h5,
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
    spec_average,
    spec_cropping,
    spec_save,
    spec_shaper,
    spec_wrapper,
    spectrum_generate,
    standard_spec,
    stitch_scans,
    ximea_correction,
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
    batch as batch,
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
    "web",
    "visualization",
    "crystal",
    "analysis",
    "batch",
    "core_batch",
    "h5io",
    "image_processing",
    "preprocessing",
    "spectrum",
    "standards",
    "utils",
    "data_io",
    "io",
    "XAS_spec",
    "spec_average",
    "norm_spec_preview",
    "plot_spectra_in_chunks",
    "preview_spectra_html",
    "EDXAS_Calibrate",
    "calibrate_regression",
    "find_shifts",
    "register_thresholding",
    "stitch_scans",
    "load_bluesky_h5",
    "load_nexus_entry",
    "load_processed",
    "load_processed_scans",
    "raw_loading",
    "saveh5",
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
