"""Dispersive XAS (DXAS) analysis package.

Provides a complete pipeline for converting raw dispersive X-ray absorption
spectroscopy images into calibrated, energy-resolved spectra:

    raw images → pre-processing → ROI selection → spectrum generation
    → normalisation → energy calibration

Submodules
----------
preprocessing
    Dark correction, median denoising, Beer-Lambert transmission/absorption.
spectrum
    Spectral generation, normalisation, interpolation, cropping, peak finding.
calibration
    Pixel-to-energy polynomial regression (:class:`EDXAS_Calibrate`).
visualization
    Interactive pyqtgraph image viewer with ROI management (:class:`PgSpec`).
analysis
    High-level :class:`XAS_spec` analysis workflow class.
io
    Data loading/saving (custom HDF5, NeXus areaDetector, Bluesky).
utils
    Date/time helpers, colour gradients, image binning, GIF/video export.
standards
    Built-in reference standard XAS spectra for calibration.
crystal
    Crystal-optics calculations (Laue / Bragg geometry).
image_processing
    Low-level image transformations (rotate, shift, flip, filter).
h5io
    HDF5 read/write library (h5rw format).

Quick-start
-----------
>>> import Dispersive_XAS as dxas
>>> result = dxas.pre_process(data, flat, savedata=False)
>>> viewer = dxas.PgSpec(result["mux"])
>>> xas_obj = dxas.XAS_spec(result["mux"], m=mask)
"""

__version__ = "0.2.0"

# --- Core submodules (always available) ---
from . import crystal
from . import h5io
from . import image_processing

# --- Functional submodules ---
from . import analysis
from . import calibration
from . import io
from . import preprocessing
from . import spectrum
from . import standards
from . import utils
from . import visualization

# --- Flat public API (mirrors numpy-style: functions available at top level) ---
from .analysis import XAS_spec
from .calibration import EDXAS_Calibrate, calibrate_regression
from .io import (
    load_bluesky_h5,
    load_nexus_entry,
    load_processed,
    load_processed_scans,
    raw_loading,
    saveh5,
)
from .preprocessing import pre_process, pre_process_scan, ximea_correction
from .spectrum import (
    atten_slope_corr,
    find_edge_jump,
    find_edge_pnts,
    intensity_at_energy,
    interpt_spec,
    norm_spec,
    peak_finder,
    spec_cropping,
    spec_save,
    spec_shaper,
    spec_wrapper,
    spectrum_generate,
)
from .standards import list_standards, standard_spec
from .utils import (
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

__all__ = [
    # submodules
    "crystal",
    "h5io",
    "image_processing",
    "analysis",
    "calibration",
    "io",
    "preprocessing",
    "spectrum",
    "standards",
    "utils",
    "visualization",
    # analysis
    "XAS_spec",
    # calibration
    "EDXAS_Calibrate",
    "calibrate_regression",
    # io
    "load_bluesky_h5",
    "load_nexus_entry",
    "load_processed",
    "load_processed_scans",
    "raw_loading",
    "saveh5",
    # preprocessing
    "pre_process",
    "pre_process_scan",
    "ximea_correction",
    # spectrum
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
    # standards
    "list_standards",
    "standard_spec",
    # utils
    "binning",
    "change_font_size",
    "color_gradient",
    "date_today",
    "make_gif",
    "make_video",
    "time_now",
    "timestamp_convert",
    # visualization
    "PgSpec",
    "show_roi",
]
