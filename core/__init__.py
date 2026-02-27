"""Core numerical namespace for Dispersive_XAS.

This namespace excludes GUI/web interactivity.
"""

from .analysis import XAS_spec, spec_average
from .batch import norm_spec_preview
from .calibration import EDXAS_Calibrate, calibrate_regression
from .data_io import (
    load_bluesky_h5,
    load_nexus_entry,
    load_processed,
    load_processed_scans,
    raw_loading,
    saveh5,
)
from .image_processing import find_shifts, register_thresholding, stitch_scans
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

__all__ = [
    "XAS_spec",
    "spec_average",
    "norm_spec_preview",
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
]
