"""Core numerical namespace for Dispersive_XAS.

This namespace excludes GUI/web interactivity.
"""

from .analysis import XAS_spec, spec_average
from .batch import (
    apply_calibration_to_scan,
    calibrate_from_reference_foil,
    find_h5_files,
    find_nearest_flatfield,
    load_calibration_model,
    norm_spec_preview,
    save_calibration_model,
)
from .calibration import EDXAS_Calibrate, calibrate_regression
from .data_io import (
    load_mask_h5,
    load_bluesky_h5,
    load_nexus_entry,
    load_processed,
    load_processed_scans,
    raw_loading,
    save_mask_h5,
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
    "find_h5_files",
    "find_nearest_flatfield",
    "calibrate_from_reference_foil",
    "save_calibration_model",
    "load_calibration_model",
    "apply_calibration_to_scan",
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
]
