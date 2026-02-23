"""Backward-compatible facade for the Dispersive_XAS package.

All symbols that were previously defined in this monolithic module are
re-exported here so that existing code using::

    import Dispersive_XAS.xas as xas

continues to work without modification.

For new code, prefer importing from the package directly::

    import Dispersive_XAS as dxas
    # or from specific submodules:
    from Dispersive_XAS.preprocessing import pre_process
    from Dispersive_XAS.spectrum import norm_spec
    from Dispersive_XAS.calibration import EDXAS_Calibrate
    from Dispersive_XAS.analysis import XAS_spec
    from Dispersive_XAS.visualization import PgSpec
"""

# Re-export everything from the reorganised submodules
from .analysis import XAS_spec, spec_average
from .batch import norm_spec_preview, plot_spectra_in_chunks
from .calibration import EDXAS_Calibrate, calibrate_regression
from .image_processing import find_shifts, register_thresholding, stitch_scans
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
