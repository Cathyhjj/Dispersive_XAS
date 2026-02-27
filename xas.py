"""Backward-compatible facade.

Prefer `import Dispersive_XAS as dxas` for new code.
"""

from .core.analysis import XAS_spec, spec_average
from .core.batch import norm_spec_preview
from .core.calibration import EDXAS_Calibrate, calibrate_regression
from .core.data_io import (
    load_bluesky_h5,
    load_nexus_entry,
    load_processed,
    load_processed_scans,
    raw_loading,
    saveh5,
)
from .core.image_processing import find_shifts, register_thresholding, stitch_scans
from .core.preprocessing import pre_process, pre_process_scan, ximea_correction
from .core.spectrum import (
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
from .core.standards import list_standards, standard_spec
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
    plot_spectra_in_chunks,
    preview_spectra_html,
    select_rect_roi,
    show_image,
    show_image_stack,
    show_line,
    show_lines,
    show_mask_overlay,
)
