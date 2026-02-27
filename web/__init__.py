"""Web UI namespace for Dispersive_XAS.

This namespace intentionally contains visualization/interaction only.
Numerical processing lives in :mod:`Dispersive_XAS.core`.
"""

from .plotting import show_image, show_image_stack, show_line, show_lines, show_mask_overlay
from .roi import PgSpec, select_rect_roi, show_roi
from .batch import plot_spectra_in_chunks, preview_spectra_html

__all__ = [
    "PgSpec",
    "select_rect_roi",
    "show_roi",
    "show_line",
    "show_lines",
    "show_image",
    "show_image_stack",
    "show_mask_overlay",
    "plot_spectra_in_chunks",
    "preview_spectra_html",
]
