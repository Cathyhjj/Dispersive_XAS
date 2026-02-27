"""Backward-compatible visualization facade.

Visualization is now implemented in :mod:`Dispersive_XAS.web`.
"""

from .web.roi import PgSpec, show_roi

__all__ = ["show_roi", "PgSpec"]
