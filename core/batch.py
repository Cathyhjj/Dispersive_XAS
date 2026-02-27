"""Core batch helpers (no plotting/UI)."""

from __future__ import annotations

import numpy as np

__all__ = ["norm_spec_preview"]

_DEFAULT_FACTOR = 200.0


def norm_spec_preview(
    spec: np.ndarray,
    x1: int,
    x2: int,
    factor: float = _DEFAULT_FACTOR,
) -> np.ndarray:
    """Normalize a 1-D spectrum by its value range within a pixel window."""
    spec = np.nan_to_num(
        spec,
        nan=float(np.nanmean(spec)),
        posinf=float(np.nanmax(spec)),
        neginf=float(np.nanmin(spec)),
    )
    ref = spec[x1:x2]
    smin, smax = float(np.min(ref)), float(np.max(ref))
    return ((spec - smin) / (smax - smin + 1e-12)) * factor
