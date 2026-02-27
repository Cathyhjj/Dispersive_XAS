"""Built-in reference standard XAS spectra for energy calibration.

Spectra were measured at BM14 of SPring-8 at 400 ms exposure.
Available standards: Ag, Ag2O, AgNO3, AgCl, Pd, PdO, Pd (ESRF), Pt-L3.
"""

import os

import numpy as np

from .spectrum import interpt_spec, norm_spec, spec_shaper

__all__ = ["standard_spec", "list_standards"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.dirname(__file__)

_SAMPLES: dict = {
    # Normalised (background-removed) spectra
    "Ag_norm": ("normalized_bk_rm", "Ag_K_standard.dat.nor", (0, 1)),
    "Ag2O_norm": ("normalized_bk_rm", "Ag2O_K_standard.dat.nor", (0, 1)),
    "AgNO3_norm": ("normalized_bk_rm", "AgNO3_k_standard.txt.nor", (0, 1)),
    "AgCl_norm": ("normalized_bk_rm", "Ag_chloride.xmu.nor", (0, 1)),
    "PdO_norm": ("normalized_bk_rm", "PdO_K_standard.dat.nor", (0, 1)),
    "Pd_norm": ("normalized_bk_rm", "Pd_K_standard.dat.nor", (0, 1)),
    "Pd_ESRF_norm": ("normalized_bk_rm", "esrf_BM23_Pd.txt.nor", (0, 1)),
    # Raw (µ·x) spectra
    "Ag": ("", "Ag_K_standard.dat", None),
    "Ag2O": ("", "Ag2O_K_standard.dat", None),
    "AgNO3": ("", "AgNO3_k_standard.txt", None),
    "Pd": ("", "Pd_K_standard.dat", None),
    "PdO": ("", "PdO_K_standard.dat", None),
    "Pt-L3": ("", "PtFoil_XAFS_Pos15empty.0001", None),
    # Flattened (post-edge normalised) spectra – columns (0, 3)
    "Ag_flat": ("normalized_bk_rm/flatened", "Ag_K_standard.dat.nor.flat", (0, 3)),
    "Ag2O_flat": ("normalized_bk_rm/flatened", "Ag2O_K_standard.dat.nor.flat", (0, 3)),
    "AgNO3_flat": ("normalized_bk_rm/flatened", "AgNO3_k_standard.txt.nor.flat", (0, 3)),
    "AgCl_flat": ("normalized_bk_rm/flatened", "Ag_chloride.xmu.nor.flat", (0, 3)),
    "PdO_flat": ("normalized_bk_rm/flatened", "PdO_K_standard.dat.nor.flat", (0, 3)),
    "Pd_flat": ("normalized_bk_rm/flatened", "Pd_K_standard.dat.nor.flat", (0, 3)),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_standards() -> list:
    """Return the names of all available built-in standards."""
    return sorted(_SAMPLES.keys())


def standard_spec(
    sample: str,
    norm: bool = True,
    intp: bool = False,
    pnts: int = 3000,
) -> np.ndarray:
    """Load a built-in reference standard XAS spectrum.

    Parameters
    ----------
    sample : str
        Sample key (see :func:`list_standards` for all available names).
        Append ``_norm`` for normalised spectra, ``_flat`` for flattened
        post-edge spectra.
    norm : bool
        If ``True``, min–max normalise the loaded spectrum (default: ``True``).
    intp : bool
        If ``True``, interpolate onto a uniform grid of *pnts* points.
    pnts : int
        Number of interpolation points (used when *intp* is ``True``).

    Returns
    -------
    ndarray, shape (2, N)

    Raises
    ------
    ValueError
        If *sample* is not found in the built-in registry.
    """
    if sample not in _SAMPLES:
        available = ", ".join(sorted(_SAMPLES.keys()))
        raise ValueError(
            f"Unknown standard '{sample}'.  Available keys: {available}"
        )

    subdir, filename, usecols = _SAMPLES[sample]
    path = os.path.join(_DATA_DIR, "standard_XAS", subdir, filename)
    print(f"Loading {path}")

    if usecols is not None:
        raw = np.loadtxt(path, usecols=usecols)
    else:
        raw = np.loadtxt(path)

    spectrum = spec_shaper(raw)  # ensure (2, N)

    if intp:
        spectrum = interpt_spec(spectrum, pnts=pnts)
    if norm:
        spectrum = norm_spec(spectrum)

    return spectrum
