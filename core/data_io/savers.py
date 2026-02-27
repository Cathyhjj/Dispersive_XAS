"""Save helpers for DXAS data products."""

from __future__ import annotations

import os

import numpy as np

from .. import h5io
from ..utils import date_today

__all__ = ["saveh5", "save_mask_h5"]


def saveh5(
    data: np.ndarray,
    file_name: str = "temp",
    folder_name: str = "_save_h5",
    date: bool = True,
) -> None:
    """Save a numpy array to an HDF5 file."""
    prefix = date_today() + "_" if date else ""
    directory = os.path.join(os.getcwd(), folder_name)
    os.makedirs(directory, exist_ok=True)
    h5io.h5write(
        os.path.join(directory, f"{prefix}{file_name}.h5"),
        var1=data[:, ::-1],
    )


def save_mask_h5(
    mask: np.ndarray,
    file_name: str = "mask",
    folder_name: str = "mask",
) -> str:
    """Save a mask to HDF5 without mixing read logic.

    Returns
    -------
    str
        Absolute path to the saved ``.h5`` file.
    """
    directory = os.path.join(os.getcwd(), folder_name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{file_name}.h5")
    h5io.h5write(path, var1=np.asarray(mask))
    return path
