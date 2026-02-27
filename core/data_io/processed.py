"""Loaders for native DXAS processed and raw HDF5 data."""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np

from .. import h5io

__all__ = ["raw_loading", "load_processed", "load_processed_scans", "load_mask_h5"]


def raw_loading(folder: str, detector: Optional[str] = None) -> np.ndarray:
    """Load all HDF5 files in a folder into a numpy array."""
    key_map = {"Ximea": "raw_data", None: "var1"}

    try:
        file_raw = h5io.h5read(f"{folder}/*.h5")
    except IOError:
        file_raw = [h5io.h5read(folder)]

    frames = []
    for file_data in file_raw:
        if detector == "Zyla":
            frame = np.array(
                file_data["entry_0000"]["measurement"]["andor-zyla"]["data"][0],
                dtype=float,
            )
        else:
            frame = np.array(file_data[key_map.get(detector, "var1")], dtype=float)
        frames.append(frame)

    return np.array(frames)


def load_processed(folder: str) -> dict:
    """Load data saved by :func:`Dispersive_XAS.preprocessing.pre_process`."""
    return {
        "data": h5io.h5read(os.path.join(folder, "00_data.h5"))["var1"],
        "flat": h5io.h5read(os.path.join(folder, "01_flat.h5"))["var1"],
        "transmission": h5io.h5read(os.path.join(folder, "02_transmission.h5"))["var1"],
        "mux": h5io.h5read(os.path.join(folder, "03_mux.h5"))["var1"],
    }


def load_processed_scans(folder: str) -> dict:
    """Load data saved by :func:`Dispersive_XAS.preprocessing.pre_process_scan`."""
    subfolders = ["00_data", "01_flat", "02_transmission", "03_mux"]
    result = {}
    for sub in subfolders:
        key = sub[3:]
        files = sorted(glob.glob(os.path.join(folder, sub, "*.h5")))
        result[key] = np.asarray([h5io.h5read(f)["var1"] for f in files])
    return result


def load_mask_h5(
    file_name: str = "mask",
    folder_name: str = "mask",
    as_bool: bool = True,
) -> np.ndarray:
    """Load a saved mask from HDF5 without mixing save logic."""
    path = os.path.join(folder_name, file_name)
    if not path.endswith(".h5"):
        path += ".h5"
    arr = np.asarray(h5io.h5read(path)["var1"])
    if as_bool:
        return arr.astype(bool)
    return arr
