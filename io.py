"""Data loading and saving utilities for DXAS.

Supports:
* Custom HDF5 format written by :mod:`~Dispersive_XAS.h5io`.
* NeXus-style HDF5 files from areaDetector (``/entry/data/data``).
* Bluesky HDF5 scan files.
"""

import glob
import os
from typing import Optional

import h5py
import numpy as np

from . import h5io
from .utils import date_today

__all__ = [
    "raw_loading",
    "load_processed",
    "load_processed_scans",
    "saveh5",
    "load_nexus_entry",
    "load_bluesky_h5",
]


# ---------------------------------------------------------------------------
# Custom h5io format
# ---------------------------------------------------------------------------


def raw_loading(folder: str, detector: Optional[str] = None) -> np.ndarray:
    """Load all HDF5 files in a folder into a numpy array.

    Parameters
    ----------
    folder : str
        Path to a directory containing ``*.h5`` files (glob pattern accepted).
        A single file path without wildcards is also accepted.
    detector : {'Ximea', 'Zyla', None}
        Detector type, which determines the HDF5 dataset key:

        * ``'Ximea'`` → key ``raw_data``
        * ``'Zyla'``  → Andor Zyla NeXus path
        * ``None``    → key ``var1`` (default)

    Returns
    -------
    ndarray, shape (N, H, W)
        Stack of images.
    """
    key_map = {"Ximea": "raw_data", None: "var1"}

    try:
        file_raw = h5io.h5read("%s/*.h5" % folder)
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
    """Load data saved by :func:`~Dispersive_XAS.preprocessing.pre_process`.

    Parameters
    ----------
    folder : str
        Directory containing ``00_data.h5``, ``01_flat.h5``,
        ``02_transmission.h5``, and ``03_mux.h5``.

    Returns
    -------
    dict
        ``{'data': ndarray, 'flat': ndarray, 'transmission': ndarray, 'mux': ndarray}``
    """
    return {
        "data": h5io.h5read(os.path.join(folder, "00_data.h5"))["var1"],
        "flat": h5io.h5read(os.path.join(folder, "01_flat.h5"))["var1"],
        "transmission": h5io.h5read(os.path.join(folder, "02_transmission.h5"))["var1"],
        "mux": h5io.h5read(os.path.join(folder, "03_mux.h5"))["var1"],
    }


def load_processed_scans(folder: str) -> dict:
    """Load data saved by :func:`~Dispersive_XAS.preprocessing.pre_process_scan`.

    Parameters
    ----------
    folder : str
        Parent directory containing ``00_data/``, ``01_flat/``,
        ``02_transmission/``, and ``03_mux/`` sub-directories.

    Returns
    -------
    dict
        ``{'data': ndarray, 'flat': ndarray, 'transmission': ndarray, 'mux': ndarray}``
        Each value is stacked over all scans.
    """
    subfolders = ["00_data", "01_flat", "02_transmission", "03_mux"]
    result = {}
    for sub in subfolders:
        key = sub[3:]  # strip leading '00_' etc.
        files = sorted(glob.glob(os.path.join(folder, sub, "*.h5")))
        result[key] = np.asarray([h5io.h5read(f)["var1"] for f in files])
    return result


def saveh5(
    data: np.ndarray,
    file_name: str = "temp",
    folder_name: str = "_save_h5",
    date: bool = True,
) -> None:
    """Save a numpy array to an HDF5 file.

    Parameters
    ----------
    data : ndarray
    file_name : str
        Base filename without extension.
    folder_name : str
        Sub-directory to create and save into.
    date : bool
        If ``True``, prepend today's date (YYYYMMDD) to the filename.
    """
    prefix = date_today() + "_" if date else ""
    directory = os.path.join(os.getcwd(), folder_name)
    os.makedirs(directory, exist_ok=True)
    h5io.h5write(
        os.path.join(directory, f"{prefix}{file_name}.h5"),
        var1=data[:, ::-1],
    )


# ---------------------------------------------------------------------------
# NeXus / areaDetector HDF5
# ---------------------------------------------------------------------------


def load_nexus_entry(filepath: str) -> dict:
    """Load a NeXus-style HDF5 file from an areaDetector source.

    Expected layout::

        /entry  (NXentry)
          /data  (NXdata)
            data : image stack, shape (N, H, W)
          /instrument
            /NDAttributes
              NDArrayEpicsTSSec, NDArrayEpicsTSnSec,
              NDArrayTimeStamp, NDArrayUniqueId
            /performance
              timestamp  (may be empty)

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.

    Returns
    -------
    dict with keys:

    ``meta``
        Dictionary of metadata (shape, dtype, NX class strings).
    ``data``
        ndarray of shape (N, H, W).
    ``nd_attributes``
        Dictionary of NDAttribute arrays.
    ``performance``
        ``{'timestamp': ndarray or None}``.
    ``timestamps``
        ``{'epics_sec', 'epics_nsec', 'epics_ts', 'nd_time'}``
        where ``epics_ts = epics_sec + epics_nsec * 1e-9``.
    ``frame_ids``
        ``NDArrayUniqueId`` array or ``None``.
    """
    out = {
        "meta": {},
        "data": None,
        "nd_attributes": {},
        "performance": {"timestamp": None},
        "timestamps": {
            "epics_sec": None,
            "epics_nsec": None,
            "epics_ts": None,
            "nd_time": None,
        },
        "frame_ids": None,
    }

    with h5py.File(filepath, "r") as f:
        entry = f["/entry"]
        data_grp = entry.get("data")
        if data_grp is None or "data" not in data_grp:
            raise KeyError("Expected dataset '/entry/data/data' not found.")

        arr = data_grp["data"][()]  # (N, H, W)
        out["data"] = arr

        nx_entry = entry.attrs.get("NX_class", b"")
        nx_data = data_grp.attrs.get("NX_class", b"")
        if isinstance(nx_entry, bytes):
            nx_entry = nx_entry.decode()
        if isinstance(nx_data, bytes):
            nx_data = nx_data.decode()

        n, h, w = arr.shape
        out["meta"].update(
            {
                "nx_entry_class": nx_entry,
                "nx_data_class": nx_data,
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "n_frames": int(n),
                "height": int(h),
                "width": int(w),
            }
        )

        nd_grp = entry.get("instrument/NDAttributes")
        if nd_grp is not None:
            for name, obj in nd_grp.items():
                if isinstance(obj, h5py.Dataset):
                    out["nd_attributes"][name] = obj[()]

            epics_sec = out["nd_attributes"].get("NDArrayEpicsTSSec")
            epics_nsec = out["nd_attributes"].get("NDArrayEpicsTSnSec")
            nd_time = out["nd_attributes"].get("NDArrayTimeStamp")
            frame_ids = out["nd_attributes"].get("NDArrayUniqueId")

            out["timestamps"]["epics_sec"] = epics_sec
            out["timestamps"]["epics_nsec"] = epics_nsec
            out["timestamps"]["nd_time"] = nd_time
            out["frame_ids"] = frame_ids

            if epics_sec is not None and epics_nsec is not None:
                out["timestamps"]["epics_ts"] = (
                    epics_sec.astype(np.float64)
                    + epics_nsec.astype(np.float64) * 1e-9
                )

        perf_grp = entry.get("instrument/performance")
        if perf_grp is not None and "timestamp" in perf_grp:
            ts = perf_grp["timestamp"][()]
            out["performance"]["timestamp"] = ts if ts.size > 0 else None

    return out


# ---------------------------------------------------------------------------
# Bluesky HDF5
# ---------------------------------------------------------------------------


def load_bluesky_h5(filepath: str) -> dict:
    """Load a Bluesky HDF5 scan file.

    Expected layout::

        /<uuid>/
          duration, entry_identifier, plan_name, sample_name,
          start_time, stop_time
          data/
            <detector arrays …>

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dict with keys:

    ``duration``, ``entry_identifier``, ``plan_name``, ``sample_name``,
    ``start_time``, ``stop_time``
        Top-level scan metadata.
    ``data``
        Dictionary of arrays from the ``data`` group.
    """
    data: dict = {}
    with h5py.File(filepath, "r") as f:
        root = f[list(f.keys())[0]]

        for key in (
            "duration",
            "entry_identifier",
            "plan_name",
            "sample_name",
            "start_time",
            "stop_time",
        ):
            val = root[key][()]
            if isinstance(val, bytes):
                val = val.decode()
            data[key] = val

        data["data"] = {name: root["data"][name][()] for name in root["data"].keys()}

    return data
