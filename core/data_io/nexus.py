"""Loaders for NeXus / areaDetector HDF5 files."""

from __future__ import annotations

import h5py
import numpy as np

__all__ = ["load_nexus_entry"]


def load_nexus_entry(filepath: str) -> dict:
    """Load a NeXus-style HDF5 file from an areaDetector source."""
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

        arr = data_grp["data"][()]
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
