"""Loaders for Bluesky HDF5 scans."""

from __future__ import annotations

import h5py

__all__ = ["load_bluesky_h5"]


def load_bluesky_h5(filepath: str) -> dict:
    """Load a Bluesky HDF5 scan file."""
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
