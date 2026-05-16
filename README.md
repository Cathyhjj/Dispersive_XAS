# Dispersive_XAS

Dispersive_XAS is a Python package for dispersive X-ray absorption spectroscopy
(DXAS) data reduction. It includes HDF5/NeXus loaders, detector preprocessing,
ROI tools, spectrum extraction, energy calibration, and large-batch preview and
analysis workflows.

The installable package name is `dispersive-xas`; the Python import name remains
`Dispersive_XAS` for compatibility with existing notebooks.

## Installation

From this repository:

```bash
python -m pip install -e .
```

For notebook and interactive ROI widgets:

```bash
python -m pip install -e ".[notebook]"
```

## Quick Start

```python
import Dispersive_XAS as dxas

roi = dxas.make_tilted_band_roi(
    shape=(512, 2048),
    left_center_row=190,
    right_center_row=210,
    half_width=35,
)

result = dxas.apply_calibration_to_scan(
    data_path="scan.h5",
    flat_path="flat.h5",
    calibration="calibration.json",
    roi=roi,
)
```

For interactive tilted-band ROI selection in a notebook:

```python
editor = dxas.select_tilted_band_roi(mux_image, save_path="roi.json")
roi = editor.get_spec()
```

## Repository Notes

- Keep raw detector data (`.h5`, `.hdf5`, `.nxs`) outside git.
- Generated preview HTML, CSV, JSON, and analysis folders should stay untracked.
- The `example/` folder contains a workflow snapshot, not raw data.
