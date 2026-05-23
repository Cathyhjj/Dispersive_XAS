# Dispersive_XAS

Dispersive_XAS is a Python package for dispersive X-ray absorption spectroscopy
(DXAS) data reduction. It covers the common APS-style workflow from raw
areaDetector HDF5 files to calibrated spectra, interactive ROI selection, batch
preview HTML files, and time-resolved transition diagnostics.

The installable package name is `dispersive-xas`; the Python import name remains
`Dispersive_XAS` for compatibility with existing notebooks and legacy analysis
scripts.

## What is included

- HDF5, NeXus, Bluesky, and legacy processed-data loaders.
- Raw detector preprocessing into transmission and absorption maps.
- Horizontal and tilted-band ROI tools, including notebook editors.
- Spectrum extraction, normalization, interpolation, peak finding, and edge
  point interpolation.
- Pixel-to-energy calibration from a reference foil and standard spectrum.
- Chunked calibration for large scans that cannot be loaded fully into memory.
- Interactive Plotly HTML previews and summary dashboards for batch workflows.
- Crystal geometry calculators used for DXAS beamline design checks.

## Installation

From this repository:

```bash
python -m pip install -e .
```

For notebook and interactive ROI widgets:

```bash
python -m pip install -e ".[notebook]"
```

For development and tests:

```bash
python -m pip install -e ".[dev,notebook]"
pytest
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

calibration = dxas.load_calibration_model("calibration.json")

result = dxas.apply_calibration_to_scan(
    data_path="scan.h5",
    flat_path="flat.h5",
    calibration=calibration,
    roi=roi,
)
```

For interactive tilted-band ROI selection in a notebook:

```python
editor = dxas.select_tilted_band_roi(mux_image, save_path="roi.json")
roi = editor.get_spec()
```

For a full large-quantity workflow:

```python
from pathlib import Path

import Dispersive_XAS as dxas

cfg = dxas.BatchAnalysisConfig(
    data_dir=Path("/path/to/data"),
    scan_file="20251011_1915_sample_001.h5",
    foil_file="20251011_1857_Cu_foil_002.h5",
    roi=roi,
    analysis_dirname="analysis_run",
)

result = dxas.run_large_quantity_analysis(cfg, make_previews=True)
print(result["summary_dashboard_html"])
```

## Documentation

- [Workflow guide](docs/WORKFLOWS.md): practical recipes for ROI selection,
  preprocessing, calibration, scan application, and large-batch analysis.
- [API reference](docs/API_REFERENCE.md): public functions/classes grouped by
  task and module.

## Data conventions

- Image stacks use shape `(frames, rows, columns)`.
- A single image uses shape `(rows, columns)`.
- Spectra use shape `(2, N)` where row 0 is the pixel/energy axis and row 1 is
  the intensity axis.
- ROI specs are dictionaries. The most common forms are:

```python
{"kind": "row_range", "row_start": 155, "row_stop": 235}

{
    "kind": "tilted_band",
    "center_row_at_col0": 190.0,
    "slope_per_col": 0.01,
    "half_width": 35.0,
}
```

## Package layout

- `Dispersive_XAS.core`: numerical loading, preprocessing, ROI, spectra,
  calibration, and crystal calculations.
- `Dispersive_XAS.web`: Plotly and ipywidgets helpers for interactive viewing.
- `Dispersive_XAS.batch`: high-level large-quantity analysis orchestration.
- `Dispersive_XAS.progress`: console and JSON progress reporting.

## Repository Notes

- Keep raw detector data (`.h5`, `.hdf5`, `.nxs`) outside git.
- Generated preview HTML, CSV, JSON, and analysis folders should stay untracked.
- The `example/` folder contains a workflow snapshot, not raw data.
