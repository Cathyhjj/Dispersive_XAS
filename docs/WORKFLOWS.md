# Dispersive_XAS workflow guide

This guide shows the common paths through the package. The examples use
`import Dispersive_XAS as dxas`, which exposes the public API used by existing
notebooks.

## 1. Load detector data

NeXus / areaDetector files are expected to contain `/entry/data/data` with
shape `(frames, rows, columns)`.

```python
import Dispersive_XAS as dxas

scan = dxas.load_nexus_entry("sample_scan.h5")
flat = dxas.load_nexus_entry("flatfield.h5")

data_stack = scan["data"]
flat_stack = flat["data"]
```

For legacy processed data saved by this package:

```python
processed = dxas.load_processed("20260227_preproc")
mux = processed["mux"]
```

## 2. Build or select an ROI

Use a horizontal row range when the beam is level:

```python
roi = dxas.normalize_roi_spec(
    shape=(512, 2048),
    row_range=(155, 235),
)
```

Use a tilted band when the beam footprint drifts across detector columns:

```python
roi = dxas.make_tilted_band_roi(
    shape=(512, 2048),
    left_center_row=190,
    right_center_row=210,
    half_width=35,
)
```

In a notebook, create a tilted-band ROI interactively:

```python
editor = dxas.select_tilted_band_roi(mux_image, save_path="roi.json")
roi = editor.get_spec()
```

Save and reuse ROI specs:

```python
dxas.save_roi_json("roi.json", roi, metadata={"scan": "sample_scan.h5"})
roi = dxas.load_roi_json("roi.json")
```

## 3. Preprocess raw images

`pre_process` converts one sample image and one flat image into transmission and
absorption products. Pass averaged frames if you want a representative static
image.

```python
sample_avg = data_stack.mean(axis=0)
flat_avg = flat_stack.mean(axis=0)

out = dxas.pre_process(
    data=sample_avg,
    flat=flat_avg,
    denoise_size=3,
    savedata=False,
)

mux = out["mux"]
```

For stacks that are already dark-corrected:

```python
out = dxas.pre_process_scan(
    data_darkcorr=data_stack,
    flat_darkcorr=flat_stack,
    denoise_size=3,
    savedata=False,
)
```

## 4. Extract and normalize spectra

For a single image or stack, `roi_weighted_column_mean` returns one spectrum per
frame using the selected ROI.

```python
spectra = dxas.roi_weighted_column_mean(out["mux"], roi=roi)
```

The lower-level spectrum helpers operate on arrays with shape `(2, N)`.

```python
spec = dxas.spectrum_generate(mux, show=False)
spec_norm = dxas.norm_spec(spec, x0=50, x1=130)
spec_interp = dxas.interpt_spec(spec_norm, pnts=1000)
```

## 5. Fit an energy calibration

Use a reference foil scan and a standard spectrum to fit a polynomial mapping
from detector pixel to energy. The standard can be a built-in reference from
`standard_spec(...)` or an external two-column text file.

```python
import numpy as np

standard = dxas.spec_shaper(np.loadtxt("CuFoil_new.0001.nor", usecols=(0, 1)))

fit, meta = dxas.calibrate_from_reference_foil(
    foil_path="cu_foil.h5",
    flat_path="cu_foil_flat.h5",
    standard_spec=standard,
    roi=roi,
    norm_range_pixels=(50, 130),
    poly_order=2,
    show=False,
)

model_path = dxas.save_calibration_model(
    "calibration_cu_foil.json",
    fit,
    metadata=meta,
)
```

To see built-in standards:

```python
print(dxas.list_standards())
standard = dxas.standard_spec("Pd_norm", norm=True)
```

## 6. Apply calibration to a large scan

`apply_calibration_to_scan` processes frames in chunks, applies flat-field
correction, extracts ROI spectra, normalizes them, and writes a compact HDF5
file when `output_h5` is supplied.

```python
model = dxas.load_calibration_model("calibration_cu_foil.json")

result = dxas.apply_calibration_to_scan(
    data_path="sample_scan.h5",
    flat_path="sample_flat.h5",
    calibration=model,
    roi=roi,
    norm_range_pixels=(50, 130),
    chunk_size=500,
    output_h5="calibrated_sample_scan.h5",
)
```

The output HDF5 file contains:

- `energy`: calibrated energy axis.
- `pixel`: original detector pixel axis.
- `spectra`: normalized spectra with shape `(frames, columns)`.

## 7. Make interactive preview HTML

Generate a full-scan HTML preview before committing to the full calibrated
analysis:

```python
dxas.preview_spectra_html(
    data_path="sample_scan.h5",
    flat_path="sample_flat.h5",
    roi=roi,
    chunk_size=1000,
    median_size=3,
    display_inline=False,
)
```

For very large scans, `plot_spectra_in_chunks` writes one preview per chunk.

```python
dxas.plot_spectra_in_chunks(
    data_path="sample_scan.h5",
    flat_path="sample_flat.h5",
    roi=roi,
    chunk_size=1000,
    max_line_traces=200,
)
```

## 8. Run the large-quantity pipeline

The high-level pipeline resolves scan/flat/foil files, creates previews,
calibrates from a foil, applies the calibration to the scan, calculates
transition metrics, and writes an HTML dashboard.

```python
from pathlib import Path

cfg = dxas.BatchAnalysisConfig(
    data_dir=Path("/path/to/data"),
    scan_file="sample_start_001.h5",
    reverse_scan_file="sample_stop_001.h5",
    use_reverse_scan=True,
    foil_file="Cu_foil_002.h5",
    cu_standard=Path("/path/to/CuFoil_new.0001.nor"),
    roi=roi,
    analysis_dirname="analysis_20260227",
    overwrite=False,
)

result = dxas.run_large_quantity_analysis(cfg, make_previews=True)
print(result["summary_dashboard_html"])
```

Useful outputs include:

- `00_analysis_summary_all.html`: dashboard with the main reports embedded.
- `analysis_manifest.json`: machine-readable run manifest.
- `*_timeseries.csv`: frame-by-frame ratio, alpha, and residual traces.
- `*_difference_map.html`: interactive difference spectra heatmap.
- `foil_calibration_report.html`: calibration fit and residual diagnostics.

## 9. Progress reporting

Long jobs can emit both console text and a JSON progress file.

```python
progress = dxas.BatchProgressReporter(
    json_path="analysis_progress.json",
    enabled=True,
)

dxas.run_large_quantity_analysis(cfg, progress=progress)
```

The JSON payload includes current stage, status, percent complete, elapsed time,
ETA when available, and workflow-specific metadata.
