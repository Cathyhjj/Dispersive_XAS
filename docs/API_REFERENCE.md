# Dispersive_XAS API reference

This reference groups the public API by task. The package re-exports the most
common functions from `Dispersive_XAS.__init__`, so the examples use:

```python
import Dispersive_XAS as dxas
```

## Core data conventions

- Image stack: `(frames, rows, columns)`.
- Single image: `(rows, columns)`.
- Spectrum: `(2, N)`, with axis in row 0 and intensity in row 1.
- Calibrated spectra HDF5: `energy`, `pixel`, and `spectra` datasets.
- ROI specs: dictionaries with `kind="row_range"` or `kind="tilted_band"`.

## Loading and saving data

### `load_nexus_entry(filepath)`

Loads an areaDetector/NeXus HDF5 file with `/entry/data/data`. Returns a dict
containing `data`, `meta`, `nd_attributes`, `timestamps`, and `frame_ids`.

### `load_bluesky_h5(filepath)`

Loads a Bluesky HDF5 scan root and returns scan metadata plus the datasets under
the root `data` group.

### `raw_loading(folder, detector=None)`

Loads legacy raw HDF5 files from a folder. Use `detector="Zyla"` for the
Andor/Zyla nested layout, `detector="Ximea"` for the `raw_data` key, and
`None` for the default `var1` key.

### `load_processed(folder)` and `load_processed_scans(folder)`

Load outputs written by `pre_process` and `pre_process_scan`.

### `saveh5(data, file_name="temp", folder_name="_save_h5", date=True)`

Save a 2-D array through the legacy HDF5 writer. The data is column-reversed to
match historical notebook orientation.

### `save_mask_h5(mask, file_name="mask", folder_name="mask")`

Save a mask array to HDF5 and return the output path.

### `load_mask_h5(file_name="mask", folder_name="mask", as_bool=True)`

Load a mask saved by `save_mask_h5`.

## Preprocessing and image correction

### `pre_process(data, flat, dark=None, denoise_size=3, savedata=True, prefix="preproc")`

Processes one sample image and one flat image. It optionally subtracts a dark
field, median filters both images, clips non-positive values, computes
transmission, and returns absorption `mux = -log(data / flat)`.

### `pre_process_scan(data_darkcorr, flat_darkcorr, denoise_size=3, savedata=True, prefix="preproc")`

Processes pre-dark-corrected stacks frame by frame and returns the same keys as
`pre_process`.

### `ximea_correction(data, crop_slice=None, show=False)`

Estimates a column-wise Ximea detector offset from a crop region and returns
both offset-subtracted and offset-divided corrected images.

### Image transforms

`rotate_image`, `shift_image`, `flip_image_horizontal`, `flip_image_vertical`,
`invert_image`, `log_image`, `threshold_image_min`, `threshold_image_max`,
`median_filter_image`, and `gaussian_filter_image` provide simple numpy/scipy
operations used by older notebooks.

### Registration helpers

- `register_thresholding(imgs, binary_lower_lm, binary_upper_lm, footprint=None, show=True)`
- `find_shifts(register_binary)`
- `stitch_scans(imgs, masks, show=True)`

These helpers threshold image stacks, estimate phase-correlation shifts, and
average registered scans under masks.

## ROI utilities

### `make_tilted_band_roi(shape, left_center_row, right_center_row, half_width)`

Builds a normalized tilted-band ROI from the center row at the left and right
detector edges.

### `normalize_roi_spec(shape, row_range=None, roi=None)`

Validates and normalizes row-range and tilted-band ROI dictionaries. Adds
`row_bounds` for downstream chunked processing.

### `build_roi_mask(shape, row_range=None, roi=None)`

Returns a boolean mask for the requested ROI.

### `prepare_roi_weights(shape, row_range=None, roi=None, dtype=np.float32)`

Returns `(roi_spec, row_bounds, row_weights, col_weight_sum)` for efficient
column-wise weighted averaging.

### `roi_weighted_column_mean(image, row_range=None, roi=None)`

Extracts a spectrum from a 2-D image or a spectrum per frame from a 3-D stack.

### `fit_tilted_band_roi(image, ...)`

Fits a tilted-band ROI automatically from a representative absorption image by
thresholding and smoothing the beam footprint column by column.

### `infer_tilted_band_roi_from_paths(data_path, flat_path, ...)`

Builds a representative absorption image from HDF5 sample/flat paths and fits a
tilted-band ROI.

### `save_roi_json(path, roi, metadata=None)` and `load_roi_json(path)`

Persist ROI specs for reuse across scans.

## Interactive ROI and plotting

### `select_tilted_band_roi(img, initial_roi=None, title="", save_path=None, show=True)`

Creates a `TiltedBandROIEditor`, displays it in a notebook when `show=True`,
and returns the editor. Call `editor.get_spec()` to retrieve the ROI.

### `TiltedBandROIEditor`

Notebook editor for tilted-band ROI center, slope, and width. Key methods:

- `get_spec()`: return the current normalized ROI dictionary.
- `save(path=None, metadata=None)`: save the current ROI JSON.
- `launch(show=True)`: create/display the widget UI.

### `select_rect_roi(img, name="main", show_selector=True)` and `PgSpec`

Notebook tools for rotated rectangular ROI selection. `PgSpec` exposes
`getMask`, `getMaskAndAngle`, `getAllMasks`, and `getArrayRegion` for legacy
ROI workflows.

### Plotting functions

`show_line`, `show_lines`, `show_image`, `show_image_stack`, and
`show_mask_overlay` return Plotly figures and display them when `show=True`.

## Spectrum operations

### `spec_shaper(spectrum)`

Ensures a spectrum has shape `(2, N)`.

### `spec_wrapper(energy, intensity, output=(2, -1))`

Stacks energy/pixel axis and intensity into a spectrum array.

### `spectrum_generate(crop_mux, mode="average", title="Spectrum", show=True, **kwargs)`

Collapses a 2-D image into one 1-D spectrum by summing or averaging rows.

### `norm_spec(spectrum, x0=None, x1=None, show=False, robust_percentile=None, **kwargs)`

Min-max normalizes intensity. `x0`/`x1` can be single bounds or parallel lists
of multiple normalization windows.

### `interpt_spec(spec, x_min=None, x_max=None, pnts=3000)`

Interpolates a spectrum onto a uniform axis.

### `spec_cropping(spec, crop_E1, crop_E2, show=False)`

Crops a spectrum to an axis range.

### `peak_finder(spec, spec_min=None, spec_max=None, peak_n=None, prominence=0.01, filtering=False, show=True, include_troughs=True, **kwargs)`

Finds prominent peaks and optionally troughs. Returns detector indices and axis
positions for selected features.

### `find_edge_jump(spec, show=True, prominence=0.005)`

Finds the strongest positive derivative feature as an edge-jump index.

### `find_edge_pnts(spec, y_pnts, edge_min=None, edge_max=None, show=True)`

Interpolates axis positions at requested intensity levels.

### `intensity_at_energy(energy, spec_1d, E_eV)`

Interpolates a 1-D spectrum at a specific energy.

### `atten_slope_corr(element, data_x, E_threshold=0, show=True)`

Uses `xraylib` cross-sections to build an attenuation-slope correction curve.

## Calibration and batch processing

### `calibrate_regression(train_spec, target_standard, peaks_train, peaks_target, order=1, sample_spec=None, show=True)`

Fits a polynomial from selected experimental feature indices to selected
standard feature indices.

### `EDXAS_Calibrate`

Polynomial pixel-to-energy calibration class. Important attributes include
`coef`, `order`, `rmse`, `train`, `target`, and `new_x`. Use
`sample_spec(sample_spec)` to apply the calibration to a spectrum.

### `calibrate_from_reference_foil(foil_path, flat_path, standard_spec, ...)`

Runs the full foil calibration workflow: preprocess foil/flat images, extract
ROI spectrum, normalize, detect edge/peaks, match them to a reference standard,
and return `(fit, metadata)`.

### `save_calibration_model(file_path, calibration, metadata=None)` and `load_calibration_model(file_path)`

Save/load JSON calibration models for reuse.

### `apply_calibration_to_scan(data_path, flat_path, calibration, ..., output_h5=None, progress=None)`

Processes a large scan in chunks. When `output_h5` is set, writes calibrated
energy and normalized spectra to HDF5; otherwise returns spectra in memory.

### `BatchProgressReporter` and `emit_progress`

Emit human-readable progress lines and optionally update a JSON status file.

## Preview and full analysis workflows

### `preview_spectra_html(data_path, flat_path, ..., display_inline=True)`

Computes spectra for a full scan in chunks and writes one interactive HTML file
with heatmap, averaged lines, and individual-frame lines.

### `plot_spectra_in_chunks(data_path, flat_path, ..., chunk_size=1000)`

Writes one interactive preview HTML per chunk. This is useful for very large
scans where a single all-frame HTML file would be too heavy.

### `BatchAnalysisConfig`

Dataclass that holds file names, ROI settings, smoothing parameters, peak
search windows, chunk sizes, output naming, and progress settings for the full
large-quantity workflow.

### `run_large_quantity_analysis(cfg, make_previews=True, progress=None)`

Runs the full workflow: resolve scan/flat/foil inputs, optionally write preview
HTML, fit foil calibration, apply calibration to forward/reverse scans,
calculate transition metrics, write Plotly reports, and emit a summary
dashboard plus manifest.

### `run_analysis(cfg, make_previews=True, progress=None)`

Backward-compatible alias for `run_large_quantity_analysis`.

## Standards and utilities

### `list_standards()` and `standard_spec(sample, norm=True, intp=False, pnts=3000)`

List and load built-in reference spectra distributed with the package.

### General utilities

`date_today`, `time_now`, `timestamp_convert`, `color_gradient`,
`change_font_size`, `binning`, `make_gif`, and `make_video` support legacy
notebook workflows.

## Crystal geometry calculators

`Crystal`, `Laue_Crystal`, and `Bragg_Crystal` estimate Bragg angle, energy
spread, footprint, focus distance, beam size, and resolution for DXAS crystal
configurations. These classes preserve historical print-summary behavior via
their `printlst` dictionary and `type_writer()` method.
