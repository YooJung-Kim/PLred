# PLred-dev mode 1 Pipeline: Functions by Step

Five-step pipeline for spatial response-map construction and image reconstruction.
Steps 1–4 are complete and tested; Step 5 (image reconstruction) is incomplete / not yet integrated into the CLI.

## Data flow

```
fastcam FITS + timestamps
       │
  Step 1 (sort)        → step1_fastcam.h5
       │
  Step 2 (ingest)      → step2_alldata.h5
       │
  Step 3 (average)     → step3_map.h5        [ROI viewer cache built first]
       │
  Step 4 (specextract) → step4_couplingmap.fits
       │
  Step 5 (imgrecon)    → polymodel.fits → reconstructed images  [incomplete]
```

**Tutorial notebooks:** `PLred/tutorials/tutorial_step{1–5}_*.ipynb`  
**CLI entry point:** `pipeline.run_mode1(configname, steps=[N])`  
**CLI step numbers differ from tutorial numbers** (see note in each step).

---

## Step 1 — Timestamp matching (`plred-sort`)

**Tutorial:** `tutorial_step1_sort.ipynb` | **CLI:** `steps=[1]`  
**Output:** `step1_fastcam.h5` — averaged PSFcam frames matched to each PLcam exposure

| File | Function | Role |
|------|----------|------|
| `sort.py` | `script_match_timestamps(configname)` | **Main entry point.** Reads `.ini`, matches fastcam/slowcam timestamps, writes output H5. |
| `sort.py` | `_load_timestamp_file(filepath)` | Loads `.txt` timestamp files for fast- or slow-cam. |
| `sort.py` | `compute_frame_durations(fastcam_timestamp, fastcam_fileinds)` | Computes per-frame exposure durations from timestamps. |
| `sort.py` | `_build_matching_dict(fastcam_timestamp, slowcam_timestamp, ...)` | Builds the frame-index mapping between fast and slow cameras. |
| `sort.py` | `plot_matching_diagnostics(h5_path)` | Plots weight/timestamp diagnostics from the output H5. |
| `_sort_base.py` | `find_data_between(datadir, obs_start, obs_end, ...)` | Finds FITS files within the observation time window. |
| `_sort_base.py` | `validate_timestamp_matching(timestamps1, timestamps2)` | Sanity-checks that matched timestamps agree within tolerance. |
| `scripts/_diagnostics.py` | `plot_step1(fastcam_h5)` | CLI diagnostic plot: nstacks and centroid coverage. |
| `scripts/_diagnostics.py` | `plot_pre_ingest_check(configname)` | CLI diagnostic: first PLcam frame preview. |
| `scripts/_diagnostics.py` | `get_plots_dir(configname)` / `save_diagnostic(fig, dir, name)` | Resolve plot output directory and save figure. |
| `pipeline.py` | `_step1(configname)` | CLI wrapper: calls `script_match_timestamps` then saves diagnostics. |

---

## Step 2 — Build giant H5 (`plred-ingest`)

**Tutorial:** `tutorial_step2_ingest.ipynb` | **CLI:** `steps=[2]`  
**Output:** `step2_alldata.h5` — combined H5 with all PLcam frames, PSFcam frames, centroids, and metadata

| File | Function | Role |
|------|----------|------|
| `ingest.py` | `ingest_to_h5(step1_h5, outpath, plcam_dark, plcam_data_dir, ...)` | **Main entry point (tutorial).** Reads Step 1 H5 + PLcam FITS, writes giant H5. |
| `ingest.py` | `ingest_from_config_unified(configname)` | **CLI entry point.** Parses `.ini` and dispatches to `_ingest_from_new_config`. |
| `ingest.py` | `_ingest_from_new_config(cfg)` | Resolves config fields and calls `ingest_to_h5`. |
| `ingest.py` | `_resolve_plcam_files(slowcam_ts_files, plcam_data_dir)` | Finds PLcam FITS files corresponding to timestamp files. |
| `ingest.py` | `_probe_plcam_shape(plcam_files, plcam_roi, dark_frame)` | Reads one PLcam file to determine cropped frame shape. |
| `ingest.py` | `_probe_plcam_shape_from_array(plcam_array, plcam_roi)` | Same as above when PLcam data is already in memory. |
| `ingest.py` | `_load_psfcam_frames_from_fits(psfcam_files, matched_sc_inds, ...)` | Loads and averages PSFcam frames for each PLcam exposure. |
| `ingest.py` | `_write_plcam_frames(outpath, plcam_files, matched_sc_inds, ...)` | Writes PLcam frames (dark-subtracted, cropped) to H5. |
| `ingest.py` | `_write_plcam_frames_from_array(outpath, plcam_array, ...)` | Same as above when PLcam data is provided as an array. |
| `ingest.py` | `_load_spectra(plcam_spectra, N)` | Loads pre-extracted spectra if provided instead of raw frames. |
| `ingest.py` | `ingest_from_config(configname)` | ⚠️ **Legacy** — old config format, superseded by `ingest_from_config_unified`. |
| `scripts/_diagnostics.py` | `plot_step2(alldata_h5)` | CLI diagnostic plot: centroid scatter and PLcam frame preview. |
| `pipeline.py` | `_step2(configname)` | CLI wrapper: calls `ingest_from_config_unified` then saves diagnostics. |

---

## Step 3 — ROI viewer cache + spatial averaging (`plred-roi` / `plred-average`)

**Tutorial:** `tutorial_step3_average.ipynb` | **CLI:** `steps=[3]` (ROI cache) and `steps=[4]` (averaging)  
**Output:** `step3_map.h5` — per-grid-bin averaged PLcam and PSFcam frames

### Step 3a — ROI viewer cache (CLI `steps=[3]`)

| File | Function | Role |
|------|----------|------|
| `average.py` | `build_ROI_access(giant_h5, roi, h5_path, ...)` | **Main entry point (tutorial).** Writes a downsampled ROI cache for the interactive HTML viewer. |
| `average.py` | `build_ROI_access_from_config(configname)` | **CLI entry point.** Parses `.ini` and calls `build_ROI_access`. |
| `average.py` | `_load_giant_meta(giant_h5)` | Reads metadata (centroids, peaks, timestamps) from the giant H5. |
| `h5_products.py` | `create_roi_access_h5(...)` | Creates a new ROI-access H5 file with the initial data. |
| `h5_products.py` | `append_roi_access_h5(...)` | Appends additional frames to an existing ROI-access H5 file. |
| `scripts/_diagnostics.py` | `plot_step3(alldata_h5)` | CLI diagnostic: centroid heatmap. |
| `pipeline.py` | `_step3(configname, pause=True)` | CLI wrapper: calls `build_ROI_access_from_config`, saves diagnostic, optionally waits for user to inspect viewer and fill config. |

### Step 3b — Spatial averaging (CLI `steps=[4]`)

| File | Function | Role |
|------|----------|------|
| `average.py` | `average_to_h5(giant_h5, outpath, map_n, map_width, xc, yc, ...)` | **Main entry point (tutorial).** Bins frames into a spatial grid and averages. |
| `average.py` | `average_to_h5_from_config(configname)` | **CLI entry point.** Parses `.ini` and calls `average_to_h5`. |
| `average.py` | `_build_grid(centroids, map_n, map_width, xc, yc, pix2mas)` | Defines the spatial grid bins in pixel and mas coordinates. |
| `average.py` | `_apply_filter(peaks, timestamps, maxpix_min, maxpix_max, ...)` | Masks frames outside peak brightness or time range. |
| `average.py` | `_average_one(...)` | Averages frames in a single grid bin (no chunking). |
| `average.py` | `_average_chunked(...)` | Averages frames in large bins in memory-efficient chunks. |
| `average.py` | `_write_averaged_h5(...)` | Writes the full averaged map to the output H5. |
| `average.py` | `explore_grid(giant_h5, map_n, map_width, ...)` | Returns per-bin frame counts and average PSF/PLcam images (used in tutorial for grid inspection). |
| `average.py` | `pixel_map(averaged_h5, py, px)` | Returns the spatial response map at one PLcam pixel (tutorial inspection). |
| `_sort_base.py` | `bin_by_centroids_from_indices(psfcamframes, centroids, xbins, ybins)` | Assigns each frame to its grid bin by centroid position. |
| `_sort_base.py` | `compute_weighted_frame_binning(fast_timestamps, slow_timestamps)` | Computes time-weighted contribution of fastcam frames to each slowcam exposure. |
| `pipeline.py` | `_step4(configname)` | CLI wrapper: calls `average_to_h5_from_config`. |

---

## Step 4 — Spectral extraction (`plred-extract`)

**Tutorial:** `tutorial_step4_specextract.ipynb` | **CLI:** `steps=[5]`  
**Output:** `step4_couplingmap_trace.fits` and/or `step4_couplingmap_optimal.fits`

### Building an extractor (run once per instrument configuration)

| File | Function | Role |
|------|----------|------|
| `specextract.py` | `make_trace_file(fits_path, nfib, dark_path, xmin, xmax, ...)` | Detects fiber peaks and traces in a calibration frame, writes a `.npz` trace file. |
| `specextract.py` | `find_peaks(image, nfib, thres, min_dist, ...)` | Finds fiber peak positions in one detector column. |
| `specextract.py` | `find_traces(image, nfib, trace_width, poly_deg, ...)` | Traces fiber positions across detector columns via polynomial fits. |
| `specextract.py` | `make_trace_extractor(traces_or_model_file, boxsize, dark, ...)` | **Tutorial entry point.** Builds a simple box-sum extractor from a trace array or file. |
| `specextract.py` | `make_simple_extractor(ylocs, width, dark)` | Builds a box extractor from fixed row positions (no wavelength tracking). |
| `specextract.py` | `load_spectrum_model(model_file)` | Loads a pre-built LSF/optimal-extraction model from a `.npz` file. |
| `specextract.py` | `make_FIRSTPL_extractor(model_file, plcam_roi, dark, ...)` | Builds an optimal extractor using a FIRSTPL LSF model matrix. |
| `specextract.py` | `make_simple_optimal_extractor(P_or_file, dark)` | Builds a simple optimal extractor from a spatial profile array. |
| `specextract.py` | `build_simple_optimal_profile(...)` | Builds the spatial profile matrix P for `make_simple_optimal_extractor`. |
| `specextract.py` | `save_simple_optimal_profile(P, traces, filename, xmin)` | Saves the profile P to disk. |

### Running extraction

| File | Function | Role |
|------|----------|------|
| `specextract.py` | `extract_to_coupling_map(averaged_h5, extractor, output_fits, ...)` | **Main entry point (tutorial).** Applies extractor to every grid bin in the averaged H5, writes coupling-map FITS. |
| `specextract.py` | `extract_from_config(configname)` | **CLI entry point.** Parses `.ini` and calls `extract_to_coupling_map`. |
| `specextract.py` | `extract_from_fits(...)` | Applies extractor to a single FITS image (not the grid H5). |
| `specextract.py` | `extract_spec(im, ylocs, width)` | Low-level box extraction from one image given row positions. |
| `specextract.py` | `_load_dark(dark_fits)` | Loads a dark frame from a FITS file. |
| `specextract.py` | `_probe_extractor_shape(extractor, avg_plcam, map_n)` | Determines output spectrum shape by running extractor on one frame. |
| `specextract.py` | `_trim_matrix_to_roi(A, xmin, xmax, ny_full, roi_x0, roi_x1)` | Crops the LSF matrix to the detector ROI. |
| `specextract.py` | `_write_extraction_header(header, extractor, extra)` | Writes metadata into the output FITS header. |
| `scripts/_diagnostics.py` | `plot_step5_extraction(cm_fits, map_h5)` | CLI diagnostic: coupling map summary plot. |
| `scripts/_diagnostics.py` | `plot_firstpl_quality(model_file, ref_image, ...)` | CLI diagnostic: FIRSTPL extractor quality check (only for `extractor=FIRSTPL`). |
| `pipeline.py` | `_step5(configname)` | CLI wrapper: calls `extract_from_config` then saves diagnostics. |

---

## Step 5 — Image reconstruction ⚠️ *Incomplete / not in CLI*

**Tutorial:** `tutorial_step5_image_reconstruction.ipynb` (runs on older data; not integrated into the unified pipeline)

This step uses `mapmodel.py`, `fit.py`, and `imgrecon.py` which were carried over from the previous pipeline version. The step currently requires manually providing a coupling-map FITS file from the old format. Integration with the Step 4 output is not yet done.

### Building the polynomial coupling-map model

| File | Function | Role |
|------|----------|------|
| `sort.py` | `make_responsemaps(filename, ...)` | ❌ **Legacy.** Old approach to building coupling maps from sorted H5 files. Predates `specextract.py`. |
| `mapmodel.py` | `CouplingMapModel(mapdata, model, min_nframes, ...)` | Loads raw coupling-map data or a saved polynomial model. |
| `mapmodel.py` | `make_polynomial_model(output_name, wav_fitrange, wav_reconrange, ...)` | Fits 2D polynomial + spectral polynomial to the coupling map cube; writes `polymodel.fits`. |
| `mapmodel.py` | `poly_design_matrix(x, y, degree)` | Constructs the spatial polynomial design matrix. |
| `mapmodel.py` | `make_interpolation_model(...)` / `make_interpolation_model_irregular(...)` | Low-level interpolation model builders called by `make_polynomial_model`. |
| `mapmodel.py` | `compute_vec(specind, fibind, xshift, yshift, n_trim)` | Evaluates the polynomial model at a given sky offset. |
| `mapmodel.py` | `diagnostic_plot_model(fibind)` | Animated GIF of model vs data per fiber. |
| `mapmodel.py` | `diagnostic_plot_residuals(specind)` / `diagnostic_plot_SN(specind)` | Residual and S/N maps for a given wavelength channel. |

### Building the convolution matrix and fitting

| File | Function | Role |
|------|----------|------|
| `fit.py` | `PLMapFit(model_file, image_ngrid, image_fov, n_trim)` | Loads polynomial model; sets up image grid. |
| `fit.py` | `PLMapFit.make_matrix(specind, fiber_inds)` | Builds the `(n_fibers, n_pixels)` convolution matrix for one wavelength. |
| `fit.py` | `PLMapFit.save_matrix_to_file(filename)` / `load_matrix_from_file(filename)` | Saves/loads a pre-built matrix to/from FITS. |
| `fit.py` | `PLMapFit.prepare_data(fiber_inds)` | Extracts and stacks data vectors for selected fibers. |
| `fit.py` | `PLMapFit.subsample_matrix(fiber_inds)` | Subsets the convolution matrix to selected fibers (used in bootstrap). |
| `fit.py` | `PLMapFit.store_hyperparams(ini_temp, tau, gamma, n_element, target_chi2, ...)` | Stores MCMC hyperparameters on the fitter. |
| `fit.py` | `PLMapFit.run(centerfrac, niter, burn_in_iter, ...)` | **Main MCMC entry point.** Runs simulated-annealing image reconstruction. |
| `fit.py` | `PLMapFit.run_fitting_gaussian(ini_params, bounds, ...)` | Gaussian model fit (scipy minimize). |
| `fit.py` | `PLMapFit.run_fitting_pointsource(n_point_sources, ini_params, ...)` | Point-source model fit (scipy minimize). |
| `fit.py` | `PLMapFit.plot_data(...)` / `plot_model(...)` / `plot_residuals(...)` / `plot_1d(...)` | Diagnostic plots of data, model, and residuals. |

### Image reconstruction (MCMC)

| File | Function | Role |
|------|----------|------|
| `imgrecon.py` | `locs2image(locs, axis_len)` | Converts a list of flux-element locations to a 2D image array. |
| `imgrecon.py` | `entropy(image)` | Computes image entropy (regularization term). |
| `imgrecon.py` | `BaseImageReconstructor` | MCMC reconstructor class (simulated annealing, SQUEEZE-inspired). |
| `imgrecon.py` | `BaseImageReconstructor.run_chain(niter, central_frac, ...)` | Runs the MCMC chain. |
| `imgrecon.py` | `BaseImageReconstructor.compute_ll(vec)` | Computes chi² log-likelihood. |
| `imgrecon.py` | `BaseImageReconstructor.compute_regul()` | Computes regularization (entropy). |
| `imgrecon.py` | `BaseImageReconstructor.move_element(ni, move_scheme)` | Proposes a single MCMC move. |
| `imgrecon.py` | `BaseImageReconstructor.set_initial_state(central_frac)` | Initializes flux-element positions. |
| `imgrecon.py` | `CouplingMapImageReconstructor` | Subclass of `BaseImageReconstructor` for coupling-map data. |
| `imgrecon.py` | `PointSourceFitter` | Subclass of `BaseModelFitter` for point-source optimization. |
| `imgrecon.py` | `GaussianBlobFitter` | Subclass of `BaseModelFitter` for Gaussian blob optimization. |

---

## Supporting / standalone modules

| File | Module / Function | Notes |
|------|-------------------|-------|
| `imageutils.py` | `subpixel_centroid_2d`, `find_centroid`, `extract_patch`, `shift_image_warpaffine`, etc. | Image utility functions used by `sort.py` (centroid detection) and `fit.py`. |
| `wavecal.py` | `make_wave_solution_neon(...)`, `make_wave_solution_laser(...)` | Standalone wavelength calibration. **Not integrated into the main pipeline.** Status: experimental. |
| `h5_products.py` | `create_roi_access_h5(...)`, `append_roi_access_h5(...)` | Low-level H5 writers used by `average.build_ROI_access`. |
| `_sort_base.py` | `find_data_between`, `validate_timestamp_matching`, `bin_by_centroids_from_indices`, `compute_weighted_frame_binning` | Shared utilities imported by `sort.py` and `average.py`. |
| `visPLred/preprocess.py` | `DetectorNonlinearityModel`, `NonlinearityFitter` | Detector nonlinearity calibration (instrument-specific). Used in pre-processing notebooks, not main pipeline. |
| `visPLred/spec.py` | `SpectrumModel`, `frame_to_spec`, `extract_spec_box`, `extract_spec_optimal` | Older spectral extraction (visPLred-specific). Superseded by `specextract.py` for the main pipeline. |

---

## CLI step vs tutorial step numbering

| Tutorial notebook | Tutorial step | CLI step (`pipeline.run_mode1`) |
|---|---|---|
| `tutorial_step1_sort.ipynb` | Step 1 | `steps=[1]` |
| `tutorial_step2_ingest.ipynb` | Step 2 | `steps=[2]` |
| `tutorial_step3_average.ipynb` | Step 3 | `steps=[3]` (ROI cache) + `steps=[4]` (averaging) |
| `tutorial_step4_specextract.ipynb` | Step 4 | `steps=[5]` |
| `tutorial_step5_image_reconstruction.ipynb` | Step 5 ⚠️ | No CLI equivalent yet |
