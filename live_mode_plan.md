# PLred Live Mode Implementation Plan

## Context

During observations, Fastcam and Slowcam run on separate computers and write data continuously. The observer wants to monitor ROI response maps with ~5–15 second latency. This is a separate pipeline from the offline science reduction (`sort → ingest → average`), which must remain unchanged.

The live mode produces an append-only `live_roi.h5` that is **compatible with the existing `roi_viewer.html`** already in `PLred-dev/PLred/viewers/`. No new viewer needs to be built — the format matches what `build_ROI_access()` in `average.py` already writes to standalone H5 files.

---

## Files to Create / Modify

| Action | Path |
|--------|------|
| **Create** | `PLred/live.py` — core live pipeline module |
| **Create** | `PLred/scripts/live_cli.py` — CLI entry point |
| **Create** | `PLred/live_template_config.ini` — standalone template config for live mode |
| **Modify** | `setup.py` — add `plred-live` console_script |
| **Modify** | `PLred/viewers/roi_viewer.html` — add 10-second auto-refresh |

Do **not** touch `sort.py`, `ingest.py`, `average.py`, or `template_config.ini`.

---

## live_roi.h5 Layout

Identical to the standalone viewer H5 written by `average.build_ROI_access(h5_path=...)` so that `roi_viewer.html` works without modification:

```
live_roi.h5
  attrs:
    layout              = "plred_roi_access_v1"   ← same key roi_viewer.html checks
    product_type        = "live_preview"
    created_by          = "PLred.live"
    dark_subtracted     = False
    roi                 = [y0, y1, x0, x1]         ← global PLcam pixel coords
    roi_coordinate_type = "global_plcam_pixels"
    t0                  = <first Unix timestamp>
    source_fastcam_dir  = ...
    source_slowcam_dir  = ...
    poll_sec            = ...
    created_time        = ...
    psfcam_is_fast      = True/False
    live_pixels_json    = <JSON string>

  /roi_access    float32  (roi_h, roi_w, N)   chunks=(1, 1, 512)  resizable
  /timestamps    float64  (N,)                chunks=(512,)        resizable
  /centroids     float32  (N, 2)              chunks=(512, 2)      resizable
  /peaks         float32  (N,)                chunks=(512,)        resizable
  /nstacks       float32  (N,)                chunks=(512,)        resizable  [optional]
```

PSF frames are **excluded** from live_roi.h5 to keep the file lightweight — the viewer reloads the whole file on refresh, so smaller is better. The viewer can show centroid scatter and peak histograms using just `/centroids` and `/peaks` without PSF thumbnails.

`roi_access[local_y, local_x, :]` gives one pixel's time series — fast for viewer response-map queries.

---

## Existing Code to Reuse

All from `PLred/sort.py`:
- `_load_timestamp_file(filepath)` — reads SCExAO/MILK `.txt` timestamp files
- `compute_frame_durations(fastcam_timestamp, fastcam_fileinds)` — dead-time correction
- `_build_matching_dict(fastcam_timestamp, slowcam_timestamp, bisect_inds, ind_start, ind_end, frame_end_times)` — core weighted timestamp matching
- `find_data_between(datadir, obs_start, obs_end)` — file discovery filtered by time window

From `PLred/imageutils.py`:
- `subpixel_centroid_2d(image)` — PSF centroid computation

---

## live.py Structure

### Module-level helpers

```python
def _load_state(path) -> set           # read live_state.json → set of processed filenames
def _save_state(path, state)           # write updated set back to live_state.json (atomic rename)
def _rsync(remote_host, remote_dir, local_dir)  # subprocess rsync call; no-op if remote_host blank
def _wait_stable(path, wait=1.0, checks=3) -> bool  # file-size stability check before reading
```

### H5 I/O

```python
def _create_live_h5(path, roi, config, live_pixels)
    # Creates resizable datasets with correct chunking, sets all file-level attrs

def _append_live_h5(path, roi_frames, timestamps, centroids, peaks, nstacks=None)
    # Opens in 'r+', resizes all datasets, writes new slice
    # roi_frames shape: (n_new, roi_h, roi_w) → stored transposed as moveaxis(0,-1)
```

### Incremental matching

```python
def _load_fastcam_state(fastcam_local_dir, obs_start, obs_end,
                        dark_path, crop_width, apply_dtc)
    -> dict with keys: timestamps, fileinds, frameinds, frame_end_times, dark, files
    # Uses find_data_between() to discover .txt files in obs window
    # Concatenates all timestamps, computes frame_end_times
    # Refreshed each poll cycle to pick up new fastcam files

def match_one_slowcam_file(slowcam_ts_file, fastcam_state, slowcam_data_dir, roi)
    -> dict or None
    # 1. Load slowcam timestamps from txt file
    # 2. Compute bisect_inds against fastcam_state['timestamps']
    # 3. Call _build_matching_dict for this file's timestamp range
    # 4. Build weighted PSFcam frames from fastcam FITS (memmap=True)
    # 5. Compute centroids via subpixel_centroid_2d, peaks via np.max
    # 6. Load PLcam ROI from slowcam FITS (memmap=True, no dark subtraction)
    # 7. Return {roi_frames, timestamps, centroids, peaks, nstacks}
    # Returns None if no overlap found (logs warning, caller marks processed and skips)
```

### Main loop

```python
def run_live(config_path, out_path=None, roi_override=None, poll_sec_override=None)
    # Parses config; creates live_roi.h5 if it doesn't exist
    # Enters while True loop:
    #   rsync new fastcam + slowcam files
    #   refresh fastcam_state (rescan using find_data_between)
    #   find slowcam .txt files in obs window not yet in live_state.json
    #   for each stable, unprocessed file:
    #     match_one_slowcam_file → append_live_h5 → save_state
    #   sleep(poll_sec)
```

---

## live_cli.py

```python
def main():
    parser = argparse.ArgumentParser(prog='plred-live', ...)
    parser.add_argument('config')
    parser.add_argument('--out', default=None)
    parser.add_argument('--roi', default=None)   # "y0,y1,x0,x1" override
    parser.add_argument('--poll-sec', type=float, default=None)
    args = parser.parse_args()
    from PLred.live import run_live
    run_live(args.config, out_path=args.out, roi_override=args.roi,
             poll_sec_override=args.poll_sec)
```

---

## setup.py Addition

```python
"plred-live = PLred.scripts.live_cli:main",
```

---

## Separate Live Config File: `live_template_config.ini`

Observer copies this per target and fills in values. Not part of the main pipeline config.

```ini
# PLred live-mode config — copy and fill in per observing target.
# Usage: plred-live my_target_live.ini

[Live]
outpath             = live_roi.h5
roi                 = 800,1200,1234,1235   # y0,y1,x0,x1 global PLcam pixel coords
poll_sec            = 2
psfcam_is_fast      = True
apply_dead_time_correction = True

[Fastcam]
remote_host         =                      # blank = no rsync, reads local_dir directly
remote_dir          =
local_dir           = incoming/fastcam
obs_start           = 12:00:00             # HH:MM:SS — skip files before this time
obs_end             = 12:30:00             # HH:MM:SS — skip files after this time
dark_file           =                      # PSFcam dark FITS; blank = no dark subtraction
crop_width          = 20                   # px around PSF center to keep

[Slowcam]
remote_host         =
remote_dir          =
local_dir           = incoming/slowcam
obs_start           = 12:00:00
obs_end             = 12:30:00

[LivePixels]
# py, px, label — global PLcam pixel coordinates (one per line)
pixels              =
#    900,1234,trace_center
#    901,1234,trace_plus1
#    902,1234,trace_plus2
```

`obs_start` / `obs_end` are passed to `find_data_between()` to filter which timestamp files are discovered, preventing pickup of data from previous targets on the same night.

---

## Processed-File Tracking

`live_state.json` alongside `live_roi.h5`:
```json
{"processed_slowcam_timestamp_files": ["slowcam_0001.txt", ...]}
```
Written atomically (write temp file → rename) to survive crashes. Updated only after a successful H5 append.

---

## Partial-File Safety

Slowcam `.txt` files are processed only when both the `.txt` and matching `.fits` file exist and have stable size (using `_wait_stable`, same pattern as `watcher_first.py`). This avoids ingesting partially written FITS files.

---

## HTML Viewer Auto-Refresh

Modify `PLred/viewers/roi_viewer.html` to reload the open H5 file every 10 seconds when a live file is detected.

Strategy: add a `setInterval` that calls the existing file-load function if the loaded file has `product_type = "live_preview"` in its attributes. Minimal change — just re-run the same read path on a timer.

```javascript
// After file load, check if live mode and start auto-refresh
if (attrs.product_type === "live_preview") {
    startLiveRefresh(filePath, intervalMs=10000);
}

function startLiveRefresh(filePath, intervalMs) {
    if (window._liveRefreshTimer) clearInterval(window._liveRefreshTimer);
    window._liveRefreshTimer = setInterval(() => loadFile(filePath), intervalMs);
}
```

A small "LIVE — refreshing every 10s" badge is shown in the UI when refresh is active; clicking it pauses the refresh.

This is fast because `live_roi.h5` contains no PSF frames — only `roi_access`, `timestamps`, `centroids`, `peaks`.

---

## Phase 1 Scope

- `PLred/live.py` with all functions above
- `PLred/scripts/live_cli.py` + setup.py `plred-live` entry point
- `PLred/live_template_config.ini` as a standalone per-target config template
- Polling-based rsync (no watchdog); `obs_start`/`obs_end` filtering via `find_data_between()`
- `live_roi.h5` without PSF frames, compatible with existing `roi_viewer.html`
- Predefined live pixels stored as JSON in file attrs
- Auto-refresh in `viewers/roi_viewer.html` when `product_type = "live_preview"`

Phase 2+ (not in this implementation): rolling-memory mode, zarr backend.

---

## Verification

1. `pip install -e .` in PLred-dev, confirm `plred-live --help` works
2. Copy `live_template_config.ini`, point `[Fastcam].local_dir` and `[Slowcam].local_dir` at tutorial test data in `PLred/tutorials/data/`, set `obs_start`/`obs_end` to bracket the test data
3. Run `plred-live my_live.ini` and confirm `live_roi.h5` is created with correct attrs and datasets
4. Confirm datasets grow after each poll cycle; `live_state.json` is updated
5. Open `PLred/viewers/roi_viewer.html`, load `live_roi.h5`, verify "LIVE" badge appears and the view refreshes every 10 s
6. Confirm re-running `plred-live` from scratch skips already-processed files
