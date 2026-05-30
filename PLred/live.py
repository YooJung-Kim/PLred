"""
PLred live observing pipeline — two-machine mode.

Two independent polling loops:

  plred-live-sort  (fastcam / PSFcam machine)
    rsync slowcam .txt files in
    timestamp match + centroid computation
    save lightweight .npz per slowcam file
    rsync .npz to PLcam machine

  plred-live-roi   (PLcam machine)
    rsync .npz match results in
    extract PLcam ROI from corresponding FITS
    append to live_roi.h5 (plred_roi_access_v1 format)

Both scripts run on one machine for local testing by leaving remote_host blank.
"""

import os
import glob
import json
import time
import logging
import subprocess
import tempfile
from datetime import datetime

import numpy as np

log = logging.getLogger('plred.live')


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rsync(remote_user, remote_host, remote_dir, local_dir, extra_args=()):
    """rsync remote → local.  No-op when remote_host is blank."""
    if not remote_host:
        return
    src = ('%s@%s:%s' % (remote_user, remote_host, remote_dir)
           if remote_user else '%s:%s' % (remote_host, remote_dir))
    os.makedirs(local_dir, exist_ok=True)
    cmd = ['rsync', '-az', '--ignore-existing'] + list(extra_args) + [src, local_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("rsync failed: %s", result.stderr.strip())


def _rsync_file(remote_user, remote_host, remote_dir, local_dir, filename):
    """rsync a single file remote → local."""
    if not remote_host:
        return
    src = ('%s@%s:%s/%s' % (remote_user, remote_host, remote_dir.rstrip('/'), filename)
           if remote_user else '%s:%s/%s' % (remote_host, remote_dir.rstrip('/'), filename))
    os.makedirs(local_dir, exist_ok=True)
    cmd = ['rsync', '-az', src, local_dir]
    subprocess.run(cmd, capture_output=True, text=True)


def _wait_stable(path, wait=1.5, checks=3):
    """Return True when file size stops changing (file fully written)."""
    prev = -1
    for _ in range(checks):
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            time.sleep(wait)
            continue
        if size == prev and size > 0:
            return True
        prev = size
        time.sleep(wait)
    return False


def _load_state(path):
    """Load processed-file set from JSON state file."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path) as f:
            return set(json.load(f).get('processed', []))
    except Exception:
        return set()


def _save_state(path, state):
    """Atomically write processed-file set to JSON state file."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'processed': sorted(state)}, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Sort-role helpers  (fastcam / PSFcam machine)
# ---------------------------------------------------------------------------

def _load_fastcam_state(fastcam_local_dir, obs_start, obs_end,
                        dark_file, crop_width, apply_dtc):
    """
    Scan fastcam directory for .txt files in [obs_start, obs_end], load all
    timestamps, and optionally compute dead-time-corrected frame end times.

    Returns dict or None if no files found.
    """
    from .sort import _load_timestamp_file, compute_frame_durations
    from ._sort_base import find_data_between
    from astropy.io import fits as pyfits

    d = fastcam_local_dir.rstrip('/') + '/'
    txt_files = find_data_between(d, obs_start, obs_end, footer='.txt')
    if not txt_files:
        return None

    fits_files = [f.replace('.txt', '.fits') for f in txt_files]

    ts_list, fi_list, fr_list = [], [], []
    for i, txt in enumerate(txt_files):
        ts, fr = _load_timestamp_file(txt)
        ts_list.append(ts)
        fi_list.append(np.full(len(ts), i, dtype=int))
        fr_list.append(fr)

    timestamps = np.concatenate(ts_list)
    fileinds   = np.concatenate(fi_list)
    frameinds  = np.concatenate(fr_list)

    frame_end_times = None
    if apply_dtc and len(timestamps) > 1:
        frame_end_times = compute_frame_durations(timestamps, fileinds)

    dark = None
    cw = int(crop_width) if crop_width else None
    if dark_file and os.path.exists(dark_file):
        d_arr = pyfits.getdata(dark_file).astype('float32')
        if d_arr.ndim == 3:
            d_arr = d_arr.mean(axis=0)
        if cw:
            cy, cx = d_arr.shape[0] // 2, d_arr.shape[1] // 2
            dark = d_arr[cy - cw:cy + cw, cx - cw:cx + cw]
        else:
            dark = d_arr

    return {
        'timestamps':      timestamps,
        'fileinds':        fileinds,
        'frameinds':       frameinds,
        'frame_end_times': frame_end_times,
        'dark':            dark,
        'files':           fits_files,
        'crop_width':      cw,
    }


def match_one_slowcam_file(slowcam_txt, fastcam_state):
    """
    Match one slowcam timestamp file against the in-memory fastcam state.

    Returns a dict suitable for save_match_result(), or None when there is
    no temporal overlap with the fastcam data.

    Keys in the return dict:
        timestamps   float64 (N,)   — slowcam Unix timestamps
        centroids    float32 (N, 2) — PSF centroid (x, y)
        peaks        float32 (N,)   — PSF max pixel
        nstacks      float32 (N,)   — total exposure weight
        frameinds    int     (N,)   — frame indices within the slowcam FITS file
    """
    from bisect import bisect as bisect_fn
    from astropy.io import fits as pyfits
    from .sort import _load_timestamp_file, _build_matching_dict
    from .imageutils import subpixel_centroid_2d

    slowcam_ts, slowcam_frameinds_all = _load_timestamp_file(slowcam_txt)

    fc_ts          = fastcam_state['timestamps']
    fc_fileinds    = fastcam_state['fileinds']
    fc_frameinds   = fastcam_state['frameinds']
    frame_end_times = fastcam_state.get('frame_end_times')
    dark           = fastcam_state.get('dark')
    fits_files     = fastcam_state['files']
    cw             = fastcam_state.get('crop_width')

    bisect_inds = [bisect_fn(fc_ts, t) for t in slowcam_ts]
    bisect_arr  = np.array(bisect_inds)

    max_bisect = int(bisect_arr.max()) if len(bisect_arr) > 0 else 0
    if max_bisect == 0:
        log.warning("No fastcam timestamps after any slowcam timestamp in %s",
                    os.path.basename(slowcam_txt))
        return None

    ind_end = int(np.argmax(bisect_arr == max_bisect))
    if ind_end == 0:
        log.warning("No temporal overlap in %s", os.path.basename(slowcam_txt))
        return None

    Dict, matched_ts, n_skipped = _build_matching_dict(
        fc_ts, slowcam_ts, bisect_inds, 0, ind_end,
        frame_end_times=frame_end_times,
    )

    if not Dict:
        log.warning("Empty match dict for %s", os.path.basename(slowcam_txt))
        return None
    if n_skipped > 0:
        log.warning("%d slowcam frame(s) skipped in %s", n_skipped,
                    os.path.basename(slowcam_txt))

    indices = list(Dict.keys())
    N = len(indices)

    # --- determine PSFcam frame crop shape ---
    if cw:
        fh = fw = 2 * cw
    else:
        # probe from first needed fastcam FITS file
        first_fc_ind  = list(Dict[indices[0]].keys())[0]
        first_file_ind = int(fc_fileinds[first_fc_ind])
        probe = pyfits.getdata(fits_files[first_file_ind])
        fh, fw = probe.shape[-2], probe.shape[-1]

    psf_frames = np.zeros((N, fh, fw), dtype='float32')
    nstacks    = np.zeros(N, dtype='float32')

    # Load fastcam FITS file by file (efficient: open each file once)
    current_file_ind = -1
    current_data     = None

    for out_idx, slow_ind in enumerate(indices):
        fc_inds  = list(Dict[slow_ind].keys())
        fc_fracs = list(Dict[slow_ind].values())

        frame_acc = np.zeros((fh, fw), dtype='float64')
        w_total   = 0.0

        for fc_ind, frac in zip(fc_inds, fc_fracs):
            file_ind  = int(fc_fileinds[fc_ind])
            frame_ind = int(fc_frameinds[fc_ind])

            if file_ind != current_file_ind:
                current_file_ind = file_ind
                raw = pyfits.getdata(fits_files[file_ind]).astype('float32')
                if cw:
                    h, w = raw.shape[-2], raw.shape[-1]
                    cy, cx = h // 2, w // 2
                    raw = raw[:, cy - cw:cy + cw, cx - cw:cx + cw]
                if dark is not None:
                    raw = raw - dark
                current_data = raw

            frame_acc += frac * current_data[frame_ind]
            w_total   += frac

        if w_total > 0:
            psf_frames[out_idx] = (frame_acc / w_total).astype('float32')
        nstacks[out_idx] = w_total

    # --- compute centroids and peaks ---
    centroids = np.zeros((N, 2), dtype='float32')
    peaks     = np.zeros(N, dtype='float32')

    for i, frame in enumerate(psf_frames):
        try:
            centroids[i] = subpixel_centroid_2d(frame)
        except Exception:
            centroids[i] = (np.nan, np.nan)
        peaks[i] = float(np.nanmax(frame))

    # frame indices within the slowcam FITS file for each matched timestamp
    matched_frameinds = np.array(
        [int(slowcam_frameinds_all[ind]) for ind in indices], dtype=int)

    return {
        'timestamps': np.array(matched_ts, dtype='float64'),
        'centroids':  centroids,
        'peaks':      peaks,
        'nstacks':    nstacks,
        'frameinds':  matched_frameinds,
    }


def save_match_result(result, slowcam_txt, output_dir):
    """
    Save match result dict as <slowcam_basename>_match.npz in output_dir.
    Returns the path of the written .npz file.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(slowcam_txt))[0]
    path = os.path.join(output_dir, base + '_match.npz')
    np.savez(path,
             timestamps=result['timestamps'],
             centroids=result['centroids'],
             peaks=result['peaks'],
             nstacks=result['nstacks'],
             frameinds=result['frameinds'])
    return path


def run_sort_loop(config_path):
    """
    Main loop for the fastcam (PSFcam) machine.

    Polls for new slowcam .txt files, does timestamp matching and centroid
    computation, writes a lightweight .npz per file, and rsyncs results to
    the PLcam machine.
    """
    from configobj import ConfigObj
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')

    cfg      = ConfigObj(config_path)
    sort_cfg = cfg.get('Sort', {})
    fc_cfg   = cfg.get('Fastcam', {})
    sc_cfg   = cfg.get('Slowcam', {})
    out_cfg  = cfg.get('Output', {})

    obs_start  = sort_cfg.get('obs_start', '00:00:00').strip()
    obs_end    = sort_cfg.get('obs_end',   '23:59:59').strip()
    poll_sec   = float(sort_cfg.get('poll_sec', 2) or 2)
    crop_width = sort_cfg.get('crop_width', '').strip()
    crop_width = int(crop_width) if crop_width else None
    dark_file  = sort_cfg.get('dark_file', '').strip() or None
    apply_dtc  = str(sort_cfg.get('apply_dead_time_correction', 'True')).lower() == 'true'

    fastcam_local_dir = fc_cfg.get('local_dir', '.').strip()

    sc_remote_host = sc_cfg.get('remote_host', '').strip()
    sc_remote_user = sc_cfg.get('remote_user', '').strip()
    sc_remote_dir  = sc_cfg.get('remote_dir',  '').strip()
    sc_local_dir   = sc_cfg.get('local_dir',   'incoming/slowcam').strip()

    out_local_dir   = out_cfg.get('local_dir',   'match_results').strip()
    out_remote_host = out_cfg.get('remote_host', '').strip()
    out_remote_user = out_cfg.get('remote_user', '').strip()
    out_remote_dir  = out_cfg.get('remote_dir',  '').strip()

    os.makedirs(sc_local_dir,  exist_ok=True)
    os.makedirs(out_local_dir, exist_ok=True)

    state_path   = os.path.join(out_local_dir, 'live_sort_state.json')
    state        = _load_state(state_path)
    fastcam_state = None

    log.info("plred-live-sort started")
    log.info("  fastcam local:  %s", fastcam_local_dir)
    log.info("  slowcam remote: %s  local: %s", sc_remote_host or '(local)', sc_local_dir)
    log.info("  output:         %s", out_local_dir)
    log.info("  obs window:     %s — %s", obs_start, obs_end)

    while True:
        # ── rsync slowcam .txt files from PLcam machine ───────────────────
        _rsync(sc_remote_user, sc_remote_host, sc_remote_dir, sc_local_dir,
               ('--include=*.txt', '--exclude=*'))

        # ── refresh fastcam state ─────────────────────────────────────────
        new_state = _load_fastcam_state(
            fastcam_local_dir, obs_start, obs_end, dark_file, crop_width, apply_dtc)
        if new_state is not None:
            fastcam_state = new_state

        if fastcam_state is None:
            log.info("No fastcam data found yet, waiting…")
            time.sleep(poll_sec)
            continue

        # ── find unprocessed slowcam .txt files ───────────────────────────
        from ._sort_base import find_data_between
        sc_dir = sc_local_dir.rstrip('/') + '/'
        all_sc_txt = find_data_between(sc_dir, obs_start, obs_end, footer='.txt')
        new_files  = [f for f in all_sc_txt
                      if os.path.basename(f) not in state]

        for slowcam_txt in new_files:
            if not _wait_stable(slowcam_txt):
                log.warning("Not stable yet: %s", os.path.basename(slowcam_txt))
                continue

            log.info("Matching: %s", os.path.basename(slowcam_txt))
            try:
                result = match_one_slowcam_file(slowcam_txt, fastcam_state)
            except Exception:
                log.exception("match_one_slowcam_file failed for %s",
                              os.path.basename(slowcam_txt))
                state.add(os.path.basename(slowcam_txt))
                _save_state(state_path, state)
                continue

            if result is None:
                log.warning("No match for %s — skipping",
                            os.path.basename(slowcam_txt))
                state.add(os.path.basename(slowcam_txt))
                _save_state(state_path, state)
                continue

            npz_path = save_match_result(result, slowcam_txt, out_local_dir)
            log.info("  → %s  (%d frames)",
                     os.path.basename(npz_path), len(result['timestamps']))

            # rsync .npz to PLcam machine
            if out_remote_host:
                _rsync_file(out_remote_user, out_remote_host, out_remote_dir,
                            out_remote_dir, os.path.basename(npz_path))

            state.add(os.path.basename(slowcam_txt))
            _save_state(state_path, state)

        time.sleep(poll_sec)


# ---------------------------------------------------------------------------
# ROI-role helpers  (PLcam machine)
# ---------------------------------------------------------------------------

def load_plcam_roi_frames(slowcam_fits, frameinds, roi):
    """
    Load specific frames from a PLcam FITS file and crop to roi.

    Parameters
    ----------
    slowcam_fits : str
        Path to the PLcam FITS file.
    frameinds : array-like of int
        Frame indices within the FITS cube.
    roi : tuple (y0, y1, x0, x1)
        Global PLcam pixel coordinates.

    Returns
    -------
    ndarray float32, shape (N, roi_h, roi_w)
    """
    from astropy.io import fits as pyfits

    y0, y1, x0, x1 = roi
    roi_h, roi_w = y1 - y0, x1 - x0
    N = len(frameinds)

    out = np.zeros((N, roi_h, roi_w), dtype='float32')
    try:
        hdul = pyfits.open(slowcam_fits, memmap=True)
        with hdul:
            data = hdul[0].data
            for i, fi in enumerate(frameinds):
                out[i] = data[fi, y0:y1, x0:x1].astype('float32')
    except (ValueError, Exception):
        # memmap blocked by BZERO/BSCALE — fall back to full load
        with pyfits.open(slowcam_fits, memmap=False) as hdul:
            data = hdul[0].data
            for i, fi in enumerate(frameinds):
                out[i] = data[fi, y0:y1, x0:x1].astype('float32')

    return out


def _create_live_h5(outpath, roi, extra_attrs):
    """Create an empty live_roi.h5 using h5_products schema."""
    from .h5_products import create_roi_access_h5
    create_roi_access_h5(outpath, roi, attrs=extra_attrs)


def run_roi_loop(config_path):
    """
    Main loop for the PLcam machine.

    Polls for new .npz match results, loads the corresponding PLcam FITS
    frames, and appends to live_roi.h5.
    """
    from configobj import ConfigObj
    from .h5_products import append_roi_access_h5
    import h5py

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')

    cfg    = ConfigObj(config_path)
    roi_cfg = cfg.get('ROI', {})
    sc_cfg  = cfg.get('Slowcam', {})
    mr_cfg  = cfg.get('MatchResults', {})
    lp_cfg  = cfg.get('LivePixels', {})

    outpath   = roi_cfg.get('outpath', 'live_roi.h5').strip()
    roi_str   = roi_cfg.get('roi', '').strip()
    if not roi_str:
        raise ValueError("[ROI] roi is required (y0,y1,x0,x1)")
    roi = tuple(int(x) for x in roi_str.split(','))

    obs_start = roi_cfg.get('obs_start', '00:00:00').strip()
    obs_end   = roi_cfg.get('obs_end',   '23:59:59').strip()
    poll_sec  = float(roi_cfg.get('poll_sec', 2) or 2)

    slowcam_local_dir = sc_cfg.get('local_dir', '.').strip()

    mr_remote_host = mr_cfg.get('remote_host', '').strip()
    mr_remote_user = mr_cfg.get('remote_user', '').strip()
    mr_remote_dir  = mr_cfg.get('remote_dir',  '').strip()
    mr_local_dir   = mr_cfg.get('local_dir',   'incoming/match_results').strip()

    # Parse [LivePixels]
    pixels_raw = lp_cfg.get('pixels', [])
    if isinstance(pixels_raw, str):
        pixels_raw = [pixels_raw]
    live_pixels = []
    for line in pixels_raw:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                live_pixels.append({
                    'py': int(parts[0]),
                    'px': int(parts[1]),
                    'label': parts[2] if len(parts) > 2 else '',
                })
            except ValueError:
                pass

    os.makedirs(mr_local_dir, exist_ok=True)
    outdir = os.path.dirname(os.path.abspath(outpath))
    os.makedirs(outdir, exist_ok=True)

    state_path = os.path.join(outdir, 'live_roi_state.json')
    state = _load_state(state_path)

    # Create live_roi.h5 if not present
    if not os.path.exists(outpath):
        _create_live_h5(outpath, roi, {
            'product_type':        'live_preview',
            'created_by':          'PLred.live',
            'dark_subtracted':     False,
            'roi_coordinate_type': 'global_plcam_pixels',
            'psfcam_is_fast':      True,
            'poll_sec':            poll_sec,
            'created_time':        datetime.now().isoformat(),
            'live_pixels_json':    json.dumps(live_pixels),
        })
        log.info("Created %s", outpath)

    log.info("plred-live-roi started")
    log.info("  roi:            %s", roi)
    log.info("  slowcam local:  %s", slowcam_local_dir)
    log.info("  match results:  remote=%s  local=%s",
             mr_remote_host or '(local)', mr_local_dir)

    first_append = True

    while True:
        # ── rsync .npz match results from fastcam machine ─────────────────
        _rsync(mr_remote_user, mr_remote_host, mr_remote_dir, mr_local_dir,
               ('--include=*.npz', '--exclude=*'))

        # ── find unprocessed .npz files ───────────────────────────────────
        all_npz  = sorted(glob.glob(os.path.join(mr_local_dir, '*_match.npz')))
        new_npz  = [f for f in all_npz if os.path.basename(f) not in state]

        for npz_path in new_npz:
            if not _wait_stable(npz_path):
                log.warning("Not stable yet: %s", os.path.basename(npz_path))
                continue

            log.info("Processing: %s", os.path.basename(npz_path))

            try:
                data       = np.load(npz_path)
                timestamps = data['timestamps']
                centroids  = data['centroids']
                peaks      = data['peaks']
                nstacks    = data['nstacks']
                frameinds  = data['frameinds'].astype(int)

                # Derive PLcam FITS path from .npz basename
                # e.g. "cropped_firstpl_15:05:17_match.npz" → "cropped_firstpl_15:05:17.fits"
                npz_base      = os.path.basename(npz_path)
                slowcam_base  = npz_base[:-len('_match.npz')] + '.fits'
                slowcam_fits  = os.path.join(slowcam_local_dir, slowcam_base)

                if not os.path.exists(slowcam_fits):
                    log.warning("PLcam FITS not found yet: %s — will retry", slowcam_base)
                    continue   # don't mark processed; retry next cycle

                if not _wait_stable(slowcam_fits):
                    log.warning("PLcam FITS not stable: %s", slowcam_base)
                    continue

                roi_frames = load_plcam_roi_frames(slowcam_fits, frameinds, roi)
                append_roi_access_h5(outpath, roi_frames, timestamps,
                                     centroids, peaks, nstacks=nstacks)

                # Set t0 from first successful append
                if first_append and len(timestamps) > 0:
                    import h5py as _h5py
                    with _h5py.File(outpath, 'r+') as fh:
                        fh.attrs['t0'] = float(timestamps[0])
                    first_append = False

                log.info("  Appended %d frames → %s  (total %d)",
                         len(timestamps), os.path.basename(outpath),
                         _current_n(outpath))

            except Exception:
                log.exception("Failed to process %s", os.path.basename(npz_path))
                state.add(os.path.basename(npz_path))
                _save_state(state_path, state)
                continue

            state.add(os.path.basename(npz_path))
            _save_state(state_path, state)

        time.sleep(poll_sec)


def _current_n(h5_path):
    """Return current N (number of frames) in a live_roi.h5."""
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            return f['timestamps'].shape[0]
    except Exception:
        return -1
