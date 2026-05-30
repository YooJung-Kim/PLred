# codes related to constructing coupling maps
# MERGED VERSION:
#   - All-at-once timestamp matching (original sort.py)
#   - Bugfixed matching logic (sort4.py: edge cases, dead-time correction)
#   - Giant H5 output with weighted frames, nstacks, metadata (sort4.py)
#   - Full FrameSorter + binning functions (sort4.py)
#
# Bug fixes applied from sort4.py:
#   - Bug 1: single fastcam frame spanning entire slowcam interval
#   - Bug 2: negative index wrap-around when bisect_inds[ind] == 0
#   - Bug 3: ind_end == 0 validation
#   - Bug 4: float tolerance matching instead of exact equality
#   - Bug 5: slowcam_nbin truncation warnings
#   - Feature: dead-time correction using frame end times

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from .imageutils import subpixel_centroid_2d
from ._sort_base import (
    find_data_between,
    validate_timestamp_matching,
    bin_by_centroids_from_indices,
    compute_weighted_frame_binning,
)
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import h5py
import json
from configobj import ConfigObj
from datetime import datetime
from bisect import bisect
import os, glob, pickle, re


# ---------------------------------------------------------------------------
# Timestamp-file helpers
# ---------------------------------------------------------------------------

def _load_timestamp_file(filepath):
    '''
    Read one timestamp txt file (SCExAO/MILK format) and return
    (timestamps, frameinds).  Avoids reading the file multiple times.

    Returns
    -------
    timestamps : ndarray float64  – col 4, absolute acquisition time (Unix s)
    frameinds  : ndarray int      – col 0, datacube frame index within this file
    '''
    data = np.genfromtxt(filepath)
    return data[:, 4], data[:, 0].astype(int)


def compute_frame_durations(fastcam_timestamp, fastcam_fileinds):
    '''
    Return per-frame exposure durations, correcting for dead-time between
    FITS files.

    For interior frames within a file, duration = timestamp[i+1] - timestamp[i]
    (no dead-time inside a file).  For the last frame of each file (and the
    final frame overall), use the median intra-file interval instead, so that
    the inter-file gap does not inflate the effective exposure duration.

    Parameters
    ----------
    fastcam_timestamp : ndarray float64, shape (N,)
    fastcam_fileinds  : ndarray int, shape (N,)  – file index per frame

    Returns
    -------
    frame_end_times : ndarray float64, shape (N,)  – Unix time when each
                      frame's exposure ends
    '''
    N = len(fastcam_timestamp)
    durations = np.empty(N, dtype='float64')

    # Default: duration from next-frame start
    durations[:-1] = np.diff(fastcam_timestamp)
    durations[-1]  = np.nan  # filled below

    # Indices of the last frame in each file
    boundary_mask = np.concatenate([np.diff(fastcam_fileinds) != 0, [True]])

    for fid in np.unique(fastcam_fileinds):
        mask  = fastcam_fileinds == fid
        inds  = np.where(mask)[0]
        if len(inds) > 1:
            median_dur = np.median(np.diff(fastcam_timestamp[inds]))
        else:
            # Single-frame file: use global median of all intra-file intervals
            all_diffs = [np.diff(fastcam_timestamp[fastcam_fileinds == f])
                         for f in np.unique(fastcam_fileinds)
                         if (fastcam_fileinds == f).sum() > 1]
            median_dur = np.median(np.concatenate(all_diffs)) if all_diffs else 0.0
        durations[inds[-1]] = median_dur  # replace inflated inter-file gap

    return fastcam_timestamp + durations


def _build_matching_dict(fastcam_timestamp, slowcam_timestamp,
                         bisect_inds, ind_start, ind_end,
                         frame_end_times=None):
    '''
    Build {slowcam_ind: {fastcam_ind: weight}} with all edge cases handled.

    Bug 1 fix: when bisect_inds[ind] == bisect_inds[ind+1] a single fastcam
               frame covers the whole slowcam interval; combined weight =
               overlap / frame_duration.
    Bug 2 fix: when bisect_inds[ind] == 0 the left-boundary index would wrap
               to fastcam_timestamp[-1]; those slowcam frames are skipped.

    Parameters
    ----------
    frame_end_times : ndarray float64, shape (N,), optional
        Unix time when each fastcam frame's exposure ends, as returned by
        compute_frame_durations().  When provided, leftfrac/rightfrac are
        computed using actual frame durations, so dead-time gaps between
        FITS files do not inflate the weights.  When None, the original
        formulation (fastcam interval = next-frame-start minus this-frame-start)
        is used.

    Returns
    -------
    Dict       : dict
    timestamps : list  – slowcam timestamps for each key in Dict
    n_skipped  : int
    '''
    Dict = {}
    timestamps = []
    n_skipped = 0

    for ind in tqdm(np.arange(ind_start, ind_end)):
        bi      = bisect_inds[ind]
        bi_next = bisect_inds[ind + 1]

        # Bug 2: skip frames that precede all fastcam timestamps
        if bi == 0:
            print("WARNING: slowcam frame %d precedes all fastcam timestamps"
                  " — skipping." % ind)
            n_skipped += 1
            continue

        # Compute leftfrac: fraction of frame (bi-1) inside the slowcam interval
        if frame_end_times is not None:
            end_left  = frame_end_times[bi - 1]
            dur_left  = end_left - fastcam_timestamp[bi - 1]
            leftfrac  = max(0.0, end_left - slowcam_timestamp[ind]) / dur_left if dur_left > 0 else 0.0
        else:
            interval_left = fastcam_timestamp[bi] - fastcam_timestamp[bi - 1]
            leftfrac = (fastcam_timestamp[bi] - slowcam_timestamp[ind]) / interval_left

        if bi == bi_next:
            # Bug 1: only one fastcam frame spans the entire slowcam interval
            if frame_end_times is not None:
                end_left     = frame_end_times[bi - 1]
                dur_left     = end_left - fastcam_timestamp[bi - 1]
                slowcam_end  = slowcam_timestamp[ind + 1]
                overlap      = max(0.0, min(end_left, slowcam_end) - slowcam_timestamp[ind])
                combined     = overlap / dur_left if dur_left > 0 else 0.0
            else:
                interval_right = fastcam_timestamp[bi_next] - fastcam_timestamp[bi_next - 1]
                rightfrac = (slowcam_timestamp[ind + 1] - fastcam_timestamp[bi_next - 1]) / interval_right
                combined = max(leftfrac + rightfrac - 1.0, 0.0)
            Dict[ind] = {bi - 1: combined}
        else:
            # Compute rightfrac: fraction of frame (bi_next-1) inside the slowcam interval
            if frame_end_times is not None:
                end_right   = frame_end_times[bi_next - 1]
                dur_right   = end_right - fastcam_timestamp[bi_next - 1]
                slowcam_end = slowcam_timestamp[ind + 1]
                if slowcam_end >= end_right:
                    rightfrac = 1.0
                else:
                    rightfrac = max(0.0, slowcam_end - fastcam_timestamp[bi_next - 1]) / dur_right if dur_right > 0 else 0.0
            else:
                interval_right = fastcam_timestamp[bi_next] - fastcam_timestamp[bi_next - 1]
                rightfrac = (slowcam_timestamp[ind + 1] - fastcam_timestamp[bi_next - 1]) / interval_right
            Dict[ind] = {bi - 1: leftfrac}
            for k in np.arange(bi, bi_next - 1):
                Dict[ind][k] = 1
            Dict[ind][bi_next - 1] = rightfrac

        timestamps.append(slowcam_timestamp[ind])

    return Dict, timestamps, n_skipped


# ---------------------------------------------------------------------------
# Main: ALL-AT-ONCE timestamp matching
# ---------------------------------------------------------------------------

def script_match_timestamps(configname):
    '''
    ALL-AT-ONCE timestamp matching: loads all slowcam and fastcam timestamps,
    matches them, and outputs a single giant H5 file with all weighted frames.

    Config file example:

        [Fastcam]
        start_time          = 11:59:00
        end_time            = 12:20:09
        dark_file           = # if empty, dark_start_time and dark_end_time are used
        dark_start_time     = 12:23:04
        dark_end_time       = 12:23:05
        path                = /mnt/sdata/20250211/palila/

        [Slowcam]
        timestamp_dir       = /mnt/userdata/yjkim/20250211_betcmi/firstcam_timestamps/
        nbin                = 1

        [Output]
        outname             = /mnt/userdata/yjkim/timestamp_matched_palila/
        filename            = betcmi_20250211_matched

        [Options]
        verbose             = False
        show_plot            = True
        crop_width           = 20
        apply_dead_time_correction = True
        psfcam_is_fast       = True   # True: PSFcam=fastcam, crop applies
                                      # False: PSFcam=slowcam, no crop on fastcam frames
    '''

    config = ConfigObj(configname)

    # ── Support unified [Sort] section (new) and legacy [Fastcam]/[Output] (old) ──
    sort_sec = config.get('Sort', {})
    cameras  = config.get('Cameras', {})

    if sort_sec.get('fastcam_dir', '').strip():
        # Unified config format
        fastcam_dir        = sort_sec['fastcam_dir'].strip()
        fastcam_start_time = sort_sec.get('fastcam_start_time', '00:00:00').strip()
        fastcam_end_time   = sort_sec.get('fastcam_end_time',   '23:59:59').strip()
        fastcam_dark_file  = sort_sec.get('fastcam_dark_file',  '').strip()
        if not fastcam_dark_file:
            fastcam_dark_start_time = sort_sec.get('fastcam_dark_start', '').strip()
            fastcam_dark_end_time   = sort_sec.get('fastcam_dark_end',   '').strip()
        slowcam_timestamps_dir = sort_sec.get('slowcam_dir', '').strip()
        try:
            slowcam_nbin = int(sort_sec.get('slowcam_nbin', 1) or 1)
        except Exception:
            slowcam_nbin = 1
        # Output: [Sort].output is a full path; split into directory + stem
        sort_output = sort_sec.get('output', 'fastcam.h5').strip() or 'fastcam.h5'
        outname  = os.path.dirname(os.path.abspath(sort_output)) or '.'
        filename = os.path.splitext(os.path.basename(sort_output))[0]
        verbose    = str(sort_sec.get('verbose', 'False')).lower() == 'true'
        show_plot  = str(sort_sec.get('show_plot', 'False')).lower() == 'true'
        crop_width = int(sort_sec.get('crop_width', 20) or 20)
        apply_dtc  = str(sort_sec.get('apply_dead_time_correction', 'True')).lower() == 'true'
        # Derive psfcam_is_fast from [Cameras].fastcam_role
        fastcam_role = cameras.get('fastcam_role', 'PSF').strip().upper()
        psfcam_is_fast = (fastcam_role == 'PSF')
    else:
        # Legacy config format
        fastcam_dir = config['Fastcam']['path']
        fastcam_start_time = config['Fastcam']['start_time']
        fastcam_end_time = config['Fastcam']['end_time']

        fastcam_dark_file = config['Fastcam']['dark_file']
        if fastcam_dark_file.strip() == '':
            fastcam_dark_start_time = config['Fastcam']['dark_start_time']
            fastcam_dark_end_time = config['Fastcam']['dark_end_time']

        slowcam_timestamps_dir = config['Slowcam']['timestamp_dir']
        try:
            slowcam_nbin = int(config['Slowcam']['nbin'])
        except Exception:
            slowcam_nbin = 1

        outname = config['Output']['outname']
        filename = config['Output']['filename']

        verbose = config['Options']['verbose'].lower() == 'true'
        show_plot = config['Options']['show_plot'].lower() == 'true'
        crop_width = int(config['Options']['crop_width'])
        try:
            apply_dtc = config['Options']['apply_dead_time_correction'].lower() == 'true'
        except Exception:
            apply_dtc = True

    if sort_sec.get('fastcam_dir', '').strip():
        pass  # psfcam_is_fast already set from [Cameras].fastcam_role above
    else:
        try:
            psfcam_is_fast = config['Options']['psfcam_is_fast'].lower() == 'true'
        except Exception:
            psfcam_is_fast = True

    os.makedirs(outname, exist_ok=True)

    # find fastcam data
    fastcam_timestampfiles = find_data_between(fastcam_dir, fastcam_start_time,
                                                fastcam_end_time, footer='.txt')
    if fastcam_dark_file.strip() == '':
        fastcam_darkframes = find_data_between(fastcam_dir, fastcam_dark_start_time,
                                               fastcam_dark_end_time, footer='.fits')

    # find slowcam timestamp data
    slowcam_timestampfiles = np.sort(glob.glob(slowcam_timestamps_dir + '*.txt'))

    if verbose:
        print("fastcam timestampfiles:", fastcam_timestampfiles)
        print("slowcam timestampfiles:", slowcam_timestampfiles)

    ######################
    # Timestamp matching #
    ######################

    fc_ts_list, fc_fi_list = [], []
    for i, f in enumerate(fastcam_timestampfiles):
        ts, fi = _load_timestamp_file(f)
        fc_ts_list.append(ts); fc_fi_list.append(fi)

    fastcam_timestamp = np.concatenate(fc_ts_list)
    fastcam_frameinds = np.concatenate(fc_fi_list)
    fastcam_fileinds  = np.concatenate(
        [np.full(len(ts), i) for i, ts in enumerate(fc_ts_list)])

    sc_ts_list, sc_fi_list = [], []
    for f in slowcam_timestampfiles:
        ts, fi = _load_timestamp_file(f)
        sc_ts_list.append(ts); sc_fi_list.append(fi)

    slowcam_timestamp = np.concatenate(sc_ts_list)
    slowcam_frameinds = np.concatenate(sc_fi_list)
    slowcam_fileinds  = np.concatenate(
        [np.full(len(ts), i) for i, ts in enumerate(sc_ts_list)])

    # Apply slowcam nbin with warning
    if slowcam_nbin > 1:
        n_total = len(slowcam_timestamp)
        n_keep  = (n_total // slowcam_nbin) * slowcam_nbin
        n_drop  = n_total - n_keep
        if n_drop > 0:
            print("WARNING: slowcam_nbin=%d — dropping last %d timestamp(s) "
                  "that do not fill a complete bin." % (slowcam_nbin, n_drop))
        slowcam_timestamp = slowcam_timestamp[:n_keep].reshape(-1, slowcam_nbin)[:, 0]
        slowcam_fileinds  = slowcam_fileinds[:n_keep].reshape(-1, slowcam_nbin)[:, 0]
        slowcam_frameinds = slowcam_frameinds[:n_keep].reshape(-1, slowcam_nbin)[:, 0]

    # Bisect matching
    bisect_inds = [bisect(fastcam_timestamp, t) for t in slowcam_timestamp]
    bisect_arr  = np.array(bisect_inds)
    max_bisect  = int(bisect_arr.max())

    if max_bisect == 0:
        raise ValueError(
            "No fastcam timestamp is later than any slowcam timestamp. "
            "Check that both cameras cover the same time interval.")

    ind_end = int(np.argmax(bisect_arr == max_bisect))
    if ind_end == 0:
        raise ValueError(
            "ind_end resolved to 0: all bisect_inds equal the maximum (%d). "
            "The fastcam timestamps may not overlap with the slowcam timestamps."
            % max_bisect)

    ind_start = 0
    print("total slowcam images: %d" % (ind_end - ind_start))

    if show_plot:
        plt.figure(figsize=(12, 4))
        plt.plot(bisect_inds, 'o-', ms=1)
        plt.axvline(ind_start); plt.axvline(ind_end)
        plt.xlabel('slowcam frameinds'); plt.ylabel('fastcam frameinds')
        plt.title('Timestamp matching')
        plt.show()


    # Build matching dict
    frame_end_times = None
    if apply_dtc:
        frame_end_times = compute_frame_durations(fastcam_timestamp, fastcam_fileinds)
        print("Applied dead-time correction using frame end times")

    Dict, timestamps, n_skipped = _build_matching_dict(
        fastcam_timestamp, slowcam_timestamp, bisect_inds, ind_start, ind_end,
        frame_end_times=frame_end_times)

    if n_skipped > 0:
        print("WARNING: %d slowcam frame(s) skipped (timestamp before all "
              "fastcam timestamps)." % n_skipped)

    ############################
    # Load dark, set crop region
    ############################
    if fastcam_dark_file.strip() == '':
        fastcam_avgdark = np.average(fits.getdata(fastcam_darkframes[0]), axis=0)
    else:
        fastcam_avgdark = fits.getdata(fastcam_dark_file)

    fastcam_files = [f.replace('.txt', '.fits') for f in fastcam_timestampfiles]
    xwidth, ywidth = fastcam_avgdark.shape
    xc, yc = xwidth // 2, ywidth // 2

    # Crop only when fastcam IS the PSF camera (we only need the PSF core)
    if psfcam_is_fast and crop_width is not None:
        xw = yw = int(crop_width)
        print("PSFcam=fastcam — saving cropped frames [%d:%d, %d:%d]"
              % (xc-xw, xc+xw, yc-yw, yc+yw))
    else:
        xw, yw = xwidth // 2, ywidth // 2
        if not psfcam_is_fast:
            print("PSFcam=slowcam — fastcam contains PLcam data, saving full frames")
        else:
            print("No crop_width specified — saving full frames")

    fastcam_avgdark = fastcam_avgdark[xc-xw:xc+xw, yc-yw:yc+yw]

    ######################################
    # Write weighted-mean frames to giant H5
    ######################################
    indices = list(Dict.keys())
    nstacks = []
    h5_path = os.path.join(outname, filename + '.h5')

    current_fc_fileind = 0
    print("reading fastcam file", fastcam_files[current_fc_fileind])
    current_fc_data = (fits.getdata(fastcam_files[current_fc_fileind])
                       [:, xc-xw:xc+xw, yc-yw:yc+yw] - fastcam_avgdark)

    with h5py.File(h5_path, 'w') as fh:
        ds = fh.create_dataset('frames', shape=(len(indices), 2*xw, 2*yw),
                               dtype='float32', chunks=(1, 2*xw, 2*yw))

        for out_idx, ind in enumerate(tqdm(indices)):
            fc_inds = np.array(list(Dict[ind].keys()))
            fc_frac = np.array(list(Dict[ind].values()))

            frame_acc = np.zeros((2*xw, 2*yw), dtype='float64')
            w_total   = 0.0

            for _fi, _ff in zip(fc_inds, fc_frac):
                if fastcam_fileinds[_fi] != current_fc_fileind:
                    current_fc_fileind = fastcam_fileinds[_fi]
                    print("reading new fastcam file", fastcam_files[current_fc_fileind])
                    current_fc_data = (fits.getdata(fastcam_files[current_fc_fileind])
                                       [:, xc-xw:xc+xw, yc-yw:yc+yw] - fastcam_avgdark)
                if verbose:
                    print("frame %d  frac %.4f" % (fastcam_frameinds[_fi], _ff))
                frame_acc += _ff * current_fc_data[fastcam_frameinds[_fi]]
                w_total   += _ff

            ds[out_idx] = (frame_acc / w_total).astype('float32') if w_total > 0 \
                          else frame_acc.astype('float32')
            nstacks.append(w_total)

        fh.create_dataset('nstacks', data=np.array(nstacks, dtype='float32'))
        fh.create_dataset('timestamps', data=np.array(timestamps, dtype='float64'))

        meta = {
            'fastcam_timestampfiles': list(fastcam_timestampfiles),
            'slowcam_timestampfiles': [str(f) for f in slowcam_timestampfiles],
            'fastcam_fileinds':  fastcam_fileinds.tolist(),
            'slowcam_fileinds':  slowcam_fileinds.tolist(),
            'fastcam_frameinds': fastcam_frameinds.tolist(),
            'slowcam_frameinds': slowcam_frameinds.tolist(),
            'matched_indices':   {str(k): {str(kk): float(vv) for kk, vv in v.items()}
                                  for k, v in Dict.items()},
            'timestamps':        [float(t) for t in timestamps],
            'slowcam_nbin':      slowcam_nbin,
            'dead_time_corrected': apply_dtc,
            'psfcam_is_fast':    psfcam_is_fast,
        }
        fh.create_dataset('metadata', data=json.dumps(meta))

    print("saved: %s  (%d frames)" % (h5_path, len(indices)))

    # Save pickle backup
    pkl_path = os.path.join(outname, filename + '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'fastcam_timestampfiles': fastcam_timestampfiles,
            'slowcam_timestampfiles': list(slowcam_timestampfiles),
            'fastcam_fileinds':  fastcam_fileinds,
            'slowcam_fileinds':  slowcam_fileinds,
            'fastcam_frameinds': fastcam_frameinds,
            'slowcam_frameinds': slowcam_frameinds,
            'matched_indices':   Dict,
            'timestamps':        timestamps,
            'nstacks':           nstacks,
        }, f)
    print("timestamp matching backup saved to %s" % pkl_path)

    fig = plot_matching_diagnostics(h5_path)
    fig.savefig(os.path.join(outname, filename + '_diagnostics.png'))

    return h5_path


# ---------------------------------------------------------------------------
# Diagnostics: verify timestamp matching quality
# ---------------------------------------------------------------------------

def plot_matching_diagnostics(h5_path, max_weight_frames=5000):
    '''
    Load a matched-timestamps HDF5 file and produce four diagnostic plots.

    Plots
    -----
    1. Raw frame intervals vs corrected frame durations
    2. w_total (nstacks) timeseries
    3. Individual weight histogram (all values must be in [0, 1])
    4. Boundary-weight scatter (leftfrac / rightfrac vs slowcam frame index)
    '''
    with h5py.File(h5_path, 'r') as fh:
        nstacks = fh['nstacks'][:]
        meta    = json.loads(fh['metadata'][()])

    Dict = {int(k): {int(kk): vv for kk, vv in v.items()}
            for k, v in meta['matched_indices'].items()}

    fastcam_timestamp = np.array([], dtype='float64')
    fastcam_fileinds  = np.array([], dtype=int)
    for i, ts_file in enumerate(meta['fastcam_timestampfiles']):
        ts, _ = _load_timestamp_file(ts_file)
        fastcam_timestamp = np.concatenate([fastcam_timestamp, ts])
        fastcam_fileinds  = np.concatenate([fastcam_fileinds,
                                             np.full(len(ts), i, dtype=int)])

    frame_end_times = compute_frame_durations(fastcam_timestamp, fastcam_fileinds)
    raw_intervals   = np.diff(fastcam_timestamp)
    corrected_durs  = frame_end_times[:-1] - fastcam_timestamp[:-1]
    boundary_inds   = np.where(np.diff(fastcam_fileinds) != 0)[0]

    slowcam_keys = sorted(Dict.keys())
    sample_keys  = slowcam_keys[:max_weight_frames]

    all_weights = []
    leftfracs, rightfracs = [], []
    for key in sample_keys:
        vals = list(Dict[key].values())
        all_weights.extend(vals)
        if len(vals) >= 2:
            leftfracs.append(vals[0]); rightfracs.append(vals[-1])
        elif len(vals) == 1:
            leftfracs.append(vals[0]); rightfracs.append(vals[0])

    all_weights = np.array(all_weights)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Timestamp matching diagnostics\n%s' % os.path.basename(h5_path),
                 fontsize=11)

    ax = axes[0, 0]
    ax.plot(raw_intervals * 1e3, color='steelblue', lw=0.7, alpha=0.7,
            label='raw diff(timestamp) [ms]')
    ax.plot(corrected_durs * 1e3, color='tomato', lw=0.8, alpha=0.9,
            label='corrected duration [ms]')
    for bi in boundary_inds:
        ax.axvline(bi, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('fastcam frame index')
    ax.set_ylabel('duration (ms)')
    ax.set_title('Frame durations: raw vs dead-time corrected')
    ax.legend(fontsize=8)
    p99 = np.percentile(raw_intervals, 99) * 1e3
    ax.set_ylim(0, min(p99 * 5, raw_intervals.max() * 1e3 * 1.05))

    ax = axes[0, 1]
    ax.plot(nstacks, lw=0.7, color='steelblue')
    ax.axhline(np.median(nstacks), color='tomato', ls='--', lw=1,
               label='median = %.3f' % np.median(nstacks))
    ax.set_xlabel('slowcam frame index')
    ax.set_ylabel('w_total')
    ax.set_title('w_total per matched frame')
    sc_ts_all = np.array(meta['timestamps'])
    if len(sc_ts_all) > 1:
        median_fc_dur = np.median(corrected_durs)
        expected = np.median(np.diff(sc_ts_all)) / median_fc_dur
        ax.axhline(expected, color='green', ls=':', lw=1,
                   label='expected ≈ %.1f' % expected)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    out_of_range = int(np.sum((all_weights < 0) | (all_weights > 1)))
    ax.hist(all_weights, bins=50, color='steelblue', edgecolor='white', lw=0.3)
    ax.set_xlabel('weight value')
    ax.set_ylabel('count')
    ax.set_title('Individual weight histogram (all should be in [0, 1])')
    ax.text(0.98, 0.95, 'out-of-range: %d' % out_of_range,
            ha='right', va='top', transform=ax.transAxes,
            color='red' if out_of_range > 0 else 'green', fontsize=9)

    ax = axes[1, 1]
    sc_x = np.arange(len(leftfracs))
    ax.scatter(sc_x, leftfracs,  s=4, alpha=0.5, color='steelblue',  label='leftfrac')
    ax.scatter(sc_x, rightfracs, s=4, alpha=0.5, color='tomato',     label='rightfrac')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axhline(1, color='k', lw=0.5, ls='--')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('slowcam frame index (first %d)' % len(sample_keys))
    ax.set_ylabel('fraction')
    ax.set_title('Boundary weights (leftfrac / rightfrac)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    outpng = os.path.splitext(h5_path)[0] + '_diagnostics.png'
    plt.savefig(outpng)
    print("Diagnostics saved to %s" % outpng)

    print("=== Diagnostic summary: %s ===" % os.path.basename(h5_path))
    print("fastcam frames:         %d  (%d files)"
          % (len(fastcam_timestamp), len(meta['fastcam_timestampfiles'])))
    print("slowcam matched frames: %d" % len(slowcam_keys))
    if len(boundary_inds):
        print("Raw interval at boundaries (ms):       %s"
              % np.round(raw_intervals[boundary_inds] * 1e3, 2))
        print("Corrected duration at boundaries (ms): %s"
              % np.round(corrected_durs[boundary_inds] * 1e3, 2))
    print("Weight range: [%.4f, %.4f]  (out-of-range: %d)"
          % (all_weights.min(), all_weights.max(), out_of_range))
    print("w_total: median=%.3f  min=%.3f  max=%.3f"
          % (np.median(nstacks), nstacks.min(), nstacks.max()))

    return fig


# ---------------------------------------------------------------------------
# Bin and write PL + PSF frames to a consolidated h5 file
# ---------------------------------------------------------------------------

def bin_by_centroids_to_h5(
    outname,
    psfcamframes,
    centroids, xbins, ybins,
    plcam_file_indices=None,
    plcam_frame_indices=None,
    plcam_files=None,
    ny=None, nx=None,
    plcam_frames_mmap=None,
    nbin=1,
    timestamps=None,
):
    '''
    Bin PL camera frames by PSF centroid position and write all bins into a
    single HDF5 file at ``outname + ".h5"``.

    H5 structure:
        outname.h5
          attrs: xbins, ybins
          /bin_i_j/
            attrs: num_frames, xbin_min, ybin_min
            rawframes  (num_frames, ny, nx)  int32, chunked by frame
            psfframes  (num_frames, h, w)    float32
            peaks      (num_frames,)         float32
            centers    (num_frames, 2)       float32
            timestamps (num_frames,)         float64  [if provided]
            fileinds   (num_frames,)         int64    [file-based path only]
            frameinds  (num_frames,)         int64    [file-based path only]

    Supply exactly one of ``plcam_files`` (file-based, PSFcam=fast) or
    ``plcam_frames_mmap`` (mmap-based, PSFcam=slow).

    Parameters
    ----------
    outname              : str  path prefix (``outname + ".h5"`` is created)
    psfcamframes         : ndarray (N, h, w)
    centroids            : ndarray (N, 2)
    xbins, ybins         : ndarray  bin edges
    plcam_file_indices   : ndarray int
    plcam_frame_indices  : ndarray int
    plcam_files          : list of str
    ny, nx               : int  PL frame shape (required for file-based path)
    plcam_frames_mmap    : numpy memmap (N, ny, nx)
    nbin                 : int  PL camera on-chip binning
    timestamps           : ndarray float64 (N,) optional

    Returns
    -------
    psfcam_binned : ndarray (nb_x, nb_y, h, w)
    num_frames    : ndarray (nb_x, nb_y)
    frame_to_bin  : ndarray int (N, 2)  per-frame (iy, ix); -1 for out-of-grid
    '''
    use_mmap  = plcam_frames_mmap is not None
    use_files = plcam_files is not None

    if use_mmap == use_files:
        raise ValueError("Supply exactly one of plcam_frames_mmap or plcam_files.")

    x, y  = centroids[:, 0], centroids[:, 1]
    nb_x, nb_y = len(xbins) - 1, len(ybins) - 1
    N = len(centroids)

    if use_mmap:
        ny, nx = plcam_frames_mmap.shape[1], plcam_frames_mmap.shape[2]

    # First pass: compute bin membership
    frame_to_bin = np.full((N, 2), -1, dtype=np.int32)
    num_frames   = np.zeros((nb_x, nb_y), dtype=np.int64)

    for i in range(nb_x):
        for j in range(nb_y):
            mask = ((x >= xbins[i]) & (x < xbins[i+1]) &
                    (y >= ybins[j]) & (y < ybins[j+1]))
            frame_to_bin[mask, 0] = i
            frame_to_bin[mask, 1] = j
            num_frames[i, j]      = mask.sum()

    # Per-bin PSF averages
    psfcam_binned = np.zeros((nb_x, nb_y,
                               psfcamframes.shape[1], psfcamframes.shape[2]))
    for i in range(nb_x):
        for j in range(nb_y):
            mask = (frame_to_bin[:, 0] == i) & (frame_to_bin[:, 1] == j)
            if num_frames[i, j] > 0:
                psfcam_binned[i, j] = np.mean(psfcamframes[mask], axis=0)

    # Create h5 with pre-allocated datasets
    h5_path = outname + '.h5'
    print("Creating consolidated h5: %s" % h5_path)

    with h5py.File(h5_path, 'w') as h5f:
        h5f.attrs['xbins'] = xbins
        h5f.attrs['ybins'] = ybins

        for i in range(nb_x):
            for j in range(nb_y):
                nf = int(num_frames[i, j])
                if nf == 0:
                    continue

                mask = (frame_to_bin[:, 0] == i) & (frame_to_bin[:, 1] == j)
                grp  = h5f.create_group('bin_%d_%d' % (i, j))
                grp.attrs['num_frames'] = nf
                grp.attrs['xbin_min']   = float(xbins[i])
                grp.attrs['ybin_min']   = float(ybins[j])

                grp.create_dataset('psfframes',
                                   data=psfcamframes[mask].astype('float32'))
                grp.create_dataset('peaks',
                                   data=np.nanmax(psfcamframes[mask],
                                                  axis=(1, 2)).astype('float32'))
                grp.create_dataset('centers',
                                   data=np.column_stack(
                                       [x[mask], y[mask]]).astype('float32'))
                if timestamps is not None:
                    grp.create_dataset('timestamps',
                                       data=timestamps[mask].astype('float64'))
                if use_files:
                    grp.create_dataset('fileinds',
                                       data=plcam_file_indices[mask].astype('int64'))
                    grp.create_dataset('frameinds',
                                       data=plcam_frame_indices[mask].astype('int64'))

                grp.create_dataset(
                    'rawframes',
                    shape=(nf, ny, nx),
                    dtype='int32',
                    chunks=(1, ny, nx),
                )

    # Second pass: write PL frames into pre-allocated datasets
    write_idx = {(i, j): 0
                 for i in range(nb_x) for j in range(nb_y)
                 if num_frames[i, j] > 0}

    if use_files:
        with h5py.File(h5_path, 'r+') as h5f:
            for fileind, fpath in enumerate(tqdm(plcam_files, desc='Writing PL frames')):
                print('Reading PL file %d: %s' % (fileind, fpath))
                _dat = fits.getdata(fpath)
                file_mask   = (plcam_file_indices == fileind)
                global_inds = np.where(file_mask)[0]

                for g_idx in global_inds:
                    iy = int(frame_to_bin[g_idx, 0])
                    ix = int(frame_to_bin[g_idx, 1])
                    if iy < 0:
                        continue
                    frame_idx = int(plcam_frame_indices[g_idx])
                    wi = write_idx[(iy, ix)]
                    try:
                        raw = _dat[frame_idx] if nbin == 1 else \
                              np.mean(_dat[frame_idx:frame_idx + nbin], axis=0)
                        h5f['bin_%d_%d' % (iy, ix)]['rawframes'][wi] = raw
                        write_idx[(iy, ix)] += 1
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print("  Failed frame %d from file %d: %s" % (frame_idx, fileind, e))
    else:
        with h5py.File(h5_path, 'r+') as h5f:
            for g_idx in tqdm(range(N), desc='Writing PL frames'):
                iy = int(frame_to_bin[g_idx, 0])
                ix = int(frame_to_bin[g_idx, 1])
                if iy < 0:
                    continue
                wi = write_idx[(iy, ix)]
                h5f['bin_%d_%d' % (iy, ix)]['rawframes'][wi] = \
                    plcam_frames_mmap[g_idx]
                write_idx[(iy, ix)] += 1

    print("Consolidated h5 written to %s" % h5_path)
    return psfcam_binned, num_frames, frame_to_bin


# ---------------------------------------------------------------------------
# Assemble response maps from per-bin spectral h5 files
# ---------------------------------------------------------------------------

def make_responsemaps(filename, footer='_spec', nfib=38, nwav=200,
                      psfframe_shape=(40, 40), nboot=50):
    '''
    Assemble coupling-map FITS from per-bin spectral h5 files and the
    consolidated raw-frame h5.

    Reads
    -----
    {filename}.h5                    – consolidated h5 (psfframes per bin)
    {filename}_bin_i_j{footer}.h5    – per-bin spectral h5 files (from visPLred)
    {filename}_info.json             – grid info written by FrameSorter
    '''
    info    = json.load(open(filename + '_info.json', 'r'))
    map_n   = info['map_n']
    map_w   = info['map_w']
    pix2mas = info['pix2mas']

    pos_mas = np.linspace(-map_w / 2, map_w / 2, map_n) * pix2mas

    specs       = np.full((map_n, map_n, nfib, nwav), np.nan)
    bootspecs   = np.full((nboot, map_n, map_n, nfib, nwav), np.nan)
    nframes_map = np.zeros((map_n, map_n))
    psfframes   = np.full((map_n, map_n, *psfframe_shape), np.nan)

    print("Reading spectral h5 files")
    for i in tqdm(range(map_n)):
        for j in range(map_n):
            spec_file = filename + '_bin_%d_%d%s.h5' % (i, j, footer)
            if os.path.exists(spec_file):
                with h5py.File(spec_file, 'r') as f:
                    specs[i, j]        = f['avgspec'][:]
                    bootspecs[:, i, j] = f['bootspecs'][:]
                    nframes_map[i, j]  = f.attrs['num_frames']

    print("Reading PSF frames from consolidated h5")
    raw_h5 = filename + '.h5'
    if os.path.exists(raw_h5):
        with h5py.File(raw_h5, 'r') as h5f:
            for i in range(map_n):
                for j in range(map_n):
                    grp_name = 'bin_%d_%d' % (i, j)
                    if grp_name in h5f:
                        psfframes[i, j] = np.nanmean(
                            h5f[grp_name]['psfframes'][:], axis=0)

    total = np.nansum(specs, axis=(0, 1))
    normspecs = specs / total

    boot_total    = np.nansum(bootspecs, axis=(1, 2))
    normbootspecs = bootspecs / boot_total[:, None, None, :, :]

    specs_var     = np.nanvar(bootspecs,     axis=0)
    normspecs_var = np.nanvar(normbootspecs, axis=0)

    header = fits.Header()
    header['MAP_N'] = map_n
    header['MAP_W'] = map_w
    header['XMIN']  = float(min(pos_mas))
    header['XMAX']  = float(max(pos_mas))
    header['YMIN']  = float(min(pos_mas))
    header['YMAX']  = float(max(pos_mas))

    hdul = fits.HDUList([
        fits.PrimaryHDU(specs,         header=header),
        fits.ImageHDU(nframes_map,     name='nframes'),
        fits.ImageHDU(psfframes,       name='psfcam'),
        fits.ImageHDU(specs_var,       name='var'),
        fits.ImageHDU(normspecs_var,   name='normvar'),
        fits.ImageHDU(normspecs,       name='normspec'),
    ])
    out_fits = filename + '_couplingmap.fits'
    hdul.writeto(out_fits, overwrite=True)
    print("Coupling map saved to %s" % out_fits)


# ---------------------------------------------------------------------------
# FrameSorter: orchestrates steps 2a / 2b for a full observation
# ---------------------------------------------------------------------------

class FrameSorter:
    '''
    Load matched fast/slow camera data, compute PSF centroids, and sort
    PL camera frames into spatial bins.

    Parameters
    ----------
    slowcam_timestamp_path     : str  directory with slowcam .txt timestamp files
    slowcam_data_path          : str  directory with slowcam FITS data files
    timestamp_matching_h5_name : str  path to the Step 1 output .h5 file
    obs_start, obs_end         : str  "%H:%M:%S"
    match_frames               : bool  call match_frames on init
    psfcam_is_fastcam          : bool  True if PSF camera is the fast camera
    psfcam_darkfile            : str   optional dark FITS for the PSF camera
    pix2mas                    : float plate scale mas/pixel of PSF camera
    slowcam_header             : str   file prefix of slowcam files
    slowcam_footer             : str   file extension of slowcam data files
    slowcam_shape              : tuple (ny, nx) of one slowcam frame
    '''

    normvar = None
    var     = None

    def __init__(self,
                 slowcam_timestamp_path,
                 slowcam_data_path,
                 timestamp_matching_h5_name,
                 obs_start, obs_end,
                 match_frames=True,
                 psfcam_is_fastcam=True,
                 psfcam_darkfile=None,
                 pix2mas=16.2,
                 slowcam_header='firstpl_',
                 slowcam_footer='.fits',
                 slowcam_shape=(412, 1896),
                 ):
        self.slowcam_timestamp_path      = slowcam_timestamp_path
        self.slowcam_data_path           = slowcam_data_path
        self.timestamp_matching_h5_name  = timestamp_matching_h5_name
        self.obs_start = obs_start
        self.obs_end   = obs_end
        self.ny, self.nx = slowcam_shape
        self.pix2mas        = pix2mas
        self.psfcam_is_fastcam = psfcam_is_fastcam

        if psfcam_darkfile is not None:
            self.psfcam_dark = fits.getdata(psfcam_darkfile)

        if match_frames:
            if self.psfcam_is_fastcam:
                self.match_frames(header=slowcam_header, footer=slowcam_footer)
            else:
                self.match_frames2(header=slowcam_header, footer=slowcam_footer)

    def match_frames(self, header='firstpl_', footer='.fits'):
        '''
        Match PL camera (slow) to the pre-averaged PSF camera (fast).

        Reads PSF frames from the Step 1 H5, then aligns PL camera FITS files
        to the same matched timestamps.

        After this call:
            self.psfcam_frames       ndarray (N, h, w)
            self.plcam_files         list of FITS paths
            self.plcam_file_indices  ndarray int (N,)
            self.plcam_frame_indices ndarray int (N,)
            self.timestamps          ndarray float64 (N,)
        '''
        plcam_ts_files = find_data_between(
            self.slowcam_timestamp_path, self.obs_start, self.obs_end,
            header=header, footer='.txt')
        plcam_data_files = find_data_between(
            self.slowcam_data_path, self.obs_start, self.obs_end,
            header=header, footer=footer)

        timestamps_plcam = np.concatenate(
            [_load_timestamp_file(f)[0] for f in plcam_ts_files])

        with h5py.File(self.timestamp_matching_h5_name, 'r') as f:
            psfcam_frames = f['frames'][:]
            meta = json.loads(f['metadata'][()])
            matched_timestamps = np.array(meta['timestamps'])

        # Build file/frame indices for PL camera
        file_inds, frame_inds = [], []
        for fi, f in enumerate(plcam_data_files):
            nf = fits.getheader(f)['NAXIS3']
            file_inds.extend([fi] * nf)
            frame_inds.extend(range(nf))
        file_inds  = np.array(file_inds)
        frame_inds = np.array(frame_inds)

        idx_plcam, idx_psf = validate_timestamp_matching(
            timestamps_plcam, matched_timestamps)

        self.plcam_file_indices  = file_inds[idx_plcam]
        self.plcam_frame_indices = frame_inds[idx_plcam]
        self.plcam_files         = plcam_data_files
        self.psfcam_frames       = np.asarray(psfcam_frames)[idx_psf]
        self.timestamps          = matched_timestamps[idx_psf]

    def match_frames2(self, header='palila_', footer='.fits', crop_width=20):
        '''
        Match PSF camera (slow) to the pre-averaged PL camera (fast).

        Reads PL frames from the Step 1 H5 (averaged fast PLcam frames),
        then loads the PSF camera FITS files (slowcam) and aligns them.

        After this call:
            self.psfcam_frames      ndarray (N, h, w)  dark-subtracted, cropped
            self.plcam_frames_mmap  ndarray (N, ny, nx)  from H5
            self.timestamps         ndarray float64 (N,)
        '''
        psfcam_ts_files = find_data_between(
            self.slowcam_timestamp_path, self.obs_start, self.obs_end,
            header=header, footer='.txt')
        psfcam_data_files = find_data_between(
            self.slowcam_data_path, self.obs_start, self.obs_end,
            header=header, footer=footer)

        timestamps_psfcam = np.concatenate(
            [_load_timestamp_file(f)[0] for f in psfcam_ts_files])

        # Load PL frames from Step 1 H5 (fastcam = PLcam in this mode)
        with h5py.File(self.timestamp_matching_h5_name, 'r') as f:
            plcam_matched_frames = f['frames'][:]
            meta = json.loads(f['metadata'][()])
            matched_timestamps = np.array(meta['timestamps'])

        cw = crop_width
        dark_crop = self.psfcam_dark[
            self.psfcam_dark.shape[0] // 2 - cw:self.psfcam_dark.shape[0] // 2 + cw,
            self.psfcam_dark.shape[1] // 2 - cw:self.psfcam_dark.shape[1] // 2 + cw]

        # Load PSF frames (slowcam, fits in memory)
        psfcam_frames = []
        for f in psfcam_data_files:
            data = fits.getdata(f)
            h, w = data.shape[1], data.shape[2]
            cropped = data[:, h // 2 - cw:h // 2 + cw, w // 2 - cw:w // 2 + cw]
            psfcam_frames.append(cropped)
        psfcam_frames = np.concatenate(psfcam_frames, axis=0).astype('float32')
        psfcam_frames -= dark_crop

        idx_psf, idx_plcam = validate_timestamp_matching(
            timestamps_psfcam, matched_timestamps)

        self.psfcam_frames     = psfcam_frames[idx_psf]
        self.plcam_frames_mmap = plcam_matched_frames[idx_plcam]
        self.timestamps        = matched_timestamps[idx_plcam]

    def compute_psfcam_centroids(self, peak=True):
        '''
        Compute sub-pixel centroid for each PSF frame.
        Results stored in self.centroids (N, 2).
        '''
        centroids = []
        for frame in self.psfcam_frames:
            try:
                cent = subpixel_centroid_2d(frame) if peak else center_of_mass(frame)
            except Exception:
                cent = (np.nan, np.nan)
            centroids.append(cent)
        self.centroids = np.array(centroids)

    def bin_by_centroids(self, map_n, map_width,
                         effective_idx=None,
                         xc=None, yc=None,
                         plot=True,
                         to_file=False,
                         filename=None,
                         nbin=1):
        '''
        Sort all frames into a (map_n × map_n) spatial grid and optionally
        write PL frames to a consolidated h5.

        Parameters
        ----------
        map_n         : int    number of grid cells along each axis
        map_width     : float  total grid width in PSF-camera pixels
        effective_idx : ndarray bool or int  subset of frames to use
        xc, yc        : float  grid centre (pixels); defaults to sigma-clipped median
        plot          : bool   save / show frame-count map
        to_file       : bool   write PL frames to h5 (requires ``filename``)
        filename      : str    output path prefix when ``to_file=True``
        nbin          : int    on-chip PL binning factor used during capture

        Returns
        -------
        psfcam_binned : ndarray (map_n, map_n, h, w)
        num_frames    : ndarray (map_n, map_n)
        idxs_or_f2b   : ndarray  idxs (in-memory) or frame_to_bin (to_file)
        '''
        from astropy.stats import sigma_clip

        if effective_idx is not None:
            centroids  = self.centroids[effective_idx]
            psfcam_frm = self.psfcam_frames[effective_idx]
            timestamps = self.timestamps[effective_idx]
            if self.psfcam_is_fastcam:
                pfi = self.plcam_file_indices[effective_idx]
                pfr = self.plcam_frame_indices[effective_idx]
            else:
                plmm = self.plcam_frames_mmap[effective_idx]
        else:
            centroids  = self.centroids
            psfcam_frm = self.psfcam_frames
            timestamps = self.timestamps
            if self.psfcam_is_fastcam:
                pfi = self.plcam_file_indices
                pfr = self.plcam_frame_indices
            else:
                plmm = self.plcam_frames_mmap

        self.map_n     = map_n
        self.map_width = map_width

        if xc is None:
            cl = sigma_clip(centroids[:, 0], sigma=3, maxiters=5)
            xc = float(np.nanmedian(cl.data[~cl.mask]))
        if yc is None:
            cl = sigma_clip(centroids[:, 1], sigma=3, maxiters=5)
            yc = float(np.nanmedian(cl.data[~cl.mask]))

        xbins = np.linspace(xc - map_width / 2, xc + map_width / 2, map_n + 1)
        ybins = np.linspace(yc - map_width / 2, yc + map_width / 2, map_n + 1)

        bin_centres_x = (xbins[:-1] + np.diff(xbins) / 2) - xc
        bin_centres_y = (ybins[:-1] + np.diff(ybins) / 2) - yc
        self.x_mas = bin_centres_x * self.pix2mas
        self.y_mas = bin_centres_y * self.pix2mas
        self.xbins = xbins
        self.ybins = ybins
        self.xmin  = (xbins[0]  - xc) * self.pix2mas
        self.xmax  = (xbins[-1] - xc) * self.pix2mas
        self.ymin  = (ybins[0]  - yc) * self.pix2mas
        self.ymax  = (ybins[-1] - yc) * self.pix2mas

        if not to_file:
            psfcam_binned, num_frames, idxs = bin_by_centroids_from_indices(
                psfcam_frm, centroids, xbins, ybins)
            result = idxs
        else:
            assert filename is not None, "filename required when to_file=True"
            infodict = {
                'xmin': self.xmin, 'ymin': self.ymin,
                'xmax': self.xmax, 'ymax': self.ymax,
                'map_n': map_n, 'map_w': map_width,
                'pix2mas': self.pix2mas, 'nbin': nbin,
                'xc': xc, 'yc': yc,
            }
            json.dump(infodict, open(filename + '_info.json', 'w'))
            print("Info saved to %s" % (filename + '_info.json'))

            if self.psfcam_is_fastcam:
                psfcam_binned, num_frames, result = bin_by_centroids_to_h5(
                    filename, psfcam_frm, centroids, xbins, ybins,
                    plcam_file_indices=pfi,
                    plcam_frame_indices=pfr,
                    plcam_files=self.plcam_files,
                    ny=self.ny, nx=self.nx,
                    nbin=nbin,
                    timestamps=timestamps,
                )
            else:
                psfcam_binned, num_frames, result = bin_by_centroids_to_h5(
                    filename, psfcam_frm, centroids, xbins, ybins,
                    plcam_frames_mmap=plmm,
                    nbin=nbin,
                    timestamps=timestamps,
                )

        self.psfcam_binned_frames = psfcam_binned
        self.num_frames           = num_frames

        if plot:
            plt.figure(figsize=(5, 5))
            plt.imshow(num_frames, origin='upper',
                       extent=(self.xmin, self.xmax, self.ymin, self.ymax))
            plt.xlabel('x (mas)'); plt.ylabel('y (mas)')
            plt.colorbar(); plt.title('Number of frames averaged')
            if to_file and filename:
                plt.savefig(filename + '_num_frames.png')
                print("Saved plot to %s" % (filename + '_num_frames.png'))
            else:
                plt.show()

        return psfcam_binned, num_frames, result
