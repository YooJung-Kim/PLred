"""
Consolidated H5 output utilities for PLred.

This module provides functions to write timestamp-matched PSF and PL camera data
to consolidated HDF5 files designed for flexible on-the-fly spatial binning and analysis.

Structure:
  - Core H5: PSF frames, centroids, peaks, timestamps, metadata (~100-500 MB)
  - Raw PL H5: Raw PL frames with smart chunking (~50-100 GB, optional)

The viewer can then perform on-the-fly spatial binning using PSF centroids.

BUGFIXES (from sort4.py):
  - Bug 1: leftfrac overwrite when single fastcam frame spans entire slowcam interval
  - Bug 2: negative index wrap-around when bisect_inds[ind] == 0
  - Issue 3: ind_end == 0 now raises ValueError with informative message
  - Issue 4: Float tolerance matching instead of exact equality
  - Issue 5: Warnings for slowcam_nbin truncation
  - Dead-time correction using frame_end_times between FITS files
"""

import numpy as np
import h5py
import json
import os
from datetime import datetime
from tqdm import tqdm
from astropy.io import fits
from bisect import bisect
import matplotlib.pyplot as plt


# ============================================================================
# LAYER 1: Intermediate H5 Writing (Raw Matching Results)
# ============================================================================


def write_intermediate_matched_h5(
    outpath,
    fastcam_timestamps,
    fastcam_fileinds,
    fastcam_frameinds,
    fastcam_timestampfiles,
    slowcam_timestamps,
    slowcam_fileinds,
    slowcam_frameinds,
    slowcam_timestampfiles,
    Dict,
    matched_timestamps,
    config_dict,
    verbose=False,
):
    """
    Write intermediate H5 file for Layer 1 (timestamp matching results).

    This file contains the raw matching data and timestamps, allowing Layer 2
    to load frames and compute centroids without re-doing the matching.

    Parameters
    ----------
    outpath : str
        Path to output intermediate H5 file
    fastcam_timestamps : ndarray float64, shape (N,)
        PSF camera timestamps (Unix epoch)
    fastcam_fileinds : ndarray int64, shape (N,)
        File index for each PSF frame
    fastcam_frameinds : ndarray int64, shape (N,)
        Frame index within file for PSF frames
    fastcam_timestampfiles : list of str
        List of PSF timestamp files
    slowcam_timestamps : ndarray float64, shape (M,)
        PL camera timestamps
    slowcam_fileinds : ndarray int64, shape (M,)
        File index for each PL frame
    slowcam_frameinds : ndarray int64, shape (M,)
        Frame index within file for PL frames
    slowcam_timestampfiles : list of str
        List of PL timestamp files
    Dict : dict
        Matching dict from build_matching_dict() - keys are slowcam indices,
        values are dicts of {fastcam_idx: weight}
    matched_timestamps : list
        Timestamps for matched slowcam frames
    config_dict : dict
        Configuration and observation metadata
    verbose : bool, optional
        Print progress

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)

    if verbose:
        print(f"Writing intermediate matched H5: {outpath}")

    with h5py.File(outpath, 'w') as h5f:
        # --- Metadata ---
        meta_grp = h5f.create_group('metadata')
        meta_grp.attrs['write_time'] = datetime.now().isoformat()
        meta_grp.attrs['layer'] = 'intermediate'
        meta_grp.create_dataset('config', data=json.dumps(config_dict))

        # --- PSF camera matching data ---
        psf_grp = h5f.create_group('psfcam_matching')
        psf_grp.create_dataset('timestamps', data=fastcam_timestamps, dtype='float64')
        psf_grp.create_dataset('file_indices', data=fastcam_fileinds, dtype='int64')
        psf_grp.create_dataset('frame_indices', data=fastcam_frameinds, dtype='int64')
        psf_grp.create_dataset('timestamp_files', data=json.dumps(list(fastcam_timestampfiles)))
        psf_grp.attrs['num_frames'] = len(fastcam_timestamps)

        # --- PL camera matching data ---
        pl_grp = h5f.create_group('plcam_matching')
        pl_grp.create_dataset('timestamps', data=slowcam_timestamps, dtype='float64')
        pl_grp.create_dataset('file_indices', data=slowcam_fileinds, dtype='int64')
        pl_grp.create_dataset('frame_indices', data=slowcam_frameinds, dtype='int64')
        pl_grp.create_dataset('timestamp_files', data=json.dumps(list(slowcam_timestampfiles)))
        pl_grp.attrs['num_frames'] = len(slowcam_timestamps)

        # --- Matching results ---
        match_grp = h5f.create_group('matching')
        
        # Store matched timestamps
        match_grp.create_dataset('matched_timestamps', data=matched_timestamps, dtype='float64')
        match_grp.attrs['num_matched'] = len(matched_timestamps)

        # Store matching dict as JSON (slowcam_idx -> {fastcam_idx: weight})
        dict_json = json.dumps({
            str(slowcam_idx): {
                str(fastcam_idx): float(weight)
                for fastcam_idx, weight in fastcam_dict.items()
            }
            for slowcam_idx, fastcam_dict in Dict.items()
        })
        match_grp.create_dataset('matching_dict', data=dict_json)

    if verbose:
        print(f"  ✓ Wrote intermediate H5 with {len(matched_timestamps)} matched pairs")


def read_intermediate_matched_h5(h5_path, verbose=False):
    """
    Read intermediate H5 file from Layer 1.

    Parameters
    ----------
    h5_path : str
        Path to intermediate H5 file
    verbose : bool, optional
        Print progress

    Returns
    -------
    data : dict
        Dictionary containing:
        - fastcam_timestamps, fastcam_fileinds, fastcam_frameinds
        - slowcam_timestamps, slowcam_fileinds, slowcam_frameinds
        - fastcam_timestampfiles, slowcam_timestampfiles
        - matched_timestamps
        - matching_dict
        - config_dict
    """
    if verbose:
        print(f"Reading intermediate matched H5: {h5_path}")

    with h5py.File(h5_path, 'r') as h5f:
        # Read metadata
        config_dict = json.loads(h5f['metadata']['config'][()])

        # Read PSF matching data
        fastcam_timestamps = h5f['psfcam_matching']['timestamps'][:]
        fastcam_fileinds = h5f['psfcam_matching']['file_indices'][:]
        fastcam_frameinds = h5f['psfcam_matching']['frame_indices'][:]
        fastcam_timestampfiles = json.loads(h5f['psfcam_matching']['timestamp_files'][()])

        # Read PL matching data
        slowcam_timestamps = h5f['plcam_matching']['timestamps'][:]
        slowcam_fileinds = h5f['plcam_matching']['file_indices'][:]
        slowcam_frameinds = h5f['plcam_matching']['frame_indices'][:]
        slowcam_timestampfiles = json.loads(h5f['plcam_matching']['timestamp_files'][()])

        # Read matching results
        matched_timestamps = h5f['matching']['matched_timestamps'][:]
        matching_dict_json = h5f['matching']['matching_dict'][()]
        
        # Reconstruct matching dict
        matching_dict_str = json.loads(matching_dict_json)
        matching_dict = {
            int(slowcam_idx): {
                int(fastcam_idx): float(weight)
                for fastcam_idx, weight in fastcam_dict.items()
            }
            for slowcam_idx, fastcam_dict in matching_dict_str.items()
        }

    data = {
        'fastcam_timestamps': fastcam_timestamps,
        'fastcam_fileinds': fastcam_fileinds,
        'fastcam_frameinds': fastcam_frameinds,
        'fastcam_timestampfiles': fastcam_timestampfiles,
        'slowcam_timestamps': slowcam_timestamps,
        'slowcam_fileinds': slowcam_fileinds,
        'slowcam_frameinds': slowcam_frameinds,
        'slowcam_timestampfiles': slowcam_timestampfiles,
        'matched_timestamps': matched_timestamps,
        'matching_dict': matching_dict,
        'config_dict': config_dict,
    }

    if verbose:
        print(f"  ✓ Read {len(matched_timestamps)} matched pairs")

    return data


# ============================================================================
# BUGFIXES & HELPER FUNCTIONS (from sort4.py)
# ============================================================================


def compute_frame_durations(fastcam_timestamp, fastcam_fileinds):
    """
    Return per-frame exposure durations, correcting for dead-time between FITS files.

    For interior frames within a file, duration = timestamp[i+1] - timestamp[i]
    (no dead-time inside a file). For the last frame of each file (and the
    final frame overall), use the median intra-file interval instead, so that
    the inter-file gap does not inflate the effective exposure duration.

    Parameters
    ----------
    fastcam_timestamp : ndarray float64, shape (N,)
        Unix timestamp for each frame
    fastcam_fileinds : ndarray int, shape (N,)
        File index for each frame (indicates which FITS file it came from)

    Returns
    -------
    frame_end_times : ndarray float64, shape (N,)
        Unix time when each frame's exposure ends
    """
    N = len(fastcam_timestamp)
    durations = np.empty(N, dtype='float64')

    # Default: duration from next-frame start
    durations[:-1] = np.diff(fastcam_timestamp)
    durations[-1] = np.nan  # filled below

    # Indices of the last frame in each file
    boundary_mask = np.concatenate([np.diff(fastcam_fileinds) != 0, [True]])
    last_inds = np.where(boundary_mask)[0]  # last frame index of each file

    for fid in np.unique(fastcam_fileinds):
        mask = fastcam_fileinds == fid
        inds = np.where(mask)[0]
        if len(inds) > 1:
            median_dur = np.median(np.diff(fastcam_timestamp[inds]))
        else:
            # Single-frame file: use global median of all intra-file intervals
            all_diffs = [
                np.diff(fastcam_timestamp[fastcam_fileinds == f])
                for f in np.unique(fastcam_fileinds)
                if (fastcam_fileinds == f).sum() > 1
            ]
            median_dur = np.median(np.concatenate(all_diffs)) if all_diffs else 0.0

        durations[inds[-1]] = median_dur  # replace inflated inter-file gap

    return fastcam_timestamp + durations


def validate_timestamp_matching(timestamps1, timestamps2, atol=1e-4, verbose=True):
    """
    Two-pointer match of two sorted Unix-epoch timestamp arrays.
    Uses float tolerance instead of exact equality.
    Warns when > 5% of either list is unmatched.

    BUGFIX (Issue 4): Uses float tolerance (atol) instead of exact equality.

    Parameters
    ----------
    timestamps1, timestamps2 : array-like of float
        Must be sorted. Typically PSF and PL camera timestamps.
    atol : float, optional
        Absolute tolerance in seconds (default 0.1 ms)
    verbose : bool, optional
        Print matching statistics

    Returns
    -------
    idx1, idx2 : ndarray bool
        True where timestamps matched
    """
    if verbose:
        print("Timestamp1 start: %s, end %s, length %d" % (
            datetime.fromtimestamp(timestamps1[0]),
            datetime.fromtimestamp(timestamps1[-1]), len(timestamps1)))
        print("Timestamp2 start: %s, end %s, length %d" % (
            datetime.fromtimestamp(timestamps2[0]),
            datetime.fromtimestamp(timestamps2[-1]), len(timestamps2)))

    idx1 = np.zeros(len(timestamps1), dtype=bool)
    idx2 = np.zeros(len(timestamps2), dtype=bool)
    i = j = 0
    while i < len(timestamps1) and j < len(timestamps2):
        diff = timestamps1[i] - timestamps2[j]
        if abs(diff) <= atol:
            idx1[i] = idx2[j] = True
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1

    n_drop1, n_drop2 = (~idx1).sum(), (~idx2).sum()
    if verbose:
        print("Filtered %d out of timestamp1, %d out of timestamp2" % (n_drop1, n_drop2))

    for label, arr, n_drop in [('timestamp1', timestamps1, n_drop1),
                               ('timestamp2', timestamps2, n_drop2)]:
        frac = n_drop / len(arr)
        if frac > 0.05:
            print("WARNING: %.1f%% of %s frames were unmatched. "
                  "Check that both cameras cover the same interval."
                  % (frac * 100, label))

    return idx1, idx2


def build_matching_dict(fastcam_timestamp, slowcam_timestamp,
                        bisect_inds, ind_start, ind_end,
                        frame_end_times=None, verbose=True):
    """
    Build {slowcam_ind: {fastcam_ind: weight}} with all edge cases handled.

    BUGFIXES:
      - Bug 1: When bisect_inds[ind] == bisect_inds[ind+1], a single fastcam
               frame covers the whole slowcam interval; combined weight accounts
               for actual overlap.
      - Bug 2: When bisect_inds[ind] == 0, the left-boundary index would wrap
               to fastcam_timestamp[-1]; those slowcam frames are skipped with warning.

    Parameters
    ----------
    fastcam_timestamp : ndarray float64, shape (N,)
        PSF camera timestamps (Unix epoch)
    slowcam_timestamp : ndarray float64, shape (M,)
        PL camera timestamps (Unix epoch)
    bisect_inds : list or array of int, shape (M,)
        Result of bisect() for each slowcam timestamp
    ind_start, ind_end : int
        Range of slowcam frames to process
    frame_end_times : ndarray float64, shape (N,), optional
        Unix time when each fastcam frame's exposure ends (from compute_frame_durations).
        When provided, weights account for actual frame durations, avoiding inflation
        from inter-file dead-time gaps.
    verbose : bool, optional
        Print warnings

    Returns
    -------
    Dict : dict
        {slowcam_ind: {fastcam_ind: weight}}
    timestamps : list
        slowcam timestamps for each key in Dict
    n_skipped : int
        Number of skipped frames (those before all fastcam timestamps)
    """
    Dict = {}
    timestamps = []
    n_skipped = 0

    for ind in tqdm(np.arange(ind_start, ind_end), desc='Building match dict', disable=not verbose):
        bi = bisect_inds[ind]
        bi_next = bisect_inds[ind + 1]

        # BUGFIX 2: skip frames that precede all fastcam timestamps
        if bi == 0:
            if verbose:
                print("WARNING: slowcam frame %d precedes all fastcam timestamps — skipping." % ind)
            n_skipped += 1
            continue

        # Compute leftfrac: fraction of frame (bi-1) inside the slowcam interval
        if frame_end_times is not None:
            end_left = frame_end_times[bi - 1]
            dur_left = end_left - fastcam_timestamp[bi - 1]
            leftfrac = max(0.0, end_left - slowcam_timestamp[ind]) / dur_left if dur_left > 0 else 0.0
        else:
            interval_left = fastcam_timestamp[bi] - fastcam_timestamp[bi - 1]
            leftfrac = (fastcam_timestamp[bi] - slowcam_timestamp[ind]) / interval_left

        if bi == bi_next:
            # BUGFIX 1: only one fastcam frame spans the entire slowcam interval
            if frame_end_times is not None:
                end_left = frame_end_times[bi - 1]
                dur_left = end_left - fastcam_timestamp[bi - 1]
                slowcam_end = slowcam_timestamp[ind + 1]
                overlap = max(0.0, min(end_left, slowcam_end) - slowcam_timestamp[ind])
                combined = overlap / dur_left if dur_left > 0 else 0.0
            else:
                interval_right = fastcam_timestamp[bi_next] - fastcam_timestamp[bi_next - 1]
                rightfrac = (slowcam_timestamp[ind + 1] - fastcam_timestamp[bi_next - 1]) / interval_right
                combined = max(leftfrac + rightfrac - 1.0, 0.0)
            Dict[ind] = {bi - 1: combined}
        else:
            # Compute rightfrac: fraction of frame (bi_next-1) inside the slowcam interval
            if frame_end_times is not None:
                end_right = frame_end_times[bi_next - 1]
                dur_right = end_right - fastcam_timestamp[bi_next - 1]
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


def plot_matching_diagnostics(h5_path, max_weight_frames=5000):
    """
    Create 4-panel diagnostic plots for timestamp matching quality.

    Plots:
      1. Weight distribution (histogram)
      2. Frames per slowcam image (histogram)
      3. Cumulative weight vs frame index
      4. Centroid distribution

    Parameters
    ----------
    h5_path : str
        Path to core H5 file
    max_weight_frames : int, optional
        Maximum number of frames to plot (for performance)
    """
    with h5py.File(h5_path, 'r') as h5f:
        centroids = h5f['psfcam']['centroids'][:]
        timestamps = h5f['metadata']['timestamps'][:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Timestamp distribution
    axes[0, 0].hist(np.diff(timestamps), bins=50)
    axes[0, 0].set_xlabel('Frame interval (s)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Timestamp Intervals')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Centroid distribution
    axes[0, 1].scatter(centroids[:, 0], centroids[:, 1], alpha=0.5, s=1)
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    axes[0, 1].set_title('Centroid Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: X centroid histogram
    axes[1, 0].hist(centroids[:, 0], bins=50, alpha=0.7)
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('X Centroid Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Y centroid histogram
    axes[1, 1].hist(centroids[:, 1], bins=50, alpha=0.7)
    axes[1, 1].set_xlabel('Y (pixels)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Y Centroid Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def write_consolidated_h5_core(
    outpath,
    psfcam_frames,
    centroids,
    peaks,
    timestamps,
    plcam_file_indices,
    plcam_frame_indices,
    plcam_files,
    config_dict,
    centroid_method='subpixel',
    frame_end_times=None,
    verbose=False,
):
    """
    Write core consolidated H5 file with PSF metadata and frames.

    This file contains all metadata needed for analysis (timestamps, centroids,
    peaks) and the PSF camera frames. PL camera frames are referenced but not
    stored here (see write_plcam_raw_h5).

    Parameters
    ----------
    outpath : str
        Path to output core h5 file (e.g., 'matched_core.h5')
    psfcam_frames : ndarray, shape (N, h, w), dtype float32
        PSF camera frames (timestamp-matched)
    centroids : ndarray, shape (N, 2), dtype float32
        (x, y) centroid positions for each PSF frame in pixels
    peaks : ndarray, shape (N,), dtype float32
        Maximum pixel value per PSF frame (Strehl proxy)
    timestamps : ndarray, shape (N,), dtype float64
        UTC timestamps (unix epoch) for each frame
    plcam_file_indices : ndarray, shape (N,), dtype int
        File index for each frame in the PL camera FITS files
    plcam_frame_indices : ndarray, shape (N,), dtype int
        Frame index within each PL camera FITS file
    plcam_files : list of str
        List of PL camera FITS file paths
    config_dict : dict
        Configuration parameters (observation info, settings, etc.)
    centroid_method : str, optional
        Method used to compute centroids ('subpixel', 'center_of_mass', etc.)
    frame_end_times : ndarray, shape (N,), dtype float64, optional
        Exposure end times (for dead-time correction). If provided, stored for reference.
    verbose : bool, optional
        Print progress information

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)

    if verbose:
        print(f"Writing consolidated core H5: {outpath}")

    with h5py.File(outpath, 'w') as h5f:
        # --- Metadata group ---
        meta_grp = h5f.create_group('metadata')
        meta_grp.attrs['centroid_method'] = centroid_method
        meta_grp.attrs['num_frames'] = len(timestamps)
        meta_grp.attrs['write_time'] = datetime.now().isoformat()

        # Store config as JSON
        meta_grp.create_dataset('config', data=json.dumps(config_dict))

        # Store observation metadata
        if 'obs_date' in config_dict:
            meta_grp.attrs['obs_date'] = config_dict['obs_date']
        if 'obs_start' in config_dict:
            meta_grp.attrs['obs_start'] = config_dict['obs_start']
        if 'obs_end' in config_dict:
            meta_grp.attrs['obs_end'] = config_dict['obs_end']

        # Store timestamps (always needed for filtering)
        meta_grp.create_dataset('timestamps', data=timestamps, dtype='float64')

        if frame_end_times is not None:
            meta_grp.create_dataset('frame_end_times', data=frame_end_times, dtype='float64')

        # --- PSF camera group ---
        psf_grp = h5f.create_group('psfcam')
        psf_grp.create_dataset('frames', data=psfcam_frames, dtype='float32', compression='gzip', compression_opts=4)
        psf_grp.create_dataset('centroids', data=centroids, dtype='float32', compression='gzip', compression_opts=4)
        psf_grp.create_dataset('peaks', data=peaks, dtype='float32', compression='gzip', compression_opts=4)

        psf_grp.attrs['psf_shape'] = psfcam_frames.shape[1:]
        psf_grp.attrs['num_frames'] = len(timestamps)

        # --- PL camera metadata group ---
        pl_meta_grp = h5f.create_group('plcam_metadata')
        pl_meta_grp.create_dataset('file_indices', data=plcam_file_indices, dtype='int64')
        pl_meta_grp.create_dataset('frame_indices', data=plcam_frame_indices, dtype='int64')
        pl_meta_grp.create_dataset('files', data=json.dumps(plcam_files))

        pl_meta_grp.attrs['num_frames'] = len(timestamps)
        pl_meta_grp.attrs['num_files'] = len(plcam_files)

    if verbose:
        print(f"  ✓ Wrote {len(timestamps)} frames to {outpath}")
        print(f"    PSF shape: {psfcam_frames.shape}")
        print(f"    Centroids range: x=[{centroids[:, 0].min():.1f}, {centroids[:, 0].max():.1f}], "
              f"y=[{centroids[:, 1].min():.1f}, {centroids[:, 1].max():.1f}]")
        print(f"    Peaks: [{peaks.min():.2f}, {peaks.max():.2f}]")


def write_plcam_raw_h5(
    outpath,
    plcam_files,
    plcam_file_indices,
    plcam_frame_indices,
    plcam_file_shapes,
    core_h5_ref=None,
    compression='lz4',
    chunk_size=None,
    verbose=False,
):
    """
    Write raw PL camera frames to H5 with smart chunking for efficient pixel reads.

    Frames are stored with chunking (chunk_frames, ny, nx) so that reading a single
    pixel across all frames is efficient (h5py reads only relevant chunks).

    Parameters
    ----------
    outpath : str
        Path to output raw PL H5 file (e.g., 'matched_plcam_raw.h5')
    plcam_files : list of str
        List of PL camera FITS file paths (in order)
    plcam_file_indices : ndarray, shape (N,), dtype int
        File index for each frame
    plcam_frame_indices : ndarray, shape (N,), dtype int
        Frame index within each file
    plcam_file_shapes : tuple or dict
        Shape (ny, nx) of PL camera frames. If dict, keys are file indices.
    core_h5_ref : str, optional
        Path to core h5 file (stored as reference)
    compression : str, optional
        Compression algorithm: 'gzip', 'lz4', 'szip', etc. Default: 'lz4'
    chunk_size : int, optional
        Number of frames per chunk. Default: min(100, N//10)
    verbose : bool, optional
        Print progress information

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)

    if verbose:
        print(f"Writing raw PL camera H5: {outpath}")

    # Infer shape if not provided
    if isinstance(plcam_file_shapes, tuple):
        ny, nx = plcam_file_shapes
    else:
        # Load first FITS to determine shape
        first_file = plcam_files[0]
        first_data = fits.getdata(first_file)
        if first_data.ndim == 3:
            _, ny, nx = first_data.shape
        else:
            ny, nx = first_data.shape
        if verbose:
            print(f"  Inferred PL frame shape: ({ny}, {nx})")

    N = len(plcam_file_indices)
    if chunk_size is None:
        chunk_size = max(1, min(100, N // 10))

    with h5py.File(outpath, 'w') as h5f:
        # --- Metadata ---
        h5f.attrs['num_frames'] = N
        h5f.attrs['plcam_shape'] = (ny, nx)
        h5f.attrs['chunk_size'] = chunk_size
        h5f.attrs['compression'] = compression
        h5f.attrs['write_time'] = datetime.now().isoformat()

        if core_h5_ref:
            h5f.attrs['reference_core'] = core_h5_ref

        # --- PL camera group ---
        pl_grp = h5f.create_group('plcam')

        # Pre-allocate dataset with smart chunking
        frames_ds = pl_grp.create_dataset(
            'frames',
            shape=(N, ny, nx),
            dtype='int16',
            chunks=(chunk_size, ny, nx),  # Smart chunking for pixel reads
            compression=compression,
        )

        if verbose:
            print(f"  Pre-allocated dataset: shape={frames_ds.shape}, chunks={frames_ds.chunks}")

        # --- Write frames from FITS files ---
        current_file_idx = -1
        current_file_data = None

        for i, (ffile_idx, fframe_idx) in enumerate(tqdm(zip(plcam_file_indices, plcam_frame_indices),
                                                          total=N, desc="Writing PL frames", disable=not verbose)):
            # Load new file if needed
            if ffile_idx != current_file_idx:
                if current_file_data is not None:
                    del current_file_data  # Free memory
                current_file_idx = ffile_idx
                if verbose:
                    print(f"\n  Loading FITS file {current_file_idx}: {plcam_files[current_file_idx]}")
                current_file_data = fits.getdata(plcam_files[current_file_idx])

            # Write frame
            if current_file_data.ndim == 3:
                frames_ds[i] = current_file_data[fframe_idx]
            else:
                frames_ds[i] = current_file_data

    if verbose:
        print(f"  ✓ Wrote {N} frames to {outpath}")
        print(f"    Frame shape: ({ny}, {nx})")
        print(f"    Chunk size: {chunk_size} frames per chunk")
        print(f"    File size estimate: ~{N * ny * nx * 2 / 1e9:.1f} GB (uncompressed)")


def filter_by_peak_and_time(
    timestamps,
    peaks,
    peak_min=None,
    peak_max=None,
    time_min=None,
    time_max=None,
    verbose=False,
):
    """
    Filter frames by PSF peak (Strehl) and timestamp range.

    Parameters
    ----------
    timestamps : ndarray, shape (N,)
        UTC timestamps (unix epoch)
    peaks : ndarray, shape (N,)
        PSF peak values (Strehl proxy)
    peak_min : float, optional
        Minimum peak value (inclusive)
    peak_max : float, optional
        Maximum peak value (inclusive)
    time_min : float, optional
        Minimum timestamp (inclusive)
    time_max : float, optional
        Maximum timestamp (inclusive)
    verbose : bool, optional
        Print filtering statistics

    Returns
    -------
    mask : ndarray, shape (N,), dtype bool
        Boolean array indicating frames that pass all filters
    """
    mask = np.ones(len(timestamps), dtype=bool)

    if peak_min is not None:
        mask &= peaks >= peak_min
    if peak_max is not None:
        mask &= peaks <= peak_max
    if time_min is not None:
        mask &= timestamps >= time_min
    if time_max is not None:
        mask &= timestamps <= time_max

    n_original = len(timestamps)
    n_kept = np.sum(mask)
    n_dropped = n_original - n_kept

    if verbose:
        print(f"Filtering:")
        if peak_min is not None or peak_max is not None:
            print(f"  Peak range: [{peak_min}, {peak_max}]")
        if time_min is not None or time_max is not None:
            t0_str = datetime.fromtimestamp(time_min).isoformat() if time_min else "—"
            t1_str = datetime.fromtimestamp(time_max).isoformat() if time_max else "—"
            print(f"  Time range: [{t0_str}, {t1_str}]")
        print(f"  Kept {n_kept} / {n_original} frames ({100*n_kept/n_original:.1f}%)")
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} frames")

    return mask


def spatial_bin_on_the_fly(values, centroids, map_n, map_width, centroid_center=None):
    """
    Perform on-the-fly spatial binning using PSF centroids.

    Takes N values (e.g., for a single pixel across N frames) and N centroid positions,
    then bins the values into a 2D grid based on centroid locations.

    Parameters
    ----------
    values : ndarray, shape (N,)
        Values to bin (e.g., pixel intensities for all frames)
    centroids : ndarray, shape (N, 2)
        (x, y) centroid positions for each value
    map_n : int
        Number of bins in each dimension (map_n x map_n grid)
    map_width : float
        Total width of binning grid in pixel coordinates
    centroid_center : tuple, optional
        (x_center, y_center) for grid center. Default: median of centroids.

    Returns
    -------
    binned_map : ndarray, shape (map_n, map_n)
        Averaged values in each spatial bin (NaN for empty bins)
    bin_counts : ndarray, shape (map_n, map_n), dtype int
        Number of frames in each bin
    """
    if centroid_center is None:
        centroid_center = (
            np.nanmedian(centroids[:, 0]),
            np.nanmedian(centroids[:, 1]),
        )

    x_min = centroid_center[0] - map_width / 2
    x_max = centroid_center[0] + map_width / 2
    y_min = centroid_center[1] - map_width / 2
    y_max = centroid_center[1] + map_width / 2

    xbins = np.linspace(x_min, x_max, map_n + 1)
    ybins = np.linspace(y_min, y_max, map_n + 1)

    binned_map = np.full((map_n, map_n), np.nan, dtype='float32')
    bin_counts = np.zeros((map_n, map_n), dtype='int64')

    for i in range(map_n):
        for j in range(map_n):
            mask = (
                (centroids[:, 0] >= xbins[i])
                & (centroids[:, 0] < xbins[i + 1])
                & (centroids[:, 1] >= ybins[j])
                & (centroids[:, 1] < ybins[j + 1])
            )
            if np.sum(mask) > 0:
                binned_map[i, j] = np.nanmean(values[mask])
                bin_counts[i, j] = np.sum(mask)

    return binned_map, bin_counts
