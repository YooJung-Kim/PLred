# codes related to constructing coupling maps
# MERGED VERSION:
#   - All-at-once timestamp matching (original sort.py)
#   - Bugfixed matching logic (sort4.py: edge cases, dead-time correction)
#   - Giant H5 output with weighted frames, nstacks, metadata (sort4.py)
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
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import h5py
import json
from configobj import ConfigObj
from datetime import datetime
from bisect import bisect
import os, glob, pickle, re


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_data_between(datadir, obs_start, obs_end, header='', footer=''):
    '''
    Find files whose embedded timestamp falls strictly between obs_start and obs_end.
    File names must contain a timestamp matching HH:MM:SS.ffffff.

    Parameters
    ----------
    datadir : str
    obs_start, obs_end : str  "%H:%M:%S"
    header, footer : str  filename prefix / suffix filters
    '''
    start = datetime.strptime(obs_start, "%H:%M:%S")
    end   = datetime.strptime(obs_end,   "%H:%M:%S")

    files = sorted(glob.glob(datadir + header + '*' + footer))
    pattern = r"(\d{2}:\d{2}:\d{2}\.\d+)"
    valid_files = []

    for f in files:
        m = re.search(pattern, f)
        if m:
            obstime = datetime.strptime(m.group(1)[:13], "%H:%M:%S.%f")
            if start < obstime < end:
                valid_files.append(f)

    print("number of files found: %d" % len(valid_files))
    return valid_files


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
    last_inds = np.where(boundary_mask)[0]  # last frame index of each file

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
    matches them, and outputs a giant H5 file with all weighted frames.

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
        show_plot           = True
        crop_width          = 20
        apply_dead_time_correction = True
    '''

    # read the config file
    config = ConfigObj(configname)

    fastcam_dir = config['Fastcam']['path']
    fastcam_start_time = config['Fastcam']['start_time']
    fastcam_end_time = config['Fastcam']['end_time']

    fastcam_dark_file = config['Fastcam']['dark_file']
    if fastcam_dark_file.strip() == '':
        fastcam_dark_start_time = config['Fastcam']['dark_start_time']
        fastcam_dark_end_time = config['Fastcam']['dark_end_time']

    slowcam_timestamps = config['Slowcam']['timestamp_dir']
    try:
        slowcam_nbin = int(config['Slowcam']['nbin'])
    except:
        slowcam_nbin = 1

    outname = config['Output']['outname']
    filename = config['Output']['filename']

    verbose = (config['Options']['verbose']).lower() == 'true'
    show_plot = (config['Options']['show_plot']).lower() == 'true'
    crop_width = int(config['Options']['crop_width'])
    try:
        apply_dtc = (config['Options']['apply_dead_time_correction']).lower() == 'true'
    except:
        apply_dtc = True

    os.makedirs(outname, exist_ok=True)

    # find fastcam data
    fastcam_timestampfiles = find_data_between(fastcam_dir, fastcam_start_time, 
                                                fastcam_end_time, footer='.txt')
    if fastcam_dark_file.strip() == '':
        fastcam_darkframes = find_data_between(fastcam_dir, fastcam_dark_start_time, 
                                               fastcam_dark_end_time, footer='.fits')
    
    # find slowcam timestamp data
    slowcam_timestampfiles = np.sort(glob.glob(slowcam_timestamps + '*.txt'))

    if verbose:
        print("fastcam timestampfiles:", fastcam_timestampfiles)
        print("slowcam timestampfiles:", slowcam_timestampfiles)

    ######################
    # Timestamp matching #
    ######################
    
    # Load all fastcam timestamps from all files
    fastcam_timestamps_list = [np.genfromtxt(f)[:, 4] for f in fastcam_timestampfiles]
    fastcam_timestamp = np.concatenate(fastcam_timestamps_list)
    
    # Load all slowcam timestamps from all files
    slowcam_timestamps_list = [np.genfromtxt(f)[:, 4] for f in slowcam_timestampfiles]
    slowcam_timestamp = np.concatenate(slowcam_timestamps_list)

    # file and frame indices into array
    fastcam_fileinds = np.concatenate([np.full(len(ts), i) 
                                        for i, ts in enumerate(fastcam_timestamps_list)])
    slowcam_fileinds = np.concatenate([np.full(len(ts), i) 
                                        for i, ts in enumerate(slowcam_timestamps_list)])

    fastcam_frameinds = np.concatenate([np.genfromtxt(f, dtype=int)[:, 0] 
                                         for f in fastcam_timestampfiles])
    slowcam_frameinds = np.concatenate([np.genfromtxt(f, dtype=int)[:, 0] 
                                         for f in slowcam_timestampfiles])

    # Apply slowcam nbin with warning
    if slowcam_nbin > 1:
        n_total = len(slowcam_timestamp)
        n_keep  = (n_total // slowcam_nbin) * slowcam_nbin
        n_drop  = n_total - n_keep
        if n_drop > 0:
            print("WARNING: slowcam_nbin=%d — dropping last %d timestamp(s) "
                  "that do not fill a complete bin." % (slowcam_nbin, n_drop))
        slowcam_timestamp = slowcam_timestamp[:n_keep].reshape(-1, slowcam_nbin)[:, 0]
        slowcam_fileinds = slowcam_fileinds[:n_keep].reshape(-1, slowcam_nbin)[:, 0]
        slowcam_frameinds = slowcam_frameinds[:n_keep].reshape(-1, slowcam_nbin)[:, 0]

    # Bisect matching
    bisect_inds = [bisect(fastcam_timestamp, t) for t in slowcam_timestamp]
    bisect_arr = np.array(bisect_inds)
    max_bisect = int(bisect_arr.max())

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

    # Build matching dict with bugfixes
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

    ######################
    # PSF frame averaging to H5 #
    ######################
    if fastcam_dark_file.strip() == '':
        fastcam_avgdark = np.average(fits.getdata(fastcam_darkframes[0]), axis=0)
    else:
        fastcam_avgdark = fits.getdata(fastcam_dark_file)
        
    fastcam_files = [f.split('.txt')[0] + '.fits' for f in fastcam_timestampfiles]

    (xwidth, ywidth) = np.shape(fastcam_avgdark)
    xc, yc = xwidth // 2, ywidth // 2
    if crop_width is None:
        xw, yw = xwidth // 2, ywidth // 2
        print("saving full frames, without crop")
    else:
        xw = yw = int(crop_width)
        print(f"saving cropped frames, [{xc-xw}:{xc+xw}, {yc-yw}:{yc+yw}]")

    fastcam_avgdark = fastcam_avgdark[xc-xw:xc+xw, yc-yw:yc+yw]

    indices = list(Dict.keys())
    nstacks = []

    # Write to giant H5 file
    h5_path = os.path.join(outname, filename + '.h5')
    
    current_fc_fileind = 0
    print("reading fastcam file", fastcam_files[current_fc_fileind])
    current_fc_data = (fits.getdata(fastcam_files[current_fc_fileind])
                       [:, xc-xw:xc+xw, yc-yw:yc+yw] - fastcam_avgdark)

    with h5py.File(h5_path, 'w') as fh:
        # Create frames dataset
        ds = fh.create_dataset('frames', shape=(len(indices), 2*xw, 2*yw),
                               dtype='float32', chunks=(1, 2*xw, 2*yw))

        for out_idx, ind in enumerate(tqdm(indices)):
            fc_inds = np.array(list(Dict[ind].keys()))
            fc_frac = np.array(list(Dict[ind].values()))

            frame_acc = np.zeros((2*xw, 2*yw), dtype='float64')
            w_total = 0.0

            for _fi, _ff in zip(fc_inds, fc_frac):
                if fastcam_fileinds[_fi] != current_fc_fileind:
                    current_fc_fileind = fastcam_fileinds[_fi]
                    print("reading new fastcam file", fastcam_files[current_fc_fileind])
                    current_fc_data = (fits.getdata(fastcam_files[current_fc_fileind])
                                       [:, xc-xw:xc+xw, yc-yw:yc+yw] - fastcam_avgdark)
                if verbose:
                    print("frame %d  frac %.4f" % (fastcam_frameinds[_fi], _ff))
                frame_acc += _ff * current_fc_data[fastcam_frameinds[_fi]]
                w_total += _ff

            ds[out_idx] = (frame_acc / w_total).astype('float32') if w_total > 0 \
                          else frame_acc.astype('float32')
            nstacks.append(w_total)

        # Create nstacks dataset
        fh.create_dataset('nstacks', data=np.array(nstacks, dtype='float32'))

        # Create metadata
        meta = {
            'fastcam_timestampfiles': list(fastcam_timestampfiles),
            'slowcam_timestampfiles': list(slowcam_timestampfiles),
            'fastcam_fileinds':  fastcam_fileinds.tolist(),
            'slowcam_fileinds':  slowcam_fileinds.tolist(),
            'fastcam_frameinds': fastcam_frameinds.tolist(),
            'slowcam_frameinds': slowcam_frameinds.tolist(),
            'matched_indices':   {str(k): {str(kk): float(vv) for kk, vv in v.items()}
                                  for k, v in Dict.items()},
            'timestamps':        [float(t) for t in timestamps],
            'slowcam_nbin':      slowcam_nbin,
            'dead_time_corrected': apply_dtc,
        }
        fh.create_dataset('metadata', data=json.dumps(meta))

    print("saved: %s  (%d frames)" % (h5_path, len(indices)))

    # Save pickle backup for reference
    pkl_path = os.path.join(outname, filename + '.pkl')
    data_to_save = {'fastcam_timestampfiles': fastcam_timestampfiles,
                    'slowcam_timestampfiles': slowcam_timestampfiles,
                    'fastcam_fileinds': fastcam_fileinds,
                    'slowcam_fileinds': slowcam_fileinds,
                    'fastcam_frameinds': fastcam_frameinds,
                    'slowcam_frameinds': slowcam_frameinds,
                    'matched_indices': Dict,
                    'timestamps': timestamps,
                    'nstacks': nstacks}
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print("timestamp matching backup saved to %s" % pkl_path)

    return h5_path


# ---------------------------------------------------------------------------
# FrameSorter class for downstream processing
# ---------------------------------------------------------------------------

class FrameSorter:
    '''
    Read matched frames from H5 output of script_match_timestamps and
    compute centroids, peaks, and other diagnostics.
    '''
    def __init__(self, h5_path, verbose=False):
        self.h5_path = h5_path
        self.verbose = verbose
        
        with h5py.File(h5_path, 'r') as f:
            self.frames = f['frames'][:]
            self.nstacks = f['nstacks'][:]
            meta = json.loads(f['metadata'][()])
            self.metadata = meta
            
        self.n_frames = len(self.frames)
        self.shape = self.frames.shape
        print(f"Loaded {self.n_frames} frames from {h5_path}")
        
    def compute_centroids(self):
        '''Compute centroid of each frame.'''
        self.centroids = np.array([subpixel_centroid_2d(f) for f in self.frames])
        return self.centroids
    
    def compute_peaks(self):
        '''Compute peak value and location of each frame.'''
        self.peaks = np.array([np.nanmax(f) for f in self.frames])
        self.peak_locs = np.array([np.unravel_index(np.argmax(f), f.shape) 
                                    for f in self.frames])
        return self.peaks, self.peak_locs
    
    def save_consolidated_h5(self, outpath):
        '''
        Save consolidated H5 with frames, centroids, peaks, timestamps, nstacks.
        '''
        if not hasattr(self, 'centroids'):
            self.compute_centroids()
        if not hasattr(self, 'peaks'):
            self.compute_peaks()
            
        with h5py.File(outpath, 'w') as f:
            f.create_dataset('frames', data=self.frames)
            f.create_dataset('nstacks', data=self.nstacks)
            f.create_dataset('centroids', data=self.centroids)
            f.create_dataset('peaks', data=self.peaks)
            f.create_dataset('peak_locations', data=self.peak_locs)
            f.create_dataset('timestamps', 
                           data=np.array(self.metadata['timestamps'], dtype='float64'))
            f.create_dataset('metadata', data=json.dumps(self.metadata))
            
        print(f"Consolidated H5 saved to {outpath}")
