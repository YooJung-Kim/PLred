"""
Step 2: H5 ingestion and peak computation.

Reads the Step 1 matched H5 (PSFcam averaged frames + metadata) and the raw
PLcam FITS data, dark-subtracts the PLcam frames, computes PSFcam centroids
and peak values, and writes everything into a single consolidated H5 file.

Output H5 structure
-------------------
giant.h5
  attrs: n_frames, psfcam_is_fast, plcam_type ('raw' or 'spectra')
  /metadata/
    timestamps      float64 (N,)   — relative timestamps in seconds from t0
    t0              float64        — absolute Unix timestamp of the first frame
    config          str            — JSON dump of all input parameters
  /psfcam/
    frames          float32 (N, h, w)   — dark-subtracted (from Step 1)
    centroids       float32 (N, 2)      — (x, y) subpixel centroid
    peaks           float32 (N,)        — max pixel per frame (Strehl proxy)
  /plcam/
    frames          int16   (N, ny, nx) — dark-subtracted raw frames  [OR]
    spectra         float32 (N, Nlambda, Nport) — if pre-extracted spectra supplied
"""

import numpy as np
import h5py
import json
import os
from datetime import datetime
from tqdm import tqdm
from astropy.io import fits
from configobj import ConfigObj

from .imageutils import subpixel_centroid_2d


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_to_h5(
    step1_h5,
    outpath,
    plcam_dark=None,
    psfcam_dark=None,
    plcam_data_dir=None,
    slowcam_data_dir=None,
    plcam_roi=None,
    plcam_spectra=None,
    compression='gzip',
    compression_opts=4,
    verbose=False,
    spectral_orientation='horizontal',
):
    """
    Build the consolidated giant H5 (Step 2).

    Each PLcam frame is stored in its own HDF5 chunk so that writes are
    independent (no read-modify-write of neighbouring frames).  Apply
    ``plcam_roi`` to crop the PLcam frames before storing — this is the
    single most effective way to reduce file size and write time.

    Parameters
    ----------
    step1_h5 : str
        Path to the H5 file produced by Step 1 (script_match_timestamps).
    outpath : str
        Output path for the giant H5 (e.g. 'obs_giant.h5').
    plcam_dark : str or ndarray, optional
        Path to PLcam dark FITS file, or a 2-D dark frame array.
        If None, dark subtraction is skipped (can be done later in Step 3).
    plcam_data_dir : str, optional
        Directory containing PLcam FITS files.  When supplied, the directory
        part of the timestamp-file paths stored in the Step 1 metadata is
        replaced with this directory.  When None, the paths are used as-is.
    slowcam_data_dir : str, optional
        Directory containing PSFcam FITS files when ``psfcam_is_fast=False``.
        If omitted, ``plcam_data_dir`` is used as a backward-compatible
        fallback.
    plcam_roi : tuple of int, optional
        (y0, y1, x0, x1) pixel crop applied to every PLcam frame before
        storing.  Strongly recommended for large detectors — reduces both
        file size and write time proportionally.  None = full frame.
    plcam_spectra : str or ndarray, optional
        Pre-extracted PLcam spectra with shape (N, Nlambda, Nport).
        May be a path to an .npy / .npz file or a numpy array.
        When provided, this is stored instead of raw PLcam frames.
    compression : str
        HDF5 compression filter ('gzip', 'lz4', 'szip', …).
    compression_opts : int
        Compression level (meaningful for gzip).
    verbose : bool
        Print per-file progress messages.

    Returns
    -------
    outpath : str
    """
    # ------------------------------------------------------------------
    # 1. Load Step 1 H5
    # ------------------------------------------------------------------
    with h5py.File(step1_h5, 'r') as f:
        psfcam_frames   = f['frames'][:]          # (N, h, w) float32
        nstacks         = f['nstacks'][:]          # (N,)
        timestamps      = f['timestamps'][:]       # (N,)
        meta1           = json.loads(f['metadata'][()])

    N = len(timestamps)
    psfcam_is_fast = meta1.get('psfcam_is_fast', True)

    slowcam_ts_files = meta1['slowcam_timestampfiles']
    slowcam_fileinds = np.array(meta1['slowcam_fileinds'])
    slowcam_frameinds = np.array(meta1['slowcam_frameinds'])
    matched_indices  = {int(k): v for k, v in meta1['matched_indices'].items()}
    matched_sc_inds  = sorted(matched_indices.keys())   # in time order

    # When psfcam_is_fast=True:  step1 frames = PSFcam,  slowcam FITS = PLcam
    # When psfcam_is_fast=False: step1 frames = PLcam,   slowcam FITS = PSFcam
    if psfcam_is_fast:
        plcam_fits_files = _resolve_plcam_files(slowcam_ts_files, plcam_data_dir)
        plcam_from_step1 = None          # PLcam comes from FITS
        psfcam_for_centroid = psfcam_frames
    else:
        # PLcam data is already in step1 H5 (fastcam = PLcam was averaged there)
        plcam_from_step1 = psfcam_frames  # rename for clarity
        plcam_fits_files = None
        # PSFcam FITS are the slowcam files; slowcam_data_dir overrides their directory.
        # Fall back to plcam_data_dir for older configs that reused the same key.
        psf_data_dir = slowcam_data_dir if slowcam_data_dir is not None else plcam_data_dir
        psfcam_fits_files = _resolve_plcam_files(slowcam_ts_files, psf_data_dir)
        print("psfcam_is_fast=False: PLcam data taken from step1 H5; "
              "loading PSFcam FITS from slowcam file paths for centroids")
        psfcam_for_centroid = _load_psfcam_frames_from_fits(
            psfcam_fits_files, matched_sc_inds, slowcam_fileinds, slowcam_frameinds, N, verbose,
            psfcam_dark=psfcam_dark)

    psfcam_h, psfcam_w = psfcam_for_centroid.shape[1], psfcam_for_centroid.shape[2]

    # ------------------------------------------------------------------
    # 2. Load PLcam dark  (only used when PLcam comes from FITS)
    # ------------------------------------------------------------------
    if not psfcam_is_fast:
        dark_frame = None
        if plcam_dark is not None:
            print("WARNING: psfcam_is_fast=False — PLcam dark was already applied "
                  "in plred-sort. Ignoring supplied plcam_dark.")
        dark_subtracted = True   # dark was applied in sort.py
        dark_source = 'applied_in_sort'
    elif plcam_dark is None:
        dark_frame = None
        if verbose:
            print("No PLcam dark supplied — skipping dark subtraction")
    elif isinstance(plcam_dark, np.ndarray):
        dark_frame = plcam_dark.astype('float32')
    else:
        dark_frame = fits.getdata(plcam_dark).astype('float32')
        if dark_frame.ndim == 3:
            dark_frame = dark_frame.mean(axis=0)
        if verbose:
            print("Loaded PLcam dark: %s  shape %s" % (plcam_dark, dark_frame.shape))

    # ------------------------------------------------------------------
    # 3. Determine PLcam frame shape (after optional ROI crop)
    # ------------------------------------------------------------------
    plcam_type = 'spectra' if plcam_spectra is not None else 'raw'

    transpose_plcam = spectral_orientation == 'vertical'

    if plcam_type == 'raw':
        if psfcam_is_fast:
            ny, nx, roi = _probe_plcam_shape(plcam_fits_files, plcam_roi, dark_frame)
        else:
            # PLcam shape comes from step1 frames directly
            ny, nx, roi = _probe_plcam_shape_from_array(plcam_from_step1, plcam_roi)
        if transpose_plcam:
            ny, nx = nx, ny
        if verbose:
            print("PLcam frame shape after ROI%s: (%d, %d)" % (
                ' + transpose' if transpose_plcam else '', ny, nx))
    else:
        spectra_array = _load_spectra(plcam_spectra, N)
        _, Nlambda, Nport = spectra_array.shape
        if verbose:
            print("Pre-extracted spectra shape: (%d, %d, %d)" % (N, Nlambda, Nport))

    # ------------------------------------------------------------------
    # 4. Compute PSFcam centroids and peaks
    # ------------------------------------------------------------------
    print("Computing PSFcam centroids and peaks (%d frames)..." % N)
    centroids = np.zeros((N, 2), dtype='float32')
    peaks     = np.zeros(N,      dtype='float32')

    for i, frame in enumerate(tqdm(psfcam_for_centroid, disable=not verbose)):
        try:
            centroids[i] = subpixel_centroid_2d(frame)
        except Exception:
            centroids[i] = (np.nan, np.nan)
        peaks[i] = float(np.nanmax(frame))

    print("  centroid range  x=[%.2f, %.2f]  y=[%.2f, %.2f]" % (
        np.nanmin(centroids[:, 0]), np.nanmax(centroids[:, 0]),
        np.nanmin(centroids[:, 1]), np.nanmax(centroids[:, 1])))
    print("  peak range  [%.1f, %.1f]" % (peaks.min(), peaks.max()))

    # ------------------------------------------------------------------
    # 5. Write output H5
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)

    if psfcam_is_fast:
        dark_subtracted = dark_frame is not None
        if isinstance(plcam_dark, np.ndarray):
            dark_source = '<array>'
        elif plcam_dark is not None:
            dark_source = os.path.abspath(str(plcam_dark))
        else:
            dark_source = 'none'
    # else: dark_subtracted and dark_source already set above for psfcam_is_fast=False

    # Dark statistics (useful for quality checks downstream)
    # In the psfcam_is_fast=False path, PLcam dark was already applied in step 1,
    # so dark_frame may still be None even though dark_subtracted=True.
    if dark_subtracted and dark_frame is not None:
        dark_mean   = float(np.mean(dark_frame))
        dark_median = float(np.median(dark_frame))
        dark_std    = float(np.std(dark_frame))
        dark_shape  = list(dark_frame.shape)
    else:
        dark_mean = dark_median = dark_std = 0.0
        dark_shape = []

    ingest_time = datetime.now().isoformat()
    t0          = float(timestamps[0])
    t_end       = float(timestamps[-1])

    config_dict = {
        # --- provenance ---
        'step1_h5':            os.path.abspath(step1_h5),
        'ingest_time':         ingest_time,
        # --- PLcam frames ---
        'plcam_type':          plcam_type,
        'plcam_transposed':    transpose_plcam,
        'spectral_orientation': spectral_orientation,
        'plcam_data_dir':      plcam_data_dir,
        'plcam_roi':           list(plcam_roi) if plcam_roi else None,
        'plcam_fits_files':    plcam_fits_files,      # full list of files used
        # --- dark subtraction ---
        'dark_subtracted':     dark_subtracted,       # bool: was dark applied here?
        'dark_source':         dark_source,           # path, '<array>', or 'none'
        'dark_mean':           dark_mean,
        'dark_median':         dark_median,
        'dark_std':            dark_std,
        'dark_shape':          dark_shape,
        # --- frame counts ---
        'n_frames_matched':    N,
        't0_unix':             t0,
        't_end_unix':          t_end,
        'duration_s':          t_end - t0,
        # --- compression ---
        'compression':         compression,
        'compression_opts':    compression_opts,
    }

    # Relative times (seconds from t0, starting at 0 for the user)
    t_start_rel = 0.0
    t_end_rel   = t_end - t0      # duration in seconds
    timestamps_rel = timestamps - t0

    print("Writing giant H5: %s" % outpath)
    with h5py.File(outpath, 'w') as h5f:
        # ------------------------------------------------------------------
        # Root attrs — everything a user needs without opening sub-groups
        # ------------------------------------------------------------------
        h5f.attrs['n_frames']              = N
        h5f.attrs['psfcam_is_fast']        = psfcam_is_fast
        h5f.attrs['plcam_type']            = plcam_type
        h5f.attrs['plcam_dark_subtracted'] = dark_subtracted
        h5f.attrs['plcam_transposed']      = transpose_plcam
        h5f.attrs['spectral_orientation']  = spectral_orientation
        h5f.attrs['ingest_time']           = ingest_time
        h5f.attrs['t_start']               = t_start_rel   # 0.0 by definition
        h5f.attrs['t_end']                 = t_end_rel     # seconds from first frame
        h5f.attrs['duration_s']            = t_end_rel
        h5f.attrs['t0_unix']               = t0            # absolute reference
        if plcam_roi:
            h5f.attrs['plcam_roi']         = list(plcam_roi)   # visible in dict(f.attrs)

        # ------------------------------------------------------------------
        # metadata group
        # ------------------------------------------------------------------
        meta_grp = h5f.create_group('metadata')
        meta_grp.create_dataset('timestamps', data=timestamps_rel, dtype='float64')
        meta_grp.create_dataset('t0',         data=t0,             dtype='float64')
        meta_grp.create_dataset('config',     data=json.dumps(config_dict))
        # String array of PLcam FITS files — readable without parsing JSON
        meta_grp.create_dataset(
            'plcam_fits_files',
            data=np.array(plcam_fits_files, dtype=h5py.special_dtype(vlen=str)),
        )
        # Convenience attrs on metadata group — mirrors root for quick access
        meta_grp.attrs['n_frames']    = N
        meta_grp.attrs['t0_unix']     = t0
        meta_grp.attrs['t_start']     = t_start_rel
        meta_grp.attrs['t_end']       = t_end_rel
        meta_grp.attrs['duration_s']  = t_end_rel
        if plcam_roi:
            meta_grp.attrs['plcam_roi'] = list(plcam_roi)

        # ------------------------------------------------------------------
        # psfcam group
        # ------------------------------------------------------------------
        psf_grp = h5f.create_group('psfcam')
        psf_grp.create_dataset('frames',
                               data=psfcam_frames,
                               dtype='float32',
                               chunks=(1, psfcam_h, psfcam_w),
                               compression=compression,
                               compression_opts=compression_opts)
        psf_grp.create_dataset('centroids', data=centroids, dtype='float32')
        psf_grp.create_dataset('peaks',     data=peaks,     dtype='float32')
        psf_grp.create_dataset('nstacks',   data=nstacks,   dtype='float32')
        psf_grp.attrs['centroid_x_range'] = [float(np.nanmin(centroids[:, 0])),
                                              float(np.nanmax(centroids[:, 0]))]
        psf_grp.attrs['centroid_y_range'] = [float(np.nanmin(centroids[:, 1])),
                                              float(np.nanmax(centroids[:, 1]))]
        psf_grp.attrs['peak_range']       = [float(peaks.min()), float(peaks.max())]

        # ------------------------------------------------------------------
        # plcam group
        # ------------------------------------------------------------------
        pl_grp = h5f.create_group('plcam')
        pl_grp.attrs['dark_subtracted'] = dark_subtracted
        pl_grp.attrs['dark_source']     = dark_source
        if dark_subtracted:
            pl_grp.attrs['dark_mean']   = dark_mean
            pl_grp.attrs['dark_median'] = dark_median
            pl_grp.attrs['dark_std']    = dark_std
            pl_grp.attrs['dark_shape']  = dark_shape
        if plcam_roi:
            pl_grp.attrs['roi']         = list(plcam_roi)   # kept for back-compat

        if plcam_type == 'spectra':
            pl_grp.create_dataset('spectra',
                                  data=spectra_array.astype('float32'),
                                  dtype='float32',
                                  chunks=(1, Nlambda, Nport),
                                  compression=compression,
                                  compression_opts=compression_opts)
        else:
            frames_ds = pl_grp.create_dataset(
                'frames',
                shape=(N, ny, nx),
                dtype='float32',
                chunks=(1, ny, nx),
                compression=compression,
                compression_opts=compression_opts,
            )

    # ------------------------------------------------------------------
    # 6. Fill PLcam frames
    # ------------------------------------------------------------------
    if plcam_type == 'raw':
        if psfcam_is_fast:
            # Normal path: PLcam comes from FITS files
            _write_plcam_frames(
                outpath, plcam_fits_files, matched_sc_inds,
                slowcam_fileinds, slowcam_frameinds,
                dark_frame, roi, N, verbose,
                transpose=transpose_plcam,
            )
        else:
            # PLcam data is already in plcam_from_step1 (fastcam averaged frames)
            _write_plcam_frames_from_array(
                outpath, plcam_from_step1, matched_sc_inds, roi, N, verbose,
                transpose=transpose_plcam,
            )

    print("Done. Giant H5 written to %s" % outpath)
    return outpath


# ---------------------------------------------------------------------------
# Config-based entry point
# ---------------------------------------------------------------------------

def ingest_from_config(configname):
    """
    Run ingest_to_h5() using parameters from a .ini config file.

    Config format
    -------------
    [Step1]
    h5_path             = /path/to/step1.h5

    [PLcam]
    dark_file           = /path/to/plcam_dark.fits
    data_dir            =         # optional override for FITS directory
    roi                 =         # optional: y0,y1,x0,x1

    [Output]
    outpath             = /path/to/giant.h5

    [Options]
    chunk_frames        = 50
    compression         = gzip
    compression_opts    = 4
    verbose             = False
    """
    config = ConfigObj(configname)

    step1_h5  = config['Step1']['h5_path']

    dark_file = config['PLcam'].get('dark_file', '').strip()
    plcam_dark = dark_file if dark_file else None

    data_dir = config['PLcam'].get('data_dir', '').strip()
    plcam_data_dir = data_dir if data_dir else None

    roi_str = config['PLcam'].get('roi', '').strip()
    if roi_str:
        plcam_roi = tuple(int(x) for x in roi_str.split(','))
    else:
        plcam_roi = None

    outpath = config['Output']['outpath']

    compression = config['Options'].get('compression', 'gzip')
    try:
        compression_opts = int(config['Options'].get('compression_opts', 4))
    except Exception:
        compression_opts = 4
    verbose = config['Options'].get('verbose', 'False').lower() == 'true'

    return ingest_to_h5(
        step1_h5=step1_h5,
        outpath=outpath,
        plcam_dark=plcam_dark,
        plcam_data_dir=plcam_data_dir,
        plcam_roi=plcam_roi,
        compression=compression,
        compression_opts=compression_opts,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_plcam_shape_from_array(plcam_array, plcam_roi):
    """Determine PLcam frame shape when frames come from the step1 H5 array."""
    full_ny, full_nx = plcam_array.shape[1], plcam_array.shape[2]
    if plcam_roi is not None:
        y0, y1, x0, x1 = plcam_roi
        ny, nx = y1 - y0, x1 - x0
        roi = (y0, y1, x0, x1)
    else:
        ny, nx = full_ny, full_nx
        roi = None
    return ny, nx, roi


def _load_psfcam_frames_from_fits(psfcam_files, matched_sc_inds,
                                   slowcam_fileinds, slowcam_frameinds, N, verbose,
                                   psfcam_dark=None):
    """Load PSFcam frames from FITS when psfcam_is_fast=False (PSFcam = slowcam)."""
    # Determine frame shape from first available file
    psf_h = psf_w = None
    for fpath in psfcam_files:
        if os.path.exists(fpath):
            sample = fits.getdata(fpath)
            psf_h, psf_w = sample.shape[-2], sample.shape[-1]
            break
    if psf_h is None:
        raise FileNotFoundError("No PSFcam FITS files found: %s" % psfcam_files[:3])

    out = np.zeros((N, psf_h, psf_w), dtype='float32')

    # Load PSF dark if provided
    dark_frame = None
    if psfcam_dark is not None:
        if isinstance(psfcam_dark, np.ndarray):
            dark_frame = psfcam_dark.astype('float32')
        else:
            try:
                dark_frame = fits.getdata(psfcam_dark).astype('float32')
            except Exception:
                dark_frame = None
    # Build file_idx → [(out_idx, frame_idx)] mapping
    file_map = {}
    for out_idx, sc_ind in enumerate(matched_sc_inds):
        file_idx  = int(slowcam_fileinds[sc_ind])
        frame_idx = int(slowcam_frameinds[sc_ind])
        file_map.setdefault(file_idx, []).append((out_idx, frame_idx))

    for file_idx, entries in tqdm(sorted(file_map.items()),
                                  desc='Loading PSFcam frames', disable=not verbose):
        fpath = psfcam_files[file_idx]
        if not os.path.exists(fpath):
            print("WARNING: PSFcam FITS not found: %s — leaving zeros" % fpath)
            continue
        try:
            hdul = fits.open(fpath, memmap=True)
            data = hdul[0].data
        except Exception:
            hdul = fits.open(fpath, memmap=False)
            data = hdul[0].data
        with hdul:
            for out_idx, frame_idx in entries:
                frame = data[frame_idx].astype('float32')
                if dark_frame is not None:
                    try:
                        frame = frame - dark_frame
                    except Exception:
                        # If shapes mismatch, attempt to collapse or crop dark
                        df = dark_frame
                        if df.ndim == 3:
                            df = df.mean(axis=0)
                        df = df.astype('float32')
                        if df.shape == frame.shape:
                            frame = frame - df
                out[out_idx] = frame

    return out


def _write_plcam_frames_from_array(outpath, plcam_array, matched_sc_inds,
                                    roi, N, verbose, transpose=False):
    """Write PLcam frames directly from an in-memory array (psfcam_is_fast=False path)."""
    n_written = 0
    with h5py.File(outpath, 'r+') as h5f:
        ds = h5f['plcam/frames']
        for out_idx, sc_ind in enumerate(tqdm(matched_sc_inds,
                                               desc='Writing PLcam frames',
                                               disable=not verbose)):
            frame = plcam_array[out_idx].astype('float32')
            if roi is not None:
                y0, y1, x0, x1 = roi
                frame = frame[y0:y1, x0:x1]
            if transpose:
                frame = frame.T
            ds[out_idx] = frame
            n_written += 1
    print("Wrote %d / %d PLcam frames (from step1 array)" % (n_written, N))


def _resolve_plcam_files(slowcam_ts_files, plcam_data_dir):
    """
    Derive PLcam FITS paths from timestamp file paths.
    Replace .txt extension with .fits; optionally override the directory.
    """
    plcam_files = []
    for ts in slowcam_ts_files:
        fits_path = ts.replace('.txt', '.fits')
        if plcam_data_dir is not None:
            fits_path = os.path.join(plcam_data_dir,
                                     os.path.basename(fits_path))
        plcam_files.append(fits_path)
    return plcam_files


def _probe_plcam_shape(plcam_files, plcam_roi, dark_frame):
    """
    Read the first available PLcam FITS file to determine frame shape,
    validate ROI bounds, and return (ny, nx, roi).
    """
    for fpath in plcam_files:
        if os.path.exists(fpath):
            data = fits.getdata(fpath)
            full_ny = data.shape[-2]
            full_nx = data.shape[-1]
            break
    else:
        raise FileNotFoundError(
            "None of the PLcam FITS files were found: %s" % plcam_files)

    if plcam_roi is not None:
        y0, y1, x0, x1 = plcam_roi
        ny, nx = y1 - y0, x1 - x0
        roi = (y0, y1, x0, x1)
    else:
        ny, nx = full_ny, full_nx
        roi = None

    # Validate dark shape
    if dark_frame is not None:
        dark_ny, dark_nx = dark_frame.shape[-2], dark_frame.shape[-1]
        if (dark_ny, dark_nx) != (full_ny, full_nx):
            raise ValueError(
                "Dark frame shape (%d, %d) does not match PLcam frame shape (%d, %d)"
                % (dark_ny, dark_nx, full_ny, full_nx))

    return ny, nx, roi


def _load_spectra(plcam_spectra, N):
    """Load pre-extracted spectra from path or array, validate shape."""
    if isinstance(plcam_spectra, np.ndarray):
        arr = plcam_spectra
    elif plcam_spectra.endswith('.npz'):
        npz = np.load(plcam_spectra)
        key = list(npz.files)[0]
        arr = npz[key]
    else:
        arr = np.load(plcam_spectra)

    if arr.ndim != 3:
        raise ValueError("plcam_spectra must be shape (N, Nlambda, Nport), got %s" % str(arr.shape))
    if arr.shape[0] != N:
        raise ValueError("plcam_spectra has %d frames but Step 1 H5 has %d matched frames"
                         % (arr.shape[0], N))
    return arr


def _write_plcam_frames(outpath, plcam_files, matched_sc_inds,
                         slowcam_fileinds, slowcam_frameinds,
                         dark_frame, roi, N, verbose, transpose=False):
    """
    Second-pass writer: open each PLcam FITS file with memmap (reads only
    the frames we need), apply ROI and optional dark subtraction, then write
    one frame at a time into the pre-allocated dataset.

    Each write hits exactly one HDF5 chunk (1, ny, nx), so there is no
    read-modify-write overhead from neighbouring frames.
    """
    # Build mapping: fits_file_idx → [(out_idx, frame_in_file), ...]
    out_idx_map = {}
    for out_idx, sc_ind in enumerate(matched_sc_inds):
        file_idx  = int(slowcam_fileinds[sc_ind])
        frame_idx = int(slowcam_frameinds[sc_ind])
        out_idx_map.setdefault(file_idx, []).append((out_idx, frame_idx))

    n_written = 0
    with h5py.File(outpath, 'r+') as h5f:
        ds = h5f['plcam/frames']

        for file_idx, entries in tqdm(sorted(out_idx_map.items()),
                                      desc='Writing PLcam frames',
                                      total=len(out_idx_map)):
            fpath = plcam_files[file_idx]
            if not os.path.exists(fpath):
                print("WARNING: PLcam FITS not found: %s — filling with zeros" % fpath)
                for out_idx, _ in entries:
                    ds[out_idx] = 0
                n_written += len(entries)
                continue

            if verbose:
                print("Reading PLcam file %d: %s" % (file_idx, fpath))

            # memmap=True pages in only the needed frames from disk.
            # Falls back to full load when BZERO/BSCALE require rescaling.
            try:
                hdul = fits.open(fpath, memmap=True)
                data = hdul[0].data
            except Exception:
                hdul = fits.open(fpath, memmap=False)
                data = hdul[0].data
            with hdul:

                for out_idx, frame_idx in entries:
                    frame = data[frame_idx].astype('float32')
                    if roi is not None:
                        y0, y1, x0, x1 = roi
                        frame = frame[y0:y1, x0:x1]
                    if dark_frame is not None:
                        if roi is not None:
                            y0, y1, x0, x1 = roi
                            frame -= dark_frame[y0:y1, x0:x1]
                        else:
                            frame -= dark_frame
                    if transpose:
                        frame = frame.T
                    ds[out_idx] = frame
                    n_written += 1

    print("Wrote %d / %d PLcam frames" % (n_written, N))


# ---------------------------------------------------------------------------
# Unified config entry point  (reads new [Ingest] / [Instrument] schema)
# ---------------------------------------------------------------------------

def ingest_from_config_unified(configname):
    """
    Run ingest_to_h5() using the unified PLred pipeline config schema.

    Reads [Instrument], [Ingest] sections.  The old ingest_from_config()
    (using [Step1]/[PLcam]/[Output] sections) is kept for backward compat.
    """
    from configobj import ConfigObj
    cfg = ConfigObj(configname)

    ingest = cfg.get('Ingest', {})
    instrument = cfg.get('Instrument', {})

    step1_h5 = ingest.get('step1_h5', '').strip()
    if not step1_h5:
        # Fall back to [Sort].output from unified config
        step1_h5 = cfg.get('Sort', {}).get('output', '').strip()
    if not step1_h5:
        raise ValueError("[Ingest] step1_h5 is required (or set [Sort] output)")

    outpath = ingest.get('output', 'alldata.h5').strip() or 'alldata.h5'

    dark = ingest.get('plcam_dark', '').strip()
    plcam_dark = dark if dark else None

    psf_dark = ingest.get('psfcam_dark', '').strip()
    psfcam_dark = psf_dark if psf_dark else None

    data_dir = ingest.get('plcam_data_dir', '').strip()
    plcam_data_dir = data_dir if data_dir else None

    slowcam_data_dir = ingest.get('slowcam_data_dir', '').strip()
    if not slowcam_data_dir:
        # Backward-compatible fallback for older configs that only provided
        # plcam_data_dir when PSF was the slow camera.
        slowcam_data_dir = plcam_data_dir

    roi_str = ingest.get('plcam_roi', '').strip()
    plcam_roi = tuple(int(x) for x in roi_str.split(',')) if roi_str else None

    orientation = instrument.get('spectral_orientation', 'horizontal').strip().lower()

    # [ROIViewer].roi falls back to plcam_roi — store for build_ROI_access_from_config
    _ = cfg.get('ROIViewer', {})

    return ingest_to_h5(
        step1_h5=step1_h5,
        outpath=outpath,
        plcam_dark=plcam_dark,
        psfcam_dark=psfcam_dark,
        plcam_data_dir=plcam_data_dir,
        slowcam_data_dir=slowcam_data_dir,
        plcam_roi=plcam_roi,
        spectral_orientation=orientation,
    )
