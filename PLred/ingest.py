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
    plcam_data_dir=None,
    plcam_roi=None,
    plcam_spectra=None,
    compression='gzip',
    compression_opts=4,
    verbose=False,
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
    psfcam_h, psfcam_w = psfcam_frames.shape[1], psfcam_frames.shape[2]
    psfcam_is_fast = meta1.get('psfcam_is_fast', True)

    slowcam_ts_files = meta1['slowcam_timestampfiles']
    slowcam_fileinds = np.array(meta1['slowcam_fileinds'])
    slowcam_frameinds = np.array(meta1['slowcam_frameinds'])
    matched_indices  = {int(k): v for k, v in meta1['matched_indices'].items()}
    matched_sc_inds  = sorted(matched_indices.keys())   # in time order

    # Resolve PLcam FITS file paths
    plcam_fits_files = _resolve_plcam_files(slowcam_ts_files, plcam_data_dir)

    # ------------------------------------------------------------------
    # 2. Load PLcam dark
    # ------------------------------------------------------------------
    if plcam_dark is None:
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

    if plcam_type == 'raw':
        ny, nx, roi = _probe_plcam_shape(plcam_fits_files, plcam_roi, dark_frame)
        if verbose:
            print("PLcam frame shape after ROI: (%d, %d)" % (ny, nx))
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

    for i, frame in enumerate(tqdm(psfcam_frames, disable=not verbose)):
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
    config_dict = {
        'step1_h5':       step1_h5,
        'plcam_dark':     str(plcam_dark) if not isinstance(plcam_dark, np.ndarray) else '<array>',
        'plcam_data_dir': plcam_data_dir,
        'plcam_roi':      list(plcam_roi) if plcam_roi else None,
        'plcam_type':     plcam_type,
        'compression':    compression,
        'write_time':     datetime.now().isoformat(),
    }

    print("Writing giant H5: %s" % outpath)
    with h5py.File(outpath, 'w') as h5f:
        h5f.attrs['n_frames']      = N
        h5f.attrs['psfcam_is_fast'] = psfcam_is_fast
        h5f.attrs['plcam_type']    = plcam_type

        # metadata group — store relative timestamps (seconds from first frame)
        t0 = float(timestamps[0])
        meta_grp = h5f.create_group('metadata')
        meta_grp.create_dataset('timestamps', data=timestamps - t0, dtype='float64')
        meta_grp.create_dataset('t0',         data=t0,              dtype='float64')
        meta_grp.create_dataset('config',     data=json.dumps(config_dict))
        meta_grp.attrs['n_frames'] = N
        meta_grp.attrs['t0']       = t0

        # psfcam group
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

        # plcam group — pre-allocate, fill below
        pl_grp = h5f.create_group('plcam')

        if plcam_type == 'spectra':
            pl_grp.create_dataset('spectra',
                                  data=spectra_array.astype('float32'),
                                  dtype='float32',
                                  chunks=(1, Nlambda, Nport),
                                  compression=compression,
                                  compression_opts=compression_opts)
        else:
            # chunk=(1, ny, nx): each frame is its own chunk so writes are
            # independent — no read-modify-write of neighbouring frames.
            frames_ds = pl_grp.create_dataset(
                'frames',
                shape=(N, ny, nx),
                dtype='float32',
                chunks=(1, ny, nx),
                compression=compression,
                compression_opts=compression_opts,
            )
            if plcam_roi:
                pl_grp.attrs['roi'] = list(plcam_roi)

    # ------------------------------------------------------------------
    # 6. Fill PLcam frames (file by file to keep RAM constant)
    # ------------------------------------------------------------------
    if plcam_type == 'raw':
        _write_plcam_frames(
            outpath, plcam_fits_files, matched_sc_inds,
            slowcam_fileinds, slowcam_frameinds,
            dark_frame, roi, N, verbose,
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
                         dark_frame, roi, N, verbose):
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
                    ds[out_idx] = frame
                    n_written += 1

    print("Wrote %d / %d PLcam frames" % (n_written, N))
