"""
H5 product schemas for PLred viewer-compatible files.

Defines the plred_roi_access_v1 layout used by both the offline ROI cache
(average.build_ROI_access) and the live mode (live.py).  Having the schema
in one place ensures both products open correctly in roi_viewer.html.
"""

import os
import numpy as np
import h5py


def create_roi_access_h5(
    path,
    roi,
    attrs=None,
    chunk_t=512,
    compression="gzip",
    compression_opts=4,
):
    """
    Create an empty resizable plred_roi_access_v1 H5.

    Parameters
    ----------
    path : str
        Output file path.  Parent directory is created if needed.
    roi : tuple of int
        (y0, y1, x0, x1) in PLcam pixel coordinates.
    attrs : dict, optional
        Extra file-level attributes (e.g. product_type, created_by,
        dark_subtracted, source_fastcam_dir, ...).  Merged with the
        mandatory layout and roi attrs.
    chunk_t : int
        Time-axis chunk size for all resizable datasets.
    compression, compression_opts : str, int
        HDF5 compression applied to every dataset.

    Datasets created (all resizable along the time axis, N starts at 0):
      /roi_access   float32  (roi_h, roi_w, 0)   chunks=(1, 1, chunk_t)
      /timestamps   float64  (0,)                 chunks=(chunk_t,)
      /centroids    float32  (0, 2)               chunks=(chunk_t, 2)
      /peaks        float32  (0,)                 chunks=(chunk_t,)
      /nstacks      float32  (0,)                 chunks=(chunk_t,)

    File-level attrs always written:
      layout = "plred_roi_access_v1"
      roi    = [y0, y1, x0, x1]
    """
    y0, y1, x0, x1 = roi
    roi_h, roi_w = y1 - y0, x1 - x0
    if roi_h <= 0 or roi_w <= 0:
        raise ValueError("roi (%d,%d,%d,%d) has non-positive extent" % (y0, y1, x0, x1))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    ct = chunk_t
    kw = dict(compression=compression, compression_opts=compression_opts)

    with h5py.File(path, 'w') as f:
        f.attrs['layout'] = 'plred_roi_access_v1'
        f.attrs['roi']    = [y0, y1, x0, x1]
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v

        f.create_dataset(
            'roi_access',
            shape=(roi_h, roi_w, 0),
            maxshape=(roi_h, roi_w, None),
            dtype='float32',
            chunks=(1, 1, ct),
            **kw,
        )
        f.create_dataset(
            'timestamps',
            shape=(0,),
            maxshape=(None,),
            dtype='float64',
            chunks=(ct,),
            **kw,
        )
        f.create_dataset(
            'centroids',
            shape=(0, 2),
            maxshape=(None, 2),
            dtype='float32',
            chunks=(ct, 2),
            **kw,
        )
        f.create_dataset(
            'peaks',
            shape=(0,),
            maxshape=(None,),
            dtype='float32',
            chunks=(ct,),
            **kw,
        )
        f.create_dataset(
            'nstacks',
            shape=(0,),
            maxshape=(None,),
            dtype='float32',
            chunks=(ct,),
            **kw,
        )


def append_roi_access_h5(
    path,
    roi_frames,
    timestamps,
    centroids,
    peaks,
    nstacks=None,
):
    """
    Append new frames to an existing plred_roi_access_v1 H5.

    Parameters
    ----------
    path : str
        Path to an H5 created by create_roi_access_h5.
    roi_frames : ndarray, shape (n_new, roi_h, roi_w), float32
        New PLcam ROI frames.  Stored transposed as (roi_h, roi_w, N)
        so that /roi_access[y, x, :] gives a fast pixel time-series.
    timestamps : ndarray, shape (n_new,), float64
        Unix timestamps for each new frame.
    centroids : ndarray, shape (n_new, 2), float32
        PSF centroid (x, y) per frame.
    peaks : ndarray, shape (n_new,), float32
        PSF peak value (Strehl proxy) per frame.
    nstacks : ndarray, shape (n_new,), float32, optional
        Total exposure weight per frame (from timestamp matching).
        If None, /nstacks is left unchanged.
    """
    roi_frames = np.asarray(roi_frames, dtype='float32')
    timestamps = np.asarray(timestamps, dtype='float64')
    centroids  = np.asarray(centroids,  dtype='float32')
    peaks      = np.asarray(peaks,      dtype='float32')

    n_new = len(timestamps)
    if roi_frames.shape[0] != n_new:
        raise ValueError(
            "roi_frames has %d frames but timestamps has %d" % (roi_frames.shape[0], n_new))
    if centroids.shape != (n_new, 2):
        raise ValueError("centroids must have shape (%d, 2)" % n_new)
    if peaks.shape != (n_new,):
        raise ValueError("peaks must have shape (%d,)" % n_new)

    with h5py.File(path, 'r+') as f:
        old_n = f['timestamps'].shape[0]
        new_n = old_n + n_new

        roi_h, roi_w = f['roi_access'].shape[:2]
        if roi_frames.shape[1:] != (roi_h, roi_w):
            raise ValueError(
                "roi_frames spatial shape %s does not match H5 roi (%d, %d)"
                % (roi_frames.shape[1:], roi_h, roi_w))

        f['roi_access'].resize((roi_h, roi_w, new_n))
        f['roi_access'][:, :, old_n:new_n] = np.moveaxis(roi_frames, 0, -1)

        f['timestamps'].resize((new_n,))
        f['timestamps'][old_n:new_n] = timestamps

        f['centroids'].resize((new_n, 2))
        f['centroids'][old_n:new_n, :] = centroids

        f['peaks'].resize((new_n,))
        f['peaks'][old_n:new_n] = peaks

        f['nstacks'].resize((new_n,))
        if nstacks is not None:
            f['nstacks'][old_n:new_n] = np.asarray(nstacks, dtype='float32')
        else:
            f['nstacks'][old_n:new_n] = np.nan
