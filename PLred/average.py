"""
Step 3: Grid exploration and PLcam averaging.

Workflow
--------
1. explore_grid()  — tune map_n / map_width / filters interactively.
                     Reads only centroids+peaks+timestamps (fast always).
                     Optionally reads a few PLcam pixels to preview response maps.
2. average_to_h5() — once happy with parameters, write compact averaged H5.
3. pixel_map()     — one-liner to extract a 2D response map from the averaged H5.

Output H5 structure (average_to_h5)
------------------------------------
averaged.h5
  attrs: map_n, map_width, xc, yc, pix2mas, plcam_type
  /metadata/
    config    str  JSON of all parameters
    xbins     float64 (map_n+1,)
    ybins     float64 (map_n+1,)
    x_mas     float64 (map_n,)
    y_mas     float64 (map_n,)
    nframes   int32   (map_n, map_n)   — 0 for empty bins
    timestamps str    JSON list-of-lists, one per bin
  /avg_PSFcam float32 (map_n, map_n, h, w)    — NaN for empty bins
  /avg_PLcam  float32 (map_n, map_n, ny, nx)  — NaN for empty bins
              OR      (map_n, map_n, Nlambda, Nport) for spectra mode
  /bootstrap/
    avg_PLcam float32 (n_bootstrap, map_n, map_n, ny, nx)  [if n_bootstrap > 0]

Timestamp convention
--------------------
Timestamps in the giant H5 (/metadata/timestamps) are stored as **relative seconds
from the first frame** so they start at 0.  The absolute Unix timestamp of the first
frame is stored as /metadata/t0 (float64) and as f.attrs['t0'].

Use relative times when calling explore_grid() and average_to_h5():
    time_min=10.0, time_max=60.0   →  frames 10–60 s after observation start

To recover the absolute time of any frame:
    abs_time = t0 + timestamps[i]
"""

import numpy as np
import h5py
import json
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

try:
    import zarr
    _ZARR_AVAILABLE = True
except ImportError:
    _ZARR_AVAILABLE = False


def _make_zarr_gzip(level):
    """Return a gzip compressor compatible with installed zarr version(s)."""
    # Preferred path (works with common zarr + numcodecs setups)
    try:
        from numcodecs import GZip as _NumcodecsGZip
        return _NumcodecsGZip(level=level)
    except Exception:
        pass

    # Older zarr exposed GZip directly
    if _ZARR_AVAILABLE and hasattr(zarr, 'GZip'):
        return zarr.GZip(level=level)

    # Newer zarr may expose codecs namespace
    if _ZARR_AVAILABLE and hasattr(zarr, 'codecs') and hasattr(zarr.codecs, 'GzipCodec'):
        return zarr.codecs.GzipCodec(level=level)

    raise ImportError(
        "Could not construct a gzip compressor for zarr. "
        "Install numcodecs (recommended): pip install numcodecs"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_giant_meta(giant_h5):
    """Read centroids, peaks, relative timestamps (seconds from t0), t0, and plcam_type."""
    with h5py.File(giant_h5, 'r') as f:
        centroids  = f['psfcam/centroids'][:]    # (N, 2)
        peaks      = f['psfcam/peaks'][:]         # (N,)
        timestamps = f['metadata/timestamps'][:]  # (N,) seconds from t0
        plcam_type = f.attrs.get('plcam_type', 'raw')
        # t0 may be absent in older giant H5 files — fall back gracefully
        t0 = float(f['metadata/t0'][()]) if 'metadata/t0' in f else None
    return centroids, peaks, timestamps, plcam_type, t0


def _build_grid(centroids, map_n, map_width, xc, yc, pix2mas):
    """Compute bin edges and per-frame bin assignments."""
    if xc is None:
        cl = sigma_clip(centroids[:, 0], sigma=3, maxiters=5)
        xc = float(np.nanmedian(cl.data[~cl.mask]))
    if yc is None:
        cl = sigma_clip(centroids[:, 1], sigma=3, maxiters=5)
        yc = float(np.nanmedian(cl.data[~cl.mask]))

    xbins = np.linspace(xc - map_width / 2, xc + map_width / 2, map_n + 1)
    ybins = np.linspace(yc - map_width / 2, yc + map_width / 2, map_n + 1)

    dx = np.diff(xbins)[0]
    dy = np.diff(ybins)[0]
    x_mas = (xbins[:-1] + dx / 2 - xc) * pix2mas
    y_mas = (ybins[:-1] + dy / 2 - yc) * pix2mas

    # Assign each frame to a bin (-1 if outside grid)
    xi = np.floor((centroids[:, 0] - xbins[0]) / dx).astype(int)
    yi = np.floor((centroids[:, 1] - ybins[0]) / dy).astype(int)
    outside = (xi < 0) | (xi >= map_n) | (yi < 0) | (yi >= map_n)
    xi[outside] = -1
    yi[outside] = -1

    return xbins, ybins, x_mas, y_mas, xc, yc, xi, yi


def _apply_filter(peaks, timestamps, strehl_min, strehl_max, time_min, time_max):
    """Boolean mask of frames that pass all filters."""
    mask = np.ones(len(peaks), dtype=bool)
    if strehl_min is not None:
        mask &= peaks >= strehl_min
    if strehl_max is not None:
        mask &= peaks <= strehl_max
    if time_min is not None:
        mask &= timestamps >= time_min
    if time_max is not None:
        mask &= timestamps <= time_max
    return mask


def _has_roi_access(f, key="plcam/roi_access"):
    """Return True if the ROI time-series cache dataset exists in the open HDF5 file."""
    return key in f


# ---------------------------------------------------------------------------
# build_ROI_access
# ---------------------------------------------------------------------------

def build_ROI_access(
    giant_h5,
    roi,
    out_key="plcam/roi_access",
    chunk_t=2048,
    compression="gzip",
    compression_opts=4,
    overwrite=False,
    zarr_path=None,
    zarr_zip=True,
    zarr_include_metadata=True,
    zarr_include_psf=False,
    zarr_psf_chunk_t=256,
    verbose=True,
):
    """
    Build a transposed ROI cache for fast pixel time-series reads in explore_grid().

    Two complementary layouts live in the same giant H5:

      /plcam/frames     (N, ny, nx)       chunks=(1, ny, nx)    — frame-wise averaging
      /plcam/roi_access (roi_h, roi_w, N) chunks=(1, 1, chunk_t) — pixel time-series

    Reading roi_access[local_y, local_x, :] decompresses ceil(N/chunk_t) chunks
    instead of N full-frame chunks, giving ~ny*nx/chunk_t speedup for explore_grid().
    ROI cache is optional and should cover only the pixels of interest, e.g. a
    single spectral trace (400×1) or a small sky region.

    Parameters
    ----------
    giant_h5 : str
        Path to the giant H5. Opened in r+ mode.
    roi : tuple of int
        (y0, y1, x0, x1) in original PLcam pixel coordinates.
    out_key : str
        HDF5 dataset path for the cache.
    chunk_t : int
        Time-axis chunk size.
    compression, compression_opts : str, int
        HDF5 compression.
    overwrite : bool
        If True, delete and rebuild any existing cache.
    zarr_path : str, optional
        If given, also write the cache to a Zarr store at this path.
        By default (`zarr_zip=True`), this is written as a zip store. If the
        provided path does not end with `.zip`, `.zip` is appended.
        Set `zarr_zip=False` to write a directory store instead.
        Useful for HTML viewers (zarr.js / zarrita compatible).
        Requires the `zarr` package.
    zarr_zip : bool
        If True (default), write Zarr as a single `.zip` file. If False,
        write a directory store at `zarr_path`.
    zarr_include_metadata : bool
        If True and zarr_path is provided, also store per-frame metadata arrays
        in the Zarr root group:
          /timestamps  (N,) float64  (seconds from t0)
          /peaks       (N,) float32
          /centroids   (N,2) float32
        and optional root attr 't0' when available.
    zarr_include_psf : bool
        If True and zarr_path is provided, also store /psf_frames
        with shape (N, h, w). This can significantly increase Zarr size.
    zarr_psf_chunk_t : int
        Time-axis chunk size for /psf_frames in Zarr when
        zarr_include_psf=True.
    verbose : bool
    """
    y0, y1, x0, x1 = roi
    roi_h, roi_w = y1 - y0, x1 - x0

    with h5py.File(giant_h5, 'r+') as f:
        plcam_type = f.attrs.get('plcam_type', 'raw')
        if plcam_type != 'raw':
            raise ValueError(
                "build_ROI_access only supports plcam_type='raw', got '%s'" % plcam_type)

        src = f['plcam/frames']   # (N, ny, nx)
        N, ny, nx = src.shape

        if y0 < 0 or y1 > ny or x0 < 0 or x1 > nx or roi_h <= 0 or roi_w <= 0:
            raise ValueError(
                "roi (%d,%d,%d,%d) out of bounds for frames shape (%d,%d)" % (
                    y0, y1, x0, x1, ny, nx))

        if out_key in f:
            if not overwrite:
                if verbose:
                    print("'%s' already exists. Pass overwrite=True to rebuild." % out_key)
                dst = f[out_key]
            else:
                if verbose:
                    print("Deleting existing '%s' (overwrite=True)" % out_key)
                del f[out_key]
                dst = None
        else:
            dst = None

        ct = min(chunk_t, N)

        if dst is None:
            if verbose:
                print("Building roi_access: roi=(%d,%d,%d,%d)  shape=(%d,%d,%d)  chunks=(1,1,%d)" % (
                    y0, y1, x0, x1, roi_h, roi_w, N, ct))
            grp_key, ds_name = out_key.rsplit('/', 1)
            grp = f.require_group(grp_key)
            dst = grp.create_dataset(
                ds_name,
                shape=(roi_h, roi_w, N),
                dtype='float32',
                chunks=(1, 1, ct),
                compression=compression,
                compression_opts=compression_opts,
            )
            dst.attrs['roi']    = [y0, y1, x0, x1]
            dst.attrs['source'] = '/plcam/frames'
            dst.attrs['layout'] = 'transposed_roi_time_access'

        # Open zarr store alongside if requested
        z_dst = None
        z_root = None
        if zarr_path is not None:
            if not _ZARR_AVAILABLE:
                raise ImportError(
                    "zarr is required for zarr_path output. "
                    "Install with: pip install zarr")
            compressor = _make_zarr_gzip(level=compression_opts)
            zarr_target = zarr_path
            if zarr_zip:
                if not zarr_target.endswith('.zip'):
                    zarr_target = zarr_target + '.zip'
                store = zarr.ZipStore(zarr_target, mode='w')
            else:
                store = zarr_target
            z_root = zarr.open_group(store, mode='w')
            z_dst = z_root.create_dataset(
                'roi_access',
                shape=(roi_h, roi_w, N), dtype='float32',
                chunks=(1, 1, ct), compressor=compressor,
            )
            z_root.attrs['layout'] = 'plred_roi_access_v1'
            z_root.attrs['roi'] = [y0, y1, x0, x1]
            z_root.attrs['source'] = '/plcam/frames'
            z_root.attrs['n_frames'] = int(N)
            z_root.attrs['plcam_type'] = str(plcam_type)
            z_dst.attrs['layout'] = 'transposed_roi_time_access'

            if zarr_include_metadata:
                ts = f['metadata/timestamps'][:].astype('float64')
                pk = f['psfcam/peaks'][:].astype('float32')
                cc = f['psfcam/centroids'][:].astype('float32')
                z_root.create_dataset(
                    'timestamps', data=ts,
                    dtype='float64', chunks=(min(ct, N),), compressor=compressor,
                )
                z_root.create_dataset(
                    'peaks', data=pk,
                    dtype='float32', chunks=(min(ct, N),), compressor=compressor,
                )
                z_root.create_dataset(
                    'centroids', data=cc,
                    dtype='float32', chunks=(min(ct, N), 2), compressor=compressor,
                )
                if 'metadata/t0' in f:
                    z_root.attrs['t0'] = float(f['metadata/t0'][()])

            if zarr_include_psf:
                psf_src = f['psfcam/frames']
                _, psf_h, psf_w = psf_src.shape
                ct_psf = max(1, min(int(zarr_psf_chunk_t), N))
                z_psf = z_root.create_dataset(
                    'psf_frames',
                    shape=(N, psf_h, psf_w),
                    dtype='float32',
                    chunks=(ct_psf, psf_h, psf_w),
                    compressor=compressor,
                )
                if verbose:
                    psf_mb = N * psf_h * psf_w * 4 / 1e6
                    print(
                        "Including psf_frames in Zarr: shape=(%d,%d,%d), chunks=(%d,%d,%d), "
                        "raw_size≈%.0f MB" % (N, psf_h, psf_w, ct_psf, psf_h, psf_w, psf_mb)
                    )
                for k0 in tqdm(range(0, N, ct_psf), desc='Writing zarr psf_frames', disable=not verbose):
                    k1 = min(k0 + ct_psf, N)
                    z_psf[k0:k1] = psf_src[k0:k1].astype('float32')

            if verbose:
                print("Also writing Zarr store: %s" % zarr_target)

        for t0 in tqdm(range(0, N, ct), desc='Building roi_access', disable=not verbose):
            t1 = min(t0 + ct, N)
            block = src[t0:t1, y0:y1, x0:x1]              # (t1-t0, roi_h, roi_w)
            transposed = np.moveaxis(block, 0, -1)          # (roi_h, roi_w, t1-t0)
            dst[:, :, t0:t1] = transposed
            if z_dst is not None:
                z_dst[:, :, t0:t1] = transposed

        if z_dst is not None and zarr_zip:
            z_dst.store.close()

        if verbose:
            print("roi_access written to '%s'" % out_key)
            if zarr_path is not None:
                zarr_final = zarr_path + '.zip' if (zarr_zip and not zarr_path.endswith('.zip')) else zarr_path
                print("Zarr cache written to '%s'" % zarr_final)
                if zarr_include_metadata:
                    print("  included: /timestamps, /peaks, /centroids (+ attr t0 if present)")
                if zarr_include_psf:
                    print("  included: /psf_frames")

    return out_key


# ---------------------------------------------------------------------------
# explore_grid
# ---------------------------------------------------------------------------

def explore_grid(
    giant_h5,
    map_n,
    map_width,
    xc=None, yc=None,
    time_min=None, time_max=None,
    strehl_min=None, strehl_max=None,
    pix2mas=16.2,
    plcam_pixels=None,
    roi_access_key="plcam/roi_access",
    plot=True,
):
    """
    Explore grid parameters and preview response maps interactively.

    Parameters
    ----------
    giant_h5 : str
        Path to Step 2 giant H5.
    map_n : int
        Grid resolution (map_n × map_n bins).
    map_width : float
        Grid FOV in PSF-camera pixels.
    xc, yc : float, optional
        Grid centre in PSF pixels. Default: sigma-clipped median of centroids.
    time_min, time_max : float, optional
        Unix timestamp bounds for frame selection.
    strehl_min, strehl_max : float, optional
        PSF peak value bounds.
    pix2mas : float
        Plate scale mas/pixel of PSF camera.
    plcam_pixels : list of (int, int), optional
        List of (py, px) pixel coordinates to extract from PLcam and bin.
        Each pixel produces one 2D response map. For spectra mode, pass
        (ilambda, iport) pairs. If None, only grid coverage is shown.
    roi_access_key : str
        HDF5 key of the transposed ROI cache built by build_ROI_access().
        If the dataset exists, pixel reads use the fast path (~ceil(N/chunk_t)
        chunk reads). If absent, falls back to the slow ds[:, py, px] path.
    plot : bool
        Show diagnostic plots.

    Returns
    -------
    result : dict
        xbins, ybins, x_mas, y_mas, xc, yc,
        mask           — bool (N,) frames passing filters
        nframes_map    — int (map_n, map_n)
        avg_psf_map    — float32 (map_n, map_n, h, w) or None
        pixel_maps     — {(py, px): float32 (map_n, map_n)} or {}
    """
    centroids, peaks, timestamps, plcam_type, t0 = _load_giant_meta(giant_h5)
    N = len(timestamps)

    if t0 is not None:
        from datetime import datetime as _dt
        print("Observation start (t0): %s  (Unix %.3f)" % (
            _dt.utcfromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S UTC'), t0))
        print("timestamps are relative seconds from t0  "
              "[0.0, %.3f]" % float(timestamps[-1]))
    else:
        print("timestamps: [%.3f, %.3f]" % (float(timestamps[0]), float(timestamps[-1])))

    # Filter
    filt_mask = _apply_filter(peaks, timestamps, strehl_min, strehl_max, time_min, time_max)
    n_kept = filt_mask.sum()
    print("Frames after filter: %d / %d  (%.1f%%)" % (n_kept, N, 100 * n_kept / N))

    # Build grid on filtered frames
    xbins, ybins, x_mas, y_mas, xc, yc, xi, yi = _build_grid(
        centroids[filt_mask], map_n, map_width, xc, yc, pix2mas)

    # nframes per bin
    nframes_map = np.zeros((map_n, map_n), dtype=int)
    xi_f = xi  # already computed on filtered frames
    yi_f = yi
    for i in range(map_n):
        for j in range(map_n):
            nframes_map[i, j] = int(np.sum((xi_f == i) & (yi_f == j)))

    print("Grid %dx%d  FOV %.1f×%.1f mas  filled bins: %d / %d" % (
        map_n, map_n,
        (xbins[-1] - xbins[0]) * pix2mas, (ybins[-1] - ybins[0]) * pix2mas,
        (nframes_map > 0).sum(), map_n * map_n))

    # Average PSF per bin
    avg_psf_map = None
    with h5py.File(giant_h5, 'r') as f:
        psfframes_all = f['psfcam/frames'][:]   # (N, h, w) — small, load all
    psfframes_filt = psfframes_all[filt_mask]
    h, w = psfframes_filt.shape[1], psfframes_filt.shape[2]
    avg_psf_map = np.full((map_n, map_n, h, w), np.nan, dtype='float32')
    for i in range(map_n):
        for j in range(map_n):
            bin_mask = (xi_f == i) & (yi_f == j)
            if bin_mask.sum() > 0:
                avg_psf_map[i, j] = np.mean(psfframes_filt[bin_mask], axis=0)

    # PLcam pixel extraction
    pixel_maps = {}
    if plcam_pixels:
        with h5py.File(giant_h5, 'r') as f:
            if plcam_type == 'raw':
                ds = f['plcam/frames']
                use_roi = _has_roi_access(f, roi_access_key)

                if use_roi:
                    roi_ds = f[roi_access_key]
                    roi_bounds = list(roi_ds.attrs['roi'])   # [y0, y1, x0, x1]
                    y0, y1, x0, x1 = roi_bounds
                    print("PLcam pixel read (%d pixels): using roi_access %s  "
                          "roi=(%d,%d,%d,%d)  fast path" % (
                              len(plcam_pixels), roi_access_key, y0, y1, x0, x1))
                else:
                    ny, nx = ds.shape[1], ds.shape[2]
                    total_mb = N * ny * nx * 4 / 1e6
                    print("WARNING: '%s' not found — falling back to ds[:, py, px], "
                          "which decompresses one full frame (%dx%d) per timestamp "
                          "(~%.0f MB per pixel). Call build_ROI_access() for fast access." % (
                              roi_access_key, ny, nx, total_mb))

                for py, px in plcam_pixels:
                    if use_roi:
                        if not (y0 <= py < y1 and x0 <= px < x1):
                            raise ValueError(
                                "pixel (%d, %d) is outside roi (%d,%d,%d,%d). "
                                "Rebuild roi_access with a larger roi." % (
                                    py, px, y0, y1, x0, x1))
                        local_y = py - y0
                        local_x = px - x0
                        values_all = roi_ds[local_y, local_x, :].astype('float32')
                    else:
                        values_all = ds[:, py, px].astype('float32')

                    values_filt = values_all[filt_mask]
                    pmap = np.full((map_n, map_n), np.nan, dtype='float32')
                    for i in range(map_n):
                        for j in range(map_n):
                            bin_mask = (xi_f == i) & (yi_f == j)
                            if bin_mask.sum() > 0:
                                pmap[i, j] = float(np.nanmean(values_filt[bin_mask]))
                    pixel_maps[(py, px)] = pmap
                    print("  pixel (%d, %d)  map range [%.3g, %.3g]" % (
                        py, px, np.nanmin(pmap), np.nanmax(pmap)))

            else:
                # Spectra mode: unchanged
                ds = f['plcam/spectra']
                ny, nx = ds.shape[1], ds.shape[2]
                total_mb = N * ny * nx * 4 / 1e6
                print("PLcam spectra pixel read: %d pixels × %d frames  "
                      "(~%.0f MB per pixel)" % (len(plcam_pixels), N, total_mb))

                for py, px in plcam_pixels:
                    values_all = ds[:, py, px].astype('float32')
                    values_filt = values_all[filt_mask]
                    pmap = np.full((map_n, map_n), np.nan, dtype='float32')
                    for i in range(map_n):
                        for j in range(map_n):
                            bin_mask = (xi_f == i) & (yi_f == j)
                            if bin_mask.sum() > 0:
                                pmap[i, j] = float(np.nanmean(values_filt[bin_mask]))
                    pixel_maps[(py, px)] = pmap
                    print("  pixel (%d, %d)  map range [%.3g, %.3g]" % (
                        py, px, np.nanmin(pmap), np.nanmax(pmap)))

    if plot:
        _plot_explore(nframes_map, avg_psf_map, pixel_maps, x_mas, y_mas, map_n)

    return {
        'xbins': xbins, 'ybins': ybins,
        'x_mas': x_mas, 'y_mas': y_mas,
        'xc': xc, 'yc': yc,
        'mask': filt_mask,
        'nframes_map': nframes_map,
        'avg_psf_map': avg_psf_map,
        'pixel_maps': pixel_maps,
        't0': t0,   # absolute Unix timestamp of first frame; timestamps are seconds from t0
    }


def _plot_explore(nframes_map, avg_psf_map, pixel_maps, x_mas, y_mas, map_n):
    """Plot nframes map, PSF thumbnails, and optional pixel response maps."""
    n_pixel_plots = len(pixel_maps)
    ncols = 2 + n_pixel_plots
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    ext = (x_mas[0], x_mas[-1], y_mas[-1], y_mas[0])

    # nframes map
    ax = axes[0]
    im = ax.imshow(nframes_map, origin='upper', extent=ext)
    plt.colorbar(im, ax=ax, label='nframes')
    ax.set_title('Frames per bin')
    ax.set_xlabel('x (mas)'); ax.set_ylabel('y (mas)')

    # Avg PSF thumbnails (mosaic)
    ax = axes[1]
    if avg_psf_map is not None:
        h, w = avg_psf_map.shape[2], avg_psf_map.shape[3]
        mosaic = np.full((map_n * h, map_n * w), np.nan)
        for i in range(map_n):
            for j in range(map_n):
                if not np.all(np.isnan(avg_psf_map[i, j])):
                    mosaic[i*h:(i+1)*h, j*w:(j+1)*w] = avg_psf_map[i, j]
        ax.imshow(mosaic, origin='upper', cmap='viridis')
    ax.set_title('Avg PSF per bin')
    ax.axis('off')

    # Pixel response maps
    for k, ((py, px), pmap) in enumerate(pixel_maps.items()):
        ax = axes[2 + k]
        im = ax.imshow(pmap, origin='upper', extent=ext)
        plt.colorbar(im, ax=ax)
        ax.set_title('PLcam pixel (%d, %d)' % (py, px))
        ax.set_xlabel('x (mas)'); ax.set_ylabel('y (mas)')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# average_to_h5
# ---------------------------------------------------------------------------

def average_to_h5(
    giant_h5,
    outpath,
    map_n,
    map_width,
    xc=None, yc=None,
    time_min=None, time_max=None,
    strehl_min=None, strehl_max=None,
    pix2mas=16.2,
    n_bootstrap=0,
    time_chunk_minutes=None,
    verbose=False,
):
    """
    Average PLcam and PSFcam frames per spatial bin and write compact H5.

    Parameters
    ----------
    giant_h5 : str
        Path to Step 2 giant H5.
    outpath : str
        Output path. For time chunking, '_chunk_000.h5' etc. are appended.
    map_n : int
        Grid resolution.
    map_width : float
        Grid FOV in PSF pixels.
    xc, yc : float, optional
        Grid centre.
    time_min, time_max : float, optional
        Unix timestamp bounds.
    strehl_min, strehl_max : float, optional
        PSF peak bounds.
    pix2mas : float
        Plate scale mas/pixel.
    n_bootstrap : int
        Number of bootstrap samples (0 = disabled).
    time_chunk_minutes : float, optional
        If set, split into chunks of this duration and write one file each.
    verbose : bool

    Returns
    -------
    str or list of str  — output path(s)
    """
    centroids, peaks, timestamps, plcam_type, t0 = _load_giant_meta(giant_h5)

    filt_mask = _apply_filter(peaks, timestamps, strehl_min, strehl_max, time_min, time_max)
    filt_indices = np.where(filt_mask)[0]

    if time_chunk_minutes is not None:
        return _average_chunked(
            giant_h5, outpath, map_n, map_width, xc, yc,
            pix2mas, n_bootstrap, time_chunk_minutes,
            centroids, peaks, timestamps, plcam_type, t0,
            filt_mask, filt_indices,
            strehl_min, strehl_max, time_min, time_max, verbose,
        )

    return _average_one(
        giant_h5, outpath, map_n, map_width, xc, yc, pix2mas, n_bootstrap,
        centroids, peaks, timestamps, plcam_type, t0,
        filt_mask, filt_indices,
        strehl_min, strehl_max, time_min, time_max, verbose,
    )


def _average_one(
    giant_h5, outpath, map_n, map_width, xc, yc, pix2mas, n_bootstrap,
    centroids, peaks, timestamps, plcam_type, t0,
    filt_mask, filt_indices,
    strehl_min, strehl_max, time_min, time_max, verbose,
):
    """Average one time window and write to outpath."""
    n_kept = len(filt_indices)
    print("Averaging %d frames → %s" % (n_kept, outpath))

    xbins, ybins, x_mas, y_mas, xc, yc, xi, yi = _build_grid(
        centroids[filt_mask], map_n, map_width, xc, yc, pix2mas)

    # Bin assignments for filtered frames (global indices)
    xi_g = np.full(len(centroids), -1, dtype=int)
    yi_g = np.full(len(centroids), -1, dtype=int)
    xi_g[filt_mask] = xi
    yi_g[filt_mask] = yi

    # Read PSF frames (small, load all at once)
    with h5py.File(giant_h5, 'r') as f:
        psfframes_all = f['psfcam/frames'][:]
        psfcam_h, psfcam_w = psfframes_all.shape[1], psfframes_all.shape[2]
        plcam_key = 'plcam/frames' if plcam_type == 'raw' else 'plcam/spectra'
        plcam_shape = f[plcam_key].shape   # (N, ny, nx) or (N, Nlambda, Nport)

    plcam_dims = plcam_shape[1:]  # (ny, nx) or (Nlambda, Nport)

    # Compute per-bin averages
    avg_psf  = np.full((map_n, map_n, psfcam_h, psfcam_w), np.nan, dtype='float64')
    avg_pl   = np.full((map_n, map_n) + plcam_dims, np.nan, dtype='float64')
    nframes  = np.zeros((map_n, map_n), dtype=int)
    ts_lists = [[[] for _ in range(map_n)] for _ in range(map_n)]

    # PSF averages (fast, all in memory)
    for i in range(map_n):
        for j in range(map_n):
            bmask = filt_mask & (xi_g == i) & (yi_g == j)
            nf = bmask.sum()
            nframes[i, j] = nf
            if nf > 0:
                avg_psf[i, j] = np.mean(psfframes_all[bmask], axis=0)
                ts_lists[i][j] = timestamps[bmask].tolist()

    # PLcam averages: read frame by frame to keep RAM constant
    print("Reading PLcam frames...")
    pl_acc   = np.zeros((map_n, map_n) + plcam_dims, dtype='float64')
    pl_count = np.zeros((map_n, map_n), dtype=int)

    with h5py.File(giant_h5, 'r') as f:
        ds = f[plcam_key]
        for idx in tqdm(filt_indices, disable=not verbose, desc='PLcam frames'):
            i, j = int(xi_g[idx]), int(yi_g[idx])
            if i < 0:
                continue
            pl_acc[i, j]   += ds[idx].astype('float64')
            pl_count[i, j] += 1

    for i in range(map_n):
        for j in range(map_n):
            if pl_count[i, j] > 0:
                avg_pl[i, j] = pl_acc[i, j] / pl_count[i, j]

    # Bootstrap
    boot_avg_pl = None
    if n_bootstrap > 0:
        print("Computing %d bootstrap samples..." % n_bootstrap)
        boot_avg_pl = np.full(
            (n_bootstrap, map_n, map_n) + plcam_dims, np.nan, dtype='float32')

        # Collect per-bin frame indices (global)
        bin_frame_inds = {}
        for i in range(map_n):
            for j in range(map_n):
                bmask = filt_mask & (xi_g == i) & (yi_g == j)
                inds = np.where(bmask)[0]
                if len(inds) > 0:
                    bin_frame_inds[(i, j)] = inds

        with h5py.File(giant_h5, 'r') as f:
            ds = f[plcam_key]
            for k in tqdm(range(n_bootstrap), desc='Bootstrap'):
                for (i, j), inds in bin_frame_inds.items():
                    sample = np.random.choice(inds, size=len(inds), replace=True)
                    acc = np.zeros(plcam_dims, dtype='float64')
                    for idx in sample:
                        acc += ds[idx].astype('float64')
                    boot_avg_pl[k, i, j] = (acc / len(sample)).astype('float32')

    # Write output
    _write_averaged_h5(
        outpath, map_n, map_width, xc, yc, pix2mas, plcam_type, t0,
        xbins, ybins, x_mas, y_mas, nframes, ts_lists,
        avg_psf.astype('float32'), avg_pl.astype('float32'),
        boot_avg_pl,
        strehl_min, strehl_max, time_min, time_max, n_bootstrap,
    )
    return outpath


def _average_chunked(
    giant_h5, outpath, map_n, map_width, xc, yc, pix2mas, n_bootstrap,
    time_chunk_minutes,
    centroids, peaks, timestamps, plcam_type, t0,
    filt_mask, filt_indices,
    strehl_min, strehl_max, time_min, time_max, verbose,
):
    """Split filtered frames into time chunks and write one file per chunk."""
    chunk_sec = time_chunk_minutes * 60.0
    ts_filt = timestamps[filt_mask]

    if len(ts_filt) == 0:
        print("WARNING: no frames pass filters, nothing to write.")
        return []

    t_start = ts_filt.min()
    t_end   = ts_filt.max()
    n_chunks = int(np.ceil((t_end - t_start) / chunk_sec))
    print("Time chunking: %.1f min chunks → %d files" % (time_chunk_minutes, n_chunks))

    base, ext = os.path.splitext(outpath)
    if not ext:
        ext = '.h5'

    out_paths = []
    for k in range(n_chunks):
        chunk_t0 = t_start + k * chunk_sec
        chunk_t1 = t_start + (k + 1) * chunk_sec

        # Build chunk mask within already-filtered frames
        chunk_mask_full = filt_mask.copy()
        chunk_mask_full &= (timestamps >= chunk_t0) & (timestamps < chunk_t1)
        chunk_inds = np.where(chunk_mask_full)[0]

        if len(chunk_inds) == 0:
            print("  Chunk %03d: no frames, skipping" % k)
            continue

        chunk_path = '%s_chunk_%03d%s' % (base, k, ext)
        print("  Chunk %03d: %d frames  [%.1f, %.1f)" % (
            k, len(chunk_inds), chunk_t0 - t_start, chunk_t1 - t_start))

        # Determine grid center from first chunk if not given
        _xc = xc
        _yc = yc

        _average_one(
            giant_h5, chunk_path, map_n, map_width, _xc, _yc, pix2mas, n_bootstrap,
            centroids, peaks, timestamps, plcam_type, t0,
            chunk_mask_full, chunk_inds,
            strehl_min, strehl_max, chunk_t0, chunk_t1, verbose,
        )
        out_paths.append(chunk_path)

    return out_paths


def _write_averaged_h5(
    outpath, map_n, map_width, xc, yc, pix2mas, plcam_type, t0,
    xbins, ybins, x_mas, y_mas, nframes, ts_lists,
    avg_psf, avg_pl, boot_avg_pl,
    strehl_min, strehl_max, time_min, time_max, n_bootstrap,
):
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    config = {
        'map_n': map_n, 'map_width': map_width, 'xc': xc, 'yc': yc,
        'pix2mas': pix2mas, 'plcam_type': plcam_type,
        'strehl_min': strehl_min, 'strehl_max': strehl_max,
        'time_min': time_min, 'time_max': time_max,
        'n_bootstrap': n_bootstrap,
        't0': t0,
        'write_time': datetime.now().isoformat(),
    }
    with h5py.File(outpath, 'w') as f:
        f.attrs['map_n']      = map_n
        f.attrs['map_width']  = map_width
        f.attrs['xc']         = xc
        f.attrs['yc']         = yc
        f.attrs['pix2mas']    = pix2mas
        f.attrs['plcam_type'] = plcam_type
        if t0 is not None:
            f.attrs['t0'] = t0   # absolute Unix timestamp of first frame

        meta = f.create_group('metadata')
        meta.create_dataset('config',  data=json.dumps(config))
        meta.create_dataset('xbins',   data=xbins,   dtype='float64')
        meta.create_dataset('ybins',   data=ybins,   dtype='float64')
        meta.create_dataset('x_mas',   data=x_mas,   dtype='float64')
        meta.create_dataset('y_mas',   data=y_mas,   dtype='float64')
        meta.create_dataset('nframes', data=nframes, dtype='int32')
        meta.create_dataset('timestamps', data=json.dumps(ts_lists))
        # bin-level timestamps are also relative seconds from t0
        if t0 is not None:
            meta.create_dataset('t0', data=t0, dtype='float64')

        f.create_dataset('avg_PSFcam', data=avg_psf, dtype='float32',
                         compression='gzip', compression_opts=4)
        f.create_dataset('avg_PLcam',  data=avg_pl,  dtype='float32',
                         compression='gzip', compression_opts=4)

        if boot_avg_pl is not None:
            boot = f.create_group('bootstrap')
            boot.create_dataset('avg_PLcam', data=boot_avg_pl, dtype='float32',
                                compression='gzip', compression_opts=4)

    print("Averaged H5 written to %s  (%dx%d grid, %d filled bins)" % (
        outpath, map_n, map_n, (nframes > 0).sum()))


# ---------------------------------------------------------------------------
# pixel_map
# ---------------------------------------------------------------------------

def pixel_map(averaged_h5, py, px):
    """
    Extract a 2D response map for one PLcam pixel from an averaged H5.

    Parameters
    ----------
    averaged_h5 : str
    py, px : int
        Pixel indices into the PLcam frame (or ilambda, iport for spectra).

    Returns
    -------
    ndarray float32 (map_n, map_n)  — NaN for empty bins
    """
    with h5py.File(averaged_h5, 'r') as f:
        pmap = f['avg_PLcam'][:, :, py, px][:]
    return pmap.astype('float32')
