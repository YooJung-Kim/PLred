"""
Diagnostic plot helpers for PLred CLI scripts.

All functions return a matplotlib Figure (no plt.show() calls).
Call save_diagnostic(fig, outdir, name) to write to disk.
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def get_plots_dir(configname):
    """Return the plots/ subdirectory from [Pipeline].outdir in the config.

    Falls back to a 'plots/' directory next to *configname* when outdir is not set.
    The path is always relative to the config file's own directory so that running
    from a different CWD still produces consistent output locations.
    """
    try:
        from configobj import ConfigObj
        cfg    = ConfigObj(configname)
        outdir = cfg.get('Pipeline', {}).get('outdir', '').strip()
    except Exception:
        outdir = ''

    config_dir = os.path.dirname(os.path.abspath(configname))
    if outdir:
        base = outdir if os.path.isabs(outdir) else os.path.join(config_dir, outdir)
    else:
        base = config_dir
    return os.path.join(base, 'plots')


def save_diagnostic(fig, outdir, filename):
    """Save *fig* to *outdir/filename* and close it.  Silent on failure."""
    try:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, filename)
        fig.savefig(path, dpi=120, bbox_inches='tight')
        _close(fig)
        print(f'Diagnostic saved: {path}')
    except Exception as e:
        print(f'Warning: could not save diagnostic plot: {e}')


def _close(fig):
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def _figure(*args, **kwargs):
    from matplotlib.figure import Figure
    return Figure(*args, **kwargs)


# ---------------------------------------------------------------------------
# Step 1 — timestamp matching
# ---------------------------------------------------------------------------

def plot_step1(fastcam_h5):
    """
    Two panels from fastcam.h5:
      left  — PSF centroid scatter coloured by peak value
      right — mosaic of 6 evenly-spaced averaged PSF frames
    """
    import h5py

    with h5py.File(fastcam_h5, 'r') as f:
        frames     = f['frames'][:]       # (N, h, w)
        centroids  = f['metadata'][()]    # metadata is a JSON string
        # centroids and peaks are not stored in step-1 H5; use frame max as proxy
        peaks = np.array([float(fr.max()) for fr in frames])
        cx = np.full(len(frames), np.nan)
        cy = np.full(len(frames), np.nan)
        try:
            import json
            meta = json.loads(f['metadata'][()])
            # step-1 H5 has no centroids — skip centroid panel if unavailable
        except Exception:
            pass

    N = len(frames)
    n_show = min(6, N)
    inds = np.linspace(0, N - 1, n_show, dtype=int)

    fig = _figure(figsize=(12, 4))
    gs  = fig.add_gridspec(1, n_show + 1, wspace=0.3)

    # Left col: peak histogram
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(peaks, bins=20, color='steelblue', edgecolor='none', alpha=0.8)
    ax0.set_xlabel('PSF peak (counts)')
    ax0.set_ylabel('Frame count')
    ax0.set_title('PSF peak distribution')
    ax0.grid(alpha=0.3)

    # Right cols: frame mosaic
    vmax = np.nanpercentile(frames, 99)
    for k, idx in enumerate(inds):
        ax = fig.add_subplot(gs[0, k + 1])
        ax.imshow(frames[idx], origin='upper', cmap='viridis',
                  vmin=0, vmax=vmax, aspect='equal')
        ax.set_title(f'frame {idx}', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f'Step 1: timestamp matching  ({N} frames)', y=1.02)
    return fig


# ---------------------------------------------------------------------------
# Step 2 — ingest
# ---------------------------------------------------------------------------

def plot_step2(alldata_h5):
    """
    Two panels from alldata.h5:
      left  — PSF centroid scatter coloured by peak value
      right — PSF peak histogram
    """
    import h5py

    with h5py.File(alldata_h5, 'r') as f:
        centroids = f['psfcam/centroids'][:]   # (N, 2)
        peaks     = f['psfcam/peaks'][:]       # (N,)

    fig = _figure(figsize=(10, 4))
    gs  = fig.add_gridspec(1, 2, wspace=0.35)

    # Centroid scatter
    ax0 = fig.add_subplot(gs[0, 0])
    sc  = ax0.scatter(centroids[:, 0], centroids[:, 1],
                      c=peaks, cmap='viridis', s=10, alpha=0.7)
    fig.colorbar(sc, ax=ax0, label='PSF peak (counts)')
    ax0.set_xlabel('Centroid x (PSFcam pixels)')
    ax0.set_ylabel('Centroid y (PSFcam pixels)')
    ax0.set_title('PSF centroids')
    ax0.set_aspect('equal')
    ax0.grid(alpha=0.3)

    # Peak histogram
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(peaks, bins=25, color='steelblue', edgecolor='none', alpha=0.8)
    ax1.axvline(float(np.median(peaks)), color='tomato', lw=1.5, ls='--',
                label=f'median = {np.median(peaks):.0f}')
    ax1.set_xlabel('PSF peak (counts)')
    ax1.set_ylabel('Frame count')
    ax1.set_title('PSF peak distribution')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    fig.suptitle(f'Step 2: ingest  ({len(peaks)} frames)', y=1.02)
    return fig


# ---------------------------------------------------------------------------
# Step 3 — ROI viewer cache
# ---------------------------------------------------------------------------

def plot_step3(alldata_h5):
    """
    Two panels from alldata.h5:
      left  — PSF centroid scatter with ±2σ ellipse
      right — mean PLcam frame (sum over all time steps)
    """
    import h5py

    with h5py.File(alldata_h5, 'r') as f:
        centroids = f['psfcam/centroids'][:]   # (N, 2)
        peaks     = f['psfcam/peaks'][:]
        plcam_ds  = f['plcam/frames']
        # Mean PLcam frame (load first 50 frames max for speed)
        n_load = min(50, plcam_ds.shape[0])
        mean_plcam = plcam_ds[:n_load].mean(axis=0)   # (ny, nx)
        roi = list(f.attrs.get('plcam_roi', [0, mean_plcam.shape[0],
                                              0, mean_plcam.shape[1]]))

    fig = _figure(figsize=(11, 4))
    gs  = fig.add_gridspec(1, 2, wspace=0.35)

    # Centroid scatter
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(centroids[:, 0], centroids[:, 1],
                c=peaks, cmap='viridis', s=10, alpha=0.7)
    cx = float(np.nanmedian(centroids[:, 0]))
    cy = float(np.nanmedian(centroids[:, 1]))
    sx = float(np.nanstd(centroids[:, 0]))
    sy = float(np.nanstd(centroids[:, 1]))
    ax0.axvline(cx, color='tomato', lw=1, ls='--', alpha=0.8)
    ax0.axhline(cy, color='tomato', lw=1, ls='--', alpha=0.8)
    ax0.set_xlabel('Centroid x')
    ax0.set_ylabel('Centroid y')
    ax0.set_title(f'PSF centroids  median=({cx:.1f},{cy:.1f})  σ=({sx:.2f},{sy:.2f})')
    ax0.grid(alpha=0.3)

    # Mean PLcam frame
    ax1 = fig.add_subplot(gs[0, 1])
    vmax = float(np.nanpercentile(mean_plcam, 99))
    im   = ax1.imshow(mean_plcam, aspect='auto', origin='upper',
                      vmin=0, vmax=vmax, cmap='inferno')
    fig.colorbar(im, ax=ax1, label='Counts')
    ax1.set_xlabel(f'x (local, det x0={int(roi[2])})')
    ax1.set_ylabel('y')
    ax1.set_title('Mean PLcam frame')

    fig.suptitle(f'Step 3: ROI cache  ({len(peaks)} frames)', y=1.02)
    return fig


# ---------------------------------------------------------------------------
# Step 4 — spatial averaging
# ---------------------------------------------------------------------------

def plot_step4(map_h5):
    """
    Two panels from map.h5:
      left  — nframes heatmap with bin counts annotated
      right — MAP_N×MAP_N grid of avg PLcam cross-dispersion profiles
    """
    import h5py

    with h5py.File(map_h5, 'r') as f:
        avg_plcam = f['avg_PLcam'][:]          # (map_n, map_n, ny, nx)
        nframes   = f['metadata/nframes'][:]   # (map_n, map_n)
        x_mas     = f['metadata/x_mas'][:]
        y_mas     = f['metadata/y_mas'][:]

    map_n = avg_plcam.shape[0]

    fig = _figure(figsize=(12, 5))
    gs  = fig.add_gridspec(1, map_n + 1, wspace=0.1)

    # Left: nframes heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    im  = ax0.imshow(nframes, origin='upper', cmap='Blues',
                     vmin=0, aspect='equal')
    fig.colorbar(im, ax=ax0, label='nframes', fraction=0.046, pad=0.04)
    for i in range(map_n):
        for j in range(map_n):
            ax0.text(j, i, str(nframes[i, j]), ha='center', va='center',
                     fontsize=8, color='black' if nframes[i, j] < nframes.max() * 0.6 else 'white')
    ax0.set_xticks(range(map_n)); ax0.set_xticklabels([f'{v:.1f}' for v in x_mas], fontsize=6)
    ax0.set_yticks(range(map_n)); ax0.set_yticklabels([f'{v:.1f}' for v in y_mas], fontsize=6)
    ax0.set_xlabel('x (mas)', fontsize=8); ax0.set_ylabel('y (mas)', fontsize=8)
    ax0.set_title('Frames per bin', fontsize=9)

    # Right: MAP_N×MAP_N grid of avg PLcam images
    fig2 = _figure(figsize=(max(8, map_n * 2.2), max(4, map_n * 2.0) + 0.6))
    vmax = float(np.nanpercentile(avg_plcam[nframes > 0], 99)) if np.any(nframes > 0) else 1.0

    for i in range(map_n):
        for j in range(map_n):
            ax = fig2.add_subplot(map_n, map_n, i * map_n + j + 1)
            if nframes[i, j] > 0:
                ax.imshow(avg_plcam[i, j], aspect='auto', origin='upper',
                          vmin=0, vmax=vmax, cmap='inferno')
            else:
                ax.set_facecolor('#111')
                ax.text(0.5, 0.5, '—', ha='center', va='center',
                        color='gray', transform=ax.transAxes, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'({x_mas[j]:.1f},{y_mas[i]:.1f})', fontsize=6)

    fig2.suptitle('Step 4: avg PLcam per spatial bin  (x,y in mas)', y=1.01, fontsize=9)
    fig2.tight_layout()

    # Combine: embed nframes heatmap as a small inset in the grid figure
    # (return grid figure; caller saves both or just the grid)
    # For simplicity return a combined figure with nframes on the left
    fig3 = _figure(figsize=(max(8, map_n * 2.2) + 3, max(4, map_n * 2.0) + 0.6))
    gs3  = fig3.add_gridspec(map_n, map_n + 1, wspace=0.08, hspace=0.3)

    ax_nfr = fig3.add_subplot(gs3[:, 0])
    im_nfr = ax_nfr.imshow(nframes, origin='upper', cmap='Blues', vmin=0, aspect='equal')
    fig3.colorbar(im_nfr, ax=ax_nfr, label='nframes', fraction=0.046, pad=0.04)
    for i in range(map_n):
        for j in range(map_n):
            ax_nfr.text(j, i, str(nframes[i, j]), ha='center', va='center',
                        fontsize=8, color='black' if nframes[i, j] < nframes.max() * 0.6 else 'white')
    ax_nfr.set_xticks(range(map_n)); ax_nfr.set_xticklabels([f'{v:.1f}' for v in x_mas], fontsize=6)
    ax_nfr.set_yticks(range(map_n)); ax_nfr.set_yticklabels([f'{v:.1f}' for v in y_mas], fontsize=6)
    ax_nfr.set_xlabel('x (mas)', fontsize=7); ax_nfr.set_ylabel('y (mas)', fontsize=7)
    ax_nfr.set_title('Frames/bin', fontsize=8)

    for i in range(map_n):
        for j in range(map_n):
            ax = fig3.add_subplot(gs3[i, j + 1])
            if nframes[i, j] > 0:
                ax.imshow(avg_plcam[i, j], aspect='auto', origin='upper',
                          vmin=0, vmax=vmax, cmap='inferno')
            else:
                ax.set_facecolor('#111')
                ax.text(0.5, 0.5, '—', ha='center', va='center',
                        color='gray', transform=ax.transAxes, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    fig3.suptitle('Step 4: spatial averaging', y=1.01)
    return fig3


# ---------------------------------------------------------------------------
# Step 5a — traces
# ---------------------------------------------------------------------------

def plot_step5_traces(flat_fits, traces_npz, dark_fits=None, xmin=None, xmax=None):
    """
    Two panels:
      left  — flat image (ROI) with traces overlaid
      right — cross-dispersion profile with detected peak positions
    """
    from astropy.io import fits as pyfits

    data = pyfits.getdata(flat_fits).astype(np.float32)
    image = np.mean(data, axis=0) if data.ndim == 3 else data

    if dark_fits is not None:
        dk = pyfits.getdata(dark_fits).astype(np.float32)
        dark = np.mean(dk, axis=0) if dk.ndim == 3 else dk
        image = image - dark

    d    = np.load(traces_npz, allow_pickle=False)
    ylocs  = d['ylocs'].astype(int)
    traces = d['traces']                  # (nfib, nwav)
    _xmin  = int(d['xmin']) if 'xmin' in d else (xmin or 0)
    _xmax  = int(d['xmax']) if 'xmax' in d else (xmax or image.shape[1])

    img_roi = image[:, _xmin:_xmax]
    nfib    = len(ylocs)

    fig  = _figure(figsize=(11, 5))
    gs   = fig.add_gridspec(1, 2, wspace=0.35)

    # Image + traces
    ax0  = fig.add_subplot(gs[0, 0])
    vmax = float(np.nanpercentile(img_roi, 99))
    ax0.imshow(img_roi, aspect='auto', origin='upper',
               vmin=0, vmax=vmax, cmap='inferno')
    x_arr = np.arange(traces.shape[1])
    for fi in range(nfib):
        ax0.plot(x_arr, traces[fi], lw=0.8, alpha=0.8, color=f'C{fi % 10}')
    ax0.set_xlabel('Spectral channel')
    ax0.set_ylabel('y (cross-dispersion)')
    ax0.set_title(f'Flat + {nfib} fiber traces  x=[{_xmin},{_xmax})')

    # Cross-dispersion profile + peaks
    ax1     = fig.add_subplot(gs[0, 1])
    profile = np.nansum(img_roi, axis=1)
    ax1.plot(profile, np.arange(len(profile)), 'k', lw=0.8)
    ax1.plot(profile[ylocs], ylocs, 'ro', ms=5, label='peaks', zorder=3)
    ax1.invert_yaxis()
    ax1.set_xlabel('Summed counts')
    ax1.set_ylabel('y pixel')
    ax1.set_title(f'Cross-dispersion profile  ({nfib} peaks)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    fig.suptitle('Step 5a: fiber trace detection', y=1.02)
    return fig


# ---------------------------------------------------------------------------
# FIRSTPL extraction quality diagnostic  (3-panel: image / model / residual)
# ---------------------------------------------------------------------------

def plot_firstpl_quality(model_file, ref_image, plcam_roi,
                         var_const=200, thresh=0.1, truncate=0,
                         zoom_rows=60):
    """
    Three-panel quality diagnostic for FIRSTPL optimal extraction.

    Parameters
    ----------
    model_file : str
        Path to model.npz.
    ref_image : ndarray (ny_roi, nx_roi)
        A single PLcam frame or averaged frame in the stored ROI layout.
    plcam_roi : tuple (y0, y1, x0, x1)
        Detector-space ROI matching ref_image.
    var_const, thresh : float
        Extraction parameters (must match what was used for the coupling map).
    truncate : int
        Number of edge channels already removed from the extraction output.
    zoom_rows : int
        Number of detector rows to show in the zoom panels (centred on
        the middle fiber trace).

    Panels
    ------
    1. Detector image (zoomed)  — ref_image in the spectral range
    2. Model reconstruction     — A^T @ spec_flat, reshaped to (ny, nwav)
    3. Residual                 — image − model
    """
    from scipy.sparse import csr_matrix
    from PLred.visPLred.spec import flatten_im, extract_spec_optimal

    d = np.load(model_file, allow_pickle=False)
    A_csr = csr_matrix(
        (d['matrix_data'], d['matrix_indices'], d['matrix_indptr']),
        shape=tuple(d['matrix_shape']),
    )
    xmin_model = int(d['xmin'])
    xmax_model = int(d['xmax'])
    nwav_model = xmax_model - xmin_model

    # ny_full may be absent in older model files — infer from matrix shape
    if 'ny_full' in d:
        ny_full = int(d['ny_full'])
    else:
        # A shape is (nfib*nwav, ny_full*nwav) → ny_full = ncols // nwav
        ny_full = A_csr.shape[1] // nwav_model

    roi_y0     = int(plcam_roi[0])
    roi_x0     = int(plcam_roi[2])
    roi_x1     = int(plcam_roi[3])

    eff_xmin = max(xmin_model, roi_x0)
    eff_xmax = min(xmax_model, roi_x1)
    if eff_xmax <= eff_xmin:
        raise ValueError("No overlap between model and ROI.")

    # Trim matrix columns to the overlap (mirrors make_FIRSTPL_extractor logic)
    nwav_orig = xmax_model - xmin_model
    nfib      = A_csr.shape[0] // nwav_orig
    new_nwav  = eff_xmax - eff_xmin

    if eff_xmin == xmin_model and eff_xmax == xmax_model:
        A = A_csr
    else:
        x_rel   = np.arange(eff_xmin - xmin_model, eff_xmax - xmin_model)
        y_arr   = np.arange(ny_full)
        row_inds = (np.arange(nfib)[:, None] * nwav_orig + x_rel[None, :]).ravel()
        col_inds = (y_arr[:, None] * nwav_orig + x_rel[None, :]).ravel()
        A = A_csr[row_inds][:, col_inds]
        nfib = A.shape[0] // new_nwav

    # Slice ref_image to the effective spectral range
    img_xmin = eff_xmin - roi_x0
    img_xmax = eff_xmax - roi_x0
    img_roi  = ref_image[:, img_xmin:img_xmax].astype(np.float32)  # (ny_roi, new_nwav)

    # Embed ROI into full detector height
    ny_img = img_roi.shape[0]
    if ny_img < ny_full:
        full = np.zeros((ny_full, new_nwav), dtype=np.float32)
        full[roi_y0 : roi_y0 + ny_img, :] = img_roi
        im_full = full
    else:
        im_full = img_roi

    imvec = im_full.ravel()
    spec_flat, recon_vec = extract_spec_optimal(A, imvec, var_const=var_const, thresh=thresh)

    recon    = recon_vec.reshape(ny_full, new_nwav)
    residual = im_full - recon

    # Zoom: centre on median fiber y-position
    if 'trace_vals' in d:
        tv = d['trace_vals']       # (nfib, nwav_model)
        xrel_mid = (eff_xmin - xmin_model + eff_xmax - xmin_model) // 2
        y_mid = int(np.nanmedian(tv[:, min(xrel_mid, tv.shape[1]-1)]))
    else:
        y_mid = ny_full // 2
    half = zoom_rows // 2
    zy0  = max(0, y_mid - half)
    zy1  = min(ny_full, y_mid + half)

    # Crop to roi for display
    disp_y0 = max(roi_y0, zy0) - roi_y0
    disp_y1 = min(roi_y0 + ny_img, zy1) - roi_y0

    det_crop  = img_roi[disp_y0:disp_y1]
    mod_crop  = recon[zy0:zy1]
    res_crop  = residual[zy0:zy1]

    # Build figure
    fig = _figure(figsize=(13, 4))
    gs  = fig.add_gridspec(1, 3, wspace=0.3)

    vmax  = float(np.nanpercentile(det_crop[det_crop > 0], 99)) if np.any(det_crop > 0) else 1.0
    vmin  = 0.0
    rlim  = float(np.nanpercentile(np.abs(res_crop), 97))
    x_ticks = [0, new_nwav // 2, new_nwav - 1]
    x_labels = [str(eff_xmin), str((eff_xmin + eff_xmax)//2), str(eff_xmax - 1)]

    for col, (img, title, kw) in enumerate([
        (det_crop, 'Detector (zoomed)', dict(vmin=vmin, vmax=vmax, cmap='inferno')),
        (mod_crop, 'Model reconstruction', dict(vmin=vmin, vmax=vmax, cmap='inferno')),
        (res_crop, 'Residual (image − model)', dict(vmin=-rlim, vmax=rlim, cmap='RdBu_r')),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(img, aspect='auto', origin='upper', **kw)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Detector x col', fontsize=8)
        ax.set_ylabel(f'y (det rows {zy0}–{zy1})', fontsize=8)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=7)

    trunc_note = f'  (edge truncation: {truncate})' if truncate else ''
    fig.suptitle(
        f'FIRSTPL extraction quality  —  {nfib} fibers, nwav={new_nwav}{trunc_note}',
        y=1.02, fontsize=9,
    )
    return fig


# ---------------------------------------------------------------------------
# Step 5b — extraction
# ---------------------------------------------------------------------------

def plot_step5_extraction(couplingmap_fits, map_h5=None):
    """
    Two panels from couplingmap.fits:
      left  — nframes heatmap
      right — mean coupling map for the fiber with the highest total response
    """
    from astropy.io import fits as pyfits
    import h5py

    with pyfits.open(couplingmap_fits) as hdul:
        data   = hdul[0].data    # (map_n, map_n, nfib, nwav)
        nfr    = hdul[1].data    # (map_n, map_n)

    map_n, _, nfib, nwav = data.shape
    mean_map = np.nanmean(data, axis=3)   # (map_n, map_n, nfib)

    x_mas = np.arange(map_n, dtype=float)
    y_mas = np.arange(map_n, dtype=float)
    if map_h5 is not None:
        try:
            with h5py.File(map_h5, 'r') as f:
                x_mas = f['metadata/x_mas'][:]
                y_mas = f['metadata/y_mas'][:]
        except Exception:
            pass

    extent = [x_mas[0], x_mas[-1], y_mas[-1], y_mas[0]]

    # Best fiber = highest mean response summed over valid bins
    with np.errstate(all='ignore'):
        fiber_totals = np.array([
            np.nansum(mean_map[:, :, fi][nfr > 0]) for fi in range(nfib)
        ])
    best_fi = int(np.argmax(fiber_totals))

    ncols = min(8, nfib)
    nrows = int(np.ceil(nfib / ncols))
    fig   = _figure(figsize=(ncols * 1.8, nrows * 1.8 + 0.8))

    axes = []
    for idx in range(nfib):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        axes.append(ax)

    for fi in range(nfib):
        ax  = axes[fi]
        m   = mean_map[:, :, fi]
        pos = m[m > 0]
        vmax = float(np.nanpercentile(pos, 98)) if len(pos) else 1.0
        ax.imshow(m, extent=extent, origin='upper',
                  cmap='inferno', vmin=0, vmax=max(vmax, 1e-9), aspect='equal')
        ax.set_title(f'fib {fi}' + (' ★' if fi == best_fi else ''), fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])

    valid = int((nfr >= 1).sum())
    fig.suptitle(
        f'Step 5b: coupling map  ({nfib} fibers, {nwav} channels, '
        f'{valid}/{map_n*map_n} valid bins)',
        y=1.01, fontsize=9,
    )
    fig.tight_layout()
    return fig
