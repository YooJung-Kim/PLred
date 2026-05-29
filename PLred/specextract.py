"""
Spectral extraction: averaged.h5 → coupling_map.fits  OR  raw FITS cube → _spec.fits.

Two extraction modes
--------------------
**Mode A — coupling map (averaged H5 → FITS)**

    extract_to_coupling_map(averaged_h5, extractor, output_fits)

Reads the output of ``average_to_h5()``, extracts a mean spectrum per spatial
grid bin, and writes the FITS format consumed by ``CouplingMapModel(mapdata=...)``.

**Mode B — time-series spectra (raw FITS cube → _spec.fits)**

    extract_from_fits(input_fits, extractor, dark_fits)

Reads raw science FITS cube(s) with shape ``(Nframes, h, w)``, subtracts a master
dark, applies the extractor frame-by-frame, and writes ``*_spec.fits`` cubes with
shape ``(Nframes, nfib, nwav)``.  The input FITS header is duplicated and augmented
with extraction metadata.

Ready-made extractor factories
-------------------------------
    import PLred.specextract as specextract

    # Fixed-aperture box extraction (any instrument)
    ylocs     = specextract.find_peaks(ref_image, nfib=38, thres=0.1)
    extractor = specextract.make_simple_extractor(ylocs, width=3)

    # Trace-following box extraction (visPLred / FIRST-PL)
    traces    = specextract.find_traces(ref_image, nfib=38, ini_ys=ylocs)
    extractor = specextract.make_trace_extractor(traces, boxsize=3)

    # Optimal extraction with nonlinearity correction (FIRST-PL)
    extractor = specextract.make_FIRSTPL_extractor(
        'specmodel/specmodel.npz',        # built by SpectrumModel.save_spectra_model()
        nonlin_modelfile='nonlin_model.fits',
    )

    # Mode A
    extract_to_coupling_map('averaged.h5', extractor, 'coupling_map.fits')

    # Mode B
    extract_from_fits(['sci1.fits', 'sci2.fits'], extractor, dark_fits='dark.fits')

Output FITS structure — Mode A (HDU indices match CouplingMapModel(mapdata=...))
----------------------------------------------------------------------------------
    [0] primary  : (map_n, map_n, nfib, nwav)  mean extracted spectra
    [1] nframes  : (map_n, map_n)               frames per bin
    [2] reserved : (map_n, map_n, nfib, nwav)   zeros (placeholder)
    [3] var      : (map_n, map_n, nfib, nwav)   variance of the mean per bin
    [4] normvar  : (map_n, map_n, nfib, nwav)   normalized variance
    [5] traces   : (nfib, nx)                   fiber traces (if available)
    header: XMIN, XMAX, MAP_N, SPEX_* extraction metadata

Output FITS structure — Mode B
--------------------------------
    [0] primary  : (Nframes, nfib, nwav)  extracted spectra
    [1] traces   : (nfib, nx)             fiber traces (if available)
    header: copied from input + SPEX_* extraction metadata

Variance estimation (Mode A, three-tier priority)
--------------------------------------------------
1. Bootstrap  — if /bootstrap/avg_PLcam exists in the H5 and use_bootstrap=True.
2. variance_extractor — user-supplied callable image → var(nfib, nwav).
3. Poisson fallback — var = spectra / nframes.
"""

import os
import datetime
from types import SimpleNamespace
import numpy as np
import h5py
from astropy.io import fits
from tqdm import tqdm


# ------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------

def _attach_info(func, info):
    """Attach a metadata dict to an extractor callable for header propagation."""
    func._info = info
    return func


def _write_extraction_header(header, extractor, extra=None):
    """Write extractor metadata into a FITS header using HIERARCH keywords."""
    info = {}
    if hasattr(extractor, '_info'):
        info.update({k: v for k, v in extractor._info.items() if k != 'traces'})
    if extra:
        info.update(extra)

    header['HIERARCH SPEX DATE'] = datetime.datetime.utcnow().isoformat()[:23]
    header['HIERARCH SPEX TYPE'] = str(info.pop('extractor', 'unknown'))

    for key, val in info.items():
        hkey = f'HIERARCH SPEX {key.upper()[:12]}'
        if isinstance(val, (list, np.ndarray)):
            s = ','.join(str(v) for v in np.asarray(val).ravel())
            if len(s) > 65:
                s = s[:62] + '...'
            try:
                header[hkey] = s
            except Exception:
                pass
        elif isinstance(val, (bool, int, float)):
            try:
                header[hkey] = val
            except Exception:
                pass
        elif isinstance(val, str):
            # For file paths use basename to stay within FITS card limits
            if len(val) > 65 and os.sep in val:
                val = os.path.basename(val)
            val = val[:65] if len(val) > 65 else val
            try:
                header[hkey] = val
            except Exception:
                pass


def _load_dark(dark_fits):
    """Load a master dark frame from a FITS file.

    If the file is a cube (Nframes, h, w), the median across frames is returned.
    """
    with fits.open(dark_fits) as hdl:
        dark = hdl[0].data.astype(np.float32)
    if dark.ndim == 3:
        dark = np.median(dark, axis=0).astype(np.float32)
    return dark


def load_spectrum_model(model_file):
    """
    Load a FIRST-PL spectrum model from a ``.npz`` file saved by
    ``SpectrumModel.save_spectra_model()``.

    Parameters
    ----------
    model_file : str
        Path to the ``.npz`` model file.

    Returns
    -------
    model : SimpleNamespace with attributes:
        ``A``          — scipy.sparse.csr_matrix, extraction matrix
        ``xmin``       — int, first detector column included in the matrix
        ``xmax``       — int, last detector column (exclusive)
        ``XMIN``       — int, global left edge used when building trace_vals
        ``trace_vals`` — ndarray (nfib, XMAX-XMIN) or None
        ``wav_map``    — ndarray (nfib, xmax-xmin) wavelength map, or None

    Notes
    -----
    If ``wav_map`` is present, the extractor factories will apply
    per-fiber wavelength interpolation to a common reference grid
    (using ``visPLred.spec.interpolate_spectrum``).
    """
    from scipy.sparse import csr_matrix
    d = np.load(model_file, allow_pickle=False)
    A = csr_matrix(
        (d['matrix_data'], d['matrix_indices'], d['matrix_indptr']),
        shape=tuple(d['matrix_shape']),
    )
    nwav = int(d['xmax']) - int(d['xmin'])
    # ny_full: full detector height used when building A.
    # Derive from matrix shape if not explicitly stored (backward compat).
    if 'ny_full' in d:
        ny_full = int(d['ny_full'])
    else:
        ny_full = A.shape[1] // nwav if nwav > 0 else None

    return SimpleNamespace(
        A          = A,
        xmin       = int(d['xmin']),
        xmax       = int(d['xmax']),
        XMIN       = int(d['XMIN']) if 'XMIN' in d else 0,
        ny_full    = ny_full,
        trace_vals = d['trace_vals'] if 'trace_vals' in d else None,
        wav_map    = d['wav_map']    if 'wav_map'    in d else None,
    )


# ------------------------------------------------------------------
# Mode A — averaged H5 → coupling-map FITS
# ------------------------------------------------------------------

def extract_to_coupling_map(
    averaged_h5,
    extractor,
    output_fits,
    variance_extractor=None,
    use_bootstrap=True,
    extractor_info=None,
    verbose=True,
):
    """
    Extract spectra from an averaged H5 file and write a coupling-map FITS.

    Parameters
    ----------
    averaged_h5 : str
        Path to the averaged H5 produced by ``average_to_h5()``.
    extractor : callable
        ``f(image: ndarray[ny, nx]) -> ndarray[nfib, nwav]``
        Applied to the mean PLcam image of each non-empty grid bin.
        Configure hyperparameters via a factory function or closure.
    output_fits : str
        Output path for the coupling-map FITS file (written with overwrite=True).
    variance_extractor : callable, optional
        ``f(image: ndarray[ny, nx]) -> ndarray[nfib, nwav]``
        Returns the variance of the *mean* spectrum for one bin.
        Falls back to bootstrap (if available) or Poisson if None.
    use_bootstrap : bool
        If True and ``/bootstrap/avg_PLcam`` exists in the H5, use bootstrap
        variance estimate.
    extractor_info : dict, optional
        Extra key/value pairs to store in the output FITS header alongside
        the metadata auto-detected from ``extractor._info``.
    verbose : bool

    Returns
    -------
    str
        Path to the written FITS file (same as ``output_fits``).
    """
    # ------------------------------------------------------------------
    # 1. Load averaged H5
    # ------------------------------------------------------------------
    with h5py.File(averaged_h5, 'r') as f:
        avg_plcam = f['avg_PLcam'][:]                        # (map_n, map_n, ny, nx)
        nframes   = f['metadata/nframes'][:]                 # (map_n, map_n)
        x_mas     = f['metadata/x_mas'][:]                   # (map_n,)
        y_mas     = f['metadata/y_mas'][:]                   # (map_n,)

        # Detector-space pixel bounds of the stored PLcam frames.
        # Present only when average_to_h5() was called with plcam_roi=.
        if 'metadata/plcam_roi' in f:
            plcam_roi = f['metadata/plcam_roi'][:]           # [y0, y1, x0, x1]
        else:
            plcam_roi = None

        # Dark-subtraction flag propagated from ingest_to_h5
        h5_dark_subtracted = f.attrs.get('plcam_dark_subtracted', None)
        h5_dark_source     = str(f.attrs.get('dark_source', 'unknown'))

        has_bootstrap = use_bootstrap and ('bootstrap/avg_PLcam' in f)
        if has_bootstrap:
            bs_plcam = f['bootstrap/avg_PLcam'][:]           # (n_bs, map_n, map_n, ny, nx)
            n_bs = bs_plcam.shape[0]
            if verbose:
                print(f"Bootstrap resamples found: n_bs={n_bs}")
        else:
            bs_plcam = None
            n_bs = 0

    map_n = avg_plcam.shape[0]
    assert avg_plcam.shape[1] == map_n, "avg_PLcam must be square in spatial dims"

    # ------------------------------------------------------------------
    # 1b. Resolve image-space column slice from model xmin/xmax + H5 ROI
    # ------------------------------------------------------------------
    # Spectral models store coordinates in detector space (e.g. xmin=1100).
    # If the H5 images are a hardware ROI crop, column 0 in the image
    # corresponds to detector column x0_roi, so we must convert:
    #   image_col = det_col - x0_roi
    _ext_info  = getattr(extractor, '_info', {})
    model_xmin = _ext_info.get('xmin',    None)
    model_xmax = _ext_info.get('xmax',    None)
    ny_full    = _ext_info.get('ny_full', None)   # full detector height (FIRSTPL)

    roi_x0 = int(plcam_roi[2]) if plcam_roi is not None else 0
    roi_y0 = int(plcam_roi[0]) if plcam_roi is not None else 0

    if model_xmin is not None and model_xmax is not None:
        img_xmin = model_xmin - roi_x0
        img_xmax = model_xmax - roi_x0
        img_nx   = avg_plcam.shape[3]
        if img_xmin < 0 or img_xmax > img_nx:
            raise ValueError(
                f"Model xmin/xmax ({model_xmin}, {model_xmax}) maps to image columns "
                f"[{img_xmin}, {img_xmax}] but H5 images have {img_nx} columns "
                f"(roi_x0={roi_x0}).  Check plcam_roi or model bounds."
            )
    else:
        img_xmin = None
        img_xmax = None

    def _crop_image(image):
        """Slice to model spectral range and y-embed for full-detector extractors."""
        # 1. x-slice to model column range
        out = image[:, img_xmin:img_xmax] if img_xmin is not None else image

        # 2. y-embed: extractors whose matrix was built on full-detector height
        #    (FIRSTPL optimal) need a (ny_full, nwav) image even when H5 stores
        #    a y-cropped ROI.  Embed the ROI at the correct detector y-offset.
        if ny_full is not None and out.shape[0] < ny_full:
            full = np.zeros((ny_full, out.shape[1]), dtype=np.float32)
            ny_roi = out.shape[0]
            full[roi_y0:roi_y0 + ny_roi, :] = out
            return full

        return out

    # Warn if the extractor will double-subtract or skip dark
    extractor_has_dark = _ext_info.get('has_dark', False)
    if h5_dark_subtracted is not None:
        if h5_dark_subtracted and extractor_has_dark:
            print(
                f"WARNING: Dark was already subtracted during ingest "
                f"(source: {h5_dark_source}) AND the extractor also applies a dark. "
                f"This will double-subtract the dark. Pass dark=None to the extractor factory."
            )
        elif not h5_dark_subtracted and not extractor_has_dark:
            print(
                f"WARNING: Dark was NOT subtracted during ingest and the extractor "
                f"has no dark. Consider passing a dark frame to the extractor factory "
                f"or re-running ingest with plcam_dark=."
            )

    if verbose:
        ds_str = f"dark_subtracted={h5_dark_subtracted} (source: {h5_dark_source})"
        print(f"Loaded {averaged_h5}: map_n={map_n}, avg_PLcam shape={avg_plcam.shape}")
        print(f"  {ds_str}")
        if plcam_roi is not None:
            print(f"  plcam_roi (det): y=[{plcam_roi[0]},{plcam_roi[1]}]  "
                  f"x=[{plcam_roi[2]},{plcam_roi[3]}]")
        if img_xmin is not None:
            print(f"  model xmin/xmax: [{model_xmin},{model_xmax}]  "
                  f"→ image cols [{img_xmin},{img_xmax}]")
        if ny_full is not None:
            print(f"  ny_full={ny_full}  roi_y0={roi_y0}  "
                  f"(y-embed={'yes' if avg_plcam.shape[2] < ny_full else 'no'})")

    # ------------------------------------------------------------------
    # 2. Extract spectra for each grid bin
    # ------------------------------------------------------------------
    # Probe on the cropped image so nfib/nwav reflect actual extractor output
    nfib, nwav = _probe_extractor_shape(
        lambda im: extractor(_crop_image(im)), avg_plcam, map_n
    )
    if verbose:
        print(f"Extractor output shape: nfib={nfib}, nwav={nwav}")

    spectra = np.full((map_n, map_n, nfib, nwav), np.nan, dtype=np.float32)
    datavar = np.full_like(spectra, np.nan)

    bs_spectra = None
    if has_bootstrap:
        bs_spectra = np.full((n_bs, map_n, map_n, nfib, nwav), np.nan, dtype=np.float32)

    if verbose:
        print("Extracting spectra per grid bin...")

    grid_iter = [(ix, iy) for ix in range(map_n) for iy in range(map_n)]

    for ix, iy in tqdm(grid_iter, disable=not verbose, desc="Extracting"):
        image = avg_plcam[ix, iy]
        if not np.any(np.isfinite(image)):
            continue
        spectra[ix, iy] = extractor(_crop_image(image)).astype(np.float32)
        if has_bootstrap:
            for k in range(n_bs):
                bs_image = bs_plcam[k, ix, iy]
                if np.any(np.isfinite(bs_image)):
                    bs_spectra[k, ix, iy] = extractor(_crop_image(bs_image)).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. Estimate variance
    # ------------------------------------------------------------------
    if has_bootstrap and bs_spectra is not None:
        datavar = np.nanvar(bs_spectra, axis=0, ddof=1).astype(np.float32)
        if verbose:
            print("Variance estimated from bootstrap resamples.")
    elif variance_extractor is not None:
        if verbose:
            print("Variance estimated from variance_extractor...")
        for ix, iy in tqdm(grid_iter, disable=not verbose, desc="Var extract"):
            image = avg_plcam[ix, iy]
            if not np.any(np.isfinite(image)):
                continue
            datavar[ix, iy] = variance_extractor(_crop_image(image)).astype(np.float32)
    else:
        if verbose:
            print("Variance estimated via Poisson approximation (spectra / nframes).")
        n = nframes[:, :, None, None].astype(np.float32)
        n[n == 0] = np.nan
        datavar = (spectra / n).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Normalized variance
    # ------------------------------------------------------------------
    spatial_sum = np.nansum(spectra, axis=(0, 1))
    spatial_sum[spatial_sum == 0] = np.nan
    datanormvar = (datavar / spatial_sum[None, None] ** 2).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Write FITS
    # ------------------------------------------------------------------
    hdu0 = fits.PrimaryHDU(spectra)
    hdu0.header['EXTNAME'] = 'data'
    hdu0.header['XMIN']    = float(x_mas.min())
    hdu0.header['XMAX']    = float(x_mas.max())
    hdu0.header['MAP_N']   = map_n
    hdu0.header['NFIB']    = nfib
    hdu0.header['NWAV']    = nwav
    hdu0.header['SRC_H5']  = os.path.basename(averaged_h5)
    _write_extraction_header(hdu0.header, extractor, extra=extractor_info)

    hdu1 = fits.ImageHDU(nframes.astype(np.int32))
    hdu1.header['EXTNAME'] = 'nframes'

    hdu2 = fits.ImageHDU(np.zeros((map_n, map_n, nfib, nwav), dtype=np.float32))
    hdu2.header['EXTNAME'] = 'reserved'

    hdu3 = fits.ImageHDU(datavar)
    hdu3.header['EXTNAME'] = 'var'

    hdu4 = fits.ImageHDU(datanormvar)
    hdu4.header['EXTNAME'] = 'normvar'

    hdulist = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4])

    # Optional traces HDU
    traces = None
    if hasattr(extractor, '_info') and 'traces' in extractor._info:
        traces = extractor._info['traces']
    if traces is not None:
        hdu_tr = fits.ImageHDU(np.asarray(traces, dtype=np.float64))
        hdu_tr.header['EXTNAME'] = 'TRACES'
        hdu_tr.header['COMMENT'] = 'Fiber trace y-centers: (nfib, nx)'
        hdulist.append(hdu_tr)

    hdulist.writeto(output_fits, overwrite=True)

    if verbose:
        print(f"Coupling map FITS written to: {output_fits}")
        print(f"  spectra shape : {spectra.shape}")
        print(f"  valid bins    : {np.sum(nframes > 0)} / {map_n * map_n}")
        if traces is not None:
            print(f"  traces stored : shape {np.asarray(traces).shape}")

    return output_fits


# ------------------------------------------------------------------
# Mode B — raw FITS cube(s) → _spec.fits
# ------------------------------------------------------------------

def extract_from_fits(
    input_fits,
    extractor,
    dark_fits,
    xmin=None,
    xmax=None,
    output_dir=None,
    output_suffix='_spec',
    extractor_info=None,
    traces=None,
    verbose=True,
):
    """
    Extract spectra from raw FITS cube(s) and write ``*_spec.fits`` files.

    Each input file is expected to contain a data cube of shape
    ``(Nframes, h, w)``.  A master dark is subtracted from every frame before
    the extractor is applied.  Output files have shape ``(Nframes, nfib, nwav)``
    and the input FITS header is duplicated and augmented with extraction
    metadata.

    Parameters
    ----------
    input_fits : str or list of str
        Path(s) to raw science FITS file(s), each with shape ``(Nframes, h, w)``.
        A single 2-D file ``(h, w)`` is treated as a one-frame cube.
    extractor : callable
        ``f(image: ndarray[h, w_roi]) -> ndarray[nfib, nwav]``
        Applied to each dark-subtracted (and optionally pre-sliced) frame.
        Use one of the factory functions to get automatic metadata propagation.
    dark_fits : str or ndarray
        Path to a dark FITS file (2-D or cube) **or** a pre-loaded 2-D dark
        array.  If a cube is given, the median frame is used as the master dark.
        Dark subtraction is mandatory to avoid negative-count artifacts.
    xmin : int or None
        First detector column to pass to the extractor.  If None, auto-detected
        from ``extractor._info['xmin']`` (set by ``make_FIRSTPL_extractor`` and
        ``make_trace_extractor`` when loaded from a model file).  If neither
        provides a value, the full frame width is used.
    xmax : int or None
        Last detector column (exclusive).  Same auto-detection as ``xmin``.
    output_dir : str or None
        Directory for output files.  Defaults to the same directory as each
        input file.
    output_suffix : str
        Suffix appended before ``.fits`` in output filenames (default ``'_spec'``).
        Example: ``'science_001.fits'`` → ``'science_001_spec.fits'``.
    extractor_info : dict or None
        Additional key/value pairs to store in the output FITS header.
    traces : ndarray (nfib, nx) or None
        Fiber traces to save as a ``TRACES`` extension HDU.  Auto-detected from
        ``extractor._info['traces']`` if present.
    verbose : bool

    Returns
    -------
    list of str
        Paths to the written ``*_spec.fits`` files, in the same order as
        ``input_fits``.

    Notes
    -----
    The output HDU list::

        [0]  PrimaryHDU  shape (Nframes, nfib, nwav)  — extracted spectra
        [1]  ImageHDU    shape (nfib, nx)              — traces  (if available)

    The header of HDU[0] is a copy of the input primary header with
    ``NAXIS1/2/3`` updated and ``HIERARCH SPEX *`` keywords added:

        SPEX TYPE  — extractor name (e.g. ``simple_box``)
        SPEX DATE  — UTC timestamp of extraction
        SPEX DARK  — dark file path or ``'array'``
        SPEX *     — any parameter stored by the factory (width, boxsize, …)

    Example
    -------
    >>> import PLred.specextract as specextract
    >>> ylocs    = specextract.find_peaks(ref_image, nfib=38, thres=0.1)
    >>> extractor = specextract.make_simple_extractor(ylocs, width=3)
    >>> out = specextract.extract_from_fits(
    ...     ['sci_001.fits', 'sci_002.fits'],
    ...     extractor,
    ...     dark_fits='master_dark.fits',
    ... )
    >>> print(out)
    ['sci_001_spec.fits', 'sci_002_spec.fits']
    """
    if isinstance(input_fits, str):
        input_fits = [input_fits]

    # ------------------------------------------------------------------
    # Load master dark
    # ------------------------------------------------------------------
    if isinstance(dark_fits, np.ndarray):
        master_dark = dark_fits.astype(np.float32)
        if master_dark.ndim == 3:
            master_dark = np.median(master_dark, axis=0).astype(np.float32)
        dark_label = 'array'
    else:
        if verbose:
            print(f"Loading dark from: {dark_fits}")
        master_dark = _load_dark(dark_fits)
        dark_label = str(dark_fits)

    if verbose:
        print(f"Master dark: shape={master_dark.shape}, "
              f"median={float(np.median(master_dark)):.1f}")

    # Auto-detect traces and xmin/xmax from extractor._info
    _info = getattr(extractor, '_info', {})
    if traces is None and 'traces' in _info:
        traces = _info['traces']
    if xmin is None:
        xmin = _info.get('xmin', None)
    if xmax is None:
        xmax = _info.get('xmax', None)

    output_paths = []

    for fpath in input_fits:
        if verbose:
            print(f"\nProcessing: {fpath}")

        with fits.open(fpath) as hdl:
            in_hdr = hdl[0].header.copy()
            cube   = hdl[0].data.astype(np.float32)

        if cube.ndim == 2:
            cube = cube[None]  # single frame → (1, h, w)

        Nframes, h, w = cube.shape

        if master_dark.shape != (h, w):
            raise ValueError(
                f"Dark shape {master_dark.shape} does not match frame shape "
                f"({h}, {w}) in {fpath}"
            )

        # Determine column slice for the extractor
        _xmin = xmin if xmin is not None else 0
        _xmax = xmax if xmax is not None else w

        if verbose:
            print(f"  cube shape  : {cube.shape}")
            if xmin is not None or xmax is not None:
                print(f"  column slice: [{_xmin}:{_xmax}]  ({_xmax - _xmin} px)")

        # For FIRSTPL extractor: y-embed ROI frame back to full detector height
        _ny_full = _info.get('ny_full', None)
        _y0_roi  = ymin if (ymin := _info.get('ymin', None)) is not None else 0

        def _prep(frame):
            """Dark-subtract, x-slice, and y-embed if needed."""
            out = (frame - master_dark)[:, _xmin:_xmax]
            if _ny_full is not None and out.shape[0] < _ny_full:
                full = np.zeros((_ny_full, out.shape[1]), dtype=np.float32)
                ny_frame = out.shape[0]
                full[_y0_roi:_y0_roi + ny_frame, :] = out
                return full
            return out

        # Probe output shape on first frame
        test_spec = extractor(_prep(cube[0]))
        if test_spec.ndim != 2:
            raise ValueError(
                f"extractor must return a 2-D array (nfib, nwav), "
                f"got shape {test_spec.shape}"
            )
        nfib, nwav = test_spec.shape
        if verbose:
            print(f"  output per frame: ({nfib}, {nwav})")
            print(f"  extracting {Nframes} frames ...")

        out_cube = np.empty((Nframes, nfib, nwav), dtype=np.float32)
        out_cube[0] = test_spec.astype(np.float32)

        for i in tqdm(range(1, Nframes), disable=not verbose, desc="  frames"):
            out_cube[i] = extractor(_prep(cube[i])).astype(np.float32)

        # ------------------------------------------------------------------
        # Build output header (copy input, update axes, add extraction meta)
        # ------------------------------------------------------------------
        out_hdr = in_hdr.copy()
        out_hdr['NAXIS']  = 3
        out_hdr['NAXIS1'] = nwav
        out_hdr['NAXIS2'] = nfib
        out_hdr['NAXIS3'] = Nframes
        dark_str = dark_label if len(dark_label) <= 65 else dark_label[-65:]
        out_hdr['HIERARCH SPEX DARK'] = dark_str
        out_hdr['HIERARCH SPEX PXMIN'] = _xmin
        out_hdr['HIERARCH SPEX PXMAX'] = _xmax
        _write_extraction_header(out_hdr, extractor, extra=extractor_info)

        # ------------------------------------------------------------------
        # Build output path:  basename_spec.fits
        # ------------------------------------------------------------------
        base = os.path.splitext(os.path.basename(fpath))[0]
        out_name = base + output_suffix + '.fits'
        if output_dir is not None:
            out_path = os.path.join(output_dir, out_name)
        else:
            out_path = os.path.join(os.path.dirname(os.path.abspath(fpath)), out_name)

        # ------------------------------------------------------------------
        # Write output FITS
        # ------------------------------------------------------------------
        hdu0 = fits.PrimaryHDU(out_cube, header=out_hdr)
        hdulist = fits.HDUList([hdu0])

        if traces is not None:
            hdu_tr = fits.ImageHDU(np.asarray(traces, dtype=np.float64))
            hdu_tr.header['EXTNAME'] = 'TRACES'
            hdu_tr.header['COMMENT'] = 'Fiber trace y-centers: shape (nfib, nx)'
            hdulist.append(hdu_tr)

        hdulist.writeto(out_path, overwrite=True)
        output_paths.append(out_path)

        if verbose:
            print(f"  written: {out_path}")
            print(f"  shape: {out_cube.shape}  dtype={out_cube.dtype}")
            if traces is not None:
                print(f"  traces stored: shape {np.asarray(traces).shape}")

    return output_paths


# ------------------------------------------------------------------
# Spectral traces and peak finding
# ------------------------------------------------------------------


def locate_spectra(im, num_spec=3, width=6, plot=True, exclude=[0]):

    im_column_stack = np.mean(im, axis=1)
    for ex in exclude: im_column_stack[ex] = 0
    ylocs = np.zeros(num_spec, dtype=int)

    for i in range(num_spec):
        _yloc = np.argmax(im_column_stack)
        ylocs[i] = int(_yloc)
        im_column_stack[_yloc - width: _yloc + width] = 0

    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(im)
        for i in range(num_spec):
            plt.axhspan(ylocs[i] - width, ylocs[i] + width, alpha=0.2, color='white')
        plt.show()

    ylocs = np.sort(ylocs)
    return ylocs


def find_peaks(image, nfib, thres=0.05, min_dist=6, ref_col=None, plot=False):
    """
    Find fiber peak positions in a detector image using peakutils.

    An alternative to ``locate_spectra()`` — useful when fibers are closely
    spaced or the iterative max-masking approach misses peaks.  Returns a 1-D
    array of y-pixel centers suitable for ``find_traces(ini_ys=...)``,
    ``make_simple_extractor(ylocs=...)``, or ``make_trace_extractor``.

    Parameters
    ----------
    image : ndarray (ny, nx) or (ny,)
        Reference detector image (dark-subtracted) or a 1-D cross-dispersion
        profile.  If 2-D, the profile at ``ref_col`` is used.
    nfib : int
        Expected number of fibers.  A ``ValueError`` is raised if a different
        number is found — use it as a signal to tune ``thres`` or ``min_dist``.
    thres : float
        Normalised detection threshold for ``peakutils.indexes`` (0–1).
        Lower → detect fainter peaks; raise if spurious peaks appear.
    min_dist : int
        Minimum pixel separation between peaks (default 6).
    ref_col : int or None
        Column to use when ``image`` is 2-D.  Defaults to the brightest column
        (edge columns excluded).
    plot : bool
        If True, plot the cross-dispersion profile with a line at each peak.

    Returns
    -------
    ylocs : ndarray (nfib,) int
        Y-pixel centers of detected peaks, sorted top-to-bottom.

    Raises
    ------
    ValueError
        If the number of detected peaks differs from ``nfib``.

    Example
    -------
    >>> import PLred.specextract as specextract
    >>> ylocs = specextract.find_peaks(avg_plcam[ix, iy], nfib=38, thres=0.1)
    >>> extractor = specextract.make_simple_extractor(ylocs, width=3)
    >>>
    >>> # or feed into find_traces for curved-trace extraction
    >>> traces = specextract.find_traces(avg_plcam[ix, iy], nfib=38, ini_ys=ylocs)
    """
    import peakutils

    image = np.asarray(image)
    if image.ndim == 2:
        ny, nx = image.shape
        if ref_col is None:
            margin = max(1, nx // 10)
            col_flux = np.nansum(image, axis=0)
            col_flux[:margin]  = 0
            col_flux[-margin:] = 0
            ref_col = int(np.argmax(col_flux))
        profile = image[:, ref_col].copy()
    else:
        profile = image.copy()

    profile = np.nan_to_num(profile, nan=0.0)
    profile = np.clip(profile, 0, None)

    found = peakutils.indexes(profile, thres=thres, min_dist=min_dist)

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, max(4, len(profile) / 40)))
        ax.plot(profile, np.arange(len(profile)), color='steelblue', lw=1)
        for y in found:
            ax.axhline(y, color='tomato', lw=0.8, alpha=0.8)
        ax.invert_yaxis()
        ax.set_xlabel('Counts')
        ax.set_ylabel('y pixel')
        title = f'find_peaks: {len(found)} peaks found'
        if image.ndim == 2:
            title += f'  (ref_col={ref_col})'
        ax.set_title(title)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    if len(found) != nfib:
        raise ValueError(
            f"find_peaks: found {len(found)} peaks (expected {nfib}). "
            f"Adjust thres (currently {thres}) or min_dist ({min_dist})."
        )
    return np.sort(found).astype(int)


def find_traces(image, nfib, trace_width=4, poly_deg=5, ref_col=None,
                ini_ys=None, max_jump=2, min_dist=6, plot=False):
    """
    Find fiber trace positions across a 2D detector image.

    Starting from initial peak positions at a reference column, traces each
    fiber left and right using subpixel 3-point peak fitting, then fits a
    polynomial to each raw trace.

    No neon calibration required — any bright dark-subtracted frame works.

    Parameters
    ----------
    image : ndarray (ny, nx)
        Reference detector image, dark-subtracted.
    nfib : int
        Number of fiber traces to find.
    trace_width : int
        Half-width of the cross-dispersion search window (default 4).
    poly_deg : int
        Polynomial degree for smoothing each raw trace (default 5).
        Automatically capped at ``max(1, nx // 8)`` to prevent overfitting
        on short spectral ranges.
    ref_col : int or None
        Column for initial peak finding.  Defaults to brightest column
        (edge columns excluded).
    ini_ys : array-like (nfib,) or None
        Initial y-pixel guesses at ``ref_col``.  Skips automatic peak finding
        when provided — useful with ``locate_spectra()`` or ``find_peaks()``.
    max_jump : float
        Maximum allowed pixel jump between adjacent columns (default 2).
    min_dist : int
        Minimum distance between peaks at reference column (default 6).
        Only used when ``ini_ys`` is None.
    plot : bool
        If True, overlay fitted traces on the image.

    Returns
    -------
    traces : ndarray (nfib, nx)
        Y-pixel center of each fiber at every spectral column.
        Pass to ``make_trace_extractor()``.

    Example
    -------
    >>> import PLred.specextract as specextract
    >>> import h5py, numpy as np
    >>> with h5py.File('averaged.h5') as f:
    ...     avg = f['avg_PLcam'][:]
    ...     nf  = f['metadata/nframes'][:]
    >>> ix, iy = np.unravel_index(np.argmax(nf), nf.shape)
    >>> traces = specextract.find_traces(avg[ix, iy], nfib=38, plot=True)
    >>> extractor = specextract.make_trace_extractor(traces, boxsize=3)
    """
    from PLred.visPLred.spec import find_multiple_peaks
    try:
        from PLred.imageutils import find_3point_peak
        _has_3pt = True
    except ImportError:
        _has_3pt = False

    ny, nx = image.shape

    if ref_col is None:
        margin = max(1, nx // 10)
        col_flux = np.nansum(image, axis=0)
        col_flux[:margin]  = 0
        col_flux[-margin:] = 0
        ref_col = int(np.argmax(col_flux))
    ref_col = int(np.clip(ref_col, 0, nx - 1))

    if ini_ys is not None:
        ini_ys = np.asarray(ini_ys, dtype=float)
        if len(ini_ys) != nfib:
            raise ValueError(f"ini_ys has {len(ini_ys)} entries but nfib={nfib}")
    else:
        profile = image[:, ref_col].copy()
        profile = np.nan_to_num(profile, nan=0.0)
        ini_ys = find_multiple_peaks(profile, n_peaks=nfib, min_dist=min_dist).astype(float)

    raw_trace = np.full((nfib, nx), np.nan)
    raw_trace[:, ref_col] = ini_ys

    def _track_direction(x_range, fib_ini_y):
        ypos = np.full(nx, np.nan)
        yint = int(np.round(fib_ini_y))
        for x in x_range:
            y0 = max(0, yint - trace_width)
            y1 = min(ny, yint + trace_width + 1)
            crop = image[y0:y1, x]
            yarr = np.arange(y0, y1)
            if len(yarr) < 3 or np.nanmax(crop) <= 0:
                ypos[x] = yint
                continue
            try:
                if _has_3pt:
                    peak = find_3point_peak(yarr, crop)
                else:
                    peak = float(yarr[np.argmax(crop)])
                if abs(peak - yint) <= max_jump:
                    ypos[x] = peak
                    yint = int(np.round(peak))
                else:
                    ypos[x] = yint
            except Exception:
                ypos[x] = yint
        return ypos

    for fibind in range(nfib):
        ini_y = ini_ys[fibind]
        left  = _track_direction(range(ref_col - 1, -1, -1), ini_y)
        right = _track_direction(range(ref_col + 1, nx),      ini_y)
        for x in range(ref_col - 1, -1, -1):
            if np.isfinite(left[x]):
                raw_trace[fibind, x] = left[x]
        for x in range(ref_col + 1, nx):
            if np.isfinite(right[x]):
                raw_trace[fibind, x] = right[x]

    effective_deg = min(poly_deg, max(1, nx // 8))
    x_arr  = np.arange(nx)
    traces = np.zeros((nfib, nx), dtype=np.float64)
    for fibind in range(nfib):
        valid = np.isfinite(raw_trace[fibind])
        if valid.sum() < effective_deg + 1:
            traces[fibind] = raw_trace[fibind]
        else:
            coeffs = np.polyfit(x_arr[valid], raw_trace[fibind, valid], deg=effective_deg)
            traces[fibind] = np.polyval(coeffs, x_arr)

    if plot:
        import matplotlib.pyplot as plt
        vmax = np.nanpercentile(image, 99)
        plt.figure(figsize=(min(12, nx * 0.05 + 4), 6))
        plt.imshow(image, aspect='auto', origin='upper', cmap='viridis',
                   vmin=0, vmax=max(vmax, 1.0))
        for fibind in range(nfib):
            plt.plot(x_arr, traces[fibind], lw=0.8, alpha=0.7, linestyle='--', color='white')
        plt.axvline(ref_col, color='white', lw=0.5, linestyle='--', alpha=0.5)
        plt.xlabel('x  (spectral)')
        plt.ylabel('y  (cross-dispersion)')
        plt.title(f'find_traces: {nfib} fibers  ref_col={ref_col}')
        plt.tight_layout()
        plt.show()

    return traces


# ------------------------------------------------------------------
# Instrument extractor factories
# ------------------------------------------------------------------

def make_simple_extractor(ylocs, width=6, dark=None):
    """
    Build a fixed-aperture box extractor.

    Sums ``±width`` rows around each fixed fiber y-center at every spectral
    column.  Works for any instrument where fiber traces are approximately
    horizontal.  Use ``make_trace_extractor`` instead when traces curve
    significantly across the detector.

    Parameters
    ----------
    ylocs : array-like (nfib,)
        Y-pixel centers of each fiber in the cross-dispersion direction.
        Typically from ``locate_spectra()``, ``find_peaks()``, or
        ``IRPLred.spec.locate_spectra()``.
    width : int
        Half-width of the extraction box in pixels (default 6).
    dark : ndarray (ny, nx) or None
        Dark frame to subtract before extraction.  Pass None if dark subtraction
        was already done upstream (e.g. in ``ingest_to_h5``).  When using
        ``extract_from_fits``, pass ``dark=None`` here and supply the dark via
        the ``dark_fits`` argument instead.

    Returns
    -------
    callable  f(image: ndarray[ny, nx]) -> ndarray[nfib, nwav]

    Example
    -------
    >>> import PLred.specextract as specextract
    >>> ylocs    = specextract.find_peaks(ref_image, nfib=38, thres=0.1)
    >>> extractor = specextract.make_simple_extractor(ylocs, width=3)
    >>> specextract.extract_to_coupling_map('averaged.h5', extractor, 'coupling_map.fits')
    """
    _ylocs = np.asarray(ylocs, dtype=int)

    def _extract(image):
        im = image - dark if dark is not None else image
        return extract_spec(im, _ylocs, width=width).astype(np.float32)

    return _attach_info(_extract, {
        'extractor': 'simple_box',
        'width':     width,
        'ylocs':     _ylocs.tolist(),
        'has_dark':  dark is not None,
    })


def make_FIRSTPL_extractor(model_file, dark=None, nonlin_modelfile=None,
                            var_const=200, thresh=0.1):
    """
    Build an optimal extractor for FIRST-PL data from a saved spectrum model.

    Loads the extraction matrix (and optionally the wavelength map) from the
    ``.npz`` file written by ``SpectrumModel.save_spectra_model()``.  If the
    model includes a wavelength map, each fiber's spectrum is interpolated onto
    the reference fiber's wavelength grid before being returned.

    The extractor expects a **pre-cropped** image of shape ``(ny, xmax-xmin)``
    where ``xmin``/``xmax`` come from the model file.  When using
    ``extract_from_fits``, pass the same ``model_file`` and the function will
    pre-slice each raw frame automatically.

    Parameters
    ----------
    model_file : str
        Path to the ``.npz`` spectrum model file produced by
        ``SpectrumModel.save_spectra_model()``
        (see ``visPLred/tutorials/pre2_spectrum_model.ipynb``).
    dark : ndarray (ny, nx) or None
        Dark frame to subtract from the **full** detector image before
        pre-cropping.  Pass None if dark subtraction is already done upstream.
        When using ``extract_from_fits`` this is handled by the ``dark_fits``
        argument instead — leave ``dark=None`` here in that case.
    nonlin_modelfile : str or None
        Path to the nonlinearity-correction FITS produced by
        ``preprocess.model_nonlinearity_from_flats()``.
        None skips nonlinearity correction.
    var_const : float
        Variance constant for the regularised least-squares solver (default 200).
    thresh : float
        Damping threshold for regularised extraction (default 0.1).

    Returns
    -------
    callable  f(image: ndarray[ny, xmax-xmin]) -> ndarray[nfib, nwav]
        If ``wav_map`` is present in the model file, ``nwav = xmax - xmin``
        and spectra are on the reference fiber's wavelength grid.
        Otherwise ``nwav = xmax - xmin`` in pixel space.

    Example
    -------
    >>> import PLred.specextract as specextract
    >>>
    >>> # Mode A — averaged H5 (images already cropped to xmin:xmax)
    >>> extractor = specextract.make_FIRSTPL_extractor(
    ...     'specmodel/specmodel.npz',
    ...     nonlin_modelfile='nonlin_model.fits',
    ... )
    >>> specextract.extract_to_coupling_map('averaged.h5', extractor, 'coupling_map.fits')
    >>>
    >>> # Mode B — raw FITS cubes (xmin/xmax auto-detected from model)
    >>> specextract.extract_from_fits(['sci.fits'], extractor, dark_fits='dark.fits')
    """
    model   = load_spectrum_model(model_file)
    A       = model.A
    wav_map = model.wav_map        # (nfib, nwav) or None
    xmin    = model.xmin
    xmax    = model.xmax
    nwav    = xmax - xmin
    ny_full = model.ny_full        # full detector height the matrix was built for
    # Derive nfib from matrix — avoids relying on the hardcoded nspec=38 in
    # frame_to_spec / vec_to_mat, which breaks for non-FIRST-PL instruments.
    nfib    = A.shape[0] // nwav

    def _extract(image):
        # image: (ny_full, nwav) — x-sliced and y-embedded by _crop_image
        im = image - dark if dark is not None else image.copy()
        if nonlin_modelfile is not None:
            from PLred.visPLred.preprocess import correct_nonlinearity_map
            im, _ = correct_nonlinearity_map(im, nonlin_modelfile)
        from PLred.visPLred.spec import extract_spec_optimal, interpolate_spectrum, flatten_im
        imvec = flatten_im(im, 0, nwav)           # (ny_full * nwav,)
        spec_flat, _ = extract_spec_optimal(
            A, imvec, var_const=var_const, thresh=thresh,
        )
        spec = spec_flat.reshape(nfib, nwav)      # (nfib, nwav) — not hardcoded
        if wav_map is not None:
            spec = interpolate_spectrum(spec, wav_map)
        return spec.astype(np.float32)

    return _attach_info(_extract, {
        'extractor':        'FIRSTPL_optimal',
        'model_file':       str(model_file),
        'xmin':             xmin,
        'xmax':             xmax,
        'ny_full':          ny_full,    # full detector height; used for y-ROI embedding
        'var_const':        var_const,
        'thresh':           thresh,
        'nonlin_modelfile': str(nonlin_modelfile) if nonlin_modelfile else 'none',
        'has_dark':         dark is not None,
        'has_wavmap':       wav_map is not None,
    })


def make_trace_extractor(traces_or_model_file, xmin=None, xmax=None,
                         boxsize=3, dark=None, nonlin_modelfile=None):
    """
    Build a trace-following box extractor.

    Wraps ``PLred.visPLred.spec.extract_spec_box``.  Unlike
    ``make_simple_extractor``, which uses a fixed y-center per fiber, this
    follows curved fiber traces across the detector.

    Accepts either a pre-computed traces array **or** a spectrum model file
    path (the ``.npz`` written by ``SpectrumModel.save_spectra_model()``).
    When a model file is given, ``trace_vals`` is sliced to ``[xmin, xmax)``
    automatically, and wavelength interpolation is applied if ``wav_map`` is
    present in the file.

    Parameters
    ----------
    traces_or_model_file : ndarray (nfib, nx) or str
        Either:

        * An array of shape ``(nfib, nx)`` — y-pixel center of each fiber at
          each spectral column, already sliced to the extraction ROI.
          Typically from ``find_traces()`` or ``SpectrumModel.trace_vals``
          pre-sliced to ``[:, xmin-XMIN : xmax-XMIN]``.
        * A path (str/Path) to a ``.npz`` spectrum model file.
          ``trace_vals`` will be loaded and sliced to ``[xmin, xmax)``
          automatically.

    xmin : int or None
        First spectral column.  Only used when loading from a model file;
        defaults to the ``xmin`` stored in the model.
    xmax : int or None
        Last spectral column (exclusive).  Only used with a model file;
        defaults to ``xmax`` from the model.
    boxsize : int
        Half-width of the extraction box in pixels (default 3).
    dark : ndarray (ny, nx) or None
        Dark frame to subtract.  None skips subtraction.  When using
        ``extract_from_fits``, pass ``dark=None`` here and supply the dark
        via ``dark_fits`` instead.
    nonlin_modelfile : str or None
        Path to nonlinearity-correction FITS.  None skips correction.

    Returns
    -------
    callable  f(image: ndarray[ny, nwav]) -> ndarray[nfib, nwav]
        Input image must be pre-cropped to the extraction range ``[xmin, xmax)``.
        ``extract_from_fits`` does this automatically from ``extractor._info``.

    Notes
    -----
    The traces array and ``xmin``/``xmax`` are stored in the extractor's
    ``_info`` dict.  ``extract_from_fits`` uses them to pre-slice each raw
    frame, and both extraction functions save the traces as a ``TRACES``
    extension HDU in the output FITS.

    Example
    -------
    >>> import PLred.specextract as specextract
    >>>
    >>> # from a model file (recommended for FIRST-PL)
    >>> extractor = specextract.make_trace_extractor(
    ...     'specmodel/specmodel.npz', boxsize=3,
    ... )
    >>> specextract.extract_to_coupling_map('averaged.h5', extractor, 'coupling_map.fits')
    >>>
    >>> # from traces computed with find_traces()
    >>> traces = specextract.find_traces(ref_image, nfib=38, ini_ys=ylocs)
    >>> extractor = specextract.make_trace_extractor(traces, boxsize=3)
    """
    from PLred.visPLred.spec import extract_spec_box

    if isinstance(traces_or_model_file, (str, os.PathLike)):
        model   = load_spectrum_model(traces_or_model_file)
        _xmin   = xmin if xmin is not None else model.xmin
        _xmax   = xmax if xmax is not None else model.xmax
        _XMIN   = model.XMIN
        if model.trace_vals is None:
            raise ValueError(
                "Model file has no trace_vals.  Run SpectrumModel.trace_spectra() "
                "before save_spectra_model()."
            )
        tv      = np.asarray(model.trace_vals, dtype=np.float64)
        _traces = tv[:, _xmin - _XMIN : _xmax - _XMIN]
        wav_map = model.wav_map
        model_label = str(traces_or_model_file)
    else:
        _traces     = np.asarray(traces_or_model_file, dtype=np.float64)
        _xmin       = xmin
        _xmax       = xmax
        wav_map     = None
        model_label = 'array'

    def _extract(image):
        # image: (ny, nwav) — already cropped to [xmin, xmax)
        im = image - dark if dark is not None else image
        if nonlin_modelfile is not None:
            from PLred.visPLred.preprocess import correct_nonlinearity_map
            im, _ = correct_nonlinearity_map(im, nonlin_modelfile)
        spec = extract_spec_box(_traces, im, boxsize=boxsize).astype(np.float32)
        if wav_map is not None:
            from PLred.visPLred.spec import interpolate_spectrum
            spec = interpolate_spectrum(spec, wav_map).astype(np.float32)
        return spec

    info = {
        'extractor':        'trace_box',
        'model_file':       model_label,
        'boxsize':          boxsize,
        'traces_shape':     str(_traces.shape),
        'nonlin_modelfile': str(nonlin_modelfile) if nonlin_modelfile else 'none',
        'has_dark':         dark is not None,
        'has_wavmap':       wav_map is not None,
        'traces':           _traces,   # stored as FITS HDU, not written to header
    }
    if _xmin is not None:
        info['xmin'] = _xmin
    if _xmax is not None:
        info['xmax'] = _xmax

    return _attach_info(_extract, info)


# ------------------------------------------------------------------
# 1-D optimal extraction (instrument-agnostic, Horne 1986 style)
# ------------------------------------------------------------------

def build_simple_optimal_profile(
    ref_image,
    traces,
    ext_width=8,
    poly_deg_loc=5,
    poly_deg_sig=10,
    plot=False,
    verbose=True,
):
    """
    Build a per-column Gaussian spatial profile from a reference image.

    For each fiber and each spectral column the cross-dispersion slice is
    described by a Gaussian.  The Gaussian center (relative to the trace
    center) and sigma are estimated via weighted moments, then smoothed with
    polynomials across the spectral axis.  The resulting unit-amplitude
    profile matrix ``P`` is ready for Horne-style optimal extraction.

    Parameters
    ----------
    ref_image : ndarray (ny, nwav)
        Dark-subtracted reference image.  A high-S/N average frame or the
        average of the best-populated grid bin works well.  Image must follow
        the ``(spatial, spectral)`` convention used throughout specextract.

        .. note::
            If your raw frames are ``(nwav, ny)`` (e.g. the Apapane IR camera),
            pass ``ref_image.T``.

    traces : ndarray (nfib, nwav)
        Y-pixel center of each fiber at each spectral column.  From
        ``find_traces()`` or loaded from a model file.
    ext_width : int
        Half-width of the extraction window in pixels (default 8).
        The profile is evaluated over ``2 * ext_width`` pixels centred on
        the trace position.
    poly_deg_loc : int
        Polynomial degree for smoothing the Gaussian center across the
        spectral axis (default 5).
    poly_deg_sig : int
        Polynomial degree for smoothing the Gaussian sigma across the
        spectral axis (default 10).
    plot : bool
        If True, plot raw and smoothed Gaussian parameters per fiber.
    verbose : bool

    Returns
    -------
    P : ndarray (nfib, nwav, 2*ext_width)
        Unit-amplitude Gaussian profiles.  ``P[fib, x, k]`` is the value of
        the profile for fiber ``fib`` at spectral column ``x`` and local
        cross-dispersion pixel ``k`` (relative to the trace center).
    smooth_params : ndarray (nfib, nwav, 2)
        Smoothed ``[center, sigma]`` used to build ``P``.  Column 0 is the
        Gaussian mean in local coordinates (ideally ≈ ext_width); column 1
        is sigma in pixels.

    Notes
    -----
    Gaussian parameters are estimated from weighted pixel moments (no
    iterative fitting), so the function runs in seconds even for large
    images.  The profile is unit-normalised (``amplitude = 1``) because only
    the *shape* enters the extraction formula.

    Save the result with ``save_simple_optimal_profile()`` and build an
    extractor with ``make_simple_optimal_extractor()``.

    Example
    -------
    >>> import h5py, numpy as np
    >>> import PLred.specextract as se
    >>> with h5py.File('averaged.h5') as f:
    ...     nf  = f['metadata/nframes'][:]
    ...     avg = f['avg_PLcam'][:]
    >>> ix, iy = np.unravel_index(np.argmax(nf), nf.shape)
    >>> ref = avg[ix, iy]
    >>> ylocs  = se.find_peaks(ref, nfib=3, thres=0.1)
    >>> traces = se.find_traces(ref, nfib=3, ini_ys=ylocs)
    >>> P, params = se.build_simple_optimal_profile(ref, traces, ext_width=8, plot=True)
    >>> se.save_simple_optimal_profile(P, traces, 'profile.npz')
    >>> extractor = se.make_simple_optimal_extractor('profile.npz')
    """
    ny, nwav = ref_image.shape
    nfib     = traces.shape[0]
    ext2     = 2 * ext_width
    local_arr = np.arange(ext2, dtype=np.float64)

    # Raw estimated parameters: [center (local coords), sigma] per fiber per col
    raw_params = np.full((nfib, nwav, 2), np.nan)

    for fibind in tqdm(range(nfib), disable=not verbose, desc="Fitting profiles"):
        for x in range(nwav):
            y_cen = int(np.round(traces[fibind, x]))
            y0, y1 = y_cen - ext_width, y_cen + ext_width
            if y0 < 0 or y1 > ny:
                continue
            profile = ref_image[y0:y1, x].astype(np.float64)
            profile = np.clip(profile, 0, None)
            total = profile.sum()
            if total <= 0:
                continue
            # weighted moment estimates — fast, no iterative fitting needed
            mean  = float(np.dot(local_arr, profile) / total)
            var   = float(np.dot((local_arr - mean) ** 2, profile) / total)
            sigma = max(np.sqrt(var), 0.3)   # floor avoids zero-sigma
            raw_params[fibind, x, 0] = mean
            raw_params[fibind, x, 1] = sigma

    # Smooth center and sigma with polynomials across the spectral axis
    smooth_params = raw_params.copy()
    x_arr = np.arange(nwav, dtype=np.float64)
    for fibind in range(nfib):
        for param_idx, deg in [(0, poly_deg_loc), (1, poly_deg_sig)]:
            valid = np.isfinite(raw_params[fibind, :, param_idx])
            eff_deg = min(deg, max(1, valid.sum() // 4))
            if valid.sum() > eff_deg + 1:
                coeffs = np.polyfit(x_arr[valid], raw_params[fibind, valid, param_idx],
                                    deg=eff_deg)
                smooth_params[fibind, :, param_idx] = np.polyval(coeffs, x_arr)

    if plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for fibind in range(nfib):
            axs[0].plot(x_arr, raw_params[fibind, :, 0], alpha=0.3, lw=0.8)
            axs[0].plot(x_arr, smooth_params[fibind, :, 0], lw=1.5,
                        label=f'fib {fibind}')
            axs[1].plot(x_arr, raw_params[fibind, :, 1], alpha=0.3, lw=0.8)
            axs[1].plot(x_arr, smooth_params[fibind, :, 1], lw=1.5)
        axs[0].set_title('Gaussian center (local coords)  raw=faint, smooth=solid')
        axs[1].set_title('Gaussian sigma  raw=faint, smooth=solid')
        for ax in axs:
            ax.set_xlabel('spectral column')
            ax.axhline(ext_width, color='k', lw=0.5, linestyle='--', alpha=0.4)
        axs[0].legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    # Build unit-amplitude profile matrix
    from astropy.modeling.functional_models import Gaussian1D
    P = np.zeros((nfib, nwav, ext2), dtype=np.float32)
    for fibind in range(nfib):
        mean_arr  = smooth_params[fibind, :, 0]
        sigma_arr = smooth_params[fibind, :, 1]
        valid = np.isfinite(mean_arr) & np.isfinite(sigma_arr) & (sigma_arr > 0)
        for x in np.where(valid)[0]:
            P[fibind, x] = Gaussian1D(
                amplitude=1, mean=mean_arr[x], stddev=sigma_arr[x]
            )(local_arr)

    if verbose:
        print(f"Profile built: P shape = {P.shape}")
        frac_valid = np.mean(np.any(P > 0, axis=2))
        print(f"  valid columns: {frac_valid * 100:.1f}%")

    return P, smooth_params


def save_simple_optimal_profile(P, traces, filename, xmin=0):
    """
    Save an optimal extraction profile matrix to a ``.npz`` file.

    Parameters
    ----------
    P : ndarray (nfib, nwav, 2*ext_width)
        Profile matrix from ``build_simple_optimal_profile()``.
    traces : ndarray (nfib, nwav)
        Fiber trace y-centers (same ``nwav`` as ``P``).
    filename : str
        Output path.  ``.npz`` extension added if missing.
    xmin : int
        Detector column offset — the spectral column corresponding to index 0
        in the ``nwav`` axis of ``P``.  Stored so ``make_simple_optimal_extractor``
        can record it in the output FITS header.

    Example
    -------
    >>> se.save_simple_optimal_profile(P, traces, 'profile.npz', xmin=40)
    """
    if not str(filename).endswith('.npz'):
        filename = str(filename) + '.npz'
    np.savez(
        filename,
        P      = np.asarray(P,      dtype=np.float32),
        traces = np.asarray(traces, dtype=np.float64),
        xmin   = np.array(xmin, dtype=int),
    )
    print(f"Profile saved: {filename}  (P shape {np.asarray(P).shape})")


def make_simple_optimal_extractor(P_or_file, dark=None):
    """
    Build a 1-D optimal extractor from a pre-built Gaussian profile matrix.

    Applies the Horne (1986) weighted-sum formula per spectral column::

        spec[fib, x] = Σ_k  data[y_cen+k, x] · P[fib, x, k]
                       ─────────────────────────────────────────
                             Σ_k  P[fib, x, k]²

    where the sum is over the ``2 * ext_width`` cross-dispersion pixels
    centred on the trace position ``y_cen = round(traces[fib, x])``.

    Parameters
    ----------
    P_or_file : str or tuple (P, traces)
        * **str/Path**: path to an ``.npz`` file written by
          ``save_simple_optimal_profile()``.  ``P`` and ``traces`` are
          loaded automatically.
        * **tuple**: ``(P_array, traces_array)`` passed directly.

    dark : ndarray (ny, nx) or None
        Dark frame to subtract before extraction.  When using
        ``extract_from_fits``, pass ``dark=None`` here and supply the dark
        via ``dark_fits`` instead.

    Returns
    -------
    callable  f(image: ndarray[ny, nwav]) -> ndarray[nfib, nwav]
        Image must be pre-cropped to the same ``nwav`` columns as ``P``.
        ``extract_from_fits`` does this automatically if ``xmin`` is stored
        in the ``.npz`` (set by ``save_simple_optimal_profile``).

    Notes
    -----
    The extraction is fully vectorised (no Python loop over spectral columns),
    so it runs in milliseconds per frame even for large images.

    Example
    -------
    >>> import PLred.specextract as se
    >>> extractor = se.make_simple_optimal_extractor('profile.npz')
    >>> # Mode A — averaged H5
    >>> se.extract_to_coupling_map('averaged.h5', extractor, 'coupling_map.fits')
    >>> # Mode B — raw FITS cubes (xmin auto-detected from profile.npz)
    >>> se.extract_from_fits(['sci.fits'], extractor, dark_fits='dark.fits')
    """
    if isinstance(P_or_file, (str, os.PathLike)):
        d      = np.load(P_or_file, allow_pickle=False)
        P      = d['P']
        traces = d['traces']
        xmin   = int(d['xmin']) if 'xmin' in d else 0
        model_label = str(P_or_file)
    else:
        P, traces   = P_or_file
        xmin        = 0
        model_label = 'array'

    P      = np.asarray(P,      dtype=np.float32)   # (nfib, nwav, 2*ext)
    traces = np.asarray(traces, dtype=np.float64)    # (nfib, nwav)
    nfib, nwav, ext2 = P.shape
    ext_width = ext2 // 2

    # Pre-compute P² sums (nfib, nwav); zero → NaN to avoid divide-by-zero
    P2_sum = np.sum(P ** 2, axis=2).astype(np.float64)   # (nfib, nwav)
    P2_sum[P2_sum == 0] = np.nan

    # Offset array for vectorised indexing: shape (2*ext_width,)
    offsets = np.arange(-ext_width, ext_width, dtype=int)

    def _extract(image):
        # image: (ny, nwav_roi)
        im = (image - dark).astype(np.float64) if dark is not None else image.astype(np.float64)
        ny_im, nx_im = im.shape
        spec = np.full((nfib, nwav), np.nan, dtype=np.float32)

        for fibind in range(nfib):
            centers = np.round(traces[fibind]).astype(int)  # (nwav,)

            # y indices for every column: (nwav, 2*ext)
            y_idx = centers[:, None] + offsets[None, :]
            x_idx = np.arange(nwav)[:, None] * np.ones(ext2, dtype=int)[None, :]

            # mask out-of-bounds
            valid_pix = (y_idx >= 0) & (y_idx < ny_im) & (x_idx < nx_im)

            y_safe = np.clip(y_idx, 0, ny_im - 1)
            x_safe = np.clip(x_idx, 0, nx_im - 1)

            data_crops = im[y_safe, x_safe].astype(np.float32)   # (nwav, 2*ext)
            P_fib      = P[fibind].copy()                         # (nwav, 2*ext)

            # zero out OOB positions so they contribute nothing
            data_crops[~valid_pix] = 0.0
            P_fib[~valid_pix]      = 0.0

            p2 = np.sum(P_fib ** 2, axis=1)
            p2[p2 == 0] = np.nan

            spec[fibind] = (np.sum(data_crops * P_fib, axis=1) / p2).astype(np.float32)

        return spec

    return _attach_info(_extract, {
        'extractor':  'simple_optimal',
        'model_file': model_label,
        'nfib':       nfib,
        'nwav':       nwav,
        'ext_width':  ext_width,
        'has_dark':   dark is not None,
        'xmin':       xmin,
    })


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _probe_extractor_shape(extractor, avg_plcam, map_n):
    """Run extractor on the first non-empty bin to determine (nfib, nwav)."""
    for ix in range(map_n):
        for iy in range(map_n):
            image = avg_plcam[ix, iy]
            if np.any(np.isfinite(image)):
                result = extractor(image)
                if result.ndim != 2:
                    raise ValueError(
                        f"extractor must return a 2D array (nfib, nwav), "
                        f"got shape {result.shape}"
                    )
                return result.shape
    raise ValueError("No non-empty bins found in avg_PLcam — cannot determine extractor output shape.")


# ------------------------------------------------------------------
# Low-level box extraction
# ------------------------------------------------------------------

def extract_spec(im, ylocs, width=6):
    """
    Extract spectra from a 2D image at fixed Y apertures.

    Parameters
    ----------
    im : ndarray (H, W)
    ylocs : array-like (nfib,)
        Y-coordinates of aperture centers.
    width : int
        Half-width of extraction box (default 6).

    Returns
    -------
    specs : ndarray (nfib, W)
    """
    specs = []
    for yloc in ylocs:
        spec = np.sum(im[yloc - width: yloc + width, :], axis=0)
        specs.append(spec)
    return np.array(specs)
