"""
Spectral extraction bridge: averaged.h5 → coupling_map.fits → CouplingMapModel.

Overview
--------
`extract_to_coupling_map()` converts the output of `average_to_h5()` into the
FITS format consumed by `CouplingMapModel(mapdata=...)` in `mapmodel.py`.

The extraction engine is fully user-supplied via a callable so that visPLred,
IRPLred, or any future instrument can plug in its own method:

    import functools
    import PLred.visPLred.spec as spec

    my_extractor = functools.partial(
        spec.extract_spec_box,
        traces=traces,
        boxsize=3,
    )

    extract_to_coupling_map(
        'averaged.h5',
        extractor=my_extractor,
        output_fits='coupling_map.fits',
    )

Output FITS structure (HDU indices match CouplingMapModel(mapdata=...))
-----------------------------------------------------------------------
    [0] primary  : (map_n, map_n, nfib, nwav)  mean extracted spectra
    [1] nframes  : (map_n, map_n)               frames per bin
    [2] reserved : (map_n, map_n, nfib, nwav)   zeros (placeholder for model-load compat)
    [3] var      : (map_n, map_n, nfib, nwav)   variance of the mean per bin
    [4] normvar  : (map_n, map_n, nfib, nwav)   normalized variance
    header keywords: XMIN, XMAX, MAP_N (on HDU[0])

Variance estimation (three-tier priority)
-----------------------------------------
1. Bootstrap  — if /bootstrap/avg_PLcam exists in the H5 and use_bootstrap=True,
                the extractor is applied to each bootstrap resample and
                np.var(resamples, axis=0, ddof=1) gives the variance of the mean.
2. variance_extractor — user-supplied callable image(ny,nx) → var(nfib,nwav).
3. Poisson fallback — var = spectra / nframes  (valid for photon-count detectors).
"""

import numpy as np
import h5py
from astropy.io import fits
from tqdm import tqdm


def extract_to_coupling_map(
    averaged_h5,
    extractor,
    output_fits,
    variance_extractor=None,
    use_bootstrap=True,
    verbose=True,
):
    """
    Extract spectra from an averaged H5 file and write a coupling-map FITS.

    Parameters
    ----------
    averaged_h5 : str
        Path to the averaged H5 produced by `average_to_h5()`.
    extractor : callable
        ``f(image: ndarray[ny, nx]) -> ndarray[nfib, nwav]``
        Applied to the mean PLcam image of each non-empty grid bin.
        Configure hyperparameters (traces, box size, regularization, …) via
        ``functools.partial`` or a closure before passing.
    output_fits : str
        Output path for the coupling-map FITS file (written with overwrite=True).
    variance_extractor : callable, optional
        ``f(image: ndarray[ny, nx]) -> ndarray[nfib, nwav]``
        Returns the variance of the *mean* spectrum for one bin.
        If None, variance falls back to bootstrap (if available) or Poisson.
    use_bootstrap : bool
        If True and ``/bootstrap/avg_PLcam`` exists in the H5, apply extractor
        to each bootstrap resample and use sample variance as the mean variance.
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        Path to the written FITS file (same as `output_fits`).
    """
    # ------------------------------------------------------------------
    # 1. Load averaged H5
    # ------------------------------------------------------------------
    with h5py.File(averaged_h5, 'r') as f:
        avg_plcam = f['avg_PLcam'][:]                        # (map_n, map_n, ny, nx)
        nframes   = f['metadata/nframes'][:]                 # (map_n, map_n)
        x_mas     = f['metadata/x_mas'][:]                   # (map_n,)
        y_mas     = f['metadata/y_mas'][:]                   # (map_n,)

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

    if verbose:
        print(f"Loaded {averaged_h5}: map_n={map_n}, avg_PLcam shape={avg_plcam.shape}")

    # ------------------------------------------------------------------
    # 2. Extract spectra for each grid bin
    # ------------------------------------------------------------------
    # Probe shape from first non-empty bin
    nfib, nwav = _probe_extractor_shape(extractor, avg_plcam, map_n)
    if verbose:
        print(f"Extractor output shape: nfib={nfib}, nwav={nwav}")

    spectra = np.full((map_n, map_n, nfib, nwav), np.nan, dtype=np.float32)
    datavar = np.full_like(spectra, np.nan)

    # Bootstrap spectra buffer (only allocated if needed)
    bs_spectra = None
    if has_bootstrap:
        bs_spectra = np.full((n_bs, map_n, map_n, nfib, nwav), np.nan, dtype=np.float32)

    if verbose:
        print("Extracting spectra per grid bin...")

    grid_iter = [
        (ix, iy)
        for ix in range(map_n)
        for iy in range(map_n)
    ]

    for ix, iy in tqdm(grid_iter, disable=not verbose, desc="Extracting"):
        image = avg_plcam[ix, iy]                            # (ny, nx)
        if not np.any(np.isfinite(image)):
            continue                                         # empty bin → leave NaN

        spectra[ix, iy] = extractor(image).astype(np.float32)

        # Bootstrap extraction
        if has_bootstrap:
            for k in range(n_bs):
                bs_image = bs_plcam[k, ix, iy]
                if np.any(np.isfinite(bs_image)):
                    bs_spectra[k, ix, iy] = extractor(bs_image).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. Estimate variance
    # ------------------------------------------------------------------
    if has_bootstrap and bs_spectra is not None:
        # Variance of the mean = sample variance across resamples
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
            datavar[ix, iy] = variance_extractor(image).astype(np.float32)
    else:
        # Poisson fallback: var(mean) = mean / nframes
        if verbose:
            print("Variance estimated via Poisson approximation (spectra / nframes).")
        n = nframes[:, :, None, None].astype(np.float32)
        n[n == 0] = np.nan
        datavar = (spectra / n).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Normalized variance
    #    normdata = data / nansum(data, axis=(0,1))
    #    var(normdata) = datavar / nansum(data, axis=(0,1))^2
    # ------------------------------------------------------------------
    spatial_sum = np.nansum(spectra, axis=(0, 1))            # (nfib, nwav)
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

    hdu1 = fits.ImageHDU(nframes.astype(np.int32))
    hdu1.header['EXTNAME'] = 'nframes'

    # HDU[2]: placeholder so HDU[3] and HDU[4] align with CouplingMapModel indices
    hdu2 = fits.ImageHDU(np.zeros((map_n, map_n, nfib, nwav), dtype=np.float32))
    hdu2.header['EXTNAME'] = 'reserved'

    hdu3 = fits.ImageHDU(datavar)
    hdu3.header['EXTNAME'] = 'var'

    hdu4 = fits.ImageHDU(datanormvar)
    hdu4.header['EXTNAME'] = 'normvar'

    hdulist = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4])
    hdulist.writeto(output_fits, overwrite=True)

    if verbose:
        print(f"Coupling map FITS written to: {output_fits}")
        print(f"  spectra shape : {spectra.shape}")
        valid_bins = np.sum(nframes > 0)
        print(f"  valid bins    : {valid_bins} / {map_n * map_n}")

    return output_fits


# ------------------------------------------------------------------
# Instrument extractor factories
# ------------------------------------------------------------------

def make_irplred_extractor(ylocs, width=6, dark=None):
    """
    Build a box extractor for IRPLred data using PLred.IRPLred.spec.extract_spec.

    Parameters
    ----------
    ylocs : array-like (nfib,)
        Y-pixel centers of each fiber in the cross-dispersion direction.
        Typically found with ``IRPLred.spec.locate_spectra()``.
    width : int
        Half-width of the extraction box in pixels (default 6).
    dark : ndarray (ny, nx) or None
        Dark frame to subtract before extraction. Pass None if dark subtraction
        was already done upstream (e.g. in ``ingest_to_h5``).

    Returns
    -------
    callable  f(image: ndarray[ny, nx]) -> ndarray[nfib, nwav]

    Example
    -------
    >>> from PLred.IRPLred import spec as irspec
    >>> import PLred.specextract as specextract
    >>> ylocs = irspec.locate_spectra(ref_image, num_spec=17, width=3, plot=False)
    >>> extractor = specextract.make_irplred_extractor(ylocs, width=3)
    >>> specextract.extract_to_coupling_map('averaged.h5', extractor, 'coupling_map.fits')
    """
    from PLred.IRPLred.spec import extract_spec
    _ylocs = np.asarray(ylocs, dtype=int)

    def _extract(image):
        im = image - dark if dark is not None else image
        return extract_spec(im, _ylocs, width=width).astype(np.float32)

    return _extract


def make_visplred_extractor(A, dark=None, nonlin_modelfile=None,
                             var_const=200, thresh=0.1):
    """
    Build an optimal extractor for visPLred data.

    Wraps ``PLred.visPLred.spec.extract_spec_optimal`` with optional dark
    subtraction and per-pixel nonlinearity correction.

    The nonlinearity model is generated by the calibration notebook
    ``visPLred/tutorials/pre1_nonlinearity_correction.ipynb`` and stored as a
    FITS file via ``preprocess.model_nonlinearity_from_flats()``.

    Parameters
    ----------
    A : scipy.sparse matrix
        Spectral extraction matrix built from the spectrum model
        (see ``visPLred/tutorials/pre2_spectrum_model.ipynb``).
    dark : ndarray (ny, nx) or None
        Dark frame to subtract. None skips subtraction.
    nonlin_modelfile : str or None
        Path to nonlinearity-correction FITS produced by
        ``preprocess.model_nonlinearity_from_flats()``.
        None skips nonlinearity correction.
    var_const : float
        Variance constant used when no variance image is available (default 200).
    thresh : float
        Damping threshold for the regularised extraction (default 0.1).

    Returns
    -------
    callable  f(image: ndarray[ny, nx]) -> ndarray[nfib, nwav]

    Notes
    -----
    ``extract_spec_optimal`` returns a flat vector of length ``nfib * nwav``.
    This factory reshapes it to ``(nfib, nwav)`` where ``nwav = image.shape[1]``.
    Ensure that ``A`` was built for the same ``nfib`` and ``nwav`` dimensions.

    Example
    -------
    >>> import PLred.specextract as specextract
    >>> extractor = specextract.make_visplred_extractor(
    ...     A               = A,
    ...     dark            = dark_frame,
    ...     nonlin_modelfile= 'nonlin_model.fits',
    ... )
    >>> specextract.extract_to_coupling_map('averaged.h5', extractor, 'coupling_map.fits')
    """
    from PLred.visPLred.spec import extract_spec_optimal

    def _extract(image):
        im = image - dark if dark is not None else image.copy()
        if nonlin_modelfile is not None:
            from PLred.visPLred.preprocess import correct_nonlinearity_map
            im, _ = correct_nonlinearity_map(im, nonlin_modelfile)
        spec, _ = extract_spec_optimal(A, im.ravel(), var_const=var_const, thresh=thresh)
        return spec.reshape(-1, image.shape[1]).astype(np.float32)

    return _extract


def find_peaks(image, nfib, thres=0.05, min_dist=6, ref_col=None):
    """
    Find fiber peak positions in a detector image using peakutils.

    An alternative to ``IRPLred.spec.locate_spectra()`` — useful when fibers
    are closely spaced or the iterative max-masking approach misses peaks.
    Returns a 1-D array of y-pixel centers suitable for passing to
    ``find_traces(ini_ys=...)``, ``make_irplred_extractor(ylocs=...)``, or
    ``make_visplred_box_extractor`` via ``find_traces``.

    Parameters
    ----------
    image : ndarray (ny, nx) or (ny,)
        Reference detector image (dark-subtracted) or a 1-D cross-dispersion
        profile.  If 2-D, the profile is formed by summing the column at
        ``ref_col``.
    nfib : int
        Expected number of fibers.  A ``ValueError`` is raised if peakutils
        finds a different number — use that as a signal to tune ``thres`` or
        ``min_dist``.
    thres : float
        Normalised detection threshold for ``peakutils.indexes`` (0–1,
        relative to the profile maximum).  Lower → detect fainter peaks;
        raise if spurious peaks appear (default 0.05).
    min_dist : int
        Minimum pixel separation between peaks (default 6).
    ref_col : int or None
        Column to use when ``image`` is 2-D.  Defaults to the brightest
        column (edge columns excluded).

    Returns
    -------
    ylocs : ndarray (nfib,) int
        Y-pixel centers of the detected peaks, sorted top-to-bottom.

    Raises
    ------
    ValueError
        If the number of detected peaks differs from ``nfib``.

    Example
    -------
    >>> import PLred.specextract as specextract
    >>> # as a drop-in for IRPLred.spec.locate_spectra
    >>> ylocs = specextract.find_peaks(avg_plcam[ix, iy], nfib=38, thres=0.1)
    >>> extractor = specextract.make_irplred_extractor(ylocs, width=3)
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

    Starting from initial peak positions at a reference column, this traces
    each fiber left and right across the detector using subpixel 3-point peak
    fitting, then fits a polynomial to each raw trace.

    No neon calibration required — any bright dark-subtracted frame works
    (a lamp flat, or the average science image from the best-populated grid bin).

    Parameters
    ----------
    image : ndarray (ny, nx)
        Reference detector image, dark-subtracted.  The brightest frame or an
        average of many frames gives the most reliable traces.
    nfib : int
        Number of fiber traces to find.
    trace_width : int
        Half-width of the cross-dispersion search window when tracking from
        one column to the next (default 4).
    poly_deg : int
        Degree of polynomial fit used to smooth each raw trace (default 5).
    ref_col : int or None
        Column index used for initial peak finding.  Defaults to the column
        with the highest total cross-dispersion flux (edge columns excluded).
    ini_ys : array-like (nfib,) or None
        Initial y-pixel guesses for each fiber at ``ref_col``.  When given,
        automatic peak finding is skipped entirely — useful when you already
        know roughly where the fibers are (e.g. from a previous run or from
        ``IRPLred.spec.locate_spectra()``).  The order must match the fiber
        order you want in the output traces.
    max_jump : float
        Maximum allowed pixel jump between adjacent columns before the new
        peak is rejected and the previous position is kept (default 2).
    min_dist : int
        Minimum distance between fiber peaks at the reference column (default 6).
        Only used when ``ini_ys`` is None.
    plot : bool
        If True, overlay the fitted traces on the image.

    Returns
    -------
    traces : ndarray (nfib, nx)
        Y-pixel center of each fiber at every spectral column.
        Ready to pass to ``make_visplred_box_extractor()``.

    Example
    -------
    >>> import PLred.specextract as specextract
    >>> # use the brightest grid bin as a reference image
    >>> import h5py, numpy as np
    >>> with h5py.File('averaged.h5') as f:
    ...     avg = f['avg_PLcam'][:]
    ...     nf  = f['metadata/nframes'][:]
    >>> ix, iy = np.unravel_index(np.argmax(nf), nf.shape)
    >>> traces = specextract.find_traces(avg[ix, iy], nfib=38, plot=True)
    >>> extractor = specextract.make_visplred_box_extractor(traces, boxsize=3)
    """
    from PLred.visPLred.spec import find_multiple_peaks
    try:
        from PLred.imageutils import find_3point_peak
        _has_3pt = True
    except ImportError:
        _has_3pt = False

    ny, nx = image.shape

    # Reference column: brightest column, but avoid the outer 10% of columns
    # where edge artefacts can produce spurious peaks.
    if ref_col is None:
        margin = max(1, nx // 10)
        col_flux = np.nansum(image, axis=0)
        col_flux[:margin]  = 0
        col_flux[-margin:] = 0
        ref_col = int(np.argmax(col_flux))
    ref_col = int(np.clip(ref_col, 0, nx - 1))

    # Initial peaks at reference column
    if ini_ys is not None:
        ini_ys = np.asarray(ini_ys, dtype=float)
        if len(ini_ys) != nfib:
            raise ValueError(
                f"ini_ys has {len(ini_ys)} entries but nfib={nfib}"
            )
    else:
        profile = image[:, ref_col].copy()
        profile = np.nan_to_num(profile, nan=0.0)
        ini_ys = find_multiple_peaks(profile, n_peaks=nfib, min_dist=min_dist).astype(float)

    raw_trace = np.full((nfib, nx), np.nan)
    raw_trace[:, ref_col] = ini_ys

    def _track_direction(x_range, fib_ini_y):
        """Track one fiber along x_range, returning per-column y positions."""
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
        # trace leftward then rightward from ref_col
        left  = _track_direction(range(ref_col - 1, -1, -1),    ini_y)
        right = _track_direction(range(ref_col + 1, nx),         ini_y)
        for x in range(ref_col - 1, -1, -1):
            if np.isfinite(left[x]):
                raw_trace[fibind, x] = left[x]
        for x in range(ref_col + 1, nx):
            if np.isfinite(right[x]):
                raw_trace[fibind, x] = right[x]

    # Polynomial fit to smooth each trace.
    # Cap degree so we never use more freedom than ~1 knot per 8 columns —
    # this prevents overfitting on short spectral ranges (e.g. nx=20 → deg 2).
    effective_deg = min(poly_deg, max(1, nx // 8))
    x_arr  = np.arange(nx)
    traces = np.zeros((nfib, nx), dtype=np.float64)
    for fibind in range(nfib):
        valid = np.isfinite(raw_trace[fibind])
        if valid.sum() < effective_deg + 1:
            traces[fibind] = raw_trace[fibind]  # not enough points; keep raw
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


def make_visplred_box_extractor(traces, boxsize=3, dark=None, nonlin_modelfile=None):
    """
    Build a trace-following box extractor for visPLred data.

    Wraps ``PLred.visPLred.spec.extract_spec_box``. Unlike
    ``make_irplred_extractor``, which uses a fixed y-center per fiber, this
    follows curved fiber traces across the detector — the y-center of each
    fiber varies with the spectral column x.

    Parameters
    ----------
    traces : ndarray (nfib, nx)
        Y-pixel center of each fiber at each spectral column, already sliced
        to match the ``nx`` width of the images in the averaged H5.
        Typically ``SpectrumModel.trace_vals`` (saved as ``trace_vals.npy``)
        sliced to the spectral range of your ROI:
        ``traces[:, xmin - XMIN : xmax - XMIN]``
    boxsize : int
        Half-width of the extraction box in pixels (default 3).
    dark : ndarray (ny, nx) or None
        Dark frame to subtract before extraction. None skips subtraction.
    nonlin_modelfile : str or None
        Path to nonlinearity-correction FITS from
        ``visPLred/tutorials/pre1_nonlinearity_correction.ipynb``.
        None skips nonlinearity correction.

    Returns
    -------
    callable  f(image: ndarray[ny, nx]) -> ndarray[nfib, nwav]

    Notes
    -----
    ``traces`` must have the same ``nx`` as the images it will be applied to.
    If the averaged H5 was built from a cropped detector region ``[xmin, xmax)``,
    slice the full-detector traces accordingly before passing them here.

    Example
    -------
    >>> import numpy as np
    >>> import PLred.specextract as specextract
    >>> # trace_vals shape: (nfib, XMAX-XMIN) from SpectrumModel.trace_spectra()
    >>> trace_vals = np.load('model/trace_vals.npy')
    >>> XMIN = 200
    >>> xmin, xmax = 600, 900          # spectral ROI used in ingest_to_h5
    >>> traces_roi = trace_vals[:, xmin - XMIN : xmax - XMIN]
    >>> extractor = specextract.make_visplred_box_extractor(traces_roi, boxsize=3)
    >>> specextract.extract_to_coupling_map('averaged.h5', extractor, 'coupling_map.fits')
    """
    from PLred.visPLred.spec import extract_spec_box
    _traces = np.asarray(traces, dtype=np.float64)

    def _extract(image):
        im = image - dark if dark is not None else image
        if nonlin_modelfile is not None:
            from PLred.visPLred.preprocess import correct_nonlinearity_map
            im, _ = correct_nonlinearity_map(im, nonlin_modelfile)
        return extract_spec_box(_traces, im, boxsize=boxsize).astype(np.float32)

    return _extract


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
