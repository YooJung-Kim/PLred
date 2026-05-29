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
