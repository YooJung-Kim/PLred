"""
Wavelength calibration utilities for PLred.

Two calibration modes
---------------------
**Neon lamp** (``make_wave_solution_neon``)
    One image with multiple known emission lines.  User specifies rough pixel
    positions and known wavelengths.  The code finds precise line centers per
    fiber and fits a per-fiber polynomial ``pixel → wavelength``.
    Output: ``wavsol.npz``

**Tunable laser** (``make_wave_solution_laser``)
    N frames at N known wavelengths.  Each frame contains one laser spot per
    fiber.  A 2D Gaussian is fitted to each spot to find its x-center, then a
    per-fiber polynomial is fitted across all wavelength steps.
    Output: ``wcal.fits``

Both outputs can be passed as ``wavsol_file`` in the pipeline config to enable
wavelength interpolation for any extractor type.
"""

import os
import datetime
import numpy as np
from astropy.io import fits


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_and_average(fits_path, dark_path=None, transpose=False):
    """Load a FITS file, average frames if cube, optionally dark-subtract."""
    with fits.open(fits_path) as hdl:
        data = hdl[0].data.astype(np.float32)
    image = np.mean(data, axis=0) if data.ndim == 3 else data

    if dark_path is not None:
        with fits.open(dark_path) as hdl:
            dk = hdl[0].data.astype(np.float32)
        dark = np.mean(dk, axis=0) if dk.ndim == 3 else dk
        image = image - dark

    if transpose:
        image = image.T
    return image


def _load_traces(traces_or_file):
    """Return (ylocs, traces) from a traces.npz path or (ylocs_array, traces_array) tuple."""
    if isinstance(traces_or_file, (str, os.PathLike)):
        d = np.load(traces_or_file, allow_pickle=False)
        ylocs  = d['ylocs'].astype(int)
        traces = d['traces'] if 'traces' in d else None
    else:
        ylocs, traces = traces_or_file
        ylocs = np.asarray(ylocs, dtype=int)
    return ylocs, traces


# ------------------------------------------------------------------
# Neon lamp wavelength solution
# ------------------------------------------------------------------

def make_wave_solution_neon(
    neon_fits,
    traces_or_file,
    line_positions,
    xmin=None,
    xmax=None,
    poly_deg=3,
    dark_path=None,
    search_width=15,
    fiber_half_width=4,
    outpath='wavsol.npz',
    transpose=False,
    plot=True,
    verbose=True,
):
    """
    Build a per-fiber wavelength solution from a Neon lamp image.

    Parameters
    ----------
    neon_fits : str
        Neon lamp FITS file (cube is averaged).
    traces_or_file : str or (ylocs, traces)
        ``traces.npz`` path from ``plred-traces``, or a ``(ylocs, traces)``
        tuple where ``ylocs`` is shape ``(nfib,)`` and ``traces`` is
        ``(nfib, nwav)``.
    line_positions : list of (float, float)
        ``[(x_approx, wavelength_nm), ...]`` — approximate pixel positions
        of known Neon emission lines and their wavelengths in nm.
        ``x_approx`` is in image-local coordinates (after any xmin crop).
    xmin, xmax : int or None
        Spectral column crop.  Should match what was used when generating
        ``traces_or_file``.
    poly_deg : int
        Polynomial degree for the ``pixel → wavelength`` fit per fiber.
    dark_path : str or None
        Dark FITS for subtraction.
    search_width : int
        Half-width in pixels to search around each ``x_approx`` when refining
        the line center (default 15).
    fiber_half_width : int
        Half-width in y around each fiber center used to collapse the
        cross-dispersion profile for line-center finding (default 4).
    outpath : str
        Output ``wavsol.npz`` path.
    transpose : bool
        Transpose image before processing (CRED2 vertical-trace detectors).
    plot : bool
        Show diagnostic plots (per-fiber wavelength fits).
    verbose : bool

    Returns
    -------
    str
        Path to the written ``wavsol.npz``.
    """
    image = _load_and_average(neon_fits, dark_path, transpose)
    ny, nx = image.shape
    _xmin = int(xmin) if xmin is not None else 0
    _xmax = int(xmax) if xmax is not None else nx
    img = image[:, _xmin:_xmax]
    nwav = _xmax - _xmin

    ylocs, _ = _load_traces(traces_or_file)
    nfib = len(ylocs)

    if verbose:
        print(f"Neon wavecal: {nfib} fibers, {len(line_positions)} lines, "
              f"poly_deg={poly_deg}")

    # Find precise line centers per fiber
    # fit_centers[fib, line] = precise x-center of that Neon line for that fiber
    fit_centers = np.full((nfib, len(line_positions)), np.nan)

    for li, (x_approx, _wav) in enumerate(line_positions):
        xi = int(round(x_approx))
        x0 = max(0, xi - search_width)
        x1 = min(nwav, xi + search_width + 1)
        x_arr = np.arange(x0, x1)

        for fi, yloc in enumerate(ylocs):
            y0 = max(0, yloc - fiber_half_width)
            y1 = min(ny, yloc + fiber_half_width + 1)
            strip = img[y0:y1, x0:x1]
            profile = np.nansum(strip, axis=0).astype(np.float64)
            profile = np.clip(profile, 0, None)
            total = profile.sum()
            if total <= 0:
                continue
            # Weighted centroid
            center = float(np.dot(x_arr, profile) / total)
            fit_centers[fi, li] = center

    # Fit polynomial per fiber
    line_wavs = np.array([lp[1] for lp in line_positions])
    poly_coeffs = np.zeros((nfib, poly_deg + 1))
    x_eval = np.arange(nwav, dtype=np.float64)
    wav_map = np.zeros((nfib, nwav), dtype=np.float64)

    for fi in range(nfib):
        valid = np.isfinite(fit_centers[fi])
        if valid.sum() < poly_deg + 1:
            if verbose:
                print(f"  fiber {fi}: only {valid.sum()} valid lines — "
                      f"need {poly_deg + 1}. Skipping.")
            continue
        coeffs = np.polyfit(fit_centers[fi, valid], line_wavs[valid], deg=poly_deg)
        poly_coeffs[fi] = coeffs
        wav_map[fi] = np.polyval(coeffs, x_eval)

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        for fi in range(nfib):
            valid = np.isfinite(fit_centers[fi])
            axes[0].plot(fit_centers[fi, valid], line_wavs[valid], 'o',
                         alpha=0.6, ms=4, color=f'C{fi % 10}')
            axes[0].plot(x_eval, wav_map[fi], lw=0.8, alpha=0.7,
                         color=f'C{fi % 10}')
        axes[0].set_xlabel('x pixel (image-local)')
        axes[0].set_ylabel('Wavelength (nm)')
        axes[0].set_title(f'Neon line fits  ({nfib} fibers, deg {poly_deg})')
        axes[0].grid(alpha=0.3)

        axes[1].imshow(wav_map, aspect='auto', origin='upper',
                       extent=[_xmin, _xmax, nfib, 0])
        axes[1].set_xlabel('x pixel (detector)')
        axes[1].set_ylabel('Fiber index')
        axes[1].set_title('wav_map (nm)')
        plt.colorbar(axes[1].images[0], ax=axes[1], label='nm')
        plt.tight_layout()
        plt.show()

    np.savez(
        outpath,
        wav_map     = wav_map.astype(np.float64),
        poly_coeffs = poly_coeffs.astype(np.float64),
        xmin        = np.array(_xmin, dtype=np.int32),
        xmax        = np.array(_xmax, dtype=np.int32),
        method      = np.array('neon'),
        poly_deg    = np.array(poly_deg, dtype=np.int32),
    )
    if verbose:
        valid_fibs = np.any(wav_map != 0, axis=1).sum()
        print(f"wavsol.npz written to: {outpath}")
        print(f"  wav_map shape: {wav_map.shape}  valid fibers: {valid_fibs}/{nfib}")
        print(f"  wavelength range: {wav_map[wav_map>0].min():.2f}–"
              f"{wav_map.max():.2f} nm")
    return outpath


# ------------------------------------------------------------------
# Tunable laser wavelength solution
# ------------------------------------------------------------------

def make_wave_solution_laser(
    fits_files,
    wavelengths,
    traces_or_file,
    dark_path=None,
    crop_size=14,
    poly_deg=4,
    xmin=None,
    xmax=None,
    outpath='wcal.fits',
    transpose=False,
    plot=True,
    verbose=False,
    maxval=1000,
    bad_thres=200,
    pixels_to_interp=None,
):
    """
    Build a per-fiber wavelength solution from tunable laser sweep data.

    For each laser wavelength a separate FITS frame is provided.  The code
    fits a 2D Gaussian to each fiber's laser spot, records the precise
    x-center, then fits a per-fiber polynomial ``pixel → wavelength``.

    This is an adaptation of the user's ``solve_wavelength_calibration``
    code, wrapped with the standard PLred file-handling conventions.

    Parameters
    ----------
    fits_files : list of str
        FITS frames in wavelength order.  Each should contain a single frame
        (or will have its first frame used) showing the laser spot at the
        corresponding wavelength.
    wavelengths : array-like
        Wavelength in nm for each frame in ``fits_files``.
    traces_or_file : str or (ylocs, traces)
        ``traces.npz`` path or ``(ylocs, traces)`` tuple.
        Only ``ylocs`` is needed here (fiber y-positions).
    dark_path : str or None
        Dark FITS (averaged if cube) subtracted from each science frame.
    crop_size : int
        Size (pixels) of the square crop used for 2D Gaussian fitting (default 14).
    poly_deg : int
        Polynomial degree for ``pixel → wavelength`` fit per fiber (default 4).
    xmin, xmax : int or None
        Spectral column range.  If provided, ``pixels_to_interp`` defaults
        to this range.
    outpath : str
        Output FITS path (default ``wcal.fits``).
    transpose : bool
        Transpose each frame before processing (CRED2 detectors).
    plot : bool
        Show residual + wavelength fit diagnostic plots.
    verbose : bool
    maxval : float
        Clip Gaussian amplitude at this value to avoid saturation artefacts.
    bad_thres : float
        Pixels above this value in the median frame are masked as bad.
    pixels_to_interp : array-like or None
        Pixel positions at which to evaluate the wavelength solution.
        Defaults to the range covered by the fitted spot centers.

    Returns
    -------
    str
        Path to the written ``wcal.fits``.
    """
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    wavs   = np.asarray(wavelengths, dtype=np.float64)
    ylocs, _ = _load_traces(traces_or_file)
    ylocs    = np.sort(ylocs)
    nfib     = len(ylocs)

    # Load all frames
    frames = []
    for fp in fits_files:
        with fits.open(fp) as hdl:
            d = hdl[0].data.astype(np.float32)
        if d.ndim == 3:
            d = d[0]
        if transpose:
            d = d.T
        frames.append(d)
    frames = np.array(frames)

    # Load dark if provided
    if dark_path is not None:
        with fits.open(dark_path) as hdl:
            dk = hdl[0].data.astype(np.float32)
        dark = np.mean(dk, axis=0) if dk.ndim == 3 else dk
        if transpose:
            dark = dark.T
        frames = frames - dark[None]

    medimg = np.median(frames, axis=0)
    bad_pix_mask = np.ones_like(medimg)
    bad_pix_mask[medimg > bad_thres] = 0

    xg, yg = np.meshgrid(np.arange(crop_size), np.arange(crop_size))

    def gauss2d(params):
        xc, yc, xsig, ysig, peak, bg = params
        return peak * np.exp(-(((xg - xc) / xsig) ** 2 + ((yg - yc) / ysig) ** 2)) + bg

    def fit_gaussian2d(image, ini_param):
        bnds = ((0, crop_size), (0, crop_size), (0.5, 5), (0.5, 5), (0, None), (0, 100))
        res = minimize(
            lambda p: np.sum((image - gauss2d(p)) ** 2),
            ini_param, bounds=bnds, method='Powell',
        )
        if verbose:
            print(res.message)
        return res.x

    def find_xc_ini(im, yloc):
        summed = np.sum((im * bad_pix_mask)[yloc - crop_size // 2 : yloc + crop_size // 2, :],
                        axis=0)
        xc = int(np.argmax(summed[crop_size:-crop_size])) + crop_size
        return xc

    nframes = len(frames)
    fit_params_all = np.zeros((nframes, nfib, 6))

    if plot:
        n_panels = nframes * nfib
        ncols = max(1, int(np.sqrt(n_panels)))
        nrows = max(1, int(np.ceil(n_panels / ncols)))
        fig_res, axs_res = plt.subplots(nrows, ncols,
                                         figsize=(ncols * 2, nrows * 2))
        axs_res = np.asarray(axs_res).ravel()

    for i, im in enumerate(frames):
        if verbose:
            print(f'=== λ={wavs[i]:.2f} nm ===')
        for j, yloc in enumerate(ylocs):
            xc = int(find_xc_ini(im, yloc))
            image_crop = (im * bad_pix_mask)[
                yloc - crop_size // 2 : yloc + crop_size // 2,
                xc  - crop_size // 2 : xc  + crop_size // 2,
            ]
            peak_ini = min(float(np.max(image_crop)), maxval)
            ini = [crop_size / 2, crop_size / 2, 1.4, 0.8, peak_ini, 0.0]
            fp  = fit_gaussian2d(image_crop, ini)
            if plot:
                ax = axs_res[i * nfib + j]
                ax.imshow((image_crop - gauss2d(fp)) / np.clip(image_crop, 1, None),
                           origin='lower', vmin=-0.5, vmax=0.5)
                ax.axis('off')
            fp[0] += xc - crop_size // 2
            fp[1] += yloc - crop_size // 2
            fit_params_all[i, j] = fp

    if plot:
        fig_res.suptitle('Gaussian fit residuals (normalised)')
        plt.tight_layout()
        plt.show()

    # Polynomial fit per fiber
    if pixels_to_interp is None:
        xc_all = fit_params_all[:, :, 0].flatten()
        x_lo = int(np.floor(xc_all.min())) + 1
        x_hi = int(np.ceil(xc_all.max()))  - 1
        if xmin is not None:
            x_lo = max(x_lo, int(xmin))
        if xmax is not None:
            x_hi = min(x_hi, int(xmax))
        pixels_to_interp = np.arange(x_lo, x_hi)

    pixels_to_interp = np.asarray(pixels_to_interp, dtype=np.float64)
    # params_interp[0] = pixel axis; params_interp[1..nfib+1] = wavelength per fiber
    params_interp = np.zeros((nfib + 1, len(pixels_to_interp)))
    params_interp[0] = pixels_to_interp

    if plot:
        fig2, axs2 = plt.subplots(1, nfib, figsize=(4 * nfib, 3))
        if nfib == 1:
            axs2 = [axs2]

    for fi in range(nfib):
        x_centers = fit_params_all[:, fi, 0]
        coeffs = np.polyfit(x_centers, wavs, deg=poly_deg)
        params_interp[fi + 1] = np.poly1d(coeffs)(pixels_to_interp)
        if plot:
            axs2[fi].plot(x_centers, wavs, 'o', color=f'C{fi}', label='data')
            axs2[fi].plot(pixels_to_interp, params_interp[fi + 1], '-k', lw=1)
            axs2[fi].set_title(f'Fiber {fi}')
            axs2[fi].set_xlabel('x pixel')
            axs2[fi].set_ylabel('λ (nm)')
            axs2[fi].grid(alpha=0.3)

    if plot:
        plt.suptitle('Wavelength calibration fits')
        plt.tight_layout()
        plt.show()

    # Write FITS
    hdu = fits.PrimaryHDU(params_interp.astype(np.float64))
    hdu.header['TIME']    = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hdu.header['METHOD']  = 'laser'
    hdu.header['POLYDEG'] = poly_deg
    hdu.header['NPOINT']  = len(wavs)
    hdu.header['NFIB']    = nfib
    hdu.header['XMIN']    = int(pixels_to_interp[0])
    hdu.header['XMAX']    = int(pixels_to_interp[-1]) + 1
    for fi, yloc in enumerate(ylocs):
        hdu.header[f'YLOC{fi}'] = int(yloc)
    hdu.writeto(outpath, overwrite=True)

    if verbose or True:
        print(f"Wavelength solution written to: {outpath}")
        print(f"  params_interp shape: {params_interp.shape}")
        print(f"  pixel range: [{int(pixels_to_interp[0])}, {int(pixels_to_interp[-1])}]")
        print(f"  wavelength range: {params_interp[1:].min():.2f}–"
              f"{params_interp[1:].max():.2f} nm")
    return outpath
