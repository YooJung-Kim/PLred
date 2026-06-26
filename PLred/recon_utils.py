"""
High-level convenience wrappers for Step 5: polynomial modeling and image
reconstruction.  Designed for interactive notebook use.

Typical workflow
----------------
1.  ``mapmodel = CouplingMapModel(mapdata='step4_couplingmap.fits', min_nframes=5)``
2.  ``plot_fiber_sum(mapmodel)``                    # survey all positions to find signal
2.  ``inspect_wav_range(mapmodel, ...)``            # pick wav_fitrange/wav_reconrange
3.  ``run_polymodel(mapmodel, ...)``                # fit + save polymodel.fits
4.  ``plot_polymodel_map(mapmodel, ...)``           # diagnostics
5.  ``plot_polymodel_spectrum(mapmodel, ...)``
6.  ``animate_polymodel(mapmodel, ...)``
7.  ``fit_mm = CouplingMapModel(model='polymodel.fits')``
8.  ``run_astrometry(fit_mm, ...)``                 # centroid track
9.  ``run_reconstruction(fit_mm, ...)``             # MCMC image
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize

import PLred.mapmodel as _mm
import PLred.fit as _fit


# ---------------------------------------------------------------------------
# Range inspection
# ---------------------------------------------------------------------------

def plot_fiber_sum(mapmodel, ix=None, iy=None, wmin=None, wmax=None,
                   normalize=True, fitmin=None, fitmax=None, ax=None):
    """
    Plot raw-count spectra summed over all fibers to identify the signal window.

    ``mapmodel.data`` holds the raw accumulated counts per (ix, iy, fiber, wav).
    Summing over fibers collapses per-fiber noise while the science signal
    (coherent across fibers) stands out, making it easy to spot ``fitmin``/``fitmax``
    before examining individual fibers with ``inspect_wav_range``.

    Parameters
    ----------
    mapmodel : CouplingMapModel
        Loaded with ``mapdata=``.  Requires ``mapmodel.data`` (raw counts).
    ix, iy : int or None
        Grid bin indices.  If *both* are None, overlay fiber-sum spectra from
        all valid grid positions (thin gray) plus their median (bold red).
        If both are given, plot the single-position fiber-sum.
    wmin, wmax : int or None
        Wavelength index window to display.  Defaults to the full range.
    normalize : bool
        If True, normalize each curve to its maximum.  Useful for comparing
        spectral shapes across positions with different total flux.
    fitmin, fitmax : int, optional
        If provided, shade this window to confirm the signal region.
    ax : matplotlib Axes, optional
        If None, a new figure is created and returned.

    Returns
    -------
    fig : Figure   (only when ax is None)
    """
    _require_attr(mapmodel, 'data', "load the coupling map with mapdata=")
    nwav = mapmodel.data.shape[3]
    if wmin is None:
        wmin = 0
    if wmax is None:
        wmax = nwav

    wav = np.arange(wmin, wmax)

    return_fig = ax is None
    if return_fig:
        fig, ax = plt.subplots(figsize=(9, 3))

    if ix is not None and iy is not None:
        spec = np.nansum(mapmodel.data[ix, iy, :, wmin:wmax], axis=0)
        if normalize and np.nanmax(spec) > 0:
            spec = spec / np.nanmax(spec)
        ax.plot(wav, spec, color='tab:red')
        ax.set_title(f'Fiber-summed counts  (ix={ix}, iy={iy})')
    else:
        all_specs = []
        for i in range(mapmodel.map_n):
            for j in range(mapmodel.map_n):
                if not np.any(np.isfinite(mapmodel.data[i, j, :, wmin:wmax])):
                    continue
                spec = np.nansum(mapmodel.data[i, j, :, wmin:wmax], axis=0)
                if normalize and np.nanmax(spec) > 0:
                    spec = spec / np.nanmax(spec)
                all_specs.append(spec)
                ax.plot(wav, spec, color='gray', alpha=0.3, lw=0.8)
        if all_specs:
            med = np.nanmedian(np.array(all_specs), axis=0)
            ax.plot(wav, med, color='tab:red', lw=2, label='Median')
            ax.legend(fontsize=8)
        ax.set_title('Fiber-summed counts — all grid positions')

    if fitmin is not None and fitmax is not None:
        ax.axvspan(fitmin, fitmax, color='tab:red', alpha=0.18,
                   label=f'Signal window ({fitmin}–{fitmax})')
        ax.legend(fontsize=8)

    ax.set_xlabel('Wavelength index')
    ylabel = 'Normalized counts (fiber sum)' if normalize else 'Counts (fiber sum)'
    ax.set_ylabel(ylabel)

    if return_fig:
        plt.tight_layout()
        return fig


def inspect_wav_range(mapmodel, ix, iy, fibind,
                      wmin, wmax, fitmin=None, fitmax=None, ax=None):
    """
    Plot the normalized coupling spectrum at one grid position and fiber.

    Shade the region that would be *excluded* from polynomial fitting
    (the signal window between ``fitmin`` and ``fitmax``).  Use this to
    visually pick ``fitmin``/``fitmax`` before calling ``run_polymodel``.

    Parameters
    ----------
    mapmodel : CouplingMapModel
        Loaded with ``mapdata=``.
    ix, iy : int
        Grid bin indices.
    fibind : int
        Fiber (port) index (0-based).
    wmin, wmax : int
        Wavelength index range to display (absolute indices into normdata).
    fitmin, fitmax : int, optional
        Signal window to shade red (excluded from polynomial fit).
    ax : matplotlib Axes, optional
        If None, a new figure is created and returned.

    Returns
    -------
    fig : Figure   (only when ax is None)
    """
    wav = np.arange(wmin, wmax)
    data = mapmodel.normdata[ix, iy, fibind, wmin:wmax]

    return_fig = ax is None
    if return_fig:
        fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(wav, data, color='tab:red')
    ax.axvspan(wmin, wmax - 1, color='gray', alpha=0.08, label='Recon range')
    if fitmin is not None and fitmax is not None:
        ax.axvspan(fitmin, fitmax, color='tab:red', alpha=0.2,
                   label=f'Excluded from fit ({fitmin}–{fitmax})')
    ax.set_xlabel('Wavelength index')
    ax.set_ylabel('Normalized coupling')
    ax.set_title(f'Port {fibind + 1},  position (ix={ix}, iy={iy})')
    ax.legend(fontsize=8)

    if return_fig:
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Polynomial model fitting
# ---------------------------------------------------------------------------

def run_polymodel(mapmodel, wav_fitrange, wav_reconrange, output_name,
                  poly_deg_spatial=9, poly_deg_spectral=6):
    """
    Fit a 2-D spatial + 1-D spectral polynomial model to the coupling maps.

    Thin wrapper around ``CouplingMapModel.make_polynomial_model``.  After
    this call ``mapmodel`` will carry ``all_map_inputs``, ``all_modeled_recons``,
    ``wav_reconrange``, and ``wav_fitrange`` — needed by the diagnostic
    functions below.

    For astrometry and image reconstruction you will need to reload the saved
    file::

        fit_mm = CouplingMapModel(model=output_name + '.fits')

    Parameters
    ----------
    mapmodel : CouplingMapModel
        Loaded with ``mapdata=``.
    wav_fitrange : array-like of int
        Absolute wavelength indices to use for fitting (exclude signal window).
    wav_reconrange : array-like of int
        Absolute wavelength indices to reconstruct (typically a superset of
        ``wav_fitrange`` that includes the signal region).
    output_name : str
        Output path without ``.fits`` extension.
    poly_deg_spatial : int
        Degree of the 2-D spatial polynomial (default 9).
    poly_deg_spectral : int
        Degree of the 1-D spectral polynomial over coefficients (default 6).

    Returns
    -------
    (all_map_inputs, all_modeled_recons, all_modeled_coeffs, model_chi2)
        Same as ``CouplingMapModel.make_polynomial_model``.
    """
    return mapmodel.make_polynomial_model(
        output_name,
        wav_fitrange,
        wav_reconrange,
        poly_deg_spatial=poly_deg_spatial,
        poly_deg_spectral=poly_deg_spectral,
    )


# ---------------------------------------------------------------------------
# Diagnostic: spatial maps
# ---------------------------------------------------------------------------

def plot_polymodel_map(mapmodel, specind, fibind=None,
                       vmax=0.012, vmax_res=None, vmax_sn=10):
    """
    Visualize the polynomial model at one wavelength.

    Parameters
    ----------
    mapmodel : CouplingMapModel
        After ``run_polymodel`` (needs ``all_map_inputs``, ``all_modeled_recons``).
    specind : int
        0-based index into ``wav_reconrange``.
    fibind : int or None
        *None* → grid of model maps for all fibers.
        *int*  → 4-panel (data | model | residual | S/N) for that fiber.
    vmax : float
        Color scale maximum for data/model panels.
    vmax_res : float or None
        Color scale for residual; defaults to ``vmax / 5``.
    vmax_sn : float
        Color scale for S/N panel (symmetric).
    """
    _require_attr(mapmodel, 'all_map_inputs',
                  'run_polymodel (or make_polynomial_model) first')

    wav_abs = mapmodel.wav_reconrange[specind]
    extent = (mapmodel.pos_mas[0], mapmodel.pos_mas[-1],
              mapmodel.pos_mas[0], mapmodel.pos_mas[-1])
    if vmax_res is None:
        vmax_res = vmax / 5

    if fibind is not None:
        data  = mapmodel.all_map_inputs[:, :, fibind, specind]
        model = mapmodel.all_modeled_recons[:, :, fibind, specind]
        res   = data - model
        normvar = _get_normvar(mapmodel, fibind, specind)
        with np.errstate(invalid='ignore', divide='ignore'):
            sn = res / np.sqrt(normvar)

        fig, axs = plt.subplots(1, 4, figsize=(13, 3), sharex=True, sharey=True)
        kw_img = dict(origin='lower', extent=extent)
        axs[0].imshow(data,  vmin=0,       vmax=vmax,     **kw_img)
        axs[1].imshow(model, vmin=0,       vmax=vmax,     **kw_img)
        axs[2].imshow(res,   vmin=-vmax_res, vmax=vmax_res, cmap='RdBu', **kw_img)
        axs[3].imshow(sn,    vmin=-vmax_sn, vmax=vmax_sn,  cmap='RdBu', **kw_img)
        for ax, title in zip(axs, ['Data', 'Model', 'Residual', 'S/N']):
            ax.set_title(title)
            ax.set_xlabel('x (mas)')
        axs[0].set_ylabel('y (mas)')
        fig.suptitle(f'Port {fibind + 1},  wav index {wav_abs}')
        plt.tight_layout()
        return fig

    # All-fiber grid (model only)
    nfib  = mapmodel.nfib
    ncols = 5
    nrows = int(np.ceil(nfib / ncols))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols * 2, nrows * 2),
                            sharex=True, sharey=True)
    axs = axs.flatten()
    kw = dict(vmin=0, vmax=vmax, origin='lower', extent=extent)
    for i in range(nfib):
        axs[i].imshow(mapmodel.all_modeled_recons[:, :, i, specind], **kw)
        axs[i].set_title(f'Port {i + 1}', fontsize=7)
        axs[i].axis('off')
    for ax in axs[nfib:]:
        ax.axis('off')
    fig.suptitle(f'Model maps,  wav index {wav_abs}')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Diagnostic: spectrum at one sky position
# ---------------------------------------------------------------------------

def plot_polymodel_spectrum(mapmodel, ix, iy, fibind, ax=None):
    """
    Plot data vs. polynomial model spectrum at one grid position and fiber.

    The region excluded from the fit is shaded in red.

    Parameters
    ----------
    mapmodel : CouplingMapModel
        After ``run_polymodel``.
    ix, iy : int
        Grid bin indices.
    fibind : int
        Fiber (port) index (0-based).
    ax : matplotlib Axes, optional
    """
    _require_attr(mapmodel, 'all_map_inputs',
                  'run_polymodel (or make_polynomial_model) first')

    data  = mapmodel.all_map_inputs[ix, iy, fibind]
    model = mapmodel.all_modeled_recons[ix, iy, fibind]
    wav   = np.asarray(mapmodel.wav_reconrange)

    return_fig = ax is None
    if return_fig:
        fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(wav, data,  color='tab:red',   label='Data')
    ax.plot(wav, model, color='black',     label='Model')

    # shade wavelengths excluded from fitting
    if hasattr(mapmodel, 'wav_fitrange'):
        in_fit = np.isin(wav, mapmodel.wav_fitrange)
        _shade_excluded(ax, wav, in_fit)

    ax.set_xlabel('Wavelength index')
    ax.set_ylabel('Normalized coupling')
    ax.set_title(f'Port {fibind + 1},  position (ix={ix}, iy={iy})')
    ax.legend(fontsize=8)

    if return_fig:
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Diagnostic: animation across wavelengths
# ---------------------------------------------------------------------------

def animate_polymodel(mapmodel, fibind, wav_indices=None,
                      interval=100, vmax=0.012, vmax_res=None, vmax_sn=10):
    """
    Return a ``FuncAnimation`` cycling through wavelengths for one fiber.

    Four panels: data | model | residual | S/N.

    Parameters
    ----------
    mapmodel : CouplingMapModel
        After ``run_polymodel``.
    fibind : int
        Fiber (port) index (0-based).
    wav_indices : array-like of int or None
        0-based indices into ``wav_reconrange``.  Defaults to all.
    interval : int
        Milliseconds between frames.
    vmax, vmax_res, vmax_sn : float
        Color scale limits.
    """
    _require_attr(mapmodel, 'all_map_inputs',
                  'run_polymodel (or make_polynomial_model) first')

    if wav_indices is None:
        wav_indices = np.arange(len(mapmodel.wav_reconrange))
    if vmax_res is None:
        vmax_res = vmax / 5

    extent = (mapmodel.pos_mas[0], mapmodel.pos_mas[-1],
              mapmodel.pos_mas[0], mapmodel.pos_mas[-1])
    kw = dict(origin='lower', extent=extent)

    fig, axs = plt.subplots(1, 4, figsize=(13, 3), sharex=True, sharey=True)

    def _update(specind):
        for ax in axs:
            ax.clear()
        wav_abs = mapmodel.wav_reconrange[specind]
        data  = mapmodel.all_map_inputs[:, :, fibind, specind]
        model = mapmodel.all_modeled_recons[:, :, fibind, specind]
        res   = data - model
        normvar = _get_normvar(mapmodel, fibind, specind)
        with np.errstate(invalid='ignore', divide='ignore'):
            sn = res / np.sqrt(normvar)

        axs[0].imshow(data,  vmin=0,        vmax=vmax,     **kw)
        axs[1].imshow(model, vmin=0,        vmax=vmax,     **kw)
        axs[2].imshow(res,   vmin=-vmax_res, vmax=vmax_res, cmap='RdBu', **kw)
        axs[3].imshow(sn,    vmin=-vmax_sn,  vmax=vmax_sn,  cmap='RdBu', **kw)
        for ax, title in zip(axs, ['Data', 'Model', 'Residual', 'S/N']):
            ax.set_title(title)
        fig.suptitle(f'Port {fibind + 1},  wav index {wav_abs}')

    ani = FuncAnimation(fig, _update, frames=wav_indices, interval=interval)
    return ani


# ---------------------------------------------------------------------------
# Astrometry: centroid fitting
# ---------------------------------------------------------------------------

def run_astrometry(mapmodel, wav_indices, fibinds=None, n_trim=1,
                   x0=0.0, y0=0.0, plot=True):
    """
    Fit the centroid shift (x, y) at each wavelength by chi² minimization.

    Requires a model-loaded ``CouplingMapModel`` (``CouplingMapModel(model=...)``),
    not a mapdata-loaded one, because it calls ``mapmodel.compute_vec()``.

    Parameters
    ----------
    mapmodel : CouplingMapModel
        Loaded with ``model=``.
    wav_indices : array-like of int
        0-based indices into ``wav_reconrange`` (same as ``specind`` for
        ``compute_vec``).
    fibinds : array-like of int or None
        Fiber indices to include in the chi² sum.  Defaults to all fibers.
    n_trim : int
        Edge pixels to exclude (should match the value used when building the
        polynomial model).
    x0, y0 : float
        Initial guess for (x, y) shift in mas.
    plot : bool
        If True, plot the centroid track automatically.

    Returns
    -------
    results : list of dict
        One dict per wavelength with keys ``specind``, ``wav``, ``x``, ``y``,
        ``chi2``, ``success``.
    """
    _require_attr(mapmodel, 'model_coeffs',
                  'load mapmodel with model=... (not mapdata=...)')

    if fibinds is None:
        fibinds = np.arange(mapmodel.nfib)

    map_n = mapmodel.map_n
    trimmed = (map_n - 2 * n_trim, map_n - 2 * n_trim)

    def _chi2(param, specind):
        x, y = param
        chi2_sum, ndfs = 0.0, 0
        for fib in fibinds:
            model_vec = mapmodel.compute_vec(specind, fib, x, y,
                                             n_trim=n_trim).reshape(trimmed)
            data    = mapmodel.normdata[n_trim:-n_trim, n_trim:-n_trim, fib, specind]
            datavar = mapmodel.datanormvar[n_trim:-n_trim, n_trim:-n_trim, fib, specind]
            valid = np.isfinite(datavar) & (datavar > 0) & np.isfinite(data)
            if not valid.any():
                continue
            chi2_sum += float(np.sum((model_vec[valid] - data[valid]) ** 2
                                     / datavar[valid]))
            ndfs += int(valid.sum())
        ndfs -= 2
        return chi2_sum / ndfs if ndfs > 0 else np.nan

    results = []
    for specind in wav_indices:
        opt = minimize(_chi2, x0=[x0, y0], args=(int(specind),),
                       method='Nelder-Mead')
        wav_abs = int(mapmodel.wav_reconrange[specind])
        row = dict(specind=int(specind), wav=wav_abs,
                   x=float(opt.x[0]), y=float(opt.x[1]),
                   chi2=float(opt.fun), success=bool(opt.success))
        results.append(row)
        print(f'wav {wav_abs:3d} (specind {specind:2d}): '
              f'x={opt.x[0]:+.3f} mas,  y={opt.x[1]:+.3f} mas,  '
              f'chi2={opt.fun:.3f}')

    if plot and results:
        wavs = [r['wav'] for r in results]
        xs   = [r['x']   for r in results]
        ys   = [r['y']   for r in results]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(wavs, xs, marker='o', ms=3, label='x shift')
        ax.plot(wavs, ys, marker='o', ms=3, label='y shift')
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        ax.set_xlabel('Wavelength index')
        ax.set_ylabel('Centroid shift (mas)')
        ax.set_title('Astrometric centroid track')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return results


# ---------------------------------------------------------------------------
# Image reconstruction
# ---------------------------------------------------------------------------

def run_reconstruction(mapmodel, wavind, image_fov, image_ngrid,
                       n_trim=1, fiber_inds=None,
                       ini_temp=1e3, tau=3e5, gamma=10000,
                       n_elements=50, target_chi2=1.0,
                       burn_in_iter=500, niter=1000,
                       prior_type='uniform', plot=True,
                       small_to_random_ratio=1, **kwargs):
    """
    Run MCMC image reconstruction at one wavelength.

    Parameters
    ----------
    mapmodel : CouplingMapModel
        Loaded with ``model=`` (needs ``compute_vec``).  You can also pass the
        path to a polymodel FITS file as a string.
    wavind : int
        0-based index into ``wav_reconrange`` (the ``specind`` argument).
    image_fov : float
        Reconstructed image field-of-view in mas.
    image_ngrid : int
        Number of pixels per side in the reconstructed image.
    n_trim : int
        Edge pixels to trim from the coupling-map grid.
    fiber_inds : array-like of int or None
        Fibers to include.  Defaults to all.
    ini_temp : float
        Initial annealing temperature.
    tau : float
        Temperature cooling timescale.
    gamma : float
        Adaptive temperature parameter.
    n_elements : int
        Number of flux elements in the MCMC image.
    target_chi2 : float
        Reduced chi² target for convergence.
    burn_in_iter : int
        Burn-in iterations (discarded from posterior).
    niter : int
        Total MCMC iterations.
    prior_type : str
        Spatial prior passed to ``make_prior`` (``'uniform'``, ``'circle'``,
        ``'gaussian'``).
    plot : bool
        Show live MCMC progress plots.
    small_to_random_ratio : float
        Fraction of MCMC steps that are small vs. random jumps.
    **kwargs
        Extra keyword arguments forwarded to ``PLMapFit.run``.

    Returns
    -------
    rc : ReconResult
        MCMC chain result (has ``post_locs``, ``axis_len``, etc.).
    fitter : PLMapFit
        The fitter object (useful for ``plot_data``, ``plot_residuals``, etc.).
    """
    if isinstance(mapmodel, str):
        model_file = mapmodel
    else:
        _require_attr(mapmodel, 'model_coeffs',
                      'load mapmodel with model=... (not mapdata=...)')
        model_file = mapmodel.model_fits.filename()

    if fiber_inds is None:
        # detect nfib from the model file
        _tmp = mapmodel if not isinstance(mapmodel, str) else \
               _mm.CouplingMapModel(model=mapmodel)
        fiber_inds = np.arange(_tmp.nfib)

    fitter = _fit.PLMapFit(
        model_file  = model_file,
        image_ngrid = image_ngrid,
        image_fov   = image_fov,
        n_trim      = n_trim,
    )
    fitter.make_matrix(wavind, fiber_inds=fiber_inds)
    fitter.prepare_data(fiber_inds=fiber_inds)
    fitter.store_hyperparams(ini_temp, tau, gamma, n_elements, target_chi2)
    rc = fitter.run(
        niter                = niter,
        burn_in_iter         = burn_in_iter,
        prior_type           = prior_type,
        plot                 = plot,
        small_to_random_ratio = small_to_random_ratio,
        **kwargs,
    )
    return rc, fitter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_attr(obj, attr, hint):
    if not hasattr(obj, attr):
        raise AttributeError(
            f"'{type(obj).__name__}' has no attribute '{attr}'. "
            f"Call {hint}."
        )


def _get_normvar(mapmodel, fibind, specind):
    """Return datanormvar for one fiber at one specind (handles both shapes)."""
    nv = mapmodel.datanormvar
    if nv.ndim == 4:
        if nv.shape[-1] == len(mapmodel.wav_reconrange):
            # model-loaded: last axis is already wav_reconrange
            return nv[:, :, fibind, specind]
        else:
            # mapdata-loaded: last axis is full wav range
            wav_abs = mapmodel.wav_reconrange[specind]
            return nv[:, :, fibind, wav_abs]
    raise ValueError('Unexpected datanormvar shape: {}'.format(nv.shape))


def _shade_excluded(ax, wav, in_fit):
    """Shade contiguous wavelength ranges where in_fit is False."""
    excluded = wav[~in_fit]
    if not len(excluded):
        return
    diffs  = np.diff(excluded)
    breaks = np.where(diffs > 1)[0] + 1
    groups = np.split(excluded, breaks)
    first  = True
    for grp in groups:
        label = 'Excluded from fit' if first else ''
        ax.axvspan(grp[0] - 0.5, grp[-1] + 0.5,
                   color='tab:red', alpha=0.18, label=label)
        first = False
    ax.legend(fontsize=8)
