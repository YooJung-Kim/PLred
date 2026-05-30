"""
plred-profile — build a 1D Gaussian optimal-extraction profile from a flat FITS.

Usage
-----
    plred-profile flat.fits --dark dark.fits --traces traces.npz \\
        --ext-width 8 --poly-deg-loc 5 --poly-deg-sig 10 --out profile.npz

    plred-profile flat.fits --traces traces.npz --out profile.npz  # no dark
"""

import argparse


def _read_transpose_from_config(config_path):
    try:
        from configobj import ConfigObj
        cfg = ConfigObj(config_path)
        ori = cfg.get('Instrument', {}).get('spectral_orientation', 'horizontal')
        return ori.strip().lower() == 'vertical'
    except Exception as e:
        print(f"Warning: could not read config {config_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        prog='plred-profile',
        description='Build a 1D Gaussian optimal-extraction profile from a flat FITS.',
    )
    parser.add_argument('flat', help='Flat or lamp FITS file. Multi-frame cubes are averaged.')
    parser.add_argument('--dark', '-d', default=None,
                        help='Dark FITS subtracted before profile fitting.')
    parser.add_argument('--traces', required=True,
                        help='traces.npz from plred-traces (fiber positions).')
    parser.add_argument('--ext-width', type=int, default=8,
                        help='Cross-dispersion extraction half-width in pixels (default 8).')
    parser.add_argument('--poly-deg-loc', type=int, default=5,
                        help='Polynomial degree for fiber center smoothing (default 5).')
    parser.add_argument('--poly-deg-sig', type=int, default=10,
                        help='Polynomial degree for Gaussian sigma smoothing (default 10).')
    parser.add_argument('--xmin', type=int, default=None,
                        help='First spectral column.')
    parser.add_argument('--xmax', type=int, default=None,
                        help='Last spectral column (exclusive).')
    parser.add_argument('--out', '-o', default='profile.npz',
                        help='Output profile.npz path (default: profile.npz).')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip diagnostic plots.')
    parser.add_argument('--transpose', action='store_true',
                        help='Transpose image before fitting (CRED2 vertical-trace detectors).')
    parser.add_argument('--config', default=None,
                        help='Pipeline .ini — reads [Instrument].spectral_orientation.')

    args = parser.parse_args()

    transpose = args.transpose
    if not transpose and args.config:
        transpose = _read_transpose_from_config(args.config)
        if transpose:
            print('Config: spectral_orientation=vertical → transposing image.')

    import numpy as np
    from astropy.io import fits as pyfits
    import PLred.specextract as se

    # Load flat
    with pyfits.open(args.flat) as hdl:
        data = hdl[0].data.astype(np.float32)
    image = np.mean(data, axis=0) if data.ndim == 3 else data

    # Load and subtract dark
    if args.dark:
        with pyfits.open(args.dark) as hdl:
            dk = hdl[0].data.astype(np.float32)
        dark = np.mean(dk, axis=0) if dk.ndim == 3 else dk
        image = image - dark

    if transpose:
        image = image.T

    # Crop spectral range
    xmin = args.xmin or 0
    if args.xmin is not None or args.xmax is not None:
        xmax = args.xmax or image.shape[1]
        image = image[:, xmin:xmax]

    # Load traces (need full traces array, not just ylocs)
    d = np.load(args.traces, allow_pickle=False)
    if 'traces' in d:
        traces = d['traces'].astype(np.float64)
        # Crop traces to match xmin/xmax if needed
        if args.xmin is not None or args.xmax is not None:
            t_xmin = int(d['xmin']) if 'xmin' in d else 0
            s = (xmin - t_xmin)
            e = s + image.shape[1]
            traces = traces[:, max(0, s):e]
    else:
        raise ValueError("traces.npz must contain a 'traces' key (from plred-traces). "
                         "Re-run plred-traces to generate it.")

    P, _smooth_params = se.build_simple_optimal_profile(
        ref_image=image,
        traces=traces,
        ext_width=args.ext_width,
        poly_deg_loc=args.poly_deg_loc,
        poly_deg_sig=args.poly_deg_sig,
        plot=not args.no_plot,
    )

    se.save_simple_optimal_profile(P, traces, args.out, xmin=xmin)
    print(f"Profile written to: {args.out}")


if __name__ == '__main__':
    main()
