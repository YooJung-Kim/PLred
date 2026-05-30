"""
plred-traces — find fiber traces from a flat/lamp FITS and save traces.npz.

Usage
-----
    plred-traces flat.fits --nfib 38 --out traces.npz
    plred-traces flat.fits --dark dark.fits --nfib 38 --thres 0.08 --out traces.npz
    plred-traces flat.fits --nfib 12 --transpose --out traces.npz
    plred-traces flat.fits --nfib 38 --config obs.ini --out traces.npz
    plred-traces flat.fits --nfib 38 --no-plot --out traces.npz
"""

import argparse
import sys


def _read_orientation_from_config(config_path):
    """Return True if [Instrument].spectral_orientation == 'vertical'."""
    try:
        from configobj import ConfigObj
        cfg = ConfigObj(config_path)
        orientation = cfg.get('Instrument', {}).get('spectral_orientation', 'horizontal')
        return orientation.strip().lower() == 'vertical'
    except Exception as e:
        print(f"Warning: could not read config {config_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        prog='plred-traces',
        description='Find fiber traces from a flat/lamp FITS and save traces.npz.',
    )
    parser.add_argument('flat', help='Flat or lamp FITS file. Multi-frame cubes are averaged.')
    parser.add_argument('--dark', '-d', default=None,
                        help='Dark FITS (also averaged if cube).')
    parser.add_argument('--nfib', '-n', type=int, required=True,
                        help='Number of fiber traces expected.')
    parser.add_argument('--out', '-o', default='traces.npz',
                        help='Output npz path (default: traces.npz).')
    parser.add_argument('--thres', type=float, default=0.05,
                        help='peakutils detection threshold 0–1 (default 0.05).')
    parser.add_argument('--min-dist', type=int, default=6,
                        help='Minimum pixel separation between peaks (default 6).')
    parser.add_argument('--trace-width', type=int, default=4,
                        help='Cross-dispersion half-width for tracking (default 4).')
    parser.add_argument('--poly-deg', type=int, default=5,
                        help='Polynomial degree for trace smoothing (default 5).')
    parser.add_argument('--xmin', type=int, default=None,
                        help='First spectral column (optional crop).')
    parser.add_argument('--xmax', type=int, default=None,
                        help='Last spectral column, exclusive (optional crop).')
    parser.add_argument('--transpose', action='store_true',
                        help='Transpose image before tracing (for vertical-trace detectors, e.g. CRED2).')
    parser.add_argument('--config', default=None,
                        help='Pipeline config .ini — reads [Instrument].spectral_orientation.')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip interactive diagnostic plot.')

    args = parser.parse_args()

    # Resolve transpose from config if not explicitly set
    transpose = args.transpose
    if not transpose and args.config is not None:
        transpose = _read_orientation_from_config(args.config)
        if transpose:
            print('Config: spectral_orientation=vertical → transposing image.')

    import os
    import PLred.specextract as se
    se.make_trace_file(
        fits_path   = args.flat,
        nfib        = args.nfib,
        dark_path   = args.dark,
        xmin        = args.xmin,
        xmax        = args.xmax,
        thres       = args.thres,
        min_dist    = args.min_dist,
        trace_width = args.trace_width,
        poly_deg    = args.poly_deg,
        outpath     = args.out,
        transpose   = transpose,
        plot        = False,    # always suppress interactive window; save file below
        verbose     = True,
    )

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    from PLred.scripts._diagnostics import plot_step5_traces, save_diagnostic, get_plots_dir

    if args.config:
        plots_dir = get_plots_dir(args.config)
    else:
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)) or '.', 'plots')

    save_diagnostic(
        plot_step5_traces(
            flat_fits  = args.flat,
            traces_npz = args.out,
            dark_fits  = args.dark,
            xmin       = args.xmin,
            xmax       = args.xmax,
        ),
        plots_dir, '5_traces.png',
    )


if __name__ == '__main__':
    main()
