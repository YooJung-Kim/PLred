"""
plred-wavesol — generate a per-fiber wavelength solution.

Two modes:

  neon   One neon lamp image with multiple known emission lines.
  laser  N frames at N known wavelengths (tunable laser sweep).

Usage
-----
# Neon mode
plred-wavesol neon.fits --dark dark.fits --traces traces.npz \\
    --lines "279:748.9,535:724.5,746:703.2" --out wavsol.npz

# Laser mode
plred-wavesol --mode laser laser_dir/ \\
    --wavelengths "650.0,651.0,652.0" --traces traces.npz --out wcal.fits

# Wavelengths from file (one per line)
plred-wavesol --mode laser laser_dir/ \\
    --wavelengths wavelengths.txt --traces traces.npz --out wcal.fits
"""

import argparse
import os
import sys


def _read_transpose_from_config(config_path):
    try:
        from configobj import ConfigObj
        cfg = ConfigObj(config_path)
        ori = cfg.get('Instrument', {}).get('spectral_orientation', 'horizontal')
        return ori.strip().lower() == 'vertical'
    except Exception as e:
        print(f"Warning: could not read config {config_path}: {e}")
        return False


def _parse_lines(lines_str):
    """Parse 'pixel:wavelength,...' string into [(pixel, wavelength), ...]."""
    pairs = []
    for token in lines_str.split(','):
        token = token.strip()
        if not token:
            continue
        parts = token.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid line spec {token!r}. Expected 'pixel:wavelength'.")
        pairs.append((float(parts[0]), float(parts[1])))
    return pairs


def _parse_wavelengths(wavelengths_str):
    """Parse comma-separated nm values or a path to a text file."""
    if os.path.isfile(wavelengths_str):
        with open(wavelengths_str) as f:
            vals = [float(line.strip()) for line in f if line.strip()]
        return vals
    return [float(v.strip()) for v in wavelengths_str.split(',') if v.strip()]


def _collect_fits_files(input_path):
    """Return sorted list of FITS files from a directory or a single file."""
    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, fn)
            for fn in os.listdir(input_path)
            if fn.lower().endswith('.fits')
        )
        if not files:
            raise FileNotFoundError(f"No FITS files found in {input_path}")
        return files
    return [input_path]


def main():
    parser = argparse.ArgumentParser(
        prog='plred-wavesol',
        description='Generate a per-fiber wavelength solution (neon lamp or laser sweep).',
    )
    parser.add_argument('input',
                        help='Neon FITS file (neon mode) or directory of FITS frames (laser mode).')
    parser.add_argument('--mode', choices=['neon', 'laser'], default='neon',
                        help='Calibration mode: neon (default) or laser.')
    parser.add_argument('--dark', '-d', default=None,
                        help='Dark FITS subtracted from science frames.')
    parser.add_argument('--traces', required=True,
                        help='traces.npz from plred-traces (fiber y-positions).')
    parser.add_argument('--lines', default=None,
                        help='[neon] Comma-separated pixel:wavelength_nm pairs, e.g. "279:748.9,535:724.5".')
    parser.add_argument('--wavelengths', default=None,
                        help='[laser] Comma-separated nm values or path to .txt file (one per frame).')
    parser.add_argument('--xmin', type=int, default=None,
                        help='First spectral column to include.')
    parser.add_argument('--xmax', type=int, default=None,
                        help='Last spectral column (exclusive).')
    parser.add_argument('--poly-deg', type=int, default=3,
                        help='Polynomial degree for pixel→wavelength fit (default 3).')
    parser.add_argument('--out', '-o', default=None,
                        help='Output path. Defaults to wavsol.npz (neon) or wcal.fits (laser).')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip diagnostic plots.')
    parser.add_argument('--transpose', action='store_true',
                        help='Transpose frames before processing (CRED2 vertical-trace detectors).')
    parser.add_argument('--config', default=None,
                        help='Pipeline .ini — reads [Instrument].spectral_orientation for transpose.')

    args = parser.parse_args()

    transpose = args.transpose
    if not transpose and args.config:
        transpose = _read_transpose_from_config(args.config)
        if transpose:
            print('Config: spectral_orientation=vertical → transposing frames.')

    from PLred.wavecal import make_wave_solution_neon, make_wave_solution_laser

    if args.mode == 'neon':
        if args.lines is None:
            parser.error('--lines is required for neon mode.')
        line_positions = _parse_lines(args.lines)
        outpath = args.out or 'wavsol.npz'
        make_wave_solution_neon(
            neon_fits=args.input,
            traces_or_file=args.traces,
            line_positions=line_positions,
            xmin=args.xmin,
            xmax=args.xmax,
            poly_deg=args.poly_deg,
            dark_path=args.dark,
            outpath=outpath,
            transpose=transpose,
            plot=not args.no_plot,
        )

    else:  # laser
        if args.wavelengths is None:
            parser.error('--wavelengths is required for laser mode.')
        wavelengths = _parse_wavelengths(args.wavelengths)
        fits_files  = _collect_fits_files(args.input)
        if len(fits_files) != len(wavelengths):
            parser.error(
                f'Number of FITS files ({len(fits_files)}) does not match '
                f'number of wavelengths ({len(wavelengths)}).'
            )
        outpath = args.out or 'wcal.fits'
        make_wave_solution_laser(
            fits_files=fits_files,
            wavelengths=wavelengths,
            traces_or_file=args.traces,
            dark_path=args.dark,
            poly_deg=args.poly_deg,
            xmin=args.xmin,
            xmax=args.xmax,
            outpath=outpath,
            transpose=transpose,
            plot=not args.no_plot,
        )


if __name__ == '__main__':
    main()
