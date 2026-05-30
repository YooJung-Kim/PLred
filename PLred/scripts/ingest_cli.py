"""
plred-ingest — Step 2: build combined giant H5 from sorted frames and PLcam FITS.

Usage
-----
    plred-ingest obs.ini
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        prog='plred-ingest',
        description='Step 2: build alldata.h5 from fastcam.h5 + PLcam FITS files.',
    )
    parser.add_argument('config', help='Pipeline config .ini file.')
    args = parser.parse_args()

    from PLred.ingest import ingest_from_config_unified
    ingest_from_config_unified(args.config)

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    try:
        from configobj import ConfigObj
        cfg = ConfigObj(args.config)
        alldata_h5 = cfg.get('Ingest', {}).get('output', 'alldata.h5').strip() or 'alldata.h5'
        outdir = os.path.dirname(os.path.abspath(alldata_h5))

        from PLred.scripts._diagnostics import plot_step2, save_diagnostic
        fig = plot_step2(alldata_h5)
        save_diagnostic(fig, outdir, '2_ingest.png')
    except Exception as e:
        print(f'Warning: could not save step 2 diagnostic: {e}')


if __name__ == '__main__':
    main()
