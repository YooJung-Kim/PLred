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
    from configobj import ConfigObj
    from PLred.scripts._diagnostics import plot_step2, save_diagnostic, get_plots_dir

    cfg        = ConfigObj(args.config)
    outputs    = cfg.get('Outputs', {})
    alldata_h5 = (
        outputs.get('ingest_output', '').strip()
        or cfg.get('Ingest', {}).get('output', 'alldata.h5').strip()
        or 'alldata.h5'
    )
    save_diagnostic(plot_step2(alldata_h5), get_plots_dir(args.config), '2_ingest.png')


if __name__ == '__main__':
    main()
