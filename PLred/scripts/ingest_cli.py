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

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    from configobj import ConfigObj
    from PLred.scripts._diagnostics import (
        plot_step2, plot_first_frame_check, save_diagnostic, get_plots_dir,
    )

    cfg        = ConfigObj(args.config)
    outputs    = cfg.get('Outputs', {})
    alldata_h5 = (
        outputs.get('ingest_output', '').strip()
        or cfg.get('Ingest', {}).get('output', 'alldata.h5').strip()
        or 'alldata.h5'
    )
    plots_dir  = get_plots_dir(args.config)

    save_diagnostic(plot_step2(alldata_h5), plots_dir, '2_ingest.png')

    try:
        pix2mas = float(cfg.get('Average', {}).get('pix2mas', 0) or 0) or None
    except Exception:
        pix2mas = None
    save_diagnostic(
        plot_first_frame_check(alldata_h5, pix2mas=pix2mas),
        plots_dir, '2_first_frame.png',
    )


if __name__ == '__main__':
    main()
