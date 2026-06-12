"""
plred-roi — Step 3: build ROI viewer cache for interactive parameter selection.

Usage
-----
    plred-roi obs.ini
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        prog='plred-roi',
        description='Step 3: build roi.h5 transposed cache and open the HTML viewer.',
    )
    parser.add_argument('config', help='Pipeline config .ini file.')
    args = parser.parse_args()

    from PLred.average import build_ROI_access_from_config
    build_ROI_access_from_config(args.config)

    print()
    print("ROI cache written.")
    print("Open PLred/scripts/h5_viewer.html in a browser to explore the data.")
    print("Fill in [Average] xc, yc, time_min, time_max in your config, then run:")
    print("    plred-average obs.ini")

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    from configobj import ConfigObj
    from PLred.scripts._diagnostics import plot_step3, save_diagnostic, get_plots_dir

    cfg        = ConfigObj(args.config)
    alldata_h5 = (
        cfg.get('Outputs', {}).get('ingest_output', '').strip()
        or cfg.get('Ingest', {}).get('output', '').strip()
        or 'alldata.h5'
    )

    save_diagnostic(plot_step3(alldata_h5), get_plots_dir(args.config), '3_roi.png')


if __name__ == '__main__':
    main()
