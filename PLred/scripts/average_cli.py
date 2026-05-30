"""
plred-average — Step 4: spatial averaging/binning into map.h5.

Usage
-----
    plred-average obs.ini
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        prog='plred-average',
        description='Step 4: spatially sort and average PLcam frames into map.h5.',
    )
    parser.add_argument('config', help='Pipeline config .ini file.')
    args = parser.parse_args()

    from PLred.average import average_to_h5_from_config
    average_to_h5_from_config(args.config)

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    from configobj import ConfigObj
    from PLred.scripts._diagnostics import plot_step4, save_diagnostic

    cfg    = ConfigObj(args.config)
    map_h5 = cfg.get('Average', {}).get('output', 'map.h5').strip() or 'map.h5'
    outdir = os.path.dirname(os.path.abspath(map_h5))

    save_diagnostic(plot_step4(map_h5), outdir, '4_average.png')


if __name__ == '__main__':
    main()
