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
    try:
        from configobj import ConfigObj
        cfg = ConfigObj(args.config)
        map_h5 = cfg.get('Average', {}).get('output', 'map.h5').strip() or 'map.h5'
        outdir = os.path.dirname(os.path.abspath(map_h5))

        from PLred.scripts._diagnostics import plot_step4, save_diagnostic
        fig = plot_step4(map_h5)
        save_diagnostic(fig, outdir, '4_average.png')
    except Exception as e:
        print(f'Warning: could not save step 4 diagnostic: {e}')


if __name__ == '__main__':
    main()
