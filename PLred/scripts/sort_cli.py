"""
plred-sort — Step 1: timestamp matching and PSFcam frame averaging.

Usage
-----
    plred-sort obs.ini
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        prog='plred-sort',
        description='Step 1: match timestamps between fast and slow cameras, average PSFcam frames.',
    )
    parser.add_argument('config', help='Pipeline config .ini file.')
    args = parser.parse_args()

    from PLred.sort import script_match_timestamps
    script_match_timestamps(args.config)

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    from configobj import ConfigObj
    from PLred.scripts._diagnostics import plot_step1, save_diagnostic

    cfg      = ConfigObj(args.config)
    outname  = cfg.get('Output', {}).get('outname', '.').strip() or '.'
    filename = cfg.get('Output', {}).get('filename', 'fastcam').strip() or 'fastcam'
    fastcam_h5 = os.path.join(outname, filename + '.h5')

    save_diagnostic(plot_step1(fastcam_h5), outname, '1_timestamp_matching.png')


if __name__ == '__main__':
    main()
