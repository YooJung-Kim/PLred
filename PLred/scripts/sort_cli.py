"""
plred-sort — Step 1: timestamp matching and PSFcam frame averaging.

Usage
-----
    plred-sort obs.ini

Reads [Sort] section (unified config) or falls back to legacy [Output] section.
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
    from PLred.scripts._diagnostics import plot_step1, save_diagnostic, get_plots_dir

    cfg = ConfigObj(args.config)

    # Unified config: [Sort].output is the step1 H5 path
    sort_sec = cfg.get('Sort', {})
    if sort_sec.get('output', '').strip():
        fastcam_h5 = sort_sec['output'].strip()
    else:
        # Legacy [Output] section
        outname  = cfg.get('Output', {}).get('outname', '.').strip() or '.'
        filename = cfg.get('Output', {}).get('filename', 'fastcam').strip() or 'fastcam'
        fastcam_h5 = os.path.join(outname, filename + '.h5')

    from PLred.scripts._diagnostics import plot_pre_ingest_check
    plots_dir = get_plots_dir(args.config)
    save_diagnostic(plot_step1(fastcam_h5), plots_dir, '1_sort.png')
    save_diagnostic(plot_pre_ingest_check(args.config), plots_dir, '1_first_frame.png')


if __name__ == '__main__':
    main()
