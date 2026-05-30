"""
plred-live-sort — Live sort role (fastcam / PSFcam machine).

Polls for new slowcam .txt files, does timestamp matching and PSF centroid
computation, writes lightweight .npz match results, and rsyncs them to the
PLcam machine.

Usage
-----
    plred-live-sort my_sort.ini
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog='plred-live-sort',
        description='Live pipeline: fastcam machine — timestamp match + centroid.',
    )
    parser.add_argument('config', help='Live sort config .ini file.')
    args = parser.parse_args()

    from PLred.live import run_sort_loop
    run_sort_loop(args.config)


if __name__ == '__main__':
    main()
