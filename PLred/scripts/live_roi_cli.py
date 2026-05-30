"""
plred-live-roi — Live ROI role (PLcam machine).

Polls for .npz match results from the fastcam machine, extracts the PLcam ROI
from the corresponding FITS files, and appends to live_roi.h5 (compatible with
roi_viewer.html).

Usage
-----
    plred-live-roi my_roi.ini
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog='plred-live-roi',
        description='Live pipeline: PLcam machine — ROI extraction and H5 append.',
    )
    parser.add_argument('config', help='Live ROI config .ini file.')
    args = parser.parse_args()

    from PLred.live import run_roi_loop
    run_roi_loop(args.config)


if __name__ == '__main__':
    main()
