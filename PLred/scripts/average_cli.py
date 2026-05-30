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


if __name__ == '__main__':
    main()
