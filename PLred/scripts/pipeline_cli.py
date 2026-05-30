"""
plred-run — master pipeline runner for PLred Mode 1.

Runs all five steps of the spatial response-map pipeline in sequence,
or a subset of steps specified by --steps or --from.

Usage
-----
    plred-run obs.ini                  # run all steps 1–5
    plred-run obs.ini --from 4         # re-run from step 4
    plred-run obs.ini --steps 4,5      # run only steps 4 and 5
    plred-run obs.ini --steps 2        # run only step 2
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog='plred-run',
        description='Run the PLred Mode 1 pipeline (steps 1–5) from a config file.',
    )
    parser.add_argument('config', help='Pipeline config .ini file.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--from', dest='from_step', type=int, default=None,
                       metavar='STEP',
                       help='Start pipeline from this step (e.g. --from 4).')
    group.add_argument('--steps', default=None,
                       help='Comma-separated list of steps to run (e.g. --steps 4,5).')

    args = parser.parse_args()

    if args.from_step is not None:
        steps = list(range(args.from_step, 6))
    elif args.steps is not None:
        steps = [int(s.strip()) for s in args.steps.split(',') if s.strip()]
    else:
        steps = None  # all

    from PLred.pipeline import run_mode1
    run_mode1(args.config, steps=steps)


if __name__ == '__main__':
    main()
