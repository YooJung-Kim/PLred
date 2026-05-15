"""
Command-line interface for IRPLred time-series pipeline.

Supports both modular and orchestrated workflows for timestamp matching,
spectral extraction, and final HDF5 ingest.

Usage:
    # Full pipeline (all steps)
    python -m PLred.IRPLred --config config.ini --all

    # Individual steps (modular/checkpoint workflow)
    python -m PLred.IRPLred --config config.ini --match-only
    python -m PLred.IRPLred --config config.ini --extract-only
    python -m PLred.IRPLred --config config.ini --ingest-only

    # With overrides
    python -m PLred.IRPLred --config config.ini --all --dataset-name my_data
    python -m PLred.IRPLred --config config.ini --all --obs-start 12:00:00 --obs-end 13:00:00
"""

import argparse
from .pipeline import (
    match_timestamps_to_h5,
    extract_spectra_to_fits,
    ingest_timeseries_to_hdf5,
    process_complete,
)


def main():
    """CLI entry point for IRPLred time-series pipeline."""
    parser = argparse.ArgumentParser(
        description="IRPLred time-series data reduction pipeline",
        epilog="Example: python -m PLred.IRPLred --config config.ini --all",
    )

    # Main config file
    parser.add_argument("--config", required=True, help="Path to configuration INI file")

    # Workflow mode (mutually exclusive: all OR individual steps)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Run complete pipeline (all 3 steps)",
    )
    mode_group.add_argument(
        "--match-only",
        action="store_true",
        help="Step 1 only: timestamp matching → {dataset}_match.h5",
    )
    mode_group.add_argument(
        "--extract-only",
        action="store_true",
        help="Step 2 only: spectral extraction → {file}_spec.fits",
    )
    mode_group.add_argument(
        "--ingest-only",
        action="store_true",
        help="Step 3 only: final ingest → {dataset}.h5",
    )

    # Optional command-line overrides
    parser.add_argument("--dataset-name", help="Override config dataset_name")
    parser.add_argument("--output-dir", help="Override config output_dir")
    parser.add_argument("--obs-start", help="Override config obs_start (HH:MM:SS format)")
    parser.add_argument("--obs-end", help="Override config obs_end (HH:MM:SS format)")

    # Intermediate file handling
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        default=True,
        help="Keep intermediate _match.h5 and _spec.fits files (default)",
    )
    parser.add_argument(
        "--no-save-intermediates",
        dest="save_intermediates",
        action="store_false",
        help="Delete intermediate files after final ingest",
    )

    # Verbosity
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide progress bars",
    )

    args = parser.parse_args()

    show_progress = not args.no_progress

    # Execute requested workflow
    try:
        if args.all:
            print("\n🔄 Running complete pipeline (Steps 1-3)...\n")
            process_complete(
                args.config,
                save_intermediates=args.save_intermediates,
                show_progress=show_progress,
            )
        elif args.match_only:
            print("\n🔄 Running Step 1: Timestamp Matching...\n")
            match_timestamps_to_h5(args.config, show_progress=show_progress)
        elif args.extract_only:
            print("\n🔄 Running Step 2: Spectral Extraction...\n")
            extract_spectra_to_fits(args.config, show_progress=show_progress)
        elif args.ingest_only:
            print("\n🔄 Running Step 3: Final Ingest...\n")
            ingest_timeseries_to_hdf5(args.config, show_progress=show_progress)

        print("\n✓ Pipeline completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

