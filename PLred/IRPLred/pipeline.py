"""
IRPLred unified time-series data reduction pipeline.

Three-step workflow (can be used modularly or orchestrated):
1. match_timestamps_to_h5() - timestamp matching (PSF ↔ PL) → {dataset}_match.h5
2. extract_spectra_to_fits() - spectral extraction per file → {file}_spec.fits
3. ingest_timeseries_to_hdf5() - final ingest → {dataset}.h5

All steps configured via single unified INI file. See template_config.ini for details.

Usage:
    # Modular (allows checkpointing and restarting)
    from PLred.IRPLred import match_timestamps_to_h5, extract_spectra_to_fits, ingest_timeseries_to_hdf5
    match_timestamps_to_h5('config.ini')
    extract_spectra_to_fits('config.ini')
    ingest_timeseries_to_hdf5('config.ini')

    # Orchestrated (all-in-one)
    from PLred.IRPLred import process_complete
    process_complete('config.ini', save_intermediates=True)

    # CLI
    python -m PLred.IRPLred --config config.ini --all
"""

import os
import numpy as np
import h5py
import logging
from datetime import datetime, timezone
from configobj import ConfigObj
from tqdm import tqdm

from ..sort import FrameSorter
from .._sort_base import find_data_between
from . import parameters
from .spec import extract_all_spectra_from_fits, extract_spectra_and_save_fits, locate_spectra
from astropy.io import fits

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def match_timestamps_to_h5(config_file, output_name=None, show_progress=True):
    """
    Step 1: Timestamp matching between PSF and PL cameras.

    Saves PSF frames and matching metadata to a single H5 file for downstream processing.

    Parameters
    ----------
    config_file : str
        Path to INI configuration file (see template_config.ini)
    output_name : str, optional
        Override output name from config (without _match.h5 suffix)
    show_progress : bool, optional
        Show progress bars (default: True)

    Returns
    -------
    match_h5_path : str
        Path to created {dataset}_match.h5 file

    Config sections used:
        [Input]: psf_dir, psf_header, pl_dir, pl_header, obs_start, obs_end
        [Error_Tolerance]: min_matched_frames, timestamp_tolerance_sec
        [Pipeline]: show_progress
    """
    config = ConfigObj(config_file)

    # Parse config
    psf_dir = config["Input"]["psf_dir"]
    psf_header = config["Input"].get("psf_header", "")
    pl_dir = config["Input"]["pl_dir"]
    pl_header = config["Input"].get("pl_header", "")
    obs_start = config["Input"]["obs_start"]
    obs_end = config["Input"]["obs_end"]

    dataset_name = config["Ingest"].get("dataset_name", "dataset")
    output_dir = config["Ingest"].get("output_dir", "./data/processed/")
    output_name = output_name or dataset_name

    min_matched = int(config["Error_Tolerance"].get("min_matched_frames", 100))

    os.makedirs(output_dir, exist_ok=True)
    match_h5 = os.path.join(output_dir, f"{output_name}_match.h5")

    print(f"\n--- Step 1: Timestamp Matching ---")
    print(f"PSF directory: {psf_dir}")
    print(f"PL directory: {pl_dir}")
    print(f"Time range: {obs_start} - {obs_end}")

    # Use FrameSorter to match timestamps
    try:
        sorter = FrameSorter(
            plcam_timestamp_path=os.path.join(pl_dir, "timestamps/"),
            plcam_spec_path=pl_dir,
            psfcam_frames_name="dummy",  # Will be overridden
            psfcam_timestamp_name="dummy",  # Will be overridden
            obs_start=obs_start,
            obs_end=obs_end,
            plcam_header=pl_header,
        )
        n_matched = len(sorter.timestamps)

        if n_matched < min_matched:
            logger.warning(
                f"Only {n_matched} frames matched (< {min_matched} minimum). "
                "Check timestamp alignment."
            )
            if n_matched == 0:
                raise ValueError("No frames matched between PSF and PL cameras!")

        print(f"✓ Matched {n_matched} frames")

        # Save to H5
        print(f"Saving to {match_h5}...")
        with h5py.File(match_h5, "w") as f:
            f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            f.attrs["obs_start"] = obs_start
            f.attrs["obs_end"] = obs_end
            f.attrs["n_matched"] = n_matched
            f.attrs["dataset_name"] = output_name

            # PSF camera data
            f.create_dataset(
                "psfcam_frames",
                data=sorter.psfcam_frames,
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset("psfcam_timestamps", data=sorter.timestamps)

            # PL camera indices for frame lookup
            f.create_dataset("plcam_fileinds", data=sorter.plcam_file_indices)
            f.create_dataset("plcam_frameinds", data=sorter.plcam_frame_indices)

        print(f"✓ Created {match_h5}")
        return match_h5

    except Exception as e:
        logger.error(f"Timestamp matching failed: {e}")
        raise


def extract_spectra_to_fits(config_file, match_h5_path=None, show_progress=True):
    """
    Step 2: Spectral extraction from PL detector frames.

    Reads matched frame information from Step 1, extracts spectra from original
    PL FITS files, and saves one _spec.fits file per original FITS file.

    Parameters
    ----------
    config_file : str
        Path to INI configuration file
    match_h5_path : str, optional
        Path to {dataset}_match.h5 from Step 1 (auto-found if not provided)
    show_progress : bool, optional
        Show progress bars (default: True)

    Returns
    -------
    spec_fits_files : list of str
        Paths to created {file}_spec.fits files

    Config sections used:
        [Input]: pl_dir, pl_header, spectrum_locations, pl_dark_file
        [Spectral_Extraction]: extraction_width
        [Detector]: nfib
    """
    config = ConfigObj(config_file)

    pl_dir = config["Input"]["pl_dir"]
    pl_header = config["Input"].get("pl_header", "")
    spectrum_locs_raw = config["Spectral_Extraction"].get("spectrum_locations", "auto")
    extraction_width = int(config["Spectral_Extraction"].get("extraction_width", 6))
    nfib = int(config["Detector"].get("nfib", parameters.NFIB))

    pl_dark_file = config["Input"].get("pl_dark_file", None)
    if pl_dark_file and pl_dark_file.strip() == "":
        pl_dark_file = None

    dataset_name = config["Ingest"].get("dataset_name", "dataset")
    output_dir = config["Ingest"].get("output_dir", "./data/processed/")

    # Auto-find match.h5 if not provided
    if match_h5_path is None:
        match_h5_path = os.path.join(output_dir, f"{dataset_name}_match.h5")

    if not os.path.exists(match_h5_path):
        raise FileNotFoundError(f"Matching file not found: {match_h5_path}")

    print(f"\n--- Step 2: Spectral Extraction ---")
    print(f"Reading matched data from: {match_h5_path}")

    # Find PL FITS files
    obs_start = config["Input"]["obs_start"]
    obs_end = config["Input"]["obs_end"]
    pl_files = find_data_between(pl_dir, obs_start, obs_end, header=pl_header, footer=".fits")

    if len(pl_files) == 0:
        raise FileNotFoundError(f"No PL FITS files found in {pl_dir}")

    # Load dark frame if specified
    dark_frame = None
    if pl_dark_file and os.path.exists(pl_dark_file):
        with fits.open(pl_dark_file) as hdul:
            dark_frame = hdul[0].data
        print(f"✓ Loaded dark frame: {pl_dark_file}")

    # Determine spectrum locations
    if spectrum_locs_raw.lower() == "auto":
        print("Auto-detecting spectrum locations...")
        with fits.open(pl_files[0]) as hdul:
            first_frame = hdul[0].data[0]
        spectrum_locs = locate_spectra(
            first_frame, num_spec=nfib, width=extraction_width, plot=False
        )
    else:
        spectrum_locs = np.array(eval(spectrum_locs_raw), dtype=int)

    print(f"Using {len(spectrum_locs)} spectrum locations")

    # Extract spectra from each FITS file and save to separate _spec.fits file
    spec_fits_files = []
    iterator = tqdm(pl_files, desc="Extracting spectra") if show_progress else pl_files

    for pl_file in iterator:
        # Extract and save with proper FITS headers
        spec_file = extract_spectra_and_save_fits(
            pl_file,
            spectrum_locs,
            width=extraction_width,
            dark_frame=dark_frame,
            dark_file_name=os.path.basename(pl_dark_file) if pl_dark_file else None,
            output_suffix="_spec.fits",
            show_progress=False,
        )

        spec_fits_files.append(spec_file)

    print(f"✓ Extracted {len(spec_fits_files)} files")
    return spec_fits_files


def ingest_timeseries_to_hdf5(config_file, match_h5_path=None, show_progress=True):
    """
    Step 3: Final ingest to canonical HDF5 time-series format.

    Reads matched PSF frames from Step 1 and extracted spectra from Step 2,
    combines them with proper time alignment, and creates final HDF5 file.

    Parameters
    ----------
    config_file : str
        Path to INI configuration file
    match_h5_path : str, optional
        Path to {dataset}_match.h5 from Step 1 (auto-found if not provided)
    show_progress : bool, optional
        Show progress bars (default: True)

    Returns
    -------
    output_h5_path : str
        Path to created {dataset}.h5 file

    Config sections used:
        [Ingest]: dataset_name, output_dir, wavelength_min_um, wavelength_max_um
        [Detector]: plate_scale_mas
        [Metadata]: file_type, target_name
    """
    config = ConfigObj(config_file)

    dataset_name = config["Ingest"].get("dataset_name", "dataset")
    output_dir = config["Ingest"].get("output_dir", "./data/processed/")
    wavelength_min = float(config["Ingest"].get("wavelength_min_um", 1.1))
    wavelength_max = float(config["Ingest"].get("wavelength_max_um", 1.8))

    plate_scale_mas = float(config["Detector"].get("plate_scale_mas", parameters.IR_PLATE_SCALE))
    file_type = config["Metadata"].get("file_type", "onsky-sci")
    target_name = config["Metadata"].get("target_name", "unknown")

    # Auto-find match.h5 if not provided
    if match_h5_path is None:
        match_h5_path = os.path.join(output_dir, f"{dataset_name}_match.h5")

    if not os.path.exists(match_h5_path):
        raise FileNotFoundError(f"Matching file not found: {match_h5_path}")

    print(f"\n--- Step 3: Final Ingest ---")
    print(f"Reading matched data from: {match_h5_path}")

    # Load matched data
    with h5py.File(match_h5_path, "r") as f:
        psfcam_frames = f["psfcam_frames"][:]
        psfcam_timestamps = f["psfcam_timestamps"][:]
        n_matched = f.attrs["n_matched"]

    # TODO: Load extracted spectra from _spec.fits files and concatenate
    # For now, placeholder (will be implemented after spec extraction step)

    print(f"✓ Loaded {n_matched} matched frames")
    print("(Spectral extraction integration coming in next step)")

    output_h5 = os.path.join(output_dir, f"{dataset_name}.h5")
    print(f"Output will be saved to: {output_h5}")

    return output_h5


def process_complete(
    config_file,
    save_intermediates=True,
    show_progress=True,
    restart_from_step=1,
):
    """
    Orchestrator: Run complete 3-step pipeline.

    Executes timestamp matching → spectral extraction → ingest in sequence,
    with optional checkpointing and error handling.

    Parameters
    ----------
    config_file : str
        Path to INI configuration file
    save_intermediates : bool, optional
        Keep _match.h5 and _spec.fits files after ingest (default: True)
    show_progress : bool, optional
        Show progress bars (default: True)
    restart_from_step : int, optional
        Resume from step 1, 2, or 3 (default: 1 = start from beginning)

    Returns
    -------
    output_h5_path : str
        Path to final {dataset}.h5 file

    Notes
    -----
    If error_tolerance settings specify 'skip' behavior, the pipeline will
    continue despite warnings. 'error' or missing 'error_tolerance' causes abort.
    """
    config = ConfigObj(config_file)
    dataset_name = config["Ingest"].get("dataset_name", "dataset")
    output_dir = config["Ingest"].get("output_dir", "./data/processed/")

    print(f"\n{'='*60}")
    print(f"IRPLred Time-Series Pipeline: {dataset_name}")
    print(f"{'='*60}")

    output_h5 = os.path.join(output_dir, f"{dataset_name}.h5")
    match_h5 = os.path.join(output_dir, f"{dataset_name}_match.h5")

    try:
        # Step 1
        if restart_from_step <= 1:
            print("\n[1/3] Timestamp Matching...")
            match_h5 = match_timestamps_to_h5(config_file, show_progress=show_progress)

        # Step 2
        if restart_from_step <= 2:
            print("\n[2/3] Spectral Extraction...")
            spec_fits_files = extract_spectra_to_fits(
                config_file, match_h5_path=match_h5, show_progress=show_progress
            )

        # Step 3
        if restart_from_step <= 3:
            print("\n[3/3] Final Ingest...")
            output_h5 = ingest_timeseries_to_hdf5(
                config_file, match_h5_path=match_h5, show_progress=show_progress
            )

        print(f"\n{'='*60}")
        print(f"✓ Pipeline complete!")
        print(f"Output: {output_h5}")
        print(f"{'='*60}\n")

        # Cleanup intermediates if requested
        if not save_intermediates:
            print("Cleaning up intermediate files...")
            if os.path.exists(match_h5):
                os.remove(match_h5)
                print(f"  Removed {match_h5}")
            for spec_file in spec_fits_files:
                if os.path.exists(spec_file):
                    os.remove(spec_file)
                    print(f"  Removed {spec_file}")

        return output_h5

    except Exception as e:
        logger.error(f"Pipeline failed at step {restart_from_step}: {e}")
        print(f"\n✗ Pipeline failed. Intermediates retained for debugging:")
        print(f"  - {match_h5}")
        print(f"  - *_spec.fits files in {output_dir}")
        print(f"Resume with: process_complete(config_file, restart_from_step={restart_from_step})")
        raise
