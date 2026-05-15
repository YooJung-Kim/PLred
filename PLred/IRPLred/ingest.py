"""
IR Photonic Lantern Mode 2: Time-series spectral extraction and ingest to HDF5.

This module implements the Mode 2 workflow:
1. Extract spectra from raw PL detector FITS files (per frame)
2. Match timestamps with pre-matched PSF camera frames
3. Ingest synchronized spectra + PSF frames into canonical HDF5 format

Usage:
    from IRPLred.ingest import ingest_mode2_to_hdf5
    ingest_mode2_to_hdf5(config_file='config.ini')

Or via CLI:
    python -m IRPLred.ingest --config config.ini --output-name WR140_mode2
"""

import os
import argparse
import numpy as np
import h5py
import pickle
from datetime import datetime, timezone
from astropy.io import fits
from tqdm import tqdm
from configobj import ConfigObj

from ..sort import FrameSorter  # Reuse existing timestamp matching
from .._sort_base import find_data_between, validate_timestamp_matching
from . import parameters
from .spec import (
    extract_spec,
    extract_spec_from_frame,
    extract_all_spectra_from_fits,
    locate_spectra,
)


def ingest_mode2_to_hdf5(
    config_file,
    output_name=None,
    obs_start=None,
    obs_end=None,
    show_progress=True,
    overrides=None,
):
    """
    Mode 2 time-series ingest: extract spectra and match timestamps.

    Parameters
    ----------
    config_file : str
        Path to INI configuration file specifying input data and parameters
    output_name : str, optional
        Override output filename from config (without _mode2.h5 suffix)
    obs_start : str, optional
        Override observation start time from config (HH:MM:SS format)
    obs_end : str, optional
        Override observation end time from config (HH:MM:SS format)
    show_progress : bool, optional
        Show progress bars (default: True)
    overrides : dict, optional
        Additional parameter overrides from config

    Returns
    -------
    output_h5_path : str
        Path to created HDF5 file

    Notes
    -----
    Expected INI sections:
        [Input]
            plcam_dir, plcam_header, plcam_footer
            psf_matched_frames_file, psf_matched_metadata_file
            pl_dark_file (optional)
        [Spectral_Extraction]
            spectrum_locations (auto or list), extraction_width
            apply_dark_subtraction
        [Detector]
            nfib, detector_height, detector_width, plate_scale_mas
        [Metadata]
            file_type, dataset_name, target_name
            pl_backend, tracking_camera
        [Output]
            output_name, output_dir
    """
    overrides = overrides or {}

    # Parse config file
    config = ConfigObj(config_file)

    # Input files
    plcam_dir = config["Input"]["plcam_dir"]
    plcam_header = config["Input"].get("plcam_header", "")
    plcam_footer = config["Input"].get("plcam_footer", ".fits")
    psf_frames_file = config["Input"]["psf_matched_frames_file"]
    psf_metadata_file = config["Input"]["psf_matched_metadata_file"]

    pl_dark_file = config["Input"].get("pl_dark_file", None)
    if pl_dark_file and pl_dark_file.strip() == "":
        pl_dark_file = None

    # Observation times
    obs_start = obs_start or config["Input"]["obs_start"]
    obs_end = obs_end or config["Input"]["obs_end"]

    # Spectral extraction
    spectrum_locations_raw = config["Spectral_Extraction"].get("spectrum_locations", "auto")
    extraction_width = int(config["Spectral_Extraction"].get("extraction_width", 6))
    apply_dark = (
        config["Spectral_Extraction"].get("apply_dark_subtraction", "True").lower()
        == "true"
    )

    # Detector parameters
    nfib = int(config["Detector"].get("nfib", parameters.NFIB))
    detector_height = int(
        config["Detector"].get("detector_height", parameters.DETECTOR_SIZE[0])
    )
    detector_width = int(
        config["Detector"].get("detector_width", parameters.DETECTOR_SIZE[1])
    )
    plate_scale_mas = float(
        config["Detector"].get("plate_scale_mas", parameters.IR_PLATE_SCALE)
    )

    # Metadata
    file_type = config["Metadata"].get("file_type", "onsky-sci")
    dataset_name = config["Metadata"].get("dataset_name", "unknown")
    target_name = config["Metadata"].get("target_name", "unknown")
    pl_backend = config["Metadata"].get("pl_backend", "IR")
    tracking_camera = config["Metadata"].get("tracking_camera", "palila")

    # Output
    output_name = output_name or config["Output"]["output_name"]
    output_dir = config["Output"].get("output_dir", "./data/processed/")
    os.makedirs(output_dir, exist_ok=True)
    output_h5 = os.path.join(output_dir, f"{output_name}_mode2.h5")

    print(f"\n--- Mode 2 Ingest: {dataset_name} ---")
    print(f"Source: {plcam_dir}")
    print(f"Output: {output_h5}")

    # Find PL camera files
    print("Finding PL camera FITS files...")
    plcam_files = find_data_between(
        plcam_dir, obs_start, obs_end, header=plcam_header, footer=plcam_footer
    )

    if len(plcam_files) == 0:
        print("ERROR: No PL camera files found in time window!")
        return None

    # Load dark frame if specified
    dark_frame = None
    if apply_dark and pl_dark_file:
        if os.path.exists(pl_dark_file):
            with fits.open(pl_dark_file) as hdul:
                dark_frame = hdul[0].data
            print(f"Loaded dark frame: {pl_dark_file}")

    # Determine spectrum locations
    if spectrum_locations_raw.lower() == "auto":
        print("Auto-detecting spectrum locations...")
        # Load first frame to auto-detect
        with fits.open(plcam_files[0]) as hdul:
            first_frame = hdul[0].data[0]
        spectrum_locs = locate_spectra(
            first_frame, num_spec=nfib, width=extraction_width, plot=False
        )
    else:
        # Parse list from config
        spectrum_locs = np.array(
            eval(spectrum_locations_raw), dtype=int
        )  # e.g., "[50, 150, 250]"

    print(f"Using {len(spectrum_locs)} spectrum locations: {spectrum_locs}")

    # Get PL detector dimensions from first frame
    with fits.open(plcam_files[0]) as hdul:
        # Assuming data layout: (nframes, height, width)
        first_data = hdul[0].data
        N_frames_total = sum(
            fits.open(f)[0].data.shape[0] for f in plcam_files
        )  # Total frames across all files
        N_lambda = first_data.shape[2]  # Wavelength axis

    print(f"PL detector: {N_frames_total} total frames, {N_lambda} wavelength pixels")

    # Extract all spectra from time-series
    print("Extracting spectra from PL detector frames...")
    intensities = extract_all_spectra_from_fits(
        plcam_files, spectrum_locs, width=extraction_width, dark_frame=dark_frame,
        show_progress=show_progress
    )
    # intensities shape: (N_frames_total, N_fibers, N_lambda)

    print(f"Extracted spectra shape: {intensities.shape}")

    # Load pre-matched PSF camera frames and timestamps
    print("Loading pre-matched PSF camera data...")
    if not os.path.exists(psf_frames_file) or not os.path.exists(psf_metadata_file):
        print(
            f"ERROR: PSF data not found:\n  {psf_frames_file}\n  {psf_metadata_file}"
        )
        print(
            "NOTE: Use PLred.sort.script_match_timestamps() or FrameSorter to pre-match PSF frames."
        )
        return None

    with open(psf_metadata_file, "rb") as f:
        matched_metadata = pickle.load(f)

    psf_frames = np.load(psf_frames_file)  # shape (N_matched, H_psf, W_psf)
    psf_timestamps = matched_metadata[
        "timestamps"
    ]  # shape (N_matched,) - Unix epoch

    N_matched = len(psf_timestamps)
    H_psf, W_psf = psf_frames.shape[1:]

    print(f"Matched {N_matched} frames between PL and PSF cameras")
    print(f"PSF frame shape: ({H_psf}, {W_psf})")

    # Ensure time dimensions match
    if intensities.shape[0] != N_matched:
        print(
            f"WARNING: PL spectra ({intensities.shape[0]}) and PSF frames ({N_matched}) have different frame counts!"
        )
        print(
            "Using minimum count and truncating. Consider re-running timestamp matching."
        )
        N_use = min(intensities.shape[0], N_matched)
        intensities = intensities[:N_use]
        psf_frames = psf_frames[:N_use]
        psf_timestamps = psf_timestamps[:N_use]

    # Create wavelength calibration (placeholder: linear interpolation)
    wavelength_min = float(config["Detector"].get("wavelength_min_um", 1.1))
    wavelength_max = float(config["Detector"].get("wavelength_max_um", 1.8))
    wavelengths_1d = np.linspace(wavelength_min * 1e-6, wavelength_max * 1e-6, N_lambda)
    wavelengths = np.tile(
        wavelengths_1d[np.newaxis, :], (len(spectrum_locs), 1)
    )  # Shape: (N_fibers, N_lambda)

    # Write to HDF5
    print(f"Writing to HDF5...")
    with h5py.File(output_h5, "w") as f:
        # File-level attributes (schema follows ingest_raw_data.py)
        f.attrs["schema_version"] = "1.0"
        f.attrs["file_type"] = file_type
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        f.attrs["PL_backend"] = pl_backend
        f.attrs["tracking_cam"] = tracking_camera
        f.attrs["have_WFS_telem"] = False
        f.attrs["simulation"] = False
        f.attrs["dataset_name"] = dataset_name
        f.attrs["target_name"] = target_name

        # Backend attributes (IR-specific)
        f.attrs["nonlinearity_correction"] = config["Metadata"].get(
            "nonlinearity_correction", "False"
        ).lower() == "true"
        f.attrs["extraction_method"] = "box"

        # Core datasets
        f.create_dataset(
            "intensities",
            data=intensities.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            chunks=(min(100, N_use), len(spectrum_locs), N_lambda),
        )
        f.create_dataset("timestamps", data=psf_timestamps.astype(np.float64))
        f.create_dataset("wavelengths", data=wavelengths.astype(np.float32))

        # Tracking camera group
        group = f.create_group("tracking_camera")
        group.create_dataset(
            "images",
            data=psf_frames.astype(np.uint16),
            compression="gzip",
            compression_opts=4,
            chunks=(min(100, N_use), H_psf, W_psf),
        )
        group.create_dataset("timestamps", data=psf_timestamps.astype(np.float64))

        # Tracking camera attributes
        group.attrs["camera_name"] = tracking_camera
        group.attrs["focal_station"] = "IR"
        group.attrs["plate_scale_mas"] = plate_scale_mas
        group.attrs["detector_height"] = H_psf
        group.attrs["detector_width"] = W_psf

    print(f"Successfully created {output_h5}")
    print(f"  - intensities shape: {intensities.shape}")
    print(f"  - PSF frames shape: {psf_frames.shape}")
    print(f"  - timestamps: {N_use} frames")

    return output_h5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IR PL Mode 2: Time-series spectral extraction and ingest"
    )
    parser.add_argument("--config", required=True, help="Path to configuration INI file")
    parser.add_argument(
        "--output-name", default=None, help="Override output filename from config"
    )
    parser.add_argument(
        "--obs-start", default=None, help="Override observation start time (HH:MM:SS)"
    )
    parser.add_argument(
        "--obs-end", default=None, help="Override observation end time (HH:MM:SS)"
    )
    args = parser.parse_args()

    ingest_mode2_to_hdf5(
        config_file=args.config,
        output_name=args.output_name,
        obs_start=args.obs_start,
        obs_end=args.obs_end,
    )
