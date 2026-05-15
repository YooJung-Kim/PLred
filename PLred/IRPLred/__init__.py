# IRPLred - Simplified IR variant of PLred spectral extraction
# Time-series pipeline with modular and orchestrated workflows

from .ingest import ingest_mode2_to_hdf5, ingest_timeseries_to_hdf5
from .pipeline import (
    match_timestamps_to_h5,
    extract_spectra_to_fits,
    process_complete,
)
from .spec import extract_spec, extract_spec_from_frame, extract_all_spectra_from_fits, locate_spectra
from . import parameters

__all__ = [
    # Pipeline functions (3-step + orchestrator)
    "match_timestamps_to_h5",
    "extract_spectra_to_fits",
    "ingest_timeseries_to_hdf5",
    "process_complete",
    # Legacy names (for backward compatibility during transition)
    "ingest_mode2_to_hdf5",  # Use ingest_timeseries_to_hdf5 instead
    # Spectral extraction utilities
    "extract_spec",
    "extract_spec_from_frame",
    "extract_all_spectra_from_fits",
    "locate_spectra",
    # Configuration
    "parameters",
]

