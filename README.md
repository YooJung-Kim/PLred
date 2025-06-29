# PLred

Data reduction and image reconstruction tools for photonic lantern (PL) spectra.

> **Structure:** PLred includes a submodule **`visPLred`** that handles all the calibration and spectral extraction functionality for the SCExAO/FIRST-PL. While PLred provides the general data processing tools including frame sorting and image reconstruction, visPLred contains the FIRST-PL-specific tools.

## Installation

To install, run:

```bash
pip install -e .
```

## Quick Start

Follow the tutorial series to get started:

### Step 1: Frame Sorting ([Tutorial](PLred/tutorials/step1_frame_sorting.ipynb))
**Organizes PL frames based on PSF peak coordinates**
- **Input:** Raw PL frames, Raw PSF frames, timestamps, configuration file (.ini)
- **Output:** Sorted frames data (`.h5` files), Frame sorting info (`_info.json`)
- **Key processes:** Timestamp matching, PSF peak detection, grid definition, frame sorting

### Step 2: Spectral Extraction ([Tutorial](PLred/visPLred/tutorials/step2_spectral_extraction.ipynb))
**Extracts calibrated spectra from sorted PL frames**
- **Input:** Sorted frame data (`.h5` files), nonlinearity models, spectrum models
- **Output:** Extracted spectra (`_spec.h5` files)
- **Key processes:** Dark subtraction, nonlinearity correction, wavelength calibration, spectrum extraction

### Step 3: Image Reconstruction ([Tutorial](PLred/tutorials/step3_image_reconstruction.ipynb))
**Reconstructs response maps from extracted spectra and performs image reconstruction**
- **Input:** Extracted spectra (`_spec.h5` files), Frame sorting info (`_info.json`)
- **Output:** Response maps and polynomial models (`.fits` files),
- **Key processes:** Response map reconstruction, polynomial modeling, centroid analysis, convolution matrix generation, image reconstruction


## Core Modules

#### Main PLred module (`PLred/`)
Frame sorting and image reconstruction

#### visPLred submodule (`PLred/visPLred/`)
FIRST-PL data calibration and spectral extraction

## File Descriptions


#### `sort.py`
Frame sorting and timestamp matching functions for organizing PL data by PSF coordinates.

#### `mapmodel.py`
Contains `CouplingMapModel` class for modeling empirical coupling maps and creating smooth polynomial interpolations.

#### `imgrecon.py`
General image reconstruction algorithms based on MCMC.

#### `fit.py`
Functions for fitting data to models and performing image reconstruction.

#### `imageutils.py`
Utility functions for image processing.
