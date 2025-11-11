# PLred

Data reduction and image reconstruction tools for photonic lantern (PL) spectra.
Sorts PL frames based on PSF peak coordinates, makes response maps, and reconstructs image.
See [Kim et al. 2025](https://iopscience.iop.org/article/10.3847/2041-8213/ae0739) for details.

> PLred includes a submodule **`visPLred`** that handles all the calibration and spectral extraction functionality for the SCExAO/FIRST-PL. After frame sorting in **`PLred`**, the outputs are used as inputs to **`visPLred`** which extracts spectra from the sorted frames. The extracted spectra are used as inputs to **`PLred`** again to make response maps and further image reconstruction. For any other instruments, their own spectral extraction tools can be used. PLred is intended to be independent of instruments.

## Installation

To install, run:

```bash
pip install git+https://github.com/YooJung-Kim/PLred.git
```

## Quick Start

Follow the tutorial series to get started:

### Step 1: Frame Sorting ([Tutorial](PLred/tutorials/step1_frame_sorting.ipynb))
**Organizes PL frames based on PSF peak coordinates**
- **Input:** Raw PL frames, Raw PSF frames, timestamps, configuration file (.ini)
- **Output:** Sorted frames data (`.h5` files), Frame sorting info (`_info.json`)
- **Key processes:** Timestamp matching, PSF peak detection, grid definition, frame sorting

### Step 2: Spectral Extraction (done in submodule `visPLred`; [Tutorial](PLred/visPLred/tutorials/step2_spectral_extraction.ipynb))
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

## Contact

Author: Yoo Jung Kim (yjkim@astro.ucla.edu)