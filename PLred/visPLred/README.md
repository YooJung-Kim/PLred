# visPLred

Module for FIRST-PL frame calibration and spectral extraction.

## Overview

visPLred handles the critical **Step 2: Spectral Extraction** of the PLred pipeline, extracting spectra from sorted PL frames. This module performs the following tasks:

### Preprocessing & Calibration
1. **Nonlinearity Correction Model Generation ([Tutorial](tutorials/pre1_nonlinearity_correction.ipynb))**
   - **Input:** Calibration data, flat fields of multiple integration times
   - **Output:** Detector nonlinearity correction models

2. **Spectrum Model Creation ([Tutorial](tutorials/pre2_spectrum_model.ipynb))**  
   - **Input:** Neon lamp data, Reference high S/N frame
   - **Output:** Spectrum models

### Spectral Extraction ([Tutorial](tutorials/step2_spectral_extraction.ipynb))
3. **Calibrated Spectrum Extraction**
   - **Input:** Sorted PL frames (`.h5` files, from Step 1), nonlinearity models, spectrum models
   - **Output:** Extracted spectra (`_spec.h5` files)
   - **Key processes:** Dark subtraction, nonlinearity correction, spectral extraction, wavelength calibration


## File Descriptions

### `preprocess.py`
Nonlinearity model generation and correction.

### `spec.py` 
Spectrum model generation and spectral extraction,

### `parameters.py`
Stores fixed parameters for FIRST-PL.

### `utils.py`
Miscellaneous functions.


## Contact

Author: Yoo Jung Kim (yjkim@astro.ucla.edu)