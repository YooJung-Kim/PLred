# PLred

Data reduction and image reconstruction tools for photonic lantern spectra.

## Overview

1. **Makes smooth coupling map models**
   - **Input:** coupling map FITS
   - **Output:** modeled coupling map FITS

2. **Performs image reconstruction**
   - **Input:** modeled coupling map FITS, .ini file
   - **Output:** to be determined!

3. **Makes 3D model scenes (spatial and spectral) and fits data**

## Installation

To install, run:

```bash
pip install -e .
```

## Usage

## File Descriptions

### `mapmodel.py`
Contains CouplingMapModel class for modeling empirical coupling maps.

### `imgrecon.py`
General image reconstruction codes based on MCMC.

### `fit.py`
Functions for fitting data to models or doing image reconstruction.

### `scene.py`
Generates model scenes.
