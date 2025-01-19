# visPLred

## Overview

This module performs the following tasks:

1. **Generates spectrum model from Neon lamp and flat data**
   - **Input:** Neon data, flat data
   - **Output:** Model files

2. **Extracts spectrum from visible PL raw frames using the spectrum model**
   - **Input:** Raw FITS files (data chunk), .ini file
   - **Output:** _spec.fits files (extracted spectrum), info file

3. **Constructs coupling maps from timestamp-matching data**
   - **Input:** .ini file
   - **Output:** Coupling map FITS file



## Usage
Here is an example of how to use visPLred:
```python
import visPLred

# Example code (to be added)
```

## File Descriptions

### `parameters.py`
Stores fixed parameters relevant to visible PL data

### `extract_spec.py`
Generates spectrum model and extracts spectrum

### `couplingmap.py`
Constructs coupling maps

### `sim.py`
Simulation tools for visible PL

### `utils.py`
Miscellaneous utility functions
