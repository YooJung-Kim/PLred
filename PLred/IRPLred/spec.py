import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os, glob
from tqdm import tqdm


def extract_spec(im, ylocs, width=6):
    """
    Extract spectra from a 2D image at multiple Y locations.

    Parameters
    ----------
    im : ndarray, shape (H, W)
        2D image
    ylocs : array-like
        Y-coordinates of aperture centers
    width : int, optional
        Half-width of extraction box (default: 6)

    Returns
    -------
    specs : ndarray, shape (len(ylocs), W)
        Extracted spectra, one per row
    """
    specs = []
    for i in range(len(ylocs)):
        spec = np.sum(im[ylocs[i] - width: ylocs[i] + width, :], axis=0)
        specs.append(spec)
    specs = np.array(specs)
    return specs


def extract_spec_from_frame(im, ylocs, width=6):
    """
    Extract spectra from a single 2D detector image.

    This is a convenience wrapper around extract_spec for single-frame extraction.

    Parameters
    ----------
    im : ndarray, shape (H, W)
        2D detector image
    ylocs : array-like
        Y-coordinates (pixel rows) of fiber apertures
    width : int, optional
        Half-width of extraction box in pixels (default: 6)

    Returns
    -------
    spec : ndarray, shape (len(ylocs), W)
        Extracted spectrum(s). Each row is spectrum from one fiber.
    """
    return extract_spec(im.T, ylocs, width=width)


def extract_all_spectra_from_fits(
    fits_files, ylocs, width=6, dark_frame=None, show_progress=True
):
    """
    Extract spectra from all frames in a list of FITS files.

    Parameters
    ----------
    fits_files : list of str
        Paths to FITS files containing raw PL detector data
    ylocs : array-like
        Y-coordinates of fiber apertures
    width : int, optional
        Half-width of extraction box (default: 6)
    dark_frame : ndarray, optional
        Dark frame to subtract (shape must match image shape)
    show_progress : bool, optional
        Show tqdm progress bar (default: True)

    Returns
    -------
    spectra : ndarray, shape (N_total_frames, N_fibers, N_wavelengths)
        Time-series of extracted spectra
    """
    spectra = []
    desc = "Extracting spectra from FITS files" if show_progress else None
    iterator = tqdm(fits_files, desc=desc) if show_progress else fits_files

    for fits_file in iterator:
        with fits.open(fits_file) as hdul:
            data = hdul[0].data  # shape (N_frames_in_file, H, W)

            for frame_idx in range(data.shape[0]):
                frame = data[frame_idx]

                # Dark subtraction
                if dark_frame is not None:
                    frame = frame - dark_frame

                # Extract spectra
                spec = extract_spec_from_frame(frame, ylocs, width=width)
                spectra.append(spec)

    spectra = np.array(spectra)  # shape (N_total, N_fibers, N_wavelengths)
    return spectra


def extract_spectra_and_save_fits(
    fits_file,
    ylocs,
    width=6,
    dark_frame=None,
    dark_file_name=None,
    output_suffix="_spec.fits",
    show_progress=False,
):
    """
    Extract spectra from a FITS file and save to new FITS file with metadata.

    Preserves original FITS header and adds extraction parameters as new keywords.

    Parameters
    ----------
    fits_file : str
        Path to input FITS file with raw detector data
    ylocs : array-like
        Y-coordinates of fiber apertures
    width : int, optional
        Half-width of extraction box (default: 6)
    dark_frame : ndarray, optional
        Dark frame to subtract (shape must match image shape)
    dark_file_name : str, optional
        Name of dark file used (for header documentation)
    output_suffix : str, optional
        Suffix for output FITS file (default: "_spec.fits")
    show_progress : bool, optional
        Show progress bar (default: False)

    Returns
    -------
    output_file : str
        Path to created output FITS file
    """
    # Load raw data and extract
    with fits.open(fits_file) as hdul_in:
        raw_data = hdul_in[0].data  # shape (N_frames, H, W)
        raw_header = hdul_in[0].header.copy()

        # Extract all spectra
        extracted = extract_all_spectra_from_fits(
            [fits_file], ylocs, width=width, dark_frame=dark_frame, show_progress=show_progress
        )
        # extracted shape: (N_frames, N_fibers, N_lambda)

    # Create output FITS with preserved + enhanced header
    output_file = fits_file.replace(".fits", output_suffix).replace(".gz", "")

    # Create primary HDU with extracted data
    primary_hdu = fits.PrimaryHDU(data=extracted.astype(np.float32), header=raw_header)

    # Add extraction parameters to header
    primary_hdu.header["EXTWIDTH"] = (width, "Half-width of spectral extraction box (pixels)")
    primary_hdu.header["HASDARK"] = (dark_frame is not None, "Dark frame subtracted")
    if dark_file_name:
        primary_hdu.header["DARKFILE"] = (dark_file_name, "Dark frame filename")
    primary_hdu.header["NFIB"] = (len(ylocs), "Number of extracted fibers")
    primary_hdu.header["SPECTLOC"] = (str(list(ylocs)), "Spectrum Y-locations (pixels)")
    primary_hdu.header["EXTMETHOD"] = ("box", "Spectral extraction method")
    primary_hdu.header["ORIGFITS"] = (os.path.basename(fits_file), "Original FITS filename")

    # Create HDU list and write to file
    hdul_out = fits.HDUList([primary_hdu])
    hdul_out.writeto(output_file, overwrite=True)

    return output_file  

def locate_spectra(im, num_spec=3, width=6, plot=True, exclude=[0]):

    im_column_stack = np.mean(im, axis=1)
    for ex in exclude: im_column_stack[ex] = 0
    ylocs = np.zeros(num_spec, dtype=int)

    for i in range(num_spec):
        _yloc = np.argmax(im_column_stack)
        ylocs[i] = int(_yloc)
        im_column_stack[_yloc - width: _yloc + width] = 0
    
    if plot:
        plt.imshow(im)
        for i in range(num_spec): plt.axhspan(ylocs[i] - width, ylocs[i] + width, alpha=0.2, color='white')
        plt.show()

    return ylocs

import h5py
from tqdm import tqdm

def process_h5_files(mapfiles, dark, locs, spec_width=6, vertical = True,
                     nframes_cut = 0, nboot=50, skip_if_exists = True, filter_negative = False):
    
    for mapfile in tqdm(mapfiles):
        
        if not os.path.exists(mapfile):
            print("file %s does not exist, skipping" % mapfile)
            continue
        
        if skip_if_exists and os.path.exists(mapfile.replace('.h5', '_spec.h5')):
            print("file %s already exists, skipping" % mapfile)
            continue

        # Load the map file
        with h5py.File(mapfile, 'r') as f:
            mapdata = f['rawframes'][:]
        
        # mapdata -= dark # subtract dark frame
        
        nframes = mapdata.shape[0]
        if nframes < nframes_cut:
            print("file %s has only %d frames, skipping" % (mapfile, nframes))
            continue
        
        if vertical:
            dim = mapdata.shape[1]
        else:
            dim = mapdata.shape[2]
            
        # create hdf5 file
        outname = mapfile.replace('.h5', '_spec.h5')
        
        with h5py.File(outname, 'w') as h5file:
            
            print("creating %s" % outname)
            h5file.attrs['num_frames'] = nframes
            
            avgspec_h5 = h5file.create_dataset('avgspec', shape = (len(locs), dim))
            bootspec_h5 = h5file.create_dataset('bootspecs', shape = (nboot, len(locs), dim))
            
            # get average
            avgframe = np.mean(mapdata, axis=0)
            darksub = avgframe - dark
            if filter_negative:
                darksub[darksub < 0] = 0
            avgspec = extract_spec((darksub).T, locs, width=spec_width) if vertical else extract_spec((darksub), locs, width=spec_width)
            avgspec_h5[:] = avgspec
            
            # bootstrap
            for i in tqdm(range(nboot)):
                boot_ind = np.random.choice(nframes, nframes, replace=True)
                bootframes = mapdata[boot_ind]
                bootavg = np.mean(bootframes, axis=0)
                darksub = bootavg - dark
                if filter_negative:
                    darksub[darksub < 0] = 0
                bootspec = extract_spec((darksub).T, locs, width=spec_width) if vertical else extract_spec((darksub), locs, width=spec_width)
                bootspec_h5[i] = bootspec
        

    