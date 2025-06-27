from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys, glob
import h5py

from PLred.visPLred import spec as sp
from PLred.visPLred import utils as du
import PLred.visPLred.preprocess as pp

import json

if __name__ == "__main__":

    datadir = '/mnt/datazpool/PL/yjkim/remapped/mizarA_20250514/grid_binned_08:31:00_08:36:59/'
    # datadir = '/mnt/datazpool/PL/yjkim/remapped/mizarB2_20250514/grid_ini_08:38:33_08:55:52/'
    info = json.load(open(datadir + 'remapped_info.json', 'r'))
    map_n = info['map_n']
    map_width = info['map_width']

    pos_mas = np.linspace(-map_width/2, map_width/2, map_n) * 16.2

    # specfiles = [datadir+'remapped_bin_%d_%d_spec.h5' % (i, j) for i in np.arange(0, map_n) for j in np.arange(0, map_n)]

    nboot = 50

    # Load the spectra
    specs = np.zeros((map_n, map_n, 38, 700))
    bootspecs = np.zeros((nboot, map_n, map_n, 38, 700))
    nframes = np.zeros((map_n, map_n))

    print("Reading spectra")
    for i in tqdm(range(map_n)):
        for j in range(map_n):
            mapfile = datadir + 'remapped_bin_%d_%d_spec.h5' % (i, j)
            if os.path.exists(mapfile):
                with h5py.File(mapfile, 'r') as f:
                    specs[i, j] = f['avgspec'][:]
                    bootspecs[:, i, j] = f['bootspecs'][:]
                    nframes[i, j] = f.attrs['num_frames']
            else:
                # If the file does not exist, fill with NaNs
                specs[i, j] = np.nan * np.ones((38, 700))
                bootspecs[:, i, j] = np.nan * np.ones((nboot, 38, 700))
    
    print("Reading PSF frames")
    psfframes = np.zeros((map_n, map_n, 40, 40))
    for i in tqdm(range(map_n)):
        for j in range(map_n):
            mapfile = datadir + 'remapped_bin_%d_%d.h5' % (i, j)
            if os.path.exists(mapfile):
                with h5py.File(mapfile, 'r') as f:
                    psfframes[i, j] = np.nanmean(f['psfframes'][:], axis=0)
            else:
                # If the file does not exist, fill with NaNs
                psfframes[i, j] = np.nan * np.ones((40, 40))

    # Normalize over the map
    normspecs = specs / np.nansum(specs, axis=(0, 1))  # normalize over the clusters
    normbootspecs = bootspecs / np.nansum(bootspecs, axis=(1, 2))[:,None,None,:,:]  # normalize over the clusters
    
    # Calculate variance
    specs_var = np.nanvar(bootspecs, axis=0)
    normspecs_var = np.nanvar(normbootspecs, axis=0)

    # save the map
    header = fits.Header()
    header['MAP_N'] = map_n
    header['MAP_W'] = map_width
    header['XMIN'] = min(pos_mas)
    header['XMAX'] = max(pos_mas)
    header['YMIN'] = min(pos_mas)
    header['YMAX'] = max(pos_mas)


    hdu = fits.PrimaryHDU(specs, header=header)
    hdu2 = fits.ImageHDU(nframes, name = 'nframes')
    hdu3 = fits.ImageHDU(psfframes, name = 'psfcam')
    hdu4 = fits.ImageHDU(specs_var, name = 'var')
    hdu5 = fits.ImageHDU(normspecs_var, name = 'normvar')
    hdu6 = fits.ImageHDU(normspecs, name = 'normspec')
    # hdu7 = fits.BinTableHDU.from_columns(cols, name='cluster_centers')
    
    hdul = fits.HDUList([hdu, hdu2, hdu3, hdu4, hdu5, hdu6])
    hdul.writeto(datadir + 'remapped_couplingmap.fits', overwrite=True)
    print("remapped_couplingmap.fits saved in %s" % datadir)