from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys, glob
import h5py

from PLred.visPLred import spec as sp
from PLred.visPLred import utils as du
import PLred.visPLred.preprocess as pp


from configobj import ConfigObj
import yaml

from scipy.sparse import load_npz
import json

from queue import Queue
import multiprocessing

if __name__ == "__main__":

    configname = sys.argv[1]

    # Parse config

    config = ConfigObj(configname)
    name = config['Inputs']['name']
    binning_method = config['Coupling_map']['binning_method']

    remapped_path = config['Coupling_map']['output_dir'] + name + '/' + config['Coupling_map']['output_name'] + '_'+config['Coupling_map']['map_start_time'] + '_' + config['Coupling_map']['map_end_time'] + '/'
    info = json.load(open(remapped_path + 'remapped_info.json', 'r'))#['n_clusters']
    
    if binning_method == 'voronoi':
        n_clusters = info['n_clusters']
        map_width = info['map_width']
        mapfiles = [remapped_path+'remapped_voronoi_bin_%d.h5' % i for i in np.arange(0, n_clusters)]
        specfiles = [remapped_path+'remapped_voronoi_bin_%d_spec.h5' % i for i in np.arange(0, n_clusters)]


        # Load the info
        xbins, ybins = [], []
        nframes = []
        psf_images = []
        for mapfile in mapfiles:
            with h5py.File(mapfile, 'r') as f:
                xbins.append(f.attrs['xbin'])
                ybins.append(f.attrs['ybin'])
                nframes.append(f.attrs['num_frames'])
                psf_images.append(np.nanmean(f['psfframes'][:], axis=0))
        xbins = np.array(xbins)
        ybins = np.array(ybins)
        nframes = np.array(nframes)
        psf_images = np.array(psf_images)

        col1 = fits.Column(name='xbin', array=xbins, format='D')
        col2 = fits.Column(name='ybin', array=ybins, format='D')
        cols = fits.ColDefs([col1, col2])

    # elif binning_method == 'grid':
    #     map_n = info['map_n']
    #     map_width = info['map_width']
    #     mapfiles = [remapped_path+'remapped_bin_%d_%d.h5' % (i, j) for i in np.arange(0, map_n) for j in np.arange(0, map_n)]
    #     specfiles = [remapped_path+'remapped_bin_%d_%d_spec.h5' % (i, j) for i in np.arange(0, map_n) for j in np.arange(0, map_n)]

    #     # Load the info
    #     nframes = []
    #     for mapfile in mapfiles:

    # Load the spectra
    specs = []
    bootspecs = []
    for n, mapfile in zip(nframes, specfiles):

        if n >= 5:
                
            with h5py.File(mapfile, 'r') as f:
                specs.append(f['avgspec'][:]) # shape: (nfib x nwav)
                bootspecs.append(f['bootspecs'][:])  # shape: (nboot x nfib x nwav)

        else:
            # If there are not enough frames, we fill with NaNs
            with h5py.File(mapfile, 'r') as f:
                specs.append(f['avgspec'][:] * np.nan) # shape: (nfib x nwav)
                bootspecs.append(f['bootspecs'][:] * np.nan)  # shape: (nboot x nfib x nwav)

    specs = np.array(specs) # shape: (n_clusters, nfib, nwav)
    bootspecs = np.array(bootspecs) # shape: (n_clusters, nboot, nfib, nwav)

    # normalize over the map
    normspecs = specs / np.nansum(specs, axis=0) # normalize over the clusters
    normbootspecs = bootspecs / np.nansum(bootspecs, axis=0) # normalize over the clusters

    # calculate variance
    specs_var = np.nanvar(bootspecs, axis=1)
    normspecs_var = np.nanvar(normbootspecs, axis=1)

    # save the map
    header = fits.Header()
    header['MAP_W'] = map_width
    header['NCLUSTER'] = n_clusters
    header['VORONOI'] = remapped_path + 'voronoi.pkl'

    hdu = fits.PrimaryHDU(specs, header=header)
    hdu2 = fits.ImageHDU(nframes, name = 'nframes')
    hdu3 = fits.ImageHDU(psf_images, name = 'psfcam')
    hdu4 = fits.ImageHDU(specs_var, name = 'var')
    hdu5 = fits.ImageHDU(normspecs_var, name = 'normvar')
    hdu6 = fits.ImageHDU(normspecs, name = 'normspec')
    hdu7 = fits.BinTableHDU.from_columns(cols, name='cluster_centers')
    
    hdul = fits.HDUList([hdu, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7])
    hdul.writeto(remapped_path + 'remapped_voronoi_couplingmap.fits', overwrite=True)
    print("remapped_voronoi_couplingmap.fits saved in %s" % remapped_path)