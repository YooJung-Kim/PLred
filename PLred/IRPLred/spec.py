import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os, glob


def extract_spec(im, ylocs, width=6):
    specs = []
    for i in range(len(ylocs)):
        spec = np.sum(im[ylocs[i] - width: ylocs[i] + width, :], axis=0)
        specs.append(spec)
    specs = np.array(specs)  
    return specs  

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
        

    