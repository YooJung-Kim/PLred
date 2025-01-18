from datetime import datetime
import os, glob
import numpy as np
import re
from astropy.io import fits
from tqdm import tqdm
from .spec import frame_to_spec

def find_data_between(datadir, obs_start, obs_end,
                      header = '', footer = ''):

    '''
    Find data between (obs_start) and (obs_end) times, in format of %H:%M:%S.
    The name of the file should contain the timestamp
    '''

    start = datetime.strptime(obs_start, "%H:%M:%S")
    end = datetime.strptime(obs_end, "%H:%M:%S")

    files = glob.glob(datadir+header+'*'+footer)
    files = sorted(files)

    pattern = r"(\d{2}:\d{2}:\d{2}\.\d+)"

    valid_files = []

    for f in files:

        match = re.search(pattern, f)

        if match:
            obstime = match.group(1)
            obstime = datetime.strptime(obstime[:13], "%H:%M:%S.%f")

            if (obstime > start) and (obstime < end):

                valid_files.append(f)

    print("number of files found: %d" % len(valid_files))
    
    return valid_files

def average_frames(files, verbose = False):
    '''
    Average all the frames of the given files.
    returns averaged frames and number of frames
    '''

    nframes = []
    avg = []
    for file in files:
        nframes.append(fits.getheader(file)['NAXIS3'])
        avg.append(np.sum(fits.getdata(file), axis=0))
    
    if verbose:
        print("number of frames: ", nframes)
    
    nframes = np.sum(nframes)
    avg = np.sum(avg, axis=0).astype(float)

    # nframes = np.sum([fits.getheader(file)['NAXIS3'] for file in files])
    # avg = np.sum([np.sum(fits.getdata(file), axis=0) for file in files], axis=0).astype(float)
    avg /= nframes 

    return avg, nframes

def filter_nans(arr):
    filtered_arr = arr.copy()

    idx = ~np.isfinite(arr)
    for i0 in range(len(idx)):
        if idx[i0]: filtered_arr[i0] = 0
    return filtered_arr

def reduce_couplingmap(couplingmap_file, modelfile, nfib = 38,
                       write_new = True):
    '''
    Extract spectrum from each frame in the couplingmap file

    Parameters
    ----------
    couplingmap_file : str
        path to the couplingmap file
    modelfile : str
        path to the model file
    nfib : int
        number of fibers in the couplingmap file
    write_new : bool
        if True, write the new couplingmap file
        otherwise, extend the file with extensions
    '''
    data = fits.getdata(couplingmap_file)
    header = fits.getheader(couplingmap_file)

    npt = header['NPT']

    model = np.load(modelfile, allow_pickle = True)
    xmin, xmax = model['info'].item()['xmin'], model['info'].item()['xmax']

    cube = np.zeros((nfib, xmax-xmin, npt, npt))

    for i in tqdm(range(npt)):
        for j in range(npt):
            cube[:,:,i,j] = frame_to_spec(data[npt*i+j], xmin, xmax, model['wav_map'], matrix = model['matrix'].item(), return_residual = False)

    sumcube = np.sum(cube, axis = 0)
    normcube = cube / sumcube

    header_cube = fits.Header()
    header_cube['XMIN'] = xmin
    header_cube['XMAX'] = xmax
    header_cube['MODEL'] = modelfile


    hdulist = fits.HDUList()

    if write_new:
        couplingmap_file = couplingmap_file.replace('.fits', '_reduced.fits')
        hdulist.append(fits.PrimaryHDU(header = header))
    else:
        hdulist.append(fits.PrimaryHDU(data = data, header = header))
    hdulist.append(fits.ImageHDU(cube, name = 'cube', header = header_cube))
    hdulist.append(fits.ImageHDU(normcube, name = 'normcube', header = header_cube))
    hdulist.writeto(couplingmap_file, overwrite = True)

    print(f"{couplingmap_file} written")