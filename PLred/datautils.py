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


def bin_by_centroids(palilaframes, firstcamframes, centroids, xbins, ybins):
    '''
    Bin frames by centroids
    '''
    x = centroids[:,0]
    y = centroids[:,1]

    palila_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, palilaframes.shape[1], palilaframes.shape[2]))
    firstcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, firstcamframes.shape[1], firstcamframes.shape[2]))
    num_frames = np.zeros((len(xbins)-1, len(ybins)-1))
    idxs = np.zeros((len(xbins)-1, len(ybins)-1, len(firstcamframes)), dtype=bool)
    
    for i in range(len(xbins)-1):
        for j in range(len(ybins)-1):
            xidx = (x >= xbins[i]) & (x < xbins[i+1])
            yidx = (y >= ybins[j]) & (y < ybins[j+1])
            idx = xidx & yidx
            idxs[i,j] = idx

            palila_binned_frames[i,j] = np.mean(palilaframes[idx], axis=0)
            firstcam_binned_frames[i,j] = np.mean(firstcamframes[idx], axis=0)
            num_frames[i,j] = np.sum(idx)
    return palila_binned_frames, firstcam_binned_frames, num_frames, idxs


def calculate_bootstrap_variance_map(firstcamframes, idxs, nbootstrap = 100):

    '''
    Calculate the bootstrap variance maps

    '''

    arr = np.arange(len(firstcamframes))
    bootstrap_frames = np.zeros((nbootstrap, idxs.shape[0], idxs.shape[1], firstcamframes.shape[1], firstcamframes.shape[2]))
    bootstrap_normframes = np.zeros((nbootstrap, idxs.shape[0], idxs.shape[1], firstcamframes.shape[1], firstcamframes.shape[2]))

    for i in range(idxs.shape[0]):

        for j in range(idxs.shape[1]):

            idx = idxs[i,j]

            for k in range(nbootstrap):
                # resample the indices with replacement
                resampled_idx = np.random.choice(arr[idx], len(arr[idx]), replace = True)
                # store the resampled frames
                bootstrap_frames[k,i,j] = np.nanmean(firstcamframes[resampled_idx], axis = 0)
    
    # calculate normalized bootstrap frames
    for k in range(nbootstrap):
        bootstrap_normframes[k] = bootstrap_frames[k] / np.nansum(bootstrap_frames[k], axis=(0,1))[None,None,:,:]

    # calculate std of bootstrap frames
    var = np.nanvar(bootstrap_frames, axis = 0)

    # calculate normalized boostrap frames
    normvar = np.nanvar(bootstrap_normframes, axis=0)

    return var, normvar


def validate_timestamp_matching(timestamps1, timestamps2):
    '''
    Validate that the timestamps in two lists match
        returns two boolean lists indicating which timestamps match
    '''

    from datetime import datetime

    print("Timestamp1 start: %s, end %s, length %d " % (datetime.fromtimestamp(timestamps1[0]),
                                            datetime.fromtimestamp(timestamps1[-1]),
                                            len(timestamps1)))
    print("Timestamp2 start: %s, end %s, length %d " % (datetime.fromtimestamp(timestamps2[0]),
                                            datetime.fromtimestamp(timestamps2[-1]),
                                            len(timestamps2)))
    

    idx1 = [False] * len(timestamps1)
    idx2 = [False] * len(timestamps2)

    i, j = 0, 0
    while i < len(timestamps1) and j < len(timestamps2):
        if timestamps1[i] == timestamps2[j]:
            idx1[i] = True
            idx2[j] = True
            i += 1
            j += 1
        elif timestamps1[i] < timestamps2[j]:
            i += 1
        else:
            j += 1
    
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    print("Filtered %d out of timestamp1, %d out of timestamp2" % (np.sum(~(idx1)), np.sum(~(idx2))))

    return idx1, idx2
    