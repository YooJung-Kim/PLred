
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import glob
import re

from .parameters import telescope_params, firstcam_params
from .spec import frame_to_spec

diameter = telescope_params['diameter']
NFIB = firstcam_params['NFIB']
zaber_microns = firstcam_params['zaber_microns']

def find_data_between(datadir, obs_start, obs_end,
                      header = '', footer = ''):

    '''
    Find data between (obs_start) and (obs_end) times, in format of %H:%M:%S.
    The name of the file should contain the timestamp

    Parameters
    ----------
    datadir : str
        path to the directory containing the data files
    obs_start : str (%H:%M:%S)
        start time of the observation
    obs_end : str (%H:%M:%S)
        end time of the observation
    header : str
        prefix of the data files
    footer : str
        suffix of the data files

    Returns
    -------
    valid_files : list
        list of files that are between the start and end times

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
    '''
    Replace NaNs with 0 in the array
    '''
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



def plot_coupling_maps(couplingmap_file, outname, norm = False, fnumber = 8,
                       specinds = None):
    
    '''
    Plot the coupling maps from a coupling map file.
    
    Parameters
    ----------
    couplingmap_file : str
        Path to the coupling map file.
    outname : str
        Path to the output file.
    norm : bool
        If True, use the normalized coupling maps.
    fnumber : float
        Focal ratio of the telescope.
    specinds : list
        List of spectral indices to average for plot. If None, sum over all spectral indices.
    '''

    if norm:
        cube = fits.open(couplingmap_file)[2].data
    else:
        cube = fits.open(couplingmap_file)[1].data

    cubeheader = fits.open(couplingmap_file)[0].header

    window_step = cubeheader['WINDOW']
    npt = cubeheader['NPT']

    x_pos = np.linspace(0 - window_step/2, 0 + window_step/2, npt) * zaber_microns
    y_pos = np.linspace(0 - window_step/2, 0 + window_step/2, npt) * zaber_microns

    plate_scale = 206265 / (fnumber * diameter * 1e6) * 1e3 # mas per micron

    x_pos_mas = x_pos * plate_scale
    y_pos_mas = y_pos * plate_scale

    if specinds is None:
        specinds = np.arange(np.shape(cube)[1])
    
    cube_sliced = np.average(cube[:, specinds, :, :], axis=1)
    
    fig, axs = plt.subplots(ncols=5, nrows=8, figsize=(10,16), sharex=True, sharey=True)
    axs = axs.flatten()


    for fibind in range(NFIB):
        axs[fibind].imshow(cube_sliced[fibind, :, :], origin='lower',
                        extent = (min(x_pos_mas), max(x_pos_mas), min(y_pos_mas), max(y_pos_mas)))
        axs[fibind].set_title('Fiber {}'.format(fibind))
        if fibind // 5 == 7: axs[fibind].set_xlabel('x (mas)')
        if fibind % 5 == 0: axs[fibind].set_ylabel('y (mas)')
    axs[39].imshow(np.sum(cube_sliced[:, :, :], axis=0), origin='lower',
                extent = (min(x_pos_mas), max(x_pos_mas), min(y_pos_mas), max(y_pos_mas)))
    axs[39].set_title('Sum')
    axs[38].axis('off')

    axs[39].set_xlabel('x (mas)')
    axs[39].set_ylabel('y (mas)')

    fig.savefig(outname+'.png')


def normalize_flatten(cube):

    if len(np.shape(cube)) == 2:
        cube = np.array([cube])
    
    sumspec = np.sum(cube, axis=1)

    flattened_cube = []

    for frameind in range(len(cube)):

        flattened_cube.append(np.array([cube[frameind][fibind] / sumspec[frameind] for fibind in range(38)]).flatten())

    return np.array(flattened_cube), sumspec


from sklearn.decomposition import PCA
import time


def normspec_PCA(datadir, obs_start, obs_end,
                 clip_sigma = 5, n_components = 50):
    
    specfiles = find_data_between(datadir, obs_start, obs_end,
                                  footer = '_spec.fits')

    all_data = []
    
    all_sumspec = []
    for i in tqdm(np.arange(len(specfiles))):
        cube = fits.getdata(specfiles[i])

        normcube, sumspec = normalize_flatten(cube)
        all_data.append(normcube)
        all_sumspec.append(np.sum(sumspec, axis=1))

        if i == 0:
            spec = np.sum(sumspec, axis=0)
        else:
            spec += np.sum(sumspec, axis=0)

    # clip
    all_sumspec = np.array(all_sumspec).flatten()
    # all_sumspec = all_sumspec.reshape(-1, all_sumspec.shape[-1])
    
    all_data = np.array(all_data)
    all_data = all_data.reshape(-1, all_data.shape[-1])

    meanspec = np.mean(all_sumspec, axis=0)
    stdspec = np.std(all_sumspec, axis=0)

    idx = (all_sumspec < meanspec + clip_sigma * stdspec) & (all_sumspec > meanspec - clip_sigma * stdspec)
    all_data = all_data[idx]

    print('number of frames:', len(all_data))

    start = time.time()
    pca = PCA(n_components = n_components)
    pca.fit_transform(all_data)
    end = time.time()
    print('time elapsed: %.2f seconds' % (end - start))

    return pca, spec
