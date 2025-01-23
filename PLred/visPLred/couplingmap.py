# functions related to constructing coupling maps

import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
from .utils import find_data_between
from ..imageutils import subpixel_centroid_2d
from .parameters import *

def bin_by_centroids(psfcamframes, firstcamframes, centroids, xbins, ybins):
    '''
    Bin frames by centroids
    '''
    x = centroids[:,0]
    y = centroids[:,1]

    psfcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, psfcamframes.shape[1], psfcamframes.shape[2]))
    firstcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, firstcamframes.shape[1], firstcamframes.shape[2]))
    num_frames = np.zeros((len(xbins)-1, len(ybins)-1))
    idxs = np.zeros((len(xbins)-1, len(ybins)-1, len(firstcamframes)), dtype=bool)
    
    for i in range(len(xbins)-1):
        for j in range(len(ybins)-1):
            xidx = (x >= xbins[i]) & (x < xbins[i+1])
            yidx = (y >= ybins[j]) & (y < ybins[j+1])
            idx = xidx & yidx
            idxs[i,j] = idx

            psfcam_binned_frames[i,j] = np.mean(psfcamframes[idx], axis=0)
            firstcam_binned_frames[i,j] = np.mean(firstcamframes[idx], axis=0)
            num_frames[i,j] = np.sum(idx)
    return psfcam_binned_frames, firstcam_binned_frames, num_frames, idxs


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


class SimultaneousData:
    '''
    Class to handle simultaneous data from two cameras
    '''
    def __init__(self, firstcam_timestamp_path, firstcam_spec_path,
                 obs_start, obs_end,
                 psfcam, 
                 psfcam_frames_name, psfcam_timestamp_name,
                 match_frames = True):
        
        '''
        Initialize the class

        Parameters
        ----------
        firstcam_timestamp_path: str
            path to the first camera timestamp files
        firstcam_spec_path: str
            path to the first camera spectrum files
        obs_start: str (HH:MM:SS)
            start time of the observation
        obs_end: str (HH:MM:SS)
            end time of the observation
        psfcam: str
            name of the psfcam (palila or vcam)
        psfcam_frames_name: str
            name of the psfcam frames file
        psfcam_timestamp_name: str
            name of the psfcam timestamp file
        match_frames: bool
            whether to match the frames    
        '''
        

        self.firstcam_timestamp_path = firstcam_timestamp_path
        self.firstcam_spec_path = firstcam_spec_path
        self.obs_start = obs_start
        self.obs_end = obs_end
        self.psfcam_frames_name = psfcam_frames_name
        self.psfcam_timestamp_name = psfcam_timestamp_name

        if psfcam == 'palila':
            self.pix2mas = palila_params['plate_scale']
        elif psfcam == 'vcam':
            self.pix2mas = vampires_params['plate_scale']
        else:
            raise ValueError("psfcam should be either palila or vcam")
        self.psfcam_name = psfcam

        if match_frames:
            self.match_frames()

    
    def match_frames(self):

        '''
        Match the frames from the two cameras
        '''

        firstcam_timestampfiles = find_data_between(self.firstcam_timestamp_path, self.obs_start, self.obs_end, header='firstpl_', footer='.txt')
        firstcam_specfiles = find_data_between(self.firstcam_spec_path, self.obs_start, self.obs_end, header='firstpl_', footer='_spec.fits')

        # load timestamps
        timestamps_matching_spec = np.concatenate([np.genfromtxt(file)[:,4] for file in firstcam_timestampfiles])

        # load psfcam frames and timestamp matching pkl file
        psfcam_frames = np.load(self.psfcam_frames_name)
        with open(self.psfcam_timestamp_name, 'rb') as f:
            psfcam_timestamp = pickle.load(f)

        # append spectrum
        all_cropped_specs = []
        for f in firstcam_specfiles:
            all_cropped_specs.append(fits.getdata(f)[:,:,:])
        all_cropped_specs = np.vstack(all_cropped_specs)

        # validate timestamps
        idx1, idx2 = validate_timestamp_matching(timestamps_matching_spec, (np.array(psfcam_timestamp['timestamps'])))

        # filter out the frames that don't match
        all_cropped_specs = all_cropped_specs[idx1]
        psfcam_frames = psfcam_frames[idx2]

        self.firstcam_frames = all_cropped_specs
        self.psfcam_frames = psfcam_frames
        self.timestamps = np.array(psfcam_timestamp['timestamps'])[idx2]

    def compute_psfcam_centroids(self):

        '''
        Compute the centroids of the psfcam frames
        '''

        centroids = []
        for t in range(len(self.psfcam_frames)):
            try:
                cent = subpixel_centroid_2d(self.psfcam_frames[t])
            except:
                cent = (np.nan, np.nan)
                
            centroids.append(cent)

        centroids = np.array(centroids)
        self.centroids = centroids

    def bin_by_centroids(self, map_n, map_width, effective_idx = None, plot = True, calculate_variance = True, nbootstrap = 100):

        '''
        Bin the frames by centroids
        
        Parameters
        ----------
        map_n: int
            number of bins in x and y
        map_width: float
            width of the map in pixels
        effective_idx: array
            indices of the frames to consider
        plot: bool
            whether to plot the number of frames averaged
        calculate_variance: bool
            whether to calculate the bootstrap variance map
        nbootstrap: int
            number of bootstrap iterations
        '''

        if effective_idx is not None:
            centroids = self.centroids[effective_idx]
            psfcam_frames = self.psfcam_frames[effective_idx]
            firstcam_frames = self.firstcam_frames[effective_idx]
        else:
            centroids = self.centroids
            psfcam_frames = self.psfcam_frames
            firstcam_frames = self.firstcam_frames

        self.map_n = map_n
        self.map_width = map_width

        xbins = np.linspace(np.nanmedian(centroids[:,0]) - map_width/2, np.nanmedian(centroids[:,0]) + map_width/2, map_n+1)
        ybins = np.linspace(np.nanmedian(centroids[:,1]) - map_width/2, np.nanmedian(centroids[:,1]) + map_width/2, map_n+1)

        self.x_mas = ((xbins[:-1] + np.diff(xbins)[0]/2) - np.nanmedian(centroids[:,0])) * self.pix2mas
        self.y_mas = ((ybins[:-1] + np.diff(ybins)[0]/2) - np.nanmedian(centroids[:,1])) * self.pix2mas

        self.xbins = xbins
        self.ybins = ybins

        self.psfcam_binned_frames, self.firstcam_binned_frames, self.num_frames, self.idxs = bin_by_centroids(psfcam_frames, firstcam_frames, centroids, xbins, ybins)

        self.xmin = (xbins[0] - np.nanmedian(centroids[:,0])) * self.pix2mas
        self.xmax = (xbins[-1]  - np.nanmedian(centroids[:,0])) * self.pix2mas
        self.ymin = (ybins[0] - np.nanmedian(centroids[:,1])) * self.pix2mas
        self.ymax = (ybins[-1]  - np.nanmedian(centroids[:,1])) * self.pix2mas

        if plot:

            fig = plt.figure(figsize=(5,5))
            plt.imshow(self.num_frames, origin='lower', extent = (self.xmin, self.xmax, self.ymin, self.ymax))
            plt.xlabel('x (mas)')
            plt.ylabel('y (mas)')
            plt.colorbar()
            plt.title('Number of frames averaged')
            plt.show()

        if calculate_variance:
            self.var, self.normvar = calculate_bootstrap_variance_map(firstcam_frames, self.idxs, nbootstrap = nbootstrap)
    
    def save(self, filename):
        '''
        Save the data to a fits file
        
        Parameters
        ----------
        filename: str
            name of the fits file
        '''
        header = fits.Header()
        header['XMIN'] = self.xmin
        header['XMAX'] = self.xmax
        header['YMIN'] = self.ymin
        header['YMAX'] = self.ymax
        header['MAP_N'] = self.map_n
        header['MAP_W'] = self.map_width

        hdu = fits.PrimaryHDU(self.firstcam_binned_frames, header = header)
        hdu2 = fits.ImageHDU(self.num_frames, name = 'nframes')
        hdu3 = fits.ImageHDU(self.psfcam_binned_frames, name='psfcam')
        hdu4 = fits.ImageHDU(self.var, name='var')
        hdu5 = fits.ImageHDU(self.normvar, name='normvar')

        hdulist = fits.HDUList([hdu, hdu2, hdu3, hdu4, hdu5])
        hdulist.writeto(filename, overwrite=True)
