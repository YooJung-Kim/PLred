# functions related to constructing coupling maps

import numpy as np
import matplotlib.pyplot as plt
import pickle, os
from astropy.io import fits
from .utils import find_data_between
from ..imageutils import subpixel_centroid_2d
from .parameters import *
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import h5py

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


def bin_by_centroids_from_indices(psfcamframes, firstcam_file_indices, firstcam_frame_indices, firstcam_files,
                                  centroids, xbins, ybins,
                                  bootstrap = False,
                                  skip_frame_reading = False):
    '''
    Bin frames by centroids, but not from already stored frames.
    Reads the frames from the firstcam_file_indices and firstcam_frame_indices.
    This is useful when the firstcam frames are too large to store in memory.
    '''
    
    x = centroids[:,0]
    y = centroids[:,1]
    ny = firstcam_params['size_y']
    nx = firstcam_params['size_x']
    
    psfcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, psfcamframes.shape[1], psfcamframes.shape[2]))
    firstcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, ny, nx))

    num_frames = np.zeros((len(xbins)-1, len(ybins)-1))
    idxs0 = np.zeros((len(xbins)-1, len(ybins)-1, len(firstcam_file_indices)), dtype=bool)

    for i in range(len(xbins)-1):
        for j in range(len(ybins)-1):
            # check if the x and y coordinates are within the bin
            xidx = (x >= xbins[i]) & (x < xbins[i+1])
            yidx = (y >= ybins[j]) & (y < ybins[j+1])
            idx = xidx & yidx

            # store the indices of the frames that match this bin
            idxs0[i,j] = idx
            psfcam_binned_frames[i,j] = np.mean(psfcamframes[idx], axis=0)
            num_frames[i,j] = np.sum(idx)
    
    if bootstrap:
        idxs = np.zeros((len(xbins)-1, len(ybins)-1, len(firstcam_file_indices)), dtype=int)

        for i in range(len(xbins)-1):
            for j in range(len(ybins)-1):      
                all_indices = np.where(idxs0[i,j] == True)[0]
                # resample the indices with replacement
                resampled_idx = np.random.choice(all_indices, len(all_indices), replace = True)
                # store the resampled frames
                psfcam_binned_frames[i,j] = np.mean(psfcamframes[resampled_idx], axis=0)
                for k in resampled_idx:
                    idxs[i,j,k] += 1
                print(f"Resampled {len(resampled_idx)} frames for bin {i},{j}")
                print(idxs, idxs0)
    else:
        idxs = idxs0

    if not skip_frame_reading:
        # now read the frames from the firstcam_file_indices and firstcam_frame_indices
        for fileind, f in tqdm(enumerate(firstcam_files)):
            print('Start reading file %d' % fileind)
            
            # indices that correspond to this file
            id = (firstcam_file_indices == fileind)
            print('ID', id, len(id))
            print(np.where(id == True)[0])

            # frame indices
            id_frame = firstcam_frame_indices[id] # get the frame indices for this file
            print('ID_FRAME', id_frame)
            # id_frame = np.where(id_frame)[0]

            _dat = fits.getdata(f)

            for i0, i in enumerate(id_frame): #range(len(id_frame)):

                if not bootstrap:
                    iy, ix = np.where(idxs[:,:,np.where(id == True)[0][i0]] == True) # get the indices of the bin that matches this frame])
                    # iy, ix = np.where(idxs[:,:,np.where(id == True)[0][i]] == True) # get the indices of the bin that matches this frame])
                    nstack = 1
                else:
                    iy, ix = np.where(idxs[:,:,np.where(id == True)[0][i0]] > 0) # get the indices of the bin that matches this frame])
                    nstack = np.sum(idxs[:,:,np.where(id == True)[0][i0]])
                try:
                    # np.where(id == True)[0][i] # get the indices of the frames that match this file

                    # idxs[]
                    iy = iy[0]
                    ix = ix[0]

                    firstcam_binned_frames[iy,ix,:,:] += _dat[i] * nstack
                    print(f"Added frame {i} from file {fileind} to bin {iy},{ix}")
                except Exception:
                    print(f"Failed to add frame {i} from file {fileind} to bin {iy},{ix}")
                    continue

        # now calculate the mean
        firstcam_binned_frames = firstcam_binned_frames / num_frames[:,:,None,None]
        return psfcam_binned_frames, firstcam_binned_frames, num_frames, idxs
    
    else:
        # if we are not reading the frames, just return the binned frames
        return psfcam_binned_frames, firstcam_binned_frames, num_frames, idxs




def bin_by_centroids_to_file(outname, psfcamframes, firstcam_file_indices, firstcam_frame_indices, firstcam_files,
                                  centroids, xbins, ybins,
                                #   maxframes = 1000,
                                #   bootstrap = False,
                                  skip_frame_reading = False):
    '''
    Bin frames by centroids, but not from already stored frames.
    Reads the frames from the firstcam_file_indices and firstcam_frame_indices.
    This is useful when the firstcam frames are too large to store in memory.
    '''
    
    x = centroids[:,0]
    y = centroids[:,1]
    ny = firstcam_params['size_y']
    nx = firstcam_params['size_x']
    
    psfcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, psfcamframes.shape[1], psfcamframes.shape[2]))
    firstcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, ny, nx))

    num_frames = np.zeros((len(xbins)-1, len(ybins)-1))
    idxs = np.zeros((len(xbins)-1, len(ybins)-1, len(firstcam_file_indices)), dtype=bool)


    for i in range(len(xbins)-1):
        for j in range(len(ybins)-1):

            # check if the x and y coordinates are within the bin
            xidx = (x >= xbins[i]) & (x < xbins[i+1])
            yidx = (y >= ybins[j]) & (y < ybins[j+1])
            idx = xidx & yidx

            # store the indices of the frames that match this bin
            idxs[i,j] = idx
            psfcam_binned_frames[i,j] = np.mean(psfcamframes[idx], axis=0)
            num_frames[i,j] = np.sum(idx)

            if num_frames[i,j] > 0:
                print(np.shape(psfcamframes[idx]))

                with h5py.File(outname+'_bin_%d_%d.h5' % (i,j), 'w') as h5f:
                    print("creating file %s" % (outname+'_bin_%d_%d.h5' % (i,j)))
                    
                    rawframes_dset = h5f.create_dataset('rawframes', 
                                            shape = (num_frames[i,j], firstcam_params['size_y'], firstcam_params['size_x']), 
                                            # chunks = (1, firstcam_params['size_y'], firstcam_params['size_x']),
                                            # maxshape=(None, firstcam_params['size_y'], firstcam_params['size_x']), 
                                            dtype='int')
                    psfframes_dset = h5f.create_dataset('psfframes',
                                                        data = psfcamframes[idx])
                    peaks_dset = h5f.create_dataset('peaks',
                                                    data = np.nanmax(psfcamframes[idx], axis=(1,2)))
                    centers_dset = h5f.create_dataset('centers',
                                                    data = np.array([x[idx], y[idx]]).T)
                    
                    fileinds_dset = h5f.create_dataset('fileinds',
                                                        data = firstcam_file_indices[idx])
                    frameidnds_dset = h5f.create_dataset('frameinds',
                                                        data = firstcam_frame_indices[idx])

                    h5f.attrs['num_frames'] = num_frames[i,j].astype(int)
                    h5f.attrs['xbin'] = xbins[i]
                    h5f.attrs['ybin'] = ybins[j]
                    
                


    write_idx = {(i, j):0 for i in range(len(xbins)-1) for j in range(len(ybins)-1)}

    if not skip_frame_reading:
        # now read the frames from the firstcam_file_indices and firstcam_frame_indices
        for fileind, f in tqdm(enumerate(firstcam_files)):
            print('Start reading file %d' % fileind)
            
            # indices that correspond to this file
            id = (firstcam_file_indices == fileind)
            # print('ID', id, len(id))
            # print(np.where(id == True)[0])

            # frame indices
            id_frame = firstcam_frame_indices[id] # get the frame indices for this file
            # print('ID_FRAME', id_frame)
            # id_frame = np.where(id_frame)[0]

            _dat = fits.getdata(f)

            # CHANGED HERE!!
            for i0, i in enumerate(id_frame): #range(len(id_frame)):
                
                iy, ix = np.where(idxs[:,:,np.where(id == True)[0][i0]] == True) # get the indices of the bin that matches this frame])


                try:

                    iy = iy[0]
                    ix = ix[0]



                    # append the frame to the h5 file
                    with h5py.File(outname+'_bin_%d_%d.h5' % (iy,ix), 'r+') as h5f:
                        dset = h5f['rawframes']
                        wi = write_idx[(iy, ix)]
                        dset[wi] = _dat[i]
                        write_idx[(iy, ix)] += 1

                    # firstcam_binned_frames[iy,ix,:,:] += _dat[i] * nstack
                    print(f"Added frame {i} from file {fileind} to bin {iy},{ix}")
                except:
                    # print(e)
                    print(f"Failed to add frame {i} from file {fileind} to bin {iy},{ix}")
                    continue

        return psfcam_binned_frames, num_frames, idxs
    
    else:
        # if we are not reading the frames, just return the binned frames
        return psfcam_binned_frames, num_frames, idxs
    

def calculate_bootstrap_variance_map(firstcamframes, idxs, nbootstrap = 100,
                                     return_bootstrap_samples = False):

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

    if return_bootstrap_samples:
        return var, normvar, bootstrap_normframes
    else:
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

    normvar = None
    var = None

    def __init__(self, firstcam_timestamp_path, firstcam_spec_path,
                 obs_start, obs_end,
                 psfcam, 
                 psfcam_frames_name, psfcam_timestamp_name,
                 match_frames = True,
                 store_spec = True):
        
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

        self.store_spec = store_spec


        if match_frames:
            if store_spec:
                # stores all the data.
                # if working with raw FIRSTcam frames, don't use this option.
                self.match_frames(footer = '_spec.fits')
            else:
                self.match_frames(location_only = True, footer = '.fits')


    
    def match_frames(self, location_only = False, footer = '_spec.fits'):

        '''
        Match the frames from the two cameras
        '''

        firstcam_timestampfiles = find_data_between(self.firstcam_timestamp_path, self.obs_start, self.obs_end, header='firstpl_', footer='.txt')
        firstcam_specfiles = find_data_between(self.firstcam_spec_path, self.obs_start, self.obs_end, header='firstpl_', footer=footer)

        # load timestamps
        timestamps_matching_spec = np.concatenate([np.genfromtxt(file)[:,4] for file in firstcam_timestampfiles])

        # load psfcam frames and timestamp matching pkl file
        psfcam_frames = np.load(self.psfcam_frames_name)
        with open(self.psfcam_timestamp_name, 'rb') as f:
            psfcam_timestamp = pickle.load(f)

        if not location_only:
            # append spectrum
            all_cropped_specs = []
            for f in firstcam_specfiles:
                all_cropped_specs.append(fits.getdata(f)[:,:,:])
            all_cropped_specs = np.vstack(all_cropped_specs)
        
        else:
            all_file_indices = []
            all_frame_indices = []
            for fi, f in enumerate(firstcam_specfiles):
                for n in range(fits.getheader(f)['NAXIS3']): #fits.getdata(f).shape[0]):
                    all_file_indices.append(fi)
                    all_frame_indices.append(n)
            all_file_indices = np.array(all_file_indices)
            all_frame_indices = np.array(all_frame_indices)


        # validate timestamps
        idx1, idx2 = validate_timestamp_matching(timestamps_matching_spec, (np.array(psfcam_timestamp['timestamps'])))

        # filter out the frames that don't match
        if not location_only:
            all_cropped_specs = all_cropped_specs[idx1]
            self.firstcam_frames = all_cropped_specs

        else:
            all_file_indices = all_file_indices[idx1]
            all_frame_indices = all_frame_indices[idx1]
            self.firstcam_file_indices = all_file_indices
            self.firstcam_frame_indices = all_frame_indices
            self.firstcam_files = firstcam_specfiles

        psfcam_frames = psfcam_frames[idx2]
        self.psfcam_frames = psfcam_frames
        self.timestamps = np.array(psfcam_timestamp['timestamps'])[idx2]

    def compute_psfcam_centroids(self, peak = True):

        '''
        Compute the centroids of the psfcam frames
        '''

        centroids = []
        for t in range(len(self.psfcam_frames)):
            try:

                if peak:
                    cent = subpixel_centroid_2d(self.psfcam_frames[t])
                
                else:
                    cent = center_of_mass(self.psfcam_frames[t])
            except:
                cent = (np.nan, np.nan)
                
            centroids.append(cent)

        centroids = np.array(centroids)
        self.centroids = centroids

    def bin_by_centroids(self, map_n, map_width, effective_idx = None, plot = True, calculate_variance = True, nbootstrap = 100,
                         return_bootstrap_samples = False,
                         skip_frame_reading = False,
                         to_file = False,
                         filename = None):

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
        return_bootstrap_samples: bool
            whether to return the bootstrap samples
        '''

        if effective_idx is not None:
            centroids = self.centroids[effective_idx]
            psfcam_frames = self.psfcam_frames[effective_idx]
            if not self.store_spec:
                firstcam_file_indices = self.firstcam_file_indices[effective_idx]
                firstcam_frame_indices = self.firstcam_frame_indices[effective_idx]
            else:
                firstcam_frames = self.firstcam_frames[effective_idx]

        else:
            centroids = self.centroids #[effective_idx]
            psfcam_frames = self.psfcam_frames #[effective_idx]

            if not self.store_spec:
                firstcam_file_indices = self.firstcam_file_indices #[effective_idx]
                firstcam_frame_indices = self.firstcam_frame_indices #[effective_idx]
            else:
                firstcam_frames = self.firstcam_frames #[effective_idx]

        self.map_n = map_n
        self.map_width = map_width

        xbins = np.linspace(np.nanmedian(centroids[:,0]) - map_width/2, np.nanmedian(centroids[:,0]) + map_width/2, map_n+1)
        ybins = np.linspace(np.nanmedian(centroids[:,1]) - map_width/2, np.nanmedian(centroids[:,1]) + map_width/2, map_n+1)

        self.x_mas = ((xbins[:-1] + np.diff(xbins)[0]/2) - np.nanmedian(centroids[:,0])) * self.pix2mas
        self.y_mas = ((ybins[:-1] + np.diff(ybins)[0]/2) - np.nanmedian(centroids[:,1])) * self.pix2mas

        self.xbins = xbins
        self.ybins = ybins

        self.xmin = (xbins[0] - np.nanmedian(centroids[:,0])) * self.pix2mas
        self.xmax = (xbins[-1]  - np.nanmedian(centroids[:,0])) * self.pix2mas
        self.ymin = (ybins[0] - np.nanmedian(centroids[:,1])) * self.pix2mas
        self.ymax = (ybins[-1]  - np.nanmedian(centroids[:,1])) * self.pix2mas


        if self.store_spec:
            self.psfcam_binned_frames, self.firstcam_binned_frames, self.num_frames, self.idxs = bin_by_centroids(psfcam_frames, firstcam_frames, centroids, xbins, ybins)
        else:
            if not to_file:
                self.psfcam_binned_frames, self.firstcam_binned_frames, self.num_frames, self.idxs = bin_by_centroids_from_indices(psfcam_frames, firstcam_file_indices, firstcam_frame_indices, self.firstcam_files, centroids, xbins, ybins, skip_frame_reading=skip_frame_reading)
            else:
                import json
                infodict = {'xmin': self.xmin, 'ymin': self.ymin, 'xmax': self.xmax, 'ymax': self.ymax, 'map_n': map_n, 'map_w': map_width}
                json.dump(infodict, open(filename+'_info.json', 'w'))
                print("Info Saved to %s" % filename+'_info.json')
                self.psfcam_binned_frames, self.num_frames, self.idxs = bin_by_centroids_to_file(filename, psfcam_frames, firstcam_file_indices, firstcam_frame_indices, self.firstcam_files, centroids, xbins, ybins, skip_frame_reading=skip_frame_reading)
                return      
             
            # this is used for bootstrap later
            self.result_psfcam_frames = psfcam_frames
            self.result_firstcam_file_indices = firstcam_file_indices
            self.result_firstcam_frame_indices = firstcam_frame_indices
            self.result_firstcam_files = self.firstcam_files
            self.result_centroids = centroids
            self.result_xbins = xbins
            self.result_ybins = ybins


        if plot:

            fig = plt.figure(figsize=(5,5))
            plt.imshow(self.num_frames, origin='lower', extent = (self.xmin, self.xmax, self.ymin, self.ymax))
            plt.xlabel('x (mas)')
            plt.ylabel('y (mas)')
            plt.colorbar()
            plt.title('Number of frames averaged')
            plt.show()

        if calculate_variance:

            if self.store_spec:

                if return_bootstrap_samples:
                    self.var, self.normvar, self.bootstrap_samples = calculate_bootstrap_variance_map(firstcam_frames, self.idxs, nbootstrap = nbootstrap, return_bootstrap_samples = True)
                else:
                    self.var, self.normvar = calculate_bootstrap_variance_map(firstcam_frames, self.idxs, nbootstrap = nbootstrap)
                    self.bootstrap_samples = None

            else:
                raise ValueError("calculate_variance is not implemented for firstcam frames stored in files. Please use store_spec = True to store the frames in memory.") 
                                                                                                                                                                                                           
                # for i in range(nbootstrap):
                #     psfcam_binned_frames, firstcam_binned_frames, num_frames, idxs = bin_by_centroids_from_indices(psfcam_frames, firstcam_file_indices, firstcam_frame_indices, self.firstcam_files, centroids, xbins, ybins,
                                                                                                                # bootstrap=True)

        else:
            self.var = None
            self.normvar = None
            self.bootstrap_samples = None


    def save_bootstrap_frames(self, filename, nbootstrap = 100):

        assert self.store_spec is False, "compute_bootstrap_frames is not implemented for firstcam frames stored in memory."

        for i in range(nbootstrap):
            self.psfcam_binned_frames, self.firstcam_binned_frames, self.num_frames, self.idxs = bin_by_centroids_from_indices(self.result_psfcam_frames, 
                                                                                                                            self.result_firstcam_file_indices, 
                                                                                                                            self.result_firstcam_frame_indices,
                                                                                                                            self.result_firstcam_files, 
                                                                                                                            self.result_centroids, 
                                                                                                                            self.result_xbins, 
                                                                                                                            self.result_ybins,
                                                                                                                            bootstrap=True)
        
            self.var = None
            self.bootstrap_samples = None
            self.save(filename + '_bootstrap_%d.fits' % i)
            print("Saved bootstrap frames to %s" % filename + '_bootstrap_%d.fits' % i)


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
        hdulist = fits.HDUList([hdu, hdu2, hdu3])
        
        if self.var is not None:
            hdu4 = fits.ImageHDU(self.var, name='var')
            hdu5 = fits.ImageHDU(self.normvar, name='normvar')
            hdulist.append(hdu4)
            hdulist.append(hdu5)

        hdulist.writeto(filename, overwrite=True)
        print("Saved to %s" % filename)

        if self.bootstrap_samples is not None:
            hdu6 = fits.PrimaryHDU(self.bootstrap_samples, header=header)
            hdu6.writeto(filename.replace('.fits', '_bootstrap.fits'), overwrite=True)
            print("Saved bootstrap samples to %s" % filename.replace('.fits', '_bootstrap.fits'))
