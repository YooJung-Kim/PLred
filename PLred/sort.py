# codes related to constructing coupling maps
# 1) Timestamp matching script
# 2) Frame sorting class (reads timestamp matched frames and sorts them by centroids)
# 3) Response map generation script (input for mapmodel)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from .imageutils import subpixel_centroid_2d
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import h5py
import json
from configobj import ConfigObj
from datetime import datetime
from bisect import bisect
import os, glob, pickle, re

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


def script_match_timestamps(configname):

    '''
    This script is copied from scexao6:/mnt/userdata/yjkim/timestamp_matched_palila/script_match_timestamps.py
    # Script matching timestamps between two cameras
    # Author: Yoo Jung Kim, Feb 27 2025

    Config file example:

        [Fastcam]
        obs_date            = 20250211
        start_time          = 11:59:00
        end_time            = 12:20:09
        dark_file           = # if empty, dark_start_time and dark_end_time are used
        dark_start_time     = 12:23:04
        dark_end_time       = 12:23:05
        path                = /mnt/sdata/20250211/palila/

        [Slowcam]
        timestamp_dir       = /mnt/userdata/yjkim/20250211_betcmi/firstcam_timestamps/
        nbin                = 1

        [Output]
        outname             = /mnt/userdata/yjkim/timestamp_matched_palila/betcmi_20250211
        filename            = first_palila_matched

        [Options]
        verbose              = False
        show_plot            = True
        crop_width           = 20

    '''

    # read the config file

    config = ConfigObj(configname)

    fastcam_dir = config['Fastcam']['path']
    fastcam_start_time = config['Fastcam']['start_time']
    fastcam_end_time = config['Fastcam']['end_time']
    obs_date = config['Fastcam']['obs_date']

    fastcam_dark_file = config['Fastcam']['dark_file']
    
    if fastcam_dark_file.strip() == '':
        fastcam_dark_start_time = config['Fastcam']['dark_start_time']
        fastcam_dark_end_time = config['Fastcam']['dark_end_time']

    slowcam_timestamps = config['Slowcam']['timestamp_dir']
    try:
        slowcam_nbin = int(config['Slowcam']['nbin'])
    except:
        slowcam_nbin = 1

    outname = config['Output']['outname']
    filename = config['Output']['filename']

    verbose = (config['Options']['verbose']).lower() == 'true'
    show_plot = (config['Options']['show_plot']).lower() == 'true'
    crop_width = int(config['Options']['crop_width'])

    os.makedirs(outname, exist_ok=True)


    # find fastcam data
    fastcam_timestampfiles = find_data_between(fastcam_dir, fastcam_start_time, fastcam_end_time, footer = '.txt') #np.sort(glob.glob(data_dir+args.fastcam_filename+'*.txt'))
    # fastcam_frames = find_data_between(fastcam_dir, fastcam_start_time, fastcam_end_time, footer = '.fits')
    if fastcam_dark_file.strip() == '':
        fastcam_darkframes = find_data_between(fastcam_dir, fastcam_dark_start_time, fastcam_dark_end_time, footer = '.fits')
    
    # find slowcam timestamp data
    slowcam_timestampfiles = np.sort(glob.glob(slowcam_timestamps+'*.txt')) ##np.sort(glob.glob(data_dir+args.slowcam_filename+'*.txt'))


    if verbose:
        print("fastcam timestampfiles", fastcam_timestampfiles)
        print("slowcam timestampfiles", slowcam_timestampfiles)
        # for f in slowcam_timestampfiles:
        #     print(np.shape(np.genfromtxt(f)))
                # print("fastcam timestamp file %s has wrong shape" % f)
                # print("expected 7 columns, got %d" % np.shape(np.genfromtxt(f))[1])

    ######################
    # Timestamp matching #
    ######################
    
    fastcam_timestamp = np.concatenate([np.genfromtxt(f)[:,4] for f in fastcam_timestampfiles])
    
    # if slowcam_nbin > 1:
    #     _ts = [np.genfromtxt(f)[:,4] for f in slowcam_timestampfiles]

    #     slowcam_timestamp = np.concatenate([np.genfromtxt(f)[:,4] for f in slowcam_timestampfiles])
    # else:
    slowcam_timestamp = np.concatenate([np.genfromtxt(f)[:,4] for f in slowcam_timestampfiles])

    # file and frame indices into array
    fastcam_fileinds = np.concatenate([np.full(len(np.genfromtxt(f)), i) for i, f in enumerate(fastcam_timestampfiles)])
    slowcam_fileinds = np.concatenate([np.full(len(np.genfromtxt(f)), i) for i, f in enumerate(slowcam_timestampfiles)])

    fastcam_frameinds = np.concatenate([np.genfromtxt(f, dtype=int)[:,0] for f in fastcam_timestampfiles])
    slowcam_frameinds = np.concatenate([np.genfromtxt(f, dtype=int)[:,0] for f in slowcam_timestampfiles])

    # optional slowcam nbin

    ## TODO!!: need to deal with situations where logging went off in the middle of the observation. 
    if slowcam_nbin > 1:
        slowcam_timestamp = slowcam_timestamp[:(len(slowcam_timestamp)//slowcam_nbin)*slowcam_nbin].reshape((-1, slowcam_nbin))[:,0]
        slowcam_fileinds = slowcam_fileinds[:(len(slowcam_fileinds)//slowcam_nbin)*slowcam_nbin].reshape((-1, slowcam_nbin))[:,0]
        slowcam_frameinds = slowcam_frameinds[:(len(slowcam_frameinds)//slowcam_nbin)*slowcam_nbin].reshape((-1, slowcam_nbin))[:,0]

    # match indices (reference here : slowcam)
    bisect_inds = [bisect(fastcam_timestamp,  slowcam_timestamp[ind]) for ind in range(len(slowcam_timestamp))]

    # define start and end indices
    ind_start = 0 
    ind_end = np.argmin(np.array(bisect_inds) < np.max(bisect_inds))

    if show_plot:
        plt.plot(bisect_inds, 'o-', ms=1)
        plt.axvline(ind_start)
        plt.axvline(ind_end)
        plt.xlabel('slowcam frameinds')
        plt.ylabel('fastcam frameinds')
        
        plt.show()

    print('total slowcam images:', (ind_end - ind_start))


    # now fill in the dictionary

    Dict = {}
    timestamps = []

    for ind in tqdm(np.arange(ind_start, ind_end-1)): 
            
        if verbose:
            print('finding fastcam frames that match with slowcam frame ind %d:%d' % (ind, ind+1))
            print('absolute times: %f to %f' % (slowcam_timestamp[ind],slowcam_timestamp[ind+1]))

            print(fastcam_timestamp[bisect_inds[ind]-1],fastcam_timestamp[bisect_inds[ind]])
            print(slowcam_timestamp[ind])
            print('left', (fastcam_timestamp[bisect_inds[ind]] - slowcam_timestamp[ind]) / (fastcam_timestamp[bisect_inds[ind]] - fastcam_timestamp[bisect_inds[ind]-1] ))

            print('right', (slowcam_timestamp[ind+1] - fastcam_timestamp[bisect_inds[ind+1]-1]) / (fastcam_timestamp[bisect_inds[ind+1]] - fastcam_timestamp[bisect_inds[ind+1]-1] ))

        leftfrac = (fastcam_timestamp[bisect_inds[ind]] - slowcam_timestamp[ind]) / (fastcam_timestamp[bisect_inds[ind]] - fastcam_timestamp[bisect_inds[ind]-1] )
        rightfrac = (slowcam_timestamp[ind+1] - fastcam_timestamp[bisect_inds[ind+1]-1]) / (fastcam_timestamp[bisect_inds[ind+1]] - fastcam_timestamp[bisect_inds[ind+1]-1] )
        
        Dict[ind] = {bisect_inds[ind]-1 :leftfrac}
        for i in np.arange(bisect_inds[ind], bisect_inds[ind+1]-1):
            Dict[ind][i] = 1
        Dict[ind][bisect_inds[ind+1]-1] = rightfrac

        timestamps.append(slowcam_timestamp[ind])

    # save
    data_to_save = {'fastcam_timestampfiles': fastcam_timestampfiles,
                    'slowcam_timestampfiles': slowcam_timestampfiles,
                    'fastcam_fileinds': fastcam_fileinds,
                    'slowcam_fileinds': slowcam_fileinds,
                    'fastcam_frameinds': fastcam_frameinds,
                    'slowcam_frameinds': slowcam_frameinds,
                    'matched_indices': Dict,
                    'timestamps': timestamps
                    }

    with open(outname+'/'+filename+'.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    print("timestamp matching file saved to %s" % (outname+'/'+filename+'.pkl'))


    ######################
    # PSF frame averaging #
    ######################
    if fastcam_dark_file.strip() == '':
        fastcam_avgdark = np.average(fits.getdata(fastcam_darkframes[0]), axis=0)
    else:
        fastcam_avgdark = fits.getdata(fastcam_dark_file)
        
    fastcam_files = [f.split('.txt')[0]+'.fits' for f in fastcam_timestampfiles] 

    (xwidth, ywidth) = np.shape(fastcam_avgdark)
    xc, yc = xwidth//2, ywidth//2
    if crop_width is None:
        xw, yw = xwidth//2, ywidth//2
        print("saving full frames, without crop")
    else:
        xw = yw = int(crop_width)
        print(f"saving cropped frames, [{xc-xw}:{xc+xw}, {yc-yw}:{yc+yw}]")

    fastcam_avgdark = fastcam_avgdark[xc-xw:xc+xw, yc-yw:yc+yw]

    indices = list(Dict.keys())

    current_fastcam_fileind = 0
    print("reading fastcam file", fastcam_files[current_fastcam_fileind])

    # apply dark correction every frame
    current_fastcamfile = fits.getdata(fastcam_files[current_fastcam_fileind])[:,xc-xw:xc+xw, yc-yw:yc+yw] - fastcam_avgdark 

    nstacks = []
    stacked_frames = []

    for ind in tqdm(indices):

        fastcam_ind = np.array(list(Dict[ind].keys()))
        fastcam_frac = np.array(list(Dict[ind].values()))

        stacked_frame = []
        nstack = []

        for (_fastcam_ind, _fastcam_frac) in zip(fastcam_ind, fastcam_frac):

            if fastcam_fileinds[_fastcam_ind] != current_fastcam_fileind:
                current_fastcam_fileind = fastcam_fileinds[_fastcam_ind]
                print("reading new fastcam file", fastcam_files[current_fastcam_fileind])
    
                # apply dark correction every frame
                current_fastcamfile = fits.getdata(fastcam_files[current_fastcam_fileind])[:,xc-xw:xc+xw, yc-yw:yc+yw] - fastcam_avgdark

            if verbose: print("appending frame", fastcam_frameinds[_fastcam_ind],'with fraction of', _fastcam_frac)
            stacked_frame.append(_fastcam_frac* current_fastcamfile[fastcam_frameinds[_fastcam_ind]])
            nstack.append(_fastcam_frac)
        
        stacked_frames.append(np.sum(stacked_frame, axis=0))
        nstacks.append(np.sum(nstack))

    np.save('%s_fastcam_matched_frames.npy' % (outname+'/'+filename), stacked_frames)
    np.save('%s_nstacks.npy'% (outname+'/'+filename), nstacks)

    print("matched fastcam frames saved to %s_fastcam_matched_frames.npy" % (outname+'/'+filename))


# def bin_by_centroids(psfcamframes, plcamframes, centroids, xbins, ybins):
#     '''
#     Bin frames by centroids
#     '''
#     x = centroids[:,0]
#     y = centroids[:,1]

#     psfcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, psfcamframes.shape[1], psfcamframes.shape[2]))
#     plcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, plcamframes.shape[1], plcamframes.shape[2]))
#     num_frames = np.zeros((len(xbins)-1, len(ybins)-1))
#     idxs = np.zeros((len(xbins)-1, len(ybins)-1, len(plcamframes)), dtype=bool)
    
#     for i in range(len(xbins)-1):
#         for j in range(len(ybins)-1):
#             xidx = (x >= xbins[i]) & (x < xbins[i+1])
#             yidx = (y >= ybins[j]) & (y < ybins[j+1])
#             idx = xidx & yidx
#             idxs[i,j] = idx

#             psfcam_binned_frames[i,j] = np.mean(psfcamframes[idx], axis=0)
#             plcam_binned_frames[i,j] = np.mean(plcamframes[idx], axis=0)
#             num_frames[i,j] = np.sum(idx)
#     return psfcam_binned_frames, plcam_binned_frames, num_frames, idxs


def bin_by_centroids_from_indices(psfcamframes, #plcam_file_indices, plcam_frame_indices, plcam_files, ny, nx,
                                  centroids, xbins, ybins):
                                #   bootstrap = False,
                                #   skip_frame_reading = False):
    '''
    Bin frames by centroids, but not from already stored frames.
    Reads the frames from the plcam_file_indices and plcam_frame_indices.
    This is useful when the plcam frames are too large to store in memory.
    '''
    
    x = centroids[:,0]
    y = centroids[:,1]
    # ny = plcam_params['size_y']
    # nx = plcam_params['size_x']
    
    psfcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, psfcamframes.shape[1], psfcamframes.shape[2]))
    # plcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, ny, nx))

    num_frames = np.zeros((len(xbins)-1, len(ybins)-1))
    idxs0 = np.zeros((len(xbins)-1, len(ybins)-1, len(centroids)), dtype=bool)

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
    
    # if bootstrap:
    #     idxs = np.zeros((len(xbins)-1, len(ybins)-1, len(plcam_file_indices)), dtype=int)

    #     for i in range(len(xbins)-1):
    #         for j in range(len(ybins)-1):      
    #             all_indices = np.where(idxs0[i,j] == True)[0]
    #             # resample the indices with replacement
    #             resampled_idx = np.random.choice(all_indices, len(all_indices), replace = True)
    #             # store the resampled frames
    #             psfcam_binned_frames[i,j] = np.mean(psfcamframes[resampled_idx], axis=0)
    #             for k in resampled_idx:
    #                 idxs[i,j,k] += 1
    #             print(f"Resampled {len(resampled_idx)} frames for bin {i},{j}")
    #             print(idxs, idxs0)
    # else:
    idxs = idxs0

    # if not skip_frame_reading:
    #     # now read the frames from the plcam_file_indices and plcam_frame_indices
    #     for fileind, f in tqdm(enumerate(plcam_files)):
    #         print('Start reading file %d' % fileind)
            
    #         # indices that correspond to this file
    #         id = (plcam_file_indices == fileind)
    #         print('ID', id, len(id))
    #         print(np.where(id == True)[0])

    #         # frame indices
    #         id_frame = plcam_frame_indices[id] # get the frame indices for this file
    #         print('ID_FRAME', id_frame)
    #         # id_frame = np.where(id_frame)[0]

    #         _dat = fits.getdata(f)

    #         for i0, i in enumerate(id_frame): #range(len(id_frame)):

    #             if not bootstrap:
    #                 iy, ix = np.where(idxs[:,:,np.where(id == True)[0][i0]] == True) # get the indices of the bin that matches this frame])
    #                 # iy, ix = np.where(idxs[:,:,np.where(id == True)[0][i]] == True) # get the indices of the bin that matches this frame])
    #                 nstack = 1
    #             else:
    #                 iy, ix = np.where(idxs[:,:,np.where(id == True)[0][i0]] > 0) # get the indices of the bin that matches this frame])
    #                 nstack = np.sum(idxs[:,:,np.where(id == True)[0][i0]])
    #             try:
    #                 # np.where(id == True)[0][i] # get the indices of the frames that match this file

    #                 # idxs[]
    #                 iy = iy[0]
    #                 ix = ix[0]

    #                 plcam_binned_frames[iy,ix,:,:] += _dat[i] * nstack
    #                 print(f"Added frame {i} from file {fileind} to bin {iy},{ix}")
    #             except Exception:
    #                 print(f"Failed to add frame {i} from file {fileind} to bin {iy},{ix}")
    #                 continue

    #     # now calculate the mean
    #     plcam_binned_frames = plcam_binned_frames / num_frames[:,:,None,None]
    #     return psfcam_binned_frames, plcam_binned_frames, num_frames, idxs
    
    # else:
        # if we are not reading the frames, just return the binned frames
    return psfcam_binned_frames, num_frames, idxs




def bin_by_centroids_to_file(outname, psfcamframes, plcam_file_indices, plcam_frame_indices, plcam_files, ny, nx,
                                  centroids, xbins, ybins,
                                #   maxframes = 1000,
                                #   bootstrap = False,
                                  skip_frame_reading = False,
                                  nbin = 1):
    '''
    Bin frames by centroids, but not from already stored frames.
    Reads the frames from the plcam_file_indices and plcam_frame_indices.
    This is useful when the plcam frames are too large to store in memory.
    '''
    
    x = centroids[:,0]
    y = centroids[:,1]
    # ny = plcam_params['size_y']
    # nx = plcam_params['size_x']
    
    psfcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, psfcamframes.shape[1], psfcamframes.shape[2]))
    # plcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, ny, nx))

    num_frames = np.zeros((len(xbins)-1, len(ybins)-1))
    idxs = np.zeros((len(xbins)-1, len(ybins)-1, len(plcam_file_indices)), dtype=bool)


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
                # print(np.shape(psfcamframes[idx]))

                with h5py.File(outname+'_bin_%d_%d.h5' % (i,j), 'w') as h5f:
                    print("creating file %s" % (outname+'_bin_%d_%d.h5' % (i,j)))
                    
                    rawframes_dset = h5f.create_dataset('rawframes', 
                                            shape = (num_frames[i,j], ny, nx), 
                                            # chunks = (1, plcam_params['size_y'], plcam_params['size_x']),
                                            # maxshape=(None, plcam_params['size_y'], plcam_params['size_x']), 
                                            dtype='int')
                    psfframes_dset = h5f.create_dataset('psfframes',
                                                        data = psfcamframes[idx])
                    peaks_dset = h5f.create_dataset('peaks',
                                                    data = np.nanmax(psfcamframes[idx], axis=(1,2)))
                    centers_dset = h5f.create_dataset('centers',
                                                    data = np.array([x[idx], y[idx]]).T)
                    
                    fileinds_dset = h5f.create_dataset('fileinds',
                                                        data = plcam_file_indices[idx])
                    frameidnds_dset = h5f.create_dataset('frameinds',
                                                        data = plcam_frame_indices[idx])

                    h5f.attrs['num_frames'] = num_frames[i,j].astype(int)
                    h5f.attrs['xbin'] = xbins[i]
                    h5f.attrs['ybin'] = ybins[j]
                    
                


    write_idx = {(i, j):0 for i in range(len(xbins)-1) for j in range(len(ybins)-1)}

    if not skip_frame_reading:
        # now read the frames from the plcam_file_indices and plcam_frame_indices
        for fileind, f in tqdm(enumerate(plcam_files)):
            print('Start reading file %d' % fileind)
            
            # indices that correspond to this file
            id = (plcam_file_indices == fileind)
            # print('ID', id, len(id))
            # print(np.where(id == True)[0])

            # frame indices
            id_frame = plcam_frame_indices[id] # get the frame indices for this file
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

                        if nbin == 1:
                            dset[wi] = _dat[i]
                        else:
                            dset[wi] = np.mean(_dat[i:i+nbin], axis=0)
                        write_idx[(iy, ix)] += 1

                    # plcam_binned_frames[iy,ix,:,:] += _dat[i] * nstack
                    if nbin == 1:
                        print(f"Added frame {i} from file {fileind} to bin {iy},{ix}")
                    else:
                        print(f"Added frames {i} to {i+nbin-1} from file {fileind} to bin {iy},{ix}")
                except:
                    # print(e)
                    if nbin == 1:
                        print(f"Failed to add frame {i} from file {fileind} to bin {iy},{ix}")
                    else:
                        print(f"Failed to add frames {i} to {i+nbin-1} from file {fileind} to bin {iy},{ix}")
                    continue

        return psfcam_binned_frames, num_frames, idxs
    
    else:
        # if we are not reading the frames, just return the binned frames
        return psfcam_binned_frames, num_frames, idxs
    

# def calculate_bootstrap_variance_map(plcamframes, idxs, nbootstrap = 100,
#                                      return_bootstrap_samples = False):

#     '''
#     Calculate the bootstrap variance maps
#     '''

#     arr = np.arange(len(plcamframes))
#     bootstrap_frames = np.zeros((nbootstrap, idxs.shape[0], idxs.shape[1], plcamframes.shape[1], plcamframes.shape[2]))
#     bootstrap_normframes = np.zeros((nbootstrap, idxs.shape[0], idxs.shape[1], plcamframes.shape[1], plcamframes.shape[2]))

#     for i in range(idxs.shape[0]):

#         for j in range(idxs.shape[1]):

#             idx = idxs[i,j]

#             for k in range(nbootstrap):
#                 # resample the indices with replacement
#                 resampled_idx = np.random.choice(arr[idx], len(arr[idx]), replace = True)
#                 # store the resampled frames
#                 bootstrap_frames[k,i,j] = np.nanmean(plcamframes[resampled_idx], axis = 0)
    
#     # calculate normalized bootstrap frames
#     for k in range(nbootstrap):
#         bootstrap_normframes[k] = bootstrap_frames[k] / np.nansum(bootstrap_frames[k], axis=(0,1))[None,None,:,:]

#     # calculate std of bootstrap frames
#     var = np.nanvar(bootstrap_frames, axis = 0)

#     # calculate normalized boostrap frames
#     normvar = np.nanvar(bootstrap_normframes, axis=0)

#     if return_bootstrap_samples:
#         return var, normvar, bootstrap_normframes
#     else:
#         return var, normvar


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


class FrameSorter:
    '''
    Class to handle simultaneous data from two cameras
    '''

    normvar = None
    var = None

    def __init__(self, plcam_timestamp_path, plcam_spec_path,
                #  psfcam, 
                 psfcam_frames_name, psfcam_timestamp_name,
                 obs_start, obs_end,
                 match_frames = True,
                #  store_spec = True,
                 pix2mas = 16.2,
                 plcam_header = 'firstpl_',
                 plcam_footer = '.fits',
                 plcam_shape = (412, 1896),
                 ):
        
        '''
        Initialize the class

        Parameters
        ----------
        plcam_timestamp_path: str
            path to the PL camera timestamp files
        plcam_spec_path: str
            path to the PL camera spectrum files
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
        

        self.plcam_timestamp_path = plcam_timestamp_path
        self.plcam_spec_path = plcam_spec_path
        self.obs_start = obs_start
        self.obs_end = obs_end
        self.psfcam_frames_name = psfcam_frames_name
        self.psfcam_timestamp_name = psfcam_timestamp_name
        self.ny, self.nx = plcam_shape

        # if psfcam == 'palila':
        #     self.pix2mas = palila_params['plate_scale']
        # elif psfcam == 'vcam':
        #     self.pix2mas = vampires_params['plate_scale']
        # else:
        #     raise ValueError("psfcam should be either palila or vcam")
        # self.psfcam_name = psfcam
        self.pix2mas = pix2mas

        # self.store_spec = store_spec


        if match_frames:
            # if store_spec:
            #     # stores all the data.
            #     # if working with raw plcam frames, don't use this option.
            #     self.match_frames(footer = '_spec.fits')
            # else:
            self.match_frames(location_only = True, header = plcam_header, footer = plcam_footer)


    
    def match_frames(self, location_only = True, header = 'firstpl_', footer = '.fits'):

        '''
        Match the frames from the two cameras
        Warning: location_only = False option is not maintained!
        '''

        plcam_timestampfiles = find_data_between(self.plcam_timestamp_path, self.obs_start, self.obs_end, header=header, footer='.txt')
        plcam_specfiles = find_data_between(self.plcam_spec_path, self.obs_start, self.obs_end, header=header, footer=footer)

        # load timestamps
        timestamps_matching_spec = np.concatenate([np.genfromtxt(file)[:,4] for file in plcam_timestampfiles])

        # load psfcam frames and timestamp matching pkl file
        psfcam_frames = np.load(self.psfcam_frames_name)
        with open(self.psfcam_timestamp_name, 'rb') as f:
            psfcam_timestamp = pickle.load(f)

        if not location_only:
            # append spectrum
            all_cropped_specs = []
            for f in plcam_specfiles:
                all_cropped_specs.append(fits.getdata(f)[:,:,:])
            all_cropped_specs = np.vstack(all_cropped_specs)
        
        else:
            all_file_indices = []
            all_frame_indices = []
            for fi, f in enumerate(plcam_specfiles):
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
            self.plcam_frames = all_cropped_specs

        else:
            all_file_indices = all_file_indices[idx1]
            all_frame_indices = all_frame_indices[idx1]
            self.plcam_file_indices = all_file_indices
            self.plcam_frame_indices = all_frame_indices
            self.plcam_files = plcam_specfiles

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

    def bin_by_centroids(self, map_n, map_width, effective_idx = None, plot = True, 
                        #  calculate_variance = True, nbootstrap = 100,
                        #  return_bootstrap_samples = False,
                         skip_frame_reading = False,
                         to_file = False,
                         filename = None,
                         nbin = 1):

        '''
        Bin the frames by centroids
        
        Parameters
        ----------
        map_n: int
            number of bins in x and y
        map_width: float
            width of the map in pixels (PSF camera)
        effective_idx: array
            indices of the frames to consider. use this to filter out bad frames. if None, use all frames.
        plot: bool
            whether to plot the number of frames averaged
        skip_frame_reading: bool
            whether to skip reading the frames from the files. 
        to_file: bool
            whether to save the binned frames to files.
            it generates big files, so use this only if map_n, map_width, and effective_idx are optimized.
        filename: str
            name of the file to save the binned frames to.
        nbin: int
            binning factor for PL camera frames that was used for timestamp matching.
        '''

        if effective_idx is not None:
            centroids = self.centroids[effective_idx]
            psfcam_frames = self.psfcam_frames[effective_idx]
            # if not self.store_spec:
            plcam_file_indices = self.plcam_file_indices[effective_idx]
            plcam_frame_indices = self.plcam_frame_indices[effective_idx]
            # else:
            #     plcam_frames = self.plcam_frames[effective_idx]

        else:
            centroids = self.centroids #[effective_idx]
            psfcam_frames = self.psfcam_frames #[effective_idx]

            # if not self.store_spec:
            plcam_file_indices = self.plcam_file_indices #[effective_idx]
            plcam_frame_indices = self.plcam_frame_indices #[effective_idx]
            # else:
                # plcam_frames = self.plcam_frames #[effective_idx]

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


        # if self.store_spec:
            # self.psfcam_binned_frames, self.plcam_binned_frames, self.num_frames, self.idxs = bin_by_centroids(psfcam_frames, plcam_frames, centroids, xbins, ybins)
        # else:
        if not to_file:
            self.psfcam_binned_frames, self.num_frames, self.idxs = bin_by_centroids_from_indices(psfcam_frames, centroids, xbins, ybins)
        
        
        else:
            infodict = {'xmin': self.xmin, 'ymin': self.ymin, 'xmax': self.xmax, 'ymax': self.ymax, 'map_n': map_n, 'map_w': map_width, 'pix2mas': self.pix2mas, 'nbin': nbin}
            json.dump(infodict, open(filename+'_info.json', 'w'))
            print("Info Saved to %s" % filename+'_info.json')
            self.psfcam_binned_frames, self.num_frames, self.idxs = bin_by_centroids_to_file(filename, psfcam_frames, plcam_file_indices, plcam_frame_indices, self.plcam_files, self.ny, self.nx, centroids, xbins, ybins, skip_frame_reading=skip_frame_reading,
                                                                                             nbin = nbin)
            # return      
            
        # # this is used for bootstrap later
        # self.result_psfcam_frames = psfcam_frames
        # self.result_plcam_file_indices = plcam_file_indices
        # self.result_plcam_frame_indices = plcam_frame_indices
        # self.result_plcam_files = self.plcam_files
        # self.result_centroids = centroids
        # self.result_xbins = xbins
        # self.result_ybins = ybins


        if plot:

            fig = plt.figure(figsize=(5,5))
            plt.imshow(self.num_frames, origin='upper', extent = (self.xmin, self.xmax, self.ymin, self.ymax))
            plt.xlabel('x (mas)')
            plt.ylabel('y (mas)')
            plt.colorbar()
            plt.title('Number of frames averaged')
            
            if to_file:
                plt.savefig(filename+'_num_frames.png')
                print("Saved plot to %s" % filename+'_num_frames.png')
            else:
                plt.show()
        
        return self.psfcam_binned_frames, self.num_frames, self.idxs

        # if calculate_variance:

        #     if self.store_spec:

        #         if return_bootstrap_samples:
        #             self.var, self.normvar, self.bootstrap_samples = calculate_bootstrap_variance_map(plcam_frames, self.idxs, nbootstrap = nbootstrap, return_bootstrap_samples = True)
        #         else:
        #             self.var, self.normvar = calculate_bootstrap_variance_map(plcam_frames, self.idxs, nbootstrap = nbootstrap)
        #             self.bootstrap_samples = None

        #     else:
        #         raise ValueError("calculate_variance is not implemented for plcam frames stored in files. Please use store_spec = True to store the frames in memory.") 
                                                                                                                                                                                                           
        #         # for i in range(nbootstrap):
        #         #     psfcam_binned_frames, plcam_binned_frames, num_frames, idxs = bin_by_centroids_from_indices(psfcam_frames, plcam_file_indices, plcam_frame_indices, self.plcam_files, centroids, xbins, ybins,
                                                                                                                # bootstrap=True)

        # else:
        #     self.var = None
        #     self.normvar = None
        #     self.bootstrap_samples = None


    # def save_bootstrap_frames(self, filename, nbootstrap = 100):

    #     # assert self.store_spec is False, "compute_bootstrap_frames is not implemented for plcam frames stored in memory."

    #     for i in range(nbootstrap):
    #         self.psfcam_binned_frames, self.plcam_binned_frames, self.num_frames, self.idxs = bin_by_centroids_from_indices(self.result_psfcam_frames, 
    #                                                                                                                         self.result_plcam_file_indices, 
    #                                                                                                                         self.result_plcam_frame_indices,
    #                                                                                                                         self.result_plcam_files, 
    #                                                                                                                         self.result_centroids, 
    #                                                                                                                         self.result_xbins, 
    #                                                                                                                         self.result_ybins,
    #                                                                                                                         bootstrap=True)
        
    #         self.var = None
    #         self.bootstrap_samples = None
    #         self.save(filename + '_bootstrap_%d.fits' % i)
    #         print("Saved bootstrap frames to %s" % filename + '_bootstrap_%d.fits' % i)


    # def save(self, filename):
    #     '''
    #     Save the data to a fits file
        
    #     Parameters
    #     ----------
    #     filename: str
    #         name of the fits file
    #     '''
    #     header = fits.Header()
    #     header['XMIN'] = self.xmin
    #     header['XMAX'] = self.xmax
    #     header['YMIN'] = self.ymin
    #     header['YMAX'] = self.ymax
    #     header['MAP_N'] = self.map_n
    #     header['MAP_W'] = self.map_width

    #     hdu = fits.PrimaryHDU(self.plcam_binned_frames, header = header)
    #     hdu2 = fits.ImageHDU(self.num_frames, name = 'nframes')
    #     hdu3 = fits.ImageHDU(self.psfcam_binned_frames, name='psfcam')
    #     hdulist = fits.HDUList([hdu, hdu2, hdu3])
        
    #     if self.var is not None:
    #         hdu4 = fits.ImageHDU(self.var, name='var')
    #         hdu5 = fits.ImageHDU(self.normvar, name='normvar')
    #         hdulist.append(hdu4)
    #         hdulist.append(hdu5)

    #     hdulist.writeto(filename, overwrite=True)
    #     print("Saved to %s" % filename)

    #     if self.bootstrap_samples is not None:
    #         hdu6 = fits.PrimaryHDU(self.bootstrap_samples, header=header)
    #         hdu6.writeto(filename.replace('.fits', '_bootstrap.fits'), overwrite=True)
    #         print("Saved bootstrap samples to %s" % filename.replace('.fits', '_bootstrap.fits'))


def make_responsemaps(filename, footer = '_spec', nfib = 38, nwav = 200, psfframe_shape = (40,40), nboot = 50):
    '''
    Make response files from _spec.h5 files.

    Parameters
    ----------
    filename: str
        name of the file to read the spectra from (header) and save the response map to (filename+ '_couplingmap.fits')
    footer: str
        footer of the file to read the spectra from (e.g. '_spec')
    nfib: int
        number of fibers in the spectra
    nwav: int
        number of wavelengths in the spectra
    psfframe_shape: tuple
        shape of the PSF frames (default: (40, 40))
    nboot: int
        number of bootstrap samples to generate (default: 50)
    '''


    info = json.load(open(filename + '_info.json', 'r'))
    map_n = info['map_n']
    map_width = info['map_w']
    pix2mas = info['pix2mas']

    pos_mas = np.linspace(-map_width/2, map_width/2, map_n) * pix2mas

    # load the spectra
    specs = np.zeros((map_n, map_n, nfib, nwav))
    bootspecs = np.zeros((nboot, map_n, map_n, nfib, nwav))
    nframes = np.zeros((map_n, map_n))

    print("Reading spectra")
    for i in tqdm(range(map_n)):
        for j in range(map_n):
            mapfile = filename + '_bin_%d_%d' % (i, j) + footer + '.h5'
            if os.path.exists(mapfile):
                with h5py.File(mapfile, 'r') as f:
                    specs[i, j] = f['avgspec'][:]
                    bootspecs[:, i, j] = f['bootspecs'][:]
                    nframes[i, j] = f.attrs['num_frames']
            else:
                # If the file does not exist, fill with NaNs
                specs[i, j] = np.nan * np.ones((nfib, nwav))
                bootspecs[:, i, j] = np.nan * np.ones((nboot, nfib, nwav))
    
    print("Reading PSF frames")
    psfframes = np.zeros((map_n, map_n, *psfframe_shape))
    for i in tqdm(range(map_n)):
        for j in range(map_n):
            mapfile = filename + '_bin_%d_%d' % (i, j) + '.h5'
            if os.path.exists(mapfile):
                with h5py.File(mapfile, 'r') as f:
                    psfframes[i, j] = np.nanmean(f['psfframes'][:], axis=0)
            else:
                # If the file does not exist, fill with NaNs
                psfframes[i, j] = np.nan * np.ones(psfframe_shape)
    

    # Normalize over the map
    normspecs = specs / np.nansum(specs, axis=(0, 1))  # normalize over the map
    normbootspecs = bootspecs / np.nansum(bootspecs, axis=(1, 2))[:,None,None,:,:]  # normalize over the map

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


    hdul = fits.HDUList([hdu, hdu2, hdu3, hdu4, hdu5, hdu6])
    hdul.writeto(filename + '_couplingmap.fits', overwrite=True)
    print("remapped_couplingmap.fits saved in %s" % filename+ '_couplingmap.fits')

    return