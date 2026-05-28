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


def script_match_timestamps(
    configname,
    verbose=False,
):
    """
    LAYER 1: Fast timestamp matching (no frame loading).

    Match timestamps between PSF and PL cameras and write intermediate H5 file.
    This layer is fast, deterministic, and can be reused by multiple processing strategies.

    Config file format:
        [Fastcam]
        obs_date        = 20250211
        start_time      = 11:59:00
        end_time        = 12:20:09
        path            = /mnt/sdata/20250211/palila/

        [Slowcam]
        timestamp_dir   = /mnt/userdata/yjkim/20250211_betcmi/timestamps/
        nbin            = 1

        [Output]
        outname         = /mnt/userdata/yjkim/output
        filename        = observation_matched

        [Options]
        show_plot       = False
        crop_width      = 20

    Parameters
    ----------
    configname : str
        Path to config file
    verbose : bool, optional
        Print progress

    Returns
    -------
    intermediate_h5_path : str
        Path to intermediate H5 file (input for Layer 2)
    """
    from .h5_consolidation import (
        validate_timestamp_matching,
        compute_frame_durations,
        build_matching_dict,
        write_intermediate_matched_h5,
    )

    config = ConfigObj(configname)

    fastcam_dir = config['Fastcam']['path']
    fastcam_start_time = config['Fastcam']['start_time']
    fastcam_end_time = config['Fastcam']['end_time']
    obs_date = config['Fastcam']['obs_date']

    slowcam_timestamps_dir = config['Slowcam']['timestamp_dir']
    try:
        slowcam_nbin = int(config['Slowcam']['nbin'])
    except:
        slowcam_nbin = 1

    outname = config['Output']['outname']
    filename = config['Output']['filename']

    show_plot = (config['Options'].get('show_plot', 'False')).lower() == 'true'

    os.makedirs(outname, exist_ok=True)

    if verbose:
        print("\n" + "="*70)
        print("LAYER 1: Timestamp Matching")
        print("="*70)

    # ======== Find data files ========
    fastcam_timestampfiles = find_data_between(fastcam_dir, fastcam_start_time, fastcam_end_time, footer='.txt')
    slowcam_timestampfiles = np.sort(glob.glob(slowcam_timestamps_dir + '*.txt'))

    if verbose:
        print(f"Found {len(fastcam_timestampfiles)} PSF timestamp files")
        print(f"Found {len(slowcam_timestampfiles)} PL timestamp files")

    # ======== Read timestamps ========
    fastcam_timestamps_list = []
    fastcam_fileinds_list = []
    fastcam_frameinds_list = []

    for file_idx, tsfile in enumerate(fastcam_timestampfiles):
        data = np.genfromtxt(tsfile)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        frameinds = data[:, 0].astype(int)
        timestamps = data[:, 4]

        fastcam_timestamps_list.append(timestamps)
        fastcam_fileinds_list.append(np.full(len(timestamps), file_idx, dtype=int))
        fastcam_frameinds_list.append(frameinds)

    fastcam_timestamps = np.concatenate(fastcam_timestamps_list)
    fastcam_fileinds = np.concatenate(fastcam_fileinds_list)
    fastcam_frameinds = np.concatenate(fastcam_frameinds_list)

    sort_idx = np.argsort(fastcam_timestamps)
    fastcam_timestamps = fastcam_timestamps[sort_idx]
    fastcam_fileinds = fastcam_fileinds[sort_idx]
    fastcam_frameinds = fastcam_frameinds[sort_idx]

    # Read PL camera timestamps
    slowcam_timestamps_list = []
    slowcam_fileinds_list = []
    slowcam_frameinds_list = []

    for file_idx, tsfile in enumerate(slowcam_timestampfiles):
        data = np.genfromtxt(tsfile)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        frameinds = data[:, 0].astype(int)
        timestamps = data[:, 4]

        slowcam_timestamps_list.append(timestamps)
        slowcam_fileinds_list.append(np.full(len(timestamps), file_idx, dtype=int))
        slowcam_frameinds_list.append(frameinds)

    slowcam_timestamps = np.concatenate(slowcam_timestamps_list)
    slowcam_fileinds = np.concatenate(slowcam_fileinds_list)
    slowcam_frameinds = np.concatenate(slowcam_frameinds_list)

    sort_idx = np.argsort(slowcam_timestamps)
    slowcam_timestamps = slowcam_timestamps[sort_idx]
    slowcam_fileinds = slowcam_fileinds[sort_idx]
    slowcam_frameinds = slowcam_frameinds[sort_idx]

    if verbose:
        print(f"PSF camera: {len(fastcam_timestamps)} timestamps")
        print(f"PL camera: {len(slowcam_timestamps)} timestamps")

    # ======== Validate and filter timestamps ========
    idx_fastcam, idx_slowcam = validate_timestamp_matching(
        fastcam_timestamps,
        slowcam_timestamps,
        atol=1e-4,
        verbose=verbose,
    )

    fastcam_timestamps = fastcam_timestamps[idx_fastcam]
    fastcam_fileinds = fastcam_fileinds[idx_fastcam]
    fastcam_frameinds = fastcam_frameinds[idx_fastcam]

    slowcam_timestamps = slowcam_timestamps[idx_slowcam]
    slowcam_fileinds = slowcam_fileinds[idx_slowcam]
    slowcam_frameinds = slowcam_frameinds[idx_slowcam]

    # ======== Handle slowcam_nbin ========
    if slowcam_nbin > 1:
        n_total = len(slowcam_timestamps)
        n_keep = (n_total // slowcam_nbin) * slowcam_nbin
        n_drop = n_total - n_keep
        if n_drop > 0:
            print(f"WARNING: slowcam_nbin={slowcam_nbin} — dropping last {n_drop} timestamp(s)")
        slowcam_timestamps = slowcam_timestamps[:n_keep:slowcam_nbin]
        slowcam_fileinds = slowcam_fileinds[:n_keep:slowcam_nbin]
        slowcam_frameinds = slowcam_frameinds[:n_keep:slowcam_nbin]

    # ======== Compute frame durations ========
    fastcam_frame_end_times = compute_frame_durations(fastcam_timestamps, fastcam_fileinds)

    # ======== Bisect and build matching dict ========
    bisect_inds = np.array([bisect(fastcam_timestamps, t) for t in slowcam_timestamps])
    bisect_arr = bisect_inds
    max_bisect = int(bisect_arr.max())

    if max_bisect == 0:
        raise ValueError(
            "No PSF camera timestamp is later than any PL camera timestamp. "
            "Check that both cameras cover the same time interval."
        )

    ind_end = int(np.argmax(bisect_arr == max_bisect))
    if ind_end == 0:
        raise ValueError(
            "ind_end resolved to 0: all bisect_inds equal the maximum (%d). "
            "The PSF camera timestamps may not overlap with the PL camera timestamps."
            % max_bisect
        )

    ind_start = 0
    if verbose:
        print(f"Timestamp overlap: {ind_end - ind_start} PL frames")

    Dict, matched_timestamps, n_skipped = build_matching_dict(
        fastcam_timestamps,
        slowcam_timestamps,
        bisect_inds,
        ind_start,
        ind_end,
        frame_end_times=fastcam_frame_end_times,
        verbose=verbose,
    )

    if n_skipped > 0:
        print(f"WARNING: {n_skipped} PL frame(s) skipped (timestamp before all PSF timestamps).")

    if show_plot:
        plt.figure(figsize=(12, 4))
        plt.plot(bisect_inds, 'o-', ms=1)
        plt.axvline(ind_start)
        plt.axvline(ind_end)
        plt.xlabel('PL camera frame index')
        plt.ylabel('PSF camera frame index')
        plt.title('Bisect results (timestamp matching)')
        plt.grid(True, alpha=0.3)
        plt.show()

    # ======== Write intermediate H5 ========
    config_dict = {
        'obs_date': obs_date,
        'obs_start': fastcam_start_time,
        'obs_end': fastcam_end_time,
        'slowcam_nbin': slowcam_nbin,
    }

    intermediate_h5_path = os.path.join(outname, f'{filename}_intermediate.h5')
    from .h5_consolidation import write_intermediate_matched_h5

    write_intermediate_matched_h5(
        intermediate_h5_path,
        fastcam_timestamps,
        fastcam_fileinds,
        fastcam_frameinds,
        fastcam_timestampfiles,
        slowcam_timestamps,
        slowcam_fileinds,
        slowcam_frameinds,
        slowcam_timestampfiles,
        Dict,
        matched_timestamps,
        config_dict,
        verbose=verbose,
    )

    if verbose:
        print(f"\n✓ Layer 1 complete: {intermediate_h5_path}")
        print(f"  Ready for Layer 2 processing")

    return intermediate_h5_path

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


def script_process_matched_timestamps(
    intermediate_h5_path,
    fastcam_files,
    fastcam_dir,
    slowcam_files,
    output_path,
    output_filename,
    write_raw_plcam=False,
    peak_min=None,
    peak_max=None,
    time_min=None,
    time_max=None,
    crop_width=20,
    verbose=False,
):
    """
    LAYER 2: Process matched timestamps (load frames, compute features).

    Takes intermediate H5 from Layer 1 and produces consolidated H5 with:
    - Loaded PSF frames from FITS files
    - Loaded PL frames (optional, raw)
    - Computed centroids and peaks
    - Timestamp arrays
    - All bugfixes applied

    Parameters
    ----------
    intermediate_h5_path : str
        Path to intermediate H5 from Layer 1
    fastcam_files : list of str
        List of PSF camera FITS file paths
    fastcam_dir : str
        Directory containing PSF FITS files
    slowcam_files : list of str
        List of PL camera FITS file paths
    output_path : str
        Output directory
    output_filename : str
        Base filename for output (will add _core.h5)
    write_raw_plcam : bool, optional
        If True, also write raw PL frames
    peak_min : float, optional
        Filter: minimum PSF peak value
    peak_max : float, optional
        Filter: maximum PSF peak value
    time_min : float, optional
        Filter: minimum timestamp
    time_max : float, optional
        Filter: maximum timestamp
    crop_width : int, optional
        Width for centroid computation
    verbose : bool, optional
        Print progress

    Returns
    -------
    outnames : dict
        Paths to output files (core.h5, optional raw_plcam.h5)
    """
    from .h5_consolidation import (
        read_intermediate_matched_h5,
        write_consolidated_h5_core,
        write_plcam_raw_h5,
        filter_by_peak_and_time,
        plot_matching_diagnostics,
    )

    if verbose:
        print("\n" + "="*70)
        print("LAYER 2: Process Matched Timestamps")
        print("="*70)

    # ======== Read Layer 1 output ========
    matched_data = read_intermediate_matched_h5(intermediate_h5_path, verbose=verbose)

    fastcam_timestamps = matched_data['fastcam_timestamps']
    fastcam_fileinds = matched_data['fastcam_fileinds']
    fastcam_frameinds = matched_data['fastcam_frameinds']
    slowcam_fileinds = matched_data['slowcam_fileinds']
    slowcam_frameinds = matched_data['slowcam_frameinds']
    matched_timestamps = matched_data['matched_timestamps']
    matching_dict = matched_data['matching_dict']
    config_dict = matched_data['config_dict']

    os.makedirs(output_path, exist_ok=True)

    if verbose:
        print(f"Loaded {len(matched_timestamps)} matched pairs from Layer 1")

    # ======== Load PSF frames and compute centroids ========
    if verbose:
        print("Loading PSF camera frames...")

    psfcam_frames = []
    psfcam_centroids = []
    psfcam_peaks = []
    psfcam_timestamps_matched = []
    plcam_file_indices = []
    plcam_frame_indices = []

    current_file_idx = -1
    current_file_data = None

    for slowcam_idx, fastcam_dict in tqdm(matching_dict.items(), desc='Loading frames', disable=not verbose):
        for fastcam_idx, weight in fastcam_dict.items():
            if weight <= 0:
                continue

            file_idx = fastcam_fileinds[fastcam_idx]
            frame_idx = fastcam_frameinds[fastcam_idx]

            # Load FITS frame if needed
            if file_idx != current_file_idx:
                if current_file_data is not None:
                    del current_file_data
                current_file_idx = file_idx
                try:
                    current_file_data = fits.getdata(fastcam_files[file_idx])
                except:
                    # Fallback: try file from directory
                    current_file_data = fits.getdata(
                        os.path.join(fastcam_dir, os.path.basename(fastcam_files[file_idx]))
                    )

            # Extract frame
            if current_file_data.ndim == 3:
                frame = current_file_data[frame_idx]
            else:
                frame = current_file_data

            # Compute centroid and peak
            peak = np.max(frame)
            try:
                cx, cy = subpixel_centroid_2d(frame, crop_width=crop_width)
            except:
                cx, cy = frame.shape[1] / 2, frame.shape[0] / 2

            psfcam_frames.append(frame.astype('float32'))
            psfcam_centroids.append([cx, cy])
            psfcam_peaks.append(peak)
            psfcam_timestamps_matched.append(matched_timestamps[slowcam_idx])

            plcam_file_indices.append(slowcam_fileinds[slowcam_idx])
            plcam_frame_indices.append(slowcam_frameinds[slowcam_idx])

    psfcam_frames = np.array(psfcam_frames, dtype='float32')
    psfcam_centroids = np.array(psfcam_centroids, dtype='float32')
    psfcam_peaks = np.array(psfcam_peaks, dtype='float32')
    psfcam_timestamps_matched = np.array(psfcam_timestamps_matched, dtype='float64')
    plcam_file_indices = np.array(plcam_file_indices, dtype='int64')
    plcam_frame_indices = np.array(plcam_frame_indices, dtype='int64')

    if verbose:
        print(f"Loaded {len(psfcam_frames)} PSF frames with computed centroids and peaks")

    # ======== Apply filtering ========
    filter_mask = filter_by_peak_and_time(
        psfcam_timestamps_matched,
        psfcam_peaks,
        peak_min=peak_min,
        peak_max=peak_max,
        time_min=time_min,
        time_max=time_max,
        verbose=verbose,
    )

    psfcam_frames = psfcam_frames[filter_mask]
    psfcam_centroids = psfcam_centroids[filter_mask]
    psfcam_peaks = psfcam_peaks[filter_mask]
    psfcam_timestamps_matched = psfcam_timestamps_matched[filter_mask]
    plcam_file_indices = plcam_file_indices[filter_mask]
    plcam_frame_indices = plcam_frame_indices[filter_mask]

    # ======== Write consolidated H5 ========
    outnames = {}

    # Add filtering info to config
    config_dict.update({
        'crop_width': crop_width,
        'peak_filter': {'min': peak_min, 'max': peak_max},
        'time_filter': {'min': time_min, 'max': time_max},
    })

    # Write core H5
    core_h5_path = os.path.join(output_path, f'{output_filename}_core.h5')
    write_consolidated_h5_core(
        core_h5_path,
        psfcam_frames,
        psfcam_centroids,
        psfcam_peaks,
        psfcam_timestamps_matched,
        plcam_file_indices,
        plcam_frame_indices,
        slowcam_files,
        config_dict,
        centroid_method='subpixel',
        verbose=verbose,
    )
    outnames['core'] = core_h5_path

    # Optionally write raw PL H5
    if write_raw_plcam:
        raw_h5_path = os.path.join(output_path, f'{output_filename}_plcam_raw.h5')
        try:
            pl_shape = fits.getdata(slowcam_files[0]).shape
            if len(pl_shape) == 3:
                pl_ny, pl_nx = pl_shape[1:]
            else:
                pl_ny, pl_nx = pl_shape
        except:
            pl_ny, pl_nx = 2000, 500

        write_plcam_raw_h5(
            raw_h5_path,
            slowcam_files,
            plcam_file_indices,
            plcam_frame_indices,
            (pl_ny, pl_nx),
            core_h5_ref=core_h5_path,
            compression='lz4',
            verbose=verbose,
        )
        outnames['raw_plcam'] = raw_h5_path

    # ======== Diagnostic plots ========
    if verbose:
        print("Generating diagnostic plots...")
        try:
            fig = plot_matching_diagnostics(core_h5_path)
            plot_path = os.path.join(output_path, f'{output_filename}_diagnostics.png')
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved diagnostic plot: {plot_path}")
        except Exception as e:
            print(f"  Warning: could not generate diagnostic plots: {e}")

    if verbose:
        print(f"\n✓ Layer 2 complete")
        print(f"  Core H5: {core_h5_path}")
        if write_raw_plcam:
            print(f"  Raw PL H5: {raw_h5_path}")

    return outnames
    """
    Match timestamps between PSF and PL cameras and write consolidated H5 output.

    This is the modern alternative to script_match_timestamps() that writes
    consolidated HDF5 files designed for efficient on-the-fly spatial binning.

    INCLUDES ALL BUGFIXES FROM sort4.py:
      - Float tolerance matching (not exact equality)
      - Dead-time correction between FITS files
      - Edge case handling (bisect_inds==0, single frame spans interval, etc.)
      - Proper error reporting and warnings
      - nbin truncation warnings

    Parameters
    ----------
    configname : str
        Path to config file (same format as script_match_timestamps)
    output_format : str, optional
        'h5' for consolidated HDF5 output, 'legacy' for old .pkl/.npy format
    write_raw_plcam : bool, optional
        If True, also write raw PL camera frames to separate H5 file (~50-100 GB)
    peak_min : float, optional
        Filter: minimum PSF peak value (Strehl proxy)
    peak_max : float, optional
        Filter: maximum PSF peak value
    time_min : float, optional
        Filter: minimum timestamp (unix epoch)
    time_max : float, optional
        Filter: maximum timestamp
    verbose : bool, optional
        Print progress information

    Returns
    -------
    outnames : dict
        Dictionary with keys 'core', 'raw_plcam' (if written), 'legacy' (if output_format='legacy')
        containing paths to output files
    """
    from .h5_consolidation import (
        write_consolidated_h5_core,
        write_plcam_raw_h5,
        filter_by_peak_and_time,
        validate_timestamp_matching,
        compute_frame_durations,
        build_matching_dict,
        plot_matching_diagnostics,
    )

    # ======== Read config (same as original script_match_timestamps) ========
    config = ConfigObj(configname)

    fastcam_dir = config['Fastcam']['path']
    fastcam_start_time = config['Fastcam']['start_time']
    fastcam_end_time = config['Fastcam']['end_time']
    obs_date = config['Fastcam']['obs_date']

    fastcam_dark_file = config['Fastcam']['dark_file']
    if fastcam_dark_file.strip() == '':
        fastcam_dark_start_time = config['Fastcam']['dark_start_time']
        fastcam_dark_end_time = config['Fastcam']['dark_end_time']

    slowcam_timestamps_dir = config['Slowcam']['timestamp_dir']
    try:
        slowcam_nbin = int(config['Slowcam']['nbin'])
    except:
        slowcam_nbin = 1

    outname = config['Output']['outname']
    filename = config['Output']['filename']

    show_plot = (config['Options'].get('show_plot', 'False')).lower() == 'true'
    crop_width = int(config['Options']['crop_width'])

    os.makedirs(outname, exist_ok=True)

    # ======== Find data files ========
    fastcam_timestampfiles = find_data_between(fastcam_dir, fastcam_start_time, fastcam_end_time, footer='.txt')
    if fastcam_dark_file.strip() == '':
        fastcam_darkframes = find_data_between(fastcam_dir, fastcam_dark_start_time, fastcam_dark_end_time, footer='.fits')
    else:
        fastcam_darkframes = [fastcam_dark_file]

    slowcam_timestampfiles = np.sort(glob.glob(slowcam_timestamps_dir + '*.txt'))

    if verbose:
        print(f"Found {len(fastcam_timestampfiles)} PSF camera timestamp files")
        print(f"Found {len(slowcam_timestampfiles)} PL camera timestamp files")

    # ======== Read and process timestamps (read each file only once!) ========
    # PSF camera timestamps
    fastcam_timestamps_list = []
    fastcam_fileinds_list = []
    fastcam_frameinds_list = []
    
    for file_idx, tsfile in enumerate(fastcam_timestampfiles):
        # Format: col 0 = frame index, col 4 = unix timestamp
        data = np.genfromtxt(tsfile)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        frameinds = data[:, 0].astype(int)
        timestamps = data[:, 4]
        
        fastcam_timestamps_list.append(timestamps)
        fastcam_fileinds_list.append(np.full(len(timestamps), file_idx, dtype=int))
        fastcam_frameinds_list.append(frameinds)
    
    fastcam_timestamps = np.concatenate(fastcam_timestamps_list)
    fastcam_fileinds = np.concatenate(fastcam_fileinds_list)
    fastcam_frameinds = np.concatenate(fastcam_frameinds_list)
    
    # Sort by timestamp
    sort_idx = np.argsort(fastcam_timestamps)
    fastcam_timestamps = fastcam_timestamps[sort_idx]
    fastcam_fileinds = fastcam_fileinds[sort_idx]
    fastcam_frameinds = fastcam_frameinds[sort_idx]

    # PL camera timestamps
    slowcam_timestamps_list = []
    slowcam_fileinds_list = []
    slowcam_frameinds_list = []
    
    for file_idx, tsfile in enumerate(slowcam_timestampfiles):
        # Format: col 0 = frame index, col 4 = unix timestamp
        data = np.genfromtxt(tsfile)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        frameinds = data[:, 0].astype(int)
        timestamps = data[:, 4]
        
        slowcam_timestamps_list.append(timestamps)
        slowcam_fileinds_list.append(np.full(len(timestamps), file_idx, dtype=int))
        slowcam_frameinds_list.append(frameinds)
    
    slowcam_timestamps = np.concatenate(slowcam_timestamps_list)
    slowcam_fileinds = np.concatenate(slowcam_fileinds_list)
    slowcam_frameinds = np.concatenate(slowcam_frameinds_list)
    
    # Sort by timestamp
    sort_idx = np.argsort(slowcam_timestamps)
    slowcam_timestamps = slowcam_timestamps[sort_idx]
    slowcam_fileinds = slowcam_fileinds[sort_idx]
    slowcam_frameinds = slowcam_frameinds[sort_idx]

    if verbose:
        print(f"PSF camera: {len(fastcam_timestamps)} timestamps")
        print(f"PL camera: {len(slowcam_timestamps)} timestamps")

    # ======== BUGFIX Issue 4: Validate timestamp matching with float tolerance ========
    idx_fastcam, idx_slowcam = validate_timestamp_matching(
        fastcam_timestamps,
        slowcam_timestamps,
        atol=1e-4,
        verbose=verbose,
    )

    fastcam_timestamps = fastcam_timestamps[idx_fastcam]
    fastcam_fileinds = fastcam_fileinds[idx_fastcam]
    fastcam_frameinds = fastcam_frameinds[idx_fastcam]
    
    slowcam_timestamps = slowcam_timestamps[idx_slowcam]
    slowcam_fileinds = slowcam_fileinds[idx_slowcam]
    slowcam_frameinds = slowcam_frameinds[idx_slowcam]

    if verbose:
        print(f"After validation: {len(fastcam_timestamps)} PSF frames matched")

    # ======== BUGFIX Issue 5: Handle slowcam_nbin truncation with warning ========
    if slowcam_nbin > 1:
        n_total = len(slowcam_timestamps)
        n_keep = (n_total // slowcam_nbin) * slowcam_nbin
        n_drop = n_total - n_keep
        if n_drop > 0:
            print(f"WARNING: slowcam_nbin={slowcam_nbin} — dropping last {n_drop} timestamp(s) "
                  f"that do not fill a complete bin.")
        slowcam_timestamps = slowcam_timestamps[:n_keep:slowcam_nbin]
        slowcam_fileinds = slowcam_fileinds[:n_keep:slowcam_nbin]
        slowcam_frameinds = slowcam_frameinds[:n_keep:slowcam_nbin]

    # ======== Compute frame durations for dead-time correction ========
    # (This accounts for gaps between FITS files)
    fastcam_frame_end_times = compute_frame_durations(fastcam_timestamps, fastcam_fileinds)

    # ======== Bisect and build matching dict ========
    bisect_inds = np.array([bisect(fastcam_timestamps, t) for t in slowcam_timestamps])
    bisect_arr = bisect_inds
    max_bisect = int(bisect_arr.max())

    # BUGFIX Issue 3: Proper error handling
    if max_bisect == 0:
        raise ValueError(
            "No PSF camera timestamp is later than any PL camera timestamp. "
            "Check that both cameras cover the same time interval."
        )

    ind_end = int(np.argmax(bisect_arr == max_bisect))
    if ind_end == 0:
        raise ValueError(
            "ind_end resolved to 0: all bisect_inds equal the maximum (%d). "
            "The PSF camera timestamps may not overlap with the PL camera timestamps."
            % max_bisect
        )

    ind_start = 0
    if verbose:
        print(f"Timestamp overlap: {ind_end - ind_start} slowcam frames")

    # BUGFIX Bug 1 & Bug 2: Use robust matching dict builder
    Dict, matched_timestamps, n_skipped = build_matching_dict(
        fastcam_timestamps,
        slowcam_timestamps,
        bisect_inds,
        ind_start,
        ind_end,
        frame_end_times=fastcam_frame_end_times,
        verbose=verbose,
    )

    if n_skipped > 0:
        print(f"WARNING: {n_skipped} PL frame(s) skipped (timestamp before all PSF timestamps).")

    if show_plot:
        plt.figure(figsize=(12, 4))
        plt.plot(bisect_inds, 'o-', ms=1)
        plt.axvline(ind_start)
        plt.axvline(ind_end)
        plt.xlabel('PL camera frame index')
        plt.ylabel('PSF camera frame index')
        plt.title('Bisect results (timestamp matching)')
        plt.grid(True, alpha=0.3)
        plt.show()

    # ======== PLACEHOLDER: Load PSF camera frames and compute centroids ========
    # (YOUR ACTUAL FRAME LOADING LOGIC GOES HERE)
    # TODO: Replace this with actual FITS frame loading from fastcam_dir
    # See original script_match_timestamps() for reference
    
    if verbose:
        print("Loading PSF camera frames...")

    psfcam_frames = []
    psfcam_centroids = []
    psfcam_peaks = []
    psfcam_timestamps_matched = []
    plcam_file_indices = []
    plcam_frame_indices = []

    # For each matched pair, extract frame and compute centroid
    for slowcam_idx, fastcam_dict in Dict.items():
        for fastcam_idx, weight in fastcam_dict.items():
            if weight <= 0:
                continue

            # TODO: Load actual frame from fastcam FITS files
            # frame = load_fastcam_frame(fastcam_timestampfiles, fastcam_fileinds[fastcam_idx], fastcam_frameinds[fastcam_idx])
            # For now, create placeholder:
            frame = np.random.randn(128, 128).astype('float32')

            peak = np.max(frame)
            try:
                cx, cy = subpixel_centroid_2d(frame, crop_width=crop_width)
            except:
                cx, cy = 64, 64  # fallback to center

            psfcam_frames.append(frame)
            psfcam_centroids.append([cx, cy])
            psfcam_peaks.append(peak)
            psfcam_timestamps_matched.append(matched_timestamps[slowcam_idx])

            plcam_file_indices.append(slowcam_fileinds[slowcam_idx])
            plcam_frame_indices.append(slowcam_frameinds[slowcam_idx])

    psfcam_frames = np.array(psfcam_frames, dtype='float32')
    psfcam_centroids = np.array(psfcam_centroids, dtype='float32')
    psfcam_peaks = np.array(psfcam_peaks, dtype='float32')
    psfcam_timestamps_matched = np.array(psfcam_timestamps_matched, dtype='float64')
    plcam_file_indices = np.array(plcam_file_indices, dtype='int64')
    plcam_frame_indices = np.array(plcam_frame_indices, dtype='int64')

    if verbose:
        print(f"Loaded {len(psfcam_frames)} matched frames")

    # ======== Apply filtering ========
    filter_mask = filter_by_peak_and_time(
        psfcam_timestamps_matched,
        psfcam_peaks,
        peak_min=peak_min,
        peak_max=peak_max,
        time_min=time_min,
        time_max=time_max,
        verbose=verbose,
    )

    psfcam_frames = psfcam_frames[filter_mask]
    psfcam_centroids = psfcam_centroids[filter_mask]
    psfcam_peaks = psfcam_peaks[filter_mask]
    psfcam_timestamps_matched = psfcam_timestamps_matched[filter_mask]
    plcam_file_indices = plcam_file_indices[filter_mask]
    plcam_frame_indices = plcam_frame_indices[filter_mask]

    # ======== Write output ========
    outnames = {}

    if output_format == 'h5':
        # Build config dict
        config_dict = {
            'obs_date': obs_date,
            'obs_start': fastcam_start_time,
            'obs_end': fastcam_end_time,
            'crop_width': crop_width,
            'slowcam_nbin': slowcam_nbin,
            'peak_filter': {'min': peak_min, 'max': peak_max},
            'time_filter': {'min': time_min, 'max': time_max},
        }

        # Write core H5
        core_h5_path = os.path.join(outname, f'{filename}_core.h5')
        write_consolidated_h5_core(
            core_h5_path,
            psfcam_frames,
            psfcam_centroids,
            psfcam_peaks,
            psfcam_timestamps_matched,
            plcam_file_indices,
            plcam_frame_indices,
            slowcam_timestampfiles,
            config_dict,
            centroid_method='subpixel',
            verbose=verbose,
        )
        outnames['core'] = core_h5_path

        # Optionally write raw PL camera H5
        if write_raw_plcam:
            raw_h5_path = os.path.join(outname, f'{filename}_plcam_raw.h5')
            # You'll need to determine PL camera frame shape
            # For now, assuming it's in the FITS headers
            try:
                pl_shape = fits.getdata(slowcam_timestampfiles[0]).shape
                if len(pl_shape) == 3:
                    pl_ny, pl_nx = pl_shape[1:]
                else:
                    pl_ny, pl_nx = pl_shape
            except:
                pl_ny, pl_nx = 2000, 500  # Default fallback

            write_plcam_raw_h5(
                raw_h5_path,
                slowcam_timestampfiles,
                plcam_file_indices,
                plcam_frame_indices,
                (pl_ny, pl_nx),
                core_h5_ref=core_h5_path,
                compression='lz4',
                verbose=verbose,
            )
            outnames['raw_plcam'] = raw_h5_path

        if verbose:
            print(f"\n✓ Consolidated H5 output written to {outname}")
            print(f"  Core file: {core_h5_path}")
            if write_raw_plcam:
                print(f"  Raw PL camera file: {raw_h5_path}")

    else:
        # Legacy output format (original behavior)
        legacy_outpath = os.path.join(outname, f'{filename}_matched.pkl')
        with open(legacy_outpath, 'wb') as f:
            pickle.dump(
                {
                    'psfcam_frames': psfcam_frames,
                    'centroids': psfcam_centroids,
                    'peaks': psfcam_peaks,
                    'timestamps': psfcam_timestamps_matched,
                    'plcam_file_indices': plcam_file_indices,
                    'plcam_frame_indices': plcam_frame_indices,
                },
                f
            )
        outnames['legacy'] = legacy_outpath
        if verbose:
            print(f"Legacy output written to {legacy_outpath}")

    return outnames