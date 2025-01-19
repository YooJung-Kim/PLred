import numpy as np


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
    