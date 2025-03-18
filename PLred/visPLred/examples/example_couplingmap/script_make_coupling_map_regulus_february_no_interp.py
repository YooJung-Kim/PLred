
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from PLred.visPLred import couplingmap as cm

firstcam_timestamp_path = '/mnt/datazpool/PL/20250211/firstpl/'
firstcam_spec_path =      '/home/first/yjkim/reduced/regulus_20250211_decemberneon2_no_wav_interp/'
obs_start = '12:40:00'
obs_end   = '12:51:56'


obs_starts = [obs_start]
obs_ends = [obs_end]

psfcam = 'palila'
psfcam_frames_name = '/home/first/yjkim/reduced/regulus_20250211/regulus_20250211/first_palila_matched_fastcam_matched_frames.npy'
psfcam_timestamp_name = '/home/first/yjkim/reduced/regulus_20250211/regulus_20250211/first_palila_matched.pkl'

for (obs_start, obs_end) in zip(obs_starts, obs_ends):
    
    sd = cm.SimultaneousData(firstcam_timestamp_path, firstcam_spec_path,
                    obs_start, obs_end,
                    psfcam, 
                    psfcam_frames_name, psfcam_timestamp_name)


    sd.compute_psfcam_centroids(peak=True)
    sd.bin_by_centroids(15, 3, calculate_variance = True)
    sd.save(f'regulus_20250211_couplingmap_no_wav_interp_{obs_start}_{obs_end}.fits')

    # threshold = 7500
    # idx = np.nanmax(sd.psfcam_frames, axis=(1,2)) > threshold
    # sd.bin_by_centroids(15, 3, effective_idx = idx, calculate_variance = True)
    # sd.save(f'regulus_20250211_couplingmap_{obs_start}_{obs_end}_thres{threshold}.fits')
