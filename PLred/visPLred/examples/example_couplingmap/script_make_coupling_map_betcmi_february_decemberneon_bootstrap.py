
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from PLred.visPLred import couplingmap as cm

firstcam_timestamp_path = '/mnt/datazpool/PL/20250211/firstpl/'
firstcam_spec_path =      '/home/first/yjkim/reduced/betcmi_20250211_decemberneon/'
obs_start = '12:11:00'
obs_end   = '12:19:09'

# obs_starts = ['14:35:00', '14:45:00', '14:45:00', '14:55:00']
# obs_ends = ['14:45:00', '14:55:00', '14:55:00', '15:10:00']

psfcam = 'palila'
psfcam_frames_name = '/home/first/yjkim/reduced/betcmi_20250211/betcmi_20250211/first_palila_matched_fastcam_matched_frames.npy'
psfcam_timestamp_name = '/home/first/yjkim/reduced/betcmi_20250211/betcmi_20250211/first_palila_matched.pkl'


sd = cm.SimultaneousData(firstcam_timestamp_path, firstcam_spec_path,
                obs_start, obs_end,
                psfcam, 
                psfcam_frames_name, psfcam_timestamp_name)


sd.compute_psfcam_centroids(peak=True)
sd.bin_by_centroids(15, 3, calculate_variance = True, return_bootstrap_samples= True)
sd.save(f'betcmi_20250211_decemberneon_couplingmap_{obs_start}_{obs_end}.fits')
