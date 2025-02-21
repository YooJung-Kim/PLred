import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from PLred.visPLred import couplingmap as cm

firstcam_timestamp_path = '/mnt/datazpool/PL/20240917/firstpl/'
firstcam_spec_path =      '/home/first/yjkim/reduced/kitalpha_20240917/'
obs_start = '08:32:13'
obs_end   = '08:37:55'


psfcam = 'palila'
psfcam_frames_name = '/home/first/yjkim/reduced/kitalpha_20240917/first_palila_matched_fastcam_matched_frames.npy'
psfcam_timestamp_name = '/home/first/yjkim/reduced/kitalpha_20240917/first_palila_matched.pkl'

sd = cm.SimultaneousData(firstcam_timestamp_path, firstcam_spec_path,
                 obs_start, obs_end,
                 psfcam, 
                 psfcam_frames_name, psfcam_timestamp_name)

sd.compute_psfcam_centroids(peak=True)


sd.bin_by_centroids(31, 3, calculate_variance = True)
sd.save('kitalpha_20240918_couplingmap_fine.fits')


threshold = 12000
idx = np.nanmax(sd.psfcam_frames, axis=(1,2)) > threshold

sd.bin_by_centroids(15, 3, effective_idx= idx, calculate_variance = True)
sd.save('kitalpha_20240918_couplingmap_thres%d.fits' % threshold)
