
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from PLred.visPLred import couplingmap as cm

firstcam_timestamp_path = '/mnt/datazpool/PL/yjkim/binned_observation/mizarA_20250514/20250514/firstpl/'
firstcam_spec_path =      '/mnt/datazpool/PL/yjkim/binned_observation/mizarA_20250514/20250514/firstpl/'

# obs_start = '08:31:00'
# obs_end   = '08:36:59' #'08:36:59'

# obs_starts = ['14:35:00', '14:45:00', '14:45:00', '14:55:00']
# obs_ends = ['14:45:00', '14:55:00', '14:55:00', '15:10:00']

psfcam = 'palila'
psfcam_frames_name = '/mnt/datazpool/PL/yjkim/reduced/mizarA_binned_20250514/first_palila_matched_fastcam_matched_frames.npy'
psfcam_timestamp_name = '/mnt/datazpool/PL/yjkim/reduced/mizarA_binned_20250514/first_palila_matched.pkl'



obs_start = '08:31:00'
obs_end   = '08:36:59' #'08:36:59'

outdir = '/mnt/datazpool/PL/yjkim/remapped_voronoi/'

sd = cm.SimultaneousData(firstcam_timestamp_path, firstcam_spec_path,
                obs_start, obs_end,
                psfcam, 
                psfcam_frames_name, psfcam_timestamp_name,
                store_spec = False,)

sd.compute_psfcam_centroids()
sd.make_voronoi_binning(n_per_bin = 20, savedir = 'voronoi_test')
sd.remap_frames_voronoi(outdir)