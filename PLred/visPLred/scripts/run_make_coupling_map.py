
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from PLred.visPLred import couplingmap as cm
from configobj import ConfigObj
import sys

if __name__ == "__main__":

    configname = sys.argv[1]

    # Parse config

    config = ConfigObj(configname)
    path_input = config['Inputs']['path']
    output_dir = config['Outputs']['output_dir']
    obs_date = config['Inputs']['obs_date']
    name = config['Inputs']['name']

    firstcam_timestamp_path = path_input+obs_date+'/firstpl/'
    firstcam_spec_path =      output_dir + name + '/' #/betcmi_20240917/'

    obs_start = config['Coupling_map']['start_time']
    obs_end = config['Coupling_map']['end_time']

    psfcam = config['Coupling_map']['PSFcam']

    psfcam_frames_name = config['Coupling_map']['PSFcam_frames']
    psfcam_timestamp_name = config['Coupling_map']['timestamp_pkl_file']

    nbins = int(config['Coupling_map']['nbins'])
    centroid_width = float(config['Coupling_map']['centroid_width'])
    nbootstrap = int(config['Coupling_map']['nbootstrap'])
    threshold = float(config['Coupling_map']['threshold'])

    map_output_dir = config['Coupling_map']['output_dir']
    map_output_name = config['Coupling_map']['output_name']

    # Define SimultaneousData object

    sd = cm.SimultaneousData(firstcam_timestamp_path, firstcam_spec_path,
                    obs_start, obs_end,
                    psfcam, 
                    psfcam_frames_name, psfcam_timestamp_name)

    # threshold by PSF peak count (~ Strehl)
    idx = np.nanmax(sd.psfcam_frames, axis=(1,2)) > threshold

    # compute centroids and binning
    sd.compute_psfcam_centroids(peak=True)
    sd.bin_by_centroids(nbins, centroid_width, effective_idx = idx,
                        calculate_variance = True, 
                        return_bootstrap_samples= True, nbootstrap=nbootstrap)
    
    # save
    sd.save(f'{map_output_dir}{name}_{map_output_name}_{obs_start}_{obs_end}.fits')
