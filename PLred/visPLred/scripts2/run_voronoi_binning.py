
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from PLred.visPLred import couplingmap as cm
from configobj import ConfigObj
import sys, os

if __name__ == "__main__":

    configname = sys.argv[1]

    # Parse config

    config = ConfigObj(configname)

    binning_method = config['Coupling_map']['binning_method']
    assert binning_method == 'voronoi', "This script is for Voronoi binning only."
    path_input = config['Inputs']['path']
    # output_dir = config['Outputs']['output_dir']
    obs_date = config['Inputs']['obs_date']
    name = config['Inputs']['name']

    firstcam_timestamp_path = path_input+obs_date+'/firstpl/'
    firstcam_spec_path =   path_input+obs_date+'/firstpl/'

    obs_start = config['Coupling_map']['map_start_time']
    obs_end = config['Coupling_map']['map_end_time']

    psfcam = config['Coupling_map']['PSFcam']

    psfcam_frames_name = config['Timestamp_matching']['output_dir'] + '/first_palila_matched_fastcam_matched_frames.npy'
    psfcam_timestamp_name = config['Timestamp_matching']['output_dir'] + '/first_palila_matched.pkl'

    width_pix = float(config['Coupling_map']['width_pix'])
    n_per_bin = int(config['Coupling_map']['n_per_bin'])
    map_output_dir = config['Coupling_map']['output_dir'] + name + '/'
    map_output_name = config['Coupling_map']['output_name']+'_'+obs_start+'_'+obs_end + '/'
    threshold = float(config['Coupling_map']['threshold'])

    os.makedirs(map_output_dir, exist_ok=True)
    os.makedirs(map_output_dir + map_output_name, exist_ok=True)


    # Define SimultaneousData object

    sd = cm.SimultaneousData(firstcam_timestamp_path, firstcam_spec_path,
                    obs_start, obs_end,
                    psfcam, 
                    psfcam_frames_name, psfcam_timestamp_name,
                    store_spec = False)
    
    sd.compute_psfcam_centroids()

    # threshold by PSF peak count (~ Strehl)
    idx = np.nanmax(sd.psfcam_frames, axis=(1,2)) > threshold

    # compute centroids and binning
    sd.compute_psfcam_centroids(peak=True)
    sd.make_voronoi_binning(n_per_bin = n_per_bin, map_width = width_pix, effective_idx = idx,
                        savedir = map_output_dir + map_output_name)
    
    sd.remap_frames_voronoi(map_output_dir + map_output_name + 'remapped')

