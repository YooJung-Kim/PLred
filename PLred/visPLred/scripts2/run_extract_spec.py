from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys, glob
import h5py

from PLred.visPLred import spec as sp
from PLred.visPLred import utils as du
import PLred.visPLred.preprocess as pp


from configobj import ConfigObj
import yaml

from scipy.sparse import load_npz
import json

from queue import Queue
import multiprocessing


def box_extract_from_rawframe(rawframe, dark,
                     xmin, xmax,
                     traces,
                     nonlinearity_model = None,
                     nonlin_xmin = None,
                     nonlin_xmax = None): #, wav_map):

    im_to_correct = (rawframe - dark)#[:,xrange]

    if nonlinearity_model is not None:
        model = pp.DetectorNonlinearityModel(modelname =nonlinearity_model)
        corrected_map, status = model.correct_map(im_to_correct[:,nonlin_xmin:nonlin_xmax])#[:, xrange])

        # patch correction
        frame = im_to_correct.copy()
        frame[:,xmin:xmax] = corrected_map
    else:
        frame = im_to_correct.copy()

    spec_box = sp.frame_to_spec(frame, xmin, xmax, traces = traces)

    return spec_box


if __name__ == "__main__":

    configname = sys.argv[1]

    # Parse config

    config = ConfigObj(configname)
    binning_method = config['Coupling_map']['binning_method']

    name = config['Inputs']['name']
    remapped_path = config['Coupling_map']['output_dir'] + name + '/' + config['Coupling_map']['output_name'] + '_'+config['Coupling_map']['map_start_time'] + '_' + config['Coupling_map']['map_end_time'] + '/'
    nonlin_model_name = config['Spec']['nonlin_model']
    spec_model_name = config['Spec']['spec_model']
    trace_model_name = config['Spec']['trace_model']
    xmin = int(config['Spec']['xmin'])
    xmax = int(config['Spec']['xmax'])
    nbootstrap = int(config['Spec']['nbootstrap'])
    restart_from = int(config['Spec']['restart_from'])
    multiprocessing_n = int(config['Spec']['multiprocessing_n'])

    if binning_method == 'voronoi':
        n_clusters = json.load(open(remapped_path + 'remapped_info.json', 'r'))['n_clusters']

        mapfiles = [remapped_path+'remapped_voronoi_bin_%d.h5' % i for i in np.arange(restart_from, n_clusters)]
        print("%d number of remapped files" % len(mapfiles))

    elif binning_method == 'grid':
        map_n = json.load(open(remapped_path + 'remapped_info.json', 'r'))['map_n']
        mapfiles = glob.glob(remapped_path + 'remapped_grid_bin_*.h5')
        mapfiles.sort()
        mapfiles = mapfiles[restart_from:]
        print("%d number of remapped files" % len(mapfiles))

    if config['Inputs']['darkfile_name'].strip() != '':
        dark = fits.getdata(config['Inputs']['darkfile_name'])
        if len(np.shape(dark)) == 3:
            dark = np.nanmean(dark, axis=0)
    else:
        dark_files = du.find_data_between(config['Inputs']['path'] + config['Inputs']['obs_date'] + '/firstpl/', 
                                          config['Inputs']['dark_start_time'], config['Inputs']['dark_end_time'], 
                                          header='firstpl_', footer='.fits')
        dark, nframes = du.average_frames(dark_files)

    if config['Spec']['nonlin_model'].strip() != '':
        nonlincorr = True
    else:
        nonlincorr = False
    
    if config['Spec']['use_optimal'] == 'True':
        use_optimal = True
        # load spectrum model
        matrix = load_npz(spec_model_name + '_matrix.npz')
        # with open(spec_model_name + '_info.json', 'r') as f:
        info = np.load(spec_model_name+'_info.npy', allow_pickle=True).item() #json.load(f)
        wav_map = np.load(spec_model_name + '_wavmap.npy')
        spec_xmin, spec_xmax = info['xmin'], info['xmax']
        if xmin != spec_xmin or xmax != spec_xmax:
            raise ValueError("xmin and xmax requested spectral extraction and spectrum extraction model are different!")
        print("xmin, xmax", xmin, xmax)
        print("spec_xmin, spec_xmax", spec_xmin, spec_xmax)

    else:
        use_optimal = False
        trace_vals = np.load(trace_model_name + 'trace_vals.npy')
        trace_info = json.load(open(trace_model_name + 'trace_info.json', 'r'))
        trace_xmin = trace_info['xmin']
        if xmin - trace_xmin < 0 or xmax - trace_xmin > np.shape(trace_vals)[1]:
            raise ValueError("xmin and xmax for box spectrum extraction are not in the range of trace model!")
        traces = np.array(trace_vals)[:, xmin - trace_xmin:xmax - trace_xmin]
        print("trace file loaded")


    # for mapfile in tqdm(mapfiles):

    def process_mapfile(mapfile):

        if not os.path.exists(mapfile):
            print("file %s does not exist" % mapfile)
            return
        
        print("Loading %s" % mapfile)

        # load map
        with h5py.File(mapfile, 'r') as f:
            mapdata = f['rawframes'][:]

        nframes = mapdata.shape[0]

        out_specs = []

        avgframe = np.nanmean(mapdata, axis=0)
        bootframes = [avgframe]
        for i in range(nbootstrap):
            boot_ind = np.random.choice(nframes, nframes, replace=True)
            bootframes.append(np.nanmean(mapdata[boot_ind], axis=0))

        for frame in tqdm(bootframes):

            if nonlincorr:
                spec_box = box_extract_from_rawframe(frame, dark, 
                                                        xmin, xmax,
                                                        traces,
                                                        nonlinearity_model = nonlin_model_name,
                                                        nonlin_xmin = xmin,
                                                        nonlin_xmax = xmax)
            else:
                spec_box = sp.frame_to_spec(frame, xmin, xmax, traces = traces)

            out_specs.append(spec_box)
            
        avgspec = out_specs[0]
        bootspecs = out_specs[1:]

        out_spec_name = mapfile.replace('.h5', '_spec.h5')
        with h5py.File(out_spec_name, 'w') as h5file:
            avgspec_h5 = h5file.create_dataset('avgspec', data = avgspec)
            bootspec_h5 = h5file.create_dataset('bootspecs', data = np.array(bootspecs))
            h5file.attrs['num_frames'] = nframes
            h5file.attrs['spec_optimal_extract'] = use_optimal
            h5file.attrs['nonlinearity_model'] = nonlin_model_name
            h5file.attrs['trace_model'] = trace_model_name
            h5file.attrs['spec_model'] = spec_model_name
        
        print("Saved %s" % out_spec_name)
    

    with multiprocessing.Pool(processes=multiprocessing_n) as pool:
        pool.map(process_mapfile, mapfiles)
        