from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys

from PLred.visPLred import spec as sp
from PLred.visPLred import utils as du


from configobj import ConfigObj
import yaml

from scipy.sparse import load_npz
import json

from queue import Queue
from multiprocessing import Process

class YAMLLogger:
    def __init__(self, output_file):
        """
        Initializes the YAML logger.

        Parameters:
        ----------
        output_file : str
            Path to the YAML file for storing reduction results.
        """
        self.output_file = output_file
        self.data = {'input': {}, 'output': [],
                     'spectral_extraction_params': {}}

    def log_target_files(self, files):
        """Logs target files to the YAML file."""
        self.data["input"]["target_files"] = files
        self._write()
    
    def log_dark_files(self, files):
        """Logs dark files to the YAML file."""
        self.data["input"]["dark_files"] = files
        self._write()
    
    def log_reduced_files(self, file):
        self.data['output'].append(file)
        self._write()
    
    # def log_reduced_file(self, file_path):
    #     """Logs a reduced file to the YAML file."""
    #     self.data["ReducedFiles"].append(file_path)
    #     self._write()

    # def log_spectral_extraction_params(self, params):
    #     """Logs spectral extraction parameters."""
    #     self.data["SpectralExtraction"] = params
    #     self._write()

    def _write(self):
        """Writes the current data to the YAML file."""
        with open(self.output_file, "w") as yaml_file:
            yaml.dump(self.data, yaml_file)


if __name__ == "__main__":

    configname = sys.argv[1]

    # Parse config

    config = ConfigObj(configname)

    path_input = config['Inputs']['path']
    obs_date = config['Inputs']['obs_date']
    name = config['Inputs']['name']
    target_start_time = config['Inputs']['target_start_time']
    target_end_time = config['Inputs']['target_end_time']
    dark_start_time = config['Inputs']['dark_start_time']
    dark_end_time = config['Inputs']['dark_end_time']

    output_dir = config['Outputs']['output_dir']
    spec_model_name = config['Spec']['spec_model']


    # Make directories
    os.makedirs(output_dir+name, exist_ok=True)

    # set up info config file
    info_config = YAMLLogger(output_dir+name+'/info.yaml')

    # Find data
    target_files = du.find_data_between(path_input + obs_date + '/firstpl/', target_start_time, target_end_time, header='firstpl_', footer='.fits')
    dark_files = du.find_data_between(path_input + obs_date + '/firstpl/', dark_start_time, dark_end_time, header='firstpl_', footer='.fits')
    
    info_config.log_target_files(target_files)
    info_config.log_dark_files(list(dark_files))

    # load spectrum model
    matrix = load_npz(spec_model_name + '_matrix.npz')
    # with open(spec_model_name + '_info.json', 'r') as f:
    info = np.load(spec_model_name+'_info.npy', allow_pickle=True).item() #json.load(f)
    wav_map = np.load(spec_model_name + '_wavmap.npy')
    xmin, xmax = info['xmin'], info['xmax']

    # average dark
    dark, nframes = du.average_frames(dark_files)
    hdu = fits.PrimaryHDU(dark)
    hdu.header['files'] = str(dark_files)
    hdu.header['nframes'] = int(nframes)
    hdu.writeto(output_dir+name+'/dark.fits', overwrite=True)
    
    # extract spectrum
    # possibly do multiprocessing
    for file in tqdm(target_files):

        data = fits.getdata(file)
        header = fits.getheader(file)

        out_specs = []
        out_res = []

        for i in range(len(data)):

            spec, res = sp.frame_to_spec(data[i] - dark, xmin, xmax, wav_map, matrix, return_residual = True)
            out_specs.append(spec)
            out_res.append(res)

        # plt.plot(np.sum(out_specs, axis=(0,1)))
        # plt.show()
        hdu = fits.PrimaryHDU(np.array(out_specs))
        hdu.header['dark'] = output_dir+name+'/dark.fits'
        hdu.header['model'] = spec_model_name
        # add more info here, related to extraction params
        # np.save(output_dir+name+'/'+file.split('/')[-1].split('.fits')[0]+'_spec.npy', out_specs)
        hdu.writeto(output_dir+name+'/'+file.split('/')[-1].split('.fits')[0]+'_spec.fits', overwrite=True)
        info_config.log_reduced_files(file.split('/')[-1].split('.fits')[0]+'_spec.fits')

        hdu2 = fits.PrimaryHDU(np.nanmean(np.array(out_res), axis=0))
        hdu2.writeto(output_dir+name+'/'+file.split('/')[-1].split('.fits')[0]+'_res.fits', overwrite=True)



    # info_config.data['dark_files'] = dark_files
    # info_config['target_files'] = target_files
    # info_config['dark_files'] = dark_files

    # info_config.write()

    # files = du.find_data_between('/mnt/datazpool/PL/20241219/', '14:27:00', '14:27:02', header='firstpl_', footer='.fits')