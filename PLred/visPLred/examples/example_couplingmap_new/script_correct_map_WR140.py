import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os, glob
from PLred.visPLred.preprocess import NonlinearityFitter, DetectorNonlinearityModel
# from nonlinearity_curve_fitter import NonlinearityFitter, DetectorNonlinearityModel
import h5py
from tqdm import tqdm

modeldir = '/mnt/datazpool/PL/yjkim/flat_characterization/2025-05-12/flux_dependent_flat/'
datadir = '/mnt/datazpool/PL/yjkim/remapped/WR140_20250515/grid_ini_12:35:22_12:50:21/'
# data = fits.getdata('/mnt/datazpool/PL/yjkim/reduced_map/betcmi_20250211/betcmi_20250211_couplingmap_frames_12:11:00_12:19:09.fits')
dark = fits.getdata('/mnt/datazpool/PL/yjkim/reduced/WR140_20250515/dark/firstpl_12:50:22.545780104_modified.fits')
dark = np.average(dark, axis=0)
# frame = data -dark[None,None,:,:]

xmin = 800
xmax = 1500

map_n = 15

corrected_map = np.zeros((map_n, map_n, 412, xmax-xmin)) #[:,:,:,1100:1300]
status = np.zeros((map_n, map_n, 412, xmax-xmin), dtype=int) #[:,:,:,1100:1300]
for i in tqdm(range(map_n)):
    for j in range(map_n):

        if os.path.exists(datadir+'remapped_bin_%d_%d.h5' % (i,j)):
            print('Loading %d %d' % (i,j))
            h5file = h5py.File(datadir+'remapped_bin_%d_%d.h5' % (i,j), 'r')
            h = h5file['rawframes'][:]
            im_to_correct = np.average(h[:,:,xmin:xmax], axis=0) - dark[:,xmin:xmax]

        
            model = DetectorNonlinearityModel(modelname = modeldir+'model3_800_1500')
            # model = DetectorNonlinearityModel(modelname = 'model')
            corrected_map[i,j], status[i,j] = model.correct_map(im_to_correct)
            h5file.close()

        else:
            print('No file %d %d' % (i,j))
            corrected_map[i,j] = np.zeros((412, xmax-xmin))
            status[i,j] = np.zeros((412, xmax-xmin), dtype=int)
np.save(datadir+f'corrected_map_{xmin}_{xmax}', corrected_map)
np.save(datadir+f'status_map_{xmin}_{xmax}', status)

