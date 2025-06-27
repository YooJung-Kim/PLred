import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os, glob
from PLred.visPLred.preprocess import NonlinearityFitter, DetectorNonlinearityModel
# from nonlinearity_curve_fitter import NonlinearityFitter, DetectorNonlinearityModel
import h5py
from tqdm import tqdm

modeldir = '/mnt/datazpool/PL/yjkim/flat_characterization/2025-05-12/flux_dependent_flat/'
datadir = '/mnt/datazpool/PL/yjkim/remapped/mizarB2_20250514_highres/'
# data = fits.getdata('/mnt/datazpool/PL/yjkim/reduced_map/betcmi_20250211/betcmi_20250211_couplingmap_frames_12:11:00_12:19:09.fits')
dark = fits.getdata('/mnt/datazpool/PL/20250514/darks/firstpl_15:30:26.681277758.fits')
dark = np.average(dark, axis=0)
# frame = data -dark[None,None,:,:]

xmin = 800
xmax = 1500

corrected_map = np.zeros((25, 25, 412, xmax-xmin)) #[:,:,:,1100:1300]
status = np.zeros((25, 25, 412, xmax-xmin), dtype=int) #[:,:,:,1100:1300]
for i in tqdm(range(25)):
    for j in range(25):

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
np.save(datadir+f'corrected_map_mizarB2_highres_{xmin}_{xmax}', corrected_map)
np.save(datadir+f'status_map_mizarB2_highres_{xmin}_{xmax}', status)

