import numpy as np
import matplotlib.pyplot as plt

datadir = '/mnt/datazpool/PL/yjkim/remapped/WR140_20250515/grid_ini_12:35:22_12:50:21/'

corrected_map = np.load(datadir+'corrected_map_800_1500.npy')
status_map = np.load(datadir+'status_map_800_1500.npy')
outname = datadir+'specmap_800_1500.npy'

n_bin = 15

from PLred.visPLred import spec as sp
import h5py
from astropy.io import fits

modelname = './mizar'
sm = sp.SpectrumModel(modelname)
datadir2 = '/mnt/datazpool/PL/yjkim/remapped/mizarA_20250514/grid_ini_08:31:00_08:36:59/'
frame = h5py.File(datadir2 + 'remapped_08:31:00_08:36:59_bin_7_7.h5', 'r')
dark = fits.getdata('/mnt/datazpool/PL/20250514/darks/firstpl_15:30:26.681277758.fits')

sm.flat = np.average(frame['rawframes'][:],axis=0) - np.average(dark,axis=0)

neon_chunk = fits.getdata('/mnt/datazpool/PL/20250513/firstpl/firstpl_15:53:02.390432600.fits')
neon_dark_chunk = fits.getdata('/mnt/datazpool/PL/20250513/firstpl/firstpl_16:02:22.983506040.fits')
neon = np.mean(neon_chunk, axis=0) - np.mean(neon_dark_chunk, axis=0)

sm.neon = neon
sm.find_peaks()

ini_wav_ind=3
sm.trace_spectra(ini_wav_ind)


xmin = 800#-200
xmax = 1500#+300
specmap = np.zeros((n_bin,n_bin,38,xmax-xmin))
from tqdm import tqdm

for i in tqdm(range(n_bin)):
    for j in range(n_bin):
        cleaned_frame0 = np.zeros((412,1896))
        cleaned_frame0[:,xmin:xmax] += (corrected_map)[i,j].copy()

        spec_box = sp.frame_to_spec(cleaned_frame0, xmin, xmax, traces = np.array(sm.trace_vals)[:,xmin - sm.XMIN:xmax - sm.XMIN])
        specmap[i,j] = spec_box

np.save(outname, specmap)
print('Saved %s' % outname)