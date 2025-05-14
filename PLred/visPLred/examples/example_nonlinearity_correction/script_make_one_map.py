outname = 'test1'
remake = False

from scipy.ndimage import median_filter
from scipy.stats import binned_statistic
from astropy.io import fits
from tqdm import tqdm
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import PLred.visPLred.spec as sp

dir = '/Volumes/T7 Touch/firstcam_flats_2025-02-11_mean_std/'
logrec = []
files = glob.glob(dir+'*_mean_std.fits')

xrange = np.arange(412) #[53]
yrange = np.arange(1100, 1300)

maps = fits.getdata('/Volumes/T7 Touch/firstpl_testmaps/betcmi_20250211_couplingmap_frames_12:11:00_12:19:09.fits')
dark = fits.getdata('/Volumes/T7 Touch/firstpl_testmaps/dark.fits')
frames = maps - dark[None,None,:,:]

xs = 7 #np.array([7])
ys = 7 #np.array([7])

frames = frames[xs, ys]

matrix = load_npz('/Volumes/T7 Touch/specmodels/model_matrix.npz')
info = np.load('/Volumes/T7 Touch/specmodels/model_info.npy', allow_pickle=True).item()
xmin, xmax = info['xmin'], info['xmax']
wav_map = np.load('/Volumes/T7 Touch/specmodels/model_wavmap.npy')


import PLred.visPLred.preprocess as pp
import importlib
importlib.reload(pp)

if os.path.exists(f'{outname}.fits') and not remake:
    print(f'{outname}.fits already exists. Use remake=True to overwrite.')
    all_poly_results = fits.getdata(f'{outname}.fits')
else:
    all_poly_results, minvals, maxvals = pp.model_nonlinearity_from_flats(files, outname=f'{outname}.fits')

corrected_map = np.zeros_like(frames)
flags = np.zeros_like(frames, dtype=int)
specmaps = np.zeros((38, xmax-xmin))
# specmaps = np.zeros((frames.shape[0], frames.shape[1], 38, xmax-xmin))


# for i in tqdm(range(frames.shape[0])):

#     for j in range(frames.shape[1]):

# nonlinearity correction
corrected_map, flags = pp.correct_nonlinearity_map(frames, f'{outname}.fits', xrange, yrange)
# corrected_map[i, j], flags[i, j] = pp.correct_nonlinearity_map(frames[i, j], f'{outname}.fits', xrange, yrange)

# bad pixel flagging
badpix = np.ones_like(flags, dtype=bool)
badpix[flags == 0] = False

# badpix = np.ones_like(flags[i, j], dtype=bool)
# badpix[flags[i, j] == 0] = False

# clean the map
cleaned_frame = (corrected_map).copy()
# cleaned_frame = (corrected_map[i, j]).copy()
cleaned_frame[cleaned_frame*0 != 0] = 0
cleaned_frame[cleaned_frame < 0] = 0

# spectral extraction

spec, res = sp.frame_to_spec(cleaned_frame, xmin, xmax, wav_map, matrix, return_residual=True,
                                badpix=badpix)

specmaps = spec

spec, res = sp.frame_to_spec(frames, xmin, xmax, wav_map, matrix, return_residual=True,
                                badpix=badpix)

specmaps0 = spec
# specmaps[i, j] = spec

# save the corrected map
np.save(f'{outname}_corrected_map.npy', corrected_map)
np.save(f'{outname}_flags.npy', flags)
np.save(f'{outname}_specmaps.npy', specmaps)
np.save(f'{outname}_specmaps0.npy', specmaps0)

