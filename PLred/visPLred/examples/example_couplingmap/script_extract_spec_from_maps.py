import PLred.visPLred.spec as sp
from scipy.sparse import load_npz
import json
import numpy as np
import PLred.visPLred.preprocess as pp
from astropy.io import fits
from tqdm import tqdm
import os

matrix = load_npz('/home/first/yjkim/specmodels/2025-02-07/model_decemberneon/model_matrix.npz')
info = np.load('/home/first/yjkim/specmodels/2025-02-07/model_decemberneon/model_info.npy', allow_pickle=True).item()
xmin, xmax = info['xmin'], info['xmax']
wav_map = np.load('/home/first/yjkim/specmodels/2025-02-07/model_decemberneon/model_wavmap.npy')

mapdata_dir = '/mnt/datazpool/PL/yjkim/reduced_map/betcmi_20250211/'
mapdata_name = mapdata_dir+'betcmi_20250211_couplingmap_frames_12:11:00_12:19:09'


xrange = np.arange(412)
yrange = np.arange(xmin,xmax)

nbootstrap = 38

# modelfile = '/mnt/datazpool/PL/yjkim/flat_characterization/2025-02-11/flux_dependent_flat/model_nonlinearity.fits'
modelfile = '/home/first/yjkim/PLred/PLred/visPLred/examples/example_nonlinearity_correction/model_nonlinearity_new.fits'
# modelfile = '/home/first/yjkim/PLred/PLred/visPLred/examples/example_nonlinearity_correction/model_nonlinearity_cutmin100.fits'
dark = fits.getdata('/mnt/datazpool/PL/yjkim/reduced/betcmi_20250211_fin/dark.fits')

avg_name = f'{mapdata_name}.fits'
bootstrap_name = [f'{mapdata_name}_bootstrap_{i}.fits' for i in range(nbootstrap)]

def extract_spec_from_map(data):

    # dark subtraction
    darksub = data - dark[None, None, :, :]
    
    corrected_map = np.zeros_like(darksub)
    flags = np.zeros_like(darksub, dtype=int)
    specmaps = np.zeros((darksub.shape[0], darksub.shape[1], 38, xmax-xmin))

    for i in tqdm(range(darksub.shape[0])):
        for j in range(darksub.shape[1]):

            # nonlinearity correction
            corrected_map[i, j], flags[i, j] = pp.correct_nonlinearity_map(darksub[i, j], modelfile, xrange, yrange)
    
            # bad pixel flagging
            badpix = np.ones_like(flags[i, j], dtype=bool)
            badpix[flags[i, j] == 0] = False

            # clean the map
            cleaned_frame = (corrected_map[i, j]).copy()
            cleaned_frame[cleaned_frame*0 != 0] = 0
            cleaned_frame[cleaned_frame < 0] = 0

            # spectral extraction

            spec, res = sp.frame_to_spec(cleaned_frame, xmin, xmax, wav_map, matrix, return_residual=True,
                                         badpix=badpix)
            
            specmaps[i, j] = spec

    return specmaps

# extract spec from avg maps
if os.path.exists(avg_name.replace('.fits', '_spec2.fits')):
    specmaps_avg = fits.getdata(avg_name.replace('.fits', '_spec2.fits'))
    print("skipping average spectrum extraction")
else:
    specmaps_avg = extract_spec_from_map(fits.getdata(avg_name))
    # save the average spectrum
    fits.writeto(avg_name.replace('.fits', '_spec2.fits'), specmaps_avg, overwrite=True)

# extract spec from bootstrap maps
specmaps_bootstrap = []
for i in tqdm(range(nbootstrap)):

    if os.path.exists(bootstrap_name[i].replace('.fits', '_spec2.fits')):
        specmaps_bootstrap.append(fits.getdata(bootstrap_name[i].replace('.fits', '_spec2.fits')))
        print("skipping bootstrap spectrum extraction for ", bootstrap_name[i])
    
    else:
        specmaps_bootstrap.append(extract_spec_from_map(fits.getdata(bootstrap_name[i])))
        # save the bootstrap spectrum
        fits.writeto(bootstrap_name[i].replace('.fits', '_spec2.fits'), specmaps_bootstrap[-1], overwrite=True)

# compute variance
specmaps_var = np.nanvar(specmaps_bootstrap, axis=0)

# calculate normalized bootstrap frames
bootstrap_normframes = np.zeros_like(specmaps_bootstrap)
for k in range(nbootstrap):
    bootstrap_normframes[k] = specmaps_bootstrap[k] / np.nansum(specmaps_bootstrap[k], axis=(0,1))[None,None,:,:]

specmaps_normvar = np.nanvar(bootstrap_normframes, axis=0)

# save
hdu = fits.PrimaryHDU(specmaps_avg)
hdu.header = fits.getheader(avg_name)
hdu.header['nonlin'] = 'corrected'

hdu2 = fits.ImageHDU(fits.getdata(avg_name, ext=1), name='nframes')
hdu3 = fits.ImageHDU(fits.getdata(avg_name, ext=2), name='psfcam')

hdu4 = fits.ImageHDU(specmaps_var, name='var')
hdu5 = fits.ImageHDU(specmaps_normvar, name='normvar')

hdulist = fits.HDUList([hdu, hdu2, hdu3, hdu4, hdu5])
hdulist.writeto(f'{mapdata_name}_combined2.fits', overwrite=True)

