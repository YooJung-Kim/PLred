import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm

import PLred.mapmodel as mm

# couplingmap_data = '/home/first/yjkim/flat_characterization/2025-02-11/flux_dependent_flat/corrected_betcmi_20250211_couplingmap.fits'
# couplingmap_data = '/home/first/yjkim/flat_characterization/2025-02-11/flux_dependent_flat/corrected_betcmi_20250211_couplingmap_badpix.fits'
couplingmap_data = '/mnt/datazpool/PL/yjkim/reduced_map/betcmi_20250211/betcmi_20250211_couplingmap_frames_12:11:00_12:19:09_combined.fits'
data = fits.open(couplingmap_data)

frames0 = data[0].data
header = data[0].header
numframes = data[1].data
var = data[3].data
normvar = data[4].data

pos_mas = np.linspace(header['XMIN'], header['XMAX'], header['MAP_N'])

smooth_spectrum = False

if smooth_spectrum:

    from scipy.ndimage import gaussian_filter

    frames = np.zeros_like(frames0)
    for ix in range(len(pos_mas)):
        for iy in range(len(pos_mas)):
            for fibind in range(38):
                frames0[ix,iy,fibind] = gaussian_filter(frames0[ix,iy,fibind], sigma=1)




normalized = frames0 / np.nansum(frames0, axis=(0,1))[None,None,:,:]

n = len(pos_mas)

mapmodel = mm.CouplingMapModel(mapdata = couplingmap_data)

# wav_fitrange = np.r_[0:95, 120:200]
wav_fitrange = np.r_[0:100, 120:200]
wav_reconrange = np.r_[0:200]
all_map_inputs, all_modeled_recons, all_modeled_coeffs, model_chi2 = \
    mapmodel.make_polynomial_model('corrected2_betcmi_20250211_polymodel', wav_fitrange, wav_reconrange)

for fibind in range(38):
    fig, axs = plt.subplots(ncols=3, figsize=(10,3))

    def animate(specind):
        for ax in axs: ax.clear()
        axs[0].imshow(all_map_inputs[:,:,fibind,specind],vmin=0,vmax=0.01, origin='lower',
                    extent = (min(pos_mas), max(pos_mas), min(pos_mas), max(pos_mas)))
        axs[1].imshow(all_modeled_recons[:,:,fibind,specind],vmin=0,vmax=0.01, origin='lower',
                    extent = (min(pos_mas), max(pos_mas), min(pos_mas), max(pos_mas)))
        axs[2].imshow(all_map_inputs[:,:,fibind,specind] - all_modeled_recons[:,:,fibind,specind],vmin=-0.002,vmax=0.002,cmap='turbo',
                    origin='lower',
                    extent = (min(pos_mas), max(pos_mas), min(pos_mas), max(pos_mas)))
        # axs[3].imshow((all_map_input[specind]-modeled_recon[specind]),vmin=-0.001,vmax=0.001,cmap='RdBu')
        fig.suptitle('specind %d' % wav_reconrange[specind])

        axs[0].set_title('data')
        axs[1].set_title('model')
        axs[2].set_title('residual')

    anim = FuncAnimation(fig, animate,np.arange(90,120), interval = 100)
    anim.save('couplingmap_example_corrected2_fibind%d.gif' % fibind)
