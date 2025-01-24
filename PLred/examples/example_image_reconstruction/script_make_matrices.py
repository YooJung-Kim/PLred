import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm

import PLred.fit as fit
from PLred.imgrecon import locs2image

model_file = '../example_model_couplingmap/betcmi_20240917_polymodel.fits'
image_ngrid = 32
image_fov = 20 # mas
n_trim = 4

fitter = fit.PLMapFit(model_file = model_file, image_ngrid = image_ngrid, 
                      image_fov = image_fov, n_trim = n_trim)

for specind in np.arange(50, 170):
    fibinds = np.arange(38) #np.arange(38) # np.arange(25,30) #

    fitter.make_matrix(specind, fibinds)
    fitter.save_matrix_to_file('matrices/matrix_specind%d.fits' % specind)
