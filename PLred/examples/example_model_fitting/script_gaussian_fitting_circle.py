import PLred.mapmodel as mm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize
from scipy.ndimage import center_of_mass
import emcee, corner
import PLred.fit as fit
from astropy.modeling.functional_models import Gaussian2D


fiber_inds = np.arange(38)
timeind = 3

for specind in np.arange(103,104):

    fitter2 = fit.PLMapFit(matrix_file = '/Users/yjkim/Documents/OnskyDemoPaper/results/matrices/matrix_specind%d_ngrid33_t%d.fits' % (specind, timeind))
    fitter2.prepare_data(fiber_inds)

    rc0 = fitter2.run_mcmc_pointsource(1, np.array([16,16]))
    x0 = np.average(rc0.sampler.get_chain(discard=100, flat=True)[:,1])
    y0 = np.average(rc0.sampler.get_chain(discard=100, flat=True)[:,0])

    rc = fitter2.run_mcmc_gaussian(np.array([x0,y0,1.5]), ini_ball_size=0.05, niter = 1000)
    chain = rc.sampler.get_chain(flat=True, discard=200)

    fig, axs = plt.subplots(nrows=4, figsize=(10,10))
    for i in range(10):
        for j in range(3): axs[j].plot(rc.sampler.get_chain()[:,i,j])
        axs[3].plot(rc.sampler.get_log_prob()[:,i]/13**2/38 * 2)
    fig.tight_layout()
    fig.savefig('circle_fitting_results/mcmc_specind%d_t%d.png' % (specind, timeind))

    fig = corner.corner(chain, labels=['x','y','sig'])
    fig.savefig('circle_fitting_results/corner_specind%d_t%d.png' % (specind, timeind))

    np.save('circle_fitting_results/chain_specind%d_t%d.npy' % (specind, timeind), chain)

    final_values = np.mean(chain, axis=0)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    gauss = Gaussian2D(amplitude=1, x_mean=final_values[0], y_mean=final_values[1], 
                x_stddev=final_values[2], y_stddev=final_values[2])

    xg, yg = np.meshgrid(np.arange(33), np.arange(33))
    ax.imshow(gauss(xg,yg))
    ax.set_title('logprob=%.3f,x=%.1f,y=%.1f,s=%.1f' % (np.mean(rc.sampler.get_log_prob(flat=True, discard=200))/38/13**2 * 2, final_values[0], final_values[1], final_values[2]))
    fig.savefig('circle_fitting_results/image_specind%d_t%d.png' % (specind, timeind))