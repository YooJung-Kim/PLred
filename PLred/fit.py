import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from .imgrecon import CouplingMapImageReconstructor, PointSourceFitter, GaussianBlobFitter, DiskFitter
from .mapmodel import CouplingMapModel

# here I intend to add model fitting as well

class PLMapFit:

    # Image reconstruction class

    def __init__(self, matrix_file = None, model_file = None, image_ngrid = None, image_fov = None, n_trim = None):

        if matrix_file is not None:
            self.load_matrix_from_file(matrix_file)
        
        else:

            assert model_file is not None, "Model file is required to create a new matrix"
            assert (image_ngrid is not None) and (image_fov is not None) and (n_trim is not None), "image_ngrid, image_fov, n_trim are required to create a new matrix"
            
            self.model_file = model_file
            self.mapmodel = CouplingMapModel(model = model_file)
            self.mapmodel.set_grid_param(image_ngrid, image_fov, n_trim)

            self.n_trim = n_trim
        

    def prepare_data(self, fiber_inds, specind = None):

        if specind is None:
            specind = self.mat_specind
            print("preparing data for specind", specind)

        if self.n_trim > 0:
            observed = self.mapmodel.normdata[self.n_trim:-self.n_trim,self.n_trim:-self.n_trim,fiber_inds, specind]
            observed_err = self.mapmodel.datanormvar[self.n_trim:-self.n_trim,self.n_trim:-self.n_trim,fiber_inds, specind]
        else:
            observed = self.mapmodel.normdata[:,:,fiber_inds, specind]
            observed_err = self.mapmodel.datanormvar[:,:,fiber_inds, specind]

        observed = np.transpose(observed, (2,0,1)).flatten()
        observed_err = np.sqrt(np.transpose(observed_err, (2,0,1)).flatten())

        self.observed = observed
        self.observed_err = observed_err

        self.idx = ~np.isfinite(self.observed_err)
        # self.observed_err[idx] = 1e6 # some random big number
      
    def make_matrix(self, specind, fiber_inds):

        self.mat = self.mapmodel.make_matrix(specind, fiber_inds)
        self.mat_specind = specind

        if len(fiber_inds) == self.mapmodel.nfib:
            self.mat_full = self.mat.copy()
            print("all the fiber indices are used. saving the matrix to mat_full")

    def subsample_matrix(self, fiber_inds):

        # self.mat_full = self.mat.copy()
        
        subsampled = []

        _n = self.mat_full.shape[1] // self.mapmodel.nfib

        for fibind in fiber_inds:
            
            subsampled.append(self.mat_full[:,fibind * _n:(fibind+1) * _n])

        subsampled = np.hstack(subsampled)

        self.mat = subsampled
        self.subsampled_fiber_inds = fiber_inds


    def save_matrix_to_file(self, filename):
        
        hdu = fits.PrimaryHDU(self.mat)
        hdu.header['NGRID'] = self.mapmodel.image_ngrid
        hdu.header['FOV'] = self.mapmodel.image_fov
        hdu.header['NTRIM'] = self.n_trim
        hdu.header['MODEL'] = self.model_file
        hdu.header['SPECIND'] = self.mat_specind
        hdu.writeto(filename, overwrite=True)
        print("Matrix saved to ", filename)


    def load_matrix_from_file(self, filename):
        
        hdu = fits.open(filename)
        self.model_file = hdu[0].header['MODEL']
        self.mapmodel = CouplingMapModel(model = self.model_file)

        self.mat = hdu[0].data
        self.mat_full = self.mat.copy()
        self.mapmodel.set_grid_param(hdu[0].header['NGRID'], hdu[0].header['FOV'], hdu[0].header['NTRIM'])
        self.n_trim = hdu[0].header['NTRIM']
        print("Matrix loaded from ", filename, "specind", hdu[0].header['SPECIND'])
        self.mat_specind = hdu[0].header['SPECIND']

    def store_hyperparams(self, ini_temp, tau, gamma, n_elemenet, target_chi2,
                          regul_dict = {}):

        self.ini_temp = ini_temp
        self.tau = tau
        self.gamma = gamma
        self.n_elemenet = n_elemenet
        self.target_chi2 = target_chi2
        self.regul_dict = regul_dict

    def run(self, centerfrac = None, move_ratio = 1, niter = 500, burn_in_iter=100, seed=12345, plot_every = 500,
            prior_type = 'circle', ini_method = 'random', 
            small_to_random_ratio = 0, **kwargs):



        self.rc = CouplingMapImageReconstructor(self.mat.T, self.observed, self.observed_err, 'imgrecon_test',
                                                axis_len = self.mapmodel.image_ngrid,
                                                ini_temp= self.ini_temp,
                                                tau = self.tau,
                                                burn_in_iter= burn_in_iter,
                                                gamma = self.gamma,
                                                seed = seed,
                                                n_element= self.n_elemenet,
                                                target_chi2= self.target_chi2,
                                                ini_method = ini_method,
                                                regul_dict= self.regul_dict
                                                # do_entropy_regul= self.do_entropy_regul,
                                                # entropy_regul_coeff = self.entropy_regul_coeff
                                                )


        self.rc.make_prior(prior_type, **kwargs)

        # if centerfrac is None:
        #     self.rc.run_chain(niter, plot_every = plot_every)
        # else:
        #     self.rc.run_chain_with_central_frac(niter, centerfrac, plot_every = plot_every)

        self.rc.run_chain(niter, move_ratio = move_ratio, central_frac=centerfrac, plot_every = plot_every,
                          small_to_random_ratio= small_to_random_ratio)

        print("Final chi2", self.rc.current_ll*2)
        return self.rc

    
    def plot_data(self, vmax=0.02, return_fig = False): 

        n = self.mapmodel.map_n - 2 * self.n_trim
        reshaped = np.reshape(self.rc.data, newshape = (-1, n, n))
        titles = ['port %d' % i for i in self.subsampled_fiber_inds]

        fig = plot_maps(reshaped, titles = titles, vmin = 0, vmax= vmax, suptitle= 'Data',
                  return_fig = return_fig)
        if return_fig: return fig
    
    def plot_model(self, vmax=0.02, return_fig = False):

        n = self.mapmodel.map_n - 2 * self.n_trim
        reshaped = np.reshape(self.rc.final_vec, newshape = (-1, n, n))
        titles = ['port %d' % i for i in self.subsampled_fiber_inds]

        fig = plot_maps(reshaped, titles = titles, vmin = 0, vmax= vmax, suptitle = 'Model',
                  return_fig = return_fig)
        if return_fig: return fig

    def plot_residuals(self, vmax=0.002, return_fig = False, SN = False):

        n = self.mapmodel.map_n - 2 * self.n_trim
        if SN:
            reshaped = np.reshape((self.rc.data - self.rc.final_vec)/self.rc.data_err, newshape = (-1, n, n))
        else:
            reshaped = np.reshape((self.rc.data - self.rc.final_vec), newshape = (-1, n, n))
        titles = ['port %d' % i for i in self.subsampled_fiber_inds]

        fig = plot_maps(reshaped, titles = titles, vmin = -vmax, vmax= vmax, suptitle = 'Residuals',
                  cmap = 'RdBu',
                  return_fig = return_fig)
        if return_fig: return fig
    
    def plot_1d(self, subsampled_fibind, return_fig = False):

        n = self.mapmodel.map_n - 2 * self.n_trim
        reshaped = np.reshape(self.rc.data, newshape = (-1, n**2))
        reshaped_err = np.reshape(self.rc.data_err, newshape = (-1, n**2))
        reshaped_vec = np.reshape(self.rc.final_vec, newshape = (-1, n**2))

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)

        ax.errorbar(np.arange(n**2), reshaped[subsampled_fibind], yerr = reshaped_err[subsampled_fibind], label = 'data', fmt='o-', color='black')
        ax.plot(np.arange(n**2), reshaped_vec[subsampled_fibind], label = 'reconstructed', color='red')
        ax.legend()

        chi2 = np.nanmean((reshaped[subsampled_fibind] - reshaped_vec[subsampled_fibind])**2/reshaped_err[subsampled_fibind]**2)
        ax.set_title(r'port %d ($\chi^2$ = %.3f)' % (self.subsampled_fiber_inds[subsampled_fibind], chi2))
        
        if return_fig:
            return fig
        
        plt.show()


    #     self.rc.plot_diagnostic()

    def run_fitting_pointsource(self, n_point_sources, ini_params,
                                mcmc = False,
                                bounds = None,
                                ini_ball_size = 0.1,
                                niter = 1000, 
                                burn_in_iter=100, 
                                seed=12345, 
                                plot_every = 500):

        self.rc = PointSourceFitter(self.mat.T, self.observed, self.observed_err, 'pointsouce_test',
                                                axis_len = self.mapmodel.image_ngrid,
                                                n_point_sources= n_point_sources,
                                                # ini_temp= self.ini_temp,
                                                # tau = self.tau,
                                                burn_in_iter= burn_in_iter,
                                                # gamma = self.gamma,
                                                seed = seed,
                                                # n_element= self.n_elemenet,
                                                # target_chi2= self.target_chi2
                                            
                                                )
        if mcmc:
            self.rc.run_chain(niter, ini_params, ini_ball_size, plot_every = plot_every)
        else:

            self.rc.run_optimization(ini_params, bounds = bounds)
        
        return self.rc
    
    def run_fitting_gaussian(self, ini_params, 
                             mcmc = False,
                             bounds = None,
                             fix_PA_value = None,
                             central_point_source_flux = 0,
                             ini_ball_size = 0.1,
                             niter = 1000, 
                             burn_in_iter=100, 
                             seed=12345, 
                             plot_every = 500):
        '''
        Run fitting for a Gaussian blob model.
        
        For circular gaussian fit, ini_params = [x, y, sigma]
        For elliptical gaussian fit, ini_params = [x, y, sigma_x, sigma_y, PA]
        For elliptical gaussian fit with fixed PA, ini_params = [x, y, sigma_x, sigma_y]
        If fix_PA_value is provided, it will be used as the fixed PA value.
        If ini_params has 3 elements, it will be treated as circular gaussian fit.
        If ini_params has 4 elements, it will be treated as elliptical gaussian fit with fixed PA.
        If ini_params has 5 elements, it will be treated as elliptical gaussian fit.
        
        '''

        if len(ini_params) == 5:
            fix_circular = False
            fix_PA = False
        elif len(ini_params) == 3:
            fix_circular = True
            fix_PA = False
        elif len(ini_params) == 4:
            fix_circular = False
            fix_PA = True
        else:
            raise ValueError("ini_params should have 3 or 5 elements")
        self.rc = GaussianBlobFitter(self.mat.T, self.observed, self.observed_err, 'gaussianblob_test',
                                                axis_len = self.mapmodel.image_ngrid,
                                                fix_circular= fix_circular,
                                                fix_PA = fix_PA,
                                                fix_PA_value = fix_PA_value,
                                                # ini_temp= self.ini_temp,
                                                # tau = self.tau,
                                                burn_in_iter= burn_in_iter,
                                                # gamma = self.gamma,
                                                seed = seed,
                                                # n_element= self.n_elemenet,
                                                # target_chi2= self.target_chi2
                                                central_point_source_flux= central_point_source_flux
                                                )
        if mcmc:
            self.rc.run_chain(niter, ini_params, ini_ball_size, plot_every = plot_every)
        else:
            self.rc.run_optimization(ini_params, bounds = bounds)
        return self.rc


class PolyPLMapFit(PLMapFit):

    # Map fitting class, but this time with multiple wavelength channels

    def __init__(self, matrix_files):

        self.load_matrix_from_file_all(matrix_files)
        

    def prepare_data_all(self, fiber_inds, specinds):

        nwav = len(specinds)

        observeds = []
        observed_errs = []
        idxs = []

        for i in range(nwav):
            self.prepare_data(fiber_inds, specind = specinds[i])
            observeds.append(self.observed)
            observed_errs.append(self.observed_err)
            idxs.append(self.idx)

        self.observeds = np.array(observeds)
        self.observed_errs = np.array(observed_errs)
        self.idxs = np.array(idxs)
    
    def load_matrix_from_file_all(self, filenames):

        mats = []
        mat_specinds = []

        for filename in filenames:
            self.load_matrix_from_file(filename)
            mats.append(self.mat)
            mat_specinds.append(self.mat_specind)
        
        self.mat = np.array(mats)
        self.mat_specind = np.array(mat_specinds)
        self.mat_full = self.mat.copy()

    def run_mcmc_disk(self, fixed_params, vgrid, ini_params, ini_ball_size = 0.1,
                             niter = 1000, 
                             apply_point_source_fraction = False,
                             point_source_fracs = None,
                             burn_in_iter=100, 
                             seed=12345, 
                             plot_every = 500,
                             outname = 'disk_test'):

        self.rc = DiskFitter(fixed_params,
                             vgrid,
                             np.transpose(self.mat, (0,2,1)), 
                             self.observeds, 
                             self.observed_errs, 
                             outname,
                             apply_point_source_fraction= apply_point_source_fraction,
                             point_source_fracs = point_source_fracs,
                             axis_len = self.mapmodel.image_ngrid,
                             image_fov = self.mapmodel.image_fov,
                             burn_in_iter= burn_in_iter,
                             seed = seed,
                            )
        self.rc.run_chain(niter, ini_params, ini_ball_size, plot_every = plot_every)

        return self.rc

    def get_logprobs(self, discard, flat=True):
        return self.rc.sampler.get_log_prob(discard=discard, flat=flat)
    
    def get_chain(self, discard, flat=True):
        return self.rc.sampler.get_chain(discard=discard, flat=flat)
    
    def get_images(self, params):
        
        params_array = self.rc.params_dict_to_array(params)
        iso_map = self.rc.compute_model_from_params(params_array, return_image = True)
        return iso_map
    
    def get_vecs(self, params):
        
        params_array = self.rc.params_dict_to_array(params)
        vecs = self.rc.compute_model_from_params(params_array, return_image = False)
        return vecs
    
    def save_mcmc_results(self, filename):
        np.savez(filename, 
                 chain = self.get_chain(discard=0, flat=False),
                 logprobs = self.get_logprobs(discard=0, flat=False),
                 free_params = self.rc.free_param_keys,
                 vgrid = self.rc.vgrid,
                 ini_params = self.rc.ini_params,
                 apply_point_source_fraction = self.rc.apply_point_source_fraction,
                 point_source_fracs = self.rc.point_source_fracs,
                 )
        print("MCMC results saved to ", filename)
    




def plot_maps(maps, titles=None, texts=None, origin='upper', vmin=None, vmax=None,
              cmap = 'viridis', suptitle=None, return_fig = False):

    n_maps = len(maps)
    ncols = min(5, n_maps)
    nrows = (n_maps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2), sharex=True, sharey=True)
    
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_maps:
            im = ax.imshow(maps[i], origin=origin, vmin=vmin, vmax=vmax,
                           cmap = cmap)
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
            # if texts is not None:
            #     ax.text(0.5, 0.5, texts[i], transform=ax.transAxes, fontsize=12, color='white', ha='center')
        else:
            ax.axis('off')

    # fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.tight_layout()
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=16)

    if return_fig:
        return fig
    else:
        plt.show()


# class PLMapFitShape:
    
#     def __init__(self, n_trim = 0):


