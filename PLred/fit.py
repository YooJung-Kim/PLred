import numpy as np

from .imgrecon import CouplingMapImageReconstructor
from .mapmodel import CouplingMapModel

class PLMapFit:

    def __init__(self, model_file, image_ngrid, image_fov, n_trim):


        self.mapmodel = CouplingMapModel(model = model_file)
        self.mapmodel.set_grid_param(image_ngrid, image_fov, n_trim)

        self.n_trim = n_trim
    

    def prepare_data(self, specind, fiber_inds):

        observed = self.mapmodel.normdata[self.n_trim:-self.n_trim,self.n_trim:-self.n_trim,fiber_inds, specind]
        observed = np.transpose(observed, (2,0,1)).flatten()

        observed_err = self.mapmodel.datanormvar[self.n_trim:-self.n_trim,self.n_trim:-self.n_trim,fiber_inds, specind]
        observed_err = np.sqrt(np.transpose(observed_err, (2,0,1)).flatten())

        self.observed = observed
        self.observed_err = observed_err

        idx = ~np.isfinite(self.observed_err)
        self.observed_err[idx] = 1e6 # some random big number
      
    def make_matrix(self, specind, fiber_inds):

        self.mat = self.mapmodel.make_matrix(specind, fiber_inds)
    
    # def subsample_matrix(self, sampled_fibs):
        


    def store_hyperparams(self, ini_temp, tau, gamma, n_elemenet, target_chi2):

        self.ini_temp = ini_temp
        self.tau = tau
        self.gamma = gamma
        self.n_elemenet = n_elemenet
        self.target_chi2 = target_chi2

    def run(self, niter = 500, burn_in_iter=100, seed=12345, plot_every = 500,
            prior_type = 'circle', **kwargs):



        self.rc = CouplingMapImageReconstructor(self.mat.T, self.observed, self.observed_err, 'imgrecon_test',
                                                axis_len = self.mapmodel.image_ngrid,
                                                ini_temp= self.ini_temp,
                                                tau = self.tau,
                                                burn_in_iter= burn_in_iter,
                                                gamma = self.gamma,
                                                seed = seed,
                                                n_element= self.n_elemenet,
                                                target_chi2= self.target_chi2,
                                                ini_method = 'random'
                                            
                                                )

        self.rc.make_prior(prior_type, **kwargs)
        self.rc.run_chain(niter, plot_every = plot_every)

        return self.rc
    
    # def plot_diagnostic(self):

    #     self.rc.plot_diagnostic()

