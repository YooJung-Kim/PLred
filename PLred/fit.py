import numpy as np
from astropy.io import fits

from .imgrecon import CouplingMapImageReconstructor
from .mapmodel import CouplingMapModel

# here I intend to add model fitting as well

class PLMapFit:

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

        observed = self.mapmodel.normdata[self.n_trim:-self.n_trim,self.n_trim:-self.n_trim,fiber_inds, specind]
        observed = np.transpose(observed, (2,0,1)).flatten()

        observed_err = self.mapmodel.datanormvar[self.n_trim:-self.n_trim,self.n_trim:-self.n_trim,fiber_inds, specind]
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

