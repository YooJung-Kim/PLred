# solve image reconstruction problem
# basically solve ill-posed problem of finding
# (observables) = (giant matrix) @ (image)
# by representing image as a collection of flux elements
# and performing global minimization of -log_likelihood
# using simulated annealing algorithm.
# inspired by SQUEEZE (https://github.com/fabienbaron/squeeze)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import logging


def locs2image(locs, axis_len):
    ''' computes image with given flux element locations '''
    image = np.zeros((axis_len,axis_len))
    for loc in locs:
        x, y = loc // axis_len, loc % axis_len
        image[x,y] += 1
    return image


class BaseImageReconstructor:
    '''
    Base class for MCMC image reconstruction using simulated annealing
    '''

    vectype = np.float32 # or np.complex_

    def __init__(self, matrix, data, data_err, outname,
                 axis_len = 33,
                 n_element = 200,
                 ndf = None,
                 model_central_frac = False,
                 ini_central_frac = None,
                 gamma = 256,
                 tau = 1e4,
                 target_chi2 = 1,
                 burn_in_iter = 200,
                 ini_temp = 1,
                 ini_method = 'random',
                 seed = 123456,
                 loglevel = logging.INFO
                 ):
        '''
        Parameters:
        - matrix : The matrix mapping image to observables
        - data   : Observed data
        - data_err : Errors of the observed data
        - outname : The output name
        - axis_len : length of image axis
        - n_element : Number of flux elements
        - ndf : Degree of freedom
        - model_central_frac : Whether to model the central star flux fraction
        - gamma : Temperature decay coefficient
        - tau : Temperature change timescale
        - target_chi2 : The target chi^2
        - burn_in_iter : The iteration to start sampling posterior distributions
        - ini_temp : Initial temperature
        - seed : Seed for computing the initial image
        
        '''
        
        if model_central_frac:
            assert axis_len % 2 == 1, "use odd number for axis_len, to enable central frac semi-parametric image reconstruction"

        # store matrix and data
        self.matrix = matrix
        self.data = data
        self.data_err = data_err
        self.len_vec = np.shape(matrix)[0]
        
        # image parameters
        self.axis_len = axis_len
        self.n_element = n_element
        self.centerloc = axis_len // 2 * axis_len + axis_len // 2

        # temperature schedule parameters
        self.gamma = gamma
        self.tau = tau
        self.target_chi2 = target_chi2
        self.ini_temp = ini_temp

        # convergence parameters
        self.burn_in_iter = burn_in_iter
        self.ndf = ndf if ndf is not None else len(data)
        self.ini_seed = seed
        self.ini_method = ini_method
        np.random.seed(seed)

        self.model_central_frac = model_central_frac
        self.ini_central_frac = ini_central_frac

        self.outname = outname

        logging.basicConfig(level = loglevel, format='%(asctime)s - %(levelname)s - %(message)s')

    def compute_ll(self, vec):
        '''
        given the vector (= (matrix) @ (image)),
        compute observables and log likelihood.
        This should be defined case-by-case.
        Recommended to implement this in a subclass of BaseImageReconstructor
        '''
        raise NotImplementedError("compute_ll must be overridden")

    def mh(self, new_ll, prior_ratio):
        '''
        Metropolis-Hastings criterion for accepting new state
        '''

        prob_move = np.min([1,prior_ratio * (np.exp((self.current_ll - new_ll)/self.temp))])
        
        if prob_move >= np.random.rand():
            # update temperature
            self.temp += 1/self.tau * (new_ll - self.gamma * self.temp) * (1 - self.target_chi2 / new_ll)    
            return True, new_ll
        else:
            
            return False, self.current_ll
        
    def set_initial_state(self):

        if self.ini_method == 'random':
            # random locations
            element_xs = np.random.choice(self.axis_len, size = self.n_element)
            element_ys = np.random.choice(self.axis_len, size = self.n_element)
            self.locs = self.axis_len * element_xs + element_ys
        
        elif self.ini_method == 'center':
            self.locs = np.array([self.centerloc] * self.n_element)

        # compute vector from locs
        self.current_vec = self.compute_model_from_locs(self.locs)
        
        # compute initial log likelihood
        self.current_ll = self.compute_ll(self.current_vec)

        # initial temperature
        self.temp = self.ini_temp

        self.lls = [self.current_ll]
        self.temps = [self.temp]
        self.post_locs = np.array([], dtype=int)
        self.all_post_locs = np.array([], dtype=int)

        if self.model_central_frac:
            self.current_central_frac = self.ini_central_frac
            self.central_fracs = []
        else:
            self.current_central_frac = None

    
    def compute_model_from_locs(self, locs, normalize = True):
        '''
        compute the model vector from the locations
        '''
        vec = np.zeros(self.len_vec, dtype = self.vectype)
        for i in range(len(locs)):
            vec += self.matrix[:,locs[i]]

        if normalize:
            return vec / len(locs)
        else:
            return vec

    def make_prior(self, method, **kwargs):
        '''
        make prior image

        Parameters:
        - method : 'uniform' or 'gaussian'
        - kwargs : sigma for gaussian, radius for circle

        '''
        self.prior_method = method
        self.prior_kwargs = kwargs

        if method == 'uniform':

            self.prior = np.ones(self.axis_len**2)

        elif method == 'gaussian':

            self.prior = np.zeros(self.axis_len**2)
            x = np.arange(self.axis_len)
            y = np.arange(self.axis_len)
            X, Y = np.meshgrid(x,y)
            self.prior = np.exp(-((X - self.axis_len//2)**2 + (Y - self.axis_len//2)**2) / kwargs['sigma']**2)
        
        elif method == 'circle':

            self.prior = np.zeros(self.axis_len**2) + 1e-6
            x = np.arange(self.axis_len) - self.axis_len/2
            y = np.arange(self.axis_len) - self.axis_len/2
            X, Y = np.meshgrid(x,y)
            self.prior[(X**2 + Y**2).flatten() < kwargs['radius']**2] = 1

        else:
            raise ValueError("prior method not recognized")

        self.prior = self.prior.flatten()

    def plot_current_state(self):
        '''
        plot the current state
        '''
        fig, axs = plt.subplots(ncols=3, figsize=(12,4))
        axs[0].imshow(locs2image(self.locs, self.axis_len), origin='lower')
        axs[1].semilogy(self.lls)
        axs[2].semilogy(self.temps)

        axs[0].set_title('current image')
        axs[1].set_title('log likelihood')
        axs[2].set_title('temperature')

        for ax in axs[1:]:
            ax.set_xlabel('iteration')

        fig.savefig(self.outname+'_current_state.png')

    def plot_final_state(self, image):
        '''
        plot the final state
        '''
        fig, axs = plt.subplots(ncols=3, figsize=(12,4))
        axs[0].imshow(image, origin='lower')
        axs[1].semilogy(self.lls)
        axs[2].semilogy(self.temps)

        axs[0].set_title('final image')
        axs[1].set_title('log likelihood')
        axs[2].set_title('temperature')

        for ax in axs[1:]:
            ax.set_xlabel('iteration')

        fig.savefig(self.outname+'_final_state.png')

    def move_element(self, ni):
        '''
        move the flux element ni
        '''
        newloc = np.random.choice(self.axis_len**2)
        delta_vec = (self.matrix[:,newloc] - self.matrix[:,self.locs[ni]]) / self.n_element
        new_vec = self.current_vec + delta_vec
        new_ll = self.compute_ll(new_vec)
        prior_ratio = self.prior[newloc] / self.prior[self.locs[ni]]
        move, new_ll = self.mh(new_ll, prior_ratio)

        if move:
            logging.debug(f"Move accepted in element {ni}")
            self.locs[ni] = newloc
            self.current_vec += delta_vec
            self.current_ll = new_ll
        else:
            # logging.info(f"Move rejected in element {ni}")
            self.current_ll = self.current_ll
        

    def run_chain(self, niter, plot_every = 100):
        '''
        Run the MCMC chain
        
        Parameters:
        - niter : Number of iterations
        '''
        self.niter = niter

        self.set_initial_state()

        for iter in tqdm(range(self.niter)):
            
            # attempt flux element move
            for ni in range(self.n_element):

                self.move_element(ni)
                

            # store the results
            self.lls.append(self.current_ll)
            self.temps.append(self.temp)
            self.post_locs = np.concatenate((self.post_locs, self.locs))
            
            if iter > self.burn_in_iter:
                self.all_post_locs = np.concatenate((self.all_post_locs, self.locs))

            if iter % plot_every == 0:
                self.plot_current_state()

        # compute final recovered parameters
        self.final_image = locs2image(self.all_post_locs, self.axis_len)
        self.final_vec = (self.matrix @ self.final_image.flatten()) / len(self.all_post_locs)
        self.plot_final_state(self.final_image)

        print("Done")



class CouplingMapImageReconstructor(BaseImageReconstructor):

    def __init__(self, matrix, data, data_err, outname,
                 axis_len = 33,
                 n_element = 200,
                 ndf = None,
                 model_central_frac = False,
                 ini_central_frac = None,
                 gamma = 256,
                 tau = 1e4,
                 target_chi2 = 1,
                 burn_in_iter = 200,
                 ini_temp = 1,
                 ini_method = 'random',
                 seed = 123456,
                 loglevel = logging.INFO
                 ):
        super().__init__(matrix, data, data_err, outname,
                         axis_len = axis_len,
                         n_element = n_element,
                         ndf = ndf,
                         model_central_frac = model_central_frac,
                         ini_central_frac = ini_central_frac,
                         gamma = gamma,
                         tau = tau,
                         target_chi2 = target_chi2,
                         burn_in_iter = burn_in_iter,
                         ini_temp = ini_temp,
                         ini_method = ini_method,
                         seed = seed,
                         loglevel=loglevel)
        
    def compute_ll(self, vec):
        '''
        Observable is the fine-sampled on-sky coupling map.
        The input vector itself is the observable, directly can be compared with data.
        '''
        current_observable = vec
        # chi^2 / 2
        current_ll = np.nansum((current_observable - self.data)**2 / self.data_err**2) / self.ndf / 2
        return current_ll



