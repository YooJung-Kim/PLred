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
        self.n_fixed = 0

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
            current_chi2 = self.current_ll * 2
            self.temp += 1/self.tau * (current_chi2 - self.gamma * self.temp) * (1 - self.target_chi2 / current_chi2)    
            # if self.temp < 1:
            #     self.temp = 1
            return True, new_ll
        else:
            
            return False, self.current_ll
        
    def set_initial_state(self, central_frac = None):
        '''
        sets initial state for the MCMC chain
        self.current_vec, self.current_ll, self.temp are set
        '''

        if self.ini_method == 'random':
            # random locations
            element_xs = np.random.choice(self.axis_len, size = self.n_element)
            element_ys = np.random.choice(self.axis_len, size = self.n_element)
            self.locs = self.axis_len * element_xs + element_ys
        
        elif self.ini_method == 'center':
            self.locs = np.array([self.centerloc] * self.n_element)

        if central_frac is not None:
            self.central_frac = central_frac
            self.n_fixed = int((self.n_element * central_frac)/ (1 - central_frac))
            print("Fixing %d elements to the center. %d elements are free to move" % (self.n_fixed, self.n_element))
            self.locs = list(np.concatenate([self.locs, [self.centerloc]*self.n_fixed]))
        
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
    
    # def set_initial_state_central_frac(self, central_frac):

    #     if self.ini_method == 'random':
    #         # random locations
    #         element_xs = np.random.choice(self.axis_len, size = self.n_element)
    #         element_ys = np.random.choice(self.axis_len, size = self.n_element)
    #         self.locs = self.axis_len * element_xs + element_ys
        
    #     elif self.ini_method == 'center':
    #         self.locs = np.array([self.centerloc] * self.n_element)

    #     self.n_fixed = int((self.n_element * central_frac)/ (1 - central_frac))
    #     print("Fixing %d elements to the center. %d elements are free to move" % (self.n_fixed, self.n_element))
    #     self.locs = list(np.concatenate([self.locs, [self.centerloc]*self.n_fixed]))
        
    #     # compute vector from locs
    #     self.current_vec = self.compute_model_from_locs(self.locs)
        
    #     # compute initial log likelihood
    #     self.current_ll = self.compute_ll(self.current_vec)

    #     # initial temperature
    #     self.temp = self.ini_temp

    #     self.lls = [self.current_ll]
    #     self.temps = [self.temp]
    #     self.post_locs = np.array([], dtype=int)
    #     self.all_post_locs = np.array([], dtype=int)

    #     # if self.model_central_frac:
    #     #     self.current_central_frac = 0.5
    #     #     self.central_fracs = []
    #     # else:
    #     #     self.current_central_frac = None

    
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

    def move_element(self, ni, move_scheme = 'random'):
        '''
        move the flux element ni
        '''

        if move_scheme == 'random':
            newloc = np.random.choice(self.axis_len**2)
            delta_vec = (self.matrix[:,newloc] - self.matrix[:,self.locs[ni]]) / (self.n_element + self.n_fixed)
            new_vec = self.current_vec + delta_vec
            new_ll = self.compute_ll(new_vec)
            prior_ratio = self.prior[newloc] / self.prior[self.locs[ni]]
            move, new_ll = self.mh(new_ll, prior_ratio)

            if move:
                logging.debug(f"Move_random accepted in element {ni}")
                self.locs[ni] = newloc
                self.current_vec += delta_vec
                self.current_ll = new_ll
            else:
                # logging.info(f"Move rejected in element {ni}")
                self.current_ll = self.current_ll
        
        elif move_scheme == 'smallstep':
            # move to adjacent pixel
            valid_move = False
            while not valid_move:
                which_axis = np.random.choice([0,1])
                if which_axis == 0:
                    newloc = self.locs[ni] + np.random.choice([-1, 1])
                else:
                    newloc = self.locs[ni] + np.random.choice([-1, 1]) * self.axis_len
                if 0 <= newloc < self.axis_len**2:
                    valid_move = True
            delta_vec = (self.matrix[:,newloc] - self.matrix[:,self.locs[ni]]) / (self.n_element + self.n_fixed)
            new_vec = self.current_vec + delta_vec
            new_ll = self.compute_ll(new_vec)
            prior_ratio = self.prior[newloc] / self.prior[self.locs[ni]]
            move, new_ll = self.mh(new_ll, prior_ratio)
            if move:
                logging.debug(f"Move_smallstep accepted in element {ni}")
                self.locs[ni] = newloc
                self.current_vec += delta_vec
                self.current_ll = new_ll
            else:
                # logging.info(f"Move rejected in element {ni}")
                self.current_ll = self.current_ll
        
        elif move_scheme == 'center':
            # move to center
            newloc = self.centerloc
            delta_vec = (self.matrix[:,newloc] - self.matrix[:,self.locs[ni]]) / (self.n_element + self.n_fixed)
            new_vec = self.current_vec + delta_vec
            new_ll = self.compute_ll(new_vec)
            prior_ratio = self.prior[newloc] / self.prior[self.locs[ni]]
            move, new_ll = self.mh(new_ll, prior_ratio)

            if move:
                logging.debug(f"Move_center accepted in element {ni}")
                self.locs[ni] = newloc
                self.current_vec += delta_vec
                self.current_ll = new_ll
            else:
                # logging.info(f"Move rejected in element {ni}")
                self.current_ll = self.current_ll
        
        else:
            raise ValueError("move_scheme not recognized")
    
    # def run_chain_with_central_frac(self, niter, central_frac, plot_every = 100):

    #     self.niter = niter

    #     self.set_initial_state_central_frac(central_frac)
    #     self.central_frac = central_frac

    #     for iter in tqdm(range(self.niter)):
            
    #         # attempt flux element move
    #         for ni in range(self.n_element):

    #             self.move_element(ni)
                

    #         # store the results
    #         self.lls.append(self.current_ll)
    #         self.temps.append(self.temp)
    #         self.post_locs = np.concatenate((self.post_locs, self.locs))
            
    #         if iter > self.burn_in_iter:
    #             self.all_post_locs = np.concatenate((self.all_post_locs, self.locs))

    #         if iter % plot_every == 0:
    #             self.plot_current_state()

    #     # compute final recovered parameters
    #     self.final_image = locs2image(self.all_post_locs, self.axis_len)
    #     self.final_vec = (self.matrix @ self.final_image.flatten()) / len(self.all_post_locs)
    #     self.plot_final_state(self.final_image)

    #     print("Done")

    def run_chain(self, niter, central_frac = None, plot_every = 100,
                  move_ratio = 0.95,
                  small_to_random_ratio = 0):
        '''
        Run the MCMC chain
        
        Parameters:
        - niter : Number of iterations
        '''
        self.niter = niter

        self.set_initial_state(central_frac=central_frac)

        for iter in tqdm(range(self.niter)):
            
            # attempt flux element move
            for ni in range(self.n_element):
                
                if np.random.rand() < move_ratio:
                    if np.random.rand() < small_to_random_ratio:
                        self.move_element(ni, move_scheme='smallstep')
                    else:
                        self.move_element(ni, move_scheme='random')
                else:
                    self.move_element(ni, move_scheme='center')                

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

import emcee

class BaseModelFitter:

    '''
    Model fitter using emcee.
    Regardless of the model, the scene is projected on a regular grid of pixels.
    "compute_model_from_params" should be implemented in a subclass of BaseModelFitter.
    '''

    vectype = np.float32 # or np.complex_

    def __init__(self, matrix, data, data_err, outname,
                 axis_len = 33,
                #  gamma = 256,
                #  tau = 1e4,
                #  target_chi2 = 1,
                 burn_in_iter = 200,
                 nwalkers = 10,
                #  ini_temp = 1,
                 seed = 123456,
                 loglevel = logging.INFO
                 ):

        self.matrix = matrix
        self.data = data
        self.data_err = data_err
        self.len_vec = np.shape(matrix)[0]

        # image parameters
        self.axis_len = axis_len
        self.center = axis_len // 2 

        # temperature schedule parameters
        # self.gamma = gamma
        # self.tau = tau
        # self.target_chi2 = target_chi2
        # self.ini_temp = ini_temp

        # convergence parameters
        self.burn_in_iter = burn_in_iter
        self.ini_seed = seed
        np.random.seed(seed)

        self.outname = outname

        self.nwalkers = nwalkers

        logging.basicConfig(level = loglevel, format='%(asctime)s - %(levelname)s - %(message)s')

    
    def compute_ll(self, params):
        '''
        compute the log likelihood of the model given the parameters
        '''
        # raise NotImplementedError("compute_ll must be overridden")
        current_observable = self.compute_model_from_params(params)

        # -chi^2 / 2
        current_ll = -0.5 * np.nansum((current_observable - self.data)**2 / self.data_err**2) #/ self.ndf / 2
        return current_ll
    
    def compute_model_from_params(self, params):
        '''
        compute the model vector from the parameters
        '''
        raise NotImplementedError("compute_model_from_params must be overridden")
    
    def run_chain(self, niter, ini_params, ini_ball_size = 1e-3, plot_every=100):

        self.nparams = len(ini_params)
        self.niter = niter

        pos = np.random.normal(ini_params, ini_ball_size, size=(self.nwalkers, self.nparams))
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.nparams, self.compute_ll)
        self.sampler.run_mcmc(pos, niter, progress=True)

    
    # def set_initial_state(self, ini_params, nwalkers = 10):
    #     '''
    #     sets initial state for the MCMC chain
    #     self.current_vec, self.current_ll, self.temp are set
    #     '''

    #     # compute vector from locs
    #     self.current_vec = self.compute_model_from_params(ini_params)
        
    #     # compute initial log likelihood
    #     self.current_ll = self.compute_ll(ini_params)

    #     # initial temperature
    #     # self.temp = self.ini_temp

    #     self.lls = [self.current_ll]
    #     # self.temps = [self.temp]

    #     self.params = [ini_params]
    #     self.all_params = [ini_params]

    #     self.nparams = len(ini_params)

    #     self.sampler = emcee.EnsembleSampler(nwalkers, self.nparams, self.compute_ll)
    
    def plot_current_state(self):
        '''
        plot the current state
        '''
        fig, axs = plt.subplots(ncols=3, figsize=(12,4))
        axs[0].imshow(self.current_vec.reshape((self.axis_len,self.axis_len)), origin='lower')
        axs[1].semilogy(self.lls)
        for i in range(self.nparams):
            axs[2].plot(self.params[:,i]) #semilogy(self.temps)

        axs[0].set_title('current image')
        axs[1].set_title('log likelihood')
        axs[2].set_title('parameters')

        for ax in axs[1:]:
            ax.set_xlabel('iteration')

        fig.savefig(self.outname+'_current_state.png')

    def plot_final_state(self, image):
        '''
        plot the final state
        '''
        fig, axs = plt.subplots(ncols=3, figsize=(12,4))
        axs[0].imshow(image.reshape((self.axis_len,self.axis_len)), origin='lower')
        axs[1].semilogy(self.lls)
        for i in range(self.nparams):
            axs[2].plot(self.params[:,i]) #semilogy(self.temps)

        axs[0].set_title('final image')
        axs[1].set_title('log likelihood')
        axs[2].set_title('parameters')

        for ax in axs[1:]:
            ax.set_xlabel('iteration')

        fig.savefig(self.outname+'_final_state.png')
    
    # def move_element(self, ni, move_scheme = 'random'):

class PointSourceFitter(BaseModelFitter):

    def __init__(self, matrix, data, data_err, outname, 
                    axis_len = 33,
                 n_point_sources = 1,
                 burn_in_iter = 200,
                 seed = 123456,
                 loglevel = logging.INFO
                 ):
        super().__init__(matrix, data, data_err, outname,
                            axis_len = axis_len,
                         # gamma = gamma,
                         # tau = tau,
                         # target_chi2 = target_chi2,
                         burn_in_iter = burn_in_iter,
                         # ini_temp = ini_temp,
                         seed = seed,
                         loglevel=loglevel)
        
        self.n_point_sources = n_point_sources

    def compute_vec_point(self, x, y):
    
        xint, yint = int(x), int(y)
        xfrac, yfrac = x - xint, y - yint

        # compute the model vector from the locations
        vec = np.zeros(self.len_vec, dtype = self.vectype)

        loc = xint * self.axis_len + yint
        vec += self.matrix[:,loc] * (1 - xfrac) * (1 - yfrac)
        if xint + 1 < self.axis_len:
            loc = (xint + 1) * self.axis_len + yint
            vec += self.matrix[:,loc] * xfrac * (1 - yfrac)
        else:
            return np.nan
        if yint + 1 < self.axis_len:
            loc = xint * self.axis_len + (yint + 1)
            vec += self.matrix[:,loc] * (1 - xfrac) * yfrac
        else:
            return np.nan
        if xint + 1 < self.axis_len and yint + 1 < self.axis_len:
            loc = (xint + 1) * self.axis_len + (yint + 1)
            vec += self.matrix[:,loc] * xfrac * yfrac
        else:
            return np.nan
        
        return vec
        
    def compute_model_from_params(self, params):
        '''
        compute the model vector from the parameters
        '''

        if self.n_point_sources == 1:
            [x, y] = params

            vec = self.compute_vec_point(x, y)
        
        else:
            vec = np.zeros(self.len_vec, dtype = self.vectype)
            fsum = 0

            for i in range(self.n_point_sources):
                x, y = params[3*i], params[3*i+1]
                if i == 0:
                    f = 1
                else:
                    f = params[3*i-1]
                vec += f * self.compute_vec_point(x, y)
                fsum += f
            
            # normalize the fluxes
            vec /= fsum

        return vec

class GaussianBlobFitter(BaseModelFitter):

    min_sigma = 1e-2
    bound_width = 1


    def __init__(self, matrix, data, data_err, outname,
                    axis_len = 33,
                    fix_circular = False,
                    # n_blobs = 1,
                    burn_in_iter = 200,
                    seed = 123456,
                    loglevel = logging.INFO
                    ):
            super().__init__(matrix, data, data_err, outname,
                            axis_len = axis_len,
                            # gamma = gamma,
                            # tau = tau,
                            # target_chi2 = target_chi2,
                            burn_in_iter = burn_in_iter,
                            # ini_temp = ini_temp,
                            seed = seed,
                            loglevel=loglevel)
            
            # self.n_blobs = n_blobs
            xa = np.arange(self.axis_len)
            self.xg, self.yg = np.meshgrid(xa, xa)

            self.fix_circular = fix_circular
    
    def compute_model_from_params(self, params):

        if not self.fix_circular:
            [x, y, sigma_x, sigma_y, theta] = params
        else:
            [x, y, sigma] = params

        # if sigma_x < 0 or sigma_y < 0:
        #     return np.inf
        
        # if x < 0 or x > self.axis_len or y < 0 or y > self.axis_len:
        #     return np.inf
        
        # if theta < -np.pi/2 or theta > np.pi/2:
        #     return np.inf
        
        # print(params)

        # compute the model vector from the parameters

        from astropy.modeling.models import Gaussian2D
        
        if not self.fix_circular:
            g = Gaussian2D(amplitude=1, x_mean=x, y_mean=y, x_stddev=sigma_x, y_stddev=sigma_y, theta=theta)
        else:
            g = Gaussian2D(amplitude=1, x_mean=x, y_mean=y, x_stddev=sigma, y_stddev=sigma)
        
        _im = g(self.xg, self.yg).flatten()
        # print(np.shape(vec))
        if np.nansum(_im) == 0:
            logging.info("zero flux encountered. params:", params)
        
        else:
            _im = _im / np.nansum(_im) # normalize the fluxes

        vec = self.matrix @ _im
        return vec

    def compute_ll(self, params):

        # check if (x,y) are within bounds
        if params[0] < self.bound_width:
            logging.info("x out of bounds. x = %.2f" % params[0])
            params[0] = self.bound_width
        if params[0] > self.axis_len-self.bound_width:
            logging.info("x out of bounds. x = %.2f" % params[0])
            params[0] = self.axis_len-self.bound_width
        if params[1] < self.bound_width:
            logging.info("y out of bounds. y = %.2f" % params[1])
            params[1] = self.bound_width
        if params[1] > self.axis_len-self.bound_width:
            logging.info("y out of bounds. y = %.2f" % params[1])
            params[1] = self.axis_len-self.bound_width

        # [x, y, sigma_x, sigma_y, theta] = params

        # if params[2] < 0:
        #     # logging.info("negative sigma_x")
        #     params[2] *= -1
        # if params[3] < 0:
        #     # logging.info("negative sigma_y")
        #     params[3] *= -1


        # check sigma
        if params[2] < self.min_sigma:
            logging.debug("sigma_x too small")
            params[2] = self.min_sigma
        
        if not self.fix_circular:
            if params[3] < self.min_sigma:
                logging.debug("sigma_y too small")
                params[3] = self.min_sigma

            if params[2] > params[3]:
                logging.debug("swap sigma")
                params[2], params[3] = params[3], params[2]
                params[4] += np.pi/2
                # params = [x, y, sigma_x, sigma_y, theta]

        # if sigma_x < 0 or sigma_y < 0:
        #     logging.info("negative sigma")
        #     return -np.inf



        # if x < 1 or x > self.axis_len-1 or y < 1 or y > self.axis_len-1:
        #     logging.info("out of bounds. (x,y)=(%.2f,%.2f)" % (x,y))
        #     return -np.inf
        
        # check if position angle is within bounds
        if not self.fix_circular:
            if params[4] < -np.pi/2:
                params[4] += np.pi
            
            if params[4] > np.pi/2:
                params[4] -= np.pi
                # logging.info("theta out of bounds")
                # return -np.inf
            
            
        return super().compute_ll(params)

class PolyBaseModelFitter(BaseModelFitter):
    '''
    Base class for wavelength-dependent model fitting
    '''

    def __init__(self, matrices, data_list, data_err_list, outname,
                 axis_len = 33,
                 burn_in_iter = 200,
                 seed = 123456,
                 loglevel = logging.INFO,
                 nwalkers = 10,

                 ):
        
        # Define the model fitters for each matrix
        self.ModelFitters = [
            BaseModelFitter(matrix, data, data_err, outname,
                            axis_len = axis_len,
                            burn_in_iter = burn_in_iter,
                            seed = seed,
                            loglevel=loglevel)
            for matrix, data, data_err in zip(matrices, data_list, data_err_list)
        ]

        self.data = data_list
        self.data_err = data_err_list

        self.nwav = len(matrices)
        self.nwalkers = nwalkers

        logging.basicConfig(level = loglevel, format='%(asctime)s - %(levelname)s - %(message)s')

    # def compute_ll(self, params):
    #     '''
    #     compute the log likelihood of the model given the parameters
    #     '''
    #     # raise NotImplementedError("compute_ll must be overridden")
    #     current_observable = self.compute_model_from_params(params)

    #     # -chi^2 / 2
    #     current_ll = -0.5 * np.nansum((current_observable - self.data)**2 / self.data_err**2) #/ self.ndf / 2
    #     return current_ll
    
    # def compute_ll(self, params):
    #     '''
    #     compute the log likelihood of the model given the parameters
    #     '''


    #     # raise NotImplementedError("compute_ll must be overridden")
        
    #     # lls = np.zeros(self.nwav)
    #     # for wavind in range(self.nwav):
    #     #     # compute log likelihood for each wavelength

    #     #     current_observable = self.compute_model_from_params(params)
    #     #     current_ll = -0.5 * np.nansum((current_observable - self.data_list[wavind])**2 / self.data_err_list[wavind]**2) #/ self.ndf / 2
    #     #     # compute the model vector from the parameters
    #     #     # self.ModelFitters[wavind].current_vec = self.ModelFitters[wavind].compute_model_from_params(params[wavind])
    #     #     # compute initial log likelihood
    #     #     lls[wavind] = current_ll #self.ModelFitters[wavind].compute_ll(params)
        
    #     # # current_observable = self.compute_model_from_params(params)

    #     # # -chi^2 / 2
    #     # # current_ll = -0.5 * np.nansum((current_observable - self.data)**2 / self.data_err**2) #/ self.ndf / 2
    #     # return np.sum(lls) #current_ll
    
    # def compute_model_from_params(self, params):
    #     '''
    #     compute the model vector from the parameters
    #     '''
    #     raise NotImplementedError("compute_model_from_params must be overridden")
    
    # def run_chain(self, niter, ini_params, ini_ball_size = 1e-3, plot_every=100):

    #     self.nparams = len(ini_params)
    #     self.niter = niter

    #     pos = np.random.normal(ini_params, ini_ball_size, size=(self.nwalkers, self.nparams))
    #     self.sampler = emcee.EnsembleSampler(self.nwalkers, self.nparams, self.compute_ll)
    #     self.sampler.run_mcmc(pos, niter, progress=True)

from PLred.scene import make_simple_powerlaw_disk, get_iso_velocity_map
class DiskFitter(PolyBaseModelFitter):



    def __init__(self, fixed_params, vgrid, matrices, data_list, data_err_list, outname,
                 apply_point_source_fraction = False,
                 point_source_fracs = None,
                 axis_len = 33,
                 image_fov = 20,
                 burn_in_iter = 200,
                 seed = 123456,
                 loglevel = logging.INFO
                 ):
        '''
        Disk fitter
        fixed_params: dictionary of fixed parameters for the disk model
        vgrid: velocity grid for the disk model
        matrices: list of matrices for each wavelength
        data_list: list of data for each wavelength
        data_err_list: list of data errors for each wavelength

        Disk params include
        - Vrot
        - Rstar
        - Rout
        - power_index
        - incl_angle
        - PA
        - beta
        '''
        super().__init__(matrices, data_list, data_err_list, outname,
                         axis_len = axis_len,
                         burn_in_iter = burn_in_iter,
                         seed = seed,
                         loglevel=loglevel)
        

        
        # grid where the disk will be defined
        self.xa = np.linspace(-image_fov/2, image_fov/2, axis_len)
        self.yg, self.xg = np.meshgrid(self.xa, self.xa, indexing='ij')

        self.vgrid = vgrid
        self.axis_len = axis_len
        self.image_fov = image_fov

        self.apply_point_source_fraction = apply_point_source_fraction
        if apply_point_source_fraction:
            assert point_source_fracs is not None, "point_source_fracs must be provided if apply_point_source_fraction is True"
            self.point_source_fracs = point_source_fracs

            

        assert len(self.point_source_fracs) == self.nwav, "point_source_fracs must have the same length as the number of wavelengths"

        self.disk_params = {
            'Vrot': None if 'Vrot' not in fixed_params else fixed_params['Vrot'],
            'Rstar': None if 'Rstar' not in fixed_params else fixed_params['Rstar'],
            'Rout': None if 'Rout' not in fixed_params else fixed_params['Rout'],
            'power_index': None if 'power_index' not in fixed_params else fixed_params['power_index'],
            'incl_angle': None if 'incl_angle' not in fixed_params else fixed_params['incl_angle'],
            'PA': None if 'PA' not in fixed_params else fixed_params['PA'],
            'beta': None if 'beta' not in fixed_params else fixed_params['beta'],
        }

        self.fixed_params = fixed_params
        self.free_params = {k: v for k, v in self.disk_params.items() if v is None}
        self.fixed_params = {k: v for k, v in self.disk_params.items() if v is not None}
        self.free_param_keys = list(self.free_params.keys())


    def params_dict_to_array(self, params_dict):
        return np.array([params_dict[k] for k in self.free_param_keys])

    def params_array_to_dict(self, params_array):
        params_dict = self.fixed_params.copy()
        params_dict.update({k: v for k, v in zip(self.free_param_keys, params_array)})
        return params_dict

    def compute_disk(self, params_array, plot=False):
        params_dict = self.params_array_to_dict(params_array)
        # override the disk parameters with the input params
        self.disk_params['Vrot'] = params_dict['Vrot']
        self.disk_params['Rstar'] = params_dict['Rstar']
        self.disk_params['Rout'] = params_dict['Rout']
        self.disk_params['power_index'] = params_dict['power_index']
        self.disk_params['incl_angle'] = params_dict['incl_angle']
        self.disk_params['PA'] = params_dict['PA']
        self.disk_params['beta'] = params_dict['beta']

        # compute the disk model
        intenmap, velmap, _, _ = make_simple_powerlaw_disk(self.disk_params['Vrot'], 
                                                    self.disk_params['Rstar'], 
                                                    self.disk_params['Rout'], 
                                                    self.disk_params['power_index'], 
                                                    self.disk_params['incl_angle'], 
                                                    self.disk_params['PA'], 
                                                    self.disk_params['beta'],
                                                    ngrid = self.axis_len,
                                                    fov = self.image_fov,
                                                    plot=plot)
        
        iso_map = get_iso_velocity_map(intenmap, velmap, self.vgrid)

        return iso_map

    def compute_model_from_params(self, params_array):
        params_dict = self.params_array_to_dict(params_array)
        # compute disk iso velocity map
        iso_map = self.compute_disk(params_array)
        iso_map = iso_map / np.nansum(iso_map, axis=(1,2))[:,None,None] # normalize the map

        # compute stellar contribution
        if self.apply_point_source_fraction:

            star = np.sqrt(self.xg**2 + self.yg**2) < params_dict['Rstar']
            star = star.astype(np.float32)
            star = star / np.nansum(star) # normalize the fluxes

            for wavind in range(self.nwav):
                iso_map[wavind] *= (1 - self.point_source_fracs[wavind])
                iso_map[wavind] += self.point_source_fracs[wavind] * star

        # compute the model vector from the parameters
        vecs = np.array([self.ModelFitters[wavind].matrix @ iso_map[wavind].flatten() for wavind in range(self.nwav)])
        return vecs

    # def compute_ll(self, params_array):
    #     params_dict = self.params_array_to_dict(params_array)
    #     return super().compute_ll(params_array)

    def run_chain(self, niter, ini_params, ini_ball_size = 1e-3, plot_every=100):

        params_array = self.params_dict_to_array(ini_params)

        return super().run_chain(niter, params_array, ini_ball_size = ini_ball_size, plot_every=plot_every)

    

    
class PointSourceModel:

    vectype = np.float32 # or np.complex_

    def __init__(self, matrix, data, data_err, outname,
                 gamma = 256,
                 tau = 1e4,
                 target_chi2 = 1,
                 burn_in_iter = 200,
                 ini_temp = 1,
                 seed = 123456,
                 loglevel = logging.INFO
                 ):


        # store matrix and data
        self.matrix = matrix
        self.data = data
        self.data_err = data_err
        self.len_vec = np.shape(matrix)[0]

        # temperature schedule parameters
        self.gamma = gamma
        self.tau = tau
        self.target_chi2 = target_chi2
        self.ini_temp = ini_temp

        # convergence parameters
        self.burn_in_iter = burn_in_iter
        self.ini_seed = seed
        np.random.seed(seed)

        self.outname = outname

        logging.basicConfig(level = loglevel, format='%(asctime)s - %(levelname)s - %(message)s')

    def compute_ll(self, pos):

        (pos_x, pos_y) = pos

        # compute the model vector from the locations
        vec = np.zeros(self.len_vec, dtype = self.vectype)
