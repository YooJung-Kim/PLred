import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.linalg import lstsq
from tqdm import tqdm

NFIB = 38

def poly_design_matrix(x, y, degree):
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x ** i) * (y ** j))
    return np.vstack(terms).T


def make_interpolation_model(normcube, pos_mas, wav_fitrange, wav_reconrange, 
                             poly_deg_spatial = 7, poly_deg_spectral = 7,
                             variance_map = None, weighted = True):

    if weighted is True: assert variance_map is not None, "for weighted least squares, variance map should be given"

    all_recon_data, all_map_input, all_coeffs = [], [], []

    x_grid, y_grid = np.meshgrid(pos_mas, pos_mas)

    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()

    
    X_poly = poly_design_matrix(x_flat, y_flat, poly_deg_spatial)

    
    for specind in (wav_reconrange):

        map_data = normcube[:,:,specind] # cube[:,:,specind] / np.nansum(cube[:,:,specind])
        weight = 1/variance_map[:,:,specind]
        idx = ~np.isfinite(map_data)
        map_data[idx] = 0
        weight[idx] = 0

        reshaped_data = map_data.ravel()
        
        if weighted:
            reshaped_weights = weight.ravel() #* 0 + 1.0
        else:
            reshaped_weights = np.ones_like(reshaped_data)



        X_poly_weighted = X_poly * np.sqrt(reshaped_weights[:,np.newaxis])
        b_weighted = reshaped_data * np.sqrt(reshaped_weights)

        coeffs, _, _, _ = lstsq(X_poly_weighted, b_weighted)
        
        recon = np.dot(X_poly, coeffs)

        # reshape to match the cube
        recon = recon.reshape((len(pos_mas), len(pos_mas)))
        map_input = reshaped_data.reshape((len(pos_mas), len(pos_mas)))

        all_recon_data.append(recon)
        all_map_input.append(map_input)

        all_coeffs.append(coeffs)


    all_coeffs = np.array(all_coeffs)
    modeled_coeffs = np.zeros_like(all_coeffs)

    for coeff_ind in range(np.shape(all_coeffs)[1]):

        poly = np.polyfit(wav_fitrange, all_coeffs[wav_fitrange,coeff_ind], deg = poly_deg_spectral)
        modeled_coeff = np.poly1d(poly)(wav_reconrange)
        modeled_coeffs[:,coeff_ind] = modeled_coeff

    modeled_recon = []
    for specind in (wav_reconrange):
        recon = np.dot(X_poly, modeled_coeffs[specind])
        modeled_recon.append(recon.reshape((len(pos_mas), len(pos_mas))))

    modeled_recon = np.array(modeled_recon)

    all_map_input = np.array(all_map_input)

    # fill in the missing values
    missing_indices_x = np.where(idx == True)[0]
    missing_indices_y = np.where(idx == True)[1]

    for (mx, my) in zip(missing_indices_x, missing_indices_y):
        (all_map_input)[:,mx, my] = modeled_recon[:,mx, my]

    return modeled_coeffs, modeled_recon, all_map_input




class CouplingMapModel:


    def __init__(self, mapdata = None, model = None,
                 min_nframes = 5):

        self.data = None
        self.datavar = None
        self.datanormvar = None

        if mapdata is not None:
            print("loading mapdata")
            self.map_fits = fits.open(mapdata)

            self.data = self.map_fits[0].data
            self.map_header = self.map_fits[0].header
            self.numframes = self.map_fits[1].data
            self.datavar = self.map_fits[3].data
            self.datanormvar = self.map_fits[4].data
            self.pos_mas = np.linspace(self.map_header['XMIN'], self.map_header['XMAX'], self.map_header['MAP_N'])

            print("masking data with less than {} frames".format(min_nframes))
            self.min_nframes = min_nframes
            idx = self.numframes > min_nframes
            self.data[~idx] = np.nan
            self.datavar[~idx] = np.nan
            self.datanormvar[~idx] = np.nan

            self.normdata = self.data / np.nansum(self.data, axis=(0,1))[None,None]
        
        if model is not None:
            print("loading model")
            self.model_fits = fits.open(model)
            self.model_coeffs = self.model_fits[0].data
            self.model_header = self.model_fits[0].header
            self.normdata = self.model_fits[1].data
            self.datavar = self.model_fits[2].data
            self.datanormvar = self.model_fits[3].data
            self.chi2 = self.model_fits[4].data
            self.pos_mas = np.linspace(self.model_header['XMIN'], self.model_header['XMAX'], self.model_header['MAP_N'])
        
            x_grid, y_grid = np.meshgrid(self.pos_mas, self.pos_mas)
            self.x_flat = x_grid.ravel()
            self.y_flat = y_grid.ravel()
            X_poly = poly_design_matrix(self.x_flat, self.y_flat, self.model_header['NPOLY1'])
            self.wav_reconrange = np.arange(self.model_header['MIN_WAV'], self.model_header['MAX_WAV']+1)
            self.all_modeled_recons = np.zeros((len(self.pos_mas), len(self.pos_mas), NFIB, len(self.wav_reconrange)))
            
            for fibind in range(NFIB):
                for specind in range(len(self.wav_reconrange)):
                    coeffs = self.model_coeffs[fibind, specind]
                    recon = np.dot(X_poly, coeffs)
                    self.all_modeled_recons[:,:,fibind,specind] = recon.reshape((len(self.pos_mas), len(self.pos_mas)))
        
        if self.data is None:
            print("No data loaded. load either mapdata or model")
        

    def make_polynomial_model(self, output_name,
                              wav_fitrange, wav_reconrange,
                              poly_deg_spatial = 9,
                              poly_deg_spectral = 9,
                              weighted = True):
        '''
        make a smooth polynomial model for the coupling map data

        Parameters
        ----------
        output_name : str
            output name for the model
        wav_fitrange : list
            wavelength range for fitting the polynomial
        wav_reconrange : list
            wavelength range for reconstructing the polynomial
        poly_deg_spatial : int
            degree of the polynomial in spatial direction
        poly_deg_spectral : int
            degree of the polynomial in spectral direction
        weighted : bool
            if True, use weighted least squares
        '''
        
        
        all_modeled_coeffs, all_modeled_recons, all_map_inputs = [], [] ,[]

        for fibind in tqdm(range(NFIB)):

            modeled_coeffs, modeled_recon, all_map_input = make_interpolation_model(self.normdata[:,:,fibind,:], 
                                                                                    self.pos_mas, 
                                                                                    wav_fitrange = wav_fitrange, 
                                                                                    wav_reconrange = wav_reconrange, 
                                                                                    poly_deg_spatial = poly_deg_spatial,
                                                                                    poly_deg_spectral = poly_deg_spectral,
                                                                                    variance_map= self.datanormvar[:,:,fibind,:],
                                                                                    weighted = weighted)
            
            all_modeled_coeffs.append(modeled_coeffs)
            all_modeled_recons.append(modeled_recon)
            all_map_inputs.append(all_map_input)

        all_modeled_coeffs = np.array(all_modeled_coeffs)
        all_modeled_recons = np.array(all_modeled_recons)
        all_map_inputs = np.array(all_map_inputs)

        all_modeled_recons = np.transpose(all_modeled_recons, axes = (2,3,0,1))
        all_map_inputs = np.transpose(all_map_inputs, axes = (2,3,0,1))

        self.all_modeled_recons = all_modeled_recons

        # compute chi2
        self.model_chi2 = (all_map_inputs - all_modeled_recons)**2 / self.datanormvar

        if output_name is not None:
            hdu = fits.PrimaryHDU(all_modeled_coeffs)
            hdu.header['XMIN'] = min(self.pos_mas)
            hdu.header['XMAX'] = max(self.pos_mas)
            hdu.header['MAP_N'] = len(self.pos_mas)
            hdu.header['NPOLY1'] = poly_deg_spatial
            hdu.header['NPOLY2'] = poly_deg_spectral
            hdu.header['WEIGHTED'] = weighted
            hdu.header['EXTNAME'] = 'coeffs'
            hdu.header['MIN_WAV'] = min(wav_reconrange)
            hdu.header['MAX_WAV'] = max(wav_reconrange)

            hdu2 = fits.ImageHDU(all_map_inputs)
            hdu2.header['EXTNAME'] = 'data'

            hdu3 = fits.ImageHDU(self.datavar)
            hdu3.header['EXTNAME'] = 'var'

            hdu4 = fits.ImageHDU(self.datanormvar)
            hdu4.header['EXTNAME'] = 'normvar'

            hdu5 = fits.ImageHDU(self.model_chi2)
            hdu5.header['EXTNAME'] = 'chi2'

            hdulist = fits.HDUList([hdu, hdu2, hdu3, hdu4, hdu5])
            hdulist.writeto(output_name+'.fits', overwrite=True)
            print(output_name+'.fits saved')
        
        return all_map_inputs, all_modeled_recons, all_modeled_coeffs, self.model_chi2

    def set_grid_param(self, image_ngrid, image_fov, n_trim):
        
        self.image_ngrid = image_ngrid
        self.image_fov = image_fov
        self.image_mas = np.linspace(-image_fov/2, image_fov/2, image_ngrid)
        self.image_xg, self.image_yg = np.meshgrid(self.image_mas, self.image_mas)
        self.n_trim = n_trim

    def make_matrix(self, specind, fiber_inds):
        '''
        make flat matrix for image reconstruction
        (response on the image grid)
        '''
        
        mat = []
        
        for (x_shift, y_shift) in tqdm(zip(self.image_xg.flatten(), self.image_yg.flatten())):

            _mat = []
            for fibind in (fiber_inds):
                

                X_poly_shifted = poly_design_matrix(self.x_flat + x_shift, self.y_flat + y_shift, self.model_header['NPOLY1'])
                recon_shifted = np.dot(X_poly_shifted, self.model_coeffs[fibind,specind])
                trimmed = recon_shifted.reshape((15,15))[self.n_trim:-self.n_trim, self.n_trim:-self.n_trim].flatten()
                _mat.append(trimmed)
            mat.append(np.array(_mat).ravel())
        
        return np.array(mat)
        


    # def make_interpolation_model(self, wav_fitrange, wav_reconrange, 
    #                              poly_deg_spatial = 9, poly_deg_spectral = 9,
    #                              weighted = True):
    #     '''
    #     Make an interpolation model
    #     '''
        
    #     # all_recon_data, all_map_input, all_coeffs = [], [], []

    #     x_grid, y_grid = np.meshgrid(self.pos_mas, self.pos_mas)

    #     x_flat = x_grid.ravel()
    #     y_flat = y_grid.ravel()

        
    #     X_poly = poly_design_matrix(x_flat, y_flat, poly_deg_spatial)
        
    #     all_map_inputs = np.zeros((NFIB, len(wav_reconrange), len(self.pos_mas), len(self.pos_mas)))
    #     all_modeled_coeffs = np.zeros((NFIB, len(wav_reconrange), X_poly.shape[1]))
    #     all_modeled_recons = np.zeros((NFIB, len(wav_reconrange), len(self.pos_mas), len(self.pos_mas)))

    #     for fibind in tqdm(range(NFIB)):

    #         all_recon_data, all_map_input, all_coeffs = [], [], []

    #         for specind0, specind in enumerate(wav_reconrange):

    #             map_data = self.normdata[:,:,fibind,specind] # cube[:,:,specind] / np.nansum(cube[:,:,specind])
    #             weight = 1/self.normvar[:,:,fibind,specind]
    #             idx = ~np.isfinite(map_data)
    #             map_data[idx] = 0
    #             weight[idx] = 0

    #             reshaped_data = map_data.ravel()
                
    #             if weighted:
    #                 reshaped_weights = weight.ravel() 
    #             else:
    #                 reshaped_weights = np.ones_like(reshaped_data)


    #             X_poly_weighted = X_poly * np.sqrt(reshaped_weights[:,np.newaxis])
    #             b_weighted = reshaped_data * np.sqrt(reshaped_weights)

    #             coeffs, _, _, _ = lstsq(X_poly_weighted, b_weighted)
                
    #             recon = np.dot(X_poly, coeffs)

    #             # reshape to match the cube
    #             recon = recon.reshape((len(self.pos_mas), len(self.pos_mas)))
    #             map_input = reshaped_data.reshape((len(self.pos_mas), len(self.pos_mas)))

    #             all_recon_data.append(recon)
    #             all_map_input.append(map_input)

    #             all_coeffs.append(coeffs)

    #         all_coeffs = np.array(all_coeffs)
    #         modeled_coeffs = np.zeros_like(all_coeffs)

    #         for coeff_ind in range(np.shape(all_coeffs)[1]):

    #             poly = np.polyfit(wav_fitrange, all_coeffs[wav_fitrange,coeff_ind], deg = poly_deg_spectral)
    #             modeled_coeff = np.poly1d(poly)(wav_reconrange)
    #             modeled_coeffs[:,coeff_ind] = modeled_coeff

    #         modeled_recon = []
    #         for specind in (wav_reconrange):
    #             recon = np.dot(X_poly, modeled_coeffs[specind])
    #             modeled_recon.append(recon.reshape((len(self.pos_mas), len(self.pos_mas))))

    #         modeled_recon = np.array(modeled_recon)


    #         all_modeled_recons[fibind, specind0] = modeled_recon
    #         all_map_inputs[fibind, specind0] = map_input
    #         all_modeled_coeffs[fibind, specind0] = modeled_coeffs

    #     self.all_modeled_recons = all_modeled_recons
    #     self.all_map_inputs = all_map_inputs
    #     self.all_modeled_coeffs = all_modeled_coeffs

    #     self.save_model()
