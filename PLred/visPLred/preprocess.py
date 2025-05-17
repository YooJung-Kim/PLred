# routines for correcting detector imperfections

import numpy as np
import matplotlib.pyplot as plt
from PLred.visPLred import utils as du
from astropy.io import fits
from tqdm import tqdm
from PLred.visPLred.parameters import *


def make_dark_fits(dark_files):

    dark, nframes = du.average_frames(dark_files)
    hdu = fits.PrimaryHDU(dark)
    hdu.header['files'] = str(dark_files)
    hdu.header['nframes'] = int(nframes)
    return hdu

def fit_nonlinearity_curve(ts, vals, refind, npoly=3, plot=False, val_to_correct = None):

    # x-axis: expected counts (from the reference frame)
    xval = ts * vals[refind] / ts[refind]

    # y-axis: observed counts
    yval = vals

    # remove NaNs and negative values
    clean_idx = np.isfinite(np.log10(xval)) & np.isfinite(np.log10(yval)) #& (np.log10(_x) > 0) & (_y > 0)
    xval = xval[clean_idx]
    yval = yval[clean_idx]

    # fit polynomial
    poly = np.polyfit(np.log10(xval), np.log10(yval), npoly)

    if plot:

        fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

        for ax in axs:
            ax.plot(xval, yval, color='red')
            ax.plot(xval, 10**np.polyval(poly, np.log10(xval)), color='black', ls='--')

            ax.set_xlabel('Expected counts')
            ax.set_ylabel('Observed counts')
        axs[0].axline((0, 0), slope=1, color='gray')
        axs[1].plot(xval, xval, color='gray')
        
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')

        # fig.suptitle('')

        if val_to_correct is not None:
            # plot the correction
            corrected_val, flag = correct_nonlinearity(val_to_correct, poly, np.min(yval), np.max(yval), verbose=True)
            axs[1].plot(corrected_val, val_to_correct, 'o', color='blue',
            label = 'corrected from %d to %d. flag %d' % (val_to_correct, corrected_val, flag))
            axs[1].legend()
            # axs[1].set_title(f'Corrected value: {corrected_val}, flag: {flag}')

        return poly, fig
    
    return poly

def read_flat_files(files):

    # Load flats with multiple exposure times
    logrec = []

    for f in files:
        with fits.open(f, mode='readonly') as hdulist:
            h = hdulist[0].header

            intstr = f.split('firstpl_')[1].split('s_')[0].split('_')[-1]
            typ = h['DATA-TYP'].lower()
            # int = h['EXPTIME']

            logrec.append((f, intstr, typ)) #{'fn': f, 'int': intstr, 'typ': typ})

    logrec = np.array(logrec, dtype=[('fn', 'U256'), ('int', 'U16'), ('typ', 'U16')])


    ts, ims, sig_ims = [], [], []
    for intstr in np.unique(logrec['int']):
        t = float(intstr.rstrip('s'))
        dfns = logrec['fn'][np.nonzero((logrec['typ']=='dark') & (logrec['int']==intstr))[0]]
        ffns = logrec['fn'][np.nonzero((logrec['typ']=='flat') & (logrec['int']==intstr))[0]]
        for dfn, ffn in zip(dfns, ffns):
            with fits.open(dfn, mode='readonly') as hdulist:
                d = hdulist[0].data
                sig_d = hdulist[1].data
            with fits.open(ffn, mode='readonly') as hdulist:
                f = hdulist[0].data
                sig_f = hdulist[1].data
            im = f-d
            sig_im = np.sqrt(sig_d**2 + sig_f**2)
            ts.append(t)
            ims.append(im)
            sig_ims.append(sig_im)
    ts = np.array(ts)
    its = np.argsort(ts)
    ts = ts[its]
    ims = np.array(ims)[its]
    sig_ims = np.array(sig_ims)[its]

    return ts, ims, sig_ims


def model_nonlinearity_from_flats(files, outname,
                                  npoly = 3,
                                  refind = -1,
                                  cut_min_maxcount = 100):
    

    ts, ims, sig_ims = read_flat_files(files)

    # now fit a polynomial to the log-log plot of the data

    all_poly_results = []

    all_poly_results = np.zeros((ims.shape[1], ims.shape[2], npoly+1)) * np.nan
    minvals = np.zeros((ims.shape[1], ims.shape[2])) * np.nan
    maxvals = np.zeros((ims.shape[1], ims.shape[2])) * np.nan

    for x in tqdm(range(firstcam_params['size_y'])): #54+3):

        for y in range(firstcam_params['size_x']): #np.arange(1896):


            try:
                # store the valid range (min and max values of the observed counts)
                minvals[x,y] = np.min(ims[:,x,y])
                maxvals[x,y] = np.max(ims[:,x,y])

                # skip if maxcount is smaller than cut_min_maxcount. don't attempt modeling.

                if maxvals[x,y] > cut_min_maxcount:

                    poly = fit_nonlinearity_curve(ts, ims[:,x,y], refind, npoly)
                    all_poly_results[x,y,:] = poly
                
                # else:
                    # print(f'Skipping pixel ({x}, {y}) due to low max count: {maxvals[x,y]}')



                # # x-axis: expected counts (from the reference frame)
                # xval = ts * ims[refind,x,y] / ts[refind]

                # # y-axis: observed counts
                # yval = ims[:,x,y]

                # # remove NaNs and negative values
                # clean_idx = np.isfinite(np.log10(xval)) & np.isfinite(np.log10(yval)) #& (np.log10(_x) > 0) & (_y > 0)
                # xval = xval[clean_idx]
                # yval = yval[clean_idx]

                # # fit polynomial
                # poly = np.polyfit(np.log10(xval), np.log10(yval), npoly)
                # all_poly_results[x,y,:] = poly




            except:
                # leave the values as NaN
                continue
    
    hdu = fits.PrimaryHDU(all_poly_results)
    hdu.header['npoly'] = npoly
    hdu.header['refind'] = refind
    hdu2 = fits.ImageHDU(minvals, name='minvals')
    hdu3 = fits.ImageHDU(maxvals, name='maxvals')
    hdulist = fits.HDUList([hdu, hdu2, hdu3])
    hdulist.writeto(outname, overwrite=True)
    print(f'Saved results to {outname}')

    return all_poly_results, minvals, maxvals


def correct_nonlinearity(observed_val, poly, minval, maxval, verbose= False):
    '''
    Corrects the nonlinearity of the observed value using the provided polynomial coefficients.
    Args:
        observed_val (float): The observed value to be corrected.
        poly (array-like): Polynomial coefficients for the nonlinearity correction.
        minval (float): Minimum valid value for the correction.
        maxval (float): Maximum valid value for the correction.
    Returns:
        tuple: A tuple containing the corrected value and a status code:
            - Corrected value
            - Status code (0: within bounds, 1: below minval, 2: above maxval, 3: error)
    '''
    try:

        logval = np.log10(observed_val)
        _root_arr = np.zeros(len(poly))
        _root_arr[-1] = logval
        root = np.roots(poly - _root_arr)
        possible_root = 10**root[np.isreal(root)]
        if verbose:
            print(f'Possible roots: {possible_root}')

        possible_root = possible_root[(possible_root > observed_val) & (possible_root < 10*observed_val)]
        
        if len(possible_root) == 0:
            if verbose: print(f'No possible roots found for {observed_val}')
            return observed_val, 3
        
        idx = np.argmin(np.abs(possible_root - observed_val))

        if possible_root[idx] < minval:
            if verbose: print(f'Value {possible_root[idx]} smaller than {minval}')
            # don't want to extrapolate
            return observed_val, 1
        elif possible_root[idx] > maxval:
            if verbose: print(f'Value {possible_root[idx]} larger than {maxval}')
            # don't want to extrapolate
            return observed_val, 2
        else:
            if verbose: print(f'Successfully corrected {observed_val} to {possible_root[idx]}')
            # return the corrected value if it was successful
            return possible_root[idx], 0
    except Exception as e:
        if verbose: print(f'Error correcting {observed_val}', e)
        return observed_val, 4
    
def correct_nonlinearity_map(frame, modelfile,
                             xrange = np.arange(firstcam_params['size_y']),
                             yrange = np.arange(firstcam_params['size_x'])):
    '''
    Corrects the nonlinearity of the data using the model file
    generated using `model_nonlinearity_from_flats`.
    '''

    modelfile = fits.open(modelfile, mode='readonly')
    poly_results = modelfile[0].data
    minvals = modelfile[1].data
    maxvals = modelfile[2].data

    corrected_data = np.zeros_like(frame)
    flags = np.zeros_like(frame, dtype=int)  # 0: within bounds, 1: below minval, 2: above maxval, 3: error

    for x in xrange:
        for y in yrange:
            corrected_data[x, y], flags[x, y] = correct_nonlinearity(frame[x, y], poly_results[x, y], minvals[x, y], maxvals[x, y])
    
    return corrected_data, flags




    


    # make nonlinearity correction file

    # correct nonlinearity

    #

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os, glob
from scipy.optimize import minimize, curve_fit
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# rawfiles = np.sort(glob.glob('../*.fits'))
# exptimes = np.unique([fits.getheader(f)['EXPTIME'] for f in rawfiles])
# allims = fits.getdata('../mean/mean_flats.fits')

# good = np.r_[0:19, 20:22]
# ims = allims[good]
# exptimes = exptimes[good]



class NonlinearityFitter:

    def __init__(self, verbose=True):

        self.verbose = verbose
        # List of (function, initial_guess, loss_function, label)
        self.trials = [
            # (self.fun_linear, None, "linear", None),
            (self.fun_trial1, [1,1,1,1], "trial1", None), #self.check_singularity_trial1),
            (self.fun_trial2, [1,1,1], "trial2", None),
            # (self.fun4, [1,1,1],  "fun4"),
            # (self.fun2, [1,1,1,1], "fun2"),
        ]

    def prep_fit(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        self.result = None
        self.best_fun = None
        self.best_params = None
        self.residual = None
    
    def eval_from_model(self, x, funind, params):
        """Evaluate the model function with the given parameters."""
        fun = self.trials[funind][0]
        return fun(x, *params[:len(self.trials[funind][1])]) 


    # --- Model functions ---
    @staticmethod
    def fun_trial1(x, p, p2, p3, p4):
        return x * (1 + (p/x) + (p2/x)**2) / (1 + p3/x + (p4/x)**2)

    @staticmethod
    def fun_trial2(x, p, p2, p3):
        return x * (1 + p / (1 + (x/p2)**p3))
    
    @staticmethod
    def check_singularity_trial1(params):
        
        root = (params[2] + np.sqrt(params[2]**2 - 4*params[3])) / 2
        # Check for singularity conditions
        if (root > 1) and (root < 1e4):
            print(f"Singularity check failed: root = {root}")
            return True
        return False
    

    def loss(self, params, fun):
        return np.nansum((np.log10(self.y) - np.log10(fun(self.x, *params)))**2)
    
    def logloss(self, params, fun):
        return np.nansum((np.log10(self.y) - np.log10(fun(self.x, *params)))**2)

    
    def fit(self, tol=1, improvement_ratio = 5, max_trials=None):
        """Try each function in order, adopt the first that fits well."""

        self.linear_fit_loss = np.nansum((np.log10(self.y) - np.log10(self.x))**2)
        if self.verbose:
            print("Linear fit loss = %.2e" % self.linear_fit_loss)

        if max_trials is None:
            max_trials = len(self.trials)

        best_overall = None
        best_residual = np.inf
        best_fun = None
        best_params = None
        best_label = None
        fit_status = False

        for i, (fun, p0, label, singularity_check) in enumerate(self.trials[:max_trials]):
            
            if fit_status: continue

            if self.verbose:
                print(f"Trying {label}...")

            # Try logloss
            logloss_fn = lambda params: np.nansum((np.log10(self.y) - np.log10(fun(self.x, *params)))**2)
            try:
                opt_log = minimize(logloss_fn, x0=p0, method='Nelder-Mead')
                res_log = logloss_fn(opt_log.x)
                singular_log = singularity_check(opt_log.x) if singularity_check else False
                if self.verbose:
                    print(f"  logloss: Residual={res_log:.2e}, Success={opt_log.success}, Singular={singular_log}")
            except Exception as e:
                if self.verbose:
                    print(f"  logloss failed: {e}")
                opt_log, res_log, singular_log = None, np.inf, True

            # Try loss
            loss_fn = lambda params: np.nansum((self.y - fun(self.x, *params))**2)
            try:
                opt_lin = minimize(loss_fn, x0=p0, method='Nelder-Mead')
                res_lin = logloss_fn(opt_lin.x)
                singular_lin = singularity_check(opt_lin.x) if singularity_check else False
                if self.verbose:
                    print(f"  loss:    Residual={res_lin:.2e}, Success={opt_lin.success}, Singular={singular_lin}")
            except Exception as e:
                if self.verbose:
                    print(f"  loss failed: {e}")
                opt_lin, res_lin, singular_lin = None, np.inf, True
            
            # Choose the better fit (lowest residual, not singular, and optimizer succeeded)
            candidates = [
                (opt_log, res_log, 'logloss', singular_log),
                (opt_lin, res_lin, 'loss', singular_lin)
            ]
            for opt, res, loss_label, singular in candidates:
                if opt is not None and not singular and res < best_residual and res < tol:
                    best_overall = opt
                    best_residual = res
                    best_fun = fun
                    best_params = opt.x
                    best_label = f"{label} ({loss_label})"
                    if self.verbose:
                        print(f"  Accepted {best_label} with params {opt.x}")
                    
                    # Stop further trials if a fit is accepted
                    self.result = best_overall
                    self.best_fun = best_fun
                    self.best_fun_ind = i
                    self.best_params = best_params
                    self.residual = best_residual
                    if self.verbose:
                        print(f"Best fit: {best_label} with residual {best_residual:.2e}")
                    
                    fit_status = True
                    
        if fit_status:

            if best_residual * improvement_ratio > self.linear_fit_loss:
                if self.verbose:
                    print(f"Best fit residual {best_residual:.2e} is not significantly better than linear fit {self.linear_fit_loss:.2e}.")
                self.best_fun = lambda x, *params: x

                return None, -1, self.linear_fit_loss, self.linear_fit_loss

            return self.result, self.best_fun_ind, best_residual, self.linear_fit_loss                   

        if self.verbose:
            print("All fits failed or residuals too high.")

        return None, np.nan, None, self.linear_fit_loss



    def predict(self, x=None):
        if self.best_fun is None or self.best_params is None:
            raise RuntimeError("No fit available. Call fit() first.")
        if x is None:
            x = self.x
        return self.best_fun(x, *self.best_params)

    def plot(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        plt.plot(x, y, 'o', color='black')
        _xarr = np.logspace(-1, 4, 100)
        plt.plot(_xarr, self.predict(_xarr), color='red', lw=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(_xarr, _xarr, color='gray')
        plt.title('Nonlinearity model')
        plt.xlabel('Observed counts')
        plt.ylabel('Corrected counts')
        plt.savefig('nonlinearity_model.png')
        plt.clf()

    def plot_ratio(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        plt.plot(x, y/x, 'o', color='black')
        _xarr = np.logspace(-1, 4, 100)
        plt.plot(_xarr, self.predict(_xarr)/_xarr, color='red', lw=2)
        plt.xscale('log')
        plt.axhline(1, color='gray')
        plt.title('Nonlinearity model')
        plt.xlabel('Observed counts')
        plt.ylabel('Corrected counts / Observed counts')
        plt.savefig('nonlinearity_model_ratio.png')
        plt.clf()

class DetectorNonlinearityModel:

    def __init__(self, modelname=None):

        if modelname is not None:
            self.model = np.load(modelname+'.npz')
            print(f"Model loaded from {modelname}.npz")

    def create_model(self, observed_ims, corrected_ims, refind = -1, mincounts=300, tol=1, improvement_ratio=5, max_trials=None, plot=False):
        # nx, ny = observed_ims.shape[1:3]
        # funinds = np.full((nx, ny), np.nan)
        # minvals = np.full((nx, ny), np.nan)
        # maxvals = np.full((nx, ny), np.nan)
        # params = np.full((nx, ny, 4), np.nan)

        # def fit_pixel(idx):
        #     x, y = idx
        #     x_to_fit = observed_ims[:, x, y]
        #     y_to_fit = corrected_ims[:, x, y]
        #     minval = np.nanmin(x_to_fit)
        #     maxval = np.nanmax(x_to_fit)
        #     if x_to_fit[refind] < mincounts:
        #         return (x, y, np.nan, minval, maxval, [np.nan]*4)
        #     fitter = NonlinearityFitter(verbose=False)
        #     fitter.prep_fit(x_to_fit, y_to_fit)
        #     result, funind, fit_loss, linear_loss = fitter.fit(tol=tol, improvement_ratio=improvement_ratio, max_trials=max_trials)
        #     best_params = [np.nan]*4
        #     if fitter.best_params is not None:
        #         best_params[:len(fitter.best_params)] = fitter.best_params
        #     return (x, y, funind, minval, maxval, best_params)

        # idxs = [(x, y) for x in range(nx) for y in range(ny)]
        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     for x, y, funind, minval, maxval, best_params in tqdm(executor.map(fit_pixel, idxs), total=len(idxs)):
        #         funinds[x, y] = funind
        #         minvals[x, y] = minval
        #         maxvals[x, y] = maxval
        #         params[x, y, :] = best_params

        # self.model = {
        #     'funinds': funinds,
        #     'params': params,
        #     'minvals': minvals,
        #     'maxvals': maxvals
        # }
        # return self.model

        funinds = np.zeros((observed_ims.shape[1], observed_ims.shape[2]))
        minvals = np.zeros((observed_ims.shape[1], observed_ims.shape[2]))
        maxvals = np.zeros((observed_ims.shape[1], observed_ims.shape[2]))
        params = np.zeros((observed_ims.shape[1], observed_ims.shape[2], 4))
        # model_array = np.zeros((observed_ims.shape[1], observed_ims.shape[2], 6))
        
        self.fitter = NonlinearityFitter(verbose=False)
        
        for x in tqdm(range(observed_ims.shape[1])):
            for y in range(observed_ims.shape[2]):
                x_to_fit = observed_ims[:, x, y]
                y_to_fit = corrected_ims[:, x, y]

                minvals[x, y] = np.nanmin(x_to_fit)
                maxvals[x, y] = np.nanmax(x_to_fit)

                if x_to_fit[refind] < mincounts:
                    # print('skipping pixel (%d, %d) due to low flux %d' % (x, y, x_to_fit[refind]))
                    funinds[x, y] = np.nan
                    continue
                
                self.fitter.prep_fit(x_to_fit, y_to_fit)

                result, funind, fit_loss, linear_loss = self.fitter.fit(tol=tol, 
                                                                        improvement_ratio=improvement_ratio,
                                                                        max_trials=max_trials)
                if plot:
                    self.fitter.plot()
                    self.fitter.plot_ratio()
                
                try:
                    funinds[x, y] = funind
                    params[x, y, :len(self.fitter.best_params)] = self.fitter.best_params
                except:
                    funinds[x, y] = np.nan
                    params[x, y, :] = np.nan
                    print(result, funind, fit_loss, linear_loss)
                    print(f"Failed to fit pixel ({x}, {y})")
        self.model = {
            'funinds': funinds,
            'params': params,
            'minvals': minvals,
            'maxvals': maxvals
        }
        return self.model
    
    def save_model(self, filename):

        np.savez(filename, **self.model)
        print(f"Model saved to {filename}.npz")

    def correct_nonlinearity(self, observed_val, funind, minval, maxval, params):

        if (observed_val < minval) or (observed_val > maxval):
            # if the observed value is out of bounds, return the observed value
            print(f"Observed value {observed_val} is out of bounds ({minval}, {maxval}).")
            return observed_val, 2
        
        if funind *0 !=  0:
            # failure case
            return observed_val, 3
        
        if funind == -1:
            # linear case
            return observed_val, 1
        
        else:
            fitter = NonlinearityFitter(verbose=False)
            corrected = fitter.eval_from_model(observed_val, int(funind), params)
            return corrected, 0
        
    # def correct_map(self, observed_map):

    #     corrected_map = np.zeros_like(observed_map)
    #     status = np.zeros_like(observed_map, dtype=int)
    #     for x in tqdm(range(observed_map.shape[0])):
    #         for y in range(observed_map.shape[1]):
    #             funind = self.model['funinds'][x, y]
    #             minval = self.model['minvals'][x, y]
    #             maxval = self.model['maxvals'][x, y]
    #             params = self.model['params'][x, y]#[:]
                
    #             corrected_map[x, y], status[x, y] = self.correct_nonlinearity(observed_map[x, y], funind, minval, maxval, params)
        
    #     return corrected_map, status

    

    def correct_map(self, observed_map, max_workers=8):

        import time
        start_t = time.time()

        corrected_map = np.copy(observed_map)
        status = np.zeros_like(observed_map, dtype=int)

        funinds = self.model['funinds']
        minvals = self.model['minvals']
        maxvals = self.model['maxvals']
        params = self.model['params']

        # Mask for linear case
        mask_linear = (funinds == -1)
        corrected_map[mask_linear] = observed_map[mask_linear]
        status[mask_linear] = 1

        # Mask for skip/failure case
        mask_skip = (np.isnan(funinds)) | (observed_map < minvals) | (observed_map > maxvals)
        corrected_map[mask_skip] = observed_map[mask_skip]
        status[mask_skip] = 2

        # Mask for nonlinear case
        mask_nonlinear = ~(mask_linear | mask_skip)
        idxs = np.argwhere(mask_nonlinear)

        def correct_pixel(idx):
            x, y = idx
            funind = int(funinds[x, y])
            param = params[x, y]
            fitter = NonlinearityFitter(verbose=False)
            return fitter.eval_from_model(observed_map[x, y], funind, param)

        # Parallel correction for nonlinear pixels
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(correct_pixel, idxs))

        for (x, y), val in zip(idxs, results):
            corrected_map[x, y] = val
            status[x, y] = 0

        end_t = time.time()
        print("time elapsed: ", end_t - start_t)
        return corrected_map, status