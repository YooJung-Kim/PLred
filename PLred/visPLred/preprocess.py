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

            intstr = f.split('s_')[0].split('_')[-1]
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
        return observed_val, 3
    
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