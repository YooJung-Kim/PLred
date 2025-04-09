import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from configobj import ConfigObj
import sys, os, glob



if __name__ == "__main__":

    configname = sys.argv[1]

    # Parse config

    config = ConfigObj(configname)

    path_input = config['Input']['input_dir']
    input_map = config['Input']['input_map']
    input_wav = config['Input']['input_wav']

    output_dir = config['Output']['output_dir']
    output_name = config['Output']['output_name']

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + output_name + '/', exist_ok=True)

    print(f'Output directory: {output_dir + output_name + "/"}')


    ########################
    # make polynomial model
    ########################

    exists = os.path.exists(output_dir + output_name + '/polymodel.fits')
    skip = config['Polymodel'].as_bool('skip_if_exists')

    if exists and skip:
        print(f'Polymodel exists. using {output_dir + output_name + "/polymodel.fits"}')
    else:
        print(f'Creating polynomial model: {output_dir + output_name + "/polymodel.fits"}')
        
        import PLred.mapmodel as mm

        min_nframes = int(config['Polymodel']['min_nframes'])

        mapmodel = mm.CouplingMapModel(mapdata = path_input + input_map,
                                       min_nframes = min_nframes)
        
        wav = np.load(path_input + input_wav)
        specinds = wav['specinds']
        # print(specinds)
        vmax = float(config['Polymodel']['vrange_max'])
        vmin = float(config['Polymodel']['vrange_min'])

        # range of wavelength to use for polynomial fitting.
        # should exclude the range of spectroastrometric signal
        wav_fitrange = specinds[(wav['v'] > vmax) | (wav['v'] < vmin)]
        wav_reconrange =  specinds

        print("making polynomial model for v range", vmin, vmax)
        print("length of recon range", len(specinds))
        print("length of fit range", len(wav_fitrange))

        # print(wav_fitrange, wav_reconrange)
        
        all_map_inputs, all_modeled_recons, all_modeled_coeffs, model_chi2 = \
            mapmodel.make_polynomial_model(output_dir + output_name + '/polymodel', wav_fitrange, wav_reconrange)
        
        # plot diagnostics
        fibinds_to_plot = config['Polymodel']['fibinds_to_plot']
        fibinds_to_plot = [int(i) for i in fibinds_to_plot]
        print('Making diagnostic plots for', fibinds_to_plot)

        if len(fibinds_to_plot) > 0:

            os.makedirs(output_dir + output_name + '/polymodel_plots/', exist_ok=True)
            for fibind in fibinds_to_plot:
                anim = mapmodel.diagnostic_plot_model(fibind)#, wav_reconrange)
                anim.save(output_dir + output_name + f'/polymodel_plots/fibind_{fibind}.gif')


    ##########################
    # make convolution matrix
    ##########################

    import PLred.fit as fit

    os.makedirs(output_dir + output_name + '/matrix/', exist_ok=True)


    wav = np.load(path_input + input_wav)
    specinds = wav['specinds']
    vmax2 = float(config['Convolution_matrix']['vrange_max'])
    vmin2 = float(config['Convolution_matrix']['vrange_min'])

    wav_to_make = specinds[(wav['v'] < vmax2) & (wav['v'] > vmin2)] - specinds[0]
    print(wav_to_make)

    image_ngrid = int(config['Convolution_matrix']['image_ngrid'])
    image_fov = float(config['Convolution_matrix']['image_fov'])
    n_trim = int(config['Convolution_matrix']['n_trim'])

    fitter = fit.PLMapFit(model_file = output_dir + output_name + '/polymodel.fits',
                          image_ngrid = image_ngrid,
                          image_fov = image_fov,
                          n_trim = n_trim)

    for specind in wav_to_make:

        print(f'Creating convolution matrix for specind {specind + specinds[0]}')

        exists = os.path.exists(output_dir + output_name + f'/matrix/matrix_{specind + specinds[0]}.fits')
        skip = config['Convolution_matrix'].as_bool('skip_if_exists')

        if exists and skip:
            print(f'Convolution matrix exists. using {output_dir + output_name + f"/matrix/matrix_{specind + specinds[0]}.fits"}')
        else:

            print(f'Creating convolution matrix: {output_dir + output_name + f"/matrix/matrix_{specind + specinds[0]}.fits"}')

            fibinds = np.arange(38)
            fitter.make_matrix(specind, fibinds)
            fitter.save_matrix_to_file(output_dir + output_name + f'/matrix/matrix_{specind + specinds[0]}.fits')
    
    ##########################
    # point source model fit
    ##########################


    from scipy.optimize import minimize

    wav = np.load(path_input + input_wav)
    specinds = wav['specinds']
    vmax2 = float(config['Gaussian_model']['vrange_max'])
    vmin2 = float(config['Gaussian_model']['vrange_min'])

    wav_to_use = specinds[(wav['v'] < vmax2) & (wav['v'] > vmin2)] 

    os.makedirs(output_dir + output_name + '/point_model/', exist_ok=True)

    nbootstrap = int(config['Point_source_model']['nbootstrap'])
    bootstrap_samples = [np.random.choice(np.arange(38), size=38, replace=True) for _ in range(nbootstrap)]

    # below added!

    n = int(config['Convolution_matrix']['n_trim'])
    def chi2_model_point(param, specind = 0, fibinds=np.arange(38)):
        chi2 = 0
        (x,y) = param
        # print(x,y)
        for fibind in fibinds:
            model = fitter.mapmodel.compute_vec(specind, fibind, x, y, n_trim=1).reshape((13,13))
            data = fitter.mapmodel.normdata[n:-n,n:-n,fibind,specind]
            datavar = fitter.mapmodel.datanormvar[n:-n,n:-n,fibind,specind]

            chi2 += np.nansum((model-data)**2 / datavar)
        
        return chi2/len(model.flatten())/len(fibinds)
    
    all_opts = []
    all_funs = []
    result_specinds = []
    bootstrap_opts = []
    for specind in wav_to_use:

        allopt = minimize(chi2_model_point, x0=[0,0], args=(specind - specinds[0],np.arange(38)))
        all_opts.append(allopt.x)

        result_specinds.append(specind)
        print('start specind', specind, "using all: (%.3f, %.3f)" % (allopt.x[0], allopt.x[1]))

        _opts = []
        for samp in bootstrap_samples:
            opt = minimize(chi2_model_point, x0=[0,0], args=(specind - specinds[0],samp))
            _opts.append(opt.x)
        
        bootstrap_opts.append(_opts)

        print("specind %d result: bootstrap std (%.3f, %.3f)" % (specind,
                                                                np.std(np.array(bootstrap_opts[-1])[:,0]),
                                                                np.std(np.array(bootstrap_opts[-1])[:,1])))



    # n_point_source = int(config['Point_source_model']['n_point_source'])

    # all_opts = []
    # all_funs = []
    # result_specinds = []
    # bootstrap_opts = []
    # bootstrap_funs = []

    # for specind in wav_to_use:

    #     fitter = fit.PLMapFit(matrix_file = output_dir + output_name + f'/matrix/matrix_{specind}.fits')

    #     fibinds = np.arange(38)
    #     fitter.prepare_data(fibinds)
    #     fitter.subsample_matrix(fibinds)

    #     ini_params =  np.array([0, 0])
    #     bounds =[(-5, 5), (-5, 5)]
        
    #     fitter.run_fitting_pointsource(n_point_source, ini_params, bounds=bounds)

    #     allopt = fitter.rc.opt
    #     all_opts.append(allopt.x)
    #     all_funs.append(allopt.fun)
    #     result_specinds.append(specind)

    #     print('start specind', specind, "using all: (%.3f, %.3f)" % (allopt.x[0], allopt.x[1]))


    #     _opts = []
    #     _funs = []
    #     for samp in bootstrap_samples:
            
    #         fitter.prepare_data(samp)
    #         fitter.subsample_matrix(samp)
    #         fitter.run_fitting_pointsource(n_point_source, ini_params, bounds=bounds)

    #         opt = fitter.rc.opt
    #         _opts.append(opt.x)
    #         _funs.append(opt.fun)
        
    #     bootstrap_opts.append(_opts)
    #     bootstrap_funs.append(_funs)

    #     print("specind %d result: bootstrap std (%.3f, %.3f)" % (specind,
    #                                                                         np.std(np.array(bootstrap_opts[-1])[:,0]),
    #                                                                         np.std(np.array(bootstrap_opts[-1])[:,1])))

        np.savez(output_dir + output_name + f'/point_model/point_model_bootstrap.npz',
                bootstrap_opts = bootstrap_opts, 
                result_specinds = result_specinds,
                all_opts = all_opts,
                all_funs = all_funs,
                boostrap_samples = bootstrap_samples,
                # bootstrap_funs = bootstrap_funs
                )


    





    ##########################
    # Gaussian model fit
    ##########################
    
    from scipy.optimize import minimize

    wav = np.load(path_input + input_wav)
    specinds = wav['specinds']
    vmax2 = float(config['Gaussian_model']['vrange_max'])
    vmin2 = float(config['Gaussian_model']['vrange_min'])

    wav_to_use = specinds[(wav['v'] < vmax2) & (wav['v'] > vmin2)] 

    os.makedirs(output_dir + output_name + '/gauss_model/', exist_ok=True)

    nbootstrap = int(config['Gaussian_model']['nbootstrap'])
    bootstrap_samples = [np.random.choice(np.arange(38), size=38, replace=True) for _ in range(nbootstrap)]

    all_opts = []
    all_funs = []
    result_specinds = []
    bootstrap_opts = []
    bootstrap_funs = []

    for specind in wav_to_use:

        fitter = fit.PLMapFit(matrix_file = output_dir + output_name + f'/matrix/matrix_{specind}.fits')

        fibinds = np.arange(38)
        fitter.prepare_data(fibinds)
        fitter.subsample_matrix(fibinds)

        ini_params =  np.array([fitter.mapmodel.image_ngrid//2, fitter.mapmodel.image_ngrid//2, 2])
        bounds = [(1, fitter.mapmodel.image_ngrid-1),
                                            (1, fitter.mapmodel.image_ngrid-1),
                                            (1.1e-2, fitter.mapmodel.image_ngrid)]
        
        fitter.run_fitting_gaussian(ini_params, bounds=bounds)

        allopt = fitter.rc.opt
        all_opts.append(allopt.x)
        all_funs.append(allopt.fun)
        result_specinds.append(specind)

        print('start specind', specind, "using all: (%.3f, %.3f, width %.3f)" % (allopt.x[0], allopt.x[1], allopt.x[2]))


        _opts = []
        _funs = []
        for samp in bootstrap_samples:
            
            fitter.prepare_data(samp)
            fitter.subsample_matrix(samp)
            fitter.run_fitting_gaussian(ini_params, bounds=bounds)

            opt = fitter.rc.opt
            _opts.append(opt.x)
            _funs.append(opt.fun)
        
        bootstrap_opts.append(_opts)
        bootstrap_funs.append(_funs)

        print("specind %d result: bootstrap std (%.3f, %.3f, width %.3f)" % (specind,
                                                                            np.std(np.array(bootstrap_opts[-1])[:,0]),
                                                                            np.std(np.array(bootstrap_opts[-1])[:,1]),
                                                                            np.std(np.array(bootstrap_opts[-1])[:,2])))

        np.savez(output_dir + output_name + f'/gauss_model/gauss_model_bootstrap.npz',
                bootstrap_opts = bootstrap_opts, 
                result_specinds = result_specinds,
                all_opts = all_opts,
                all_funs = all_funs,
                boostrap_samples = bootstrap_samples,
                bootstrap_funs = bootstrap_funs
                )


    


