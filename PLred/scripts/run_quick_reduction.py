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
    os.system('cp ' + configname + ' ' + output_dir + output_name + '/')

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

    exists = os.path.exists(output_dir + output_name + '/point_model/point_model_bootstrap.npz')
    skip = config['Point_source_model'].as_bool('skip_if_exists')

    if exists and skip:
        print('Point source model exists. skipping...')

    else:


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
        matsize = (fitter.mapmodel.map_n) - 2*n
        def chi2_model_point(param, specind = 0, fibinds=np.arange(38)):
            chi2 = 0
            (x,y) = param
            # print(x,y)
            for fibind in fibinds:
                model = fitter.mapmodel.compute_vec(specind, fibind, x, y, n_trim=n).reshape((matsize,matsize))
                data = fitter.mapmodel.normdata[n:-n,n:-n,fibind,specind]
                datavar = fitter.mapmodel.datanormvar[n:-n,n:-n,fibind,specind]

                chi2 += np.nansum((model-data)**2 / datavar)
                ndf = np.sum(np.isfinite(datavar))
            return chi2/ndf/len(fibinds)
        
        all_opts = []
        all_funs = []
        result_specinds = []
        bootstrap_opts = []
        for specind in wav_to_use:

            allopt = minimize(chi2_model_point, x0=[0,0], args=(specind - specinds[0],np.arange(38)))
            all_opts.append(allopt.x)
            all_funs.append(allopt.fun)

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

        # plot
        fig, axs = plt.subplots(figsize= (8,4), ncols=2)
        bootstrap_std = np.nanstd(bootstrap_opts, axis=1)
        scale = 1 #fitter.mapmodel.image_fov / fitter.mapmodel.image_ngrid
        for i in range(2):
            axs[i].errorbar(result_specinds,
                            np.array(all_opts)[:,i] * scale,
                            yerr = bootstrap_std[:,i] * scale,
                            fmt = 'o-', ms=2,
                            color='C%d' % i)
        axs[0].set_title('x (mas)')
        axs[1].set_title('y (mas)')
        fig.savefig(output_dir + output_name + '/point_model/point_model_results.png')

    


    ##########################
    # Gaussian model fit
    ##########################

    exists = os.path.exists(output_dir + output_name + '/gauss_model/gauss_model_bootstrap.npz')
    skip = config['Gaussian_model'].as_bool('skip_if_exists')

    if exists and skip:
        print('Gaussian model exists. skipping...')
    else:
    
        from scipy.optimize import minimize

        wav = np.load(path_input + input_wav)
        specinds = wav['specinds']
        vmax2 = float(config['Gaussian_model']['vrange_max'])
        vmin2 = float(config['Gaussian_model']['vrange_min'])

        wav_to_use = specinds[(wav['v'] < vmax2) & (wav['v'] > vmin2)] 

        # point source fraction
        point_source_frac_file = config['Gaussian_model']['point_source_frac'].strip()
        print('filename',point_source_frac_file, type(point_source_frac_file))
        if point_source_frac_file != '':
            point_source_frac_file = np.load(point_source_frac_file)
            fracs = []
            for specind in wav_to_use:
                w = np.where(specind == point_source_frac_file['specinds'])[0][0]
                frac = point_source_frac_file['frac'][w]
                if frac < 0.90:
                    fracs.append(frac)
                else:
                    fracs.append(0)
        else:
            fracs = [0] * len(wav_to_use)
        fracs = np.array(fracs)

        os.makedirs(output_dir + output_name + '/gauss_model/', exist_ok=True)

        nbootstrap = int(config['Gaussian_model']['nbootstrap'])
        bootstrap_samples = [np.random.choice(np.arange(38), size=38, replace=True) for _ in range(nbootstrap)]

        all_opts = []
        all_funs = []
        result_specinds = []
        bootstrap_opts = []
        bootstrap_funs = []
        central_fracs = []

        for specind in wav_to_use:

            fitter = fit.PLMapFit(matrix_file = output_dir + output_name + f'/matrix/matrix_{specind}.fits')

            fibinds = np.arange(38)
            fitter.prepare_data(fibinds)
            fitter.subsample_matrix(fibinds)

            ini_params =  np.array([fitter.mapmodel.image_ngrid//2, fitter.mapmodel.image_ngrid//2, 2])
            bounds = [(1, fitter.mapmodel.image_ngrid-1),
                                                (1, fitter.mapmodel.image_ngrid-1),
                                                (1.1e-2, fitter.mapmodel.image_ngrid)]
            
            fitter.run_fitting_gaussian(ini_params, bounds=bounds,
                                        central_point_source_flux=fracs[np.where(wav_to_use == specind)[0][0]])

            allopt = fitter.rc.opt
            all_opts.append(allopt.x)
            all_funs.append(allopt.fun)
            result_specinds.append(specind)
            central_fracs.append(fitter.rc.central_point_source_flux)

            print('start specind', specind, "using all: (%.3f, %.3f, width %.3f)" % (allopt.x[0], allopt.x[1], allopt.x[2]))


            _opts = []
            _funs = []
            for samp in bootstrap_samples:
                
                fitter.prepare_data(samp)
                fitter.subsample_matrix(samp)
                fitter.run_fitting_gaussian(ini_params, bounds=bounds,
                                            central_point_source_flux=fracs[np.where(wav_to_use == specind)[0][0]])

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
                    bootstrap_funs = bootstrap_funs,
                    central_fracs = central_fracs
                    )
            
        # plot
        fig, axs = plt.subplots(figsize= (16,4), ncols=4)
        bootstrap_std = np.nanstd(bootstrap_opts, axis=1)
        offsets = [fitter.mapmodel.image_ngrid//2, fitter.mapmodel.image_ngrid//2, 0]
        scale = fitter.mapmodel.image_fov / fitter.mapmodel.image_ngrid
        for i in range(3):
            axs[i].errorbar(result_specinds,
                            (np.array(all_opts)[:,i] - offsets[i]) * (scale),
                            yerr = bootstrap_std[:,i] * scale,
                            fmt = 'o-', ms=2,
                            color='C%d' % i)

        axs[3].plot(result_specinds, central_fracs, 'o-')
        axs[0].set_title('x (mas)')
        axs[1].set_title('y (mas)')
        axs[2].set_title('width (mas)')
        axs[3].set_title('central point source fraction')
        fig.savefig(output_dir + output_name + '/gauss_model/gauss_model_results.png')

    


    ##########################
    # Image reconstruction
    ##########################

    imgrecon_name = 'imgrecon_' + config['Image_reconstruction']['name']
    
    exists = os.path.exists(output_dir + output_name + '/' + imgrecon_name +'/img_recon.npz')
    skip = config['Image_reconstruction'].as_bool('skip_if_exists')

    if exists and skip:
        print('Image reconstruction exists. skipping...')
    else:

        from PLred.imgrecon import locs2image

        wav = np.load(path_input + input_wav)
        specinds = wav['specinds']
        vmax2 = float(config['Image_reconstruction']['vrange_max'])
        vmin2 = float(config['Image_reconstruction']['vrange_min'])

        wav_to_use = specinds[(wav['v'] < vmax2) & (wav['v'] > vmin2)] 

        # point source fraction
        point_source_frac_file = config['Image_reconstruction']['point_source_frac'].strip()
        print('filename',point_source_frac_file, type(point_source_frac_file))
        if point_source_frac_file != '':
            point_source_frac_file = np.load(point_source_frac_file)
            fracs = []
            for specind in wav_to_use:
                w = np.where(specind == point_source_frac_file['specinds'])[0][0]
                frac = point_source_frac_file['frac'][w]
                if frac < 0.90:
                    fracs.append(frac)
                else:
                    fracs.append(None)
        else:
            fracs = [None] * len(wav_to_use)
        fracs = np.array(fracs)

        os.makedirs(output_dir + output_name + '/' + imgrecon_name, exist_ok=True)
        os.makedirs(output_dir + output_name + '/' + imgrecon_name + '/plots/', exist_ok=True)

        plot_residuals = config['Image_reconstruction'].as_bool('plot_residuals')
        if plot_residuals:
            os.makedirs(output_dir + output_name + '/' + imgrecon_name + '/residuals/', exist_ok=True)
            os.makedirs(output_dir + output_name + '/' + imgrecon_name + '/residuals2d/', exist_ok=True)
            os.makedirs(output_dir + output_name + '/' + imgrecon_name + '/residuals2d_SN/', exist_ok=True)

        # hyperparameters

        ini_temp = float(config['Image_reconstruction']['ini_temp'])
        n_iter = int(config['Image_reconstruction']['n_iter'])
        tau = float(config['Image_reconstruction']['tau'])
        gamma = float(config['Image_reconstruction']['gamma'])
        n_element = int(config['Image_reconstruction']['n_element'])
        burn_in_iter = int(config['Image_reconstruction']['burn_in_iter'])
        target_chi2 = float(config['Image_reconstruction']['target_chi2'])

        hyperparam_dict = {'ini_temp': ini_temp,
                            'n_iter': n_iter,
                            'tau': tau,
                            'gamma': gamma,
                            'n_element': n_element,
                            'target_chi2': target_chi2,
                            'burn_in_iter': burn_in_iter}

        all_ims = []
        all_lls = []
        all_fracs = []

        for specind in wav_to_use:

            fitter = fit.PLMapFit(matrix_file = output_dir + output_name + f'/matrix/matrix_{specind}.fits')

            fibinds = np.arange(38)
            fitter.prepare_data(fibinds)
            fitter.subsample_matrix(fibinds)
            fitter.store_hyperparams(ini_temp, tau, gamma, n_element, target_chi2, regul_dict={})

            frac = fracs[np.where(wav_to_use == specind)[0][0]]
            rc = fitter.run(small_to_random_ratio = 1,
                            centerfrac = frac,
                            niter = n_iter,
                            radius = 40,
                            burn_in_iter = burn_in_iter)
            
            all_ims.append(locs2image(rc.post_locs, rc.axis_len))
            all_lls.append(rc.current_ll)
            all_fracs.append(frac)

            # save the results
            fig, axs = plt.subplots(figsize= (10,5), ncols=2)
            im = locs2image(rc.post_locs, rc.axis_len).copy()
            axs[0].imshow(im)
            
            im[fitter.mapmodel.image_ngrid//2, fitter.mapmodel.image_ngrid//2] = None
            axs[1].imshow(im)
            
            
            fig.suptitle(f'specind = %d, ll = %.4f, frac = {frac}' % (specind, rc.current_ll))
                
            fig.savefig(output_dir + output_name + '/' + imgrecon_name + '/plots/specind_%d.png' % (specind))


            if plot_residuals:

                # plot residuals
                for fibind in fibinds:
                    fig = fitter.plot_1d(fibind, return_fig = True)
                    fig.savefig(output_dir + output_name + '/' + imgrecon_name + '/residuals/residuals_specind%d_fibind%d.png' % (specind, fibind))

                fig = fitter.plot_residuals(return_fig = True)
                fig.savefig(output_dir + output_name + '/' + imgrecon_name + '/residuals2d/residuals_specind%d.png' % (specind))

                fig = fitter.plot_residuals(return_fig = True, SN= True, vmax = 10)
                fig.savefig(output_dir + output_name + '/' + imgrecon_name + '/residuals2d_SN/residuals_specind%d.png' % (specind))


            np.savez(output_dir + output_name + '/' + imgrecon_name + '/img_recon.npz', 
                     all_ims = np.array(all_ims), 
                     all_lls = np.array(all_lls),
                     specinds = wav_to_use, 
                     fibinds = fibinds,
                     hyperparam_dict = hyperparam_dict)
            
            

    ##########################
    # Elliptical Gaussian model fit
    ##########################

    exists = os.path.exists(output_dir + output_name + '/elliptical_gauss_model/elliptical_gauss_model_bootstrap.npz')
    skip = config['Elliptical_Gaussian_model'].as_bool('skip_if_exists')

    fix_PA_value = float(config['Elliptical_Gaussian_model']['fix_PA_value'])

    if exists and skip:
        print('Elliptical Gaussian model exists. skipping...')
    else:
    
        from scipy.optimize import minimize

        wav = np.load(path_input + input_wav)
        specinds = wav['specinds']
        vmax2 = float(config['Elliptical_Gaussian_model']['vrange_max'])
        vmin2 = float(config['Elliptical_Gaussian_model']['vrange_min'])

        wav_to_use = specinds[(wav['v'] < vmax2) & (wav['v'] > vmin2)] 

        # point source fraction
        point_source_frac_file = config['Elliptical_Gaussian_model']['point_source_frac'].strip()
        print('filename',point_source_frac_file, type(point_source_frac_file))
        if point_source_frac_file != '':
            point_source_frac_file = np.load(point_source_frac_file)
            fracs = []
            for specind in wav_to_use:
                w = np.where(specind == point_source_frac_file['specinds'])[0][0]
                frac = point_source_frac_file['frac'][w]
                if frac < 0.90:
                    fracs.append(frac)
                else:
                    fracs.append(0)
        else:
            fracs = [0] * len(wav_to_use)
        fracs = np.array(fracs)

        os.makedirs(output_dir + output_name + '/elliptical_gauss_model/', exist_ok=True)

        nbootstrap = int(config['Elliptical_Gaussian_model']['nbootstrap'])
        bootstrap_samples = [np.random.choice(np.arange(38), size=38, replace=True) for _ in range(nbootstrap)]

        all_opts = []
        all_funs = []
        result_specinds = []
        bootstrap_opts = []
        bootstrap_funs = []
        central_fracs = []

        for specind in wav_to_use:

            fitter = fit.PLMapFit(matrix_file = output_dir + output_name + f'/matrix/matrix_{specind}.fits')

            fibinds = np.arange(38)
            fitter.prepare_data(fibinds)
            fitter.subsample_matrix(fibinds)

            ini_params =  np.array([fitter.mapmodel.image_ngrid//2, fitter.mapmodel.image_ngrid//2, 
                                    2, 2])
            bounds = [(1, fitter.mapmodel.image_ngrid-1),
                        (1, fitter.mapmodel.image_ngrid-1),
                        (1.1e-2, fitter.mapmodel.image_ngrid),
                        (1.1e-2, fitter.mapmodel.image_ngrid)]
            
            fitter.run_fitting_gaussian(ini_params, bounds=bounds,
                                        fix_PA_value=fix_PA_value,
                                        central_point_source_flux=fracs[np.where(wav_to_use == specind)[0][0]])

            allopt = fitter.rc.opt
            all_opts.append(allopt.x)
            all_funs.append(allopt.fun)
            result_specinds.append(specind)
            central_fracs.append(fitter.rc.central_point_source_flux)

            print('start specind', specind, "using all: (%.3f, %.3f, width %.3f, %.3f)" % (allopt.x[0], allopt.x[1], allopt.x[2], allopt.x[3]))


            _opts = []
            _funs = []
            for samp in bootstrap_samples:
                
                fitter.prepare_data(samp)
                fitter.subsample_matrix(samp)
                fitter.run_fitting_gaussian(ini_params, bounds=bounds,
                                            fix_PA_value=fix_PA_value,
                                            central_point_source_flux=fracs[np.where(wav_to_use == specind)[0][0]])

                opt = fitter.rc.opt
                _opts.append(opt.x)
                _funs.append(opt.fun)
            
            bootstrap_opts.append(_opts)
            bootstrap_funs.append(_funs)

            print("specind %d result: bootstrap std (%.3f, %.3f, width %.3f, %.3f)" % (specind,
                                                                                np.std(np.array(bootstrap_opts[-1])[:,0]),
                                                                                np.std(np.array(bootstrap_opts[-1])[:,1]),
                                                                                np.std(np.array(bootstrap_opts[-1])[:,2]),
                                                                                np.std(np.array(bootstrap_opts[-1])[:,3])))

            np.savez(output_dir + output_name + f'/elliptical_gauss_model/elliptical_gauss_model_bootstrap.npz',
                    bootstrap_opts = bootstrap_opts, 
                    result_specinds = result_specinds,
                    all_opts = all_opts,
                    all_funs = all_funs,
                    boostrap_samples = bootstrap_samples,
                    bootstrap_funs = bootstrap_funs,
                    central_fracs = central_fracs
                    )
            
        # plot
        fig, axs = plt.subplots(figsize= (16,4), ncols=5)
        bootstrap_std = np.nanstd(bootstrap_opts, axis=1)
        offsets = [fitter.mapmodel.image_ngrid//2, fitter.mapmodel.image_ngrid//2, 0, 0]
        scale = fitter.mapmodel.image_fov / fitter.mapmodel.image_ngrid
        for i in range(4):
            axs[i].errorbar(result_specinds,
                            (np.array(all_opts)[:,i] - offsets[i]) * (scale),
                            yerr = bootstrap_std[:,i] * scale,
                            fmt = 'o-', ms=2,
                            color='C%d' % i)

        axs[3].plot(result_specinds, central_fracs, 'o-')
        axs[0].set_title('x (mas)')
        axs[1].set_title('y (mas)')
        axs[2].set_title('width x (mas)')
        axs[3].set_title('width y (mas)')
        axs[4].set_title('central point source fraction')
        fig.savefig(output_dir + output_name + '/elliptical_gauss_model/elliptical_gauss_model_results.png')




