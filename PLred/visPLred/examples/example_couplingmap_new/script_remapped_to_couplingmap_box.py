import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import h5py
import os, glob
import PLred.visPLred.preprocess as pp
from PLred.visPLred import spec as sp
from scipy.sparse import load_npz
from tqdm import tqdm

# data directory
data_dir = '/mnt/datazpool/PL/yjkim/remapped/mizarA_20250514/grid_binned_08:31:00_08:36:59/'
data_header = 'remapped'

# dark file
# dark = fits.getdata('/mnt/datazpool/PL/20250514/darks/firstpl_15:33:15.921714388.fits')
# dark = np.average(dark, axis=0)

dark = fits.getdata('/mnt/datazpool/PL/20250514/darks/firstpl_15:30:26.681277758.fits')
dark = np.average(dark, axis=0)

# boostrap number
nboot = 50
nframes_cut = 5

# map_n
n = 15

# nonlinearity model name
modelname = '/mnt/datazpool/PL/yjkim/flat_characterization/2025-05-12/flux_dependent_flat/model3_800_1500'
# xrange = np.r_[1100:1300]
xmin = 800
xmax = 1500

from PLred.visPLred import spec as sp
specmodelname = './mizar'
sm = sp.SpectrumModel(specmodelname)
datadir = '/mnt/datazpool/PL/yjkim/remapped/mizarA_20250514/'
frame = h5py.File(datadir + 'grid_binned_08:31:00_08:36:59/remapped_bin_7_7.h5', 'r')
dark2 = fits.getdata('/mnt/datazpool/PL/20250514/darks/firstpl_15:30:26.681277758.fits')
sm.flat = np.average(frame['rawframes'][:],axis=0) - np.average(dark2,axis=0)
neon_chunk = fits.getdata('/mnt/datazpool/PL/20250513/firstpl/firstpl_15:53:02.390432600.fits')
neon_dark_chunk = fits.getdata('/mnt/datazpool/PL/20250513/firstpl/firstpl_16:02:22.983506040.fits')
neon = np.mean(neon_chunk, axis=0) - np.mean(neon_dark_chunk, axis=0)

sm.neon = neon
sm.find_peaks()

ini_wav_ind=3
sm.trace_spectra(ini_wav_ind)

# # specmodel
# specmodel = '/home/first/yjkim/specmodels/2025-02-07/model_decemberneon/'

# matrix = load_npz(specmodel+'model_matrix.npz')
# info = np.load(specmodel+'model_info.npy', allow_pickle=True).item()
# xmin, xmax = int(info['xmin']), int(info['xmax'])
# wav_map = np.load(specmodel+'model_wavmap.npy')

def rawframe_to_spec(rawframe, dark,
                     nonlinearity_model,
                    #  matrix,
                     xmin, xmax): #, wav_map):

    im_to_correct = (rawframe - dark)#[:,xrange]
    # print(np.shape(im_to_correct))

    print("start correcting nonlinearity")
    # nonlinearity correction
    model = pp.DetectorNonlinearityModel(modelname =nonlinearity_model)
    corrected_map, status = model.correct_map(im_to_correct[:,xmin:xmax])#[:, xrange])
    # mask = status < 2
    # mask = corrected_map *0 != 0

    # patch correction
    frame = im_to_correct.copy()
    frame[:,xmin:xmax] = corrected_map
    # badpix = np.zeros_like(frame, dtype=bool)
    # badpix[:,xmin:xmax] = mask

    print("start extracting spectrum")
    spec_box = sp.frame_to_spec(frame, xmin, xmax, traces = np.array(sm.trace_vals)[:,xmin - sm.XMIN:xmax - sm.XMIN])

    return spec_box



# mapfiles
mapfiles = [data_dir+data_header+'_bin_%d_%d.h5' % (i, j) for i in range(n) for j in range(n)]

# mapfiles
# mapfiles = sorted(glob.glob(data_dir+data_header+'_bin_%.h5'))

for mapfile in tqdm(mapfiles):

    if not os.path.exists(mapfile):
        print("file %s does not exist" % mapfile)
        continue

    # load map
    with h5py.File(mapfile, 'r') as f:
        mapdata = f['rawframes'][:]

    nframes = mapdata.shape[0]
    if nframes < nframes_cut:
        print("file %s has only %d frames, skipping" % (mapfile, nframes))
        continue

    # create hdf5 file
    outname = mapfile.replace('.h5', '_spec.h5')
    with h5py.File(outname, 'w') as h5file:
        print("creating %s" % outname)
        h5file.attrs['num_frames'] = nframes

        avgspec_h5 = h5file.create_dataset('avgspec', shape = (38, xmax-xmin))
        bootspec_h5 = h5file.create_dataset('bootspecs', shape = (nboot, 38, xmax-xmin))

        # get average
        avgframe = np.nanmean(mapdata, axis=0)

        # get average spectrum
        avgspec = rawframe_to_spec(avgframe, dark, modelname, xmin, xmax)
        # save average spectrum
        # with h5py.File(outname, 'r+') as h5f:
        #     h5f['avgspec'][:] = avgspec
        avgspec_h5[:] = avgspec

        # bootspecs = []
        for i in tqdm(range(nboot)):
            print("bootstrapping %d" % i)
            # get bootstrapped index
            boot_ind = np.random.choice(nframes, nframes, replace=True)

            # get bootstrapped spectrum
            bootframes = mapdata[boot_ind]
            bootframe = np.average(bootframes, axis=0)
            spec = rawframe_to_spec(bootframe, dark, modelname, xmin, xmax)
            # bootspecs.append(spec)
            # save bootstrapped spectrum
            # bootspec_h5[i] = spec

            # with h5py.File(outname, 'r+') as h5f:
            #     h5f['bootspecs'][i] = spec
            bootspec_h5[i] = spec

        


    


    
    





    # # get the spectrum
    # spec = rawframe_to_spec(rawframe, dark, modelname, matrix, xmin, xmax, wav_map)

    # # save the spectrum
    # np.save(data_dir+mapname+'_spec.npy', spec)

    







    

    

# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
# import os, glob
# from nonlinearity_curve_fitter import NonlinearityFitter, DetectorNonlinearityModel


# data = fits.getdata('/mnt/datazpool/PL/yjkim/reduced_map/betcmi_20250211/betcmi_20250211_couplingmap_frames_12:11:00_12:19:09.fits')
# dark = fits.getdata('/mnt/datazpool/PL/yjkim/reduced/betcmi_20250211/dark.fits')
# frame = data -dark[None,None,:,:]

# corrected_map = np.zeros_like(frame)[:,:,:,1100:1300]
# status = np.zeros_like(frame, dtype=int)[:,:,:,1100:1300]
# for i in range(np.shape(frame)[0]):
#     for j in range(np.shape(frame)[1]):
        

#         im_to_correct = frame[i,j,:, 1100:1300]
#         model = DetectorNonlinearityModel(modelname = 'model')
#         corrected_map[i,j], status[i,j] = model.correct_map(im_to_correct)
