
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from PLred.visPLred import couplingmap as cm

firstcam_timestamp_path = '/mnt/datazpool/PL/yjkim/binned_observation/mizarA_20250514/20250514/firstpl/'
firstcam_spec_path =      '/mnt/datazpool/PL/yjkim/binned_observation/mizarA_20250514/20250514/firstpl/'

# obs_start = '08:31:00'
# obs_end   = '08:36:59' #'08:36:59'

# obs_starts = ['14:35:00', '14:45:00', '14:45:00', '14:55:00']
# obs_ends = ['14:45:00', '14:55:00', '14:55:00', '15:10:00']

psfcam = 'palila'
psfcam_frames_name = '/mnt/datazpool/PL/yjkim/reduced/mizarA_binned_20250514/first_palila_matched_fastcam_matched_frames.npy'
psfcam_timestamp_name = '/mnt/datazpool/PL/yjkim/reduced/mizarA_binned_20250514/first_palila_matched.pkl'



obs_start = '08:31:00'
obs_end   = '08:36:59' #'08:36:59'

sd = cm.SimultaneousData(firstcam_timestamp_path, firstcam_spec_path,
                obs_start, obs_end,
                psfcam, 
                psfcam_frames_name, psfcam_timestamp_name,
                store_spec = False,)


sd.compute_psfcam_centroids(peak=True)
sd.bin_by_centroids(15, 3, calculate_variance = False,
                    to_file = True,
                    filename = f'/mnt/datazpool/PL/yjkim/remapped/mizarA_20250514/remapped_binned_{obs_start}_{obs_end}')#, return_bootstrap_samples= True)


# sd.save(f'betcmi_20250211_couplingmap_frames_{obs_start}_{obs_end}.fits')
# sd.save_bootstrap_frames(f'betcmi_20250211_couplingmap_frames_{obs_start}_{obs_end}')



# import matplotlib
# matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits

# from PLred.visPLred import couplingmap as cm
# import pickle
# from PLred.visPLred.parameters import *
# import h5py
# from tqdm import tqdm

# firstcam_timestamp_path = '/mnt/datazpool/PL/20250514/firstpl/'
# firstcam_spec_path =      '/mnt/datazpool/PL/20250514/firstpl/'

# obs_start = '08:31:00'
# obs_end   = '08:36:59' #'08:36:59'

# psfcam = 'palila'
# psfcam_frames_name = '/mnt/datazpool/PL/yjkim/reduced/mizarA_20250514/first_palila_matched_fastcam_matched_frames.npy'
# psfcam_timestamp_name = '/mnt/datazpool/PL/yjkim/reduced/mizarA_20250514/first_palila_matched.pkl'




# def bin_by_centroids_to_file(outname, psfcamframes, firstcam_file_indices, firstcam_frame_indices, firstcam_files,
#                                   centroids, xbins, ybins,
#                                 #   maxframes = 1000,
#                                 #   bootstrap = False,
#                                   skip_frame_reading = False):
#     '''
#     Bin frames by centroids, but not from already stored frames.
#     Reads the frames from the firstcam_file_indices and firstcam_frame_indices.
#     This is useful when the firstcam frames are too large to store in memory.
#     '''
    
#     x = centroids[:,0]
#     y = centroids[:,1]
#     ny = firstcam_params['size_y']
#     nx = firstcam_params['size_x']
    
#     psfcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, psfcamframes.shape[1], psfcamframes.shape[2]))
#     firstcam_binned_frames = np.zeros((len(xbins)-1, len(ybins)-1, ny, nx))

#     num_frames = np.zeros((len(xbins)-1, len(ybins)-1))
#     idxs = np.zeros((len(xbins)-1, len(ybins)-1, len(firstcam_file_indices)), dtype=bool)



#     for i in range(len(xbins)-1):
#         for j in range(len(ybins)-1):

#             # assert os.path.exists(outname+'_bin_%d_%d.fits' % (i,j)) is False, "File %s already exists. Try another name for outname." % (outname+'_bin_%d_%d.fits' % (i,j))
            
#             # create a new fits file for this bin
#             # hdu = fits.PrimaryHDU(np.zeros((0, firstcam_params['size_y'], firstcam_params['size_x']), dtype=int))
#             # hdu.writeto(outname+'_bin_%d_%d.fits' % (i,j), overwrite=True)

#             # check if the x and y coordinates are within the bin
#             xidx = (x >= xbins[i]) & (x < xbins[i+1])
#             yidx = (y >= ybins[j]) & (y < ybins[j+1])
#             idx = xidx & yidx

#             # store the indices of the frames that match this bin
#             idxs[i,j] = idx
#             psfcam_binned_frames[i,j] = np.mean(psfcamframes[idx], axis=0)
#             num_frames[i,j] = np.sum(idx)

#             if num_frames[i,j] > 0:
#                 print(np.shape(psfcamframes[idx]))

#                 with h5py.File(outname+'_bin_%d_%d.h5' % (i,j), 'w') as h5f:
#                     print("creating file %s" % (outname+'_bin_%d_%d.h5' % (i,j)))
                    
#                     rawframes_dset = h5f.create_dataset('rawframes', 
#                                             shape = (num_frames[i,j], firstcam_params['size_y'], firstcam_params['size_x']), 
#                                             # chunks = (1, firstcam_params['size_y'], firstcam_params['size_x']),
#                                             # maxshape=(None, firstcam_params['size_y'], firstcam_params['size_x']), 
#                                             dtype='int')
#                     psfframes_dset = h5f.create_dataset('psfframes',
#                                                         data = psfcamframes[idx])
#                     peaks_dset = h5f.create_dataset('peaks',
#                                                     data = np.nanmax(psfcamframes[idx], axis=(1,2)))
#                     centers_dset = h5f.create_dataset('centers',
#                                                     data = np.array([x[idx], y[idx]]).T)
                    
#                     fileinds_dset = h5f.create_dataset('fileinds',
#                                                         data = firstcam_file_indices[idx])
#                     frameidnds_dset = h5f.create_dataset('frameinds',
#                                                         data = firstcam_frame_indices[idx])

#                     h5f.attrs['num_frames'] = num_frames[i,j].astype(int)
#                     h5f.attrs['xbin'] = xbins[i]
#                     h5f.attrs['ybin'] = ybins[j]
                    
                


#     write_idx = {(i, j):0 for i in range(len(xbins)-1) for j in range(len(ybins)-1)}

#     if not skip_frame_reading:
#         # now read the frames from the firstcam_file_indices and firstcam_frame_indices
#         for fileind, f in tqdm(enumerate(firstcam_files)):
#             print('Start reading file %d' % fileind)
            
#             # indices that correspond to this file
#             id = (firstcam_file_indices == fileind)
#             # print('ID', id, len(id))
#             # print(np.where(id == True)[0])

#             # frame indices
#             id_frame = firstcam_frame_indices[id] # get the frame indices for this file
#             # print('ID_FRAME', id_frame)
#             # id_frame = np.where(id_frame)[0]

#             _dat = fits.getdata(f)

#             # CHANGED HERE!!
#             for i0, i in enumerate(id_frame): #range(len(id_frame)):

                
#                 iy, ix = np.where(idxs[:,:,np.where(id == True)[0][i0]] == True) # get the indices of the bin that matches this frame])
#                 # iy, ix = np.where(idxs[:,:,np.where(id == True)[0][i]] == True) # get the indices of the bin that matches this frame])
#                 # nstack = 1

#                 try:
#                     # np.where(id == True)[0][i] # get the indices of the frames that match this file

#                     # idxs[]
#                     iy = iy[0]
#                     ix = ix[0]

#                     # append the frame to the fits file
#                     # print(_dat[i].shape)

#                     # append_frame_to_fits(outname+'_bin_%d_%d.fits' % (iy,ix), _dat[i])
#                     # ds = datasets['bin_%d_%d' % (iy,ix)]
#                     # ds.resize((ds.shape[0] + 1, firstcam_params['size_y'], firstcam_params['size_x']))
#                     # ds[-1] = _dat[i]

#                     # append the frame to the h5 file
#                     with h5py.File(outname+'_bin_%d_%d.h5' % (iy,ix), 'r+') as h5f:
#                         dset = h5f['rawframes']
#                         wi = write_idx[(iy, ix)]
#                         dset[wi] = _dat[i]
#                         write_idx[(iy, ix)] += 1

#                     # firstcam_binned_frames[iy,ix,:,:] += _dat[i] * nstack
#                     print(f"Added frame {i} from file {fileind} to bin {iy},{ix}")
#                 except:
#                     # print(e)
#                     print(f"Failed to add frame {i} from file {fileind} to bin {iy},{ix}")
#                     continue
#         # # close the h5 file
#         # h5f.close()

#         # # dump to fits file
#         # with h5py.File(outname+'_bins.h5', 'r') as h5f:
#         #     for i in range(len(xbins)-1):
#         #         for j in range(len(ybins)-1):
#         #             ds = h5f['bin_%d_%d' % (i,j)]
#         #             # create a new fits file for this bin
#         #             hdu = fits.PrimaryHDU(ds[:])
#         #             hdu.writeto(outname+'_bin_%d_%d.fits' % (i,j), overwrite=True)

#         # now calculate the mean
#         # firstcam_binned_frames = firstcam_binned_frames / num_frames[:,:,None,None]
#         return psfcam_binned_frames, num_frames, idxs
    


# sd = cm.SimultaneousData(firstcam_timestamp_path, firstcam_spec_path,
#                 obs_start, obs_end,
#                 psfcam, 
#                 psfcam_frames_name, psfcam_timestamp_name,
#                 store_spec = False,
#                 matched_frames = True,)

# # bin!!

# sd.psfcam_frames = arr2 = sd.psfcam_frames.reshape(-1, 2, *sd.psfcam_frames.shape[1:]).sum(axis=1)
# sd.firstcam_file_indices = sd.firstcam_file_indices.reshape(-1, 2)[:,0] #.sum(axis=1)
# sd.firstcam_frame_indices1 = sd.firstcam_frame_indices.reshape(-1, 2)[:,0] #.sum(axis=1)
# sd.firstcam_frame_indices2 = sd.firstcam_frame_indices.reshape(-1, 2)[:,1] #.sum(axis=1)

# sd.compute_psfcam_centroids(peak=True)

# map_n = 15
# map_width = 3


# centroids = sd.centroids #[effective_idx]
# psfcam_frames = sd.psfcam_frames #[effective_idx]
# firstcam_file_indices = sd.firstcam_file_indices #[effective_idx]
# firstcam_frame_indices1 = sd.firstcam_frame_indices1 #[effective_idx]
# firstcam_frame_indices2 = sd.firstcam_frame_indices2 #[effective_idx]

# sd.map_n = map_n
# sd.map_width = map_width

# xbins = np.linspace(np.nanmedian(centroids[:,0]) - map_width/2, np.nanmedian(centroids[:,0]) + map_width/2, map_n+1)
# ybins = np.linspace(np.nanmedian(centroids[:,1]) - map_width/2, np.nanmedian(centroids[:,1]) + map_width/2, map_n+1)

# sd.x_mas = ((xbins[:-1] + np.diff(xbins)[0]/2) - np.nanmedian(centroids[:,0])) * sd.pix2mas
# sd.y_mas = ((ybins[:-1] + np.diff(ybins)[0]/2) - np.nanmedian(centroids[:,1])) * sd.pix2mas

# sd.xbins = xbins
# sd.ybins = ybins

# self.psfcam_binned_frames, self.num_frames, self.idxs = bin_by_centroids_to_file(filename, psfcam_frames, firstcam_file_indices, firstcam_frame_indices, self.firstcam_files, centroids, xbins, ybins, skip_frame_reading=skip_frame_reading)


# # from PLred.visPLred.utils import find_data_between
# # from PLred.visPLred.coupligmap import validate_timestamp_matching

# # firstcam_timestampfiles = find_data_between(sd.firstcam_timestamp_path, sd.obs_start, sd.obs_end, header='firstpl_', footer='.txt')
# # firstcam_specfiles = find_data_between(sd.firstcam_spec_path, sd.obs_start, sd.obs_end, header='firstpl_', footer='.fits')

# # # load timestamps
# # timestamps_matching_spec = np.concatenate([np.genfromtxt(file)[:,4] for file in firstcam_timestampfiles])

# # # load psfcam frames and timestamp matching pkl file
# # psfcam_frames = np.load(sd.psfcam_frames_name)
# # with open(sd.psfcam_timestamp_name, 'rb') as f:
# #     psfcam_timestamp = pickle.load(f)

# # all_file_indices = []
# # all_frame_indices = []
# # for fi, f in enumerate(firstcam_specfiles):
# #     for n in range(fits.getheader(f)['NAXIS3']): #fits.getdata(f).shape[0]):
# #         all_file_indices.append(fi)
# #         all_frame_indices.append(n)
# # all_file_indices = np.array(all_file_indices)
# # all_frame_indices = np.array(all_frame_indices)

# # # validate timestamps
# # idx1, idx2 = validate_timestamp_matching(timestamps_matching_spec, (np.array(psfcam_timestamp['timestamps'])))
# # all_file_indices = all_file_indices[idx1]
# # all_frame_indices = all_frame_indices[idx1]
# # sd.firstcam_file_indices = all_file_indices
# # sd.firstcam_frame_indices = all_frame_indices
# # sd.firstcam_files = firstcam_specfiles





# sd.compute_psfcam_centroids(peak=True)
# sd.bin_by_centroids(15, 3, calculate_variance = False,
#                     to_file = True,
#                     filename = f'/mnt/datazpool/PL/yjkim/remapped/mizarA_20250514/remapped_{obs_start}_{obs_end}')#, return_bootstrap_samples= True)

