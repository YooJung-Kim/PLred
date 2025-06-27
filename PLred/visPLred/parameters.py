# fixed parameters used for data reduction

telescope_params = {
    'diameter' : 8.2, # telescope diameter in m
}

firstcam_params = {
    'NFIB'          : 38,
    'zaber_microns' : 0.047, # 1 zaber step = 0.047 microns
    'polar_1'       : [0,1,2,4, 6, 8,10,12,14,16,18,20,22,24,26,28,30,32,34],
    'polar_2'       : [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,36,37],
    'resolving_power' : 3000, # spectrograph resolving power for sim.py
    'gain'          : 0.1, # e- / ADU
    'readnoise'     : 0.45, # e-
    'size_y'        : 412, # size of the detector in pixels
    'size_x'        : 1896, # size of the detector in pixels
}

palila_params = {
    'plate_scale'   : 16.2, # mas / pixel
    'pa_offset'     : 2.7, # deg, offset for D_IMRPAD
}
