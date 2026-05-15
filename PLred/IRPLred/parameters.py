# fixed parameters used for data reduction

telescope_params = {
    'diameter' : 8.2, # telescope diameter in m
}

CRED2_params ={
    'orientation'   : 'vertical'

}

CRED1_params = {
    'orientation'   : 'horizontal'
}


palila_params = {
    'plate_scale'   : 16.2, # mas / pixel
    'pa_offset'     : 2.7, # deg, offset for D_IMRPAD
}

# IR Spectrometer Detector Parameters (Mode 2 pipeline)
# These are default values used by ingest_mode2_to_hdf5() when not overridden in config
NFIB = 38  # Number of fibers from IR spectrometer
DETECTOR_SIZE = (384, 512)  # (height, width) of IR detector in pixels
IR_PLATE_SCALE = 20.0  # Plate scale of PSF camera in mas/pixel
BOX_EXTRACTION_WIDTH = 6  # Default half-width for box extraction in pixels

