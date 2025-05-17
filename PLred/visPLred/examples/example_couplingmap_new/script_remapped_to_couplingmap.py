import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import h5py
import os, glob


# data directory
data_dir = '/mnt/datazpool/PL/yjkim/remapped/mizarA_20250514/betcmi_20250211/'

# map_n
n = 15

def rawframe_to_spec(rawframe):

    
