import sys
import os
import shutil

from PLred.visPLred import utils as du

from configobj import ConfigObj

if __name__ == "__main__":

    configname = sys.argv[1]
    config = ConfigObj(configname)

    path_input = config['Inputs']['path']
    obs_date = config['Inputs']['obs_date']

    target_start_time = config['Inputs']['target_start_time']
    target_end_time = config['Inputs']['target_end_time']

    destination = config['Outputs']['timestamp_dir']

    # Find data
    target_files = du.find_data_between(path_input + obs_date + '/firstpl/', target_start_time, target_end_time, header='firstpl_', footer='.txt')


    # Write to file
    print("generating temporary directory")
    os.makedirs('firstcam_timestamps')
    for target_file in target_files:
        shutil.copy(target_file, 'firstcam_timestamps/')
    
    # send the directory over to scexao6
    os.system('scp -r firstcam_timestamps '+destination)
    print("done!")

    os.system('rm -r firstcam_timestamps')
    print("removed the directory")


