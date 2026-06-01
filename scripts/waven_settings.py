import numpy as np
import os
import utils

# spike_times_dir = utils.derivitive_dir+'spike_times/'
# files_in_spike_times_dir = os.listdir(spike_times_dir)
# print(files_in_spike_times_dir)

"""
Parameters Gabor Library:
    N_thetas (int): number of orientation equally spaced between 0 and 180 degree.
    Sigmas (list): standart deviation of theb gabor filters expressed in pixels (radius of the gaussian half peak wigth).
    Frequencies (list): spatial frequencies expressed in pixels per cycles.
    Phases (list): 0 and pi/2.
    NX (int): number of azimuth positions (pix) (x shape of the downsampled stimuli).
    NY (int): number of elevation positions (pix) (y shape of the downsampled stimuli).
    Save Path (string): where to save the gabor library

Parameters alignement:
    Dirs (string): where the raw data are.
    Experiment Info: (mouse name, data, experiment number)
    Number of Planes (int): number of acquisition planes.
    Block End (int): timeframe where the experiment starts.
    Number of Frames (int): number of frames stim 30 Hz -> 1800 frame/min.
    Number of Trials to Keep(int): Number of Trials to Keep.

Parameters analysis:
    screen_x: stimulus screen x size inn pixels.
    screen_y: stimulus screen y size inn pixels.
    NX (int): number of azimuth positions (pix) (x shape of the downsampled stimuli).
    NY (int): number of elevation positions (pix) (y shape of the downsampled stimuli).
    Resolution (float): microscope resolution (um per pixels)
    Sigmas (list): standart deviation of theb gabor filters expressed in pixels (radius of the gaussian half peak wigth).
    Visual Coverage (list): [azimuth left, azimuth right, elevation top , elevation bottom] in visual degree.
    Analysis Coverage (list): [azimuth left, azimuth right, elevation top , elevation bottom] in visual degree.
    Movie Path: path to the stimulus (.mp4)
    Library Path: path to Gabor library (same as save path if ran)
    Spks Path (opt): path to the spks.npy file to skip the alignement procedure, if set ignores Parameter alignment
"""

# List of default parameters for the Gabor Library
gabor_param={
    "N_thetas":"10",
    "Sigmas": "[2, 3, 4, 5, 6, 7, 8]",
    # "Frequencies": "[0.015, 0.04, 0.07, 0.1]",
    "Frequencies":"[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 1.1, 1.2, 1.3]", #"0.036" #0.15
    "Phases": "[0, 90]",
    # "NX": "135",
    # "NY": "54",
    # "NX": "107", # TODO
    # "NY": "85",  # TODO
    "NX" : 67,
    "NY" : 53,
    "Save Path":"../../waven/gabors_library.npy"
}

# List of default parameters
param_defaults = {
    "Path Directory": "/user/raabe14/u19361/workspace-allen/rawdata/stimulus/zebra/",
    "Dirs": "../../rawdata/allen_open_scope/sub-820454/",
    "Experiment Info": "('820454', '2025-11-05', 3)",
    "Number of Planes": "1",
    "Block End": "0",
    "screen_x":"1072",
    "screen_y":"848",
    # "NX": "135", # TODO
    # "NY": "54",  # TODO
    # "NX": "120", # TODO
    # "NY": "95",  # TODO
    "Resolution":"1.3671",
    "Sigmas": "[2, 3, 4, 5, 6, 8]",
    # "Frequencies": "[0.015, 0.04, 0.07, 0.1]",
    # "Frequencies": "0.08",
    # "Visual Coverage":"[-135, 45, 34, -34]", # TODO
    "Visual Coverage":"[-60, 60, 47.5, -47.5]", # TODO
    # "Analysis Coverage":"[-   5, 45, 34, -34]", # TODO
    "Analysis Coverage": "[-60, 60, 47.5, -47.5]",
    "Number of Frames": "9000",
    "Number of Trials to Keep": "1",
    "Movie Path": "/user/raabe14/u19361/workspace-allen/rawdata/stimulus/zebra/zebra_allen_screen_tscale_30_scale_10.mp4",
    "Library Path": "../../waven/filter_libraries/",
    # "Spks Path": utils.derivitive_dir+'spike_times/'+files_in_spike_times_dir[0],
}



sigmas = eval(gabor_param["Sigmas"])
nx = int(gabor_param["NX"])
ny = int(gabor_param["NY"])
n_thetas = int(gabor_param["N_thetas"])
offsets= eval(gabor_param["Phases"])
path_save = gabor_param["Save Path"]
xs = np.arange(nx)
ys = np.arange(ny)
thetas = np.array([(i * np.pi) / n_thetas for i in range(n_thetas)])
sigmas = np.array(sigmas)
offsets=np.array(offsets)
frequencies =  eval(gabor_param["Frequencies"])

path_directory = param_defaults["Path Directory"]
dirs = [param_defaults["Dirs"]]
exp_info = eval(param_defaults["Experiment Info"])
sigmas = eval(param_defaults["Sigmas"])
sigmas=np.array(sigmas)
visual_coverage = eval(param_defaults["Visual Coverage"])
analysis_coverage = eval(param_defaults["Analysis Coverage"])
n_planes = int(param_defaults["Number of Planes"])
block_end = int(param_defaults["Block End"])
screen_x = int(param_defaults["screen_x"])
screen_y = int(param_defaults["screen_y"])
ns = len(sigmas)
resolution=float(param_defaults["Resolution"])
# spks_path = param_defaults["Spks Path"]
nb_frames = int(param_defaults["Number of Frames"])
n_trial2keep = int(param_defaults["Number of Trials to Keep"])
movpath = param_defaults["Movie Path"]
library_path = param_defaults["Library Path"]
screen_ratio = abs(visual_coverage[0]-visual_coverage[1])/nx
xM, xm, yM, ym = analysis_coverage

pathdata = os.path.join(os.path.join(os.path.join(dirs[0] , exp_info[0]) , exp_info[1]) , str(exp_info[2]))
pathsuite2p = os.path.join(pathdata , 'suite2p')

deg_per_pix=abs(xM-xm)/nx
sigmas_deg=np.trunc(2*deg_per_pix*sigmas*100)/100


## define visual coverage for the analysis
if (visual_coverage!=analysis_coverage):
    visual_coverage=np.array(visual_coverage)
    analysis_coverage=np.array(analysis_coverage)
    ratio_x=1-((visual_coverage[0]-visual_coverage[1])-(analysis_coverage[0]-analysis_coverage[1]))/(visual_coverage[0]-visual_coverage[1])
    ratio_y=1-((visual_coverage[2]-visual_coverage[3])-(analysis_coverage[2]-analysis_coverage[3]))/(visual_coverage[2]-visual_coverage[3])
else:
    ratio_x=1
    ratio_y=1
