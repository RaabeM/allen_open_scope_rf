import torch

def dummy_cuda(self, device=None, non_blocking=False):
    return self  # just return the same tensor (on CPU)

torch.Tensor.cuda = dummy_cuda
torch.nn.Module.cuda = dummy_cuda

from Waven import WaveletGenerator as wg
from Waven import Analysis_Utils as au
from Waven import LoadPinkNoise as lpn
from Waven import zebraGUI as ui
import numpy as np
import gc
import os

from pathlib import Path
# from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt
from dandi import dandiapi, download
import os   
import pandas as pd
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from pynwb import NWBHDF5IO

from tqdm import tqdm

import pickle
import utils


from numba import njit, typed
from waven_settings import *
import h5py


def create_gabor_library():
    # create a new gabor library
    if not os.path.exists(path_save):
        freq=True
        L = wg.makeFilterLibrary(xs, ys, thetas, sigmas, offsets, f, freq=freq)
        np.save(path_save, L)
        lib_path=path_save


def downsample_video():
    # downsample the video according to the analysis coverage
    downsampled_path = movpath[:-4]+'_downsampled.npy'
    if not os.path.exists(downsampled_path):
        wg.downsample_video_binary(os.path.abspath(movpath),
                                np.array(visual_coverage),  
                                np.array(analysis_coverage), 
                                shape=(ny, nx), 
                                chunk_size=500, 
                                ratios=(ratio_x, ratio_y)
                                )

def wavelet_decomposition():
    # Video decomposition
    waveletDecomposition_files = os.listdir(os.path.dirname(path_directory))
    waveletDecomposition_files = [f for f in waveletDecomposition_files if f[:3] == 'dwt']

    folder_path = '../../rawdata/allen_open_scope/stimulus/zebra'

        # if len(waveletDecomposition_files) < 2:
    if True:
        videodata=np.load(movpath[:-4]+'_downsampled.npy')
        wg.waveletDecomposition(videodata=videodata, phase=0, sigmas=sigmas, folder_path=folder_path, library_path=lib_path)
        wg.waveletDecomposition(videodata=videodata, phase=1, sigmas=sigmas, folder_path=folder_path, library_path=lib_path)



# convert object-array of spike lists -> numba typed list of float64 arrays
def make_typed_spike_list(spike_times_obj_array):
    tl = typed.List()
    for arr in spike_times_obj_array:
        tl.append(np.asarray(arr, dtype=np.float64))
    return tl

@njit
def compute_neuron_rate_to_zebra_frames(frame_onset_times, frame_offset_times, spike_times_list, delay=0.0):
    

    num_units = len(spike_times_list)
    num_trials = len(frame_onset_times)
    out = np.zeros((num_units, num_trials), dtype=np.float64)

    for u in range(num_units):
        unit_spike_times = spike_times_list[u]
        for i in range(num_trials):
            start_time = frame_onset_times[i] + delay
            stop_time  = frame_offset_times[i] + delay
            rate = np.sum((unit_spike_times >= start_time) & (unit_spike_times < stop_time)) / (stop_time - start_time)
            out[u, i] = rate

    return out


def save_results(rfs_zebra, unit_names_list, delay, filename, results_dir='/home/marcelraabe/projects/SPP2205_local/results/allen_open_scope/rf/waven/zebra/'):

    
    # Save results
    
    outpath = os.path.join(results_dir, filename)

    with h5py.File(outpath, 'a') as hf:
        # create or replace a group for this unit
        for idx, unit_name in enumerate(unit_names_list):
            
            grp_name = unit_name
            if grp_name in hf:
                del hf[grp_name]
            grp = hf.create_group(grp_name)

            opt_sigma_idx = np.array(rfs_zebra[1])[3, idx]
            grp.create_dataset('opt_sigma', data=sigmas[opt_sigma_idx])

            theta_idx = 0
            cc_f_1_xy=rfs_zebra[0][idx, :, :, theta_idx, opt_sigma_idx]
            grp.create_dataset('response_theta_0', data=cc_f_1_xy)

            theta_idx = 1
            cc_f_1_xy=rfs_zebra[0][idx, :, :, theta_idx, opt_sigma_idx]
            grp.create_dataset('response_theta_45', data=cc_f_1_xy)

            theta_idx = 2
            cc_f_1_xy=rfs_zebra[0][idx, :, :, theta_idx, opt_sigma_idx]
            grp.create_dataset('response_theta_90', data=cc_f_1_xy)




def full_pipeline(frame_onset_times, frame_offset_times, spike_times_list, delay, unit_names_list, results_dir='/home/marcelraabe/projects/SPP2205_local/results/allen_open_scope/rf/waven/zebra/'):

    print(f'Running Waven Pipeline for delay: {delay} seconds')

    print('Creating Gabor Library...')
    create_gabor_library()

    print('Downsampling Video...')
    downsample_video()

    print('Wavelet Decomposition...')
    wavelet_decomposition()

    print('Computing Neuron Rates to Zebra Frames...')
    spike_times_list = make_typed_spike_list(spike_times_list)
    neuron_rate_to_zebra_frames = compute_neuron_rate_to_zebra_frames(frame_onset_times, frame_offset_times, spike_times_list, delay)

    ## the spikes data have to be time registered to the stimulus frames
    ## MR Only one trial thus no repeatability
    # respcorr_zebra = au.repetability_trial3(spks, neuron_pos, plotting=True)
    wavelets0, wavelets1, wavelet_c = lpn.coarseWavelet(path=path_directory,
                                                        downsampling=False, 
                                                        nx0=nx, 
                                                        ny0=ny, 
                                                        nx=13, 
                                                        ny=11, 
                                                        no=n_theta, 
                                                        ns=ns,
                                                        chunk_size=200,
                                                        )

    print('Running Correlation Analysis...')
    ## runs correlation analysis
    rfs_zebra = au.PearsonCorrelationPinkNoise(stim=wavelet_c.reshape(9000, -1), 
                                            resp=neuron_rate_to_zebra_frames.reshape(9000, -1),
                                            neuron_pos=np.zeros((1,2)),  # dummy value for neuron_pos, 
                                            nx=13, 
                                            ny=11, 
                                            ns=ns, 
                                            visual_coverage=analysis_coverage, 
                                            screen_ratio=screen_ratio, 
                                            sigmas=sigmas_deg,
                                            plotting=False,
                                            n_thetas=n_theta
                                            )


    print('Saving Results...')
    filename =  f'waven-zebra-rfs__freq-{f}__delay-{delay}.h5' 
    save_results(rfs_zebra, unit_names_list, delay, filename, results_dir)