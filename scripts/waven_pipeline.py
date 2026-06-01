import torch

def dummy_cuda(self, device=None, non_blocking=False):
    return self  # just return the same tensor (on CPU)

# If no no GPU is available prevent cuda calls 
if not torch.cuda.is_available():
    print("CUDA is not available. Running on CPU.")
    torch.Tensor.cuda = dummy_cuda
    torch.nn.Module.cuda = dummy_cuda
else:
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")

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
import json
import hashlib
import numpy as np
from typing import Dict, Any, Tuple
import time



def create_gabor_library(xs, ys, thetas, sigmas, offsets, frequencies,
                                 library_dir: str = library_path,
                                 registry_name: str = "filter_registry.json",
                                 overwrite: bool = False) -> Path:
    """
    Creates a filter library or loads it, if it already exists with the same parameters. 
    Existing libraries and their parameteres are tracked in a JSON registry file. 

    Parameters
    ----------
    library_dir :
        directory to save libraries and registry.
    registry_name :
        filename of the json registry (inside library_dir).
    overwrite :
        if True and a matching file exists, recreate and overwrite it.

    Returns
    -------
    library_path : 
        Path to library
    """
    lib_dir = Path(library_dir)
    lib_dir.mkdir(parents=True, exist_ok=True)
    registry_path = lib_dir / registry_name

    props = {
        "xs": list(map(int, xs)),  # Ensure all elements are Python int
        "ys": list(map(int, ys)),  # Ensure all elements are Python int
        "thetas": list(map(float, thetas)),
        "sigmas": list(map(float, sigmas)),
        "offsets": list(map(float, offsets)),
        "frequencies": list(map(float, frequencies)),
    }

    serialized = json.dumps(props, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]
    filename = f"gabor_filter_library_{digest}.npy"
    file_path = lib_dir / filename

    # load or init registry (list of entries)
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            try:
                registry = json.load(f)
            except Exception:
                registry = []
    else:
        registry = []

    # search for exact serialized match
    for entry in registry:
        if entry.get("serialized") == serialized:
            entry_path = Path(entry.get("filename"))
            # if stored path is relative to other dir, resolve relative to registry location
            if not entry_path.is_absolute():
                entry_path = (registry_path.parent / entry_path).resolve()
            if entry_path.exists() and not overwrite:
                print(f"Library already exists at {entry_path}")
                return entry_path
            # else: file missing or overwrite requested -> break and recreate below
            break

    # create library with user-provided function
    lib = wg.makeFilterLibrary2(xs, ys, thetas, sigmas, offsets, frequencies)
    if not isinstance(lib, np.ndarray):
        lib = np.asarray(lib)

    # save compressed
    # np.savez_compressed(file_path, library=lib)
    np.save(file_path, lib)

    # add/update registry entry
    new_entry = {
        "serialized": serialized,
        "filename": str(file_path.name),  # store relative filename for portability
        "hash": digest,
        "props": props,
    }
    # remove previous entries with same serialized if present
    registry = [e for e in registry if e.get("serialized") != serialized]
    registry.append(new_entry)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    return file_path


# def create_gabor_library():
#     # create a new gabor library
#     path_library = os.path.join(lib_path, '/filter_library')
#     os.makedirs(path_library, exist_ok=True)
#     path_save = os.path.join(, filename)
#     if not os.path.exists(path_save):
#         freq=True
#         L = wg.makeFilterLibrary2(xs, ys, thetas, sigmas, offsets, frequencies)
#         np.save(path_save, L)
#         lib_path=path_save


def downsample_video():
    # downsample the video according to the analysis coverage
    downsampled_path = movpath[:-4]+f'_downsampled__{ny}_{nx}.npy'
    if not os.path.exists(downsampled_path):
        wg.downsample_video_binary(os.path.abspath(movpath),    
                                np.array(visual_coverage),  
                                np.array(analysis_coverage), 
                                shape=(ny, nx), 
                                chunk_size=500, 
                                ratios=(ratio_x, ratio_y)
                                )


def get_path_wavelet_decomposition(library_path=library_path, movpath=movpath):
    # Video decomposition
    lib_name = os.path.basename(library_path)
    lib_id = lib_name.split('_')[-1].split('.')[0]

    movpath = Path(movpath)
    print(movpath)
    path_decompositions = movpath.parent / 'wavelet_decompositions' / f'lib-{lib_id}'
    Path(path_decompositions).mkdir(exist_ok=True)

    for phase in [0, 1]:
        filename = f'dwt_videodata_{phase}.npy'
        file_path = os.path.join(path_decompositions, filename)
        
        if os.path.exists(file_path):
            print(f"Wavelet decomposition for lib {lib_id} and phase {phase} already exists at {file_path}. Skipping decomposition.")
        else:
            video_path = f'{movpath.parent/movpath.stem}_downsampled__{ny}_{nx}.npy'
            if not os.path.exists(video_path):
                print(f"Downsampled video not found at {video_path}. Running downsampling...")
                downsample_video()
            videodata=np.load(video_path)

            wg.waveletDecomposition(videodata=videodata, phase=phase, sigmas=sigmas, folder_path=path_decompositions, library_path=library_path)
    
    return path_decompositions


# convert object-array of spike lists -> numba typed list of float64 arrays
def make_typed_spike_list(spike_times_obj_array):
    tl = typed.List()
    for arr in spike_times_obj_array:
        tl.append(np.asarray(arr, dtype=np.float64))
    return tl


def compute_neuron_rate_to_zebra_frames(frame_onset_times, spike_times_list, delay=0.0, duration=1/30):
    """
    Compute the firing rate of each neuron in spike_times_list for each frame defined by frame_onset_times, given a delay and duration.
    Assumes that spike times are sorted. 
    """
    starts = frame_onset_times + delay      # vectorized, shape (num_trials,)
    stops  = starts + duration
    num_units = len(spike_times_list)
    out = np.zeros((num_units, len(frame_onset_times)), dtype=np.float64)

    for u in range(num_units):
        spk = spike_times_list[u]
        out[u] = (np.searchsorted(spk, stops) - np.searchsorted(spk, starts)) / duration
    return out


# @njit
# def compute_neuron_rate_to_zebra_frames(frame_onset_times, spike_times_list, delay=0.0, duration=1/30):
#     num_units = len(spike_times_list)
#     num_trials = len(frame_onset_times)
#     out = np.zeros((num_units, num_trials), dtype=np.float64)
#     # out = np.zeros((num_trials, num_units), dtype=np.float64)


#     for u in range(num_units):
#         unit_spike_times = spike_times_list[u]
#         for i in range(num_trials):
#             start_time = frame_onset_times[i] + delay
#             # stop_time  = frame_offset_times[i] + delay
#             stop_time  = start_time + duration
#             rate = np.sum((unit_spike_times >= start_time) & (unit_spike_times < stop_time)) / (stop_time - start_time)
#             out[u, i] = rate

#     return out


def save_results(rfs,
                     unit_names_list,
                    #  delay,
                    #  duration,
                    #  sigmas,
                    #  thetas,
                    #  frequencies,
                     results_path='./',
                     results_filename='',
                     attributes: Dict[str, Any] = {}
                     ):
    import tempfile

    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    if not results_filename.endswith('.h5'):
        results_filename += '.h5'

    outpath = results_path / results_filename

    # Write to a temp file first; rename atomically on success so that any
    # interruption (OOM kill, wall-time exceeded) never leaves a corrupt file.
    tmp_fd, tmp_name = tempfile.mkstemp(suffix='.h5', dir=results_path)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        with h5py.File(tmp_path, 'w') as hf:
            for k, v in attributes.items():
                hf.attrs[k] = v

            hf.create_dataset('unit_ids', data=list(unit_names_list))
            hf.create_dataset('correlation_matrix', data=rfs[0], compression='gzip', compression_opts=4)
            hf.create_dataset('best_gabor_params_idx', data=rfs[1])
            hf.create_dataset('best_gabor_params_degree', data=rfs[2])
            hf.create_dataset('abs_max_value', data=rfs[3])

        if outpath.exists():
            outpath.unlink()
        tmp_path.rename(outpath)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return outpath


def subsample_wavelet_responses(wavelets_complex, xis, yis):

    wavelets_complex_subsampled = np.zeros((wavelets_complex.shape[0], len(xis), len(yis), wavelets_complex.shape[3], wavelets_complex.shape[4], wavelets_complex.shape[5]))
    for i, x_i in enumerate(tqdm(xis)):
        for j, y_i in enumerate(yis):
            wavelets_complex_subsampled[:, i, j, :, :, :] = wavelets_complex[:, int(x_i), int(y_i)]
    
    return wavelets_complex_subsampled


def load_phase_dependent_wavelet_decompositions(path_decompositions, phase, xis=[], yis=[]):

    file_path = os.path.join(path_decompositions, f'dwt_videodata_{phase}.npy')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Wavelet decomposition files not found at {file_path}. Please run wavelet_decomposition() first.")
    
    wavelet = np.load(file_path)

    if len(xis) > 0 and len(yis) > 0:
        wavelet = subsample_wavelet_responses(wavelet, xis, yis)

    # stack phase 0 and 1 into a new phase dimension: result shape will be
    # (n_frames, n_phases=2, Nx, Ny, Ntheta, Nsigma, Nfreq, Noffset, ...)
    # combined = np.stack((wavelet_0, wavelet_1), axis=-1)

    return wavelet


def load_complex_wavelet_decomposition(path_decompositions, xis=[], yis=[]):

    pc = os.path.join(path_decompositions, 'dwt_videodata_c.npy')
    if not os.path.exists(pc):
        # Compute wavelet_c in chunks and avoid loading full arrays into RAM.
        p0 = os.path.join(path_decompositions, 'dwt_videodata_0.npy')
        p1 = os.path.join(path_decompositions, 'dwt_videodata_1.npy')

        # open memmaps to read shapes (these do not load the whole file)
        wavelet_0_mm = np.load(p0, mmap_mode='r')
        wavelet_1_mm = np.load(p1, mmap_mode='r')

        n_frames = wavelet_0_mm.shape[0]
        frame_shape = wavelet_0_mm.shape[1:]  # (Nx, Ny, Ntheta, Nsigma, Nfreq, Noffset) or similar
        dtype = np.float32

        # prepare an on-disk memmap for the computed magnitude (wavelet_c)
        c_path = os.path.join(path_decompositions, 'dwt_videodata_c.npy')
        # remove existing file if present to ensure open_memmap can create cleanly
        if os.path.exists(c_path):
            os.remove(c_path)
        wavelet_c_mm = np.lib.format.open_memmap(c_path, mode='w+', dtype=dtype, shape=(n_frames, *frame_shape))

        chunk_frames = 512  # adjust if you want larger/smaller chunks

        for start in tqdm(range(0, n_frames, chunk_frames), desc="Computing wavelet_c chunks"):
            end = min(start + chunk_frames, n_frames)
            # read just the slice from the memmaps (this loads only the chunk into RAM)
            a0 = wavelet_0_mm[start:end].astype(dtype)
            a1 = wavelet_1_mm[start:end].astype(dtype)

            # magnitude / power: a0**2 + a1**2
            c_chunk = a0 * a0 + a1 * a1

            # write chunk directly to the output memmap
            wavelet_c_mm[start:end] = c_chunk

            # free local refs for this iteration
            del a0, a1, c_chunk
            torch.cuda.empty_cache()

        # close/delete temporary memmap references to free resources
        del wavelet_0_mm, wavelet_1_mm, wavelet_c_mm

    print(f"Loading precomputed wavelet_c from {pc}")
    wavelet_c = np.load(pc)

    if len(xis) > 0 and len(yis) > 0:
        wavelet_c = subsample_wavelet_responses(wavelet_c, xis, yis)

    # add a new axis for phase to match expected shape (n_frames, Nx, Ny, Ntheta, Nsigma, Nfreq, Noffset, n_phases) in au.PearsonCorrelationPinkNoise()
    # wavelet_c = wavelet_c[..., np.newaxis]  

    return wavelet_c


def full_pipeline(frame_onset_times,
                #   frame_offset_times,
                  spike_times_list,
                  delays,
                  durations,
                  unit_names_list,
                  xis=[],
                  yis=[],
                  results_path='./',
                  results_filename='',
                  attributes={}, 
                  recompute=False, 
                  phase='complex',
                  ):

    # ensure delays and durations are numpy 1D arrays of floats (support scalar input)
    delays = np.atleast_1d(np.asarray(delays, dtype=float))
    durations = np.atleast_1d(np.asarray(durations, dtype=float))

    # create_gabor_library()
    print('Creating/Loading Gabor Filter Library...')
    library_path = create_gabor_library(xs, ys, thetas, sigmas, offsets, frequencies)
    lib_id = os.path.basename(library_path).split('_')[-1].split('.')[0]

    print('Downsampling Video...')
    downsample_video()

    print('Wavelet Decomposition...')
    path_decompositions =  get_path_wavelet_decomposition(library_path=library_path)
    # path_decompositions =  wavelet_decomposition_gpu(library_path=library_path, device='cuda')

    print('Loading and Subsampling Wavelet Decompositions...')
    assert phase in ['complex', '0', '1'], "phase parameter must be 'complex', '0', or '1'"
    if phase in ['0', '1']:
        wavelet_decomposition = load_phase_dependent_wavelet_decompositions(path_decompositions, phase, xis, yis)
    else:
        wavelet_decomposition = load_complex_wavelet_decomposition(path_decompositions, xis, yis)

    results_path = Path(results_path) / f'lib_{lib_id}'

    for delay in delays:
        for duration in durations:

            # Check whether results already exist for this parameter combination
            # Skip if yes and recompute is False, else compute and save results
            stem = Path(results_filename).stem if results_filename else ''
            filename = f'{stem}__lib_{lib_id}__delay_{delay}__dur_{duration}.h5' if stem else f'lib_{lib_id}__delay_{delay}__dur_{duration}.h5'
            filepath = results_path/filename

            # Due to timeout and memory constraints it happened that some runs were killed mid-way, leaving behind corrupt result files.
            # There we check here, wheter previous results exist and can be opened, before deciding to skip or recompute.
            if filepath.exists() and not recompute:
                try:
                    with h5py.File(filepath, 'r'):
                        pass
                    print(f"Results for lib {lib_id}, delay {delay}, duration {duration} already exist at {filepath}. Skipping computation.")
                    continue
                except OSError:
                    print(f"WARNING: corrupt results file at {filepath} — recomputing.")    
                    filepath.unlink()

            print(f'Computing Neuron Rates to Zebra Frames for delay {delay}, duration {duration} and phase {phase}...')
            # spike_times_list = make_typed_spike_list(spike_times_list)
            _t0 = time.perf_counter()
            neuron_rate_to_zebra_frames = compute_neuron_rate_to_zebra_frames(frame_onset_times, spike_times_list, delay, duration)
            _elapsed = time.perf_counter() - _t0
            print(f"---> compute_neuron_rate_to_zebra_frames took {_elapsed:.4f} s for {len(spike_times_list)} units")

            print('Running Correlation Analysis...')
            ## runs correlation analysis
            rfs = au.PearsonCorrelationPinkNoise(#stim=wavelet_c.reshape(wavelet_c.shape[0], -1), 
                                                    stim=wavelet_decomposition,        
                                                    resp=neuron_rate_to_zebra_frames.T,
                                                    neuron_pos=np.zeros((1,2)),  # dummy value for neuron_pos, 
                                                    nx=wavelet_decomposition.shape[1], 
                                                    ny=wavelet_decomposition.shape[2], 
                                                    ns=ns, 
                                                    n_frequencies=len(frequencies),
                                                    n_phases=wavelet_decomposition.shape[-1],  # use actual number of phases from loaded decomposition
                                                    visual_coverage=analysis_coverage, 
                                                    screen_ratio=screen_ratio, 
                                                    sigmas=sigmas_deg,
                                                    plotting=False,
                                                    n_thetas=n_thetas
                                                    )
            
            # Add additional attributes
            add_to_attributes = {
                "library_id": lib_id,
                "delay": delay,
                "duration": duration,
                "sigmas": list(sigmas),
                "thetas": list(thetas),
                "frequencies": list(frequencies),
                "phase": phase,
                "num_units": len(spike_times_list),
                "num_frames": wavelet_decomposition.shape[0],
            }
            attributes.update(add_to_attributes)

            print('Saving Results...')
            outpath = save_results(rfs,
                                    unit_names_list,
                                    # delay,
                                    # duration,
                                    # sigmas,
                                    # thetas,
                                    # frequencies,
                                    results_path=results_path,
                                    results_filename=filename,
                                    attributes=attributes
                                    )
    return results_path