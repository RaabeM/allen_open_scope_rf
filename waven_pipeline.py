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


def wavelet_decomposition(library_path=library_path, movpath=movpath):
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


@njit
def compute_neuron_rate_to_zebra_frames(frame_onset_times, spike_times_list, delay=0.0, duration=1/30):
    num_units = len(spike_times_list)
    num_trials = len(frame_onset_times)
    out = np.zeros((num_units, num_trials), dtype=np.float64)
    # out = np.zeros((num_trials, num_units), dtype=np.float64)


    for u in range(num_units):
        unit_spike_times = spike_times_list[u]
        for i in range(num_trials):
            start_time = frame_onset_times[i] + delay
            # stop_time  = frame_offset_times[i] + delay
            stop_time  = start_time + duration
            rate = np.sum((unit_spike_times >= start_time) & (unit_spike_times < stop_time)) / (stop_time - start_time)
            out[u, i] = rate

    return out



def save_results(rfs_zebra, unit_names_list, delay, filename, results_dir='../../results/allen_open_scope/rf/waven/zebra/'):
    # Save results
    
    outpath = os.path.join(results_dir, filename)
    if os.path.exists(outpath):
        os.remove(outpath)


    with h5py.File(outpath, 'a') as hf:
        # create or replace a group for this unit
        for idx, unit_name in enumerate(unit_names_list):
            
            grp_name = unit_name
            if grp_name in hf:
                del hf[grp_name]
            grp = hf.create_group(grp_name)

            grp.create_dataset('delay', data=delay)

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


def read_results(filename,results_dir):

    outpath = os.path.join(results_dir, filename)
    with h5py.File(outpath, 'r') as file:
        unit_names_saved = list(file.keys())
        opt_sigmas = []
        response_theta_0s = []
        response_theta_45s = []
        response_theta_90s = []

        for unit_name in unit_names_saved:
            grp = file[unit_name]
            opt_simga = grp['opt_sigma'][()]
            delay = grp['delay'][()]
            response_theta_0 = grp['response_theta_0'][:]
            response_theta_45 = grp['response_theta_45'][:]
            response_theta_90 = grp['response_theta_90'][:]

            
            opt_sigmas.append( opt_simga )
            response_theta_0s.append(response_theta_0)
            response_theta_45s.append(response_theta_45)
            response_theta_90s.append(response_theta_90)



    df_res_waven = pd.DataFrame({'unit_name':unit_names_saved, 
                                'opt_simgas':opt_sigmas, 
                                'response_theta_0':response_theta_0s,
                                'response_theta_45':response_theta_45s,
                                'response_theta_90':response_theta_90s,
                                'delay':delay,
                                })

    df_res_waven['max_response_0'] = df_res_waven['response_theta_0'].apply(lambda x: np.max(x))
    df_res_waven['max_response_45'] = df_res_waven['response_theta_45'].apply(lambda x: np.max(x))
    df_res_waven['max_response_90'] = df_res_waven['response_theta_90'].apply(lambda x: np.max(x))
    df_res_waven['overall_max'] = df_res_waven[['max_response_0', 'max_response_45', 'max_response_90',]].abs().max(axis=1)

    return df_res_waven


def save_results_new(rfs, unit_names_list, delay, duration, sigmas, thetas, frequencies,
                     results_path='./', results_filename=''):
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    if not results_filename.endswith('.h5'):
        results_filename += '.h5'

    outpath = results_path / results_filename
    if outpath.exists():
        outpath.unlink()

    with h5py.File(outpath, 'w') as hf:
        for name, val in (('sigmas', sigmas), ('thetas', thetas), ('frequencies', frequencies),
                          ('delay', delay), ('duration', duration)):
            hf.attrs[name] = val

        hf.create_dataset('unit_ids', data=list(unit_names_list))
        hf.create_dataset('correlation_matrix', data=rfs[0], compression='gzip', compression_opts=4)
        hf.create_dataset('best_gabor_params_idx', data=rfs[1])
        hf.create_dataset('best_gabor_params_degree', data=rfs[2])
        hf.create_dataset('abs_max_value', data=rfs[3])

    return outpath


def subsample_wavelet_responses(wavelets_complex, xis, yis):

    wavelets_complex_subsampled = np.zeros((wavelets_complex.shape[0], len(xis), len(yis), wavelets_complex.shape[3], wavelets_complex.shape[4], wavelets_complex.shape[5]))
    for i, x_i in enumerate(tqdm(xis)):
        for j, y_i in enumerate(yis):
            wavelets_complex_subsampled[:, i, j, :, :, :] = wavelets_complex[:, int(x_i), int(y_i)]
    
    return wavelets_complex_subsampled


def convolve_gabor_library_with_video(library_np, video_np, device='cuda'):
    """
    library_np: np.ndarray shaped (Nx, Ny, Ntheta, Nsigma, Nfrequency, Noffset, K)
                where K == Nx*Ny and last axis is flattened kernel for each output pixel.
    video_np:   np.ndarray shaped (Nframes, Nx, Ny)
    returns:    torch.Tensor shaped (Nframes, Nx, Ny, Ntheta, Nsigma, Nfrequency, Noffset)
    """
    # convert & check
    lib = torch.as_tensor(library_np, dtype=torch.float32, device=device)
    vid = torch.as_tensor(video_np, dtype=torch.float32, device=device)

    Nframes = vid.shape[0]
    K = lib.shape[-1]
    assert vid.shape[1] * vid.shape[2] == K, "Nx*Ny must equal library last dim"

    # flatten frames: (Nframes, K)
    frames = vid.reshape(Nframes, -1)

    # compute dot product contracting the last axis of lib with frames
    # einsum indices: 'fk,xyabcdk->fxyabcd'
    out = torch.einsum('fk,xyabcdk->fxyabcd', frames, lib)
    # out shape: (Nframes, Nx, Ny, Ntheta, Nsigma, Nfrequency, Noffset)
    return out



def convolve_library_inmemory(library_np, video_np, device='cuda', block_pixels=256, dtype=np.float32):
    """
    Compute convolution in GPU in blocks, keep each computed block in RAM and
    append; return full array of shape (Nframes, Nx, Ny, Ntheta, Nsigma, Nfreq, Noffset).
    WARNING: final result is kept in memory.
    """
    Nx, Ny, Ntheta, Nsigma, Nfreq, Noffset, K = library_np.shape
    Nframes = video_np.shape[0]
    assert K == Nx * Ny
    n_filters_per_pixel = Ntheta * Nsigma * Nfreq * Noffset

    # flatten frames once (Nframes, K) and move to device
    frames = video_np.reshape(Nframes, K).astype(dtype)
    dev = torch.device(device)
    frames_t = torch.as_tensor(frames, dtype=torch.float32, device=dev)

    # prepare pixel index order and accumulator
    indices = [(i, j) for i in range(Nx) for j in range(Ny)]
    chunks = []  # list of numpy arrays shaped (Nframes, bs, Ntheta, Nsigma, Nfreq, Noffset)

    for start in range(0, len(indices), block_pixels):
        block = indices[start:start + block_pixels]
        bs = len(block)

        # build filters matrix (K, bs * n_filters_per_pixel)
        filters_block = np.empty((K, bs * n_filters_per_pixel), dtype=dtype)
        for bi, (ix, iy) in enumerate(block):
            fvals = library_np[ix, iy, :, :, :, :, :]                  # (Ntheta,Nsigma,Nfreq,Noffset,K)
            fvals_rs = fvals.reshape(n_filters_per_pixel, K)           # (n_filters_per_pixel, K)
            filters_block[:, bi*n_filters_per_pixel:(bi+1)*n_filters_per_pixel] = fvals_rs.T

        # GPU matmul: (Nframes, K) @ (K, M) -> (Nframes, M)
        filters_t = torch.as_tensor(filters_block, dtype=torch.float32, device=dev)
        out_block = torch.matmul(frames_t, filters_t)  # (Nframes, bs * n_filters_per_pixel)
        out_block = out_block.cpu().numpy().astype(dtype)

        # reshape to (Nframes, bs, Ntheta, Nsigma, Nfreq, Noffset) and append to list
        out_block = out_block.reshape(Nframes, bs, Ntheta, Nsigma, Nfreq, Noffset)
        chunks.append(out_block)

        # cleanup GPU temporaries
        del filters_t, out_block
        torch.cuda.empty_cache()

    # concatenate all pixel blocks along pixel axis -> (Nframes, Nx*Ny, Ntheta,...)
    concatenated = np.concatenate(chunks, axis=1)  # axis=1 is pixel axis
    # reshape to final shape
    final = concatenated.reshape(Nframes, Nx, Ny, Ntheta, Nsigma, Nfreq, Noffset)
    return final

import h5py
from typing import Optional

def convolve_per_frame_wholeframe(library_np: np.ndarray,
                                  video_np: np.ndarray,
                                  out_path: Optional[str] = None,
                                  device: str = 'cuda',
                                  block_pixels: int = 128,
                                  dtype=np.float32,
                                  to_hdf5: bool = True):
    """
    For each frame: move the whole flattened frame to device once, then iterate
    over small blocks of output pixels and matmul frame @ filters_block.

    library_np: shape (Nx, Ny, Ntheta, Nsigma, Nfreq, Noffset, K) with K == Nx*Ny
    video_np:   shape (Nframes, Nx, Ny)
    out_path:   HDF5 filename if to_hdf5 True
    block_pixels: number of output pixels per block
    """
    Nx, Ny, Ntheta, Nsigma, Nfreq, Noffset, K = library_np.shape
    Nframes = video_np.shape[0]
    assert K == Nx * Ny
    n_filters = Ntheta * Nsigma * Nfreq * Noffset
    dev = torch.device(device)

    # prepare pixel index blocks
    indices = [(ix, iy) for ix in range(Nx) for iy in range(Ny)]
    blocks = [indices[s:s+block_pixels] for s in range(0, len(indices), block_pixels)]

    # prepare output store
    out_shape = (Nframes, Nx, Ny, Ntheta, Nsigma, Nfreq, Noffset)
    if to_hdf5:
        if out_path is None:
            raise ValueError("out_path must be provided when to_hdf5=True")
        h5f = h5py.File(out_path, 'w')
        dset = h5f.create_dataset('out', shape=out_shape, dtype='f4',
                                  chunks=(1, min(Nx,16), min(Ny,16), Ntheta, Nsigma, Nfreq, Noffset),
                                  compression="gzip")
    else:
        out_frames = []

    for fi in range(Nframes):
        # move full flattened frame to device once
        frame = video_np[fi].reshape(K).astype(dtype)               # (K,)
        frame_t = torch.as_tensor(frame, dtype=torch.float32, device=dev)  # (K,)

        # buffer for this frame's pixel outputs (flat pixel axis)
        frame_pixels_out = np.empty((len(indices), n_filters), dtype=dtype)
        pos = 0

        for block in blocks:
            bs = len(block)
            # build filter block on CPU: shape (K, bs * n_filters)
            fb = np.empty((K, bs * n_filters), dtype=dtype)
            for bi, (ix, iy) in enumerate(block):
                fvals = library_np[ix, iy, :, :, :, :, :]     # (Ntheta,Nsigma,Nfreq,Noffset,K)
                fvals_rs = fvals.reshape(n_filters, K)        # (n_filters, K)
                fb[:, bi*n_filters:(bi+1)*n_filters] = fvals_rs.T  # (K, n_filters) per pixel

            # move filters block to device and matmul with the full frame
            filters_t = torch.as_tensor(fb, dtype=torch.float32, device=dev)   # (K, M)
            out_block = torch.matmul(frame_t.unsqueeze(0), filters_t)          # (1, M)
            out_block = out_block.squeeze(0).cpu().numpy().astype(dtype)       # (M,)
            out_block = out_block.reshape(bs, n_filters)                       # (bs, n_filters)

            frame_pixels_out[pos:pos+bs, :] = out_block
            pos += bs

            # free GPU mem for this block
            del filters_t, out_block, fb
            torch.cuda.empty_cache()

        # reshape flat pixel axis to (Nx, Ny, Ntheta, ...)
        frame_full = frame_pixels_out.reshape(Nx, Ny, Ntheta, Nsigma, Nfreq, Noffset)

        if to_hdf5:
            dset[fi, :, :, :, :, :, :] = frame_full
        else:
            out_frames.append(frame_full)

        # free per-frame GPU memory
        del frame_t
        torch.cuda.empty_cache()

    if to_hdf5:
        h5f.close()
        return out_path
    else:
        return np.stack(out_frames, axis=0)


def wavelet_decomposition_gpu(library_path=library_path, movpath=movpath, device='cuda'):

    lib_name = os.path.basename(library_path)
    lib_id = lib_name.split('_')[-1].split('.')[0]

    movpath = Path(movpath)
    path_decompositions = movpath.parent / 'wavelet_decompositions' / f'lib-{lib_id}'
    Path(path_decompositions).mkdir(exist_ok=True)

    # Load downsampled video
    video_path = f'{movpath.parent/movpath.stem}_downsampled__{ny}_{nx}.npy'
    if not os.path.exists(video_path):
        print(f"Downsampled video not found at {video_path}. Running downsampling...")
        downsample_video()
    videodata=np.load(video_path)

    # Load filter library
    filter_library = np.load(library_path)

    # Make the composition
    wavelet_decomposition = convolve_per_frame_wholeframe(filter_library, videodata, device=device, to_hdf5=False)
    print(wavelet_decomposition.shape)

    for phase in [0, 1]:
        filename = f'dwt_videodata_{phase}.npy'
        file_path = os.path.join(path_decompositions, filename)
        np.save(file_path, wavelet_decomposition[..., phase])

    return path_decompositions


def load_wavelet_decomposition(path_decompositions, xis=[], yis=[]):

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

    return wavelet_c
                                                            

def full_pipeline(frame_onset_times,
                #   frame_offset_times,
                  spike_times_list,
                  delay,
                  duration,
                  unit_names_list,
                  xis=[],
                  yis=[],
                  results_path='./',
                  results_filename=''):

    print(f'Running Waven Pipeline for delay={delay} seconds und duration={duration} seconds...')

    print('Creating Gabor Library...')
    # create_gabor_library()
    library_path = create_gabor_library(xs, ys, thetas, sigmas, offsets, frequencies)
    lib_id = os.path.basename(library_path).split('_')[-1].split('.')[0]

    print('Downsampling Video...')
    downsample_video()

    print('Wavelet Decomposition...')
    path_decompositions =  wavelet_decomposition(library_path=library_path)
    # path_decompositions =  wavelet_decomposition_gpu(library_path=library_path, device='cuda')




    print('Computing Neuron Rates to Zebra Frames...')
    spike_times_list = make_typed_spike_list(spike_times_list)
    neuron_rate_to_zebra_frames = compute_neuron_rate_to_zebra_frames(frame_onset_times, spike_times_list, delay, duration)
    # neuron_rate_to_zebra_frames = neuron_rate_to_zebra_frames[:10, :]
    print(neuron_rate_to_zebra_frames.shape)

    ## the spikes data have to be time registered to the stimulus frames
    ## MR Only one trial thus no repeatability
    # respcorr_zebra = au.repetability_trial3(spks, neuron_pos, plotting=True)
    # wavelets0, wavelets1, wavelet_c = lpn.coarseWavelet(path=path_decompositions,
    #                                                     downsampling=False, 
    #                                                     nx0=nx, 
    #                                                     ny0=ny, 
    #                                                     nx=nx, 
    #                                                     ny=ny, 
    #                                                     no=n_thetas, 
    #                                                     ns=ns,
    #                                                     nf=len(frequencies),
    #                                                     chunk_size=1000)
    print('Loading and Subsampling Wavelet Decompositions...')
    wavelet_c = load_wavelet_decomposition(path_decompositions, xis, yis)
    

    print('Running Correlation Analysis...')
    ## runs correlation analysis
    rfs = au.PearsonCorrelationPinkNoise(stim=wavelet_c.reshape(wavelet_c.shape[0], -1), 
                                            resp=neuron_rate_to_zebra_frames.T,
                                            neuron_pos=np.zeros((1,2)),  # dummy value for neuron_pos, 
                                            nx=wavelet_c.shape[1], 
                                            ny=wavelet_c.shape[2], 
                                            ns=ns, 
                                            n_frequencies=len(frequencies),
                                            visual_coverage=analysis_coverage, 
                                            screen_ratio=screen_ratio, 
                                            sigmas=sigmas_deg,
                                            plotting=False,
                                            n_thetas=n_thetas
                                            )


    print('Saving Results...')
    stem = Path(results_filename).stem if results_filename else ''
    filename = f'{stem}__lib_{lib_id}__delay_{delay}__dur_{duration}' if stem else f'lib_{lib_id}__delay_{delay}__dur_{duration}'

    outpath = save_results_new(rfs,
                               unit_names_list,
                               delay,
                               duration,
                               sigmas,
                               thetas,
                               frequencies,
                               results_path=results_path,
                               results_filename=filename)
    