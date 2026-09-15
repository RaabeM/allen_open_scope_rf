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

import sys
sys.path.append('..')
import utils


from numba import njit, typed
from waven_settings import *
# the SLAP2 branch reuses the gap handling already written for the Siegle maps
# (drop_missing_samples / min_samples_for / window_means) rather than growing a
# second convention for the same blanking periods
import rf_siegle_ophys as siegle
import h5py
import json
import hashlib
import numpy as np
from typing import Dict, Any, Tuple
import time

# Fewest imaged movie frames a SLAP2 (delay, duration) cell must retain before
# the correlation is worth running at all. A Zebra repeat is ~1800 frames and a
# session images roughly a third of it, so anything near this floor means the
# window landed almost entirely in the blanking gaps.
MIN_COVERED_FRAMES = 300



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
    lib = wg.makeFilterLibrary2(xs, ys, thetas, sigmas, np.deg2rad(offsets), frequencies) # Note: Converting offsets to radians here is not very beautiful, but allow only minor mixes to prior analysis
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


def get_downsampled_video_path(movie_path=None):
    """Path of the downsampled .npy for `movie_path` (defaults to the settings movpath)."""
    movie_path = Path(movie_path if movie_path is not None else movpath)
    return movie_path.parent / f'{movie_path.stem}_downsampled__{ny}_{nx}.npy'


def downsample_video(movie_path=None):
    # downsample the video according to the analysis coverage
    movie_path = Path(movie_path if movie_path is not None else movpath)
    downsampled_path = get_downsampled_video_path(movie_path)
    if not os.path.exists(downsampled_path):
        wg.downsample_video_binary(os.path.abspath(movie_path),
                                np.array(visual_coverage),
                                np.array(analysis_coverage),
                                shape=(ny, nx),
                                chunk_size=500,
                                ratios=(ratio_x, ratio_y)
                                )


def get_path_wavelet_decomposition(library_path=library_path, movpath=None):
    # Video decomposition
    lib_name = os.path.basename(library_path)
    lib_id = lib_name.split('_')[-1].split('.')[0]

    movpath = Path(movpath if movpath is not None else globals()['movpath'])
    print(movpath)
    # The movie stem has to be part of the path: several stimulus movies live in
    # the same folder (the zebra scale variants), and a decomposition is specific
    # to the movie *and* the filter library. Keyed on lib_id alone, the second
    # movie would silently reuse the first one's dwt_videodata_*.npy via the
    # skip-if-exists branch below and every variant would yield identical RFs.
    path_decompositions = movpath.parent / 'wavelet_decompositions' / movpath.stem / f'lib-{lib_id}'
    Path(path_decompositions).mkdir(exist_ok=True, parents=True)

    for phase in [0, 1]:
        filename = f'dwt_videodata_{phase}.npy'
        file_path = os.path.join(path_decompositions, filename)

        if os.path.exists(file_path):
            print(f"Wavelet decomposition for lib {lib_id} and phase {phase} already exists at {file_path}. Skipping decomposition.")
        else:
            video_path = get_downsampled_video_path(movpath)
            if not os.path.exists(video_path):
                print(f"Downsampled video not found at {video_path}. Running downsampling...")
                downsample_video(movpath)
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


def average_signal_over_window(times, signal, sample_times, duration):
    """
    Average a continuously sampled signal over a window of length `duration`
    starting at each time in `sample_times`.

    This is the continuous-signal counterpart of the spike-rate window in
    compute_neuron_rate_to_zebra_frames: a negative duration averages backwards
    from the sample time instead of forwards. The mean (rather than a sum over
    `duration`) keeps the sign of the response independent of the sign of the
    window, which matters because Pearson correlation is sign-sensitive.

    times        : (n_times,) sorted acquisition times of `signal`
    signal       : (n_units, n_times) continuous signal, e.g. dF/F
    sample_times : (n_samples,) window start times
    duration     : window length in seconds; sign sets the direction

    Returns (n_units, n_samples). Windows shorter than the sampling interval
    fall back to the single sample they start on, so the output never contains
    empty-window NaNs as long as every `sample_times` entry is in `times`.
    """
    lo = np.minimum(sample_times, sample_times + duration)
    hi = np.maximum(sample_times, sample_times + duration)

    i_lo = np.searchsorted(times, lo, side='left')
    i_hi = np.searchsorted(times, hi, side='right')
    # a window is empty only if it falls between two samples; anchor it to the
    # nearest preceding sample so every window averages at least one value
    i_hi = np.maximum(i_hi, i_lo + 1)
    i_lo = np.minimum(i_lo, signal.shape[1] - 1)
    i_hi = np.minimum(i_hi, signal.shape[1])

    # cumulative sums turn the per-window means into two gathers
    csum = np.zeros((signal.shape[0], signal.shape[1] + 1), dtype=np.float64)
    np.cumsum(signal, axis=1, dtype=np.float64, out=csum[:, 1:])

    counts = (i_hi - i_lo).astype(np.float64)
    return (csum[:, i_hi] - csum[:, i_lo]) / counts


def remove_bout_means(times, signal, bout_bounds):
    """Subtract each acquisition bout's own mean from every unit.

    SLAP2 records in ~30 s bouts separated by blanking periods, and the dF/F
    baseline is not guaranteed to return to the same level after a gap. Left in,
    that step change is a slow signal spanning whole bouts, and it correlates
    with whatever the stimulus happens to do slowly over the same stretch.
    Centring each bout removes it while leaving the within-bout modulation - the
    part that carries the receptive field - untouched.

    times       : (n_times,) sorted acquisition times
    signal      : (n_units, n_times)
    bout_bounds : (n_bouts, 2) array of (start_time, stop_time), e.g. the
                  start_time/stop_time columns of NWBStream.slap2_segments()

    Returns a new (n_units, n_times) array; `signal` is not modified.
    """
    out = np.array(signal, dtype=np.float64, copy=True)
    for lo, hi in np.asarray(bout_bounds, dtype=float):
        m = (times >= lo) & (times <= hi)
        if m.any():
            out[:, m] -= out[:, m].mean(axis=1, keepdims=True)
    return out


def compute_slap2_response_to_zebra_frames(times, signal, frame_onset_times,
                                           delay=0.0, duration=1/30,
                                           min_coverage=siegle.MIN_COVERAGE):
    """Mean dF/F per movie frame, dropping frames that were not imaged.

    The counterpart of compute_neuron_rate_to_zebra_frames for SLAP2. SLAP2
    samples at ~200 Hz against a 30 Hz movie, so - unlike the mesoscope - there
    are several samples per frame and the resampling runs frame-first: one
    response per movie frame, averaged over [onset + delay, onset + delay +
    duration).

    The catch is that imaging is not continuous, so most frames of a session
    have no samples at all. average_signal_over_window deliberately never
    returns NaN: it anchors an empty window to the nearest preceding sample.
    That is right for a continuously sampled mesoscope and wrong here, where it
    would fill an un-imaged frame with a value from before the gap. So coverage
    is tested first and uncovered frames are dropped from the design matrix
    entirely, rather than being filled in.

    times            : (n_times,) sorted acquisition times
    signal           : (n_units, n_times) dF/F, already free of NaN samples
    frame_onset_times: (n_frames,) movie frame onsets
    duration         : window length in seconds; sign sets the direction, as in
                       average_signal_over_window

    Returns (responses, covered):
        responses : (n_units, covered.sum()) mean dF/F per retained frame
        covered   : (n_frames,) bool mask, to apply to the stimulus rows
    """
    starts = np.asarray(frame_onset_times, dtype=float) + delay
    lo = np.minimum(starts, starts + duration)
    hi = np.maximum(starts, starts + duration)

    min_samples = siegle.min_samples_for(times, abs(duration), min_coverage)
    resp = siegle.window_means(signal, times, lo, hi, min_samples=min_samples)

    # window_means blanks a whole column at once when a window is undersampled
    covered = np.isfinite(resp).all(axis=0)
    return resp[:, covered], covered


# def compute_mean_signal_to_zebra_frames(frame_onset_times, signal, delay=0.0, duration=1/30):
#     starts = frame_onset_times + delay      # vectorized, shape (num_trials,)
#     stops  = starts + duration

#     num_units = len(signal)
#     out = np.zeros((num_units, len(frame_onset_times)), dtype=np.float64)
#     times = signal.columns.to_numpy()
#     idx_starts = np.searchsorted(times, starts)
#     idx_stops = np.searchsorted(times, stops)

#     signals = signal.to_numpy() # (n_units, n_times)

#     out = np.array([np.mean(signals[:, s:e], axis=1) for s,e in zip(idx_starts, idx_stops)]).T # (n_units, n_frame_onset_times)
#     return out

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
    print(f"Saving results to {outpath}...")

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
            hf.create_dataset('correlation_matrix', data=rfs[0].astype(np.float16), compression='gzip', compression_opts=4, dtype=np.float16)
            hf.create_dataset('best_gabor_params_idx', data=rfs[1])
            hf.create_dataset('best_gabor_params_degree', data=rfs[2])
            hf.create_dataset('abs_max_value', data=rfs[3])

        if outpath.exists():
            outpath.unlink()
        tmp_path.rename(outpath)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        print(f"Error saving results: {e}")
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
                  input_type='spiketimes',
                  movie_path=None,
                  frame_index=None,
                  bout_bounds=None,
                  min_coverage=siegle.MIN_COVERAGE
                  ):
    """
    movie_path: stimulus movie to correlate against. Defaults to the `movpath`
    from waven_settings; pass it explicitly when a session shows more than one
    movie (e.g. the zebra scale variants), so that each gets its own
    downsampling and wavelet decomposition.

    frame_index: (n_onsets,) row index into the wavelet decomposition for each
    entry of `frame_onset_times`. Required for input_type='slap2', where the
    onsets pool several repeats of the movie and so cannot be matched to the
    decomposition by position - the same stimulus row recurs on every repeat.

    bout_bounds: (n_bouts, 2) array of (start, stop) acquisition times, used by
    input_type='slap2' to centre each imaging bout separately; see
    remove_bout_means. Pass None to skip.

    min_coverage: fraction of a response window that must actually be sampled
    for the corresponding movie frame to be kept (input_type='slap2' only).
    """

    # ensure delays and durations are numpy 1D arrays of floats (support scalar input)
    delays = np.atleast_1d(np.asarray(delays, dtype=float))
    durations = np.atleast_1d(np.asarray(durations, dtype=float))

    movie_path = Path(movie_path) if movie_path is not None else Path(movpath)

    # create_gabor_library()
    print('Creating/Loading Gabor Filter Library...')
    library_path = create_gabor_library(xs, ys, thetas, sigmas, offsets, frequencies)
    lib_id = os.path.basename(library_path).split('_')[-1].split('.')[0]

    print('Downsampling Video...')
    downsample_video(movie_path)

    print('Wavelet Decomposition...')
    path_decompositions =  get_path_wavelet_decomposition(library_path=library_path, movpath=movie_path)
    # path_decompositions =  wavelet_decomposition_gpu(library_path=library_path, device='cuda')

    print('Loading and Subsampling Wavelet Decompositions...')
    assert phase in ['complex', '0', '1'], "phase parameter must be 'complex', '0', or '1'"
    if phase in ['0', '1']:
        wavelet_decomposition = load_phase_dependent_wavelet_decompositions(path_decompositions, phase, xis, yis)
    else:
        wavelet_decomposition = load_complex_wavelet_decomposition(path_decompositions, xis, yis)

    # Stimulus and response must be sampled at the same timepoints. For spike
    # times, one frame onset corresponds to exactly one movie frame, so a length
    # mismatch means the onsets were built for a different movie than the one
    # decomposed here - silent garbage rather than an error further down.
    if input_type == 'spiketimes' and len(frame_onset_times) != wavelet_decomposition.shape[0]:
        raise ValueError(
            f'{len(frame_onset_times)} frame onset times but {wavelet_decomposition.shape[0]} '
            f'frames in the decomposition of {movie_path.name}. '
            'The onsets must be generated from this movie.'
        )

    # Dropping blanked samples and centring the bouts does not depend on delay
    # or duration, so it is done once rather than per grid cell.
    if input_type == 'slap2':
        if frame_index is None:
            raise ValueError(
                "input_type='slap2' needs frame_index: the onsets pool several "
                'repeats of the movie, so the matching stimulus row cannot be '
                "inferred from position. Use the 'frame' column of "
                'NWBStream.zebra_frame_times().'
            )
        frame_index = np.asarray(frame_index, dtype=int)
        if len(frame_index) != len(frame_onset_times):
            raise ValueError(
                f'{len(frame_index)} frame indices but {len(frame_onset_times)} '
                'frame onset times; they must describe the same rows.'
            )
        if frame_index.max() >= wavelet_decomposition.shape[0]:
            raise ValueError(
                f'frame_index reaches {frame_index.max()} but the decomposition of '
                f'{movie_path.name} has only {wavelet_decomposition.shape[0]} frames. '
                'The onsets must be generated from this movie.'
            )

        slap2_times = np.asarray(spike_times_list.columns.to_numpy(), dtype=float)
        slap2_dff = spike_times_list.to_numpy(dtype=np.float64)
        n_raw = slap2_dff.shape[1]
        slap2_dff, slap2_times = siegle.drop_missing_samples(slap2_dff, slap2_times)
        print(f'{slap2_dff.shape[0]} ROIs, {slap2_dff.shape[1]} samples '
              f'({n_raw - slap2_dff.shape[1]} blanked), '
              f'{len(frame_onset_times)} movie frames')

        if bout_bounds is not None:
            slap2_dff = remove_bout_means(slap2_times, slap2_dff, bout_bounds)
            print(f'Centred {len(np.asarray(bout_bounds))} acquisition bouts')

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
            if input_type == 'spiketimes':
                stim = wavelet_decomposition
                neuron_rate_to_zebra_frames = compute_neuron_rate_to_zebra_frames(frame_onset_times, spike_times_list, delay, duration)
            elif input_type == 'mesoscope':
                # mesoscope data is a DataFrame with time in columns and ROIs in rows
                times = spike_times_list.columns.to_numpy()
                dff = spike_times_list.to_numpy()  # (n_rois, n_times)
                # we have to keep in mind that the mesoscope sampling rate is ~10Hz, so larger then the zebra frame rate
                # Therefore only select the frame immediately before a mesoscope data point and drop the rest.
                # The response at time t is attributed to the frame shown `delay` seconds earlier.
                t_lookup = times - delay
                keep = (t_lookup > frame_onset_times[0]) & (t_lookup < frame_onset_times[-1])
                # index of the last frame whose onset is strictly before t_lookup
                idxs = np.searchsorted(frame_onset_times, t_lookup[keep], side='left') - 1
                # stimulus and response must be restricted to the *same* samples,
                # otherwise the two matrices have different numbers of timepoints
                stim = wavelet_decomposition[idxs]
                # average dF/F over `duration` from each retained sample; windows
                # shorter than the ~0.1 s mesoscope interval reduce to that sample
                neuron_rate_to_zebra_frames = average_signal_over_window(times, dff, times[keep], duration)
            elif input_type == 'slap2':
                # SLAP2 samples ~6x faster than the movie, so the resampling runs
                # the other way round from the mesoscope: one response per movie
                # frame, with the frames that fall in a blanking gap dropped from
                # stimulus and response alike.
                neuron_rate_to_zebra_frames, covered = compute_slap2_response_to_zebra_frames(
                    slap2_times, slap2_dff, frame_onset_times, delay, duration,
                    min_coverage=min_coverage)
                # The decomposition is ~100 GB at the current settings and the
                # fancy index below copies most of it. Drop the previous cell's
                # copy first, or the assignment holds two at once and the peak
                # doubles for no reason.
                stim = None
                rows = frame_index[covered]
                # a fully covered session indexes every frame in order, where the
                # copy buys nothing over a view of the decomposition itself
                if len(rows) == wavelet_decomposition.shape[0] and np.array_equal(
                        rows, np.arange(wavelet_decomposition.shape[0])):
                    stim = wavelet_decomposition
                else:
                    stim = wavelet_decomposition[rows]
                print(f'{covered.sum()} of {len(covered)} movie frames imaged '
                      f'({100 * covered.mean():.0f} %)')
                if covered.sum() < MIN_COVERED_FRAMES:
                    print(f'WARNING: only {covered.sum()} covered frames for delay {delay}, '
                          f'duration {duration} — skipping, the correlation would be noise.')
                    continue
            else:
                raise ValueError(
                    f'unknown input_type {input_type!r}; expected one of '
                    "'spiketimes', 'mesoscope', 'slap2'"
                )


            print('Running Correlation Analysis...')
            ## runs correlation analysis
            rfs = au.PearsonCorrelationPinkNoise(#stim=wavelet_c.reshape(wavelet_c.shape[0], -1),
                                                    stim=stim,
                                                    resp=neuron_rate_to_zebra_frames.T,
                                                    neuron_pos=np.zeros((1,2)),  # dummy value for neuron_pos,
                                                    nx=stim.shape[1],
                                                    ny=stim.shape[2],
                                                    ns=ns,
                                                    n_frequencies=len(frequencies),
                                                    n_phases=stim.shape[-1],  # use actual number of phases from loaded decomposition
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
                "num_frames": stim.shape[0],
                "movie": movie_path.name,
            }
            if input_type == 'slap2':
                # without these, a weak map cannot be told apart from one
                # estimated off a handful of frames that happened to be imaged
                add_to_attributes.update({
                    "n_frames_covered": int(covered.sum()),
                    "n_frames_total": int(covered.size),
                    "frac_frames_covered": float(covered.mean()),
                    "min_coverage": float(min_coverage),
                    "bouts_centred": bout_bounds is not None,
                })
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
    