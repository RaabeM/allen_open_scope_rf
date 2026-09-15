
from pathlib import Path
# from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt
from dandi import dandiapi, download
import os   
import pandas as pd
from scipy.signal import medfilt
from scipy.interpolate import interp1d


import pickle
import utils

import neo
import quantities as pq

import sys
import argparse
from datetime import datetime
import h5py







def compute_rf_sta_zebra(video_loader,
                         spike_times, 
                         zebra_onsets, 
                         zebra_TrialInSequence,
                         window_size = 10,
                         ):

    stim_shape = video_loader.shape

    frames = np.zeros((window_size, stim_shape[0], stim_shape[1]))
    for st in spike_times:
        # Find frame spike falls into 
        index = np.digitize(st, zebra_onsets) - 1
        frame_index = zebra_TrialInSequence[index] - 1 # zebra_TrialInSequence starts with 1 
        
        for i in np.arange(window_size):
            frames[i] += video_loader.get_frame(frame_index - i)
    frames /= len(spike_times)

    return frames



def main(args):

    probe = args.probe
    recompute = args.recompute.lower() == 'true'
    window_size = int(args.window_size)
    results_dir = args.results_dir
    nwb_dir = args.nwb_dir
    logfile = args.log_file
    if logfile != 'None':
        log_f = open(logfile, 'a')
        sys.stdout = log_f
        sys.stderr = log_f
    

    video_loader = utils.VideoLoader()

    # Probe dict contains all dandi filenames as keys. Values are all idx of units on that probe
    probe_a, probe_b, probe_c, probe_d = utils.load_probe_dicts()
    if probe == 'a':
        probe_dict = probe_a
    elif probe == 'b':
        probe_dict = probe_b
    elif probe == 'c':
        probe_dict = probe_c
    elif probe == 'd':
        probe_dict = probe_d
    else:
        raise ValueError(f"Invalid probe: {probe}")

    if args.i == 'None':
        dandi_filepaths_to_process = list(probe_dict.keys())
    else:
        dandi_filepaths_to_process = list(probe_dict.keys())[ int(args.i) : int(args.i) + 1 ]
    

    print(f">>> Processing probe {probe}...")
    for dandi_filepath in dandi_filepaths_to_process:
        print(f">>> Processing file {dandi_filepath}...new")

        nwb = utils.load_dandi(dandi_filepath = dandi_filepath.split('/')[-1],
                                dandiset_id = utils.dandiset_id,
                                dandi_dirpath = nwb_dir
                            )
        df_units = nwb.units.to_dataframe()
        df_zebra = nwb.intervals['Zebra_presentations'].to_dataframe()
        zebra_TrialInSequence = df_zebra['TrialInSequence'].values.astype(float).astype(int)

        zebra_onsets = df_zebra['start_time'].values
        zebra_onsets = zebra_onsets * pq.s

        idx_single_units = [k for k in probe_dict[dandi_filepath] if df_units.iloc[k]['decoder_label']=='sua']
        
        for n, unit_index in enumerate(idx_single_units):
            unit_name = df_units.loc[unit_index, 'unit_name']
            
            result_filename = f"{dandi_filepath.split('/')[-1]}__probe-{probe}__{unit_name}.h5"
            outpath = os.path.join(results_dir, result_filename)

            os.makedirs(results_dir, exist_ok=True)

            # Skip if output already exists
            if os.path.exists(outpath) and not recompute:
                print(f"    - Output already exists, skipping: {outpath}")
                continue
            else:
                print(f"    - [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing unit {n} / {len(idx_single_units)}...")

                spike_times = df_units.loc[unit_index, 'spike_times']
                spike_times = neo.core.SpikeTrain(spike_times, t_start=spike_times[0], t_stop=spike_times[-1], units=pq.s)

                if spike_times.t_start > zebra_onsets[-1] or spike_times.t_stop < zebra_onsets[0]:
                    continue

                spike_times = spike_times.time_slice(zebra_onsets[window_size-1], zebra_onsets[-1])

                frames = compute_rf_sta_zebra(video_loader=video_loader,
                            spike_times=spike_times, 
                            zebra_onsets=zebra_onsets, 
                            zebra_TrialInSequence=zebra_TrialInSequence,
                            window_size = 10,
                            )             
                
                # number of spikes (ensure plain int)
                n_spikes = int(len(spike_times))

                dandi_file = dandi_filepath.split('/')[-1]
                with h5py.File(outpath, 'a') as hf:
                    # create or replace a group for this unit
                    grp_name = unit_name
                    if grp_name in hf:
                        del hf[grp_name]
                    grp = hf.create_group(grp_name)

                    # store frames and metadata under the unit group
                    grp.create_dataset('frames', data=frames, compression='gzip')
                    grp.create_dataset('n_spikes', data=np.int32(n_spikes))
                    grp.create_dataset('probe', data=probe)

                # print(f"    - Saved results to {outpath}")
                # np.save(outpath, frames)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute STA receptive fields from zebra noise stimulus")
    parser.add_argument("--probe", required=True, help="Probe to analyze: a, b, c, or d")
    parser.add_argument("--nwb-dir", required=True, dest="nwb_dir", help="Directory containing NWB files (e.g., /data/allen_open_scope/)")
    parser.add_argument("--results-dir", required=True, dest="results_dir", help="Directory to save results (e.g., /results/rf/sta/zebra/)")
    parser.add_argument("--recompute", nargs='?', default='False', help="Recompute results even if they exist")
    parser.add_argument("--window_size", nargs='?', default='10', help="Optional: Number of frames to include in STA")
    parser.add_argument("--log_file", nargs='?', default='None', help="Optional: Path to logfile for stdout/stderr")
    parser.add_argument("--i", nargs='?', default='None', help="Optional: Index of single file to process")

    args = parser.parse_args()
    main(args)