import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
# from Waven.LoadPinkNoise import *
from Waven import WaveletGenerator as wg

import sys
sys.path.append('..')

import utils
from waven_settings import *
from waven_pipeline import *
import argparse

from datetime import datetime
from pathlib import Path

import optimize_waven_parameters as owp


def main(nwb_path=None, results_dir=None, probe_idx=0):

    results_dir = Path(results_dir)
    session_name = Path(nwb_path).stem

    delays = np.arange(0.0, 0.35, 0.05)
    durations = np.arange(0.03, 0.28, 0.05)

    print('Loading Dandi NWB file...')
    stream = utils.open_local(nwb_path)
    nwb = stream.nwb

    units_df = stream.units_df()
    probes = units_df['probe'].unique()
    probe = probes[probe_idx]

    # for probe in units_df['probe'].unique():
    unit_names = units_df.loc[units_df['probe'] == probe, 'unit_name'].values
    spiketimes = units_df.loc[units_df['probe'] == probe, 'spike_times'].values

    zebra_df = stream.zebra_df()
    for i_trial, trialnumber in enumerate(zebra_df['TrialNumber'].unique()):

        frame_onset_times = zebra_df.loc[zebra_df['TrialNumber'] == trialnumber, 'start_time'].values


        attributes = {'session' : session_name, 
                        'probe' : probe, 
                        'date_computed' : datetime.today().strftime('%Y-%m-%d'),
                        'nwb_trialnumber' : trialnumber,
                        'trial' : i_trial,
                        }

        for phase in ['1']:#['0', '1']:
            results_path = results_dir/probe/f'trial_{i_trial}'/f'phase_{phase}'
            results_path = full_pipeline(frame_onset_times,
                    spiketimes,
                    delays,
                    durations,
                    unit_names,
                #   xis=xis,
                #   yis=yis,
                    results_path=results_path,
                    results_filename='',
                    attributes=attributes,
                    recompute=True,
                    phase=phase,
                    )
            
            # Optimize parameters 
            print('Optimizing parameters...')
            print(results_path)
            outpath = Path(f"/mnt/ceph-hdd/projects/cidbn_wibral_neuro_nonhuman/SPP2205_mraabe/results/allen_open_scope/rf/waven/zebra/optimized/{session_name}")
            stem = f"optimized__{session_name}__{probe}__trial_{i_trial}__phase_{phase}"
            csv_out = outpath / (stem + ".csv")
            h5_out = outpath / (stem + ".h5")
            df = owp.optimize_parameters(results_dir=results_path, output_path=csv_out)
            df = owp.load_rf_maps(df)
            owp.save_rf_results(df, h5_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run waven pipeline for a given probe")
    # parser.add_argument('--probe', nargs='?', default='ProbeB', help='Probe name (e.g., ProbeB)')
    parser.add_argument('--nwb-path', required=True, help='Path to the NWB file (e.g., /data/sub-820454.nwb)')
    parser.add_argument('--results-dir', required=True, help='Root directory for saving results')
    parser.add_argument('--probe_idx', required=True, type=int, help='Index of the probe to process (0-based)')
    args = parser.parse_args()
    main(nwb_path=args.nwb_path, results_dir=args.results_dir, probe_idx=args.probe_idx)