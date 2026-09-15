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




def main(probe='ProbeB', nwb_path=None, results_dir=None):

    # delays = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    delays = np.arange(0.0, 0.35, 0.05)
    durations = np.arange(0.03, 0.28, 0.05)

    print('Loading Dandi NWB file...')
    stream = utils.open_local(nwb_path)
    nwb = stream.nwb
    # nwb = NWBHDF5IO(nwb_path, 'r').read()

    df_units = nwb.units.to_dataframe()
    unit_names = [unit['unit_name'] for i, unit in df_units.iterrows() if probe in unit['electrodes']['group_name'].unique()]
    spiketimes = [df_units.loc[df_units['unit_name']==name, 'spike_times'].values[0] for name in unit_names]
    df_zebra = nwb.intervals['Zebra_presentations'].to_dataframe()


    print(df_units.head())
    # df_merge = pd.merge(df_gabors, df_units.reset_index(), on='unit_name', how='inner')
    # df_merge = df_units[df_units['unit_name'].isin(df_gabors['unit_name'].unique())]
    # df_merge.reset_index(drop=True, inplace=True)

    # print('Getting spike times, frame onset/offset times, and unit names...')
    # spike_times_list = df_merge['spike_times'].values
    frame_onset_times = df_zebra['start_time'].values
    # # frame_offset_times = df_zebra['stop_time'].values
    # unit_names_list = df_merge['unit_name'].values

    # Get Gabor RF locations and compute corresponding pixel locations on the screen
    # df_rf = nwb.intervals['RF mapping_presentations'].to_dataframe()
    # x_gabors = np.sort(df_rf['X'].unique().astype(np.float32))
    # y_gabors = np.sort(df_rf['Y'].unique().astype(np.float32))
    # nx_px = 107
    # ny_px = 85
    # angl_y_zebra = 90
    # angl_x_zebra = 120
    # x_zebra_i = np.rint(x_gabors*nx_px/angl_x_zebra + nx_px/2)
    # y_zebra_i = np.rint(y_gabors*ny_px/angl_y_zebra + ny_px/2)
    # print(x_zebra_i)
    # print(y_zebra_i)
    
    # x_zebra_i_double = np.sort(np.concatenate([x_zebra_i, x_zebra_i[:-1] + np.diff(x_zebra_i)/2]))
    # y_zebra_i_double = np.sort(np.concatenate([y_zebra_i, y_zebra_i[:-1] + np.diff(y_zebra_i)/2]))

    # xis = np.arange(0, nx_px)
    # yis = np.arange(0, ny_px)

    # xis = np.arange(0, 105, 2)
    # yis = np.arange(0, 85, 2)


    attributes = {'session' : os.path.basename(nwb_path), 'probe' : probe, 'date_computed' : datetime.today().strftime('%Y-%m-%d')}

    # for delay in delays:
    #     for duration in tqdm(durations):
    for phase in ['0', '1']:
        full_pipeline(frame_onset_times,
                spiketimes,
                delays,
                durations,
                unit_names,
            #   xis=xis,
            #   yis=yis,
                results_path=os.path.join(results_dir, probe),
                results_filename='',
                attributes=attributes,
                recompute=False,
                phase=phase,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run waven pipeline for a given probe")
    parser.add_argument('--probe', nargs='?', default='ProbeB', help='Probe name (e.g., ProbeB)')
    parser.add_argument('--nwb-path', required=True, help='Path to the NWB file (e.g., /data/sub-820454.nwb)')
    parser.add_argument('--results-dir', required=True, help='Root directory for saving results (e.g., /results/waven/)')
    args = parser.parse_args()
    main(probe=args.probe, nwb_path=args.nwb_path, results_dir=args.results_dir)