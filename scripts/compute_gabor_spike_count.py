import numpy as np
import pandas as pd
import neo
import os
import h5py
from tqdm import tqdm
import quantities as pq
from pathlib import Path

import sys
sys.path.append('..')

import utils
import argparse
from datetime import datetime
import multiprocessing
from functools import partial

PROBE = 'c'
DELAY = 100 *pq.ms
DURATION = 200 * pq.ms
RESULTS_DIR = f'../../results/allen_open_scope/rf/siegle/gabors/z-score_responses/{dandi_filepath[:-4]}/{DELAY.magnitude}/'

def get_stim_onsets_offsets(df_rf, combination_xy, orientation='0'):
    x, y  = combination_xy
    df_sub = df_rf[
        (df_rf['X'] == str(float(x))) &
        (df_rf['Y'] == str(float(y))) &
        (df_rf['Orientation'] == str(float(orientation)))
    ]
    return (df_sub['start_time'].values * pq.s, df_sub['stop_time'].values * pq.s)


def time_slice_spike_train(spike_train, start, stop):
    if (stop < spike_train.t_start) or ( start > spike_train.t_stop):
        return []
    else:
        if start < spike_train.t_start:
            start = spike_train.t_start
        if stop > spike_train.t_stop:
            stop = spike_train.t_stop
    
    return spike_train.time_slice(start, stop)


def index_to_angle(index, df_rf=df_rf):
    unique_x = np.unique(df_rf['X'])
    x_pos = np.sort(unique_x.astype(float))

    if isinstance(index, (list, np.ndarray)):
        return [x_pos[i] for i in index]
    else:
        return x_pos[index]



def compute_spike_count_gabors(spike_times,#df_units, 
                               df_rf, 
                               combinations_xy, 
                               unit_name, #unit_idxs, 
                               p_threshold=0.05,
                               duration=DURATION,
                               delay=DELAY,
                               results_dir=RESULTS_DIR, 
                               probe=None,
                               attributes=None,
                               ):
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    unique_x = np.unique(df_rf['X'])
    unique_y = np.unique(df_rf['Y'])
    x_pos = np.sort(unique_x.astype(float))
    y_pos = np.sort(unique_y.astype(float))
    unique_orientations = np.unique(df_rf['Orientation'])
    combinations_xy = [(x, y) for x in unique_x for y in unique_y]

    results = []
    for idx in tqdm(unit_idxs):

        unit_name = df_units.loc[idx, 'unit_name']

        spike_times = df_units.loc[idx, 'spike_times']
        spike_times = neo.SpikeTrain(spike_times, t_start=spike_times[0], t_stop=spike_times[-1], units=pq.s)

        for orientation in unique_orientations.astype(float):
            
            outpath = os.path.join(results_dir, f'gabors_spikecount_{unit_name}_orientation_{orientation}.h5')
            if os.path.exists(os.path.join(results_dir, f'gabors_spikecount_{unit_name}.h5')):
                continue

            R = np.zeros((5, len(x_pos), len(y_pos)))
            for (x,y) in combinations_xy:
                x_i = np.digitize(float(x), x_pos)-1
                y_i = np.digitize(float(y), y_pos)-1

                onsets, offsets = get_stim_onsets_offsets(df_rf, combination_xy=(x,y), orientation=orientation)
                for i, (onset, offset) in enumerate(zip(onsets, offsets)):
                    n_spikes = len(time_slice_spike_train(spike_times, onset+delay, onset+delay+duration))
                    R[i, x_i, y_i] = n_spikes / ((offset - onset - delay).rescale(pq.s).magnitude)
                
                
            E = np.mean(R)  # Grand mean
            O = np.mean(R, axis=0) # mean response grid
            chi_squared = np.sum((O - E)**2 / (E+1e-10))

            n_shuffle = 1000
            chi_squared_shuffled = np.zeros(n_shuffle)
            for i in range(n_shuffle):
                R_shuffled = np.random.permutation(R.flatten())
                R_shuffled = R_shuffled.reshape(R.shape)
                O_shuffled = np.mean(R_shuffled, axis=0)
                chi_squared_shuffled[i] = np.sum((O_shuffled - E)**2 / (E+1e-10))

            p_value = (np.sum(chi_squared_shuffled > chi_squared)+1 ) / (n_shuffle+1)

            std = np.std(R)
            z_score_response = (O - E) / (std+1e-10)  # add small value to avoid division by zero

            # results.append(pd.DataFrame({
            #                         'unit_name': [unit_name],
            #                         'file': [dandi_filepath],
            #                         'mean_response' : [E],
            #                         'z_score_response': [z_score_response],
            #                         'orientation': [orientation],
            #                         'p_value': [p_value],
            #                         'mean_rate': [np.mean(R)],
            #                         'probe': [probe],
            #                         })
            # )


            with h5py.File(outpath, 'a') as hf:
                # create or replace a group for this unit
                grp_name = unit_name
                if grp_name in hf:
                    del hf[grp_name]
                grp = hf.create_group(grp_name)

                # store frames and metadata under the unit group
                grp.create_dataset('file', data=dandi_filepath)
                grp.create_dataset('orientation', data=np.float32(orientation))
                grp.create_dataset('mean_rate', data=np.float32(E))
                grp.create_dataset('mean_response', data=np.float32(O))
                grp.create_dataset('z_score_response', data=np.float32(z_score_response))
                grp.create_dataset('p_value', data=np.float32(p_value))
                grp.create_dataset('probe', data=str(probe))


def compute_spike_count_gabors_parallel(spiketimes, onsets, ):
    df_units = df_units.copy()
    df_units['spike_times'] = [np.asarray(st) for st in df_units['spike_times']]
    df_rf = df_rf[['X', 'Y', 'Orientation', 'start_time', 'stop_time']].copy()

    worker = partial(_compute_unit, x_pos=x_pos, y_pos=y_pos, unique_orientations=unique_orientations,
                     df_units=df_units, df_rf=df_rf, dandi_filepath=dandi_filepath, combinations_xy=combinations_xy)

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(worker, unit_idx), total=len(unit_idx), desc='units'))

    all_rows = [row for rows in results for row in rows]
    df_p_values = pd.DataFrame(all_rows)

    if save_results:
        print(f'Saving results to {results_path}...')
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df_p_values.to_csv(results_path, index=False)
    else:
        return df_p_values




def main(probe_idx=0, nwb_path=None, results_dir=None):
    stream = utils.open_local(nwb_path)
    # nwb = stream.nwb

    units_df = stream.units_df()
    probes = units_df['probe'].unique()
    probe = probes[probe_idx]

    df_rf = stream.gabor_rf_df()

    # for probe in units_df['probe'].unique():
    unit_names = units_df.loc[units_df['probe'] == probe, 'unit_name'].values
    spiketimes = units_df.loc[units_df['probe'] == probe, 'spike_times'].values
    # unit_idx = units_df.loc[units_df['probe'] == probe].index.values

    unique_x = np.unique(df_rf['X'])
    unique_y = np.unique(df_rf['Y'])
    unique_orientations = np.unique(df_rf['Orientation'])

    x_pos = np.sort(unique_x.astype(float))
    y_pos = np.sort(unique_y.astype(float))
    combinations_xy = [(x, y) for x in unique_x for y in unique_y]


    results_dir = Path(results_dir) 
    nwb_path = Path(nwb_path)
    results_dir = results_dir / nwb_path.stem
    filename = f'{nwb_path.stem}__rf-spike-count__{probe}.csv'
    result_path = results_dir / filename

    compute_spike_count_gabors(x_pos, 
               y_pos, 
               unique_orientations, 
               units_df, 
               unit_idx, 
               df_rf, 
               nwb_path.stem, 
               combinations_xy, 
               result_path,
               save_results=True)
