import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from pathlib import Path
import matplotlib.pyplot as plt

from scripts.optimize_waven_parameters import load_rf_maps, load_rf_results
import utils



import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from scripts.waven_settings import *
from matplotlib.ticker import MultipleLocator
import Waven.WaveletGenerator as wg




def get_statistics_per_probe(probe:str, nwb_path, results_path,  rate_threshold=2): 

    # nwb_path = '../../../rawdata/allen_open_scope/sub-830794/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys.nwb'
    stream = utils.open_local(nwb_path)
    units_df = stream.units_df()
    units_df['unit_name'] = units_df['unit_name'].astype(str)

    zebra_df = stream.zebra_df()

    # RESULTS_PATH = Path('../../../results/allen_open_scope/rf/waven/zebra/optimized/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys')
    dfs = []
    for trial in [0,1]:
        for phase in [0, 1]:
            ov_file = list(results_path.glob(f'*{probe}__trial_{trial}__phase_{phase}.h5'))[0]
            tmp_df = load_rf_results(ov_file)
            tmp_df['trial'] = trial
            tmp_df['phase'] = phase
            dfs.append(tmp_df)
            
    ov_df = pd.concat(dfs, ignore_index=True)
    ov_df = ov_df.sort_values('abs_max_value', ascending=False)
    ov_df.reset_index(inplace=True, drop=True)


    zebra_df['TrialNumber'].unique()
    t_start = zebra_df.loc[zebra_df['TrialNumber']==  '59558.0', 'start_time'].values[0]
    t_stop  = zebra_df.loc[zebra_df['TrialNumber']=='59558.0', 'stop_time'].values[-1]
    dt = t_stop - t_start

    def rate(unit):
        spk = units_df.loc[units_df['unit_name']==unit, 'spike_times'].values[0]
        rate = (np.searchsorted(spk, t_stop) - np.searchsorted(spk, t_start))/dt
        return rate
    ov_df['rate'] = ov_df['unit_id'].apply(rate)
    ov_df = ov_df[ov_df['rate'] > rate_threshold]
    ov_df.reset_index(inplace=True, drop=True)



    dfs = []
    df_all = []
    for trial in [0,1]:
        for phase in [0, 1]:
            file = list(results_path.glob(f'*{probe}__trial_{trial}__phase_{phase}_all_values.csv'))[0]
            df = pd.read_csv(file)
            df['trial'] = trial
            df['phase'] = phase
            df_all.append(df)
            
            df_best = pd.DataFrame(df.groupby('unit_id')['abs_max_value'].idxmax()).reset_index()
            df_best['duration'] = [df.loc[idx, 'duration'] for idx in df_best['abs_max_value']]
            df_best['delay'] = [df.loc[idx, 'delay'] for idx in df_best['abs_max_value']]
            df_best['trial'] = trial
            df_best['phase'] = phase
            dfs.append(df_best)
    df_best = pd.concat(dfs, ignore_index=True)
    df_all = pd.concat(df_all, ignore_index=True)


    trial_phase_00 = ov_df.query('trial==0 and phase==0').sort_values('abs_max_value', ascending=False).reset_index(drop=True)

    return ov_df, df_best, df_all, trial_phase_00


def plot_optimized_results(probe:str, NY, nwb_path, results_path,  rate_threshold=2, path_printing=None): 

    ov_df, df_best, df_all, trial_phase_00 = get_statistics_per_probe(probe, nwb_path, results_path, rate_threshold)

    f, axs = plt.subplots(NY, 4, figsize=(NY*1.2, NY*2.5))

    # delays = np.arange(0.0, 0.35, 0.05)
    # durations = np.arange(0.03, 0.28, 0.05)

    trial_handles = [
        Line2D([0], [0], color='k', linestyle='-', lw=2, label='Trial 1'),
        Line2D([0], [0], color='k', linestyle='--', lw=2, label='Trial 2'),
    ]
    highest_abs_max = np.round(ov_df['abs_max_value'].max(),1)


    # PLOTTING 
    for idx in range(NY):
        ax = axs[idx]

        # Split axes[1] into a 2x2 grid
        inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax[0].get_subplotspec(), wspace=0.05, hspace=0.05)
        # ax[0].remove()  # remove the original axes
        ax_inner = np.array([[f.add_subplot(inner[0, 0]), f.add_subplot(inner[0, 1])],
                            [f.add_subplot(inner[1, 0]), f.add_subplot(inner[1, 1])]])
        # ax00 = f.add_subplot(inner[0, 0])
        # ax01 = f.add_subplot(inner[0, 1])
        # ax10 = f.add_subplot(inner[1, 0])
        # ax11 = f.add_subplot(inner[1, 1])
        unit = trial_phase_00.loc[idx, 'unit_id']
        for trial in [0,1]:
            for phase in [0,1]:
                a = ax_inner[trial, phase]
                map = ov_df.query(f"trial=={trial} and phase=={phase} and unit_id=='{unit}'")['rf_map'].values[0].T
                abs_max = ov_df.query(f"trial=={trial} and phase=={phase} and unit_id=='{unit}'")['abs_max_value'].values[0]
                a.imshow(map, cmap='coolwarm', vmin=-abs_max, vmax=abs_max)

                a.set_xticks([])
                a.set_yticks([])
                # rate = ov_df.loc[idx, 'rate']
                # ax[0].set_title(f"Abs. max correlation: {ov_df.loc[idx, 'abs_max_value']:.2f}")#\nRate: {rate:.1f} Hz")
        ax_inner[0,0].set_title("Phase 0", fontsize=12, color='b')
        ax_inner[0,1].set_title("Phase $\pi$/2", fontsize=12, color='r')
        ax_inner[0,0].set_ylabel("Trial 1", fontsize=12)
        ax_inner[1,0].set_ylabel("Trial 2", fontsize=12)
        ax[0].set_axis_off()


        inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax[1].get_subplotspec(), wspace=0.05, hspace=0.05)
        # ax[0].remove()  # remove the original axes
        ax_inner = np.array([[f.add_subplot(inner[0, 0]), f.add_subplot(inner[0, 1])],
                            [f.add_subplot(inner[1, 0]), f.add_subplot(inner[1, 1])]])
        lx = 67
        ly = 53

        for trial in [0,1]:
            for phasei, phase in enumerate([0,np.pi/2]):
                a = ax_inner[trial, phasei]

                theta = thetas[ov_df.query(f"trial=={trial} and phase=={phasei} and unit_id=='{unit}'")['theta_idx'].values[0]]
                sigma = sigmas[ov_df.query(f"trial=={trial} and phase=={phasei} and unit_id=='{unit}'")['sigma_idx'].values[0]]
                frequency = frequencies[ov_df.query(f"trial=={trial} and phase=={phasei} and unit_id=='{unit}'")['frequency_idx'].values[0]]
                best_filter = wg.makeGaborFilter(lx//2, #ov_df.loc[idx, 'xi'],
                                            ly//2, #ov_df.loc[idx, 'yi'], 
                                            theta, 
                                            sigma, 
                                            phase, 
                                            frequency, 
                                            lx, ly)
                a.imshow(best_filter, cmap='coolwarm', vmin=-best_filter.max(), vmax=best_filter.max())
                a.set_xticks([])
                a.set_yticks([])
                # rate = ov_df.loc[idx, 'rate']
                # ax[0].set_title(f"Abs. max correlation: {ov_df.loc[idx, 'abs_max_value']:.2f}")#\nRate: {rate:.1f} Hz")
        ax_inner[0,0].set_title("Phase 0", fontsize=12, color='b')
        ax_inner[0,1].set_title("Phase $\pi$/2", fontsize=12, color='r')
        ax_inner[0,0].set_ylabel("Trial 1", fontsize=12)
        ax_inner[1,0].set_ylabel("Trial 2", fontsize=12)
        ax[1].set_axis_off()




        for trial, linestyle in zip([0,1], ['-', '--']):
            for (phasei, phase), color in zip(enumerate([0,np.pi/2]), ['b', 'r']):
                delay = df_best.query(f"trial=={trial} and phase=={phasei} and unit_id=='{unit}'")['delay'].values[0]
                durations = df_all.query(f"trial=={trial} and phase=={phasei} and unit_id=='{unit}' and delay=={delay}")['duration'].values
                abs_max_values = df_all.query(f"trial=={trial} and phase=={phasei} and unit_id=='{unit}' and delay=={delay}")['abs_max_value'].values

                ax[2].plot(durations, abs_max_values, alpha=1, marker='o', linestyle=linestyle, color=color, label=f'Trial {trial+1}, phase {phase:.2f}')
                ax[2].set_xlabel('Duration (s)')
                ax[2].set_ylabel('Absolute max\ncorrelation')
                ax[2].set_title(f"Durations for best delay")
                ax[2].set_xticks(durations)
                ax[2].legend(handles=trial_handles, title='Trial',frameon=False)
                ax[2].set_yticks(np.arange(0, highest_abs_max, 0.1))
                ax[2].set_yticklabels(np.round(np.arange(0, highest_abs_max, 0.1),1))



        for trial, linestyle in zip([0,1], ['-', '--']):
            for (phasei, phase), color in zip(enumerate([0,np.pi/2]), ['b', 'r']):
                duration = df_best.query(f"trial=={trial} and phase=={phasei} and unit_id=='{unit}'")['duration'].values[0]
                delays = df_all.query(f"trial=={trial} and phase=={phasei} and unit_id=='{unit}' and duration=={duration}")['delay'].values
                abs_max_values = df_all.query(f"trial=={trial} and phase=={phasei} and unit_id=='{unit}' and duration=={duration}")['abs_max_value'].values

                ax[3].plot(delays, abs_max_values, alpha=1, marker='o', markersize=3, linestyle=linestyle, color=color, label=f'Trial {trial+1}, phase {phase:.2f}')
        ax[3].set_xlabel('Delay (s)')
        ax[3].set_ylabel('Absolute max\ncorrelation')
        ax[3].set_title(f"Delays for best duration")
        # ax[3].set_xticks(np.unique(delays), np.unique(delays), rotation=90)
        # ax[3].set_xticks(np.round(np.arange(0,.4,.05),2), np.round(np.arange(0,.4,.05),2), rotation=90)
        major_ticks = np.round(np.arange(0, .35, .05), 2)
        ax[3].set_xticks(major_ticks)
        ax[3].set_xticklabels(major_ticks, rotation=90)

        step = major_ticks[1] - major_ticks[0] if len(major_ticks) > 1 else 0.05
        ax[3].xaxis.set_minor_locator(MultipleLocator(step / 2))
        ax[3].tick_params(axis='x', which='minor', length=4)

        # print(ax[3].get_xticks())
        ax[3].legend(handles=trial_handles, title='Trial', frameon=False)
        ax[3].set_yticks(np.arange(0, highest_abs_max, 0.1))
        ax[3].set_yticklabels(np.round(np.arange(0, highest_abs_max, 0.1),1))

        ax[3].grid()

    #     duration = df_best[df_best['unit_id'] == unit]['duration'].values[0]
    #     delays = df.loc[(df['unit_id'] == unit) & (df['duration']==duration), 'delay'].values
    #     abs_max_values = df.loc[(df['unit_id'] == unit) & (df['duration']==duration), 'abs_max_value'].values
    #     ax[3].plot(delays, abs_max_values, alpha=1, color='b', marker='o')
    #     ax[3].set_xlabel('Delay (s)')
    #     ax[3].set_ylabel('Absolute max\ncorrelation')
    #     ax[3].set_title(f"Delays for best duration")
    #     ax[3].set_xticks(delays)

    axs[0,0].set_title("Receptive field\n\n", fontsize=12)
    axs[0,1].set_title("Best filter\n\n", fontsize=12)

    f.suptitle(probe, fontsize=30, y=1.02)
    f.tight_layout()


    if path_printing is not None:
        path_printing = Path(path_printing)
        path_printing.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_printing/f'{probe}_optimized_results.png', dpi=300, bbox_inches='tight')


def plot_avg_rf(probe, nwb_path, results_path, rate_threshold, f, ax):
    try:
        ov_df, df_best, df_all, trial_phase_00 = get_statistics_per_probe(probe, nwb_path, results_path, rate_threshold)
    except Exception as e:
        print(f"Error processing probe {probe}: {e}")
        ax.text(0.5, 0.5, f'{probe} failed', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return None

    ax.set_axis_off()
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec(), wspace=0.05, hspace=0.05)
    ax_inner = np.array([[f.add_subplot(inner[0, 0]), f.add_subplot(inner[0, 1])],
                            [f.add_subplot(inner[1, 0]), f.add_subplot(inner[1, 1])]])

    for phase in [0,1]:
        for trial in [0,1]:
            subset = ov_df.query(f'trial=={trial} and phase=={phase}')
            avg_rf = np.sum(subset['abs_max_value'].values * subset['rf_map'].values) / np.sum(subset['abs_max_value'].values)

            axi = ax_inner[trial, phase]
            axi.imshow(avg_rf.T, cmap='coolwarm', vmin=-avg_rf.max(), vmax=avg_rf.max())
            axi.set_xticks([])
            axi.set_yticks([])

    ax.set_title(probe, fontsize=12)
    ax_inner[0,0].set_title("Phase 0", fontsize=12, color='b')
    ax_inner[0,1].set_title("Phase $\pi$/2", fontsize=12, color='r')
    ax_inner[0,0].set_ylabel("Trial 1", fontsize=12)
    ax_inner[1,0].set_ylabel("Trial 2", fontsize=12)



    return ax_inner