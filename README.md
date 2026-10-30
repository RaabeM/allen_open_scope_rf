# [Allen Open Scope] Receptive Field Mapping

Standalone package of the receptive field mapping for the [Allen Open Scope P3 Data Release paper](https://github.com/AllenNeuralDynamics/openscope_p3_data_release_paper). It supports analyzsing all three recording modalities - **ecephys** (spike times), **mesoscope** (dF/F or deconvolved events) and **SLAP2** (dF/F or deconvolved events, green and red channel) - ,and Gabor and Zebra stimulus. 


## Setup

### 1. Install the environment

```bash
conda env create -f envs/waven_dandi.yml
conda activate waven_dandi
```

This installs the `waven` package from the `allen_open_scope` branch of [github.com/RaabeM/waven](https://github.com/RaabeM/waven).

### 2. Configure paths and the filter bank

All analysis parameters live in `scripts/waven_settings.py`; its docstring explains each one. Set at least:

```python
movpath = '/path/to/zebra_allen_screen_tscale_30_scale_10.mp4'
library_path = '../../waven/filter_libraries/'
```

- `movpath`: the Zebra stimulus, available from the [Open Scope repository](https://github.com/AllenNeuralDynamics/openscope-community-predictive-processing/blob/main/code/stimulus-control/src/Mindscope/movies/zebra_allen_screen_tscale_30_scale_10.mp4).
- `library_path`: where the Gabor filter libraries are stored.


## Running the pipeline

Run the scripts from `scripts/`. Each modality has its own driver; each streams the session from DANDI by default (`--local` reads a file from disk instead):

```bash
python compute_waven_pipeline_ecephys.py   --nwb-path <asset path> --results-dir <root> --probe-idx 0
python compute_waven_pipeline_mesoscope.py --nwb-path <asset path> --results-dir <root> --plane-idx 4 --signal events
python compute_waven_pipeline_slap2.py     --nwb-path <asset path> --results-dir <root> --dmd DMD1 --channel green
```

All three share `--optimized-dir`, `--phases`, `--per-trial`, `--recompute`, `--no-optimize`, `--dandiset-id` and `--local`; see `--help`. Zebra repeats are **pooled** into one map by default; `--per-trial` gives one map per repeat.

**Memory:** one phase of the wavelet decomposition (~54 GB at the current settings) is loaded whole, and the correlation adds a copy of similar size, so plan for 300–350 GB.

### Result layout

```
<results-dir>/<session>/<signal>/<group>/<pooled_trials|trial_<i>>/phase_<p>/lib_<id>/
    lib_<id>__delay_<delay>__dur_<duration>.h5      one file per (delay, duration)
<optimized-dir>/<session>/<signal>/
    optimized__<session>__<group>__<pooled_trials|trial_<i>>__phase_<p>.{csv,h5}
```

`<signal>` is `spikes` for ecephys and `events` or `dff` for the imaging modalities; `<group>` is the probe, the imaging plane, or `<DMD>/<channel>`.

### Per-(delay, duration) file

```
/attrs:  session, modality, signal, group, phase, delay, duration, library_id,
         sigmas, thetas, frequencies, movie, pooled_repeats, n_repeats, trial, ...
/unit_ids                 — (n_units,) unit identifiers
/correlation_matrix       — (n_units, nx, ny, n_thetas, n_sigmas, n_frequencies)  float16, gzip-4
/best_gabor_params_idx    — (5, n_units) best [x, y, theta, sigma, frequency] indices
/best_gabor_params_degree — (4, n_units) best [x, y, theta, sigma] in visual degrees
/abs_max_value            — (n_units,) peak |Pearson r|
```

### Optimized file (one row per unit)

Each driver reduces its delay × duration grid with `optimize_waven_parameters.py` (also usable on its own: `python optimize_waven_parameters.py <lib_<id> folder> --output best.csv`):

```
/unit_ids      — (n_units,) unit identifier strings
/rf_maps       — (n_units, nx, ny)  RF map at the best filter
/delay, /duration, /abs_max_value
/x_deg, /y_deg — RF centre in visual degrees
/theta_deg, /theta_rad — orientation
/sigma_deg, /sigma     — Gabor size in visual degrees / pixels
/frequency     — spatial frequency in cycles/pixel
/xi, /yi, /theta_idx, /sigma_idx, /frequency_idx — grid indices of the best filter
/lib_id, /source_file
```

## Significance

`compute_null_distribution_<modality>.py` builds a per-unit null of the sweep-wide maximum from circularly rolled responses (same pooling and grid as the driver), and `filter_significant_maps.py` turns observed maxima and nulls into per-unit p-values with Benjamini–Hochberg across units.

## Local Gabor RFs

`compute_rf_gabors_{ecephys,mesoscope,slap2}.py` map receptive fields from the local Gabor patches: the response to each presentation (spike rate, or the mean ΔF/F / event amplitude over a window) is tested for spatial structure with a permutation test, once per (delay, duration), and reduced to the best window per unit and orientation. The delay x duration grid and the window rule of each modality are those of its Zebra analysis (imported from `compute_waven_pipeline_<modality>.py`), so both stimuli are tested over the same windows. The shared code is `rf_gabors.py`. Like the waven drivers they stream from DANDI (`--local` reads from disk), pool the whole RF block by default (`--per-trial` scores its two halves at the pooled run's windows), skip existing files and refuse files computed on another unit list (`--recompute` replaces them). Results go to `<results-path>/<session>/<signal>/` below a modality root, as for the Zebra results: `gabors/<ecephys|mesoscope|slap2>/<session>/<spikes|dff|events>/`.

`compute_gabor_unit_significance.py` adds a unit-level p-value after the full search (`p_value_gumbel`, a Gumbel fit to the studentized maximum over windows and orientations, see `gabor_significance.py`) and its BH q-value (`p_value_bh`) to the pooled optimized files in `--results-dir` (default `results/gabors/`), with the BH families of `filter_significant_maps.py`.

## Data-release figures

```bash
python plotting/plot_data_release_panels.py
```

Example sessions: ecephys 830794, mesoscope 832700, SLAP2 829704. Figures go to `plots/data_release__*.png`. Everything is read from `results/`, which is not part of this repository, so compute these inputs first:

1. **Zebra maps**, pooled: `results/zebra/optimized/<modality>/<session>/<signal>/optimized__<session>__<group>__pooled_trials__phase_0.h5`, from the waven drivers. The signal per modality is `ZEBRA_SIGNAL` in the script (ecephys `spikes`, mesoscope `events`, SLAP2 `dff`).
2. **Zebra nulls** of the same signal, block and library: `results/zebra/nullresults/...`, from `compute_null_distribution_<modality>.py`. The script checks that each null's lag-0 maximum reproduces the map's `abs_max_value`.
3. **Local Gabor maps**, pooled: `results/gabors/<modality>/<session>/<signal>/*__optimized.h5` (`spikes` for ecephys, `events` for imaging), from `compute_rf_gabors_<modality>.py`.
4. **Gabor unit significance**: `compute_gabor_unit_significance.py`. Needed for the overlap, pie and Gaussian-summary panels.
5. **Ephys areas**: `export_ephys_unit_areas.py` writes `results/ecephys/<session>/unit_areas.csv`, used to keep visual cortex layer 2/3.

`--trial 0` or `--trial 1` plots one Zebra repeat instead of the pooled maps (the `trial_<i>` maps and nulls from `--per-trial` runs); those figures are named `data_release__zebra-trial-<i>__*.png`. The local Gabor maps stay pooled.

A figure whose inputs are missing or do not fit together is skipped and listed at the end of the run.
