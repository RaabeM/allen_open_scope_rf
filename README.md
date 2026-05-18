# allen_open_scope — Wavelet RF Mapping

Receptive field mapping pipeline applied to [Allen Open Scope](https://dandiarchive.org/dandiset/001637) electrophysiology data. Zebra noise (pink noise video) stimuli are decomposed by a Gabor wavelet filter bank, and the resulting wavelet responses are cross-correlated with single-unit spike rates to estimate receptive field maps.

## Method overview

```
Zebra noise video (.mp4)
  → Gabor filter library (waven)
  → Wavelet decomposition per frame
  → Pearson correlation with signal (here: spike rates)
  → Receptive field map per unit (.h5)
```

## Setup

### 1. Install the environment

```bash
conda env create -f envs/waven_dandi.yml
conda activate waven_dandi
```

This installs the `waven` package directly from the `develop` branch of [github.com/RaabeM/waven](https://github.com/RaabeM/waven).

### 2. Configure paths

Open `waven_settings.py` and update the two path entries to point to your data:

```python
"Movie Path": "/path/to/zebra_allen_screen_tscale_30_scale_10.mp4",
"Library Path": "/path/to/filter_libraries/",
```


## Running the pipeline

### Wavelet RF mapping (main pipeline)

```bash
python compute_waven_pipeline.py ProbeB \***
    --nwb-path /data/sub-820454.nwb \
    --results-dir /results/waven/
```

- `probe` (positional, default `ProbeB`): probe to process
- `--nwb-path`: path to the NWB file
- `--results-dir`: root directory for output HDF5 files

Results are saved to `<results-dir>/<probe>/lib_<hash>/` as one `.h5` file per (delay, duration) combination.

**Caution:** Under the current settings loading the filter libraries take **256 GB of RAM!**

### Optimize parameters

After running the pipeline across the full delay × duration grid:

```bash
python optimize_waven_parameters.py /results/waven/ProbeB/lib_<hash>/ \
    --output /results/waven/ProbeB/lib_<hash>/best_params.h5
```


## Output format

Each result file is an HDF5 with the following structure:

```
/attrs:  sigmas, thetas, frequencies, delay, duration, session, probe
/unit_ids                — (n_units,) unit identifiers
/correlation_matrix      — (n_units, nx, ny, n_thetas, ns, n_freqs)  gzip-4
/best_gabor_params_idx   — (5, n_units)
/best_gabor_params_degree — (4, n_units)
/abs_max_value           — (n_units,)
```


## Dependencies

See `envs/waven_dandi.yml`. Key packages: PyTorch, PyNWB, DANDI, NumPy, SciPy, Numba, Elephant/Neo, h5py.
