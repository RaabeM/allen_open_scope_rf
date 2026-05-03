# allen_open_scope

Analysis pipeline applying the waven RF-mapping approach to Allen Open Scope
electrophysiology sessions. Data is downloaded from DANDI and processed with
the waven package (`../waven/`).

## What the pipeline does

Zebra noise stimulus (pink noise video) → wavelet decomposition via Gabor
filter library → Pearson cross-correlation with neural spike rates →
receptive field maps per unit.

Target scale: many sessions × ~10k unique units per session, run across
multiple Gabor library configurations.

## Source files

| File | Purpose |
|---|---|
| `waven_pipeline.py` | Core pipeline functions; main entry point |
| `waven_settings.py` | All analysis parameters (sigmas, nx/ny, coverage, paths) |
| `utils.py` | Path helpers and shared utilities |
| `compute_waven_pipeline.py` | Script to run the pipeline on a session |

## Result file layout

One HDF5 file per (session, library, delay, duration) combination:

```
results_path/
└── session_{session_id}/
    └── {session_id}__lib_{lib_id}__delay_{delay}__dur_{duration}.h5
```

The caller constructs the filename; `save_results_new` only creates the
directory and enforces the `.h5` suffix.

### HDF5 structure

```
/attrs:  sigmas, thetas, frequencies, delay, duration
/unit_ids               — (n_units,) unit identifiers
/correlation_matrix     — (n_units, nx, ny, n_thetas, ns, n_frequencies) gzip-compressed
/best_gabor_params_idx  — (5, n_units) best [x,y,theta,sigma,frequency] indices
/best_gabor_params_degree — (4, n_units) best params in visual degrees
/abs_max_value          — (n_units,) peak correlation per unit
```

## Key design decisions

- **One file per (session, library, delay, duration)** — different parameter
  sweeps never overwrite each other; most sessions will have one file
- **lib_id** is the 10-character SHA-1 hash of the library parameters,
  extracted from the library filename (computed by `create_gabor_library`)
- **Overwrite semantics**: existing file is deleted before writing — results
  on disk are always complete or absent, never partial
- **gzip compression on correlation_matrix** (level 4) — ~5–10× size
  reduction on the dominant dataset; small derived datasets left uncompressed

## Gabor library management

`create_gabor_library()` hashes the filter parameters (xs, ys, thetas,
sigmas, offsets, frequencies) and looks up an existing library in
`filter_registry.json` before recomputing. Libraries are stored in
`../waven/filter_libraries/` (gitignored, large .npy files).

## Parameters (waven_settings.py)

- `nx=107, ny=85` — downsampled stimulus grid size
- `n_thetas=4` — orientations (0°, 45°, 90°, 135°)
- `sigmas=[2,3,4,5,6,8]` — Gabor sigma values in pixels
- `frequencies=[0.015, 0.036, 0.072]` — spatial frequencies (cycles/pixel)
- `analysis_coverage=[-60, 60, 47.5, -47.5]` — visual field in degrees
- `movpath` — zebra noise stimulus video (.mp4)

## Environment

```bash
conda activate waven_dandi
```

Same environment as the waven package. Run from this directory so that
`waven_settings.py` and `utils.py` are on the path.
