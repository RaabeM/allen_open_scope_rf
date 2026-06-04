"""
Optimize waven pipeline parameters per unit.

Scans a directory tree of waven result .h5 files and, for each unit,
identifies the (delay, duration, theta, sigma, frequency) combination
that maximises the peak Pearson correlation.

Within each .h5 file the correlation matrix has already been reduced to
  abs_max_value[unit]          — peak |correlation| across x,y,θ,σ,f
  best_gabor_params_idx[:, u]  — argmax indices [xi, yi, θi, σi, fi]
  best_gabor_params_degree[:, u] — best params in visual degrees [x, y, σ, f]

So optimising over delay × duration amounts to picking the file with
the highest abs_max_value for each unit and reading out its stored best
gabor parameters.

Usage
-----
  python optimize_waven_parameters.py <results_dir> [--output path/to/out.csv]

  or import and call optimize_parameters() directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_lib_id(filepath: Path) -> str:
    """Extract library id from filename (e.g. lib_5d7b94947f__delay…)."""
    stem = filepath.stem
    for part in stem.split("__"):
        if part.startswith("lib_") or part.startswith("lib-"):
            return part.split("_", 1)[1].split("-", 1)[-1]
    return ""


def discover_files(results_dir: str | Path) -> list[Path]:
    """Return all .h5 files under results_dir that look like waven outputs."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.rglob("*.h5"))
    valid = []
    for f in files:
        try:
            with h5py.File(f, "r") as hf:
                if "abs_max_value" in hf and "unit_ids" in hf:
                    valid.append(f)
        except OSError:
            pass
    return valid


# ---------------------------------------------------------------------------
# core optimisation
# ---------------------------------------------------------------------------

def optimize_parameters(
    results_dir: str | Path,
    output_path: str | Path | None = None,
    all_values_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Scan all waven result files under results_dir and return a DataFrame
    with one row per unit containing the best parameters across all
    (delay, duration) combinations.

    Columns
    -------
    unit_id, delay, duration, lib_id,
    abs_max_value,
    x_deg, y_deg,           — RF centre in visual degrees
    theta_deg,              — orientation in Waven's degree convention
                              (0–225° for n_thetas=10; theta_rad is the
                               canonical radian value from attrs)
    sigma_deg,              — Gabor σ in visual degrees (2 * deg/pix * σ)
    theta_rad,              — orientation in radians (from attrs lookup)
    sigma,                  — Gabor σ in pixels (from attrs lookup)
    frequency,              — spatial frequency in cycles/pixel (from attrs)
    xi, yi, theta_idx, sigma_idx, frequency_idx,  — raw grid indices
    source_file             — which .h5 file gave the best result

    A secondary CSV with columns (unit_id, delay, duration, abs_max_value) is
    written to all_values_path (or <output_path stem>_all_values.csv when
    output_path is given and all_values_path is None).
    """
    files = discover_files(results_dir)
    if not files:
        raise FileNotFoundError(f"No valid waven .h5 files found under {results_dir}")

    print(f"Found {len(files)} result files — scanning for per-unit optima…")

    best: dict[str, dict] = {}
    all_rows: list[dict] = []

    for filepath in tqdm(files):
        lib_id = _parse_lib_id(filepath)
        try:
            with h5py.File(filepath, "r") as hf:
                delay = float(hf.attrs["delay"])
                duration = float(hf.attrs["duration"])
                thetas = hf.attrs["thetas"][:]
                sigmas = hf.attrs["sigmas"][:]
                frequencies = hf.attrs["frequencies"][:]

                unit_ids = hf["unit_ids"][:].astype(str)
                abs_max = hf["abs_max_value"][:].astype(float)
                best_idx = hf["best_gabor_params_idx"][:]    # (5, n_units)
                best_deg = hf["best_gabor_params_degree"][:] # (4, n_units)
        except OSError as e:
            print(f"WARNING: skipping {filepath}: {e}")
            continue

        for i, uid in enumerate(unit_ids):
            val = abs_max[i]
            all_rows.append({"unit_id": uid, "delay": delay, "duration": duration, "abs_max_value": val})
            if uid not in best or val > best[uid]["abs_max_value"]:
                xi, yi, ti, si, fi = best_idx[:, i]
                best[uid] = {
                    "unit_id": uid,
                    "delay": delay,
                    "duration": duration,
                    "lib_id": lib_id,
                    "abs_max_value": val,
                    "x_deg": best_deg[0, i],
                    "y_deg": best_deg[1, i],
                    "theta_deg": best_deg[2, i],
                    "sigma_deg": best_deg[3, i],
                    "xi": int(xi),
                    "yi": int(yi),
                    "theta_idx": int(ti),
                    "theta_rad": float(thetas[ti]),
                    "sigma_idx": int(si),
                    "sigma": float(sigmas[si]),
                    "frequency_idx": int(fi),
                    "frequency": float(frequencies[fi]),
                    "source_file": str(filepath),
                }

    df = pd.DataFrame(list(best.values()))
    df = df.sort_values("unit_id").reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} units → {output_path}")

    # resolve all_values_path: explicit > sibling of output_path > skip
    all_values_path = output_path.parent / (output_path.stem + "_all_values.csv")
    all_values_path.parent.mkdir(parents=True, exist_ok=True)
    df_all = (
        pd.DataFrame(all_rows)
        .sort_values(["unit_id", "delay", "duration"])
        .reset_index(drop=True)
    )
    df_all.to_csv(all_values_path, index=False)
    print(f"Saved {len(df_all)} (unit, delay, duration) rows → {all_values_path}")

    return output_path if output_path is not None else df


# ---------------------------------------------------------------------------
# RF map loader
# ---------------------------------------------------------------------------

def load_rf_maps(
    best_params: str | Path | pd.DataFrame
) -> pd.DataFrame:
    """
    Attach the 2D RF correlation map to each unit in best_params.

    For every unit the function reads the slice
        correlation_matrix[unit_row, :, :, theta_idx, sigma_idx, frequency_idx]
    from its source .h5 file, giving the spatial RF map (nx × ny) at the
    optimal orientation, scale, and frequency found by optimize_parameters.

    Parameters
    ----------
    best_params :
        Path to best_params.csv produced by optimize_parameters

    Returns
    -------
    DataFrame identical to the input with one extra column:
        rf_map  — list of (nx, ny) float32 arrays, one per unit
    """
    if isinstance(best_params, pd.DataFrame):
        df = best_params
    else:
        df = pd.read_csv(best_params)

    rf_maps = [None] * len(df)

    # with h5py.File(f'{best_params.parent/best_params.stem}.h5', "r") as hf:
    #     rf_maps = hf['rf_maps'][:]
    #     df["rf_map"] = list(rf_maps)

    # group by source file so each file is opened exactly once
    for source_file, grp in tqdm(df.groupby("source_file"), desc="Loading RF maps"):
        try:
            with h5py.File(source_file, "r") as hf:
                # build uid → row-index map for this file
                file_unit_ids = hf["unit_ids"][:].astype(str)
                uid_to_row = {uid: row for row, uid in enumerate(file_unit_ids)}
                cm = hf["correlation_matrix"]   # (n_units, nx, ny, n_theta, n_sigma, n_freq)

                for df_row, unit in grp.iterrows():
                    row_idx = uid_to_row.get(unit["unit_id"])
                    if row_idx is None:
                        continue
                    ti = int(unit["theta_idx"])
                    si = int(unit["sigma_idx"])
                    fi = int(unit["frequency_idx"])
                    rf_maps[df_row] = cm[row_idx, :, :, ti, si, fi]   # (nx, ny) float32

        except OSError as e:
            print(f"WARNING: could not open {source_file}: {e}")

    df["rf_map"] = rf_maps
    return df


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

_SCALAR_COLS = [
    "delay", "duration", "abs_max_value",
    "x_deg", "y_deg", "theta_deg", "sigma_deg",
    "theta_rad", "sigma", "frequency",
    "xi", "yi", "theta_idx", "sigma_idx", "frequency_idx",
]
_STRING_COLS = ["lib_id", "source_file"]


def save_rf_results(df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Save the DataFrame returned by load_rf_maps to a single HDF5 file.

    Layout
    ------
    /unit_ids   — (n_units,) unit identifier strings
    /rf_maps    — (n_units, nx, ny) float32, gzip-compressed
    /<col>      — one dataset per scalar / string column
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    maps = np.stack(df["rf_map"].values)   # (n_units, nx, ny)

    with h5py.File(output_path, "w") as hf:
        dt = h5py.string_dtype()
        hf.create_dataset("unit_ids", data=df["unit_id"].tolist(), dtype=dt)
        hf.create_dataset("rf_maps", data=maps, compression="gzip", compression_opts=4)

        for col in _SCALAR_COLS:
            if col in df.columns:
                hf.create_dataset(col, data=df[col].values)

        for col in _STRING_COLS:
            if col in df.columns:
                hf.create_dataset(col, data=df[col].tolist(), dtype=dt)

    print(f"Saved {len(df)} units → {output_path}")
    return output_path


def load_rf_results(path: str | Path) -> pd.DataFrame:
    """
    Load a file written by save_rf_results.

    Returns the same DataFrame structure as load_rf_maps, with
    an 'rf_map' column containing (nx, ny) float32 arrays.
    """
    path = Path(path)
    with h5py.File(path, "r") as hf:
        unit_ids = hf["unit_ids"][:].astype(str)
        maps = hf["rf_maps"][:]   # (n_units, nx, ny)

        data: dict = {"unit_id": unit_ids}
        for col in _SCALAR_COLS:
            if col in hf:
                data[col] = hf[col][:]
        for col in _STRING_COLS:
            if col in hf:
                data[col] = hf[col][:].astype(str)

    df = pd.DataFrame(data)
    df["rf_map"] = [maps[i] for i in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimise waven RF parameters per unit across delay × duration sweeps."
    )
    parser.add_argument(
        "results_dir",
        help="Root directory containing waven .h5 result files (searched recursively).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save results. Defaults to <results_dir>/best_params.h5.",
    )
    args = parser.parse_args()

    out = args.output or str(Path(args.results_dir) / "best_params.h5")

    df = optimize_parameters(args.results_dir, args.output)
    df = load_rf_maps(df)
    save_rf_results(df, out)

    print(f"\nSummary ({len(df)} units):")
    print(df[["unit_id", "delay", "duration", "abs_max_value",
              "theta_rad", "sigma", "frequency"]].describe())


if __name__ == "__main__":
    main()
