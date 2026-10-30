"""Data-release RF panels for one example session of each modality.

Sessions match ``compare_zebra_gabor_rfs.ipynb``:

* ecephys 830794
* mesoscope 832700
* SLAP2 829704, green and red kept separate (different indicators)

For each session, the top 3 units are plotted twice: once ranked by Zebra
peak correlation (``abs_max_value``, the notebook's Selection A), and once
ranked by local-Gabor significance (lowest unit-level ``p_value_gumbel``, or
the per-orientation ``p_value`` where that is missing, ties broken by peak
|z|, the notebook's Selection B). Each ranking writes two PNGs with the same
unit order: the three Zebra maps, then the three Gabor maps.

Ephys units are restricted to visual cortex layer 2/3 using the saved
extremum-channel table from ``export_ephys_unit_areas.py``
(``results/ecephys/<session>/unit_areas.csv``). Mesoscope units are restricted
to plane ``VISp_0``. SLAP2 ROIs are all kept: that recording is already V1
layer 2/3.

Zebra maps are the **pooled** results (both Zebra repeats in one map), phase 0:

    results/zebra/optimized/<modality>/<session>/<signal>/
        optimized__<session>__<group>__pooled_trials__phase_0.h5

Zebra units are then limited to those that pass the null test after
Benjamini–Hochberg correction at q = 0.05. Only null files of the same signal,
pooled block, phase and filter library as the plotted maps are used, and their
lag-0 maximum must reproduce the maps' ``abs_max_value``. The p-value is the
Gumbel tail probability of ``filter_significant_maps.py``, and the BH families
are the same as there: the whole session for ecephys and mesoscope, each
DMD x channel on its own for SLAP2. The Zebra signal of each modality is
``ZEBRA_SIGNAL``; the local Gabor maps use ``GABOR_SIGNAL``.

A paired figure (Gabor above, pooled Zebra below, one shared colorbar per
row) is written for ephys VISp layer 2/3, mesoscope VISp_0, and SLAP2 green.

Inputs, all in the repository's ``results/`` (see ``results/README.md``):
optimized and null Zebra results, the pooled Gabor ``*__optimized.h5`` in
``results/gabors/<modality>/<session>/<signal>/`` with ``p_value_gumbel`` /
``p_value_bh`` (``scripts/compute_gabor_unit_significance.py``)
and the ephys unit areas (``scripts/export_ephys_unit_areas.py``).

    python plotting/plot_data_release_panels.py              # pooled Zebra maps
    python plotting/plot_data_release_panels.py --trial 0    # first Zebra repeat only

``--trial 0`` / ``--trial 1`` swap the pooled Zebra maps and nulls for those of
one repeat (``trial_<i>`` instead of ``pooled_trials`` in the paths above) and
prefix the figure names with ``zebra-trial-<i>``. The local Gabor maps stay pooled.
"""
from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))
sys.path.append(str(REPO / "scripts"))

from analysis.gaussian_envelope import fit_gaussian_envelope
from filter_significant_maps import bh_fdr as _bh_fdr, gumbel_mom, load_null
from optimize_waven_parameters import load_rf_results
from waven_settings import analysis_coverage

# The repository's results/. Layout is documented in results/README.md.
RESULTS = REPO / "results"
RF_ROOT = RESULTS
PLOT_DIR = REPO / "plots"
FILE_PREFIX = "data_release"

N_TOP = 3
PHASE = 0
# Zebra maps and their nulls: the pooled block of both repeats.
ZEBRA_BLOCK = "pooled_trials"


def set_zebra_block(trial: str | int = "pooled") -> None:
    """Choose the Zebra maps: ``"pooled"`` (both repeats) or one repeat, 0 or 1.

    A single repeat reads the ``trial_<i>`` maps and nulls (``--per-trial`` runs
    of the waven drivers and null scripts). The local Gabor maps stay pooled.
    """
    global ZEBRA_BLOCK
    ZEBRA_BLOCK = "pooled_trials" if str(trial) == "pooled" else f"trial_{int(trial)}"
    _Q_CACHE.clear()


def zebra_label() -> str:
    """``Zebra`` for the pooled maps, ``Zebra trial <i>`` for one repeat."""
    return "Zebra" if ZEBRA_BLOCK == "pooled_trials" else f"Zebra {ZEBRA_BLOCK.replace('_', ' ')}"
GABOR_SIGNAL = "events"
# Signal of the Zebra maps and nulls per modality. Maps and nulls must both
# exist for it: results/zebra/{optimized,nullresults}/<modality>/<session>/<signal>/.
ZEBRA_SIGNAL = {"ecephys": "spikes", "mesoscope": "events", "slap2": "dff"}
# The lag-0 maximum of a null file and the plotted map's abs_max_value must agree
# to this, or the two were not computed on the same data (float16 maps).
CHECK_TOL = 1e-3
Q_LEVEL = 0.05
FONTSIZE = 14
TITLE_FONTSIZE = FONTSIZE + 1
PAIRED_TITLE_SIZE = 22
PAIRED_LABEL_SIZE = 27
PAIRED_TICK_SIZE = 22.5
# map size of the paired panels (example summaries), inches per visual degree
PAIRED_INCH_PER_DEG = 0.038
SAVE_DPI = 200

xM, xm, yM, ym = analysis_coverage
WAVEN_EXTENT = (xM, xm, ym, yM)

# One example session per modality, as in the notebook's SESSIONS / UNIT_LABEL.
# label -> (modality, session, channel kept or None)
SESSIONS = {
    "ecephys 830794": (
        "ecephys",
        "sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys",
        None,
    ),
    "mesoscope 832700": (
        "mesoscope",
        "sub-832700_ses-multiplane-ophys-832700-2026-01-24-12-06-12_ophys",
        None,
    ),
    "slap2 829704 green": (
        "slap2",
        "sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys",
        "green",
    ),
    "slap2 829704 red": (
        "slap2",
        "sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys",
        "red",
    ),
}

# Saved by export_ephys_unit_areas.py. Join on unit_id; do not stream the NWB.
EPHYS_SESSION = "sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys"
UNIT_AREAS_CSV = RESULTS / "ecephys" / EPHYS_SESSION / "unit_areas.csv"

WAVEN_COLS = [
    "unit_id", "abs_max_value", "rf_map", "delay", "duration",
    "x_deg", "y_deg", "theta_rad", "theta_deg", "sigma_deg", "frequency", "lib_id",
]
GABOR_COLS = [
    "unit_id", "p_value", "p_value_gumbel", "p_value_bh", "z_max", "ori_idx", "ori_rad", "delay",
    "duration", "rf_map", "rf_maps_all", "p_all", "rate_map",
]


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def in_filter(group: str, keep: str | None) -> bool:
    return keep is None or group.split("__")[-1] == keep


def canonical_unit_id(uid) -> str:
    m = re.match(r"^.*_roi(\d+)$", str(uid))
    return m.group(1) if m else str(uid)


def _gabor_dir(modality: str, session: str) -> Path:
    """``results/gabors/<modality>/<session>/<signal>``: ``spikes`` for ecephys, else ``GABOR_SIGNAL``."""
    signal = "spikes" if modality == "ecephys" else GABOR_SIGNAL
    return RESULTS / "gabors" / modality / session / signal


def _zebra_dir(kind: str, session: str, signal: str) -> Path | None:
    """``results/zebra/<kind>/<modality>/<session>/<signal>``, or None if absent."""
    hits = sorted(
        p / signal for p in (RESULTS / "zebra" / kind).glob(f"*/{session}")
        if (p / signal).is_dir()
    )
    if len(hits) > 1:
        raise ValueError(f"{session}/{signal} is under several modality folders of {kind}: {hits}")
    return hits[0] if hits else None


def discover_waven(session: str, signal: str) -> dict[str, Path]:
    """Pooled, phase-``PHASE`` optimized Zebra files of one session and signal, by group."""
    out = {}
    d = _zebra_dir("optimized", session, signal)
    if d is None:
        return out
    prefix = f"optimized__{session}__"
    suffix = f"__{ZEBRA_BLOCK}__phase_{PHASE}"
    for f in sorted(d.glob(f"{prefix}*{suffix}.h5")):
        out[f.stem[len(prefix):-len(suffix)]] = f
    return out


def discover_gabor(modality: str, session: str) -> dict[str, Path]:
    """Pooled Gabor ``*__optimized.h5`` of one session, by group."""
    out = {}
    d = _gabor_dir(modality, session)
    if not d.is_dir():
        print(f"  no Gabor results at {d}")
        return out
    for f in sorted(d.glob(f"{session}__rf-spike-count__*__optimized.h5")):
        if "__trial_" in f.stem:
            continue
        out[f.stem.split("__")[-2]] = f

    channel_prefix = f"rf-{GABOR_SIGNAL}-"
    for f in sorted(d.glob(f"{session}__*__optimized.h5")):
        if "__trial_" in f.stem:
            continue
        parts = f.stem.split("__")
        if len(parts) < 4:
            continue
        tag, group = parts[-3], parts[-2]
        if tag.startswith(channel_prefix):
            group = f"{group}__{tag[len(channel_prefix):]}"
        out[group] = f
    return out


def load_waven(path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=WAVEN_COLS)
    df = load_rf_results(path)
    if "rf_map" not in df.columns:
        print(f"  {Path(path).name}: no rf_maps stored — skipping")
        return pd.DataFrame(columns=WAVEN_COLS)
    df = df[[c for c in WAVEN_COLS if c in df.columns]].copy()
    df["unit_id"] = df["unit_id"].map(canonical_unit_id)
    return df


def _extent_from_centres(x_pos, y_pos):
    dx = np.median(np.diff(x_pos)) if len(x_pos) > 1 else 1.0
    dy = np.median(np.diff(y_pos)) if len(y_pos) > 1 else 1.0
    return (
        x_pos[0] - dx / 2, x_pos[-1] + dx / 2,
        y_pos[0] - dy / 2, y_pos[-1] + dy / 2,
    )


def load_gabor(path) -> pd.DataFrame:
    """Best orientation per unit: lowest p, then largest peak |z| on a tie."""
    if path is None or not Path(path).exists():
        empty = pd.DataFrame(columns=GABOR_COLS)
        empty.attrs["extent"] = (-45, 45, -45, 45)
        return empty

    with h5py.File(path, "r") as hf:
        unit_names = hf["unit_names"][:].astype(str)
        p_values = hf["p_value"][:]
        z_scores = hf["z_score_response"][:]
        # Mean response per position in the best window, laid out like the z-map:
        # Hz for spikes, the mean trace value for imaging.
        mean_response = hf["mean_response"][:]
        orientations = hf["orientations"][:]
        x_pos, y_pos = hf["x_positions"][:], hf["y_positions"][:]
        if "p_value_bh" in hf and hf["p_value_bh"].shape == (len(unit_names),):
            p_value_bh = np.asarray(hf["p_value_bh"][:], dtype=float)
        else:
            p_value_bh = np.full(len(unit_names), np.nan)
        if "p_value_gumbel" in hf and hf["p_value_gumbel"].shape == (len(unit_names),):
            p_value_gumbel = np.asarray(hf["p_value_gumbel"][:], dtype=float)
        else:
            p_value_gumbel = np.full(len(unit_names), np.nan)

    z_peak = np.nanmax(np.abs(z_scores), axis=(2, 3))
    with np.errstate(invalid="ignore"):
        at_min = p_values <= np.nanmin(p_values, axis=1, keepdims=True) + 1e-12
    best_ori = np.argmax(np.where(at_min, z_peak, -np.inf), axis=1)
    u = np.arange(len(unit_names))

    df = pd.DataFrame({
        "unit_id": [canonical_unit_id(x) for x in unit_names],
        "p_value": p_values[u, best_ori],
        "p_value_gumbel": p_value_gumbel,
        "p_value_bh": p_value_bh,
        "ori_idx": best_ori,
        "ori_rad": orientations[best_ori],
    })
    df["rf_map"] = [z_scores[i, best_ori[i]] for i in u]
    df["rate_map"] = [np.asarray(mean_response[i, best_ori[i]], dtype=float) for i in u]
    df["z_max"] = z_peak[u, best_ori]
    with np.errstate(invalid="ignore"):
        df["z_max_all"] = np.nanmax(z_peak, axis=1)
    df.attrs["extent"] = _extent_from_centres(x_pos, y_pos)
    df.attrs["x_positions"] = np.asarray(x_pos, dtype=float)
    df.attrs["y_positions"] = np.asarray(y_pos, dtype=float)
    df.attrs["orientations"] = tuple(float(o) for o in orientations)
    return df


def load_session(modality: str, session: str, keep: str | None) -> pd.DataFrame:
    """Pooled Zebra maps (``ZEBRA_SIGNAL``) and Gabor maps of one session, merged per unit."""
    signal = ZEBRA_SIGNAL[modality]
    waven_files, gabor_files = discover_waven(session, signal), discover_gabor(modality, session)
    if not waven_files:
        print(f"  no {ZEBRA_BLOCK} phase-{PHASE} Zebra maps for {session}/{signal}")
    frames, extent, orientations = [], None, ()
    x_positions = y_positions = None
    for group in sorted(
        g for g in set(waven_files) | set(gabor_files) if in_filter(g, keep)
    ):
        w = load_waven(waven_files.get(group))
        s = load_gabor(gabor_files.get(group))
        if extent is None and len(s):
            extent = s.attrs["extent"]
            orientations = s.attrs.get("orientations", ())
            x_positions = s.attrs.get("x_positions")
            y_positions = s.attrs.get("y_positions")
        merged = w.merge(s, on="unit_id", how="outer", suffixes=("_zebra", "_gabor"))
        merged.insert(0, "group", group)
        if len(merged):
            frames.append(merged)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["group", "unit_id", "abs_max_value", "rf_map_zebra", "p_value", "rf_map_gabor"]
    )
    duplicated = df.duplicated(["group", "unit_id"])
    if duplicated.any():
        raise ValueError(
            f"{session}: {int(duplicated.sum())} (group, unit) pairs occur twice, "
            f"e.g. {df.loc[duplicated, ['group', 'unit_id']].head(3).values.tolist()}"
        )
    df.attrs["extent"] = extent or (-45, 45, -45, 45)
    df.attrs["orientations"] = tuple(float(o) for o in orientations)
    if x_positions is not None:
        df.attrs["x_positions"] = np.asarray(x_positions, dtype=float)
        df.attrs["y_positions"] = np.asarray(y_positions, dtype=float)
    return df


def load_ephys_unit_areas() -> pd.DataFrame:
    """Unit id, probe, area, and layer from the saved table."""
    if not UNIT_AREAS_CSV.is_file():
        raise FileNotFoundError(
            f"No unit-area table at {UNIT_AREAS_CSV}. "
            "Run export_ephys_unit_areas.py once to build it."
        )
    anatomy = pd.read_csv(UNIT_AREAS_CSV, dtype=str, keep_default_na=False)
    anatomy["unit_id"] = anatomy["unit_id"].map(_text)
    anatomy["probe"] = anatomy["probe"].map(_text)
    return anatomy


def ephys_vis_layer23(df: pd.DataFrame) -> pd.DataFrame:
    """Keep visual-cortex layer 2/3 units, labeled from the saved area table."""
    anatomy = load_ephys_unit_areas()
    print(f"Unit areas from {UNIT_AREAS_CSV.name}")
    n_match = int(df["unit_id"].isin(set(anatomy["unit_id"])).sum())
    print(f"  unit-id overlap with saved areas: {n_match} / {len(df)}")
    keep = (
        anatomy["area"].astype(str).str.startswith("VIS")
        & (anatomy["layer"].astype(str) == "2/3")
    )
    anatomy = anatomy.loc[keep, ["probe", "unit_id", "area", "layer", "structure"]]
    print(f"  visual cortex layer 2/3: {len(anatomy)} units")
    out = df.merge(anatomy.drop(columns=["probe"]), on="unit_id", how="inner")
    print(f"  visual cortex layer 2/3 with a zebra map: {out['rf_map_zebra'].map(lambda m: isinstance(m, np.ndarray)).sum() if len(out) else 0}")
    return out


def restrict_population(label: str, df: pd.DataFrame) -> pd.DataFrame:
    if label.startswith("ecephys"):
        return ephys_vis_layer23(df)
    if label.startswith("mesoscope"):
        out = df[df["group"] == "VISp_0"].copy()
        print(f"  VISp_0: {len(out)} paired units")
        return out
    print(f"  all SLAP2 ROIs (V1 layer 2/3): {len(df)} paired units")
    return df


def _null_files(modality: str, session: str, lib_ids) -> list[tuple[str, Path]]:
    """Null files of the plotted maps as ``(group, path)``.

    Same signal, pooled block, phase and filter library as the maps. The group
    is the path between the signal and the block: ``ProbeA``, ``VISp_0`` or
    ``DMD1__green``.
    """
    signal = ZEBRA_SIGNAL[modality]
    base = _zebra_dir("nullresults", session, signal)
    if base is None:
        raise FileNotFoundError(
            f"No null results for {session}/{signal} under {RESULTS / 'zebra' / 'nullresults'}"
        )
    found = []
    for path in sorted(base.rglob("null__lib_*.h5")):
        rel = path.relative_to(base).parts      # (*group, block, phase, lib_<id>, file)
        if len(rel) < 5 or rel[-4] != ZEBRA_BLOCK or rel[-3] != f"phase_{PHASE}":
            continue
        if rel[-2].removeprefix("lib_") not in lib_ids:
            continue
        found.append(("__".join(rel[:-4]), path))
    return found


def _unit_null_rows(group: str, path: Path) -> pd.DataFrame:
    """Gumbel p-value of every tested unit of one null file, as filter_significant_maps.py."""
    null = load_null(path)
    mu, beta = gumbel_mom(null["M_sur"])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p_values = -np.expm1(-np.exp(-(null["max_obs_null"] - mu) / beta))
    p_values = np.where(np.isfinite(p_values), p_values, 1.0)
    return pd.DataFrame({
        "group": group,
        "unit_id": [canonical_unit_id(unit_id) for unit_id in null["unit_ids"]],
        "max_obs_null": null["max_obs_null"],
        "p_null": p_values,
    })


_Q_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
# Inputs a figure step was skipped for: missing results (FileNotFoundError) or
# results that do not fit together (ValueError). Listed at the end of main().
SKIPPED: list[str] = []
SKIPPABLE = (FileNotFoundError, ValueError)


def _skip(what: str, exc: Exception) -> None:
    print(f"  skipping {what}: {exc}")
    SKIPPED.append(f"{what}: {exc}")


def zebra_q_table(modality: str, session: str) -> pd.DataFrame:
    """Every tested unit, with its Gumbel p-value and BH q-value.

    The nulls are matched to the plotted maps by signal, block, phase and
    library, and their lag-0 maximum must reproduce the maps' abs_max_value.
    BH families as in filter_significant_maps.py: the session for ecephys and
    mesoscope, each group (DMD x channel) for SLAP2.
    """
    key = (modality, session)
    if key in _Q_CACHE:
        return _Q_CACHE[key]
    signal = ZEBRA_SIGNAL[modality]
    maps = discover_waven(session, signal)
    if not maps:
        raise FileNotFoundError(f"No {ZEBRA_BLOCK} phase-{PHASE} Zebra maps for {session}/{signal}")
    libraries, observed = set(), {}
    for group, path in maps.items():
        with h5py.File(path, "r") as hf:
            libraries.update(_text(value) for value in hf["lib_id"][:])
            ids = [canonical_unit_id(_text(value)) for value in hf["unit_ids"][:]]
            observed.update(((group, uid), float(v)) for uid, v in zip(ids, hf["abs_max_value"][:]))
    files = _null_files(modality, session, libraries)
    if not files:
        raise FileNotFoundError(
            f"No {ZEBRA_BLOCK} phase-{PHASE} null files of library {sorted(libraries)} "
            f"for {session}/{signal}"
        )
    table = pd.concat([_unit_null_rows(group, path) for group, path in files], ignore_index=True)
    duplicated = table.duplicated(["group", "unit_id"])
    if duplicated.any():
        raise ValueError(f"{session}: {int(duplicated.sum())} units have more than one null file")

    mapped = np.array([observed.get(k, np.nan) for k in zip(table["group"], table["unit_id"])])
    diff = np.abs(mapped - table["max_obs_null"].to_numpy())
    wrong = np.isfinite(mapped) & (diff > CHECK_TOL)
    if wrong.any():
        raise ValueError(
            f"{session}/{signal}: the null lag-0 maximum disagrees with the maps' abs_max_value "
            f"for {int(wrong.sum())} units (up to {diff[wrong].max():.2e}). The nulls and the "
            "maps were not computed on the same data; nothing here is valid."
        )
    if np.isnan(mapped).any():
        print(f"  {int(np.isnan(mapped).sum())} tested units have no plotted map")

    p_values = table["p_null"].to_numpy()
    if modality == "slap2":
        table["q"] = np.nan
        for group, rows in table.groupby("group").groups.items():
            table.loc[rows, "q"] = _bh_fdr(p_values[table.index.get_indexer(rows)])
    else:
        table["q"] = _bh_fdr(p_values)
    table["significant"] = table["q"] <= Q_LEVEL
    family = "per DMD x channel" if modality == "slap2" else "over the session"
    print(
        f"  null test ({signal}, {ZEBRA_BLOCK}), BH {family}, q ≤ {Q_LEVEL}: "
        f"{int(table['significant'].sum())} / {len(table)} tested units"
    )
    _Q_CACHE[key] = table
    return table


def zebra_significant_units(modality: str, session: str) -> pd.DataFrame:
    """Units with BH q ≤ ``Q_LEVEL`` in their family."""
    table = zebra_q_table(modality, session)
    return table.loc[table["significant"], ["group", "unit_id", "p_null", "q"]].copy()


def keep_significant_zebra(df: pd.DataFrame, modality: str, session: str) -> pd.DataFrame:
    """Keep rows whose (group, unit) passes the Zebra null test."""
    significant = zebra_significant_units(modality, session)
    out = df.merge(significant, on=["group", "unit_id"], how="inner")
    out.attrs = dict(df.attrs)
    print(f"  plotted units passing the null test: {len(out)} / {len(df)}")
    return out


def rank_units(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Top units. ``zebra`` uses |r|; ``gabor`` uses the Gabor p-value, then |z|."""
    if by == "zebra":
        both = df.dropna(subset=["abs_max_value"])
        both = both[both["rf_map_zebra"].map(lambda m: isinstance(m, np.ndarray))]
        ordered = both.sort_values("abs_max_value", ascending=False)
    elif by == "gabor":
        if "p_value" not in df.columns or "z_max" not in df.columns:
            return df.iloc[0:0]
        # the unit-level p after the full search where it exists; the
        # per-orientation p of the best window ties at 1 / (n_shuffle + 1)
        p_col = (
            "p_value_gumbel"
            if "p_value_gumbel" in df.columns and df["p_value_gumbel"].notna().any()
            else "p_value"
        )
        both = df.dropna(subset=[p_col, "z_max"])
        if not len(both) or "rf_map_gabor" not in both.columns:
            return both.iloc[0:0]
        both = both[both["rf_map_gabor"].map(lambda m: isinstance(m, np.ndarray))]
        if not len(both):
            return both
        ordered = both.sort_values([p_col, "z_max"], ascending=[True, False])
    else:
        raise ValueError(by)
    return ordered.head(N_TOP)


def _show(fig, ax, image, extent, origin, cbar_label, xlabel):
    v = np.nanmax(np.abs(image))
    v = v if np.isfinite(v) and v > 0 else 1.0
    im = ax.imshow(
        image.T, cmap="coolwarm", vmin=-v, vmax=v, extent=extent,
        origin=origin, aspect="equal", interpolation="nearest",
    )
    ax.axhline(0, color=".4", lw=.4)
    ax.axvline(0, color=".4", lw=.4)
    ax.set_ylabel("Elevation (deg)", fontsize=FONTSIZE)
    if xlabel:
        ax.set_xlabel("Azimuth (deg)", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.set_xticks(np.arange(-60, 80, 20))
    ax.set_yticks(np.arange(-40, 60, 20))
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=FONTSIZE)
    cbar.set_label(cbar_label, fontsize=FONTSIZE)


def _panel_id(unit_id) -> str:
    """Break a long unit id so neighbouring panel titles do not run together."""
    text = str(unit_id)
    if len(text) <= 22:
        return text
    parts = text.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:2]) + "-\n" + "-".join(parts[2:])
    return text[: len(text) // 2] + "\n" + text[len(text) // 2 :]


def plot_maps(rows: pd.DataFrame, kind: str, extent, orientations, suptitle: str) -> plt.Figure:
    """One row of maps, unit order left to right matching ``rows``."""
    n = len(rows)
    fig, axs = plt.subplots(1, n, figsize=(5.0 * n, 5.0), squeeze=False, sharex=True, sharey=True)
    zebra = kind == "zebra"
    for i, (_, row) in enumerate(rows.iterrows()):
        ax = axs[0, i]
        where = row["structure"] if "structure" in row and pd.notna(row["structure"]) else row["group"]
        uid = _panel_id(row["unit_id"])
        if zebra:
            _show(fig, ax, row["rf_map_zebra"], WAVEN_EXTENT, "upper", "Correlation", True)
            for edge in (-40, 40):
                ax.axvline(edge, color=".75", lw=.6, zorder=1)
                ax.axhline(edge, color=".75", lw=.6, zorder=1)
            ax.set_title(
                f"{where}\n{uid}\n|r|max = {row['abs_max_value']:.3f}",
                fontsize=TITLE_FONTSIZE,
            )
        else:
            _show(fig, ax, row["rf_map_gabor"], extent, "lower", "response (z-scored)", True)
            ori = ""
            if pd.notna(row.get("ori_idx")) and len(orientations):
                ori = f", θ = {np.rad2deg(orientations[int(row['ori_idx'])]):.0f}°"
            ax.set_title(
                f"{where}\n{uid}{ori}\n"
                f"p = {row['p_value']:.4f}, |z|max = {row['z_max']:.2f}",
                fontsize=TITLE_FONTSIZE,
            )
    # Shared axes would otherwise crop to whichever panel was drawn last.
    if zebra:
        x0, x1, y0, y1 = WAVEN_EXTENT
    else:
        x0, x1, y0, y1 = extent
    axs[0, 0].set_xlim(x0, x1)
    axs[0, 0].set_ylim(y0, y1)
    # Leave the top strip for the figure title; tight_layout otherwise
    # places panel titles in the same band.
    fig.suptitle(suptitle, fontsize=TITLE_FONTSIZE + 2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def save_figure(fig: plt.Figure, stem: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    # every figure depends on the Zebra block (maps or significance), so a single
    # repeat gets its own file names and never overwrites the pooled figures
    block = "" if ZEBRA_BLOCK == "pooled_trials" else f"__zebra-{ZEBRA_BLOCK.replace('_', '-')}"
    path = PLOT_DIR / f"{FILE_PREFIX}{block}__{stem}.png"
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


def plot_ranked_maps() -> None:
    for label, (modality, session, keep) in SESSIONS.items():
        print(f"\n=== {label} ===")
        loaded = load_session(modality, session, keep)
        extent = loaded.attrs.get("extent", (-45, 45, -45, 45))
        orientations = loaded.attrs.get("orientations", ())
        try:
            df = restrict_population(label, loaded)
            df = keep_significant_zebra(df, modality, session)
        except SKIPPABLE as exc:
            _skip(f"ranked maps, {label}", exc)
            continue
        for by, note in (("zebra", "ranked-by-zebra"), ("gabor", "ranked-by-local-gabors")):
            top = rank_units(df, by)
            if not len(top):
                print(f"  no paired units to rank by {by}")
                continue
            print(f"  top {len(top)} by {by}:")
            for rank, row in enumerate(top.itertuples(index=False), start=1):
                print(
                    f"    {rank}. {row.group} / {row.unit_id}  "
                    f"|r|={row.abs_max_value:.3f}  p={row.p_value:.4f}  |z|={row.z_max:.2f}"
                )
            stem = f"{slug(label)}__{note}"
            rank_title = (
                f"significant {zebra_label()}, ranked by |r|"
                if by == "zebra"
                else f"significant {zebra_label()}, ranked by local Gabors"
            )
            has_zebra = top["rf_map_zebra"].map(lambda m: isinstance(m, np.ndarray)).any()
            has_gabor = (
                "rf_map_gabor" in top.columns
                and top["rf_map_gabor"].map(lambda m: isinstance(m, np.ndarray)).any()
            )
            if has_zebra:
                fig_z = plot_maps(top, "zebra", extent, orientations, f"{label} — {rank_title}")
                save_figure(fig_z, f"{stem}__zebra-rfs")
            if has_gabor:
                fig_g = plot_maps(top, "gabor", extent, orientations, f"{label} — {rank_title}")
                save_figure(fig_g, f"{stem}__gabor-rfs")
            elif by == "gabor":
                print("  no Gabor maps for these units (unit ids do not match the zebra/NWB set)")


# Sessions (keys of SESSIONS) of the RF summaries and the overlap pies.
SUMMARY_LABELS = ("ecephys 830794", "mesoscope 832700", "slap2 829704 green")

# DMD1 is the apical plane and DMD2 the basal plane. The line break is the
# tick label on the summary figures.
DMD1_LABEL = "DMD1\n(apical)"
DMD2_LABEL = "DMD2\n(basal)"

GROUP_COLORS = {
    "VISp": "#2166ac",
    "VISl": "#b2182b",
    "V1": "#2166ac",
    "LM": "#b2182b",
    DMD1_LABEL: "#2166ac",
    DMD2_LABEL: "#b2182b",
    "apical": "#2166ac",
    "basal": "#b2182b",
    "DMD1": "#2166ac",
    "DMD2": "#b2182b",
}


def label_summary_groups(modality: str, df: pd.DataFrame) -> pd.DataFrame:
    """VISp vs VISl, or SLAP2 apical (DMD1 green) vs basal (DMD2 green)."""
    if modality == "ecephys":
        anatomy = load_ephys_unit_areas()
        print(f"Unit areas from {UNIT_AREAS_CSV.name}")
        n_match = int(df["unit_id"].isin(set(anatomy["unit_id"])).sum())
        print(f"  unit-id overlap with saved areas: {n_match} / {len(df)}")
        if n_match == 0:
            print(f"  summary id sample: {df['unit_id'].head(3).tolist()}")
            print(f"  NWB id sample: {anatomy['unit_id'].head(3).tolist()}")
            return df.iloc[0:0].assign(structure=pd.Series(dtype=str))
        # Summaries have no area. Walk every probe, match that probe's unit
        # ids, and keep VISp / VISl from the updated extremum-channel labels.
        pieces = []
        probes = sorted(set(anatomy["probe"]) | set(df["group"]))
        for probe in probes:
            summary = df[df["group"] == probe]
            labeled = anatomy[anatomy["probe"] == probe]
            vis = labeled[labeled["area"].isin(["VISp", "VISl"])]
            matched = summary.merge(
                vis[["unit_id", "area"]], on="unit_id", how="inner"
            )
            print(
                f"  {probe}: {len(summary)} summary units, "
                f"{len(vis)} VISp/VISl in NWB, {len(matched)} matched"
            )
            if len(matched):
                pieces.append(matched)
        if not pieces:
            print(f"  summary id sample: {df['unit_id'].head(3).tolist()}")
            print(f"  NWB id sample: {anatomy['unit_id'].head(3).tolist()}")
            return df.iloc[0:0].assign(structure=pd.Series(dtype=str))
        out = pd.concat(pieces, ignore_index=True)
        out["structure"] = out["area"]
        return out

    if modality == "mesoscope":
        out = df[df["group"].str.match(r"VIS[pl]")].copy()
        out["structure"] = np.where(out["group"].str.startswith("VISp"), "VISp", "VISl")
        return out

    out = df[df["group"].isin(["DMD1__green", "DMD2__green"])].copy()
    out["structure"] = np.where(
        out["group"] == "DMD1__green", DMD1_LABEL, DMD2_LABEL
    )
    return out


STRUCTURE_ORDER = (
    "VISp", "VISl", "V1", "LM",
    DMD1_LABEL, DMD2_LABEL, "DMD1", "DMD2", "apical", "basal",
)
# Summary-figure text is twice the original size. Tick labels are 1.5×.
SUMMARY_LABEL = 26
SUMMARY_TICK = 20


def _structure_order(structures) -> list[str]:
    """VISp left of VISl. Other groups keep their listed order."""
    present = set(structures)
    return [name for name in STRUCTURE_ORDER if name in present]


def _summary_ci(values: np.ndarray, stat: str = "median", n_boot: int = 2000) -> tuple[float, float, float]:
    """Point estimate and bootstrap 95% CI. ``stat`` is ``median`` or ``mean``."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    reduce = np.median if stat == "median" else np.mean
    center = float(reduce(values))
    if len(values) == 1:
        return center, center, center
    rng = np.random.default_rng(0)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    lo, hi = np.percentile(reduce(draws, axis=1), [2.5, 97.5])
    return center, float(lo), float(hi)


def _star_label(p_value: float) -> str:
    if p_value > 0.05:
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def _median_permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = 10000) -> float:
    """Two-sided permutation p-value for a difference of medians."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    observed = abs(float(np.median(a) - np.median(b)))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(0)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(float(np.median(pooled[:n_a]) - np.median(pooled[n_a:])))
        count += diff >= observed - 1e-15
    return (count + 1) / (n_perm + 1)


# Summary figures name the same comparison on both rows.
_SUMMARY_RENAME = {
    "VISp": "V1",
    "VISl": "LM",
}


def _fit_columns(df: pd.DataFrame, map_col: str, mode: str, centres) -> pd.DataFrame:
    """Gaussian centre, one-sigma ellipse area and fit r2 of each map; NaN where the fit fails."""
    rows = []
    for image in df[map_col]:
        fit = None
        if isinstance(image, np.ndarray):
            x_pos, y_pos = centres(image.shape)
            fit = _fit_map(image, x_pos, y_pos, mode)
            values = (fit["sigma_x"], fit["sigma_y"], fit["x0"], fit["y0"])
            if not fit["success"] or not np.isfinite(values).all() or min(fit["sigma_x"], fit["sigma_y"]) <= 0:
                fit = None
        rows.append((np.nan,) * 4 if fit is None else (
            float(fit["x0"]), float(fit["y0"]),
            float(np.pi * fit["sigma_x"] * fit["sigma_y"]), float(fit["r2"]),
        ))
    return pd.DataFrame(rows, columns=["x0", "y0", "area", "fit_r2"], index=df.index)


# A Gabor-significant unit whose fitted centre lies beyond this many degrees in
# azimuth or elevation, or whose fit fails, is left out of the corpus.
GABOR_CENTRE_LIMIT = 40.0

_CORPUS_CACHE: dict[str, pd.DataFrame] = {}


def rf_corpus(label: str) -> pd.DataFrame:
    """Units of the RF summaries and the overlap pies, with both tests and fits.

    Every unit of the plotted areas (VISp / VISl, or the SLAP2 DMDs), tested on
    Zebra or not (`attach_tests`). Significant maps carry their Gaussian fit in
    ``gabor_*`` / ``zebra_*`` columns. Gabor-significant units whose fit fails
    or whose centre lies beyond ``GABOR_CENTRE_LIMIT`` are dropped.
    """
    if label in _CORPUS_CACHE:
        return _CORPUS_CACHE[label]
    modality, session, keep = SESSIONS[label]
    loaded = load_session(modality, session, keep)
    labeled = label_summary_groups(modality, loaded)
    labeled.attrs = dict(loaded.attrs)
    tested = attach_tests(labeled, modality, session)
    extent = loaded.attrs.get("extent", (-45.0, 45.0, -45.0, 45.0))
    gabor = _fit_columns(
        tested.loc[tested["pass_gabor"]], "rf_map_gabor", "signed",
        lambda shape: _gabor_centres(loaded, extent, shape),
    )
    zebra = _fit_columns(
        tested.loc[tested["pass_zebra"]], "rf_map_zebra", "abs", _zebra_centres,
    )
    tested = tested.join(gabor.add_prefix("gabor_")).join(zebra.add_prefix("zebra_"))
    inside = tested[["gabor_x0", "gabor_y0"]].abs().le(GABOR_CENTRE_LIMIT).all(axis=1)
    dropped = tested["pass_gabor"] & ~inside
    print(
        f"  Gabor q ≤ {Q_LEVEL}, fit failed or centre beyond ±{GABOR_CENTRE_LIMIT:g}°, excluded: "
        f"{tested.loc[dropped].groupby('structure').size().to_dict()}"
    )
    out = tested.loc[~dropped].reset_index(drop=True)
    out.attrs = dict(loaded.attrs)
    _CORPUS_CACHE[label] = out
    return out


def _fit_table(df: pd.DataFrame, kind: str, strength_col: str) -> pd.DataFrame:
    """One row per fitted map of ``kind`` (``gabor`` or ``zebra``) for the summary panels."""
    fitted = pd.DataFrame({
        "structure": df["structure"],
        "x0": df[f"{kind}_x0"],
        "y0": df[f"{kind}_y0"],
        "area": df[f"{kind}_area"],
        "strength": df[strength_col].astype(float),
        "fit_r2": df[f"{kind}_fit_r2"],
    })
    failed = fitted["x0"].isna()
    if failed.any():
        print(f"  {kind}: {int(failed.sum())} of {len(fitted)} fits failed, not drawn")
    return fitted.loc[~failed].reset_index(drop=True)


def prepare_fit_summaries() -> dict[str, dict]:
    """Gaussian summaries for the units of `rf_corpus` with q ≤ 0.05 on that stimulus.

    Position is the fitted centre. Area is the one-sigma ellipse. Gabor is the
    signed z-map of the best orientation; Zebra is the absolute correlation.
    """
    out = {}
    for label in SUMMARY_LABELS:
        modality = SESSIONS[label][0]
        print(f"\n=== fit summary {modality} ===")
        try:
            corpus = rf_corpus(label)
        except SKIPPABLE as exc:
            _skip(f"RF summary, {modality}", exc)
            continue
        if not len(corpus):
            print(f"  {modality}: no units in the plotted areas")
            continue
        corpus = corpus.assign(structure=corpus["structure"].replace(_SUMMARY_RENAME))
        gabor = corpus.loc[corpus["pass_gabor"]]
        zebra = corpus.loc[corpus["pass_zebra"]]
        print(f"  q ≤ {Q_LEVEL}: Gabor {gabor.groupby('structure').size().to_dict()}, "
              f"Zebra {zebra.groupby('structure').size().to_dict()}")
        gabor_fit = _fit_table(gabor, "gabor", "z_max_all")
        zebra_fit = _fit_table(zebra, "zebra", "abs_max_value")
        for name, fitted in (("Gabor", gabor_fit), ("Zebra", zebra_fit)):
            if len(fitted):
                print(f"  {name} median fit r2 {fitted['fit_r2'].median():.2f}")
        extent = corpus.attrs.get("extent", (-45.0, 45.0, -45.0, 45.0))
        gabor_sample = next((m for m in gabor["rf_map_gabor"] if isinstance(m, np.ndarray)), None)
        zebra_sample = next((m for m in zebra["rf_map_zebra"] if isinstance(m, np.ndarray)), None)
        gabor_extent = (
            _fit_window(*_gabor_centres(corpus, extent, gabor_sample.shape))
            if gabor_sample is not None else extent
        )
        zebra_extent = (
            _fit_window(*_zebra_centres(zebra_sample.shape))
            if zebra_sample is not None else WAVEN_EXTENT
        )
        out[modality] = {
            "gabor": gabor_fit,
            "zebra": zebra_fit,
            "gabor_extent": gabor_extent,
            "zebra_extent": zebra_extent,
        }
    return out


def _fit_window(x_pos, y_pos) -> tuple[float, float, float, float]:
    """Bounds of a fitted centre: the grid plus one step, as in `fit_gaussian_envelope`.

    The fitter uses the azimuth step on both axes.
    """
    x_pos, y_pos = np.asarray(x_pos, dtype=float), np.asarray(y_pos, dtype=float)
    step = float(np.median(np.diff(np.sort(x_pos))))
    return (x_pos.min() - step, x_pos.max() + step, y_pos.min() - step, y_pos.max() + step)


def _scatter_legend_handles(df: pd.DataFrame):
    from matplotlib.lines import Line2D

    structures = _structure_order(df["structure"]) if len(df) else []
    return [
        Line2D(
            [0], [0], linestyle="none", marker="o", markersize=16,
            markerfacecolor=GROUP_COLORS[structure], markeredgecolor="none",
            # Compartment (apical/basal) stays on the box-plot ticks only.
            label=f"{structure.split(chr(10), 1)[0]} (n={(df['structure'] == structure).sum()})",
        )
        for structure in structures
    ]


def _draw_position(ax, df: pd.DataFrame, extent, xlabel: bool) -> None:
    structures = _structure_order(df["structure"]) if len(df) else []
    for structure in structures:
        sub = df[df["structure"] == structure]
        ax.scatter(
            sub["x0"], sub["y0"],
            s=14 * 1.5 ** 2, alpha=0.55, linewidths=0,
            color=GROUP_COLORS[structure], zorder=2,
        )
    x0, x1, y0, y1 = (float(v) for v in extent)
    ax.axhline(0, color=".75", lw=.7, zorder=1)
    ax.axvline(0, color=".75", lw=.7, zorder=1)
    # Ticks outside the limits would make matplotlib widen the axis to them.
    xticks, yticks = np.arange(-60, 80, 20), np.arange(-40, 60, 20)
    ax.set_xticks(xticks[(xticks >= min(x0, x1)) & (xticks <= max(x0, x1))])
    ax.set_yticks(yticks[(yticks >= min(y0, y1)) & (yticks <= max(y0, y1))])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.tick_params(labelsize=SUMMARY_TICK, length=6, width=1.1)
    if xlabel:
        ax.set_xlabel("Azimuth (deg)", fontsize=SUMMARY_LABEL)
    ax.set_ylabel("Elevation (deg)", fontsize=SUMMARY_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _place_scatter_legends(fig, axes, frames, fig_w: float, fig_h: float, center_in: float) -> None:
    """Put one legend above each scatter, sharing the same left edge.

    The pair is centered on the position panels, in the band the row titles
    used to occupy. A shared left edge keeps the group dots in one column.
    """
    legends = []
    for ax, df in zip(axes, frames):
        handles = _scatter_legend_handles(df)
        if not handles:
            legends.append(None)
            continue
        legends.append(ax.legend(
            handles=handles, frameon=False, fontsize=SUMMARY_LABEL - 4,
            loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0,
            handlelength=0.7, handletextpad=0.35, labelspacing=0.3, borderpad=0.15,
        ))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [
        0.0 if leg is None else leg.get_window_extent(renderer).width / fig.dpi
        for leg in legends
    ]
    left = center_in - max(widths, default=0.0) / 2
    pad = 0.12
    for leg, ax in zip(legends, axes):
        if leg is None:
            continue
        top = ax.get_position().y1 * fig_h
        leg.set_bbox_to_anchor(
            (left / fig_w, (top + pad) / fig_h), transform=fig.transFigure,
        )


def _draw_comparison(ax, df: pd.DataFrame, column: str, ylabel: str, ymax: float | None = None) -> None:
    """Box plot of the two groups, with a bootstrap CI on the median.

    ``ymax`` fixes the top of the axis, and the top tick, at that value.
    Points above it are left outside the frame.
    """
    structures = _structure_order(df["structure"]) if len(df) else []
    grouped = []
    for structure in structures:
        values = df.loc[df["structure"] == structure, column].to_numpy(dtype=float)
        grouped.append(values[np.isfinite(values)])
    if ymax is not None:
        above = sum(int((values > ymax).sum()) for values in grouped)
        if above:
            n_all = sum(len(values) for values in grouped)
            print(f"    {ylabel}: {above} of {n_all} values lie above the axis limit {ymax:g}")
    boxed = [(i, values) for i, values in enumerate(grouped) if len(values) >= 2]
    if boxed:
        bp = ax.boxplot(
            [values for _, values in boxed],
            positions=[i for i, _ in boxed],
            widths=0.55,
            patch_artist=True,
            whis=1.5,
            showfliers=True,
            medianprops={"color": "k", "linewidth": 1.6},
            whiskerprops={"color": "k", "linewidth": 1.0},
            capprops={"color": "k", "linewidth": 1.0},
            flierprops={
                "marker": "o", "markersize": 3.5, "markerfacecolor": "0.35",
                "markeredgecolor": "none", "alpha": 0.55,
            },
        )
        for patch, (i, _) in zip(bp["boxes"], boxed):
            patch.set_facecolor(GROUP_COLORS[structures[i]])
            patch.set_edgecolor("k")
            patch.set_alpha(0.45)
            patch.set_linewidth(1.0)
    for i, values in enumerate(grouped):
        if len(values) == 1:
            ax.scatter(
                i, values[0], s=28, color=GROUP_COLORS[structures[i]],
                edgecolors="k", linewidths=0.6, zorder=4,
            )
        if len(values):
            center, lo, hi = _summary_ci(values, stat="median")
            ax.errorbar(
                i, center, yerr=[[max(center - lo, 0.0)], [max(hi - center, 0.0)]],
                fmt="none", ecolor="k", elinewidth=1.3, capsize=5, capthick=1.3, zorder=6,
            )
    if len(grouped) == 2 and all(len(values) for values in grouped):
        p_value = _median_permutation_p(grouped[0], grouped[1])
        print(f"    {ylabel}: median permutation p = {p_value:.4g}")
        pooled = np.concatenate(grouped)
        if ymax is None:
            y_top = float(np.nanmax(pooled))
            y_span = y_top - float(np.nanmin(pooled))
            if y_span <= 0:
                y_span = abs(y_top) if y_top else 1.0
            bracket = y_top + 0.08 * y_span
            rise = 0.035 * y_span
            y_lo = float(np.nanmin(pooled)) - 0.05 * y_span
            y_hi = bracket + 0.28 * y_span
        else:
            y_lo, y_hi = 0.0, float(ymax)
            y_span = y_hi - y_lo
            bracket = y_hi - 0.16 * y_span
            rise = 0.03 * y_span
        ax.plot(
            [0, 0, 1, 1], [bracket, bracket + rise, bracket + rise, bracket],
            color="k", lw=0.8, clip_on=ymax is not None,
        )
        ax.text(
            0.5, bracket + 0.045 * y_span, _star_label(p_value),
            ha="center", va="bottom", fontsize=SUMMARY_LABEL - 2,
            clip_on=ymax is not None,
        )
        ax.set_ylim(y_lo, y_hi)
        if ymax is not None:
            ax.set_yticks(np.arange(0, ymax + 1, 1000))
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(range(len(structures)))
    ax.set_xticklabels(structures, fontsize=SUMMARY_TICK)
    ax.set_ylabel(ylabel, fontsize=SUMMARY_LABEL)
    ax.tick_params(axis="y", labelsize=SUMMARY_TICK, length=6, width=1.1)
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def _ylabel_overhang(ax) -> float:
    """Inches from the left edge of the y-axis label to the axes spine."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    spine = ax.get_window_extent(renderer).x0
    label = ax.yaxis.label.get_window_extent(renderer).x0
    return max((spine - label) / fig.dpi, 0.0)


def plot_fit_summaries(tables: dict[str, dict]) -> None:
    """One figure per modality: Gabor on top, Zebra below.

    Each row is the Gaussian centre, the one-sigma ellipse area, and the
    response strength. A degree has the same length in both position panels,
    and the Zebra window is the wider one. On the Zebra row the clear gap
    from the scatter to the area label equals the gap from the area panel to
    the correlation label. Area and correlation panels share that spacing in
    both rows, and share the vertical span of the position panel in their row.
    """
    inch = 0.034
    # Clear space between a panel edge and the next panel's y-axis label.
    clear = 0.42
    group_w, gap_row = 3.2, 1.65
    margin_l, margin_r = 1.7, 0.45
    margin_b, margin_t = 1.55, 1.95
    # Draft gaps are wide enough that the labels can be measured without overlap.
    draft_gap = 2.3

    for modality, packed in tables.items():
        gabor, zebra = packed["gabor"], packed["zebra"]
        # Each scatter spans its fit window in azimuth, and both share the wider
        # elevation range, so no fitted centre is clipped.
        gabor_extent, zebra_extent = packed["gabor_extent"], packed["zebra_extent"]
        y_lo = min(gabor_extent[2], zebra_extent[2])
        y_hi = max(gabor_extent[3], zebra_extent[3])
        gabor_extent = (*gabor_extent[:2], y_lo, y_hi)
        zebra_extent = (*zebra_extent[:2], y_lo, y_hi)
        gabor_w, gabor_h = (span * inch for span in _span(gabor_extent))
        zebra_w, zebra_h = (span * inch for span in _span(zebra_extent))
        top_h = max(gabor_h, zebra_h)
        draft_w = margin_l + zebra_w + draft_gap + group_w + draft_gap + group_w + margin_r
        fig_h = margin_b + zebra_h + gap_row + top_h + margin_t
        fig = plt.figure(figsize=(draft_w, fig_h))

        def place(x, y, w, h, fig_w):
            return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])

        gabor_y = margin_b + zebra_h + gap_row + (top_h - gabor_h) / 2
        zebra_y = margin_b
        # Same azimuth above the same azimuth: offset by the gap between the left limits.
        gabor_x = margin_l + (gabor_extent[0] - zebra_extent[0]) * inch
        gabor_ax = place(gabor_x, gabor_y, gabor_w, gabor_h, draft_w)
        zebra_ax = place(margin_l, zebra_y, zebra_w, zebra_h, draft_w)
        print(f"\n{modality}")
        _draw_position(gabor_ax, gabor, gabor_extent, xlabel=False)
        _draw_position(zebra_ax, zebra, zebra_extent, xlabel=True)

        draft_area_x = margin_l + zebra_w + draft_gap
        draft_strength_x = draft_area_x + group_w + draft_gap
        rows = (
            (gabor, gabor_y, gabor_h, "Peak |z score|"),
            (zebra, zebra_y, zebra_h, "Peak |correlation|"),
        )
        area_axes, strength_axes = [], []
        for frame, y, height, strength_label in rows:
            area_ax = place(draft_area_x, y, group_w, height, draft_w)
            strength_ax = place(draft_strength_x, y, group_w, height, draft_w)
            _draw_comparison(area_ax, frame, "area", "RF area (deg²)", ymax=2000)
            _draw_comparison(strength_ax, frame, "strength", strength_label)
            area_axes.append(area_ax)
            strength_axes.append(strength_ax)

        # Zebra is the wider scatter, so its labels set the column positions.
        area_x = margin_l + zebra_w + clear + _ylabel_overhang(area_axes[1])
        strength_x = area_x + group_w + clear + _ylabel_overhang(strength_axes[1])
        fig_w = strength_x + group_w + margin_r
        fig.set_size_inches(fig_w, fig_h, forward=True)
        for ax, x, y, w, h in (
            (gabor_ax, gabor_x, gabor_y, gabor_w, gabor_h),
            (zebra_ax, margin_l, zebra_y, zebra_w, zebra_h),
        ):
            ax.set_position([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
        for area_ax, strength_ax, (_, y, height, _) in zip(area_axes, strength_axes, rows):
            area_ax.set_position([area_x / fig_w, y / fig_h, group_w / fig_w, height / fig_h])
            strength_ax.set_position([
                strength_x / fig_w, y / fig_h, group_w / fig_w, height / fig_h,
            ])
        fig.suptitle(f"{modality} — q ≤ {Q_LEVEL}", fontsize=SUMMARY_LABEL + 4, y=0.985)
        _place_scatter_legends(
            fig, (gabor_ax, zebra_ax), (gabor, zebra),
            fig_w, fig_h, margin_l + zebra_w / 2,
        )
        save_figure(fig, f"{modality}__rf-summary")


PAIRED_PANELS = ("ecephys 830794", "mesoscope 832700", "slap2 829704 green")


def _lookup(df: pd.DataFrame, group: str, unit_id: str):
    if df is None or not len(df) or "unit_id" not in df.columns:
        return None
    hit = df[(df["group"] == group) & (df["unit_id"].astype(str) == str(unit_id))]
    if not len(hit):
        return None
    return hit.iloc[0]


def _map_image(row, column):
    if row is None or column not in row.index:
        return None
    image = row[column]
    return image if isinstance(image, np.ndarray) else None


def _span(extent) -> tuple[float, float]:
    x0, x1, y0, y1 = (float(v) for v in extent)
    return abs(x1 - x0), abs(y1 - y0)


def _peak(image) -> float:
    if image is None:
        return 1.0
    peak = np.nanmax(np.abs(image))
    if not np.isfinite(peak) or peak <= 0:
        return 1.0
    return float(peak)


def _row_scale(maps) -> float:
    peaks = [_peak(image) for image in maps if image is not None]
    return max(peaks) if peaks else 1.0


def _paint_selection(ax, color: str) -> None:
    """Frame one map in the pie-chart color of its selection column."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(3.6)


def _draw_paired(ax, image, extent, origin, vmax, xlabel, ylabel, faint_zero=False):
    im = ax.imshow(
        image.T, cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=extent,
        origin=origin, aspect="equal", interpolation="nearest",
    )
    zero = ".75" if faint_zero else ".4"
    ax.axhline(0, color=zero, lw=.7, zorder=1)
    ax.axvline(0, color=zero, lw=.7, zorder=1)
    ax.tick_params(labelsize=PAIRED_TICK_SIZE)
    ax.set_xticks(np.arange(-60, 80, 20))
    ax.set_yticks(np.arange(-40, 60, 20))
    if xlabel:
        ax.set_xlabel("Azimuth (deg)", fontsize=PAIRED_LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PAIRED_LABEL_SIZE)
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return im


def _draw_empty(ax, extent, ylabel, text=None):
    """A map slot with no unit: no ticks, no frame, optionally a note in the middle."""
    ax.set_xlim(*extent[:2])
    ax.set_ylim(*extent[2:])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PAIRED_LABEL_SIZE)
    if text:
        ax.text(
            0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=PAIRED_TICK_SIZE,
        )


def _zebra_centres(shape):
    """Pixel centres of a Zebra map, in the same degrees as the panel.

    The map is drawn with ``origin='upper'`` and ``WAVEN_EXTENT`` as the
    pixel edges, so index 0 of the elevation axis is the top edge. The stored
    ``x_deg`` / ``y_deg`` are the grid index times the step, half a pixel off
    these centres.
    """
    nx, ny = shape
    left, right, bottom, top = (float(v) for v in WAVEN_EXTENT)
    x = left + (np.arange(nx) + 0.5) * (right - left) / nx
    y = top - (np.arange(ny) + 0.5) * (top - bottom) / ny
    return x, y


def _draw_gaussian(ax, fit):
    """Mark the Gaussian centre and the one-sigma ellipse of the main axes."""
    from matplotlib.patches import Ellipse

    if not fit.get("success"):
        return False
    sx, sy = float(fit["sigma_x"]), float(fit["sigma_y"])
    x0, y0 = float(fit["x0"]), float(fit["y0"])
    if not np.isfinite([sx, sy, x0, y0, fit["theta"]]).all() or min(sx, sy) <= 0:
        return False
    # sigma_x lies along angle theta: see the rotation in gaussian2d.
    kw = dict(
        xy=(x0, y0), width=2 * sx, height=2 * sy,
        angle=np.degrees(float(fit["theta"])), fill=False, zorder=3,
    )
    ax.add_patch(Ellipse(edgecolor="black", linewidth=3.2, **kw))
    ax.add_patch(Ellipse(edgecolor="white", linewidth=1.5, **kw))
    ax.plot(
        x0, y0, marker="o", ms=7, mfc="white", mec="black", mew=1.3,
        linestyle="none", zorder=4,
    )
    return True


def _fit_map(image, x_pos, y_pos, mode):
    """``mode`` is ``signed`` for a Gabor z-map, or ``abs`` for a Zebra envelope."""
    values = np.abs(image) if mode == "abs" else image
    return fit_gaussian_envelope(values, x_pos, y_pos, nonnegative=mode == "abs")


def _gabor_centres(frame, extent, shape):
    x_pos = frame.attrs.get("x_positions")
    y_pos = frame.attrs.get("y_positions")
    if x_pos is None or len(x_pos) != shape[0] or len(y_pos) != shape[1]:
        from analysis.gaussian_envelope import patch_centres
        x_pos = patch_centres(extent[:2], shape[0])
        y_pos = patch_centres(extent[2:], shape[1])
    return np.asarray(x_pos, dtype=float), np.asarray(y_pos, dtype=float)


def _overlay_gaussian(ax, image, frame, extent, mode, unit="", verbose=True):
    if image is None:
        return
    if mode == "signed":
        x_pos, y_pos = _gabor_centres(frame, extent, image.shape)
        kind = "Gabor z"
    else:
        x_pos, y_pos = _zebra_centres(image.shape)
        kind = "Zebra |r|"
    fit = _fit_map(image, x_pos, y_pos, mode)
    prefix = f"    {unit}: " if unit else "    "
    if not _draw_gaussian(ax, fit):
        if verbose:
            print(f"{prefix}{kind} Gaussian fit did not converge")
    elif verbose:
        print(
            f"{prefix}{kind} centre ({fit['x0']:.1f}, {fit['y0']:.1f}) deg, "
            f"sigma {fit['sigma_x']:.1f} x {fit['sigma_y']:.1f}, fit r2 {fit['r2']:.2f}"
        )


def _mark_q(ax, passed: bool, fontsize: float = PAIRED_TICK_SIZE) -> None:
    """``q < Q_LEVEL`` or ``n.s.`` in the map's upper-right corner."""
    ax.text(
        0.97, 0.97, f"q < {Q_LEVEL:g}" if passed else "n.s.",
        transform=ax.transAxes, ha="right", va="top", fontsize=fontsize, zorder=5,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )


def _mark_peak_rate(ax, row, z_map, frame, extent) -> None:
    """Highest rate of any position, with an arrow to the largest-z position."""
    rate_map = row.get("rate_map")
    if not isinstance(rate_map, np.ndarray) or not np.isfinite(rate_map).any():
        return
    x_pos, y_pos = _gabor_centres(frame, extent, z_map.shape)
    zi, zj = np.unravel_index(np.nanargmax(z_map), z_map.shape)
    ri, rj = np.unravel_index(np.nanargmax(rate_map), rate_map.shape)
    # The text sits in the half of the map away from the peak.
    tx = 0.72 if x_pos[zi] < np.mean(x_pos) else 0.28
    ty = 0.62 if y_pos[zj] < np.mean(y_pos) else 0.3
    ax.annotate(
        f"peak rate\n{rate_map[ri, rj]:.2g} Hz",
        xy=(x_pos[zi], y_pos[zj]), xycoords="data",
        xytext=(tx, ty), textcoords="axes fraction", ha="center", va="center",
        fontsize=PAIRED_TICK_SIZE, zorder=6,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 2, "mutation_scale": 22,
                    "shrinkA": 4, "shrinkB": 10},
    )


def _paired_colorbar(fig, im, cax, label):
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=PAIRED_TICK_SIZE)
    cbar.set_label(label, fontsize=PAIRED_LABEL_SIZE)
    return cbar


def _both_maps(df: pd.DataFrame) -> pd.DataFrame:
    if "rf_map_gabor" not in df.columns or "rf_map_zebra" not in df.columns:
        return df.iloc[0:0]
    keep = df["rf_map_gabor"].map(lambda m: isinstance(m, np.ndarray)) & df["rf_map_zebra"].map(
        lambda m: isinstance(m, np.ndarray)
    )
    out = df.loc[keep].copy()
    out.attrs = dict(df.attrs)
    return out


def primary_visual(label: str, df: pd.DataFrame) -> pd.DataFrame:
    """Primary visual cortex for the paired figure: VISp layer 2/3, or VISp_0."""
    if label.startswith("ecephys"):
        anatomy = load_ephys_unit_areas()
        visp = anatomy[(anatomy["area"] == "VISp") & (anatomy["layer"] == "2/3")]
        out = df.merge(
            visp[["unit_id", "area", "layer", "structure"]], on="unit_id", how="inner",
        )
        print(f"  VISp layer 2/3: {len(out)} units")
    elif label.startswith("mesoscope"):
        out = df[df["group"] == "VISp_0"].copy()
        print(f"  VISp_0: {len(out)} units")
    else:
        out = df
        print(f"  SLAP2 V1 layer 2/3: {len(out)} units")
    for key in ("extent", "orientations", "x_positions", "y_positions"):
        if key in df.attrs:
            out.attrs[key] = df.attrs[key]
    if "extent" not in out.attrs:
        out.attrs["extent"] = (-45, 45, -45, 45)
    return out


def plot_paired_panel(
    label, selected, frame, extent, criterion, overlay_gaussian=False, column_colors=None,
    column_titles=None, mark_q=False, peak_rate_columns=(), empty_text=None,
):
    """One column per unit: Gabor on top, pooled Zebra below.

    A degree has the same length on both stimuli. Zebra covers more azimuth,
    so it is wider and the Gabor map is centered on the same vertical meridian.
    Each row has one shared color scale and one colorbar. ``column_colors``
    paints both maps in a column with that selection color, except in a column
    without a unit. ``column_titles``
    go above the Gabor maps. ``mark_q`` writes on every map whether that
    stimulus's RF passed BH (``q < Q_LEVEL``) or not. In ``peak_rate_columns``
    the Gabor map names the highest rate of any position, with an arrow to the
    position of the largest z. ``empty_text`` holds one ``(gabor, zebra)`` pair
    per column: a column without a unit then draws both maps without ticks, with
    that text (or nothing, for None) in the middle.
    """
    n = len(selected)
    az_g, el_g = _span(extent)
    az_z, el_z = _span(WAVEN_EXTENT)
    gabor_w, gabor_h = az_g * PAIRED_INCH_PER_DEG, el_g * PAIRED_INCH_PER_DEG
    zebra_w, zebra_h = az_z * PAIRED_INCH_PER_DEG, el_z * PAIRED_INCH_PER_DEG
    cbar_w, cbar_pad = 0.32, 0.28
    panel_gap, row_gap = 1.05, 0.95
    margin_l, margin_r = 2.55, 1.35
    margin_b, margin_t = 1.55, 0.95
    if column_titles is not None:
        margin_t += 0.55

    fig_w = margin_l + n * zebra_w + (n - 1) * panel_gap + cbar_pad + cbar_w + margin_r
    fig_h = margin_b + zebra_h + row_gap + gabor_h + margin_t
    fig = plt.figure(figsize=(fig_w, fig_h))

    def column_x(index: int) -> float:
        return margin_l + index * (zebra_w + panel_gap)

    rows = [_lookup(frame, picked.group, picked.unit_id) for picked in selected.itertuples(index=False)]
    gabor_maps = [_map_image(row, "rf_map_gabor") for row in rows]
    zebra_maps = [_map_image(row, "rf_map_zebra") for row in rows]
    gabor_vmax = _row_scale(gabor_maps)
    zebra_vmax = _row_scale(zebra_maps)
    gabor_im = zebra_im = None

    for col, (gabor_map, zebra_map) in enumerate(zip(gabor_maps, zebra_maps)):
        zx = column_x(col)
        zy = margin_b
        gx = zx + (zebra_w - gabor_w) / 2
        gy = margin_b + zebra_h + row_gap
        gabor_ax = fig.add_axes([gx / fig_w, gy / fig_h, gabor_w / fig_w, gabor_h / fig_h])
        zebra_ax = fig.add_axes([zx / fig_w, zy / fig_h, zebra_w / fig_w, zebra_h / fig_h])
        color = None if column_colors is None else column_colors[col]
        gabor_ylabel = "Gabor\nElevation (deg)" if col == 0 else None
        zebra_ylabel = f"{zebra_label()}\nElevation (deg)" if col == 0 else None
        empty = rows[col] is None and empty_text is not None
        if empty:
            _draw_empty(gabor_ax, extent, gabor_ylabel, empty_text[col][0])
        elif gabor_map is not None:
            gabor_im = _draw_paired(
                gabor_ax, gabor_map, extent, "lower", gabor_vmax,
                xlabel=False, ylabel="Gabor\nElevation (deg)" if col == 0 else None,
            )
        elif color is None:
            gabor_ax.set_axis_off()
        else:
            _draw_paired(
                gabor_ax, np.full((2, 2), np.nan), extent, "lower", gabor_vmax,
                xlabel=False, ylabel="Gabor\nElevation (deg)" if col == 0 else None,
            )
        if empty:
            _draw_empty(zebra_ax, WAVEN_EXTENT, zebra_ylabel, empty_text[col][1])
        elif zebra_map is not None:
            zebra_im = _draw_paired(
                zebra_ax, zebra_map, WAVEN_EXTENT, "upper", zebra_vmax,
                xlabel=True, ylabel=f"{zebra_label()}\nElevation (deg)" if col == 0 else None,
                faint_zero=True,
            )
        elif color is None:
            zebra_ax.set_axis_off()
        else:
            _draw_paired(
                zebra_ax, np.full((2, 2), np.nan), WAVEN_EXTENT, "upper", zebra_vmax,
                xlabel=True, ylabel=f"{zebra_label()}\nElevation (deg)" if col == 0 else None,
                faint_zero=True,
            )
        # an empty column has no selection to frame
        if color is not None and not empty:
            _paint_selection(gabor_ax, color)
            _paint_selection(zebra_ax, color)
        if column_titles is not None:
            gabor_ax.set_title(column_titles[col], fontsize=PAIRED_LABEL_SIZE, pad=12)
        # if col in peak_rate_columns and rows[col] is not None and gabor_map is not None:
        #     _mark_peak_rate(gabor_ax, rows[col], gabor_map, frame, extent)
        if mark_q and rows[col] is not None:
            for ax, passed in (
                (gabor_ax, rows[col].get("pass_gabor", False)),
                (zebra_ax, rows[col].get("pass_zebra", False)),
            ):
                _mark_q(ax, bool(passed))
        if overlay_gaussian and rows[col] is not None:
            unit = f"{rows[col].group} / {rows[col].unit_id}"
            # A fit is a position and a size only when that stimulus's RF passed.
            if bool(rows[col].get("pass_gabor", False)):
                _overlay_gaussian(gabor_ax, gabor_map, frame, extent, "signed", unit)
            if bool(rows[col].get("pass_zebra", False)):
                _overlay_gaussian(zebra_ax, zebra_map, frame, extent, "abs", unit)

    cbar_x = (column_x(n - 1) + zebra_w + cbar_pad) / fig_w
    if gabor_im is not None:
        gabor_cax = fig.add_axes([
            cbar_x, (margin_b + zebra_h + row_gap) / fig_h, cbar_w / fig_w, gabor_h / fig_h,
        ])
        _paired_colorbar(fig, gabor_im, gabor_cax, "response (z)")
    if zebra_im is not None:
        zebra_cax = fig.add_axes([
            cbar_x, margin_b / fig_h, cbar_w / fig_w, zebra_h / fig_h,
        ])
        _paired_colorbar(fig, zebra_im, zebra_cax, "Correlation")
    fig.suptitle(f"{label} — {criterion}", fontsize=PAIRED_TITLE_SIZE, y=0.98)
    return fig


def plot_paired_release() -> None:
    """Gabor above Zebra for primary visual cortex, among null-test passers."""
    populations = {
        "ecephys 830794": "VISp layer 2/3",
        "mesoscope 832700": "VISp_0",
        "slap2 829704 green": "V1 layer 2/3",
    }
    for label in PAIRED_PANELS:
        modality, session, keep = SESSIONS[label]
        print(f"\n=== paired {label} ===")
        try:
            loaded = load_session(modality, session, keep)
            frame = keep_significant_zebra(primary_visual(label, loaded), modality, session)
        except SKIPPABLE as exc:
            _skip(f"paired panels, {label}", exc)
            continue
        paired = _both_maps(frame)
        extent = frame.attrs.get("extent") or (-45, 45, -45, 45)
        if not len(paired):
            print("  no significant units have both a Gabor map and a Zebra map")
            continue
        population = populations[label]
        for by, note in (("zebra", "ranked-by-zebra"), ("gabor", "ranked-by-local-gabors")):
            criterion = "ranked by Zebra" if by == "zebra" else "ranked by Gabor"
            pool = paired
            if by == "gabor" and "z_max" in pool.columns:
                pool = pool[pool["z_max"] > 0]
            top = rank_units(pool, by)
            if not len(top):
                print(f"  no units to rank by {by}")
                continue
            print(f"  top {len(top)} by {by}:")
            for rank, row in enumerate(top.itertuples(index=False), start=1):
                print(f"    {rank}. {row.group} / {row.unit_id}")
            fig = plot_paired_panel(
                f"{label} — {population}", top, paired, extent, criterion,
            )
            save_figure(fig, f"{slug(label)}__{note}__full-gabor")


def attach_tests(df: pd.DataFrame, modality: str, session: str) -> pd.DataFrame:
    """Add the Zebra q and the Gabor BH q.

    A unit without a test on one stimulus (e.g. below the Zebra null's spike
    minimum) is kept with p = q = 1 there, so it counts as not significant.
    """
    if "p_value_bh" not in df.columns or df["p_value_bh"].isna().all():
        raise FileNotFoundError(
            f"{session}: the Gabor files carry no p_value_bh. "
            "Run scripts/compute_gabor_unit_significance.py first."
        )
    q_table = zebra_q_table(modality, session)
    q_table = q_table[["group", "unit_id", "p_null", "q"]].rename(
        columns={"q": "q_zebra", "p_null": "p_zebra"}
    )
    out = df.merge(q_table, on=["group", "unit_id"], how="left").reset_index(drop=True)
    untested = out[["p_value_bh", "q_zebra"]].isna()
    print(
        f"  untested, set to p = q = 1: Gabor {int(untested['p_value_bh'].sum())}, "
        f"Zebra {int(untested['q_zebra'].sum())}"
    )
    out[["p_value_bh", "p_zebra", "q_zebra"]] = out[["p_value_bh", "p_zebra", "q_zebra"]].fillna(1.0)
    out["pass_zebra"] = out["q_zebra"] <= Q_LEVEL
    out["pass_gabor"] = out["p_value_bh"] <= Q_LEVEL
    out.attrs = dict(df.attrs)
    return out


def _top_zebra(pool: pd.DataFrame, n: int = N_TOP) -> pd.DataFrame:
    """Lowest Zebra p-value first, then largest peak |r|."""
    ordered = pool.dropna(subset=["p_zebra"]).sort_values(
        ["p_zebra", "abs_max_value"], ascending=[True, False],
    )
    return ordered.head(n)


def _top_gabor(pool: pd.DataFrame, n: int = N_TOP) -> pd.DataFrame:
    """Lowest unit-level Gabor p-value first, then largest peak |z|."""
    ordered = pool.dropna(subset=["p_value_gumbel"]).sort_values(
        ["p_value_gumbel", "z_max"], ascending=[True, False],
    )
    return ordered.head(n)


def _top_both(pool: pd.DataFrame, n: int = N_TOP) -> pd.DataFrame:
    """Smallest sum of the Zebra rank and the Gabor rank.

    Rank 1 is the best unit on that stimulus. Ties share the better rank.
    """
    work = pool.dropna(subset=["p_zebra", "p_value_gumbel"]).copy()
    if not len(work):
        return work
    work["zebra_rank"] = work["p_zebra"].rank(ascending=True, method="min")
    work["gabor_rank"] = work["p_value_gumbel"].rank(ascending=True, method="min")
    work["combined_rank"] = work["zebra_rank"] + work["gabor_rank"]
    ordered = work.sort_values(
        ["combined_rank", "zebra_rank", "p_zebra"],
        ascending=[True, True, True],
    )
    return ordered.head(n)


def plot_overlap_examples() -> None:
    """Top units for Zebra only, Gabor only, and both.

    Zebra only and Gabor only each keep the three lowest unit-level p-values.
    Both adds those two ranks and keeps the three smallest sums. Ephys is
    VISp layer 2/3, mesoscope is VISp_0, and SLAP2 is the green channel.
    """
    jobs = (
        ("ecephys 830794", "VISp layer 2/3"),
        ("mesoscope 832700", "VISp_0"),
        ("slap2 829704 green", "V1 layer 2/3"),
    )
    categories = (
        ("zebra-only", "Zebra only, lowest p", lambda df: df["pass_zebra"] & ~df["pass_gabor"], _top_zebra),
        ("gabor-only", "Gabor only, lowest p", lambda df: df["pass_gabor"] & ~df["pass_zebra"], _top_gabor),
        ("both", "both, lowest rank sum", lambda df: df["pass_zebra"] & df["pass_gabor"], _top_both),
    )
    for label, population in jobs:
        modality, session, keep = SESSIONS[label]
        print(f"\n=== overlap examples {label} ===")
        try:
            loaded = load_session(modality, session, keep)
            tested = attach_tests(primary_visual(label, loaded), modality, session)
        except SKIPPABLE as exc:
            _skip(f"overlap examples, {label}", exc)
            continue
        paired = _both_maps(tested)
        extent = paired.attrs.get("extent") or (-45, 45, -45, 45)
        print(
            f"  tested by both: {len(tested)}; with both maps: {len(paired)}; "
            f"Zebra {(paired['pass_zebra']).sum() if len(paired) else 0}, "
            f"Gabor {(paired['pass_gabor']).sum() if len(paired) else 0}"
        )
        for stem, criterion, choose, ranker in categories:
            pool = paired.loc[choose(paired)] if len(paired) else paired
            selected = ranker(pool) if len(pool) else pool
            print(f"  {stem}: {len(pool)} eligible, drawing {len(selected)}")
            for row in selected.itertuples(index=False):
                extra = ""
                if stem == "both" and hasattr(row, "combined_rank"):
                    extra = (
                        f"  p {row.p_zebra:.3g}+{row.p_value_gumbel:.3g}"
                        f"  ranks {row.zebra_rank:.0f}+{row.gabor_rank:.0f}={row.combined_rank:.0f}"
                    )
                elif stem == "zebra-only":
                    extra = f"  p {row.p_zebra:.3g}"
                elif stem == "gabor-only":
                    extra = f"  p {row.p_value_gumbel:.3g}"
                print(f"    {row.group} / {row.unit_id}{extra}")
            if not len(selected):
                continue
            fig = plot_paired_panel(
                f"{label} — {population}", selected, paired, extent, criterion,
                overlay_gaussian=True,
            )
            save_figure(fig, f"{slug(label)}__{stem}__full-gabor")


_PIE_ORDER = ("neither", "Zebra only", "Gabor only", "both")
_PIE_COLORS = {
    "neither": "#c8c8c8",
    "Zebra only": "#4C78A8",
    "Gabor only": "#E45756",
    "both": "#54A24B",
}
# Selections of the example figures, in column order:
# (file stem, pie-chart category for the color, title, selection, ranking)
OVERLAP_COLUMNS = (
    ("both", "both", "Both significant", lambda df: df["pass_zebra"] & df["pass_gabor"], _top_both),
    ("gabor-only", "Gabor only", "Gabor significant", lambda df: df["pass_gabor"] & ~df["pass_zebra"], _top_gabor),
    ("zebra-only", "Zebra only", "Zebra significant", lambda df: df["pass_zebra"] & ~df["pass_gabor"], _top_zebra),
)
N_TOP_GRID = 10
# Column titles of the example summaries, by file stem
EXAMPLE_TITLES = {
    "both": "Both\nsignificant",
    "gabor-only": "Only Gabor\nsignificant",
    "zebra-only": "Only Zebra\nsignificant",
}


def plot_overlap_example_summaries() -> None:
    """One example per selection, three columns by two rows.

    Columns are both, Gabor only, then Zebra only. Ephys takes the second
    ranked example in each column. Mesoscope and SLAP2 take the first. Both
    maps in a column share that selection's pie-chart color, and each row
    uses one color scale.
    """
    jobs = (
        ("ecephys 830794", "VISp layer 2/3", 1),
        ("mesoscope 832700", "VISp_0", 0),
        ("slap2 829704 green", "V1 layer 2/3", 0),
    )
    for label, population, index in jobs:
        modality, session, keep = SESSIONS[label]
        print(f"\n=== example summary {label} ===")
        try:
            loaded = load_session(modality, session, keep)
            tested = attach_tests(primary_visual(label, loaded), modality, session)
        except SKIPPABLE as exc:
            _skip(f"example summary, {label}", exc)
            continue
        paired = _both_maps(tested)
        extent = paired.attrs.get("extent") or (-45, 45, -45, 45)
        picked = []
        colors = []
        titles = []
        for stem, pie_name, title, choose, ranker in OVERLAP_COLUMNS:
            pool = paired.loc[choose(paired)] if len(paired) else paired
            selected = ranker(pool) if len(pool) else pool
            colors.append(_PIE_COLORS[pie_name])
            titles.append(EXAMPLE_TITLES[stem])
            if len(selected) > index:
                row = selected.iloc[index]
                print(f"  {stem}: example {index + 1}  {row.group} / {row.unit_id}")
                picked.append({"group": row.group, "unit_id": row.unit_id})
            else:
                print(f"  {stem}: no example at position {index + 1}")
                picked.append({"group": None, "unit_id": None})
        summary = pd.DataFrame(picked)
        # An empty column says so on the stimulus it selects on; the other map stays blank.
        note = "No significant result"
        empty_text = [
            (note if stem != "zebra-only" else None, note if stem != "gabor-only" else None)
            for stem, *_ in OVERLAP_COLUMNS
        ]
        fig = plot_paired_panel(
            f"{label} — {population}", summary, paired, extent, "one example",
            overlay_gaussian=True, column_colors=colors, column_titles=titles, mark_q=True,
            empty_text=empty_text,
            # Ephys only: the Zebra-only unit's Gabor map has too few spikes to pass.
            peak_rate_columns=(
                {[stem for stem, *_ in OVERLAP_COLUMNS].index("zebra-only")}
                if modality == "ecephys" else ()
            ),
        )
        save_figure(fig, f"{slug(label)}__example-summary")


def _short_unit(group: str, unit_id) -> str:
    """``<probe/plane/DMD>`` over the unit id, a UUID cut to its first 8 characters."""
    uid = str(unit_id)
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f-]{27}", uid):
        uid = uid[:8]
    return f"{str(group).split('__')[0]}\n{uid}"


def plot_top_maps(n: int = N_TOP_GRID) -> None:
    """The ``n`` best units of each selection, one figure per modality.

    Blocks top to bottom are both, Gabor only, then Zebra only, ranked as in the
    overlap examples. In each block the Gabor maps sit above the Zebra maps of
    the same units, one color scale and colorbar per row, framed in the
    selection's pie-chart color. Every map says whether that stimulus passed BH,
    and a passing map carries its Gaussian fit.
    """
    jobs = (
        ("ecephys 830794", "VISp layer 2/3"),
        ("mesoscope 832700", "VISp_0"),
        ("slap2 829704 green", "V1 layer 2/3"),
    )
    inch_per_deg = 0.02
    tick, label_size, title_size = 11, 13, 15
    cbar_w, cbar_pad = 0.16, 0.18
    col_gap, map_gap, block_gap = 0.3, 0.35, 1.25
    margin_l, margin_r, margin_b, margin_t = 1.25, 1.0, 0.75, 1.0
    for label, population in jobs:
        modality, session, keep = SESSIONS[label]
        print(f"\n=== top {n} maps {label} ===")
        try:
            loaded = load_session(modality, session, keep)
            tested = attach_tests(primary_visual(label, loaded), modality, session)
        except SKIPPABLE as exc:
            _skip(f"top-{n} maps, {label}", exc)
            continue
        paired = _both_maps(tested)
        extent = paired.attrs.get("extent") or (-45, 45, -45, 45)
        gabor_w, gabor_h = (span * inch_per_deg for span in _span(extent))
        zebra_w, zebra_h = (span * inch_per_deg for span in _span(WAVEN_EXTENT))
        # Room above each Gabor row for the unit names.
        name_h = 0.45
        block_h = name_h + gabor_h + map_gap + zebra_h
        fig_w = margin_l + n * zebra_w + (n - 1) * col_gap + cbar_pad + cbar_w + margin_r
        fig_h = margin_b + 3 * block_h + 2 * block_gap + margin_t
        fig = plt.figure(figsize=(fig_w, fig_h))

        def add(x, y, w, h):
            return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])

        for b, (stem, pie_name, title, choose, ranker) in enumerate(OVERLAP_COLUMNS):
            pool = paired.loc[choose(paired)] if len(paired) else paired
            selected = ranker(pool, n) if len(pool) else pool
            print(f"  {stem}: {len(pool)} eligible, drawing {len(selected)}")
            color = _PIE_COLORS[pie_name]
            zebra_y = margin_b + (2 - b) * (block_h + block_gap)
            gabor_y = zebra_y + zebra_h + map_gap
            fig.text(
                margin_l / fig_w, (gabor_y + gabor_h + name_h + 0.12) / fig_h,
                f"{title}: top {len(selected)} of {len(pool)}",
                fontsize=title_size, color=color, fontweight="bold", ha="left", va="bottom",
            )
            if not len(selected):
                continue
            rows = [_lookup(paired, r.group, r.unit_id) for r in selected.itertuples(index=False)]
            gabor_vmax = _row_scale([_map_image(r, "rf_map_gabor") for r in rows])
            zebra_vmax = _row_scale([_map_image(r, "rf_map_zebra") for r in rows])
            gabor_im = zebra_im = None
            for col, row in enumerate(rows):
                zx = margin_l + col * (zebra_w + col_gap)
                gabor_ax = add(zx + (zebra_w - gabor_w) / 2, gabor_y, gabor_w, gabor_h)
                zebra_ax = add(zx, zebra_y, zebra_w, zebra_h)
                first = col == 0
                gabor_im = _draw_paired(
                    gabor_ax, _map_image(row, "rf_map_gabor"), extent, "lower", gabor_vmax,
                    xlabel=False, ylabel="Gabor\nElev. (deg)" if first else None,
                )
                zebra_im = _draw_paired(
                    zebra_ax, _map_image(row, "rf_map_zebra"), WAVEN_EXTENT, "upper", zebra_vmax,
                    xlabel=False, ylabel="Zebra\nElev. (deg)" if first else None,
                    faint_zero=True,
                )
                for ax in (gabor_ax, zebra_ax):
                    ax.tick_params(labelsize=tick, length=3)
                    ax.yaxis.label.set_size(label_size)
                    if not first:
                        ax.set_yticklabels([])
                    _paint_selection(ax, color)
                # Fewer azimuth ticks than the paired panels, so neighbours do not collide.
                gabor_ax.set_xticks([-40, 0, 40])
                gabor_ax.set_xticklabels([])
                zebra_ax.set_xticks([-50, 0, 50])
                if b == 2:
                    zebra_ax.set_xlabel("Azimuth (deg)", fontsize=label_size)
                gabor_ax.set_title(
                    f"{col + 1}. {_short_unit(row.group, row.unit_id)}", fontsize=tick, pad=4,
                )
                unit = f"{row.group} / {row.unit_id}"
                for ax, image, mode, passed in (
                    (gabor_ax, _map_image(row, "rf_map_gabor"), "signed", row.get("pass_gabor", False)),
                    (zebra_ax, _map_image(row, "rf_map_zebra"), "abs", row.get("pass_zebra", False)),
                ):
                    _mark_q(ax, bool(passed), fontsize=tick - 1)
                    if bool(passed):
                        _overlay_gaussian(ax, image, paired, extent, mode, unit, verbose=False)
            cbar_x = margin_l + n * zebra_w + (n - 1) * col_gap + cbar_pad
            for im, y, h, name in (
                (gabor_im, gabor_y, gabor_h, "z"), (zebra_im, zebra_y, zebra_h, "r"),
            ):
                cbar = fig.colorbar(im, cax=add(cbar_x, y, cbar_w, h))
                cbar.ax.tick_params(labelsize=tick)
                cbar.set_label(name, fontsize=label_size)
        fig.suptitle(
            f"{label} — {population} — top {n} per selection",
            fontsize=title_size + 3, y=1 - 0.3 / fig_h,
        )
        save_figure(fig, f"{slug(label)}__top-{n}-maps")


def _overlap_counts(df: pd.DataFrame) -> dict[str, int]:
    if not len(df):
        return {name: 0 for name in _PIE_ORDER}
    return {
        "neither": int((~df["pass_zebra"] & ~df["pass_gabor"]).sum()),
        "Zebra only": int((df["pass_zebra"] & ~df["pass_gabor"]).sum()),
        "Gabor only": int((df["pass_gabor"] & ~df["pass_zebra"]).sum()),
        "both": int((df["pass_zebra"] & df["pass_gabor"]).sum()),
    }


_PIE_AREA = {
    "VISp": "V1",
    "VISl": "LM",
    "VISp_0": "V1",
    "DMD1": DMD1_LABEL,
    "DMD2": DMD2_LABEL,
}


def _draw_pie(ax, title: str, counts: dict[str, int]) -> None:
    """Percentages sit in the slices. The legend is the upper right of the pie."""
    from matplotlib.patches import Patch

    present = [name for name in _PIE_ORDER if counts[name] > 0]
    sizes = [counts[name] for name in present]
    colors = [_PIE_COLORS[name] for name in present]
    total = sum(counts.values())
    if total == 0:
        ax.set_axis_off()
        ax.set_title(title, fontsize=SUMMARY_LABEL * 1.3, pad=12)
        return
    _, _, autotexts = ax.pie(
        sizes, colors=colors, startangle=90,
        autopct=lambda pct: f"{pct:.0f}%" if pct >= 6 else "",
        pctdistance=0.62,
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
    )
    for text in autotexts:
        text.set_fontsize(SUMMARY_LABEL)
    handles = [
        Patch(
            facecolor=_PIE_COLORS[name], edgecolor="0.15", linewidth=0.6,
            label=f"{name} ({counts[name]})",
        )
        for name in _PIE_ORDER
    ]
    ax.legend(
        handles=handles, frameon=False, fontsize=SUMMARY_LABEL * 1.3,
        loc="upper left", bbox_to_anchor=(0.98, 1.05), borderaxespad=0,
    )
    ax.set_title(title, fontsize=SUMMARY_LABEL * 1.3, pad=14)


def _save_overlap_pies(modality: str, scope: str, sides, stem: str) -> None:
    """One pie per group. ``sides`` is ``(name, frame)`` pairs."""
    n = len(sides)
    fig, axes = plt.subplots(1, n, figsize=(10.4 * n, 8.8), squeeze=False)
    fig.subplots_adjust(wspace=0.42, top=0.84, bottom=0.04, left=0.04, right=0.97)
    for ax, (name, subset) in zip(axes[0], sides):
        counts = _overlap_counts(subset)
        zebra = counts["Zebra only"] + counts["both"]
        print(f"  {modality} {name} ({scope}): {counts} (n={len(subset)}, zebra {zebra})")
        label = _PIE_AREA.get(name, name)
        # one-line labels start with an empty line, so every title is as tall as
        # the two-line DMD labels and the pies of all modalities line up
        if "\n" not in label:
            label = "\n" + label
        _draw_pie(ax, f"{label} ({len(subset)})", counts)
    note = "" if scope in ("all layers", "all planes", "green") else f", {scope}"
    fig.suptitle(
        f"{modality}{note} — significant at q ≤ {Q_LEVEL}",
        fontsize=SUMMARY_LABEL, y=1.02,
    )
    save_figure(fig, stem)


def plot_overlap_pies() -> None:
    """Two pies per modality: VISp and VISl, or DMD1 and DMD2.

    Ephys pools every layer. Mesoscope pools every plane of that area.
    SLAP2 is the green channel only. A second pair keeps ephys layer 2/3
    and mesoscope plane VISp_0, the populations matched to SLAP2. The units
    are those of the RF summaries (`rf_corpus`).
    """
    print("\n=== overlap pies ===")
    frames = {}
    for label in SUMMARY_LABELS:
        try:
            frames[label] = rf_corpus(label)
        except SKIPPABLE as exc:
            _skip(f"overlap pies, {label}", exc)
    panels = []
    if "ecephys 830794" in frames:
        anatomy = load_ephys_unit_areas()[["unit_id", "layer"]]
        ephys = frames["ecephys 830794"].merge(anatomy, on="unit_id", how="left")
        panels.append(("ecephys", "all layers", (
            ("VISp", ephys[ephys["structure"] == "VISp"]),
            ("VISl", ephys[ephys["structure"] == "VISl"]),
        ), "ecephys__significance-overlap"))
        # SLAP2 is already V1 layer 2/3. These pies hold depth to that population.
        layer23 = ephys[ephys["layer"].astype(str) == "2/3"]
        panels.append(("ecephys", "layer 2/3", (
            ("VISp", layer23[layer23["structure"] == "VISp"]),
            ("VISl", layer23[layer23["structure"] == "VISl"]),
        ), "ecephys__significance-overlap__layer23"))
    if "mesoscope 832700" in frames:
        meso = frames["mesoscope 832700"]
        panels.append(("mesoscope", "all planes", (
            ("VISp", meso[meso["group"].astype(str).str.startswith("VISp_")]),
            ("VISl", meso[meso["group"].astype(str).str.startswith("VISl_")]),
        ), "mesoscope__significance-overlap"))
        panels.append(("mesoscope", "VISp_0", (
            ("VISp_0", meso[meso["group"].astype(str) == "VISp_0"]),
        ), "mesoscope__significance-overlap__VISp_0"))
    if "slap2 829704 green" in frames:
        slap = frames["slap2 829704 green"]
        panels.append(("slap2", "green", (
            ("DMD1", slap[slap["group"].astype(str).str.startswith("DMD1")]),
            ("DMD2", slap[slap["group"].astype(str).str.startswith("DMD2")]),
        ), "slap2__significance-overlap"))
    for modality, scope, sides, stem in panels:
        _save_overlap_pies(modality, scope, sides, stem)


def main(trial: str | int = "pooled") -> None:
    """Every figure; a step that fails is reported and the others still run."""
    set_zebra_block(trial)
    print(f"RESULTS = {RESULTS}")
    print(f"Zebra maps: {ZEBRA_BLOCK}, phase {PHASE}")
    steps = (
        ("RF summaries", lambda: plot_fit_summaries(prepare_fit_summaries())),
        ("ranked maps", plot_ranked_maps),
        ("paired panels", plot_paired_release),
        ("overlap examples", plot_overlap_examples),
        ("example summaries", plot_overlap_example_summaries),
        ("top-10 maps", plot_top_maps),
        ("overlap pies", plot_overlap_pies),
    )
    failed = []
    for name, step in steps:
        try:
            step()
        except Exception:
            print(f"\n{name} failed:")
            traceback.print_exc()
            failed.append(name)
    if SKIPPED:
        print("\nSkipped:")
        for line in SKIPPED:
            print(f"  {line}")
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", default="pooled", choices=["pooled", "0", "1"],
                        help="Zebra maps to plot: 'pooled' (default, both repeats) or one "
                             "repeat, 0 or 1 (the trial_<i> results). Figures of a single "
                             "repeat are named data_release__zebra-trial-<i>__*.png.")
    main(parser.parse_args().trial)
