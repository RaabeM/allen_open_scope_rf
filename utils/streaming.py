from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pynwb
import remfile
import requests
import pandas as pd
from dandi.dandiapi import DandiAPIClient
from tqdm.auto import tqdm

DANDISET_ID = "001637"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_url(asset_id: str) -> str:
    r = requests.get(
        f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
        allow_redirects=True, stream=True, timeout=30,
    )
    url = r.url
    r.close()
    return url


def _detect_modality(nwb: pynwb.NWBFile) -> str:
    if nwb.units is not None:
        return "ecephys"
    ophys = nwb.processing.get("ophys")
    if ophys and any("DMD" in k for k in ophys.data_interfaces):
        return "slap2"
    if ophys or any(k.startswith("VIS") for k in nwb.processing):
        return "mesoscope"
    return "unknown"


def _require_modality(actual: str, expected: str, method: str) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{method}() requires modality='{expected}', got '{actual}'"
        )


# ---------------------------------------------------------------------------
# NWBStream — unified context-managed handle
# ---------------------------------------------------------------------------

@dataclass
class NWBStream:
    """Lazy streaming handle for a DANDI NWB file. Use as a context manager."""

    nwb:      pynwb.NWBFile
    io:       pynwb.NWBHDF5IO
    _h5:      h5py.File
    modality: Literal["ecephys", "mesoscope", "slap2", "unknown"]
    asset_id: str

    def __enter__(self) -> "NWBStream":
        return self

    def __exit__(self, *_) -> None:
        try:
            self.io.close()
        finally:
            self._h5.close()

    # --- Ecephys ---

    def units_df(self, probe: str | None = None) -> pd.DataFrame:
        """Units table as a DataFrame, optionally filtered to one probe.

        Avoids pynwb's to_dataframe() which triggers one HTTP request per
        HDF5 chunk for spike_times and per-unit DynamicTableRegion expansion
        for electrodes. Instead, reads both as single bulk h5py reads. 

        probe: substring matched against electrode group_name, e.g. 'ProbeB'.
        """
        _require_modality(self.modality, "ecephys", "units_df")
        h5u = self._h5["units"]

        # --- scalar columns (small, one read each) ---
        skip = {"spike_times", "spike_times_index", "electrodes", "electrodes_index",
                "waveform_mean", "waveform_sd", "waveform_mean_index", "waveform_sd_index"}
        data = {}
        for col in self.nwb.units.colnames:
            if col in skip:
                continue
            try:
                raw = h5u[col][:]
                data[col] = raw.tolist() if hasattr(raw, "tolist") else list(raw)
            except KeyError:
                pass
        df = pd.DataFrame(data)

        # --- probe column: one bulk read of the electrodes table ---
        elec_h5     = self.nwb.electrodes["group_name"].data.parent
        group_names = np.array(elec_h5["group_name"].asstr()[:])
        elec_flat   = h5u["electrodes"][:]
        elec_bounds = np.concatenate([[0], h5u["electrodes_index"][:]])
        df["probe"] = [
            group_names[elec_flat[elec_bounds[i]:elec_bounds[i + 1]]][0]
            for i in range(len(df))
        ]

        # --- probe filter before loading spike_times ---
        if probe is not None:
            df = df[df["probe"].str.contains(probe, na=False)]

        # preserve original row numbers before reset — used to index spike_bounds
        orig_indices = df.index.tolist()
        df = df.reset_index(drop=True)

        # --- spike_times: two bulk reads, reconstruct ragged array ---
        spike_flat   = h5u["spike_times"][:]
        spike_bounds = np.concatenate([[0], h5u["spike_times_index"][:]])
        df["spike_times"] = [
            spike_flat[spike_bounds[i]:spike_bounds[i + 1]]
            for i in orig_indices
        ]

        return df

    def zebra_df(self) -> pd.DataFrame:
        """Zebra noise stimulus presentation table (start_time, stop_time, ...)."""
        _require_modality(self.modality, "ecephys", "zebra_df")
        return self.nwb.intervals["Zebra_presentations"].to_dataframe()

    def gabor_rf_df(self) -> pd.DataFrame:
        """Gabor RF mapping stimulus table (X, Y, Orientation, start_time, ...)."""
        _require_modality(self.modality, "ecephys", "gabor_rf_df")
        return self.nwb.intervals["RF mapping_presentations"].to_dataframe()

    # --- Mesoscope ---

    def imaging_planes(self) -> list[str]:
        """Names of processing groups that represent imaging planes (e.g. VISp_0)."""
        _require_modality(self.modality, "mesoscope", "imaging_planes")
        return [k for k in self.nwb.processing if k.startswith("VIS")]

    def dff_df(self, plane: str) -> pd.DataFrame:
        """ΔF/F traces for a given imaging plane.

        Returns a DataFrame with shape (n_rois, n_timepoints).
        plane: one of the names returned by imaging_planes().
        """
        _require_modality(self.modality, "mesoscope", "dff_df")
        module = self.nwb.processing[plane]
        dff_ts = module["dff_timeseries"]["dff_timeseries"]
        return pd.DataFrame(
            dff_ts.data[:].T,
            columns=dff_ts.timestamps[:],
        )

    # --- SLAP2 ---

    def slap2_dff(self) -> pd.DataFrame:
        raise NotImplementedError("SLAP2 NWBs not yet on DANDI")


# ---------------------------------------------------------------------------
# DandiSession — asset discovery and opening
# ---------------------------------------------------------------------------

class DandiSession:
    """Lists and opens NWB assets from a DANDI dandiset."""

    def __init__(self, dandiset_id: str = DANDISET_ID) -> None:
        self._dandiset_id = dandiset_id
        client = DandiAPIClient()
        self._ds = client.get_dandiset(dandiset_id)

    def assets(self):
        """All assets sorted by path. Each entry has .identifier, .path, .size."""
        return sorted(self._ds.get_assets(), key=lambda a: a.path)

    def open(self, asset_id: str) -> NWBStream:
        """Resolve DANDI redirect → remfile → h5py → pynwb → NWBStream."""
        url = _resolve_url(asset_id)
        h5  = h5py.File(remfile.File(url), "r")
        io  = pynwb.NWBHDF5IO(file=h5, load_namespaces=True, mode="r")
        nwb = io.read()
        return NWBStream(
            nwb=nwb,
            io=io,
            _h5=h5,
            modality=_detect_modality(nwb),
            asset_id=asset_id,
        )

    def download(self, asset_id: str, dest: str | Path) -> Path:
        """Download a DANDI asset to a local file via the DANDI API. Returns the saved path."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        asset = self._ds.get_asset(asset_id)
        downloader = asset.get_download_file_iter()
        with dest.open("wb") as f, tqdm(
            total=asset.size, unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            for chunk in downloader(0):
                f.write(chunk)
                bar.update(len(chunk))
        return dest


# ---------------------------------------------------------------------------
# Public shortcut
# ---------------------------------------------------------------------------

def open_nwb(asset_id: str, dandiset_id: str = DANDISET_ID) -> NWBStream:
    """Stream a DANDI NWB file by asset ID without downloading it."""
    return DandiSession(dandiset_id).open(asset_id)


def open_local(path: str | Path, asset_id: str = "") -> NWBStream:
    """Open a locally saved NWB file as an NWBStream.

    Identical interface to open_nwb() — all NWBStream methods work the same.
    asset_id is optional metadata; leave blank if the file wasn't from DANDI.
    """
    h5  = h5py.File(Path(path), "r")
    io  = pynwb.NWBHDF5IO(file=h5, load_namespaces=True, mode="r")
    nwb = io.read()
    return NWBStream(
        nwb=nwb,
        io=io,
        _h5=h5,
        modality=_detect_modality(nwb),
        asset_id=asset_id,
    )
