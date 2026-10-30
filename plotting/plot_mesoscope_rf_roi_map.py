#!/usr/bin/env python3
# Author: Marcel C. Raabe
# Date: 2026-10-29
"""Interactive map of the mesoscope ROIs, coloured by their receptive field.

Draws the segmented ROIs of every imaging plane of the mesoscope session of
``plot_data_release_panels.py`` (``mesoscope 832700``) over the plane's average
projection, and fills each ROI by a feature chosen from a drop-down menu:

* Zebra RF centre azimuth / elevation, RF size, orientation and peak |r|
* local-Gabor RF centre azimuth / elevation, RF size, orientation and peak |z|
* RF type: significant on both stimuli, only Zebra, only Gabor, or neither

A second menu picks the plane. Hovering an ROI shows its name and the value of
the selected feature. The page is one self-contained HTML file.

Significance and RF centre / size come from ``rf_corpus`` of
``plot_data_release_panels.py``, the units of the RF summaries and overlap pies:
BH q ≤ 0.05 per stimulus, and the Gaussian fit of each significant map (centre,
and one-sigma ellipse area as the size). Peak |r| (``abs_max_value``) and peak
|z| (``z_max_all``) come from ``load_session`` for every unit with results.
ROI masks and average projections are read from the session NWB; ROI ``i`` of a
plane's ``roi_table`` is unit ``i`` of that plane.

ROIs missing from ``rf_corpus`` are drawn as "excluded" (Gabor significant, but
the fit failed or its centre lies beyond ±40°) or "untested" (no results).

    python plotting/plot_mesoscope_rf_roi_map.py
    python plotting/plot_mesoscope_rf_roi_map.py --nwb <session .nwb> --out <page .html>
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import zlib
from pathlib import Path

import h5py
import numpy as np

from plot_data_release_panels import (
    FILE_PREFIX,
    PLOT_DIR,
    REPO,
    SESSIONS,
    load_session,
    rf_corpus,
)

LABEL = "mesoscope 832700"
MODALITY, SESSION, KEEP = SESSIONS[LABEL]
NWB = (
    REPO.parents[1] / "rawdata" / "allen_open_scope" / "meso"
    / SESSION.split("_")[0] / f"{SESSION}.nwb"
)
OUTPUT = PLOT_DIR / f"{FILE_PREFIX}__mesoscope__rf-roi-map.html"
# Colour limits of the sequential features: these percentiles over the session.
CLIP_PERCENTILES = (1.0, 99.0)
# Robust stretch of the average projection behind the ROIs.
PROJECTION_CLIP = (1.0, 99.5)
AREA_ORDER = ("VISp", "VISl")

# RF type codes; the colours are those of the significance-overlap pies.
RF_TYPES = (
    ("both", "Both significant", "#54A24B"),
    ("zebra", "Only Zebra", "#4C78A8"),
    ("gabor", "Only Gabor", "#E45756"),
    ("none", "None", "#c8c8c8"),
    ("excluded", "Excluded (Gabor fit failed or centre beyond ±40°)", "#7a7f7d"),
    ("untested", "Untested", "#4a4f4d"),
)
# (key, menu group, label, unit, scale, digits); scale "diverging" is centred on 0.
FEATURES = (
    ("rf_type", "Both stimuli", "RF type", "", "category", 0),
    ("zebra_azimuth", "Zebra noise", "RF centre azimuth", "°", "diverging", 1),
    ("zebra_elevation", "Zebra noise", "RF centre elevation", "°", "diverging", 1),
    ("zebra_size", "Zebra noise", "RF size", "deg²", "sequential", 0),
    ("zebra_orientation", "Zebra noise", "RF orientation", "°", "cyclic", 0),
    ("zebra_peak", "Zebra noise", "Peak |r|", "", "sequential", 3),
    ("gabor_azimuth", "Local Gabors", "RF centre azimuth", "°", "diverging", 1),
    ("gabor_elevation", "Local Gabors", "RF centre elevation", "°", "diverging", 1),
    ("gabor_size", "Local Gabors", "RF size", "deg²", "sequential", 0),
    ("gabor_orientation", "Local Gabors", "RF orientation", "°", "cyclic", 0),
    ("gabor_peak", "Local Gabors", "Peak |z|", "", "sequential", 2),
)


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _finite(value, digits: int = 4) -> float | None:
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


# ---------------------------------------------------------------------------
# ROI masks and projections, as the openscope_p3 extract_mesoscope_plane_snr.py
# ---------------------------------------------------------------------------


def mask_runs(mask: np.ndarray) -> list[int]:
    """Flatten a boolean ROI mask to [row, col_start, length, ...] runs."""
    runs: list[int] = []
    for row in np.flatnonzero(mask.any(axis=1)):
        cols = np.flatnonzero(mask[row])
        breaks = np.flatnonzero(np.diff(cols) > 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [cols.size - 1]))
        for s, e in zip(starts, ends, strict=True):
            runs.extend((int(row), int(cols[s]), int(cols[e] - cols[s] + 1)))
    return runs


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def encode_gray_png(image: np.ndarray) -> bytes:
    """8-bit grayscale PNG, written with zlib alone."""
    arr = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = arr.shape
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )


def normalize_projection(projection: np.ndarray) -> np.ndarray:
    arr = np.asarray(projection, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[finite], PROJECTION_CLIP)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.rint((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)


def plane_depth(nwb: h5py.File, plane: str) -> tuple[str, float | None]:
    """Area and depth (µm) from ``Structure: VISp Depth: 150``."""
    location = _text(nwb[f"general/optophysiology/{plane}/location"][()])
    match = re.search(r"Structure:\s*(\S+)\s+Depth:\s*([\d.]+)", location)
    if not match:
        return plane.split("_")[0], None
    return match.group(1), float(match.group(2))


def plane_order(plane: dict) -> tuple:
    """VISp first, then ascending depth within area."""
    area = plane["structure"]
    rank = AREA_ORDER.index(area) if area in AREA_ORDER else len(AREA_ORDER)
    depth = plane["depthUm"]
    return (rank, area, float("inf") if depth is None else depth, plane["plane"])


# ---------------------------------------------------------------------------
# Features per ROI
# ---------------------------------------------------------------------------


def roi_features() -> tuple[dict, dict]:
    """RF type and feature values per (plane, unit id), and the peaks of every tested unit.

    Returns ``(rows, tested)``: ``rows`` for the units of ``rf_corpus``, ``tested``
    the peaks of every unit with results (``load_session``).
    """
    corpus = rf_corpus(LABEL)
    loaded = load_session(MODALITY, SESSION, KEEP)
    tested = {
        (g, str(u)): (float(r), float(z))
        for g, u, r, z in zip(
            loaded["group"], loaded["unit_id"], loaded["abs_max_value"], loaded["z_max_all"],
            strict=True,
        )
    }
    rows = {}
    for row in corpus.itertuples(index=False):
        zebra, gabor = bool(row.pass_zebra), bool(row.pass_gabor)
        rows[(row.group, str(row.unit_id))] = {
            "rf_type": (
                "both" if zebra and gabor else "zebra" if zebra else "gabor" if gabor else "none"
            ),
            "zebra_azimuth": row.zebra_x0,
            "zebra_elevation": row.zebra_y0,
            "zebra_size": row.zebra_area,
            # best filter of the sweep; an orientation only where the map is significant
            "zebra_orientation": row.theta_deg if zebra else np.nan,
            "gabor_azimuth": row.gabor_x0,
            "gabor_elevation": row.gabor_y0,
            "gabor_size": row.gabor_area,
            # orientation of the Gabor map with the lowest p (ties: largest |z|)
            "gabor_orientation": np.degrees(row.ori_rad) if gabor else np.nan,
        }
    return rows, tested


def build_plane(nwb: h5py.File, plane: str, rows: dict, tested: dict) -> dict:
    module = nwb[f"processing/{plane}"]
    roi_table = module["image_segmentation/roi_table"]
    roi_ids = roi_table["id"][:]
    if not np.array_equal(roi_ids, np.arange(len(roi_ids))):
        raise ValueError(f"{plane}: ROI ids are not the row index; unit ids would not match")

    masks = roi_table["image_mask"]
    runs, offsets = [], [0]
    for i in range(len(roi_ids)):
        runs.extend(mask_runs(np.asarray(masks[i]) > 0))
        offsets.append(len(runs))
    height, width = masks.shape[1:]

    values = {key: [] for key, *_ in FEATURES if key != "rf_type"}
    types = []
    for roi in roi_ids:
        key = (plane, str(int(roi)))
        row = rows.get(key, {})
        peak_r, peak_z = tested.get(key, (np.nan, np.nan))
        types.append(row.get("rf_type", "excluded" if key in tested else "untested"))
        for name in values:
            if name.endswith("_peak"):
                value = peak_r if name == "zebra_peak" else peak_z
            else:
                value = row.get(name, np.nan)
            values[name].append(_finite(value, 2 if name.endswith("_size") else 4))

    structure, depth = plane_depth(nwb, plane)
    return {
        "plane": plane,
        "structure": structure,
        "depthUm": depth,
        "width": int(width),
        "height": int(height),
        "projectionPng": base64.b64encode(
            encode_gray_png(normalize_projection(module["images/average_projection"][:]))
        ).decode("ascii"),
        "maskRunsBase64": base64.b64encode(np.asarray(runs, dtype="<u2").tobytes()).decode(),
        "maskRunOffsets": offsets,
        "rfType": types,
        "values": values,
    }


def feature_scales(planes: list[dict]) -> list[dict]:
    """Colour limits per feature, shared by all planes so a colour means one value."""
    out = []
    for key, group, label, unit, scale, digits in FEATURES:
        entry = {"key": key, "group": group, "label": label, "unit": unit,
                 "scale": scale, "digits": digits}
        if scale == "cyclic":
            # orientation is periodic in 180°, so the scale spans one period
            entry["lo"], entry["hi"] = 0.0, 180.0
        elif scale != "category":
            pooled = np.array([
                v for plane in planes for v in plane["values"][key] if v is not None
            ])
            if scale == "diverging":
                limit = float(np.percentile(np.abs(pooled), CLIP_PERCENTILES[1]))
                entry["lo"], entry["hi"] = -limit, limit
            else:
                lo, hi = np.percentile(pooled, CLIP_PERCENTILES)
                entry["lo"], entry["hi"] = float(lo), float(hi)
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nwb", type=Path, default=NWB, help=f"session NWB (default: {NWB})")
    parser.add_argument("--out", type=Path, default=OUTPUT, help=f"page (default: {OUTPUT})")
    args = parser.parse_args()

    rows, tested = roi_features()
    planes = []
    with h5py.File(args.nwb, "r") as nwb:
        names = sorted(k for k in nwb["processing"] if re.match(r"VIS[a-z]+_\d+$", k))
        for name in names:
            plane = build_plane(nwb, name, rows, tested)
            counts = {code: plane["rfType"].count(code) for code, *_ in RF_TYPES}
            print(f"  {name}: {len(plane['rfType'])} ROIs  {counts}")
            planes.append(plane)

    payload = {
        "session": SESSION,
        "clipPercentiles": CLIP_PERCENTILES,
        "rfTypes": [{"code": c, "label": label, "color": color} for c, label, color in RF_TYPES],
        "features": feature_scales(planes),
        "planes": sorted(planes, key=plane_order),
    }
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    html = TEMPLATE.replace("__DATA__", data.replace("</", "<\\/"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8", newline="\n")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mesoscope ROI receptive fields</title>
<style>
:root {
  color-scheme: light;
  --accent: #167f8c;
  --ink: #293133;
  --muted: #68716f;
  --line: #d3d8d6;
  --paper: #fff;
  --wash: #f4f7f6;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 16px; background: var(--paper); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.viewer { max-width: 760px; margin: 0 auto; }
h1 { margin: 0 0 2px; font-size: 18px; font-weight: 650; }
.sub { margin: 0 0 12px; color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; margin-bottom: 12px; }
.controls label { display: flex; flex-direction: column; gap: 3px;
  font-size: 12px; color: var(--muted); font-weight: 600; }
select {
  min-width: 200px; padding: 6px 8px; border: 1px solid var(--line); border-radius: 4px;
  background: var(--paper); color: var(--ink); font: inherit;
}
select:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.check { flex-direction: row !important; align-items: center; align-self: end;
  padding-bottom: 7px; color: var(--ink) !important; font-weight: 500 !important; }
.canvas-wrap { position: relative; width: 100%; max-width: 640px; }
canvas {
  display: block; width: 100%; height: auto; border: 1px solid var(--line);
  border-radius: 4px; image-rendering: pixelated; cursor: crosshair; background: #000;
}
.tip {
  position: absolute; pointer-events: none; padding: 5px 8px; border-radius: 4px;
  background: rgba(41, 49, 51, 0.94); color: #fff; font-size: 12.5px; line-height: 1.35;
  white-space: nowrap; transform: translate(12px, 12px);
}
.tip b { font-weight: 650; }
.legend { margin-top: 10px; max-width: 640px; }
.legend-title { font-weight: 600; margin-bottom: 4px; }
.bar { height: 12px; border-radius: 2px; border: 1px solid var(--line); }
.ticks { display: flex; justify-content: space-between; color: var(--muted);
  font-size: 12px; font-variant-numeric: tabular-nums; }
.swatches { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12.5px; }
.swatch { display: inline-flex; align-items: center; gap: 6px; }
.swatch i { width: 12px; height: 12px; border-radius: 2px; display: inline-block;
  border: 1px solid rgba(0, 0, 0, 0.2); }
.n { color: var(--muted); font-variant-numeric: tabular-nums; }
.note { margin: 10px 0 0; max-width: 640px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<main class="viewer">
  <h1>Mesoscope ROI receptive fields</h1>
  <p class="sub" id="subtitle"></p>
  <div class="controls">
    <label>Imaging plane <select id="plane"></select></label>
    <label>Colour ROIs by <select id="feature"></select></label>
    <label class="check"><input type="checkbox" id="projection" checked>
      Average projection</label>
  </div>
  <div class="canvas-wrap">
    <canvas id="map"
      aria-label="Segmented ROIs of the selected plane, filled by the selected feature"></canvas>
    <div class="tip" id="tip" hidden></div>
  </div>
  <div class="legend" id="legend"></div>
  <p class="note" id="note"></p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
// Viridis from 0.1 up: its darkest purple vanishes on the dimmed projection.
const VIRIDIS = [[72, 36, 117], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
const DIVERGING = [[33, 102, 172], [103, 169, 207], [240, 240, 240], [239, 138, 98], [178, 24, 43]];
// OKLCH colour to sRGB, clipped to the gamut.
function oklch(L, C, hue) {
  const h = hue * Math.PI / 180, a = C * Math.cos(h), b = C * Math.sin(h);
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  ].map((c) => {
    c = Math.min(1, Math.max(0, c));
    return Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055));
  });
}
// Orientation: one turn of hue at constant lightness, so 0° and 180° meet.
const CYCLIC = Array.from({ length: 37 }, (_, i) => oklch(0.7, 0.13, 30 + 10 * i));
const EMPTY = [150, 150, 150];
const STOPS = { diverging: DIVERGING, sequential: VIRIDIS, cyclic: CYCLIC };
const TYPE = Object.fromEntries(DATA.rfTypes.map((t) => [t.code, t]));
const FEATURE = Object.fromEntries(DATA.features.map((f) => [f.key, f]));

const planeSelect = document.getElementById("plane");
const featureSelect = document.getElementById("feature");
const projectionBox = document.getElementById("projection");
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const tip = document.getElementById("tip");

const state = { plane: null, feature: "rf_type", hover: -1 };

function hexRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function ramp(stops, t) {
  t = Math.min(1, Math.max(0, t));
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const w = x - i;
  return stops[i].map((c, k) => Math.round(c + w * (stops[i + 1][k] - c)));
}
function cssGradient(stops) {
  return `linear-gradient(to right, ${stops.map((s) => `rgb(${s.join(",")})`).join(", ")})`;
}
function fmt(value, digits) {
  return value.toFixed(digits).replace(/^-(0\.?0*)$/, "$1");
}

// Decode each plane once: a label image (ROI index + 1 per pixel) and the projection.
function preparePlane(plane) {
  if (plane.labels) return Promise.resolve(plane);
  const bytes = Uint8Array.from(atob(plane.maskRunsBase64), (c) => c.charCodeAt(0));
  const runs = new Uint16Array(bytes.buffer);
  const labels = new Int32Array(plane.width * plane.height);
  const n = plane.rfType.length;
  for (let roi = 0; roi < n; roi++) {
    for (let r = plane.maskRunOffsets[roi]; r < plane.maskRunOffsets[roi + 1]; r += 3) {
      const start = runs[r] * plane.width + runs[r + 1];
      labels.fill(roi + 1, start, start + runs[r + 2]);
    }
  }
  plane.runs = runs;
  plane.labels = labels;
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const off = document.createElement("canvas");
      off.width = plane.width; off.height = plane.height;
      const octx = off.getContext("2d");
      octx.drawImage(img, 0, 0);
      const rgba = octx.getImageData(0, 0, plane.width, plane.height).data;
      plane.gray = new Uint8Array(plane.width * plane.height);
      for (let i = 0; i < plane.gray.length; i++) plane.gray[i] = rgba[i * 4];
      resolve(plane);
    };
    img.src = `data:image/png;base64,${plane.projectionPng}`;
  });
}

function roiColors(plane, key) {
  const feature = FEATURE[key];
  if (feature.scale === "category") return plane.rfType.map((code) => hexRgb(TYPE[code].color));
  const stops = STOPS[feature.scale];
  return plane.values[key].map((v) =>
    v === null ? EMPTY : ramp(stops, (v - feature.lo) / (feature.hi - feature.lo)));
}

function draw() {
  const plane = state.plane;
  const { width, height, labels, gray } = plane;
  const colors = roiColors(plane, state.feature);
  const showProjection = projectionBox.checked;
  const image = ctx.createImageData(width, height);
  const px = image.data;
  for (let i = 0; i < labels.length; i++) {
    const g = showProjection ? gray[i] : 0;
    const roi = labels[i] - 1;
    let r = g * 0.6, gg = g * 0.6, b = g * 0.6;
    if (roi >= 0) {
      const c = colors[roi];
      r = 0.85 * c[0] + 0.15 * g; gg = 0.85 * c[1] + 0.15 * g; b = 0.85 * c[2] + 0.15 * g;
    }
    px[i * 4] = r; px[i * 4 + 1] = gg; px[i * 4 + 2] = b; px[i * 4 + 3] = 255;
  }
  if (state.hover >= 0) outline(plane, state.hover, px);
  ctx.putImageData(image, 0, 0);
}

// White edge around the hovered ROI: its pixels with a 4-neighbour outside it.
function outline(plane, roi, px) {
  const { width, height, labels, runs, maskRunOffsets } = plane;
  const id = roi + 1;
  for (let r = maskRunOffsets[roi]; r < maskRunOffsets[roi + 1]; r += 3) {
    const y = runs[r];
    for (let x = runs[r + 1]; x < runs[r + 1] + runs[r + 2]; x++) {
      const i = y * width + x;
      const edge = x === 0 || x === width - 1 || y === 0 || y === height - 1
        || labels[i - 1] !== id || labels[i + 1] !== id
        || labels[i - width] !== id || labels[i + width] !== id;
      if (edge) { px[i * 4] = 255; px[i * 4 + 1] = 255; px[i * 4 + 2] = 255; }
    }
  }
}

function valueText(plane, roi) {
  const key = state.feature;
  const feature = FEATURE[key];
  const code = plane.rfType[roi];
  if (feature.scale === "category") return TYPE[code].label;
  const value = plane.values[key][roi];
  if (value !== null) {
    const unit = !feature.unit || feature.unit === "°" ? feature.unit : ` ${feature.unit}`;
    return `${fmt(value, feature.digits)}${unit}`;
  }
  if (code === "untested") return "not tested";
  if (code === "excluded" && key.startsWith("gabor")) {
    return "excluded (Gabor fit failed or centre beyond ±40°)";
  }
  const stimulus = key.startsWith("zebra") ? "Zebra" : "Gabor";
  const significant = code === "both" || code === stimulus.toLowerCase();
  return significant ? "Gaussian fit failed" : `no significant ${stimulus} RF`;
}

function featureName(key) {
  const f = FEATURE[key];
  if (f.scale === "category") return f.label;
  return `${f.group === "Zebra noise" ? "Zebra" : "Gabor"} ${f.label}`;
}

function legend() {
  const plane = state.plane;
  const feature = FEATURE[state.feature];
  const box = document.getElementById("legend");
  const note = document.getElementById("note");
  if (feature.scale === "category") {
    const counts = {};
    plane.rfType.forEach((c) => { counts[c] = (counts[c] || 0) + 1; });
    const swatches = DATA.rfTypes.filter((t) => counts[t.code]).map((t) =>
      `<span class="swatch"><i style="background:${t.color}"></i>${t.label}`
      + ` <span class="n">${counts[t.code]}</span></span>`).join("");
    box.innerHTML = '<div class="legend-title">RF type (BH q ≤ 0.05 per stimulus)</div>'
      + `<div class="swatches">${swatches}</div>`;
    note.textContent = "Significance per stimulus: Zebra, Gumbel null of the sweep-wide "
      + "maximum |r|; "
      + "local Gabors, unit-level Gumbel p after the search over windows and orientations. "
      + "Both Benjamini–Hochberg corrected over the session.";
    return;
  }
  const values = plane.values[state.feature];
  const missing = values.filter((v) => v === null).length;
  const cyclic = feature.scale === "cyclic";
  const unit = feature.unit ? ` (${feature.unit})` : "";
  const mid = (feature.lo + feature.hi) / 2;
  const ticks = cyclic
    ? ["0", "45", "90", "135", "180"]
    : [`≤ ${fmt(feature.lo, feature.digits)}`, fmt(mid, feature.digits),
      `≥ ${fmt(feature.hi, feature.digits)}`];
  box.innerHTML = `<div class="legend-title">${featureName(state.feature)}${unit}</div>
    <div class="bar" style="background:${cssGradient(STOPS[feature.scale])}"></div>
    <div class="ticks">${ticks.map((t) => `<span>${t}</span>`).join("")}</div>
    <div class="swatches" style="margin-top:6px"><span class="swatch">
      <i style="background:rgb(${EMPTY.join(",")})"></i>No value
      <span class="n">${missing} of ${values.length}</span></span></div>`;
  if (cyclic) {
    note.textContent = (state.feature === "zebra_orientation"
      ? "Orientation of the best filter of the Zebra wavelet library (10 orientations in "
        + "18° steps). "
      : "Orientation of the local-Gabor map with the lowest p-value, ties broken by the "
        + "largest |z| (3 orientations: 0°, 45°, 90°). ")
      + "Shown for ROIs with a significant RF on this stimulus, in that stimulus's own "
      + "angle convention. Cyclic colour scale: 0° and 180° are the same orientation.";
    return;
  }
  const limits = feature.scale === "diverging"
    ? `symmetric about 0 at the ${DATA.clipPercentiles[1]}th percentile of |value|`
    : `${DATA.clipPercentiles[0]}st–${DATA.clipPercentiles[1]}th percentile`;
  const definition = state.feature.endsWith("_peak")
    ? (state.feature === "zebra_peak"
      ? "Peak |r|: largest |Pearson r| over the Zebra filter bank and delay × duration "
        + "sweep (every tested ROI)."
      : "Peak |z|: largest |z| of the local-Gabor maps over orientations and positions "
        + "(every tested ROI).")
    : "Centre and size of a 2D Gaussian fit to the significant map (size: one-sigma "
      + "ellipse area, π σx σy); azimuth positive to the right, elevation positive upwards. "
      + "ROIs without a significant RF on this stimulus have no value.";
  note.textContent = `${definition} Colour scale shared by all planes, clipped ${limits}.`;
}

function subtitle() {
  const p = state.plane;
  const depth = p.depthUm === null ? "" : ` · ${Math.round(p.depthUm)} µm`;
  document.getElementById("subtitle").textContent =
    `${DATA.session.split("_")[0]} · ${p.plane} (${p.structure}${depth}) · ${p.rfType.length} ROIs`;
}

function update() {
  subtitle();
  legend();
  draw();
}

async function selectPlane(index) {
  const plane = await preparePlane(DATA.planes[index]);
  state.plane = plane;
  state.hover = -1;
  tip.hidden = true;
  canvas.width = plane.width;
  canvas.height = plane.height;
  update();
}

function onMove(event) {
  const plane = state.plane;
  if (!plane) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / rect.width * plane.width);
  const y = Math.floor((event.clientY - rect.top) / rect.height * plane.height);
  const inside = x >= 0 && y >= 0 && x < plane.width && y < plane.height;
  const roi = inside ? plane.labels[y * plane.width + x] - 1 : -1;
  if (roi !== state.hover) { state.hover = roi; draw(); }
  if (roi < 0) { tip.hidden = true; return; }
  tip.innerHTML = `<b>${plane.plane} · ROI ${roi}</b><br>`
    + `${featureName(state.feature)}: ${valueText(plane, roi)}`;
  tip.style.left = `${event.clientX - rect.left}px`;
  tip.style.top = `${event.clientY - rect.top}px`;
  tip.hidden = false;
  // Keep the tooltip inside the map near the right edge.
  const over = tip.offsetLeft + 12 + tip.offsetWidth - canvas.parentElement.clientWidth;
  if (over > 0) tip.style.left = `${event.clientX - rect.left - over - 16}px`;
}

DATA.planes.forEach((p, i) => {
  const depth = p.depthUm === null ? "" : ` · ${Math.round(p.depthUm)} µm`;
  planeSelect.add(new Option(`${p.plane} (${p.structure}${depth}, ${p.rfType.length} ROIs)`, i));
});
const groups = {};
DATA.features.forEach((f) => {
  if (!groups[f.group]) {
    groups[f.group] = document.createElement("optgroup");
    groups[f.group].label = f.group;
    featureSelect.append(groups[f.group]);
  }
  groups[f.group].append(new Option(f.label, f.key));
});
featureSelect.value = state.feature;

planeSelect.addEventListener("change", () => selectPlane(Number(planeSelect.value)));
featureSelect.addEventListener("change", () => { state.feature = featureSelect.value; update(); });
projectionBox.addEventListener("change", draw);
canvas.addEventListener("mousemove", onMove);
canvas.addEventListener("mouseleave", () => {
  tip.hidden = true;
  if (state.hover >= 0) { state.hover = -1; draw(); }
});
selectPlane(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
