"""Paper figure assembled from the data-release panels of each modality.

Columns are the modalities (Neuropixels, Mesoscope, SLAP2 glutamate), rows are
the panels written by ``plot_data_release_panels.py``:

* A  example summary       ``<prefix>__<session label>__example-summary.png``
* B  significance overlap  ``<prefix>__<modality>__significance-overlap.png``
* C  RF summary            ``<prefix>__<modality>__rf-summary.png``

Left of rows A and C, an example of each stimulus sits beside the sub-row of
its maps: a Gabor patch next to the Gabor maps (top) and a Zebra movie frame
next to the Zebra maps (bottom). The Gabor patch is drawn from the parameters
of the RF mapping table (20 deg aperture, 0.08 cycles/deg, contrast 0.8, on
mean grey); the Zebra frame is read from ``waven_settings.movpath``, or from
the local copy in ``rawdata/`` when that path is not mounted. Both show the
whole screen, ``visual_coverage``.

The panels are read from ``plots/`` as PNGs, so run
``plot_data_release_panels.py`` first. Each panel's own suptitle (session or
modality and the q threshold) repeats the column header and is cropped off
unless ``--keep-titles`` is given.

    python plotting/plot_paper_figure.py
    python plotting/plot_paper_figure.py --prefix data_release__zebra-trial-0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO / "scripts"))

sys.path.append(str(Path(__file__).resolve().parent))

from plot_data_release_panels import PAIRED_INCH_PER_DEG, PAIRED_LABEL_SIZE, SAVE_DPI, WAVEN_EXTENT
from waven_settings import movpath, visual_coverage

PLOT_DIR = REPO / "plots"
PREFIX = "data_release"
# local copy of the Zebra movie, used when movpath is not mounted
LOCAL_MOVIE = REPO.parents[1] / "rawdata" / "allen_open_scope" / "stimulus" / "zebra" / Path(movpath).name
ZEBRA_FRAME = 1000
# example Gabor patch, parameters of the "RF mapping" stimulus table
GABOR_X, GABOR_Y = -20.0, -20.0     # deg, a grid position near the example RFs
GABOR_ORIENTATION = np.pi / 4       # rad
GABOR_SF = 0.08                     # cycles/deg
GABOR_DIAMETER = 20.0               # deg
GABOR_CONTRAST = 0.8
# rows that get the stimulus examples beside their Gabor (top) and Zebra (bottom) sub-rows
STIMULUS_ROWS = ("A", "C")
STIMULUS_WIDTH = 0.3  # width of the stimulus column relative to a modality column
# the stimuli take the size of the Zebra maps and the font of the column titles
# in the example summaries (row SCALE_ROW), as they appear in this figure
SCALE_ROW = "A"
GABOR_MAP_DEG = 90  # azimuth span of the Gabor maps, the RF mapping grid -45..45 deg
# rows of a map frame broken by less than this (pixels, e.g. by the q label) stay one frame
MAP_GAP = 150

# column header -> (session label slug, modality), as in plot_data_release_panels.SESSIONS
COLUMNS = {
    "Neuropixels": ("ecephys-830794", "ecephys"),
    "Mesoscope": ("mesoscope-832700", "mesoscope"),
    "SLAP2 glutamate": ("slap2-829704-green", "slap2"),
}
# row letter -> file stem, given (session label slug, modality)
ROWS = {
    "A": lambda label, modality: f"{label}__example-summary",
    "B": lambda label, modality: f"{modality}__significance-overlap",
    "C": lambda label, modality: f"{modality}__rf-summary",
}

COLUMN_WIDTH = 7.0  # inches per modality column
TITLE_SIZE = 20
LETTER_SIZE = 26
# a pixel counts as content when any channel is darker than this (0-1 scale)
WHITE = 0.97
# a white band at least this tall (pixels) separates the suptitle from the panel
TITLE_GAP = 20


def _content_rows(image: np.ndarray) -> np.ndarray:
    rgb = image[..., :3]
    return (rgb < WHITE).any(axis=(1, 2))


def trim_whitespace(image: np.ndarray, pad: int = 10) -> np.ndarray:
    """Crop the white margin around the content, leaving ``pad`` pixels."""
    rgb = image[..., :3] < WHITE
    rows = np.flatnonzero(rgb.any(axis=(1, 2)))
    cols = np.flatnonzero(rgb.any(axis=(0, 2)))
    if rows.size == 0:
        return image
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, image.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, image.shape[1])
    return image[r0:r1, c0:c1]


def drop_suptitle(image: np.ndarray) -> np.ndarray:
    """Remove the first line of content when a white band separates it from the rest.

    The data-release panels all start with a one-line suptitle followed by a
    gap; the panel content begins below that gap.
    """
    content = _content_rows(image)
    rows = np.flatnonzero(content)
    if rows.size == 0:
        return image
    # end of the first run of content rows
    first = rows[0]
    end = first + np.argmin(content[first:]) if not content[first:].all() else image.shape[0]
    following = np.flatnonzero(content[end:])
    if following.size == 0 or following[0] < TITLE_GAP:
        return image
    return image[end + following[0]:]


def subrow_split(image: np.ndarray) -> float:
    """Row (pixels) of the widest white band in the middle of a two-row panel."""
    content = _content_rows(image)
    lo, hi = int(0.3 * len(content)), int(0.7 * len(content))
    best, start, best_mid = 0, None, len(content) / 2
    for i in range(lo, hi + 1):
        if i < hi and not content[i]:
            start = i if start is None else start
        elif start is not None:
            if i - start > best:
                best, best_mid = i - start, (start + i) / 2
            start = None
    return best_mid


def zebra_frame(index: int = ZEBRA_FRAME) -> np.ndarray:
    path = Path(movpath) if Path(movpath).exists() else LOCAL_MOVIE
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise OSError(f"cannot read frame {index} of {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0


def gabor_frame(shape: tuple[int, int]) -> np.ndarray:
    """One Gabor patch on mean grey, over the whole screen at ``shape`` pixels."""
    x0, x1, y0, y1 = visual_coverage
    az, el = np.meshgrid(np.linspace(x0, x1, shape[1]), np.linspace(y0, y1, shape[0]))
    dx, dy = az - GABOR_X, el - GABOR_Y
    along = dx * np.cos(GABOR_ORIENTATION) + dy * np.sin(GABOR_ORIENTATION)
    grating = GABOR_CONTRAST * np.cos(2 * np.pi * GABOR_SF * along)
    inside = np.hypot(dx, dy) <= GABOR_DIAMETER / 2
    return 0.5 + 0.5 * np.where(inside, grating, 0.0)


def map_rows(image: np.ndarray, width: float) -> list[tuple[int, int]]:
    """Row spans (half-open) in which some run of content is at least ``width`` pixels.

    In an example summary these are the map frames: the colour maps are not
    white anywhere, so every row of a map is one run as wide as the map.
    """
    content = (image[..., :3] < WHITE).any(axis=2)
    longest = np.zeros(len(content), dtype=int)
    for r, row in enumerate(content):
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
        if edges.size:
            longest[r] = (edges[1::2] - edges[::2]).max()
    wide = np.concatenate(([False], longest >= width, [False]))
    edges = np.flatnonzero(np.diff(wide.astype(np.int8)))
    spans = []
    for start, stop in zip(edges[::2], edges[1::2]):
        if spans and start - spans[-1][1] < MAP_GAP:
            spans[-1] = (spans[-1][0], stop)
        else:
            spans.append((start, stop))
    return spans


def _show_stimulus(fig, image, x1, y_mid, width, height, title, fontsize):
    ax = fig.add_axes((x1 - width, y_mid - height / 2, width, height))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize, pad=4)
    return ax


def load_panel(path: Path, keep_title: bool) -> np.ndarray:
    image = plt.imread(path)
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image / 255.0
    image = trim_whitespace(image)
    if not keep_title:
        image = trim_whitespace(drop_suptitle(image))
    return image


def build_figure(prefix: str, keep_titles: bool) -> plt.Figure:
    panels: dict[tuple[str, str], np.ndarray | None] = {}
    for letter, stem in ROWS.items():
        for header, (label, modality) in COLUMNS.items():
            path = PLOT_DIR / f"{prefix}__{stem(label, modality)}.png"
            if path.exists():
                panels[letter, header] = load_panel(path, keep_titles)
            else:
                print(f"  missing {path.name}")
                panels[letter, header] = None

    # each row is as tall as its tallest panel at the common column width
    ratios = []
    for letter in ROWS:
        aspects = [img.shape[0] / img.shape[1] for (row, _), img in panels.items()
                   if row == letter and img is not None]
        ratios.append(max(aspects, default=0.5))
    fig_w = COLUMN_WIDTH * (len(COLUMNS) + STIMULUS_WIDTH)
    fig_h = COLUMN_WIDTH * sum(ratios) * 1.08 + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(len(ROWS), len(COLUMNS) + 1, height_ratios=ratios,
                            width_ratios=[STIMULUS_WIDTH] + [1] * len(COLUMNS),
                            left=0.025, right=0.995, top=1 - 0.6 / fig_h, bottom=0.005,
                            hspace=0.08, wspace=0.04)
    zebra = zebra_frame()
    stimuli = (("Gabor patch", gabor_frame(zebra.shape)), ("Zebra noise", zebra))
    firsts = {}
    for i, letter in enumerate(ROWS):
        first = None
        for j, header in enumerate(COLUMNS, start=1):
            ax = fig.add_subplot(grid[i, j])
            ax.axis("off")
            image = panels[letter, header]
            if image is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", color="0.5")
            else:
                ax.imshow(image, interpolation="lanczos")
                # bottom-aligned, so panels of different heights end on one line
                ax.set_anchor("S")
                if first is None:
                    first = ax, image
            if i == 0:
                ax.set_title(header, fontsize=TITLE_SIZE, fontweight="bold", pad=12)
        top = grid[i, 0].get_position(fig).y1
        fig.text(0.005, top, letter, fontsize=LETTER_SIZE, fontweight="bold",
                 ha="left", va="top")
        if first is not None:
            firsts[letter] = (grid[i, 0].get_position(fig), *first)

    if SCALE_ROW in firsts:
        # inches of this figure per pixel of the example-summary PNG
        _, panel_ax, panel = firsts[SCALE_ROW]
        panel_ax.apply_aspect()
        scale = panel_ax.get_position().width * fig_w / panel.shape[1]
        for letter in STIMULUS_ROWS:
            if letter in firsts:
                _place_stimuli(fig, *firsts[letter], stimuli, scale, letter == SCALE_ROW)
    return fig


def _place_stimuli(fig, cell, panel_ax, panel, stimuli, scale, on_maps) -> None:
    """Put each stimulus beside its sub-row of the first panel of the row.

    ``scale`` is inches per pixel of the example-summary PNG. The stimuli are
    drawn as large as its Zebra maps, titled in the size of its column titles.
    With ``on_maps`` (the example summary itself) each stimulus is centred on the
    Gabor or Zebra maps; otherwise on the upper or lower half of the panel.
    """
    panel_ax.apply_aspect()
    box = panel_ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    az, el = (abs(WAVEN_EXTENT[1] - WAVEN_EXTENT[0]), abs(WAVEN_EXTENT[3] - WAVEN_EXTENT[2]))
    width = az * PAIRED_INCH_PER_DEG * SAVE_DPI * scale
    height = el * PAIRED_INCH_PER_DEG * SAVE_DPI * scale
    fontsize = PAIRED_LABEL_SIZE * SAVE_DPI * scale

    n_rows = panel.shape[0]
    spans = None
    if on_maps:
        # Gabor maps are narrower than Zebra maps; both are wider than any other content
        found = map_rows(panel, 0.95 * GABOR_MAP_DEG * PAIRED_INCH_PER_DEG * SAVE_DPI)
        if len(found) == 2:
            spans = found
    if spans is None:
        split = subrow_split(panel)
        spans = ((0, split), (split, n_rows))
    x1 = cell.x1 - 0.04 * cell.width
    for (title, image), (r0, r1) in zip(stimuli, spans):
        y_mid = box.y1 - (r0 + r1) / 2 / n_rows * box.height
        _show_stimulus(fig, image, x1, y_mid, width / fig_w, height / fig_h, title, fontsize)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default=PREFIX,
                        help="file prefix of the panels in plots/, e.g. "
                             "'data_release__zebra-trial-0' for one Zebra repeat")
    parser.add_argument("--keep-titles", action="store_true",
                        help="keep each panel's own suptitle")
    args = parser.parse_args()

    fig = build_figure(args.prefix, args.keep_titles)
    for ext in ("png", "pdf"):
        path = PLOT_DIR / f"{args.prefix}__paper-figure.{ext}"
        fig.savefig(path, dpi=300)
        print(f"  -> {path.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
