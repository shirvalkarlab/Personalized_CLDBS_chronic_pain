"""
plot_figureS8_composite.py
================================================================================
Figure S8: ambulatory biomarker stability, all nine implanted device sides in one
3 x 3 figure (panels a-i).

Each panel is the AUC grid written by analyze_ambulatory_biomarker_stability.py
for one device side: leave-one-out LDA AUC of low vs high SF-MPQv2 pain from the
mean power in each 5 Hz window (columns), using every survey from the first
2, 4, 6, ... months after the first ambulatory survey (rows). Rows with fewer
than 2 low- or 2 high-pain surveys have no AUC and are drawn grey.

The heatmaps are drawn with bilinear interpolation between grid cells, which
smooths the display only; the AUC values are those in the CSVs. The y-axis label
and the colour-bar label are written once, on panel a.

INPUT   data/figureS8_biomarker_stability/<DEVICE_SIDE>_auc_grid.csv  (nine files)
OUTPUT  figures/FigureS8_biomarker_stability.pdf

Usage (from python_code/stability_and_closed_loop/):
    python plot_figureS8_composite.py
"""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.environ.get("S8_DATA_DIR", os.path.join(REPO, "data", "figureS8_biomarker_stability"))
FIG_DIR = os.environ.get("FIGURE_DIR", os.path.join(REPO, "figures"))

SIDES = ["P1R", "P2L", "P2R", "P3L", "P3R", "P4L", "P4R", "P5L", "P5R"]
AUC_RANGE = (0.5, 0.75)


def side_title(side):
    """'P4R' -> 'P4 Right'."""
    return f"{side[:2]} {'Left' if side.endswith('L') else 'Right'}"


def load_grids(folder=DATA_DIR):
    """{side: (auc array, rows = windows and columns = frequencies, frequencies, months)}."""
    grids = {}
    for side in SIDES:
        table = pd.read_csv(os.path.join(folder, f"{side}_auc_grid.csv"))
        auc = table.pivot(index="months_included", columns="center_frequency_hz", values="auc")
        grids[side] = (auc.to_numpy(), auc.columns.to_numpy(dtype=float), auc.index.to_numpy())
    return grids


def build_figure(grids):
    """Draw the 3 x 3 figure; returns (figure, heatmap axes, colour bars)."""
    colour_map = copy.copy(plt.get_cmap("Reds"))
    colour_map.set_bad("0.85")
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.0), constrained_layout=True)
    heatmaps, colorbars = [], []
    for k, (ax, side) in enumerate(zip(axes.ravel(), SIDES)):
        auc, frequencies, months = grids[side]
        month_edge = months[-1]
        image = ax.imshow(auc, cmap=colour_map, vmin=AUC_RANGE[0], vmax=AUC_RANGE[1],
                          origin="lower", aspect="auto", interpolation="bilinear",
                          extent=(frequencies[0] - 0.5, frequencies[-1] + 0.5, 0, month_edge))
        ax.set_xlim(0, 50)
        ax.set_yticks(np.arange(0, month_edge + 1, 6))
        ax.set_xlabel("Center frequency of 5 Hz window (Hz)")
        ax.set_title(side_title(side))
        ax.text(-0.12, 1.04, "abcdefghi"[k], transform=ax.transAxes,
                fontweight="bold", fontsize=13, ha="left", va="bottom")
        colorbar = fig.colorbar(image, ax=ax)
        if k == 0:
            ax.set_ylabel("Months since first ambulatory survey")
            colorbar.set_label("AUC, low vs high SF-MPQv2")
        heatmaps.append(ax)
        colorbars.append(colorbar)
    return fig, heatmaps, colorbars


def main():
    grids = load_grids()
    for side in SIDES:
        auc, _, months = grids[side]
        print(f"{side}: {len(months)} windows (to {months[-1]} months), "
              f"{np.isnan(auc).all(axis=1).sum()} without enough surveys, "
              f"AUC {np.nanmin(auc):.3f}-{np.nanmax(auc):.3f}")
    fig, _, _ = build_figure(grids)
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "FigureS8_biomarker_stability.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
