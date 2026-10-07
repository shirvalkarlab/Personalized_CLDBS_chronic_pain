#!/usr/bin/env python3
"""
plot_extended_figure4_ancillary.py
================================================================================
Extended Figure 4 a + b + c: ancillary behavioral scores workflow and results.

  a)  Construct-score schematic: how the raw questionnaires (SF-36 subscales,
      BDI, BAI, HamD-6) are oriented (higher = better; depression/anxiety
      scales sign-flipped), z-scored within each participant x measure, and
      pooled within construct (all member z-scores in one model) to give the
      PAIN / FUNCTION / MOOD construct effects tested in panel b.
  b)  Symptom domains: within-patient z (higher = better), grey (baseline) vs
      blue (chronic), pain / function / mood side-by-side, with the pooled
      mixed-model p-value above each domain.
  c)  MCID responders by measure: number of participants (of 5) whose
      first-baseline -> last-chronic change reached the minimal clinically
      important difference in the better direction.

Layout, colours and the box + swarm conventions are shared with the other
cohort panels through `_ancillary_helpers.py`, which also holds the statistics.
Responder status is shown as MCID counts.

INPUT   data/figure_ancillary_behavioral/ancillary_scores_long.csv
OUTPUT  figures/FigureExtended4abc_ancillary_behavioral.pdf
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# _ancillary_helpers.py sits beside this file. Put that folder on sys.path so
# the script runs from any working directory, not only from python_code/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ancillary_helpers import (
    PARTICIPANT_COLORS,
    add_panel_letter, draw_box_swarm, fit_constructs,
    format_p, mcid_responders,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "ANCILLARY_DATA_DIR",
    os.path.join(HERE, os.pardir, "data", "figure_ancillary_behavioral"),
)
FIG_DIR = os.environ.get("FIGURE_DIR", os.path.join(HERE, os.pardir, "figures"))


def draw_panel_a(ax):
    """Panel a: schematic of the construct-score workflow.

    Four columns (raw questionnaires; orient + z-score; average within
    construct; construct score), one row per construct, connected by
    construct-coloured arrows. Depression/anxiety
    scales are labelled 'sign-flipped' because the analysis multiplies them
    by -1 before z-scoring so a positive change always means improvement.
    """
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    ok = {"pain":     (0.835, 0.369, 0.000),
          "function": (0.337, 0.706, 0.914),
          "mood":     (0.000, 0.620, 0.451)}

    # Column headers
    ax.text(1.6, 9.6, "Raw questionnaires", ha="center", fontweight="bold", fontsize=8)
    ax.text(4.7, 9.6, "Orient + z-score\n(within patient)", ha="center", fontweight="bold", fontsize=8)
    ax.text(7.1, 9.6, "Pool within\nconstruct", ha="center", fontweight="bold", fontsize=8)
    ax.text(9.0, 9.6, "Construct\nscore", ha="center", fontweight="bold", fontsize=8)

    groups = {
        "pain":     ["SF-36 Pain"],
        "function": ["SF-36 Physical Fxn", "SF-36 Role-Physical",
                     "SF-36 Energy", "SF-36 Social", "SF-36 General Health"],
        "mood":     ["BDI", "BAI", "HamD6", "SF-36 Emotion", "SF-36 Role-Emotional"],
    }
    flip_set = {"BDI", "BAI", "HamD6"}

    y = 9.0
    positions = {}
    for g in ("pain", "function", "mood"):
        yy = []
        for it in groups[g]:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.35, y - 0.16), 2.5, 0.32,
                boxstyle="round,pad=0.02",
                fc=(0.96, 0.96, 0.96), ec=ok[g], lw=1.0))
            lab = it + ("  (sign-flipped)" if it in flip_set else "")
            ax.text(1.6, y, lab, ha="center", va="center", fontsize=6)
            yy.append(y); y -= 0.42
        positions[g] = yy; y -= 0.35

    for g in ("pain", "function", "mood"):
        yy = positions[g]; ymid = float(np.mean(yy))
        for yv in yy:
            ax.add_patch(FancyArrowPatch(
                (2.9, yv), (3.9, yv), arrowstyle="-|>",
                mutation_scale=7, color=(0.6, 0.6, 0.6), lw=0.6))
        ax.add_patch(mpatches.FancyBboxPatch(
            (3.95, min(yy) - 0.16), 1.5, max(yy) - min(yy) + 0.32,
            boxstyle="round,pad=0.02", fc="white", ec=ok[g], lw=1.2))
        ax.text(4.7, ymid, "z = (x - mean) / SD\nper patient x measure",
                ha="center", va="center", fontsize=5.6)
        ax.add_patch(FancyArrowPatch(
            (5.5, ymid), (6.3, ymid), arrowstyle="-|>",
            mutation_scale=8, color=ok[g], lw=1.3))
        ax.add_patch(mpatches.FancyBboxPatch(
            (6.35, ymid - 0.35), 1.5, 0.7,
            boxstyle="round,pad=0.02", fc="white", ec=ok[g], lw=1.2))
        ax.text(7.1, ymid, "mean of\nz-scores", ha="center", va="center", fontsize=6)
        ax.add_patch(FancyArrowPatch(
            (7.9, ymid), (8.4, ymid), arrowstyle="-|>",
            mutation_scale=8, color=ok[g], lw=1.3))
        ax.add_patch(mpatches.FancyBboxPatch(
            (8.45, ymid - 0.4), 1.4, 0.8,
            boxstyle="round,pad=0.02", fc=ok[g], ec="k", lw=0.8))
        ax.text(9.15, ymid, g.upper(), ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="w")

    ax.text(0.35, 0.55,
            "Higher = better for all constructs. Depression/anxiety scales "
            "(BDI, BAI, HamD6) are sign-flipped (multiply by -1) before "
            "z-scoring\nso an increase always means improvement. Each "
            "measure is z-scored within patient so patients on different "
            "absolute scales combine\non a common footing. All member "
            "z-scores of a construct enter one model; phase change is "
            "tested with z ~ phase + (1|patient).",
            ha="left", va="center", fontsize=5.4, color=(0.25, 0.25, 0.25))


def _add_participant_legend(ax):
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=col,
                          markersize=5, label=pat)
               for pat, col in PARTICIPANT_COLORS.items()]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              fontsize=6, handlelength=0.8, handletextpad=0.3,
              labelspacing=0.25, borderpad=0.2)


def draw_panel_b(ax, long_z: pd.DataFrame, constructs: pd.DataFrame):
    """Panel b: within-patient z per domain, baseline vs chronic side by side."""
    domain_order = ["pain", "function", "mood"]
    xbase = [0.0, 1.4, 2.8]                     # per-domain group centres
    phase_offset = [-0.25, +0.25]               # baseline (left), chronic (right)
    tick_labels = ["pain", "function", "mood"]

    for ci, cn in enumerate(domain_order):
        d = long_z[(long_z["construct"] == cn) & long_z["z"].notna()]
        if d.empty:
            continue
        for j, ph in enumerate(("baseline", "chronic")):
            xc = xbase[ci] + phase_offset[j]
            sub = d[d["phase"] == ph]
            draw_box_swarm(ax, sub, [ph], xc, "z",
                           jitter=0.09, msize=8, malpha=0.55, rng_seed=ci * 10 + j)

        # Domain-level p-value above the pair, inside the axes
        crow = constructs[constructs["construct"] == cn]
        if not crow.empty:
            ax.text(xbase[ci], 2.9,
                    format_p(crow["p"].iloc[0]),
                    ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="k", linestyle=":", linewidth=0.6)
    ax.set_xticks(xbase)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(xbase[0] - 0.7, xbase[-1] + 0.7)
    ax.set_ylim(-6, 3.4)
    ax.set_ylabel("within-patient z (higher = better)")
    ax.set_title("Symptom domains:\nbaseline (grey) vs chronic (blue)",
                 fontsize=9, fontweight="bold")
    _add_participant_legend(ax)


def draw_panel_c(ax, responders: pd.DataFrame):
    """Panel c: MCID responder counts (participants of 5, sorted ascending)."""
    show = ["randPain", "randPhysFxn", "randEnergy", "BAI", "sf36_composite"]
    S = responders.set_index("measure").reindex(show).dropna(subset=["n"]).reset_index()
    S = S.sort_values("reached_mcid").reset_index(drop=True)

    green = (0.00, 0.62, 0.451)
    for i, row in S.iterrows():
        ax.barh(i, row["reached_mcid"], color=green, alpha=0.85,
                edgecolor="none")

    pretty = {"sf36_composite": "composite",
              "randPain": "Pain", "randPhysFxn": "PhysFxn",
              "randEnergy": "Energy", "BAI": "BAI"}
    labs = [pretty.get(m, m) for m in S["measure"]]
    ax.set_yticks(range(len(S)))
    ax.set_yticklabels(labs)
    ax.set_xlim(0, 5.2)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("patients improved (of 5)")
    ax.set_title("MCID Responders by domain", fontsize=9, fontweight="bold")


def main():
    long_path = os.path.join(DATA_DIR, "ancillary_scores_long.csv")
    if not os.path.exists(long_path):
        raise FileNotFoundError(f"Missing extract: {long_path}")
    long = pd.read_csv(long_path)
    print(f"Loaded {len(long)} per-survey rows "
          f"({long['patient'].nunique()} participants, "
          f"{long['measure'].nunique()} measures).")

    constructs, long_z = fit_constructs(long)
    _, resp = mcid_responders(long)

    print("\nConstruct pooling:")
    for _, r in constructs.iterrows():
        print(f"  {r['construct']:9s} {r['coef_sd']:+.3f} SD  "
              f"95%CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  p={r['p']:.4g}")
    print("\nMCID responder counts (of 5):")
    for _, r in resp.iterrows():
        print(f"  {r['measure']:20s} {int(r['reached_mcid'])}/{int(r['n'])}")

    # Two-row layout: schematic on top (panel a), analytic panels below (b, c).
    fig = plt.figure(figsize=(8.4, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.4, 1.0])

    ax_a = fig.add_subplot(gs[0, :])
    draw_panel_a(ax_a)
    add_panel_letter(ax_a, "a")

    ax_b = fig.add_subplot(gs[1, 0])
    draw_panel_b(ax_b, long_z, constructs)
    add_panel_letter(ax_b, "b")
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)

    ax_c = fig.add_subplot(gs[1, 1])
    draw_panel_c(ax_c, resp)
    add_panel_letter(ax_c, "c")
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "FigureExtended4abc_ancillary_behavioral.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
