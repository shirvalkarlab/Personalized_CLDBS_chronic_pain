#!/usr/bin/env python3
"""
plot_figure6_ancillary_main.py
================================================================================
Main Figure 6 d + e: ancillary behavioral scores, baseline vs chronic.

  d)  SF-36 composite baseline vs chronic (per-survey box + patient swarm).
  e)  Symptom-domain forest (pain / function / mood) in SD units with 95% CI.

Box + swarm conventions are shared with the other cohort panels (see
_ancillary_helpers.py). Refits the mixed models at run time from the
long-format CSV.

INPUT   data/figure_ancillary_behavioral/ancillary_scores_long.csv
OUTPUT  figures/Figure6de_ancillary_behavioral_main.pdf
        data/figure_ancillary_behavioral/{mixed_effects,construct_pooling}_reference.csv
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# _ancillary_helpers.py sits beside this file. Put that folder on sys.path so
# the script runs from any working directory, not only from python_code/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ancillary_helpers import (
    CHRONIC_LINE, GREY_LINE, PARTICIPANT_COLORS,
    add_panel_letter, draw_box_swarm, fit_constructs, fit_mixed_effects,
    format_p, write_reference_tables,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "ANCILLARY_DATA_DIR",
    os.path.join(HERE, os.pardir, "data", "figure_ancillary_behavioral"),
)
FIG_DIR = os.environ.get("FIGURE_DIR", os.path.join(HERE, os.pardir, "figures"))


def _add_participant_legend(ax):
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=col,
                          markersize=5, label=pat)
               for pat, col in PARTICIPANT_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=6, handlelength=0.8, handletextpad=0.3,
              labelspacing=0.25, borderpad=0.2)


def draw_panel_d(ax, long: pd.DataFrame, mixed: pd.DataFrame):
    """Panel d: SF-36 composite baseline vs chronic."""
    comp = long[long["measure"] == "sf36_composite"]
    draw_box_swarm(ax, comp, ["baseline", "chronic"], [0, 1], "value")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["baseline", "chronic"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("SF-36 composite (0-100)")
    ax.set_title("Quality of life (exploratory)", fontsize=9, fontweight="bold")

    m = mixed[mixed["measure"] == "sf36_composite"]
    if not m.empty:
        p = m["p"].iloc[0]
        y_top = comp["value"].max() + 4
        ax.plot([0, 1], [y_top, y_top], "k-", linewidth=0.6)
        ax.text(0.5, y_top + 1.5, f"mixed model {format_p(p)}",
                ha="center", va="bottom", fontsize=7)
    _add_participant_legend(ax)


def draw_panel_e(ax, constructs: pd.DataFrame):
    """Panel e: construct-level forest (pain / function / mood)."""
    want = ["mood", "function", "pain"]   # bottom -> top
    C = constructs.set_index("construct").reindex(want).dropna(subset=["coef_sd"]).reset_index()

    for i, row in C.iterrows():
        col = CHRONIC_LINE if row["p"] < 0.05 else GREY_LINE
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                "-", color=col, linewidth=2.2)
        ax.scatter(row["coef_sd"], i, s=52, color=col, zorder=3)
        ax.text(row["ci_hi"] + 0.08, i, format_p(row["p"]),
                va="center", fontsize=7)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_yticks(range(len(C)))
    ax.set_yticklabels(C["construct"])
    ax.set_ylim(-0.5, len(C) - 0.5)
    ax.set_xlabel("baseline \u2192 chronic (SD, 95% CI)")
    ax.set_title("Symptom domains\n(higher = better)", fontsize=9, fontweight="bold")
    xl = ax.get_xlim()
    ax.set_xlim(min(-0.4, xl[0]), xl[1] + 0.5)


def main():
    long_path = os.path.join(DATA_DIR, "ancillary_scores_long.csv")
    if not os.path.exists(long_path):
        raise FileNotFoundError(f"Missing extract: {long_path}")
    long = pd.read_csv(long_path)
    print(f"Loaded {len(long)} per-survey rows "
          f"({long['patient'].nunique()} participants, "
          f"{long['measure'].nunique()} measures).")

    mixed = fit_mixed_effects(long)
    constructs, _ = fit_constructs(long)
    write_reference_tables(mixed, constructs, DATA_DIR)
    print("\nSF-36 composite mixed model:")
    m = mixed[mixed["measure"] == "sf36_composite"].iloc[0]
    print(f"  coef={m['coef']:+.2f} 95%CI[{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}]  p={m['p']:.4g}")
    print("\nConstruct pooling:")
    for _, r in constructs.iterrows():
        print(f"  {r['construct']:9s} {r['coef_sd']:+.3f} SD  "
              f"95%CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  p={r['p']:.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    draw_panel_d(axes[0], long, mixed)
    draw_panel_e(axes[1], constructs)

    for ax, letter in zip(axes, "de"):
        add_panel_letter(ax, letter)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "Figure6de_ancillary_behavioral_main.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
