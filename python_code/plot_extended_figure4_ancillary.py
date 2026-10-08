#!/usr/bin/env python3
"""
plot_extended_figure4_ancillary.py
================================================================================
Extended Figure 4 a-d: ancillary behavioral scores and daily step counts.

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
  d)  Percent change in daily step count from each participant's pre-trial
      baseline, closed-loop (blue) vs sham (grey) days, for the three
      participants with wearable data in both arms (P1, P2, P4): summary box per
      participant with every day overlaid as a dot. This is the original
      analysis of this panel: closed-loop vs sham days pooled across
      participants, one-sided two-sample t-test (closed-loop > sham). The script
      also prints a one-sided Wilcoxon rank-sum test per participant and each
      participant's closed-loop minus sham difference from a participant x
      condition linear model.

Layout, colours and the box + swarm conventions are shared with the other
cohort panels through `_ancillary_helpers.py`, which also holds the statistics.
Responder status is shown as MCID counts.

INPUT   data/figure_ancillary_behavioral/ancillary_scores_long.csv     (a-c)
        data/figure5_double_blind_outcomes/double_blind_outcomes.xlsx  (d, sheet 'daily')
OUTPUT  figures/FigureExtended4abcd_ancillary_behavioral.pdf
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, ttest_ind

# _ancillary_helpers.py sits beside this file. Put that folder on sys.path so
# the script runs from any working directory, not only from python_code/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ancillary_helpers import (
    BASELINE_FACE, CHRONIC_FACE, PARTICIPANT_COLORS,
    add_panel_letter, draw_box_swarm, draw_summary_box, fit_constructs,
    format_p, mcid_responders,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "ANCILLARY_DATA_DIR",
    os.path.join(HERE, os.pardir, "data", "figure_ancillary_behavioral"),
)
FIG_DIR = os.environ.get("FIGURE_DIR", os.path.join(HERE, os.pardir, "figures"))
STEP_DATA = os.environ.get(
    "STEP_DATA",
    os.path.join(HERE, os.pardir, "data", "figure5_double_blind_outcomes",
                 "double_blind_outcomes.xlsx"),
)


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


# ---------------------------------------------------------------------------
# Panel d: daily step count, closed-loop vs sham
# ---------------------------------------------------------------------------
def load_step_days(path: str | None = None) -> pd.DataFrame:
    """One row per participant-day with a step value: participant, condition, steps.

    steps is the percent change in daily step count from the participant's
    pre-trial baseline. Days without a step value are dropped.
    """
    daily = pd.read_excel(path or STEP_DATA, sheet_name="daily")
    days = daily.rename(columns={"pt_id": "participant", "stim_group": "condition"})
    days = days[days["condition"].isin(["closed-loop", "sham"])]
    return days.dropna(subset=["steps"])[["participant", "condition", "steps"]].reset_index(drop=True)


def pooled_step_test(days: pd.DataFrame) -> dict:
    """One-sided two-sample t-test (closed-loop > sham), all participants' days pooled."""
    closed_loop = days.loc[days["condition"] == "closed-loop", "steps"]
    sham = days.loc[days["condition"] == "sham", "steps"]
    result = ttest_ind(closed_loop, sham, alternative="greater")
    return {"t": result.statistic, "df": len(closed_loop) + len(sham) - 2, "p": result.pvalue,
            "n_closed_loop": len(closed_loop), "n_sham": len(sham)}


def participant_step_tests(days: pd.DataFrame) -> pd.DataFrame:
    """One-sided Wilcoxon rank-sum (closed-loop > sham) per participant, unadjusted."""
    rows = []
    for participant, rows_p in days.groupby("participant"):
        closed_loop = rows_p.loc[rows_p["condition"] == "closed-loop", "steps"]
        sham = rows_p.loc[rows_p["condition"] == "sham", "steps"]
        result = mannwhitneyu(closed_loop, sham, alternative="greater")
        rows.append({"participant": participant,
                     "n_closed_loop": len(closed_loop), "n_sham": len(sham),
                     "mean_closed_loop": closed_loop.mean(), "mean_sham": sham.mean(),
                     "U": result.statistic, "p": result.pvalue})
    return pd.DataFrame(rows)


def participant_step_contrasts(days: pd.DataFrame) -> pd.DataFrame:
    """Closed-loop minus sham per participant from steps ~ participant * condition.

    One linear model with a shared residual variance; two-sided P per participant.
    """
    data = days.assign(closed_loop=(days["condition"] == "closed-loop").astype(float))
    fit = smf.ols("steps ~ C(participant) * closed_loop", data).fit()
    names = list(fit.params.index)
    participants = sorted(data["participant"].unique())
    rows = []
    for participant in participants:
        contrast = np.zeros(len(names))
        contrast[names.index("closed_loop")] = 1
        if participant != participants[0]:
            contrast[names.index(f"C(participant)[T.{participant}]:closed_loop")] = 1
        test = fit.t_test(contrast)
        rows.append({"participant": participant, "estimate": float(test.effect[0]),
                     "se": float(test.sd[0][0]), "t": float(test.tvalue[0][0]),
                     "df": int(fit.df_resid), "p": float(test.pvalue)})
    return pd.DataFrame(rows)


def draw_panel_d(ax, days: pd.DataFrame, pooled: dict):
    """Panel d: closed-loop group and sham group, one box + daily dots per participant."""
    rng = np.random.default_rng(4)
    participants = sorted(days["participant"].unique())
    group_start = {"closed-loop": 0.0, "sham": len(participants) * 0.45 + 0.5}
    faces = {"closed-loop": CHRONIC_FACE, "sham": BASELINE_FACE}
    centres = {}
    for condition, start in group_start.items():
        xs = [start + 0.45 * i for i in range(len(participants))]
        centres[condition] = float(np.mean(xs))
        for x, participant in zip(xs, participants):
            steps = days.loc[(days["participant"] == participant)
                             & (days["condition"] == condition), "steps"].to_numpy()
            draw_summary_box(ax, steps, x, 0.34, faces[condition])
            ax.scatter(x + rng.uniform(-0.1, 0.1, steps.size), steps, s=8,
                       color=PARTICIPANT_COLORS[participant], alpha=0.7,
                       edgecolors="none", zorder=4)
    ax.axhline(0, color="k", linestyle=":", linewidth=0.6)
    top = days["steps"].max()
    bracket = top * 1.08
    ax.plot([centres["closed-loop"], centres["closed-loop"], centres["sham"], centres["sham"]],
            [bracket * 0.97, bracket, bracket, bracket * 0.97], "k-", linewidth=0.8)
    ax.text(np.mean(list(centres.values())), bracket * 1.01,
            f"p={pooled['p']:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(centres.values()))
    ax.set_xticklabels(["closed-loop", "sham"])
    ax.set_ylim(days["steps"].min() * 1.1, bracket * 1.12)
    ax.set_ylabel("% change in daily steps from baseline")
    ax.set_title("Daily step count:\nclosed-loop (blue) vs sham (grey)",
                 fontsize=9, fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=PARTICIPANT_COLORS[p],
                          markersize=5, label=p) for p in participants]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6,
              handlelength=0.8, handletextpad=0.3, labelspacing=0.25)


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

    steps = load_step_days()
    pooled = pooled_step_test(steps)
    print(f"\nDaily steps, % change from baseline, closed-loop vs sham, days pooled over "
          f"{', '.join(sorted(steps['participant'].unique()))}:")
    print(f"  one-sided two-sample t-test (closed-loop > sham): t({pooled['df']}) = {pooled['t']:.3f}, "
          f"p = {pooled['p']:.4f}  (n = {pooled['n_closed_loop']} closed-loop, {pooled['n_sham']} sham days)")
    print("  per participant, one-sided Wilcoxon rank-sum (closed-loop > sham), unadjusted:")
    for _, r in participant_step_tests(steps).iterrows():
        print(f"    {r['participant']}  closed-loop n={r['n_closed_loop']} mean={r['mean_closed_loop']:+.1f}%  "
              f"sham n={r['n_sham']} mean={r['mean_sham']:+.1f}%  U={r['U']:.1f}  p={r['p']:.4f}")
    print("  per participant, steps ~ participant x condition, two-sided:")
    for _, r in participant_step_contrasts(steps).iterrows():
        print(f"    {r['participant']}  closed-loop - sham = {r['estimate']:+.1f} points (SE {r['se']:.1f}), "
              f"t({r['df']}) = {r['t']:.2f}, p = {r['p']:.4f}")

    # Two-row layout: schematic on top (panel a), analytic panels below (b, c, d).
    fig = plt.figure(figsize=(11.0, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], width_ratios=[1.4, 1.0, 1.0])

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

    ax_d = fig.add_subplot(gs[1, 2])
    draw_panel_d(ax_d, steps, pooled)
    add_panel_letter(ax_d, "d")
    ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "FigureExtended4abcd_ancillary_behavioral.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
