"""
Figure 5f: blinding assessment.

Participants were asked to guess whether stimulation was on ("stim") or off
("sham") just before and after each program change, while both they and the
assessor were blinded to the true condition. This script reports how often each
guess was made under each true condition, and runs the three tests reported in
the manuscript.

The unit of analysis is the participant, not the individual guess: each
participant contributes one percentage per (ground truth x guess) cell, and
every test is paired within participant.

  1. Main effect of guess. Averaging over both true conditions, did
     participants favour one answer? Each participant's two ground-truth
     percentages are averaged to a single value per guess, then the two guesses
     are compared with a paired t test. F = t^2 on 1 and n-1 degrees of freedom.

  2. Within "stim" guesses. Was a participant more likely to answer "stim"
     when stimulation was truly on than when it was truly off? Paired t test
     across participants.

  3. Within "sham" guesses. The same comparison for the "sham" answer. Because
     the two answers are complementary within a true condition (they sum to
     100%), this test is the mirror of test 2: same magnitude, opposite sign.

Tests 2 and 3 are the blinding check proper. A participant who could detect
stimulation would answer "stim" more often when it was truly on; no difference
means the blind held.

P3 did not undergo sham testing and so contributes only true-CL-DBS guesses.
P3 therefore appears in test 1, which averages within participant over the
conditions that participant has, and not in tests 2 and 3, which require both
conditions from the same participant.

INPUT
    data/figure5_double_blind_outcomes/double_blind_outcomes.xlsx, sheet
    "stim_guess": one row per participant x ground truth x guess, with
    percent_guessed giving how often that participant gave that answer under
    that true condition.

OUTPUT
    figures/Figure5f_blinding_guess.pdf   editable vector, for figure assembly
    figures/Figure5f_blinding_guess.svg   editable vector
    figures/Figure5f_blinding_guess.png   300 dpi raster, for quick viewing
    Test statistics are printed to the console.

Usage:
    python plot_figure5f_blinding_guess.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Export a true vector PDF whose labels stay editable. Font type 42 embeds
# TrueType fonts, so each label opens in Illustrator as one selectable text
# object; the default type 3 converts glyphs to outlines and cannot be edited.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

# Repository layout: this file lives in python_code/, data and figures are
# siblings of it at the repository root.
HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(HERE, os.pardir))
GUESS_WORKBOOK = os.path.join(REPOSITORY, "data", "figure5_double_blind_outcomes",
                              "double_blind_outcomes.xlsx")
FIGURE_DIRECTORY = os.path.join(REPOSITORY, "figures")

# Bars are coloured by the TRUE condition, so a reader compares the two colours
# within each guess to see whether the true condition shifted the answer.
COLOR_FOR_TRUTH = {"sham": "#4C72B0", "stim": "#C44E52"}
READABLE_TRUTH = {"sham": "sham", "stim": "CL-DBS"}


def load_guesses():
    """Read the blinding-guess sheet, one row per participant x truth x guess."""
    guesses = pd.read_excel(GUESS_WORKBOOK, sheet_name="stim_guess")
    expected = {"pt_id", "ground_truth", "pt_guess", "percent_guessed"}
    missing = expected - set(guesses.columns)
    if missing:
        raise KeyError(f"stim_guess sheet is missing columns: {sorted(missing)}")
    return guesses


def test_main_effect_of_guess(guesses):
    """Test 1: collapsing across true condition, was one answer given more often?

    Each participant is first reduced to one percentage per guess (the mean of
    their true-sham and true-CL-DBS percentages), so participants contribute
    equally regardless of how many blocks of each type they had.
    """
    per_participant = (guesses
                       .groupby(["pt_id", "pt_guess"])["percent_guessed"]
                       .mean()
                       .unstack()
                       .dropna())
    comparison = stats.ttest_rel(per_participant["stim"], per_participant["sham"])
    return {
        "stim_mean": per_participant["stim"].mean(),
        "stim_sem": per_participant["stim"].sem(),
        "sham_mean": per_participant["sham"].mean(),
        "sham_sem": per_participant["sham"].sem(),
        "f_statistic": comparison.statistic ** 2,   # F(1, n-1) = t^2
        "p_value": comparison.pvalue,
        "n_participants": len(per_participant),
    }


def test_truth_effect_within_guess(guesses, guess):
    """Tests 2 and 3: within one answer, did the true condition change how often
    participants gave it?

    Only participants with both true conditions can be paired, which excludes
    any participant who did not undergo sham testing.
    """
    within = (guesses[guesses["pt_guess"] == guess]
              .pivot_table(index="pt_id", columns="ground_truth", values="percent_guessed")
              .dropna())
    comparison = stats.ttest_rel(within["sham"], within["stim"])
    return {
        "under_sham_mean": within["sham"].mean(),
        "under_sham_sem": within["sham"].sem(),
        "under_stim_mean": within["stim"].mean(),
        "under_stim_sem": within["stim"].sem(),
        "t_statistic": comparison.statistic,
        "p_value": comparison.pvalue,
        "degrees_of_freedom": len(within) - 1,
        "n_participants": len(within),
    }


def plot_blinding_guess(guesses):
    """Draw the panel: guess on the x axis, bars coloured by true condition."""
    cell_means = guesses.pivot_table(index="pt_id",
                                     columns=["pt_guess", "ground_truth"],
                                     values="percent_guessed")

    figure, axes = plt.subplots(figsize=(3.6, 3.1))
    group_gap, bar_gap, bar_width = 1.35, 0.55, 0.50
    jitter = np.random.RandomState(7)

    for group_index, guess in enumerate(["sham", "stim"]):
        for bar_index, truth in enumerate(["sham", "stim"]):
            position = group_index * group_gap + bar_index * bar_gap
            if (guess, truth) not in cell_means.columns:
                continue
            values = cell_means[(guess, truth)].dropna()
            axes.bar(position, values.mean(), bar_width, yerr=values.sem(),
                     color=COLOR_FOR_TRUTH[truth], capsize=2.5, zorder=2,
                     error_kw=dict(lw=0.85, ecolor="#333333"))
            # Individual participants over the summary bar.
            axes.scatter(position + jitter.uniform(-0.10, 0.10, len(values)),
                         values.values, s=10, color="#2B2B2B", alpha=0.55,
                         zorder=3, linewidths=0)

    axes.axhline(50, color="#8A8A8A", lw=0.8, ls="--", zorder=1)   # chance
    axes.set_xticks([bar_gap / 2, group_gap + bar_gap / 2])
    axes.set_xticklabels(['guessed "sham"', 'guessed "stim"'])
    axes.set_xlim(-0.45, 2.25)
    axes.set_ylim(0, 100)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_ylabel("likelihood of guess (%)")
    axes.spines[["top", "right"]].set_visible(False)

    handles = [Patch(color=COLOR_FOR_TRUTH[t], label=READABLE_TRUTH[t])
               for t in ["sham", "stim"]]
    figure.legend(handles=handles, frameon=False, fontsize=6.4, loc="lower center",
                  ncol=2, bbox_to_anchor=(0.58, 0.955), title="ground truth",
                  title_fontsize=6.4, handlelength=1.1, columnspacing=1.2)
    figure.tight_layout(rect=[0, 0, 1, 0.90])
    return figure


def main():
    guesses = load_guesses()

    main_effect = test_main_effect_of_guess(guesses)
    within_stim = test_truth_effect_within_guess(guesses, "stim")
    within_sham = test_truth_effect_within_guess(guesses, "sham")

    print("Figure 5f: blinding assessment\n")
    print("1. Main effect of guess (collapsed across true condition)")
    print(f'   guessed "stim" {main_effect["stim_mean"]:.0f} +/- {main_effect["stim_sem"]:.0f}%   '
          f'guessed "sham" {main_effect["sham_mean"]:.0f} +/- {main_effect["sham_sem"]:.0f}%')
    print(f'   F(1,{main_effect["n_participants"] - 1}) = {main_effect["f_statistic"]:.2f}, '
          f'p = {main_effect["p_value"]:.3f}   (n = {main_effect["n_participants"]})\n')

    for guess, result in [("stim", within_stim), ("sham", within_sham)]:
        print(f'{2 if guess == "stim" else 3}. Effect of true condition within "{guess}" guesses')
        print(f'   under true sham {result["under_sham_mean"]:.0f} +/- {result["under_sham_sem"]:.0f}%   '
              f'under true CL-DBS {result["under_stim_mean"]:.0f} +/- {result["under_stim_sem"]:.0f}%')
        print(f'   t({result["degrees_of_freedom"]}) = {result["t_statistic"]:+.2f}, '
              f'p = {result["p_value"]:.3f}   (n = {result["n_participants"]})\n')

    figure = plot_blinding_guess(guesses)
    os.makedirs(FIGURE_DIRECTORY, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        output = os.path.join(FIGURE_DIRECTORY, f"Figure5f_blinding_guess.{extension}")
        figure.savefig(output, dpi=300, bbox_inches="tight")
        print("wrote", output)


if __name__ == "__main__":
    main()
