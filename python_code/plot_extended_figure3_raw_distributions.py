"""Extended Data Figure 3 -- raw pain scores by stimulation arm.

Shows every pain survey, not only the summary statistics of the blinded
comparison. This figure draws all of them, for each of the three pain
scales, split into the six stimulation arms of the trial.

Two versions are written. The full version shows all six arms. The reduced
version omits open-loop optimisation, which is the largest arm and can crowd the
panel; the statistics are the same for both, because they are computed once from
all six arms.

Design
    One violin per participant per arm, with every survey drawn as a point on
    top, the median as a horizontal rule, and the arm encoded by fill colour.
    An arm with fewer than three surveys is drawn as its points and n only: a
    kernel density over one or two observations invents a distribution. An arm
    with no surveys is marked with an open circle rather than left blank, so an
    empty slot is not read as a measurement of zero.

Statistics
    An omnibus Kruskal-Wallis across every arm with at least three surveys,
    then two prespecified one-tailed contrasts asking whether the double-blind
    closed-loop window is lower than its reference: versus baseline, and versus
    double-blind sham. Mann-Whitney U, Holm corrected within each participant x
    scale family. The four remaining arms enter the omnibus test only; they are
    not contrasted pairwise, which keeps the correction family to the two
    prespecified contrasts, and the trial was not designed to compare them.

Arm membership
    The arms are mutually exclusive and two flags outrank the `condition`
    column. `is_baseline` marks each participant's reference window, which is
    not a single stimulation state: for most participants it selects part of the
    0 mA period before stimulation began, and for one the pre-operative study.
    Those surveys are removed from the washout arm so none is drawn twice.
    `is_published_closed_loop` marks the curated closed-loop window that the
    published blinded-window table reports; the remaining closed-loop surveys
    form the optimisation arm. Both are checked at run time.

Reads   data/extended_figures_raw_pain/pain_surveys.csv
Writes  figures/FigureExtended3_raw_pain_distributions.{pdf,png,svg}
        figures/FigureExtended3_raw_pain_distributions_no_open_loop.{pdf,png,svg}
        data/extended_figures_raw_pain/ExtFig3_omnibus_condition_stats.csv
        data/extended_figures_raw_pain/ExtFig3_pairwise_condition_stats.csv
        data/extended_figures_raw_pain/ExtFig3_arm_counts_medians.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO = os.environ.get("NATMED_REPO",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIRECTORY = os.path.join(REPO, "data", "extended_figures_raw_pain")
FIGURE_DIRECTORY = os.path.join(REPO, "figures")
SURVEY_CSV = os.path.join(DATA_DIRECTORY, "pain_surveys.csv")

OMNIBUS_OUT = os.path.join(DATA_DIRECTORY, "ExtFig3_omnibus_condition_stats.csv")
PAIRWISE_OUT = os.path.join(DATA_DIRECTORY, "ExtFig3_pairwise_condition_stats.csv")
COUNTS_OUT = os.path.join(DATA_DIRECTORY, "ExtFig3_arm_counts_medians.csv")

PARTICIPANTS = ["P1", "P2", "P3", "P4", "P5"]
MEASURES = ["mayoNRS", "painVAS", "MPQtotal"]
MEASURE_LABEL = {"mayoNRS": "Pain NRS (0-10)",
                 "painVAS": "Pain intensity VAS (0-100)",
                 "MPQtotal": "MPQ total"}
MEASURE_TICK_STEP = {"mayoNRS": 2, "painVAS": 20, "MPQtotal": 10}

# The six arms, in trial order. Display capitalisation lives here, in the
# plotting code, and never in the data file.
BASELINE_ARM = "Baseline"
OPEN_LOOP_ARM = "Open-loop optimization, unblinded"
CLOSED_LOOP_OPEN_ARM = "Closed-loop optimization, unblinded"
BLINDED_ARM = "Closed-loop, double-blind"
SHAM_ARM = "Sham, double-blind (0 mA)"
WASHOUT_ARM = "Washout + carryover (0 mA)"

ALL_ARMS = [BASELINE_ARM, OPEN_LOOP_ARM, CLOSED_LOOP_OPEN_ARM,
            BLINDED_ARM, SHAM_ARM, WASHOUT_ARM]
ARMS_WITHOUT_OPEN_LOOP = [arm for arm in ALL_ARMS if arm != OPEN_LOOP_ARM]

# Arms whose membership is a plain `condition` value; the other two are decided
# by flag in select_arm().
CONDITION_OF_ARM = {OPEN_LOOP_ARM: "open-loop",
                    CLOSED_LOOP_OPEN_ARM: "closed-loop",
                    SHAM_ARM: "sham",
                    WASHOUT_ARM: "washout"}

TEST_REFERENCES = (BASELINE_ARM, SHAM_ARM)
MINIMUM_FOR_VIOLIN = 3

ARM_COLOUR = {BASELINE_ARM:         (0.62, 0.62, 0.62),
              OPEN_LOOP_ARM:        (0.55, 0.75, 0.60),
              CLOSED_LOOP_OPEN_ARM: (0.55, 0.75, 0.93),
              BLINDED_ARM:          (0.08, 0.32, 0.58),
              SHAM_ARM:             (0.90, 0.55, 0.20),
              WASHOUT_ARM:          (0.80, 0.80, 0.80)}
POINT_COLOUR = (0.25, 0.25, 0.25)

mpl.rcParams.update({"font.family": "Arial", "axes.titlesize": 10,
                     "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def select_arm(surveys, participant, arm):
    """Surveys of one participant in one arm.

    Two flags outrank `condition`. `is_baseline` wins first: a baseline survey's
    `condition` records where it physically sits (0 mA, or the pre-operative
    study), not the arm it is drawn as, so those surveys are also removed from
    the washout arm. `is_published_closed_loop` wins next, marking the curated
    window the published blinded-window table reports; the rest of the
    closed-loop surveys form the optimisation arm.
    """
    rows = surveys[surveys.participant == participant]
    is_baseline = rows.is_baseline == 1
    is_blinded = (rows.is_published_closed_loop == 1) & ~is_baseline

    if arm == BASELINE_ARM:
        return rows[is_baseline]
    if arm == BLINDED_ARM:
        return rows[is_blinded]
    return rows[(rows.condition == CONDITION_OF_ARM[arm])
                & ~is_baseline & ~is_blinded]


def load_surveys():
    surveys = pd.read_csv(SURVEY_CSV)
    required = ["participant", "condition", "is_baseline",
                "is_published_closed_loop", "time"] + MEASURES
    missing = [column for column in required if column not in surveys.columns]
    if missing:
        raise ValueError(f"{SURVEY_CSV} is missing column(s): {missing}")

    absent = [p for p in PARTICIPANTS if p not in set(surveys.participant)]
    if absent:
        raise ValueError(f"{SURVEY_CSV} has no surveys for: {absent}")

    surveys = surveys.assign(day=pd.to_datetime(surveys.time).dt.date)

    # No survey may be in two arms: it would be drawn twice without the figure
    # changing shape, a silent failure, so overlaps are checked here. Surveys
    # in no arm are left out on purpose: the Stage 0 inpatient surveys and the
    # pre-implant surveys outside the baseline window (1,236 and 299 of the
    # 11,986 surveys in the shipped data).
    for participant in PARTICIPANTS:
        selected = [set(select_arm(surveys, participant, arm).index)
                    for arm in ALL_ARMS]
        for first in range(len(selected)):
            for second in range(first + 1, len(selected)):
                shared = selected[first] & selected[second]
                if shared:
                    raise ValueError(
                        f"{participant}: {len(shared)} survey(s) in both "
                        f"'{ALL_ARMS[first]}' and '{ALL_ARMS[second]}'")
        if not selected[ALL_ARMS.index(BASELINE_ARM)]:
            raise ValueError(f"{participant} has no surveys with is_baseline == 1")
        if not selected[ALL_ARMS.index(BLINDED_ARM)]:
            raise ValueError(f"{participant} has no published closed-loop surveys")
    return surveys


def compute_statistics(surveys, arms):
    """Omnibus test across arms, then the two prespecified contrasts."""
    omnibus_rows = []
    pairwise_rows = []
    for measure in MEASURES:
        for participant in PARTICIPANTS:
            values = {arm: select_arm(surveys, participant, arm)[measure].dropna().values
                      for arm in arms}
            testable = {arm: v for arm, v in values.items()
                        if len(v) >= MINIMUM_FOR_VIOLIN}

            if len(testable) >= 2:
                h_statistic, p_omnibus = stats.kruskal(*testable.values())
                omnibus_rows.append({
                    "measure": measure, "participant": participant,
                    "conditions": ", ".join(testable),
                    "n_total": int(sum(len(v) for v in testable.values())),
                    "KruskalWallis_H": h_statistic,
                    "df": len(testable) - 1,
                    "p_omnibus": p_omnibus})

            if BLINDED_ARM not in testable:
                continue
            family = []
            for reference in TEST_REFERENCES:
                if reference not in testable:
                    continue
                u_statistic, p_raw = stats.mannwhitneyu(
                    testable[BLINDED_ARM], testable[reference], alternative="less")
                family.append((reference, u_statistic, p_raw))
            if not family:
                continue
            p_adjusted = multipletests([row[2] for row in family], method="holm")[1]
            for (reference, u_statistic, p_raw), p_holm in zip(family, p_adjusted):
                pairwise_rows.append({
                    "measure": measure, "participant": participant,
                    "comparison": f"{BLINDED_ARM} < {reference}",
                    "MannWhitney_U": u_statistic, "p_raw": p_raw, "p_holm": p_holm,
                    "n_test": len(testable[BLINDED_ARM]),
                    "n_ref": len(testable[reference]),
                    "median_test": float(np.median(testable[BLINDED_ARM])),
                    "median_ref": float(np.median(testable[reference]))})
    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def arm_counts(surveys):
    """n, days, median and IQR per participant x arm x scale."""
    rows = []
    for participant in PARTICIPANTS:
        for arm in ALL_ARMS:
            selected = select_arm(surveys, participant, arm)
            for measure in MEASURES:
                values = selected[measure].dropna()
                rows.append({
                    "participant": participant, "arm": arm, "measure": measure,
                    "n_surveys": len(values),
                    "n_days": selected.loc[values.index, "day"].nunique() if len(values) else 0,
                    "median": values.median() if len(values) else np.nan,
                    "q25": values.quantile(0.25) if len(values) else np.nan,
                    "q75": values.quantile(0.75) if len(values) else np.nan})
    return pd.DataFrame(rows)


def significance_label(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def draw_figure(surveys, pairwise, arms, output_stem, title):
    plt.close("all")
    arm_count = len(arms)
    offset = {arm: (index - (arm_count - 1) / 2) * (0.86 / arm_count)
              for index, arm in enumerate(arms)}
    violin_width = 0.80 / arm_count

    figure, axes = plt.subplots(3, 1, figsize=(10.2, 10.8))
    for axis, measure in zip(axes, MEASURES):
        highest = max([select_arm(surveys, p, arm)[measure].max()
                       for p in PARTICIPANTS for arm in arms
                       if select_arm(surveys, p, arm)[measure].notna().any()] + [1])

        for position, participant in enumerate(PARTICIPANTS):
            for arm in arms:
                values = select_arm(surveys, participant, arm)[measure].dropna().values
                centre = position + offset[arm]
                if values.size == 0:
                    # An empty slot would read as a measurement of zero.
                    axis.plot(centre, highest * 0.02, marker="o", mfc="none",
                              mec=(0.4, 0.4, 0.4), ms=4, mew=0.7)
                    continue
                if values.size >= MINIMUM_FOR_VIOLIN:
                    parts = axis.violinplot([values], positions=[centre],
                                            widths=violin_width,
                                            showextrema=False, showmedians=False)
                    for body in parts["bodies"]:
                        body.set_facecolor(ARM_COLOUR[arm])
                        body.set_alpha(0.55)
                        body.set_edgecolor((0.25, 0.25, 0.25))
                        body.set_linewidth(0.5)
                else:
                    # Too few observations for a density; print n instead.
                    axis.annotate(f"n={values.size}", (centre, highest * 0.06),
                                  ha="center", fontsize=6, color=(0.3, 0.3, 0.3))
                jitter = np.random.default_rng(position * 17 + len(arm)).normal(
                    0, 0.020, values.size)
                axis.scatter(centre + jitter, values, s=3.0, color=POINT_COLOUR,
                             alpha=0.30, lw=0, zorder=3)
                axis.plot([centre - 0.30 / arm_count, centre + 0.30 / arm_count],
                          [np.median(values)] * 2, color="k", lw=1.2, zorder=4)
            if position < len(PARTICIPANTS) - 1:
                axis.axvline(position + 0.5, color=(0.85, 0.85, 0.85), lw=0.6, zorder=0)

        bracket_height = highest * 1.02
        for position, participant in enumerate(PARTICIPANTS):
            rows = pairwise[(pairwise.measure == measure)
                            & (pairwise.participant == participant)]
            for _, row in rows.iterrows():
                reference = row.comparison.split(" < ")[1]
                if reference not in arms:
                    continue
                left = position + offset[BLINDED_ARM]
                right = position + offset[reference]
                axis.plot([left, left, right, right],
                          [bracket_height, bracket_height + highest * 0.022,
                           bracket_height + highest * 0.022, bracket_height],
                          color="k", lw=0.6, clip_on=False)
                axis.annotate(significance_label(row.p_holm),
                              ((left + right) / 2, bracket_height + highest * 0.026),
                              ha="center", fontsize=6.5, clip_on=False)
                bracket_height += highest * 0.062

        axis.set_xticks(range(len(PARTICIPANTS)))
        axis.set_xticklabels(PARTICIPANTS)
        axis.set_ylabel(MEASURE_LABEL[measure])
        axis.set_xlim(-0.55, len(PARTICIPANTS) - 0.45)
        axis.set_ylim(0, bracket_height + highest * 0.02)
        axis.yaxis.set_major_locator(MultipleLocator(MEASURE_TICK_STEP[measure]))
        axis.set_yticks([tick for tick in axis.get_yticks() if 0 <= tick <= highest])

    # Title above the legend, both in figure coordinates so neither can collide
    # with the other or with the top panel.
    figure.suptitle(title, y=0.995, fontsize=10.5)
    figure.legend(handles=[Line2D([], [], marker="s", ls="", ms=6,
                                  mfc=ARM_COLOUR[arm], mec=(0.25, 0.25, 0.25),
                                  label=arm) for arm in arms],
                  loc="upper center", bbox_to_anchor=(0.5, 0.972),
                  ncol=3, frameon=False, fontsize=7.5)
    figure.tight_layout(rect=[0, 0, 1, 0.925])

    for extension in ("pdf", "png", "svg"):
        path = os.path.join(FIGURE_DIRECTORY, f"{output_stem}.{extension}")
        figure.savefig(path, dpi=200, bbox_inches="tight")
        print("wrote", path)
    return figure


def main():
    surveys = load_surveys()
    drawn = sum(len(select_arm(surveys, p, arm))
                for p in PARTICIPANTS for arm in ALL_ARMS)
    print(f"surveys in file: {len(surveys)}; drawn across {len(ALL_ARMS)} arms: {drawn}")
    for arm in ALL_ARMS:
        per_participant = {p: len(select_arm(surveys, p, arm)) for p in PARTICIPANTS}
        print(f"  {arm}: n={sum(per_participant.values())} "
              + " ".join(f"{p} {n}" for p, n in per_participant.items()))

    omnibus, pairwise = compute_statistics(surveys, ALL_ARMS)
    # repr keeps every p-value's full double precision, so the statistics tables
    # and Table S11 reprint exactly what was computed.
    full_precision = lambda value: repr(float(value))
    omnibus.to_csv(OMNIBUS_OUT, index=False, float_format=full_precision)
    pairwise.to_csv(PAIRWISE_OUT, index=False, float_format=full_precision)
    arm_counts(surveys).to_csv(COUNTS_OUT, index=False, float_format=full_precision)
    print(f"wrote {OMNIBUS_OUT}\nwrote {PAIRWISE_OUT}\nwrote {COUNTS_OUT}")
    print(f"contrasts: {len(pairwise)}, significant after Holm: "
          f"{int((pairwise.p_holm < 0.05).sum())}")

    draw_figure(surveys, pairwise, ALL_ARMS,
                "FigureExtended3_raw_pain_distributions",
                "Raw pain scores by stimulation arm")
    draw_figure(surveys, pairwise, ARMS_WITHOUT_OPEN_LOOP,
                "FigureExtended3_raw_pain_distributions_no_open_loop",
                "Raw pain scores by stimulation arm (open-loop omitted)")


if __name__ == "__main__":
    main()