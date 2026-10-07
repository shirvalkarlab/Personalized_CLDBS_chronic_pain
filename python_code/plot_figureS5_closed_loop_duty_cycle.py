"""
Figure S5 - Closed-loop effective duty cycle.

Panel a shows the duty cycle per participant during the blinded testing window.
Panel b, described below, covers the whole unblinded record.

This script asks whether the participants whose pain did not improve on
closed-loop stimulation show a different day-to-day spread in how hard their
device worked than the participants whose pain did improve. A device that is
progressively losing its effect might be expected to stimulate more, or more
erratically, over time.

WHAT THE DUTY CYCLE IS
    While a device runs in closed-loop (adaptive) mode it decides moment to
    moment whether to stimulate. The device logs, for every hour, the fraction
    of that hour spent actually delivering stimulation. Averaging those hourly
    fractions over a calendar day gives the daily effective duty cycle, in
    percent. Zero percent means the device chose not to stimulate all day; one
    hundred percent means it stimulated continuously.

WHICH DAYS ARE COUNTED
    Only days the device genuinely spent in closed-loop mode are used. The
    programming history is read interval by interval and a day is kept only if
    more than ninety percent of it was spent with the device in adaptive mode,
    with therapy switched on, and with a stimulation contact actually active.
    This gate matters: many days in the raw duty-cycle logs were recorded while
    the device was in open-loop mode or switched off, and those days would
    otherwise be counted as closed-loop days with a duty cycle of zero.

    Blinded sham days are removed. Everything else is kept, so this is the
    whole unblinded chronic closed-loop record for each participant, not just
    the double-blind windows. Participants with two implanted devices have the
    two hemispheres averaged within each calendar day, so a bilateral
    participant contributes one value per day like everyone else.

WHO COUNTS AS A RESPONDER
    A responder is a participant whose pain improved by at least thirty percent
    from baseline, during closed-loop stimulation, on the pain scale chosen for
    them in advance as their index measure. Those percent changes and threshold
    flags are published in Extended Data Table 2 and are read from it here
    rather than recomputed, so the figure follows the published table.

    This gives three responders (P1, P3, P5) and two non-responders (P2, P4),
    the same grouping used elsewhere in the paper.

WHAT PANEL B SHOWS
    One box per participant with every daily value swarmed on top, responders
    first and non-responders second. Each participant's standard deviation and
    interquartile range are printed under their box so the dispersion can be
    read directly off the panel rather than
    inferred from the box heights.

WHAT IS TESTED
    Two questions are asked, and they have different answers.

    Spread does not separate the groups. The standard deviations run from 17 to
    31 percentage points across all five participants with the two groups fully
    interleaved, so there is nothing to test and nothing is claimed.

    The typical level does separate them: every responder's median duty cycle
    falls below every non-responder's. Four models are fitted to that
    separation and all are written to duty_cycle_group_tests.csv, including the
    ones that overstate it, because the gap between them is the point. Five
    participants split three against two can be relabelled only ten ways, so an
    exact test cannot go below p = 0.1 one-sided however clean the separation
    is. A mixed-effects model over all 1,259 days returns p = 0.007 if its days
    are treated as independent and p = 0.07 once the degrees of freedom match a
    predictor that varies only between participants; permuting participant
    labels through that same model returns the exact-test floor again. A
    logistic model has no finite solution here because the groups do not
    overlap, and the penalized version that does converge gives p = 0.21.

    The consistent reading is that the separation is complete but the study is
    too small to certify it, and that any p below 0.1 from these data comes
    from counting days rather than people.

DATA INPUTS (the first three in data/figureS11_programming_history/)
    <hemisphere>_stim_parameters.csv    one row per programming interval, used
        to decide which days were truly closed-loop.
    <hemisphere>_daily_duty_cycle.csv   the measured daily duty cycle.
    <participant>_sham_windows.csv      blinded sham episodes, which are removed.
    data/figureS5_closed_loop_duty_cycle/blinded_daily_duty_cycle.csv
                                        daily duty cycle in the blinded window
                                        (panel a).
    figures/ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham.csv
                                        responder status.

DATA OUTPUTS (data/figureS5_closed_loop_duty_cycle/)
    daily_duty_cycle_closed_loop.csv    one row per participant per day.
    duty_cycle_dispersion_summary.csv   one row per participant with the
        dispersion statistics and the change over time.
    duty_cycle_group_tests.csv          one row per responder-group statistic.

FIGURE OUTPUT
    figures/FigureS5_closed_loop_duty_cycle.{pdf,png,svg}

USAGE
    python plot_figureS5_closed_loop_duty_cycle.py
    # optional: set the repository root if running from elsewhere
    NATMED_REPO=/path/to/Shirvalkar_2026_NatMed_deposit \
        python plot_figureS5_closed_loop_duty_cycle.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Repository root: the folder that contains data/, figures/, python_code/.
# Defaults to two levels up from this file (python_code/ -> repository root).
repository_root = os.environ.get(
    "NATMED_REPO",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

programming_directory = os.path.join(
    repository_root, "data", "figureS11_programming_history")
output_directory = os.path.join(
    repository_root, "data", "figureS5_closed_loop_duty_cycle")
figure_directory = os.path.join(repository_root, "figures")

os.makedirs(output_directory, exist_ok=True)
os.makedirs(figure_directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Study structure
# ---------------------------------------------------------------------------
# Implanted hemispheres. P1's left device was explanted, so P1 is right only.
hemispheres = ["P1R", "P2L", "P2R", "P3L", "P3R", "P4L", "P4R", "P5L", "P5R"]
participants = ["P1", "P2", "P3", "P4", "P5"]

# Responder status is read from Extended Data Table 2 at run time rather than
# hard-coded, so the figure follows the published table if that table changes.
outcome_table_path = os.path.join(
    repository_root, "figures",
    "ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham.csv")

# Panel a plots the blinded-window duty cycle, one file, participants in order.
blinded_duty_cycle_path = os.path.join(
    repository_root, "data", "figureS5_closed_loop_duty_cycle",
    "blinded_daily_duty_cycle.csv")

# Panel b draws responders first so the two groups read as contiguous blocks.
plot_order = ["P1", "P3", "P5", "P2", "P4"]

# A day counts as closed-loop only if this much of it was spent in adaptive
# mode with therapy on and a contact active.
minimum_closed_loop_fraction = 0.9


# ---------------------------------------------------------------------------
# Figure appearance
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "Arial",
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,      # keep text editable in the exported PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# Okabe-Ito colours. Blue marks responders, vermillion marks non-responders;
# the pair stays distinguishable under every common form of colour blindness.
responder_colour = "#0072B2"
non_responder_colour = "#D55E00"
box_fill = "#F0F0F0"
grey = "#4D4D4D"


# ---------------------------------------------------------------------------
# Reading the device record
# ---------------------------------------------------------------------------
def read_response_groups():
    """Responder status per participant, from the published outcome table.

    Extended Data Table 2 carries one row per participant per pain scale, a tick
    in the index-metric column on the scale chosen for that participant in
    advance, and a yes/no flag for thirty percent improvement during
    closed-loop. A participant is a responder when that flag is set on their
    index scale.
    """
    table = pd.read_csv(outcome_table_path)
    index_rows = table[table["prospective_index_metric"].notna()]
    groups = {}
    for participant, row in index_rows.set_index("participant").iterrows():
        met_threshold = str(row["responder_30pct_closed_loop"]).strip().upper() == "Y"
        groups[participant] = "Responder" if met_threshold else "Non-responder"
    missing = [p for p in participants if p not in groups]
    if missing:
        raise ValueError(
            f"No index-metric row in Extended Data Table 2 for: {missing}")
    return groups


def read_programming_history(hemisphere):
    """Programming intervals for one hemisphere, oldest first."""
    path = os.path.join(
        programming_directory, f"{hemisphere}_stim_parameters.csv")
    if not os.path.isfile(path):
        return None
    history = pd.read_csv(path)
    history["datetime"] = pd.to_datetime(
        history["datetime"], format="%d-%b-%Y %H:%M:%S")
    return history.sort_values("datetime").reset_index(drop=True)


def read_measured_duty_cycle(hemisphere):
    """Measured daily effective duty cycle for one hemisphere."""
    path = os.path.join(
        programming_directory, f"{hemisphere}_daily_duty_cycle.csv")
    if not os.path.isfile(path):
        return None
    measured = pd.read_csv(path)
    measured["day"] = pd.to_datetime(measured["day"], format="%d-%b-%Y")
    return measured


def read_sham_windows(participant):
    """Blinded sham episodes for one participant, as (start, stop) pairs."""
    path = os.path.join(
        programming_directory, f"{participant}_sham_windows.csv")
    if not os.path.isfile(path):
        return []
    windows = pd.read_csv(path)
    return list(zip(
        pd.to_datetime(windows["sham_start"], format="%d-%b-%Y"),
        pd.to_datetime(windows["sham_stop"], format="%d-%b-%Y")))


def closed_loop_fraction_per_day(history):
    """How much of each calendar day the device spent truly in closed-loop.

    Each row of the programming history states the settings that took effect at
    that moment and stay in force until the next row. Rows where the device was
    in adaptive mode, with therapy on and a contact active, are spread across
    the calendar days they span, and the spanned time is summed per day.
    """
    interval_start = history["datetime"].values
    # The final interval has no successor; give it one nominal day.
    interval_stop = np.append(
        interval_start[1:], interval_start[-1] + np.timedelta64(1, "D"))
    is_closed_loop = (
        (history["is_adaptive"] == 1)
        & (history["therapyStatus"] == 1)
        & (history["effectively_active"] == 1)).values

    spans = []
    for start, stop, active in zip(interval_start, interval_stop, is_closed_loop):
        if not active:
            continue
        start, stop = pd.Timestamp(start), pd.Timestamp(stop)
        day = start.normalize()
        while day < stop:
            next_day = day + pd.Timedelta(days=1)
            covered = min(stop, next_day) - max(start, day)
            spans.append((day, covered.total_seconds() / 86400))
            day = next_day

    if not spans:
        return pd.DataFrame(columns=["day", "closed_loop_fraction"])

    per_day = (pd.DataFrame(spans, columns=["day", "closed_loop_fraction"])
                 .groupby("day", as_index=False)["closed_loop_fraction"].sum())
    # Overlapping intervals can nudge a day past one; a day cannot exceed itself.
    per_day["closed_loop_fraction"] = per_day["closed_loop_fraction"].clip(upper=1.0)
    return per_day


def build_daily_series(response_group):
    """One duty-cycle value per participant per closed-loop day."""
    per_hemisphere = {}
    for hemisphere in hemispheres:
        history = read_programming_history(hemisphere)
        measured = read_measured_duty_cycle(hemisphere)
        if history is None or measured is None:
            continue

        kept = measured.merge(
            closed_loop_fraction_per_day(history), on="day", how="left")
        kept["closed_loop_fraction"] = kept["closed_loop_fraction"].fillna(0)
        kept = kept[kept["closed_loop_fraction"] > minimum_closed_loop_fraction]

        for sham_start, sham_stop in read_sham_windows(hemisphere[:2]):
            kept = kept[~((kept["day"] >= sham_start) & (kept["day"] <= sham_stop))]

        per_hemisphere[hemisphere] = kept[["day", "percentDutyCycle"]]

    daily = []
    for participant in participants:
        sides = [table for name, table in per_hemisphere.items()
                 if name.startswith(participant)]
        if not sides:
            continue
        # Average the two hemispheres within a day so bilateral participants
        # contribute one value per day, like the unilateral participant.
        combined = (pd.concat(sides)
                      .groupby("day", as_index=False)["percentDutyCycle"].mean()
                      .sort_values("day")
                      .reset_index(drop=True))
        combined.insert(0, "participant", participant)
        combined.insert(1, "response_group", response_group[participant])
        combined["days_since_start"] = (
            combined["day"] - combined["day"].min()).dt.days
        daily.append(combined)

    daily = pd.concat(daily, ignore_index=True)
    return daily.rename(columns={"percentDutyCycle": "duty_cycle_percent"})


# ---------------------------------------------------------------------------
# Dispersion statistics
# ---------------------------------------------------------------------------
def summarise_dispersion(daily, response_group):
    """Per-participant spread of the daily duty cycle, plus its drift in time.

    Spread is reported three ways because they answer slightly different
    questions: the standard deviation and interquartile range are absolute, in
    percentage points, while the coefficient of variation expresses spread
    relative to each participant's own typical duty cycle. The trend statistics
    ask the separate question of whether the device worked steadily harder as
    the months passed, which is what habituation would look like.
    """
    rows = []
    for participant, block in daily.groupby("participant"):
        values = block["duty_cycle_percent"].to_numpy()
        elapsed = block["days_since_start"].to_numpy()
        first_quartile, third_quartile = np.percentile(values, [25, 75])

        # Drift across the record, at the level of single days and of months.
        daily_rho, daily_p = stats.spearmanr(elapsed, values)
        by_month = (block.assign(month=block["days_since_start"] // 30)
                         .groupby("month")["duty_cycle_percent"]
                         .agg(["median", "size"])
                         .reset_index())
        by_month = by_month[by_month["size"] >= 5]
        if len(by_month) >= 3:
            month_rho, month_p = stats.spearmanr(
                by_month["month"], by_month["median"])
        else:
            month_rho, month_p = np.nan, np.nan

        rows.append({
            "participant": participant,
            "response_group": response_group[participant],
            "days": len(values),
            "median": np.median(values),
            "mean": values.mean(),
            "standard_deviation": values.std(ddof=1),
            "interquartile_range": third_quartile - first_quartile,
            "first_quartile": first_quartile,
            "third_quartile": third_quartile,
            "coefficient_of_variation_percent":
                100 * values.std(ddof=1) / values.mean(),
            "minimum": values.min(),
            "maximum": values.max(),
            "trend_rho_daily": daily_rho,
            "trend_p_daily": daily_p,
            "trend_rho_monthly": month_rho,
            "trend_p_monthly": month_p,
            "first_3_months_median": by_month["median"].iloc[:3].mean(),
            "last_3_months_median": by_month["median"].iloc[-3:].mean(),
        })

    summary = pd.DataFrame(rows)
    summary["change_first_to_last"] = (
        summary["last_3_months_median"] - summary["first_3_months_median"])
    return summary.set_index("participant").loc[plot_order].reset_index()


# ---------------------------------------------------------------------------
# Testing whether the two groups differ
# ---------------------------------------------------------------------------
def firth_logistic(design, outcome, max_iterations=200, tolerance=1e-8):
    """Logistic regression with Firth's penalty.

    Ordinary logistic regression has no finite answer when one group's values
    all sit above the other's, which is the case here: the coefficient simply
    grows without limit. Firth's penalty removes that divergence and returns a
    finite estimate, at the cost of shrinking it toward zero.
    """
    coefficients = np.zeros(design.shape[1])
    information = np.eye(design.shape[1])
    for _ in range(max_iterations):
        probability = 1 / (1 + np.exp(-(design @ coefficients)))
        weight = probability * (1 - probability)
        weighted = design * weight[:, None]
        information = design.T @ weighted
        inverse = np.linalg.pinv(information)
        leverage = np.diag(weighted @ inverse @ design.T)
        score = design.T @ (outcome - probability + leverage * (0.5 - probability))
        step = inverse @ score
        coefficients = coefficients + step
        if np.max(np.abs(step)) < tolerance:
            break
    return coefficients, np.sqrt(np.diag(np.linalg.pinv(information)))


def test_group_difference(daily, summary, response_group):
    """Does the duty cycle distinguish responders from non-responders?

    Four models are fitted, deliberately including the ones that overstate the
    result, because the gap between them is the finding.

    The unit of replication is the participant, not the day. Five participants
    split three against two can be relabelled in only ten ways, so an exact
    test cannot return a p-value below 0.1 no matter how cleanly the groups
    separate. That floor is a property of the sample
    size rather than of the data.

    1. Exact permutation on the five participant medians.
    2. A mixed-effects model on all days with a per-participant random
       intercept, reported twice: once with the p-value the software prints,
       which treats every day as independent, and once with the degrees of
       freedom appropriate to a predictor that varies only between
       participants. Responder status is a property of a person, so the second
       is the honest one; the first is reported to show how far pseudo-
       replication inflates the result.
    3. The same mixed model with participant labels permuted, which respects
       the clustering and assumes nothing about degrees of freedom.
    4. Firth penalized logistic regression predicting responder status from a
       participant's median duty cycle.
    """
    from itertools import combinations

    medians = summary.set_index("participant")["median"]
    responders = [p for p in medians.index if response_group[p] == "Responder"]
    non_responders = [p for p in medians.index
                      if response_group[p] == "Non-responder"]
    labellings = list(combinations(medians.index, len(non_responders)))

    results = {}

    # --- 1. Exact permutation on the participant medians --------------------
    observed_gap = medians[non_responders].mean() - medians[responders].mean()
    gaps = np.array([
        medians[list(candidate)].mean()
        - medians[[p for p in medians.index if p not in candidate]].mean()
        for candidate in labellings])
    results["median_gap_percentage_points"] = observed_gap
    results["permutation_p_one_sided"] = (gaps >= observed_gap).mean()
    results["permutation_p_two_sided"] = (np.abs(gaps) >= abs(observed_gap)).mean()
    results["separation_is_complete"] = bool(
        medians[responders].max() < medians[non_responders].min())

    # --- 2. Mixed-effects model on every day --------------------------------
    days = daily.copy()
    days["is_non_responder"] = (
        days["response_group"] == "Non-responder").astype(int)
    model = smf.mixedlm("duty_cycle_percent ~ is_non_responder", days,
                        groups=days["participant"]).fit(reml=True)
    effect = model.params["is_non_responder"]
    standard_error = model.bse["is_non_responder"]
    t_statistic = effect / standard_error
    between_participant_df = days["participant"].nunique() - 2

    between_variance = float(model.cov_re.iloc[0, 0])
    results["mixed_model_effect_percentage_points"] = effect
    results["mixed_model_standard_error"] = standard_error
    results["mixed_model_p_treating_days_as_independent"] = \
        model.pvalues["is_non_responder"]
    results["mixed_model_p_between_participants"] = \
        2 * stats.t.sf(abs(t_statistic), between_participant_df)
    results["intraclass_correlation"] = \
        between_variance / (between_variance + model.scale)

    # --- 3. The same model with participant labels permuted -----------------
    permuted_t = []
    for candidate in labellings:
        relabelled = days.assign(
            is_non_responder=days["participant"].isin(candidate).astype(int))
        fitted = smf.mixedlm("duty_cycle_percent ~ is_non_responder", relabelled,
                             groups=relabelled["participant"]).fit(reml=True)
        permuted_t.append(
            fitted.params["is_non_responder"] / fitted.bse["is_non_responder"])
    permuted_t = np.array(permuted_t)
    results["mixed_model_permutation_p_one_sided"] = \
        (permuted_t >= t_statistic).mean()
    results["mixed_model_permutation_p_two_sided"] = \
        (np.abs(permuted_t) >= abs(t_statistic)).mean()

    # --- 4. Firth logistic on the participant medians -----------------------
    # Duty cycle is scaled to tens of percentage points so the odds ratio is
    # reported per ten-point increment rather than per single point.
    design = np.column_stack([np.ones(len(medians)), medians.values / 10.0])
    outcome = np.array([1.0 if response_group[p] == "Non-responder" else 0.0
                        for p in medians.index])
    coefficients, standard_errors = firth_logistic(design, outcome)
    results["firth_odds_ratio_per_10_points"] = np.exp(coefficients[1])
    results["firth_p"] = 2 * stats.norm.sf(abs(coefficients[1] / standard_errors[1]))

    results["smallest_attainable_p_one_sided"] = 1 / len(labellings)
    results["smallest_attainable_p_two_sided"] = 1 / len(labellings)
    return results


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def density_jitter(values, width=0.34, bins=26):
    """Horizontal offsets that widen the swarm where points pile up."""
    counts, edges = np.histogram(values, bins=bins)
    slot = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
    crowding = counts[slot] / max(counts.max(), 1)
    spread = width * np.sqrt(crowding)
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, size=len(values)) * spread


def read_blinded_duty_cycle():
    """Daily duty cycle during the blinded testing window, for panel a."""
    blinded = pd.read_csv(blinded_duty_cycle_path)
    blinded["day"] = pd.to_datetime(blinded["day"], format="%d-%b-%Y")
    return blinded


def draw_blinded_panel(axes, blinded):
    """Panel a: duty cycle across the blinded window, hemispheres averaged."""
    for position, participant in enumerate(participants, start=1):
        values = blinded.loc[blinded["participant"] == participant,
                             "duty_cycle_percent"].to_numpy()
        box = axes.boxplot(
            values, positions=[position], widths=0.56, patch_artist=True,
            showfliers=False, medianprops=dict(color="black", linewidth=1.0),
            boxprops=dict(facecolor=box_fill, edgecolor="black", linewidth=0.5),
            whiskerprops=dict(color="black", linewidth=0.5),
            capprops=dict(color="black", linewidth=0.5))
        for patch in box["boxes"]:
            patch.set_zorder(2)
        axes.scatter(position + density_jitter(values), values,
                     s=5, color=grey, alpha=0.55, linewidths=0.15,
                     edgecolors="white", zorder=3)

    axes.set_xticks(range(1, len(participants) + 1))
    axes.set_xticklabels(participants)
    axes.set_xlim(0.4, len(participants) + 0.6)
    axes.set_ylim(-4, 112)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_ylabel("Daily effective duty cycle (%)")
    axes.set_title("Blinded testing window", loc="left", fontweight="normal")
    axes.tick_params(direction="out", length=2.5, width=0.5)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_linewidth(0.5)


def draw_dispersion_panel(axes, daily, summary, response_group):
    """Panel b: duty cycle across the whole unblinded closed-loop record."""
    for position, participant in enumerate(plot_order, start=1):
        values = daily.loc[daily["participant"] == participant,
                           "duty_cycle_percent"].to_numpy()
        colour = (responder_colour
                  if response_group[participant] == "Responder"
                  else non_responder_colour)

        # Box first, raw points on top: the summary never hides the data.
        box = axes.boxplot(
            values, positions=[position], widths=0.56, patch_artist=True,
            showfliers=False, medianprops=dict(color="black", linewidth=1.0),
            boxprops=dict(facecolor=box_fill, edgecolor="black", linewidth=0.5),
            whiskerprops=dict(color="black", linewidth=0.5),
            capprops=dict(color="black", linewidth=0.5))
        for patch in box["boxes"]:
            patch.set_zorder(2)

        axes.scatter(position + density_jitter(values), values,
                     s=5, color=colour, alpha=0.55, linewidths=0.15,
                     edgecolors="white", zorder=3)

    # Separate the two groups without drawing a heavy divider.
    axes.axvline(3.5, color=grey, linewidth=0.5, linestyle=(0, (2, 2)),
                 zorder=1)
    axes.text(2.0, 106, "Responders", ha="center", va="bottom",
              fontsize=8, color=responder_colour)
    axes.text(4.5, 106, "Non-responders", ha="center", va="bottom",
              fontsize=8, color=non_responder_colour)

    # The dispersion becomes part of the tick label,
    # so it is read with the participant rather than floating beneath the axis.
    indexed = summary.set_index("participant")
    tick_labels = [
        f"{participant}\nSD {indexed.loc[participant, 'standard_deviation']:.0f}"
        f"   IQR {indexed.loc[participant, 'interquartile_range']:.0f}"
        for participant in plot_order]

    axes.set_xticks(range(1, len(plot_order) + 1))
    axes.set_xticklabels(tick_labels, fontsize=7, linespacing=1.6)
    axes.set_xlim(0.4, len(plot_order) + 0.6)
    axes.set_ylim(-4, 112)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_ylabel("Daily effective duty cycle (%)")
    axes.set_title("Whole unblinded closed-loop record",
                   loc="left", fontweight="normal")
    axes.tick_params(direction="out", length=2.5, width=0.5)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_linewidth(0.5)


def draw_figure(blinded, daily, summary, response_group):
    """Both panels side by side on a shared duty-cycle axis."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    draw_blinded_panel(left, blinded)
    draw_dispersion_panel(right, daily, summary, response_group)
    right.set_ylabel("")

    for axes, letter in ((left, "a"), (right, "b")):
        axes.text(-0.13, 1.06, letter, transform=axes.transAxes,
                  fontsize=11, fontweight="bold", va="bottom", ha="left")

    figure.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.17,
                           wspace=0.14)
    return figure


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def main():
    response_group = read_response_groups()
    daily = build_daily_series(response_group)
    summary = summarise_dispersion(daily, response_group)

    daily_out = daily.copy()
    daily_out["day"] = daily_out["day"].dt.strftime("%Y-%m-%d")
    daily_out.to_csv(
        os.path.join(output_directory, "daily_duty_cycle_closed_loop.csv"),
        index=False)
    summary.round(4).to_csv(
        os.path.join(output_directory, "duty_cycle_dispersion_summary.csv"),
        index=False)

    tests = test_group_difference(daily, summary, response_group)
    pd.Series(tests).rename("value").rename_axis("statistic").to_frame().round(4).to_csv(
        os.path.join(output_directory, "duty_cycle_group_tests.csv"))

    blinded = read_blinded_duty_cycle()
    figure = draw_figure(blinded, daily, summary, response_group)
    stem = os.path.join(figure_directory, "FigureS5_closed_loop_duty_cycle")
    for extension in ("pdf", "png", "svg"):
        figure.savefig(f"{stem}.{extension}", bbox_inches="tight")

    columns = ["participant", "response_group", "days", "median",
               "standard_deviation", "interquartile_range",
               "coefficient_of_variation_percent", "change_first_to_last"]
    print(summary[columns].round(1).to_string(index=False))
    print()
    for name, value in tests.items():
        print(f"{name:52s} {value:.4g}" if isinstance(value, float)
              else f"{name:52s} {value}")
    return figure, daily, summary, tests


if __name__ == "__main__":
    main()
