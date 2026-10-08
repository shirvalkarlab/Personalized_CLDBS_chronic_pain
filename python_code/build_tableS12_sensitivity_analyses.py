"""Sensitivity of the closed-loop versus sham model to sham-exposure exclusions.

Refits the primary closed-loop versus sham mixed-effects model under a series of
exclusions, to show whether the reported effect depends on any single participant
or on either of the two participants who contribute most of the sham observations.

Model (identical to the primary analysis):
    pain ~ 1 + condition + (1 + condition | participant)
fitted by restricted maximum likelihood (REML) on daily percent change from
baseline, uniform numerical rating scale, closed-loop and sham days only.

Several of these fits have their best solution on a boundary, where one
random-effects variance is close to zero. A single optimizer run can stop
short of it there and still report success, and where it stops depends on the
computer. Each fit is therefore started from a grid of starting values with
four optimizers, and the solution with the highest restricted likelihood is
kept; on every row this matches MATLAB's fitlme. Where that solution lies on
the boundary, the usual standard error is undefined, and the standard error
of the condition effect is computed given the fitted variances instead, as
fitlme and lme4 report it. p values are Wald z tests.

Writes:
    figures/TableS12_sensitivity_analyses.csv
"""

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "figure5_double_blind_outcomes" / "cohort_daily_percent_change.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "figures"

PAIN_COLUMN = "baseline_percent_diff_mayoNRS"
CONDITIONS = ["closed-loop", "sham"]
EFFECT = "condition[T.closed-loop]"

# Search for the REML optimum: every optimizer from the default start and from
# each random-effects covariance on this grid (variances relative to the
# residual variance, and the intercept-slope correlation).
OPTIMIZERS = ("lbfgs", "nm", "powell", "bfgs")
START_VARIANCES = (0.05, 0.5, 2.0)
START_CORRELATIONS = (-0.9, 0.0, 0.9)


def load_model_data():
    """Return the daily observations that enter the closed-loop versus sham model."""
    daily = pd.read_csv(INPUT_FILE)
    daily = daily[daily["stim_group"].isin(CONDITIONS)]
    daily = daily.rename(columns={PAIN_COLUMN: "pain",
                                  "patient_id": "participant",
                                  "stim_group": "condition"})
    daily = daily.dropna(subset=["pain"])
    daily["condition"] = pd.Categorical(daily["condition"], categories=CONDITIONS[::-1])
    return daily


def start_values(daily, variance_intercept, correlation, variance_slope):
    """Packed start values: fixed effects, then the Cholesky factor of the relative random-effects covariance."""
    off_diagonal = correlation * np.sqrt(variance_intercept * variance_slope)
    covariance = np.array([[variance_intercept, off_diagonal], [off_diagonal, variance_slope]])
    return np.r_[daily["pain"].mean(), 0.0, np.linalg.cholesky(covariance)[np.tril_indices(2)]]


def fit_reml(daily):
    """Return the REML fit with the highest restricted likelihood over all starts and optimizers."""
    model = smf.mixedlm("pain ~ condition", daily,
                        groups=daily["participant"], re_formula="~condition")
    with warnings.catch_warnings():
        # Starts that stop short warn; only the best fit is kept.
        warnings.simplefilter("ignore")
        best = model.fit(reml=True, method="lbfgs")
        grid = [None] + list(itertools.product(START_VARIANCES, START_CORRELATIONS, START_VARIANCES))
        for method, start in itertools.product(OPTIMIZERS, grid):
            options = {} if start is None else {"start_params": start_values(daily, *start)}
            try:
                result = model.fit(reml=True, method=method, maxiter=5000, **options)
            except (np.linalg.LinAlgError, ValueError):
                continue
            # A fit replaces the current best only if it is clearly better, so
            # rows the first fit already solves keep exactly that fit.
            if np.isfinite(result.llf) and result.llf > best.llf + 1e-6:
                best = result
    return best


def conditional_standard_error(result, daily):
    """Standard error of the condition effect given the fitted variances: sqrt of (X' V^-1 X)^-1."""
    information = np.zeros((2, 2))
    random_effects_covariance = np.asarray(result.cov_re)
    for _, group in daily.groupby("participant", observed=True):
        design = np.column_stack([np.ones(len(group)), (group["condition"] == "closed-loop").to_numpy(float)])
        covariance = design @ random_effects_covariance @ design.T + result.scale * np.eye(len(group))
        information += design.T @ np.linalg.solve(covariance, design)
    return float(np.sqrt(np.linalg.inv(information)[1, 1]))


def fit_model(daily):
    """Fit the primary model and return its condition effect, standard error and p value."""
    result = fit_reml(daily)
    estimate = result.params[EFFECT]
    standard_error = result.bse[EFFECT]
    p_value = result.pvalues[EFFECT]
    if not np.isfinite(standard_error):
        # The optimum is on the boundary (a random-effects variance at zero).
        standard_error = conditional_standard_error(result, daily)
        p_value = 2 * stats.norm.sf(abs(estimate / standard_error))
    return {"estimate_percent": round(estimate, 1),
            "standard_error": round(standard_error, 1),
            "p_value": round(p_value, 3),
            "observations": len(daily),
            "closed_loop_days": int((daily["condition"] == "closed-loop").sum()),
            "sham_days": int((daily["condition"] == "sham").sum())}


def build_sensitivity_table(daily):
    """Refit the model under each exclusion and collect the results into one table."""
    rows = []

    def add(label, subset):
        rows.append({"analysis": label, **fit_model(subset)})

    add("Full model (as reported)", daily)

    # Exclude each participant's sham observations in turn. Participants who
    # contributed no sham days are skipped, since removing nothing would simply
    # reproduce the full model.
    sham_days = daily[daily["condition"] == "sham"]["participant"].value_counts()
    for participant in sorted(sham_days[sham_days > 0].index):
        without_sham = daily[~((daily["participant"] == participant) &
                               (daily["condition"] == "sham"))]
        add(f"Excluding {participant} sham observations", without_sham)

    for participant in sorted(daily["participant"].unique()):
        add(f"Leave-one-out: {participant} removed",
            daily[daily["participant"] != participant])

    return pd.DataFrame(rows)


def main():
    daily = load_model_data()
    table = build_sensitivity_table(daily)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIRECTORY / "TableS12_sensitivity_analyses.csv", index=False)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
