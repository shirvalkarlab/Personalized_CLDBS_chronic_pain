"""Sensitivity of the closed-loop versus sham model to sham-exposure exclusions.

Refits the primary closed-loop versus sham mixed-effects model under a series of
exclusions, to show whether the reported effect depends on any single participant
or on either of the two participants who contribute most of the sham observations.

Model (identical to the primary analysis):
    pain ~ 1 + condition + (1 + condition | participant)
fitted by restricted maximum likelihood on daily percent change from baseline,
uniform numerical rating scale, closed-loop and sham days only.

Writes:
    figures/TableS12_sensitivity_analyses.csv
"""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "figure6_double_blind_outcomes" / "cohort_daily_percent_change.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "figures"

PAIN_COLUMN = "baseline_percent_diff_mayoNRS"
CONDITIONS = ["closed-loop", "sham"]


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


def fit_model(daily):
    """Fit the primary model and return its condition effect, standard error and p value."""
    model = smf.mixedlm("pain ~ condition", daily,
                        groups=daily["participant"], re_formula="~condition")
    result = model.fit(reml=True, method="lbfgs")
    effect_name = "condition[T.closed-loop]"
    return {"estimate_percent": round(result.params[effect_name], 1),
            "standard_error": round(result.bse[effect_name], 1),
            "p_value": round(result.pvalues[effect_name], 3),
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
