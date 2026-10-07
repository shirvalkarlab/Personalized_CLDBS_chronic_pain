"""Dispersion and time trends of delivered stimulation parameters.

Summarises how much the delivered stimulation settings varied within each
chronically stimulated hemisphere over the therapy period, and fits a linear
time trend to delivered amplitude. Together these describe whether the dose
required to maintain therapy changed over years of use.

Records are restricted to periods when therapy was on and delivering current
(amplitude above zero). Brief low-frequency test records (below 10 Hz) are
excluded from the frequency summary because they are not therapeutic settings.

Writes:
    data/stimulation_parameter_summary/parameter_dispersion.csv
    data/stimulation_parameter_summary/amplitude_time_trends.csv
"""

from pathlib import Path

import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = PROJECT_ROOT / "data" / "figureS11_programming_history"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "stimulation_parameter_summary"

MINIMUM_THERAPEUTIC_FREQUENCY_HZ = 10
MINIMUM_RECORDS_FOR_TREND = 20
DATE_FORMAT = "%d-%b-%Y %H:%M:%S"
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


def load_delivering_records(parameter_file):
    """Return records for which therapy was on and current was being delivered."""
    records = pd.read_csv(parameter_file)
    is_delivering = (records["therapyStatus"] == 1) & (records["amplitude_mA"] > 0)
    if "effectively_active" in records.columns:
        is_delivering &= records["effectively_active"] == 1
    return records[is_delivering]


def summarise_parameter(values, name):
    """Return median, interquartile range and coefficient of variation."""
    return {f"{name}_median": round(values.median(), 1),
            f"{name}_lower_quartile": round(values.quantile(0.25), 1),
            f"{name}_upper_quartile": round(values.quantile(0.75), 1),
            f"{name}_coefficient_of_variation_percent":
                round(100 * values.std() / values.mean(), 1) if values.mean() else None}


def describe_dispersion(hemisphere, records):
    """Summarise the spread of delivered settings for one hemisphere."""
    setting_columns = ["amplitude_mA", "pulse_width_us", "frequency_Hz"]
    summary = {"hemisphere": hemisphere,
               "records": len(records),
               "distinct_settings": len(records[setting_columns].drop_duplicates())}
    summary.update(summarise_parameter(records["amplitude_mA"], "amplitude_mA"))
    summary.update(summarise_parameter(records["pulse_width_us"], "pulse_width_us"))

    therapeutic = records[records["frequency_Hz"] >= MINIMUM_THERAPEUTIC_FREQUENCY_HZ]
    summary.update(summarise_parameter(therapeutic["frequency_Hz"], "frequency_Hz"))
    return summary


def describe_amplitude_trend(hemisphere, records):
    """Fit a linear trend of delivered amplitude against time, in milliamps per year."""
    timestamps = pd.to_datetime(records["datetime"], format=DATE_FORMAT, errors="coerce")
    records = records.assign(timestamp=timestamps).dropna(subset=["timestamp"])
    records = records.sort_values("timestamp")
    if len(records) < MINIMUM_RECORDS_FOR_TREND:
        return None

    elapsed_years = (records["timestamp"] - records["timestamp"].min()).dt.total_seconds() / SECONDS_PER_YEAR
    regression = stats.linregress(elapsed_years, records["amplitude_mA"])
    half_width = 1.96 * regression.stderr
    return {"hemisphere": hemisphere,
            "years_observed": round(elapsed_years.max(), 2),
            "records": len(records),
            "slope_mA_per_year": round(regression.slope, 3),
            "confidence_interval_low": round(regression.slope - half_width, 3),
            "confidence_interval_high": round(regression.slope + half_width, 3),
            "p_value": round(regression.pvalue, 4)}


def main():
    dispersion_rows = []
    trend_rows = []

    for parameter_file in sorted(INPUT_DIRECTORY.glob("*_stim_parameters.csv")):
        hemisphere = parameter_file.name.split("_")[0]
        records = load_delivering_records(parameter_file)
        if records.empty:
            continue

        dispersion_rows.append(describe_dispersion(hemisphere, records))
        trend = describe_amplitude_trend(hemisphere, records)
        if trend is not None:
            trend_rows.append(trend)

    dispersion = pd.DataFrame(dispersion_rows)
    trends = pd.DataFrame(trend_rows)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    dispersion.to_csv(OUTPUT_DIRECTORY / "parameter_dispersion.csv", index=False)
    trends.to_csv(OUTPUT_DIRECTORY / "amplitude_time_trends.csv", index=False)

    print(dispersion.to_string(index=False))
    print()
    print(trends.to_string(index=False))


if __name__ == "__main__":
    main()
