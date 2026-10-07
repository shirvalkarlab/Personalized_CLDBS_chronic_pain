"""Render Supplementary Table S11 as a Word document.

Reads the two statistics tables written by
plot_extended_figure3_raw_distributions.py and formats them as Table S11.1
(Kruskal-Wallis omnibus screen) and Table S11.2 (directional post-hoc
contrasts).

Reads   data/extended_figures_raw_pain/ExtFig3_omnibus_condition_stats.csv
        data/extended_figures_raw_pain/ExtFig3_pairwise_condition_stats.csv
        data/extended_figures_raw_pain/pain_surveys.csv (data-quality check:
        a caveat is added if an arm's scores are mostly zeros contradicted by
        the paired NRS; none is added for the shipped data)
Writes  figures/TableS11_extended_data_figure3_statistics.docx

Usage:
    python export_tableS11_word.py
"""

import os

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATISTICS_DIRECTORY = os.path.join(PROJECT_ROOT, "data",
                                    "extended_figures_raw_pain")
FIGURE_DIRECTORY = os.path.join(PROJECT_ROOT, "figures")
OUTPUT_NAME = "TableS11_extended_data_figure3_statistics.docx"

# Printed measure names, in the order the panels appear in Extended Data Figure 3.
# Bare abbreviations: the scales and their ranges are named in the preamble, so
# repeating "Pain intensity VAS (0-100)" on 15 rows only widens the column.
MEASURE_LABELS = {
    "mayoNRS": "NRS",
    "painVAS": "VAS",
    "MPQtotal": "MPQ",
}

TITLE = ("Supplementary Table S11 | Full statistics for Extended Data Figure 3 "
         "(raw pain-score distributions by stimulation arm).")

PREAMBLE = (
    "Scales are the numeric rating scale (NRS, 0-10), pain intensity visual "
    "analogue scale (VAS, 0-100) and McGill Pain Questionnaire short-form total "
    "(MPQ). Extended Data Figure 3 draws six mutually exclusive arms: baseline, "
    "open-loop optimization (unblinded), closed-loop optimization (unblinded), "
    "closed-loop double-blind, sham double-blind (0 mA), and washout with "
    "carryover (0 mA). Per participant and scale, all arms were screened with a "
    "two-sided Kruskal-Wallis test (Table S11.1; Arms gives how many entered "
    "that test). The closed-loop double-blind arm -- the window reported in "
    "Extended Data Table 2 -- was then compared against two references, "
    "baseline and sham, with one-tailed Mann-Whitney U tests (alternative "
    "hypothesis, closed-loop lower), Holm-corrected within each "
    "participant-by-scale family (Table S11.2). n, surveys; centre, median. "
    "The optimization and washout arms are shown in the figure for "
    "completeness and were not tested: they preceded blinding or are defined by "
    "carryover, and contrasting them would enlarge the correction family. An "
    "arm enters a test only where it has at least three surveys, so "
    "participants differ in which contrasts are available."
)

OMNIBUS_TITLE = "Table S11.1 | Kruskal-Wallis omnibus tests (two-sided)."

PAIRWISE_TITLE = ("Table S11.2 | Directional post-hoc contrasts, one-tailed "
                  "Mann-Whitney U testing closed-loop double-blind < "
                  "reference, Holm-corrected within family. CL, closed-loop "
                  "double-blind.")

FOOTNOTE = ("*p < 0.05, **p < 0.01, ***p < 0.001; ns, not significant. "
            "Source data: ExtFig3_omnibus_condition_stats.csv, "
            "ExtFig3_pairwise_condition_stats.csv.")

SURVEY_CSV = os.path.join(STATISTICS_DIRECTORY, "pain_surveys.csv")

# A pain score of exactly 0 recorded while the same survey's NRS is above 0 is not
# a plausible reading: a respondent reporting pain on one instrument cannot report
# none on another. Where such rows dominate an arm, the scale was not collected and
# the zeros stand in for absent data, which biases that arm's median toward 0.
IMPLAUSIBLE_ZERO_SHARE = 0.5


def implausible_zero_arms(surveys):
    """Arms whose scores are mostly zeros contradicted by the paired NRS.

    Returns [(participant, measure, n_zero_with_pain, n_scored)], one entry per
    participant x measure whose share exceeds IMPLAUSIBLE_ZERO_SHARE. An empty
    list means no arm looks affected, which is the expected result once the
    source data is sound.
    """
    flagged = []
    for measure in MEASURE_LABELS:
        if measure == "mayoNRS":
            continue
        for participant in sorted(surveys.participant.unique()):
            arm = surveys[(surveys.participant == participant)
                          & surveys[measure].notna() & surveys.mayoNRS.notna()]
            if arm.empty:
                continue
            contradicted = int(((arm[measure] == 0) & (arm.mayoNRS > 0)).sum())
            if contradicted / len(arm) > IMPLAUSIBLE_ZERO_SHARE:
                flagged.append((participant, measure, contradicted, len(arm)))
    return flagged


def data_quality_caveat(flagged):
    """Sentence naming the affected arms, or '' when none are flagged."""
    if not flagged:
        return ""
    parts = [f"{participant} {MEASURE_LABELS[measure]} "
             f"({contradicted} of {scored} scored surveys)"
             for participant, measure, contradicted, scored in flagged]
    return ("Data caveat: in the source survey table, the following record a "
            "score of exactly 0 on most surveys where the same survey's NRS is "
            "above 0, which indicates the scale was not collected rather than "
            "scored 0 — " + "; ".join(parts) + ". Statistics for those "
            "participant-by-measure rows are biased toward 0 and should not be "
            "interpreted until the source column is corrected.")

OMNIBUS_HEADERS = ["Scale", "Participant", "Arms", "n", "H",
                   "df", "p", "Sig."]

PAIRWISE_HEADERS = ["Scale", "Participant", "Reference", "n (CL)", "n (ref)",
                    "Median CL", "Median ref", "U", "p (Holm)", "Sig."]

BODY_FONT = "Arial"
BODY_SIZE = Pt(8)
TITLE_SIZE = Pt(9)


def significance_marker(p_value):
    """Asterisk code for a p-value, matching the table footnote."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def format_p_value(p_value):
    """p-values print to three decimals, with a floor of <0.001."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def format_number(value, decimals=2):
    """Trim trailing zeros so 4.00 prints as 4 and 8.50 as 8.5.

    Trimming is applied only to the fractional part: rstrip on the whole
    string would turn an integer like 2670 into 267.
    """
    if pd.isna(value):
        return ""
    if decimals == 0:
        return f"{float(value):.0f}"
    text = f"{float(value):.{decimals}f}"
    whole, _, fraction = text.partition(".")
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def write_cell(cell, text, bold=False):
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = BODY_FONT
    run.font.size = BODY_SIZE


def add_paragraph(document, text, bold=False, size=BODY_SIZE):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = BODY_FONT
    run.font.size = size
    return paragraph


def add_table(document, headers, rows):
    word_table = document.add_table(rows=1, cols=len(headers))
    word_table.style = "Table Grid"
    for position, header in enumerate(headers):
        write_cell(word_table.rows[0].cells[position], header, bold=True)
    for row_values in rows:
        cells = word_table.add_row().cells
        for position, value in enumerate(row_values):
            write_cell(cells[position], value)
    return word_table


def reference_of(comparison):
    """The reference arm of a contrast.

    Every contrast tests the same arm, so printing the full
    "Closed-loop, double-blind < Baseline" on each row repeats 28 characters and
    widens the column for no information. The test arm is named in the caption.
    """
    return str(comparison).split(" < ")[-1]


def omnibus_rows(omnibus):
    """One printed row per measure and participant, measures in panel order."""
    rows = []
    for measure, label in MEASURE_LABELS.items():
        subset = omnibus[omnibus.measure == measure].sort_values("participant")
        for _, record in subset.iterrows():
            rows.append([
                label,
                str(record.participant),
                # Arms that entered the test. Taken from df, not by splitting
                # the arm-name list: the names themselves contain commas
                # ("Sham, double-blind (0 mA)"), so a split inflates the count.
                str(int(record.df) + 1),
                str(int(record.n_total)),
                format_number(record.KruskalWallis_H),
                str(int(record.df)),
                format_p_value(record.p_omnibus),
                significance_marker(record.p_omnibus),
            ])
    return rows


def pairwise_rows(pairwise):
    """One printed row per contrast, measures in panel order."""
    rows = []
    for measure, label in MEASURE_LABELS.items():
        subset = pairwise[pairwise.measure == measure].sort_values(
            ["participant", "comparison"])
        for _, record in subset.iterrows():
            rows.append([
                label,
                str(record.participant),
                reference_of(record.comparison),
                str(int(record.n_test)),
                str(int(record.n_ref)),
                format_number(record.median_test),
                format_number(record.median_ref),
                format_number(record.MannWhitney_U, decimals=1),
                format_p_value(record.p_holm),
                significance_marker(record.p_holm),
            ])
    return rows


def main():
    omnibus = pd.read_csv(os.path.join(STATISTICS_DIRECTORY,
                                       "ExtFig3_omnibus_condition_stats.csv"))
    pairwise = pd.read_csv(os.path.join(STATISTICS_DIRECTORY,
                                        "ExtFig3_pairwise_condition_stats.csv"))

    unknown = (set(omnibus.measure) | set(pairwise.measure)) - set(MEASURE_LABELS)
    if unknown:
        raise ValueError(f"no printed label for measure(s): {sorted(unknown)}")

    document = Document()
    add_paragraph(document, TITLE, bold=True, size=TITLE_SIZE)
    add_paragraph(document, PREAMBLE)

    add_paragraph(document, OMNIBUS_TITLE, bold=True)
    omnibus_table = omnibus_rows(omnibus)
    add_table(document, OMNIBUS_HEADERS, omnibus_table)

    add_paragraph(document, PAIRWISE_TITLE, bold=True)
    pairwise_table = pairwise_rows(pairwise)
    add_table(document, PAIRWISE_HEADERS, pairwise_table)

    add_paragraph(document, FOOTNOTE)

    flagged = implausible_zero_arms(pd.read_csv(SURVEY_CSV))
    caveat = data_quality_caveat(flagged)
    if caveat:
        add_paragraph(document, caveat, bold=True)

    output_path = os.path.join(FIGURE_DIRECTORY, OUTPUT_NAME)
    document.save(output_path)
    print(f"Table S11.1: {len(omnibus_table)} rows")
    print(f"Table S11.2: {len(pairwise_table)} rows")
    print(f"arms flagged for implausible zeros: {len(flagged)}")
    for participant, measure, contradicted, scored in flagged:
        print(f"  {participant} {measure}: {contradicted} of {scored} "
              "scored surveys are 0 with NRS above 0")
    print(f"written to {output_path}")


if __name__ == "__main__":
    main()
