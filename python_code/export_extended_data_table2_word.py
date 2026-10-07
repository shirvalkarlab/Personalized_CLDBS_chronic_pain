"""Render Extended Data Table 2 as a Word document.

Reads the table built by matlab_code/build_extended_data_table2.m and writes a
formatted .docx alongside it. Participant labels and index-metric ticks are shown
once per participant block so the table reads as grouped rows.

Reads   figures/ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham.csv
Writes  figures/ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham.docx

Usage:
    python export_extended_data_table2_word.py
"""

import os

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIRECTORY = os.path.join(PROJECT_ROOT, "figures")
TABLE_STEM = "ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham"

CAPTION = ("Extended Data Table 2: All pain scales for all participants, "
           "closed-loop versus sham.")

# Source column -> printed header. Order here is the column order in the table.
COLUMNS = [
    ("participant", "Participant"),
    ("pain_scale", "Pain scale"),
    ("prospective_index_metric", "Index metric"),
    ("baseline_median_iqr_n_days", "Baseline median [IQR] (n days)"),
    ("closed_loop_median_iqr_n_days", "Closed-loop median [IQR] (n days)"),
    ("closed_loop_percent_change_from_baseline", "Closed-loop % change"),
    ("sham_median_iqr_n_days", "Sham median [IQR] (n days)"),
    ("sham_percent_change_from_baseline", "Sham % change"),
    ("responder_30pct_closed_loop", "\u226530% responder (CL)"),
    ("responder_50pct_closed_loop", "\u226550% responder (CL)"),
]

BODY_FONT = "Arial"
BODY_SIZE = Pt(8)
HEADER_SIZE = Pt(8)
CAPTION_SIZE = Pt(9)


def cell_text(column, value, previous_participant):
    """Printed text for one cell, blanking repeated participant labels."""
    text = "" if pd.isna(value) else str(value).strip()
    if column == "participant" and text == previous_participant:
        return ""
    if column == "prospective_index_metric":
        return "\u2713" if text.lower() == "yes" else ""
    return text


def write_cell(cell, text, bold=False):
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = BODY_FONT
    run.font.size = HEADER_SIZE if bold else BODY_SIZE


def main():
    table_data = pd.read_csv(os.path.join(FIGURE_DIRECTORY, TABLE_STEM + ".csv"))

    document = Document()
    caption = document.add_paragraph()
    caption_run = caption.add_run(CAPTION)
    caption_run.bold = True
    caption_run.font.name = BODY_FONT
    caption_run.font.size = CAPTION_SIZE

    word_table = document.add_table(rows=1, cols=len(COLUMNS))
    word_table.style = "Table Grid"
    for position, (_, header) in enumerate(COLUMNS):
        write_cell(word_table.rows[0].cells[position], header, bold=True)

    previous_participant = None
    for _, source_row in table_data.iterrows():
        cells = word_table.add_row().cells
        for position, (column, _) in enumerate(COLUMNS):
            write_cell(cells[position],
                       cell_text(column, source_row[column], previous_participant))
        previous_participant = str(source_row["participant"]).strip()

    output_path = os.path.join(FIGURE_DIRECTORY, TABLE_STEM + ".docx")
    document.save(output_path)
    print(f"Extended Data Table 2: {len(table_data)} rows written to {output_path}")


if __name__ == "__main__":
    main()
