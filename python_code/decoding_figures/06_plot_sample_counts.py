"""
06_plot_sample_counts.py - Create Sample Count Tables

This script creates publication-ready tables showing the number of samples
used for each model (pain metric x time window combination).

For LDA classification:
    - Shows "X LP / Y HP" (low pain / high pain sample counts)
    - Based on 30th/70th percentile thresholding

For LASSO regression:
    - Shows total sample count per model
    - After outlier removal

Tables are displayed as grid images with white cells and black borders.

Inputs:
    - <participant>/lda_num_samples_per_model.csv: LDA sample counts
    - <participant>/regression_num_sample_per_model.csv: LASSO sample counts

Outputs:
    - *_LDA_sample_counts.pdf: LDA sample table
    - *_REG_sample_counts.pdf: LASSO sample table

Usage:
    python 06_plot_sample_counts.py

    Modify ALL_PARTICIPANTS to process a subset of participants.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths. Share the same results root as scripts 01-05 (DECODING_RESULTS_ROOT,
# default: decoding_results/ beside this script).
_RESULTS_ROOT = os.environ.get('DECODING_RESULTS_ROOT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decoding_results'))
# Sample counts are read from each participant's own results folder, which is
# where script 01 writes them. Reading a separate staged copy would let this
# figure keep showing old counts after 01 is re-run.
COUNTS_DIR = _RESULTS_ROOT
OUTPUT_DIR = os.path.join(_RESULTS_ROOT, 'combined_results', 'num_samps') + '/'

# All participants, in order (de-identified labels).
ALL_PARTICIPANTS = ["P1", "P2", "P3", "P4", "P5"]

# Display name mappings
METRIC_DISPLAY_NAMES = {
    "intensity_vas_s0": "Intensity VAS",
    "nrs_s0": "NRS",
    "unpleasantness_vas_s0": "Unpleasantness VAS",
    "relief_s0": "Relief",
}

TIME_WINDOW_DISPLAY_NAMES = {
    "one_minute": "1 Minute",
    "five_minute": "5 Minutes",
    "ten_minute": "10 Minutes",
    "thirty_minute": "30 Minutes",
}

# Ordered lists for display
ROWS = list(METRIC_DISPLAY_NAMES.values())
COLS = list(TIME_WINDOW_DISPLAY_NAMES.values())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_lda_counts(counts_dir, participant):
    """
    Load and format LDA sample counts.

    Args:
        counts_dir: Directory containing count CSVs
        participant: Participant label (e.g., 'P5')

    Returns:
        DataFrame formatted for display (rows = metrics, columns = windows)
    """
    lda = pd.read_csv(os.path.join(counts_dir, participant,
                                   "lda_num_samples_per_model.csv"))

    # Map to display names
    lda["Metric"] = lda["Metric"].map(METRIC_DISPLAY_NAMES)
    lda["Window"] = lda["Time Window"].map(TIME_WINDOW_DISPLAY_NAMES)

    # Create display text: "X LP\nY HP"
    lda["Text"] = (
        lda["Low Pain Samples"].astype(int).astype(str) + " LP\n" +
        lda["High Pain Samples"].astype(int).astype(str) + " HP"
    )

    # Pivot to table format
    lda_display = (
        lda.pivot_table(index="Metric", columns="Window", values="Text", aggfunc="first")
        .reindex(index=ROWS, columns=COLS)
    )

    return lda_display


def load_regression_counts(counts_dir, participant):
    """
    Load and format regression sample counts.

    Args:
        counts_dir: Directory containing count CSVs
        participant: Participant label (e.g., 'P5')

    Returns:
        DataFrame formatted for display (rows = metrics, columns = windows)
    """
    reg = pd.read_csv(os.path.join(counts_dir, participant,
                                   "regression_num_sample_per_model.csv"))

    # Map to display names
    reg["Metric"] = reg["Metric"].map(METRIC_DISPLAY_NAMES)
    reg["Window"] = reg["Time Window"].map(TIME_WINDOW_DISPLAY_NAMES)

    # Pivot to table format
    reg_display = (
        reg.pivot_table(index="Metric", columns="Window", values="Num Samples", aggfunc="first")
        .reindex(index=ROWS, columns=COLS)
        .astype("Int64")
        .astype(str)
    )

    return reg_display


def create_table_figure(text_df, title, model_type, output_path):
    """
    Create a white grid table figure with text annotations.

    Args:
        text_df: DataFrame with text to display in each cell
        title: Figure title
        model_type: 'LDA' or 'REG' (affects font size)
        output_path: Path to save figure
    """
    n_rows, n_cols = text_df.shape

    # Create figure
    fig, ax = plt.subplots(figsize=(1.6 * n_cols, 0.8 * n_rows))

    # White background
    ax.imshow(np.ones((n_rows, n_cols)), cmap='gray', vmin=0, vmax=1)

    # Add grid lines
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Set tick labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(COLS, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ROWS, fontsize=8)

    # Add cell annotations
    font_size = 11 if model_type == "REG" else 8

    for i in range(n_rows):
        for j in range(n_cols):
            txt = text_df.iloc[i, j]
            display_txt = txt if pd.notnull(txt) else ""
            ax.text(j, i, display_txt, ha='center', va='center', fontsize=font_size)

    # Set title
    ax.set_title(title, pad=4, fontsize=9)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300)
    plt.close()

    print(f"  Saved: {output_path}")


# =============================================================================
# BATCH PROCESSING (all participants)
# =============================================================================

def process_all_participants():
    """
    Process sample count tables for all participants.

    Call this function to generate tables for everyone at once.
    """
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for pid in ALL_PARTICIPANTS:
        print(f"\n{'='*60}")
        print(f"PROCESSING {pid}")
        print("="*60)

        display_name = pid

        try:
            lda_display = load_lda_counts(COUNTS_DIR, pid)
            create_table_figure(
                lda_display,
                f"{display_name} - LDA sample counts",
                "LDA",
                f"{OUTPUT_DIR}{pid}_LDA_sample_counts.pdf"
            )
        except Exception as e:
            print(f"  LDA table failed: {e}")

        try:
            reg_display = load_regression_counts(COUNTS_DIR, pid)
            create_table_figure(
                reg_display,
                f"{display_name} - LASSO sample counts",
                "REG",
                f"{OUTPUT_DIR}{pid}_REG_sample_counts.pdf"
            )
        except Exception as e:
            print(f"  Regression table failed: {e}")

    print("\n" + "="*60)
    print("ALL PARTICIPANTS COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    process_all_participants()
