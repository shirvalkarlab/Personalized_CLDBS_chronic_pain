"""
03_plot_significance_heatmaps.py - Create Model Performance Heatmaps

This script creates publication-ready heatmaps showing:
- LDA classification AUC values with significance stars
- LASSO regression R² values with significance stars

Heatmaps display:
- Rows: Pain metrics (NRS, Intensity VAS, Unpleasantness VAS, Relief)
- Columns: Time windows (1, 5, 10, 30 minutes)
- Cell color: Significant (dark) vs non-significant (light)
- Cell text: Metric value with significance stars (*, **, ***)

Significance levels:
    *   : p < 0.05
    **  : p < 0.01
    *** : p < 0.001

Inputs (from decoding_results/{participant}/):
    - lda_sig_*.npy, lda_all_*.npy: LDA significance results
    - lasso_sig_*.npy, lasso_all_*.npy: LASSO significance results
    - *_all_p_vals.npy: P-values for statistical annotations

Outputs:
    - *_AUC_sig_models_stars.pdf: LDA heatmap
    - *_R2_sig_models_stars.pdf: LASSO heatmap

Usage:
    python 03_plot_significance_heatmaps.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bm_helper_functions import foldcheck

# =============================================================================
# CONFIGURATION
# =============================================================================

# Participants to process (de-identified labels). Edit to process a subset.
ALL_PARTICIPANTS = ["P1", "P2", "P3", "P4", "P5"]

# Metrics to display (subset of all metrics)
SELECT_METRICS = ['intensity_vas_s0', 'nrs_s0', 'unpleasantness_vas_s0', 'relief_s0']
ALL_TIME_WINDOWS = ['one_minute', 'five_minute', 'ten_minute', 'thirty_minute']

# Display name mappings
METRIC_DISPLAY_NAMES = {
    'intensity_vas_s0': 'Intensity VAS',
    'nrs_s0': 'NRS',
    'unpleasantness_vas_s0': 'Unpleasantness VAS',
    'relief_s0': 'Relief'
}

TIME_WINDOW_DISPLAY_NAMES = {
    'one_minute': '1 Minute',
    'five_minute': '5 Minutes',
    'ten_minute': '10 Minutes',
    'thirty_minute': '30 Minutes'
}

# Ordered column names for consistent display
ORDERED_COLUMNS = ['1 Minute', '5 Minutes', '10 Minutes', '30 Minutes']

# Heatmap colors (light = non-significant, dark = significant)
HEATMAP_COLORS = ["#cacceb", "#4e56cc"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_significance_stars(p_value):
    """
    Convert p-value to significance stars.

    Args:
        p_value: Statistical p-value

    Returns:
        String with appropriate number of stars
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def build_value_dataframe(all_sizes, all_metrics, all_values, select_metrics, all_windows):
    """
    Build a DataFrame of values organized by metric (rows) and time window (columns).

    Args:
        all_sizes: Array of time window names for each model
        all_metrics: Array of metric names for each model
        all_values: Array of values (AUC or R²) for each model
        select_metrics: List of metrics to include
        all_windows: List of all time windows

    Returns:
        DataFrame with metrics as rows and windows as columns
    """
    df = pd.DataFrame(index=select_metrics, columns=all_windows)

    for i in range(len(all_sizes)):
        size = all_sizes[i]
        metric = all_metrics[i]
        if metric in df.index:
            df.loc[metric, size] = str(all_values[i])

    return df


def build_significance_mask(sig_sizes, sig_metrics, select_metrics, all_windows):
    """
    Build a binary mask indicating which models are significant.

    Args:
        sig_sizes: Time windows of significant models
        sig_metrics: Metrics of significant models
        select_metrics: List of metrics to include
        all_windows: List of all time windows

    Returns:
        DataFrame with 0.5 (significant) or 0.1 (non-significant) values
    """
    # Initialize with all windows
    df_init = pd.DataFrame(index=select_metrics, columns=all_windows)

    # Create pivot table of significant models
    sig_df = pd.DataFrame({'Metrics': sig_metrics, 'Size': sig_sizes})
    sig_pivot = sig_df.pivot_table(index='Metrics', columns='Size', aggfunc=len)

    # Update with significant models
    df_init.update(sig_pivot)

    # Convert to binary mask (0.5 for significant, 0.1 for non-significant)
    mask = df_init.notna().astype('float') * 0.4 + 0.1

    return mask


def build_annotation_dataframe(all_values_df, pvals_df):
    """
    Build annotation strings combining values and significance stars.

    Args:
        all_values_df: DataFrame of metric values (AUC or R²)
        pvals_df: DataFrame of p-values

    Returns:
        DataFrame with formatted annotation strings
    """
    annot_df = pd.DataFrame(index=all_values_df.index, columns=all_values_df.columns)

    for row in annot_df.index:
        for col in annot_df.columns:
            value_str = all_values_df.loc[row, col]
            p_val = pvals_df.loc[row, col]

            # Handle missing values
            if value_str == 'nan' or pd.isnull(value_str):
                annot_df.loc[row, col] = "N/A"
                continue

            # Parse numeric value
            try:
                value = float(value_str)
            except ValueError:
                annot_df.loc[row, col] = "N/A"
                continue

            # Handle missing p-value
            if pd.isnull(p_val):
                annot_df.loc[row, col] = f"{value:.3f} (N/A)"
                continue

            # Combine value with significance stars
            stars = get_significance_stars(p_val)
            annot_df.loc[row, col] = f"{value:.3f}{stars}"

    return annot_df


def create_heatmap(mask_df, annot_df, title, output_path, colors=HEATMAP_COLORS):
    """
    Create and save a heatmap figure.

    Args:
        mask_df: DataFrame for coloring (significant vs non-significant)
        annot_df: DataFrame with annotation text
        title: Plot title
        output_path: Path to save the figure
        colors: Color palette for heatmap
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        mask_df,
        cmap=sns.color_palette(colors),
        cbar=False,
        annot=annot_df,
        fmt="",
        annot_kws={'fontsize': 12, 'color': 'black'},
        square=True,
        vmin=0,
        vmax=1,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')
    plt.xlabel('Data chunk lengths', fontsize=12, fontweight='bold')
    plt.ylabel('Pain Metric', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300)
    plt.close()


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

for idx, participant in enumerate(ALL_PARTICIPANTS):
    print(f"\n{'='*60}")
    print(f"PROCESSING {participant}")
    print("="*60)

    # Define paths. Read the model results written by 01_run_ml_pipeline.py,
    # which live in DECODING_RESULTS_ROOT/{participant}/ (default: decoding_results/).
    _root = os.environ.get('DECODING_RESULTS_ROOT',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decoding_results'))
    results_dir = os.path.join(_root, participant) + '/'
    fig_fold = foldcheck(results_dir + "figs/")

    # -------------------------------------------------------------------------
    # Load LDA results
    # -------------------------------------------------------------------------
    print("\n[1] Loading LDA results...")

    lda_sig_sizes = np.load(results_dir + 'lda_sig_sizes.npy', allow_pickle=True)
    lda_sig_metrics = np.load(results_dir + 'lda_sig_metrics.npy', allow_pickle=True)
    lda_sig_aucs = np.load(results_dir + 'lda_sig_aucs.npy', allow_pickle=True)

    lda_all_sizes = np.load(results_dir + 'lda_all_sizes.npy', allow_pickle=True)
    lda_all_metrics = np.load(results_dir + 'lda_all_metrics.npy', allow_pickle=True)
    lda_all_aucs = np.load(results_dir + 'lda_all_aucs.npy', allow_pickle=True)
    lda_all_pvals = np.load(results_dir + 'lda_all_p_vals.npy', allow_pickle=True)

    print(f"  Significant LDA models: {len(lda_sig_sizes)}")

    # -------------------------------------------------------------------------
    # Load LASSO results
    # -------------------------------------------------------------------------
    print("[2] Loading LASSO results...")

    lasso_sig_sizes = np.load(results_dir + 'lasso_sig_sizes.npy', allow_pickle=True)
    lasso_sig_metrics = np.load(results_dir + 'lasso_sig_metrics.npy', allow_pickle=True)
    lasso_sig_r2s = np.load(results_dir + 'lasso_sig_r2s.npy', allow_pickle=True)

    lasso_all_sizes = np.load(results_dir + 'lasso_all_sizes.npy', allow_pickle=True)
    lasso_all_metrics = np.load(results_dir + 'lasso_all_metrics.npy', allow_pickle=True)
    lasso_all_r2s = np.load(results_dir + 'lasso_all_r2s.npy', allow_pickle=True)
    lasso_all_pvals = np.load(results_dir + 'lasso_all_p_vals.npy', allow_pickle=True)

    print(f"  Significant LASSO models: {len(lasso_sig_sizes)}")

    # -------------------------------------------------------------------------
    # Create LDA AUC heatmap
    # -------------------------------------------------------------------------
    print("\n[3] Creating LDA AUC heatmap...")

    # Build DataFrames
    df_all_aucs = build_value_dataframe(
        lda_all_sizes, lda_all_metrics, lda_all_aucs,
        SELECT_METRICS, ALL_TIME_WINDOWS
    )

    auc_mask = build_significance_mask(
        lda_sig_sizes, lda_sig_metrics,
        SELECT_METRICS, ALL_TIME_WINDOWS
    )

    # Build p-value DataFrame
    df_pvals = pd.DataFrame(index=SELECT_METRICS, columns=ALL_TIME_WINDOWS)
    for i in range(len(lda_all_sizes)):
        size = lda_all_sizes[i]
        metric = lda_all_metrics[i]
        if metric in df_pvals.index:
            df_pvals.loc[metric, size] = lda_all_pvals[i]

    # Apply display name mappings
    auc_mask = auc_mask.rename(index=METRIC_DISPLAY_NAMES, columns=TIME_WINDOW_DISPLAY_NAMES)
    df_all_aucs = df_all_aucs.rename(index=METRIC_DISPLAY_NAMES, columns=TIME_WINDOW_DISPLAY_NAMES)
    df_pvals = df_pvals.rename(index=METRIC_DISPLAY_NAMES, columns=TIME_WINDOW_DISPLAY_NAMES)

    # Reorder columns
    auc_mask = auc_mask[ORDERED_COLUMNS]
    df_all_aucs = df_all_aucs[ORDERED_COLUMNS]
    df_pvals = df_pvals[ORDERED_COLUMNS]

    # Build annotations
    auc_annot = build_annotation_dataframe(df_all_aucs, df_pvals)

    # Create and save heatmap
    create_heatmap(
        auc_mask, auc_annot,
        f'{participant} LDA Model AUCs',
        fig_fold + f"{participant}_AUC_sig_models_stars.pdf"
    )
    print(f"  Saved: {fig_fold}{participant}_AUC_sig_models_stars.pdf")

    # -------------------------------------------------------------------------
    # Create LASSO R² heatmap
    # -------------------------------------------------------------------------
    print("[4] Creating LASSO R² heatmap...")

    # Build DataFrames
    df_all_r2s = build_value_dataframe(
        lasso_all_sizes, lasso_all_metrics, lasso_all_r2s,
        SELECT_METRICS, ALL_TIME_WINDOWS
    )

    r2_mask = build_significance_mask(
        lasso_sig_sizes, lasso_sig_metrics,
        SELECT_METRICS, ALL_TIME_WINDOWS
    )

    # Build p-value DataFrame
    df_pvals_r2 = pd.DataFrame(index=SELECT_METRICS, columns=ALL_TIME_WINDOWS)
    for i in range(len(lasso_all_sizes)):
        size = lasso_all_sizes[i]
        metric = lasso_all_metrics[i]
        if metric in df_pvals_r2.index:
            df_pvals_r2.loc[metric, size] = lasso_all_pvals[i]

    # Apply display name mappings
    r2_mask = r2_mask.rename(index=METRIC_DISPLAY_NAMES, columns=TIME_WINDOW_DISPLAY_NAMES)
    df_all_r2s = df_all_r2s.rename(index=METRIC_DISPLAY_NAMES, columns=TIME_WINDOW_DISPLAY_NAMES)
    df_pvals_r2 = df_pvals_r2.rename(index=METRIC_DISPLAY_NAMES, columns=TIME_WINDOW_DISPLAY_NAMES)

    # Reorder columns
    r2_mask = r2_mask[ORDERED_COLUMNS]
    df_all_r2s = df_all_r2s[ORDERED_COLUMNS]
    df_pvals_r2 = df_pvals_r2[ORDERED_COLUMNS]

    # Build annotations
    r2_annot = build_annotation_dataframe(df_all_r2s, df_pvals_r2)

    # Create and save heatmap
    create_heatmap(
        r2_mask, r2_annot,
        f"{participant} LASSO models R$^2$",
        fig_fold + f"{participant}_R2_sig_models_stars.pdf"
    )
    print(f"  Saved: {fig_fold}{participant}_R2_sig_models_stars.pdf")

print(f"\n{'='*60}")
print("ALL HEATMAPS COMPLETE")
print("="*60 + "\n")
