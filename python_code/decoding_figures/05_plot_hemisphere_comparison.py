"""
05_plot_hemisphere_comparison.py - Left vs Right Hemisphere Feature Comparison

This script creates side-by-side bar plots comparing feature importance
between left and right brain hemispheres.

For each participant:
1. Loads channel-ordered feature counts from step 02
2. Groups channels by anatomical region
3. Sums counts within each anatomical region
4. Separates into left (L*) and right (R*) hemispheres
5. Creates paired bar plots showing top 10 regions per hemisphere

Generates plots for:
- LDA coefficients
- LASSO coefficients
- Combined (LDA + LASSO)

Inputs (from decoding_results/{participant}/):
    - lda_counts_ch_ordered.csv: LDA feature counts by channel
    - lasso_counts_ch_ordered.csv: LASSO feature counts by channel

Outputs:
    - *_per_hem_anat_features_LDA_norm.pdf
    - *_per_hem_anat_features_LASSO_norm.pdf
    - *_per_hem_anat_features_Combined_norm.pdf
    - *_summed_counts_df.csv: Summary counts by region

Usage:
    python 05_plot_hemisphere_comparison.py
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bm_helper_functions import foldcheck

# =============================================================================
# CONFIGURATION
# =============================================================================

# Participants to process
# Participants to process (de-identified labels). Edit to process a subset.
ALL_PARTICIPANTS = ["P1", "P2", "P3", "P4", "P5"]

# Analysis methods to plot
ANALYSIS_METHODS = ["lda", "lasso", "Combined"]

# Number of top regions to show per hemisphere
TOP_N_REGIONS = 10

# Plot styling
BAR_COLOR = 'lightblue'
SQUISHED_MODE = False  # If True, creates more compact figures


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_counts_data(results_dir, method):
    """
    Load channel-ordered counts for the specified method.

    Args:
        results_dir: Path to results directory
        method: 'lda', 'lasso', or 'Combined'

    Returns:
        DataFrame with 'count' and 'anat' columns, indexed by channel
    """
    if method == "lda":
        df = pd.read_csv(results_dir + 'lda_counts_ch_ordered.csv', index_col=0)
    elif method == "lasso":
        df = pd.read_csv(results_dir + 'lasso_counts_ch_ordered.csv', index_col=0)
    elif method == "Combined":
        lda_df = pd.read_csv(results_dir + 'lda_counts_ch_ordered.csv', index_col=0)
        lasso_df = pd.read_csv(results_dir + 'lasso_counts_ch_ordered.csv', index_col=0)
        # Combine counts (add LDA and LASSO)
        df = lda_df.copy()
        df['count'] = lda_df['count'] + lasso_df['count']
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


def split_by_hemisphere(anat_count_df):
    """
    Split anatomical counts into left and right hemisphere DataFrames.

    Anatomical labels starting with 'L' are left hemisphere,
    those starting with 'R' are right hemisphere.

    Args:
        anat_count_df: DataFrame with 'anat' and 'count' columns

    Returns:
        left_df: Left hemisphere counts (top N)
        right_df: Right hemisphere counts (top N)
    """
    # Split by first character of anatomical label
    left_df = anat_count_df[anat_count_df['anat'].str.startswith('L')]
    right_df = anat_count_df[anat_count_df['anat'].str.startswith('R')]

    # Sort by count and take top N
    left_df = left_df.sort_values(by='count', ascending=False).head(TOP_N_REGIONS)
    right_df = right_df.sort_values(by='count', ascending=False).head(TOP_N_REGIONS)

    return left_df, right_df


def create_hemisphere_plot(left_df, right_df, title, output_path, squished=False):
    """
    Create side-by-side bar plots comparing hemispheres.

    Args:
        left_df: Left hemisphere data
        right_df: Right hemisphere data
        title: Plot supertitle
        output_path: Path to save figure
        squished: If True, create more compact figure
    """
    # Calculate shared y-axis maximum
    y_max = max(left_df['count'].max(), right_df['count'].max())

    # Create figure with two subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2)

    # Left hemisphere plot
    sns.barplot(x='anat', y='count', data=left_df, color=BAR_COLOR, ax=ax_left)
    ax_left.set_ylim(0, y_max)
    ax_left.set_title('Prevalent left hem. regions for decoding')
    ax_left.set_xlabel('Left Anat')
    ax_left.set_ylabel('Count (Feature Prominence)')

    if not squished:
        ax_left.set_axisbelow(True)
        ax_left.yaxis.grid(True, linestyle='-', linewidth=0.7, color='gray', alpha=0.5)
        plt.sca(ax_left)
        plt.xticks(rotation=45, ha='right', fontsize=12)
    else:
        plt.sca(ax_left)
        plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')

    # Right hemisphere plot
    sns.barplot(x='anat', y='count', data=right_df, color=BAR_COLOR, ax=ax_right)
    ax_right.set_ylim(0, y_max)
    ax_right.set_title('Prevalent right hem. regions for decoding')
    ax_right.set_xlabel('Right Anat')
    ax_right.set_ylabel('')  # No ylabel on right plot

    if not squished:
        ax_right.set_axisbelow(True)
        ax_right.yaxis.grid(True, linestyle='-', linewidth=0.7, color='gray', alpha=0.5)
        plt.sca(ax_right)
        plt.xticks(rotation=45, ha='right', fontsize=12)
    else:
        plt.sca(ax_right)
        plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')

    # Add supertitle
    plt.suptitle(title, fontsize=15)

    # Adjust figure size and spacing
    if not squished:
        fig.set_size_inches(12, 6)
    else:
        fig.set_size_inches(12, 4)

    plt.subplots_adjust(wspace=0.4, bottom=0.3)
    fig.align_xlabels()
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

for participant in ALL_PARTICIPANTS:
    print(f"\n{'='*60}")
    print(f"PROCESSING {participant}")
    print("="*60)

    # Define paths
    # Read the model results written by 01/02, which live in
    # DECODING_RESULTS_ROOT/{participant}/ (default: decoding_results/).
    _root = os.environ.get('DECODING_RESULTS_ROOT',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decoding_results'))
    results_dir = os.path.join(_root, participant) + '/'
    fig_fold = foldcheck(results_dir + "figs/final_figs/")
    combined_results = foldcheck(os.path.join(_root, 'combined_results') + '/')

    # -------------------------------------------------------------------------
    # Process each analysis method
    # -------------------------------------------------------------------------
    for method in ANALYSIS_METHODS:
        print(f"\n  Processing {method}...")

        # Load counts data
        try:
            df_counts = load_counts_data(results_dir, method)
        except FileNotFoundError as e:
            print(f"    WARNING: Could not load data for {method}: {e}")
            continue

        # Aggregate by anatomical region
        anat_counts = df_counts.groupby('anat')['count'].sum().reset_index()

        # Split by hemisphere
        left_df, right_df = split_by_hemisphere(anat_counts)

        print(f"    Left hemisphere: {len(left_df)} regions")
        print(f"    Right hemisphere: {len(right_df)} regions")

        # Create combined summary DataFrame
        summed_df = pd.concat([left_df, right_df], axis=0)
        summed_df['hemisphere'] = summed_df['anat'].apply(
            lambda x: 'left' if x.startswith('L') else 'right'
        )

        # Save summary CSV
        if method == "Combined":
            summed_df.to_csv(combined_results + f'{participant}_summed_counts_df.csv', index=False)
            print("    Saved summary to combined_results/")

        # Determine analysis display name
        if method == "lda":
            analysis_name = "LDA"
        elif method == "lasso":
            analysis_name = "LASSO"
        else:
            analysis_name = "Combined"

        # Create plot
        title = f"{participant}, model used: {analysis_name}"

        if SQUISHED_MODE:
            output_path = fig_fold + f"{participant}_per_hem_anat_features_{analysis_name}_squished.pdf"
        else:
            output_path = fig_fold + f"{participant}_per_hem_anat_features_{analysis_name}_norm.pdf"

        create_hemisphere_plot(
            left_df, right_df, title, output_path, squished=SQUISHED_MODE
        )

        # Also save to combined results folder
        combined_output = combined_results + f"{participant}_per_hem_anat_features_{analysis_name}_norm.pdf"
        create_hemisphere_plot(
            left_df, right_df, title, combined_output, squished=SQUISHED_MODE
        )

        print(f"    Saved: {output_path}")

print(f"\n{'='*60}")
print("ALL HEMISPHERE PLOTS COMPLETE")
print("="*60 + "\n")
