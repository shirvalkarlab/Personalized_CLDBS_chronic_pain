"""
02_extract_significant_models.py - Extract Significant Models and Feature Importance

This script identifies statistically significant models from the ML pipeline output
and calculates channel-level feature importance:

1. Loads coefficient dataframes and bootstrap distributions from step 01
2. Identifies significant models (performance > 95th percentile of bootstrap)
3. Calculates p-values for all models
4. Extracts top contributing channels/features for each significant model
5. Aggregates channel importance counts across models and pain metrics

Inputs (from decoding_results/{participant}/):
    - lda_coef_df.pkl, lasso_coef_df.pkl: Model coefficients
    - lda_boots_aucs.npy, lasso_boots_r2s.npy: Bootstrap distributions

Outputs (saved to decoding_results/{participant}/):
    - lda_sig_*.npy: Significant model info (sizes, metrics, AUCs)
    - lasso_sig_*.npy: Significant model info (sizes, metrics, R²s)
    - *_all_p_vals.npy: P-values for all models
    - *_combined_counts.csv: Feature importance counts
    - *_counts_ch_ordered.csv: Counts ordered by electrode position

Usage:
    python 02_extract_significant_models.py

    Modify ALL_PARTICIPANTS to run for different participants.
"""

import sys
import os
import numpy as np
import pandas as pd

# Import helper functions for channel labels
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bm_helper_functions import get_ch_labels

# =============================================================================
# CONFIGURATION
# =============================================================================

# Participants to process (de-identified labels). Edit to process a subset.
ALL_PARTICIPANTS = ["P1", "P2", "P3", "P4", "P5"]

# Pain metrics to analyze
SELECT_METRICS = ['intensity_vas_s0', 'nrs_s0', 'unpleasantness_vas_s0', 'relief_s0']

# Significance threshold (percentile of bootstrap distribution)
SIGNIFICANCE_PERCENTILE = 95

# Number of top features to extract per model
TOP_N_FEATURES = 10


# =============================================================================
# SIGNIFICANCE TESTING FUNCTIONS
# =============================================================================

def identify_significant_models(actual_values, bootstrap_values, percentile=95):
    """
    Identify models where actual performance exceeds the bootstrap threshold.

    Args:
        actual_values: Array of actual model performance (AUC or R²)
        bootstrap_values: Array of shape (n_bootstrap, n_models)
        percentile: Percentile threshold for significance

    Returns:
        sig_indices: Array of indices for significant models
        p_values: Array of p-values for all models
    """
    sig_indices = []
    p_values = []

    for i in range(len(actual_values)):
        # Calculate threshold from bootstrap distribution
        threshold = np.percentile(bootstrap_values[:, i], percentile)

        # Check significance
        if actual_values[i] > threshold:
            sig_indices.append(i)
            print(f"  Model {i}: {actual_values[i]:.3f} > {threshold:.3f} (SIGNIFICANT)")

        if np.isnan(actual_values[i]):
            p_values.append(np.nan)
            continue

        p_val = (np.sum(bootstrap_values[:, i] > actual_values[i]) + 1) / (bootstrap_values.shape[0] + 1)
        p_values.append(p_val)

    return np.array(sig_indices), np.array(p_values)


# =============================================================================
# FEATURE IMPORTANCE FUNCTIONS
# =============================================================================

def extract_electrode_names(column_names):
    """
    Extract electrode names from feature column names.

    Feature names are formatted as 'ELECTRODE_frequency' (e.g., 'RACC1_delta').
    This extracts just the electrode part.

    Args:
        column_names: Array of feature column names

    Returns:
        Array of electrode names (one per feature)
    """
    electrode_names = []
    for name in column_names:
        # Split on underscore and take first part
        electrode = name.split("_")[0]
        electrode_names.append(electrode)
    return np.array(electrode_names)


def calculate_channel_importance(coef_array, column_names, electrode_names):
    """
    Calculate average coefficient magnitude per electrode.

    For each electrode, averages the coefficients across all frequency bands
    to get a single importance score.

    Args:
        coef_array: 2D array of coefficients (n_models, n_features)
        column_names: Feature column names
        electrode_names: Electrode name for each feature

    Returns:
        DataFrame with mean coefficient per electrode for each model
    """
    unique_electrodes = np.unique(electrode_names)
    all_channel_means = []

    for electrode in unique_electrodes:
        # Find columns belonging to this electrode
        electrode_mask = electrode_names == electrode
        electrode_coefs = coef_array[:, electrode_mask]

        # Average across frequency bands for each model
        mean_coefs = np.mean(electrode_coefs, axis=1)
        all_channel_means.append(mean_coefs)

    # Convert to DataFrame (electrodes as columns, models as rows)
    channel_df = pd.DataFrame(
        np.array(all_channel_means).T,
        columns=unique_electrodes
    )

    return channel_df


def count_top_features(channel_df, n_top=10):
    """
    Count how often each electrode appears in top/bottom features.

    For each model, identifies the electrodes with the highest and lowest
    coefficients, then counts their frequency across all models.

    Args:
        channel_df: DataFrame with mean coefficients per electrode
        n_top: Number of top/bottom features to extract per model

    Returns:
        Series with counts per electrode
    """
    top_features = []
    bottom_features = []

    for i in range(channel_df.shape[0]):
        model_coefs = channel_df.iloc[i, :]
        top_features.extend(model_coefs.nlargest(n_top).index.tolist())
        bottom_features.extend(model_coefs.nsmallest(n_top).index.tolist())

    # Count occurrences
    all_features = top_features + bottom_features
    counts = pd.Series(all_features).value_counts()

    return counts


def process_metric_coefficients(coef_array, sig_indices, metric_labels, column_names,
                                 metric, method_name):
    """
    Process coefficients for a single pain metric.

    Args:
        coef_array: Full coefficient array (n_models, n_features)
        sig_indices: Indices of significant models
        metric_labels: Array of metric names for each model
        column_names: Feature column names
        metric: Target pain metric to process
        method_name: 'LDA' or 'LASSO' for printing

    Returns:
        DataFrame with feature counts for this metric
    """
    # Find significant models for this metric
    metric_mask = np.where(metric_labels[sig_indices] == metric)[0]
    metric_indices = sig_indices[metric_mask]

    if len(metric_indices) == 0:
        print(f"    No significant {method_name} models for {metric}")
        return None

    print(f"    {method_name} significant models for {metric}: {len(metric_indices)}")

    # Extract coefficients for significant models
    selected_coefs = coef_array[metric_indices, :]
    electrode_names = extract_electrode_names(column_names)

    # Calculate channel-level importance
    channel_df = calculate_channel_importance(selected_coefs, column_names, 
                                              electrode_names)

    # Count top features
    counts = count_top_features(channel_df, TOP_N_FEATURES)

    # Convert to DataFrame
    counts_df = pd.DataFrame(counts, columns=['count'])

    return counts_df


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

for participant in ALL_PARTICIPANTS:
    print(f"\n{'='*60}")
    print(f"PROCESSING {participant}")
    print("="*60)

    # Define paths. Read the model results written by 01_run_ml_pipeline.py,
    # which live in DECODING_RESULTS_ROOT/{participant}/ (default: decoding_results/).
    _root = os.environ.get('DECODING_RESULTS_ROOT',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decoding_results'))
    results_dir = os.path.join(_root, participant) + '/'
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load coefficient data from ML pipeline
    # -------------------------------------------------------------------------
    print("\n[1] Loading coefficient data...")

    # LDA data
    lda_df = pd.read_pickle(results_dir + 'lda_coef_df.pkl')
    lda_all_coefs = lda_df['coef'].values
    lda_all_aucs = lda_df['auc'].values
    lda_all_sizes = lda_df['size_org'].values
    lda_metric_org = lda_df['metric_org'].values
    lda_colnames = np.load(results_dir + 'lda_colnames.npy', allow_pickle=True)

    # Reshape LDA coefficients to 2D array
    lda_coef_array = np.zeros([len(lda_all_coefs), lda_all_coefs[0].shape[1]])
    for i in range(len(lda_all_coefs)):
        lda_coef_array[i, :] = lda_all_coefs[i]

    # LASSO data
    lasso_df = pd.read_pickle(results_dir + 'lasso_coef_df.pkl')
    lasso_all_coefs = lasso_df['coef'].values
    lasso_all_r2s = lasso_df['r-squared'].values
    lasso_all_sizes = lasso_df['size_org'].values
    lasso_metric_org = lasso_df['metric_org'].values
    lasso_colnames = np.load(results_dir + 'lasso_colnames.npy', allow_pickle=True)

    # Reshape LASSO coefficients
    lasso_coef_array = np.zeros([len(lasso_all_coefs), lasso_all_coefs[0].shape[0]])
    for i in range(len(lasso_all_coefs)):
        lasso_coef_array[i, :] = lasso_all_coefs[i]

    print(f"  LDA: {lda_coef_array.shape[0]} models, {lda_coef_array.shape[1]} features")
    print(f"  LASSO: {lasso_coef_array.shape[0]} models, {lasso_coef_array.shape[1]} features")

    # -------------------------------------------------------------------------
    # Load bootstrap distributions and identify significant models
    # -------------------------------------------------------------------------
    print("\n[2] Identifying significant models (bootstrap threshold)...")

    # LDA significance testing
    lda_boots = np.load(results_dir + 'lda_boots_aucs.npy')
    print("\n  LDA Models:")
    lda_sig_indices, lda_p_vals = identify_significant_models(
        lda_all_aucs, lda_boots, SIGNIFICANCE_PERCENTILE
    )

    lda_sig_aucs = lda_all_aucs[lda_sig_indices]
    lda_sig_sizes = lda_all_sizes[lda_sig_indices]
    lda_sig_metrics = lda_metric_org[lda_sig_indices]

    print(f"\n  Total significant LDA models: {len(lda_sig_indices)}")

    # LASSO significance testing
    lasso_boots = np.load(results_dir + 'lasso_boots_r2s.npy')
    print("\n  LASSO Models:")
    lasso_sig_indices, lasso_p_vals = identify_significant_models(
        lasso_all_r2s, lasso_boots, SIGNIFICANCE_PERCENTILE
    )

    lasso_sig_r2s = lasso_all_r2s[lasso_sig_indices]
    lasso_sig_sizes = lasso_all_sizes[lasso_sig_indices]
    lasso_sig_metrics = lasso_metric_org[lasso_sig_indices]

    print(f"\n  Total significant LASSO models: {len(lasso_sig_indices)}")

    # -------------------------------------------------------------------------
    # Save significance results
    # -------------------------------------------------------------------------
    print("\n[3] Saving significance results...")

    # LDA results
    np.save(results_dir + 'lda_sig_sizes.npy', lda_sig_sizes)
    np.save(results_dir + 'lda_sig_metrics.npy', lda_sig_metrics)
    np.save(results_dir + 'lda_sig_aucs.npy', lda_sig_aucs)
    np.save(results_dir + 'lda_all_sizes.npy', lda_all_sizes)
    np.save(results_dir + 'lda_all_metrics.npy', lda_metric_org)
    np.save(results_dir + 'lda_all_aucs.npy', lda_all_aucs)
    np.save(results_dir + 'lda_all_p_vals.npy', lda_p_vals)

    # LASSO results
    np.save(results_dir + 'lasso_sig_sizes.npy', lasso_sig_sizes)
    np.save(results_dir + 'lasso_sig_metrics.npy', lasso_sig_metrics)
    np.save(results_dir + 'lasso_sig_r2s.npy', lasso_sig_r2s)
    np.save(results_dir + 'lasso_all_sizes.npy', lasso_all_sizes)
    np.save(results_dir + 'lasso_all_metrics.npy', lasso_metric_org)
    np.save(results_dir + 'lasso_all_r2s.npy', lasso_all_r2s)
    np.save(results_dir + 'lasso_all_p_vals.npy', lasso_p_vals)

    # -------------------------------------------------------------------------
    # Calculate feature importance counts
    # -------------------------------------------------------------------------
    print("\n[4] Calculating feature importance counts...")

    lda_all_metric_counts = []
    lasso_all_metric_counts = []

    for metric in SELECT_METRICS:
        print(f"\n  Processing metric: {metric}")

        # LDA feature counts
        lda_counts = process_metric_coefficients(
            lda_coef_array, lda_sig_indices, lda_metric_org,
            lda_colnames, metric, "LDA"
        )
        if lda_counts is not None:
            lda_all_metric_counts.append(lda_counts)

        # LASSO feature counts
        lasso_counts = process_metric_coefficients(
            lasso_coef_array, lasso_sig_indices, lasso_metric_org,
            lasso_colnames, metric, "LASSO"
        )
        if lasso_counts is not None:
            lasso_all_metric_counts.append(lasso_counts)

    # Combine counts across metrics
    if lda_all_metric_counts:
        lda_combined = pd.concat(lda_all_metric_counts).groupby(level=0).sum()
        lda_combined.to_csv(results_dir + 'lda_combined_counts.csv')
        print(f"\n  Saved LDA combined counts: {len(lda_combined)} electrodes")
    else:
        lda_combined = pd.DataFrame(columns=['counts'])
        print("\n  No significant LDA models found")

    if lasso_all_metric_counts:
        lasso_combined = pd.concat(lasso_all_metric_counts).groupby(level=0).sum()
        lasso_combined.to_csv(results_dir + 'lasso_combined_counts.csv')
        print(f"  Saved LASSO combined counts: {len(lasso_combined)} electrodes")
    else:
        lasso_combined = pd.DataFrame(columns=['counts'])
        print("  No significant LASSO models found")

    # -------------------------------------------------------------------------
    # Order counts by electrode position (from anatomy CSV)
    # -------------------------------------------------------------------------
    print("\n[5] Ordering counts by electrode position...")

    # Load channel labels from JSON
    chl_labs, anat_labs, long_labs = get_ch_labels(participant)

    # LDA ordered counts
    lda_counts_ordered = []
    for chl in chl_labs:
        if chl in lda_combined.index:
            lda_counts_ordered.append(lda_combined.loc[chl, 'count'])
        else:
            lda_counts_ordered.append(0)

    lda_ordered_df = pd.DataFrame({
        'count': lda_counts_ordered,
        'anat': anat_labs
    }, index=chl_labs)

    np.save(results_dir + 'lda_counts_ch_ordered.npy', np.array(lda_counts_ordered))
    lda_ordered_df.to_csv(results_dir + 'lda_counts_ch_ordered.csv')

    # LASSO ordered counts
    lasso_counts_ordered = []
    for chl in chl_labs:
        if chl in lasso_combined.index:
            lasso_counts_ordered.append(lasso_combined.loc[chl, 'count'])
        else:
            lasso_counts_ordered.append(0)

    lasso_ordered_df = pd.DataFrame({
        'count': lasso_counts_ordered,
        'anat': anat_labs
    }, index=chl_labs)

    np.save(results_dir + 'lasso_counts_ch_ordered.npy', np.array(lasso_counts_ordered))
    lasso_ordered_df.to_csv(results_dir + 'lasso_counts_ch_ordered.csv')

    print(f"  Saved channel-ordered counts for {len(chl_labs)} electrodes")

print(f"\n{'='*60}")
print("PROCESSING COMPLETE")
print("="*60 + "\n")
