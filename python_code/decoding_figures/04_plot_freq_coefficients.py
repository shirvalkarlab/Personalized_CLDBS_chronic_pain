"""
Plot frequency band counts from significant models.

Generates bar plots showing how often each frequency band appears in top features
for both LDA (classification) and LASSO (regression) models.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


# Define frequency band order for plotting
FREQ_ORDER = ['delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma']
FREQ_LABELS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
SIGNIFICANCE_PERCENTILE = 95

# The four reported pain metrics (the same list as scripts 02 and 03).
SELECT_METRICS = ['intensity_vas_s0', 'nrs_s0', 'unpleasantness_vas_s0', 'relief_s0']

all_ppts = ["P1", "P2", "P3", "P4", "P5"]


# Store counts per participant for stacked bar plots
all_lda_counts = {ppt: {freq: 0 for freq in FREQ_ORDER} for ppt in all_ppts}
all_lasso_counts = {ppt: {freq: 0 for freq in FREQ_ORDER} for ppt in all_ppts}

for participant in all_ppts:
    print(f"\n{'='*50}")
    print(f"Processing {participant}")
    print('='*50)

    # Read the model results written by 01_run_ml_pipeline.py, which live in
    # DECODING_RESULTS_ROOT/{participant}/ (default: decoding_results/).
    _root = os.environ.get('DECODING_RESULTS_ROOT',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decoding_results'))
    results_dir = os.path.join(_root, participant) + '/'
    fig_dir = results_dir + 'frequency_coefficient_figs/'

    # Create figs directory if it doesn't exist
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Load LDA data
    lda_coef_df = pd.read_pickle(results_dir + 'lda_coef_df.pkl')
    lda_all_coefs = lda_coef_df['coef'].values
    lda_all_auc = lda_coef_df['auc'].values
    lda_colnames = np.load(results_dir + 'lda_colnames.npy', allow_pickle=True)
    lda_boots = np.load(results_dir + 'lda_boots_aucs.npy')

    # Load LASSO data
    lasso_coef_df = pd.read_pickle(results_dir + 'lasso_coef_df.pkl')
    lasso_all_coefs = lasso_coef_df['coef'].values
    lasso_all_r2s = lasso_coef_df['r-squared'].values
    lasso_colnames = np.load(results_dir + 'lasso_colnames.npy', allow_pickle=True)
    lasso_boots = np.load(results_dir + 'lasso_boots_r2s.npy')

    # Identify significant LDA models (AUC > 95th percentile of bootstrap)
    # lda_sig_idx = []
    # for i in range(len(lda_all_auc)):
    #     if lda_all_auc[i] > np.percentile(lda_boots_aucs[:, i], 95):
    #         lda_sig_idx.append(i)
    # print(f"LDA significant models: {len(lda_sig_idx)}/{len(lda_all_auc)}")

    lda_sig_idx, lda_p_vals = identify_significant_models(
        lda_all_auc, lda_boots, SIGNIFICANCE_PERCENTILE
    )
    lda_sig_idx = np.array([i for i in lda_sig_idx
                            if lda_coef_df['metric_org'].values[i] in SELECT_METRICS], dtype=int)

    # Identify significant LASSO models (R2 > 95th percentile of bootstrap)
    # lasso_sig_idx = []
    # for i in range(len(lasso_all_r2s)):
    #     if lasso_all_r2s[i] > np.percentile(lasso_boots_r2s[:, i], 95):
    #         lasso_sig_idx.append(i)
    # print(f"LASSO significant models: {len(lasso_sig_idx)}/{len(lasso_all_r2s)}")

    lasso_sig_idx, lasso_p_vals = identify_significant_models(
        lasso_all_r2s, lasso_boots, SIGNIFICANCE_PERCENTILE
    )
    lasso_sig_idx = np.array([i for i in lasso_sig_idx
                              if lasso_coef_df['metric_org'].values[i] in SELECT_METRICS], dtype=int)

    # Extract frequency from feature names
    def get_freq_from_name(name):
        """Extract frequency band from feature name like 'RACC1_delta'"""
        parts = name.split('_')
        if len(parts) >= 2:
            return parts[-1]
        return None

    # Count frequency bands in top/bottom features for LDA
    lda_freq_counts = {freq: 0 for freq in FREQ_ORDER}
    if len(lda_sig_idx) > 0:
        for idx in lda_sig_idx:
            coef = lda_all_coefs[idx].flatten()
            # Count top and bottom 10 features by frequency
            coef_series = pd.Series(coef, index=lda_colnames)
            top_10 = coef_series.nlargest(10)
            bottom_10 = coef_series.nsmallest(10)
            for feat_name in top_10.index.tolist() + bottom_10.index.tolist():
                freq = get_freq_from_name(feat_name)
                if freq in lda_freq_counts:
                    lda_freq_counts[freq] += 1
        # Store for stacked bar plot
        all_lda_counts[participant] = lda_freq_counts.copy()

    # Count frequency bands in top/bottom features for LASSO
    lasso_freq_counts = {freq: 0 for freq in FREQ_ORDER}
    if len(lasso_sig_idx) > 0:
        for idx in lasso_sig_idx:
            coef = lasso_all_coefs[idx].flatten()
            # Count top and bottom 10 features by frequency
            coef_series = pd.Series(coef, index=lasso_colnames)
            top_10 = coef_series.nlargest(10)
            bottom_10 = coef_series.nsmallest(10)
            for feat_name in top_10.index.tolist() + bottom_10.index.tolist():
                freq = get_freq_from_name(feat_name)
                if freq in lasso_freq_counts:
                    lasso_freq_counts[freq] += 1
        # Store for stacked bar plot
        all_lasso_counts[participant] = lasso_freq_counts.copy()

    # Create LDA count plot
    if len(lda_sig_idx) > 0:
        lda_count_df = pd.DataFrame({
            'Frequency': FREQ_LABELS,
            'Count': [lda_freq_counts[freq] for freq in FREQ_ORDER]
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Frequency', y='Count', data=lda_count_df,
                    order=FREQ_LABELS, color='lightcoral', ax=ax)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='-', linewidth=0.7, color='gray', alpha=0.5)
        ax.set_title(f'{participant} - LDA: Frequency Band Prevalence in Top Features\n(n={len(lda_sig_idx)} significant models, top/bottom 10 features each)',
                     fontsize=14)
        ax.set_xlabel('Frequency Band', fontsize=12)
        ax.set_ylabel('Count (Feature Prominence)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(fig_dir + f'{participant}_lda_freq_counts.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(fig_dir + f'{participant}_lda_freq_counts.png', dpi=300, bbox_inches='tight')
        plt.close()

        lda_count_df.to_csv(fig_dir + f'{participant}_lda_freq_counts.csv', index=False)
        print("Saved LDA frequency counts")
    else:
        print(f"No significant LDA models for {participant}, skipping LDA plot")

    # Create LASSO count plot
    if len(lasso_sig_idx) > 0:
        lasso_count_df = pd.DataFrame({
            'Frequency': FREQ_LABELS,
            'Count': [lasso_freq_counts[freq] for freq in FREQ_ORDER]
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Frequency', y='Count', data=lasso_count_df,
                    order=FREQ_LABELS, color='lightcoral', ax=ax)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='-', linewidth=0.7, color='gray', alpha=0.5)
        ax.set_title(f'{participant} - LASSO: Frequency Band Prevalence in Top Features\n(n={len(lasso_sig_idx)} significant models, top/bottom 10 features each)',
                     fontsize=14)
        ax.set_xlabel('Frequency Band', fontsize=12)
        ax.set_ylabel('Count (Feature Prominence)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(fig_dir + f'{participant}_lasso_freq_counts.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(fig_dir + f'{participant}_lasso_freq_counts.png', dpi=300, bbox_inches='tight')
        plt.close()

        lasso_count_df.to_csv(fig_dir + f'{participant}_lasso_freq_counts.csv', index=False)
        print("Saved LASSO frequency counts")
    else:
        print(f"No significant LASSO models for {participant}, skipping LASSO plot")

    # Combined counts (LDA + LASSO)
    if len(lda_sig_idx) > 0 or len(lasso_sig_idx) > 0:
        combined_counts = {freq: lda_freq_counts.get(freq, 0) + lasso_freq_counts.get(freq, 0) for freq in FREQ_ORDER}
        combined_count_df = pd.DataFrame({
            'Frequency': FREQ_LABELS,
            'Count': [combined_counts[freq] for freq in FREQ_ORDER]
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Frequency', y='Count', data=combined_count_df,
                    order=FREQ_LABELS, color='lightcoral', ax=ax)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='-', linewidth=0.7, color='gray', alpha=0.5)
        ax.set_title(f'{participant} - Combined: Frequency Band Prevalence in Top Features\n(LDA + LASSO, top/bottom 10 features per model)',
                     fontsize=14)
        ax.set_xlabel('Frequency Band', fontsize=12)
        ax.set_ylabel('Count (Feature Prominence)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(fig_dir + f'{participant}_combined_freq_counts.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(fig_dir + f'{participant}_combined_freq_counts.png', dpi=300, bbox_inches='tight')
        plt.close()

        combined_count_df.to_csv(fig_dir + f'{participant}_combined_freq_counts.csv', index=False)
        print("Saved Combined frequency counts")

print("\n" + "="*50)
print("Done processing all patients!")
print("="*50)

# Create stacked bar plots across all participants
combined_fig_dir = os.environ.get('COMBINED_FIG_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_figures')) + '/'
if not os.path.exists(combined_fig_dir):
    os.makedirs(combined_fig_dir)

# Define colors for each participant
ppt_colors = sns.color_palette("husl", len(all_ppts))
ppt_color_map = {ppt: ppt_colors[i] for i, ppt in enumerate(all_ppts)}

# Helper function to create stacked bar plot
def create_stacked_bar(counts_dict, title, filename, normalize=False, ylabel=None):
    """Create a stacked bar plot from counts dictionary.

    If normalize=True, each participant is normalized to contribute equally
    (1/N each, where N = number of participants), so the total sums to 1.
    ylabel can be specified to override the default label.
    """
    # Normalize if requested - each participant sums to 1/N, total sums to 1
    if normalize:
        n_ppts = len(all_ppts)
        normalized_counts = {}
        for ppt in all_ppts:
            ppt_total = sum(counts_dict[ppt].values())
            if ppt_total == 0:
                raise ValueError(f"Participant {ppt} has zero significant models. "
                                 f"Normalization would give unequal weight to remaining participants.")
            # Each participant contributes 1/N to the total
            normalized_counts[ppt] = {freq: counts_dict[ppt][freq] / ppt_total / n_ppts
                                      for freq in FREQ_ORDER}
        counts_dict = normalized_counts

    # Build dataframe for stacking
    data = []
    for ppt in all_ppts:
        for i, freq in enumerate(FREQ_ORDER):
            data.append({
                'Frequency': FREQ_LABELS[i],
                'Participant': ppt,
                'Count': counts_dict[ppt][freq]
            })
    df = pd.DataFrame(data)

    # Pivot for stacking
    pivot_df = df.pivot(index='Frequency', columns='Participant', values='Count')
    pivot_df = pivot_df.reindex(FREQ_LABELS)  # Ensure correct order
    pivot_df = pivot_df[all_ppts]  # Ensure participant order

    fig, ax = plt.subplots(figsize=(6, 4))
    pivot_df.plot(kind='bar', stacked=True, ax=ax,
                  color=[ppt_color_map[ppt] for ppt in all_ppts],
                  edgecolor='white', linewidth=0.5)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Frequency Band', fontsize=9)
    if ylabel is None:
        ylabel = 'Proportion' if normalize else 'Count (Feature Prominence)'
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(title='Participant', bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=8, title_fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(combined_fig_dir + filename + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(combined_fig_dir + filename + '.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSV
    pivot_df.to_csv(combined_fig_dir + filename + '.csv')
    print(f"Saved {filename}")

# Create LDA stacked bar plot
print("\nCreating stacked bar plots...")
create_stacked_bar(all_lda_counts,
                   'LDA: Frequency Band Prevalence Across Participants\n(Top/Bottom 10 Features per Significant Model, Normalized)',
                   'stacked_lda_freq_counts',
                   normalize=True)

# Create LASSO stacked bar plot
create_stacked_bar(all_lasso_counts,
                   'LASSO: Frequency Band Prevalence Across Participants\n(Top/Bottom 10 Features per Significant Model, Normalized)',
                   'stacked_lasso_freq_counts',
                   normalize=True)

# Create combined (LDA + LASSO) stacked bar plot
# Normalize each method per-participant first (each participant = 0.1 per method),
# then add. This way both participants and methods contribute equally.
n_ppts = len(all_ppts)

all_combined_counts = {}
for ppt in all_ppts:
    all_combined_counts[ppt] = {}

    # Get per-participant totals for each method
    lda_ppt_total = sum(all_lda_counts[ppt].values())
    lasso_ppt_total = sum(all_lasso_counts[ppt].values())

    # Check that participant has significant models in both methods
    if lda_ppt_total == 0:
        raise ValueError(f"Participant {ppt} has zero significant LDA models. "
                         f"Combined plot normalization would give unequal weight to remaining participants.")
    if lasso_ppt_total == 0:
        raise ValueError(f"Participant {ppt} has zero significant LASSO models. "
                         f"Combined plot normalization would give unequal weight to remaining participants.")

    for freq in FREQ_ORDER:
        # Each participant contributes 0.1 per method (0.5 / 5 participants)
        lda_prop = all_lda_counts[ppt][freq] / lda_ppt_total / n_ppts * 0.5
        lasso_prop = all_lasso_counts[ppt][freq] / lasso_ppt_total / n_ppts * 0.5
        all_combined_counts[ppt][freq] = lda_prop + lasso_prop

create_stacked_bar(all_combined_counts,
                   'Combined (LDA + LASSO): Frequency Band Prevalence Across Participants\n(Top/Bottom 10 Features per Significant Model, Normalized)',
                   'stacked_combined_freq_counts',
                   normalize=False,  # Already normalized: each ppt = 0.2, total = 1
                   ylabel='Proportion')

print("\n" + "="*50)
print("All stacked bar plots saved!")
print("="*50)
