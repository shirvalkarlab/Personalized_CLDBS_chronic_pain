import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# HELPER FUNCTIONS (process_figure4bcd_inpatient_pain_decoding.ipynb)
# =============================================================================

# SIGNIFICANCE TESTING FUNCTIONS
def identify_significant_models(mdl, bootstrap_values, metric, SIGNIFICANCE_PERCENTILE=95):
    """
    Mark models as significant if actual performance exceeds bootstrap threshold.
    
    Args:
        mdl: DataFrame containing model performance ('metric_org' column expected)
        bootstrap_values: n_bootstrap x n_models array of performance from resampling
        SIGNIFICANCE_PERCENTILE: percentile of bootstrap distribution defining significance

    Returns:
        mdl: original dataframe with added columns:
            - 'sig_indices': boolean indicating significant models
            - 'p_values': p-values for each model
    """
    sig_indices = np.zeros(len(mdl), dtype=bool)
    p_values    = np.ones(len(mdl))
    
    mdl_perform = mdl.loc[:, metric]
    
    for i in range(len(mdl_perform)):
        threshold = np.percentile(bootstrap_values[:, i], SIGNIFICANCE_PERCENTILE)

        if mdl_perform[i] > threshold:
            sig_indices[i] = True
           
        if np.isnan(mdl_perform[i]):
            sig_indices[i] = False
            p_values[i]    = np.nan
        else:
            p_values[i] = (np.sum(bootstrap_values[:, i] > mdl_perform[i]) + 1) / (bootstrap_values.shape[0] + 1)

    mdl["p_values"]    = p_values
    mdl["sig_indices"] = sig_indices

    return mdl

# FEATURE IMPORTANCE FUNCTIONS
def extract_electrode_names(column_names):
    """
    Extract electrode names from feature columns formatted as 'ELECTRODE_freq'.
    
    Args:
        column_names: list/array of feature names
    
    Returns:
        numpy array of electrode names
    """
    return np.array([name.split("_")[0] for name in column_names])

def calculate_channel_importance(coef_array, column_names, electrode_names):
    """
    Compute average coefficient magnitude per electrode across all frequency bands.

    Args:
        coef_array: 2D array (n_models x n_features)
        column_names: feature names
        electrode_names: electrode corresponding to each feature
    
    Returns:
        DataFrame: models x electrodes with mean coefficients
    """
    unique_electrodes = np.unique(electrode_names)
    all_channel_means = []

    for electrode in unique_electrodes:
        mask = electrode_names == electrode
        mean_coefs = np.mean(coef_array[:, mask], axis=1)
        all_channel_means.append(mean_coefs)

    return pd.DataFrame(np.array(all_channel_means).T, columns=unique_electrodes)

def count_top_features(channel_df, n_top=10):
    """
    Count how frequently each electrode appears among top/bottom features across models.

    Args:
        channel_df: DataFrame (models x electrodes) of mean coefficients
        n_top: number of top/bottom features to consider per model
    
    Returns:
        Series: electrode counts
    """
    top_features = []
    bottom_features = []

    for i in range(channel_df.shape[0]):
        coefs = channel_df.iloc[i, :]
        top_features.extend(coefs.nlargest(n_top).index.tolist())
        bottom_features.extend(coefs.nsmallest(n_top).index.tolist())

    return pd.Series(top_features + bottom_features).value_counts()

def process_metric_coefficients(coef_array, mdl, column_names, metric, method_name, TOP_N_FEATURES=10):
    """
    Process coefficients for a single pain metric and count feature importance.

    Args:
        coef_array: array of model coefficients
        mdl: model DataFrame with significance info
        column_names: feature names
        metric: target pain metric
        method_name: 'LDA' or 'LASSO'
    
    Returns:
        DataFrame with electrode counts or None if no significant models
    """
    metric_labels = mdl.metric_org
    sig_indices = np.where(mdl.sig_indices)[0]
    metric_mask = np.where(metric_labels[sig_indices] == metric)[0]
    metric_indices = sig_indices[metric_mask]

    if len(metric_indices) == 0:
        print(f"    No significant {method_name} models for {metric}")
        return None

    print(f"    {method_name} significant models for {metric}: {len(metric_indices)}")
    selected_coefs = coef_array[metric_indices, :]
    electrode_names = extract_electrode_names(column_names)
    channel_df = calculate_channel_importance(selected_coefs, column_names, electrode_names)
    counts_df = pd.DataFrame(count_top_features(channel_df, TOP_N_FEATURES), columns=['count'])
    
    return counts_df



# =============================================================================
# Helper functions for Figure 4B (LASSO) and Figure 4C (LDA)
# =============================================================================

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
PT_LABELS = ['P1', 'P2', 'P3', 'P4', 'P5']

MATLAB_COLORS = [
    (0.0000, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.9290, 0.6940, 0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
]

# -----------------------------------------------------------------------------
# INTERNAL HELPERS (not meant to be called directly)
# -----------------------------------------------------------------------------
def _get_target_model(model_config, pt, method):
    """Return model string like 'five_minute_nrs_s0'."""
    metric, size = model_config[pt][method]
    return f"{size}_{metric}"


def _load_and_subset(data_dir, pt, filename, target_model):
    """Load dataframe and subset rows for desired model."""
    df = pd.read_excel(
        data_dir / 'figure4_processed_inpatient_pain_decoding' / filename,
        index_col=0
    )
    return df[df['model'] == target_model]


# -----------------------------------------------------------------------------
# PUBLIC FUNCTION: FIGURE 4C (LDA ROC)
# -----------------------------------------------------------------------------
def plot_lda_roc(model_config, data_dir, fig_dir=None, save_fig=True):

    plt.figure(figsize=(4, 4))

    for i, pt in enumerate(PT_LABELS):
        color = MATLAB_COLORS[i % len(MATLAB_COLORS)]

        target_model = _get_target_model(model_config, pt, 'LDA')

        df = _load_and_subset(
            data_dir,
            pt,
            f'{pt}_LDA_ROCs_per_model.xlsx',
            target_model
        )

        fpr = df['FPR'].values
        tpr = df['TPR'].values

        auc = np.trapz(tpr, fpr)

        plt.plot(
            fpr,
            tpr,
            color=color,
            alpha=0.5,
            linewidth=2,
            label=f"{pt} (AUC={auc:.2f})"
        )

    # formatting
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("LDA Classification", fontsize=12)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(loc='lower right', fontsize=9, frameon=False)

    plt.gca().set_aspect('equal')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()

    if save_fig and fig_dir is not None:
        plt.savefig(fig_dir / 'Figure4C.pdf', dpi=300, bbox_inches='tight')

    plt.show()


# -----------------------------------------------------------------------------
# PUBLIC FUNCTION: FIGURE 4B (LASSO SCATTER)
# -----------------------------------------------------------------------------
def plot_lasso_scatter(model_config, data_dir, fig_dir=None, save_fig=False):

    plt.figure(figsize=(4, 4))

    for i, pt in enumerate(PT_LABELS):
        color = MATLAB_COLORS[i % len(MATLAB_COLORS)]

        target_model = _get_target_model(model_config, pt, 'LASSO')

        df = _load_and_subset(
            data_dir,
            pt,
            f'{pt}_lasso_predicted_values_per_model.xlsx',
            target_model
        )

        y_true = df['ground_truth'].values
        y_pred = df['predicted_values'].values

        # remove NaNs
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]

        # scale NRS → 0–100
        if "nrs" in target_model.lower():
            y_true *= 10
            y_pred *= 10

        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2

        plt.scatter(
            y_true,
            y_pred,
            s=12,
            alpha=0.5,
            color=color,
            edgecolors='none',
            label=f"{pt} (R²={r2:.2f})"
        )

    # formatting
    plt.plot([0, 100], [0, 100], 'k--', linewidth=1)
    plt.xlabel("Ground Truth Pain", fontsize=11)
    plt.ylabel("Predicted Pain", fontsize=11)
    plt.title("LASSO Regression", fontsize=12)

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(-0.02, 1.0),
        fontsize=9,
        frameon=False,
        borderaxespad=0.0
    )

    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    if save_fig and fig_dir is not None:
        plt.savefig(fig_dir / 'Figure4B.pdf', dpi=300, bbox_inches='tight')

    plt.show()
    
# ### Function: Split by hemisphere
def split_by_hemisphere(feature_counts, TOP_N_REGIONS=10):
    """
    Return top-N left and right hemisphere rows sorted by count.

    Args:
        feature_counts: DataFrame with columns ['anatomy', 'count_combined']

    Returns:
        left_df: top-N left hemisphere regions
        right_df: top-N right hemisphere regions
    """
    anat_count_df = feature_counts.groupby('anatomy')['count_combined'].sum().reset_index()
    
    def get_side(prefix, TOP_N_REGIONS):
        # Select regions starting with prefix (L or R), sort by prominence, keep top-N
        return (anat_count_df[anat_count_df['anatomy'].str.startswith(prefix)]
                .sort_values('count_combined', ascending=False)
                .head(TOP_N_REGIONS))
    
    return get_side('L',TOP_N_REGIONS), get_side('R',TOP_N_REGIONS)

# ### Function: Generate bar colors with highlights
def make_bar_colors(n_bars, red_indices, base_color='lightblue', highlight_color='red'):
    """
    Return color list for bars, highlighting selected indices.

    Args:
        n_bars: total number of bars
        red_indices: list of bar indices to highlight
        base_color: default bar color
        highlight_color: highlighted bar color

    Returns:
        list of colors for each bar
    """
    return [highlight_color if i in red_indices else base_color for i in range(n_bars)]

# ### Function: Create hemisphere plots
def create_hemisphere_plot(left_df, right_df, pt, output_path,
                           left_red_bars=None, right_red_bars=None, save_fig=False):
    """
    Create a side-by-side bar plot of top left/right hemisphere regions.

    Args:
        left_df: DataFrame of left hemisphere top regions
        right_df: DataFrame of right hemisphere top regions
        pt: participant identifier
        output_path: path to save figure
        left_red_bars: indices to highlight on left hemisphere
        right_red_bars: indices to highlight on right hemisphere
    """
    print(f"    Left hemisphere: {len(left_df)} regions")
    print(f"    Right hemisphere: {len(right_df)} regions")

    left_red_bars = left_red_bars or []
    right_red_bars = right_red_bars or []

    left_colors = make_bar_colors(len(left_df), left_red_bars)
    right_colors = make_bar_colors(len(right_df), right_red_bars)

    y_max = max(left_df['count_combined'].max(), right_df['count_combined'].max())

    # Create subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True)
    fig.subplots_adjust(wspace=0.1)
    fig.set_size_inches(8, 3)

    for ax, df, colors, label in [
        (ax_left, left_df, left_colors, 'Left hemisphere'),
        (ax_right, right_df, right_colors, 'Right hemisphere')
    ]:
        # Plot bar chart
        sns.barplot(x='anatomy', y='count_combined', data=df, palette=colors, ax=ax)
        ax.set_ylim(0, y_max)
        ax.text(0.2, 0.825, label, transform=ax.transAxes, fontsize=12)
        ax.set_xlabel('')
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='-', linewidth=0.7, color='gray', alpha=0.5)
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', length=0)

    ax_left.set_ylabel('Count (Feature Prominence)')
    ax_right.set_ylabel('')
    
    title_text = f"Choosing the Optimal Pain Decoding Brain Region ({pt})"
    plt.suptitle(title_text, fontsize=12)
    fig.align_xlabels()
    if save_fig:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
    else:
        print ('    NOT Saved')