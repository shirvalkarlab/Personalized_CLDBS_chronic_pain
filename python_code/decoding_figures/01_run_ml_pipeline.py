import os
import pickle
import random
from random import shuffle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Select the participant to analyze. Choose one of P1-P5, in order of preference:
#   1. command line:   python 01_run_ml_pipeline.py --participant P3
#   2. environment:    DECODING_PARTICIPANT=P3 python 01_run_ml_pipeline.py
#   3. the default below (edit if you prefer running without arguments)
# To regenerate all participants, loop in the shell:
#   for p in P1 P2 P3 P4 P5; do python 01_run_ml_pipeline.py --participant $p; done
import argparse

_DEFAULT_PARTICIPANT = "P5"
_parser = argparse.ArgumentParser(description="Fit LASSO + LDA pain-decoding models for one participant.")
_parser.add_argument("--participant", "-p", default=os.environ.get("DECODING_PARTICIPANT", _DEFAULT_PARTICIPANT),
                     choices=["P1", "P2", "P3", "P4", "P5"],
                     help="De-identified participant label (P1-P5). Default: %(default)s")
# parse_known_args so the script still runs inside notebooks/other launchers.
_args, _ = _parser.parse_known_args()
PARTICIPANT_LABEL = _args.participant

# Directory paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

# INPUT: neural_features/ holds one subfolder per participant. Override the root
# with DECODING_FEATURES_ROOT if your copy lives elsewhere; by default it is the
# repository's neural_features/ directory.
_FEATURES_ROOT = os.environ.get(
    "DECODING_FEATURES_ROOT", os.path.join(_REPO, "neural_features"))
FEATURE_FOLD = os.path.join(_FEATURES_ROOT, PARTICIPANT_LABEL) + "/"
FILE_PREFIX = f"{PARTICIPANT_LABEL}_"   # feature/label filename prefix (e.g. P3_)

# OUTPUT: model results are written here, one subfolder per participant label,
# separate from the input data. Override the root with DECODING_RESULTS_ROOT if
# desired.
_RESULTS_ROOT = os.environ.get(
    "DECODING_RESULTS_ROOT", os.path.join(_HERE, "decoding_results"))
RESULTS_DIR = os.path.join(_RESULTS_ROOT, PARTICIPANT_LABEL) + "/"

# Pain metrics to analyze
PAIN_METRICS = ['nrs_s0', 'intensity_vas_s0', 'unpleasantness_vas_s0',
                'mood_vas_s0', 'relief_s0']

# Time windows
SIZE_LIST = ['one_minute', 'five_minute', 'ten_minute', 'thirty_minute']

# Number of bootstrap iterations for significance testing
N_BOOTSTRAP = 1000

# Create results directory if needed
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def remove_trials_func(rm_trials, record_id, arrays):
    rm_trials = np.sort(np.unique(rm_trials).astype(int))
    for z in rm_trials[::-1]:
        id_loc = np.where(record_id == z)[0][0]
        for i in range(len(arrays)):
            if isinstance(arrays[i], np.ndarray):
                arrays[i] = np.delete(arrays[i], id_loc, 0)
            elif isinstance(arrays[i], pd.DataFrame):
                arrays[i] = arrays[i].set_index(np.arange(0,arrays[i].shape[0]))
                arrays[i] = arrays[i].drop(index = id_loc, axis=0)
    return arrays

def get_nonmin_surveys(labels, min_len_minutes):
    """
    Identify surveys that are too close together in time.

    Surveys closer than min_len_minutes apart don't have enough neural data
    preceding them to be valid samples for that time window.

    Args:
        labels: DataFrame with survey timestamps
        min_len_minutes: Minimum time between surveys in minutes

    Returns:
        Array of record_ids that should be removed
    """
    # Survey times are stored under one of two column names depending on which
    # survey instrument the participant used: P1 uses survey_start_timestamp,
    # the others scales_vasmpq_timestamp. Selecting by whichever column is
    # present keeps this independent of any participant-to-schema mapping.
    for timestamp_column in ('survey_start_timestamp', 'scales_vasmpq_timestamp'):
        if timestamp_column in labels.columns:
            break
    else:
        raise KeyError(
            "No survey-time column found in the label file. Expected one of "
            "'survey_start_timestamp' or 'scales_vasmpq_timestamp'; got "
            f"{list(labels.columns)}")
    date_strings = np.array(labels[timestamp_column])

    # Parse timestamps
    dates = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in date_strings]

    # Calculate time differences between consecutive surveys
    differences = []
    for i in range(len(dates) - 1):
        diff_minutes = (dates[i+1] - dates[i]).total_seconds() / 60
        differences.append(diff_minutes)
    differences = np.array(differences)

    # Find surveys that are too close to the previous one
    record_ids = labels['record_id'].values
    invalid_ids = record_ids[np.where(differences < min_len_minutes)[0] + 1]

    print(f"  Removing {len(invalid_ids)} surveys < {min_len_minutes} min apart")
    return invalid_ids


def drop_saved_index(table):
    """
    Drop a leading unnamed index column, if the CSV carries one.

    A CSV written with `to_csv(index=True)` stores the row number as a nameless
    first column that pandas reads back as "Unnamed: 0". That column is not data
    and must not reach the models. Matching on the column name rather than
    dropping position 0 works whether or not the index is present; a positional
    drop would discard the first real measurement when it is absent.
    """
    return table.drop(columns=[column for column in table.columns
                               if str(column).startswith("Unnamed:")])


def load_and_clean_features(feature_fold, time_window, min_len):
    """
    Load feature and label data for a specific time window.

    Removes:
        - Rows with NaN values in features
        - Surveys that are too close together in time

    Args:
        feature_fold: Path to folder containing feature CSVs
        time_window: 'one_min', 'five_min', etc.
        min_len: Minimum minutes between surveys

    Returns:
        features: Cleaned feature DataFrame
        labels: Cleaned labels DataFrame
    """
    # Map the internal window name to the filename tag and load. Feature files
    # are named "<prefix><NN>min_features.csv" (e.g. "P3_05min_features.csv").
    _tag = {"one_min": "01min", "five_min": "05min",
            "ten_min": "10min", "thirty_min": "30min"}[time_window]
    features = drop_saved_index(
        pd.read_csv(f"{feature_fold}{FILE_PREFIX}{_tag}_features.csv"))
    labels = drop_saved_index(
        pd.read_csv(f"{feature_fold}{FILE_PREFIX}{_tag}_labels.csv"))

    # Remove rows where features contain NaN
    valid_mask = ~features.iloc[:, 1].isna()
    labels = labels[valid_mask]
    features = features[valid_mask]

    # Remove surveys too close together
    invalid_ids = get_nonmin_surveys(labels, min_len)
    record_ids = labels['record_id'].values
    features, labels = remove_trials_func(invalid_ids, record_ids, [features, labels])

    return features, labels


# =============================================================================
# LASSO REGRESSION FUNCTIONS
# =============================================================================

def prepare_regression_data(feature_df, label_df, label_col):
    """
    Prepare feature matrix X and target vector Y for regression.

    Preprocessing:
        - Remove rows with NaN in the target label
        - Remove outliers (z-score > 3)
        - Note: Standardization is done inside CV loop to avoid data leakage

    Args:
        feature_df: DataFrame of neural features
        label_df: DataFrame of pain labels
        label_col: Name of the target column

    Returns:
        X: Feature matrix (n_samples, n_features) - NOT standardized
        Y: Target vector (n_samples,)
        success: Boolean indicating if enough samples remain
    """
    # Remove rows with NaN in target
    valid_mask = ~label_df[label_col].isna()
    feature_df = feature_df[valid_mask]
    label_df = label_df[valid_mask]

    # Remove outliers (any feature with z-score > 3)
    outlier_mask = (np.abs(stats.zscore(feature_df)) < 3).all(axis=1)
    label_df = label_df[outlier_mask]
    feature_df = feature_df[outlier_mask]

    x = np.array(feature_df)

    if len(label_df) <= 4:
        print("  NOT ENOUGH SAMPLES")
        return np.nan, np.nan, False

    y = np.array(label_df[label_col])

    return x, y, True


def run_lasso_models(feature_list, label_list, size_list, pain_metrics, save=True):
    """
    Run LASSO regression for all pain metrics and time windows.

    Uses leave-one-out cross-validation to compute R² values.
    Uses fresh data for each metric (no cumulative shrinking).

    Args:
        feature_list: List of feature DataFrames (one per time window)
        label_list: List of label DataFrames (one per time window)
        size_list: List of time window names
        pain_metrics: List of pain metric column names
        save: Whether to save results to disk

    Returns:
        all_coefs: Array of coefficients (metrics x windows x features)
        all_r2s: List of R² values for all models
    """
    # Initialize output storage
    all_r2s = []
    all_coefs = []
    metric_org = []
    size_org = []
    sample_counts = {}

    # Create prediction vs ground truth figure
    fig, axes = plt.subplots(nrows=len(pain_metrics), ncols=len(size_list),
                             figsize=(20, 40))

    for i, (features, labels, size) in enumerate(zip(feature_list, label_list, 
                                                     size_list)):

            for j, metric in enumerate(pain_metrics):
                print(f"Working on {size} and {metric}")
                size_org.append(size)
                metric_org.append(metric)
                print('>>> {}'.format(metric))

                X, Y, success = prepare_regression_data(
                    features.copy(),
                    labels.copy(),
                    metric
                )

                if not success:
                    sample_counts[(size, metric)] = 0
                    all_r2s.append(np.nan)
                    all_coefs.append(np.full(features.shape[1], np.nan))
                    continue

                print(f"X shape: {X.shape}")
                print(f"Y shape: {Y.shape}")
                sample_counts[(size, metric)] = X.shape[0]

                # Leave-one-out cross-validation
                cv = LeaveOneOut()
                all_preds, all_gts, coefs = [], [], []

                for train_idx, test_idx in cv.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    Y_train, Y_test = Y[train_idx], Y[test_idx]

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    model = Lasso(alpha=1e-2, max_iter=int(1e4), tol=1e-4)
                    model.fit(X_train, Y_train)

                    coefs.append(model.coef_)
                    all_preds.extend(model.predict(X_test))
                    all_gts.extend(Y_test)

                # Calculate R² (correlation squared)
                r2 = np.round(np.corrcoef(all_gts, all_preds)[0, 1] ** 2, 3)
                rmse = np.round(np.sqrt(np.mean((np.array(all_preds) - np.array(all_gts))**2)), 4)

                all_r2s.append(r2)
                all_coefs.append(np.mean(np.array(coefs), axis=0))

                print(f"    R² = {r2}, RMSE = {rmse}")

                # Plot predictions vs ground truth
                ax = axes[j, i]
                if metric in ['nrs_s0', 'best_s0', 'worst_s0']:
                    max_val, min_val = 11, -1
                    ticks = [0, 2, 4, 6, 8, 10]
                else:
                    max_val, min_val = 110, -10
                    ticks = [0, 20, 40, 60, 80, 100]

                ax.scatter(all_gts, all_preds)
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_aspect('equal')
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.75, zorder=0)
                ax.set_title(f'{size}\n{metric}\nR²: {r2} | RMSE: {rmse}', size=14)
                ax.set_xlabel('Ground Truth', size=12)
                ax.set_ylabel('Predictions', size=12)

    # Save results
    if save:
        # Save prediction plots
        fig.savefig(RESULTS_DIR + 'lasso_preds_vs_gt.pdf', bbox_inches="tight", dpi=100)

        # Save sample counts
        sample_df = pd.DataFrame([
            {'Time Window': sz, 'Metric': met, 'Num Samples': count}
            for (sz, met), count in sample_counts.items()
        ])
        sample_df.to_csv(RESULTS_DIR + 'regression_num_sample_per_model.csv', index=False)

        # Save coefficients DataFrame
        coef_df = pd.DataFrame({
            'r-squared': all_r2s,
            #'metric': [pm for _ in size_list for pm in pain_metrics],
            #'size': [sz for sz in size_list for _ in pain_metrics],
            'coef': all_coefs,
            'metric_org': metric_org,
            'size_org': size_org
        })
        coef_df.to_pickle(RESULTS_DIR + 'lasso_coef_df.pkl')

        # Save column names for later feature identification
        np.save(RESULTS_DIR + 'lasso_colnames.npy', features.columns)

        # Create summary bar plot
        result_df = pd.DataFrame({
            'r-squared': all_r2s,
            'metric': [pm for _ in size_list for pm in pain_metrics],
            'size': [sz for sz in size_list for _ in pain_metrics]
        })

        plt.figure(figsize=(20, 5))
        sns.barplot(data=result_df, y='r-squared', x='size', hue='metric',
                    palette='dark', alpha=0.6)
        plt.legend(ncol=len(pain_metrics), fontsize=12)
        plt.title(f'{PARTICIPANT_LABEL} Regression Results Summary')
        plt.ylim(0, 1)
        plt.savefig(RESULTS_DIR + 'lasso_results_summary.pdf', bbox_inches="tight", dpi=100)
        plt.close()

    plt.close(fig)
    return np.array(all_coefs), all_r2s


# =============================================================================
# LDA CLASSIFICATION FUNCTIONS
# =============================================================================

def binarize_labels(labels):
    """
    Convert continuous pain labels to binary (low/high pain).

    Uses 30th and 70th percentiles as thresholds.
    Samples in the middle tercile are marked as NaN (excluded).

    Args:
        labels: Array of continuous pain values

    Returns:
        labels_: Binary labels (0=low, 1=high, NaN=middle)
        num_low_pain_samples: Count of low pain samples
        num_high_pain_samples: Count of high pain samples
    """
    labels_ = np.float32(labels.copy())

    lower_third = np.percentile(labels_, 30)
    upper_third = np.percentile(labels_, 70)

    # Use labels_ throughout
    print("{} low pain samples".format(sum(labels_ <= lower_third)))
    print("{} high pain samples".format(sum(labels_ >= upper_third)))

    num_low_pain_samples = sum(labels_ <= lower_third)
    num_high_pain_samples = sum(labels_ >= upper_third)

    # Use labels_ for all conditions
    labels_[labels_ <= lower_third] = 0.0
    labels_[(labels_ > lower_third) & (labels_ < upper_third)] = np.nan
    labels_[labels_ >= upper_third] = 1.0

    return labels_, num_low_pain_samples, num_high_pain_samples


def prepare_classification_data(feature_df, label_df, label_col):
    """
    Prepare feature matrix X and binary target Y for classification.

    Args:
        feature_df: DataFrame of neural features
        label_df: DataFrame of pain labels
        label_col: Name of the target column

    Returns:
        X: Feature matrix
        Y: Binary target vector (0 = low pain, 1 = high pain)
        n_low: Number of low pain samples
        n_high: Number of high pain samples
        success: Boolean indicating if enough samples remain
    """
    # Remove rows with NaN in target
    valid_mask = ~label_df[label_col].isna()
    feature_df = feature_df[valid_mask]
    label_df = label_df[valid_mask]

    # Remove outliers
    outlier_mask = (np.abs(stats.zscore(feature_df)) < 3).all(axis=1)
    label_df = label_df[outlier_mask]
    feature_df = feature_df[outlier_mask]

    x = np.array(feature_df)

    if len(label_df) <= 4:
        print("  NOT ENOUGH SAMPLES")
        return np.nan, np.nan, np.nan, np.nan, False

    y, n_lp, n_hp = binarize_labels(np.array(label_df[label_col]))
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    return x, y, n_lp, n_hp, True


def run_lda_models(feature_list, label_list, size_list, pain_metrics, save=True):
    """
    Run LDA classification for all pain metrics and time windows.

    Uses leave-one-out cross-validation to compute AUC values.

    Args:
        feature_list: List of feature DataFrames (one per time window)
        label_list: List of label DataFrames (one per time window)
        size_list: List of time window names
        pain_metrics: List of pain metric column names
        save: Whether to save results to disk

    Returns:
        all_coefs: Array of coefficients
        all_aucs: List of AUC values for all models
    """
    # Initialize output storage
    all_aucs = []
    all_coefs = []
    metric_org = []
    size_org = []
    sample_counts = {}

    # Create ROC curve figure
    fig, axes = plt.subplots(nrows=len(pain_metrics), ncols=len(size_list),
                             figsize=(20, 40))
    for i, (features, labels, size) in enumerate(zip(feature_list, label_list, size_list)):
            for j, metric in enumerate(pain_metrics):
                print("length: {}".format(size))
                print("metric: {}".format(metric))
                size_org.append(size)
                metric_org.append(metric)

                X, Y, n_low, n_high, success = prepare_classification_data(
                    features.copy(), labels.copy(), metric
                )

                if not success:
                    sample_counts[(size, metric)] = {'low': 0, 'high': 0}
                    all_aucs.append(np.nan)
                    all_coefs.append(np.full((1, features.shape[1]), np.nan))
                    continue

                print("X SHAPE: ", X.shape)
                print("Y SHAPE: ", Y.shape)
                print("CONTINUING")
                sample_counts[(size, metric)] = {'low': n_low, 'high': n_high}

                # Leave-one-out cross-validation
                cv = LeaveOneOut()
                all_preds, all_gts, coefs = [], [], []

                for train_idx, test_idx in cv.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    Y_train, Y_test = Y[train_idx], Y[test_idx]

                    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    model.fit(X_train, Y_train)

                    coefs.append(model.coef_)
                    all_preds.append(model.predict_proba(X_test))
                    all_gts.append(Y_test)

                # Calculate AUC
                all_preds = [j for i in all_preds for j in i]
                all_preds = np.array(all_preds)[:, 1]
                model_auc = np.round(roc_auc_score(Y, all_preds), 3)
                all_aucs.append(model_auc)
                all_coefs.append(np.mean(np.array(coefs), axis=0))

                print(f"    AUC = {model_auc}")

                # Plot ROC curve
                ax = axes[j, i]
                fpr, tpr, _ = roc_curve(all_gts, all_preds)
                ax.plot(fpr, tpr, 'b', label=f'AUC = {model_auc:.2f}')
                ax.plot([0, 1], [0, 1], 'r--')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.set_aspect('equal')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{size}\n{metric}\nAUC: {model_auc}', size=14)

    # Save results
    if save:
        # Save ROC plots
        fig.savefig(RESULTS_DIR + 'lda_preds_vs_gt.pdf')

        # Save sample counts
        sample_rows = [
            {'Time Window': sz, 'Metric': met,
             'Low Pain Samples': vals['low'], 'High Pain Samples': vals['high'],
             'Total Used': vals['low'] + vals['high']}
            for (sz, met), vals in sample_counts.items()
        ]
        pd.DataFrame(sample_rows).to_csv(RESULTS_DIR + 'lda_num_samples_per_model.csv', index=False)

        # Save coefficients DataFrame
        coef_df = pd.DataFrame({
            'auc': all_aucs,
            #'metric': [pm for _ in size_list for pm in pain_metrics],
            #'size': [sz for sz in size_list for _ in pain_metrics],
            'coef': all_coefs,
            'metric_org': metric_org,
            'size_org': size_org
        })
        coef_df.to_pickle(RESULTS_DIR + 'lda_coef_df.pkl')
        np.save(RESULTS_DIR + 'lda_colnames.npy', features.columns)

        # Save sample count dict for later use
        with open(RESULTS_DIR + 'count_dict.pkl', 'wb') as f:
            pickle.dump(sample_counts, f)

        # Create summary bar plot
        result_df = pd.DataFrame({
            'auc': all_aucs,
            'metric': [pm for _ in size_list for pm in pain_metrics],
            'size': [sz for sz in size_list for _ in pain_metrics]
        })
        plt.figure(figsize=(20, 5))
        sns.barplot(data=result_df, y='auc', x='size', hue='metric',
                    palette='dark', alpha=0.6)
        plt.legend(ncol=len(pain_metrics), fontsize=12)
        plt.title(f'{PARTICIPANT_LABEL} Classification Results Summary')
        plt.savefig(RESULTS_DIR + 'lda_results_summary.png', bbox_inches="tight", dpi=100)
        plt.close()

    plt.close(fig)
    return np.array(all_coefs), all_aucs


# =============================================================================
# BOOTSTRAP SIGNIFICANCE TESTING
# =============================================================================

def run_bootstrap_lasso(feature_list, label_list, size_list, pain_metrics, n_bootstrap):
    """
    Run bootstrap permutation testing for LASSO models.

    Shuffles features to break feature-label relationship, establishing
    a null distribution for R² values.

    Returns:
        Array of shape (n_bootstrap, n_models) with permuted R² values
    """
    boots_r2s = []

    for iteration in range(n_bootstrap):
        print(iteration)

        # Shuffle each time window's features independently
        shuffled_features = []
        for features in feature_list:
            idx = list(range(len(features)))
            shuffle(idx)
            shuffled_features.append(features.iloc[idx, :])

        # Run LASSO without saving
        _, all_r2 = run_lasso_models(shuffled_features, label_list, size_list,
                                      pain_metrics, save=False)
        boots_r2s.append(all_r2)

    return np.array(boots_r2s)


def run_bootstrap_lda(feature_list, label_list, size_list, pain_metrics, n_bootstrap):
    """
    Run bootstrap permutation testing for LDA models.

    Shuffles features to break feature-label relationship, establishing
    a null distribution for AUC values.

    Returns:
        Array of shape (n_bootstrap, n_models) with permuted AUC values
    """
    boots_aucs = []

    for iteration in range(n_bootstrap):

        # Shuffle each time window's features independently
        shuffled_features = []
        for features in feature_list:
            idx = list(range(len(features)))
            shuffle(idx)
            shuffled_features.append(features.iloc[idx, :])

        # Run LDA without saving
        _, all_aucs = run_lda_models(shuffled_features, label_list, size_list,
                                      pain_metrics, save=False)
        boots_aucs.append(all_aucs)

    return np.array(boots_aucs)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"  BIOMARKER DECODING PIPELINE - {PARTICIPANT_LABEL}")
    print(f"{'#'*60}")

    # -------------------------------------------------------------------------
    # STEP 1: Load and clean data for all time windows
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading and cleaning feature data...")

    one_min_features, one_min_labels = load_and_clean_features(
        FEATURE_FOLD, 'one_min', 1)
    five_min_features, five_min_labels = load_and_clean_features(
        FEATURE_FOLD, 'five_min', 5)
    ten_min_features, ten_min_labels = load_and_clean_features(
        FEATURE_FOLD, 'ten_min', 10)
    thirty_min_features, thirty_min_labels = load_and_clean_features(
        FEATURE_FOLD, 'thirty_min', 30)

    feature_list = [one_min_features, five_min_features,
                    ten_min_features, thirty_min_features]

    label_list = [one_min_labels, five_min_labels,
                  ten_min_labels, thirty_min_labels]

    print("\n  Loaded data shapes:")
    for size, features in zip(SIZE_LIST, feature_list):
        print(f"    {size}: {features.shape}")

    # -------------------------------------------------------------------------
    # STEP 2: Run LASSO regression
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Running LASSO regression models...")
    all_coefs_lasso, all_r2 = run_lasso_models(
        feature_list, label_list, SIZE_LIST, PAIN_METRICS)

    # -------------------------------------------------------------------------
    # STEP 3: Bootstrap significance testing for LASSO
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Running LASSO bootstrap permutation testing...")
    boots_r2s = run_bootstrap_lasso(
        feature_list, label_list, SIZE_LIST, PAIN_METRICS, N_BOOTSTRAP)

    np.save(RESULTS_DIR + 'lasso_boots_r2s.npy', boots_r2s)
    print(f"  Saved bootstrap R² distributions to: {RESULTS_DIR}lasso_boots_r2s.npy")

    # -------------------------------------------------------------------------
    # STEP 4: Run LDA classification
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Running LDA classification models...")
    all_coefs_lda, all_aucs = run_lda_models(
        feature_list, label_list, SIZE_LIST, PAIN_METRICS)

    # -------------------------------------------------------------------------
    # STEP 5: Bootstrap significance testing for LDA
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Running LDA bootstrap permutation testing...")
    boots_aucs = run_bootstrap_lda(
        feature_list, label_list, SIZE_LIST, PAIN_METRICS, N_BOOTSTRAP)

    np.save(RESULTS_DIR + 'lda_boots_aucs.npy', boots_aucs)
    print(f"  Saved bootstrap AUC distributions to: {RESULTS_DIR}lda_boots_aucs.npy")

    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------
    print(f"\n{'#'*60}")
    print("  PIPELINE COMPLETE")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'#'*60}\n")
