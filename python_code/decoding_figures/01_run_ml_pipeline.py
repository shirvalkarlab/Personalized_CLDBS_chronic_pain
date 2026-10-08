import os

# One linear-algebra thread per process. The permutation nulls run one process
# per worker; letting each process also start one thread per core
# oversubscribes the machine (one P5 permutation round took 603 s with 16
# threads and 105 s with one). Set these variables yourself to override.
# They must be set before numpy is imported.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import contextlib
import io
import multiprocessing
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import shuffle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit
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
_parser.add_argument("--permutations", type=int, default=1000,
                     help="Permutations in each null distribution. 0 skips the nulls and "
                          "keeps the null files already in the results folder. Default: %(default)s")
# Two cores are left free by default so the machine stays usable during a run.
_parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2),
                     help="Processes used for the permutation nulls. "
                          "Default: two fewer than the CPU cores (%(default)s)")
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

# Number of permutations in each null distribution (the output files keep their
# historical "boots" names, but the null is a permutation test).
N_PERMUTATIONS = _args.permutations
N_WORKERS = max(1, _args.workers)

# Permutations handed to a worker process at a time.
LASSO_PERMUTATIONS_PER_TASK = 10
LDA_PERMUTATIONS_PER_TASK = 25

# LDA needs at least this many surveys in each pain class. With fewer, a
# leave-one-out training fold holds a single survey of one class, its covariance
# is singular, and the AUC is set by rounding error; such models are reported as
# not estimable (NaN).
MIN_SURVEYS_PER_CLASS = 3

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
                sample_counts[(size, metric)] = {'low': n_low, 'high': n_high}

                if min(n_low, n_high) < MIN_SURVEYS_PER_CLASS:
                    print("  NOT ESTIMABLE: fewer than "
                          f"{MIN_SURVEYS_PER_CLASS} surveys in a pain class")
                    all_aucs.append(np.nan)
                    all_coefs.append(np.full((1, features.shape[1]), np.nan))
                    continue

                print("CONTINUING")

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
# PERMUTATION NULL DISTRIBUTIONS
# =============================================================================
#
# Each permutation shuffles every time window's feature rows against the pain
# labels and re-runs all 20 models, giving one null score per model. The
# permutations are independent, so they run in parallel worker processes.
#
# LASSO: each permutation re-runs run_lasso_models() unchanged.
#
# LDA: each permutation is scored with lda_loo_aucs(), a batched closed form of
# LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') evaluated by
# leave-one-out. It computes the same model as scikit-learn without the
# features x features matrix solve that dominates scikit-learn's run time, and
# scores many permutations at once. On the shipped data it reproduces every
# observed AUC of run_lda_models() exactly (tests/test_permutation_null.py
# checks it against scikit-learn).

def draw_permutation_orders(feature_list, n_permutations):
    """
    Draw the feature-row order of every permutation up front.

    The same random.shuffle calls are made in the same order as a serial loop
    over permutations would make, so the permutations do not depend on how many
    worker processes later score them.

    Returns:
        List of n_permutations lists, each holding one row order per time window
    """
    orders = []
    for _ in range(n_permutations):
        per_window = []
        for features in feature_list:
            idx = list(range(len(features)))
            shuffle(idx)
            per_window.append(idx)
        orders.append(per_window)
    return orders


_LDA_SCALE_FLOOR = 10 * np.finfo(np.float64).eps   # StandardScaler treats smaller scales as 1


def _lda_class_terms(X):
    """
    Pieces of scikit-learn's Ledoit-Wolf class covariance for a batch of classes.

    For X of shape (K, m, p) -- K independent sets of m samples -- scikit-learn's
    _cov(X, shrinkage='auto') standardizes each feature, applies Ledoit-Wolf
    shrinkage s toward mu times the identity, and rescales. The result equals
    (1 - s) / m * Zt' Zt + s * mu * diag(scale**2), where Zt is the centred data.

    Returns:
        mean (K, p), Zt (K, m, p), s (K,), mu (K,), scale (K, p)
    """
    _, m, p = X.shape
    mean = X.mean(axis=1)
    centred = X - mean[:, None, :]
    scale = np.sqrt((centred ** 2).mean(axis=1))
    scale[scale < _LDA_SCALE_FLOOR] = 1.0
    standardized = centred / scale[:, None, :]
    standardized = standardized - standardized.mean(axis=1, keepdims=True)  # ledoit_wolf re-centres

    squared = standardized ** 2
    emp_cov_trace = squared.sum(axis=1) / m
    mu = emp_cov_trace.sum(axis=1) / p
    beta_ = (squared.sum(axis=2) ** 2).sum(axis=1)
    gram = standardized @ standardized.transpose(0, 2, 1)
    delta_ = (gram ** 2).sum(axis=(1, 2)) / m ** 2
    beta = (beta_ / m - delta_) / (p * m)
    delta = (delta_ - 2.0 * mu * emp_cov_trace.sum(axis=1) + p * mu ** 2) / p
    beta = np.minimum(beta, delta)
    s = np.where(beta == 0, 0.0, beta / np.where(delta == 0, 1.0, delta))
    return mean, standardized * scale[:, None, :], s, mu, scale


def _lda_decision(train_low, train_high, test):
    """
    Decision value of a two-class LDA (solver='lsqr', shrinkage='auto') for a batch.

    The pooled covariance is a diagonal matrix plus U'U, with U the stacked,
    weighted, centred training samples, so the Woodbury identity replaces the
    features x features solve with a samples x samples one.

    Args:
        train_low: (K, m0, p) low-pain training samples of K independent fits
        train_high: (K, m1, p) high-pain training samples
        test: (K, p) one held-out sample per fit

    Returns:
        (K,) decision values; scikit-learn's predict_proba is expit() of these
    """
    m0, m1 = train_low.shape[1], train_high.shape[1]
    priors = np.array([m0, m1], dtype=float) / (m0 + m1)
    terms = [_lda_class_terms(train_low), _lda_class_terms(train_high)]
    diagonal = sum(priors[c] * terms[c][2][:, None] * terms[c][3][:, None] * terms[c][4] ** 2
                   for c in range(2))
    U = np.concatenate([np.sqrt(priors[c] * (1 - terms[c][2]) / (m0, m1)[c])[:, None, None] * terms[c][1]
                        for c in range(2)], axis=1)
    means = np.stack([terms[0][0], terms[1][0]], axis=2)              # (K, p, 2)

    inverse_diagonal = 1.0 / diagonal
    scaled_means = inverse_diagonal[:, :, None] * means
    scaled_U = U * inverse_diagonal[:, None, :]
    capacitance = np.eye(U.shape[1])[None] + scaled_U @ U.transpose(0, 2, 1)
    coef = scaled_means - scaled_U.transpose(0, 2, 1) @ np.linalg.solve(capacitance, U @ scaled_means)

    intercept = -0.5 * np.einsum("kpc,kpc->kc", means, coef) + np.log(priors)[None, :]
    return (np.einsum("kp,kp->k", test, coef[:, :, 1] - coef[:, :, 0])
            + intercept[:, 1] - intercept[:, 0])


def lda_loo_decisions(X, y):
    """
    Leave-one-out decision values of LDA(solver='lsqr', shrinkage='auto').

    Args:
        X: (K, n, p) features of K data sets that share the labels y
        y: (n,) labels, 0 = low pain, 1 = high pain

    Returns:
        (K, n) held-out decision value of every sample in every data set
    """
    n_sets, n = X.shape[0], X.shape[1]
    decisions = np.empty((n_sets, n))
    for i in range(n):
        train = np.arange(n) != i
        decisions[:, i] = _lda_decision(X[:, train & (y == 0)], X[:, train & (y == 1)], X[:, i])
    return decisions


def lda_loo_aucs(X, y):
    """Leave-one-out AUC, rounded to 3 decimals as in run_lda_models(), for each of K data sets."""
    probabilities = expit(lda_loo_decisions(X, y))
    return np.array([np.round(roc_auc_score(y, probabilities[k]), 3)
                     for k in range(X.shape[0])])


# Data each worker process scores permutations against (set by _init_null_worker).
_NULL_FEATURES = None
_NULL_LABELS = None


def _init_null_worker(feature_list, label_list):
    global _NULL_FEATURES, _NULL_LABELS
    _NULL_FEATURES, _NULL_LABELS = feature_list, label_list
    # Shuffled feature rows make pandas warn on every boolean selection.
    warnings.filterwarnings("ignore", category=UserWarning)


def _lasso_null_task(orders):
    """Null R² of every LASSO model for each permutation in orders."""
    scores = []
    for order in orders:
        shuffled = [features.iloc[idx, :] for features, idx in zip(_NULL_FEATURES, order)]
        with contextlib.redirect_stdout(io.StringIO()):
            _, r2s = run_lasso_models(shuffled, _NULL_LABELS, SIZE_LIST, PAIN_METRICS, save=False)
        scores.append(r2s)
    return np.array(scores, dtype=float)


def _lda_null_task(orders):
    """Null AUC of every LDA model for each permutation in orders."""
    scores = np.full((len(orders), len(SIZE_LIST) * len(PAIN_METRICS)), np.nan)
    column = 0
    for window, (features, labels) in enumerate(zip(_NULL_FEATURES, _NULL_LABELS)):
        for metric in PAIN_METRICS:
            # Prepare each permutation exactly as run_lda_models() does, then
            # score together the permutations that share one label vector.
            groups = {}
            for k, order in enumerate(orders):
                with contextlib.redirect_stdout(io.StringIO()):
                    X, y, n_low, n_high, success = prepare_classification_data(
                        features.iloc[order[window], :].copy(), labels.copy(), metric)
                if not success or min(n_low, n_high) < MIN_SURVEYS_PER_CLASS:
                    continue
                key = y.tobytes()
                groups.setdefault(key, (y, [], []))
                groups[key][1].append(k)
                groups[key][2].append(X)
            for y, rows, matrices in groups.values():
                scores[rows, column] = lda_loo_aucs(np.stack(matrices).astype(float), y)
            column += 1
    return scores


def run_permutation_null(task, orders, per_task, feature_list, label_list, n_workers, name):
    """
    Score every permutation in orders with task(), in parallel, in order.

    Returns:
        Array of shape (n_permutations, n_models)
    """
    # Smaller tasks when there are few permutations, so every worker gets some.
    per_task = max(1, min(per_task, -(-len(orders) // n_workers)))
    chunks = [orders[i:i + per_task] for i in range(0, len(orders), per_task)]
    results = [None] * len(chunks)
    started = time.time()
    done = 0

    def report():
        elapsed = time.time() - started
        remaining = elapsed / done * (len(orders) - done)
        print(f"  {name} null: {done}/{len(orders)} permutations, "
              f"{elapsed / 60:.1f} min elapsed, about {remaining / 60:.1f} min left")

    if n_workers == 1:
        _init_null_worker(feature_list, label_list)
        for j, chunk in enumerate(chunks):
            results[j] = task(chunk)
            done += len(chunk)
            report()
    else:
        # "spawn" starts each worker fresh on every platform, so each picks up
        # the one-thread setting at the top of this script.
        with ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_null_worker,
                                 initargs=(feature_list, label_list)) as pool:
            futures = {pool.submit(task, chunk): j for j, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                j = futures[future]
                results[j] = future.result()
                done += len(chunks[j])
                report()
    return np.vstack(results)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Print each line as it happens, also when output goes to a log file.
    sys.stdout.reconfigure(line_buffering=True)
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
    # STEP 3: Permutation null for LASSO
    # -------------------------------------------------------------------------
    # The LASSO orders are drawn here and the LDA orders in step 5, in the
    # order a serial loop would draw them.
    lasso_orders = draw_permutation_orders(feature_list, N_PERMUTATIONS)
    if N_PERMUTATIONS > 0:
        print(f"\n[STEP 3] LASSO permutation null: {N_PERMUTATIONS} permutations "
              f"on {N_WORKERS} worker processes...")
        boots_r2s = run_permutation_null(
            _lasso_null_task, lasso_orders, LASSO_PERMUTATIONS_PER_TASK,
            feature_list, label_list, N_WORKERS, "LASSO")
        np.save(RESULTS_DIR + 'lasso_boots_r2s.npy', boots_r2s)
        print(f"  Saved permutation R² distributions to: {RESULTS_DIR}lasso_boots_r2s.npy")
    else:
        print("\n[STEP 3] Skipped (--permutations 0); keeping the existing lasso_boots_r2s.npy")

    # -------------------------------------------------------------------------
    # STEP 4: Run LDA classification
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Running LDA classification models...")
    all_coefs_lda, all_aucs = run_lda_models(
        feature_list, label_list, SIZE_LIST, PAIN_METRICS)

    # -------------------------------------------------------------------------
    # STEP 5: Permutation null for LDA
    # -------------------------------------------------------------------------
    lda_orders = draw_permutation_orders(feature_list, N_PERMUTATIONS)
    if N_PERMUTATIONS > 0:
        print(f"\n[STEP 5] LDA permutation null: {N_PERMUTATIONS} permutations "
              f"on {N_WORKERS} worker processes...")
        boots_aucs = run_permutation_null(
            _lda_null_task, lda_orders, LDA_PERMUTATIONS_PER_TASK,
            feature_list, label_list, N_WORKERS, "LDA")
        np.save(RESULTS_DIR + 'lda_boots_aucs.npy', boots_aucs)
        print(f"  Saved permutation AUC distributions to: {RESULTS_DIR}lda_boots_aucs.npy")
    else:
        print("\n[STEP 5] Skipped (--permutations 0); keeping the existing lda_boots_aucs.npy")

    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------
    print(f"\n{'#'*60}")
    print("  PIPELINE COMPLETE")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'#'*60}\n")
