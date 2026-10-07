# Pain decoding pipeline (Supplementary Figures S1, S2, S3)

This folder holds the commented pain-decoding analysis scripts, numbered in the
order they must be run.

## Overview

The pipeline analyzes neural features from the temporary sEEG inpatient trial to
predict pain scores in the five participants (P1–P5).

**Analysis methods:**
- LASSO regression: predicts continuous pain scores
- LDA (linear discriminant analysis) classification: classifies high vs low pain states

## Which script makes which figure

| Script | Figure in the manuscript |
|---|---|
| `03_plot_significance_heatmaps.py` | **Supplementary Figure S1** — pain decoding performance, temporary sEEG inpatient trial |
| `06_plot_sample_counts.py` | **Supplementary Figure S2** — sample counts used for sEEG pain decoding |
| `05_plot_hemisphere_comparison.py` | **Supplementary Figure S3** — feature prominence per participant |
| `04_plot_freq_coefficients.py` | no numbered figure — frequency-band count plots kept as a supporting analysis |
| `01_run_ml_pipeline.py`, `02_extract_significant_models.py` | no figure — these fit the models and select the significant ones |

## Script execution order

Scripts 03, 05 and 06 run from the shipped results; 02 and 04 need the
coefficient pickles written by 01, which are not shipped.

```
01_run_ml_pipeline.py
   ├──▶ 02_extract_significant_models.py
   │       ├──▶ 03_plot_significance_heatmaps.py
   │       └──▶ 05_plot_hemisphere_comparison.py
   ├──▶ 04_plot_freq_coefficients.py
   └──▶ 06_plot_sample_counts.py
```

## Script descriptions

### 01_run_ml_pipeline.py
**Fits the decoding models**

Runs LASSO regression and LDA classification with:
- Leave-one-out cross-validation
- Permutation testing (feature rows shuffled against the labels, 1000 permutations)

**Inputs:** `neural_features/P{n}/*.csv` (neural features and pain labels;
override the root with the `DECODING_FEATURES_ROOT` environment variable)
**Outputs:** into `decoding_results/P{n}/` — `lasso_*.npy`, `lda_*.npy`, the
coefficient pickles (`lasso_coef_df.pkl`, `lda_coef_df.pkl`), `count_dict.pkl`, and the
sample counts `lda_num_samples_per_model.csv` and `regression_num_sample_per_model.csv`.
01 also fits `mood_vas_s0`; it is not used by any figure. The coefficient pickles
are not shipped, so 02 and 04 run only after 01.

---

### 02_extract_significant_models.py
**Identifies statistically significant models**

A model is significant when its score exceeds the 95th percentile of its permutation null.
Computes p-values and feature-importance counts per electrode.

**Inputs:** coefficient pickles and bootstrap distributions from step 01
**Outputs:** `*_sig_*.npy`, `*_all_p_vals.npy`, `*_counts_ch_ordered.csv`

---

### 03_plot_significance_heatmaps.py
**Supplementary Figure S1 — AUC and squared-correlation (R²) heatmaps**

Draws the decoding-performance heatmaps (pain metric x time window) with
significance stars (*, **, ***).

**Inputs:** significance arrays and p-values from step 02
**Outputs:** into `decoding_results/P{n}/figs/` —
`P{n}_AUC_sig_models_stars.pdf`, `P{n}_R2_sig_models_stars.pdf`

---

### 04_plot_freq_coefficients.py
**Frequency-band counts from the significant models**

Counts how often each frequency band (delta, theta, alpha, beta, low gamma, high
gamma) appears among the top features of the significant models, and draws those
counts per participant and pooled across participants. This is a supporting
analysis; it is not one of the numbered supplementary figures.

**Inputs:** coefficient pickles and bootstrap distributions from step 01
**Outputs:** into `decoding_results/P{n}/frequency_coefficient_figs/` —
`P{n}_{lda,lasso,combined}_freq_counts.{pdf,png,csv}`, plus pooled figures in
`combined_figures/` (set `COMBINED_FIG_DIR` to redirect)

---

### 05_plot_hemisphere_comparison.py
**Supplementary Figure S3 — left vs right hemisphere feature prominence**

Draws side-by-side bar plots comparing summed feature counts between
hemispheres, grouped by anatomical region.

**Inputs:** channel-ordered counts from step 02
**Outputs:** into `decoding_results/P{n}/figs/final_figs/` and
`decoding_results/combined_results/` —
`P{n}_per_hem_anat_features_{LDA,LASSO,Combined}_norm.pdf`,
`P{n}_summed_counts_df.csv`

---

### 06_plot_sample_counts.py
**Supplementary Figure S2 — sample-count tables**

Draws grid tables of the number of samples per model, for LDA and for LASSO.

**Inputs:** the per-participant sample-count CSVs collected in
`decoding_results/<participant>/`
**Outputs:** into `decoding_results/combined_results/num_samps/` —
`P{n}_LDA_sample_counts.pdf`, `P{n}_REG_sample_counts.pdf`

---

## Configuration

Each script has a configuration section at the top where you can change:
- `PARTICIPANT_LABEL` or `ALL_PARTICIPANTS`: which participant(s) to process (e.g. `"P3"`)
- directory paths (or set `DECODING_FEATURES_ROOT` / `DECODING_RESULTS_ROOT`)
- analysis parameters (significance threshold, number of top features, and so on)

## Dependencies

- numpy, pandas, matplotlib, seaborn
- scipy, scikit-learn
- `bm_helper_functions.py` — the utility module in this folder (output-folder
  creation and the channel/anatomy label lookup)

## Directory structure

Directories marked "created on run" do not exist until the corresponding script
writes them; everything else is present with its precomputed contents.

```
Shirvalkar_2026_NatMed_deposit/
├── neural_features/                   # INPUT: per-participant feature and label CSVs
│   └── P{n}/                          # P1..P5
└── python_code/
    └── decoding_figures/              # this folder
        ├── 01_run_ml_pipeline.py .. 06_plot_sample_counts.py
        ├── bm_helper_functions.py     # shared utility functions
        ├── channel_anatomy_labels.json  # channel -> anatomy lookup per participant
        ├── combined_figures/          # step 04 pooled figures (created on run)
        └── decoding_results/          # OUTPUT (precomputed)
            ├── P{n}/                  # step 01-02 arrays and counts
            │   ├── figs/              # step 03 heatmaps
            │   │   └── final_figs/    # step 05 hemisphere figures
            │   └── frequency_coefficient_figs/   # step 04 figures (created on run)
            └── combined_results/      # cross-participant summaries
                └── num_samps/         # step 06 sample-count tables
```
