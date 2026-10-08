# Data dictionary

Field-level description of every processed data file in this repository, for
**Personalized closed-loop deep brain stimulation for chronic pain: a randomized,
double-blind crossover pilot trial** (Shirvalkar et al.). Participants are labelled **P1–P6**
throughout; no device serial numbers or other identifying information are used.
Pain-scale abbreviations retained as data labels: NRS (numeric rating scale),
VAS (visual analog scale), MPQ (McGill Pain Questionnaire).

Tables are keyed by the de-identified `participant` label (P1–P6); where a
hemisphere must be distinguished, a `participant_hemisphere` column gives the
label with a left/right suffix (for example `P1R`). No device serial or model
codes appear in any shipped file.

Figure and table numbers here are the published numbering used in the article.

---

## Figure data (`data/`)

### Figure 1 timeline
Main Figure 1 (P3 pain-intensity VAS) is drawn by the same plotter as Extended
Data Figure 2 (`plot_pain_timecourse.m`) and reads the authoritative per-survey
table `extended_figures_raw_pain/pain_surveys.csv` (documented below), not a
separate spreadsheet.

### figure2_inpatient_analgesia.xlsx
Inpatient acute analgesia by stimulation target (feeds
`plot_figure2_inpatient_analgesia.m`); one sheet per participant, `P1` through
`P6`.
- `target` — stimulated brain region (bipolar contact site), or `PRETRIAL` / `SHAM` for the baseline and sham trials. Some sheets spell a region differently from the manuscript (for example `LeftCaudate` for L Caudate, `RightCMThal` for R CMThal, `R VPN` for R VPThal); the plotting function maps each to its manuscript name.
- `change_painVAS` — change in pain VAS from that stimulation's own pre-stimulation baseline (negative = analgesia).

### figure3_processed_inpatient_pain_decoding/
Processed inpatient pain-decoding results for Figure 3 b–d, nine spreadsheets per
participant for P1–P5 (45 files). Read by
`plot_figure3bcd_inpatient_pain_decoding.ipynb`.
`process_figure3bcd_inpatient_pain_decoding.ipynb` adds significance columns to the
two `_across_models.xlsx` files and writes `_inpatient_feature_counts.xlsx`; the other
six files per participant are provided as data.

- `P#_lasso_across_models.xlsx`, `P#_lasso_bootstrap_distribution.xlsx`,
  `P#_lasso_predicted_values_per_model.xlsx` — LASSO regression performance
  across candidate models, its bootstrap null distribution, and the per-model
  predicted values (Figure 3b).
- `P#_lda_across_models.xlsx`, `P#_lda_bootstrap_distribution.xlsx`,
  `P#_LDA_ROCs_per_model.xlsx` — the same three products for the
  linear-discriminant classifier, including the per-model ROC curves (Figure 3c).
- `P#_feature_names.xlsx`, `P#_electrode_labels.xlsx` — the feature and recording
  contact labels that index the model outputs.
- `P#_inpatient_feature_counts.xlsx` — decoding feature counts by anatomy;
  Figure 3d shows P3.
  - `anatomy` — sensing region.
  - `count_lda`, `count_lasso`, `count_combined` — number of selected decoding features from the linear-discriminant model, the LASSO model, and their combination.

### figure4_adaptive_dbs_performance.xlsx
Adaptive-DBS performance traces for participant P3's left hemisphere (main
Figure 4 d/e; feeds `plot_figure4_adaptive_dbs.m`).
- `[aDBS_spon_tbl]` `time`, `on_off`, `state`, `ampInMilliamps` — timestamp, therapy on/off, adaptive detector state, and delivered stimulation amplitude (mA). Drives panel e (effective duty cycle and state raster).
- `[LD_tbl]` `time`, `LD_output` — linear-discriminant detector output over time. Drives panel d (control signal with its two decision thresholds).

### figure4_processed_ambulatory_decoding/
Processed ambulatory pain-decoding data for the Figure 4 a–c example participant
(P1 only): `P1_neural_spectra.xlsx` (survey-locked power spectra, rows =
surveys, columns = frequency in Hz), `P1_pain_surveys.xlsx` (every survey paired
with a recording, plotted in panel a, with `time`, `MPQ`, `VAS` and
`in_analysis`; the spectra rows are the `in_analysis` surveys, in the same
order, which remain after outlier removal and are used in panels b–c), and `P1_pain_decoding_AUC.xlsx`
(leave-one-out decoding area under the ROC curve per frequency bin, high versus
low pain on the 724 analysis surveys at or above the 70th or at or below the 30th
MPQ percentile). Only
P1 ships here because Figure 4 a–c shows P1 as the worked example; the ambulatory
decoding inputs for the other participants are provided in a different form under
`neural_features/` and `neural_data/ambulatory_stages123/` (which drive
Supplementary Figures S8 and S4), not in this folder — so the absence of P2–P5
files here is by design, not missing data. `P1_pain_decoding_AUC.xlsx` is written by
`process_figure4_ambulatory_pain_decoding.ipynb` from the other two files.

### figureS6_battery_efficiency/ (CSV extracts)
Device energy accounting for Figure S6. The Python script
`python_code/plot_figureS6_battery_efficiency.py` reads two CSVs from this
folder directly; their fields are listed under "Further figure data extracts"
below. `breakeven_duty.csv` is also the source table behind Supplementary
Table S10.

---

## Further figure data extracts (`data/`)

Each of the figures below reads a de-identified data extract in CSV or HDF5
form, so every script runs from this deposit alone. The extracts are documented
field by field in the sections that follow.

### figure5_double_blind_outcomes/
Ambulatory double-blind closed-loop-versus-sham outcomes (main Figure 5 a–c).
Two products: the per-day percent-change CSV that draws the panels and feeds
the statistics, and a workbook whose `stim_guess` sheet feeds Figure 5f.

**`cohort_daily_percent_change.csv`** — per-participant daily percent-change
values for every ambulatory outcome measure, split by stimulation condition
(feeds `plot_figure5_double_blind_outcomes.m`, which draws the uniform
numeric-rating-scale, index-metric and response-curve panels, and
`compute_figure5_statistics.m`, which fits the closed-loop-versus-sham model).
- `patient_id` — de-identified participant label (P1–P5).
- `stim_group`, `stim_group_long` — stimulation condition: `pre-trial`, `closed-loop`, `sham`, `intermittent-loop`, `continuous-open-loop` or `active control` (short and long labels).
- `baseline_percent_diff_<measure>` — percent change from that participant's pre-trial baseline for each measure: pain numeric-rating and visual-analog scales (`mayoNRS`, `painVAS`, `worstNRS`, `worstVAS`, neuropathic/nociceptive `np*`/`noc*`, regional `left*NRS`), the McGill Pain Questionnaire total (`MPQtotal`), Fitbit step total (`FB_StepTotal`), and mood/function instruments (`HamD6`, `PROMISPainInterference`, `PCGIS`).

Each participant's prospective index metric is not a column in this file; it is
declared in `plot_figure5_double_blind_outcomes.m`, which names the
`baseline_percent_diff_*` column to use for each of P1–P5.

**`double_blind_outcomes.xlsx`** — summary workbook with three sheets; `daily` and
`stim_guess` are read by the code:
- `[summary]` `stim_group`, `mean_pain`, `SEM_pain_sem` — group mean pain and its standard error. Not read by the code.
- `[daily]` `pt_id`, `stim_group`, `pain`, `steps`, `duty cycle` — per-participant daily pain, Fitbit step count, and closed-loop duty cycle, by stimulation group (closed-loop, sham, and 46 P3 washout (0 mA) rows). `steps` is the percent change in that day's Fitbit step count from the participant's pre-trial baseline; it has values for P1, P2 and P4 only (205 closed-loop or sham days). Only `steps` is read, by `python_code/plot_extended_figure4_ancillary.py` (Extended Data Figure 4d: a one-sided two-sample t-test pooling the closed-loop and sham days of all three participants, and per-participant contrasts); the Figure 5 model reads `cohort_daily_percent_change.csv`.
- `[stim_guess]` `pt_id`, `ground_truth`, `pt_guess`, `percent_guessed` — how often each participant guessed each condition during the blinded period. Plotted by `python_code/plot_figure5f_blinding_guess.py` (Figure 5f), which also runs the blinding tests.

### figureS7_acute_predicts_chronic/
Paired acute-versus-chronic percent changes tested in Figure S7 (feeds
`plot_figureS7_acute_predicts_chronic.m`, which computes all statistics — rank
correlation, permutation p, bias-corrected bootstrap interval, and jackknives —
from these tables). The two single-scale files share the same columns;
`_hemisphere` is per implanted hemisphere, `_participant` is pooled per
participant.
- `patient_id` — de-identified participant label (P1–P5); `hemisphere`, `target` (hemisphere file only).
- `index_metric` — the participant's prospective index pain scale.
- `acute_pct_NRS`, `acute_pct_VAS` — acute (temporary-trial) percent change at the implanted target, numeric-rating and visual-analog.
- `chronic_pct_NRS`, `chronic_pct_index` — chronic closed-loop percent change, numeric-rating and index metric (negative = benefit).

The two `_pooled_` files stack the three collected pain scales into one long
table, one row per unit and scale, so the acute predictor spans a wider range of
values. They drive the second series on the jackknife panels (c–e).
- `patient_id`, `hemisphere`, `target` — as above (`hemisphere`, `target` in the hemisphere file only).
- `pain_scale` — `mayoNRS`, `painVAS` or `MPQtotal`.
- `acute_pct`, `chronic_pct` — acute and chronic percent change for that scale (negative = benefit).
- `n_days` — closed-loop participant-days contributing to `chronic_pct` (hemisphere file only).

### figureS11_programming_history/
Full device programming history per implanted hemisphere (feeds
`plot_figureS11_programming_history.m`). Two per-hemisphere files and one
per-participant file:
- `<P><side>_stim_parameters.csv` — one row per settings interval: `datetime`, `activeGroup` (A–D; blank on the first row of P2L and of P2R, where therapy is off), `therapyStatus`, `is_adaptive` (closed-loop flag), `amplitude_mA`, `pulse_width_us`, `frequency_Hz`, `contacts`, `effectively_active`, `cathode`, `anode`, `polarity`, `lead`, `duty_prog` (programmed duty cycle).
- `<P><side>_daily_duty_cycle.csv` — `day`, `percentDutyCycle` (daily realized closed-loop duty cycle).
- `<P>_sham_windows.csv` — `sham_start`, `sham_stop` (blinded sham-arm intervals
  for the participant). Present for P1, P2, P4 and P5. P3 did not undergo sham
  testing, so no file exists for that participant and no sham episodes are
  marked on that participant's panel.

Nine hemispheres carry stimulation parameters (P1R; P2L/R; P3L/R; P4L/R; P5L/R)
and eight of those also carry a daily duty-cycle file — P4L has no duty-cycle
record. Sham-window files are present for P1, P2, P4 and P5.

These same three file types are also read by
`python_code/plot_figureS5_closed_loop_duty_cycle.py`, which uses the interval
table to decide which days the device genuinely spent in closed-loop mode
before summarising the measured duty cycle on those days.

### stimulation_parameter_summary/
Written by `python_code/summarize_stimulation_parameters.py` from the
`figureS11_programming_history/*_stim_parameters.csv` files; one row per hemisphere.
- `parameter_dispersion.csv` — `hemisphere`, `records`, `distinct_settings`, and for each of amplitude (`amplitude_mA_*`), pulse width (`pulse_width_us_*`) and frequency (`frequency_Hz_*`): `median`, `lower_quartile`, `upper_quartile`, `coefficient_of_variation_percent`.
- `amplitude_time_trends.csv` — `hemisphere`, `years_observed`, `records`, `slope_mA_per_year`, `confidence_interval_low`, `confidence_interval_high`, `p_value`.

### figureS5_closed_loop_duty_cycle/
Day-to-day variability of the closed-loop effective duty cycle, written by
`python_code/plot_figureS5_closed_loop_duty_cycle.py` from the programming
history above. `blinded_daily_duty_cycle.csv` is an input to that script; the
other three files are its outputs.
- `blinded_daily_duty_cycle.csv` — panel a. One row per participant per closed-loop day of the blinded window that has a duty-cycle record (P3 has 28 such days and P4 37, against 29 and 49 closed-loop days in the pain analysis): `participant`, `day`, `duty_cycle_percent` (hemispheres averaged within the day).
- `daily_duty_cycle_closed_loop.csv` — one row per participant per closed-loop day: `participant`, `response_group` (Responder / Non-responder), `day`, `duty_cycle_percent` (hemispheres averaged within the day), `days_since_start` (days elapsed from that participant's first closed-loop day). Blinded sham days are excluded, and a day is included only if more than 90% of it was spent in adaptive mode with therapy on and a contact active.
- `duty_cycle_dispersion_summary.csv` — one row per participant: `days`, `median`, `mean`, `standard_deviation`, `interquartile_range`, `first_quartile`, `third_quartile`, `coefficient_of_variation_percent` (spread relative to that participant's own mean), `minimum`, `maximum`, `trend_rho_daily`/`trend_p_daily` (Spearman correlation of duty cycle against elapsed days), `trend_rho_monthly`/`trend_p_monthly` (same test on monthly medians, months with fewer than five days dropped), `first_3_months_median`, `last_3_months_median`, `change_first_to_last` (percentage points; positive means the device stimulated more as time passed).

- `duty_cycle_group_tests.csv` — one row per statistic, testing whether the duty cycle separates responders from non-responders. `median_gap_percentage_points` and `permutation_p_one_sided`/`_two_sided` come from an exact permutation over all ten ways of splitting five participants three against two; `separation_is_complete` records whether every responder median fell below every non-responder median. `mixed_model_effect_percentage_points` and `mixed_model_standard_error` come from a day-level mixed model with a per-participant random intercept, reported with two p-values: `mixed_model_p_treating_days_as_independent` is what the software prints and is inflated, while `mixed_model_p_between_participants` uses degrees of freedom appropriate to a predictor that varies only between participants. `mixed_model_permutation_p_one_sided`/`_two_sided` refit that model under permuted participant labels. `intraclass_correlation` quantifies how much of the variance is between participants. `firth_odds_ratio_per_10_points` and `firth_p` come from penalized logistic regression, which is used because the groups do not overlap and ordinary logistic regression has no finite solution. `smallest_attainable_p_one_sided`/`_two_sided` state the floor imposed by the sample size.

Responder status is read at run time from
`figures/ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham.csv`:
a responder improved at least 30% from baseline during closed-loop on their own
prospectively chosen index pain scale.

### figureS6_battery_efficiency/
Device energy accounting (feeds `python_code/plot_figureS6_battery_efficiency.py`,
which refits the drain model from these tables).
- `discharge_segments.csv` — one row per battery-discharge segment: `hem`, `patient`, `side`, `t0`/`t1` (segment start/stop), `dur_days`, `n_pts`, `soc_start`/`soc_end` (state-of-charge), `discharge_mAh_per_day`, `teed_J_per_day` (total electrical energy delivered), `eff_duty_frac`, `frac_off`/`frac_open_loop`/`frac_closed_loop`, `streaming_per_day`, `dominant_loop`, `patient_V`, `battery_J_per_day`.
- `breakeven_duty.csv` — per hemisphere: `E_full_Jday`, `therapy_drain_full_mAhday`, `sensing_overhead_mAhday`, `breakeven_cl_duty` (with `breakeven_lo`/`_hi`), `actual_cl_duty`, and a `status` label (whether closed-loop saves energy at the observed duty cycle). This is also the source table behind Supplementary Table S10.

### figureS8_biomarker_stability/

Figure S8 AUC grids, one CSV per implanted device side
(`<DEVICE_SIDE>_auc_grid.csv`: P1R, P2L, P2R, P3L, P3R, P4L, P4R, P5L, P5R),
written by `python_code/stability_and_closed_loop/analyze_ambulatory_biomarker_stability.py`
and drawn as one figure by `plot_figureS8_composite.py`. One row per window x
frequency:
- `months_included` — the window uses every survey from the first this-many months (30-day months) after the first ambulatory survey; steps of 2.
- `center_frequency_hz` — center of the 5 Hz power window.
- `auc` — leave-one-out LDA area under the curve, low vs high SF-MPQv2 pain (30th/70th percentiles of all surveys). Empty when the window has fewer than 2 low- or 2 high-pain surveys.
- `n_low_pain`, `n_high_pain` — surveys in each pain group for that window.

### figure_ancillary_behavioral/

Baseline (Stage 0 research visits) vs chronic on-therapy ancillary behavioral
questionnaire scores. Feeds both `python_code/plot_figure5_ancillary_main.py`
(main Figure 5 d/e) and `python_code/plot_extended_figure4_ancillary.py`
(Extended Data Figure 4 a–c; panel d reads `figure5_double_blind_outcomes/double_blind_outcomes.xlsx`).

This CSV is the raw ancillary-score data (one row per completed survey) and is
the source both plotters read. Baseline rows were assembled from the study's
REDCap survey database (the peri-implant comprehensive survey for SF-36 /
BDI / BAI and the pain-related survey for HamD-6); chronic rows are the
weekly RC+S survey deliveries during the on-therapy period. Both plotters
compute all reported statistics (mixed-effects, construct pooling, MCID
counts) directly from this long CSV at run time.

- `ancillary_scores_long.csv` — 1,909 rows in long format.
  Columns:
  - `patient` (P1–P5)
  - `measure` (sf36_composite | randPhysFxn | randRolePhysical |
    randRoleEmotional | randEnergy | randEmotion | randSocial | randPain |
    randGeneral | BDI | BAI | HamD6). `sf36_composite` = mean of the eight
    SF-36 subscales for that survey (0–100).
  - `phase` (baseline | chronic)
  - `date` (ISO date of the survey)
  - `value` (native scale: SF-36 subscales 0–100 with higher = better;
    BDI/BAI raw sum 0–63 with lower = better; HamD-6 raw sum 0–22 with
    lower = better)

  Baseline rows for SF-36 / BDI / BAI come from the study's REDCap survey
  database (the peri-implant comprehensive survey, one survey per participant
  near implant); baseline HamD-6 rows come from the same database's
  pain-related survey, sampled densely in the peri-implant window. Chronic rows
  are the weekly RC+S survey deliveries during the on-therapy period.

- `mixed_effects_reference.csv` — per-measure mixed-effects results (columns:
  measure, direction, coef, ci_lo, ci_hi, p, n_obs, n_base).
  This file and `construct_pooling_reference.csv` are written by
  `plot_figure5_ancillary_main.py` at the precision printed in the paper
  (estimates and bounds to 1 decimal, construct SDs to 2; p to 3 decimals, 4
  below 0.001, `<0.0001` below that).
- `construct_pooling_reference.csv` — pain / function / mood construct-level
  mixed-effects output (SD units, 95% CI, p).
- `responder_detail_mcid.csv` — per-participant per-measure change and MCID
  flag (`reached_mcid`). This is a reference for checking a recomputation: `plot_extended_figure4_ancillary.py` recounts
  MCID responders from the long CSV rather than reading this file.
- `responders_mcid_summary.csv` — per-measure count of MCID responders; the same
  reference role.

Responder status is distributed as MCID counts only. The pre-implant baseline
sample is too small to support a stable reliable-change index.

### extended_figures_raw_pain/

Raw, per-survey pain scores behind the two raw-data Extended Data figures, which
show the full distribution and full time course of every collected pain measure
for every participant.

All values are the raw scores as collected (no averaging, no percent-change).
Both plotters compute their statistics from these CSVs at run time; the shipped
statistics CSVs are a reference the plotters regenerate.

Only P-labels appear in this table; no device serial codes. Dates are shifted by
a fixed per-participant offset (see the README de-identification note), which
preserves every interval.

Condition labels are taken from the trial's own stimulation-group logs, not
assigned ad hoc. A survey is labelled **sham** if its calendar day is one of the
curated `sham|BLIND||` days in the trial's blinded-groups log, **washout** if
stimulation is off while unblinded during Stages 1–2, **baseline** before
therapy begins, and **closed-loop** or **open-loop** according to the active
stimulation mode at that time. Where the curated stimulation-group record does
not cover a survey's day, the state is taken from the per-survey device
timeline, so every survey carries a condition.
The capitalised forms that appear in figure legends (`Sham (0 mA)`,
`Washout (0 mA)`, `Open-loop`) are display labels applied in the plotting code;
the stored values are lower-case.

The number of blinded-sham surveys differs per participant, straight from each
participant's log (P1 3, P2 73, P4 9, P5 4 NRS surveys). Participant P5 provided
pain-VAS on only one of the participant's sham days, so that participant's sham is represented on the
NRS-type measures. P3 did not undergo sham testing and so is absent from
that list; no survey of P3's is classified **sham**, and every 0 mA survey of
P3's recorded after therapy began is classified **washout**.

- `pain_surveys.csv` — per-survey pain table, **one row per survey** (11,986
  rows), and the single pain data file for the whole deposit: main Figure 1,
  Extended Data Figure 2, Extended Data Figure 3 and Table S11 all read it. Each pain scale is its own column, so a
  plotter selects the column it needs and drops the blanks; nothing is stored in
  long form and no derived or extracted copy is written anywhere. Columns:
  - `participant` (P1–P5)
  - `time` — timestamp, study-site local, no zone offset (relative, date-shifted for
    de-identification; see the de-identification note)
  - `condition` — the physical stimulation state for that survey:
    `pre-implant`, `stage0 inpatient`, `closed-loop`, `open-loop`, `sham`, or
    `washout`. Values are lower-case; the capitalised forms that appear on
    figures are display labels applied in the plotting code.
  - `stim_group_raw` — the stimulation-group string as recorded in the trial log,
    before it is collapsed onto `condition`
  - `stage` (`pre-trial` | `stage0` | `stage1` | `stage2` | `stage3` |
    `study graduation` | `unknown`)
  - `is_baseline` (0/1) — the participant's prespecified reference window. This
    is a flag rather than a `condition` value because it is not one stimulation
    state: for P1–P4 it selects part of the 0 mA period preceding stimulation,
    and for P5 the pre-operative study. A figure drawing baseline as its own arm
    must therefore exclude these surveys from the washout arm.
  - `is_published_closed_loop` (0/1) — the curated closed-loop window reported in
    Extended Data Table 2. A subset of `condition == closed-loop`, so the two
    together separate the double-blind window from optimisation.
  - `in_published_analysis` (0/1) — whether the survey was carried in the
    published analysis set
  - `blinded` (1 = survey collected while blinded; 0 otherwise)
  - `stimulation_contact`, `duty_cycle_percent`, `frequency_hz`, `amplitude_ma`,
    `pulse_width_us` — stimulation settings in force, populated only for surveys
    with an active programme on record
  - `mayoNRS` (0–10), `painVAS` (0–100), `unpleasantVAS` (0–100), `MPQtotal`
    (MPQ total, 0–45), `nocNRS` (0–10) — the raw scores as collected
  - `baseline_percent_diff_<scale>` for each of the five scales — percent change
    from that participant's own baseline

  Study phase is **not** a column: it is derived where it is needed. Extended
  Data Figure 2 derives it from `stage` (`pre-trial` → pre-operative, `stage0` →
  Stage 0 inpatient, everything else → chronic) and reports the result at run
  time. Note that P1–P4's baseline surveys carry stage labels from Stages 1–2
  while every one of P5's is `pre-trial`; `is_baseline` marks both.

  Main Figure 1 and Extended Data Figure 2
  (`matlab_code/plot_pain_timecourse.m`) plot the full time course from this
  file and are measure-selectable via the second argument, which names the
  scale column to read (`mayoNRS`, `painVAS`, `unpleasantVAS` or `MPQtotal`),
  defaulting to `painVAS`. Surveys with no value in the selected
  column are dropped. Survey times and the boundaries in
  `longitudinal_phase_metadata.csv` are both study-site local without a zone, so
  they compare directly.

- `longitudinal_phase_metadata.csv` — one row per participant giving the
  double-blind crossover (Stage 3) boundaries used by main Figure 1 and
  Extended Data Figure 2:
  `stage1_implant_date`; `stage3_onset_datetime`, the instant the double-blind
  crossover begins, which is also where the blue crossover block starts;
  `endpoint_datetime`, where each timeline is cut so every participant has a
  defined end; and `endpoint_rule` (P1 = 3.5 years after Stage 1 implant;
  P2–P4 = Stage 3 end; P5 = last survey, because that participant's Stage 3 discontinuation is
  followed by a single-blinded phase that continues to the end of follow-up).
  Both boundaries are absolute timestamps on the same study-site local clock as
  `time` in `pain_surveys.csv`, and neither carries a zone offset, so they
  compare directly with no conversion. They are stored
  to the second rather than the day because two participants' Stage 3 onsets
  fall mid-afternoon.
- `sham_days.csv` — audit companion, one row per blinded-sham calendar day per
  participant: `participant`, `sham_date` (date-shifted), `condition`. Lists
  exactly which days the authoritative sham labelling picked, so the sham
  assignment in the main CSV can be checked. P2 includes two sham-window days
  with no survey (2175-01-23, 2175-01-28), so the file has 60 days while 58
  carry a survey.

- `ExtFig3_omnibus_condition_stats.csv` — reference statistics for Extended Data
  Figure 3, per participant × measure: Kruskal–Wallis test across every arm with
  at least three surveys (arms listed in `conditions`): `n_total`,
  `KruskalWallis_H`, `df`, `p_omnibus`. Regenerated by the plotter at run time.

- `ExtFig3_pairwise_condition_stats.csv` — the two directional post-hoc tests
  per participant × measure: **closed-loop < baseline** and **closed-loop <
  sham**, each a one-tailed Mann–Whitney U test (a one-tailed test is used
  because the clinical hypothesis is directional), Holm-corrected within each
  family. Columns: `comparison`, `MannWhitney_U`, `p_raw`, `p_holm`, `n_test`,
  `n_ref`, `median_test`, `median_ref`. Regenerated by the plotter at run time.

- `ExtFig3_arm_counts_medians.csv` — per participant, arm and measure:
  `n_surveys`, `n_days`, `median`, `q25`, `q75`. Regenerated by the plotter at
  run time.

The Word version of these statistics, laid out as it appears in the article's
supplement, is `figures/TableS11_extended_data_figure3_statistics.docx`
(Supplementary Table S11).

### Ambulatory biomarker features
The per-survey ambulatory extracts for Figures S8 and S4 are documented under
"Neural data (`neural_data/`)" below, with field-level descriptions and
per-hemisphere counts.

---

## Data tables (`data/supplementary_tables/`)

Eight spreadsheets: Supplementary Tables S2–S9.

### TableS2 — inpatient frequencies for PAG/PVG and thalamic targets
- `participant`, `target_category`, `brain_region` — target grouping.
- `frequencies_hertz` — set of stimulation frequencies tested.
- `n_frequencies`, `n_stim_epochs` — number of distinct frequencies and stimulation epochs.

### TableS3–TableS8 — inpatient stimulation parameters (P1–P6)
One table per participant, in order: `TableS3_P1`, `TableS4_P2`, `TableS5_P3`,
`TableS6_P4`, `TableS7_P5`, `TableS8_P6`. All six share these columns.
- `brain_region` — stimulated region.
- `contact_pair_anode_cathode` — bipolar contact pair (anode +, cathode −).
- `frequency_hertz`, `pulse_width_microseconds`, `amplitude_milliamps`, `duration_seconds` — stimulation parameters and delivery duration.

### TableS9 — optimal ambulatory sensing biomarkers
- `participant`, `sensing_region` — implanted hemisphere (e.g. `P1 (R)`) and sensing region.
- `biomarker_band_low_hertz`, `biomarker_band_high_hertz`, `biomarker_center_hertz` — biomarker frequency band.
- `canonical_band` — canonical band name (theta/alpha/beta/…).
- `closed_loop_use` — how the biomarker was used for closed-loop control.

### Tables shipped outside this folder
- **Extended Data Table 2** (all pain scales, closed-loop versus sham) — `figures/ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham.csv`, built by `matlab_code/build_extended_data_table2.m` from `data/extended_data_table2/blinded_window_pain_by_day.csv`, documented above; the formatted `.docx` beside it is rendered from that CSV by `python_code/export_extended_data_table2_word.py`. Columns: `participant`, `pain_scale`, `prospective_index_metric`, `baseline_median_iqr_n_days`, `closed_loop_median_iqr_n_days`, `closed_loop_percent_change_from_baseline`, `sham_median_iqr_n_days`, `sham_percent_change_from_baseline`, `responder_30pct_closed_loop`, `responder_50pct_closed_loop`. Every parenthetical count is a number of participant-days.
- **Supplementary Table S10** (marginal battery drainage and break-even duty cycle) — no spreadsheet is shipped; the per-hemisphere source table is `data/figureS6_battery_efficiency/breakeven_duty.csv`, documented above.
- **Supplementary Table S11** (full Extended Data Figure 3 statistics) — `figures/TableS11_extended_data_figure3_statistics.docx`, with the two source CSVs in `data/extended_figures_raw_pain/`.
- **Extended Data Table 1** (closed-loop algorithm settings per hemisphere: stimulation parameters, sensing bands, detector thresholds and durations) — `figures/ExtendedDataTable1_closed_loop_algorithm_settings.docx` and `.csv`, with a rendered preview in `ExtendedDataTable1_preview.png`.
- **Supplementary Table S12** (sensitivity analyses of the closed-loop versus sham model) — `figures/TableS12_sensitivity_analyses.csv`, written by `python_code/build_tableS12_sensitivity_analyses.py` from `data/figure5_double_blind_outcomes/cohort_daily_percent_change.csv`. Columns: `analysis`, `estimate_percent`, `standard_error`, `p_value`, `observations`, `closed_loop_days`, `sham_days`.
- **Supplementary Table S1** and **Table 1** are not shipped.

---

## Neural data (`neural_data/`)

Ambulatory biomarker extracts live under `neural_data/ambulatory_stages123/`.
Each file carries survey-locked spectra with pain scores attached (about 0.5 to
1.5 MB each). The precomputed decoding results under
`python_code/decoding_figures/decoding_results/` ship in full as well, so the
decoding figures can be regenerated without re-fitting any model.

### `neural_data/ambulatory_stages123/`

Survey-locked biomarker extract for Figures S8 and S4, one HDF5 per implanted
device side (`{DEVICE_SIDE}_ambulatory_features.h5`), covering the ambulatory
stages 1–3 of the trial.

**HDF5 datasets**
- `psd` — off-stimulation mean power spectral density, one row per pain survey × 200 frequency bins.
- `frequency_hz` — the 200-point frequency axis (0–99.5 Hz at 0.5 Hz).
- `pain_timestamp_unix` — survey timestamp (Unix seconds); `cluster_row` — index of the matched REDCap survey.
- `pain_<scale>` — the pain score attached to each survey (`MPQtotal`, `mayoNRS`, `painVAS`, `unpleasantVAS`, `worstNRS`, `worstVAS`).
- File attributes record `participant`, `device_side`, `channel_of_interest`, `window_seconds` (300), `n_events`, `n_cache_files`, and a `description` naming the spectral method.

`ambulatory_feature_index.csv` in the same folder lists one row per file with
`device_side`, `participant`, `n_events`, `n_cache_files`, `file`, and `size_mb`.
The nine shipped files span all five chronically implanted participants (P1–P5):

| File | Participant | Events | Source cache files | Size |
|------|-------------|-------:|-------------------:|-----:|
| `P1R_ambulatory_features.h5` | P1 | 610  | 1036 | 0.484 MB |
| `P2L_ambulatory_features.h5` | P2 | 771  | 1107 | 0.619 MB |
| `P2R_ambulatory_features.h5` | P2 | 837  | 1190 | 0.664 MB |
| `P3L_ambulatory_features.h5` | P3 | 1621 | 2128 | 1.235 MB |
| `P3R_ambulatory_features.h5` | P3 | 1666 | 2142 | 1.264 MB |
| `P4L_ambulatory_features.h5` | P4 | 776  | 874  | 0.614 MB |
| `P4R_ambulatory_features.h5` | P4 | 776  | 889  | 0.614 MB |
| `P5L_ambulatory_features.h5` | P5 | 1952 | 2159 | 1.498 MB |
| `P5R_ambulatory_features.h5` | P5 | 1973 | 2132 | 1.512 MB |

(10,982 pain-survey spectra in ~8.5 MB total.)

`python_code/stability_and_closed_loop/P3LThresholdAUC.txt` and
`P4LThresholdAUC.txt` hold the 1000-value permutation null of the pain-feature
AUC that `analyze_ambulatory_biomarker_stability.py` uses for those two sides;
for any other side the script computes a seeded null and saves it beside itself.

---

## Neural spectral features (`neural_features/`)

Narrowband spectral features for **all four window lengths (1, 5, 10, and 30
minutes)**, provided for every participant. These are the direct inputs to the
pain-decoding models.

**How the features were computed.** Each survey-locked recording window was
segmented, stimulation-contaminated and artifactual segments were rejected, and
the power spectral density was averaged within each canonical band per contact.
Only these aggregated band powers are shared; no time-domain signal is included,
since the continuous intracranial recordings contain protected health
information.

**Layout.** One subfolder per participant (`P{n}/`), each containing, for every
window length, a features file and an aligned labels file:
```
neural_features/
  P{n}/
    P{n}_01min_features.csv   P{n}_01min_labels.csv
    P{n}_05min_features.csv   P{n}_05min_labels.csv
    P{n}_10min_features.csv   P{n}_10min_labels.csv
    P{n}_30min_features.csv   P{n}_30min_labels.csv
  feature_file_index.csv
```

**`*_features.csv`** — one row per retained survey; columns are `CONTACT_band`,
the average power in one canonical frequency band on one recording contact. The
six canonical bands are delta, theta, alpha, beta, low-gamma (`lgamma`), and
high-gamma (`hgamma`); a participant with *N* contacts therefore has *N* × 6
feature columns (for example `LACC1_delta`, `LACC1_theta`, …).

**`*_labels.csv`** — one row per retained survey, aligned row-for-row with the
features file. Every column falls into one of the following groups; the files
carry nothing else.

- **Decoding targets** — `nrs_s0`, `intensity_vas_s0`, `unpleasantness_vas_s0`,
  `mood_vas_s0`, `relief_s0`. These five are the `PAIN_METRICS` consumed by
  `01_run_ml_pipeline.py`. A target may be entirely empty where a participant
  did not report that scale (for example `relief_s0` for P5's 30-minute window);
  the column is still present so every file has the same target schema.
- **Join key and survey time** — `record_id`, plus one survey-time column. P1's
  export names it `survey_start_timestamp`; the other participants use
  `scales_vasmpq_timestamp`. The pipeline selects whichever is present.
- **Pain summary scores** — `best_s0`, `worst_s0` (P3–P5).
- **McGill pain descriptors** — the individual descriptor ratings, retained
  because they are pain measurements contemporaneous with each neural window.
  P3–P5 use `<descriptor>_s0`: `throbbing_s0`, `shooting_s0`, `stabbing_s0`,
  `sharp_s0`, `cramping_s0`, `gnawing_s0`, `hot_burning_s0`, `aching_s0`,
  `heavy_s0`, `tender_s0`, `splitting_s0`, `tiring_exhausting_s0`,
  `sickening_s0`, `fearful_s0`, `punishing_cruel_s0`. P1 uses the equivalent
  `arm1_<descriptor>_mpq` names: `arm1_throbbing_mpq`, `arm1_shooting_mpq`,
  `arm1_stabbing_mpq`, `arm1_sharp_mpq`, `arm1_cramping_mpq`,
  `arm1_gnawing_mpq`, `arm1_hotburning_mpq`, `arm1_aching_mpq`,
  `arm1_heavy_mpq`, `arm1_tender_mpq`, `arm1_splitting_mpq`,
  `arm1_tiringexhausting_mpq`, `arm1_sickening_mpq`, `arm1_fearful_mpq`, and
  `arm1_punishing_cruel_s0`. Not every participant reported every descriptor.
- **Regional numeric rating scores** — `nrs_left_arm`, `nrs_left_face`,
  `nrs_left_leg` (P5 only).
- **Stimulation state at the survey** — `stimonoff` (P3–P5), or for P1 the
  finer-grained `stim_amp`, `stim_freq`, `stim_pw`, `stim_region`,
  `stim_pos_contact`, `stim_neg_contact`, `stim_site_id`, `stim_testing`.
- **Condition labels** — `short_vs_long` (P3–P5); `verum_or_sham`,
  `sham_controlled`, `verbal_pain_score` (P1).
- **Window-eligibility QC** — `valid_time_before_survey` (P2–P5), the minutes of
  usable recording preceding the survey; `1_mins_prior`, `5_mins_prior`,
  `10_mins_prior`, `30_mins_prior` (P1), the per-window availability flags.

Questionnaire scores are not carried in these files; they are shipped in
`data/figure_ancillary_behavioral/ancillary_scores_long.csv`, documented above.

**`feature_file_index.csv`** — index of every feature file with `participant`,
`window`, `surveys` (retained-survey count), and `feature_cols` (feature-column
count).

**Why the survey count differs by window length.** A survey is retained for a
given window only when it is preceded by enough continuous, artifact-free
recording of that length. Longer windows are available for fewer surveys:
surveys spaced closer together in time than the window length are dropped, and
windows overlapping stimulation-testing periods or other flagged artifacts are
excluded during featurization. Retained surveys per participant and window
(from `feature_file_index.csv`):

| Participant | 1-min | 5-min | 10-min | 30-min | Feature columns |
|---|---|---|---|---|---|
| P1 | 167 | 132 | 106 | 63 | 306 |
| P2 | 157 | 114 | 86  | 75 | 522 |
| P3 | 215 | 101 | 88  | 79 | 660 |
| P4 | 224 | 96  | 94  | 86 | 588 |
| P5 | 170 | 81  | 66  | 6  | 798 |

These counts are much smaller than the per-device-side `n_events` counts in
`neural_data/ambulatory_stages123/` (for example 132 five-minute surveys for P1
against 610 events in `P1R_ambulatory_features.h5`). The two products are built
by separate pipelines and their survey sets should not be assumed to match; use
each product's own index file (`feature_file_index.csv`,
`ambulatory_feature_index.csv`) for the authoritative counts.

### `data/extended_data_table2/blinded_window_pain_by_day.csv`

Day-level pain observations inside the double-blinded crossover window; the input
to Extended Data Table 2. One row per participant, stimulation condition and
calendar day.

| column | meaning |
|---|---|
| `participant` | P1-P5 |
| `stim_group` | `pre-trial` (baseline), `closed-loop`, or `sham`. P3 has no `sham` rows: P3 has no sham condition |
| `mayoNRS`, `painVAS`, `unpleasantVAS`, `MPQtotal`, `nocNRS` | that day's mean score on each scale; blank where the scale was not collected |
| `baseline_percent_diff_<scale>` | percent change from the participant's pre-trial baseline |

Each row is one participant-day: a day enters a condition only if at least one
report was given, and the value is the mean of that day's reports. Day counts are
therefore smaller than the survey counts in `ExtFig3_*_condition_stats.csv`.
