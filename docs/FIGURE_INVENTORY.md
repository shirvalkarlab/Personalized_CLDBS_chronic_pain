# Shirvalkar 2026 Nature Medicine — Deposit Figure / Table Inventory

_Inventory of the data and code deposit accompanying:_
**"Personalized closed-loop deep brain stimulation for chronic pain: a randomized, double-blind crossover pilot trial"** (Shirvalkar et al., 2026, Nature Medicine).

Cohort: P1–P6 (P1–P5 chronically implanted; P6 inpatient investigational only).

All figure and table numbers in this document are the published numbering used
in the article.

## Participant mapping
| Paper ID | Phase |
|---|---|
| P1 | inpatient + implanted |
| P2 | inpatient + implanted |
| P3 | inpatient + implanted |
| P4 | inpatient + implanted |
| P5 | inpatient + implanted |
| P6 | inpatient only (not implanted) |

Implanted-cohort figures (Fig 4, Fig 5, ambulatory) cover P1–P5. Fig 2 covers P1–P6; the inpatient decoding figures (Fig 3b–d, Fig S1–S3) cover P1–P5.

## Reading the tables
Each row gives the item, the language it is written in, the script that produces
it, and the data that script reads. A row with an em-dash in those columns is a
published item that no script here produces — anatomical renderings and
adverse-event summaries are assembled separately.

## MAIN figures
| ID | Title | Language | Code | Data source |
|---|---|---|---|---|
| Fig 1 | Pain timeline (P3, pain-intensity VAS): pre-op → inpatient → chronic closed-loop | MATLAB | matlab_code/plot_pain_timecourse.m (same plotter as Extended Data Fig 2, called for P3) | data/extended_figures_raw_pain/pain_surveys.csv, longitudinal_phase_metadata.csv |
| Fig 2 | Inpatient analgesia by stimulation region (P1–P6) | MATLAB | matlab_code/plot_figure2_inpatient_analgesia.m | data/figure2_inpatient_analgesia.xlsx (six sheets, P1–P6) |
| Fig 3b | Inpatient pain decoding — LASSO predicted versus reported pain | PYTHON | python_code/plot_figure3bcd_inpatient_pain_decoding.ipynb | data/figure3_processed_inpatient_pain_decoding/ |
| Fig 3c | Inpatient pain decoding — LDA ROC | PYTHON | python_code/plot_figure3bcd_inpatient_pain_decoding.ipynb | data/figure3_processed_inpatient_pain_decoding/ |
| Fig 3d | Inpatient feature prominence (P3) | PYTHON | python_code/plot_figure3bcd_inpatient_pain_decoding.ipynb | data/figure3_processed_inpatient_pain_decoding/P3_inpatient_feature_counts.xlsx |
| Fig 4a | Ambulatory decoding — McGill total vs pain VAS correlation (P1) | PYTHON | python_code/plot_figure4abc_ambulatory_pain_decoding.ipynb | data/figure4_processed_ambulatory_decoding/P1_pain_surveys.xlsx |
| Fig 4b | Ambulatory decoding — neural spectra, high vs low pain, with AUC across frequencies | PYTHON | python_code/plot_figure4abc_ambulatory_pain_decoding.ipynb | data/figure4_processed_ambulatory_decoding/P1_neural_spectra.xlsx, P1_pain_decoding_AUC.xlsx |
| Fig 4c | Ambulatory decoding — 5–10 Hz biomarker band power, high vs low pain | PYTHON | python_code/plot_figure4abc_ambulatory_pain_decoding.ipynb | data/figure4_processed_ambulatory_decoding/P1_neural_spectra.xlsx |
| Fig 4d | Closed-loop control signal example, linear-discriminant output with both decision thresholds (P3 Left) | MATLAB | matlab_code/plot_figure4_adaptive_dbs.m | data/figure4_adaptive_dbs_performance.xlsx (sheet `LD_tbl`) |
| Fig 4e | Effective duty cycle over about six days of continuous recording, with closed-loop state raster (P3 Left) | MATLAB | matlab_code/plot_figure4_adaptive_dbs.m | data/figure4_adaptive_dbs_performance.xlsx (sheet `aDBS_spon_tbl`) |
| Fig 5a | Double-blind outcomes — percent change in pain NRS, closed-loop vs sham (P1–P5) | MATLAB | matlab_code/plot_figure5_double_blind_outcomes.m (+ process_/compute_ for the statistics) | data/figure5_double_blind_outcomes/cohort_daily_percent_change.csv; double_blind_outcomes.xlsx (statistics) |
| Fig 5b | Double-blind outcomes — same contrast on the prespecified personalized index metric | MATLAB | matlab_code/plot_figure5_double_blind_outcomes.m | data/figure5_double_blind_outcomes/cohort_daily_percent_change.csv |
| Fig 5c | Response curve: proportion of closed-loop and sham participant-days reaching each reduction in NRS | MATLAB | matlab_code/plot_figure5_double_blind_outcomes.m | data/figure5_double_blind_outcomes/cohort_daily_percent_change.csv |
| Fig 5d | SF-36 composite, baseline vs chronic | PYTHON | python_code/plot_figure5_ancillary_main.py | data/figure_ancillary_behavioral/ancillary_scores_long.csv |
| Fig 5e | Domain-level effect sizes (pain / function / mood) | PYTHON | python_code/plot_figure5_ancillary_main.py | data/figure_ancillary_behavioral/ancillary_scores_long.csv |
| Fig 5f | Participant blinding guesses by true condition | PYTHON | python_code/plot_figure5f_blinding_guess.py | data/figure5_double_blind_outcomes/double_blind_outcomes.xlsx (sheet `stim_guess`) |

## EXTENDED DATA figures
| ID | Title | Language | Code | Data source |
|---|---|---|---|---|
| Ext Data Fig 1 | CONSORT diagram | — | — | — |
| Ext Data Fig 2 | Longitudinal pain time course, all five implanted participants (Figure-1 broken-axis style: baseline / Stage 0 / chronic panels derived in the plotter from `stage`; condition-coloured dots, thin-plus washouts, hollow-circle shams, per-panel moving average; measure-selectable by scale column, default painVAS) | MATLAB | matlab_code/plot_pain_timecourse.m | data/extended_figures_raw_pain/pain_surveys.csv, longitudinal_phase_metadata.csv |
| Ext Data Fig 3 | Raw pain-score distributions, all participants (violin + raw swarm; NRS, intensity VAS, MPQ total; six stimulation arms, with directional one-tailed statistics on the closed-loop double-blind arm). Also writes a reduced version omitting open-loop optimization | PYTHON | python_code/plot_extended_figure3_raw_distributions.py | data/extended_figures_raw_pain/pain_surveys.csv |
| Ext Data Fig 4 a/b/c | Construct-score schematic (a); symptom domains baseline vs chronic (b); MCID responders by measure (c) | PYTHON | python_code/plot_extended_figure4_ancillary.py | data/figure_ancillary_behavioral/ancillary_scores_long.csv |
| Ext Data Fig 4 d | Percent change in daily step count from baseline, closed-loop vs sham days (P1, P2, P4) | PYTHON | python_code/plot_extended_figure4_ancillary.py | data/figure5_double_blind_outcomes/double_blind_outcomes.xlsx (sheet daily) |

## SUPPLEMENTARY figures
| ID | Title | Language | Code | Data source |
|---|---|---|---|---|
| Fig S1 | Pain decoding performance, temporary sEEG inpatient trial (AUC + R² significance heatmaps) | PYTHON | python_code/decoding_figures/03_plot_significance_heatmaps.py | python_code/decoding_figures/decoding_results/P1–P5/ |
| Fig S2 | Sample counts used for sEEG pain decoding; writes decoding_results/combined_results/num_samps/P#_{LDA,REG}_sample_counts.pdf | PYTHON | python_code/decoding_figures/06_plot_sample_counts.py | python_code/decoding_figures/decoding_results/P1–P5/{lda_num_samples,regression_num_sample}_per_model.csv |
| Fig S3 | Feature prominence per participant, left vs right hemisphere (LDA + LASSO) | PYTHON | python_code/decoding_figures/05_plot_hemisphere_comparison.py | python_code/decoding_figures/decoding_results/P1–P5/ |
| Fig S4 | Closed-loop algorithm and threshold-selection simulation (P3 Left); writes figures/FigureS4_P3L_feature_signal_and_histogram.pdf and figures/FigureS4_P3L_closed_loop_threshold.pdf; the other legend panels are not produced by this script | PYTHON | python_code/stability_and_closed_loop/simulate_closed_loop_threshold_selection.py | neural_data/ambulatory_stages123/P3L_ambulatory_features.h5 |
| Fig S5a | Closed-loop effective duty cycle per participant during the blinded testing window (hemispheres averaged within each day) | PYTHON | python_code/plot_figureS5_closed_loop_duty_cycle.py | data/figureS5_closed_loop_duty_cycle/blinded_daily_duty_cycle.csv |
| Fig S5b | Closed-loop effective duty cycle across the whole unblinded record: per-participant day-to-day spread, responders vs non-responders | PYTHON | python_code/plot_figureS5_closed_loop_duty_cycle.py | data/figureS11_programming_history/ (`P#<side>_stim_parameters.csv`, `P#<side>_daily_duty_cycle.csv`, `P#_sham_windows.csv`); figures/ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham.csv (responder status) |
| Fig S6 | Battery efficiency: closed-loop vs open-loop (panels A–C) | PYTHON | python_code/plot_figureS6_battery_efficiency.py | data/figureS6_battery_efficiency/{discharge_segments,breakeven_duty}.csv |
| Fig S7 | Acute-trial analgesia vs chronic closed-loop benefit | MATLAB | matlab_code/plot_figureS7_acute_predicts_chronic.m | data/figureS7_acute_predicts_chronic/acute_predicts_chronic_{hemisphere,participant}.csv and acute_predicts_chronic_pooled_{hemisphere,participant}.csv |
| Fig S8 | Ambulatory biomarker stability: leave-one-out LDA AUC of each 5 Hz power window (0-50 Hz) on cumulative 2-month windows of surveys, 30th/70th percentile pain thresholds fixed on all surveys; writes figures/FigureS8_<DEVICE_SIDE>.pdf, one heatmap per device side, and data/figureS8_biomarker_stability/<DEVICE_SIDE>_auc_grid.csv | PYTHON | python_code/stability_and_closed_loop/analyze_ambulatory_biomarker_stability.py | neural_data/ambulatory_stages123/*.h5, one per implanted device side (side selected by the `DEVICE_SIDE` variable) |
| Fig S8 a–i | Nine-panel composite of the same heatmaps (P1 Right, P2 Left/Right, P3 Left/Right, P4 Left/Right, P5 Left/Right); writes figures/FigureS8_biomarker_stability.pdf | PYTHON | python_code/stability_and_closed_loop/plot_figureS8_composite.py | data/figureS8_biomarker_stability/*_auc_grid.csv |
| Fig S9 | Serious adverse events | — | — | — |
| Fig S10 | Adverse events summary | — | — | — |
| Fig S11 | Chronic DBS programming history (P1–P5) | MATLAB | matlab_code/plot_figureS11_programming_history.m | data/figureS11_programming_history/ (`P#<side>_stim_parameters.csv` and `P#<side>_daily_duty_cycle.csv` per hemisphere; `P#_sham_windows.csv` per participant) |
| Fig S12 | Contact localization, temporary inpatient sEEG | — | — | — |
| Fig S13 | Electrode coregistration and localization examples | — | — | — |
| Fig S14 | Example volume of tissue activated, sEEG + permanent | — | — | — |
| Fig S15 | Example volume of tissue activated, P3 left sEEG | — | — | — |

`python_code/decoding_figures/04_plot_freq_coefficients.py` draws frequency-band
counts from the significant models. It is a supporting analysis that informs the
decoding figures; it is not itself a numbered figure.

## SUPPLEMENTARY tables
| ID | Title | Code / export | Data file |
|---|---|---|---|
| Table S1 | Participant medications and psychiatric diagnoses | — | — |
| Table S2 | Stimulation frequencies at PAG/PVG & thalamic targets | provided as data | data/supplementary_tables/TableS2_inpatient_frequencies_PAG_PVG_thalamic.xlsx |
| Table S3 | Inpatient sEEG stimulation parameters — P1 | provided as data | data/supplementary_tables/TableS3_P1_inpatient_stim_parameters.xlsx |
| Table S4 | Inpatient sEEG stimulation parameters — P2 | provided as data | data/supplementary_tables/TableS4_P2_inpatient_stim_parameters.xlsx |
| Table S5 | Inpatient sEEG stimulation parameters — P3 | provided as data | data/supplementary_tables/TableS5_P3_inpatient_stim_parameters.xlsx |
| Table S6 | Inpatient sEEG stimulation parameters — P4 | provided as data | data/supplementary_tables/TableS6_P4_inpatient_stim_parameters.xlsx |
| Table S7 | Inpatient sEEG stimulation parameters — P5 | provided as data | data/supplementary_tables/TableS7_P5_inpatient_stim_parameters.xlsx |
| Table S8 | Inpatient sEEG stimulation parameters — P6 | provided as data | data/supplementary_tables/TableS8_P6_inpatient_stim_parameters.xlsx |
| Table S9 | Optimal ambulatory pain-decoding biomarkers by implanted hemisphere | provided as data | data/supplementary_tables/TableS9_optimal_ambulatory_biomarkers.xlsx |
| Table S10 | Marginal battery drainage and break-even duty cycle | provided as data | data/figureS6_battery_efficiency/breakeven_duty.csv (per-hemisphere source table) |
| Table S11 | Full statistics for Extended Data Figure 3 (S11.1 omnibus, S11.2 post-hoc) | PYTHON, python_code/plot_extended_figure3_raw_distributions.py for the statistics and python_code/export_tableS11_word.py for the Word copy | data/extended_figures_raw_pain/ExtFig3_omnibus_condition_stats.csv, ExtFig3_pairwise_condition_stats.csv |
| Table S12 | Sensitivity of the closed-loop versus sham comparison to sham-exposure assumptions: model refits excluding each participant's sham observations and excluding each participant entirely | PYTHON, python_code/build_tableS12_sensitivity_analyses.py | figures/TableS12_sensitivity_analyses.csv (from data/figure5_double_blind_outcomes/cohort_daily_percent_change.csv) |

## MAIN tables
| ID | Title | Code / export | Data file |
|---|---|---|---|
| Table 1 | Participant details, diagnoses and implant targets | — | — |

## EXTENDED DATA tables
| ID | Title | Code / export | Data file |
|---|---|---|---|
| Ext Data Table 1 | Closed-loop algorithm settings per hemisphere: stimulation parameters, sensing bands, detector thresholds and durations | provided as data; the .csv is the authoritative copy and the .docx is a formatted rendering of it | figures/ExtendedDataTable1_closed_loop_algorithm_settings.csv, with .docx and ExtendedDataTable1_preview.png rendered from it |
| Ext Data Table 2 | Pain outcomes on all scales, all participants (closed-loop vs sham) during the double-blinded crossover window: median [IQR] and percent change from baseline in participant-days, 30% / 50% responder flags, index-metric flag | MATLAB, matlab_code/build_extended_data_table2.m; Word copy from python_code/export_extended_data_table2_word.py | data/extended_data_table2/blinded_window_pain_by_day.csv |

## Shipped neural data
- `neural_data/ambulatory_stages123/` — per-survey off-stimulation power spectral density extract for Figures S8 and S4, one HDF5 per implanted device side (9 files, ~8.5 MB total, 10,982 pain-survey spectra, spanning P1–P5). These ship as terminal data. `python_code/stability_and_closed_loop/extract_minimal_biomarker_features.py` (serial) or `extract_minimal_biomarker_features_parallel.py` (multi-core) documents how they were produced, and reads the continuous Summit recordings and pain-cluster spreadsheets, which are withheld as protected health information.
- `neural_features/` — per-participant narrowband spectral features at four window lengths (1/5/10/30 min); the sole input to the pain-decoding pipeline (`python_code/decoding_figures/01_run_ml_pipeline.py`).
- `python_code/decoding_figures/decoding_results/` — precomputed decoding results (185 files): each participant's model outputs and figures under `P1`–`P5/`, and the cross-participant summaries under `combined_results/`. These are what let scripts `03`, `05` and `06` run without re-fitting any model.

## Notes on scope
This deposit includes the figures and tables listed above. Final published figures involve substantial manual refinement by the authors in Adobe Illustrator (colour, panel arrangement, typography); the code here regenerates the underlying raw figures that are the starting point for that polishing.

`python_code/summarize_stimulation_parameters.py` writes `data/stimulation_parameter_summary/parameter_dispersion.csv` and `amplitude_time_trends.csv`; they support Supplementary Section S5 and are not numbered figures or tables.
