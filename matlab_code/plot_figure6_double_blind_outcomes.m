function figure_handle = plot_figure6_double_blind_outcomes(configuration)
% PLOT_FIGURE6_DOUBLE_BLIND_OUTCOMES  Double-blind closed-loop versus sham
% pain outcomes (manuscript Figure 6, panels a-c).
%
%   figure_handle = PLOT_FIGURE6_DOUBLE_BLIND_OUTCOMES(configuration) draws the
%   three-panel cohort figure comparing closed-loop stimulation against sham:
%       a) Uniform numeric rating scale - each participant's daily percent
%          change from their pre-trial baseline on the numeric rating scale,
%          closed-loop versus sham.
%       b) Personalized index metric - the same comparison on each
%          participant's prespecified index metric.
%       c) Response cumulative distribution - the proportion of participant-
%          days achieving at least a given percent reduction in the numeric
%          rating scale, closed-loop versus sham.
%
%   Closed-loop is drawn in blue and sham in neutral grey (never black); the
%   individual daily scores are shown as a black swarm over each summary box.
%
%   The drawing is done by the shared helper plot_cohort_index_and_response_cdf
%   (in matlab_code/helper). This function loads the cohort table, sets the
%   per-participant index-metric assignment and the sham-validity rules, and
%   hands them to the plotter.
%
%   INPUT
%     configuration   Struct from configure_repository. Uses .data_directory,
%                     .index_metric, .figure_directory, and
%                     .figure_export_options.
%
%   OUTPUT
%     figure_handle   Handle to the created figure.
%
%   DATA (in data/figure6_double_blind_outcomes/)
%     cohort_daily_percent_change.csv  one row per participant-day-condition,
%         with patient_id, stim_group, and the percent-change-from-baseline
%         columns (baseline_percent_diff_<scale>) the figure plots.

% ---- Load the cohort table ---------------------------------------------
data_file = fullfile(configuration.data_directory, ...
    'figure6_double_blind_outcomes', 'cohort_daily_percent_change.csv');
cohort_groups = readtable(data_file, 'TextType', 'string');

% ---- Analysis settings the figure needs --------------------------------
% (1) Each participant's prespecified index pain metric (configure_repository),
%     as the percent-change column the plotter reads.
index_metric = table();
participants = fieldnames(configuration.index_metric);
for participant_index = 1:numel(participants)
    participant = participants{participant_index};
    index_metric.(participant) = "baseline_percent_diff_" + configuration.index_metric.(participant);
end
configuration.analysis.patient_best_metric_table = index_metric;

% (2) Which participants have no sham data on which scale families, so those
%     sham boxes are drawn as an explicit "no sham" marker rather than an
%     unexplained gap.
configuration.analysis.sham_nan_exclusions = struct( ...
    'P3', {{'NRS', 'VAS', 'MPQ'}}, ...   % no sham condition
    'P5', {{'VAS', 'MPQ'}});             % VAS and MPQ on only one of four sham days

% ---- Draw the figure with the shared plotting code ---------------------
% The plotter expects participant labels P1..P5 in patient_id and column names
% of the form baseline_percent_diff_<scale>, which is how the cohort table is
% organised. It saves the PDF into configuration.figure_directory.
figure_handle = plot_cohort_index_and_response_cdf(configuration, cohort_groups);

end
