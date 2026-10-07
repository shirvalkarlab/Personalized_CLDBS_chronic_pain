function figure6_statistics = compute_figure6_statistics(configuration)
% COMPUTE_FIGURE6_STATISTICS  Mixed-effects test of closed-loop versus sham.
%
%   figure6_statistics = COMPUTE_FIGURE6_STATISTICS(configuration) fits the
%   pre-specified linear mixed-effects model comparing the closed-loop and
%   sham conditions on the daily percent change from pre-trial baseline,
%   once on the uniform numeric rating scale (mayoNRS) and once on each
%   participant's index metric, and prints both tests.
%
%   MODEL (pre-specified primary efficacy analysis)
%     outcome ~ 1 + stim_group + (1 + stim_group | patient_id)
%   Fixed effect:  stimulation condition (closed-loop versus sham).
%   Random effects: a per-participant intercept AND a per-participant slope
%     for the condition effect. The random slope is essential: it lets each
%     participant have their own closed-loop-versus-sham difference, which
%     avoids treating the many days within a participant as if they were
%     independent observations. A model without the random slope would
%     overstate significance.
%
%   The F test with residual degrees of freedom is the primary test; the
%   Satterthwaite F test is printed alongside.
%
%   The comparison is closed-loop versus sham only. P3 did not undergo sham
%   testing and so contributes no sham observations to this model.
%
%   INPUT
%     configuration      Struct from configure_repository. Uses
%                        .data_directory and .index_metric.
%
%   OUTPUT
%     figure6_statistics Struct with one field per outcome ('uniform_nrs',
%                        'index_metric'), each holding the fitted model, the
%                        residual and Satterthwaite F tests, and the p-value
%                        of the condition effect.
%
%   DATA
%     data/figure6_double_blind_outcomes/cohort_daily_percent_change.csv, the
%     per-day table that Figure 6a-c also plots.

data_file = fullfile(configuration.data_directory, ...
    'figure6_double_blind_outcomes', 'cohort_daily_percent_change.csv');
daily_table = readtable(data_file, 'TextType', 'string');

% ---- Keep only the closed-loop and sham conditions ---------------------
daily_table = daily_table(ismember(daily_table.stim_group, ["closed-loop", "sham"]), :);

% ---- Check: P3 has no sham condition -----------------------------------
assert(~any(daily_table.patient_id == "P3" & daily_table.stim_group == "sham"), ...
    'P3 sham rows found in the Figure 6 daily table; P3 has no sham condition.');

% ---- Index-metric outcome: each participant's own scale ----------------
index_percent_change = nan(height(daily_table), 1);
participants = fieldnames(configuration.index_metric);
for participant_index = 1:numel(participants)
    participant = participants{participant_index};
    rows   = daily_table.patient_id == participant;
    column = "baseline_percent_diff_" + configuration.index_metric.(participant);
    index_percent_change(rows) = daily_table.(column)(rows);
end
daily_table.index_percent_change = index_percent_change;

daily_table.stim_group = categorical(daily_table.stim_group, ["closed-loop", "sham"]);
daily_table.patient_id = categorical(daily_table.patient_id);

outcome_names   = {'uniform_nrs', 'index_metric'};
outcome_columns = {'baseline_percent_diff_mayoNRS', 'index_percent_change'};

figure6_statistics = struct();

for outcome_index = 1:numel(outcome_names)
    outcome_name   = outcome_names{outcome_index};
    outcome_column = outcome_columns{outcome_index};
    model_data     = daily_table(~isnan(daily_table.(outcome_column)), :);

    model_formula = sprintf('%s ~ 1 + stim_group + (1 + stim_group | patient_id)', ...
                            outcome_column);

    fitted_model = fitlme(model_data, model_formula, ...
        'FitMethod', 'REML', ...
        'DummyVarCoding', 'reference');

    % Residual-based F test (primary).
    analysis_of_variance = anova(fitted_model, 'DFMethod', 'residual');

    % Satterthwaite F test (reported alongside).
    analysis_of_variance_satterthwaite = anova(fitted_model, 'DFMethod', 'satterthwaite');

    % The condition effect is the second row (after the intercept).
    condition_row = 2;

    outcome_result = struct();
    outcome_result.model                          = fitted_model;
    outcome_result.analysis_of_variance           = analysis_of_variance;
    outcome_result.analysis_of_variance_satterthwaite = analysis_of_variance_satterthwaite;
    outcome_result.condition_p_value              = analysis_of_variance.pValue(condition_row);
    outcome_result.condition_p_value_satterthwaite = ...
        analysis_of_variance_satterthwaite.pValue(condition_row);

    fprintf(['Figure 6, %s: %d closed-loop and %d sham participant-days; ' ...
             'F(%d, %d) = %.2f, p = %.3f (Satterthwaite p = %.3f)\n'], ...
        outcome_name, sum(model_data.stim_group == "closed-loop"), ...
        sum(model_data.stim_group == "sham"), ...
        analysis_of_variance.DF1(condition_row), analysis_of_variance.DF2(condition_row), ...
        analysis_of_variance.FStat(condition_row), outcome_result.condition_p_value, ...
        outcome_result.condition_p_value_satterthwaite);

    figure6_statistics.(outcome_name) = outcome_result;
end

end
