function output_file = build_extended_data_table2(configuration)
% BUILD_EXTENDED_DATA_TABLE2  Pain outcomes on every scale, closed-loop vs sham.
%
%   For each participant and each pain scale they provided, reports the
%   baseline, closed-loop and sham distributions and the percent change from
%   baseline during the double-blinded crossover window.
%
%   EVERY COUNT IN THIS TABLE IS A NUMBER OF PARTICIPANT-DAYS. A day enters a
%   distribution if the participant gave at least one report that day; the
%   value used is that day's mean across their reports. Day counts therefore
%   differ from survey counts, which are larger (a participant may report
%   several times per day) and are what Table S11 reports.
%
%   INPUT
%     configuration   Struct from configure_repository (data path and
%                     .index_metric).
%
%   OUTPUT
%     output_file     Path to the written .csv.
%
%   Reads  data/extended_data_table2/blinded_window_pain_by_day.csv
%   Writes figures/ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham.csv

input_file = fullfile(configuration.data_directory, 'extended_data_table2', ...
                      'blinded_window_pain_by_day.csv');
observations = readtable(input_file, 'VariableNamingRule', 'preserve');

% Scale column, printed label, and the family name used by the sham-exclusion
% rule below.
scales = { ...
    'mayoNRS',       'NRS (0-10)',             'NRS'; ...
    'painVAS',       'VAS intensity (0-100)',  'VAS'; ...
    'unpleasantVAS', 'VAS unpleasantness',     'VAS'; ...
    'MPQtotal',      'MPQ (short-form total)', 'MPQ'; ...
    'nocNRS',        'Nociceptive NRS (0-10)', 'NRS'};

% Each participant's prospectively specified index metric (Statistical
% Analysis Plan; one scale per participant) comes from configure_repository.
index_metric = configuration.index_metric;

% Sham distributions withheld where too few days were collected for a
% distribution to be meaningful: the visual analogue and MPQ scales for P5,
% which provided those scales on a single sham day.
sham_withheld = containers.Map({'P5'}, {{'VAS', 'MPQ'}});

participants = {'P1', 'P2', 'P3', 'P4', 'P5'};
rows = {};

for participant_index = 1:numel(participants)
    participant = participants{participant_index};
    is_participant = strcmp(observations.participant, participant);

    for scale_index = 1:size(scales, 1)
        column = scales{scale_index, 1};
        label  = scales{scale_index, 2};
        family = scales{scale_index, 3};

        baseline    = collect(observations, is_participant, 'pre-trial',   column);
        closed_loop = collect(observations, is_participant, 'closed-loop', column);
        sham        = collect(observations, is_participant, 'sham',        column);

        % A scale appears for a participant only if they provided it, or if it
        % is their index metric.
        is_index = strcmp(index_metric.(participant), column);
        if isempty(baseline) && isempty(closed_loop) && ~is_index
            continue
        end

        closed_loop_change = collect(observations, is_participant, 'closed-loop', ...
                                     ['baseline_percent_diff_' column]);
        sham_change        = collect(observations, is_participant, 'sham', ...
                                     ['baseline_percent_diff_' column]);

        withhold = isKey(sham_withheld, participant) && ...
                   any(strcmp(sham_withheld(participant), family));
        if withhold
            sham        = [];
            sham_change = [];
        end

        rows(end+1, :) = { ...
            participant, label, index_marker(is_index), ...
            median_with_range(baseline), ...
            median_with_range(closed_loop), ...
            percent_change(closed_loop_change), ...
            median_with_range(sham), ...
            percent_change(sham_change), ...
            responder(closed_loop_change, 30), ...
            responder(closed_loop_change, 50)};
    end
end

table_out = cell2table(rows, 'VariableNames', { ...
    'participant', 'pain_scale', 'prospective_index_metric', ...
    'baseline_median_iqr_n_days', 'closed_loop_median_iqr_n_days', ...
    'closed_loop_percent_change_from_baseline', 'sham_median_iqr_n_days', ...
    'sham_percent_change_from_baseline', ...
    'responder_30pct_closed_loop', 'responder_50pct_closed_loop'});

if ~isfolder(configuration.figure_directory); mkdir(configuration.figure_directory); end
output_file = fullfile(configuration.figure_directory, ...
    'ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham.csv');
writetable(table_out, output_file);

fprintf('Extended Data Table 2: %d rows written to %s\n', height(table_out), output_file);
end



function values = collect(observations, is_participant, group, column)
%COLLECT Day-level values for one participant, stimulation group and column.
if ~ismember(column, observations.Properties.VariableNames)
    values = [];
    return
end
selected = is_participant & strcmp(observations.stim_group, group);
values = observations.(column)(selected);
values = values(~isnan(values));
end


function text = median_with_range(values)
%MEDIAN_WITH_RANGE Median [interquartile range] (number of days).
if isempty(values)
    text = '-';
    return
end
text = sprintf('%.1f [%.1f-%.1f] (%d)', median(values), ...
               prctile(values, 25), prctile(values, 75), numel(values));
end


function text = percent_change(values)
%PERCENT_CHANGE Median percent change from baseline, signed.
if isempty(values)
    text = '-';
    return
end
text = sprintf('%+.0f%%', median(values));
end


function text = responder(values, threshold)
%RESPONDER Whether the median reduction from baseline met a threshold.
if isempty(values)
    text = '';
elseif median(values) <= -threshold
    text = 'Y';
else
    text = 'N';
end
end


function text = index_marker(is_index)
%INDEX_MARKER Tick for the participant's prospectively specified index metric.
if is_index
    text = 'yes';
else
    text = '';
end
end
