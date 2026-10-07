function statistics_table = compute_analgesia_statistics( ...
        patient_table, metric_name, participant_label, ...
        target_regions_of_interest, group_labels, statistics_table)
% COMPUTE_ANALGESIA_STATISTICS  Rank-sum test of each region against SHAM.
%
%   statistics_table = COMPUTE_ANALGESIA_STATISTICS(patient_table,
%   metric_name, participant_label, target_regions_of_interest,
%   group_labels, statistics_table) runs a one-sided Wilcoxon rank-sum test
%   comparing each region of interest against the SHAM condition for one
%   participant, draws a significance bracket over the plotted boxes, and
%   appends one result row per region to statistics_table.
%
%   The test is one-sided (tail 'left') because the hypothesis is that
%   active stimulation produces a greater pain reduction (a more negative
%   change) than sham.
%
%   INPUT
%     patient_table               Table for this participant.
%     metric_name                 Numeric column being tested.
%     participant_label           Participant identifier, for example 'P1'.
%     target_regions_of_interest  String array of the regions to test for
%                                 this participant.
%     group_labels                Categorical target order, used to place the
%                                 significance bracket at the right box.
%     statistics_table            Table to append to (may be empty on the
%                                 first call).
%
%   OUTPUT
%     statistics_table            Input table with one appended row per
%                                 tested region: participant, target, the
%                                 p-value, the rank-sum statistic, and the
%                                 sample size of each group.

target        = string(patient_table.target);
metric_values = patient_table.(metric_name);

sham_values = metric_values(target == "SHAM" & ~isnan(metric_values));

for region_index = 1:numel(target_regions_of_interest)
    this_region  = target_regions_of_interest(region_index);
    region_values = metric_values(target == this_region & ~isnan(metric_values));

    if isempty(region_values)
        warning("compute_analgesia_statistics:regionNotFound", ...
            "No surveys found for region %s in %s. " + ...
            "Check that the region label matches the workbook target column.", ...
            this_region, participant_label);
    end
    if isempty(region_values) || isempty(sham_values)
        p_value        = NaN;
        rank_sum_value = NaN;
    else
        [p_value, ~, test_stats] = ranksum(region_values, sham_values, 'tail', 'left');
        rank_sum_value = test_stats.ranksum;
    end

    % ---- Append the result row -----------------------------------------
    statistics_table = [statistics_table; table( ...
        string(participant_label), string(this_region), p_value, rank_sum_value, ...
        numel(region_values), numel(sham_values), ...
        'VariableNames', {'participant', 'target', 'p_value', ...
                          'rank_sum_statistic', 'n_region', 'n_sham'})]; %#ok<AGROW>

    % ---- Draw a significance bracket between this region and SHAM -------
    region_position = find(group_labels == this_region);
    sham_position   = find(group_labels == "SHAM");
    if ~isempty(region_position) && ~isempty(sham_position)
        significance_stars([region_position, sham_position], p_value, ...
                           max(region_values));
    end
end

end
