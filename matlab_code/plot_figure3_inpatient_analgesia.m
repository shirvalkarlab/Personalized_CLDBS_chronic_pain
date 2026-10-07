function [figure_handle, statistics_table] = ...
        plot_figure3_inpatient_analgesia(configuration, analgesia_data)
% PLOT_FIGURE3_INPATIENT_ANALGESIA  Pain relief by stimulation region.
%
%   [figure_handle, statistics_table] =
%   PLOT_FIGURE3_INPATIENT_ANALGESIA(configuration, analgesia_data) draws
%   one stacked panel per participant showing the change in pain (visual
%   analogue score) produced by stimulating each brain region during the
%   inpatient stereo-EEG mapping trial, and tests each region against the
%   sham condition.
%
%   Within each participant's panel the regions are ordered from most to
%   least analgesic (most negative change first) with the sham condition
%   drawn last as a reference. Every box shows its individual data points
%   swarmed on top. A significance bracket marks each region whose pain
%   reduction differs from sham.
%
%   INPUT
%     configuration    Struct from configure_repository.
%     analgesia_data   Struct of tables from read_excel_sheets, one field
%                      per participant (P1..P6). Each table has columns
%                      'target' (region name or 'SHAM') and 'change_painVAS'.
%
%   OUTPUT
%     figure_handle     Handle to the created figure (saved as Figure3.pdf).
%     statistics_table  Table of rank-sum test results, one row per tested
%                       region per participant.
%
%   The regions tested against sham for each participant are the pre-defined
%   candidate targets that were carried forward to consideration for
%   permanent implant. Participant P6 completed mapping but is not tested
%   here because P6 did not proceed to a permanent implant.

metric_name = 'change_painVAS';

% ---- Candidate implant regions tested against sham, per participant ----
% These are the brain regions that produced enough analgesia during mapping
% to be considered as permanent targets; they are the regions the figure
% marks with a statistical comparison against sham.
regions_by_participant = struct( ...
    'P1', ["L VCVS", "R VPThal",  "R ACC"], ...
    'P2', ["L Caudate", "R CMThal", "R ACC"], ...
    'P3', ["L Caudate", "R CMThal"], ...
    'P4', ["L ACC", "R CMThal"], ...
    'P5', ["L GPi", "R CMThal"]);

% Some workbook sheets spell regions differently from the manuscript. Every
% region is renamed to its manuscript name (right) as each sheet is read, so the
% tested regions above, the boxes and the axis labels all refer to one name.
manuscript_region_names = containers.Map( ...
    {'LeftCaudate', 'LeftACC', 'RightACC', 'RightCMThal', 'R VPN', 'L VPL', ...
     'R CMPf', 'L Motor', 'R Motor'}, ...
    {'L Caudate', 'L ACC', 'R ACC', 'R CMThal', 'R VPThal', 'L VPThal', ...
     'R CMThal', 'L M1', 'R M1'});

participant_labels = fieldnames(analgesia_data);
number_of_panels   = numel(participant_labels);

% Minimum number of valid re-baselined surveys a region needs to be shown.
minimum_surveys = 5;

% ---- Figure: one tall column, one row per participant ------------------
figure_handle = figure('Units', 'inches', 'Position', [1 1 3.5 11], 'Color', 'w');
tiledlayout(number_of_panels, 1, 'TileSpacing', 'tight', 'Padding', 'tight');

statistics_table = table();

for panel_index = 1:number_of_panels
    participant_label = participant_labels{panel_index};
    patient_table     = analgesia_data.(participant_label);
    patient_table.target = rename_regions(string(patient_table.target), ...
                                          manuscript_region_names);

    nexttile();
    hold("on");
    current_axes = gca;

    % ---- Rank regions most-analgesic first, keep only well-sampled ones -
    % Regions are ordered by their mean pain change (most negative = most
    % analgesic first). A region must have at least minimum_surveys valid
    % re-baselined surveys to appear. The sham condition is always placed
    % last as the rightmost reference box, and is never dropped by the count
    % filter. This matches the manuscript's cohort figure.
    region_order = order_regions_by_mean(patient_table, metric_name, ...
        minimum_surveys);

    % ---- Draw the box + swarm for every region in order ----------------
    draw_box_and_swarm(current_axes, patient_table, metric_name, region_order);

    % ---- Test each candidate region against sham (if this participant) -
    % One-tailed Wilcoxon rank-sum against sham; the helper also draws the
    % significance brackets above the boxes.
    if isfield(regions_by_participant, participant_label)
        best_regions = regions_by_participant.(participant_label);
        statistics_table = compute_analgesia_statistics( ...
            patient_table, metric_name, participant_label, ...
            best_regions, string(region_order), statistics_table);
    end

    % ---- Panel formatting ----------------------------------------------
    format_participant_panel(participant_label, patient_table, ...
        metric_name, region_order);
end

% ---- Save --------------------------------------------------------------
output_file = fullfile(configuration.figure_directory, 'Figure3.pdf');
exportgraphics(figure_handle, output_file, configuration.figure_export_options{:});

end


function region_order = order_regions_by_mean(patient_table, metric_name, minimum_surveys)
% Return region names ordered most-analgesic first (lowest mean change),
% keeping only regions with at least minimum_surveys valid surveys, with the
% sham condition appended last as the rightmost reference box.
regions = string(patient_table.target);
values  = patient_table.(metric_name);

unique_regions = unique(regions, 'stable');
unique_regions = unique_regions(~strcmpi(unique_regions, 'SHAM'));
unique_regions = unique_regions(~strcmpi(unique_regions, 'PRETRIAL'));

region_mean  = nan(numel(unique_regions), 1);
region_count = zeros(numel(unique_regions), 1);
for region_index = 1:numel(unique_regions)
    in_region = regions == unique_regions(region_index) & ~isnan(values);
    region_count(region_index) = sum(in_region);
    region_mean(region_index)  = mean(values(in_region));
end

keep = region_count >= minimum_surveys;
unique_regions = unique_regions(keep);
region_mean    = region_mean(keep);

[~, sort_index] = sort(region_mean, 'ascend');   % most analgesic first
region_order    = [cellstr(unique_regions(sort_index)); {'SHAM'}]';
end


function draw_box_and_swarm(current_axes, patient_table, metric_name, region_order)
% Box (interquartile range) with the raw points swarmed on top, in the
% manuscript house style: a pale-blue box, a red mean line, black whiskers
% to the full data range, and black semi-transparent swarm points. The high
% contrast between the black points and the pale box keeps every individual
% observation clearly visible.
regions    = string(patient_table.target);
box_width  = 0.55;
light_fill = [0.85, 0.89, 0.95];   % pale blue box fill

for region_index = 1:numel(region_order)

    values = patient_table.(metric_name)(regions == string(region_order{region_index}));
    values = values(~isnan(values));
    if isempty(values); continue; end

    x_center = region_index;

    quartiles      = quantile(values, [0.25, 0.50, 0.75]);
    lower_quartile = quartiles(1);
    upper_quartile = quartiles(3);
    mean_value     = mean(values);
    lowest_value   = min(values);
    highest_value  = max(values);
    inter_quartile = upper_quartile - lower_quartile;
    half           = box_width / 2;

    % Box (interquartile range), white edge on the pale fill.
    rectangle(current_axes, 'Position', ...
        [x_center - half, lower_quartile, box_width, max(inter_quartile, eps)], ...
        'FaceColor', light_fill, 'EdgeColor', 'w', 'LineWidth', 0.5);
    % Mean line (red).
    plot(current_axes, [x_center - half, x_center + half], ...
        [mean_value, mean_value], 'r-', 'LineWidth', 0.5);
    % Whiskers to the full data range, with caps.
    plot(current_axes, [x_center, x_center], [upper_quartile, highest_value], ...
        'k-', 'LineWidth', 0.5);
    plot(current_axes, [x_center, x_center], [lowest_value, lower_quartile], ...
        'k-', 'LineWidth', 0.5);
    plot(current_axes, [x_center - half/2, x_center + half/2], ...
        [highest_value, highest_value], 'k-', 'LineWidth', 0.5);
    plot(current_axes, [x_center - half/2, x_center + half/2], ...
        [lowest_value, lowest_value], 'k-', 'LineWidth', 0.5);

    % Swarm of raw points on top: black, size 10, semi-transparent, spread
    % horizontally by local density so overlapping points remain visible.
    swarmchart(current_axes, repmat(x_center, numel(values), 1), values, 10, ...
        'k', 'filled', 'MarkerFaceAlpha', 0.75, ...
        'XJitter', 'density', 'XJitterWidth', 0.35);
end

xlim(current_axes, [0.5, numel(region_order) + 0.5]);
yline(current_axes, 0, ':', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 0.5);
end


function format_participant_panel(participant_label, patient_table, metric_name, region_order)
% Apply the shared per-panel formatting: each region labelled with its valid
% survey count, a zero reference line, and a bold participant label.
number_of_groups = numel(region_order);
region_labels    = string(patient_table.target);
values           = patient_table.(metric_name);

new_labels = cell(1, number_of_groups);
for region_index = 1:number_of_groups
    n_valid = sum(~isnan(values(region_labels == string(region_order{region_index}))));
    new_labels{region_index} = sprintf('%s (%d)', region_order{region_index}, n_valid);
end

current_axes = gca;
xticks(current_axes, 1:number_of_groups);
xticklabels(current_axes, new_labels);
xtickangle(current_axes, 30);

set(current_axes, 'TickLength', [0 0], 'TickDir', 'out', 'Box', 'off', 'FontSize', 9);
ylabel(participant_label, 'FontWeight', 'bold', 'FontSize', 11);
end


function region_names = rename_regions(region_names, manuscript_region_names)
% Replace each workbook region name that has a manuscript name; leave the rest.
for name_index = 1:numel(region_names)
    workbook_name = char(region_names(name_index));
    if isKey(manuscript_region_names, workbook_name)
        region_names(name_index) = manuscript_region_names(workbook_name);
    end
end
end
