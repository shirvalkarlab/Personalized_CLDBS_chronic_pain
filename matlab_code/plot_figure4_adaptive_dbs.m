function [figure4d_handle, figure4e_handle] = ...
        plot_figure4_adaptive_dbs(configuration, adaptive_dbs_data)
% PLOT_FIGURE4_ADAPTIVE_DBS  Closed-loop stimulation performance (P3 left).
%
%   [figure4d_handle, figure4e_handle] =
%   PLOT_FIGURE4_ADAPTIVE_DBS(configuration, adaptive_dbs_data) draws the
%   two closed-loop performance panels of Figure 4, both from participant
%   P3's left hemisphere:
%     Panel d  A zoomed example of the closed-loop control signal, showing
%              how the linear-discriminant output crosses the two decision
%              thresholds to switch stimulation states.
%     Panel e  Effective duty cycle (percent of time stimulation is on)
%              across days, above a raster of the closed-loop states.
%
%   In this closed-loop system a linear-discriminant classifier reads the
%   neural signal and outputs a single control value. Two thresholds (a
%   lower and an upper bias) turn that value into a stimulation state, which
%   in turn sets whether stimulation is on. Panel d shows the control value
%   and thresholds; Panel e shows the resulting long-run duty cycle.
%
%   INPUT
%     configuration       Struct from configure_repository.
%     adaptive_dbs_data   Struct from read_excel_sheets with two tables
%                         (the field names are the sheet names in
%                         data/figure4_adaptive_dbs_performance.xlsx):
%       .aDBS_spon_tbl   columns time, on_off, state, ampInMilliamps
%       .LD_tbl          columns time, LD_output
%
%   OUTPUT
%     figure4d_handle   Handle to Panel d (saved as Figure4d.pdf).
%     figure4e_handle   Handle to Panel e (saved as Figure4e.pdf).

% ---- Two decision thresholds for the control signal --------------------
% These are the classifier bias values that bracket the control signal:
% below the lower threshold and above the upper threshold the system
% switches stimulation state.
lower_threshold = 1522;
upper_threshold = 1810;

% ---- Zoom window for Panel d, in hours from the session start ----------
zoom_window_hours = hours([0.55, 0.70]);

state_table         = adaptive_dbs_data.aDBS_spon_tbl;
control_signal_table = adaptive_dbs_data.LD_tbl;

% ---- Shared x-axis: daily ticks over whole days ------------------------
time_step        = min(diff(state_table.time));
plot_start_day   = dateshift(state_table.time(1),   'start', 'day');
plot_end_day     = dateshift(state_table.time(end), 'end',   'day');
daily_tick_times = plot_start_day:hours(24):plot_end_day;

% ---- Effective duty cycle: trailing one-hour average of on/off ---------
% Averaging the on/off indicator over the previous hour gives the fraction
% of the last hour during which stimulation was on.
samples_per_hour     = duration('01:00:00') / time_step;
one_hour_duty_cycle  = movmean(state_table.on_off, [samples_per_hour, 0]);

% ---- Build the state raster matrix -------------------------------------
% One row per possible state (-1 through 8); each column is a time sample,
% with a value only in the row matching that sample's state. This lets
% pcolor draw each state as its own coloured band.
state_sequence   = state_table.state(2:end);
number_of_states = 10;                       % rows for states -1..8
state_matrix     = nan(number_of_states, numel(state_sequence));
for state_value = -1:8
    is_this_state = state_sequence == state_value;
    state_matrix(state_value + 2, is_this_state) = state_value;
end

% Colormap: grey for "off", then Set2 colours for the numbered states.
state_colormap = [0.5 0.5 0.5; set2_qualitative_colormap(8)];

% =======================================================================
%  PANEL d: zoomed control signal with decision thresholds
% =======================================================================
figure4d_handle = figure('Units', 'inches', 'Position', [1 1 5 4]);

% ---- State raster on the left axis -------------------------------------
draw_state_raster(state_table, state_matrix, state_colormap);
set(gca, 'TickDir', 'both');

% ---- Control signal and thresholds on the right axis -------------------
yyaxis("right");
hold("on");
threshold_style = {'LineWidth', 2, 'LabelHorizontalAlignment', 'left', ...
                   'FontSize', configuration.figure_font_size, ...
                   'Color', 'k', 'LineStyle', '--'};
yline(lower_threshold, 'Label', '\beta_{0}', threshold_style{:});
yline(upper_threshold, 'Label', '\beta_{1}', threshold_style{:});
plot(control_signal_table.time, control_signal_table.LD_output, ...
     'r-', 'LineWidth', 2);
ylim([lower_threshold * 0.5, upper_threshold * 1.5]);

% ---- Zoom to the example window ----------------------------------------
zoom_start   = dateshift(min(control_signal_table.time), 'end', 'hour');
zoom_window  = zoom_start + zoom_window_hours;
zoom_ticks   = linspace(zoom_window(1), zoom_window(2), 4);
all_axes     = findall(figure4d_handle, 'Type', 'axes');
for axis_index = 1:numel(all_axes)
    set(all_axes(axis_index), 'XLim', zoom_window, 'XTick', zoom_ticks);
end

ylabel(['Linear discriminant output (arbitrary units)' newline newline], ...
       'Rotation', 270, 'VerticalAlignment', 'middle');
xticklabels(0:3:9);
xlabel('Time (minutes)');
title('Closed-loop control signal example (P3 left)', 'FontWeight', 'normal');
set(gca, 'YColor', 'r', 'Box', 'off', ...
    'FontSize', configuration.figure_font_size, 'TickDir', 'both');

output_file_d = fullfile(configuration.figure_directory, 'Figure4d.pdf');
exportgraphics(figure4d_handle, output_file_d, configuration.figure_export_options{:});

% =======================================================================
%  PANEL e: duty cycle over days, with the state raster beneath
% =======================================================================
figure4e_handle = figure('Units', 'inches', 'Position', [1 1 5 4]);
tiledlayout(4, 1, 'TileSpacing', 'compact');

% ---- Top: trailing one-hour duty cycle ---------------------------------
nexttile([3, 1]);
hold("on");
plot(state_table.time, one_hour_duty_cycle, 'k', 'LineWidth', 1.5);
ylim([-2.5, 50]); yticks([0 20 40]);
xlim([plot_start_day, plot_end_day]); xticks([]);
ylabel('Percent time on');
box("on");
title('Effective duty cycle during closed-loop stimulation (P3 left)', ...
      'FontWeight', 'normal');
set(gca, 'FontSize', configuration.figure_font_size, 'TickDir', 'both');

% ---- Bottom: state raster ----------------------------------------------
nexttile();
hold("on");
draw_state_raster(state_table, state_matrix, state_colormap);
xlim([plot_start_day, plot_end_day]);
xticks(daily_tick_times);
xticklabels(1:numel(daily_tick_times));
xlabel('Time (days)');
set(gca, 'FontSize', configuration.figure_font_size, 'TickDir', 'both');

output_file_e = fullfile(configuration.figure_directory, 'Figure4e.pdf');
exportgraphics(figure4e_handle, output_file_e, configuration.figure_export_options{:});

end


function draw_state_raster(state_table, state_matrix, state_colormap)
% Draw the closed-loop state occupancy as a coloured raster over time, with
% the y-axis limited to the lowest few states (the ones actually used).
raster = pcolor(state_table.time(1:end-1), -1:8, state_matrix);
colormap(gca, state_colormap);
set(raster, 'EdgeColor', 'none');
yticks(-0.5:8.5);
yticklabels(['Off', compose('%g', 0:8)]);
ylim([0, 3]);
ylabel('States');
set(gca, 'YColor', 'k');
end
