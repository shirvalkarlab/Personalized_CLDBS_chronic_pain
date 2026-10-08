function figure_handle = plot_pain_timecourse(configuration, measure, participants, ...
    output_basename, figure_title, vas50_run_threshold)
%PLOT_PAIN_TIMECOURSE  Longitudinal pain time course, one row per participant.
%
%   figure_handle = plot_pain_timecourse(configuration, measure, participants,
%                                        output_basename, figure_title,
%                                        vas50_run_threshold)
%
%   Draws the full pain record for one or more participants, one participant
%   per stacked row. Each row is a three-panel broken x-axis that follows the
%   study timeline in real order:
%
%       [ baseline ]  //  [ Stage 0 ]  //  [ ------ chronic (Stage 1-3) ------ ]
%
%     - baseline : pre-operative Pain Fluctuation Study surveys (per survey),
%                  the year before the temporary electrode implant.
%     - Stage 0  : the temporary externalised-electrode inpatient trial
%                  (dense, ~10 days), all surveys pooled (not split by stim).
%     - chronic  : every ambulatory survey from permanent implant (Stage 1)
%                  through open-loop optimisation (Stage 2) and the
%                  double-blind crossover (Stage 3).
%
%   All three panels share one x origin (days since the first baseline survey)
%   and a common y scale. This single function makes BOTH manuscript figures:
%     - main Figure 1            : one participant (P3), pain VAS
%     - Extended Data Figure 2   : all five participants, any selectable
%                                  pain measure
%   so there is one plotter and no hand-curated figure data.
%
%   Dots are coloured by stimulation condition in the chronic panel:
%       open-loop grey, closed-loop yellow, double-blind crossover blue.
%   The double-blind crossover is drawn as one clean terminal block (blue dots
%   from Stage 3 onset to the participant's endpoint). Washout (0 mA) surveys
%   are thin plus signs, drawn only before Stage 3 (inside the crossover a 0 mA
%   survey is a blinded sham, not a washout). Blinded shams are hollow circles.
%   A light-red centred moving-average line is drawn on top.
%
%   The moving-average window is set PER PANEL, in survey points (not days):
%   14 in baseline, 21 in Stage 0, 21 in chronic.
%
%   PANEL ASSIGNMENT is derived here, in the plotter, from the survey table's
%   `stage` column -- the data file records facts about surveys and carries no
%   presentation scaffolding:
%       stage "pre-trial"                     -> baseline panel
%       stage "stage0"                        -> Stage 0 panel
%       every other stage ("stage1", "stage2",
%       "stage3", "study graduation",
%       "unknown")                            -> chronic panel
%
%   The moving-average line is computed over every plotted survey in the
%   panel, regardless of condition.
%
%   INPUTS
%     configuration   - struct from configure_repository(); uses
%                       configuration.data_directory and .figure_directory.
%     measure         - optional char/string naming the pain-scale COLUMN to
%                       plot: "mayoNRS", "painVAS" (default),
%                       "unpleasantVAS", or "MPQtotal".
%     participants    - optional string array of participant labels to draw,
%                       one row each (default ["P1" "P2" "P3" "P4" "P5"]).
%     output_basename - optional output file name without extension
%                       (default "FigureExtended2_" + measure + "_timecourse").
%     figure_title    - optional overall title (default describes the measure).
%     vas50_run_threshold - optional scalar (default 4). The pain-intensity VAS
%                       slider opens at 50 and, when not moved, records 50
%                       again; a value of 50 REPEATED across several consecutive
%                       surveys is the signature of a slider left at its
%                       default. When plotting painVAS, any run of MORE THAN
%                       this many consecutive painVAS == 50 surveys (per
%                       participant, in time order) is set to NaN so it is
%                       neither plotted nor smoothed. Isolated and short runs of
%                       50 are kept as potentially genuine moderate ratings. Set
%                       to 0 to disable. Applied at plot time only; the
%                       pain-score CSV is never modified.
%
%   OUTPUT
%     figure_handle   - handle to the created figure (also written to
%                       figure_directory as <output_basename>.pdf, .png and
%                       .svg).

% ---- arguments ------------------------------------------------------------
if nargin < 2 || isempty(measure);      measure = "painVAS"; end
if nargin < 3 || isempty(participants);  participants = ["P1", "P2", "P3", "P4", "P5"]; end
if nargin < 6 || isempty(vas50_run_threshold); vas50_run_threshold = 4; end
measure      = string(measure);
participants = string(participants);

% ---- appearance constants -------------------------------------------------
measure_axis_maximum = struct("mayoNRS", 10, "painVAS", 100, ...
                              "unpleasantVAS", 100, "MPQtotal", 50);
measure_axis_label   = struct("mayoNRS", "Pain NRS (0-10)", ...
                              "painVAS", "Pain intensity VAS (0-100)", ...
                              "unpleasantVAS", "Unpleasantness VAS (0-100)", ...
                              "MPQtotal", "MPQ total");

baseline_color   = [0.60 0.60 0.60];   % pre-op baseline dots (grey)
stage0_color     = [0.45 0.45 0.45];   % Stage 0 inpatient dots (darker grey)

open_loop_color  = [0.42 0.42 0.42];   % open-loop optimisation (grey)
closed_loop_color = [0.95 0.80 0.15];  % closed-loop (yellow)
crossover_color  = [0.20 0.55 0.85];   % double-blind crossover (blue)
washout_color    = [0.62 0.62 0.62];   % washout crosses (light grey, drawn behind the dots)
sham_edge_color  = [0.10 0.10 0.10];   % blinded sham hollow-circle edge
trend_color      = [0.85 0.15 0.15];   % moving-average line (red)
trend_alpha      = 0.45;               % light alpha on the trend line
point_size       = 5;                  % scatter dot area
% One participant's pre-operative baseline was collected in two visits far
% apart; only the later visit is drawn. A gap longer than this many days
% separates the two visits.
baseline_split_participants = "P5";
baseline_split_gap_days     = 200;

point_alpha      = 0.70;               % scatter dot transparency
sham_size        = 18;                 % hollow-circle / plus marker area
gap_break_days   = 45;                 % break the trend across gaps > this

% Per-participant chronic-panel cutoff, as a CALENDAR DATE.
% Some participants keep reporting long after their informative follow-up ends,
% leaving a sparse tail that stretches the axis. Where a hard cutoff reads
% better than the metadata endpoint, it is set here; participants not listed
% use their endpoint_datetime from longitudinal_phase_metadata.csv.
%
% These are dates, not day offsets, on purpose. A day offset is measured from
% the participant's first surviving survey, so adding earlier surveys to the
% record silently slides every offset-based cutoff later in the trial. The
% dates below are fixed points in each participant's timeline and are
% converted to plot days against whatever origin the data give.
%
% Cutoffs chosen from the survey record: P1 where reporting thins to a sparse
% tail; P3 at the start of a 185-day reporting gap; P4 where monthly survey
% counts collapse from 73 in 2178-04 to 5 in 2178-05.
plot_cutoff_date = struct("P1", datetime(2174, 1,13), ...
                          "P3", datetime(2176, 9,10), ...
                          "P4", datetime(2178, 5,27));

% Moving-average window per panel, in survey points (not days): 14 in
% baseline, 21 in Stage 0 and in chronic. Stage 0 uses the same span as the
% chronic panel so the inpatient line tracks the within-stay variation rather
% than averaging most of the stay into one slope.
baseline_smooth_span = 14;
stage0_smooth_span   = 21;
chronic_smooth_span  = 21;

if ~isfield(measure_axis_maximum, measure)
    error("plot_pain_timecourse:unknownMeasure", ...
          "Unknown measure '%s'. Supported: %s.", measure, ...
          strjoin(string(fieldnames(measure_axis_maximum))', ", "));
end
y_maximum = measure_axis_maximum.(measure);
y_label   = measure_axis_label.(measure);

if nargin < 4 || isempty(output_basename)
    output_basename = "FigureExtended2_" + measure + "_timecourse";
end
output_basename = string(output_basename);
if nargin < 5 || isempty(figure_title)
    % The axis label carries its own parenthesised range ("Pain intensity VAS
    % (0-100)"); wrapping it whole would nest a second pair of brackets. The
    % inner ones are dropped so the title reads with one pair.
    title_measure = erase(y_label, ["(", ")"]);
    figure_title = "Longitudinal pain time course (" + title_measure + ")";
end
figure_title = string(figure_title);

% ---- data -----------------------------------------------------------------
% One per-survey table feeds every pain figure: pain_surveys.csv, one row per
% survey, one column per pain scale. The requested measure names the column to
% read; surveys with no value on that scale are dropped.
data_folder = fullfile(configuration.data_directory, "extended_figures_raw_pain");
survey_file = fullfile(data_folder, "pain_surveys.csv");
scores = readtable(survey_file, "TextType", "string");

% Fail loudly on a survey table that cannot draw this figure, rather than
% rendering blank panels that look like a participant simply stopped reporting.
required_columns = ["participant", "time", "condition", "stage", measure];
missing_columns  = required_columns(~ismember(required_columns, ...
                                              string(scores.Properties.VariableNames)));
if ~isempty(missing_columns)
    error("plot_pain_timecourse:missingColumn", ...
          "%s lacks required column(s): %s. Available: %s", survey_file, ...
          strjoin(missing_columns, ", "), ...
          strjoin(string(scores.Properties.VariableNames), ", "));
end

% Survey times and the phase boundaries in longitudinal_phase_metadata.csv are
% both written as study-site local time without a zone, so they compare
% directly. readtable may hand back the column already parsed as datetime or
% still as text, so both are handled. Should a zoned column ever arrive, it is
% put on one clock before comparison, since a zoned time compared against an
% unzoned boundary shifts every boundary by the offset.
if ~isdatetime(scores.time)
    scores.time = datetime(scores.time, "InputFormat", "yyyy-MM-dd HH:mm:ss");
end
if ~isempty(scores.time.TimeZone)
    % A zoned column is put on one clock with the unzoned phase boundaries;
    % comparing a zoned time against an unzoned one shifts every boundary.
    scores.time.TimeZone = "UTC";
    scores.time.TimeZone = "";
end

% P5 recorded the NRS on all four sham days and the other scales on one. Keep
% the NRS sham surveys aside (before the requested scale is selected) so they
% can be rescaled onto whatever measure is being plotted and stay visible.
p5_sham_nrs_rows = scores(scores.participant == "P5" & ...
                          scores.condition == "sham" & ...
                          ~isnan(scores.mayoNRS), :);
p5_sham_nrs_rows.value = p5_sham_nrs_rows.mayoNRS;

% Select the requested scale into a single `value` column and drop surveys that
% did not record it. Everything downstream reads `value`.
scores.value = scores.(measure);
scores = scores(~isnan(scores.value), :);
if isempty(scores)
    error("plot_pain_timecourse:noSurveys", ...
          "No survey in %s records a value for measure '%s'.", survey_file, measure);
end

% Optional pain-VAS "stuck slider" cleaning. The pain-intensity VAS slider
% opens at 50; when it is not moved it emits exactly 50 again. A genuine 50 is
% possible, but a value of 50 REPEATED across several consecutive surveys is the
% signature of a slider left at its default. When vas50_run_threshold > 0 and
% the requested measure is painVAS, any run of MORE THAN vas50_run_threshold
% consecutive painVAS == 50 surveys (per participant, in time order) is set to
% NaN so it is neither plotted nor smoothed. A threshold of 0 disables this.
if vas50_run_threshold > 0 && measure == "painVAS"
    scores = nan_frozen_vas50_runs(scores, vas50_run_threshold);
end

% ---- panel assignment (derived here, not stored in the data file) ---------
% The survey table records the trial stage of each survey; the broken-axis
% layout needs three panels. Pre-trial surveys are the pre-operative baseline,
% stage0 surveys are the inpatient trial, and every later stage -- including
% "study graduation" and "unknown" -- belongs to the ambulatory chronic panel.
is_baseline_panel = scores.stage == "pre-trial";
is_stage0_panel   = scores.stage == "stage0";
is_chronic_panel  = ~is_baseline_panel & ~is_stage0_panel;

baseline = scores(is_baseline_panel, :);
stage0   = scores(is_stage0_panel, :);
chronic  = scores(is_chronic_panel, :);

% Every requested participant must have surveys in every panel. An empty panel
% here means the survey table or the participant list is wrong, not that the
% figure should show a blank tile.
for check_index = 1:numel(participants)
    check_participant = participants(check_index);
    panel_counts = [sum(baseline.participant == check_participant), ...
                    sum(stage0.participant   == check_participant), ...
                    sum(chronic.participant  == check_participant)];
    if any(panel_counts == 0)
        error("plot_pain_timecourse:emptyPanel", ...
              ["Participant %s has no '%s' surveys in %s for the baseline / " ...
               "Stage 0 / chronic panels respectively: %d / %d / %d."], ...
              check_participant, measure, survey_file, panel_counts);
    end
end

phase = readtable(fullfile(data_folder, "longitudinal_phase_metadata.csv"), ...
                  "TextType", "string");
phase.stage3_onset_datetime = datetime(phase.stage3_onset_datetime);
phase.endpoint_datetime     = datetime(phase.endpoint_datetime);

% ---- figure layout: one row per participant, 10 tile columns each ---------
% Row height is fixed so a one-participant figure is short and wide while the
% five-participant figure is tall; tiles butt together (spacing none) to give
% the broken-axis look.
tile_columns   = 10;
row_height_in  = 2.08;                 % keeps vertical white space small, for a
                                       % wide:short row aspect ratio
figure_height  = 1.2 + numel(participants) * row_height_in;
figure_handle  = figure("Units", "inches", "Position", [1, 1, 9, figure_height], ...
                        "Color", "w");
% "tight" keeps the three phase panels of each row visually adjacent (the
% broken-axis look) while leaving just enough room for per-row x-tick labels;
% a double-slash glyph is drawn at each seam to mark the axis break.
layout = tiledlayout(numel(participants), tile_columns, ...
                     "TileSpacing", "tight", "Padding", "compact");

for row = 1:numel(participants)
    participant = participants(row);
    base_offset = (row - 1) * tile_columns;

    % The CSVs store absolute timestamps only. Plot positions are days from
    % this participant's first survey, computed here so every series and every
    % phase boundary share one origin by construction.
    base_rows = baseline(baseline.participant == participant, :);
    s0_rows   = stage0(stage0.participant == participant, :);
    ch_rows   = chronic(chronic.participant == participant, :);

    % One participant's pre-operative baseline was collected in two visits
    % separated by a long gap. Both visits are part of that participant's
    % baseline window and both are used in the analysis, but plotting them
    % together devotes most of the panel to empty months, so only the later
    % visit is drawn and the timeline is numbered from it. This is a
    % presentation choice: it changes nothing in the data file.
    base_rows = keep_recent_baseline_visit(base_rows, ...
                                           baseline_split_gap_days, ...
                                           participant, ...
                                           baseline_split_participants);

    % The CSVs store absolute timestamps only. Plot positions are days from
    % this participant's first plotted survey, computed here so every series and
    % every phase boundary share one origin by construction.
    plotted_times = [base_rows.time; s0_rows.time; ch_rows.time];
    time_origin = min(plotted_times);
    to_days = @(t) days(t - time_origin);

    base_days = to_days(base_rows.time);
    s0_days   = to_days(s0_rows.time);
    ch_days   = to_days(ch_rows.time);

    prow = phase(phase.participant == participant, :);
    if height(prow) ~= 1
        error("plot_pain_timecourse:missingPhaseRow", ...
              "longitudinal_phase_metadata.csv holds %d rows for %s; expected exactly 1.", ...
              height(prow), participant);
    end
    stage3_onset_shared = to_days(prow.stage3_onset_datetime);
    endpoint_shared     = to_days(prow.endpoint_datetime);

    % ===================== panel A: baseline (pre-op) =====================
    axis_base = nexttile(base_offset + 1, [1, 1]);
    hold(axis_base, "on");
    base_values = double(base_rows.value);
    scatter(axis_base, base_days, base_values, point_size, baseline_color, ...
            "filled", "MarkerFaceAlpha", point_alpha);
    draw_trend(axis_base, base_days, base_values, baseline_smooth_span, ...
               gap_break_days, trend_color, trend_alpha);
    % Participant label as the row's y label, so the layout reserves room for
    % it and places the shared y label (added after the loop) further left.
    ylabel(axis_base, participant, "FontWeight", "bold", ...
           "FontSize", configuration.figure_font_size + 1);
    ylim(axis_base, [0, y_maximum]);
    if ~isempty(base_days)
        xlim(axis_base, [min(base_days) - 2, max(base_days) + 2]);
    end
    axis_base.Box = "off";
    set_panel_ticks(axis_base, base_days, configuration.figure_font_size);
    draw_axis_break(axis_base);   % // glyph at the right seam
    if row == 1; title(axis_base, "Baseline", "FontWeight", "normal"); end

    % ===================== panel B: Stage 0 inpatient =====================
    axis_s0 = nexttile(base_offset + 2, [1, 1]);
    hold(axis_s0, "on");
    s0_values = double(s0_rows.value);
    scatter(axis_s0, s0_days, s0_values, point_size, stage0_color, ...
            "filled", "MarkerFaceAlpha", point_alpha);
    draw_trend(axis_s0, s0_days, s0_values, stage0_smooth_span, ...
               gap_break_days, trend_color, trend_alpha);
    ylim(axis_s0, [0, y_maximum]);
    if ~isempty(s0_days)
        xlim(axis_s0, [min(s0_days) - 1, max(s0_days) + 1]);
    end
    axis_s0.YAxis.Visible = "off";
    axis_s0.Box = "off";
    set_panel_ticks(axis_s0, s0_days, configuration.figure_font_size);
    draw_axis_break(axis_s0);     % // glyph at the right seam
    if row == 1; title(axis_s0, "Stage 0", "FontWeight", "normal"); end

    % ===================== panel C: chronic (Stage 1-3) ===================
    axis_chronic = nexttile(base_offset + 3, [1, tile_columns - 2]);
    hold(axis_chronic, "on");
    if row == 1; first_chronic_axis = axis_chronic; end
    ch_values = double(ch_rows.value);
    condition = ch_rows.condition;

    % Cut at the plot cutoff (per-participant override date, else the metadata
    % endpoint). The override is a calendar date and is converted to this
    % participant's plot-day axis here.
    if isfield(plot_cutoff_date, participant)
        cutoff_shared = to_days(plot_cutoff_date.(participant));
    else
        cutoff_shared = endpoint_shared;
    end
    keep = ch_days <= cutoff_shared;
    % P5 gave VAS and MPQ on only one of four sham days. When a measure other
    % than the NRS is plotted, that single sham survey is removed from the
    % panel -- neither drawn nor smoothed -- and the four rescaled NRS sham
    % values are drawn instead (added below).
    if participant == "P5" && measure ~= "mayoNRS"
        keep = keep & condition ~= "sham";
    end
    ch_days = ch_days(keep); ch_values = ch_values(keep); condition = condition(keep);

    is_crossover = ch_days >= stage3_onset_shared;
    is_sham      = condition == "sham";
    is_washout   = condition == "washout" & ~is_crossover;   % washouts pre-Stage 3 only
    is_closed    = condition == "closed-loop" & ~is_crossover;
    % Every other chronic survey before Stage 3 is drawn as open-loop.
    is_open      = ~is_crossover & ~is_sham & ~is_washout & ~is_closed;

    % Every chronic survey must fall in exactly one class; overlapping or
    % unclaimed conditions would silently drop points from the panel.
    class_membership = double(is_crossover) + double(is_sham & ~is_crossover) + ...
                       double(is_washout) + double(is_closed) + double(is_open);
    if any(class_membership ~= 1)
        error("plot_pain_timecourse:conditionClassOverlap", ...
              ["%d of %d chronic surveys for %s are not in exactly one plotting " ...
               "class. Conditions present: %s."], sum(class_membership ~= 1), ...
              numel(class_membership), participant, ...
              strjoin(unique(condition)', ", "));
    end

    % Washout crosses are drawn FIRST, in light grey, so they read as background
    % against the condition dots rather than dominating a panel that carries
    % many of them. Pre-Stage 3 only.
    scatter(axis_chronic, ch_days(is_washout), ch_values(is_washout), sham_size, ...
            washout_color, "+", "LineWidth", 0.5);
    % open-loop / optimisation (grey), closed-loop (yellow)
    scatter(axis_chronic, ch_days(is_open), ch_values(is_open), point_size, ...
            open_loop_color, "filled", "MarkerFaceAlpha", point_alpha);
    scatter(axis_chronic, ch_days(is_closed), ch_values(is_closed), point_size, ...
            closed_loop_color, "filled", "MarkerFaceAlpha", point_alpha);
    % double-blind crossover terminal block (blue dots)
    scatter(axis_chronic, ch_days(is_crossover & ~is_sham), ...
            ch_values(is_crossover & ~is_sham), point_size, crossover_color, ...
            "filled", "MarkerFaceAlpha", point_alpha);
    % blinded shams: hollow circles (P5 sham NRS values are rescaled below)
    scatter(axis_chronic, ch_days(is_sham), ch_values(is_sham), sham_size, ...
            "MarkerEdgeColor", sham_edge_color, "MarkerFaceColor", "none", ...
            "LineWidth", 0.8);

    % When another measure is plotted, rescale P5's four sham NRS values onto
    % that measure's range so the sham days stay visible.
    if participant == "P5" && measure ~= "mayoNRS"
        scale = struct("painVAS", 10, "unpleasantVAS", 10, "MPQtotal", 4.5);
        if isfield(scale, measure) && ~isempty(p5_sham_nrs_rows)
            sham_day = to_days(p5_sham_nrs_rows.time);
            sham_val = double(p5_sham_nrs_rows.value) * scale.(measure);
            visible = sham_day <= cutoff_shared;
            scatter(axis_chronic, sham_day(visible), sham_val(visible), ...
                    sham_size, "MarkerEdgeColor", sham_edge_color, ...
                    "MarkerFaceColor", "none", "LineWidth", 0.8);
            % These circles are NRS values on the plotted measure's scale, not
            % measurements of that measure. Say so on the panel: a reader
            % cannot otherwise tell them from the surrounding data.
            if any(visible)
                text(axis_chronic, max(sham_day(visible)) + 20, ...
                     max(sham_val(visible)), ...
                     "sham: NRS \times" + string(scale.(measure)), ...
                     "Color", sham_edge_color, ...
                     "FontSize", configuration.figure_font_size - 1, ...
                     "HorizontalAlignment", "left", ...
                     "VerticalAlignment", "middle");
            end
        end
    end

    % centred moving-average on top (light red)
    draw_trend(axis_chronic, ch_days, ch_values, chronic_smooth_span, ...
               gap_break_days, trend_color, trend_alpha);
    % dashed blue line at Stage 3 onset, labelled once in the first row: the
    % same boundary is drawn in every row, so repeating the label would be
    % redundant ink.
    xline(axis_chronic, stage3_onset_shared, "--", "Color", crossover_color, ...
          "LineWidth", 1.0, "Alpha", 0.8);
    if row == 1
        % Horizontal, left of the boundary and low in the panel: rotated text on
        % the line itself crossed the line it labels and overran the axis.
        text(axis_chronic, stage3_onset_shared - 25, y_maximum * 0.06, ...
             "crossover begins", "Color", crossover_color, ...
             "FontSize", configuration.figure_font_size - 1, ...
             "HorizontalAlignment", "right", "VerticalAlignment", "bottom");
        % The three stimulation conditions are named on the marks in the top row
        % so the reader decodes colour here rather than travelling to the footer
        % legend. Each label is placed at the median day of its own points.
        % Open- and closed-loop are interleaved through the optimisation period,
        % so their medians nearly coincide and a label at each median collides.
        % They are stacked on separate lines instead, left of the crossover
        % boundary, with the crossover label over its own block.
        label_specs = {is_open,   "open-loop",    open_loop_color,   0.97; ...
                       is_closed, "closed-loop",  closed_loop_color, 0.88; ...
                       is_crossover & ~is_sham, "double-blind", crossover_color, 0.97};
        for spec = 1:size(label_specs, 1)
            mask = label_specs{spec, 1};
            if ~any(mask); continue; end
            text(axis_chronic, median(ch_days(mask)), ...
                 y_maximum * label_specs{spec, 4}, ...
                 label_specs{spec, 2}, "Color", label_specs{spec, 3}, ...
                 "FontSize", configuration.figure_font_size - 1, ...
                 "FontWeight", "bold", "HorizontalAlignment", "center", ...
                 "VerticalAlignment", "top");
        end
    end

    ylim(axis_chronic, [0, y_maximum]);
    xlim(axis_chronic, [min(ch_days) - 5, cutoff_shared + 5]);
    axis_chronic.YAxis.Visible = "off";
    axis_chronic.Box = "off";
    % Uniform chronic ticks: the participant's own first chronic day, then every
    % 500 days, plus the participant's final day so the last x value is always
    % labelled even though it falls on an irregular spacing.
    set_chronic_ticks(axis_chronic, min(ch_days), max(ch_days), cutoff_shared, ...
                      configuration.figure_font_size);
    if row == 1; title(axis_chronic, "Stages 1-3 (chronic)", "FontWeight", "normal"); end
    if row == numel(participants); xlabel(axis_chronic, "Days since first baseline survey"); end
end

title(layout, figure_title);
% The shared y label belongs to the layout, which places it left of the
% participant labels. A figure annotation would be left out of the exported
% file by R2024b.
ylabel(layout, y_label, "FontSize", configuration.figure_font_size + 1);
draw_condition_legend(layout, first_chronic_axis, open_loop_color, ...
                      closed_loop_color, crossover_color, washout_color, ...
                      sham_edge_color);

% ---- export ---------------------------------------------------------------
% Vector PDF is the archival copy; a light PNG is written alongside it for quick
% viewing (a raster preview does not need the vector export options).
output_file = fullfile(configuration.figure_directory, output_basename + ".pdf");
exportgraphics(figure_handle, output_file, configuration.figure_export_options{:});
output_png = fullfile(configuration.figure_directory, output_basename + ".png");
exportgraphics(figure_handle, output_png, "Resolution", 200);

% SVG for figure editors: every string stays a <text> element, so titles,
% labels and annotations remain selectable and editable rather than outlined
% into paths.
% exportgraphics writes SVG only from R2025a, so earlier releases use print.
output_svg = fullfile(configuration.figure_directory, output_basename + ".svg");
if isMATLABReleaseOlderThan("R2025a")
    print(figure_handle, output_svg, "-dsvg", "-vector");
else
    exportgraphics(figure_handle, output_svg, "ContentType", "vector");
end
end


% =========================================================================
% helpers
% =========================================================================
function set_panel_ticks(target_axis, day_values, font_size)
%SET_PANEL_TICKS  Force a few readable x-tick labels on a phase panel.
% With tight tile spacing MATLAB drops shared tick labels, so we set an
% explicit small set of ticks (min, middle, max day) per panel and keep the
% labels visible for every participant row.
if isempty(day_values); return; end
low  = min(day_values);
high = max(day_values);
if high <= low
    ticks = round(low);
else
    % The interior tick goes at the panel midpoint, but a participant whose
    % pre-operative surveys come in two widely separated blocks has no data
    % near that midpoint -- a tick there labels empty axis. Snap the interior
    % tick to the nearest real survey day so every tick marks a day on which
    % the participant actually reported. If the nearest survey day sits within
    % a fifth of the span of either end tick, its label would collide with that
    % end label, so the interior tick is dropped and the panel keeps just its
    % first and last day.
    midpoint = (low + high) / 2;
    [~, nearest_index] = min(abs(day_values - midpoint));
    interior = day_values(nearest_index);
    if (interior - low) < 0.2 * (high - low) || (high - interior) < 0.2 * (high - low)
        ticks = unique(round([low, high]));
    else
        ticks = unique(round([low, interior, high]));
    end
end
target_axis.XTick = ticks;
target_axis.XTickLabel = string(ticks);
target_axis.XAxis.FontSize = font_size - 1;
target_axis.XTickLabelRotation = 0;
end


function set_chronic_ticks(target_axis, first_day, last_day, axis_max, font_size)
%SET_CHRONIC_TICKS  Uniform chronic x-ticks plus the participant's final day.
% The participant's own first chronic day is always marked (a unique value per
% participant), subsequent ticks fall on a fixed 500-day grid so intervals read
% identically across the cohort, and the participant's LAST day is always added
% as a terminal tick so the final x value is labelled even though it lands on an
% irregular spacing. Grid ticks within 150 d of the first day, and any within
% 150 d of the terminal day, are dropped so labels do not crowd.
first_day = round(first_day);
last_day  = round(last_day);
if isempty(first_day)
    return;
elseif isempty(last_day) || last_day <= first_day
    ticks = first_day;
else
    grid_ticks = 500 * ceil(first_day / 500) : 500 : axis_max;
    grid_ticks = grid_ticks(grid_ticks - first_day > 150 & last_day - grid_ticks > 150);
    ticks = unique([first_day, grid_ticks, last_day]);
end
target_axis.XTick = ticks;
target_axis.XTickLabel = string(ticks);
target_axis.XAxis.FontSize = font_size - 1;
target_axis.XTickLabelRotation = 0;
end


function draw_axis_break(target_axis)
%DRAW_AXIS_BREAK  Draw a double-slash break glyph at the panel's right edge.
% Marks that the x-axis is discontinuous between this panel and the next. The
% glyph is lifted just above the axis baseline (5% of the y-range) so it does
% not sit on top of the x-tick labels, which live below the axis line.
x_limits = xlim(target_axis);
y_limits = ylim(target_axis);
x_at = x_limits(2);
y_at = y_limits(1) + 0.05 * (y_limits(2) - y_limits(1));
text(target_axis, x_at, y_at, "//", "HorizontalAlignment", "center", ...
     "VerticalAlignment", "middle", "FontSize", 10, "FontWeight", "bold", ...
     "Clipping", "off", "BackgroundColor", "none");
end


function draw_trend(target_axis, day_values, measure_values, smooth_span, ...
                    gap_days, line_color, line_alpha)
%DRAW_TREND  Centred moving average over SMOOTH_SPAN survey points.
% The span is a fixed number of survey points, not a number of days: 14 in
% baseline, 21 in Stage 0, 21 in chronic. The line is broken across gaps
% longer than GAP_DAYS so it is not drawn straight across long silences.
finite = isfinite(day_values) & isfinite(measure_values);
day_values = day_values(finite); measure_values = measure_values(finite);
if numel(day_values) < 3; return; end
[day_sorted, order] = sort(day_values);
value_sorted = measure_values(order);

% Centred moving average over a fixed number of survey points.
span  = max(3, round(smooth_span));
trend = movmean(value_sorted, span);

% Break the plotted line wherever two consecutive surveys are more than
% gap_days apart, so the trend does not bridge multi-month gaps.
segment_start = 1;
for k = 2:numel(day_sorted)
    if day_sorted(k) - day_sorted(k - 1) > gap_days
        plot_segment(target_axis, day_sorted, trend, segment_start, k - 1, ...
                     line_color, line_alpha);
        segment_start = k;
    end
end
plot_segment(target_axis, day_sorted, trend, segment_start, ...
             numel(day_sorted), line_color, line_alpha);
end


function plot_segment(target_axis, day_sorted, trend, from_index, to_index, ...
                      line_color, line_alpha)
%PLOT_SEGMENT  Draw one unbroken piece of the moving-average line.
if to_index < from_index; return; end
line_handle = plot(target_axis, day_sorted(from_index:to_index), ...
                   trend(from_index:to_index), "Color", line_color, ...
                   "LineWidth", 1.5);
line_handle.Color(4) = line_alpha;
end


function draw_condition_legend(layout, host_axis, open_color, closed_color, ...
                               crossover_color, washout_color, sham_color)
%DRAW_CONDITION_LEGEND  Shared legend beneath the stacked rows.
% Proxy handles are drawn (as NaN points) on an existing axis so no extra tile
% is consumed; the legend is then attached to the whole layout and placed in
% its south tile. Display labels are set here: the survey table stores
% lower-case condition names and carries no presentation text.
hold(host_axis, "on");
h1 = scatter(host_axis, NaN, NaN, 30, open_color, "filled");
h2 = scatter(host_axis, NaN, NaN, 30, closed_color, "filled");
h3 = scatter(host_axis, NaN, NaN, 30, crossover_color, "filled");
h4 = scatter(host_axis, NaN, NaN, 40, washout_color, "+", "LineWidth", 0.5);
h5 = scatter(host_axis, NaN, NaN, 40, "MarkerEdgeColor", sham_color, ...
             "MarkerFaceColor", "none", "LineWidth", 0.8);
shared_legend = legend(host_axis, [h1, h2, h3, h4, h5], ...
       ["Open-loop", "Closed-loop", "Double-blind crossover", ...
        "Washout (0 mA)", "Sham (0 mA)"], ...
       "Orientation", "horizontal", "NumColumns", 5, "Box", "off");
shared_legend.Layout.Tile = "south";
end


function scores = nan_frozen_vas50_runs(scores, run_threshold)
%NAN_FROZEN_VAS50_RUNS  Blank painVAS == 50 values stuck across consecutive surveys.
%   For each participant, surveys are taken in time order and consecutive runs
%   of painVAS == 50 are found. Any run LONGER THAN run_threshold surveys is set
%   to NaN (the un-moved-slider artifact); shorter runs, including isolated
%   single 50s, are kept as potentially genuine moderate ratings.
scores = sortrows(scores, ["participant", "time"]);
participant_list = unique(scores.participant, "stable");
for participant_index = 1:numel(participant_list)
    rows = find(scores.participant == participant_list(participant_index));
    is_fifty = scores.value(rows) == 50;
    % Walk the participant's surveys, marking runs of consecutive 50s.
    run_start = 1;
    while run_start <= numel(rows)
        if ~is_fifty(run_start)
            run_start = run_start + 1;
            continue
        end
        run_stop = run_start;
        while run_stop < numel(rows) && is_fifty(run_stop + 1)
            run_stop = run_stop + 1;
        end
        run_length = run_stop - run_start + 1;
        if run_length > run_threshold
            scores.value(rows(run_start:run_stop)) = NaN;
        end
        run_start = run_stop + 1;
    end
end
end


function visit_rows = keep_recent_baseline_visit(visit_rows, gap_days, ...
                                                participant, split_participants)
%KEEP_RECENT_BASELINE_VISIT  Drop an early pre-operative visit for plotting.
%   Returns VISIT_ROWS unchanged unless PARTICIPANT is one of
%   SPLIT_PARTICIPANTS and its surveys contain a gap longer than GAP_DAYS. In
%   that case only the rows on or after the last such gap are returned, so the
%   panel spans the later visit rather than the empty months between visits.
%
%   This affects the figure only. The dropped surveys remain in the data file
%   and in every analysis that reads it.
if ~ismember(participant, split_participants) || height(visit_rows) < 2
    return
end
visit_rows = sortrows(visit_rows, "time");
gaps = [0; days(diff(visit_rows.time))];
last_break = find(gaps > gap_days, 1, "last");
if ~isempty(last_break)
    visit_rows = visit_rows(last_break:end, :);
end
end
