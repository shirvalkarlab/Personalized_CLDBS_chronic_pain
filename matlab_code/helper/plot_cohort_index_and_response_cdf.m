function figure_handle = plot_cohort_index_and_response_cdf(config, cohort_groups)
% PLOT_COHORT_INDEX_AND_RESPONSE_CDF  Figure 6 panels a-c, closed-loop versus sham.
%
%   figure_handle = PLOT_COHORT_INDEX_AND_RESPONSE_CDF(config, cohort_groups)
%   draws three panels:
%     a) Uniform NRS -- per-participant box (interquartile range and median)
%        of the daily percent change from pre-trial baseline in mayoNRS,
%        closed-loop versus sham.
%     b) Personalized index metric -- the same, on each participant's
%        prespecified index metric.
%     c) Response curve -- proportion of closed-loop and sham participant-days
%        reaching each reduction in NRS (daily values pooled across
%        participants).
%
%   Data and definitions match Extended Data Table 2: the index metric per
%   participant comes from config.analysis.patient_best_metric_table, the
%   percent change from the baseline_percent_diff_* columns of cohort_groups,
%   and the sham exclusions from config.analysis.sham_nan_exclusions (P3 did
%   not undergo sham testing; P5 recorded VAS and MPQ on only one of four sham
%   days).
%
%   INPUTS
%     config        - configuration struct from configure_repository, extended
%                     by the caller with .analysis.patient_best_metric_table and
%                     .analysis.sham_nan_exclusions; also uses .figure_directory
%                     and .figure_export_options.
%     cohort_groups - per-observation table (one row per participant-day-
%                     condition) read from
%                     data/figure6_double_blind_outcomes/cohort_daily_percent_change.csv,
%                     with columns patient_id, stim_group and
%                     baseline_percent_diff_<metric>.
%
%   DEPENDENCIES
%     stim_condition_colors, condition_box_face, draw_summary_box

    % Participants are identified by their de-identified labels throughout; the
    % cohort table, the index-metric table, and the sham-exclusion rules all key
    % on these same labels.
    labs = {'P1','P2','P3','P4','P5'};
    pats = labs;
    npat = numel(pats);

    cond_colors = stim_condition_colors();
    CL_COL   = cond_colors('closed-loop');
    SHAM_COL = cond_colors('sham');
    GRN      = [0.180 0.545 0.341];   % response-threshold green (annotation only)

    % Panels a & b colour by CONDITION (not participant): every closed-loop box
    % + swarm uses the shared blue and every sham the shared neutral grey, so the
    % closed-loop-vs-sham contrast matches the response-CDF panel and the other
    % cohort/subject figures.

    pbm   = config.analysis.patient_best_metric_table;
    shamx = config.analysis.sham_nan_exclusions;
    nrs_col = 'baseline_percent_diff_mayoNRS';

    pid = string(cohort_groups.patient_id);
    sg  = string(cohort_groups.stim_group);

    %% Gather per-participant daily distributions (index + uniform NRS), applying
    %  the sham exclusions so an excluded participant simply has no sham data.
    idx_col  = cell(1,npat);  idx_abbr = cell(1,npat);
    cl_idx = cell(1,npat); sh_idx = cell(1,npat);   % daily % change, index metric
    cl_nrs = cell(1,npat); sh_nrs = cell(1,npat);   % daily % change, uniform NRS
    for k = 1:npat
        idx_col{k}  = char(pbm.(pats{k}));
        idx_abbr{k} = abbrev_metric(idx_col{k});
        cl_idx{k} = daily_vals(cohort_groups, pid, sg, pats{k}, idx_col{k}, 'closed-loop', shamx);
        sh_idx{k} = daily_vals(cohort_groups, pid, sg, pats{k}, idx_col{k}, 'sham',        shamx);
        cl_nrs{k} = daily_vals(cohort_groups, pid, sg, pats{k}, nrs_col,    'closed-loop', shamx);
        sh_nrs{k} = daily_vals(cohort_groups, pid, sg, pats{k}, nrs_col,    'sham',        shamx);
    end

    %% Figure (single row of 3 panels, publication proportions)
    fig = figure('Units','inches','Position',[0 0 11 3.3],'Color','w');
    % Arial for every text object in the figure (axes, ticks, titles, labels,
    % legend, annotations) so the exported figure has a single consistent font.
    set(fig, 'DefaultAxesFontName', 'Arial', 'DefaultTextFontName', 'Arial', ...
             'DefaultLegendFontName', 'Arial');
    tl  = tiledlayout(fig, 1, 3, 'TileSpacing','compact', 'Padding','compact');
    % Font sizes: axes tick labels = 10, titles = 11 (title uses ax_font+1).
    ax_font = 10; lab_font = 8; title_font = 11;

    % Two-line participant/metric labels. Use tex \newline (NOT a raw newline,
    % which MATLAB flattens into separate tick labels).
    index_labels = cellfun(@(l,m) sprintf('%s\\newline%s', l, m), labs, idx_abbr, 'UniformOutput', false);

    % Shared y-limits for panels a & b (directly comparable), sized to show every
    % box + whisker with no clipping.
    ylims = box_ylim([cl_nrs, sh_nrs, cl_idx, sh_idx]);

    ax_a = nexttile(tl, 1);
    draw_box_panel(ax_a, cl_nrs, sh_nrs, labs, CL_COL, SHAM_COL, GRN, ax_font, lab_font, ylims);
    ylabel(ax_a, sprintf('%% change from baseline\n(negative = pain reduction)'), 'FontSize', ax_font);
    title(ax_a, 'a  Uniform NRS', 'FontSize', title_font, 'FontWeight','bold');

    ax_b = nexttile(tl, 2);
    draw_box_panel(ax_b, cl_idx, sh_idx, index_labels, CL_COL, SHAM_COL, GRN, ax_font, lab_font, ylims);
    title(ax_b, 'b  Personalized index metric', 'FontSize', title_font, 'FontWeight','bold');

    ax_c = nexttile(tl, 3);
    draw_cdf_panel(ax_c, cl_nrs, sh_nrs, CL_COL, SHAM_COL, GRN, ax_font, lab_font);
    title(ax_c, 'c  Response CDF (daily NRS): closed-loop vs sham', 'FontSize', title_font, 'FontWeight','bold');

    %% Save the figure as a PDF into the repository figures/ folder.
    figure_handle = fig;
    output_file = fullfile(config.figure_directory, ...
        'Figure6_double_blind_outcomes.pdf');
    if isfield(config, 'figure_export_options')
        exportgraphics(fig, output_file, config.figure_export_options{:});
    else
        exportgraphics(fig, output_file, 'Resolution', 600);
    end
end

% =========================================================================
function v = daily_vals(cg, pid, sg, patient, col, cond, shamx)
%DAILY_VALS Daily % change values for one participant/condition/metric column,
%   NaNs dropped. Sham is emptied when the metric family is excluded for that
%   participant (config.analysis.sham_nan_exclusions).
    if ~ismember(col, cg.Properties.VariableNames); v = []; return; end
    m = pid == patient & sg == cond;
    v = cg.(col)(m);
    v = v(~isnan(v));
    if strcmp(cond, 'sham') && sham_excluded(shamx, patient, col); v = []; end
end

% =========================================================================
function tf = sham_excluded(shamx, patient, col)
%SHAM_EXCLUDED True if this participant's sham is invalid for the metric's scale
%   family (NRS / VAS / MPQ), matching the Extended Data Table 2 / cohort
%   convention.
    fam = 'NRS';
    if contains(col, 'VAS'); fam = 'VAS'; elseif contains(col, 'MPQ'); fam = 'MPQ'; end
    tf = isfield(shamx, patient) && ismember(fam, shamx.(patient));
end

% =========================================================================
function draw_box_panel(ax, cl, sh, xlabs, CL_COL, SHAM_COL, GRN, ax_font, lab_font, ylims)
%DRAW_BOX_PANEL Closed-loop vs sham IQR boxes of daily % change, one box per
%   participant per condition, with 0 reference, >=30%/>=50% response zones, and
%   "no sham" markers. Colour encodes CONDITION: every closed-loop box + swarm is
%   the shared blue (CL_COL), every sham the shared neutral grey (SHAM_COL).
    hold(ax, 'on');
    n = numel(cl);
    off = 0.19;
    y_nosham = ylims(2) - 0.10 * diff(ylims);

    % >=30% / >=50% response thresholds, marked by dashed lines only (no
    % shaded fill below them -- the shading added little and muddied the box
    % colours).
    yline(ax, 0,   '-',  'Color', [0 0 0],   'LineWidth', 0.7);
    yline(ax, -30, '--', 'Color', GRN, 'LineWidth', 0.7, 'Alpha', 0.9);
    yline(ax, -50, '--', 'Color', GRN, 'LineWidth', 0.7, 'Alpha', 0.9);

    % Closed-loop and sham boxes (one per participant, offset side by side).
    % Colour encodes condition: closed-loop blue, sham neutral grey.
    draw_boxes(ax, cl, (1:n) - off, CL_COL);
    draw_boxes(ax, sh, (1:n) + off, SHAM_COL);

    % "no sham" where a participant has no valid sham data.
    for k = 1:n
        if isempty(sh{k})
            text(ax, k + off, y_nosham, sprintf('no\nsham'), 'HorizontalAlignment','center', ...
                'VerticalAlignment','top', 'FontSize', lab_font-1.5, 'Color', SHAM_COL);
        end
    end

    % Plain ASCII '>=' with Interpreter 'none' (NOT tex '\geq'): a tex math
    % symbol forces MATLAB's vector backend to outline EVERY label in the figure
    % to <path> glyphs, so words can no longer be selected/edited. ASCII text
    % keeps every label a contiguous, editable <text> object in the SVG.
    text(ax, n+0.55, -30, '>=30%', 'Color', GRN, 'FontSize', lab_font-1, ...
        'HorizontalAlignment','right', 'VerticalAlignment','bottom', 'Interpreter','none');
    text(ax, n+0.55, -50, '>=50%', 'Color', GRN, 'FontSize', lab_font-1, ...
        'HorizontalAlignment','right', 'VerticalAlignment','bottom', 'Interpreter','none');

    set(ax, 'XTick', 1:n, 'XTickLabel', xlabs, 'FontSize', ax_font, ...
        'TickLabelInterpreter', 'tex', 'XTickLabelRotation', 0, ...
        'Box', 'off', 'TickDir', 'out', 'TickLength', [0.02 0.02]);
    xlim(ax, [0.4, n+0.6]);
    ylim(ax, ylims);
    hold(ax, 'off');
end

% =========================================================================
function ylims = box_ylim(groups)
%BOX_YLIM Y-limits covering every group's box + whiskers (whisker = most extreme
%   value within 1.5*IQR of the box, i.e. what boxchart actually draws), with a
%   little padding. Guarantees the 0 / -30 / -50 reference lines stay in view.
    lo = inf; hi = -inf;
    for k = 1:numel(groups)
        v = groups{k};
        if numel(v) < 1; continue; end
        q1 = prctile(v,25); q3 = prctile(v,75); w = 1.5*(q3-q1);
        hi = max(hi, max(v(v <= q3 + w)));
        lo = min(lo, min(v(v >= q1 - w)));
    end
    if ~isfinite(lo); lo = -100; hi = 20; end
    lo = min(lo, -50); hi = max(hi, 5);           % keep response zones/0 visible
    r  = hi - lo;
    ylims = [lo - 0.05*r, hi + 0.09*r];           % pad; extra top room for labels
end

% =========================================================================
function draw_boxes(ax, groups, xpos, col)
%DRAW_BOXES One IQR box + median per participant for a single condition.
%   Shared cohort box look (Figure 6a,b): the summary box uses the pale grey
%   (sham) / pale
%   blue (closed-loop) fill with a black outline (draw_summary_box), keyed off
%   the condition colour COL. The condition is carried by the BOX fill; the
%   individual daily scores are drawn as a single-colour BLACK swarm on top so
%   the points read cleanly against either box fill (grey sham or pale blue
%   closed-loop) without a per-condition colour that would clash with the box.
    cond     = condition_from_color(col);
    box_face = condition_box_face(cond);
    swarm_col = [0 0 0];              % all daily-score dots black
    for k = 1:numel(groups)
        v = groups{k};
        if numel(v) < 1; continue; end
        % Box first, swarm second, so the individual daily-score dots sit in
        % the FOREGROUND -- in front of the box fill and the median line.
        % Marker size is 4.2, small enough that the dots do not obscure the
        % box statistics they overlay.
        draw_summary_box(ax, v(:), xpos(k), 0.30, box_face);
        swarmchart(ax, xpos(k) * ones(numel(v),1), v(:), 4.2, swarm_col, 'filled', ...
            'MarkerFaceAlpha', 0.45, 'MarkerEdgeColor', 'none', ...
            'XJitter', 'density', 'XJitterWidth', 0.24);
    end
end

% =========================================================================
function cond = condition_from_color(col)
%CONDITION_FROM_COLOR Map the shared sham grey to 'sham', everything else to
%   'closed-loop', so condition_box_face picks the right pale fill.
    cond = 'closed-loop';
    if isequal(round(col(:)', 3), [0.60 0.60 0.60]); cond = 'sham'; end
end

% =========================================================================
function draw_cdf_panel(ax, cl_nrs, sh_nrs, CL_COL, SHAM_COL, GRN, ax_font, lab_font)
%DRAW_CDF_PANEL Response curve built from pooled daily NRS values (not the
%   five per-participant medians): the proportion of participant-days whose NRS
%   reduction from baseline is >= x, closed-loop vs sham (228 closed-loop and
%   58 sham days in the shipped data).
    cl_red = -vertcat(cl_nrs{:});   % daily % reduction (sign flipped), closed-loop
    sh_red = -vertcat(sh_nrs{:});   % daily % reduction (sign flipped), sham
    xg  = (0:0.5:100)';
    Scl = arrayfun(@(xx) mean(cl_red >= xx), xg);
    Ssh = arrayfun(@(xx) mean(sh_red >= xx), xg);

    hold(ax, 'on');
    xline(ax, 30, '--', 'Color', GRN, 'LineWidth', 0.7, 'Alpha', 0.9);
    xline(ax, 50, '--', 'Color', GRN, 'LineWidth', 0.7, 'Alpha', 0.9);
    % ASCII '>=' (Interpreter 'none') keeps text editable in the SVG; a tex
    % '\geq' would outline every label in the figure to path glyphs.
    text(ax, 30, 1.02, '>=30%', 'Color', GRN, 'FontSize', lab_font-1, ...
        'HorizontalAlignment','center', 'Interpreter','none');
    text(ax, 50, 1.02, '>=50%', 'Color', GRN, 'FontSize', lab_font-1, ...
        'HorizontalAlignment','center', 'Interpreter','none');
    % Daily-value pooling makes the step size tiny, so a plain line reads as a
    % smooth curve (stairs would only add invisible micro-steps).
    hCL = plot(ax, xg, Scl, 'Color', CL_COL,   'LineWidth', 1.8);
    hSH = plot(ax, xg, Ssh, 'Color', SHAM_COL, 'LineWidth', 1.8);

    set(ax, 'FontSize', ax_font, 'Box', 'off', 'TickDir', 'out', 'TickLength', [0.02 0.02]);
    xlim(ax, [0 100]); ylim(ax, [0 1.06]);
    xlabel(ax, 'NRS reduction from baseline (%)', 'FontSize', ax_font);
    ylabel(ax, sprintf('Proportion of days\nachieving >= reduction'), 'FontSize', ax_font, 'Interpreter','none');
    legend(ax, [hCL hSH], {sprintf('Closed-loop (%d days)', numel(cl_red)), ...
        sprintf('Sham (%d days)', numel(sh_red))}, 'Location','northeast', ...
        'FontSize', lab_font, 'Box','off');
    hold(ax, 'off');
end

% =========================================================================
function s = abbrev_metric(col)
%ABBREV_METRIC Short scale label for a baseline_percent_diff_<metric> column.
    s = strrep(col, 'baseline_percent_diff_', '');
    switch s
        case 'MPQtotal'; s = 'MPQ';
        case 'mayoNRS';  s = 'NRS';
        case 'nocNRS';   s = 'nocNRS';
        case 'painVAS';  s = 'VAS';
        otherwise           % other columns keep their own name
    end
end
