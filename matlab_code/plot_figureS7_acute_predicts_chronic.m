function figure_handle = plot_figureS7_acute_predicts_chronic(configuration)
% PLOT_FIGURES7_ACUTE_PREDICTS_CHRONIC  Does acute (temporary-trial) analgesia
% predict chronic closed-loop benefit? (manuscript Supplementary Figure S7,
% exploratory).
%
%   figure_handle = plot_figureS7_acute_predicts_chronic(configuration) draws
%   the five-panel figure testing whether the analgesia seen acutely during the
%   temporary Stage 0 trial, at the site that was later implanted, predicts the
%   long-term benefit that same site delivers during chronic closed-loop
%   stimulation:
%       a) Same-metric scatter  - acute numeric rating scale percent change
%          versus chronic closed-loop numeric rating scale percent change, one
%          point per implanted hemisphere.
%       b) Cross-metric scatter - acute pain visual-analog percent change versus
%          chronic closed-loop index-metric percent change.
%       c) Participant jackknife - primary correlation recomputed dropping each
%          participant.
%       d) Hemisphere jackknife - dropping each hemisphere.
%       e) Cluster jackknife     - dropping each participant's hemispheres
%   Panels c-e each carry two series: the single-scale (NRS) result as an open
%   circle and the pooled three-scale result (NRS, intensity VAS, McGill total)
%   as a filled blue circle.
%
%   HOW THE STATISTICS ARE COMPUTED (all done here, from the data files, so a
%   reader can follow every step; nothing is hard-coded):
%     - Spearman rank correlation (robust to the acute/chronic scale mismatch
%       and to outliers).
%     - A two-sided permutation p value (exact enumeration when the sample is
%       small enough, otherwise a large random sample).
%     - A bias-corrected and accelerated bootstrap 95% confidence interval.
%     - Leave-one-out and cluster (leave-one-participant-out) jackknives, and a
%       cluster-aware permutation p that keeps each participant's two
%       hemispheres together.
%
%   WHY EXPLORATORY: only five participants were implanted, and the implanted
%   target was chosen because it gave the best acute response, so the acute
%   predictor is range-restricted. Both facts bias any correlation toward zero,
%   so results are reported as effect sizes with confidence intervals, never as a
%   single p value.
%
%   INPUT
%     configuration   Struct from configure_repository. Uses .data_directory,
%                     .figure_directory, and .figure_export_options.
%
%   OUTPUT
%     figure_handle   Handle to the created Figure S7.
%
%   DATA (in data/figureS7_acute_predicts_chronic/)
%     acute_predicts_chronic_hemisphere.csv   one row per implanted hemisphere,
%         with the acute and chronic percent-change values that are correlated.
%     acute_predicts_chronic_participant.csv  one row per participant (acute and
%         chronic values pooled across that participant's implanted targets).
%     acute_predicts_chronic_pooled_participant.csv and
%     acute_predicts_chronic_pooled_hemisphere.csv  the same, with the three
%         pain scales (NRS, intensity VAS, McGill total) stacked.
%
%   The random-number generator is seeded (rng(0, "twister")) for the
%   bootstrap intervals and sampled permutations, and restored on exit. The
%   statistics are printed to the Command Window.

% ---- Fixed random seed; the caller's generator state is restored on exit --
previous_generator = rng();
restore_generator  = onCleanup(@() rng(previous_generator));
rng(0, "twister");

% ---- Load the paired acute-vs-chronic tables ---------------------------
data_directory = fullfile(configuration.data_directory, ...
    'figureS7_acute_predicts_chronic');
hemi = readtable(fullfile(data_directory, ...
    'acute_predicts_chronic_hemisphere.csv'), 'TextType', 'string');
part = readtable(fullfile(data_directory, ...
    'acute_predicts_chronic_participant.csv'), 'TextType', 'string');

% Pooled tables stack the three collected pain scales (NRS, intensity VAS,
% McGill total) so the acute predictor spans a wider range of values.
pooled_part = readtable(fullfile(data_directory, ...
    'acute_predicts_chronic_pooled_participant.csv'), 'TextType', 'string');
pooled_hemi = readtable(fullfile(data_directory, ...
    'acute_predicts_chronic_pooled_hemisphere.csv'), 'TextType', 'string');

% Participants are labelled P1..P5; the label map is the identity.
labels = unique([string(hemi.patient_id); string(part.patient_id)], 'stable');
label_of = containers.Map(cellstr(labels), cellstr(labels));

% ---- Compute the correlations and jackknives (participant + hemisphere) -
participant_stats = struct();
participant_stats.primary   = correlation_report(part.acute_pct_NRS, part.chronic_pct_NRS);

hemisphere_stats = struct();
hemisphere_stats.primary   = correlation_report(hemi.acute_pct_NRS, hemi.chronic_pct_NRS);
hemisphere_stats.secondary = correlation_report(hemi.acute_pct_VAS, hemi.chronic_pct_index);
% Cluster-aware permutation (permute whole participants) and the cluster drop.
hemisphere_stats.primary.cluster_p = cluster_permutation_p(hemi.acute_pct_NRS, ...
    hemi.chronic_pct_NRS, hemi.patient_id);
hemisphere_stats.secondary.cluster_p = cluster_permutation_p(hemi.acute_pct_VAS, ...
    hemi.chronic_pct_index, hemi.patient_id);
hemisphere_stats.primary.drop_patient = leave_one_group_out(hemi.acute_pct_NRS, ...
    hemi.chronic_pct_NRS, hemi.patient_id);

% Pooled-scale versions of the same three jackknives, overlaid on panels c-e.
pooled_stats = struct();
pooled_stats.participant = correlation_report(pooled_part.acute_pct, ...
    pooled_part.chronic_pct);
pooled_stats.participant.drop_patient = leave_one_group_out(pooled_part.acute_pct, ...
    pooled_part.chronic_pct, pooled_part.patient_id);
pooled_stats.hemisphere = correlation_report(pooled_hemi.acute_pct, ...
    pooled_hemi.chronic_pct);
pooled_stats.hemisphere.drop_hemisphere = leave_one_group_out(pooled_hemi.acute_pct, ...
    pooled_hemi.chronic_pct, pooled_hemi.patient_id + pooled_hemi.hemisphere);
pooled_stats.hemisphere.drop_patient = leave_one_group_out(pooled_hemi.acute_pct, ...
    pooled_hemi.chronic_pct, pooled_hemi.patient_id);

% ---- Print the statistics ----------------------------------------------
print_report("participant", "NRS vs NRS", participant_stats.primary);
print_report("hemisphere", "NRS vs NRS", hemisphere_stats.primary);
print_report("hemisphere", "VAS vs index", hemisphere_stats.secondary);
print_report("pooled participant", "three scales", pooled_stats.participant);
print_report("pooled hemisphere", "three scales", pooled_stats.hemisphere);

% ---- Draw Figure S7 ----------------------------------------------------
figure_handle = combined_figure(part, participant_stats, hemi, ...
    hemisphere_stats, label_of, pooled_stats);

% ---- Save as a PDF into the repository figures/ folder -----------------
output_file = fullfile(configuration.figure_directory, ...
    'FigureS7_acute_predicts_chronic.pdf');
if isfield(configuration, 'figure_export_options')
    exportgraphics(figure_handle, output_file, configuration.figure_export_options{:});
else
    exportgraphics(figure_handle, output_file, 'Resolution', 600);
end

end


% =========================================================================
% Statistics and plotting helpers for this figure (participant colours are
% keyed on the de-identified P labels).
% =========================================================================
function print_report(level, scales, report)
%PRINT_REPORT Print one line of Figure S7 statistics to the Command Window.
    fprintf("Figure S7, %s level, %s: n = %d, rho = %+.3f, 95%% CI [%+.3f, %+.3f], p = %.4f", ...
        level, scales, report.n, report.rho, report.ci(1), report.ci(2), report.p);
    if isfield(report, "cluster_p")
        fprintf(", cluster p = %.4f", report.cluster_p);
    end
    fprintf("\n");
end

% =========================================================================
function report = correlation_report(x, y)
%CORRELATION_REPORT Spearman rho + exact/sampled permutation p + BCa 95% CI +
%   leave-one-out jackknife, for one pair of columns.
    x = x(:); y = y(:);
    keep = ~isnan(x) & ~isnan(y);
    x = x(keep); y = y(keep);
    n = numel(x);
    report.n = n;
    report.rho = corr(x, y, 'Type', 'Spearman');
    report.p   = permutation_p(x, y);
    report.ci  = bca_interval(x, y);
    report.leave_one_out = zeros(n, 1);
    for i = 1:n
        keep_i = true(n, 1); keep_i(i) = false;
        report.leave_one_out(i) = corr(x(keep_i), y(keep_i), 'Type', 'Spearman');
    end
end

% =========================================================================
function p = permutation_p(x, y)
%PERMUTATION_P Two-sided permutation p for Spearman rho. Exact (all orderings)
%   when n! is small, otherwise a large random sample.
    n = numel(x);
    observed = corr(x, y, 'Type', 'Spearman');
    if factorial(n) <= 100000
        orders = perms(1:n);
    else
        n_draw = 100000;
        orders = zeros(n_draw, n);
        for k = 1:n_draw; orders(k, :) = randperm(n); end
    end
    null_rho = zeros(size(orders, 1), 1);
    for k = 1:size(orders, 1)
        null_rho(k) = corr(x, y(orders(k, :)), 'Type', 'Spearman');
    end
    p = mean(abs(null_rho) >= abs(observed) - 1e-12);
end

% =========================================================================
function ci = bca_interval(x, y)
%BCA_INTERVAL Bias-corrected accelerated bootstrap 95% CI for Spearman rho.
    spearman = @(a, b) corr(a, b, 'Type', 'Spearman');
    try
        ci = bootci(10000, {spearman, x, y}, 'type', 'bca', 'Alpha', 0.05)';
    catch
        ci = [NaN NaN];   % degenerate (e.g. all-tied) resamples
    end
end

% =========================================================================
function p = cluster_permutation_p(x, y, groups)
%CLUSTER_PERMUTATION_P Permutation p that respects within-participant
%   dependence: permute WHICH participant's chronic block pairs with which
%   acute block, keeping each participant's hemispheres together.
    x = x(:); y = y(:); groups = string(groups(:));
    keep = ~isnan(x) & ~isnan(y);
    x = x(keep); y = y(keep); groups = groups(keep);
    observed = corr(x, y, 'Type', 'Spearman');
    keys = unique(groups, 'stable');
    n_group = numel(keys);
    y_block = arrayfun(@(k) y(groups == k), keys, 'UniformOutput', false);
    if factorial(n_group) <= 100000
        orders = perms(1:n_group);
    else
        n_draw = 100000; orders = zeros(n_draw, n_group);
        for k = 1:n_draw; orders(k, :) = randperm(n_group); end
    end
    null_rho = zeros(size(orders, 1), 1);
    for k = 1:size(orders, 1)
        y_perm = zeros(size(y));
        for g = 1:n_group
            rows = find(groups == keys(g));
            src  = y_block{orders(k, g)};
            y_perm(rows) = src(mod(0:numel(rows)-1, numel(src)) + 1);
        end
        null_rho(k) = corr(x, y_perm, 'Type', 'Spearman');
    end
    p = mean(abs(null_rho) >= abs(observed) - 1e-12);
end

% =========================================================================
function drops = leave_one_group_out(x, y, groups)
%LEAVE_ONE_GROUP_OUT Spearman rho recomputed dropping each participant (all of
%   that participant's hemispheres) -- the cluster jackknife.
    groups = string(groups(:));
    keys = unique(groups, 'stable');
    drops = table();
    for g = 1:numel(keys)
        keep = groups ~= keys(g);
        row = table();
        row.dropped = keys(g);
        row.rho = corr(x(keep), y(keep), 'Type', 'Spearman');
        drops = [drops; row]; %#ok<AGROW>
    end
end

% =========================================================================
function draw_scatter(ax, paired, xcol, ycol, colors, label_of, stat)
%DRAW_SCATTER One scatter panel with per-participant colour, hemisphere marker,
%   0 reference lines, point labels, and a rho / CI / n annotation box.
    has_hemi = ismember('hemisphere', paired.Properties.VariableNames);
    hold(ax, 'on');
    yline(ax, 0, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.6);
    xline(ax, 0, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.6);
    for r = 1:height(paired)
        pt = char(paired.patient_id(r));
        col = colors(pt);
        x = paired.(xcol)(r); y = paired.(ycol)(r);
        marker = 'o'; tag = label_of(pt);
        if has_hemi
            if strcmp(paired.hemisphere(r), 'R'); marker = 's'; end
            tag = sprintf('%s%s', label_of(pt), char(paired.hemisphere(r)));
        end
        scatter(ax, x, y, 95, marker, 'MarkerFaceColor', col, ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        text(ax, x, y, ['  ' tag], 'FontSize', 8, 'FontWeight','bold', ...
            'VerticalAlignment','middle');
    end
    % rho / CI / n box. Plain ASCII only (no tex, no Unicode math) so the SVG
    % keeps every label as editable text. Placed in whichever corner is
    % farthest from every data point, so it never occludes a marker.
    txt = sprintf('rho = %+.2f\n95%% CI [%+.2f, %+.2f]\nn = %d', ...
        stat.rho, stat.ci(1), stat.ci(2), stat.n);
    [bx, by, valign, halign] = emptiest_corner(paired.(xcol), paired.(ycol));
    text(ax, bx, by, txt, 'Units','normalized', 'VerticalAlignment',valign, ...
        'HorizontalAlignment',halign, 'FontSize', 8, 'BackgroundColor','w', ...
        'EdgeColor',[0.7 0.7 0.7], 'Interpreter','none');
    set(ax, 'Box','off', 'TickDir','out', 'FontSize', 10);
    axis(ax, 'padded');
    hold(ax, 'off');
end

% =========================================================================
function [bx, by, valign, halign] = emptiest_corner(x, y)
%EMPTIEST_CORNER Normalised position + alignment for an annotation box, chosen
%   as the axes corner whose nearest data point is farthest away (so the box
%   never sits on a marker). Corners are the four 0/1 combinations in the
%   normalised [0 1] data range; a small inset keeps the box off the axes edge.
    x = x(:); y = y(:); keep = ~isnan(x) & ~isnan(y);
    x = x(keep); y = y(keep);
    xn = normalise01(x); yn = normalise01(y);
    corners = [0 1; 1 1; 0 0; 1 0];   % NW, NE, SW, SE (in normalised data space)
    best = 1; best_gap = -inf;
    for c = 1:4
        gap = min(hypot(xn - corners(c,1), yn - corners(c,2)));
        if gap > best_gap; best_gap = gap; best = c; end
    end
    inset = 0.04;
    bx = corners(best,1) * (1 - 2*inset) + inset;
    by = corners(best,2) * (1 - 2*inset) + inset;
    if corners(best,2) == 1; valign = 'top'; else; valign = 'bottom'; end
    if corners(best,1) == 0; halign = 'left'; else; halign = 'right'; end
end

% =========================================================================
function u = normalise01(v)
%NORMALISE01 Scale a vector to [0 1]; a constant vector maps to 0.5.
    lo = min(v); hi = max(v);
    if hi > lo; u = (v - lo) / (hi - lo); else; u = 0.5 * ones(size(v)); end
end

% =========================================================================
function fig = combined_figure(part, ps, hemi, hs, label_of, pooled)
%COMBINED_FIGURE Single Figure S7 (panels a-e): the two hemisphere scatters on
%   top (a same-metric NRS, b cross-metric painVAS->index) and the three
%   jackknife sensitivity panels beneath (c participant leave-one-out,
%   d leave-one-hemisphere-out, e leave-one-participant-out cluster). Each
%   jackknife panel shows the single-scale result as an open circle and the
%   pooled three-scale result as a filled blue circle. Layout mirrors the
%   manuscript legend; all text is plain ASCII for editable SVG.
    colors = participant_colors();
    fig = figure('Units','inches','Position',[0 0 11 8.5],'Color','w');
    set(fig, 'DefaultAxesFontName','Arial', 'DefaultTextFontName','Arial', ...
             'DefaultLegendFontName','Arial');
    tl = tiledlayout(fig, 2, 6, 'TileSpacing','compact', 'Padding','compact');

    % ---- Top row: two hemisphere scatters (each spans 3 of 6 columns) ----
    ax_a = nexttile(tl, 1, [1 3]);
    draw_scatter(ax_a, hemi, 'acute_pct_NRS', 'chronic_pct_NRS', colors, label_of, hs.primary);
    xlabel(ax_a, sprintf('Acute NRS %% change at implanted target\n(negative = analgesia)'));
    ylabel(ax_a, sprintf('Chronic closed-loop NRS %% change\n(negative = pain reduction)'));
    title(ax_a, 'a  Same-metric (NRS to NRS)', 'FontWeight','bold');

    ax_b = nexttile(tl, 4, [1 3]);
    draw_scatter(ax_b, hemi, 'acute_pct_VAS', 'chronic_pct_index', colors, label_of, hs.secondary);
    xlabel(ax_b, sprintf('Acute painVAS %% change at implanted target\n(negative = analgesia)'));
    ylabel(ax_b, sprintf('Chronic closed-loop index-metric %% change\n(negative = benefit)'));
    title(ax_b, 'b  Cross-metric (painVAS to index)', 'FontWeight','bold');

    % ---- Bottom row: three jackknife panels (each spans 2 of 6 columns) ----
    pooled_color = [0.121 0.435 0.921];

    ax_c = nexttile(tl, 7, [1 2]); hold(ax_c, 'on');
    draw_reference_lines(ax_c, ps.primary.rho);
    for i = 1:height(part)
        scatter(ax_c, ps.primary.leave_one_out(i), i, 70, 'o', ...
            'MarkerFaceColor','none', 'MarkerEdgeColor',[0.45 0.45 0.45], 'LineWidth',1.2);
    end
    dpc = pooled.participant.drop_patient;
    for i = 1:height(dpc)
        scatter(ax_c, dpc.rho(i), i, 70, 'o', 'MarkerFaceColor', pooled_color, ...
            'MarkerEdgeColor','w', 'LineWidth',0.6);
    end
    set(ax_c, 'YTick', 1:height(part), 'YTickLabel', ...
        cellfun(@(p) ['drop ' label_of(char(p))], num2cell(part.patient_id), 'UniformOutput', false));
    xlabel(ax_c, 'Spearman rho'); xlim(ax_c, [-0.5 1.05]);
    title(ax_c, 'c  Participant (N = 5)', 'FontWeight','bold');
    key_single = plot(ax_c, NaN, NaN, 'o', 'MarkerFaceColor','none', ...
        'MarkerEdgeColor',[0.45 0.45 0.45], 'LineWidth',1.2, 'MarkerSize',7);
    key_pooled = plot(ax_c, NaN, NaN, 'o', 'MarkerFaceColor', pooled_color, ...
        'MarkerEdgeColor','w', 'LineWidth',0.6, 'MarkerSize',7);
    legend(ax_c, [key_single key_pooled], {'NRS only','three scales pooled'}, ...
        'Location','southwest', 'Box','off', 'FontSize', 7);
    finish_drop_axis(ax_c);

    ax_d = nexttile(tl, 9, [1 2]); hold(ax_d, 'on');
    draw_reference_lines(ax_d, hs.primary.rho);
    for i = 1:height(hemi)
        marker = 'o';
        if strcmp(hemi.hemisphere(i), 'R'); marker = 's'; end
        scatter(ax_d, hs.primary.leave_one_out(i), i, 70, marker, ...
            'MarkerFaceColor','none', 'MarkerEdgeColor',[0.45 0.45 0.45], 'LineWidth',1.2);
    end
    dph = pooled.hemisphere.drop_hemisphere;
    for i = 1:height(hemi)
        key = string(hemi.patient_id(i)) + string(hemi.hemisphere(i));
        j = find(dph.dropped == key, 1);
        if isempty(j); continue; end
        marker = 'o';
        if strcmp(hemi.hemisphere(i), 'R'); marker = 's'; end
        scatter(ax_d, dph.rho(j), i, 70, marker, 'MarkerFaceColor', pooled_color, ...
            'MarkerEdgeColor','w', 'LineWidth',0.6);
    end
    set(ax_d, 'YTick', 1:height(hemi), 'YTickLabel', ...
        arrayfun(@(i) sprintf('drop %s%s', label_of(char(hemi.patient_id(i))), ...
            char(hemi.hemisphere(i))), (1:height(hemi))', 'UniformOutput', false));
    xlabel(ax_d, 'Spearman rho'); xlim(ax_d, [-0.5 1.05]);
    title(ax_d, 'd  Leave-one-hemisphere-out', 'FontWeight','bold'); finish_drop_axis(ax_d);

    ax_e = nexttile(tl, 11, [1 2]); hold(ax_e, 'on');
    draw_reference_lines(ax_e, hs.primary.rho);
    dp = hs.primary.drop_patient;
    for i = 1:height(dp)
        scatter(ax_e, dp.rho(i), i, 70, 'o', 'MarkerFaceColor','none', ...
            'MarkerEdgeColor',[0.45 0.45 0.45], 'LineWidth',1.2);
    end
    dpe = pooled.hemisphere.drop_patient;
    for i = 1:height(dp)
        j = find(dpe.dropped == dp.dropped(i), 1);
        if isempty(j); continue; end
        scatter(ax_e, dpe.rho(j), i, 70, 'o', 'MarkerFaceColor', pooled_color, ...
            'MarkerEdgeColor','w', 'LineWidth',0.6);
    end
    set(ax_e, 'YTick', 1:height(dp), 'YTickLabel', ...
        cellfun(@(p) ['drop ' label_of(char(p))], num2cell(dp.dropped), 'UniformOutput', false));
    xlabel(ax_e, 'Spearman rho'); xlim(ax_e, [-0.6 1.05]);
    title(ax_e, 'e  Leave-one-participant-out (cluster)', 'FontWeight','bold'); finish_drop_axis(ax_e);

    % The panel is titled by content only; the manuscript carries the figure
    % numbering.
    sgtitle(fig, ['Acute (temporary trial) analgesia vs chronic ' ...
        'closed-loop benefit (8 hemispheres / 5 participants, exploratory)'], ...
        'FontName','Arial', 'FontSize', 11);
end

% =========================================================================
function draw_reference_lines(ax, full_rho)
%DRAW_REFERENCE_LINES Vertical lines at the full-sample rho (black) and at 0 (grey).
    xline(ax, full_rho, '-', 'Color', [0 0 0], 'LineWidth', 1.0);
    xline(ax, 0, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.6);
end

% =========================================================================
function finish_drop_axis(ax)
%FINISH_DROP_AXIS Shared axis styling for the jackknife panels.
    set(ax, 'Box','off', 'TickDir','out', 'FontSize', 9); hold(ax, 'off');
end

% =========================================================================
function colors = participant_colors()
%PARTICIPANT_COLORS Okabe-Ito colour per participant (colour-vision-safe).
    okabe = [0.902 0.624 0;      0.337 0.706 0.914;  0 0.620 0.451; ...
             0.941 0.894 0.259;  0 0.447 0.698];
    ids = {'P1','P2','P3','P4','P5'};
    colors = containers.Map();
    for k = 1:numel(ids); colors(ids{k}) = okabe(k, :); end
end
