function [fig, duty_summary] = plot_stimulation_parameter_timeline( ...
    patient_label, dL, dR, anat_L, anat_R, ...
    x_max_months, amp_max_ov, freq_max_ov, pw_override, note, duty_cl, sham_windows)
% PLOT_STIMULATION_PARAMETER_TIMELINE  Longitudinal stimulation programming history.
%
%   Draws one participant's deep brain stimulation programming history as 8
%   vertically stacked panels sharing one x-axis:
%     (legend strip at top)
%     a) Status bar (hemisphere, polarity, loop mode over time)
%     b) Left DBS device — active contact timeline
%     c) Right DBS device — active contact timeline
%     d) Amplitude (mA) over time
%     e) Frequency (Hz) over time
%     f) Pulse width (us) over time
%     g) Duty cycle (%) over time — closed-loop measured (solid points) and
%        open-loop programmed cycling (lighter step); mean closed-loop in text.
%
% INPUTS:
%   patient_label  — display name, e.g. 'P1'
%   dL, dR         — per-side stimulation parameter tables for the left and
%                     right hemispheres. Pass [] if no data for that side.
%   anat_L, anat_R — struct with fields .lead1, .lead2 giving anatomy
%                     labels for contacts E0-E3 and E8-E11. Pass [] if
%                     no device on that side.
%   x_max_months   — x-axis limit in months from implant
%   amp_max_ov     — amplitude y-axis max override ([] = auto)
%   freq_max_ov    — frequency y-axis max override ([] = auto)
%   pw_override    — pulse width values to show ([] = auto from data)
%   note           — text note to display in subtitle ('' = none)
%   duty_cl        — struct with fields .L and .R, each a table (day,
%                    percentDutyCycle) of daily measured closed-loop duty cycle
%                    per hemisphere. Pass [] or omit for none; a missing side is
%                    drawn without closed-loop points.
%   sham_windows   — Nx2 datetime [start, stop] of this participant's sham
%                    episodes, using the same definition as the cohort
%                    mixed-effects model (the 'sham' level of stim_group), so the
%                    figure and the statistics agree. Drawn as a black bar in
%                    the Loop row of panel a.
%                    Pass [] or omit for none.
%
% OUTPUTS:
%   fig           — figure handle
%   duty_summary  — struct: .patient, and .L/.R structs with closed-loop duty
%                    cycle mean, sd, cv, n (empty side if no closed-loop data).
%
% DEPENDENCIES:
%   Called by plot_figureS11_programming_history.m

if nargin < 11 || isempty(duty_cl); duty_cl = struct(); end
if nargin < 12 || isempty(sham_windows); sham_windows = datetime.empty(0,2); end

%% 0. Constants
L_COL  = [0.130 0.400 0.675];   % #2166ac
R_COL  = [0.694 0.094 0.169];   % #b2182b
BI_COL = [0.482 0.200 0.580];   % #7b3294
MONO_COL = [0.333 0.333 0.333]; % #555555
BIP_COL  = [0.831 0.628 0.090]; % #d4a017
OL_COL   = [0.180 0.800 0.443]; % #2ecc71
CL_COL   = [0.773 0.106 0.490]; % #c51b7d
CC_L     = [0.130 0.400 0.675]; % #2166ac
CC_R     = [0.753 0.224 0.169]; % #c0392b
SHAM_COL = [0 0 0];             % black — sham is a black bar in the Loop row (a)

% Contact Y-mapping: physical contacts 0-3,8-11 → plot Y 0-7
y_map = containers.Map({0,1,2,3,8,9,10,11}, {0,1,2,3,4,5,6,7});
y_ticks = 0:7;
y_labels = {'E0','E1','E2','E3','E8','E9','E10','E11'};
SEP_Y    = 3.5;
BAR_LW   = 10;
ANAT_FS  = 18;

%% 1. Per-side stimulation parameter tables ([] if no device on that side)
sL = dL;
sR = dR;

%% 2. Determine time origin and end
dates = [];
if ~isempty(sL); dates = [dates; sL.datetime]; end
if ~isempty(sR); dates = [dates; sR.datetime]; end
t_origin = min(dates);
t_end    = t_origin + caldays(round(x_max_months * 30.44));

% Clip to time window
if ~isempty(sL); sL = sL(sL.datetime <= t_end, :); end
if ~isempty(sR); sR = sR(sR.datetime <= t_end, :); end

%% 3. Build intervals and washouts
intervals = build_intervals(sL, sR, t_origin);
wm        = build_washouts(sL, sR, t_origin, t_end);

%% 4. Compute summary statistics
nL = 0; nR = 0;
if ~isempty(sL); nL = sum(sL.effectively_active); end
if ~isempty(sR); nR = sum(sR.effectively_active); end


%% 5. Parameter ranges
pw_vals = []; amp_max = 0; freq_max = 0;
for side_tbl = {sL, sR}
    st = side_tbl{1};
    if isempty(st); continue; end
    act = st(st.effectively_active, :);
    if isempty(act); continue; end
    pw_vals  = unique([pw_vals; act.pulse_width_us(act.pulse_width_us > 0)]);
    amp_max  = max(amp_max,  max(act.amplitude_mA));
    freq_max = max(freq_max, max(act.frequency_Hz));
end
pw_vals = sort(pw_vals);
if ~isempty(pw_override); pw_vals = sort(pw_override(:)); end
if ~isempty(amp_max_ov);  amp_max  = amp_max_ov;  end
if ~isempty(freq_max_ov); freq_max = freq_max_ov; end

x_min = -0.3;
x_max = x_max_months + 0.3;

% Sham episodes (the 'sham' group of the mixed-effects model) -> months-from-
% implant bands. Short episodes are padded to a minimum on-screen width so a few
% sham days stay visible on the multi-year axis; all bands are clipped to the
% plotted window.
sham_spans = zeros(0, 2);
if ~isempty(sham_windows)
    sm = [(datenum(sham_windows(:,1)) - datenum(t_origin)) / 30.44, ...
          (datenum(sham_windows(:,2)) - datenum(t_origin)) / 30.44]; %#ok<DATNM>
    min_w  = 0.5;                                   % months
    narrow = (sm(:,2) - sm(:,1)) < min_w;
    ctr    = mean(sm(narrow, :), 2);
    sm(narrow, 1) = ctr - min_w/2;
    sm(narrow, 2) = ctr + min_w/2;
    sm(:,1) = max(sm(:,1), x_min);
    sm(:,2) = min(sm(:,2), x_max);
    sham_spans = sm(sm(:,2) > sm(:,1), :);
end

%% 6. Create figure and layout
fig = figure('Units','inches', 'Position',[1 1 20 15.5], ...
    'Color','w', 'PaperPositionMode','auto');

% Height ratios for 8 panels (legend, bar, L device, R device, amp, freq,
% pulse width, duty cycle)
pw_ratio  = 0.30; if numel(pw_vals) > 2; pw_ratio = 0.50; end
dL_ratio  = 0.3;  if ~isempty(anat_L); dL_ratio = 1.1; end
dR_ratio  = 0.3;  if ~isempty(anat_R); dR_ratio = 1.1; end
amp_ratio = 0.75; if strcmp(patient_label, 'P5'); amp_ratio = 0.60; end
frq_ratio = 0.75; if strcmp(patient_label, 'P5'); frq_ratio = 0.60; end
dc_ratio  = 0.60;

ratios = [0.20, 0.22, dL_ratio, dR_ratio, amp_ratio, frq_ratio, pw_ratio, dc_ratio];
ratios = ratios / sum(ratios);

% Margins (bot_m gives the 8th panel's x-label room to clear the footer).
% left_m is wide enough that the longest status-bar row label ('Hemisphere')
% clears the left figure edge and the bold panel letters (see PANEL_X below).
left_m  = 0.115; right_m = 0.94;
top_m   = 0.945; bot_m  = 0.110;   % bottom room for x-label + blinding note + footer
hspace  = 0.004;  % fraction of total

n_panels = numel(ratios);
total_h  = top_m - bot_m - hspace * (n_panels - 1);
panel_h  = ratios * total_h;

% Create axes bottom-up
ax = gobjects(n_panels, 1);
y_cursor = bot_m;
for k = n_panels:-1:1
    ax(k) = axes('Position', [left_m, y_cursor, right_m - left_m, panel_h(k)]);
    y_cursor = y_cursor + panel_h(k) + hspace;
end

ax_leg = ax(1); ax_bar = ax(2); ax_dL = ax(3); ax_dR = ax(4);
ax_amp = ax(5); ax_frq = ax(6); ax_pw = ax(7); ax_dc = ax(8);
axes_data = [ax_bar, ax_dL, ax_dR, ax_amp, ax_frq, ax_pw, ax_dc];

for a = axes_data
    set(a, 'XLim', [x_min, x_max], 'Box', 'off', 'TickDir', 'out');
    a.XAxis.TickDirection = 'out';
    a.YAxis.TickDirection = 'out';
end

% (Sham is drawn as its own Loop-row category in the status bar below, not as a
% full-height background column.)

% Left-margin columns, left to right: bold panel letter (PANEL_X), rotated
% y-axis title (YLABEL_X), then the axis tick labels. Kept far enough apart
% that the letters never overprint the titles ('g' vs 'Pulse width' etc.).
YLABEL_X = 0.050;
PANEL_X  = 0.008;

%% 7a. Legend panel
axis(ax_leg, 'off');
hold(ax_leg, 'on');
leg_items = {
    L_COL,    0.5, 'Left'
    R_COL,    0.5, 'Right'
    BI_COL,   0.5, 'Bilateral'
    [1 1 1],  0.8, 'Washout'          % white with edge
    MONO_COL, 0.35,'Monopolar'
    BIP_COL,  0.4, 'Bipolar'
    OL_COL,   0.3, 'Open-loop'
    CL_COL,   0.4, 'Closed-loop'
    SHAM_COL, 0.9, 'Sham (blinded)'   % black bar in the Loop row
};
h_leg = gobjects(size(leg_items,1), 1);
for k = 1:size(leg_items,1)
    if k == 4  % washout: white patch with black edge
        h_leg(k) = patch(ax_leg, [0 0 0 0], [0 0 0 0], [1 1 1], ...
            'EdgeColor', 'k', 'LineWidth', 0.8, 'DisplayName', leg_items{k,3});
    else
        h_leg(k) = patch(ax_leg, [0 0 0 0], [0 0 0 0], leg_items{k,1}, ...
            'FaceAlpha', leg_items{k,2}, 'EdgeColor', 'none', ...
            'DisplayName', leg_items{k,3});
    end
end
legend(ax_leg, h_leg, 'Location', 'north', 'Orientation', 'horizontal', ...
    'NumColumns', 5, 'FontSize', 12, 'Box', 'off');

%% 7b. Title and subtitle
anat_parts = {};
if ~isempty(anat_L)
    anat_parts{end+1} = sprintf('Left: %s (E0-E3) / %s (E8-E11)', anat_L.lead1, anat_L.lead2);
end
if ~isempty(anat_R)
    anat_parts{end+1} = sprintf('Right: %s (E0-E3) / %s (E8-E11)', anat_R.lead1, anat_R.lead2);
end
note_str = '';
if ~isempty(note); note_str = sprintf('   .   %s', note); end

annotation(fig, 'textbox', [left_m 0.97 0.9 0.03], 'String', ...
    ['\bf' patient_label '\rm  Deep brain stimulation program history'], ...
    'FontSize', 16, 'EdgeColor', 'none', 'VerticalAlignment', 'top', ...
    'Interpreter', 'tex');
subtitle_str = sprintf('%s   .   L: %d  R: %d active%s', ...
    strjoin(anat_parts, '   .   '), nL, nR, note_str);
annotation(fig, 'textbox', [left_m 0.955 0.9 0.02], 'String', subtitle_str, ...
    'FontSize', 11, 'Color', [0.33 0.33 0.33], 'EdgeColor', 'none', ...
    'VerticalAlignment', 'top');

%% 7c. Status bar (panel a)
hold(ax_bar, 'on');
sc = struct('L', L_COL, 'R', R_COL, 'bilateral', BI_COL);
for i = 1:numel(intervals)
    iv = intervals(i);
    if ~iv.L_active && ~iv.R_active; continue; end
    ms = iv.ms; me = iv.me;
    if iv.L_active && iv.R_active
        state = 'bilateral';
    elseif iv.L_active
        state = 'L';
    else
        state = 'R';
    end
    col = sc.(state);

    % Check polarity and loop mode
    bip = false; adp = false;
    for sk = {'L', 'R'}
        s = sk{1};
        if iv.([s '_active']) && ~isempty(iv.(s))
            if strcmp(iv.(s).polarity, 'bipolar'); bip = true; end
            if iv.(s).is_adaptive; adp = true; end
        end
    end

    % Hemisphere row (top third)
    patch(ax_bar, [ms me me ms], [2/3 2/3 1 1], col, ...
        'FaceAlpha', 0.5, 'EdgeColor', 'none');
    % Polarity row (middle third)
    pol_col = MONO_COL; if bip; pol_col = BIP_COL; end
    patch(ax_bar, [ms me me ms], [1/3 1/3 2/3 2/3], pol_col, ...
        'FaceAlpha', 0.35, 'EdgeColor', 'none');
    % Loop row (bottom third): open-loop vs closed-loop
    loop_col = OL_COL; if adp; loop_col = CL_COL; end
    patch(ax_bar, [ms me me ms], [0 0 1/3 1/3], loop_col, ...
        'FaceAlpha', 0.3, 'EdgeColor', 'none');
end
% Sham (blinded) is its own Loop-row category: a solid black bar during each
% sham episode (mixed-effects sham group). The device is off/0 mA here, so there is no
% open/closed-loop patch to overwrite. Drawn last so it sits on top.
for bi = 1:size(sham_spans, 1)
    xs = sham_spans(bi, :);
    patch(ax_bar, [xs(1) xs(2) xs(2) xs(1)], [0 0 1/3 1/3], SHAM_COL, ...
        'FaceAlpha', 0.9, 'EdgeColor', 'none');
end
set(ax_bar, 'YLim', [0 1], 'YTick', [1/6, 3/6, 5/6], ...
    'YTickLabel', {'Loop','Polarity','Hemisphere'}, 'FontSize', 11);
ax_bar.YAxis.TickLength = [0 0];
ax_bar.XAxis.TickLength = [0 0];
set(ax_bar, 'XTickLabel', []);
ax_bar.YAxis.Color = [0.5 0.5 0.5];
line(ax_bar, [x_min x_max], [1/3 1/3], 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
line(ax_bar, [x_min x_max], [2/3 2/3], 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
pos_bar = get(ax_bar, 'Position');
annotation(fig, 'textbox', [PANEL_X pos_bar(2)+pos_bar(4)-0.01 0.03 0.02], ...
    'String', '\bfa', 'FontSize', 16, 'EdgeColor', 'none', 'Interpreter', 'tex');

%% 7d. Device panels (panels b, c)
draw_device_panel(fig, ax_dL, intervals, 'L', 'Left', anat_L, 'b', ...
    CC_L, y_map, y_ticks, y_labels, SEP_Y, BAR_LW, ANAT_FS, ...
    wm, x_min, YLABEL_X, PANEL_X);
draw_device_panel(fig, ax_dR, intervals, 'R', 'Right', anat_R, 'c', ...
    CC_R, y_map, y_ticks, y_labels, SEP_Y, BAR_LW, ANAT_FS, ...
    wm, x_min, YLABEL_X, PANEL_X);

% Bipolar anode legend on right device panel
if ~isempty(anat_R)
    hold(ax_dR, 'on');
    h_bip = plot(ax_dR, NaN, NaN, 'Color', BIP_COL, 'LineWidth', BAR_LW-2, ...
        'DisplayName', 'Bipolar anode (+)');
    legend(ax_dR, h_bip, 'Location', 'southeast', 'FontSize', 11, 'Box', 'off');
end

%% 7e-g. Parameter panels (amplitude, frequency, pulse width)
draw_param_panel(fig, ax_amp, intervals, 'amplitude_mA', ...
    sprintf('Amplitude\n(mA)'), 'd', L_COL, R_COL, wm, YLABEL_X, PANEL_X);
draw_param_panel(fig, ax_frq, intervals, 'frequency_Hz', ...
    sprintf('Frequency\n(Hz)'), 'e', L_COL, R_COL, wm, YLABEL_X, PANEL_X);
draw_param_panel(fig, ax_pw,  intervals, 'pulse_width_us', ...
    sprintf('Pulse width\n(us)'), 'f', L_COL, R_COL, wm, YLABEL_X, PANEL_X);

%% 7h. Duty cycle panel (track g) + companion closed-loop summary
duty_summary = draw_duty_cycle_panel(fig, ax_dc, intervals, duty_cl, t_origin, ...
    L_COL, R_COL, wm, x_min, x_max, YLABEL_X, PANEL_X, patient_label);

% Amplitude legend
hold(ax_amp, 'on');
hL = plot(ax_amp, NaN, NaN, 'Color', L_COL, 'LineWidth', 5, 'DisplayName', 'Left');
hR = plot(ax_amp, NaN, NaN, 'Color', R_COL, 'LineWidth', 5, 'DisplayName', 'Right');
legend(ax_amp, [hL hR], 'Location', 'northeast', 'FontSize', 12, ...
    'Box', 'off', 'NumColumns', 2);

%% 8. Axes limits and ticks
% Amplitude
set(ax_amp, 'YLim', [-0.15, max(amp_max*1.15, 5)]);
ytk = 0:floor(amp_max*1.2);
set(ax_amp, 'YTick', ytk(ytk <= amp_max*1.2));

% Frequency
set(ax_frq, 'YLim', [-3, max(freq_max*1.08, 160)]);
ftk = [0 25 50 75 100 125 150 175 200];
set(ax_frq, 'YTick', ftk(ftk <= freq_max*1.1));

% Low-frequency reference line at 2 Hz, drawn without a label.
hold(ax_frq, 'on');
line(ax_frq, [x_min x_max], [2 2], 'Color', [0.6 0.6 0.6], ...
    'LineWidth', 0.5, 'LineStyle', ':');

% Pulse width. Some patients cycle through many closely-spaced pulse widths
% (e.g. 100-200 in 10-us steps), which makes the y-tick labels collide. Thin
% the labels to a non-overlapping subset based on the panel's true height,
% keeping the rounder values; the data bars still sit at every real value.
if ~isempty(pw_vals)
    ylim_pw = [min(pw_vals)-20, max(pw_vals)+20];
    set(ax_pw, 'YLim', ylim_pw);
    old_units = get(ax_pw, 'Units');
    set(ax_pw, 'Units', 'points');
    pw_h_pts = get(ax_pw, 'Position');
    pw_h_pts = pw_h_pts(4);
    set(ax_pw, 'Units', old_units);
    % Require ~1.2 label heights (13 pt font) of clear space between labels.
    min_gap = (15.5 / max(pw_h_pts, 1)) * diff(ylim_pw);
    set(ax_pw, 'YTick', thin_close_ticks(pw_vals, min_gap));
end

%% 9. X-axis formatting
mt = 0:6:(x_max_months + 6);
for a = axes_data
    set(a, 'XTick', mt);
    % Minor ticks every 3 months
    a.XAxis.MinorTick = 'on';
    a.XAxis.MinorTickValues = 0:3:(x_max_months + 6);
end
set(ax_dc, 'XTickLabel', arrayfun(@num2str, mt, 'UniformOutput', false));
xlabel(ax_dc, 'Months from implant', 'FontSize', 16, 'FontWeight', 'bold');

% Hide x-tick labels on all but the bottom (duty-cycle) panel
for a = axes_data(1:end-1)
    set(a, 'XTickLabel', []);
end
% Y tick font size
for a = axes_data(2:end)
    set(a, 'FontSize', 13);
end

% No footer text is drawn below the axes: the figure carries only the panels
% and their own labels.

end


%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function intervals = build_intervals(sL, sR, t_origin)
% Build a merged event timeline tracking which hemisphere is active.

events = struct('datetime',{},'month',{},'side',{},'active',{}, ...
    'cathode',{},'anode',{},'polarity',{},'amplitude_mA',{}, ...
    'frequency_Hz',{},'pulse_width_us',{},'is_adaptive',{},'duty_prog',{});

for side_info = {{'L', sL}, {'R', sR}}
    sk = side_info{1}{1};
    df = side_info{1}{2};
    if isempty(df); continue; end
    for j = 1:height(df)
        e.datetime      = df.datetime(j);
        e.month         = days(df.datetime(j) - t_origin) / 30.44;
        e.side          = sk;
        e.active        = df.effectively_active(j);
        e.cathode       = df.cathode(j);
        e.anode         = df.anode(j);
        e.polarity      = df.polarity{j};
        e.amplitude_mA  = df.amplitude_mA(j);
        e.frequency_Hz  = df.frequency_Hz(j);
        e.pulse_width_us = df.pulse_width_us(j);
        e.is_adaptive   = df.is_adaptive(j);
        e.duty_prog     = df.duty_prog(j);
        events(end+1)   = e; %#ok<AGROW>
    end
end

% Sort by datetime
[~, idx] = sort([events.month]);
events = events(idx);

% Walk through events tracking bilateral state
Ls = struct('active', false);
Rs = struct('active', false);
n_ev = numel(events);

intervals = struct('ms',{},'me',{},'L_active',{},'R_active',{},'L',{},'R',{});
for i = 1:n_ev
    evt = events(i);
    if i < n_ev
        em = events(i+1).month;
    else
        em = evt.month + 0.25;
    end

    if strcmp(evt.side, 'L')
        Ls = evt;
    else
        Rs = evt;
    end

    iv.ms = evt.month;
    iv.me = em;
    iv.L_active = Ls.active;
    iv.R_active = Rs.active;
    if Ls.active; iv.L = Ls; else; iv.L = []; end
    if Rs.active; iv.R = Rs; else; iv.R = []; end
    intervals(end+1) = iv; %#ok<AGROW>
end
end


function wm = build_washouts(sL, sR, t_origin, t_end)
% Washout periods (>=3 consecutive days with no active stim on either
% hemisphere), returned as Nx2 [start_month, end_month].
spans = all_inactive_spans(sL, sR, t_origin, t_end);
if isempty(spans); wm = []; return; end
keep  = days(spans(:,2) - spans(:,1)) >= 3;      % first->last-day span (>= 3)
spans = spans(keep, :);
if isempty(spans); wm = []; return; end
wm = [(datenum(spans(:,1))            - datenum(t_origin)) / 30.44, ...
      (datenum(spans(:,2)+caldays(1)) - datenum(t_origin)) / 30.44]; %#ok<DATNM>
end


function spans = all_inactive_spans(sL, sR, t_origin, t_end)
% Nx2 datetime [firstDay, lastDay] of maximal runs of consecutive calendar
% days with NO active stim on either hemisphere (therapy off / 0 mA). Empty
% if none. Runs are found by group_consecutive_days.
act_dates = [];
for side_tbl = {sL, sR}
    df = side_tbl{1};
    if isempty(df); continue; end
    act = df(df.effectively_active, :);
    act_dates = [act_dates; dateshift(act.datetime, 'start', 'day')]; %#ok<AGROW>
end
act_dates = unique(act_dates);
all_days  = (dateshift(t_origin,'start','day') : caldays(1) : dateshift(t_end,'start','day'))';
inactive  = all_days(~ismember(all_days, act_dates));
spans     = group_consecutive_days(inactive);
end


function spans = group_consecutive_days(d)
% Group day-resolution datetimes into maximal consecutive runs, returned as
% Nx2 datetime [firstDay, lastDay]. Fully vectorized (logical run-edge masks).
if isempty(d); spans = [d(:), d(:)]; return; end
d      = sort(d(:));
newRun = [true;  days(diff(d)) > 1];   % first day of each run
runEnd = [days(diff(d)) > 1; true];    % last day of each run
spans  = [d(newRun), d(runEnd)];
end


function segs = clip_bar(ms, me, wm)
% Clip a bar segment [ms, me] to exclude washout periods in wm (Nx2).
segs = [ms, me];
if isempty(wm); return; end

for i = 1:size(wm, 1)
    ws = wm(i,1); we = wm(i,2);
    new_segs = [];
    for j = 1:size(segs, 1)
        s = segs(j,1); e = segs(j,2);
        if we <= s || ws >= e
            new_segs = [new_segs; s, e]; %#ok<AGROW>
        else
            if s < ws; new_segs = [new_segs; s, ws]; end %#ok<AGROW>
            if we < e; new_segs = [new_segs; we, e]; end %#ok<AGROW>
        end
    end
    segs = new_segs;
    if isempty(segs); return; end
end
end


function draw_device_panel(fig, ax, intervals, sk, side_label, anat, ...
    panel_label, bar_color, y_map, y_ticks_v, y_labels_v, SEP_Y, BAR_LW, ...
    ANAT_FS, wm, x_min, YLABEL_X, PANEL_X)
% Draw one device contact panel.

hold(ax, 'on');
pos = get(ax, 'Position');

if isempty(anat)
    % No data for this side
    text(ax, 0.5, 0.5, sprintf('%s: No data (explanted)', side_label), ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'FontSize', 14, 'Color', [0.6 0.6 0.6], ...
        'FontAngle', 'italic');
    set(ax, 'YTick', [], 'YColor', 'none');
    add_rotated_ylabel(ax, YLABEL_X, sprintf('%s DBS\nDevice', side_label));
    annotation(fig, 'textbox', [PANEL_X pos(2)+pos(4)-0.01 0.03 0.02], ...
        'String', ['\bf' panel_label], 'FontSize', 16, 'EdgeColor', 'none', ...
        'Interpreter', 'tex');
    return
end

% Grid lines at each contact level
for y = y_ticks_v
    line(ax, ax.XLim, [y y], 'Color', [0.91 0.91 0.91], 'LineWidth', 0.3);
end
% Separator between lead1 and lead2
line(ax, ax.XLim, [SEP_Y SEP_Y], 'Color', [0.67 0.67 0.67], 'LineWidth', 0.8);

% Plot contact bars
BIP_COL = [0.831 0.628 0.090];
for i = 1:numel(intervals)
    iv = intervals(i);
    if ~iv.([sk '_active']) || isempty(iv.(sk)); continue; end
    ms_iv = iv.ms; me_iv = iv.me;
    info = iv.(sk);
    c = info.cathode;
    if isnan(c); continue; end
    c = round(c);
    if ~y_map.isKey(c); continue; end
    cy = y_map(c);

    segs = clip_bar(ms_iv, me_iv, wm);
    for j = 1:size(segs,1)
        plot(ax, [segs(j,1) segs(j,2)], [cy cy], 'Color', bar_color, ...
            'LineWidth', BAR_LW);
    end

    % Bipolar anode
    if strcmp(info.polarity, 'bipolar') && ~isnan(info.anode)
        a = round(info.anode);
        if y_map.isKey(a)
            ay = y_map(a);
            for j = 1:size(segs,1)
                plot(ax, [segs(j,1) segs(j,2)], [ay ay], 'Color', BIP_COL, ...
                    'LineWidth', BAR_LW-2);
            end
        end
    end
end

set(ax, 'YTick', y_ticks_v, 'YTickLabel', y_labels_v, 'FontSize', 13, ...
    'YLim', [-0.6 7.6]);

% Axis label and anatomy annotations
add_rotated_ylabel(ax, YLABEL_X, sprintf('%s DBS\nDevice', side_label));
set(ax, 'YLabel', []);

% Anatomy labels
text(ax, x_min + 0.2, -0.4, anat.lead1, 'FontSize', ANAT_FS, ...
    'Color', [0 0 0 0.35], 'FontWeight', 'bold', 'VerticalAlignment', 'bottom', ...
    'HorizontalAlignment', 'left', 'FontAngle', 'italic');
text(ax, x_min + 0.2, SEP_Y + 0.1, anat.lead2, 'FontSize', ANAT_FS, ...
    'Color', [0 0 0 0.35], 'FontWeight', 'bold', 'VerticalAlignment', 'bottom', ...
    'HorizontalAlignment', 'left', 'FontAngle', 'italic');

% Panel label
annotation(fig, 'textbox', [PANEL_X pos(2)+pos(4)-0.01 0.03 0.02], ...
    'String', ['\bf' panel_label], 'FontSize', 16, 'EdgeColor', 'none', ...
    'Interpreter', 'tex');
end


function draw_param_panel(fig, ax, intervals, param, ylabel_str, ...
    panel_label, L_COL, R_COL, wm, YLABEL_X, PANEL_X)
% Draw a parameter timeline panel (amplitude, frequency, or pulse width).

hold(ax, 'on');
for i = 1:numel(intervals)
    iv = intervals(i);
    ms_iv = iv.ms; me_iv = iv.me;
    for side_info = {{'L', L_COL}, {'R', R_COL}}
        sk  = side_info{1}{1};
        col = side_info{1}{2};
        if ~iv.([sk '_active']) || isempty(iv.(sk)); continue; end
        v = iv.(sk).(param);
        if isnan(v) || v == 0; continue; end
        segs = clip_bar(ms_iv, me_iv, wm);
        for j = 1:size(segs,1)
            plot(ax, [segs(j,1) segs(j,2)], [v v], 'Color', col, ...
                'LineWidth', 5);
        end
    end
end

% Y gridlines
ytk = get(ax, 'YTick');
for y = ytk
    if y >= 0
        line(ax, ax.XLim, [y y], 'Color', [0.94 0.94 0.94], 'LineWidth', 0.3);
    end
end

pos = get(ax, 'Position');
add_rotated_ylabel(ax, YLABEL_X, ylabel_str);
set(ax, 'YLabel', []);

annotation(fig, 'textbox', [PANEL_X pos(2)+pos(4)-0.01 0.03 0.02], ...
    'String', ['\bf' panel_label], 'FontSize', 16, 'EdgeColor', 'none', ...
    'Interpreter', 'tex');
end


function duty_summary = draw_duty_cycle_panel(fig, ax, intervals, duty_cl, ...
    t_origin, L_COL, R_COL, wm, x_min, x_max, YLABEL_X, PANEL_X, patient_label)
% Track g: duty cycle (%). Open-loop periods are drawn as a lighter step at the
% programmed cycle-on/off duty cycle; closed-loop periods as the measured daily
% effective duty cycle (solid points). Days without data are gaps (never plotted
% as 0). The per-hemisphere mean closed-loop duty cycle is reported as in-panel
% text ('mean CL ...'). Returns per-hemisphere mean/SD/CV for the companion summary.
hold(ax, 'on');

for y = [0 25 50 75 100]
    line(ax, [x_min x_max], [y y], 'Color', [0.94 0.94 0.94], 'LineWidth', 0.3);
end

% (1) Open-loop programmed duty cycle -> lighter step segments
OL_L = L_COL + 0.55 * ([1 1 1] - L_COL);
OL_R = R_COL + 0.55 * ([1 1 1] - R_COL);
for i = 1:numel(intervals)
    iv = intervals(i);
    for si = {{'L', OL_L}, {'R', OL_R}}
        sk = si{1}{1}; col = si{1}{2};
        if ~iv.([sk '_active']) || isempty(iv.(sk)); continue; end
        info = iv.(sk);
        if info.is_adaptive; continue; end          % closed-loop handled below
        d = info.duty_prog;
        if isnan(d) || d <= 0; continue; end
        segs = clip_bar(iv.ms, iv.me, wm);
        for j = 1:size(segs, 1)
            plot(ax, [segs(j,1) segs(j,2)], [d d], 'Color', col, 'LineWidth', 4);
        end
    end
end

% (2) Closed-loop measured daily duty cycle -> solid points
duty_summary = struct('patient', patient_label, 'L', [], 'R', []);
sides = {'L', L_COL; 'R', R_COL};
for s = 1:size(sides, 1)
    sk = sides{s,1}; col = sides{s,2};
    if ~isfield(duty_cl, sk) || isempty(duty_cl.(sk)) || height(duty_cl.(sk)) == 0
        continue
    end
    dc   = duty_cl.(sk);
    % datenum is timezone-agnostic; the aDBS 'day' and the parsed-db t_origin
    % may differ in whether they carry a time zone. Daily resolution makes any
    % sub-day offset negligible for the months-from-implant mapping.
    mo   = (datenum(dc.day) - datenum(t_origin)) / 30.44; %#ok<DATNM>
    keep = mo >= x_min & mo <= x_max;
    scatter(ax, mo(keep), dc.percentDutyCycle(keep), 7, col, 'filled', ...
        'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', 'none');
    v  = dc.percentDutyCycle;
    mu = mean(v); sd = std(v);
    duty_summary.(sk) = struct('mean', mu, 'sd', sd, ...
        'cv', sd / max(mu, eps), 'n', numel(v));
end

% mean +/- SD annotation (small, top-right)
lbl = {};
if ~isempty(duty_summary.L)
    lbl{end+1} = sprintf('L %.0f\\pm%.0f%%', duty_summary.L.mean, duty_summary.L.sd); %#ok<AGROW>
end
if ~isempty(duty_summary.R)
    lbl{end+1} = sprintf('R %.0f\\pm%.0f%%', duty_summary.R.mean, duty_summary.R.sd); %#ok<AGROW>
end
if ~isempty(lbl)
    text(ax, x_max - 0.2, 99, ['mean CL  ' strjoin(lbl, '   ')], ...
        'FontSize', 10, 'Color', [0.35 0.35 0.35], 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'top', 'Interpreter', 'tex');
end

set(ax, 'YLim', [-3 103], 'YTick', [0 25 50 75 100], 'FontSize', 13);
pos = get(ax, 'Position');
add_rotated_ylabel(ax, YLABEL_X, sprintf('Duty cycle\n(%%)'));
set(ax, 'YLabel', []);
annotation(fig, 'textbox', [PANEL_X pos(2)+pos(4)-0.01 0.03 0.02], ...
    'String', '\bfg', 'FontSize', 16, 'EdgeColor', 'none', 'Interpreter', 'tex');
end


function ticks = thin_close_ticks(vals, min_gap)
% Return unique rounded tick values with labels that would overlap removed:
% walking low->high, a candidate is dropped if it is within MIN_GAP (data
% units) of the last kept tick. When two collide, the rounder value wins
% (multiple of 100 > 50 > 10 > other) so e.g. 100 is kept over 110/120.
% The underlying data bars are unaffected — this only thins the axis labels.
ticks_all = unique(round(vals(:)))';
if numel(ticks_all) <= 1
    ticks = ticks_all;
    return
end
kept = ticks_all(1);
for t = ticks_all(2:end)
    if t - kept(end) >= min_gap
        kept(end+1) = t; %#ok<AGROW>
    elseif tick_roundness(t) > tick_roundness(kept(end))
        kept(end) = t;   % replace last kept with the rounder colliding value
    end
end
ticks = kept;
end


function r = tick_roundness(v)
% Rank how "round" a tick value is (higher = rounder), used to break ties
% when two candidate ticks are too close to both be labelled.
if mod(v, 100) == 0
    r = 3;
elseif mod(v, 50) == 0
    r = 2;
elseif mod(v, 10) == 0
    r = 1;
else
    r = 0;
end
end


function add_rotated_ylabel(ax, YLABEL_X, str)
% Draw a rotated, bold panel y-title in the left margin. Uses an axes text
% object anchored at a point (not a rotated annotation textbox, which pivots
% about its corner and swings the tall box off the left edge of the figure).
% YLABEL_X is the target figure-x; convert it to axes-normalized x so the label
% lands in the margin just left of the tick labels.
pos = get(ax, 'Position');
nx  = (YLABEL_X - pos(1)) / pos(3);
text(ax, nx, 0.5, str, 'Units', 'normalized', 'Rotation', 90, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', 15, 'FontWeight', 'bold', 'Clipping', 'off');
end


function v = safe_num(x)
% Safely extract a numeric scalar from a table cell.
if iscell(x); x = x{1}; end
if islogical(x); x = double(x); end   % logicals are NOT numeric in MATLAB
if ischar(x) || isstring(x); x = str2double(x); end
if isempty(x) || ~isnumeric(x) || isnan(x)
    v = 0;
else
    v = double(x);
end
end


function s = safe_str(x)
% Safely extract a string from a table cell.
if iscell(x); x = x{1}; end
if isnumeric(x); x = num2str(x); end
x = string(x);
if ismissing(x); s = ''; else; s = char(x); end
end
