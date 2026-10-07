function draw_summary_box(ax, values, x_center, box_width, face_color)
% DRAW_SUMMARY_BOX  Draw one interquartile box with median and whiskers.
%
%   DRAW_SUMMARY_BOX(ax, values, x_center, box_width, face_color) draws the
%   box used in Figure 5a,b: a filled rectangle spanning the interquartile
%   range with a black outline, a black median line, and black whiskers to
%   the furthest points within 1.5*IQR. No outlier markers are drawn (the raw
%   values are shown as a swarm on top).
%
% INPUTS:
%   ax         - target axes handle.
%   values     - numeric vector of raw values (NaNs are ignored).
%   x_center   - x-coordinate of the box centre.
%   box_width  - full width of the box in x-units.
%   face_color - 1x3 RGB fill (see condition_box_face for the standard tints).

v = values(~isnan(values));
if isempty(v); return; end

quartiles = quantile(v, [0.25, 0.50, 0.75]);
q1 = quartiles(1); med = quartiles(2); q3 = quartiles(3);
inter_quartile = q3 - q1;
whisker_low  = max(min(v), q1 - 1.5 * inter_quartile);
whisker_high = min(max(v), q3 + 1.5 * inter_quartile);

half = box_width / 2;
% Interquartile box (filled, black outline)
rectangle(ax, 'Position', [x_center - half, q1, box_width, inter_quartile], ...
    'FaceColor', face_color, 'EdgeColor', 'k', 'LineWidth', 0.8);
% Median line (heavier)
plot(ax, [x_center - half, x_center + half], [med, med], 'k-', 'LineWidth', 1.2);
% Whiskers and their caps
plot(ax, [x_center, x_center], [q3, whisker_high], 'k-', 'LineWidth', 0.8);
plot(ax, [x_center, x_center], [whisker_low, q1], 'k-', 'LineWidth', 0.8);
plot(ax, [x_center - half/2, x_center + half/2], [whisker_high, whisker_high], 'k-', 'LineWidth', 0.8);
plot(ax, [x_center - half/2, x_center + half/2], [whisker_low,  whisker_low],  'k-', 'LineWidth', 0.8);
end