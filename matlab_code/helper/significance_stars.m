function significance_stars(box_positions, p_value, marker_height)
% SIGNIFICANCE_STARS  Draw a bracket to sham and a significance marker over a box.
%
%   SIGNIFICANCE_STARS(box_positions, p_value, marker_height) draws a
%   horizontal bracket spanning the two x-axis positions in box_positions
%   (the tested region first, then sham) and writes a star code that reflects
%   p_value just above the tested region's box, at marker_height:
%       p < 0.001  ***
%       p < 0.01   **
%       p < 0.05   *
%       otherwise  n.s.
%
%   INPUT
%     box_positions   [tested region position, sham position] on the x axis.
%     p_value         The p-value that sets the marker.
%     marker_height   y value at which the marker is written (the top of the
%                     tested region's data).
%
%   The bracket is placed near the top of the current axis limits; each
%   further call on the same axes draws its bracket below the previous one.

if numel(box_positions) ~= 2
    error('significance_stars:input', 'box_positions must have two elements.');
end

% ---- Choose the star code ---------------------------------------------
if     p_value < 0.001
    marker_text = '***';
elseif p_value < 0.01
    marker_text = '**';
elseif p_value < 0.05
    marker_text = '*';
else
    marker_text = 'n.s.';
end

% ---- Vertical position near the top of the axis limits ---------------
current_axes  = gca;
y_limits      = current_axes.YLim;
axis_height   = y_limits(2) - y_limits(1);

% Count the brackets already drawn so each new one sits lower. The brackets
% are hidden from the legend, so findall (not findobj) is needed to see them.
existing_brackets = findall(current_axes, 'Type', 'line', 'Tag', 'significance_bracket');
bracket_level     = numel(existing_brackets);

bracket_y = y_limits(2) - axis_height * (0.05 + 0.07 * bracket_level);
tick_drop = axis_height * 0.015;

left_x  = min(box_positions);
right_x = max(box_positions);

% ---- Draw the bracket (two short verticals joined by a horizontal) -----
line([left_x,  left_x,  right_x, right_x], ...
     [bracket_y - tick_drop, bracket_y, bracket_y, bracket_y - tick_drop], ...
     'Color', 'k', 'LineWidth', 0.5, ...
     'Tag', 'significance_bracket', 'HandleVisibility', 'off');

% ---- Label the tested region's box ---------------------------------------
text(box_positions(1), marker_height + axis_height * 0.02, marker_text, ...
     'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'bottom', ...
     'FontSize', 9, ...
     'Tag', 'significance_marker', ...
     'HandleVisibility', 'off');

end
