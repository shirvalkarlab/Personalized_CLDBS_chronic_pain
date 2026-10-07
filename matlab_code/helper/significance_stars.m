function significance_stars(box_positions, p_value)
% SIGNIFICANCE_STARS  Draw a bracket and significance marker over two boxes.
%
%   SIGNIFICANCE_STARS(box_positions, p_value) draws a horizontal bracket
%   spanning the two x-axis positions in box_positions and labels it with a
%   star code that reflects p_value:
%       p < 0.001  ***
%       p < 0.01   **
%       p < 0.05   *
%       otherwise  n.s.
%
%   INPUT
%     box_positions   Two-element vector of x positions to connect.
%     p_value         The p-value that sets the marker.
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

% Count the brackets already drawn so each new one sits lower.
existing_brackets = findobj(current_axes, 'Tag', 'significance_bracket');
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

% ---- Label it ----------------------------------------------------------
text(mean(box_positions), bracket_y, marker_text, ...
     'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'bottom', ...
     'FontSize', 9, ...
     'Tag', 'significance_bracket', ...
     'HandleVisibility', 'off');

end
