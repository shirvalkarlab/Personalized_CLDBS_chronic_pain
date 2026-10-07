function color_map = stim_condition_colors()
% STIM_CONDITION_COLORS  Colour for each stimulation condition.
%
%   color_map = STIM_CONDITION_COLORS() returns the condition colours for the
%   Figure 6 cohort panels. Blue, orange, bluish green and black are Okabe-Ito
%   colours, which stay distinguishable under the common forms of colour-vision
%   deficiency; sham is neutral grey. Blue (closed-loop) against grey (sham) is
%   the main contrast.
%
% OUTPUTS:
%   color_map - [containers.Map] condition name (char) -> 1x3 RGB in [0,1].

    color_map = containers.Map( ...
        {'closed-loop', 'intermittent-loop', 'continuous-open-loop', 'sham', 'pre-trial'}, ...
        {[0.00, 0.447, 0.698], ...   % Okabe-Ito blue         — closed-loop
         [0.902, 0.624, 0.00], ...   % Okabe-Ito orange       — intermittent-loop
         [0.00, 0.620, 0.451], ...   % Okabe-Ito bluish green — continuous-open-loop
         [0.60, 0.60, 0.60], ...     % neutral grey           — sham
         [0.00, 0.00, 0.00]});       % black                  — pre-trial (baseline)
end
