function face = condition_box_face(condition)
% CONDITION_BOX_FACE  Pale box-fill colour for a stimulation condition.
%
%   face = CONDITION_BOX_FACE(condition) returns the pale fill used by the
%   cohort box + swarm figures (Figure 6a,b). Each summary box gets a light
%   neutral tint, so the black outline and the black swarm read clearly on
%   top, rather than a saturated condition colour:
%     - baseline / sham / pre-trial   -> light grey
%     - closed-loop                   -> light blue
%     - any other active-stim loop    -> light blue (same "on therapy" tint)
%
%   This complements stim_condition_colors (the saturated line/edge palette):
%   use condition_box_face for the box FILL, and black for the box OUTLINE.
%
% INPUT:
%   condition - char/string; a condition label. Matched case-insensitively;
%               anything containing 'sham', 'baseline', or 'pre' -> grey.
%
% OUTPUT:
%   face - 1x3 RGB in [0,1], the pale fill colour.

grey_face = [0.90, 0.90, 0.90];   % baseline / sham / pre-trial
blue_face = [0.80, 0.87, 0.95];   % closed-loop and other on-therapy loops
c = lower(char(condition));
if contains(c, 'sham') || contains(c, 'baseline') || contains(c, 'pre')
    face = grey_face;
else
    face = blue_face;
end
end