function colors = set2_qualitative_colormap(number_of_colors)
% SET2_QUALITATIVE_COLORMAP  The ColorBrewer "Set2" qualitative palette.
%
%   colors = SET2_QUALITATIVE_COLORMAP(number_of_colors) returns the first
%   number_of_colors rows of the eight-colour ColorBrewer "Set2" palette as
%   an RGB matrix in [0,1]. Set2 is a soft, evenly balanced qualitative
%   palette used here to colour the discrete stimulation-state categories in
%   the closed-loop raster.
%
%   INPUT
%     number_of_colors   How many colours to return (1 to 8).
%
%   OUTPUT
%     colors             number_of_colors-by-3 RGB matrix.
%
%   Reference: Brewer, ColorBrewer 2.0 (https://colorbrewer2.org),
%   qualitative scheme "Set2".

set2 = [
    0.400 0.761 0.647;
    0.988 0.553 0.384;
    0.553 0.627 0.796;
    0.906 0.541 0.765;
    0.651 0.847 0.329;
    1.000 0.851 0.184;
    0.898 0.769 0.580;
    0.702 0.702 0.702];

if nargin < 1
    number_of_colors = size(set2, 1);
end

if number_of_colors > size(set2, 1)
    error('set2_qualitative_colormap:tooMany', ...
        'Set2 provides at most %d colours.', size(set2, 1));
end

colors = set2(1:number_of_colors, :);

end
