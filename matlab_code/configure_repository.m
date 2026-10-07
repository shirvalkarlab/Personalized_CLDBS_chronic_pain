function configuration = configure_repository(repository_directory)
% CONFIGURE_REPOSITORY  Central settings for the Shirvalkar 2026 figure code.
%
%   configuration = CONFIGURE_REPOSITORY(repository_directory) returns a
%   single struct that every figure function reads from, keeping all paths,
%   the participant labels, and shared plotting defaults in one place.
%
%   INPUT
%     repository_directory   Full path to the root of this repository
%                            (the folder that contains matlab_code, data,
%                            figures, and neural_data).
%
%   OUTPUT
%     configuration          Struct with these fields:
%       .repository_directory   Root path (as supplied).
%       .data_directory         Folder holding the per-figure spreadsheets.
%       .figure_directory       Folder where generated figure PDFs are saved.
%       .neural_data_directory  Folder holding the ambulatory neural extracts.
%       .participants           Participant labels used throughout (P1..P6).
%       .implanted_participants The five who received a permanent implant and
%                               contribute to the ambulatory figures.
%       .index_metric           Each implanted participant's prespecified
%                               index pain scale (a pain_surveys.csv column).
%       .figure_font_size       Default font size for figure text.
%       .figure_export_options  Cell array passed to exportgraphics so every
%                               figure is written at the same resolution.
%
%   The participant lists reflect the trial: six participants completed the
%   inpatient stereo-EEG mapping phase, and the five who had meaningful
%   acute pain relief received permanent implants for the ambulatory phase.

% ---- Root and sub-folder paths -----------------------------------------
configuration.repository_directory = repository_directory;
configuration.data_directory       = fullfile(repository_directory, 'data');
configuration.figure_directory     = fullfile(repository_directory, 'figures');
configuration.neural_data_directory = fullfile(repository_directory, 'neural_data');

% ---- Participants ------------------------------------------------------
% Six participants completed the inpatient stereo-EEG mapping phase (P1..P6);
% the five with meaningful acute pain relief received permanent implants and
% contribute to the ambulatory figures (P1..P5). Every shipped file is keyed
% by the participant label alone.
configuration.participants          = {'P1'; 'P2'; 'P3'; 'P4'; 'P5'; 'P6'};
configuration.implanted_participants = {'P1'; 'P2'; 'P3'; 'P4'; 'P5'};

% Each participant's prespecified index pain scale (trial analysis plan), used
% by Figure 5, its statistics and Extended Data Table 2.
configuration.index_metric = struct('P1', "MPQtotal", 'P2', "mayoNRS", ...
    'P3', "MPQtotal", 'P4', "nocNRS", 'P5', "mayoNRS");

% ---- Shared plotting defaults ------------------------------------------
% A single font size and export setting keeps every panel visually
% consistent. Final colour and style polishing is done by hand afterward.
configuration.figure_font_size     = 9;
configuration.figure_export_options = {'Resolution', 600};

end
