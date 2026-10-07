function figure_handles = plot_figureS11_programming_history(configuration)
% PLOT_FIGURES11_PROGRAMMING_HISTORY  Chronic deep brain stimulation
% programming history, one landscape figure per participant (manuscript
% Supplementary Figure S11).
%
%   figure_handles = plot_figureS11_programming_history(configuration) draws,
%   for each implanted participant, the full longitudinal record of how their
%   two stimulation devices were programmed over the trial. Each figure has
%   seven vertically stacked tracks:
%       a) Status bar    - hemisphere, polarity, and loop mode over time,
%                          with the blinded sham episodes marked in black.
%       b) Left device   - which contact was active over time.
%       c) Right device  - which contact was active over time.
%       d) Amplitude     - stimulation amplitude (milliamps) over time.
%       e) Frequency     - stimulation frequency (hertz) over time.
%       f) Pulse width   - stimulation pulse width (microseconds) over time.
%       g) Duty cycle    - daily measured closed-loop effective duty cycle.
%
%   The drawing is done by the shared helper
%   plot_stimulation_parameter_timeline (in matlab_code/helper). This function
%   loads the per-hemisphere data tables and hands them to it.
%
%   INPUT
%     configuration   Struct from configure_repository. Uses
%                     .data_directory, .figure_directory, and
%                     .figure_export_options.
%
%   OUTPUT
%     figure_handles  Array of figure handles, one per participant drawn.
%
%   DATA (all in data/figureS11_programming_history/)
%     <hemisphere>_stim_parameters.csv    per-hemisphere stimulation parameter
%         interval table, one row per programming interval (e.g.
%         P1R_stim_parameters.csv).
%     <hemisphere>_daily_duty_cycle.csv   per-hemisphere daily measured
%         closed-loop duty cycle (day, percentDutyCycle).
%     <participant>_sham_windows.csv      blinded sham episodes for the
%         participant (sham_start, sham_stop), drawn as black bars in track a
%         (e.g. P1_sham_windows.csv).

% ---- Where the data tables live and where PDFs go ----------------------
data_directory   = fullfile(configuration.data_directory, ...
                            'figureS11_programming_history');
figure_directory = configuration.figure_directory;

% ---- Participants, device hemispheres, and per-participant plot settings
% Each participant has one or two implanted hemispheres. Anatomy names the two
% leads per device (contacts E0-E3 = lead1, E8-E11 = lead2). The plot settings
% cap the time axis and, where needed, the amplitude / frequency axes, matching
% the manuscript figure.
participants = {'P1', 'P2', 'P3', 'P4', 'P5'};

hemispheres = struct( ...
    'P1', {{'P1R'}}, ...
    'P2', {{'P2L', 'P2R'}}, ...
    'P3', {{'P3L', 'P3R'}}, ...
    'P4', {{'P4L', 'P4R'}}, ...
    'P5', {{'P5L', 'P5R'}});

anatomy = struct( ...
    'P1R', struct('lead1', 'ACC',      'lead2', 'VPL'), ...
    'P2L', struct('lead1', 'ACC',      'lead2', 'Caudate'), ...
    'P2R', struct('lead1', 'ACC',      'lead2', 'Thalamus'), ...
    'P3L', struct('lead1', 'Caudate',  'lead2', 'ACC'), ...
    'P3R', struct('lead1', 'Thalamus', 'lead2', 'IFG'), ...
    'P4L', struct('lead1', 'ACC',      'lead2', 'Caudate'), ...
    'P4R', struct('lead1', 'Thalamus', 'lead2', 'SFG'), ...
    'P5L', struct('lead1', 'GPi',      'lead2', 'SGC'), ...
    'P5R', struct('lead1', 'Thalamus', 'lead2', 'SGC'));

% x_max_months, amp_max, freq_max, pulse_width_override, note
plot_settings = struct( ...
    'P1', struct('x_max_months', 39.42, 'amp_max', 4,  'freq_max', 150, 'pw_override', [], ...
                 'note', 'Left hemisphere explanted'), ...
    'P2', struct('x_max_months', 40, 'amp_max', [], 'freq_max', [],  'pw_override', [],  'note', ''), ...
    'P3', struct('x_max_months', 38, 'amp_max', [], 'freq_max', [],  'pw_override', 200, 'note', ''), ...
    'P4', struct('x_max_months', 38, 'amp_max', [], 'freq_max', [],  'pw_override', [],  'note', ''), ...
    'P5', struct('x_max_months', 33, 'amp_max', [], 'freq_max', [],  'pw_override', [],  'note', ''));

figure_handles = gobjects(0);

for participant_index = 1:numel(participants)
    participant = participants{participant_index};
    settings    = plot_settings.(participant);
    device_hemispheres = hemispheres.(participant);

    % ---- Load the per-side parameter tables and duty cycles ------------
    left_table  = [];  right_table = [];
    left_anatomy = []; right_anatomy = [];
    left_hemisphere = ''; right_hemisphere = '';
    duty_cycle = struct();

    for hemisphere_index = 1:numel(device_hemispheres)
        hemisphere = device_hemispheres{hemisphere_index};
        side = hemisphere(end);   % 'L' or 'R'

        % Files are hemisphere-prefixed (e.g. P1R_stim_parameters.csv); the
        % participant tag is already the leading two characters of hemisphere.
        parameter_file = fullfile(data_directory, ...
            sprintf('%s_stim_parameters.csv', hemisphere));
        duty_file = fullfile(data_directory, ...
            sprintf('%s_daily_duty_cycle.csv', hemisphere));

        if ~isfile(parameter_file); continue; end
        parameter_table = read_stim_parameters(parameter_file);

        daily_duty = [];
        if isfile(duty_file)
            daily_duty = readtable(duty_file);
            daily_duty.day = datetime(daily_duty.day);
        end

        if side == 'L'
            left_table = parameter_table; left_hemisphere = hemisphere;
            if isfield(anatomy, hemisphere); left_anatomy = anatomy.(hemisphere); end
            if ~isempty(daily_duty); duty_cycle.L = daily_duty; end
        else
            right_table = parameter_table; right_hemisphere = hemisphere;
            if isfield(anatomy, hemisphere); right_anatomy = anatomy.(hemisphere); end
            if ~isempty(daily_duty); duty_cycle.R = daily_duty; end
        end
    end

    if (isempty(left_table) || height(left_table) == 0) && ...
       (isempty(right_table) || height(right_table) == 0)
        continue
    end

    % ---- Load the blinded sham episodes for this participant -----------
    % Sham-episode files are per participant (e.g. P1_sham_windows.csv). A
    % participant with no sham condition has no such file and no sham bars: P3
    % did not undergo sham testing. Starting from an empty window list makes
    % that the normal case rather than an error.
    sham_windows = datetime.empty(0, 2);
    sham_file = fullfile(data_directory, ...
        sprintf('%s_sham_windows.csv', participant));
    if isfile(sham_file)
        sham_table = readtable(sham_file);
        sham_windows = [datetime(sham_table.sham_start), ...
                        datetime(sham_table.sham_stop)];
    end

    % ---- Draw the seven-track timeline with the shared plotting code ---
    figure_handle = plot_stimulation_parameter_timeline( ...
        participant, left_table, right_table, left_anatomy, right_anatomy, ...
        settings.x_max_months, settings.amp_max, settings.freq_max, ...
        settings.pw_override, settings.note, duty_cycle, sham_windows);

    % ---- Save as a landscape US-Letter PDF into figures/ ---------------
    output_file = fullfile(figure_directory, ...
        sprintf('FigureS11_%s_programming_history.pdf', participant));
    save_landscape_letter(figure_handle, output_file);

    figure_handles(end + 1) = figure_handle; %#ok<AGROW>
end

end


function parameter_table = read_stim_parameters(parameter_file)
% Read a per-hemisphere stimulation-parameter CSV and restore the column
% types the plotter expects (datetime, logical flags, and text columns).
parameter_table = readtable(parameter_file, 'TextType', 'string');
parameter_table.datetime = datetime(parameter_table.datetime);

% Restore logical columns saved as 0/1 or "true"/"false".
for name = ["is_adaptive", "effectively_active"]
    if ismember(name, parameter_table.Properties.VariableNames)
        column = parameter_table.(name);
        if ~islogical(column)
            parameter_table.(name) = logical(double(string(column)));
        end
    end
end
end


function save_landscape_letter(figure_handle, output_file)
% Render the dense timeline to a high-resolution raster and place it on a
% single landscape US-Letter PDF page, matching the manuscript's supplement
% layout.
temporary_png = [tempname, '.png'];
exportgraphics(figure_handle, temporary_png, 'Resolution', 600, ...
    'BackgroundColor', 'white');
raster = imread(temporary_png);

letter_page = figure('Units', 'inches', 'Position', [0 0 11 8.5], ...
    'Color', 'w', 'Visible', 'off', 'PaperUnits', 'inches', ...
    'PaperOrientation', 'landscape', 'PaperType', 'usletter', ...
    'PaperPositionMode', 'auto');
page_axes = axes(letter_page, 'Position', [0.01 0.01 0.98 0.98]);
imshow(raster, 'Parent', page_axes); axis(page_axes, 'off');
exportgraphics(letter_page, output_file, 'ContentType', 'image', ...
    'Resolution', 600, 'BackgroundColor', 'white');
close(letter_page);
delete(temporary_png);
end
