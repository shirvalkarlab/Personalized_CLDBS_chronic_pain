%{
=========================================================================
 MAIN_wrapper.m
=========================================================================
 Generates the MATLAB figures and supplementary tables for:

   "Bespoke closed-loop deep brain stimulation for chronic pain:
    a randomized, double-blind crossover pilot trial"
   Shirvalkar et al., 2026, Nature Medicine.

 Each figure has its own function. This wrapper only sets the repository
 path, loads the per-figure data file, and calls the matching function.
 Every generated figure is written as a PDF into the figures/ folder.

 WHICH FIGURES ARE MADE HERE (MATLAB):
   Figure 1    Pain timeline for one participant, pre-operative through
               chronic closed-loop stimulation.
   Figure 3    Inpatient pain relief by stimulation brain region.
   Figure 5    (panels d and e) Closed-loop stimulation performance:
               control signal and effective duty cycle.
   Figure 6    (panels a to c) Double-blind closed-loop versus sham pain
               outcomes.
   Figure S7   Acute analgesia versus chronic benefit.
   Figure S11  Chronic stimulation programming history.
   Extended Data Figure 1  Longitudinal pain time course, all participants.

 The remaining panels are produced by the Python code in python_code/:
 Figures 4b-d, 5a-c, 6d-f, S1-S6, S8 and Extended Data Figures 2 and 3.

 HOW TO RUN
   Run this file from anywhere; it finds the repository root from its own
   location. Run it section by section, or all at once.

 NOTE ON STYLE
   These functions produce the raw figures. Final colour choices, panel
   arrangement, and typographic clean-up are done afterward by hand in
   Adobe Illustrator; the code here generates the starting material for
   that polishing.
=========================================================================
%}

% =======================================================================
%  REPOSITORY PATH
% =======================================================================
% By default the repository root is detected automatically from the
% location of this script (matlab_code/MAIN_wrapper.m lives one level below
% the root). No editing is required if you run this file from within the
% cloned repository. To point at a different copy, replace the automatic
% line below with an explicit path, e.g.:
%     repository_directory = '/full/path/to/Shirvalkar_2026_NatMed_deposit';
% Locate this file (also works in the Live Editor).
wrapper_file = which('MAIN_wrapper');
if isempty(wrapper_file)
    try
        wrapper_file = matlab.desktop.editor.getActiveFilename;
    catch
        error('Cannot determine MAIN_wrapper location. Run from file or add repository to path.');
    end
end

this_file_directory  = fileparts(wrapper_file);   % .../matlab_code
repository_directory = fileparts(this_file_directory);
% Add every function in matlab_code (including helper/) to the path,
% then load the shared configuration struct.
cd(repository_directory);
addpath(genpath(fullfile(repository_directory, 'matlab_code')));
configuration = configure_repository(repository_directory);

% =======================================================================
%  FIGURE 1  Pain timeline: pre-operative to chronic closed-loop (P3, pain VAS)
% =======================================================================
% Figure 1 and Extended Data Figure 1 are drawn by the SAME plotter
% (plot_pain_timecourse) straight from the per-survey pain-score CSV in
% data/extended_figures_raw_pain/, so there is no hand-curated figure data.
% Main Figure 1 is the single-participant case: participant P3, pain
% visual-analogue scale.
figure1_handle = plot_pain_timecourse(configuration, "painVAS", "P3", ...
    "Figure1", "Participant P3 - pain over the full trial (pain intensity VAS)");

% =======================================================================
%  FIGURE 3  Inpatient pain relief by stimulation brain region
% =======================================================================
figure3_data = read_excel_sheets(fullfile(configuration.data_directory, ...
                                           'figure3_inpatient_analgesia.xlsx'));
[figure3_handle, figure3_statistics] = ...
    plot_figure3_inpatient_analgesia(configuration, figure3_data);

% =======================================================================
%  FIGURE 5, PANELS d AND e  Closed-loop stimulation performance
% =======================================================================
% Panel d is the closed-loop control signal with its two decision
% thresholds; panel e is the resulting effective duty cycle over about six days,
% both from participant P3's left hemisphere. Panels a to c of Figure 5
% (ambulatory pain decoding) come from the Python notebooks.
figure5_data = read_excel_sheets(fullfile(configuration.data_directory, ...
                                           'figure5_adaptive_dbs_performance.xlsx'));
[figure5d_handle, figure5e_handle] = ...
    plot_figure5_adaptive_dbs(configuration, figure5_data);

% =======================================================================
%  FIGURE 6, PANELS a TO c  Double-blind closed-loop versus sham outcomes
% =======================================================================
% plot_figure6_double_blind_outcomes reads the per-day cohort table in
% data/figure6_double_blind_outcomes/ and draws the three panels (uniform
% numeric rating scale, index metric, and response curve).
figure6_handle     = plot_figure6_double_blind_outcomes(configuration);
% The pre-specified closed-loop-versus-sham mixed-effects tests are fitted to
% the same per-day table and printed.
figure6_statistics = compute_figure6_statistics(configuration);
% Panels d and e of Figure 6 (ancillary behavioural outcomes) are drawn by
% python_code/plot_figure6_ancillary_main.py.

% =======================================================================
%  EXTENDED DATA TABLE 2  All pain scales, closed-loop versus sham
% =======================================================================
% Writes figures/ExtendedDataTable2_all_pain_scales_closed_loop_vs_sham.csv,
% which python_code/plot_figureS5_closed_loop_duty_cycle.py also reads.
extended_table2_file = build_extended_data_table2(configuration);

% =======================================================================
%  SUPPLEMENTARY FIGURE S7  Acute analgesia versus chronic benefit
% =======================================================================
% Reads data/figureS7_acute_predicts_chronic/ itself and writes
% figures/FigureS7_acute_predicts_chronic.pdf.
figureS7_handle = plot_figureS7_acute_predicts_chronic(configuration);

% =======================================================================
%  SUPPLEMENTARY FIGURE S11  Chronic stimulation programming history
% =======================================================================
% plot_figureS11_programming_history loads the per-participant CSV files
% under data/figureS11_programming_history/ itself, so it takes only the
% configuration struct. Returns one figure per participant (saved as
% FigureS11_P1_programming_history.pdf, ...).
figureS11_handles = plot_figureS11_programming_history(configuration);

% Supplementary Figure S6 (battery efficiency) is produced by the Python
% script python_code/plot_figureS6_battery_efficiency.py, which fits the
% closed-loop drain model from the device-telemetry CSVs in
% data/figureS6_battery_efficiency/ and writes
% its PDF directly. It is intentionally not called from this MATLAB
% wrapper — the fit and figure live in Python.

% =======================================================================
%  EXTENDED DATA FIGURE 1  Longitudinal pain time course, all participants
% =======================================================================
% Draws the full ambulatory pain record for every participant (P1..P5),
% stacked on identical axes in the broken-axis style of main Figure 1, with
% dots coloured by stimulation condition (open-loop grey, closed-loop yellow,
% double-blind crossover blue), washouts as plus signs, and blinded shams as
% hollow circles. This is the SAME plotter as Figure 1, called with all five
% participants. Reads data/extended_figures_raw_pain/. The default measure is
% the pain intensity VAS; the figure is saved as
% figures/FigureExtended1_painVAS_timecourse.pdf.
figureExtended1_handle = plot_pain_timecourse(configuration, "painVAS", ...
    ["P1", "P2", "P3", "P4", "P5"]);

% =======================================================================
%  END
% =======================================================================


% =======================================================================
%  OPTIONAL  Extended Data Figure 1 for any pain measure
% =======================================================================
% Run this cell on its own (Ctrl+Enter in the MATLAB editor) to regenerate the
% longitudinal time course for a pain measure other than the default. Set
% selected_measure to one of "mayoNRS", "painVAS", "unpleasantVAS", or
% "MPQtotal". Each run writes figures/FigureExtended1_<measure>_timecourse.pdf.
%
%   selected_measure = "mayoNRS";
%   plot_pain_timecourse(configuration, selected_measure, ["P1","P2","P3","P4","P5"]);
%
% To build every available measure at once:
%
%   for selected_measure = ["mayoNRS", "painVAS", "unpleasantVAS", "MPQtotal"]
%       plot_pain_timecourse(configuration, selected_measure, ["P1","P2","P3","P4","P5"]);
%   end
