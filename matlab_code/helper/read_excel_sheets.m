function sheets = read_excel_sheets(spreadsheet_file)
% READ_EXCEL_SHEETS  Load every sheet of a workbook into a struct of tables.
%
%   sheets = READ_EXCEL_SHEETS(spreadsheet_file) opens the Excel workbook at
%   spreadsheet_file and returns a struct whose field names are the sheet
%   names and whose values are the corresponding tables. This lets a figure
%   function refer to data by a readable name, for example
%   sheets.daily or sheets.stim_guess, instead of tracking sheet indices.
%
%   INPUT
%     spreadsheet_file   Full path to an .xlsx workbook.
%
%   OUTPUT
%     sheets             Struct with one field per worksheet. Column names
%                        from the spreadsheet are preserved exactly.
%
%   Sheet names are converted to valid MATLAB field names when necessary
%   (for example a leading space or a hyphen is removed), so the returned
%   field name may differ slightly from the displayed sheet name.

sheet_names = sheetnames(spreadsheet_file);
sheets      = struct();

for sheet_index = 1:numel(sheet_names)
    this_sheet_name = sheet_names{sheet_index};
    field_name      = matlab.lang.makeValidName(this_sheet_name);

    sheets.(field_name) = readtable(spreadsheet_file, ...
        'Sheet', this_sheet_name, ...
        'VariableNamingRule', 'preserve');
end

end
