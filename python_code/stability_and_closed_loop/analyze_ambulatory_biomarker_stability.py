#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_ambulatory_biomarker_stability.py
================================================================================
Supplementary Figure S8 — Stability of the ambulatory pain-decoding biomarker
                          over time.

Code for:
  "Bespoke closed-loop deep brain stimulation for chronic pain: a randomized,
   double-blind crossover pilot trial" (Shirvalkar et al., 2026, Nature Medicine).

WHAT THIS SCRIPT DOES
--------------------------------------------------------------------------------
For one implanted device side it reconstructs the ambulatory pain biomarker and
asks how stable it is across the months of chronic recording:
  1. PREPROCESSING (cell "Preprocessing"): reads Summit RC+S streaming
     sessions, segments the LFP around each pain-survey timestamp (from the
     participant's REDCap pain-cluster spreadsheet), and caches one segmented
     ".dat" file per pain event.
  2. SPECTRAL FEATURES (cells "Extract Spectral Features" / "Building Feature"):
     computes the power spectral density of each pain-locked segment, split by
     whether stimulation was on or off.
  3. FEATURE-PAIN RELATIONSHIP (cells "Feature Correlate with Pain Score",
     "Feature Comparison - t-test Method", "Threshold AUC Plot"): quantifies how
     well each frequency's power separates high- from low-pain surveys (AUC),
     with a surrogate/permutation null.
  4. STABILITY OVER TIME (cell "Figure S8 - AUC on cumulative 2-month windows"):
     recomputes the leave-one-out LDA AUC of each 5 Hz power window (0-50 Hz)
     on the first 2 months of surveys, then adds 2 months per step until all
     surveys are used, with the 30th/70th percentile pain thresholds fixed on
     all surveys. This is the Figure S8 result.

INPUTS (edit the CONFIGURATION block below)
--------------------------------------------------------------------------------
  DEVICE_SIDE       : which participant hemisphere to analyse. In the default
                     configuration this is the only setting that matters; it
                     selects the per-hemisphere feature file under
                     neural_data/ambulatory_stages123/.
  ClusterExcelPath : folder with the REDCap pain-cluster spreadsheets
  RCSDataPath      : Summit continuous bilateral streaming sessions for the device
  ProcessedFolder  : where segmented per-event ".dat" caches are written/read

OUTPUTS
--------------------------------------------------------------------------------
  Supplementary Figure S8 is written to the repository figures/ directory
  (override with OUTPUT_DIR), one heatmap per device side:
  FigureS8_<DEVICE_SIDE>.pdf.
  Every other figure (<DEVICE_SIDE>_PainFeature.pdf, the 180-day window, PSD,
  violin and LDA plots) is exploratory and goes to exploratory_output/ beside
  this script (override with EXPLORATORY_DIR). The permutation null is cached as
  <DEVICE_SIDE>ThresholdAUC.txt beside this script; P3L and P4L ship.

DEPENDENCIES
--------------------------------------------------------------------------------
  The signal-processing, graphing and Summit RC+S loader modules under
  ../shared_utilities: SignalProcessingUtility, GraphingUtility,
  PythonUtility and decoder/Summit.

NOTE
--------------------------------------------------------------------------------
  This is a research analysis script organised as Spyder cells (# %%). It
  contains several exploratory and alternative cells alongside the published
  ones. The cells named above are the ones that generate the Figure S8 panels;
  run the script cell by cell for one device side at a time.
================================================================================
"""
######################################################################
######################### Importing Libraries ########################
######################################################################
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import copy
import pandas as pd
pd.options.mode.chained_assignment = None
import json
import math
import pickle
from datetime import datetime
from pytz import timezone
from matplotlib import ticker, colors
from mplcursors import cursor
from scipy import stats, signal, io, linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics, model_selection
from statsmodels.stats import multitest

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# The analysis utility modules are imported as bare module names, so put their
# folders on sys.path before importing them.
_HERE = os.path.dirname(os.path.abspath(__file__))
# Every figure is written to the repository's figures/ directory, so output does
# not depend on the working directory the script happens to be launched from.
_REPO = os.path.dirname(os.path.dirname(_HERE))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(_REPO, "figures"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _out(filename):
    """Absolute path in the repository figures/ directory."""
    return os.path.join(OUTPUT_DIR, filename)


EXPLORATORY_DIR = os.environ.get("EXPLORATORY_DIR", os.path.join(_HERE, "exploratory_output"))


def _exploratory_out(filename):
    """Absolute path for figures that are not part of the paper."""
    os.makedirs(EXPLORATORY_DIR, exist_ok=True)
    return os.path.join(EXPLORATORY_DIR, filename)


_SHARED_UTILITIES = os.path.join(_HERE, os.pardir, "shared_utilities")
sys.path.insert(0, os.path.join(_SHARED_UTILITIES, "utility"))
sys.path.insert(0, os.path.join(_SHARED_UTILITIES, "decoder"))
sys.path.insert(0, _SHARED_UTILITIES)   # repo root, for `from utility.X import ...`

import SignalProcessingUtility as SPU   # smoothing / spectral helpers
import GraphingUtility as GU            # figure sizing helper
from PythonUtility import *             # misc list/array helpers

# ----------------------------------------------------------------------------
#  FEATURE-FILE FAST PATH
# ----------------------------------------------------------------------------
# neural_data/ambulatory_stages123/ holds one de-identified gzip HDF5 file per
# device side. Each file stores, for every pain survey, the OFF-stimulation mean
# power spectral density (200 bins, 0-99.5 Hz at 0.5 Hz, sensing channel 2)
# computed by the same spectral method used below, plus the REDCap pain scores
# attached to that survey. That is exactly the input the "Building Feature" cell
# onward needs, so Figure S8 can be reproduced from these files alone, with no
# per-event LFP caches and no Summit streaming sessions.
#
# Set USE_MINIMAL_EXTRACT = True (the default) to load the per-survey feature
# file and skip the three LFP cells (Preprocessing, Event Marking, Extract
# Spectral Features). Set it False to rebuild everything from per-event caches
# on a machine that has them.
USE_MINIMAL_EXTRACT = os.environ.get("USE_MINIMAL_EXTRACT", "1") == "1"
if not USE_MINIMAL_EXTRACT:
    import Summit                       # Medtronic Summit RC+S data loader (raw recordings only)

def load_biomarker_extract_h5(extract_path):
    """Rebuild the (PainEpochs, Frequency, Cluster, ChannelOfInterest) inputs
    that the downstream S8 analysis expects, from one per-survey HDF5 file.

    PainEpochs   list of {"PSD", "Window", "PainTimestamp", "ExcelRow"} dicts,
                 one per pain event, in the same shape the
                 "Extract Spectral Features" cell produces.
    Frequency    the 0-99.5 Hz (0.5 Hz) frequency axis of every PSD.
    Cluster      a DataFrame indexed so Cluster["<scale>"][ExcelRow] returns the
                 pain score attached to each event (the same access pattern used
                 against the REDCap pain-cluster spreadsheet).
    ChannelOfInterest  the sensing channel the PSD was computed on.
    """
    import h5py
    with h5py.File(extract_path, "r") as handle:
        frequency = handle["frequency_hz"][:].astype(float)
        psd = handle["psd"][:].astype(float)                    # (n_events, n_freq)
        cluster_row = handle["cluster_row"][:].astype(int)      # REDCap row per event
        pain_timestamp_unix = handle["pain_timestamp_unix"][:]
        channel_of_interest = int(handle.attrs.get("channel_of_interest", 2))
        # Every attached pain scale is stored as pain_<scale>.
        pain_scores = {key[len("pain_"):]: handle[key][:]
                       for key in handle.keys()
                       if key.startswith("pain_") and key != "pain_timestamp_unix"}

    # One PainEpoch per event. Window = 2*300 s to match the LFP path's
    # duration*2 tagging of the 5-minute (300 s either side) analysis window.
    # The per-event cache files are read in DESCENDING time order
    # (np.flip(sorted(...))), and the stability cell relies on that ordering
    # (it sweeps np.arange(PainTimestamp[-1], PainTimestamp[0], +step), i.e.
    # from the earliest at index -1 to the latest at index 0). The per-survey
    # file stores events ascending, so reverse them here to match.
    descending = np.argsort(pain_timestamp_unix)[::-1]
    psd = psd[descending, :]
    cluster_row = cluster_row[descending]
    pain_timestamp_unix = pain_timestamp_unix[descending]
    for key in list(pain_scores.keys()):
        pain_scores[key] = pain_scores[key][descending]

    pain_epochs = []
    for i in range(psd.shape[0]):
        pain_epochs.append({
            "PSD": psd[i, :],
            "Window": 300 * 2,
            "PainTimestamp": datetime.fromtimestamp(float(pain_timestamp_unix[i])),
            "ExcelRow": int(cluster_row[i]),
        })

    # Build a Cluster table addressable by ExcelRow. Column names are the REDCap
    # pain-scale names (MPQtotal, mayoNRS, painVAS, ...).
    # ExcelRow -1 marks an event with no matching survey row.
    has_row = cluster_row >= 0
    n_rows = int(cluster_row.max()) + 1 if len(cluster_row) else 0
    cluster = pd.DataFrame(index=range(n_rows))
    for scale, values in pain_scores.items():
        column = np.full(n_rows, np.nan)
        column[cluster_row[has_row]] = values[has_row]
        cluster[scale] = column
    event_time = np.full(n_rows, np.nan)
    event_time[cluster_row[has_row]] = pain_timestamp_unix[has_row]
    cluster["time"] = [datetime.fromtimestamp(t) if np.isfinite(t) else pd.NaT
                       for t in event_time]
    return pain_epochs, frequency, cluster, channel_of_interest


# ============================ CONFIGURATION ==============================
# Which participant hemisphere to analyze. With the default fast path this is
# the ONLY setting that matters: it selects the per-survey feature file
# {DEVICE_SIDE}_ambulatory_features.h5. Give a P-label + hemisphere,
# e.g. "P4L", "P2R". Override without editing the file:
#     DEVICE_SIDE=P2R python analyze_ambulatory_biomarker_stability.py
DEVICE_SIDE = os.environ.get("DEVICE_SIDE", "P4L")  # implanted hemisphere (P#L / P#R)
# Cached 1,000-value permutation null of the AUC for this side (read if present).
THRESHOLD_AUC_FILE = os.path.join(_HERE, DEVICE_SIDE + "ThresholdAUC.txt")

# ---- LFP-MODE ONLY (ignored when USE_MINIMAL_EXTRACT=1, the default) ----
# These paths are used only when rebuilding the per-survey features from
# per-event RC+S caches (USE_MINIMAL_EXTRACT=0). Point them at your own copies
# in that case; otherwise leave as-is.
PARTICIPANT_ID = DEVICE_SIDE[:-1]  # derived from DEVICE_SIDE; not set by hand
ClusterExcelPath = os.environ.get("CLUSTER_EXCEL_PATH", "/path/to/processed/REDcap")
RCSDataPath = os.environ.get(
    "RCS_DATA_PATH",
    f"/path/to/{PARTICIPANT_ID}/SummitData/SummitContinuousBilateralStreaming",
)
# ========================================================================
Side = f"{PARTICIPANT_ID}/{DEVICE_SIDE}"
# Path to the per-survey feature file for this device side (the default input).
MINIMAL_EXTRACT_PATH = os.environ.get(
    "MINIMAL_EXTRACT_PATH",
    os.path.join(_HERE, os.pardir, os.pardir, "neural_data",
                 "ambulatory_stages123", f"{DEVICE_SIDE}_ambulatory_features.h5"))

if not USE_MINIMAL_EXTRACT:
    # LFP mode: read the REDCap pain-cluster spreadsheet.
    Cluster = pd.read_excel(ClusterExcelPath + os.path.sep + f"{PARTICIPANT_ID}_clustered.xlsx")

# ProcessedFolder holds one segmented "cacheSession<unix>.0.dat" pickle per pain
# event (a 15-min-either-side spontaneous window, 4-channel ambulatory LFP at
# 250 Hz, with device + therapy configuration). The preprocessing cell below
# writes these from the streaming sessions; if they already exist it SKIPS that
# read, so pointing ProcessedFolder at a pre-built cache lets the analysis run
# without any streaming data. The cacheSession*.dat files are read from
# <ProcessedFolder>/<Side>/, where Side = "<PARTICIPANT_ID>/<DEVICE_SIDE>" (set just
# below). A cache whose hemisphere folders sit at the top level must be arranged
# as <PARTICIPANT_ID>/<DEVICE_SIDE>/ for the join to resolve.
ProcessedFolder = os.environ.get("PROCESSED_FOLDER", os.path.join(_HERE, "preprocessed_segments"))
if not USE_MINIMAL_EXTRACT:
    os.makedirs(ProcessedFolder + os.path.sep + Side, exist_ok=True)

# Anatomical target of each sensing channel, per participant hemisphere.
# Keyed by "P#/P#<side>" to match the `Side` selector above.
LeadLocations = {
    "P5/P5L": ['GPi', 'GPi', 'SGC', 'SGC'],
    "P5/P5R": ['Thal', 'Thal', 'SGC', 'SGC'],
    "P4/P4L": ['ACC', 'ACC', 'Caud', 'Caud'],
    "P4/P4R": ['Thal', 'Thal', 'SFG', 'SFG'],
    "P3/P3L": ['Caud', 'Caud', 'ACC', 'ACC'],
    "P3/P3R": ['Thal', 'Thal', 'IFG', 'IFG'],
    "P2/P2L": ['ACC', 'ACC', 'Caud', 'Caud'],
    "P2/P2R": ['ACC', 'ACC', 'THAL', 'THAL'],
    "P1/P1R": ['ACC', 'ACC', 'VPL', 'VPL']
}

# %% Preprocessing (To go from Medtronic JSON to segmented data)

# With the default fast path the LFP cells below (Preprocessing, Event Marking,
# monthly PainScore Thresholds, and Spectral Feature extraction) are SKIPPED:
# they read per-event .dat caches and Summit streaming sessions, which are on
# the order of 100 GB per participant. Their only outputs the rest of the script
# needs -- PainEpochs, Frequency, Cluster, ChannelOfInterest -- are loaded from
# the per-survey HDF5 feature file in the loader cell that follows. Set
# USE_MINIMAL_EXTRACT = False, with those caches in place, to run them and
# rebuild the features from the LFP.
if not USE_MINIMAL_EXTRACT:
    Description = list()

    # Loop over every pain event submitted. 
    for i in range(0,len(Cluster["time"])):
        print(i)
        DataDescription = dict()

        # A switch statement is used because different Excels by default uses different data type
        if type(Cluster["time"][i]) == pd._libs.tslibs.nattype.NaTType:
            continue
        elif type(Cluster["time"][i]) == pd._libs.tslibs.timestamps.Timestamp:
            PainTimestamp = datetime.fromtimestamp(datetime.fromisoformat(Cluster["time"][i].isoformat()).timestamp()+3600*3)
        else:
            PainTimestamp = datetime.fromtimestamp(datetime.fromisoformat(Cluster["time"][i]).timestamp()+3600*3)
        PainOnset_UnixTime = PainTimestamp.timestamp()
    
        # This loop is used to identify the closest session file related to the pain event. 
        DataFolder = Side
        for DataFolderName in np.flip(sorted(os.listdir(RCSDataPath + os.path.sep + DEVICE_SIDE))):
            if DataFolderName.startswith("Session"):
                SessionTimestamp = int(DataFolderName.replace("Session","")) / 1000
                if SessionTimestamp < PainOnset_UnixTime:
                    DataFolder = RCSDataPath + os.path.sep + DEVICE_SIDE + os.path.sep + DataFolderName
                    break
    
        # If DataFolder is never updated, means that a session file was not found
        if DataFolder == Side:
            print(f"Can't find session for Row {i}")
            continue
    
        # Create the cache file that holds the data. If cache is found, skip the file. 
        CacheFilename = ProcessedFolder + os.path.sep + Side + f"/cacheSession{PainOnset_UnixTime}.dat"
        if os.path.exists(CacheFilename):
            print("Exists " + CacheFilename)
            continue
            #os.remove(CacheFilename)
    
        # If folder exist but without JSON file, skip because not properly synced. 
        try:
            os.listdir(DataFolder)
        except FileNotFoundError:
            print("Session not Found: " + DataFolder)
            Description.append(DataDescription)
            continue
    
        # Load data using the Summit Converter in Python. Load only the LFP data type 
        try:
            Data = Summit.LoadData(DataFolder , DataType=["Lfp"])
            DataDescription["LFP_Data"] = "Lfp" in Data.keys()
        except Exception as e:
            print(e)
            continue 
    
        # If LFP data is not found in the Data folder, skip because no data. 
        if not DataDescription["LFP_Data"]:
            Description.append(DataDescription)
            print("Data not Found: " + DataFolder)
            continue
    
        # Extract Therapy configurations from the Data structure. 
        Therapy = Summit.TherapyReconstruction(Data)
        for therapyIndex in range(len(Therapy)-1,-1,-1):
            if PainOnset_UnixTime > Therapy[therapyIndex]["Time"]:
                break
    
        # Loop over all therapy, identify the group marked as Active, which will contain the active stimulation parameters used in this session. 
        for therapy in Therapy[therapyIndex]["Therapy"]:
            if therapy["ActiveGroup"] and therapy["Valid"]:
                DataDescription["StimFrequency"] = therapy["Frequency"]
                DataDescription["StimStatus"] = therapy["TherapyStatus"]
                for program in therapy["Programs"]:
                    if program["Valid"]:
                        DataDescription["StimContact"] = program["Electrode"]
                        DataDescription["StimPulsewidth"] = program["PulseWidth"]
                        DataDescription["StimCycling"] = program["Cycling"]
                        DataDescription["StimAmplitude"] = program["Amplitude"]
    
        # If LFP data is not found in the Data folder, skip because no data. 
        if "Lfp" not in Data.keys():
            Description.append(DataDescription)
            print("No Lfp Data in " + DataFolder)
            continue
    
        # Define the Cache to be 15 minutes before until 15 minutes after. Cache to be longer file but can be narrowed (typically to 10 minutes window) in further processing. 
        PreOnset = 15*60 # seconds
        PostOnset = 15*60 # seconds
    
        # Each session folder may contain multiple recording trials. 
        # Loop over each recording trial
        for nTrial in range(len(Data["Lfp"])):

            # If time variable is not in the data, indicating that data preprocessing failed (may be caused by multiple reasons but most likely due to too much missing data)
            if "Time" not in Data["Lfp"][nTrial].keys():
                print("No Timestamps")
                continue
        
            # The relative timestamp is the time of Pain Event in relative to the first timepoint of the respective trial. 
            RelativeTimestamp = PainOnset_UnixTime - Data["Lfp"][nTrial]["UnixTime"][0]

            # Select data based on window defined above
            TimeSelection = np.bitwise_and(Data["Lfp"][nTrial]["Time"] > RelativeTimestamp - PreOnset, Data["Lfp"][nTrial]["Time"] < RelativeTimestamp + PostOnset)
            SamplingRate = Data["Lfp"][nTrial]["SamplingRate"][0]
        
            # If there are data within the time window
            if np.any(TimeSelection):
                print(DataFolder + f" Lfp Trial {nTrial} Excel Row {i}")
                print(DataFolder + f" - Relative Timetsamp: {RelativeTimestamp} - Total Length: {Data['Lfp'][nTrial]['UnixTime'][-1]-Data['Lfp'][nTrial]['UnixTime'][0]}")
            
                # Create cache data structure. 
                PainEpoch = {"PainTimestamp": PainTimestamp,
                            "ReportTime": RelativeTimestamp,
                            "Data": Data["Lfp"][nTrial]["LFP"][TimeSelection,:],
                            "Time": Data["Lfp"][nTrial]["Time"][TimeSelection],
                            "Missing": Data["Lfp"][nTrial]["Missing"][TimeSelection],
                            "DriftSlope": Data["Lfp"][nTrial]["DriftSlope"],
                            "Configuration": Data["Lfp"][nTrial]["Configuration"],
                            "DeviceConfiguration": Data["Config"][0],
                            "SamplingRate": SamplingRate,
                            "PreOnset": PreOnset,
                            "PostOnset": PostOnset,
                            "RelativeTimestamp": RelativeTimestamp,
                            "Therapy": DataDescription}
            
                # Save cache. 
                CacheFilename = ProcessedFolder + os.path.sep + Side + f"/cacheSession{PainOnset_UnixTime}.dat"
                with open(CacheFilename, "wb+") as filehandler:
                    pickle.dump(PainEpoch, filehandler)
            else:
                print(np.any(TimeSelection))
                print(f"Lfp Trial {nTrial} Excel Row {i} Not Sufficient Data")
            
    # %% Event Marking Overtime

    # Loop over all cache files created above 
    CacheFiles = []
    ExistingSessionFolders = np.flip(sorted(os.listdir(ProcessedFolder + os.path.sep + Side)))
    for i in range(0,len(ExistingSessionFolders)):
        if not ExistingSessionFolders[i].startswith("cacheSession"):
            continue
    
        filename = ExistingSessionFolders[i]
        with open(ProcessedFolder + os.path.sep + Side + "/" + filename, "rb+") as filehandler:
            PainEpoch = pickle.load(filehandler)
    
        if np.sum(PainEpoch["Missing"] == 0) > 250*5*60:
            CacheFiles.append(int(ExistingSessionFolders[i].replace("cacheSession","").replace(".0.dat","")))

    # Calculate number of pain event over each monthly period. 
    EventTime = [datetime.fromisoformat(time.isoformat()).timestamp()+3600*3 for time in Cluster["time"]]
    MonthlyEventMarks, MonthlyBinMark = np.histogram(EventTime, np.arange(EventTime[0], EventTime[-1], 3600*24*30))
    MonthlyCacheMarks, MonthlyBinMark = np.histogram(CacheFiles, np.arange(EventTime[0], EventTime[-1], 3600*24*30))

    # Graphing
    fig = GU.largeFigure(1,[1280,600])
    ax = GU.addAxes(fig)

    ax.bar([datetime.fromtimestamp(time) for time in MonthlyBinMark[:-1]], MonthlyEventMarks, width=12)
    ax.bar([datetime.fromtimestamp(time+12*3600*24) for time in MonthlyBinMark[:-1]], MonthlyCacheMarks, width=12)

    ax.set_ylabel(f"Event Counts", fontsize=15)
    ax.set_xlabel("Time (PDT)", fontsize=15)
    ax.set_ylim(0,np.max(MonthlyEventMarks))

    ax.legend(["REDCap Events", "ProcessedFiles"])

    # %% PainScore Thresholds

    fig = GU.largeFigure(1,[1280,600])
    ax = GU.addAxes(fig)

    # Loop over all recordings in each month, calculate the pain threshold within each month to see the trend of pain score changes over time.
    HighPainThreshold = []
    LowPainThreshold = []
    for i in range(len(MonthlyBinMark)-1):
        Scores = Cluster["MPQtotal"][(np.array(EventTime) >= MonthlyBinMark[i]) & (np.array(EventTime) < MonthlyBinMark[i+1])]
        if len(Scores) == 0:
            HighPainThreshold.append(np.nan)
            LowPainThreshold.append(np.nan)
        else:
            HighPainThreshold.append(np.percentile(Scores, 70))
            LowPainThreshold.append(np.percentile(Scores, 30))

    ax.plot([datetime.fromtimestamp(time) for time in MonthlyBinMark[:-1]], HighPainThreshold, "ro-", linewidth=3)
    ax.plot([datetime.fromtimestamp(time) for time in MonthlyBinMark[:-1]], LowPainThreshold, "bo-", linewidth=3)

    ax.set_ylabel(f"MPQ", fontsize=15)
    ax.set_xlabel("Time (PDT)", fontsize=15)
    ax.set_ylim(0,36)

    ax.legend(["High Pain Threshold", "Low Pain Threshold"])

    # %% Extract Spectral Features

    TherapySettings = list()
    PainEpochs = list()
    ExistingSessionFolders = np.flip(sorted(os.listdir(ProcessedFolder + os.path.sep + Side)))
    Frequency = np.array(range(200))/2

    # The Channel of Interest is which LFP channels you want to analyze. 
    # Older code will attempt to run through all channels but since our features were mostly narrowed down to one channel nowaday, 
    #   Running only 1 channel make the code faster. This is 0-based, so Channel 2 = First LFP Channel of 2nd Lead. 
    ChannelOfInterest = 2

    # Duration of interest, this is # of minutes before and after pain event that we used to calculate PSDs. 
    DurationData = [300]

    RawData = []

    # Loop over all cache file
    for i in range(0,len(ExistingSessionFolders)):
        print(f"{i}/{len(ExistingSessionFolders)}")
    
        if not ExistingSessionFolders[i].startswith("cacheSession"):
            continue
    
        # Load cache file
        filename = ExistingSessionFolders[i]
        with open(ProcessedFolder + os.path.sep + Side + "/" + filename, "rb+") as filehandler:
            PainEpoch = pickle.load(filehandler)
    
        # If less than 4 seconds of data available, skip 
        if len(PainEpoch["Data"][:,ChannelOfInterest]) < 1000:
            continue
    
        # Create a 5th order Buttherworth bandpass filter between 1-100Hz. 
        [b,a] = signal.butter(5, np.array([1,100])*2/PainEpoch["SamplingRate"], 'bandpass', output='ba')
        PainEpoch["Data"][:,ChannelOfInterest] = signal.filtfilt(b,a,PainEpoch["Data"][:,ChannelOfInterest])

        # Calculate Short-time Fourier Transform using Summit's method proposed by one of the UCSF presentation, but I do not think this method actually reproduced the FFT onboard...
        # Using 2-second window, 1-second overlap, 256-point FFT (6 zero-padding). 
        PainEpoch["Spectrogram"] = Summit.TimeDomain2PowerChannel(PainEpoch["Data"][:,ChannelOfInterest].flatten(), 
                                                                time=PainEpoch["Time"], interval=500, gain=250, nFFT=256, 
                                                                fs=PainEpoch["SamplingRate"])
    
        # The missing data indicate that if a PSD calculated above contains segment of data where package lost is found. These typically lead to edge artifacts and should be removed. 
        PainEpoch["Spectrogram"]["Missing"] = np.zeros(PainEpoch["Spectrogram"]["Time"].shape)  
        for t in range(len(PainEpoch["Spectrogram"]["Missing"])):
            TimeSelection = np.bitwise_and(PainEpoch["Time"] > PainEpoch["Spectrogram"]["Time"][t]-2.5, 
                                           PainEpoch["Time"] <= PainEpoch["Spectrogram"]["Time"][t]+2.5)
            PainEpoch["Spectrogram"]["Missing"][t] = np.sum(PainEpoch["Missing"][TimeSelection])/np.sum(TimeSelection) > 0.01
    
        """
        ##### THIS SEGMENT ACTUALLY IS NOT USED, A NEW MOETHOD IS USED BELOW
        # Stimulation trend calculate the amount of variability in power for each frequency. 
        # Typically this is used to identify which frequency contains stimulation-induced artifacts. 
        StimulationTrend = np.zeros(PainEpoch["Spectrogram"]["Frequency"].shape)
        for f in range(len(PainEpoch["Spectrogram"]["Frequency"])):
            StimulationTrend[f] = SPU.stderr(np.log10(PainEpoch["Spectrogram"]["Power"][f,:]))
    
        # Find the frequency with the strongest artifacts 
        f = np.argmax(StimulationTrend[PainEpoch["Spectrogram"]["Frequency"] > 40])
        f = np.where(PainEpoch["Spectrogram"]["Frequency"] > 40)[0][0] + f

        # Moving window calculation of 5Hz window power around the stimulation band. 
        StimulationAmplitude = SPU.smooth(np.log10(np.mean(PainEpoch["Spectrogram"]["Power"][f-5:f+5,:],axis=0)),5)

        # High power = stimulation ON.
        StimulationSelection = StimulationAmplitude < ((np.max(StimulationAmplitude) + np.min(StimulationAmplitude)) / 2)
        """
    
        # Identify which electrode is used for stimulation. This give us an idea of which channel may have the strongest artifacts. 
        therapyStatus = False
        if not "StimFrequency" in PainEpoch["Therapy"].keys():
            continue 
        else:
            therapyFrequency = PainEpoch["Therapy"]["StimFrequency"]
            StimContacts = [int(i) for i in PainEpoch["Therapy"]["StimContact"].replace("+"," ").replace("-"," ").split(" ") if i.isdigit()]
        if np.any(np.array(StimContacts) > 7):
            StimLead = 2
        else:
            StimLead = 0
    
        # Stimulation PSD is calculated for the channel containing most stimulation artifacts 
        PainEpoch["StimSpectrogram"] = Summit.TimeDomain2PowerChannel(PainEpoch["Data"][:,StimLead].flatten(), 
                                                                time=PainEpoch["Time"], interval=500, gain=250, nFFT=256, 
                                                                fs=PainEpoch["SamplingRate"])
    
        # Calculate the power around the center frequency of stmulation. This didn't account for aliasing 
        FrequencySelection = np.bitwise_and(PainEpoch["StimSpectrogram"]["Frequency"] >= therapyFrequency - 2,
                                            PainEpoch["StimSpectrogram"]["Frequency"] <= therapyFrequency + 2)
    
        # Calculate the power changes on the stimulation frequency band. 
        StimPower = np.mean(PainEpoch["StimSpectrogram"]["Power"][FrequencySelection,:],axis=0)

        # If the power change is large, use the median power indicating ON/OFF separation. 
        if np.std((StimPower)) > 0.001:
            StimulationSelection = StimPower > np.median(StimPower)
            StimulationSelection = SPU.smooth(StimulationSelection, 5) > 0

        # If the power changes is minimum, either the whole recording is ON or OFF. A fixed threshold is used. 
        else:
            StimulationSelection = StimPower > 10
            StimulationSelection = SPU.smooth(StimulationSelection, 5) > 0
    
        # Loop over Pain Epoch LFP again so that we can identify which pain score to use. Excel Row is the index of the corresponding pain event in the PainReport Excel. 
        ExcelRow = -1
        for j in range(len(Cluster["time"])):
            if type(Cluster["time"][j]) == pd._libs.tslibs.timestamps.Timestamp:
                PainTimestamp = datetime.fromtimestamp(datetime.fromisoformat(Cluster["time"][j].isoformat()).timestamp()+3600*3)
            else:
                PainTimestamp = datetime.fromtimestamp(datetime.fromisoformat(Cluster["time"][j]).timestamp()+3600*3)
        
            if np.abs(PainTimestamp.timestamp() - PainEpoch["PainTimestamp"].timestamp()) < 60:
                ExcelRow = j
                break
            
        for duration in DurationData:
            # Select data based on the duration
            DataSelection = ((PainEpoch["Spectrogram"]["Time"] < PainEpoch["RelativeTimestamp"] + duration) & 
                        (PainEpoch["Spectrogram"]["Time"] > PainEpoch["RelativeTimestamp"] - duration))
        
            # If at least 2 minutes of data is available. 
            if np.sum(DataSelection) > 240:

                # Calculate the mean power from OFF stimulation and NO MISSING data period. 
                Power = np.mean(PainEpoch["Spectrogram"]["Power"][:,DataSelection & ~StimulationSelection & (PainEpoch["Spectrogram"]["Missing"] == 0)],axis=1)
                PSD = np.interp(Frequency, PainEpoch["Spectrogram"]["Frequency"], Power)
                PainEpochs.append({
                    "PSD": PSD,
                    "Window": duration*2,
                    "PainTimestamp": PainEpoch["PainTimestamp"],
                    "ExcelRow": ExcelRow
                })

# %% Load the per-survey feature file (fast path)
# When USE_MINIMAL_EXTRACT is on, rebuild exactly the variables the analysis
# from 'Building Feature' onward consumes -- PainEpochs (per-event OFF-stim
# PSD + pain-score row + timestamp), the Frequency axis, the Cluster pain-score
# table, and ChannelOfInterest -- directly from the HDF5 feature file for this
# device side, so no LFP is read.
if USE_MINIMAL_EXTRACT:
    if not os.path.exists(MINIMAL_EXTRACT_PATH):
        raise FileNotFoundError(
            f'Minimal extract not found: {MINIMAL_EXTRACT_PATH}. Build it with '
            'extract_minimal_biomarker_features.py, or set USE_MINIMAL_EXTRACT=0 '
            'to run from the raw caches.')
    PainEpochs, Frequency, Cluster, ChannelOfInterest = \
        load_biomarker_extract_h5(MINIMAL_EXTRACT_PATH)
    print(f'Loaded {len(PainEpochs)} pain events from {os.path.basename(MINIMAL_EXTRACT_PATH)}')

# %% Building Feature 
# Define Threshold used for calculating pain score feature
Threshold = [30,70]

# Create Channel Name String for graphing
ChannelName = f"Channel {ChannelOfInterest}"

# Calculate the time at which pains were calculated. 
PainTimestamp = [epoch["PainTimestamp"] for epoch in PainEpochs]

duration = 300
SelectedData = [PainEpochs[i] for i in range(len(PainEpochs)) if PainEpochs[i]["Window"] == duration*2]
PowerSpectralDensity = np.zeros((len(Frequency), len(SelectedData)))
EpochSelection = np.ones(len(SelectedData), dtype=bool)
PainScore = np.array([Cluster["MPQtotal"][epoch["ExcelRow"]] for epoch in PainEpochs if epoch["Window"] == duration*2]) 

# Loop over all recordings, 
Counter = 0
for i in range(len(PainEpochs)):
    if PainEpochs[i]["Window"] == duration*2:

        # Normalise each PSD by its mean power in 40-80 Hz; a broad band keeps narrow-band
        # artifacts from influencing the 1/f slope.
        PowerSpectralDensity[:,Counter] = PainEpochs[i]["PSD"] / np.mean(PainEpochs[i]["PSD"][rangeSelection(Frequency, [40,80])])
        if np.any(np.isnan(PowerSpectralDensity[:,Counter])):
            EpochSelection[i] = False
        Counter += 1

PowerSpectralDensity = PowerSpectralDensity[:,EpochSelection]
PainScore = PainScore[EpochSelection]

# Run zscore outlier removal on ~50Hz band to identify if a run is outlier. This typically indicates a failure in stimulation period selection done above...
# Loop until all data are within 3-standard deviations from each other. 
while True:
    OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[90:110, :],axis=0))) <= 3
    PainScore = PainScore[OffStimAccepted]
    PowerSpectralDensity = PowerSpectralDensity[:,OffStimAccepted]
    
    if np.sum(OffStimAccepted) == len(OffStimAccepted):
        break

# The AUC calculation, actual LDA building code is in this function. Feature is defined as a single power band currently. 
def calculateAUC(Feature, Label):
    PredictProbability = np.zeros(len(Label))
    LDAOutput = np.zeros(len(Label))

    # Leave-one-out, means we loop over every pain event, and predict this event by building a model using everything except this one event. 
    for i in range(len(Label)):
        TrainingSelection = np.array(range(len(Label))) != i
        lda = LinearDiscriminantAnalysis().fit(Feature[TrainingSelection,:], Label[TrainingSelection] >= np.mean(Label))
        #PredictProbability[i] = lda.predict_proba(Feature[i,:].reshape(1,-1))[0][1]
        # Instead of using the probability in Python, we only utilize the coefficients from LDA model so that we can build our customized threshold for AUC
        LDAOutput[i] = np.round(np.sum(Feature[i,:].reshape(1,-1) * lda.coef_))
    
    # Run SKLearn's ROC function by providing true label and LDA output for thresholding. 
    fpr, tpr, thresholds = metrics.roc_curve(Label >= np.mean(Label) , LDAOutput)

    # Trapz method for area-under-the-curve. 
    AUC = np.trapz(y=tpr, x=fpr)
    return AUC

# Uncomment the next line only for a participant whose pain labels have three classes.
#PainScore = np.array(PainScore == 3, dtype=float)

# Calculate AUC for every frequency band, to create AUC vs Freq graph. 
AUC = np.zeros(len(Frequency))
for f in range(0,len(Frequency),2):
    Freq = Frequency[f]
    FeatureValue = np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Freq-3, Freq+3]), :].T,axis=1).reshape((PowerSpectralDensity.shape[1],1))
    SelectionInclusion = np.bitwise_or(rangeSelection(PainScore, [0,np.percentile(PainScore, Threshold[0])], type="inclusive"), rangeSelection(PainScore, [np.percentile(PainScore, Threshold[1]),200], type="inclusive"))
    #SelectionInclusion = np.ones(PainScore.shape,dtype=bool)
    FeatureValue_Limit = FeatureValue[SelectionInclusion, :]
    PainValue_Limit = PainScore[SelectionInclusion] >= np.percentile(PainScore, Threshold[1])
    AUC[f] = calculateAUC(FeatureValue_Limit, PainValue_Limit)
    print(f"{f}/{len(Frequency)}")
AUC = AUC[::2]
AUCFrequency = Frequency[::2]

fig = GU.largeFigure(0, [800,1200])
ax = fig.subplots(2,1, sharex=True)
FigureLegends = []

GU.shadedErrorBar(Frequency, 
                  np.mean(PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], axis=1), 
                  SPU.stderr(PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], axis=1)*2, 
                  lineprops={"linewidth":1, "color": "b"}, alpha=0.3, ax=ax[0])
FigureLegends.append("Low Pain Stim Off")

GU.shadedErrorBar(Frequency, 
                  np.mean(PowerSpectralDensity[:,PainScore >= np.percentile(PainScore, Threshold[1])], axis=1), 
                  SPU.stderr(PowerSpectralDensity[:,PainScore >= np.percentile(PainScore, Threshold[1])], axis=1)*2, 
                  lineprops={"linewidth":1, "color": "r"}, alpha=0.3, ax=ax[0])
FigureLegends.append("High Pain Stim Off")

ax[0].axes.set_yscale("linear")
ax[0].set_ylim(0,400)
ax[0].set_xlim(0,80)
ax[0].set_title(f"{Side} {ChannelName} {LeadLocations[Side][ChannelOfInterest]} n={len(PainScore)}", fontsize=18)
ax[0].legend(ax[0].get_lines(), FigureLegends)
ax[0].set_xlabel("Frequency (Hz)", fontsize=15)
ax[0].set_ylabel("Power (a.u.)", fontsize=15)

DataSelection = np.bitwise_or(PainScore <= np.percentile(PainScore, Threshold[0]), PainScore > np.percentile(PainScore, Threshold[1]))
statistic, pvalue = SPU.PSDStatistic(PowerSpectralDensity[:,PainScore >= np.percentile(PainScore, Threshold[1])], PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], window=2)
rejection, corrected_pvalue = multitest.fdrcorrection(pvalue)

ax[1].plot(AUCFrequency, SPU.smooth(AUC,3), "k", linewidth=2)
ax[1].set_ylim(0, 1)

subAx = ax[1].twinx()
subAx.plot(Frequency[:200], SPU.smooth(statistic,3), "r", linewidth=2)

peakIndex = signal.find_peaks(SPU.smooth(AUC[:200],3), distance=25)[0]
peakIndex = peakIndex[peakIndex<80]
if len(peakIndex) < 2:
    peakIndex = np.concatenate((peakIndex, [0]))
peakValue = AUC[peakIndex]
peakOrders = np.flip(np.argsort(np.abs(peakValue)))
firstPeak = peakIndex[peakOrders[0]]
secondPeak = peakIndex[peakOrders[1]]

FirstPeakPValue = stats.ttest_ind(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]),:],axis=0)[PainScore >= np.percentile(PainScore, Threshold[1])], 
                np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]),:],axis=0)[PainScore <= np.percentile(PainScore, Threshold[0])])
SecondPeakPValue = stats.ttest_ind(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]),:],axis=0)[PainScore >= np.percentile(PainScore, Threshold[1])], 
                np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]),:],axis=0)[PainScore <= np.percentile(PainScore, Threshold[0])])

"""
ax[1].scatter(Frequency[firstPeak], AUC[firstPeak])
ax[1].scatter(Frequency[secondPeak], AUC[secondPeak])
ax[1].set_title(f"Peak 1 = {Frequency[firstPeak]}Hz (p={FirstPeakPValue.pvalue:.3}) | Peak 2 = {Frequency[secondPeak]}Hz (p={SecondPeakPValue.pvalue:.3})", fontsize=15)
"""

ax[1].set_ylabel("AUC", fontsize=15)
subAx.set_ylabel("t-value", fontsize=15, color="r")
ax[1].set_xlabel("Frequency (Hz)", fontsize=15)
subAx.set_ylim(-0.4, 0.4)
subAx.set_ylim(-10, 10)
ax[1].set_ylim(0.3,0.7)
ax[1].set_xlim(0,50)
cursor(multiple=True)

if not os.path.exists(THRESHOLD_AUC_FILE):
    null_generator = np.random.default_rng(0)
    ThresholdAUC = np.zeros(1000)
    for i in range(1000):
        newPainValue = null_generator.permutation(PainValue_Limit)
        ThresholdAUC[i] = calculateAUC(FeatureValue_Limit, newPainValue)
    np.savetxt(THRESHOLD_AUC_FILE, ThresholdAUC)
else:
    ThresholdAUC = np.loadtxt(THRESHOLD_AUC_FILE)

ax[1].plot(AUCFrequency[[0,-1]], [np.percentile(ThresholdAUC, 99.5), np.percentile(ThresholdAUC, 99.5)], "b--", linewidth=5)
fig.savefig(_exploratory_out(DEVICE_SIDE + "_PainFeature.pdf"))

# %% Feature Correlate with Pain Score
fig = GU.largeFigure(100, [800,600])
ax = fig.subplots(1,1)

FeatureSelection = rangeSelection(Frequency, [5,10], "inclusive")
Feature = np.mean(PowerSpectralDensity[FeatureSelection,:],axis=0)
ax.scatter(Feature, PainScore, 5, color="k")
coe = np.polyfit(Feature, PainScore, 1)
fittedLine = np.polyval(coe, [0,500])

r, p = stats.pearsonr(Feature, PainScore)

ax.plot([0,500], fittedLine, "r")
ax.set_xlabel("Spectral Feature (a.u.)", fontsize=15)
ax.set_ylabel("Pain Score (MPQ)", fontsize=15)
ax.set_title(f"r={r:.2f} p={p:.3f}")

ax.set_xlim(0,np.max(Feature)*1.1)
ax.set_ylim(0,40)

fig = GU.largeFigure(101, [800,600])
ax = fig.subplots(1,1)

blockSize = np.round(np.std(Feature))/2

for i in np.arange(0, np.max(Feature)*1.1, blockSize):
    if np.sum(rangeSelection(Feature, [i, i+blockSize], "inclusive")) > 3:
        GU.singleViolin(i + blockSize/2, PainScore[rangeSelection(Feature, [i, i+blockSize], "inclusive")], width=blockSize/2, showmeans=True)

coe = np.polyfit(Feature, PainScore, 1)
fittedLine = np.polyval(coe, [0,500])

r, p = stats.pearsonr(Feature, PainScore)

ax.plot([0,500], fittedLine, "r")
ax.set_xlabel("Spectral Feature (a.u.)", fontsize=15)
ax.set_ylabel("Pain Score (MPQ)", fontsize=15)
ax.set_title(f"r={r:.2f} p={p:.3f}")

ax.set_xlim(np.min(Feature)*0.9,np.max(Feature)*1.1)
ax.set_ylim(0,np.max(PainScore)*1.1)

# %% Heatmap of Selected Window
# NOTE: this cell rebuilds spectrograms and uses per-sample stimulation state,
# which the per-survey feature file does not carry. It is skipped in the fast
# path (the two Figure S8 results -- the AUC-vs-frequency biomarker and the
# stability-over-time AUC -- do not need it). Set USE_MINIMAL_EXTRACT = False,
# with per-event caches in place, to run it.
if not USE_MINIMAL_EXTRACT:
    Threshold = [30,70]
 
    def calculateAUC(Feature, Label):
        PredictProbability = np.zeros(len(Label))
        LDAOutput = np.zeros(len(Label))
        for i in range(len(Label)):
            TrainingSelection = np.array(range(len(Label))) != i
            lda = LinearDiscriminantAnalysis().fit(Feature[TrainingSelection,:], Label[TrainingSelection] >= np.mean(Label))
            #PredictProbability[i] = lda.predict_proba(Feature[i,:].reshape(1,-1))[0][1]
            LDAOutput[i] = np.round(np.sum(Feature[i,:].reshape(1,-1) * lda.coef_))
        fpr, tpr, thresholds = metrics.roc_curve(Label >= np.mean(Label) , LDAOutput)
        AUC = np.trapz(y=tpr, x=fpr)
        return AUC
    
    ChannelName = f"Channel {ChannelOfInterest}"
    PainTimestamp = [epoch["PainTimestamp"] for epoch in PainEpochs]

    xData = []
    yData = []
    AUCs = []
    Statistics = []

    for duration in DurationData:
        SelectedData = [PainEpochs[i] for i in range(len(PainEpochs)) if PainEpochs[i]["Window"] == duration*2]
        PowerSpectralDensity = np.zeros((len(Frequency), len(SelectedData)))
        EpochSelection = np.ones(len(SelectedData), dtype=bool)
        PainScore = np.array([Cluster["painVAS"][epoch["ExcelRow"]] for epoch in PainEpochs if epoch["Window"] == duration*2]) 

        Counter = 0
        for i in range(len(PainEpochs)):
            if PainEpochs[i]["Window"] == duration*2:
                
                PowerSpectralDensity[:,Counter] = PainEpochs[i]["PSD"] / np.mean(PainEpochs[i]["PSD"][rangeSelection(Frequency, [60,80])])
                Counter += 1
    
        PowerSpectralDensity = PowerSpectralDensity[:,EpochSelection]
        PainScore = PainScore[EpochSelection]
        
        while True:
            OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[0:40, :],axis=0))) <= 3
            PainScore = PainScore[OffStimAccepted]
            PowerSpectralDensity = PowerSpectralDensity[:,OffStimAccepted]
        
            if np.sum(OffStimAccepted) == len(OffStimAccepted):
                break
        
            OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[90:110, :],axis=0))) <= 3
            PainScore = PainScore[OffStimAccepted]
            PowerSpectralDensity = PowerSpectralDensity[:,OffStimAccepted]
        
            if np.sum(OffStimAccepted) == len(OffStimAccepted):
                break
       
        AUC = np.zeros(len(Frequency))
        for f in range(0,len(Frequency),2):
            Freq = Frequency[f]
            FeatureValue = np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Freq-3, Freq+3]), :].T,axis=1).reshape((PowerSpectralDensity.shape[1],1))
            SelectionInclusion = np.bitwise_or(rangeSelection(PainScore, [0,np.percentile(PainScore, Threshold[0])], type="inclusive"), rangeSelection(PainScore, [np.percentile(PainScore, Threshold[1]),200], type="inclusive"))
            #SelectionInclusion = np.ones(PainScore.shape,dtype=bool)
            FeatureValue_Limit = FeatureValue[SelectionInclusion, :]
            PainValue_Limit = PainScore[SelectionInclusion]
            AUC[f] = calculateAUC(FeatureValue_Limit, PainValue_Limit)
            print(f"{f}/{len(Frequency)}")
        statistic, pvalue = SPU.PSDStatistic(PowerSpectralDensity[:,PainScore >= np.percentile(PainScore, Threshold[1])], PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], window=2)

        AUCs.append(AUC[::2])
        xData = Frequency[::2]
        yData.append(duration)
        Statistics.append(statistic[::2])

# %% Graphing 
# NOTE: plots the per-duration heatmaps built by the raw-only 'Heatmap of
# Selected Window' cell above, so it is skipped in the fast path.
if not USE_MINIMAL_EXTRACT:
    fig = GU.largeFigure(0, [800,1200])
    ax = fig.subplots(1,1, sharex=True)

    image = GU.imagesc(xData, range(len(yData)), np.array(Statistics), clim=[-8,8], ax=ax)

    ax.set_ylim(0, len(yData)-1)
    #ax.set_xlim([xData[0], xData[-1]])
    ax.set_xlim(0, 50)
    ax.set_yticks(range(len(yData)))
    ax.set_yticklabels([str(y) + " seconds" for y in yData])

    GU.addColorbar(ax, image, "t statistics")

    fig = GU.largeFigure(0, [800,1200])
    ax = fig.subplots(1,1, sharex=True)

    image = GU.imagesc(xData, range(len(yData)), np.array(AUCs), clim=[0.3,0.7], ax=ax)

    ax.set_ylim(0, len(yData)-1)
    #ax.set_xlim([xData[0], xData[-1]])
    ax.set_xlim(0, 50)
    ax.set_yticks(range(len(yData)))
    ax.set_yticklabels([str(y) + " seconds" for y in yData])

    GU.addColorbar(ax, image, "AUC Curve")

# %% Figure S8 - AUC on cumulative 2-month windows
# Row k of the heatmap uses every survey from the first 2*(k+1) months after the
# first ambulatory survey (1 month = 30 days); the last row uses all surveys.
# The 30th/70th percentile pain thresholds are set once on all surveys, so the
# low- and high-pain groups are defined the same way in every row. Each column is
# the mean power in a 5 Hz window (centre +/- 2.5 Hz), centres 0-50 Hz.
Threshold = [30,70]
PainName = "MPQtotal"
duration = 300
StepSeconds = 30*24*3600*2
HalfWindow = 2.5
CentreFrequency = Frequency[::2][Frequency[::2] <= 50]

def calculateAUC(Feature, Label):
    """Leave-one-out LDA AUC. Label is True for high-pain surveys."""
    LDAOutput = np.zeros(len(Label))
    for i in range(len(Label)):
        TrainingSelection = np.arange(len(Label)) != i
        lda = LinearDiscriminantAnalysis().fit(Feature[TrainingSelection,:], Label[TrainingSelection])
        LDAOutput[i] = np.sum(Feature[i,:] * lda.coef_)
    fpr, tpr, thresholds = metrics.roc_curve(Label, LDAOutput)
    return np.trapz(y=tpr, x=fpr)

# Surveys without a pain score are left out.
Epochs = [epoch for epoch in PainEpochs if epoch["Window"] == duration*2
          and epoch["ExcelRow"] >= 0 and np.isfinite(Cluster[PainName][epoch["ExcelRow"]])]
EpochTime = np.array([epoch["PainTimestamp"].timestamp() for epoch in Epochs])
AllPainScore = np.array([Cluster[PainName][epoch["ExcelRow"]] for epoch in Epochs])
AllPower = np.array([epoch["PSD"] / np.mean(epoch["PSD"][rangeSelection(Frequency, [40,80])])
                     for epoch in Epochs]).T
LowPainThreshold = np.percentile(AllPainScore, Threshold[0])
HighPainThreshold = np.percentile(AllPainScore, Threshold[1])

TimeIndex = np.arange(np.min(EpochTime), np.max(EpochTime), StepSeconds)
AUCs = np.full((len(TimeIndex), len(CentreFrequency)), np.nan)
highData = np.zeros(len(TimeIndex), dtype=int)
lowData = np.zeros(len(TimeIndex), dtype=int)

for k, startTime in enumerate(TimeIndex):
    Include = (EpochTime <= startTime + StepSeconds) & ~np.any(np.isnan(AllPower), axis=0)
    PowerSpectralDensity = AllPower[:, Include]
    PainScore = AllPainScore[Include]

    # Remove recordings whose 45-55 Hz power is more than 3 SD from the rest.
    while True:
        OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[90:110, :],axis=0))) <= 3
        PainScore = PainScore[OffStimAccepted]
        PowerSpectralDensity = PowerSpectralDensity[:,OffStimAccepted]
        if np.sum(OffStimAccepted) == len(OffStimAccepted):
            break

    IsLow = PainScore <= LowPainThreshold
    IsHigh = rangeSelection(PainScore, [HighPainThreshold,200], type="exclusive")
    lowData[k] = np.sum(IsLow)
    highData[k] = np.sum(IsHigh)
    # A row needs at least 2 low- and 2 high-pain surveys; otherwise it is grey.
    if lowData[k] < 2 or highData[k] < 2:
        continue
    Selected = IsLow | IsHigh
    for f, Freq in enumerate(CentreFrequency):
        Band = rangeSelection(Frequency, [Freq-HalfWindow, Freq+HalfWindow], type="inclusive")
        FeatureValue = np.mean(PowerSpectralDensity[Band, :], axis=0)[Selected].reshape(-1,1)
        AUCs[k, f] = calculateAUC(FeatureValue, IsHigh[Selected])
    print(f"Figure S8 row {k+1}/{len(TimeIndex)}: {lowData[k]} low / {highData[k]} high pain")

fig = plt.figure(figsize=(4.2, 4.2))
ax = fig.subplots(1,1)
colourMap = copy.copy(plt.get_cmap("Reds"))
colourMap.set_bad("0.85")
MonthEdge = 2*len(TimeIndex)
image = ax.imshow(AUCs, cmap=colourMap, vmin=0.5, vmax=0.75, origin="lower", aspect="auto",
                  interpolation="none", extent=(CentreFrequency[0]-0.5, CentreFrequency[-1]+0.5, 0, MonthEdge))
ax.set_xlim(0, 50)
ax.set_yticks(np.arange(0, MonthEdge+1, 6))
ax.set_xlabel("Centre frequency of 5 Hz window (Hz)")
ax.set_ylabel("Months since first ambulatory survey")
ax.set_title(f"{PARTICIPANT_ID} {'Left' if DEVICE_SIDE.endswith('L') else 'Right'}")
colourBar = fig.colorbar(image, ax=ax)
colourBar.set_label("AUC, low vs high SF-MPQv2")
fig.tight_layout()
fig.savefig(_out(f"FigureS8_{DEVICE_SIDE}.pdf"))

# %% Save Results
# NOTE: this cell rebuilds spectrograms and uses per-sample stimulation state,
# which the per-survey feature file does not carry. It is skipped in the fast
# path (the two Figure S8 results -- the AUC-vs-frequency biomarker and the
# stability-over-time AUC -- do not need it). Set USE_MINIMAL_EXTRACT = False,
# with per-event caches in place, to run it.
if not USE_MINIMAL_EXTRACT:

    # Per-event feature summary written alongside the caches in LFP mode.
    FeatureFile = os.environ.get("FEATURE_OVERVIEW_CSV", "FeatureOverview.csv")
    FeatureTable = pd.read_csv(FeatureFile)

    Freq = Frequency[firstPeak]
    FeatureValue = np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Freq-3, Freq+3]), :].T,axis=1).reshape((PowerSpectralDensity.shape[1],1))
    FeatureValue_Limit = FeatureValue[SelectionInclusion, :]
    PainValue_Limit = PainScore[SelectionInclusion]

    if not os.path.exists(THRESHOLD_AUC_FILE):
        null_generator = np.random.default_rng(0)
        ThresholdAUC = np.zeros(1000)
        for i in range(1000):
            newPainValue = null_generator.permutation(PainValue_Limit)
            ThresholdAUC[i] = calculateAUC(FeatureValue_Limit, newPainValue)
        np.savetxt(THRESHOLD_AUC_FILE, ThresholdAUC)
    else:
        ThresholdAUC = np.loadtxt(THRESHOLD_AUC_FILE)

    FirstPeakPercentile = np.sum(ThresholdAUC < AUC[firstPeak]) / 1000
    SecondPeakPercentile = np.sum(ThresholdAUC < AUC[secondPeak]) / 1000

    RowFound = -1
    for n in range(len(FeatureTable["PARTICIPANT_ID"])):
        if FeatureTable["PARTICIPANT_ID"][n] == Side and FeatureTable["Channel"][n] == ChannelName:
            RowFound = n
            break

    if RowFound < 0:
        FeatureTable = FeatureTable.append({
            "PARTICIPANT_ID": Side,
            "Channel": ChannelName,
            "Target": LeadLocations[Side][SensingChannel],
            "PrimaryFeature": Frequency[firstPeak],
            "PrimaryFeatureAUC": SPU.smooth(AUC[:200],3)[firstPeak],
            "PrimaryFeatureTTestPValue":FirstPeakPValue.pvalue,
            "SecondFeature": Frequency[secondPeak],
            "SecondFeatureAUC": SPU.smooth(AUC[:200],3)[secondPeak],
            "SecondFeatureTTestPValue": SecondPeakPValue.pvalue,
            "AUCMedian": np.median(AUC)
            }, ignore_index=True)

    else:
        FeatureTable["PrimaryFeature"][RowFound] = Frequency[firstPeak]
        FeatureTable["PrimaryFeatureAUC"][RowFound] = SPU.smooth(AUC[:200],5)[firstPeak]
        FeatureTable["PrimaryFeatureTTestPValue"][RowFound] = FirstPeakPValue.pvalue
        FeatureTable["SecondFeature"][RowFound] = Frequency[secondPeak]
        FeatureTable["SecondFeatureAUC"][RowFound] = SPU.smooth(AUC[:200],5)[secondPeak]
        FeatureTable["SecondFeatureTTestPValue"][RowFound] = SecondPeakPValue.pvalue
        FeatureTable["AUCMedian"][RowFound] = np.median(AUC)

    FeatureTable.to_csv(FeatureFile, index=False)


# %% Extract Spectral Features
# NOTE: this cell rebuilds spectrograms and uses per-sample stimulation state,
# which the per-survey feature file does not carry. It is skipped in the fast
# path (the two Figure S8 results -- the AUC-vs-frequency biomarker and the
# stability-over-time AUC -- do not need it). Set USE_MINIMAL_EXTRACT = False,
# with per-event caches in place, to run it.
if not USE_MINIMAL_EXTRACT:
    TherapySettings = list()
    PainEpochs = list()
    ExistingSessionFolders = np.flip(sorted(os.listdir(ProcessedFolder + os.path.sep + Side)))
    Frequency = np.array(range(200))/2

    for i in range(0,len(ExistingSessionFolders)):
        print(f"{i}/{len(ExistingSessionFolders)}")
    
        if not ExistingSessionFolders[i].startswith("cacheSession"):
            continue
    
        filename = ExistingSessionFolders[i]
        with open(ProcessedFolder + os.path.sep + Side + "/" + filename, "rb+") as filehandler:
            PainEpoch = pickle.load(filehandler)
                
        [b,a] = signal.butter(5, np.array([1,100])*2/PainEpoch["SamplingRate"], 'bandpass', output='ba')
    
        PainEpoch["Spectrogram"] = list()
        PainEpoch["SpectrogramTD"] = list()
        for j in range(4):
            PainEpoch["Data"][:,j] = signal.filtfilt(b,a,PainEpoch["Data"][:,j])
            PainEpoch["Spectrogram"].append(Summit.TimeDomain2PowerChannel(PainEpoch["Data"][:,j].flatten(), 
                                                                           time=PainEpoch["Time"], interval=500, gain=250, nFFT=256, 
                                                                           fs=PainEpoch["SamplingRate"]))
            PainEpoch["Spectrogram"][j]["Time"] += PainEpoch["Time"][0] - 0.5
            PainEpoch["Spectrogram"][j]["Missing"] = np.zeros(PainEpoch["Spectrogram"][j]["Time"].shape)  
            for t in range(len(PainEpoch["Spectrogram"][j]["Missing"])):
                TimeSelection = np.bitwise_and(PainEpoch["Time"] > PainEpoch["Spectrogram"][j]["Time"][t]-2.5, 
                                               PainEpoch["Time"] <= PainEpoch["Spectrogram"][j]["Time"][t]+2.5)
                PainEpoch["Spectrogram"][j]["Missing"][t] = np.sum(PainEpoch["Missing"][TimeSelection])/np.sum(TimeSelection) > 0.01
        
        SelectedDuration = np.ones(PainEpoch["Spectrogram"][0]["Time"].shape, dtype=bool)
    
        if np.sum(SelectedDuration) < 300 / 2:
            print("Short Duration")
            continue
    
        PowerSpectralDensity = np.zeros((len(Frequency), 4))
        for j in range(4):
            TimeSelection = np.bitwise_and(SelectedDuration, PainEpoch["Spectrogram"][j]["Missing"]==0)
            Power = np.mean(PainEpoch["Spectrogram"][j]["Power"][:,TimeSelection],axis=1)
            PowerSpectralDensity[:,j] = np.interp(Frequency, PainEpoch["Spectrogram"][j]["Frequency"], Power)
    
        if np.any(np.isnan(PowerSpectralDensity)):
            print("Bad Data")
            continue
        
        therapyStatus = False
        if not "StimFrequency" in PainEpoch["Therapy"].keys():
            continue 
        else:
            therapyFrequency = PainEpoch["Therapy"]["StimFrequency"]
            StimContacts = [int(i) for i in PainEpoch["Therapy"]["StimContact"].replace("+"," ").replace("-"," ").split(" ") if i.isdigit()]
    
        if np.any(np.array(StimContacts) > 7):
            StimLead = 2
        else:
            StimLead = 0

        FrequencySelection = np.bitwise_and(PainEpoch["Spectrogram"][StimLead]["Frequency"] >= therapyFrequency - 2,
                                            PainEpoch["Spectrogram"][StimLead]["Frequency"] <= therapyFrequency + 2)
    
        StimPower = np.mean(PainEpoch["Spectrogram"][StimLead]["Power"][FrequencySelection,:],axis=0)
        PainEpoch["StimulationSelection"] = StimPower
    
        for j in range(len(Cluster["time"])):
            if type(Cluster["time"][j]) == pd._libs.tslibs.timestamps.Timestamp:
                PainTimestamp = datetime.fromtimestamp(datetime.fromisoformat(Cluster["time"][j].isoformat()).timestamp()+3600*3)
            else:
                PainTimestamp = datetime.fromtimestamp(datetime.fromisoformat(Cluster["time"][j]).timestamp()+3600*3)
        
            if np.abs(PainTimestamp.timestamp() - PainEpoch["PainTimestamp"].timestamp()) < 60:
                PainEpochs.append({"PSD": PowerSpectralDensity,
                                   "Spectrogram": PainEpoch["Spectrogram"],
                                   "Configuration": PainEpoch["Configuration"],
                                   "PainTimestamp": PainEpoch["PainTimestamp"],
                                   "TherapyStatus": therapyStatus,
                                   "TherapyFrequency": therapyFrequency,
                                   "StimulationSelection": PainEpoch["StimulationSelection"],
                                   "ExcelRow": j,})
                break

    UniqueConfigurations = uniqueList(TherapySettings)
    ConfigurationSums = np.zeros(len(UniqueConfigurations))

    for i in range(len(TherapySettings)):
        index = np.where(iterativeCompare(UniqueConfigurations, TherapySettings[i], "equal").flatten())[0][0]
        ConfigurationSums[index] += 1

# %% Pain Score Over Time
# NOTE: this cell rebuilds spectrograms and uses per-sample stimulation state,
# which the per-survey feature file does not carry. It is skipped in the fast
# path (the two Figure S8 results -- the AUC-vs-frequency biomarker and the
# stability-over-time AUC -- do not need it). Set USE_MINIMAL_EXTRACT = False,
# with per-event caches in place, to run it.
if not USE_MINIMAL_EXTRACT:
    Threshold = [30,70]
    #PainScoreKey = "MPQtotal"
    PainScoreKey = "painVAS"
    PainTimestamp = [epoch["PainTimestamp"] for epoch in PainEpochs if Cluster[PainScoreKey][epoch["ExcelRow"]] != 50]
    TherapyStatus = [epoch["TherapyStatus"] for epoch in PainEpochs if Cluster[PainScoreKey][epoch["ExcelRow"]] != 50]
    PainScore = np.array([Cluster[PainScoreKey][epoch["ExcelRow"]] for epoch in PainEpochs if Cluster[PainScoreKey][epoch["ExcelRow"]] != 50])

    fig = GU.largeFigure(1,[1280,600])
    ax = GU.addAxes(fig)

    ax.plot(PainTimestamp, PainScore, "k-o")
    ax.set_ylabel(f"Pain Score ({PainScoreKey})", fontsize=15)
    ax.set_xlabel("Time (PDT)", fontsize=15)
    ax.set_ylim(0,100)

    TimeRange = ax.get_xlim()
    ax.plot(TimeRange,[np.percentile(PainScore,Threshold[1]), np.percentile(PainScore,Threshold[1])], "r--")
    ax.plot(TimeRange,[np.percentile(PainScore,Threshold[0]), np.percentile(PainScore,Threshold[0])], "b--")
    ax.set_xlim(TimeRange)

    ax.set_title(Side, fontsize=18)
    fig.savefig(_exploratory_out("PainScoreOverTime.pdf"))

# %% VAS vs MPQ
# NOTE: exploratory scatter of the REDCap pain-cluster labels (i_clusters),
# which the per-survey feature file does not carry; skipped in the fast path.
if not USE_MINIMAL_EXTRACT:

    MPQSum = np.array([Cluster["MPQtotal"][epoch["ExcelRow"]] for epoch in PainEpochs])
    VAS = np.array([Cluster["painVAS"][epoch["ExcelRow"]] for epoch in PainEpochs])
    NRS = np.array([Cluster["mayoNRS"][epoch["ExcelRow"]] for epoch in PainEpochs])
    ClusterLabel = np.array([Cluster["i_clusters"][epoch["ExcelRow"]] for epoch in PainEpochs])

    fig = GU.largeFigure(1,[800,600])
    ax = GU.addAxes(fig)

    ax.scatter(MPQSum[ClusterLabel==1], VAS[ClusterLabel==1], 5, color="b")
    ax.scatter(MPQSum[ClusterLabel==2], VAS[ClusterLabel==2], 5, color="r")
    ax.scatter(MPQSum[ClusterLabel==3], VAS[ClusterLabel==3], 5, color="g")
    ax.plot([np.percentile(PainScore,Threshold[1])+0.5, np.percentile(PainScore,Threshold[1])+0.5], [-10,110], "r--")
    ax.plot([np.percentile(PainScore,Threshold[0])-0.5, np.percentile(PainScore,Threshold[0])-0.5], [-10,110], "b--")

    ax.set_ylabel("Pain Score (Pain VAS)", fontsize=15)
    ax.set_xlabel("Pain Score (MPQ Total)", fontsize=15)
    ax.set_xlim(-2,32)
    ax.set_ylim(-2,102)
    ax.set_title(Side, fontsize=18)
    fig.savefig(_exploratory_out("VASvsMPQ.pdf"))

# %% Histogram Pain Score Split
# NOTE: draws the low/high pain-threshold overlay using TimeRange from the
# exploratory 'Pain Score Over Time' cell above; skipped in the fast path.
if not USE_MINIMAL_EXTRACT:
    Threshold = [30,70]
    PainScoreKey = "painVAS"
    PainTimestamp = [epoch["PainTimestamp"] for epoch in PainEpochs]
    PainScore = np.array([Cluster[PainScoreKey][epoch["ExcelRow"]] for epoch in PainEpochs])

    fig = GU.largeFigure(1,[800,500])
    ax = GU.addAxes(fig)

    count, PainIndex = np.histogram(PainScore, np.arange(100))
    ax.bar(PainIndex[:-1], count)

    ax.plot(TimeRange,[np.percentile(PainScore,Threshold[1]), np.percentile(PainScore,Threshold[1])], "r--")
    ax.plot(TimeRange,[np.percentile(PainScore,Threshold[0]), np.percentile(PainScore,Threshold[0])], "b--")
    ax.set_xlabel(f"Pain Score ({PainScoreKey})", fontsize=15)
    ax.set_ylabel("Count", fontsize=15)
    ax.set_xlim(-0.5,100.5)

    #ax.plot([np.percentile(PainScore,Threshold[1])+0.5, np.percentile(PainScore,Threshold[1])+0.5], [0,300], "r--")
    #ax.plot([np.percentile(PainScore,Threshold[0])-0.5, np.percentile(PainScore,Threshold[0])-0.5], [0,300], "b--")
    #ax.set_ylim([0,300])

    ax.set_title(DEVICE_SIDE, fontsize=18)
    fig.savefig(_exploratory_out("PainHistogram.pdf"))

# %% Feature Comparison - Time Comparison
# NOTE: this cell rebuilds spectrograms and uses per-sample stimulation state,
# which the per-survey feature file does not carry. It is skipped in the fast
# path (the two Figure S8 results -- the AUC-vs-frequency biomarker and the
# stability-over-time AUC -- do not need it). Set USE_MINIMAL_EXTRACT = False,
# with per-event caches in place, to run it.
if not USE_MINIMAL_EXTRACT:
    Threshold = [30,70]
    for SensingChannel in [2]:
        TimeDomainSettings = [PainEpochs[i]["Configuration"][SensingChannel] for i in range(len(PainEpochs))]
        ChannelName = f"Channel {SensingChannel}"
    
        PainTimestamp = [epoch["PainTimestamp"] for epoch in PainEpochs]
        TherapyStatus = [epoch["TherapyStatus"] for epoch in PainEpochs]
        #PainScore = np.array([Cluster["i_clusters"][epoch["ExcelRow"]] for epoch in PainEpochs])
        PainScore = np.array([Cluster["MPQtotal"][epoch["ExcelRow"]] for epoch in PainEpochs]) 
    
        PowerSpectralDensity = np.zeros((len(Frequency), len(PainEpochs)))
        PowerSpectralDensity_Stim = np.zeros((len(Frequency), len(PainEpochs)))
        EpochSelection = np.ones([PowerSpectralDensity.shape[1],2], dtype=bool)
    
        for i in range(len(PainEpochs)):
            
            if not PainEpochs[i]["Channel"].startswith(ChannelName):
                #EpochSelection[i] = False
                pass
        
            if not TimeDomainSettings[i]["Channels"] == ("E02","E00") or not TimeDomainSettings[i]["SamplingRate"] == 250:
                #EpochSelection[i] = False
                pass
        
            if PainScore[i] == 2:
                #EpochSelection[i] = False
                pass
            
            for label in [0, 1]:
                if np.std((PainEpochs[i]["StimulationSelection"])) > 0.001:
                    StimulationSelection = PainEpochs[i]["StimulationSelection"] > np.median(PainEpochs[i]["StimulationSelection"])
                    StimulationSelection = SPU.smooth(StimulationSelection, 5) > 0
                else:
                    StimulationSelection = PainEpochs[i]["StimulationSelection"] > 10
                    StimulationSelection = SPU.smooth(StimulationSelection, 5) > 0
        
                TimeSelection = np.bitwise_and(StimulationSelection==label, PainEpochs[i]["Spectrogram"][0]["Missing"]==0)
                Power = np.mean(PainEpochs[i]["Spectrogram"][SensingChannel]["Power"][:, TimeSelection],axis=1)
                Power = Power / np.mean(Power[rangeSelection(PainEpochs[i]["Spectrogram"][SensingChannel]["Frequency"], [60,80])])
            
                if label == 0:    
                    PowerSpectralDensity[:,i] = np.interp(Frequency, PainEpochs[i]["Spectrogram"][SensingChannel]["Frequency"], Power)
                    if np.any(np.isnan(PowerSpectralDensity[:,i])) or np.isnan(PainScore[i]):
                        EpochSelection[i,label] = False
            
                else:
                    PowerSpectralDensity_Stim[:,i] = np.interp(Frequency, PainEpochs[i]["Spectrogram"][SensingChannel]["Frequency"], Power)
                    if np.any(np.isnan(PowerSpectralDensity_Stim[:,i])) or np.isnan(PainScore[i]):
                        #pass
                        EpochSelection[i,label] = False
    
        PowerSpectralDensity = PowerSpectralDensity[:,EpochSelection[:,0]]
        PowerSpectralDensity_Stim = PowerSpectralDensity_Stim[:,EpochSelection[:,1]]
        PainScore_Stim = PainScore[EpochSelection[:,1]]
        PainScore = PainScore[EpochSelection[:,0]]
        print(np.sum(EpochSelection[:,0]))
    
        while True:
            OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[0:40, :],axis=0))) <= 3
            OnStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity_Stim[0:40, :],axis=0))) <= 3
            PainScore_Stim = PainScore_Stim[OnStimAccepted]
            PainScore = PainScore[OffStimAccepted]
            PowerSpectralDensity_Stim = PowerSpectralDensity_Stim[:,OnStimAccepted]
            PowerSpectralDensity = PowerSpectralDensity[:,OffStimAccepted]
        
            if np.sum(OffStimAccepted) == len(OffStimAccepted):
                break
        
            OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[90:110, :],axis=0))) <= 3
            OnStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity_Stim[90:110, :],axis=0))) <= 3
            PainScore_Stim = PainScore_Stim[OnStimAccepted]
            PainScore = PainScore[OffStimAccepted]
            PowerSpectralDensity_Stim = PowerSpectralDensity_Stim[:,OnStimAccepted]
            PowerSpectralDensity = PowerSpectralDensity[:,OffStimAccepted]
        
            if np.sum(OffStimAccepted) == len(OffStimAccepted):
                break
        
        def calculateAUC(Feature, Label):
            PredictProbability = np.zeros(len(Label))
            LDAOutput = np.zeros(len(Label))
            for i in range(len(Label)):
                TrainingSelection = np.array(range(len(Label))) != i
                lda = LinearDiscriminantAnalysis().fit(Feature[TrainingSelection,:], Label[TrainingSelection] >= np.mean(Label))
                #PredictProbability[i] = lda.predict_proba(Feature[i,:].reshape(1,-1))[0][1]
                LDAOutput[i] = np.round(np.sum(Feature[i,:].reshape(1,-1) * lda.coef_))
            fpr, tpr, thresholds = metrics.roc_curve(Label >= np.mean(Label) , LDAOutput)
            AUC = np.trapz(y=tpr, x=fpr)
            return AUC
    
        #PainScore = np.array(PainScore == 3, dtype=float)
    
        AUC = np.zeros(len(Frequency))
        for f in range(len(Frequency)):
            Freq = Frequency[f]
            FeatureValue = np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Freq-3, Freq+3]), :].T,axis=1).reshape((PowerSpectralDensity.shape[1],1))
            SelectionInclusion = np.bitwise_or(rangeSelection(PainScore, [0,np.percentile(PainScore, Threshold[0])], type="inclusive"), rangeSelection(PainScore, [np.percentile(PainScore, Threshold[1]),200], type="inclusive"))
            SelectionInclusion = np.ones(PainScore.shape,dtype=bool)
            FeatureValue_Limit = FeatureValue[SelectionInclusion, :]
            PainValue_Limit = PainScore[SelectionInclusion]
            AUC[f] = calculateAUC(FeatureValue_Limit, PainValue_Limit)
    
        fig = GU.largeFigure(0, [800,1200])
        ax = fig.subplots(2,1, sharex=True)
        FigureLegends = []
    
        if len(np.unique(PainScore)) == 2:
            GU.shadedErrorBar(Frequency, 
                              np.mean(PowerSpectralDensity[:,PainScore == np.min(PainScore)], axis=1), 
                              SPU.stderr(PowerSpectralDensity[:,PainScore == np.min(PainScore)], axis=1)*2, 
                              lineprops={"linewidth":1, "color": "b"}, alpha=0.3, ax=ax[0])
            FigureLegends.append("Low Pain Stim Off")
        
            GU.shadedErrorBar(Frequency, 
                              np.mean(PowerSpectralDensity[:,PainScore == np.max(PainScore)], axis=1), 
                              SPU.stderr(PowerSpectralDensity[:,PainScore == np.max(PainScore)], axis=1)*2, 
                              lineprops={"linewidth":1, "color": "r"}, alpha=0.3, ax=ax[0])
            FigureLegends.append("High Pain Stim Off")
        else:
            GU.shadedErrorBar(Frequency, 
                              np.mean(PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], axis=1), 
                              SPU.stderr(PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], axis=1)*2, 
                              lineprops={"linewidth":1, "color": "b"}, alpha=0.3, ax=ax[0])
            FigureLegends.append("Low Pain Stim Off")
        
            GU.shadedErrorBar(Frequency, 
                              np.mean(PowerSpectralDensity[:,PainScore > np.percentile(PainScore, Threshold[1])], axis=1), 
                              SPU.stderr(PowerSpectralDensity[:,PainScore > np.percentile(PainScore, Threshold[1])], axis=1)*2, 
                              lineprops={"linewidth":1, "color": "r"}, alpha=0.3, ax=ax[0])
            FigureLegends.append("High Pain Stim Off")
    
        """
        GU.shadedErrorBar(Frequency, 
                          np.mean(PowerSpectralDensity_Stim[:,PainScore_Stim <= np.percentile(PainScore_Stim, Threshold[0])], axis=1), 
                          SPU.stderr(PowerSpectralDensity_Stim[:,PainScore_Stim <= np.percentile(PainScore_Stim, Threshold[0])], axis=1), 
                          lineprops={"linewidth":2, "color": "b", "linestyle": "--"}, alpha=0.3, ax=ax[0])
        FigureLegends.append("Low Pain Stim On")
    
        GU.shadedErrorBar(Frequency, 
                          np.mean(PowerSpectralDensity_Stim[:,PainScore_Stim > np.percentile(PainScore_Stim, Threshold[1])], axis=1), 
                          SPU.stderr(PowerSpectralDensity_Stim[:,PainScore_Stim > np.percentile(PainScore_Stim, Threshold[1])], axis=1), 
                          lineprops={"linewidth":2, "color": "r", "linestyle": "--"}, alpha=0.3, ax=ax[0])
        FigureLegends.append("High Pain Stim On")
    
        """
    
        ax[0].axes.set_yscale("log")
        ax[0].set_ylim(2e0,5e2)
        ax[0].set_xlim(0,50)
        ax[0].set_title(f"{Side} {ChannelName} {LeadLocations[Side][SensingChannel]} n={len(PainScore)}", fontsize=18)
        ax[0].legend(ax[0].get_lines(), FigureLegends)
        ax[0].set_xlabel("Frequency (Hz)", fontsize=15)
        ax[0].set_ylabel("Power (a.u.)", fontsize=15)
    
        DataSelection = np.bitwise_or(PainScore <= np.percentile(PainScore, Threshold[0]), PainScore > np.percentile(PainScore, Threshold[1]))
        statistic, pvalue = SPU.PSDStatistic(PowerSpectralDensity[:,PainScore >= np.percentile(PainScore, Threshold[1])], PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], window=2)
        rejection, corrected_pvalue = multitest.fdrcorrection(pvalue)
    
        ax[1].plot(Frequency[:200], SPU.smooth(AUC[:200],3), "k", linewidth=2)
        ax[1].set_ylim(0, 1)
    
        subAx = ax[1].twinx()
        subAx.plot(Frequency[:200], SPU.smooth(statistic,3), "r", linewidth=2)
    
        peakIndex = signal.find_peaks(SPU.smooth(AUC[:200],3), distance=25)[0]
        peakIndex = peakIndex[peakIndex<80]
        if len(peakIndex) < 2:
            peakIndex = np.concatenate((peakIndex, [0]))
        peakValue = AUC[peakIndex]
        peakOrders = np.flip(np.argsort(np.abs(peakValue)))
        firstPeak = peakIndex[peakOrders[0]]
        secondPeak = peakIndex[peakOrders[1]]
    
        FirstPeakPValue = stats.ttest_ind(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]),:],axis=0)[PainScore >= np.percentile(PainScore, Threshold[1])], 
                        np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]),:],axis=0)[PainScore <= np.percentile(PainScore, Threshold[0])])
        SecondPeakPValue = stats.ttest_ind(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]),:],axis=0)[PainScore >= np.percentile(PainScore, Threshold[1])], 
                        np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]),:],axis=0)[PainScore <= np.percentile(PainScore, Threshold[0])])
    
        """
        ax[1].scatter(Frequency[firstPeak], AUC[firstPeak])
        ax[1].scatter(Frequency[secondPeak], AUC[secondPeak])
        ax[1].set_title(f"Peak 1 = {Frequency[firstPeak]}Hz (p={FirstPeakPValue.pvalue:.3}) | Peak 2 = {Frequency[secondPeak]}Hz (p={SecondPeakPValue.pvalue:.3})", fontsize=15)
        """
    
        ax[1].set_ylabel("AUC", fontsize=15)
        subAx.set_ylabel("t-value", fontsize=15, color="r")
        ax[1].set_xlabel("Frequency (Hz)", fontsize=15)
        subAx.set_ylim(-0.4, 0.4)
        subAx.set_ylim(-8, 8)
        ax[1].set_ylim(0.3,0.7)
        ax[1].set_xlim(0,50)
        cursor(multiple=True)
        fig.savefig(_exploratory_out("PSDComparison_by_pain_state.pdf"))
    
        # Per-event feature summary written alongside the caches in LFP mode.
        FeatureFile = os.environ.get("FEATURE_OVERVIEW_CSV", "FeatureOverview.csv")
        FeatureTable = pd.read_csv(FeatureFile)
    
        Freq = Frequency[firstPeak]
        FeatureValue = np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Freq-3, Freq+3]), :].T,axis=1).reshape((PowerSpectralDensity.shape[1],1))
        FeatureValue_Limit = FeatureValue[SelectionInclusion, :]
        PainValue_Limit = PainScore[SelectionInclusion]
    
        if not os.path.exists(THRESHOLD_AUC_FILE):
            null_generator = np.random.default_rng(0)
            ThresholdAUC = np.zeros(1000)
            for i in range(1000):
                newPainValue = null_generator.permutation(PainValue_Limit)
                ThresholdAUC[i] = calculateAUC(FeatureValue_Limit, newPainValue)
            np.savetxt(THRESHOLD_AUC_FILE, ThresholdAUC)
        else:
            ThresholdAUC = np.loadtxt(THRESHOLD_AUC_FILE)
    
        FirstPeakPercentile = np.sum(ThresholdAUC < AUC[firstPeak]) / 1000
        SecondPeakPercentile = np.sum(ThresholdAUC < AUC[secondPeak]) / 1000
    
        RowFound = -1
        for n in range(len(FeatureTable["PARTICIPANT_ID"])):
            if FeatureTable["PARTICIPANT_ID"][n] == Side and FeatureTable["Channel"][n] == ChannelName:
                RowFound = n
                break
    
        if RowFound < 0:
            FeatureTable = FeatureTable.append({
                "PARTICIPANT_ID": Side,
                "Channel": ChannelName,
                "Target": LeadLocations[Side][SensingChannel],
                "PrimaryFeature": Frequency[firstPeak],
                "PrimaryFeatureAUC": SPU.smooth(AUC[:200],3)[firstPeak],
                "PrimaryFeatureTTestPValue":FirstPeakPValue.pvalue,
                "SecondFeature": Frequency[secondPeak],
                "SecondFeatureAUC": SPU.smooth(AUC[:200],3)[secondPeak],
                "SecondFeatureTTestPValue": SecondPeakPValue.pvalue,
                "AUCMedian": np.median(AUC)
                }, ignore_index=True)
    
        else:
            FeatureTable["PrimaryFeature"][RowFound] = Frequency[firstPeak]
            FeatureTable["PrimaryFeatureAUC"][RowFound] = SPU.smooth(AUC[:200],5)[firstPeak]
            FeatureTable["PrimaryFeatureTTestPValue"][RowFound] = FirstPeakPValue.pvalue
            FeatureTable["SecondFeature"][RowFound] = Frequency[secondPeak]
            FeatureTable["SecondFeatureAUC"][RowFound] = SPU.smooth(AUC[:200],5)[secondPeak]
            FeatureTable["SecondFeatureTTestPValue"][RowFound] = SecondPeakPValue.pvalue
            FeatureTable["AUCMedian"][RowFound] = np.median(AUC)
    
        FeatureTable.to_csv(FeatureFile, index=False)

# %% Feature Comparison - t-test Method
# NOTE: this cell rebuilds spectrograms and uses per-sample stimulation state,
# which the per-survey feature file does not carry. It is skipped in the fast
# path (the two Figure S8 results -- the AUC-vs-frequency biomarker and the
# stability-over-time AUC -- do not need it). Set USE_MINIMAL_EXTRACT = False,
# with per-event caches in place, to run it.
if not USE_MINIMAL_EXTRACT:
    Threshold = [30,70]
    #Threshold = [10,10]
    for SensingChannel in range(4):
        #SensingChannel = 1
        TimeDomainSettings = [PainEpochs[i]["Configuration"][SensingChannel] for i in range(len(PainEpochs))]
        ChannelName = f"Channel {SensingChannel}"
    
        PainTimestamp = [epoch["PainTimestamp"] for epoch in PainEpochs]
        TherapyStatus = [epoch["TherapyStatus"] for epoch in PainEpochs]
        PainScore = np.array([Cluster["i_clusters"][epoch["ExcelRow"]] for epoch in PainEpochs])
        #PainScore = np.array([Cluster["MPQtotal"][epoch["ExcelRow"]] for epoch in PainEpochs])
        MPQ = np.array([Cluster["MPQtotal"][epoch["ExcelRow"]] for epoch in PainEpochs])
        VAS = np.array([Cluster["painVAS"][epoch["ExcelRow"]] for epoch in PainEpochs])
        NRS = np.array([Cluster["mayoNRS"][epoch["ExcelRow"]] for epoch in PainEpochs])
    
        PowerSpectralDensity = np.zeros((len(Frequency), len(PainEpochs)))
        PowerSpectralDensity_Stim = np.zeros((len(Frequency), len(PainEpochs)))
        EpochSelection = np.ones([PowerSpectralDensity.shape[1],2], dtype=bool)
    
        for i in range(len(PainEpochs)):
            
            if not PainEpochs[i]["Channel"].startswith(ChannelName):
                pass
        
            if not TimeDomainSettings[i]["Channels"] == ("E03","E00") or not TimeDomainSettings[i]["SamplingRate"] == 250:
                #EpochSelection[i] = False
                pass
            
            if PainScore[i] == 3:
                EpochSelection[i] = False
            
            for label in [0, 1]:
                if np.std((PainEpochs[i]["StimulationSelection"])) > 0.001:
                    StimulationSelection = PainEpochs[i]["StimulationSelection"] > np.median(PainEpochs[i]["StimulationSelection"])
                    StimulationSelection = SPU.smooth(StimulationSelection, 5) > 0
                else:
                    StimulationSelection = PainEpochs[i]["StimulationSelection"] > 10
                    StimulationSelection = SPU.smooth(StimulationSelection, 5) > 0
        
                TimeSelection = np.bitwise_and(StimulationSelection==label, PainEpochs[i]["Spectrogram"][0]["Missing"]==0)
                Power = np.mean(PainEpochs[i]["Spectrogram"][SensingChannel]["Power"][:, TimeSelection],axis=1)
                Power = Power / np.mean(Power[rangeSelection(PainEpochs[i]["Spectrogram"][SensingChannel]["Frequency"], [60,80])])
            
                if label == 0:    
                    PowerSpectralDensity[:,i] = np.interp(Frequency, PainEpochs[i]["Spectrogram"][SensingChannel]["Frequency"], Power)
                    if np.any(np.isnan(PowerSpectralDensity[:,i])) or np.isnan(PainScore[i]):
                        EpochSelection[i,label] = False
            
                else:
                    PowerSpectralDensity_Stim[:,i] = np.interp(Frequency, PainEpochs[i]["Spectrogram"][SensingChannel]["Frequency"], Power)
                    if np.any(np.isnan(PowerSpectralDensity_Stim[:,i])) or np.isnan(PainScore[i]):
                        #pass
                        EpochSelection[i,label] = False
    
        PowerSpectralDensity = PowerSpectralDensity[:,EpochSelection[:,0]]
        PowerSpectralDensity_Stim = PowerSpectralDensity_Stim[:,EpochSelection[:,1]]
        PainScore_Stim = PainScore[EpochSelection[:,1]]
        PainScore = PainScore[EpochSelection[:,0]]
        print(np.sum(EpochSelection[:,0]))
        MPQ = MPQ[EpochSelection[:,0]]
        VAS = VAS[EpochSelection[:,0]]
        NRS = NRS[EpochSelection[:,0]]
    
        while True:
            OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[0:40, :],axis=0))) <= 3
            OnStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity_Stim[0:40, :],axis=0))) <= 3
            PainScore_Stim = PainScore_Stim[OnStimAccepted]
            PainScore = PainScore[OffStimAccepted]
            MPQ = MPQ[OffStimAccepted]
            VAS = VAS[OffStimAccepted]
            NRS = NRS[OffStimAccepted]
            PowerSpectralDensity_Stim = PowerSpectralDensity_Stim[:,OnStimAccepted]
            PowerSpectralDensity = PowerSpectralDensity[:,OffStimAccepted]
        
            if np.sum(OffStimAccepted) == len(OffStimAccepted):
                break
        
            OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[90:110, :],axis=0))) <= 3
            OnStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity_Stim[90:110, :],axis=0))) <= 3
            PainScore_Stim = PainScore_Stim[OnStimAccepted]
            PainScore = PainScore[OffStimAccepted]
            MPQ = MPQ[OffStimAccepted]
            VAS = VAS[OffStimAccepted]
            NRS = NRS[OffStimAccepted]
            PowerSpectralDensity_Stim = PowerSpectralDensity_Stim[:,OnStimAccepted]
            PowerSpectralDensity = PowerSpectralDensity[:,OffStimAccepted]
        
            if np.sum(OffStimAccepted) == len(OffStimAccepted):
                break
        
        def calculateAUC(Feature, Label):
            PredictProbability = np.zeros(len(Label))
            LDAOutput = np.zeros(len(Label))
            for i in range(len(Label)):
                TrainingSelection = np.array(range(len(Label))) != i
                lda = LinearDiscriminantAnalysis().fit(Feature[TrainingSelection,:], Label[TrainingSelection] >= np.mean(Label))
                #PredictProbability[i] = lda.predict_proba(Feature[i,:].reshape(1,-1))[0][1]
                LDAOutput[i] = np.round(np.sum(Feature[i,:].reshape(1,-1) * lda.coef_))
            fpr, tpr, thresholds = metrics.roc_curve(Label >= np.mean(Label) , LDAOutput)
            AUC = np.trapz(y=tpr, x=fpr)
            return AUC
    
        PainScore = np.array(PainScore == 2, dtype=float)
    
        AUC = np.zeros(len(Frequency))
        for f in range(len(Frequency)):
            Freq = Frequency[f]
            FeatureValue = np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Freq-3, Freq+3]), :].T,axis=1).reshape((PowerSpectralDensity.shape[1],1))
            SelectionInclusion = np.bitwise_or(rangeSelection(PainScore, [0,np.percentile(PainScore, Threshold[0])], type="inclusive"), rangeSelection(PainScore, [np.percentile(PainScore, Threshold[1]),200], type="inclusive"))
            SelectionInclusion = np.ones(PainScore.shape,dtype=bool)
            FeatureValue_Limit = FeatureValue[SelectionInclusion, :]
            PainValue_Limit = PainScore[SelectionInclusion]
            AUC[f] = calculateAUC(FeatureValue_Limit, PainValue_Limit)
    
        fig = GU.largeFigure(0, [800,1200])
        ax = fig.subplots(2,1, sharex=True)
        FigureLegends = []
    
        if len(np.unique(PainScore)) == 2:
            GU.shadedErrorBar(Frequency, 
                              np.mean(PowerSpectralDensity[:,PainScore == np.min(PainScore)], axis=1), 
                              SPU.stderr(PowerSpectralDensity[:,PainScore == np.min(PainScore)], axis=1)*2, 
                              lineprops={"linewidth":1, "color": "b"}, alpha=0.3, ax=ax[0])
            FigureLegends.append("Low Pain Stim Off")
        
            GU.shadedErrorBar(Frequency, 
                              np.mean(PowerSpectralDensity[:,PainScore == np.max(PainScore)], axis=1), 
                              SPU.stderr(PowerSpectralDensity[:,PainScore == np.max(PainScore)], axis=1)*2, 
                              lineprops={"linewidth":1, "color": "r"}, alpha=0.3, ax=ax[0])
            FigureLegends.append("High Pain Stim Off")
        else:
            GU.shadedErrorBar(Frequency, 
                              np.mean(PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], axis=1), 
                              SPU.stderr(PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], axis=1)*2, 
                              lineprops={"linewidth":1, "color": "b"}, alpha=0.3, ax=ax[0])
            FigureLegends.append("Low Pain Stim Off")
        
            GU.shadedErrorBar(Frequency, 
                              np.mean(PowerSpectralDensity[:,PainScore > np.percentile(PainScore, Threshold[1])], axis=1), 
                              SPU.stderr(PowerSpectralDensity[:,PainScore > np.percentile(PainScore, Threshold[1])], axis=1)*2, 
                              lineprops={"linewidth":1, "color": "r"}, alpha=0.3, ax=ax[0])
            FigureLegends.append("High Pain Stim Off")
    
        """
        GU.shadedErrorBar(Frequency, 
                          np.mean(PowerSpectralDensity_Stim[:,PainScore_Stim <= np.percentile(PainScore_Stim, Threshold[0])], axis=1), 
                          SPU.stderr(PowerSpectralDensity_Stim[:,PainScore_Stim <= np.percentile(PainScore_Stim, Threshold[0])], axis=1), 
                          lineprops={"linewidth":2, "color": "b", "linestyle": "--"}, alpha=0.3, ax=ax[0])
        FigureLegends.append("Low Pain Stim On")
    
        GU.shadedErrorBar(Frequency, 
                          np.mean(PowerSpectralDensity_Stim[:,PainScore_Stim > np.percentile(PainScore_Stim, Threshold[1])], axis=1), 
                          SPU.stderr(PowerSpectralDensity_Stim[:,PainScore_Stim > np.percentile(PainScore_Stim, Threshold[1])], axis=1), 
                          lineprops={"linewidth":2, "color": "r", "linestyle": "--"}, alpha=0.3, ax=ax[0])
        FigureLegends.append("High Pain Stim On")
    
        """
    
        #ax[0].axes.set_yscale("log")
        ax[0].set_xlim(0,100)
        ax[0].set_ylim(0,30)
        ax[0].set_title(f"{Side} {ChannelName} {LeadLocations[Side][SensingChannel]} n={len(PainScore)}", fontsize=18)
        ax[0].legend(ax[0].get_lines(), FigureLegends)
        ax[0].set_xlabel("Frequency (Hz)", fontsize=15)
        ax[0].set_ylabel("Power (a.u.)", fontsize=15)
    
        DataSelection = np.bitwise_or(PainScore <= np.percentile(PainScore, Threshold[0]), PainScore > np.percentile(PainScore, Threshold[1]))
        statistic, pvalue = SPU.PSDStatistic(PowerSpectralDensity[:,PainScore >= np.percentile(PainScore, Threshold[1])], PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], window=2)
        rejection, corrected_pvalue = multitest.fdrcorrection(pvalue)
    
        peakIndex = signal.find_peaks(SPU.smooth(AUC[:200],3)*statistic[:200], distance=15)[0]
        peakIndex = peakIndex[peakIndex<80]
        peakValue = AUC[peakIndex]
        peakOrders = np.flip(np.argsort(np.abs(peakValue)))
        firstPeak = peakIndex[peakOrders[0]]
        secondPeak = peakIndex[peakOrders[1]]
    
        FirstPeakPValue = stats.ttest_ind(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]),:],axis=0)[PainScore >= np.percentile(PainScore, Threshold[1])], 
                        np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]),:],axis=0)[PainScore <= np.percentile(PainScore, Threshold[0])])
        SecondPeakPValue = stats.ttest_ind(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]),:],axis=0)[PainScore >= np.percentile(PainScore, Threshold[1])], 
                        np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]),:],axis=0)[PainScore <= np.percentile(PainScore, Threshold[0])])
    
        ax[1].plot(Frequency[:200], SPU.smooth(AUC[:200],3), "k", linewidth=2)
        ax[1].set_ylim(0, 1)
        ax[1].set_title(f"Peak 1 = {Frequency[firstPeak]}Hz (p={FirstPeakPValue.pvalue:.3}) | Peak 2 = {Frequency[secondPeak]}Hz (p={SecondPeakPValue.pvalue:.3})", fontsize=15)
    
        subAx = ax[1].twinx()
        subAx.plot(Frequency[:200], statistic, "r", linewidth=2)
    
        ax[1].scatter(Frequency[firstPeak], AUC[firstPeak])
        ax[1].scatter(Frequency[secondPeak], AUC[secondPeak])
    
        ax[1].set_ylabel("AUC", fontsize=15)
        subAx.set_ylabel("t-value", fontsize=15, color="r")
        ax[1].set_xlabel("Frequency (Hz)", fontsize=15)
        subAx.set_ylim(-0.4, 0.4)
        subAx.set_ylim(-5, 5)
        ax[1].set_ylim(0.3,0.7)
        ax[1].set_xlim(0,50)
        cursor(multiple=True)
        fig.savefig(_exploratory_out("PSDComparison_by_stimulation_state.pdf"))
    
        # Per-event feature summary written alongside the caches in LFP mode.
        FeatureFile = os.environ.get("FEATURE_OVERVIEW_CSV", "FeatureOverview.csv")
        FeatureTable = pd.read_csv(FeatureFile)
    
        Freq = Frequency[firstPeak]
        FeatureValue = np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Freq-3, Freq+3]), :].T,axis=1).reshape((PowerSpectralDensity.shape[1],1))
        FeatureValue_Limit = FeatureValue[SelectionInclusion, :]
        PainValue_Limit = PainScore[SelectionInclusion]
    
        if not os.path.exists(THRESHOLD_AUC_FILE):
            null_generator = np.random.default_rng(0)
            ThresholdAUC = np.zeros(1000)
            for i in range(1000):
                newPainValue = null_generator.permutation(PainValue_Limit)
                ThresholdAUC[i] = calculateAUC(FeatureValue_Limit, newPainValue)
            np.savetxt(THRESHOLD_AUC_FILE, ThresholdAUC)
        else:
            ThresholdAUC = np.loadtxt(THRESHOLD_AUC_FILE)
    
        FirstPeakPercentile = np.sum(ThresholdAUC < AUC[firstPeak]) / 1000
        SecondPeakPercentile = np.sum(ThresholdAUC < AUC[secondPeak]) / 1000
    
        RowFound = -1
        for n in range(len(FeatureTable["PARTICIPANT_ID"])):
            if FeatureTable["PARTICIPANT_ID"][n] == Side and FeatureTable["Channel"][n] == ChannelName:
                RowFound = n
                break
    
        if RowFound < 0:
            FeatureTable = FeatureTable.append({
                "PARTICIPANT_ID": Side,
                "Channel": ChannelName,
                "Target": LeadLocations[Side][SensingChannel],
                "PrimaryFeature": Frequency[firstPeak],
                "PrimaryFeatureAUC": AUC[firstPeak],
                "PrimaryFeatureAUCPercentile": FirstPeakPercentile,
                "PrimaryFeatureTTestValue": FirstPeakPValue.statistic,
                "SecondFeature": Frequency[secondPeak],
                "SecondFeatureAUC": AUC[secondPeak],
                "SecondFeatureAUCPercentile": SecondPeakPercentile,
                "SecondFeatureTTestValue": SecondPeakPValue.statistic,
                }, ignore_index=True)
    
        else:
            FeatureTable["PrimaryFeature"][RowFound] = Frequency[firstPeak]
            FeatureTable["PrimaryFeatureAUC"][RowFound] = AUC[firstPeak]
            FeatureTable["PrimaryFeatureAUCPercentile"][RowFound] = FirstPeakPercentile
            FeatureTable["PrimaryFeatureTTestValue"][RowFound] = FirstPeakPValue.statistic
            FeatureTable["SecondFeature"][RowFound] = Frequency[secondPeak]
            FeatureTable["SecondFeatureAUC"][RowFound] = AUC[secondPeak]
            FeatureTable["SecondFeatureAUCPercentile"][RowFound] = SecondPeakPercentile
            FeatureTable["SecondFeatureTTestValue"][RowFound] = SecondPeakPValue.statistic
    
        FeatureTable.to_csv(FeatureFile, index=False)

# %% Threshold AUC Plot
Threshold = [30,70]

fig = GU.largeFigure(1,[800,500])
ax = GU.addAxes(fig)

count, thresholdIndex = np.histogram(ThresholdAUC, np.arange(0,1,0.01))
ax.bar(thresholdIndex[:-1], count, width=0.008)

ax.plot([AUC[firstPeak], AUC[firstPeak]],[0, 1000], "k--")
ax.plot([AUC[secondPeak], AUC[secondPeak]],[0, 1000], "k--")
ax.set_xlabel(f"AUC Surrogate", fontsize=15)
ax.set_ylabel("Count", fontsize=15)
ax.set_ylim([0,600])

fig.savefig(_exploratory_out("SurrogateDistribution.pdf"))

# %% Feature Comparison - AUC every 180 days
Threshold = [30,70]
SensingChannel = 2  # the channel stored in neural_data/ambulatory_stages123 (attribute channel_of_interest)
ChannelName = f"Channel {SensingChannel}"

PainTimestamp = [epoch["PainTimestamp"] for epoch in PainEpochs]
# TherapyStatus is only present when running from the per-event caches; the
# feature file stores the OFF-stim PSD directly, so it is not needed here.
TherapyStatus = ([epoch["TherapyStatus"] for epoch in PainEpochs]
                 if not USE_MINIMAL_EXTRACT else [None] * len(PainEpochs))
FinalAUC = []

for timestamps in np.arange(PainTimestamp[-1].timestamp(), PainTimestamp[0].timestamp(), 3600*24*180):
    # LFP mode: rebuild OFF-stim PSD from each event's time-resolved spectrogram.
    if not USE_MINIMAL_EXTRACT:
        PainScore = np.array([Cluster["MPQtotal"][epoch["ExcelRow"]] for epoch in PainEpochs])
        PowerSpectralDensity = np.zeros((len(Frequency), len(PainEpochs)))
        PowerSpectralDensity_Stim = np.zeros((len(Frequency), len(PainEpochs)))
        EpochSelection = np.ones([PowerSpectralDensity.shape[1],2], dtype=bool)

        for i in range(len(PainEpochs)):
            if not PainTimestamp[i].timestamp() < timestamps:
                EpochSelection[i] = False
                pass
            
            if not PainEpochs[i]["Channel"].startswith(ChannelName):
                #EpochSelection[i] = False
                pass
            
            if PainScore[i] == 0:
                EpochSelection[i] = False
        
            for label in [0, 1]:
                if np.std((PainEpochs[i]["StimulationSelection"])) > 0.001:
                    StimulationSelection = PainEpochs[i]["StimulationSelection"] > np.median(PainEpochs[i]["StimulationSelection"])
                    StimulationSelection = SPU.smooth(StimulationSelection, 5) > 0
                else:
                    StimulationSelection = PainEpochs[i]["StimulationSelection"] > 10
                    StimulationSelection = SPU.smooth(StimulationSelection, 5) > 0
        
                TimeSelection = np.bitwise_and(StimulationSelection==label, PainEpochs[i]["Spectrogram"][0]["Missing"]==0)
                Power = np.mean(PainEpochs[i]["Spectrogram"][SensingChannel]["Power"][:, TimeSelection],axis=1)
                Power = Power / np.mean(Power[rangeSelection(PainEpochs[i]["Spectrogram"][SensingChannel]["Frequency"], [70,80])])
        
                if label == 0:    
                    PowerSpectralDensity[:,i] = np.interp(Frequency, PainEpochs[i]["Spectrogram"][SensingChannel]["Frequency"], Power)
                    if np.any(np.isnan(PowerSpectralDensity[:,i])) or np.isnan(PainScore[i]):
                        EpochSelection[i,label] = False
            
                else:
                    PowerSpectralDensity_Stim[:,i] = np.interp(Frequency, PainEpochs[i]["Spectrogram"][SensingChannel]["Frequency"], Power)
                    if np.any(np.isnan(PowerSpectralDensity_Stim[:,i])) or np.isnan(PainScore[i]):
                        #pass
                        EpochSelection[i,label] = False

        PowerSpectralDensity = PowerSpectralDensity[:,EpochSelection[:,0]]
        PowerSpectralDensity_Stim = PowerSpectralDensity_Stim[:,EpochSelection[:,1]]
        PainScore_Stim = PainScore[EpochSelection[:,1]]
        PainScore = PainScore[EpochSelection[:,0]]
    
        while True:
            OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[10:40, :],axis=0))) <= 3
            OnStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity_Stim[10:40, :],axis=0))) <= 3
            PainScore_Stim = PainScore_Stim[OnStimAccepted]
            PainScore = PainScore[OffStimAccepted]
            PowerSpectralDensity_Stim = PowerSpectralDensity_Stim[:,OnStimAccepted]
            PowerSpectralDensity = PowerSpectralDensity[:,OffStimAccepted]
        
            if np.sum(OffStimAccepted) == len(OffStimAccepted):
                break
        
            OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[90:110, :],axis=0))) <= 3
            OnStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity_Stim[90:110, :],axis=0))) <= 3
            PainScore_Stim = PainScore_Stim[OnStimAccepted]
            PainScore = PainScore[OffStimAccepted]
            PowerSpectralDensity_Stim = PowerSpectralDensity_Stim[:,OnStimAccepted]
            PowerSpectralDensity = PowerSpectralDensity[:,OffStimAccepted]
        
            if np.sum(OffStimAccepted) == len(OffStimAccepted):
                break
            
    else:
        # FAST PATH: the feature file stores each event's OFF-stim mean PSD, so
        # build the frequency x event matrix from it directly and apply the same
        # broadband (70-80 Hz) normalisation and z-score outlier removal the LFP
        # path uses. This figure uses only the OFF-stim biomarker, so the stim-on
        # arrays are left empty.
        PainScore = np.array([Cluster['MPQtotal'][epoch['ExcelRow']] for epoch in PainEpochs])
        within_window = np.array([epoch['PainTimestamp'].timestamp() < timestamps
                                  for epoch in PainEpochs])
        psd_matrix = np.zeros((len(Frequency), len(PainEpochs)))
        keep = within_window & np.isfinite(PainScore) & (PainScore != 0)
        norm_band = rangeSelection(Frequency, [70, 80])
        for i, epoch in enumerate(PainEpochs):
            spectrum = np.asarray(epoch['PSD'], dtype=float)
            spectrum = spectrum / np.mean(spectrum[norm_band])
            psd_matrix[:, i] = spectrum
            if np.any(np.isnan(spectrum)):
                keep[i] = False
        PowerSpectralDensity = psd_matrix[:, keep]
        PainScore = PainScore[keep]
        PowerSpectralDensity_Stim = np.zeros((len(Frequency), 0))
        PainScore_Stim = np.array([])
        # Iterative z-score outlier removal on the low- and high-band means.
        while True:
            OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[10:40, :], axis=0))) <= 3
            PainScore = PainScore[OffStimAccepted]
            PowerSpectralDensity = PowerSpectralDensity[:, OffStimAccepted]
            if np.sum(OffStimAccepted) == len(OffStimAccepted):
                break
            OffStimAccepted = np.abs(stats.zscore(np.mean(PowerSpectralDensity[90:110, :], axis=0))) <= 3
            PainScore = PainScore[OffStimAccepted]
            PowerSpectralDensity = PowerSpectralDensity[:, OffStimAccepted]
            if np.sum(OffStimAccepted) == len(OffStimAccepted):
                break
    if len(OffStimAccepted) < 10:
        continue
        
    def calculateAUC(Feature, Label):
        PredictProbability = np.zeros(len(Label))
        LDAOutput = np.zeros(len(Label))
        for i in range(len(Label)):
            TrainingSelection = np.array(range(len(Label))) != i
            lda = LinearDiscriminantAnalysis().fit(Feature[TrainingSelection,:], Label[TrainingSelection] > np.percentile(Label, 50))
            PredictProbability[i] = lda.predict_proba(Feature[i,:].reshape(1,-1))[0][1]
            LDAOutput[i] = np.sum(Feature[i,:].reshape(1,-1) * lda.coef_)
        fpr, tpr, thresholds = metrics.roc_curve(Label >= np.percentile(Label, 50) , LDAOutput)
        AUC = np.trapz(y=tpr, x=fpr)
        return AUC
    
    AUC = np.zeros(len(Frequency))
    for f in range(len(Frequency)):
        Freq = Frequency[f]
        FeatureValue = np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Freq-3, Freq+3]), :].T,axis=1).reshape((PowerSpectralDensity.shape[1],1))
        FeatureValue_Limit = FeatureValue[np.bitwise_or(rangeSelection(PainScore, [0,np.percentile(PainScore, Threshold[0])], type="inclusive"), rangeSelection(PainScore, [np.percentile(PainScore, Threshold[1]),200], type="inclusive")), :]
        PainValue_Limit = PainScore[np.bitwise_or(rangeSelection(PainScore, [0,np.percentile(PainScore, Threshold[0])], type="inclusive"), rangeSelection(PainScore, [np.percentile(PainScore, Threshold[1]),200], type="inclusive"))]
        AUC[f] = calculateAUC(FeatureValue_Limit, PainValue_Limit)
        
    FinalAUC.append(AUC)
    
    fig = GU.largeFigure(0, [800,800])
    ax = fig.subplots(2,1, sharex=True)
    
    FigureLegends = []
    
    GU.shadedErrorBar(Frequency, 
                      np.mean(PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], axis=1), 
                      SPU.stderr(PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], axis=1), 
                      lineprops={"linewidth":2, "color": "b"}, alpha=0.3, ax=ax[0])
    FigureLegends.append("Low Pain Stim Off")
    
    GU.shadedErrorBar(Frequency, 
                      np.mean(PowerSpectralDensity[:,PainScore > np.percentile(PainScore, Threshold[1])], axis=1), 
                      SPU.stderr(PowerSpectralDensity[:,PainScore > np.percentile(PainScore, Threshold[1])], axis=1), 
                      lineprops={"linewidth":2, "color": "r"}, alpha=0.3, ax=ax[0])
    FigureLegends.append("High Pain Stim Off")
    
    """
    GU.shadedErrorBar(Frequency, 
                      np.mean(PowerSpectralDensity_Stim[:,PainScore_Stim <= np.percentile(PainScore_Stim, Threshold[0])], axis=1), 
                      SPU.stderr(PowerSpectralDensity_Stim[:,PainScore_Stim <= np.percentile(PainScore_Stim, Threshold[0])], axis=1), 
                      lineprops={"linewidth":2, "color": "b", "linestyle": "--"}, alpha=0.3, ax=ax[0])
    FigureLegends.append("Low Pain Stim On")
    
    GU.shadedErrorBar(Frequency, 
                      np.mean(PowerSpectralDensity_Stim[:,PainScore_Stim > np.percentile(PainScore_Stim, Threshold[1])], axis=1), 
                      SPU.stderr(PowerSpectralDensity_Stim[:,PainScore_Stim > np.percentile(PainScore_Stim, Threshold[1])], axis=1), 
                      lineprops={"linewidth":2, "color": "r", "linestyle": "--"}, alpha=0.3, ax=ax[0])
    FigureLegends.append("High Pain Stim On")
    
    """
    
    ax[0].axes.set_yscale("log")
    ax[0].set_xlim(0,100)
    #ax[0].set_ylim(0,5e5)
    ax[0].set_title(f"{Side} {ChannelName} {datetime.fromtimestamp(timestamps).strftime('%Y-%m-%d')} {LeadLocations[Side][SensingChannel]} n={len(PainScore)}", fontsize=18)
    ax[0].legend(ax[0].get_lines(), FigureLegends)
    ax[0].set_xlabel("Frequency (Hz)", fontsize=15)
    ax[0].set_ylabel("Power (a.u.)", fontsize=15)
    
    DataSelection = np.bitwise_or(PainScore <= np.percentile(PainScore, Threshold[0]), PainScore > np.percentile(PainScore, Threshold[1]))
    statistic, pvalue = SPU.PSDStatistic(PowerSpectralDensity[:,PainScore >= np.percentile(PainScore, Threshold[1])], PowerSpectralDensity[:,PainScore <= np.percentile(PainScore, Threshold[0])], window=2)
    rejection, corrected_pvalue = multitest.fdrcorrection(pvalue)
    
    peakIndex = signal.find_peaks(AUC[:80])[0]
    if len(peakIndex) == 0:
        continue
    peakValue = AUC[peakIndex]
    peakOrders = np.flip(np.argsort(np.abs(peakValue)))
    firstPeak = peakIndex[peakOrders[0]]
    secondPeak = peakIndex[peakOrders[1]]
    
    FirstPeakPValue = stats.ttest_ind(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]),:],axis=0)[PainScore >= np.percentile(PainScore, Threshold[1])], 
                    np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]),:],axis=0)[PainScore <= np.percentile(PainScore, Threshold[0])])
    SecondPeakPValue = stats.ttest_ind(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]),:],axis=0)[PainScore >= np.percentile(PainScore, Threshold[1])], 
                    np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]),:],axis=0)[PainScore <= np.percentile(PainScore, Threshold[0])])
    
    ax[1].plot(Frequency[:200], AUC[:200], "k", linewidth=2)
    ax[1].set_ylim(0, 1)
    ax[1].set_title(f"Peak 1 = {Frequency[firstPeak]}Hz (p={FirstPeakPValue.pvalue:.3}) | Peak 2 = {Frequency[secondPeak]}Hz (p={SecondPeakPValue.pvalue:.3})", fontsize=15)
    
    subAx = ax[1].twinx()
    subAx.plot(Frequency[:200], statistic, "r", linewidth=2)
    
    ax[1].scatter(Frequency[firstPeak], AUC[firstPeak])
    ax[1].scatter(Frequency[secondPeak], AUC[secondPeak])
    
    ax[1].set_ylabel("AUC", fontsize=15)
    subAx.set_ylabel("t-value", fontsize=15, color="r")
    ax[1].set_xlabel("Frequency (Hz)", fontsize=15)
    #subAx.set_ylim(-0.4, 0.4)
    subAx.set_ylim(-5, 5)
    ax[1].set_xlim(0,50)
    # One exploratory PDF per 180-day window, named by the window's end date.
    window_date = datetime.fromtimestamp(timestamps).strftime('%Y%m%d')
    fig.savefig(_exploratory_out(f"{DEVICE_SIDE}_AUC_180_day_window_{window_date}.pdf"))

# %% Correlate Pain Score vs Power 
Features = list()
Features.append(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]), :].T,axis=1))
Features.append(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]), :].T,axis=1))

fig = GU.largeFigure(100, [800,600])
ax = fig.subplots(1,1)

Label = PainScore
ax.scatter(Features[0], Label, 5, color="k")
coe = np.polyfit(Features[0], Label, 1)
fittedLine = np.polyval(coe, [0,500])

r, p = stats.pearsonr(Features[0], Label)

ax.plot([0,500], fittedLine, "r")
ax.set_xlabel("Spectral Feature (a.u.)", fontsize=15)
ax.set_ylabel("Pain Score (MPQ)", fontsize=15)
ax.set_title(f"r={r:.2f} p={p:.3f}")

ax.set_xlim(0,500)
ax.set_ylim(0,40)

fig.savefig(_exploratory_out("8HzCorrelateWithPainNRS.pdf"))

# %% LDAs Feature Seperation
import seaborn as snb

Features = list()
Features.append(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]), :].T,axis=1))
Features.append(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]), :].T,axis=1))

fig = GU.largeFigure(100, [600,400])
ax = fig.subplots(1,1)

snb.violinplot(x=PainScore > np.mean(PainScore), y=Features[0], saturation=0.8, inner="quartile", ax=ax)

#ax.set_title(f"{Frequency[firstPeak]}Hz (p={SecondPeakPValue.pvalue:.3})", fontsize=15)
ax.set_xticklabels(["Low Pain", "High Pain"])
ax.set_xlim(-1,2)
ax.set_ylim(0,600)
#ax[1].set_ylim(0,10)

fig.savefig(_exploratory_out("ViolinComparison_feature_by_pain_state.pdf"))

# %% LDAs Feature Seperation
import seaborn as snb

Features = list()
Features.append(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]), :].T,axis=1))
Features.append(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]), :].T,axis=1))

fig = GU.largeFigure(100, [1280,400])
ax = fig.subplots(1,2)

snb.violinplot(x=PainScore > np.mean(PainScore), y=Features[0], saturation=0.8, inner="quartile", ax=ax[0])
print(stats.ttest_ind(Features[0][PainScore > np.mean(PainScore)], Features[0][PainScore < np.mean(PainScore)]))
ax[0].set_title(f"{Frequency[firstPeak]}Hz (p={FirstPeakPValue.pvalue:.3})", fontsize=15)
ax[0].set_xticklabels(["Low Pain", "High Pain"])
#ax[0].set_ylim(0,800)

snb.violinplot(x=PainScore > np.mean(PainScore), y=Features[1], saturation=0.8, inner="quartile", ax=ax[1])

ax[1].set_title(f"{Frequency[secondPeak]}Hz (p={SecondPeakPValue.pvalue:.3})", fontsize=15)
ax[1].set_xticklabels(["Low Pain", "High Pain"])
#ax[1].set_ylim(0,10)

fig.savefig(_exploratory_out("ViolinComparison_feature_by_stimulation_state.pdf"))

Features = list()
Features.append(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]), :].T,axis=1))
Features.append(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]), :].T,axis=1))

fig = GU.largeFigure(100, [1280,400])
ax = fig.subplots(1,2)

GU.standardErrorBox(0, Features[0][PainScore <= np.percentile(PainScore, Threshold[0])], width=0.3, color="b", showfliers=True, flierprops=None, ax=ax[0])
GU.standardErrorBox(1, Features[0][PainScore >= np.percentile(PainScore, Threshold[1])], width=0.3, color="r", showfliers=True, flierprops=None, ax=ax[0])
ax[0].set_title(f"{Frequency[firstPeak]}Hz (p={FirstPeakPValue.pvalue:.3})", fontsize=15)
ax[0].set_xticklabels(["Low Pain", "High Pain"])
ax[0].set_ylim(0,20)

GU.standardErrorBox(0, Features[1][PainScore <= np.percentile(PainScore, Threshold[0])], width=0.3, color="b", showfliers=True, flierprops=None, ax=ax[1])
GU.standardErrorBox(1, Features[1][PainScore >= np.percentile(PainScore, Threshold[1])], width=0.3, color="r", showfliers=True, flierprops=None, ax=ax[1])
ax[1].set_title(f"{Frequency[secondPeak]}Hz (p={SecondPeakPValue.pvalue:.3})", fontsize=15)
ax[1].set_xticklabels(["Low Pain", "High Pain"])
ax[1].set_ylim(0,10)

fig.savefig(_exploratory_out("BoxplotComparison.pdf"))

fig = GU.largeFigure(101, [800,400])
ax = fig.subplots(1,1)
count, thresholdIndex = np.histogram(Features[0][PainScore <= np.percentile(PainScore, Threshold[0])], np.arange(0,20,0.2))
ax.bar(thresholdIndex[:-1]+0.1, count, width=0.1)
count, thresholdIndex = np.histogram(Features[0][PainScore >= np.percentile(PainScore, Threshold[1])], np.arange(0,20,0.2))
ax.bar(thresholdIndex[:-1], count, width=0.1)
ax.set_xlim(0,20)
ax.set_ylim(0,20)
fig.savefig(_exploratory_out("Feature1Distribution.pdf"))

fig = GU.largeFigure(102, [800,400])
ax = fig.subplots(1,1)
count, thresholdIndex = np.histogram(Features[1][PainScore <= np.percentile(PainScore, Threshold[0])], np.arange(0,10,0.1))
ax.bar(thresholdIndex[:-1]+0.05, count, width=0.05)
count, thresholdIndex = np.histogram(Features[1][PainScore >= np.percentile(PainScore, Threshold[1])], np.arange(0,10,0.1))
ax.bar(thresholdIndex[:-1], count, width=0.05)
ax.set_xlim(0,10)
ax.set_ylim(0,30)
fig.savefig(_exploratory_out("Feature2Distribution.pdf"))

# %% LDAs with 
fig = GU.largeFigure(101, [1250,600])
ax = fig.subplots(1,2)

ax[0].scatter(Features[0][PainScore < np.percentile(PainScore, Threshold[0])], Features[1][PainScore < np.percentile(PainScore, Threshold[0])], color="b")
ax[0].scatter(Features[0][PainScore > np.percentile(PainScore, Threshold[1])], Features[1][PainScore > np.percentile(PainScore, Threshold[1])], color="r")

FeatureValue = np.array([Features[0],Features[1]]).T
FeatureValue_Limit = FeatureValue[np.bitwise_or(rangeSelection(PainScore, [0,np.percentile(PainScore, Threshold[0])], type="inclusive"), rangeSelection(PainScore, [np.percentile(PainScore, Threshold[1]),200], type="inclusive")), :]
PainValue_Limit = PainScore[np.bitwise_or(rangeSelection(PainScore, [0,np.percentile(PainScore, Threshold[0])], type="inclusive"), rangeSelection(PainScore, [np.percentile(PainScore, Threshold[1]),200], type="inclusive"))]
lda = LinearDiscriminantAnalysis().fit(FeatureValue_Limit, PainValue_Limit >= np.percentile(PainValue_Limit, 50))
x1 = np.array([np.min(FeatureValue_Limit[:,0], axis=0), np.max(FeatureValue_Limit[:,0], axis=0)])
x2 = np.array([np.min(FeatureValue_Limit[:,-1], axis=0), np.max(FeatureValue_Limit[:,-1], axis=0)])

LowPainRatio = np.sum(PainValue_Limit < np.percentile(PainScore,50)) / len(PainValue_Limit < np.percentile(PainScore,50))

b, w1, w2 = lda.intercept_[0], lda.coef_[0][0], lda.coef_[0][1]
y1 = -(b+x1*w1)/w2
ax[0].plot(x1,y1,c="black")
#ax[0].set_xlim([np.percentile(Features[0], 0),np.percentile(Features[0], 100)])
#ax[0].set_ylim([np.percentile(Features[1], 0),np.percentile(Features[1], 100)])
ax[0].set_xlim(0,20)
ax[0].set_ylim(0,10)
ax[0].set_title(f"{Side} {ChannelName} {LeadLocations[Side][SensingChannel]}", fontsize=18)
ax[0].set_xlabel(f"Feature 1: {Frequency[firstPeak]:.1f}Hz", fontsize=15)
ax[0].set_ylabel(f"Feature 2: {Frequency[secondPeak]:.1f}Hz", fontsize=15)

PredictProbability = np.zeros(len(PainValue_Limit))
LDAOutput = np.zeros(len(PainValue_Limit))
for i in range(len(PainValue_Limit)):
    TrainingSelection = np.array(range(len(PainValue_Limit))) != i
    lda = LinearDiscriminantAnalysis().fit(FeatureValue_Limit[TrainingSelection,:], PainValue_Limit[TrainingSelection] >= np.mean(PainValue_Limit))
    PredictProbability[i] = lda.predict_proba(FeatureValue_Limit[i,:].reshape(1,-1))[0][1]
    LDAOutput[i] = np.sum(FeatureValue_Limit[i,:].reshape(1,-1) * lda.coef_)
    
fpr, tpr, thresholds = metrics.roc_curve(PainValue_Limit >= np.mean(PainValue_Limit) , LDAOutput)
lda = LinearDiscriminantAnalysis().fit(FeatureValue_Limit, PainValue_Limit >= np.mean(PainValue_Limit))

ax[1].plot(fpr, tpr, linewidth=2)
ax[1].plot([0,1],[0,1],"k", linewidth=2)
AUC = np.trapz(y=tpr, x=fpr)
ax[1].set_title(f"AUC: {AUC:.2} - Ratio {LowPainRatio:.2f}:{1-LowPainRatio:.2f}", fontsize=18)
ax[1].set_xlim(-0.0,1.0)
ax[1].set_xlabel("False Positive Rate", fontsize=15)
ax[1].set_ylim(-0.0,1.0)
ax[1].set_xlabel("True Positive Rate", fontsize=15)

fig.savefig(_exploratory_out("DoubleFeatureLDA.pdf"))

# %% Single Feature LDA

Features = list()
Features.append(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[firstPeak]-3, Frequency[firstPeak]+3]), :].T,axis=1))
Features.append(np.mean(PowerSpectralDensity[rangeSelection(Frequency, [Frequency[secondPeak]-3, Frequency[secondPeak]+3]), :].T,axis=1))

fig = GU.largeFigure(101, [600,600])
ax = fig.subplots(1,1)

FeatureValue = np.array([Features[0],Features[0]]).T
FeatureValue_Limit = FeatureValue[np.bitwise_or(rangeSelection(PainScore, [0,np.percentile(PainScore, Threshold[0])], type="inclusive"), rangeSelection(PainScore, [np.percentile(PainScore, Threshold[1]),200], type="inclusive")), :]
PainValue_Limit = PainScore[np.bitwise_or(rangeSelection(PainScore, [0,np.percentile(PainScore, Threshold[0])], type="inclusive"), rangeSelection(PainScore, [np.percentile(PainScore, Threshold[1]),200], type="inclusive"))]
#lda = LinearDiscriminantAnalysis().fit(FeatureValue_Limit, PainValue_Limit >= np.percentile(PainValue_Limit, 50))
lda = LinearDiscriminantAnalysis().fit(FeatureValue_Limit, PainValue_Limit >= 2)

PredictProbability = np.zeros(len(PainValue_Limit))
LDAOutput = np.zeros(len(PainValue_Limit))
for i in range(len(PainValue_Limit)):
    TrainingSelection = np.array(range(len(PainValue_Limit))) != i
    lda = LinearDiscriminantAnalysis().fit(FeatureValue_Limit[TrainingSelection,:], PainValue_Limit[TrainingSelection] >= np.mean(PainValue_Limit))
    PredictProbability[i] = lda.predict_proba(FeatureValue_Limit[i,:].reshape(1,-1))[0][1]
    LDAOutput[i] = np.sum(FeatureValue_Limit[i,:].reshape(1,-1) * lda.coef_)

true_label = PainValue_Limit >= np.mean(PainValue_Limit)
thresholds = np.flip(np.array(sorted(LDAOutput)))
fpr = np.zeros(thresholds.shape)
tpr = np.zeros(thresholds.shape)
f1 = np.zeros(thresholds.shape)
fnr = np.zeros(thresholds.shape)
for i in range(len(thresholds)):
    predict_label = LDAOutput >= thresholds[i]
    fpr[i] = np.sum(np.bitwise_and(true_label == 0, predict_label == 1)) / np.sum(true_label == 0)
    tpr[i] = np.sum(np.bitwise_and(true_label == 1, predict_label == 1)) / np.sum(true_label == 1)
    fnr[i] = np.sum(np.bitwise_and(true_label == 1, predict_label == 0)) / np.sum(true_label == 0)
    f1[i] = tpr[i]/(tpr[i] + fpr[i] * 0.5 + fnr[i] * 0.5)

#fpr, tpr, thresholds = metrics.roc_curve(PainValue_Limit >= np.mean(PainValue_Limit) , LDAOutput)
lda = LinearDiscriminantAnalysis().fit(FeatureValue_Limit, PainValue_Limit >= np.mean(PainValue_Limit))

ax.plot(fpr, tpr, linewidth=2)
ax.plot([0,1],[0,1],"k", linewidth=2)
for i in np.arange(0,100,10):
    tileThreshold = np.percentile(LDAOutput, i)
    index = np.argmin(np.abs(thresholds-tileThreshold))
    ax.scatter(fpr[index], tpr[index], 50, "r")

#tileThreshold = np.percentile(LDAOutput, 65)
#index = np.argmin(np.abs(thresholds-tileThreshold))
#ax.scatter(fpr[index], tpr[index], 80, "b")

AUCs = [np.trapz(y=tpr[:i], x=fpr[:i]) for i in range(len(tpr))]

AUC = np.trapz(y=tpr, x=fpr)
ax.set_title(f"AUC: {AUC:.2}", fontsize=18)
ax.set_xlim(-0.0,1.0)
ax.set_xlabel("False Positive Rate", fontsize=15)
ax.set_ylim(-0.0,1.0)
ax.set_ylabel("True Positive Rate", fontsize=15)

fig.savefig(_exploratory_out("Feature1LDA.pdf"))

# %% Simulate with Most Recent Data

AUCRatio = np.array([tpr[i]/fpr[i] for i in range(len(tpr))])
AUCs = np.array([np.trapz(y=tpr[:len(tpr)-1-i], x=fpr[:len(tpr)-1-i])/(fpr[len(tpr)-1-i]*fpr[len(tpr)-1-i]/2) for i in range(len(tpr))])
tileThreshold = [np.sum(LDAOutput > t)/len(LDAOutput) for t in thresholds]
#thresholds = thresholds[~np.isnan(AUCs)]
#AUCs = AUCs[~np.isnan(AUCs)]
AUCs[np.isnan(AUCs)] = 0
AUCRatio[np.isnan(AUCRatio)] = 0

fig = GU.largeFigure(101, [600,600])
ax = fig.subplots(1,1)

#ax.plot(thresholds, SPU.smooth(AUCs,10))
ax.plot(thresholds, SPU.smooth(f1,10))
#ax.set_xlim(0, 1)
ax.set_xlim(thresholds[-1], thresholds[0])

ThresholdValue = thresholds[np.argmax(SPU.smooth(f1,10))]
Percentile = (ThresholdValue - thresholds[-1]) / (thresholds[0] - thresholds[-1])

ax.set_title(f"{Percentile} Max")
print(Percentile)

fig.savefig(_exploratory_out(f"{DEVICE_SIDE}_f1_by_lda_threshold.pdf"))
