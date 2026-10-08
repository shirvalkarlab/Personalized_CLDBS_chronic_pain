#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_closed_loop_threshold_selection.py
================================================================================
Supplementary Figure S4 — Closed-loop algorithm development and threshold
                          selection simulation.

Code for:
  "Personalized closed-loop deep brain stimulation for chronic pain: a randomized,
   double-blind crossover pilot trial" (Shirvalkar et al., 2026, Nature Medicine).

WHAT THIS SCRIPT DOES
--------------------------------------------------------------------------------
For one implanted device side, it:
  1. Loads the on-device Power / Detector / AdaptiveStim streams that the
     Medtronic Summit RC+S recorded during ambulatory closed-loop operation.
  2. Extracts the single power-band feature that the closed-loop detector used
     (selected by sensing channel and frequency band).
  3. Removes statistical outliers (samples with |z-score| > 3).
  4. Plots the feature time course and its amplitude histogram.
  5. Simulates what the closed-loop controller would do for a candidate
     threshold: it walks through the feature, applies onset/offset hysteresis
     (a stimulation turn-on requires the feature to stay above threshold for a
     minimum number of samples, and turn-off likewise), and reports the
     resulting stimulation duty cycle. This is the calculation used to choose a
     patient-specific threshold before enabling closed-loop therapy.

INPUTS (edit the CONFIGURATION block below)
--------------------------------------------------------------------------------
  DEVICE_SIDE
      Which participant hemisphere to analyse; it selects the per-hemisphere
      feature file under neural_data/ambulatory_stages123/. FEATURE_CHANNEL and
      FEATURE_BAND are set for P3L, the published example; another side needs
      its own values (Extended Data Table 1).
  RCS_DATA_ROOT / DEVICE
      Location of the Summit continuous bilateral streaming session folders for
      one device side. Each session folder contains a DeviceNPC* subfolder that
      Summit.LoadData() reads.
  FEATURE_CHANNEL / FEATURE_BAND
      Which sensing channel (0-3) and frequency band [low, high] Hz define the
      closed-loop feature.
  DATE_WINDOW_START / DATE_WINDOW_END
      Raw recordings only: restrict to sessions recorded in this window
      (None keeps every session).

OUTPUTS
--------------------------------------------------------------------------------
  FigureS4_<DEVICE_SIDE>_feature_signal_and_histogram.pdf  feature time course
                                                            + histogram
  FigureS4_<DEVICE_SIDE>_closed_loop_threshold.pdf         simulated control signal
                                                            + stimulation state, titled
                                                            with the resulting duty cycle
  The other panels of the Figure S4 legend are not produced by this script.
The figures are written to OUTPUT_DIR.

DEPENDENCIES
--------------------------------------------------------------------------------
  The signal-processing and graphing modules under ../shared_utilities:
  SignalProcessingUtility, GraphingUtility and PythonUtility. The raw-recording
  path (USE_MINIMAL_EXTRACT=0) also needs the Medtronic Summit RC+S loader,
  Summit.py, which is Medtronic proprietary code and is not distributed here.
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
import json
import math
import pickle
from datetime import datetime
from pytz import timezone
from matplotlib import ticker, colors, dates
from scipy import stats, signal, io, linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from statsmodels.stats import multitest
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# The analysis utility modules are imported as bare module names (e.g.
# `import SignalProcessingUtility`), so put their folders on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED_UTILITIES = os.path.join(_HERE, os.pardir, "shared_utilities")
sys.path.insert(0, os.path.join(_SHARED_UTILITIES, "utility"))
sys.path.insert(0, os.path.join(_SHARED_UTILITIES, "decoder"))
sys.path.insert(0, _SHARED_UTILITIES)   # repo root, for `from utility.X import ...`

import SignalProcessingUtility as SPU   # smoothing, spectral helpers
import GraphingUtility as GU            # figure sizing helper (largeFigure)
from PythonUtility import *             # misc list/array helpers used below


# ----------------------------------------------------------------------------
#  FEATURE-FILE FAST PATH
# ----------------------------------------------------------------------------
# The dense-stream form of this simulation walks the on-device Power / Detector
# stream (many samples per session) through the controller's onset/offset
# hysteresis, which requires the full Summit streaming sessions.
# neural_data/ambulatory_stages123/ holds one de-identified HDF5 file per device
# side, giving for every pain survey the OFF-stimulation power spectral density
# in the sensing channel the detector used. Averaging each survey's PSD over the
# detector's feature band gives one feature-power value per survey -- the same
# quantity the on-device detector thresholds. Feeding that per-survey feature
# series into the same histogram + threshold + duty-cycle simulation below
# gives a per-survey approximation of the Figure S4 threshold-selection analysis.
#
# Set USE_MINIMAL_EXTRACT = True (the default) to use the per-survey feature file
# and skip the session load. Set it False, with the streaming sessions in place,
# to run the dense-stream simulation.
USE_MINIMAL_EXTRACT = os.environ.get("USE_MINIMAL_EXTRACT", "1") == "1"
if not USE_MINIMAL_EXTRACT:
    # The Medtronic Summit RC+S loader (Summit.py) is Medtronic proprietary code
    # and is not redistributed with this deposit. The raw-recording path needs it
    # importable as `Summit`; the default per-survey path above does not.
    try:
        import Summit                   # Medtronic Summit RC+S data loader (raw recordings only)
    except ImportError as error:
        raise ImportError(
            "USE_MINIMAL_EXTRACT=0 needs the Medtronic Summit RC+S loader "
            "(Summit.py), which is not distributed with this deposit. Place it at "
            "python_code/shared_utilities/decoder/Summit.py, or run with "
            "USE_MINIMAL_EXTRACT=1 (the default).") from error

def load_feature_band_power_from_extract(extract_path, feature_band):
    """Return (feature_power, unix_timestamps) for the closed-loop detector band.

    For each pain survey in the HDF5 feature file, average that survey's
    OFF-stim PSD over feature_band = [low_Hz, high_Hz] to obtain the single
    power value the on-device detector would have thresholded, plus the survey's
    timestamp. This is the per-survey analogue of the dense on-device Power
    stream the streaming-session form of the simulation consumes.
    """
    import h5py
    with h5py.File(extract_path, "r") as handle:
        frequency = handle["frequency_hz"][:].astype(float)
        psd = handle["psd"][:].astype(float)                      # (n_events, n_freq)
        unix_timestamps = handle["pain_timestamp_unix"][:].astype(float)
    band = (frequency >= feature_band[0]) & (frequency <= feature_band[1])
    feature_power = np.nanmean(psd[:, band], axis=1)
    finite = np.isfinite(feature_power)
    order = np.argsort(unix_timestamps[finite])
    return feature_power[finite][order], unix_timestamps[finite][order]



# ============================================================================
#  CONFIGURATION  — edit these for the device side you want to analyse.
# ============================================================================
# Root folder holding the RC+S streaming sessions for this participant.
# Point it at your local copy.
RCS_DATA_ROOT = os.environ.get(
    "RCS_DATA_ROOT",
    "/path/to/SummitContinuousBilateralStreaming",  # <-- set this
)
# Which participant hemisphere to analyze. With the default fast path this is
# the only setting that matters: it selects the per-survey feature file
# {DEVICE_SIDE}_ambulatory_features.h5. Give a P-label + hemisphere, e.g.
# "P3L", "P2R". Override without editing the file:
#     DEVICE_SIDE=P2L python simulate_closed_loop_threshold_selection.py
DEVICE_SIDE = os.environ.get("DEVICE_SIDE", "P3L")  # implanted hemisphere (P#L / P#R)
DEVICE = DEVICE_SIDE[:-1]  # participant label (streaming-session mode only)

# Which sensing channel (0-3) and power band [low_Hz, high_Hz] define the
# closed-loop feature. This must match the on-device detector configuration.
FEATURE_CHANNEL = 2       # 0-3
FEATURE_BAND = [9, 12]    # Hz

# Raw recordings only: include sessions recorded within this window. Set both to
# datetime values for your own recordings; None keeps every session.
DATE_WINDOW_START = None
DATE_WINDOW_END = None

# Figures are written to the repository's figures/ directory under their
# published names, so a rerun replaces the shipped file rather than leaving
# differently-named copies beside the script.
_REPO = os.path.dirname(os.path.dirname(_HERE))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(_REPO, "figures"))


######################################################################
############################ Helper function #########################
######################################################################
def removeOutliers(array):
    """Iteratively drop samples whose z-score exceeds 3 (in absolute value).

    Returns the cleaned array and the indices (into the original array) of the
    samples that were kept, so a parallel series (e.g. timestamps, stimulation
    amplitude) can be subset the same way.
    """
    newIndex = np.arange(len(array))
    newArray = copy.deepcopy(array)
    while True:
        zscore = stats.zscore(newArray)
        if np.any(np.abs(zscore) > 3):
            newIndex = newIndex[np.abs(zscore) <= 3]
            newArray = newArray[np.abs(zscore) <= 3]
            pass
        else:
            break
    return newArray, newIndex


if not USE_MINIMAL_EXTRACT:
    # STREAMING mode: load every qualifying session's dense Power/Detector stream.
    RCS_Path = os.path.join(RCS_DATA_ROOT, "")
    Side = f"/{DEVICE_SIDE}/"
    existingFolders = os.listdir(RCS_Path + Side)


    # ============================================================================
    #  LOAD ALL DATA  — gather the closed-loop feature, stimulation amplitude, and
    #  adaptive state across every qualifying session in the date window.
    # ============================================================================
    DataFolders = []
    for folder in existingFolders:
        try:
            # Session folder names are "Session<unix_ms>"; keep those in the window.
            FolderTime = datetime.fromtimestamp(int(folder.replace("Session", "")) / 1000)
            if ((DATE_WINDOW_START is None or FolderTime > DATE_WINDOW_START)
                    and (DATE_WINDOW_END is None or FolderTime < DATE_WINDOW_END)):
                DataFolders.append(Side + folder + "/")
        except Exception as e:
            continue

    AllPowers = []             # closed-loop feature (power band) samples
    UnixTimestamps = []        # matching sample timestamps
    StimulationAmplitude = []  # delivered adaptive stimulation amplitude
    AdaptiveStates = []        # detector state index at each sample

    for DataFolder in DataFolders:
        DeviceFolder = [f for f in os.listdir(RCS_Path + DataFolder) if f.startswith("DeviceNPC")]
        try:
            Data = Summit.LoadData(
                RCS_Path + DataFolder + DeviceFolder[0] + os.path.sep,
                ["Power", "Detector", "AdaptiveStim"],
            )
        except Exception as e:
            print(e)
            continue

        # Copy the anatomical lead location from the session config (paired montages).
        for Configuration in Data["Config"]:
            if "LeadLocation" in Configuration.keys():
                Data["LeadLocation"] = Configuration["LeadLocation"]
                Data["LeadLocation"][1] = Data["LeadLocation"][0]
                Data["LeadLocation"][3] = Data["LeadLocation"][2]

        if "Power" not in Data.keys():
            print("No Power Band Data")
            continue

        # Session start time (device clock, minus a 3 h offset).
        TimeStruct = datetime.fromtimestamp(Data["Power"][0]["UnixTime"][0] - 3 * 3600)
        print(TimeStruct.strftime("%Y %B %d - %H:%M:%S"))

        for nStream in range(len(Data["Power"])):
            if "Power" not in Data["Power"][nStream].keys():
                # Missing data in this stream
                continue

            # Find which of the 8 power-band slots matches the requested channel+band.
            # The 4 sensing channels map to slot pairs (0-1, 2-3, 4-5, 6-7).
            TargetChannel = None
            slot_pairs = {0: (0, 2), 1: (2, 4), 2: (4, 6), 3: (6, 8)}
            lo, hi = slot_pairs[FEATURE_CHANNEL]
            for i in range(lo, hi):
                if (np.array(Data["Config"][0]["SensingConfiguration"]["Power"][i]) == FEATURE_BAND).all():
                    TargetChannel = i

            if not TargetChannel:
                print("Power Band asked for is not available in this recording.")
                continue

            Configuration = Data["Power"][nStream]["Configuration"]
            SamplingRate = 1000 / Data["Power"][nStream]["Configuration"]["Interval"]

            # Power==0 marks a dropped/invalid sample; keep only accepted samples.
            Accepted = Data["Power"][nStream]["Power"][:, 0] != 0
            PowerBand = Data["Power"][nStream]["Power"][Accepted, :][:, TargetChannel]
            UnixTimestamps.extend(Data["Power"][nStream]["UnixTime"][Accepted])
            AllPowers.extend(PowerBand)

            # Align adaptive stimulation amplitude + state onto the power timestamps.
            if "AdaptiveStim" in Data.keys():
                Amp = np.interp(
                    Data["Power"][nStream]["UnixTime"][Accepted],
                    Data["AdaptiveStim"][nStream]["UnixTime"],
                    Data["AdaptiveStim"][nStream]["Amplitude"][:, 0],
                )
                StimulationAmplitude.extend(Amp)
                Amp = np.interp(
                    Data["Power"][nStream]["UnixTime"][Accepted],
                    Data["AdaptiveStim"][nStream]["UnixTime"],
                    Data["AdaptiveStim"][nStream]["CurrentAdaptiveState"],
                )
                AdaptiveStates.extend(Amp)
            else:
                StimulationAmplitude.extend(np.ones(PowerBand.shape))
                AdaptiveStates.extend(np.zeros(PowerBand.shape))

    AllChannelPowers = np.array(AllPowers)
else:
    # FAST PATH: build the feature-power series from the per-survey feature file.
    MINIMAL_EXTRACT_PATH = os.environ.get(
        'MINIMAL_EXTRACT_PATH',
        os.path.join(_HERE, os.pardir, os.pardir, 'neural_data',
                     'ambulatory_stages123',
                     f'{DEVICE_SIDE}_ambulatory_features.h5'))
    if not os.path.exists(MINIMAL_EXTRACT_PATH):
        raise FileNotFoundError(
            f'Minimal extract not found: {MINIMAL_EXTRACT_PATH}. Build it with '
            'extract_minimal_biomarker_features.py, or set USE_MINIMAL_EXTRACT=0 '
            'to run from the raw sessions.')
    _feature_power, _feature_times = load_feature_band_power_from_extract(
        MINIMAL_EXTRACT_PATH, FEATURE_BAND)
    AllPowers = list(_feature_power)
    UnixTimestamps = list(_feature_times)
    # One feature value per survey, with no per-sample stim amplitude, so use a
    # flat placeholder (the amplitude overlay is illustrative only).
    StimulationAmplitude = list(np.ones(len(_feature_power)))
    AdaptiveStates = list(np.zeros(len(_feature_power)))
    SamplingRate = 1.0   # one feature sample per survey
    AllChannelPowers = np.array(AllPowers)
    print(f'Loaded {len(AllPowers)} per-survey feature values from '
          f'{os.path.basename(MINIMAL_EXTRACT_PATH)} (band {FEATURE_BAND} Hz)')



# ============================================================================
#  VIEW FEATURE POWER  — feature time course (with delivered stim) + histogram.
# ============================================================================
# Display smoothing window. The streamed Power signal is dense (many samples per
# session) so a 30-sample window is appropriate; the feature file carries one
# value per survey, already at detector scale, so no smoothing.
AverageDuration = 1 if USE_MINIMAL_EXTRACT else 30
AllPowers = AllChannelPowers * 1   # sign coefficient: use -1 to invert feature
AllPowers, newIndex = removeOutliers(AllPowers)

fig = GU.largeFigure(1, [800, 400])
ax = fig.subplots(2, 1, sharex=False)

PowerDates = np.arange(len(AllPowers)) / SamplingRate
ax[0].plot(PowerDates, SPU.smooth(AllPowers, AverageDuration))
ax[0].set_xlabel("Survey index" if USE_MINIMAL_EXTRACT else "Time (s)", fontsize=13)
ax[0].set_ylabel("Power", fontsize=13)
ax[0].set_xlim(PowerDates[0], PowerDates[-1])
ax[0].set_ylim(np.percentile(AllPowers, 5) - 0.5 * np.std(AllPowers),
               np.percentile(AllPowers, 95) + 0.5 * np.std(AllPowers))

# Overlay the delivered stimulation amplitude on a second y-axis.
AmplitudeDates = np.arange(len(StimulationAmplitude))[newIndex] / SamplingRate
twinx = ax[0].twinx()
twinx.plot(AmplitudeDates, np.array(StimulationAmplitude)[newIndex], "r")
twinx.set_ylabel("Stimulation", fontsize=15)

# Histogram of the (smoothed) feature amplitude.
maxRange = np.round(np.percentile(AllPowers, 95) + 0.5 * np.std(AllPowers))
width = int(maxRange / 10000) if maxRange > 100000 else 10
bins = np.arange(0, maxRange, width)

FeaturePower = 1 * np.array(AllPowers)
FeaturePower = SPU.smooth(FeaturePower, AverageDuration)
count = np.histogram(FeaturePower, bins=bins)[0]
ax[1].bar(bins[:-1] + width / 2, count / np.sum(count) * 100, width=width / 2)
ax[1].set_ylabel("Percent (%)", fontsize=13)
fig.savefig(os.path.join(OUTPUT_DIR,
            f"FigureS4_{DEVICE_SIDE}_feature_signal_and_histogram.pdf"),
            transparent=False, format="pdf")

# Standalone larger histogram of the same data.
fig = GU.largeFigure(101, [600, 600])
ax = fig.subplots(1, 1)
ax.bar(bins[:-1] + width / 2, count / np.sum(count) * 100, width=width / 2, color="r")
ax.set_xlim(np.percentile(FeaturePower, 0), np.percentile(FeaturePower, 100))
ax.set_ylabel("Percent (%)", fontsize=13)


# ============================================================================
#  SIMULATION  — apply a candidate closed-loop threshold with onset/offset
#  hysteresis and measure the resulting stimulation duty cycle.
# ============================================================================
# Simulation smoothing/decimation window. Dense streamed signal -> 30 samples;
# the per-survey feature file is already one sample per detector decision -> 1.
AverageDuration = 1 if USE_MINIMAL_EXTRACT else 30
ThresholdPercent = 0.57    # threshold as a fraction of the feature's range

Indexes = np.arange(len(AllChannelPowers))
FeaturePower = SPU.smooth(AllChannelPowers[Indexes] * 1, AverageDuration)  # scaling coeff

# Controller timing parameters (in decimated samples):
OnsetDuration = 2          # consecutive above-threshold samples required to turn ON
OffsetDuration = 0         # consecutive below-threshold samples required to turn OFF
ResponsiveDuration = 0     # minimum ON dwell before an OFF is allowed

FeaturePower, Indexes = removeOutliers(FeaturePower)
SimulatedTimestamp = np.array(UnixTimestamps)[Indexes]
SimulatedTimestamp = SimulatedTimestamp[::AverageDuration]

# Decimate to one value per averaging window; this is the detector's "LDA" output.
LDAValues = np.round(FeaturePower)[::AverageDuration] * 1
# Absolute threshold = ThresholdPercent of the way from min to max feature value.
LDAThreshold = ThresholdPercent * (np.percentile(LDAValues, 100) - np.percentile(LDAValues, 0)) \
               + np.percentile(LDAValues, 0)
StimulationState = np.zeros(len(LDAValues))

# Walk the detector output and emulate the on-device state machine:
#   Counter > OnsetDuration (sustained above threshold) -> turn stimulation ON
#   Counter < -OffsetDuration (sustained below threshold) -> turn OFF,
#   but only after ResponsiveDuration samples of ON dwell.
Counter = 0
ResponsiveCounter = 0
CurrentlyStimulating = False
for i in range(0, len(LDAValues), AverageDuration):
    if LDAValues[i] >= LDAThreshold and LDAValues[i] < 999999999:
        Counter = Counter + 1 if Counter >= 0 else 1
    else:
        Counter = -1 if Counter >= 0 else Counter - 1

    if ResponsiveCounter > 0:
        ResponsiveCounter += 1

    if not CurrentlyStimulating and Counter > OnsetDuration:
        StimulationState[i:] = 1
        CurrentlyStimulating = True
        ResponsiveCounter += 1

    if CurrentlyStimulating and ResponsiveCounter > ResponsiveDuration and Counter < -OffsetDuration:
        StimulationState[i:] = 0
        CurrentlyStimulating = False
        ResponsiveCounter = 0

# Plot the detector output and the resulting simulated stimulation state; the
# title reports the chosen threshold and the achieved duty cycle.
fig = GU.largeFigure(7, [1280, 500])
ax = fig.subplots(2, 1, sharex=True)
# One sample per survey on the default path; the raw stream keeps its original scale (index / 2).
sample_axis = np.arange(len(LDAValues)) if USE_MINIMAL_EXTRACT else np.arange(len(LDAValues)) / 2
ax[0].plot(sample_axis, LDAValues)
ax[1].plot(sample_axis, StimulationState)
ax[1].set_xlabel("Survey index" if USE_MINIMAL_EXTRACT else "Time (s)", fontsize=15)
ax[1].set_ylabel("Stimulation State", fontsize=15)
ax[0].set_ylabel("LDA Output", fontsize=15)
ax[0].set_title(f"Averaging:{AverageDuration} | Onset:{OnsetDuration+1} | "
                f"Termination: {OffsetDuration+1}", fontsize=18)
ax[1].set_title(f"Threshold: {np.round(LDAThreshold)} | "
                f"Duty Cycle: {np.sum(StimulationState)/len(StimulationState)*100:.2f}%", fontsize=18)
ax[0].set_xlim([sample_axis[0], sample_axis[-1]])
fig.savefig(os.path.join(OUTPUT_DIR,
            f"FigureS4_{DEVICE_SIDE}_closed_loop_threshold.pdf"),
            transparent=False, format="pdf")
