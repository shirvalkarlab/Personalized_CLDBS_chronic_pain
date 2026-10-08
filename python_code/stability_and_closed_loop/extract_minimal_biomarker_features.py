#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_minimal_biomarker_features.py
=====================================

PURPOSE
-------
Reduce a per-pain-event ambulatory RC+S cache (one 4-channel, ~30-minute
time-domain recording per event, on the order of 100 GB per participant) to the
per-survey quantities the supplementary figure and analysis scripts consume:

    * one 200-point power spectral density (PSD, 0-99.5 Hz at 0.5 Hz spacing)
      per pain event, computed on the single channel of interest, averaged
      over the OFF-stimulation / no-missing-data samples of the event window,
    * the pain scores for that event (joined from the participant's REDCap
      cluster spreadsheet), and
    * the event timestamp and a few book-keeping fields.

Every published panel in `analyze_ambulatory_biomarker_stability.py` (Figure S8 -
biomarker AUC / t-test / stability-over-time) and
`simulate_closed_loop_threshold_selection.py` (Figure S4 - closed-loop threshold
simulation) is a function of these per-event PSDs and their attached pain scores;
neither re-reads the time-domain arrays. The reduction therefore preserves the
scientific content of those analyses while shrinking each event from ~7 MB to
~0.8 kB (a 200-element float32 vector), so all nine hemispheres together come to
a few MB.

WHEN TO RUN IT
--------------
The per-survey feature files this script writes are already present as
`neural_data/ambulatory_stages123/*.h5`, and that is what the stability and
threshold-simulation scripts read by default. Run this script only to rebuild
them on a machine that holds the per-event ".dat" caches and the REDCap
spreadsheets; the caches themselves contain protected health information and are
handled under the trial's data-use terms.

SPECTRAL COMPUTATION
--------------------
Power is computed with `TimeDomain2PowerChannel` from the Medtronic Summit RC+S
loader (`Summit.py`), which is Medtronic proprietary code and is not distributed
with this deposit. The stim-ON detection and missing-data
handling match `analyze_ambulatory_biomarker_stability.py` in this folder, so the
feature files and the stability figure are computed the same way.

STEP-BY-STEP (the same per-event loop the stability analysis uses)
-------------------------------------------------
For each `cacheSession<unix>.0.dat` under <CACHE_ROOT>/<DeviceSide>/:
  1. Load the PainEpoch pickle (a dict of numpy arrays + metadata).
  2. Skip events with < 1000 samples (< 4 s) of data.
  3. Band-pass the channel of interest (0-based index 2 = first LFP channel of
     the second lead) with a 5th-order Butterworth filter, 1-100 Hz.
  4. Compute the short-time power spectrogram with the Summit method
     (2 s window, 1 s overlap, 256-pt FFT), giving Power[freq, time].
  5. Flag spectrogram time bins whose underlying samples are >1% missing.
  6. Detect stimulation-ON time bins: pick the lead carrying the stim
     artifact, take the mean power in a +/-2 Hz band around the therapy
     frequency, and threshold it (median split if the power varies, else a
     fixed threshold) - the same rule the stability analysis applies.
  7. Join the event to the REDCap cluster row whose timestamp is within 60 s
     (a +3 h offset is applied; see TIMEZONE_OFFSET_SECONDS).
  8. Over the +/-300 s window around the pain report, average Power across the
     OFF-stim, no-missing time bins and interpolate onto the fixed 0-99.5 Hz
     / 0.5 Hz grid -> the 200-point PSD for this event.
  9. Attach the pain scores for the joined cluster row.

OUTPUT
------
One HDF5 file per device-hemisphere, plus a combined index CSV:
    <OUTPUT_DIR>/<DeviceSide>_ambulatory_features.h5
containing:
    /psd                 float32 [n_events x 200]   OFF-stim mean PSD per event
    /frequency_hz        float32 [200]              0.0 .. 99.5 Hz (0.5 Hz step)
    /pain_timestamp_unix float64 [n_events]         event time (unix seconds)
    /cluster_row         int32   [n_events]         matched REDCap row (-1 = none)
    /channel_of_interest scalar attr (int)          0-based LFP channel index
  plus one dataset per pain scale (nan where unmatched):
    /pain_mayoNRS, /pain_painVAS, /pain_unpleasantVAS, /pain_MPQtotal, ...

CONFIGURATION (environment variables, with safe defaults)
---------------------------------------------------------
    CACHE_ROOT        folder holding one subfolder per device-hemisphere, each
                      containing the cacheSession*.dat pickles.
    CLUSTER_XLSX_DIR  folder holding <PatientID>_clustered.xlsx REDCap files.
    OUTPUT_DIR        where the per-survey feature files are written
                      (default: neural_data/ambulatory_stages123_rebuilt/).
    SHARED_UTILITIES        path to shared_utilities
                      (default: ../shared_utilities relative to this file).

Set CACHE_ROOT and CLUSTER_XLSX_DIR to wherever your local copies live before
running.
"""

import os
import sys
import glob
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal
import h5py

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

# Root of the per-pain-event cache. One subfolder per device-hemisphere, each
# holding that hemisphere's cacheSession*.dat pickles.
CACHE_ROOT = os.environ.get("CACHE_ROOT", "/path/to/rcs_event_caches")

# Folder with the REDCap pain-cluster spreadsheets, one <PatientID>_clustered.xlsx.
CLUSTER_XLSX_DIR = os.environ.get("CLUSTER_XLSX_DIR", "/path/to/processed/REDcap")

# Where the per-survey feature files are written. The default is a separate
# folder, so a rebuild never replaces the shipped files.
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", os.path.join(_HERE, os.pardir, os.pardir, "neural_data",
                               "ambulatory_stages123_rebuilt"))

# Location of the Summit loader (for TimeDomain2PowerChannel only).
SHARED_UTILITIES = os.environ.get("SHARED_UTILITIES", os.path.join(_HERE, os.pardir,
                                                       "shared_utilities"))

# That folder uses package-style imports ("from utility.X import ..."), so put
# its root on sys.path, then import the Summit loader. The loader (Summit.py) is
# Medtronic proprietary code and is not distributed with this deposit; place it
# at <SHARED_UTILITIES>/decoder/Summit.py to run this script.
sys.path.insert(0, os.path.abspath(SHARED_UTILITIES))
try:
    from decoder import Summit   # noqa: E402  (path set above)
except ImportError as error:
    raise ImportError(
        "This script needs the Medtronic Summit RC+S loader at "
        f"{os.path.join(os.path.abspath(SHARED_UTILITIES), 'decoder', 'Summit.py')}, "
        "which is not distributed with this deposit.") from error

# ---------------------------------------------------------------------------
# Fixed analysis constants, matching the stability-analysis script
# ---------------------------------------------------------------------------
CHANNEL_OF_INTEREST = 2          # 0-based; = first LFP channel of the 2nd lead
FREQUENCY_GRID = np.arange(200) / 2.0   # 0.0 .. 99.5 Hz at 0.5 Hz spacing
WINDOW_SECONDS = 300             # +/- seconds around the pain report to average
MIN_SAMPLES = 1000               # skip events with < 4 s of data (@250 Hz)
MIN_SPECTROGRAM_BINS = 240       # need >= this many usable bins to keep an event
PAIN_TIME_MATCH_SECONDS = 60     # join tolerance to a REDCap cluster row
TIMEZONE_OFFSET_SECONDS = 3600 * 3   # +3 h offset applied to cache timestamps

# Pain-scale columns to carry from the REDCap cluster file (those actually
# present are exported; missing ones are silently skipped).
PAIN_SCALE_COLUMNS = [
    "mayoNRS", "painVAS", "unpleasantVAS", "worstNRS", "worstVAS", "MPQtotal",
]

# Map each device-hemisphere folder name to its participant id, so we can load
# the right <PatientID>_clustered.xlsx. The hemisphere suffix (L/R) is dropped.
def participant_id_for(device_side):
    """Map a device-hemisphere folder name to its participant id by stripping a
    trailing L/R from the folder's own basename. Accepting a nested path such as
    '<participant>/<participant>L' lets the caller point at a cache whose
    hemisphere folders are grouped under a participant folder."""
    base = os.path.basename(device_side.rstrip("/"))
    if base and base[-1] in ("L", "R"):
        base = base[:-1]
    return base


# ---------------------------------------------------------------------------
# Centered moving-average smoother: a window-length centered mean, with the
# window shrinking at the two ends.
#
# This matches SignalProcessingUtility.smooth(), which the stim-ON detection
# uses, and is defined inline so this script does not depend on the `spectrum`
# package that SignalProcessingUtility imports.
# ---------------------------------------------------------------------------
def _smooth(data, window):
    data = np.asarray(data, dtype=float)
    pre = int(window / 2)
    post = window - pre
    leading = np.array([np.mean(data[:i + post]) for i in range(pre)])
    center = np.array([np.mean(data[i - pre:i + post])
                       for i in range(pre, len(data) - post + 1)])
    tailing = np.array([np.mean(data[i - pre:])
                        for i in range(len(data) - post + 1, len(data))])
    return np.concatenate((leading, center, tailing))


# ---------------------------------------------------------------------------
# Per-event feature computation, following the stability-analysis loop
# ---------------------------------------------------------------------------
def distill_one_event(pain_epoch, cluster_times_unix):
    """Return (psd_200, cluster_row) for one PainEpoch dict, or (None, None)
    if the event lacks enough OFF-stim / non-missing data to featurize.

    `cluster_times_unix` is a precomputed array of REDCap cluster timestamps
    (unix seconds, with the +3 h offset already applied) used for the join.
    """
    data = pain_epoch["Data"]
    sampling_rate = float(np.ravel(pain_epoch["SamplingRate"])[0])

    # (2) too little data -> skip
    if len(data[:, CHANNEL_OF_INTEREST]) < MIN_SAMPLES:
        return None, None

    # (3) 5th-order Butterworth band-pass, 1-100 Hz, on the channel of interest.
    # IMPORTANT: the filtered signal is written back IN PLACE into
    # data[:, CHANNEL_OF_INTEREST]. That matters below: when the stim lead turns
    # out to be the same column (StimLead == 2 == CHANNEL_OF_INTEREST), the stim
    # spectrogram is computed from the FILTERED column, not the unfiltered one.
    # The stability analysis does the same, so the two agree.
    data = np.array(data, dtype=float, copy=True)   # own copy; don't touch caller's
    b, a = signal.butter(5, np.array([1, 100]) * 2 / sampling_rate,
                         "bandpass", output="ba")
    data[:, CHANNEL_OF_INTEREST] = signal.filtfilt(
        b, a, data[:, CHANNEL_OF_INTEREST])

    # (4) short-time power spectrogram (Summit: 2 s window, 1 s overlap)
    spec = Summit.TimeDomain2PowerChannel(
        data[:, CHANNEL_OF_INTEREST].flatten(), time=pain_epoch["Time"],
        interval=500, gain=250, nFFT=256, fs=sampling_rate)

    # (5) mark spectrogram bins that overlap >1% missing raw samples
    missing_flag = np.zeros(spec["Time"].shape)
    for t in range(len(missing_flag)):
        in_bin = np.bitwise_and(pain_epoch["Time"] > spec["Time"][t] - 2.5,
                                pain_epoch["Time"] <= spec["Time"][t] + 2.5)
        if np.sum(in_bin) > 0:
            missing_flag[t] = (np.sum(pain_epoch["Missing"][in_bin])
                               / np.sum(in_bin)) > 0.01

    # (6) stimulation-ON detection. Requires a known therapy frequency.
    therapy = pain_epoch["Therapy"]
    if "StimFrequency" not in therapy:
        return None, None
    therapy_frequency = therapy["StimFrequency"]
    stim_contacts = [int(c) for c in therapy["StimContact"]
                     .replace("+", " ").replace("-", " ").split(" ")
                     if c.isdigit()]
    # Pick the lead that carries the stim artifact (contacts >7 => second lead).
    stim_lead = 2 if np.any(np.array(stim_contacts) > 7) else 0

    stim_spec = Summit.TimeDomain2PowerChannel(
        data[:, stim_lead].astype(float).flatten(), time=pain_epoch["Time"],
        interval=500, gain=250, nFFT=256, fs=sampling_rate)
    # Mean power in a +/-2 Hz band around the therapy frequency, per time bin.
    freq_sel = np.bitwise_and(stim_spec["Frequency"] >= therapy_frequency - 2,
                              stim_spec["Frequency"] <= therapy_frequency + 2)
    stim_power = np.mean(stim_spec["Power"][freq_sel, :], axis=0)
    # If stim power varies, ON = above the median; otherwise fall back to a fixed
    # threshold (whole recording is either ON or OFF). Smooth to fill gaps.
    if np.std(stim_power) > 0.001:
        stim_on = stim_power > np.median(stim_power)
    else:
        stim_on = stim_power > 10
    stim_on = _smooth(stim_on.astype(float), 5) > 0

    # (7) join to a REDCap cluster row within +/-60 s. This takes the FIRST row
    # (in iteration order) whose timestamp is within 60 s, i.e. first-match rather
    # than nearest-match, which is what the stability analysis also does.
    event_unix = pain_epoch["PainTimestamp"].timestamp()
    cluster_row = -1
    deltas = np.abs(cluster_times_unix - event_unix)
    hits = np.where(deltas < PAIN_TIME_MATCH_SECONDS)[0]
    if len(hits):
        cluster_row = int(hits[0])

    # (8) average Power over the +/-300 s window, OFF-stim, no-missing bins.
    # The gate is on the in-window bin count (>240 = "at least 2 min of data
    # available"); the average is then taken over the further-restricted OFF-stim /
    # no-missing subset. Keep that ordering: gate on `in_window`, average over
    # `keep`. The stability analysis gates the same way.
    in_window = ((spec["Time"] < pain_epoch["RelativeTimestamp"] + WINDOW_SECONDS)
                 & (spec["Time"] > pain_epoch["RelativeTimestamp"] - WINDOW_SECONDS))
    if np.sum(in_window) <= MIN_SPECTROGRAM_BINS:
        return None, None
    keep = in_window & ~stim_on & (missing_flag == 0)
    if np.sum(keep) == 0:
        # No usable bins survive the OFF-stim / no-missing restriction, which
        # would make the mean all-NaN. Drop such events rather than emit a NaN
        # PSD row.
        return None, None
    power = np.mean(spec["Power"][:, keep], axis=1)
    psd = np.interp(FREQUENCY_GRID, spec["Frequency"], power)
    return psd.astype(np.float32), cluster_row


# ---------------------------------------------------------------------------
# Per-hemisphere driver
# ---------------------------------------------------------------------------
def extract_hemisphere(device_side):
    """Featurize every cache event under CACHE_ROOT/<device_side>/ and write one
    HDF5 file with PSDs + attached pain scores."""
    side_dir = os.path.join(CACHE_ROOT, device_side)
    cache_files = sorted(glob.glob(os.path.join(side_dir, "cacheSession*.dat")))
    if not cache_files:
        print(f"[{device_side}] no cacheSession*.dat found under {side_dir}")
        return None

    # Load the REDCap cluster spreadsheet for this participant (pain labels).
    participant = participant_id_for(device_side)
    cluster_path = os.path.join(CLUSTER_XLSX_DIR, f"{participant}_clustered.xlsx")
    cluster = pd.read_excel(cluster_path)

    # Precompute the cluster timestamps in unix seconds (with the +3 h offset).
    cluster_times_unix = []
    for t in cluster["time"]:
        if isinstance(t, pd._libs.tslibs.nattype.NaTType):
            cluster_times_unix.append(np.nan)
        elif isinstance(t, pd._libs.tslibs.timestamps.Timestamp):
            cluster_times_unix.append(
                datetime.fromisoformat(t.isoformat()).timestamp()
                + TIMEZONE_OFFSET_SECONDS)
        else:
            cluster_times_unix.append(
                datetime.fromisoformat(t).timestamp() + TIMEZONE_OFFSET_SECONDS)
    cluster_times_unix = np.array(cluster_times_unix, dtype=float)

    psd_rows, ts_rows, row_rows = [], [], []
    n_total = len(cache_files)
    for i, path in enumerate(cache_files):
        if i % 200 == 0:
            print(f"[{device_side}] {i}/{n_total}")
        with open(path, "rb") as fh:
            pain_epoch = pickle.load(fh)
        psd, cluster_row = distill_one_event(pain_epoch, cluster_times_unix)
        if psd is None:
            continue
        psd_rows.append(psd)
        ts_rows.append(pain_epoch["PainTimestamp"].timestamp())
        row_rows.append(cluster_row)

    if not psd_rows:
        print(f"[{device_side}] no usable events after distillation")
        return None

    psd_matrix = np.vstack(psd_rows).astype(np.float32)          # [n x 200]
    timestamps = np.array(ts_rows, dtype=np.float64)             # [n]
    cluster_rows = np.array(row_rows, dtype=np.int32)            # [n]

    # Gather pain scores for each matched cluster row (nan where unmatched).
    pain_scores = {}
    for col in PAIN_SCALE_COLUMNS:
        if col not in cluster.columns:
            continue
        values = np.full(len(cluster_rows), np.nan, dtype=np.float64)
        for k, r in enumerate(cluster_rows):
            if r >= 0:
                values[k] = float(cluster[col].iloc[r]) if pd.notna(
                    cluster[col].iloc[r]) else np.nan
        pain_scores[col] = values

    # Write the HDF5. Use the folder's own basename so a nested device-side such
    # as '<participant>/<participant>L' still writes a flat
    # '<participant>L_ambulatory_features.h5' rather than a nested subfolder.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    side_name = os.path.basename(device_side.rstrip("/"))
    out_path = os.path.join(OUTPUT_DIR, f"{side_name}_ambulatory_features.h5")
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("psd", data=psd_matrix, compression="gzip")
        h5.create_dataset("frequency_hz", data=FREQUENCY_GRID.astype(np.float32))
        h5.create_dataset("pain_timestamp_unix", data=timestamps)
        h5.create_dataset("cluster_row", data=cluster_rows)
        for col, values in pain_scores.items():
            h5.create_dataset(f"pain_{col}", data=values)
        h5.attrs["device_side"] = device_side
        h5.attrs["participant"] = participant
        h5.attrs["channel_of_interest"] = CHANNEL_OF_INTEREST
        h5.attrs["window_seconds"] = WINDOW_SECONDS
        h5.attrs["n_events"] = psd_matrix.shape[0]
        h5.attrs["n_cache_files"] = n_total
        h5.attrs["description"] = (
            "Per-survey ambulatory biomarker features: off-stimulation mean "
            "power spectral density (0-99.5 Hz at 0.5 Hz) per pain event, "
            "with the pain scores for that survey attached. Spectral method: "
            "Summit.TimeDomain2PowerChannel.")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[{device_side}] wrote {psd_matrix.shape[0]} events "
          f"({n_total} cache files) -> {out_path} ({size_mb:.2f} MB)")
    return out_path


def main(device_sides=None):
    """Run the extraction for the given device-hemispheres (default: every
    subfolder of CACHE_ROOT). Also writes an index CSV."""
    if device_sides is None:
        device_sides = sorted(
            d for d in os.listdir(CACHE_ROOT)
            if os.path.isdir(os.path.join(CACHE_ROOT, d)))
    written = []
    for device_side in device_sides:
        out = extract_hemisphere(device_side)
        if out:
            written.append(out)

    # Index of what was produced.
    if written:
        rows = []
        for path in written:
            with h5py.File(path, "r") as h5:
                rows.append({
                    "device_side": h5.attrs["device_side"],
                    "participant": h5.attrs["participant"],
                    "n_events": int(h5.attrs["n_events"]),
                    "n_cache_files": int(h5.attrs["n_cache_files"]),
                    "file": os.path.basename(path),
                    "size_mb": round(os.path.getsize(path) / 1e6, 3),
                })
        index_path = os.path.join(OUTPUT_DIR, "ambulatory_feature_index.csv")
        pd.DataFrame(rows).to_csv(index_path, index=False)
        print(f"\nWrote index -> {index_path}")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    # Optional: pass specific device-hemispheres as command-line args, e.g.
    #   python extract_minimal_biomarker_features.py P3L P3R
    args = sys.argv[1:]
    main(args if args else None)
