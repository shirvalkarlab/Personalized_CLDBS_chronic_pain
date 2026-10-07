"""Parallel driver for the per-survey biomarker feature extraction.

Reuses extract_minimal_biomarker_features.py (its picklable distill_one_event,
its spectral constants, and its HDF5 layout) but replaces the serial
per-cache-file loop with a multiprocessing Pool, so the pickle-load and
spectral computation of each pain event runs across many cores. Hemispheres are
independent and are processed one after another, each fully parallel across its
own file list.

Usage:
    python extract_minimal_biomarker_features_parallel.py <N_WORKERS> \
            <device_side> [<device_side> ...]
"""
import os, sys, glob, pickle
from datetime import datetime
from multiprocessing import Pool
import numpy as np, pandas as pd, h5py

# Import the serial extractor, which lives in the same folder as this driver, so
# its per-event logic is reused unchanged.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_minimal_biomarker_features as E   # noqa: E402

# Worker: load one cache file and featurize it. Returns (psd, cluster_row, ts),
# or None when the event is unusable. cluster_times_unix is passed once per
# hemisphere through the pool initializer, so the array is not re-pickled per task.
_CLUSTER_TIMES = None
def _init(cluster_times_unix):
    global _CLUSTER_TIMES
    _CLUSTER_TIMES = cluster_times_unix

def _distill_file(path):
    try:
        with open(path, "rb") as fh:
            pain_epoch = pickle.load(fh)
        psd, cluster_row = E.distill_one_event(pain_epoch, _CLUSTER_TIMES)
        if psd is None:
            return None
        return (psd, int(cluster_row), float(pain_epoch["PainTimestamp"].timestamp()))
    except Exception as exc:
        return ("ERROR", str(exc), path)

def extract_hemisphere_parallel(device_side, n_workers):
    side_dir = os.path.join(E.CACHE_ROOT, device_side)
    cache_files = sorted(glob.glob(os.path.join(side_dir, "cacheSession*.dat")))
    if not cache_files:
        print(f"[{device_side}] no cacheSession*.dat found under {side_dir}", flush=True)
        return None
    n_total = len(cache_files)

    # Load the cluster spreadsheet and precompute its timestamps, as the serial
    # extractor does.
    participant = E.participant_id_for(device_side)
    cluster = pd.read_excel(os.path.join(E.CLUSTER_XLSX_DIR, f"{participant}_clustered.xlsx"))
    cluster_times_unix = []
    for t in cluster["time"]:
        if isinstance(t, pd._libs.tslibs.nattype.NaTType):
            cluster_times_unix.append(np.nan)
        elif isinstance(t, pd._libs.tslibs.timestamps.Timestamp):
            cluster_times_unix.append(datetime.fromisoformat(t.isoformat()).timestamp() + E.TIMEZONE_OFFSET_SECONDS)
        else:
            cluster_times_unix.append(datetime.fromisoformat(t).timestamp() + E.TIMEZONE_OFFSET_SECONDS)
    cluster_times_unix = np.array(cluster_times_unix, dtype=float)

    print(f"[{device_side}] distilling {n_total} cache files on {n_workers} workers", flush=True)
    psd_rows, ts_rows, row_rows = [], [], []
    done = 0
    with Pool(n_workers, initializer=_init, initargs=(cluster_times_unix,)) as pool:
        # chunksize keeps per-task IPC overhead low; imap preserves streaming progress.
        for res in pool.imap(_distill_file, cache_files, chunksize=8):
            done += 1
            if done % 200 == 0:
                print(f"[{device_side}] {done}/{n_total}", flush=True)
            if res is None:
                continue
            if isinstance(res, tuple) and len(res) == 3 and isinstance(res[0], str) and res[0] == "ERROR":
                # worker caught an exception on this file; skip it
                continue
            psd, cluster_row, ts = res
            psd_rows.append(psd); row_rows.append(cluster_row); ts_rows.append(ts)

    if not psd_rows:
        print(f"[{device_side}] no usable events after distillation", flush=True)
        return None

    psd_matrix = np.vstack(psd_rows).astype(np.float32)
    timestamps = np.array(ts_rows, dtype=np.float64)
    cluster_rows = np.array(row_rows, dtype=np.int32)

    # Gather pain scores per matched row, as the serial extractor does.
    pain_scores = {}
    for col in E.PAIN_SCALE_COLUMNS:
        if col not in cluster.columns:
            continue
        values = np.full(len(cluster_rows), np.nan, dtype=np.float64)
        for k, r in enumerate(cluster_rows):
            if r >= 0:
                v = cluster[col].iloc[r]
                values[k] = float(v) if pd.notna(v) else np.nan
        pain_scores[col] = values

    os.makedirs(E.OUTPUT_DIR, exist_ok=True)
    side_name = os.path.basename(device_side.rstrip("/"))
    out_path = os.path.join(E.OUTPUT_DIR, f"{side_name}_ambulatory_features.h5")
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("psd", data=psd_matrix, compression="gzip")
        h5.create_dataset("frequency_hz", data=E.FREQUENCY_GRID.astype(np.float32))
        h5.create_dataset("pain_timestamp_unix", data=timestamps)
        h5.create_dataset("cluster_row", data=cluster_rows)
        for col, values in pain_scores.items():
            h5.create_dataset(f"pain_{col}", data=values)
        h5.attrs["device_side"] = device_side
        h5.attrs["participant"] = participant
        h5.attrs["channel_of_interest"] = E.CHANNEL_OF_INTEREST
        h5.attrs["window_seconds"] = E.WINDOW_SECONDS
        h5.attrs["n_events"] = psd_matrix.shape[0]
        h5.attrs["n_cache_files"] = n_total
        h5.attrs["description"] = (
            "Per-survey ambulatory biomarker features: off-stimulation mean "
            "power spectral density (0-99.5 Hz at 0.5 Hz) per pain event, "
            "with the pain scores for that survey attached. Spectral method: "
            "Summit.TimeDomain2PowerChannel.")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[{device_side}] wrote {psd_matrix.shape[0]} events "
          f"({n_total} cache files) -> {out_path} ({size_mb:.2f} MB)", flush=True)
    return out_path

if __name__ == "__main__":
    n_workers = int(sys.argv[1])
    sides = sys.argv[2:]
    for s in sides:
        extract_hemisphere_parallel(s, n_workers)
    print("PARALLEL_EXTRACT_DONE", flush=True)
