"""
Small utility helpers for the pain-decoding figure scripts.

Holds the output-folder helper and the channel/anatomy lookup used by the
numbered scripts in this folder, which together regenerate the pain-decoding
figures (Supplementary Figures S1, S2 and S3) from the precomputed decoding
results.
"""

import os
import json


def foldcheck(folder_path):
    """Create folder_path (and any parents) if it does not exist, then return it.

    Every script calls this before writing figures, so the output directory is
    guaranteed to exist.
    """
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


# Location of the per-participant channel/anatomy lookup table (a JSON file that
# sits beside these scripts).
_CHANNEL_ANATOMY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "channel_anatomy_labels.json")


def get_ch_labels(participant):
    """Return (channel_labels, anatomy_labels, long_labels) for one participant.

    Parameters
    ----------
    participant : str
        De-identified participant label, one of "P1"-"P5".

    Returns
    -------
    channel_labels : list of str
        Recording-channel names in the fixed order used to arrange the
        feature-importance counts (for example "LACC2", "RVPN1").
    anatomy_labels : list of str
        The anatomical location of each channel, aligned element-for-element
        with channel_labels (for example "L ACC", "R VPthal").
    long_labels : list of str
        A human-readable "channel (anatomy)" string for each channel, used for
        axis tick labels.

    Notes
    -----
    The mapping is stored in ``channel_anatomy_labels.json`` beside this module.
    It matches the ``*_counts_ch_ordered.csv`` result files under
    ``decoding_results/`` (whose row index is the channel label and whose
    ``anat`` column is the anatomy), which fix the channel ordering used in the
    published analysis, so re-running the pipeline reproduces that ordering.
    """
    with open(_CHANNEL_ANATOMY_FILE, "r") as handle:
        table = json.load(handle)
    if participant not in table:
        raise KeyError(
            f"No channel/anatomy labels for participant {participant!r}; "
            f"available participants are {sorted(table)}.")
    channel_labels = list(table[participant]["chl"])
    anatomy_labels = list(table[participant]["anat"])
    long_labels = [f"{chl} ({anat})" for chl, anat in zip(channel_labels, anatomy_labels)]
    return channel_labels, anatomy_labels, long_labels
