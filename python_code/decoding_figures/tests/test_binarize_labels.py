"""
Tests that pin binarize_labels() in 01_run_ml_pipeline.py to its published behaviour.

binarize_labels() splits pain scores into a low class (at or below the 30th
percentile) and a high class (at or above the 70th percentile) for the LDA
models, and returns the low and high counts written to
lda_num_samples_per_model.csv (Figure S2). These tests fail if a change to the
function would alter the published classes or counts.

Run from python_code/decoding_figures/:
    python -m unittest discover -s tests -v
"""
import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.join(_HERE, os.pardir, "01_run_ml_pipeline.py")


def load_pipeline():
    """Import 01_run_ml_pipeline.py (its name starts with a digit, so not by `import`)."""
    # Importing the script creates its results folder; keep that out of the repository.
    os.environ["DECODING_RESULTS_ROOT"] = tempfile.mkdtemp(prefix="decoding_test_")
    saved_argv = sys.argv
    sys.argv = [_PIPELINE]
    try:
        spec = importlib.util.spec_from_file_location("run_ml_pipeline", _PIPELINE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved_argv
    return module


pipeline = load_pipeline()


def classes(labels):
    """Return (binary labels, low count, high count) for a list of scores."""
    return pipeline.binarize_labels(np.array(labels, dtype=float))


class BinarizeLabelsTest(unittest.TestCase):

    def test_untied_percentiles_split_bottom_and_top_30_percent(self):
        # 0..9: 30th percentile 2.7, 70th percentile 6.3.
        y, n_low, n_high = classes(range(10))
        np.testing.assert_array_equal(y[:3], [0, 0, 0])
        self.assertTrue(np.isnan(y[3:7]).all())
        np.testing.assert_array_equal(y[7:], [1, 1, 1])
        self.assertEqual((n_low, n_high), (3, 3))

    def test_tied_value_is_trained_as_low_pain(self):
        # Both percentiles are 7: the six surveys at 7 are trained as low pain.
        y, _, _ = classes([3, 5, 7, 7, 7, 7, 7, 7, 8, 9])
        np.testing.assert_array_equal(y, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    def test_counts_are_surveys_at_or_below_30th_and_at_or_above_70th(self):
        # The published counts: surveys at or below the 30th percentile, and
        # surveys at or above the 70th. When the percentiles are equal, surveys
        # at the shared value are in both counts.
        _, n_low, n_high = classes([3, 5, 7, 7, 7, 7, 7, 7, 8, 9])
        self.assertEqual((n_low, n_high), (8, 8))


if __name__ == "__main__":
    unittest.main()
