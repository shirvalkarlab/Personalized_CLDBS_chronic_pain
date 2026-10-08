"""
Tests for stability_and_closed_loop/plot_figureS8_composite.py.

Run from python_code/:
    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                "stability_and_closed_loop"))
import plot_figureS8_composite as s8  # noqa: E402


def write_grid(folder, side, n_windows):
    frequencies = np.arange(0, 51, 1.0)
    rng = np.random.default_rng(len(side) + n_windows)
    auc = rng.uniform(0.5, 0.75, size=(n_windows, frequencies.size))
    auc[0, :] = np.nan                       # first window too small to fit
    pd.DataFrame({
        "months_included": np.repeat(2 * (np.arange(n_windows) + 1), frequencies.size),
        "center_frequency_hz": np.tile(frequencies, n_windows),
        "auc": auc.ravel(),
        "n_low_pain": 5, "n_high_pain": 5,
    }).to_csv(os.path.join(folder, f"{side}_auc_grid.csv"), index=False)
    return auc


class FigureS8CompositeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.mkdtemp()
        cls.truth = {side: write_grid(cls.folder, side, 4 + k) for k, side in enumerate(s8.SIDES)}
        cls.grids = s8.load_grids(cls.folder)
        cls.fig, cls.heatmaps, cls.colorbars = s8.build_figure(cls.grids)

    def test_panels_are_in_order_with_corrected_titles(self):
        self.assertEqual(s8.SIDES, ["P1R", "P2L", "P2R", "P3L", "P3R", "P4L", "P4R", "P5L", "P5R"])
        titles = [ax.get_title() for ax in self.heatmaps]
        self.assertEqual(titles[0], "P1 Right")
        self.assertEqual(titles[6], "P4 Right")
        letters = [t.get_text() for ax in self.heatmaps for t in ax.texts]
        self.assertEqual(letters, list("abcdefghi"))

    def test_axis_labels_only_on_the_first_panel(self):
        self.assertEqual(self.heatmaps[0].get_ylabel(), "Months since first ambulatory survey")
        self.assertTrue(all(ax.get_ylabel() == "" for ax in self.heatmaps[1:]))
        self.assertEqual(self.colorbars[0].ax.get_ylabel(), "AUC, low vs high SF-MPQv2")
        self.assertTrue(all(cb.ax.get_ylabel() == "" for cb in self.colorbars[1:]))
        self.assertTrue(all("Center frequency" in ax.get_xlabel() for ax in self.heatmaps))

    def test_smoothing_is_display_only(self):
        for ax, side in zip(self.heatmaps, s8.SIDES):
            image = ax.get_images()[0]
            self.assertEqual(image.get_interpolation(), "bilinear")
            np.testing.assert_array_equal(np.ma.filled(image.get_array(), np.nan), self.truth[side])


if __name__ == "__main__":
    unittest.main()
