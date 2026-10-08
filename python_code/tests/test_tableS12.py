"""
Tests for build_tableS12_sensitivity_analyses.py.

Reference values are MATLAB fitlme fits of the same model (REML, closed-loop
and sham days, uniform NRS), which reach the highest restricted likelihood on
every row of the table.

Run from python_code/:
    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import build_tableS12_sensitivity_analyses as s12  # noqa: E402

DAILY = s12.load_model_data()


def without_sham(participant):
    return DAILY[~((DAILY["participant"] == participant) & (DAILY["condition"] == "sham"))]


class TableS12Test(unittest.TestCase):

    def test_excluding_p1_sham_reaches_the_highest_reml_likelihood(self):
        fit = s12.fit_reml(without_sham("P1"))
        self.assertAlmostEqual(fit.llf, -1152.3478, places=3)
        self.assertAlmostEqual(fit.params["condition[T.closed-loop]"], -68.15, places=1)

    def test_excluding_p5_sham_reaches_the_highest_reml_likelihood(self):
        fit = s12.fit_reml(without_sham("P5"))
        self.assertAlmostEqual(fit.llf, -1134.8499, places=3)
        self.assertAlmostEqual(fit.params["condition[T.closed-loop]"], -42.26, places=1)

    def test_boundary_fit_reports_a_finite_standard_error(self):
        # Without P5 the random-effects covariance is singular at the optimum.
        row = s12.fit_model(DAILY[DAILY["participant"] != "P5"])
        self.assertAlmostEqual(row["estimate_percent"], -29.8, places=1)
        self.assertAlmostEqual(row["standard_error"], 19.3, delta=0.2)

    def test_full_model_row_is_unchanged(self):
        row = s12.fit_model(DAILY)
        self.assertEqual((row["estimate_percent"], row["standard_error"], row["p_value"]), (-64.4, 31.3, 0.04))


if __name__ == "__main__":
    unittest.main()
