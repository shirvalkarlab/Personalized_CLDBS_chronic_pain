"""
Tests for the Extended Data Figure 4d step-count panel in
plot_extended_figure4_ancillary.py.

Reference values are the original analysis of this panel (pooled one-sided
two-sample t-test, t = 2.9322, df = 203, sd = 41.4129, p = 0.0019).

Run from python_code/:
    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import plot_extended_figure4_ancillary as ed4  # noqa: E402


class StepCountPanelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.days = ed4.load_step_days()

    def test_days_are_the_original_analysis_days(self):
        counts = self.days.groupby(["participant", "condition"]).size().to_dict()
        self.assertEqual(counts, {("P1", "closed-loop"): 21, ("P1", "sham"): 7,
                                  ("P2", "closed-loop"): 61, ("P2", "sham"): 69,
                                  ("P4", "closed-loop"): 11, ("P4", "sham"): 36})

    def test_pooled_t_test_reproduces_the_original(self):
        pooled = ed4.pooled_step_test(self.days)
        self.assertAlmostEqual(pooled["t"], 2.9322, places=4)
        self.assertEqual(pooled["df"], 203)
        self.assertAlmostEqual(pooled["p"], 0.0019, places=4)

    def test_per_participant_rank_sum_tests(self):
        tests = ed4.participant_step_tests(self.days).set_index("participant")
        self.assertEqual(tests.index.tolist(), ["P1", "P2", "P4"])
        self.assertAlmostEqual(tests.loc["P1", "p"], 0.0473, places=4)
        self.assertAlmostEqual(tests.loc["P2", "p"], 0.7325, places=4)
        self.assertAlmostEqual(tests.loc["P4", "p"], 0.0426, places=4)

    def test_participant_contrasts_match_matlab(self):
        # MATLAB: fitlm(T, 'steps ~ pt_id*stim_group'), closed-loop minus sham per participant.
        contrasts = ed4.participant_step_contrasts(self.days).set_index("participant")
        self.assertEqual(contrasts["df"].unique().tolist(), [199])
        for participant, estimate, p in (("P1", 55.5, 0.0017), ("P2", 1.2, 0.8680), ("P4", 24.3, 0.0789)):
            self.assertAlmostEqual(contrasts.loc[participant, "estimate"], estimate, places=1)
            self.assertAlmostEqual(contrasts.loc[participant, "p"], p, places=4)


if __name__ == "__main__":
    unittest.main()
