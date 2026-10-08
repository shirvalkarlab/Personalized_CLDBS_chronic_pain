"""
Tests for the permutation-null machinery in 01_run_ml_pipeline.py.

The LDA permutation null is computed with a batched closed-form version of
scikit-learn's LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
evaluated by leave-one-out. These tests check it against scikit-learn itself,
and check that the permutation orders are the ones a serial loop draws.

Run from python_code/decoding_figures/:
    python -m unittest discover -s tests -v
"""
import random
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

from test_binarize_labels import pipeline


def sklearn_loo_decisions(X, y):
    """Held-out decision values of scikit-learn LDA, leave-one-out."""
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        train = np.arange(n) != i
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        model.fit(X[train], y[train])
        out[i] = model.decision_function(X[i:i + 1])[0]
    return out


class BatchedLdaTest(unittest.TestCase):

    def check_against_sklearn(self, n, p, seed):
        rng = np.random.default_rng(seed)
        y = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=float)
        X = rng.normal(size=(3, n, p)) + 0.8 * y[None, :, None] * (rng.random(p) < 0.1)
        X[:, :, 0] = 2.5                       # a constant feature, as after z-scoring a flat channel
        decisions = pipeline.lda_loo_decisions(X, y)
        aucs = pipeline.lda_loo_aucs(X, y)
        for k in range(X.shape[0]):
            reference = sklearn_loo_decisions(X[k], y)
            np.testing.assert_allclose(decisions[k], reference, rtol=1e-7, atol=1e-7)
            self.assertEqual(aucs[k], np.round(roc_auc_score(y, expit(reference)), 3))

    def test_batched_lda_matches_sklearn_when_features_outnumber_surveys(self):
        self.check_against_sklearn(n=24, p=80, seed=1)

    def test_batched_lda_matches_sklearn_when_surveys_outnumber_features(self):
        self.check_against_sklearn(n=40, p=12, seed=2)

    def test_batched_lda_matches_sklearn_with_unequal_classes(self):
        rng = np.random.default_rng(3)
        y = np.array([0] * 9 + [1] * 21, dtype=float)
        X = rng.normal(size=(2, 30, 50))
        decisions = pipeline.lda_loo_decisions(X, y)
        for k in range(2):
            np.testing.assert_allclose(decisions[k], sklearn_loo_decisions(X[k], y), rtol=1e-7, atol=1e-7)


class PermutationOrdersTest(unittest.TestCase):

    def test_orders_equal_the_shuffles_of_a_serial_loop(self):
        windows = [pd.DataFrame(np.zeros((n, 2))) for n in (7, 5, 9)]
        random.seed(11)
        expected = []
        for _ in range(4):
            per_window = []
            for frame in windows:
                idx = list(range(len(frame)))
                random.shuffle(idx)
                per_window.append(idx)
            expected.append(per_window)
        random.seed(11)
        self.assertEqual(pipeline.draw_permutation_orders(windows, 4), expected)


if __name__ == "__main__":
    unittest.main()
