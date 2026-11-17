"""Unit tests for Phase 6 CV statistical tests (Nadeau-Bengio & Friedman/Holm)."""

from __future__ import annotations

import unittest
import numpy as np

from src.statistics.cv_statistics import nadeau_bengio_corrected_ttest


class TestCVStatisticsModule(unittest.TestCase):
    def test_nadeau_bengio_corrected_ttest_is_more_conservative_than_naive(self) -> None:
        diffs = np.array([0.05, 0.04, 0.06, 0.05, 0.04], dtype=np.float64)
        t_nb, p_nb = nadeau_bengio_corrected_ttest(diffs, train_test_ratio=1.0 / 3.0)
        from scipy import stats
        t_naive, p_naive = stats.ttest_1samp(diffs, 0.0)
        self.assertLess(abs(t_nb), abs(t_naive))
        self.assertGreater(p_nb, p_naive)

    def test_zero_variance_diff_handled_cleanly(self) -> None:
        zeros = np.zeros(5, dtype=np.float64)
        t_stat, p_val = nadeau_bengio_corrected_ttest(zeros)
        self.assertEqual(t_stat, 0.0)
        self.assertEqual(p_val, 1.0)


if __name__ == "__main__":
    unittest.main()

