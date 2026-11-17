"""Unit tests for Phase 5 extended CV metrics, calibration, and sequence aggregation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.cv_metrics import aggregate_sequence_level_predictions, compute_ece, compute_extended_metrics


class TestCVEvaluationModule(unittest.TestCase):
    def test_extended_metrics_and_calibration(self) -> None:
        yt = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        yp = np.array([0, 1, 1, 1, 2, 2, 3, 2], dtype=np.int64)
        probs = np.array([
            [0.85, 0.10, 0.03, 0.02],
            [0.35, 0.55, 0.05, 0.05],
            [0.05, 0.85, 0.05, 0.05],
            [0.05, 0.80, 0.10, 0.05],
            [0.05, 0.05, 0.85, 0.05],
            [0.02, 0.08, 0.85, 0.05],
            [0.02, 0.03, 0.05, 0.90],
            [0.05, 0.05, 0.55, 0.35],
        ], dtype=np.float64)
        m = compute_extended_metrics(yt, yp, probs, ["G1", "G2", "G3", "G4"], is_severity=True)
        for key in ("accuracy", "error_rate", "micro_f1", "weighted_f1", "macro_f1", "mcc", "cohen_kappa", "log_loss", "brier_score", "ece", "qwk", "mae", "mse"):
            self.assertIsNotNone(m[key])
        self.assertAlmostEqual(m["accuracy"] + m["error_rate"], 1.0, places=5)
        self.assertGreaterEqual(compute_ece(yt, probs), 0.0)
        self.assertLessEqual(compute_ece(yt, probs), 1.0)

    def test_sequence_level_aggregation(self) -> None:
        df = pd.DataFrame([
            {"group_id": "seq_A", "true_idx": 0, "prob_Normal": 0.9, "prob_Papilledema": 0.05, "prob_Pseudopapilledema": 0.05},
            {"group_id": "seq_A", "true_idx": 0, "prob_Normal": 0.8, "prob_Papilledema": 0.10, "prob_Pseudopapilledema": 0.10},
            {"group_id": "seq_B", "true_idx": 1, "prob_Normal": 0.1, "prob_Papilledema": 0.80, "prob_Pseudopapilledema": 0.10},
        ])
        seq_df, seq_m = aggregate_sequence_level_predictions(df, ["Normal", "Papilledema", "Pseudopapilledema"])
        self.assertEqual(len(seq_df), 2)
        self.assertEqual(seq_m["num_samples"], 2)
        self.assertAlmostEqual(seq_m["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

