"""Tests for metrics, statistics, visualizations, experiment manager, and notebooks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import compute_classification_metrics
from src.experiments import ExperimentManager
from src.statistics import (
    cohens_d_paired,
    compare_models_for_track,
    compute_descriptive_statistics,
    grouped_bootstrap_ci,
    holm_bonferroni,
    mcnemar_test,
    rank_biserial_paired,
)
from src.utils import load_all_configs
from src.visualization import (
    plot_confidence_intervals,
    plot_confusion_matrices,
    plot_model_comparisons,
    plot_roc_and_pr_curves,
    plot_statistical_significance_heatmap,
    plot_training_curves,
)


class TestEvalStatsVizAndNotebooks(unittest.TestCase):
    def test_metrics_and_statistics(self) -> None:
        classes = ["Normal", "Papilledema", "Pseudopapilledema"]
        yt = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
        yp = np.array([0, 0, 1, 2, 2, 2, 0, 1, 2, 1, 1, 2])
        probs = np.eye(3)[yp] * 0.8 + 0.2 / 3.0

        m = compute_classification_metrics(yt, yp, probs, classes)
        self.assertAlmostEqual(m["accuracy"], 10 / 12, places=4)
        self.assertAlmostEqual(m["balanced_accuracy"], m["macro_recall"], places=4)
        self.assertIn("weighted_precision", m)
        self.assertIn("weighted_recall", m)
        self.assertIsNotNone(m["roc_auc_ovr"])
        self.assertIsNotNone(m["pr_auc_ovr"])
        self.assertIn("Normal", m["per_class"])
        self.assertIn("specificity", m["per_class"]["Normal"])

        # Grouped bootstrap CI
        pred_df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(len(yt))],
            "group_id": [f"g{i // 2}" for i in range(len(yt))],
            "true_idx": yt,
            "pred_idx": yp,
            "correct": yt == yp,
        })
        ci = grouped_bootstrap_ci(pred_df, classes, n_resamples=100, seed=42)
        self.assertTrue(any(r["metric"] == "macro_f1" for r in ci))

        # Holm-Bonferroni, Cohen's d, rank-biserial, McNemar
        adj = holm_bonferroni([0.01, 0.04, 0.20])
        self.assertAlmostEqual(adj[0], 0.03)
        self.assertAlmostEqual(adj[1], 0.08)
        self.assertAlmostEqual(adj[2], 0.20)

        a = [0.81, 0.83, 0.85, 0.84, 0.86]
        b = [0.75, 0.77, 0.79, 0.78, 0.80]
        d = cohens_d_paired(a, b)
        rb = rank_biserial_paired(a, b)
        self.assertIsNotNone(d)
        self.assertGreater(d, 0.0)
        self.assertEqual(rb, 1.0)

        mcn = mcnemar_test([True] * 25 + [False] * 5, [False] * 20 + [True] * 5 + [False] * 5)
        self.assertLess(mcn["p_value"], 0.01)

        # Track comparison
        runs_df = pd.DataFrame([
            {"track": "fundus_3class", "model": mdl, "seed": sd, "macro_f1": val, "accuracy": val}
            for mdl, vals in [("m1", a), ("m2", b), ("m3", [0.70, 0.72, 0.71, 0.73, 0.74])]
            for sd, val in zip([1, 2, 3, 4, 5], vals)
        ])
        desc = compute_descriptive_statistics(runs_df, metric_cols=("macro_f1",))
        self.assertEqual(len(desc), 3)
        res = compare_models_for_track(runs_df, {"m1": pred_df, "m2": pred_df.assign(correct=~pred_df["correct"])}, "fundus_3class")
        self.assertFalse(res["hypothesis_tests"].empty)
        self.assertFalse(res["pairwise_comparisons"].empty)

    def test_visualizations_and_experiment_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdir = Path(tmp) / "figures"
            hdf = pd.DataFrame({
                "epoch": [1, 2], "train_loss": [1.0, 0.5], "val_loss": [0.9, 0.6],
                "train_accuracy": [0.5, 0.8], "val_accuracy": [0.55, 0.75],
                "train_macro_f1": [0.48, 0.78], "val_macro_f1": [0.52, 0.74],
            })
            saved = plot_training_curves({"fundus:resnet50": hdf}, fdir)
            self.assertEqual(len(saved), 4)
            cm_p = plot_confusion_matrices([{"title": "m1", "confusion_matrix": [[5, 1], [0, 4]], "class_names": ["A", "B"]}], fdir)
            self.assertTrue(cm_p.exists())

            runs_df = pd.DataFrame([{
                "track": "fundus_3class", "category": "cnn", "model": "resnet50",
                "macro_f1": 0.8, "accuracy": 0.82, "total_params": 23_000_000,
                "train_duration_sec": 12.0, "inference_latency_ms": 1.5,
            }])
            m_figs = plot_model_comparisons(runs_df, fdir)
            self.assertGreaterEqual(len(m_figs), 4)

            ci_df = pd.DataFrame([{"track": "fundus_3class", "model": "resnet50", "metric": "macro_f1", "point_estimate": 0.8, "ci95_low": 0.7, "ci95_high": 0.9}])
            self.assertTrue(plot_confidence_intervals(ci_df, fdir).exists())
            pw_df = pd.DataFrame([{"track": "fundus_3class", "model_a": "m1", "model_b": "m2", "test_used": "McNemar (exact)", "p_adjusted": 0.04}])
            self.assertTrue(plot_statistical_significance_heatmap(pw_df, fdir).exists())

            pred_df = pd.DataFrame({"true_idx": [0, 1, 0, 1], "prob_A": [0.9, 0.2, 0.8, 0.1], "prob_B": [0.1, 0.8, 0.2, 0.9]})
            roc_figs = plot_roc_and_pr_curves({"resnet50": pred_df}, ["A", "B"], fdir)
            self.assertEqual(len(roc_figs), 2)

        # Experiment manager non-overwrite test
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_all_configs()
            cfg["paths"]["experiments_dir"] = str(Path(tmp) / "experiments")
            cfg["paths"]["checkpoints_dir"] = str(Path(tmp) / "checkpoints")
            cfg["paths"]["results_dir"] = str(Path(tmp) / "results")
            mgr = ExperimentManager(cfg)
            mgr.register_start("t__m__seed1__smoke", {"track": "fundus_3class", "model": "mlp", "seed": 1, "is_smoke": True})
            mgr.mark_completed("t__m__seed1__smoke", {"train_duration_sec": 1.0, "best_epoch": 1, "best_val_macro_f1": 0.4, "best_checkpoint_path": "x"}, {"metrics": {"macro_f1": 0.4, "accuracy": 0.4}, "parameters": {"total_params": 10, "trainable_params": 10}, "error_analysis": {"total_errors": 1}})
            self.assertEqual(len(mgr.collect_completed_runs()), 1)
            mgr.register_start("t__m__seed1", {"track": "fundus_3class", "model": "mlp", "seed": 1, "is_smoke": False})
            mgr.mark_completed("t__m__seed1", {"train_duration_sec": 1.0, "best_epoch": 1, "best_val_macro_f1": 0.5, "best_checkpoint_path": "x"}, {"metrics": {"macro_f1": 0.5, "accuracy": 0.5}, "parameters": {"total_params": 10, "trainable_params": 10}, "error_analysis": {"total_errors": 1}})
            self.assertTrue(mgr.is_completed("t__m__seed1"))
            self.assertEqual(mgr.collect_completed_runs()["exp_id"].tolist(), ["t__m__seed1"])

    def test_notebooks_exist_valid_and_no_duplicate_defs(self) -> None:
        nb_dir = Path("notebooks")
        nbs = sorted(nb_dir.glob("*.ipynb"))
        self.assertGreaterEqual(len(nbs), 1, f"Expected central research notebook in notebooks/, found {[p.name for p in nbs]}")
        self.assertTrue(
            (nb_dir / "Papilledema_Complete_Research_Analysis.ipynb").exists(),
            "Central notebook Papilledema_Complete_Research_Analysis.ipynb is missing.",
        )
        for nb_path in nbs:
            data = json.loads(nb_path.read_text(encoding="utf-8"))
            self.assertIn("cells", data)
            self.assertEqual(data["nbformat"], 4)
            src_text = "\n".join("".join(c.get("source", [])) for c in data["cells"] if c["cell_type"] == "code")
            # Ensure no model or dataset classes are duplicated in notebooks
            self.assertNotIn("class CNNClassifier", src_text)
            self.assertNotIn("class HybridCNNRNN", src_text)
            self.assertNotIn("def train_model(", src_text)


if __name__ == "__main__":
    unittest.main()

