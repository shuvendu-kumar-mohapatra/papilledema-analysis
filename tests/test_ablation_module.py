"""CPU-only unit tests for Phase 7 ablation suite compilation and plotting."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.ablation.runner import run_ablation_suite


class TestAblationModule(unittest.TestCase):
    def test_ablation_suite_skip_new_training_generates_tables_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            out_dir = root / "ablation"
            ckpt_dir = root / "ckpt_abl"
            pub_dir = root / "pub"
            eval_dir.mkdir(parents=True, exist_ok=True)

            sum_rows = []
            fold_rows = []
            tracks = ["fundus_3class", "oct_individual", "oct_sequence", "oct_severity"]
            models_by_track = {
                "fundus_3class": [
                    ("resnet50", "cnn"), ("densenet121", "cnn"), ("efficientnet_b0", "cnn"),
                    ("lstm", "rnn"), ("bilstm", "rnn"),
                    ("resnet50_lstm", "hybrid"), ("resnet50_bilstm", "hybrid"), ("efficientnet_b0_bilstm", "hybrid"),
                ],
                "oct_individual": [
                    ("resnet50", "cnn"), ("densenet121", "cnn"), ("efficientnet_b0", "cnn"),
                    ("lstm", "rnn"), ("bilstm", "rnn"),
                    ("resnet50_lstm", "hybrid"), ("resnet50_bilstm", "hybrid"), ("efficientnet_b0_bilstm", "hybrid"),
                ],
                "oct_sequence": [
                    ("resnet50", "cnn"), ("densenet121", "cnn"), ("efficientnet_b0", "cnn"),
                    ("lstm", "rnn"), ("bilstm", "rnn"),
                    ("resnet50_bilstm", "hybrid"),
                ],
                "oct_severity": [
                    ("resnet50", "cnn"), ("densenet121", "cnn"), ("efficientnet_b0", "cnn"),
                    ("lstm", "rnn"), ("bilstm", "rnn"),
                    ("resnet50_lstm", "hybrid"), ("resnet50_bilstm", "hybrid"), ("efficientnet_b0_bilstm", "hybrid"),
                ],
            }

            for ti, trk in enumerate(tracks):
                for mi, (mdl, cat) in enumerate(models_by_track[trk]):
                    base = 0.70 + 0.02 * (mi % 4) - 0.03 * ti
                    sum_rows.append({
                        "track": trk,
                        "model": mdl,
                        "category": cat,
                        "total_params": 5_000_000 + mi * 1_000_000,
                        "train_time_mean_sec": 30.0 + mi * 5.0,
                        "macro_f1_mean": base,
                        "macro_f1_std": 0.03,
                        "balanced_accuracy_mean": base + 0.01,
                        "balanced_accuracy_std": 0.025,
                        "accuracy_mean": base + 0.02,
                        "mcc_mean": base - 0.1,
                    })
                    for k in range(1, 6):
                        fold_rows.append({
                            "track": trk,
                            "model": mdl,
                            "category": cat,
                            "fold": k,
                            "n_samples": 30 if trk == "oct_sequence" else 200,
                            "total_params": 5_000_000 + mi * 1_000_000,
                            "train_duration_sec": 30.0 + mi * 5.0,
                            "best_epoch": 10 + k,
                            "accuracy": base + 0.01 * (k - 3),
                            "balanced_accuracy": base + 0.008 * (k - 3),
                            "macro_f1": base + 0.01 * (k - 3),
                            "weighted_f1": base + 0.01 * (k - 3),
                            "mcc": base - 0.1 + 0.01 * (k - 3),
                        })

            pd.DataFrame(sum_rows).to_csv(eval_dir / "cv_model_summary.csv", index=False)
            pd.DataFrame(fold_rows).to_csv(eval_dir / "cv_fold_metrics.csv", index=False)
            pd.DataFrame([{
                "model": "resnet50_bilstm",
                "n_sequences": 21,
                "seq_accuracy": 0.81,
                "seq_macro_f1": 0.78,
                "seq_mcc": 0.65,
            }]).to_csv(eval_dir / "oct_sequence_level_summary.csv", index=False)

            manifest = run_ablation_suite(
                eval_dir=eval_dir,
                output_dir=out_dir,
                ckpt_dir=ckpt_dir,
                pub_fig_dir=pub_dir,
                skip_new_training=True,
                overwrite=False,
            )

            self.assertEqual(len(manifest["ablations_completed"]), 5)
            self.assertEqual(manifest["new_runs_executed"], 0)
            self.assertTrue((out_dir / "abl01_cnn_backbone_comparison.csv").exists())
            self.assertTrue((out_dir / "abl02_rnn_lstm_vs_bilstm.csv").exists())
            self.assertTrue((out_dir / "abl03_hybrid_cnn_rnn_comparison.csv").exists())
            self.assertTrue((out_dir / "abl04_backbone_trainability_folds.csv").exists())
            self.assertTrue((out_dir / "abl05_oct_sequence_length_folds.csv").exists())
            self.assertTrue((out_dir / "abl05_oct_sequence_length_seq_level.csv").exists())
            self.assertTrue((out_dir / "ablation_manifest.json").exists())
            self.assertEqual(len(manifest["figures_generated"]), 12)
            for fp in manifest["figures_generated"]:
                self.assertTrue(Path(fp).exists(), msg=f"Missing generated figure {fp}")
                self.assertTrue((pub_dir / Path(fp).name).exists(), msg=f"Missing pub figure {Path(fp).name}")


if __name__ == "__main__":
    unittest.main()

