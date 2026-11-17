"""Unit tests for Phase 2 preprocessing comparison module."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src.preprocessing.comparison import _build_before_after_summary, _compute_gray_and_quality_metrics
from src.preprocessing.plots import plot_class_distributions


class TestPreprocessingModule(unittest.TestCase):
    def test_gray_and_quality_metrics_synthetic(self) -> None:
        img = np.tile(np.linspace(20, 220, 64, dtype=np.uint8), (64, 1))
        m = _compute_gray_and_quality_metrics(img)
        self.assertGreater(m["intensity_mean"], 100.0)
        self.assertGreater(m["rms_contrast"], 0.1)
        self.assertGreater(m["sobel_grad_mean"], 0.0)
        self.assertTrue(all(np.isfinite(list(m.values()))))

    def test_before_after_summary_and_plot_creation(self) -> None:
        f_df = pd.DataFrame({
            "label": ["Normal", "Papilledema", "Pseudopapilledema"],
            "raw_intensity_mean": [100.0, 110.0, 105.0],
            "raw_intensity_std": [40.0, 42.0, 39.0],
            "raw_rms_contrast": [0.16, 0.17, 0.15],
            "raw_michelson_contrast": [0.95, 0.96, 0.94],
            "raw_laplacian_var": [80.0, 75.0, 82.0],
            "raw_sobel_grad_mean": [12.0, 11.5, 12.2],
            "raw_snr_proxy": [2.5, 2.6, 2.7],
            "prep_intensity_mean": [101.0, 111.0, 106.0],
            "prep_intensity_std": [39.0, 41.0, 38.0],
            "prep_rms_contrast": [0.15, 0.16, 0.14],
            "prep_michelson_contrast": [0.94, 0.95, 0.93],
            "prep_laplacian_var": [70.0, 65.0, 72.0],
            "prep_sobel_grad_mean": [11.0, 10.5, 11.2],
            "prep_snr_proxy": [2.6, 2.7, 2.8],
            "clahe_intensity_mean": [115.0, 122.0, 118.0],
            "clahe_intensity_std": [45.0, 47.0, 44.0],
            "clahe_rms_contrast": [0.18, 0.19, 0.17],
            "clahe_michelson_contrast": [0.99, 0.99, 0.98],
            "clahe_laplacian_var": [130.0, 125.0, 128.0],
            "clahe_sobel_grad_mean": [18.0, 17.5, 18.2],
            "clahe_snr_proxy": [2.5, 2.6, 2.6],
            "norm_tensor_mean": [0.05, 0.1, 0.08],
            "norm_tensor_std": [1.0, 1.02, 0.98],
        })
        o_df = f_df.iloc[:3].copy()
        o_df["label"] = ["Normal", "Papilledema", "Pseudopapilledema"]
        o_df["label_severity"] = [np.nan, "G2", np.nan]
        summary = _build_before_after_summary(f_df, o_df)
        self.assertIn("fundus_3class", summary["track"].values)
        self.assertIn("clahe", summary["stage"].values)

        cls_df = pd.DataFrame([
            {"track": "fundus_3class", "class": "Normal", "count": 10, "pct": 50.0},
            {"track": "fundus_3class", "class": "Papilledema", "count": 5, "pct": 25.0},
            {"track": "fundus_3class", "class": "Pseudopapilledema", "count": 5, "pct": 25.0},
            {"track": "oct_individual", "class": "Normal", "count": 4, "pct": 20.0},
            {"track": "oct_individual", "class": "Papilledema", "count": 12, "pct": 60.0},
            {"track": "oct_individual", "class": "Pseudopapilledema", "count": 4, "pct": 20.0},
            {"track": "oct_sequence", "class": "Normal", "count": 2, "pct": 20.0},
            {"track": "oct_sequence", "class": "Papilledema", "count": 6, "pct": 60.0},
            {"track": "oct_sequence", "class": "Pseudopapilledema", "count": 2, "pct": 20.0},
            {"track": "oct_severity", "class": "G1", "count": 1, "pct": 10.0},
            {"track": "oct_severity", "class": "G2", "count": 5, "pct": 50.0},
            {"track": "oct_severity", "class": "G3", "count": 3, "pct": 30.0},
            {"track": "oct_severity", "class": "G4", "count": 1, "pct": 10.0},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_class_distributions(cls_df, Path(tmp))
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(p.exists() and p.stat().st_size > 1000 for p in paths))


if __name__ == "__main__":
    unittest.main()

