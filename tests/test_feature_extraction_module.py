"""Unit tests for Phase 3 handcrafted and deep feature extraction modules."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src.feature_extraction.analysis import _evaluate_handcrafted_ml, _run_dim_reduction
from src.feature_extraction.deep_features import DEEP_BACKBONE_DIMS
from src.feature_extraction.handcrafted import extract_single_handcrafted


class TestFeatureExtractionModule(unittest.TestCase):
    def test_handcrafted_fundus_and_oct_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f_path = Path(tmp) / "fundus.png"
            o_path = Path(tmp) / "oct.png"
            rng = np.random.default_rng(42)
            Image.fromarray(rng.integers(20, 230, (64, 64, 3), dtype=np.uint8), mode="RGB").save(f_path)
            Image.fromarray(rng.integers(10, 200, (64, 64), dtype=np.uint8), mode="L").save(o_path)

            f_feats, f_hog = extract_single_handcrafted(f_path, modality="fundus", image_size=224)
            o_feats, o_hog = extract_single_handcrafted(o_path, modality="oct", image_size=224)

            self.assertIn("rgb_rg_ratio", f_feats)
            self.assertNotIn("rgb_rg_ratio", o_feats)
            self.assertIn("glcm_contrast_d1", o_feats)
            self.assertIn("lbp_entropy", f_feats)
            self.assertIn("edge_canny_density", o_feats)
            self.assertGreater(len(f_hog), 100)
            self.assertEqual(len(f_hog), len(o_hog))
            self.assertTrue(all(np.isfinite(list(f_feats.values()))))
            self.assertTrue(all(np.isfinite(list(o_feats.values()))))

    def test_dim_reduction_and_leakfree_ml(self) -> None:
        self.assertEqual(DEEP_BACKBONE_DIMS, {"resnet50": 2048, "densenet121": 1024, "efficientnet_b0": 1280})
        rng = np.random.default_rng(7)
        X = rng.normal(size=(30, 12)).astype(np.float32)
        labels = ["Normal"] * 10 + ["Papilledema"] * 10 + ["Pseudopapilledema"] * 10
        X[:10, 0] += 2.0
        X[10:20, 1] += 2.0
        df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(30)],
            "label": labels,
            "cluster_id": [f"c{i}" for i in range(30)],
        })
        pca2, tsne2, umap2, cum_var = _run_dim_reduction(X, seed=42)
        self.assertEqual(pca2.shape, (30, 2))
        self.assertEqual(tsne2.shape, (30, 2))
        self.assertEqual(umap2.shape, (30, 2))
        self.assertLessEqual(float(cum_var[-1]), 1.0001)

        clf_rows, imp_df = _evaluate_handcrafted_ml(df, X, [f"f{i}" for i in range(12)], "fundus_3class", "label")
        self.assertEqual(len(clf_rows), 2)
        self.assertEqual(len(imp_df), 12)
        self.assertGreater(clf_rows[0]["oof_macro_f1"], 0.4)


if __name__ == "__main__":
    unittest.main()

