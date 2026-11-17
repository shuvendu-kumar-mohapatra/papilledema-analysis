"""Unit tests for Phase 4 5-fold CV splits, leakage checks, and dataset chunking."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd

from src.cross_validation.runner import PrechunkedOCTSequenceDataset, make_cv_exp_id
from src.cross_validation.splits import build_and_validate_cv_splits


class TestCrossValidationModule(unittest.TestCase):
    def test_cv_splits_valid_and_zero_fundus_cluster_leakage(self) -> None:
        out = build_and_validate_cv_splits()
        for trk, diag in out["diagnostics"].items():
            self.assertTrue(diag["validation_passed"], f"{trk} failed validation")
            self.assertFalse(diag["patient_independent"])

        f_df = pd.read_csv("results/cross_validation/splits/fundus_3class_cv_folds.csv")
        # Each cluster_id must belong to exactly 1 fold
        cluster_folds = f_df.groupby("cluster_id")["cv_fold"].nunique()
        self.assertTrue((cluster_folds == 1).all())

        # Papilledema OCT prefixes must stay grouped within 1 fold
        o3_df = pd.read_csv("results/cross_validation/splits/oct_individual_cv_folds.csv")
        pap = o3_df[o3_df["label"] == "Papilledema"]
        self.assertTrue((pap.groupby("inferred_subject_prefix")["cv_fold"].nunique() == 1).all())

    def test_prechunked_oct_sequence_dataset_shapes(self) -> None:
        seq_df = pd.read_csv("results/cross_validation/splits/oct_sequence_cv_folds.csv")
        ds = PrechunkedOCTSequenceDataset(seq_df.head(4), seq_len=8, image_size=64, is_train=False, return_meta=True)
        (frames, mask), y, meta = ds[0]
        self.assertEqual(tuple(frames.shape), (8, 1, 64, 64))
        self.assertEqual(tuple(mask.shape), (8,))
        self.assertTrue(bool(mask[0].item()))
        self.assertIn("cv__oct_sequence__resnet50_bilstm__fold1__seed42", make_cv_exp_id("oct_sequence", "resnet50_bilstm", 1))


if __name__ == "__main__":
    unittest.main()

