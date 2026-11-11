"""Tests for dataset loading, preprocessing, label mapping, split integrity, and leakage checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.data import (
    FUNDUS_CLASSES,
    OCT_3CLASSES,
    OCT_SEVERITY_CLASSES,
    MedicalImageDataset,
    OCTSequenceDataset,
    apply_fundus_clahe,
    create_fundus_splits,
    create_oct_splits,
    is_ir_slo_image,
    load_preprocessed_pil,
    pil_to_normalized_tensor,
    validate_no_leakage,
)


def _make_dummy_img(path: Path, size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (120, 80, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = color
    arr[0:4, 0:4] = (200, 10, 10)  # distinct watermark-like corner
    Image.fromarray(arr).save(path)


class TestDataAndSplits(unittest.TestCase):
    def test_preprocessing_and_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            p_rgb = tdir / "fundus.jpg"
            p_oct = tdir / "oct.jpg"
            _make_dummy_img(p_rgb, (80, 80), (150, 90, 60))
            _make_dummy_img(p_oct, (80, 80), (100, 100, 100))

            im_f = load_preprocessed_pil(p_rgb, "fundus", image_size=64, use_clahe=True)
            self.assertEqual(im_f.size, (64, 64))
            self.assertEqual(im_f.mode, "RGB")
            t_f = pil_to_normalized_tensor(im_f, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            self.assertEqual(tuple(t_f.shape), (3, 64, 64))

            im_o = load_preprocessed_pil(p_oct, "oct", image_size=64)
            self.assertEqual(im_o.mode, "L")
            t_o = pil_to_normalized_tensor(im_o, [0.5], [0.5])
            self.assertEqual(tuple(t_o.shape), (1, 64, 64))

            df_f = pd.DataFrame([{
                "sample_id": "s1", "image_path": str(p_rgb), "label": "Normal",
                "label_idx": 0, "cluster_id": "c0",
            }])
            ds_f = MedicalImageDataset(df_f, modality="fundus", image_size=64, is_train=False, return_meta=True)
            x, y, meta = ds_f[0]
            self.assertEqual(tuple(x.shape), (3, 64, 64))
            self.assertEqual(y, 0)
            self.assertEqual(meta["group_id"], "c0")

            # OCT sequence dataset test
            rows = []
            for idx in range(5):
                sp = tdir / f"seq_{idx}.jpg"
                _make_dummy_img(sp, (64, 64), (100, 100, 100))
                rows.append({
                    "sample_id": f"o_{idx}", "image_path": str(sp), "label": "Normal",
                    "label_idx": 0, "inferred_sequence_id": "Normal__test__under6",
                    "slice_index": idx,
                })
            ds_seq = OCTSequenceDataset(pd.DataFrame(rows), seq_len=8, stride=8, image_size=64, is_train=False)
            (frames, mask), label = ds_seq[0]
            self.assertEqual(tuple(frames.shape), (8, 1, 64, 64))
            self.assertEqual(int(mask.sum().item()), 5)
            self.assertEqual(label, 0)

            # Verify train sequence augmentation advances RNG across calls instead of freezing
            ds_seq_tr = OCTSequenceDataset(pd.DataFrame(rows), seq_len=8, stride=8, image_size=64, is_train=True)
            (f_a, _), _ = ds_seq_tr[0]
            (f_b, _), _ = ds_seq_tr[0]
            self.assertFalse(torch.allclose(f_a, f_b))

    def test_ir_slo_detector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            slo = Path(tmp) / "slo.jpg"
            arr = np.full((701, 269, 3), 90, dtype=np.uint8)
            Image.fromarray(arr).save(slo)
            flagged, _ = is_ir_slo_image(slo)
            self.assertTrue(flagged)

    def test_grouped_splits_and_leakage_detection(self) -> None:
        rows = []
        for c_i, cls in enumerate(FUNDUS_CLASSES):
            for k in range(10):
                cid = f"c_{cls}_{k // 2}"
                rows.append({
                    "sample_id": f"{cls}_{k}",
                    "image_path": f"fake/{cls}/{k}.jpg",
                    "label": cls,
                    "label_idx": c_i,
                    "pixel_sha256": f"hash_{cls}_{k}",
                    "cluster_id": cid,
                })
        manifest = pd.DataFrame(rows)
        splits = create_fundus_splits(manifest, seed=42)
        rep = validate_no_leakage(splits, "fundus_3class", expected_classes=FUNDUS_CLASSES)
        self.assertTrue(rep["passed"])

        # Inject duplicate cluster leakage and verify exception
        leaked = splits.copy()
        val_idx = leaked[leaked["split"] == "val"].index[0]
        train_cluster = leaked[leaked["split"] == "train"]["cluster_id"].iloc[0]
        leaked.loc[val_idx, "cluster_id"] = train_cluster
        with self.assertRaises(RuntimeError):
            validate_no_leakage(leaked, "fundus_3class", expected_classes=FUNDUS_CLASSES)

        # Inject forbidden augmented path and verify exception
        bad_path = splits.copy()
        bad_path.loc[0, "image_path"] = "data/raw/OCT/Augmented Data/Normal/1.jpg"
        with self.assertRaises(RuntimeError):
            validate_no_leakage(bad_path, "fundus_3class", expected_classes=FUNDUS_CLASSES)

    def test_real_manifests_if_present(self) -> None:
        f_sp = Path("data/splits/fundus_splits.csv")
        o3_sp = Path("data/splits/oct_3class_splits.csv")
        osv_sp = Path("data/splits/oct_severity_splits.csv")
        if f_sp.exists():
            rep = validate_no_leakage(pd.read_csv(f_sp), "fundus_3class", expected_classes=FUNDUS_CLASSES)
            self.assertTrue(rep["passed"])
        if o3_sp.exists():
            rep = validate_no_leakage(pd.read_csv(o3_sp), "oct_3class", expected_classes=OCT_3CLASSES)
            self.assertTrue(rep["passed"])
        if osv_sp.exists():
            rep = validate_no_leakage(pd.read_csv(osv_sp), "oct_severity", expected_classes=OCT_SEVERITY_CLASSES)
            self.assertTrue(rep["passed"])


if __name__ == "__main__":
    unittest.main()

