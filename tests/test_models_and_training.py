"""Tests for model init, forward/backward passes, checkpoints, resume, and smoke training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import SUPPORTED_MODELS, build_model, count_parameters
from src.training import build_loss_fn, compute_class_weights, load_checkpoint, save_checkpoint, train_model


class TestModelsAndTraining(unittest.TestCase):
    def test_all_models_forward_backward_and_freeze_modes(self) -> None:
        x3 = torch.randn(2, 3, 64, 64)
        x1 = torch.randn(2, 1, 64, 64)
        y = torch.tensor([0, 2], dtype=torch.long)
        crit = torch.nn.CrossEntropyLoss()

        for name in SUPPORTED_MODELS:
            m3 = build_model(name, num_classes=3, in_channels=3, image_size=64, pretrained=False)
            out3 = m3(x3)
            self.assertEqual(tuple(out3.shape), (2, 3), msg=f"3ch shape failed for {name}")
            loss = crit(out3, y)
            loss.backward()
            params = count_parameters(m3)
            self.assertGreater(params["total_params"], 0)
            self.assertGreater(params["trainable_params"], 0)

            m1 = build_model(name, num_classes=4, in_channels=1, image_size=64, pretrained=False)
            out1 = m1(x1)
            self.assertEqual(tuple(out1.shape), (2, 4), msg=f"1ch shape failed for {name}")

        # Sequence inputs for RNN & Hybrid models (verify padded zero frames are masked out of BN)
        x_seq = torch.randn(2, 5, 1, 64, 64)
        x_seq_alt_pad = x_seq.clone()
        x_seq_alt_pad[1, 3:, ...] = 99.0  # masked-out frames must not alter train BN or outputs
        mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
        for name in ("lstm", "bilstm", "resnet50_lstm", "resnet50_bilstm", "efficientnet_b0_bilstm"):
            ms = build_model(name, num_classes=3, in_channels=1, image_size=64, pretrained=False)
            ms.eval()
            with torch.no_grad():
                o1 = ms((x_seq, mask))
                o2 = ms((x_seq_alt_pad, mask))
            self.assertTrue(torch.allclose(o1, o2, atol=1e-5), msg=f"Masked pad frames leaked for {name}")
            ms.train()
            outs = ms((x_seq, mask))
            self.assertEqual(tuple(outs.shape), (2, 3), msg=f"seq shape failed for {name}")
            crit(outs, y).backward()

        # Backbone freeze & partial fine-tuning checks
        m_frozen = build_model("resnet50", num_classes=3, in_channels=3, pretrained=False, backbone_mode="frozen")
        m_partial = build_model("resnet50", num_classes=3, in_channels=3, pretrained=False, backbone_mode="partial")
        p_frozen = count_parameters(m_frozen)
        p_partial = count_parameters(m_partial)
        self.assertLess(p_frozen["trainable_params"], p_partial["trainable_params"])
        self.assertLess(p_partial["trainable_params"], p_partial["total_params"])

    def test_class_weights_checkpoint_and_smoke_resume(self) -> None:
        w = compute_class_weights([0, 0, 0, 1, 2], num_classes=3)
        self.assertEqual(tuple(w.shape), (3,))
        self.assertGreater(float(w[1]), float(w[0]))

        # Synthetic tiny loaders for end-to-end train + checkpoint + resume
        torch.manual_seed(42)
        x_tr = torch.randn(12, 3, 64, 64)
        y_tr = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
        x_va = torch.randn(6, 3, 64, 64)
        y_va = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
        loaders = {
            "train": DataLoader(TensorDataset(x_tr, y_tr), batch_size=4, shuffle=True),
            "val": DataLoader(TensorDataset(x_va, y_va), batch_size=4),
        }

        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "ckpt"
            model = build_model("autoencoder_classifier", num_classes=3, in_channels=3, image_size=64)
            res1 = train_model(
                model, loaders, num_classes=3, checkpoint_dir=cdir,
                device=torch.device("cpu"), epochs=2, lr=1e-3,
                use_mixed_precision=False, autoencoder_pretrain_epochs=1, resume=False,
            )
            self.assertEqual(res1["epochs_completed"], 2)
            self.assertTrue((cdir / "best.pt").exists())
            self.assertTrue((cdir / "last.pt").exists())

            # Resume to epoch 3
            model2 = build_model("autoencoder_classifier", num_classes=3, in_channels=3, image_size=64)
            res2 = train_model(
                model2, loaders, num_classes=3, checkpoint_dir=cdir,
                device=torch.device("cpu"), epochs=3, lr=1e-3,
                use_mixed_precision=False, resume=True,
            )
            self.assertEqual(res2["epochs_completed"], 3)
            ckpt = load_checkpoint(cdir / "best.pt")
            self.assertIn("model_state", ckpt)


if __name__ == "__main__":
    unittest.main()

