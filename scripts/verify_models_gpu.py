"""Real-data GPU verification across all compatible architectures and experimental tracks."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import create_dataloaders, load_split_df
from src.models import build_model, count_parameters
from src.training import TRACK_TO_TASK, build_loss_fn, load_checkpoint, save_checkpoint
from src.utils import get_device, load_all_configs, seed_everything
from src.utils.device import clear_gpu_memory, move_inputs_to_device


def main() -> int:
    seed_everything(42)
    cfg = load_all_configs()
    device = get_device(prefer_gpu=True)
    print(f"Verified Device: {device}")

    s1 = load_split_df(cfg, "fundus_3class")
    s2 = load_split_df(cfg, "fundus_3class")
    assert s1["sample_id"].equals(s2["sample_id"]) and s1["split"].equals(s2["split"])

    total_checks = 0
    for track in ("fundus_3class", "oct_individual", "oct_sequence", "oct_severity"):
        _, task, mode, class_names, in_ch = TRACK_TO_TASK[track]
        bs = 2 if mode == "sequence" else 4
        n_cls = len(class_names)
        loaders, _ = create_dataloaders(cfg, task=task, mode=mode, batch_size=bs, image_size=224)
        assert "Random" not in str(loaders["val"].dataset.transform), f"Val aug leak in {track}"
        assert "Random" not in str(loaders["test"].dataset.transform), f"Test aug leak in {track}"
        assert "Random" in str(loaders["train"].dataset.transform), f"Missing train aug in {track}"

        raw_in, raw_y = next(iter(loaders["train"]))
        train_labels = (
            [s["label_idx"] for s in loaders["train"].dataset.sequences]
            if mode == "sequence"
            else loaders["train"].dataset.df["label_idx"].tolist()
        )
        criterion = build_loss_fn(train_labels, n_cls, use_class_weights=True, device=device)

        print(f"\n=== {track} (mode={mode}, in_ch={in_ch}, classes={n_cls}) ===")
        for mname in cfg["experiments"][track]["models"]:
            clear_gpu_memory()
            model = build_model(mname, num_classes=n_cls, in_channels=in_ch, image_size=224, cfg=cfg).to(device)
            model.train()
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=(device.type == "cuda"))
            inp = move_inputs_to_device(raw_in, device)
            tgt = raw_y.to(device)

            is_seq = isinstance(raw_in, (tuple, list))
            inactive_prefix = (
                "patch_proj." if (mname in ("lstm", "bilstm") and is_seq)
                else ("slice_stem." if (mname in ("lstm", "bilstm") and not is_seq) else None)
            )
            active_params = [
                (n, p) for n, p in model.named_parameters()
                if p.requires_grad and (inactive_prefix is None or not n.startswith(inactive_prefix))
            ]
            before = {n: p.detach().clone() for n, p in active_params}

            updated = False
            for attempt in range(4):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    if hasattr(model, "forward_with_recon"):
                        logits, rloss = model.forward_with_recon(inp)
                        loss = criterion(logits, tgt) + 0.1 * rloss
                    else:
                        logits = model(inp)
                        loss = criterion(logits, tgt)
                assert logits.shape == (tgt.size(0), n_cls), f"Bad output shape {logits.shape}"
                assert torch.isfinite(loss), f"Non-finite loss for {mname}"

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for _, p in active_params], 1.0)
                missing = [n for n, p in active_params if p.grad is None]
                assert not missing, f"Missing active grads in {mname}: {missing}"
                unscaled_grads = [p.grad for _, p in active_params]
                finite_all = all(torch.isfinite(g).all().item() for g in unscaled_grads)
                max_grad = max((g.abs().max().item() for g in unscaled_grads if torch.isfinite(g).all()), default=0.0)
                scaler.step(opt)
                scaler.update()

                max_delta = max(float((p.detach() - before[n]).abs().max().item()) for n, p in active_params)
                if finite_all and max_grad > 0.0 and max_delta > 0.0:
                    updated = True
                    break
            assert updated, f"Optimizer never applied finite grad update for {mname}"

            with tempfile.TemporaryDirectory() as tmp:
                ckpt_p = Path(tmp) / f"{mname}.pt"
                save_checkpoint(ckpt_p, {"epoch": 1, "model_state": model.state_dict()})
                model.eval()
                with torch.no_grad():
                    ref_out = model(inp).detach().cpu()
                reloaded = build_model(mname, num_classes=n_cls, in_channels=in_ch, image_size=224, cfg=cfg, pretrained=False).to(device)
                ckpt = load_checkpoint(ckpt_p, map_location=device)
                reloaded.load_state_dict(ckpt["model_state"])
                reloaded.eval()
                with torch.no_grad():
                    load_out = reloaded(inp).detach().cpu()
                diff = float((ref_out - load_out).abs().max().item())
                assert diff < 1e-5, f"Checkpoint load mismatch for {mname}: {diff}"

            in_shape = tuple(raw_in[0].shape) if is_seq else tuple(raw_in.shape)
            cnt = count_parameters(model)
            total_checks += 1
            print(
                f"PASS | {track:14s} | {mname:22s} | in={in_shape} | out={tuple(logits.shape)} | "
                f"loss={loss.item():.4f} | steps={attempt + 1} | active_tensors={len(active_params)} | "
                f"max_grad={max_grad:.4f} | max_dW={max_delta:.2e} | ckpt_diff={diff:.1e} | params={cnt['total_params']:,}"
            )
            del model, reloaded, opt
            clear_gpu_memory()

    print(f"\nALL {total_checks} TRACK-MODEL REAL-DATA VERIFICATIONS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

