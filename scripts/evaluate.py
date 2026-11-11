"""Evaluate or re-evaluate a completed experiment checkpoint on its untouched test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import create_dataloaders
from src.evaluation import evaluate_model_on_test
from src.experiments import ExperimentManager
from src.models import build_model
from src.training import TRACK_TO_TASK
from src.utils import get_device, load_all_configs, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an existing experiment checkpoint.")
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()

    cfg = load_all_configs()
    mgr = ExperimentManager(cfg)
    edir = mgr.exp_dir(args.experiment_id)
    cfg_path = edir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {cfg_path}")
    ecfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    track = ecfg["track"]
    _, task, mode, class_names, in_channels = TRACK_TO_TASK[track]
    img_sz = int(ecfg.get("image_size", 224))
    loaders, _ = create_dataloaders(
        cfg,
        task=task,
        mode=mode,
        batch_size=ecfg.get("batch_size"),
        image_size=img_sz,
        use_clahe=ecfg.get("use_clahe"),
        return_meta_eval=True,
    )
    extra = {"backbone_mode": ecfg["backbone_mode"]} if ecfg.get("backbone_mode") else {}
    model = build_model(ecfg["model"], num_classes=len(class_names), in_channels=in_channels, image_size=img_sz, cfg=cfg, pretrained=False, **extra)
    device = get_device(prefer_gpu=True)
    res = evaluate_model_on_test(
        model=model,
        test_loader=loaders["test"],
        checkpoint_path=mgr.ckpt_dir(args.experiment_id) / "best.pt",
        class_names=class_names,
        device=device,
        predictions_dir=resolve_path(cfg["paths"]["results_dir"]) / "predictions",
        exp_id=args.experiment_id,
        use_amp=bool(ecfg.get("use_mixed_precision", True)),
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

