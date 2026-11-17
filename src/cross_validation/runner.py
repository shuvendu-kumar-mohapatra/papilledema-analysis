"""Resumable 5-fold cross-validation training and outer test evaluation runner."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.cross_validation.splits import build_and_validate_cv_splits
from src.data.augmentation import build_eval_transform, build_train_augmentations
from src.data.datasets import MedicalImageDataset
from src.data.preprocessing import pil_to_normalized_tensor
from src.data.validation import FUNDUS_CLASSES, OCT_3CLASSES, OCT_SEVERITY_CLASSES
from src.evaluation.evaluator import evaluate_model_on_test
from src.models.model_factory import build_model
from src.training.trainer import train_model
from src.utils.config import load_all_configs, load_config, resolve_path
from src.utils.device import clear_gpu_memory, get_device
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import seed_everything

CV_TRACK_SPECS: dict[str, tuple[str, str, list[str], int]] = {
    "fundus_3class": ("fundus", "image", FUNDUS_CLASSES, 3),
    "oct_individual": ("oct", "image", OCT_3CLASSES, 1),
    "oct_sequence": ("oct", "sequence", OCT_3CLASSES, 1),
    "oct_severity": ("oct", "image", OCT_SEVERITY_CLASSES, 1),
}


class PrechunkedOCTSequenceDataset(Dataset):
    """Dataset built directly from pre-validated CV sequence windows (paths_json)."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 8,
        image_size: int = 224,
        is_train: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        return_meta: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.image_size = image_size
        self.is_train = is_train
        self.mean = mean or [0.5]
        self.std = std or [0.5]
        self.return_meta = return_meta
        self.transform = build_train_augmentations("oct", image_size) if is_train else build_eval_transform(image_size)
        self.sequences = [
            {
                "sample_id": str(r.sample_id),
                "sequence_id": str(r.sequence_id),
                "paths": json.loads(r.paths_json),
                "label": str(r.label),
                "label_idx": int(r.label_idx),
            }
            for r in self.df.itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Any:
        item = self.sequences[idx]
        frames = torch.zeros((self.seq_len, 1, self.image_size, self.image_size), dtype=torch.float32)
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        seq_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if self.is_train else 0
        seed_state = torch.random.get_rng_state() if self.is_train else None
        for t, rel_path in enumerate(item["paths"][: self.seq_len]):
            with Image.open(resolve_path(rel_path)) as im:
                gray = im.convert("L")
                if self.is_train:
                    torch.manual_seed(seq_seed)
                gray = self.transform(gray)
            frames[t] = pil_to_normalized_tensor(gray, self.mean, self.std)
            mask[t] = True
        if seed_state is not None:
            torch.random.set_rng_state(seed_state)

        y = item["label_idx"]
        if not self.return_meta:
            return (frames, mask), y
        return (frames, mask), y, {
            "sample_id": item["sample_id"],
            "image_path": item["paths"][0],
            "label": item["label"],
            "group_id": item["sequence_id"],
        }


def make_cv_exp_id(track: str, model_name: str, fold: int, seed: int = 42, tag: str = "") -> str:
    base = f"cv__{track}__{model_name}__fold{fold}__seed{seed}"
    return f"{base}__{tag}" if tag else base


def _build_fold_loaders(
    cfg: dict[str, Any],
    cv_df: pd.DataFrame,
    track: str,
    fold_idx: int,
    n_folds: int = 5,
    batch_size: int | None = None,
) -> dict[str, DataLoader]:
    """Create train_inner / val_inner / test_outer DataLoaders for zero-based fold_idx."""
    dataset_name, mode, _, _ = CV_TRACK_SPECS[track]
    ds_cfg = cfg[dataset_name]
    img_size = int(ds_cfg["image_size"])
    mean = [float(x) for x in ds_cfg["normalize_mean"]]
    std = [float(x) for x in ds_cfg["normalize_std"]]
    use_clahe = bool(ds_cfg.get("use_clahe", False)) if dataset_name == "fundus" else False
    num_workers = int(cfg["training"].get("num_workers", 0))
    eff_bs = int(batch_size or (cfg["experiments"]["oct_sequence"].get("batch_size", 4) if mode == "sequence" else cfg["training"]["batch_size"]))

    val_fold = (fold_idx + 1) % n_folds
    test_df = cv_df[cv_df["cv_fold"] == fold_idx].copy().reset_index(drop=True)
    val_df = cv_df[cv_df["cv_fold"] == val_fold].copy().reset_index(drop=True)
    train_df = cv_df[~cv_df["cv_fold"].isin([fold_idx, val_fold])].copy().reset_index(drop=True)

    loaders: dict[str, DataLoader] = {}
    if mode == "image":
        for name, sdf, is_tr, ret_meta in (
            ("train", train_df, True, False),
            ("val", val_df, False, False),
            ("test", test_df, False, True),
        ):
            ds = MedicalImageDataset(sdf, dataset_name, img_size, is_tr, use_clahe, mean, std, return_meta=ret_meta)
            loaders[name] = DataLoader(ds, batch_size=eff_bs, shuffle=is_tr, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    else:
        seq_len = int(ds_cfg.get("sequence_length", 8))
        for name, sdf, is_tr, ret_meta in (
            ("train", train_df, True, False),
            ("val", val_df, False, False),
            ("test", test_df, False, True),
        ):
            ds = PrechunkedOCTSequenceDataset(sdf, seq_len=seq_len, image_size=img_size, is_train=is_tr, mean=mean, std=std, return_meta=ret_meta)
            loaders[name] = DataLoader(ds, batch_size=eff_bs, shuffle=is_tr, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loaders


def run_single_cv_fold(
    cfg: dict[str, Any],
    cv_cfg: dict[str, Any],
    track: str,
    model_name: str,
    fold: int,
    epochs: int | None = None,
    overwrite: bool = False,
    resume: bool = True,
    tag: str = "",
) -> dict[str, Any]:
    """Run one 5-fold CV fold (1..5): train on train_inner, early stop on val_inner, test on test_outer."""
    cv_p = cv_cfg.get("cross_validation", cv_cfg)
    seed = int(cv_p["seed"])
    n_folds = int(cv_p["n_folds"])
    paths = cv_p["paths"]
    dataset_name, mode, class_names, in_channels = CV_TRACK_SPECS[track]

    cv_exp_id = make_cv_exp_id(track, model_name, fold, seed=seed, tag=tag)
    run_dir = resolve_path(paths["runs_dir"]) / cv_exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    eval_path = run_dir / "evaluation.json"
    pred_csv = resolve_path(paths["predictions_dir"]) / f"{cv_exp_id}_predictions.csv"

    if not overwrite and status_path.exists() and eval_path.exists() and pred_csv.exists():
        st = json.loads(status_path.read_text(encoding="utf-8"))
        if st.get("status") == "completed":
            return json.loads(eval_path.read_text(encoding="utf-8"))

    splits_path = resolve_path(paths["splits_dir"]) / f"{track}_cv_folds.csv"
    if not splits_path.exists():
        build_and_validate_cv_splits(cfg=cfg, cv_cfg=cv_cfg)
    cv_df = pd.read_csv(splits_path)

    log_dir = resolve_path(paths["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"cv.{cv_exp_id}", log_file=log_dir / f"{cv_exp_id}.log")

    seed_everything(seed + fold)
    device = get_device(prefer_gpu=True)
    tr_cfg = cfg["training"]
    m_cfg = cfg.get("models", {}).get(model_name, {})
    eff_epochs = int(epochs if epochs is not None else tr_cfg["epochs"])

    start_time = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps({"cv_exp_id": cv_exp_id, "status": "running", "start_time": start_time}, indent=2), encoding="utf-8")

    try:
        loaders = _build_fold_loaders(cfg, cv_df, track, fold_idx=fold - 1, n_folds=n_folds)
        model = build_model(
            model_name,
            num_classes=len(class_names),
            in_channels=in_channels,
            model_cfg=m_cfg,
            image_size=int(cfg[dataset_name]["image_size"]),
        )
        ckpt_dir = resolve_path(paths["checkpoints_dir"]) / cv_exp_id
        train_res = train_model(
            model=model,
            loaders=loaders,
            num_classes=len(class_names),
            device=device,
            checkpoint_dir=ckpt_dir,
            epochs=eff_epochs,
            lr=float(tr_cfg["learning_rate"]),
            weight_decay=float(tr_cfg["weight_decay"]),
            grad_clip_norm=float(tr_cfg["grad_clip_norm"]),
            early_stopping_patience=int(tr_cfg["early_stopping_patience"]),
            use_mixed_precision=bool(tr_cfg["use_mixed_precision"]),
            use_class_weights=bool(tr_cfg["use_class_weights"]),
            scheduler_type=str(tr_cfg["scheduler"]),
            autoencoder_pretrain_epochs=int(tr_cfg.get("autoencoder_pretrain_epochs", 5)),
            recon_loss_weight=float(m_cfg.get("recon_loss_weight", 0.1)),
            resume=resume and not overwrite,
            logger=logger,
        )

        # Evaluate untouched outer test fold using best val checkpoint
        best_ckpt = Path(train_res["best_checkpoint_path"])
        ckpt_size_mb = round(best_ckpt.stat().st_size / (1024 * 1024), 3) if best_ckpt.exists() else 0.0
        peak_gpu_mb = max((float(r.get("gpu_peak_memory_mb", 0.0)) for r in train_res.get("history", [])), default=0.0)

        eval_model = build_model(
            model_name,
            num_classes=len(class_names),
            in_channels=in_channels,
            model_cfg=m_cfg,
            image_size=int(cfg[dataset_name]["image_size"]),
        )
        eval_res = evaluate_model_on_test(
            model=eval_model,
            test_loader=loaders["test"],
            checkpoint_path=best_ckpt,
            class_names=class_names,
            device=device,
            predictions_dir=resolve_path(paths["predictions_dir"]),
            exp_id=cv_exp_id,
            use_amp=bool(tr_cfg["use_mixed_precision"]),
            low_conf_threshold=float(cfg["evaluation"]["low_conf_threshold"]),
            high_conf_error_threshold=float(cfg["evaluation"]["high_conf_error_threshold"]),
        )

        eval_res.update({
            "cv_exp_id": cv_exp_id,
            "track": track,
            "model": model_name,
            "category": m_cfg.get("category", "unknown"),
            "fold": fold,
            "val_fold": ((fold - 1 + 1) % n_folds) + 1,
            "seed": seed,
            "best_epoch": train_res["best_epoch"],
            "epochs_completed": train_res["epochs_completed"],
            "best_val_macro_f1": train_res["best_val_macro_f1"],
            "train_duration_sec": train_res["train_duration_sec"],
            "peak_gpu_mem_mb": round(peak_gpu_mb, 2),
            "ckpt_size_mb": ckpt_size_mb,
            "checkpoint_path": str(best_ckpt),
            "history_path": str(ckpt_dir / "history.csv"),
        })
        eval_path.write_text(json.dumps(eval_res, indent=2), encoding="utf-8")
        status_path.write_text(json.dumps({
            "cv_exp_id": cv_exp_id,
            "status": "completed",
            "start_time": start_time,
            "completion_time": datetime.now(timezone.utc).isoformat(),
            "train_duration_sec": train_res["train_duration_sec"],
            "best_epoch": train_res["best_epoch"],
            "best_val_macro_f1": train_res["best_val_macro_f1"],
            "test_macro_f1": eval_res["metrics"]["macro_f1"],
            "peak_gpu_mem_mb": round(peak_gpu_mb, 2),
            "ckpt_size_mb": ckpt_size_mb,
        }, indent=2), encoding="utf-8")
        last_ckpt = ckpt_dir / "last.pt"
        if last_ckpt.exists():
            last_ckpt.unlink()
        return eval_res
    except Exception as exc:
        logger.exception("CV run %s failed: %s", cv_exp_id, exc)
        status_path.write_text(json.dumps({
            "cv_exp_id": cv_exp_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "failure_time": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
        raise
    finally:
        clear_gpu_memory()


def collect_cv_runs_summary(cv_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Aggregate all completed CV fold status + evaluation metrics into a summary CSV."""
    cv_raw = cv_cfg or load_config("configs/cross_validation.yaml")
    cv_p = cv_raw.get("cross_validation", cv_raw)
    runs_dir = resolve_path(cv_p["paths"]["runs_dir"])
    out_dir = resolve_path(cv_p["paths"]["output_dir"])
    rows: list[dict[str, Any]] = []
    if runs_dir.exists():
        for eval_p in sorted(runs_dir.glob("cv__*/evaluation.json")):
            if "__smoke" in eval_p.parent.name:
                continue
            d = json.loads(eval_p.read_text(encoding="utf-8"))
            m = d.get("metrics", {})
            rows.append({
                "cv_exp_id": d["cv_exp_id"],
                "track": d["track"],
                "model": d["model"],
                "category": d.get("category", "unknown"),
                "fold": int(d["fold"]),
                "val_fold": int(d["val_fold"]),
                "seed": int(d["seed"]),
                "best_epoch": int(d["best_epoch"]),
                "epochs_completed": int(d["epochs_completed"]),
                "best_val_macro_f1": float(d["best_val_macro_f1"]),
                "test_accuracy": float(m.get("accuracy", 0.0)),
                "test_macro_f1": float(m.get("macro_f1", 0.0)),
                "test_weighted_f1": float(m.get("weighted_f1", 0.0)),
                "test_macro_precision": float(m.get("macro_precision", 0.0)),
                "test_macro_recall": float(m.get("macro_recall", 0.0)),
                "test_roc_auc_ovr": m.get("roc_auc_ovr"),
                "train_duration_sec": float(d["train_duration_sec"]),
                "inference_latency_ms": float(d.get("inference_latency_ms", 0.0)),
                "throughput_samples_per_sec": float(d.get("throughput_samples_per_sec", 0.0)),
                "total_params": int(d.get("total_params", 0)),
                "trainable_params": int(d.get("trainable_params", 0)),
                "peak_gpu_mem_mb": float(d.get("peak_gpu_mem_mb", 0.0)),
                "ckpt_size_mb": float(d.get("ckpt_size_mb", 0.0)),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["track", "model", "fold"]).reset_index(drop=True)
        df.to_csv(out_dir / "cv_runs_summary.csv", index=False)
    return df


def run_cv_suite(
    tracks: list[str] | None = None,
    models: list[str] | None = None,
    folds: list[int] | None = None,
    epochs: int | None = None,
    overwrite: bool = False,
    tag: str = "",
) -> pd.DataFrame:
    """Run all configured tracks x models x 5 folds sequentially with resume support."""
    cfg = load_all_configs()
    cv_cfg = load_config("configs/cross_validation.yaml")
    cv_p = cv_cfg["cross_validation"]
    build_and_validate_cv_splits(cfg=cfg, cv_cfg=cv_cfg)

    target_tracks = tracks or list(cv_p["tracks"])
    target_folds = folds or list(range(1, int(cv_p["n_folds"]) + 1))

    for trk in target_tracks:
        trk_models = cfg["experiments"][trk]["models"]
        selected_models = [m for m in trk_models if models is None or m in models]
        for model_name in selected_models:
            for k in target_folds:
                run_single_cv_fold(
                    cfg=cfg,
                    cv_cfg=cv_cfg,
                    track=trk,
                    model_name=model_name,
                    fold=k,
                    epochs=epochs,
                    overwrite=overwrite,
                    resume=True,
                    tag=tag,
                )
    return collect_cv_runs_summary(cv_cfg)

