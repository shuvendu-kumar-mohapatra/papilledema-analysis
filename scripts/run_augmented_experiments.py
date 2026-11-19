"""Pipeline runner for Condition B: Cross-validation with offline augmented training folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.cross_validation.runner import CV_TRACK_SPECS, PrechunkedOCTSequenceDataset, make_cv_exp_id
from src.data.datasets import MedicalImageDataset
from src.data.validation import FUNDUS_CLASSES, OCT_3CLASSES, OCT_SEVERITY_CLASSES
from src.evaluation.cv_metrics import compute_extended_metrics
from src.evaluation.evaluator import evaluate_model_on_test
from src.models.model_factory import build_model
from src.training.trainer import train_model
from src.utils.config import load_all_configs, resolve_path
from src.utils.device import clear_gpu_memory, get_device
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import seed_everything


def build_augmented_fold_loaders(
    cfg: dict[str, Any],
    cv_df: pd.DataFrame,
    track: str,
    fold_idx: int,
    n_folds: int = 5,
    aug_manifest_path: str | Path = "results/augmentation/manifests/augmentation_manifest.csv",
    batch_size: int | None = None,
) -> dict[str, DataLoader]:
    """Build loaders where train partition includes augmented descendants, while val/test are strictly original."""
    dataset_name, mode, class_names, in_channels = CV_TRACK_SPECS[track]
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

    # Augment training partition only (never val or test)
    if mode == "image":
        man_path = resolve_path(aug_manifest_path)
        if man_path.exists():
            man_df = pd.read_csv(man_path)
            train_orig_ids = set(train_df["sample_id"].astype(str))
            syn_matches = man_df[
                (man_df["dataset"] == track) &
                (man_df["is_synthetic"]) &
                (man_df["original_image_id"].astype(str).isin(train_orig_ids))
            ]
            if not syn_matches.empty:
                label_to_idx = {c: i for i, c in enumerate(class_names)}
                syn_df = pd.DataFrame({
                    "sample_id": syn_matches["image_id"],
                    "image_path": syn_matches["output_path"],
                    "label": syn_matches["class_label"],
                    "label_idx": syn_matches["class_label"].map(label_to_idx).astype(int),
                    "is_synthetic": True,
                })
                train_df = pd.concat([train_df, syn_df], ignore_index=True)

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


def run_single_augmented_cv_fold(
    cfg: dict[str, Any],
    track: str,
    model_name: str,
    fold: int,
    n_folds: int = 5,
    splits_path: str | Path | None = None,
    output_root: str | Path = "results/cross_validation/augmented_5fold",
    epochs: int | None = None,
    overwrite: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Run one CV fold with offline augmented training set and original validation/test sets."""
    dataset_name, mode, class_names, in_channels = CV_TRACK_SPECS[track]
    out_dir = resolve_path(output_root)
    runs_dir = out_dir / "runs"
    pred_dir = out_dir / "predictions"
    ckpt_dir = resolve_path("checkpoints_cv") / f"aug__{track}__{model_name}__fold{fold}__seed{seed}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    exp_id = f"cv_aug__{track}__{model_name}__fold{fold}__seed{seed}"
    run_dir = runs_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    eval_path = run_dir / "evaluation.json"
    pred_csv = pred_dir / f"{exp_id}_predictions.csv"

    if not overwrite and status_path.exists() and eval_path.exists() and pred_csv.exists():
        st = json.loads(status_path.read_text(encoding="utf-8"))
        if st.get("status") == "completed":
            return json.loads(eval_path.read_text(encoding="utf-8"))

    s_path = resolve_path(splits_path or f"results/cross_validation/splits/{track}_cv_folds.csv")
    cv_df = pd.read_csv(s_path)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"aug.{exp_id}", log_file=log_dir / f"{exp_id}.log")

    seed_everything(seed + fold)
    device = get_device(prefer_gpu=True)
    tr_cfg = cfg["training"]
    m_cfg = cfg.get("models", {}).get(model_name, {})
    eff_epochs = int(epochs if epochs is not None else tr_cfg["epochs"])

    loaders = build_augmented_fold_loaders(cfg, cv_df, track, fold_idx=fold - 1, n_folds=n_folds)
    model = build_model(model_name, num_classes=len(class_names), in_channels=in_channels, image_size=int(cfg[dataset_name]["image_size"]), **m_cfg)

    start_time = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps({"exp_id": exp_id, "status": "running", "start_time": start_time}, indent=2), encoding="utf-8")

    tr_res = train_model(
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
        resume=not overwrite,
        logger=logger,
    )

    eval_model = build_model(model_name, num_classes=len(class_names), in_channels=in_channels, image_size=int(cfg[dataset_name]["image_size"]), **m_cfg)
    ev_res = evaluate_model_on_test(
        model=eval_model,
        test_loader=loaders["test"],
        num_classes=len(class_names),
        device=device,
        checkpoint_dir=ckpt_dir,
        class_names=class_names,
        output_predictions_csv=pred_csv,
        exp_id=exp_id,
        use_amp=bool(tr_cfg["use_mixed_precision"]),
    )

    last_pt = ckpt_dir / "last.pt"
    if last_pt.exists():
        last_pt.unlink()

    pdf = pd.read_csv(pred_csv)
    prob_cols = [f"prob_{c}" for c in class_names]
    ext_m = compute_extended_metrics(
        pdf["true_idx"].to_numpy(dtype=np.int64),
        pdf["pred_idx"].to_numpy(dtype=np.int64),
        pdf[prob_cols].to_numpy(dtype=np.float64),
        class_names,
        is_severity=(track == "oct_severity"),
    )

    payload = {
        "exp_id": exp_id,
        "track": track,
        "model": model_name,
        "fold": fold,
        "condition": "augmented",
        "train_samples": len(loaders["train"].dataset),
        "val_samples": len(loaders["val"].dataset),
        "test_samples": len(loaders["test"].dataset),
        "train_duration_sec": tr_res["train_duration_sec"],
        "best_epoch": tr_res["best_epoch"],
        "inference_latency_ms": ev_res.get("inference_latency_ms", 0.0),
        "total_params": ev_res.get("parameters", {}).get("total_params", 0),
        "peak_gpu_mem_mb": tr_res.get("peak_gpu_mem_mb", 0.0),
        **ext_m,
    }
    eval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    status_path.write_text(json.dumps({"exp_id": exp_id, "status": "completed", "end_time": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    clear_gpu_memory()
    return payload


def compile_augmented_cv_summaries(
    output_root: str | Path = "results/cross_validation/augmented_5fold",
    baseline_eval_dir: str | Path = "results/evaluation",
) -> pd.DataFrame:
    """Compile fold metrics and out-of-fold summaries for Condition B (augmented), and compare against Condition A."""
    out_dir = resolve_path(output_root)
    runs_dir = out_dir / "runs"
    eval_files = list(runs_dir.glob("*/evaluation.json"))
    rows = []
    for ef in eval_files:
        try:
            d = json.loads(ef.read_text(encoding="utf-8"))
            rows.append(d)
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if df.empty:
        print("No completed augmented runs found.")
        return df

    df.to_csv(out_dir / "cv_augmented_runs_summary.csv", index=False)

    # Group by track and model to calculate mean and std across folds
    metrics_to_agg = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc", "cohen_kappa"]
    agg_funcs = {m: ["mean", "std"] for m in metrics_to_agg if m in df.columns}
    agg_funcs["train_duration_sec"] = ["mean"]
    agg_funcs["peak_gpu_mem_mb"] = ["max"]

    summary_df = df.groupby(["track", "model"]).agg(agg_funcs)
    summary_df.columns = [f"{col}_{stat}" for col, stat in summary_df.columns]
    summary_df = summary_df.reset_index()
    summary_df.to_csv(out_dir / "cv_augmented_model_summary.csv", index=False)

    # Compare with Baseline Condition A
    base_dir = resolve_path(baseline_eval_dir)
    if (base_dir / "cv_model_summary.csv").exists():
        base_df = pd.read_csv(base_dir / "cv_model_summary.csv")
        merged = pd.merge(
            base_df[["track", "model", "macro_f1_mean", "accuracy_mean", "balanced_accuracy_mean", "mcc_mean"]],
            summary_df[["track", "model", "macro_f1_mean", "accuracy_mean", "balanced_accuracy_mean", "mcc_mean"]],
            on=["track", "model"],
            suffixes=("_orig", "_aug"),
            how="inner"
        )
        merged["delta_macro_f1"] = (merged["macro_f1_mean_aug"] - merged["macro_f1_mean_orig"]).round(4)
        merged["delta_accuracy"] = (merged["accuracy_mean_aug"] - merged["accuracy_mean_orig"]).round(4)
        merged["delta_mcc"] = (merged["mcc_mean_aug"] - merged["mcc_mean_orig"]).round(4)
        merged.to_csv(out_dir / "condition_a_vs_condition_b_comparison.csv", index=False)
        print(f"Condition A vs Condition B comparison saved ({len(merged)} models).")

    return summary_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Condition B (offline augmented) CV training.")
    parser.add_argument("--tracks", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default="results/cross_validation/augmented_5fold")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_all_configs()
    target_tracks = args.tracks or ["fundus_3class", "oct_individual", "oct_severity"]
    target_folds = args.folds or [1, 2, 3, 4, 5]

    for trk in target_tracks:
        trk_models = cfg["experiments"][trk]["models"]
        selected_models = [m for m in trk_models if args.models is None or m in args.models]
        for m_name in selected_models:
            for k in target_folds:
                print(f"\n>>> Running Augmented CV: Track={trk}, Model={m_name}, Fold={k} <<<")
                res = run_single_augmented_cv_fold(
                    cfg=cfg,
                    track=trk,
                    model_name=m_name,
                    fold=k,
                    output_root=args.output_dir,
                    epochs=args.epochs,
                    overwrite=args.overwrite,
                )
                print(f"Completed {res['exp_id']}: Macro-F1={res.get('macro_f1', 0.0):.4f}")

    compile_augmented_cv_summaries(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

