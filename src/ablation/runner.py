"""Controlled Phase 7 ablation experiment runner reusing CV folds and existing models."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

from src.cross_validation.runner import CV_TRACK_SPECS, PrechunkedOCTSequenceDataset
from src.cross_validation.splits import generate_oct_sequence_cv_splits, validate_cv_dataframe
from src.data.datasets import MedicalImageDataset
from src.data.validation import OCT_3CLASSES
from src.evaluation.cv_metrics import aggregate_sequence_level_predictions, compute_extended_metrics
from src.evaluation.evaluator import evaluate_model_on_test
from src.models.model_factory import build_model
from src.training.trainer import train_model
from src.utils.config import load_all_configs, load_config, resolve_path
from src.utils.device import clear_gpu_memory, get_device
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import seed_everything


def _run_controlled_image_mode_fold(
    cfg: dict[str, Any],
    cv_df: pd.DataFrame,
    track: str,
    model_name: str,
    backbone_mode: str,
    fold: int,
    seed: int,
    out_root: Path,
    ckpt_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one controlled 5-fold experiment varying ONLY cnn backbone_mode ('frozen'|'partial')."""
    dataset_name, _, class_names, in_channels = CV_TRACK_SPECS[track]
    abl_id = f"abl__bbmode__{track}__{model_name}__{backbone_mode}__fold{fold}__seed{seed}"
    run_dir = out_root / "runs" / abl_id
    pred_dir = out_root / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    eval_path = run_dir / "evaluation.json"
    pred_csv = pred_dir / f"{abl_id}_predictions.csv"

    if not overwrite and eval_path.exists() and pred_csv.exists():
        return json.loads(eval_path.read_text(encoding="utf-8"))

    seed_everything(seed + fold)
    device = get_device(prefer_gpu=True)
    ds_cfg = cfg[dataset_name]
    img_size = int(ds_cfg["image_size"])
    mean = [float(x) for x in ds_cfg["normalize_mean"]]
    std = [float(x) for x in ds_cfg["normalize_std"]]
    use_clahe = bool(ds_cfg.get("use_clahe", False)) if dataset_name == "fundus" else False
    tr_cfg = cfg["training"]
    m_cfg = dict(cfg.get("models", {}).get(model_name, {}))
    m_cfg["backbone_mode"] = backbone_mode

    f_idx = fold - 1
    v_idx = (f_idx + 1) % 5
    test_df = cv_df[cv_df["cv_fold"] == f_idx].reset_index(drop=True)
    val_df = cv_df[cv_df["cv_fold"] == v_idx].reset_index(drop=True)
    train_df = cv_df[~cv_df["cv_fold"].isin([f_idx, v_idx])].reset_index(drop=True)

    loaders = {
        name: DataLoader(
            MedicalImageDataset(sdf, dataset_name, img_size, is_tr, use_clahe, mean, std, return_meta=ret_m),
            batch_size=int(tr_cfg["batch_size"]),
            shuffle=is_tr,
            num_workers=int(tr_cfg.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )
        for name, sdf, is_tr, ret_m in (("train", train_df, True, False), ("val", val_df, False, False), ("test", test_df, False, True))
    }

    logger = get_logger(f"abl.{abl_id}", log_file=out_root / "logs" / f"{abl_id}.log")
    model = build_model(model_name, num_classes=len(class_names), in_channels=in_channels, model_cfg=m_cfg, image_size=img_size)
    ckpt_dir = ckpt_root / abl_id
    tr_res = train_model(
        model=model,
        loaders=loaders,
        num_classes=len(class_names),
        device=device,
        checkpoint_dir=ckpt_dir,
        epochs=int(tr_cfg["epochs"]),
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
    eval_model = build_model(model_name, num_classes=len(class_names), in_channels=in_channels, model_cfg=m_cfg, image_size=img_size)
    ev_res = evaluate_model_on_test(
        model=eval_model,
        test_loader=loaders["test"],
        checkpoint_path=tr_res["best_checkpoint_path"],
        class_names=class_names,
        device=device,
        predictions_dir=pred_dir,
        exp_id=abl_id,
        use_amp=bool(tr_cfg["use_mixed_precision"]),
    )
    last_p = ckpt_dir / "last.pt"
    if last_p.exists():
        last_p.unlink()

    pdf = pd.read_csv(pred_csv)
    prob_cols = [f"prob_{c}" for c in class_names]
    ext_m = compute_extended_metrics(pdf["true_idx"].to_numpy(), pdf["pred_idx"].to_numpy(), pdf[prob_cols].to_numpy(), class_names)
    payload = {
        "ablation_group": "cnn_backbone_trainability",
        "abl_id": abl_id,
        "track": track,
        "model": model_name,
        "backbone_mode": backbone_mode,
        "fold": fold,
        "total_params": ev_res.get("parameters", {}).get("total_params", ev_res.get("total_params", 0)),
        "trainable_params": ev_res.get("parameters", {}).get("trainable_params", ev_res.get("trainable_params", 0)),
        "train_duration_sec": tr_res["train_duration_sec"],
        "best_epoch": tr_res["best_epoch"],
        "accuracy": ext_m["accuracy"],
        "balanced_accuracy": ext_m["balanced_accuracy"],
        "macro_f1": ext_m["macro_f1"],
        "weighted_f1": ext_m["weighted_f1"],
        "mcc": ext_m["mcc"],
    }
    eval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    clear_gpu_memory()
    return payload


def _run_controlled_seq_len_fold(
    cfg: dict[str, Any],
    seq_df: pd.DataFrame,
    model_name: str,
    seq_len: int,
    fold: int,
    seed: int,
    out_root: Path,
    ckpt_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one controlled 5-fold experiment varying ONLY OCT sequence window length."""
    abl_id = f"abl__seqlen{seq_len}__{model_name}__fold{fold}__seed{seed}"
    run_dir = out_root / "runs" / abl_id
    pred_dir = out_root / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    eval_path = run_dir / "evaluation.json"
    pred_csv = pred_dir / f"{abl_id}_predictions.csv"

    if not overwrite and eval_path.exists() and pred_csv.exists():
        return json.loads(eval_path.read_text(encoding="utf-8"))

    seed_everything(seed + fold)
    device = get_device(prefer_gpu=True)
    o_cfg = cfg["oct"]
    img_size = int(o_cfg["image_size"])
    mean = [float(x) for x in o_cfg["normalize_mean"]]
    std = [float(x) for x in o_cfg["normalize_std"]]
    tr_cfg = cfg["training"]
    m_cfg = dict(cfg.get("models", {}).get(model_name, {}))
    bs = int(cfg["experiments"]["oct_sequence"].get("batch_size", 4))

    f_idx = fold - 1
    v_idx = (f_idx + 1) % 5
    test_df = seq_df[seq_df["cv_fold"] == f_idx].reset_index(drop=True)
    val_df = seq_df[seq_df["cv_fold"] == v_idx].reset_index(drop=True)
    train_df = seq_df[~seq_df["cv_fold"].isin([f_idx, v_idx])].reset_index(drop=True)

    loaders = {
        name: DataLoader(
            PrechunkedOCTSequenceDataset(sdf, seq_len=seq_len, image_size=img_size, is_train=is_tr, mean=mean, std=std, return_meta=ret_m),
            batch_size=bs,
            shuffle=is_tr,
            num_workers=int(tr_cfg.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )
        for name, sdf, is_tr, ret_m in (("train", train_df, True, False), ("val", val_df, False, False), ("test", test_df, False, True))
    }

    logger = get_logger(f"abl.{abl_id}", log_file=out_root / "logs" / f"{abl_id}.log")
    model = build_model(model_name, num_classes=3, in_channels=1, model_cfg=m_cfg, image_size=img_size)
    ckpt_dir = ckpt_root / abl_id
    tr_res = train_model(
        model=model,
        loaders=loaders,
        num_classes=3,
        device=device,
        checkpoint_dir=ckpt_dir,
        epochs=int(tr_cfg["epochs"]),
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
    eval_model = build_model(model_name, num_classes=3, in_channels=1, model_cfg=m_cfg, image_size=img_size)
    ev_res = evaluate_model_on_test(
        model=eval_model,
        test_loader=loaders["test"],
        checkpoint_path=tr_res["best_checkpoint_path"],
        class_names=OCT_3CLASSES,
        device=device,
        predictions_dir=pred_dir,
        exp_id=abl_id,
        use_amp=bool(tr_cfg["use_mixed_precision"]),
    )
    last_p = ckpt_dir / "last.pt"
    if last_p.exists():
        last_p.unlink()

    pdf = pd.read_csv(pred_csv)
    prob_cols = [f"prob_{c}" for c in OCT_3CLASSES]
    ext_m = compute_extended_metrics(pdf["true_idx"].to_numpy(), pdf["pred_idx"].to_numpy(), pdf[prob_cols].to_numpy(), OCT_3CLASSES)
    payload = {
        "ablation_group": "oct_sequence_length",
        "abl_id": abl_id,
        "track": "oct_sequence",
        "model": model_name,
        "seq_len": seq_len,
        "fold": fold,
        "n_windows": int(len(pdf)),
        "total_params": ev_res.get("parameters", {}).get("total_params", ev_res.get("total_params", 0)),
        "trainable_params": ev_res.get("parameters", {}).get("trainable_params", ev_res.get("trainable_params", 0)),
        "train_duration_sec": tr_res["train_duration_sec"],
        "best_epoch": tr_res["best_epoch"],
        "accuracy": ext_m["accuracy"],
        "balanced_accuracy": ext_m["balanced_accuracy"],
        "macro_f1": ext_m["macro_f1"],
        "weighted_f1": ext_m["weighted_f1"],
        "mcc": ext_m["mcc"],
    }
    eval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    clear_gpu_memory()
    return payload


def run_ablation_suite(
    eval_dir: str | Path = "results/evaluation",
    output_dir: str | Path = "results/ablation_study",
    ckpt_dir: str | Path = "checkpoints_ablation",
    pub_fig_dir: str | Path = "results/publication_figures",
    skip_new_training: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compile controlled CV ablations (1-3 from CV baseline, 4-5 via controlled 5-fold runs)."""
    from src.ablation.plots import generate_all_ablation_figures

    cfg = load_all_configs()
    cv_raw = load_config("configs/cross_validation.yaml")
    cv_p = cv_raw["cross_validation"]
    seed = int(cv_p["seed"])
    out_root = resolve_path(output_dir)
    ckpt_root = resolve_path(ckpt_dir)
    fig_dir = out_root / "figures"
    pub_dir = resolve_path(pub_fig_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)

    cv_sum = pd.read_csv(resolve_path(eval_dir) / "cv_model_summary.csv")
    cv_folds = pd.read_csv(resolve_path(eval_dir) / "cv_fold_metrics.csv")

    # Ablation 1: CNN Backbone Comparison (on identical CV folds per track)
    abl1 = cv_sum[cv_sum["model"].isin(["resnet50", "densenet121", "efficientnet_b0"])].copy()
    abl1.to_csv(out_root / "abl01_cnn_backbone_comparison.csv", index=False)

    # Ablation 2: LSTM vs BiLSTM Comparison (identical patch/slice representation per track)
    abl2 = cv_sum[cv_sum["model"].isin(["lstm", "bilstm"])].copy()
    abl2.to_csv(out_root / "abl02_rnn_lstm_vs_bilstm.csv", index=False)

    # Ablation 3: Hybrid Architecture Comparison (CNN-only vs CNN+LSTM vs CNN+BiLSTM on comparable tracks)
    comp_tracks = ["fundus_3class", "oct_individual", "oct_severity"]
    abl3 = cv_sum[
        cv_sum["model"].isin(["resnet50", "resnet50_lstm", "resnet50_bilstm", "efficientnet_b0", "efficientnet_b0_bilstm"])
        & cv_sum["track"].isin(comp_tracks)
    ].copy()
    abl3.to_csv(out_root / "abl03_hybrid_cnn_rnn_comparison.csv", index=False)

    # Ablation 4: Controlled CNN Trainability ('frozen', 'partial', 'full') on best modality CNNs
    bb_specs = [("fundus_3class", "efficientnet_b0"), ("oct_individual", "densenet121")]
    bb_rows: list[dict[str, Any]] = []
    # Baseline 'full' rows directly from Phase 5 cv_fold_metrics
    for trk, m_name in bb_specs:
        sub_f = cv_folds[(cv_folds["track"] == trk) & (cv_folds["model"] == m_name)]
        for r in sub_f.itertuples(index=False):
            bb_rows.append({
                "ablation_group": "cnn_backbone_trainability",
                "track": trk,
                "model": m_name,
                "backbone_mode": "full",
                "fold": int(r.fold),
                "total_params": int(r.total_params),
                "trainable_params": int(r.total_params),
                "train_duration_sec": float(r.train_duration_sec),
                "best_epoch": int(r.best_epoch),
                "accuracy": float(r.accuracy),
                "balanced_accuracy": float(r.balanced_accuracy),
                "macro_f1": float(r.macro_f1),
                "weighted_f1": float(r.weighted_f1),
                "mcc": float(r.mcc),
            })

    if not skip_new_training:
        splits_root = resolve_path(cv_p["paths"]["splits_dir"])
        for trk, m_name in bb_specs:
            cv_df = pd.read_csv(splits_root / f"{trk}_cv_folds.csv")
            for mode in ("frozen", "partial"):
                for k in range(1, 6):
                    bb_rows.append(_run_controlled_image_mode_fold(cfg, cv_df, trk, m_name, mode, k, seed, out_root, ckpt_root, overwrite=overwrite))
    abl4_df = pd.DataFrame(bb_rows)
    abl4_df.to_csv(out_root / "abl04_backbone_trainability_folds.csv", index=False)

    # Ablation 5: Controlled OCT Sequence Length (4 vs 8 vs 12 slices with resnet50_bilstm)
    seq_rows: list[dict[str, Any]] = []
    seq_level_rows: list[dict[str, Any]] = []
    # Baseline seq_len=8 directly from Phase 5
    sub_s8 = cv_folds[(cv_folds["track"] == "oct_sequence") & (cv_folds["model"] == "resnet50_bilstm")]
    for r in sub_s8.itertuples(index=False):
        seq_rows.append({
            "ablation_group": "oct_sequence_length",
            "track": "oct_sequence",
            "model": "resnet50_bilstm",
            "seq_len": 8,
            "fold": int(r.fold),
            "n_windows": int(r.n_samples),
            "train_duration_sec": float(r.train_duration_sec),
            "best_epoch": int(r.best_epoch),
            "accuracy": float(r.accuracy),
            "balanced_accuracy": float(r.balanced_accuracy),
            "macro_f1": float(r.macro_f1),
            "weighted_f1": float(r.weighted_f1),
            "mcc": float(r.mcc),
        })
    # Include baseline L=8 sequence-level metrics from Phase 5
    seq_lvl_path = resolve_path(eval_dir) / "oct_sequence_level_summary.csv"
    if seq_lvl_path.exists():
        s8_lvl = pd.read_csv(seq_lvl_path)
        s8_row = s8_lvl[s8_lvl["model"] == "resnet50_bilstm"]
        if not s8_row.empty:
            r0 = s8_row.iloc[0]
            seq_level_rows.append({"seq_len": 8, "n_sequences": int(r0["n_sequences"]), "seq_accuracy": float(r0["seq_accuracy"]), "seq_macro_f1": float(r0["seq_macro_f1"]), "seq_mcc": float(r0["seq_mcc"])})

    if not skip_new_training:
        oct3_raw = pd.read_csv(resolve_path(cfg["paths"]["splits_dir"]) / "oct_3class_splits.csv")
        for seq_l in (4, 12):
            s_cv_df = generate_oct_sequence_cv_splits(oct3_raw, seq_len=seq_l, stride=seq_l, min_seq_len=2, n_folds=5)
            validate_cv_dataframe(s_cv_df, f"oct_sequence_len{seq_l}", OCT_3CLASSES, group_col="inferred_subject_prefix", n_folds=5)
            s_cv_df.to_csv(out_root / f"oct_sequence_len{seq_l}_cv_folds.csv", index=False)
            fold_pdfs = []
            for k in range(1, 6):
                res_k = _run_controlled_seq_len_fold(cfg, s_cv_df, "resnet50_bilstm", seq_l, k, seed, out_root, ckpt_root, overwrite=overwrite)
                seq_rows.append(res_k)
                fold_pdfs.append(pd.read_csv(out_root / "predictions" / f"{res_k['abl_id']}_predictions.csv"))
            oof_cat = pd.concat(fold_pdfs, ignore_index=True)
            seq_agg_df, seq_agg_m = aggregate_sequence_level_predictions(oof_cat, OCT_3CLASSES)
            seq_level_rows.append({
                "seq_len": seq_l,
                "n_sequences": int(len(seq_agg_df)),
                "seq_accuracy": float(seq_agg_m["accuracy"]),
                "seq_macro_f1": float(seq_agg_m["macro_f1"]),
                "seq_mcc": float(seq_agg_m["mcc"]),
            })

    abl5_df = pd.DataFrame(seq_rows)
    abl5_lvl_df = pd.DataFrame(seq_level_rows).sort_values("seq_len").reset_index(drop=True)
    abl5_df.to_csv(out_root / "abl05_oct_sequence_length_folds.csv", index=False)
    abl5_lvl_df.to_csv(out_root / "abl05_oct_sequence_length_seq_level.csv", index=False)

    figs = generate_all_ablation_figures(abl1, abl2, abl3, abl4_df, abl5_df, abl5_lvl_df, fig_dir, pub_dir)
    manifest = {
        "ablations_completed": ["cnn_backbone", "lstm_vs_bilstm", "hybrid_vs_cnn", "backbone_trainability", "oct_sequence_length"],
        "new_runs_executed": int(len(abl4_df[abl4_df["backbone_mode"] != "full"]) + len(abl5_df[abl5_df["seq_len"] != 8])),
        "figures_generated": [str(p) for p in figs],
        "output_dir": str(out_root),
    }
    (out_root / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

