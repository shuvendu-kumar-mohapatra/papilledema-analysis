"""Comprehensive pipeline for Condition B (Augmented) and Multi-Fold (10/15/20) CV evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
import torch

from src.cross_validation.runner import (
    CV_TRACK_SPECS,
    _build_fold_loaders,
    make_cv_exp_id,
)
from src.data.validation import FUNDUS_CLASSES, OCT_3CLASSES, OCT_SEVERITY_CLASSES
from src.evaluation.cv_metrics import compute_extended_metrics
from src.evaluation.evaluator import evaluate_model_on_test
from src.evaluation.table_generator import generate_all_documentation_tables
from src.models.model_factory import build_model
from scripts.run_augmented_experiments import build_augmented_fold_loaders, run_single_augmented_cv_fold
from src.training.trainer import train_model
from src.utils.config import load_all_configs, resolve_path
from src.utils.device import clear_gpu_memory, get_device
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import seed_everything


def run_generic_cv_fold(
    cfg: dict[str, Any],
    cv_df: pd.DataFrame,
    track: str,
    model_name: str,
    fold: int,
    n_folds: int,
    output_root: Path,
    use_augmented: bool = False,
    epochs: int | None = None,
    overwrite: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate a single fold for any fold count (5, 10, 15, 20), augmented or original."""
    dataset_name, mode, class_names, in_channels = CV_TRACK_SPECS[track]
    tag = f"aug_{n_folds}f" if use_augmented else f"orig_{n_folds}f"
    exp_id = f"cv__{track}__{model_name}__{tag}__fold{fold}__seed{seed}"
    run_dir = output_root / "runs" / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    eval_path = run_dir / "evaluation.json"
    pred_csv = pred_dir / f"{exp_id}_predictions.csv"
    ckpt_dir = resolve_path("checkpoints_cv") / exp_id

    if not overwrite and status_path.exists() and eval_path.exists() and pred_csv.exists():
        try:
            st = json.loads(status_path.read_text(encoding="utf-8"))
            if st.get("status") == "completed":
                return json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"cv.{exp_id}", log_file=log_dir / f"{exp_id}.log")

    seed_everything(seed + fold)
    device = get_device(prefer_gpu=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    tr_cfg = cfg["training"]
    m_cfg = cfg.get("models", {}).get(model_name, {})
    eff_epochs = int(epochs if epochs is not None else tr_cfg["epochs"])

    if use_augmented:
        loaders = build_augmented_fold_loaders(
            cfg, cv_df, track, fold_idx=fold - 1, n_folds=n_folds
        )
    else:
        loaders = _build_fold_loaders(
            cfg, cv_df, track, fold_idx=fold - 1, n_folds=n_folds
        )

    model = build_model(
        model_name,
        num_classes=len(class_names),
        in_channels=in_channels,
        image_size=int(cfg[dataset_name]["image_size"]),
        **m_cfg,
    )

    start_time = datetime.now(timezone.utc).isoformat()
    status_path.write_text(
        json.dumps({"exp_id": exp_id, "status": "running", "start_time": start_time}, indent=2),
        encoding="utf-8",
    )

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

    eval_model = build_model(
        model_name,
        num_classes=len(class_names),
        in_channels=in_channels,
        image_size=int(cfg[dataset_name]["image_size"]),
        **m_cfg,
    )
    ev_res = evaluate_model_on_test(
        model=eval_model,
        test_loader=loaders["test"],
        checkpoint_path=ckpt_dir / "best.pt",
        class_names=class_names,
        device=device,
        predictions_dir=pred_dir,
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
        "n_folds": n_folds,
        "fold": fold,
        "condition": "augmented" if use_augmented else "original",
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
    status_path.write_text(
        json.dumps(
            {"exp_id": exp_id, "status": "completed", "end_time": datetime.now(timezone.utc).isoformat()},
            indent=2,
        ),
        encoding="utf-8",
    )
    clear_gpu_memory()
    return payload


def compute_statistical_tests(orig_scores: list[float], aug_scores: list[float]) -> dict[str, float]:
    """Compute paired t-test, Wilcoxon signed-rank test, Cohen's d, and 95% CI."""
    a = np.array(orig_scores, dtype=np.float64)
    b = np.array(aug_scores, dtype=np.float64)
    diff = b - a
    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se_diff = std_diff / math.sqrt(n) if n > 0 else 0.0

    # 95% Confidence Interval for mean difference (t-distribution)
    if n > 1 and se_diff > 0:
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        ci_lower = mean_diff - tcrit * se_diff
        ci_upper = mean_diff + tcrit * se_diff
    else:
        ci_lower, ci_upper = mean_diff, mean_diff

    # Paired t-test
    if n > 1 and np.var(diff) > 1e-12:
        t_stat, t_pval = stats.ttest_rel(b, a)
    else:
        t_stat, t_pval = 0.0, 1.0

    # Wilcoxon signed-rank test
    try:
        if n >= 5 and np.any(diff != 0):
            w_stat, w_pval = stats.wilcoxon(b, a)
        else:
            w_stat, w_pval = 0.0, 1.0
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    # Cohen's d (paired)
    cohen_d = mean_diff / std_diff if std_diff > 1e-12 else 0.0

    return {
        "mean_diff": round(mean_diff, 4),
        "std_diff": round(std_diff, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "t_statistic": round(float(t_stat), 4),
        "t_pvalue": round(float(t_pval), 6),
        "wilcoxon_stat": round(float(w_stat), 4),
        "wilcoxon_pvalue": round(float(w_pval), 6),
        "cohens_d": round(cohen_d, 4),
    }


def run_condition_b_evaluations(
    cfg: dict[str, Any],
    tracks: list[str] | None = None,
    epochs: int | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run Condition B 5-fold CV for key architectures across all tracks."""
    out_dir = resolve_path("results/cross_validation/augmented_5fold")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tracks = tracks or ["fundus_3class", "oct_individual", "oct_severity", "oct_sequence"]

    key_models = {
        "fundus_3class": ["efficientnet_b0", "resnet50_bilstm", "bilstm", "mlp"],
        "oct_individual": ["densenet121", "resnet50_bilstm", "bilstm", "mlp"],
        "oct_severity": ["densenet121", "resnet50_bilstm", "bilstm", "mlp"],
        "oct_sequence": ["resnet50_bilstm", "bilstm"],
    }

    results = []
    for trk in all_tracks:
        models = key_models.get(trk, ["resnet50_bilstm"])
        cv_df = pd.read_csv(resolve_path(f"results/cross_validation/splits/{trk}_cv_folds.csv"))
        for m_name in models:
            for fold in range(1, 6):
                print(f"[Condition B] Track={trk}, Model={m_name}, Fold={fold}/5")
                res = run_generic_cv_fold(
                    cfg=cfg,
                    cv_df=cv_df,
                    track=trk,
                    model_name=m_name,
                    fold=fold,
                    n_folds=5,
                    output_root=out_dir,
                    use_augmented=True,
                    epochs=epochs,
                    overwrite=overwrite,
                )
                results.append(res)

    # Save summary and compare with Condition A
    runs_df = pd.DataFrame(results)
    runs_df.to_csv(out_dir / "cv_augmented_runs_summary.csv", index=False)
    return runs_df


ALL_TRACK_MODELS: dict[str, list[str]] = {
    "fundus_3class": [
        "resnet50",
        "densenet121",
        "efficientnet_b0",
        "lstm",
        "bilstm",
        "resnet50_lstm",
        "resnet50_bilstm",
        "efficientnet_b0_bilstm",
        "mlp",
        "autoencoder_classifier",
    ],
    "oct_individual": [
        "resnet50",
        "densenet121",
        "efficientnet_b0",
        "lstm",
        "bilstm",
        "resnet50_lstm",
        "resnet50_bilstm",
        "efficientnet_b0_bilstm",
        "mlp",
        "autoencoder_classifier",
    ],
    "oct_sequence": [
        "lstm",
        "bilstm",
        "resnet50_lstm",
        "resnet50_bilstm",
        "efficientnet_b0_bilstm",
    ],
    "oct_severity": [
        "resnet50",
        "densenet121",
        "efficientnet_b0",
        "lstm",
        "bilstm",
        "resnet50_lstm",
        "resnet50_bilstm",
        "efficientnet_b0_bilstm",
        "mlp",
        "autoencoder_classifier",
    ],
}


def update_multifold_progress_report(
    n_folds_list: list[int] = [10, 15],
    progress_csv_path: Path | None = None,
) -> pd.DataFrame:
    """Generate or update the master progress tracking table for all 35 models."""
    if progress_csv_path is None:
        progress_csv_path = resolve_path("results/cross_validation/multifold_execution_progress.csv")
    progress_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for n_folds in n_folds_list:
        summary_p = resolve_path(f"results/cross_validation/{n_folds}_fold/cv_{n_folds}fold_runs_summary.csv")
        completed_counts: dict[tuple[str, str], int] = {}
        if summary_p.exists():
            try:
                sdf = pd.read_csv(summary_p)
                for (trk, m_name), grp in sdf.groupby(["track", "model"]):
                    completed_counts[(trk, m_name)] = len(grp)
            except Exception:
                pass

        for trk, models in ALL_TRACK_MODELS.items():
            for m_name in models:
                done = completed_counts.get((trk, m_name), 0)
                rem = max(0, n_folds - done)
                if done == n_folds:
                    status = "Completed"
                elif done > 0:
                    status = "In Progress"
                else:
                    status = "Pending"

                rows.append({
                    "Track": trk,
                    "Model": m_name,
                    "CV Regime": f"{n_folds}-Fold CV",
                    "Total Folds": n_folds,
                    "Completed": done,
                    "Remaining": rem,
                    "Status": status,
                    "Last Updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                })

    df = pd.DataFrame(rows)
    df.to_csv(progress_csv_path, index=False)
    return df


def run_multifold_evaluations(
    cfg: dict[str, Any],
    n_folds_list: list[int] = [10],
    epochs: int | None = None,
    overwrite: bool = False,
) -> dict[int, pd.DataFrame]:
    """Execute complete 10-fold CV across ALL 35 implemented model configurations."""
    summaries = {}
    master_prog_p = resolve_path("results/cross_validation/multifold_execution_progress.csv")

    for n_folds in n_folds_list:
        out_dir = resolve_path(f"results/cross_validation/{n_folds}_fold")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = out_dir / f"cv_{n_folds}fold_runs_summary.csv"

        existing_runs: dict[tuple[str, str, int], dict[str, Any]] = {}
        if summary_csv.exists() and not overwrite:
            try:
                prev_df = pd.read_csv(summary_csv)
                for _, row in prev_df.iterrows():
                    key = (str(row["track"]), str(row["model"]), int(row["fold"]))
                    existing_runs[key] = row.to_dict()
                print(f"[{n_folds}-Fold CV] Loaded {len(existing_runs)} existing fold records from {summary_csv.name}.")
            except Exception as e:
                print(f"[{n_folds}-Fold CV] Warning loading existing summary: {e}")

        results_dict: dict[tuple[str, str, int], dict[str, Any]] = dict(existing_runs)
        total_track_models = sum(len(m) for m in ALL_TRACK_MODELS.values())
        print(f"\n{'='*70}\nSTARTING {n_folds}-FOLD CROSS-VALIDATION SUITE ({total_track_models} models, {n_folds} folds each)\n{'='*70}")

        for trk, models in ALL_TRACK_MODELS.items():
            splits_p = resolve_path(f"results/cross_validation/{n_folds}_fold/splits/{trk}_cv_folds.csv")
            if not splits_p.exists():
                print(f"ERROR: Splits file not found for {trk} {n_folds}-fold at {splits_p}. Skipping track.")
                continue

            cv_df = pd.read_csv(splits_p)

            for m_name in models:
                model_completed_folds = sum(1 for f in range(1, n_folds + 1) if (trk, m_name, f) in results_dict)
                if model_completed_folds == n_folds and not overwrite:
                    print(f"[{n_folds}-Fold CV] Track={trk}, Model={m_name}: Already fully complete ({n_folds}/{n_folds} folds). Skipping.")
                    continue

                print(f"\n--- [{n_folds}-Fold CV] Track={trk}, Model={m_name} (Completed {model_completed_folds}/{n_folds}) ---")

                for fold in range(1, n_folds + 1):
                    key = (trk, m_name, fold)
                    if key in results_dict and not overwrite:
                        continue

                    print(f"[{n_folds}-Fold CV] Executing: Track={trk}, Model={m_name}, Fold={fold}/{n_folds}")
                    try:
                        res = run_generic_cv_fold(
                            cfg=cfg,
                            cv_df=cv_df,
                            track=trk,
                            model_name=m_name,
                            fold=fold,
                            n_folds=n_folds,
                            output_root=out_dir,
                            use_augmented=False,
                            epochs=epochs,
                            overwrite=overwrite,
                        )
                        results_dict[key] = res
                        f1_val = res.get("macro_f1", 0.0)
                        acc_val = res.get("accuracy", 0.0)
                        print(f"[{n_folds}-Fold CV] Finished Fold {fold}/{n_folds}: Macro-F1={f1_val:.4f}, Accuracy={acc_val:.4f}")
                    except Exception as exc:
                        print(f"[ERROR] [{n_folds}-Fold CV] Failed: Track={trk}, Model={m_name}, Fold={fold}/{n_folds}: {exc}")
                        traceback.print_exc()
                        clear_gpu_memory()
                        gc.collect()
                        continue

                    # Persist summary immediately after every completed fold
                    df_current = pd.DataFrame(list(results_dict.values()))
                    df_current.to_csv(summary_csv, index=False)
                    update_multifold_progress_report(n_folds_list, master_prog_p)

                    clear_gpu_memory()
                    gc.collect()

        final_df = pd.DataFrame(list(results_dict.values()))
        final_df.to_csv(summary_csv, index=False)
        summaries[n_folds] = final_df
        print(f"\n{'='*70}\nCOMPLETED {n_folds}-FOLD CV SUITE: {len(final_df)} total records saved to {summary_csv}\n{'='*70}\n")

    update_multifold_progress_report(n_folds_list, master_prog_p)
    return summaries


def compile_all_comparisons_and_tables() -> None:
    """Aggregate all runs, compute statistical significance and effect sizes, and generate documentation tables."""
    doc_dir = resolve_path("results/tables/documentation_tables")
    doc_dir.mkdir(parents=True, exist_ok=True)
    aug_dir = resolve_path("results/cross_validation/augmented_5fold")
    base_dir = resolve_path("results/evaluation")

    # 1. Condition A vs Condition B Comparison
    aug_runs_p = aug_dir / "cv_augmented_runs_summary.csv"
    base_runs_p = resolve_path("results/cross_validation/cv_runs_summary.csv")

    if aug_runs_p.exists() and base_runs_p.exists():
        aug_runs = pd.read_csv(aug_runs_p)
        base_runs = pd.read_csv(base_runs_p)

        # Merge on track, model, fold
        comp_rows = []
        stat_rows = []

        for (trk, m_name), grp_aug in aug_runs.groupby(["track", "model"]):
            grp_base = base_runs[(base_runs["track"] == trk) & (base_runs["model"] == m_name)]
            if len(grp_base) == 0:
                continue

            merged = pd.merge(grp_base, grp_aug, on="fold", suffixes=("_orig", "_aug"))
            if len(merged) < 3:
                continue

            # Statistical tests
            f1_orig = (merged["macro_f1_orig"] if "macro_f1_orig" in merged.columns else merged["test_macro_f1"]).tolist()
            f1_aug = (merged["macro_f1_aug"] if "macro_f1_aug" in merged.columns else merged["macro_f1"]).tolist()
            acc_orig = (merged["accuracy_orig"] if "accuracy_orig" in merged.columns else merged["test_accuracy"]).tolist()
            acc_aug = (merged["accuracy_aug"] if "accuracy_aug" in merged.columns else merged["accuracy"]).tolist()

            f1_stats = compute_statistical_tests(f1_orig, f1_aug)
            acc_stats = compute_statistical_tests(acc_orig, acc_aug)

            stat_rows.append({
                "track": trk,
                "model": m_name,
                "metric": "Macro-F1",
                "condition_a_mean": round(float(np.mean(f1_orig)), 4),
                "condition_a_std": round(float(np.std(f1_orig, ddof=1)), 4),
                "condition_b_mean": round(float(np.mean(f1_aug)), 4),
                "condition_b_std": round(float(np.std(f1_aug, ddof=1)), 4),
                **f1_stats,
            })

            comp_rows.append({
                "Track": trk,
                "Model": m_name,
                "Condition A Macro-F1": f"{np.mean(f1_orig):.4f} ± {np.std(f1_orig, ddof=1):.4f}",
                "Condition B Macro-F1": f"{np.mean(f1_aug):.4f} ± {np.std(f1_aug, ddof=1):.4f}",
                "Delta Macro-F1": round(float(np.mean(f1_aug) - np.mean(f1_orig)), 4),
                "Condition A Accuracy": f"{np.mean(acc_orig):.4f} ± {np.std(acc_orig, ddof=1):.4f}",
                "Condition B Accuracy": f"{np.mean(acc_aug):.4f} ± {np.std(acc_aug, ddof=1):.4f}",
                "Delta Accuracy": round(float(np.mean(acc_aug) - np.mean(acc_orig)), 4),
                "p-value (paired-t)": f1_stats["t_pvalue"],
                "Cohen's d": f1_stats["cohens_d"],
            })

        pd.DataFrame(stat_rows).to_csv(aug_dir / "statistical_tests_a_vs_b.csv", index=False)
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(aug_dir / "condition_a_vs_condition_b_comparison.csv", index=False)
        comp_df.to_csv(doc_dir / "A05_condition_a_vs_condition_b_summary.csv", index=False)
        print("Condition A vs B comparison and statistical analysis saved.")

    # 2. Multi-Fold Comparisons (Table 13 / D01-D04)
    multifold_rows = []
    for k in [5, 10, 15]:
        if k == 5:
            sum_p = base_dir / "cv_model_summary.csv"
            if sum_p.exists():
                sdf = pd.read_csv(sum_p)
                for trk, models in ALL_TRACK_MODELS.items():
                    for m_name in models:
                        sub = sdf[(sdf["track"] == trk) & (sdf["model"] == m_name)]
                        if not sub.empty:
                            r = sub.iloc[0]
                            multifold_rows.append({
                                "Configuration": "5-Fold CV",
                                "Track": trk,
                                "Model": m_name,
                                "Macro-F1": f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}",
                                "Accuracy": f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
                                "Balanced Accuracy": f"{r.get('balanced_accuracy_mean', 0.0):.4f} ± {r.get('balanced_accuracy_std', 0.0):.4f}",
                                "MCC": f"{r.get('mcc_mean', 0.0):.4f} ± {r.get('mcc_std', 0.0):.4f}",
                                "Fold Variance (F1 SD)": round(r["macro_f1_std"], 4),
                                "Status": "Feasible & Completed",
                            })
        else:
            k_runs_p = resolve_path(f"results/cross_validation/{k}_fold/cv_{k}fold_runs_summary.csv")
            if k_runs_p.exists():
                k_df = pd.read_csv(k_runs_p)
                for (trk, m_name), grp in k_df.groupby(["track", "model"]):
                    multifold_rows.append({
                        "Configuration": f"{k}-Fold CV",
                        "Track": trk,
                        "Model": m_name,
                        "Macro-F1": f"{grp['macro_f1'].mean():.4f} ± {grp['macro_f1'].std(ddof=1):.4f}",
                        "Accuracy": f"{grp['accuracy'].mean():.4f} ± {grp['accuracy'].std(ddof=1):.4f}",
                        "Balanced Accuracy": f"{grp['balanced_accuracy'].mean():.4f} ± {grp['balanced_accuracy'].std(ddof=1):.4f}",
                        "MCC": f"{grp['mcc'].mean():.4f} ± {grp['mcc'].std(ddof=1):.4f}",
                        "Fold Variance (F1 SD)": round(float(grp["macro_f1"].std(ddof=1)), 4),
                        "Status": "Feasible & Completed",
                    })

    mf_df = pd.DataFrame(multifold_rows)
    mf_df.to_csv(doc_dir / "D05_cross_validation_multifold_comparison.csv", index=False)

    # 3. Regenerate all documentation tables
    generate_all_documentation_tables()
    print("All documentation tables generated and updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["condition_b", "multifold", "compile", "all"], default="multifold")
    parser.add_argument("--folds", type=int, nargs="+", default=[10, 15])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_all_configs()
    if args.action in ["condition_b", "all"]:
        run_condition_b_evaluations(cfg, epochs=args.epochs, overwrite=args.overwrite)
    if args.action in ["multifold", "all"]:
        run_multifold_evaluations(cfg, n_folds_list=args.folds, epochs=args.epochs, overwrite=args.overwrite)
    compile_all_comparisons_and_tables()
