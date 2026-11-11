"""Sequential experiment runner coordinating data, model, training, eval, and logging."""

from __future__ import annotations

from typing import Any

from src.data.datasets import create_dataloaders
from src.data.validation import FUNDUS_CLASSES, OCT_3CLASSES, OCT_SEVERITY_CLASSES
from src.evaluation.evaluator import evaluate_model_on_test
from src.experiments.manager import ExperimentManager, make_experiment_id
from src.models.model_factory import build_model
from src.training.trainer import train_model
from src.utils.config import resolve_path
from src.utils.device import clear_gpu_memory, get_device
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import seed_everything

TRACK_TO_TASK = {
    "fundus_3class": ("fundus", "fundus_3class", "image", FUNDUS_CLASSES, 3),
    "oct_individual": ("oct", "oct_3class", "image", OCT_3CLASSES, 1),
    "oct_sequence": ("oct", "oct_3class", "sequence", OCT_3CLASSES, 1),
    "oct_severity": ("oct", "oct_severity", "image", OCT_SEVERITY_CLASSES, 1),
}


def run_single_experiment(
    cfg: dict[str, Any],
    track: str,
    model_name: str,
    seed: int = 42,
    epochs: int | None = None,
    batch_size: int | None = None,
    image_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    use_mixed_precision: bool | None = None,
    early_stopping_patience: int | None = None,
    use_clahe: bool | None = None,
    backbone_mode: str | None = None,
    overwrite: bool = False,
    resume: bool = True,
    is_smoke: bool = False,
    tag: str = "",
) -> dict[str, Any]:
    """Run one complete reproducible train -> best-ckpt test eval cycle sequentially."""
    if track not in TRACK_TO_TASK:
        raise ValueError(f"Unknown track '{track}'. Valid: {list(TRACK_TO_TASK)}")
    dataset_name, task, mode, class_names, in_channels = TRACK_TO_TASK[track]

    exp_id = make_experiment_id(track, model_name, seed, tag=tag)
    mgr = ExperimentManager(cfg)
    if not overwrite and mgr.is_completed(exp_id):
        return mgr.load_evaluation(exp_id)

    log_file = resolve_path(cfg["paths"]["logs_dir"]) / f"{exp_id}.log"
    logger = get_logger(f"exp.{exp_id}", log_file=log_file)
    seed_everything(seed)
    device = get_device(prefer_gpu=True)

    tr_cfg = cfg["training"]
    m_cfg = cfg.get("models", {}).get(model_name, {})
    eff_epochs = int(epochs if epochs is not None else tr_cfg["epochs"])
    eff_img_size = int(image_size if image_size is not None else cfg[dataset_name]["image_size"])
    eff_lr = float(lr if lr is not None else tr_cfg["learning_rate"])
    eff_wd = float(weight_decay if weight_decay is not None else tr_cfg["weight_decay"])
    eff_amp = bool(use_mixed_precision if use_mixed_precision is not None else tr_cfg["use_mixed_precision"])
    eff_patience = int(early_stopping_patience if early_stopping_patience is not None else tr_cfg["early_stopping_patience"])
    default_bs = cfg["experiments"]["oct_sequence"].get("batch_size", 4) if mode == "sequence" else tr_cfg["batch_size"]
    eff_bs = int(batch_size if batch_size is not None else default_bs)
    eff_clahe = bool(cfg[dataset_name].get("use_clahe", False) if use_clahe is None else use_clahe)
    eff_backbone_mode = str(backbone_mode if backbone_mode is not None else m_cfg.get("backbone_mode", "full"))

    config_payload = {
        "exp_id": exp_id,
        "track": track,
        "dataset": dataset_name,
        "modality": dataset_name,
        "task": task,
        "mode": mode,
        "model": model_name,
        "category": m_cfg.get("category", "unknown"),
        "seed": seed,
        "epochs": eff_epochs,
        "batch_size": eff_bs,
        "image_size": eff_img_size,
        "use_clahe": eff_clahe,
        "backbone_mode": eff_backbone_mode,
        "lr": eff_lr,
        "weight_decay": eff_wd,
        "use_mixed_precision": eff_amp,
        "early_stopping_patience": eff_patience,
        "is_smoke": is_smoke,
        "device": str(device),
    }
    mgr.register_start(exp_id, config_payload)
    logger.info("Starting %s on device=%s | epochs=%d", exp_id, device, eff_epochs)

    try:
        loaders, _ = create_dataloaders(
            cfg,
            task=task,
            mode=mode,
            batch_size=eff_bs,
            image_size=eff_img_size,
            use_clahe=use_clahe,
            return_meta_eval=True,
        )
        model_overrides = {"backbone_mode": backbone_mode} if backbone_mode else {}
        model = build_model(
            model_name,
            num_classes=len(class_names),
            in_channels=in_channels,
            image_size=eff_img_size,
            cfg=cfg,
            **model_overrides,
        )
        ae_epochs = 1 if is_smoke else int(tr_cfg.get("autoencoder_pretrain_epochs", 5))
        train_res = train_model(
            model=model,
            loaders=loaders,
            num_classes=len(class_names),
            checkpoint_dir=mgr.ckpt_dir(exp_id),
            device=device,
            epochs=eff_epochs,
            lr=eff_lr,
            weight_decay=eff_wd,
            grad_clip_norm=float(tr_cfg.get("grad_clip_norm", 1.0)),
            early_stopping_patience=eff_patience,
            use_mixed_precision=eff_amp,
            use_class_weights=bool(tr_cfg.get("use_class_weights", True)),
            scheduler_type=str(tr_cfg.get("scheduler", "cosine")),
            autoencoder_pretrain_epochs=ae_epochs,
            recon_loss_weight=float(m_cfg.get("recon_loss_weight", 0.1)),
            resume=resume and not overwrite,
            logger=logger,
        )
        eval_res = evaluate_model_on_test(
            model=model,
            test_loader=loaders["test"],
            checkpoint_path=train_res["best_checkpoint_path"],
            class_names=class_names,
            device=device,
            predictions_dir=resolve_path(cfg["paths"]["results_dir"]) / "predictions",
            exp_id=exp_id,
            use_amp=eff_amp,
            low_conf_threshold=float(cfg["evaluation"]["low_conf_threshold"]),
            high_conf_error_threshold=float(cfg["evaluation"]["high_conf_error_threshold"]),
        )
        mgr.mark_completed(exp_id, train_res, eval_res)
        logger.info(
            "Completed %s | test macro_f1=%.4f | acc=%.4f | latency=%.2fms",
            exp_id,
            eval_res["metrics"]["macro_f1"],
            eval_res["metrics"]["accuracy"],
            eval_res["inference_latency_ms"],
        )
        return eval_res
    except Exception as exc:
        mgr.mark_failed(exp_id, f"{type(exc).__name__}: {exc}")
        logger.exception("Experiment %s failed: %s", exp_id, exc)
        raise
    finally:
        clear_gpu_memory()


def run_experiment_suite(
    cfg: dict[str, Any],
    track: str,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run all selected models across seeds sequentially for one track."""
    track_models = models or list(cfg["experiments"][track]["models"])
    track_seeds = seeds or [int(cfg["project"]["default_seed"])]
    results = []
    for model_name in track_models:
        for seed in track_seeds:
            res = run_single_experiment(cfg, track=track, model_name=model_name, seed=seed, **kwargs)
            results.append(res)
    return results

