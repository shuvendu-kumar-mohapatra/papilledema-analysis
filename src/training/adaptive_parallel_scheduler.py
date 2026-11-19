"""Adaptive parallel training scheduler for 10-fold cross-validation.

Dedicated exclusively to 10-fold cross-validation (15-fold and 20-fold discontinued).
Implements sequential execution for all remaining heavy models (one heavy model at a time,
completing all 10 folds of each architecture before moving to the next), while enabling
safe adaptive concurrency for lightweight models when hardware resources permit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiments_pipeline import ALL_TRACK_MODELS, resolve_path

# ==============================================================================
# Model Architecture Classification & Resource Profiles
# ==============================================================================
HEAVY_MODELS: set[str] = {
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "resnet50_lstm",
    "resnet50_bilstm",
    "efficientnet_b0_bilstm",
}

LIGHTWEIGHT_MODELS: set[str] = {
    "mlp",
    "autoencoder_classifier",
    "lstm",
    "bilstm",
}

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    # Lightweight Tabular & Autoencoder
    "mlp": {"category": "tabular_ae", "is_heavy": False, "max_concurrent": 6, "base_vram_mb": 350.0, "safe_bs": 64},
    "autoencoder_classifier": {"category": "tabular_ae", "is_heavy": False, "max_concurrent": 6, "base_vram_mb": 450.0, "safe_bs": 64},

    # Lightweight Standalone Sequence
    "lstm": {"category": "sequence", "is_heavy": False, "max_concurrent": 4, "base_vram_mb": 400.0, "safe_bs": 64},
    "bilstm": {"category": "sequence", "is_heavy": False, "max_concurrent": 4, "base_vram_mb": 400.0, "safe_bs": 64},

    # Heavy Standalone CNN
    "efficientnet_b0": {"category": "cnn_standalone", "is_heavy": True, "max_concurrent": 1, "base_vram_mb": 950.0, "safe_bs": 32},
    "resnet50": {"category": "cnn_standalone", "is_heavy": True, "max_concurrent": 1, "base_vram_mb": 1800.0, "safe_bs": 32},
    "densenet121": {"category": "cnn_standalone", "is_heavy": True, "max_concurrent": 1, "base_vram_mb": 1800.0, "safe_bs": 32},

    # Heavy CNN-Sequence Hybrids
    "efficientnet_b0_bilstm": {"category": "cnn_hybrid", "is_heavy": True, "max_concurrent": 1, "base_vram_mb": 1250.0, "safe_bs": 32},
    "resnet50_lstm": {"category": "cnn_hybrid", "is_heavy": True, "max_concurrent": 1, "base_vram_mb": 2000.0, "safe_bs": 32},
    "resnet50_bilstm": {"category": "cnn_hybrid", "is_heavy": True, "max_concurrent": 1, "base_vram_mb": 2200.0, "safe_bs": 32},
}

# Strict Hardware Safety Ceilings
MAX_SAFE_VRAM_MB = 6500.0          # Max estimated VRAM sum
VRAM_HEAVY_THROTTLE_PCT = 85.0     # > 85% VRAM: avoid launching heavy models
VRAM_HARD_PAUSE_PCT = 92.0         # > 92% VRAM: pause all new dispatches
MIN_AVAILABLE_RAM_GB = 3.0         # < 3.0 GB free physical RAM: pause all new jobs
MAX_ACCEPTABLE_EPOCH_SEC = 1200.0  # 20 minutes per epoch stall threshold


@dataclass
class JobSpec:
    track: str
    model: str
    fold: int
    n_folds: int  # Strictly 10
    category: str
    is_heavy: bool
    batch_size: int
    est_vram_mb: float
    epochs: int | None = None
    seed: int = 42
    retry_count: int = 0


@dataclass
class RunningJob:
    spec: JobSpec
    process: subprocess.Popen
    start_time: float
    log_path: Path
    log_file: Any = None


class AdaptiveParallelScheduler:
    def __init__(
        self,
        n_folds_list: list[int] | None = None,
        max_concurrency: int = 4,
        threads_per_job: int = 2,
        epochs_override: int | None = None,
        sequential_heavy: bool = True,
        policy_file: Path | None = None,
    ) -> None:
        # Strictly enforce 10-fold CV as the only active regime
        self.n_folds_list = [10]
        self.max_concurrency = max_concurrency
        self.threads_per_job = threads_per_job
        self.epochs_override = epochs_override
        self.sequential_heavy = sequential_heavy
        self.policy_file = policy_file or (ROOT / "configs" / "adaptive_scheduler_policy.json")
        self.queue: list[JobSpec] = []
        self.running: list[RunningJob] = []
        self.completed_count = 0
        self.failed_jobs: list[dict[str, Any]] = []
        self.t_start = time.perf_counter()
        self.current_heavy_target: tuple[str, str] | None = None  # (track, model) lock

    def load_dynamic_policy(self) -> None:
        """Read dynamic policy file from disk if present."""
        if self.policy_file.exists():
            try:
                cfg = json.loads(self.policy_file.read_text(encoding="utf-8"))
                if "sequential_heavy" in cfg:
                    self.sequential_heavy = bool(cfg["sequential_heavy"])
            except Exception:
                pass

    def get_hardware_telemetry(self) -> dict[str, Any]:
        """Query live hardware telemetry (VRAM, RAM, swap, CPU)."""
        vram_allocated_mb = 0.0
        vram_total_mb = 8151.0
        vram_pct = 0.0

        if torch.cuda.is_available():
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                vram_total_mb = total_b / (1024 ** 2)
                vram_used_b = total_b - free_b
                vram_allocated_mb = vram_used_b / (1024 ** 2)
                vram_pct = (vram_used_b / max(1.0, total_b)) * 100.0
            except Exception:
                vram_allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
                vram_pct = (vram_allocated_mb / max(1.0, vram_total_mb)) * 100.0

        vm = psutil.virtual_memory()
        available_ram_gb = vm.available / (1024 ** 3)
        ram_used_gb = vm.used / (1024 ** 3)
        ram_total_gb = vm.total / (1024 ** 3)
        ram_pct = vm.percent

        sm = psutil.swap_memory()
        swap_used_gb = sm.used / (1024 ** 3)
        swap_pct = sm.percent

        cpu_pct = psutil.cpu_percent(interval=None)

        return {
            "vram_allocated_mb": round(vram_allocated_mb, 1),
            "vram_total_mb": round(vram_total_mb, 1),
            "vram_pct": round(vram_pct, 1),
            "available_ram_gb": round(available_ram_gb, 2),
            "ram_used_gb": round(ram_used_gb, 2),
            "ram_total_gb": round(ram_total_gb, 2),
            "ram_pct": round(ram_pct, 1),
            "swap_used_gb": round(swap_used_gb, 2),
            "swap_pct": round(swap_pct, 1),
            "cpu_pct": round(cpu_pct, 1),
        }

    def detect_slow_workers(self) -> list[str]:
        """Detect workers whose last epoch exceeded acceptable duration threshold."""
        slow = []
        for j in self.running:
            if j.log_path.exists():
                try:
                    lines = j.log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    epoch_lines = [l for l in lines if "Epoch" in l]
                    if epoch_lines:
                        last_line = epoch_lines[-1]
                        if "|" in last_line:
                            time_part = last_line.split("|")[-1].strip().replace("s", "")
                            dur = float(time_part)
                            if dur > MAX_ACCEPTABLE_EPOCH_SEC:
                                slow.append(f"{j.spec.model}_f{j.spec.fold} ({dur:.0f}s/ep)")
                except Exception:
                    pass
        return slow

    def load_completed_keys(self, n_folds: int = 10) -> set[tuple[str, str, int]]:
        """Load completed (track, model, fold) keys from canonical 10-fold summary CSV."""
        p = resolve_path(f"results/cross_validation/10_fold/cv_10fold_runs_summary.csv")
        if not p.exists():
            return set()
        try:
            df = pd.read_csv(p)
            return set(zip(df["track"], df["model"], df["fold"].astype(int)))
        except Exception:
            return set()

    def build_job_queue(self) -> None:
        """Build priority queue exclusively for remaining 10-fold CV jobs."""
        all_specs: list[JobSpec] = []
        done_keys = self.load_completed_keys(10)

        for trk, models in ALL_TRACK_MODELS.items():
            for m_name in models:
                cfg_info = MODEL_PROFILES.get(
                    m_name,
                    {
                        "category": "cnn_hybrid",
                        "is_heavy": True,
                        "base_vram_mb": 2000.0,
                        "safe_bs": 32,
                    },
                )

                for fold in range(1, 11):
                    # Check if already completed in summary CSV
                    if (trk, m_name, fold) in done_keys:
                        continue

                    # Check if valid evaluation.json exists on disk
                    exp_id = f"cv__{trk}__{m_name}__orig_10f__fold{fold}__seed42"
                    eval_p = resolve_path(f"results/cross_validation/10_fold/runs/{exp_id}/evaluation.json")
                    if eval_p.exists():
                        continue

                    spec = JobSpec(
                        track=trk,
                        model=m_name,
                        fold=fold,
                        n_folds=10,
                        category=cfg_info["category"],
                        is_heavy=cfg_info["is_heavy"],
                        batch_size=cfg_info["safe_bs"],
                        est_vram_mb=cfg_info["base_vram_mb"],
                        epochs=self.epochs_override,
                    )
                    all_specs.append(spec)

        # Ordering strategy:
        # Group all 10 folds of each model together sequentially!
        # First finish models that already have partial completed folds (e.g. fundus efficientnet_b0_bilstm with 8/10 done),
        # then complete remaining models architecture by architecture.
        def priority_key(s: JobSpec) -> tuple[int, str, str, int]:
            # How many folds already done for this (track, model)?
            comp_count = sum(1 for f in range(1, 11) if (s.track, s.model, f) in done_keys)
            # Higher completion first (negative rank so 8/10 runs before 0/10)
            comp_rank = -comp_count
            return (comp_rank, s.track, s.model, s.fold)

        all_specs.sort(key=priority_key)
        self.queue = all_specs
        print(f"[Scheduler] Initialized 10-fold queue with {len(self.queue)} pending jobs across {len(set((s.track, s.model) for s in self.queue))} models.", flush=True)

    def current_running_vram(self) -> float:
        return sum(j.spec.est_vram_mb for j in self.running)

    def can_dispatch(self, spec: JobSpec) -> tuple[bool, str]:
        """Determine dispatch eligibility with strict sequential heavy-model execution."""
        self.load_dynamic_policy()
        state = self.get_hardware_telemetry()

        # 1. Available physical RAM must be >= 3.0 GB
        if state["available_ram_gb"] < MIN_AVAILABLE_RAM_GB:
            return False, f"Available RAM {state['available_ram_gb']:.2f} GB is below {MIN_AVAILABLE_RAM_GB} GB threshold"

        # 2. VRAM Hard Ceiling: >= 92% pauses ALL dispatches
        if state["vram_pct"] >= VRAM_HARD_PAUSE_PCT:
            return False, f"VRAM utilization {state['vram_pct']:.1f}% >= {VRAM_HARD_PAUSE_PCT}% hard ceiling"

        # 3. SEQUENTIAL HEAVY MODEL RULE:
        # Only ONE heavy model training process is allowed at a time across the entire pipeline.
        active_heavy_count = sum(1 for j in self.running if j.spec.is_heavy)
        if spec.is_heavy:
            if self.sequential_heavy and active_heavy_count >= 1:
                return False, f"Sequential heavy policy: 1 heavy model already running ({active_heavy_count}/1)"

            # If a heavy target lock exists, ensure we finish all folds of that model first
            if self.current_heavy_target is not None:
                curr_trk, curr_m = self.current_heavy_target
                # Check if current heavy target still has remaining folds in queue or running
                target_remaining = any(
                    (s.track == curr_trk and s.model == curr_m) for s in self.queue
                ) or any(
                    (j.spec.track == curr_trk and j.spec.model == curr_m) for j in self.running
                )
                if target_remaining and (spec.track != curr_trk or spec.model != curr_m):
                    return False, f"Sequential heavy lock: completing all folds of {curr_trk}.{curr_m} first"

            # VRAM Heavy Throttle at 85%
            if state["vram_pct"] >= VRAM_HEAVY_THROTTLE_PCT:
                return False, f"VRAM utilization {state['vram_pct']:.1f}% >= {VRAM_HEAVY_THROTTLE_PCT}% heavy throttle threshold"

        # 4. Global Concurrency Limit for lightweight models
        if len(self.running) >= self.max_concurrency:
            return False, f"Global concurrency limit reached ({len(self.running)}/{self.max_concurrency})"

        # 5. Estimated VRAM Budget Ceiling
        if self.current_running_vram() + spec.est_vram_mb > MAX_SAFE_VRAM_MB:
            return False, f"Estimated VRAM ({self.current_running_vram() + spec.est_vram_mb:.0f}MB) exceeds budget ({MAX_SAFE_VRAM_MB:.0f}MB)"

        # 6. Slow worker protection
        slow_workers = self.detect_slow_workers()
        if slow_workers and spec.is_heavy:
            return False, f"Slow workers detected ({', '.join(slow_workers)}). Holding heavy dispatch"

        return True, "Eligible"

    def dispatch_job(self, spec: JobSpec) -> None:
        if spec.is_heavy:
            self.current_heavy_target = (spec.track, spec.model)

        log_dir = ROOT / "results" / "cross_validation" / "10_fold" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_p = log_dir / f"worker_{spec.track}_{spec.model}_f{spec.fold}.log"
        log_f = open(log_p, "w", encoding="utf-8")

        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "run_single_fold_worker.py"),
            "--track", spec.track,
            "--model", spec.model,
            "--fold", str(spec.fold),
            "--n-folds", "10",
            "--batch-size", str(spec.batch_size),
            "--num-threads", str(self.threads_per_job),
        ]
        if spec.epochs is not None:
            cmd.extend(["--epochs", str(spec.epochs)])

        p = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
        )
        running_job = RunningJob(
            spec=spec,
            process=p,
            start_time=time.perf_counter(),
            log_path=log_p,
            log_file=log_f,
        )
        self.running.append(running_job)
        heavy_status = "HEAVY [Sequential]" if spec.is_heavy else "LIGHT [Adaptive]"
        print(
            f"[Scheduler] Dispatched [PID {p.pid}]: 10F | {spec.track} | {spec.model} | Fold {spec.fold} "
            f"({heavy_status}, Concurrency: {len(self.running)}/{self.max_concurrency}, Est VRAM: {self.current_running_vram():.0f}MB)",
            flush=True,
        )

    def update_multifold_progress_file(self) -> None:
        """Maintain master live report strictly for 10-fold CV."""
        master_prog_p = resolve_path("results/cross_validation/multifold_execution_progress.csv")
        rows = []
        done_keys = self.load_completed_keys(10)
        for trk, models in ALL_TRACK_MODELS.items():
            for m in models:
                comp = sum(1 for f in range(1, 11) if (trk, m, f) in done_keys)
                rem = 10 - comp
                st = (
                    "Completed" if rem == 0
                    else ("In Progress" if any(j.spec.model == m and j.spec.track == trk for j in self.running)
                    else "Pending")
                )
                rows.append({
                    "Track": trk,
                    "Model": m,
                    "CV Regime": "10-Fold CV",
                    "Total Folds": 10,
                    "Completed": comp,
                    "Remaining": rem,
                    "Status": st,
                    "Last Updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                })
        df = pd.DataFrame(rows)
        master_prog_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(master_prog_p, index=False)

    def record_completed_fold(self, payload: dict[str, Any]) -> None:
        trk = payload["track"]
        model = payload["model"]
        fold = payload["fold"]
        exp_id = f"cv__{trk}__{model}__orig_10f__fold{fold}__seed42"

        eval_p = resolve_path(f"results/cross_validation/10_fold/runs/{exp_id}/evaluation.json")
        if eval_p.exists():
            try:
                with open(eval_p, "r", encoding="utf-8") as f:
                    eval_dict = json.load(f)
                new_row = pd.DataFrame([eval_dict])
            except Exception:
                new_row = None
        else:
            new_row = None

        if new_row is None:
            return

        summary_csv = resolve_path("results/cross_validation/10_fold/cv_10fold_runs_summary.csv")
        if summary_csv.exists():
            df_existing = pd.read_csv(summary_csv)
            mask = (
                (df_existing["track"] == trk)
                & (df_existing["model"] == model)
                & (df_existing["fold"] == fold)
            )
            df_clean = df_existing[~mask & df_existing["exp_id"].notna()]
            df_combined = pd.concat([df_clean, new_row], ignore_index=True)
        else:
            df_combined = new_row

        col_order = [
            "exp_id", "track", "model", "n_folds", "fold", "condition",
            "train_samples", "val_samples", "test_samples", "train_duration_sec",
            "best_epoch", "inference_latency_ms", "total_params", "peak_gpu_mem_mb",
            "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1",
            "macro_sensitivity", "macro_specificity", "roc_auc_ovr", "pr_auc_ovr",
            "per_class_auc", "per_class_pr_auc", "confusion_matrix", "per_class",
            "num_samples", "error_rate", "micro_precision", "micro_recall", "micro_f1",
            "mcc", "cohen_kappa", "log_loss", "brier_score", "ece", "qwk", "mae", "mse"
        ]
        for c in col_order:
            if c not in df_combined.columns:
                df_combined[c] = None

        df_combined = df_combined[col_order]
        df_combined = df_combined.sort_values(["track", "model", "fold"]).reset_index(drop=True)
        df_combined.to_csv(summary_csv, index=False)
        self.completed_count += 1
        self.update_multifold_progress_file()

    def run(self) -> None:
        self.build_job_queue()
        if not self.queue:
            print("[Scheduler] All 350 valid 10-fold CV experiments are already complete!")
            return

        print(f"\n{'='*70}\nSTARTING EXCLUSIVE 10-FOLD CV SCHEDULER\n{'='*70}\n")
        print(f"Policy: Sequential Heavy-Model Execution (1 heavy at a time, completing all 10 folds before switching)")
        print(f"Safety Limits: VRAM < {VRAM_HEAVY_THROTTLE_PCT}% (Heavy), < {VRAM_HARD_PAUSE_PCT}% (Pause All) | Available RAM >= {MIN_AVAILABLE_RAM_GB} GB\n", flush=True)

        last_status_print = time.perf_counter()

        while self.queue or self.running:
            # 1. Dispatch eligible jobs
            while self.queue:
                dispatched_any = False
                for idx, candidate_spec in enumerate(self.queue):
                    eligible, reason = self.can_dispatch(candidate_spec)
                    if eligible:
                        job_to_dispatch = self.queue.pop(idx)
                        self.dispatch_job(job_to_dispatch)
                        dispatched_any = True
                        break

                if not dispatched_any:
                    break

            # 2. Check running processes
            still_running: list[RunningJob] = []
            for job in self.running:
                poll_code = job.process.poll()
                if poll_code is None:
                    still_running.append(job)
                else:
                    if job.log_file and not job.log_file.closed:
                        job.log_file.close()

                    elapsed = time.perf_counter() - job.start_time
                    exp_id = f"cv__{job.spec.track}__{job.spec.model}__orig_10f__fold{job.spec.fold}__seed42"
                    eval_p = resolve_path(f"results/cross_validation/10_fold/runs/{exp_id}/evaluation.json")

                    if poll_code == 0 or eval_p.exists():
                        payload = {
                            "status": "completed",
                            "track": job.spec.track,
                            "model": job.spec.model,
                            "fold": job.spec.fold,
                            "n_folds": 10,
                            "duration_sec": elapsed,
                        }
                        self.record_completed_fold(payload)
                        folds_per_hr = (self.completed_count / max(1.0, time.perf_counter() - self.t_start)) * 3600.0
                        print(
                            f"[Scheduler] Completed: 10F | {job.spec.track} | {job.spec.model} | Fold {job.spec.fold} "
                            f"(Elapsed: {elapsed:.1f}s) | Rate: {folds_per_hr:.1f} folds/hr | Remaining in queue: {len(self.queue)}",
                            flush=True,
                        )
                    else:
                        print(f"[Scheduler] Error: Worker PID {job.process.pid} exited with code {poll_code}.", flush=True)
                        err_tail = ""
                        if job.log_path.exists():
                            try:
                                lines = job.log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                                err_tail = "\n".join(lines[-15:])
                            except Exception:
                                pass
                        if "out of memory" in err_tail.lower():
                            print(f"[Scheduler] Detected CUDA OOM for {job.spec.model}. Reducing batch size from {job.spec.batch_size} to {max(16, job.spec.batch_size // 2)}", flush=True)
                            if job.spec.retry_count < 2:
                                job.spec.batch_size = max(16, job.spec.batch_size // 2)
                                job.spec.retry_count += 1
                                self.queue.insert(0, job.spec)
                        self.failed_jobs.append({"spec": job.spec, "error": err_tail[:500]})

            self.running = still_running

            # 3. Status heartbeat every 60 seconds
            if time.perf_counter() - last_status_print > 60.0:
                last_status_print = time.perf_counter()
                st = self.get_hardware_telemetry()
                active_heavy = sum(1 for j in self.running if j.spec.is_heavy)
                target_str = f"{self.current_heavy_target[0]}.{self.current_heavy_target[1]}" if self.current_heavy_target else "None"
                print(
                    f"[Scheduler Heartbeat] Running: {len(self.running)} (Heavy: {active_heavy}, Target: {target_str}) | "
                    f"Queue: {len(self.queue)} | VRAM: {st['vram_pct']}% ({st['vram_allocated_mb']:.0f}MB) | "
                    f"RAM Avail: {st['available_ram_gb']:.1f}GB ({st['ram_pct']}%) | CPU: {st['cpu_pct']}%",
                    flush=True,
                )

            time.sleep(1.0)

        print(f"\n{'='*70}\nALL 350 VALID 10-FOLD CV EXPERIMENTS COMPLETED!\n{'='*70}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exclusive 10-Fold CV Scheduler with Sequential Heavy Models.")
    parser.add_argument("--concurrency", type=int, default=4, help="Global concurrency limit (default: 4)")
    parser.add_argument("--threads-per-job", type=int, default=2, help="CPU threads per job")
    args = parser.parse_args()

    scheduler = AdaptiveParallelScheduler(
        max_concurrency=args.concurrency,
        threads_per_job=args.threads_per_job,
        sequential_heavy=True,
    )
    scheduler.run()
