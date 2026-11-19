# Discontinued Cross-Validation Regimes (15-Fold & 20-Fold) Cleanup Manifest

**Date of Execution:** 2026-11-18  
**Author:** Antigravity Autonomous Research Agent  
**Status:** Completed & Verified  

---

## 1. Executive Summary

In accordance with Phase Three of the research master directive, all 15-fold and 20-fold cross-validation regimes have been permanently discontinued. From this point forward, **10-fold cross-validation is the ONLY active expanded cross-validation regime** across the entire project (35 model configurations $\times$ 10 folds = 350 total experiments).

All artifacts exclusive to 15-fold and 20-fold experiments were audited, cataloged, and safely purged without touching:
- Any 10-fold experiment data or checkpoints (224 completed folds preserved).
- Original 5-fold baseline and Condition B experiments (70 completed runs preserved).
- Shared dataset splits, tabular feature sets, and preprocessed imagery.
- Common model factories, training routines, and evaluation metrics.

---

## 2. Inventory of Purged Artifacts

### A. Checkpoint Directories (`checkpoints_cv/`)
A total of **60 fold checkpoint directories** belonging exclusively to 15-fold experiments (`*orig_15f*`) were deleted, freeing ~3.2 GB of disk space:
- `cv__fundus_3class__mlp__orig_15f__fold[1-15]__seed42` (15 directories)
- `cv__fundus_3class__autoencoder_classifier__orig_15f__fold[1-15]__seed42` (15 directories)
- `cv__fundus_3class__lstm__orig_15f__fold[1-15]__seed42` (15 directories)
- `cv__fundus_3class__bilstm__orig_15f__fold[1-15]__seed42` (15 directories)
*(Zero 20-fold checkpoints existed on disk).*

### B. Results & Prediction Directories (`results/cross_validation/15_fold/`)
The entire 15-fold result tree was removed:
- `results/cross_validation/15_fold/runs/` (60 individual run directories containing `train.log`, `status.json`, `evaluation.json`)
- `results/cross_validation/15_fold/predictions/` (60 prediction CSVs: `*_predictions.csv`)
- `results/cross_validation/15_fold/logs/` (60 worker log files)
- `results/cross_validation/15_fold/splits/` (4 cross-validation split files: `fundus_3class_cv_folds.csv`, `oct_individual_cv_folds.csv`, `oct_sequence_cv_folds.csv`, `oct_severity_cv_folds.csv`)
- `results/cross_validation/15_fold/cv_15fold_runs_summary.csv` (15-fold runs summary table)

### C. 20-Fold Directory Tree (`results/cross_validation/20_fold/`)
- Checked `results/cross_validation/20_fold/` — confirmed completely absent / purged.

### D. Progress & Pipeline Tracking (`results/cross_validation/multifold_execution_progress.csv`)
- Purged all 35 rows corresponding to `15-Fold CV`.
- Purged all rows corresponding to `20-Fold CV`.
- Preserved strictly the 35 active `10-Fold CV` tracking records.

### E. Code & Scheduler Configurations
- `scripts/run_experiments_pipeline.py`: Default `n_folds_list` changed from `[10, 15]` to `[10]`.
- `src/training/adaptive_parallel_scheduler.py`: Hardcoded `n_folds_list = [10]`.
- `configs/adaptive_scheduler_policy.json`: Set `"active_cv_regimes": [10]` and `"discontinued_regimes": [15, 20]`.

---

## 3. Verification Audit
A recursive scan of the project workspace confirmed:
- **Remaining 15-fold artifacts:** 0
- **Remaining 20-fold artifacts:** 0
- **10-Fold experiments intact:** 224 completed, 8 valid interrupted checkpoints preserved, 118 pending.
