# Comparative Analysis of CNN, RNN, and Hybrid CNN–RNN Architectures for Papilledema Detection and Severity Grading

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.10+](https://img.shields.io/badge/PyTorch-2.10%2B-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-9%20passing-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/Experiments-175%20Runs%20Completed-blue.svg)](results/reports/final_research_summary.md)

> **Clinical Disclaimer:** This repository contains academic research code and experimental benchmarks created strictly for computational research and educational evaluation. **It is not a certified medical device and must not be used for clinical diagnosis, triage, or treatment decisions.** See [Limitations & Clinical Caveats](#18-limitations--clinical-caveats) for important dataset constraints.

---

## 1. Project Overview

This repository provides a complete, reproducible PyTorch research pipeline comparing **Convolutional Neural Networks (CNNs)**, **standalone Recurrent Neural Networks (RNNs)**, **Hybrid CNN–RNN architectures**, and **lightweight non-convolutional/autoencoder baselines** across **Fundus Photography** and **Optical Coherence Tomography (OCT)** for papilledema detection and exploratory severity grading.

All **175 primary experimental runs** (`35` track-model configurations × `5` random seeds, excluding `8` earlier smoke checks) have been executed end-to-end on GPU, evaluated on held-out test splits, analyzed with non-parametric and paired statistical tests, and consolidated into CSV tables, publication-style figures, Markdown reports, and a central Jupyter Notebook.

---

## 2. Research Motivation

Clinically distinguishing **true papilledema** (optic disc edema secondary to raised intracranial pressure) from **normal optic discs** and **pseudopapilledema** (benign structural disc elevation such as optic disc drusen) is visually challenging and carries high diagnostic stakes. Deep learning studies frequently evaluate only isolated CNN classifiers on a single modality, often overlooking duplicate-image leakage, pre-augmented dataset leakage, or how spatial-patch vs. sequential-slice architectures behave across modalities.

This project investigates:
1. Whether CNN backbones outperform standalone patch/slice RNNs and whether Hybrid CNN–RNN coupling improves performance on 2D single-image classification vs. ordered OCT B-scan sequences.
2. How strictly enforcing duplicate-cluster grouping, IR/SLO filtering, and pre-augmented folder exclusion affects real-world performance across both modalities.
3. Whether standard deep architectures can reliably grade **G1–G4 severity** under severe class imbalance and limited patient diversity.

---

## 3. Problem Statement & Experimental Tracks

Ten models are benchmarked across **four research tracks**:

| Track ID | Modality | Task | Classes | Input Mode | Models | Seeds | Total Runs |
|---|---|---|---|---|---:|---:|---:|
| `fundus_3class` | Fundus Photography | 3-Class Classification | Normal, Papilledema, Pseudopapilledema | 2D RGB (`3×224×224`) | 10 | 5 | 50 |
| `oct_individual` | OCT B-Scans | 3-Class Classification | Normal, Papilledema, Pseudopapilledema | 2D Grayscale (`1×224×224`) | 10 | 5 | 50 |
| `oct_sequence` | OCT B-Scans | Sequence Classification | Normal, Papilledema, Pseudopapilledema | Ordered Window (`8×1×224×224`) | 5 | 5 | 25 |
| `oct_severity` | OCT B-Scans | **Exploratory** 4-Class Grading | G1, G2, G3, G4 | 2D Grayscale (`1×224×224`) | 10 | 5 | 50 |
| **Total** | — | — | — | — | **35** | **5** | **175** |

---

## 4. Methodology & Leakage-Aware Data Pipeline

Implemented in [`src/data/`](src/data/):

1. **Automated Data Verification ([`src/data/validation.py`](src/data/validation.py)):**
   * Scans raw Fundus and OCT directories, computes file SHA-256 and pixel SHA-256 hashes, verifies readability, and logs dimensions.
   * **Fundus Duplicate & Cross-Class Conflict Detection:** Computes perceptual similarity clusters (`near_dup_threshold: 0.65`, `cross_class_conflict_threshold: 0.90`). Out of `1,369` raw Fundus images, `110` near-duplicate pairs (`74` multi-image clusters covering `171` images) were grouped into `1,270` cluster units, and **`2` conflicting cross-class near-duplicate images** (`Normal/452.jpg` and `Pseudopapilledema/149.jpg`, normalized cross-correlation `similarity_ncc = 0.9998`, `exact_pixel_match = False`) were excluded, leaving **`1,367` images**.
   * **OCT Leakage & Artifact Exclusion:** Strictly excludes `data/raw/OCT/Augmented Data` (`30,000` files), `data/raw/OCT/Original Data/Original Data` (`1,273` files), and `data/raw/OCT/Original Data/OCT Data with Patient ID` (`1,146` files) per [`configs/base.yaml`](configs/base.yaml). Detects and excludes `1` non-B-scan IR/SLO reflectance image (`269×701`, `max_ch_diff=0`), retaining **`1,140` B-scan images**.
2. **Leakage-Aware Splitting ([`src/data/split_manager.py`](src/data/split_manager.py)):**
   * Fixed `70% / 15% / 15%` (`train / val / test`) stratified grouped splits generated with `seed=42` (`split_validation.json`).
   * **Fundus:** Split strictly by near-duplicate cluster ID so near-identical images never straddle splits.
   * **OCT:** Grouped by filename-derived `inferred_sequence_id`. Where a class has only one inferred subject prefix (`Normal`, `Pseudopapilledema`, `G1`, `G4`), splits are partitioned into **contiguous spatial slice blocks** rather than random slice shuffling, and explicitly flagged (`is_verified_patient=False`, `split_unit_type=contiguous_slice_block_single_subject`).
   * **Runtime Leakage Guards (`validate_no_leakage`):** Asserts zero cluster/sequence overlap, zero pixel hash overlap, expected class labels, and zero paths from excluded folders.
3. **Preprocessing & Augmentation ([`src/data/preprocessing.py`](src/data/preprocessing.py), [`src/data/augmentation.py`](src/data/augmentation.py)):**
   * Fundus images use RGB (`normalize_mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`); optional CLAHE is implemented (`use_clahe: false` in default config).
   * OCT B-scans are converted to single-channel grayscale (`normalize_mean=[0.5]`, `std=[0.5]`), resized to `224×224`.
   * Training augmentation (`RandomResizedCrop`, `RandomHorizontalFlip`, slight rotation/color jitter) is applied exclusively to the training split; validation and test transforms are deterministic (`Resize` + normalize). OCT sequences advance augmentation deterministically per window sample while maintaining frame consistency.

---

## 5. Datasets & Data Privacy Disclaimer

> **Data Redistribution & Privacy Notice:** Raw medical images (`data/raw/`), patient-ID directories, image-level manifests (`data/manifests/*.csv`), split CSVs (`data/splits/*.csv`), and per-sample prediction CSVs (`results/predictions/`) are **not included in this repository** via [`.gitignore`](.gitignore). Aggregate counts and validation summaries are preserved in [`data/manifests/dataset_audit.json`](data/manifests/dataset_audit.json), [`data/splits/split_validation.json`](data/splits/split_validation.json), and [`results/tables/dataset_summary.csv`](results/tables/dataset_summary.csv).

### Dataset Composition After Cleaning & Splitting (`seed=42`)

| Dataset / Split Table | Class | Raw | Retained | Train | Val | Test | Grouping Strategy |
|---|---|---:|---:|---:|---:|---:|---|
| **Fundus (`fundus_3class`)** | Normal | 779 | 778 | 544 | 117 | 117 | Near-duplicate clusters (`1,270` clusters; `190` in test) |
| | Papilledema | 295 | 295 | 206 | 45 | 44 | |
| | Pseudopapilledema | 295 | 294 | 206 | 44 | 44 | |
| **Total Fundus** | | **1,369** | **1,367** | **956** | **206** | **205** | |
| **OCT 3-Class (`oct_individual`)** | Normal | 132 | 132 | 92 | 20 | 20 | Inferred series + contiguous block (`1` prefix) |
| | Papilledema | 895 | 895 | 603 | 146 | 146 | Inferred series (`10` prefixes, `17` series) |
| | Pseudopapilledema | 114 | 113 | 79 | 17 | 17 | Inferred series + contiguous block (`1` prefix) |
| **Total OCT 3-Class (Slices)** | | **1,141** | **1,140** | **774** | **183** | **183** | |
| **OCT Sequences (`oct_sequence`)** | All 3 Classes (`seq_len=8, stride=8`) | — | 147 windows | 101 | 23 | 23 | Non-overlapping ordered slice windows (`17 / 4 / 4` seqs) |
| **OCT Severity (`oct_severity`, Exploratory)** | G1 | — | 40 | 28 | 6 | 6 | Contiguous block (`1` prefix) |
| | G2 | — | 598 | 306 | 146 | 146 | Inferred series (`6` prefixes) |
| | G3 | — | 184 | 146 | 25 | 13 | Inferred series (`3` prefixes) |
| | G4 | — | 73 | 51 | 11 | 11 | Contiguous block (`1` prefix) |
| **Total OCT Severity (Slices)** | | — | **895** | **531** | **188** | **176** | |

### Data Acquisition for Reproduction

To run training or verification locally, obtain the external Fundus ("Kim Fundus" 3-class collection) and Original OCT datasets from their original providers and place them under:
* `data/raw/fundus/{Normal,Papilledema,Pseudopapilledema}/`
* `data/raw/OCT/Original Data/OCT/{Normal,Papilledema/G1..G4,Pseudo papilledema}/`

Because dataset licensing and provenance reside with the original dataset creators, users must verify source terms independently before use.

---

## 6. Model Architectures

Implemented in [`src/models/`](src/models/) and configured in [`configs/models.yaml`](configs/models.yaml):

| Category | Model Identifier | Architecture Summary | Params (Fundus 3-ch) | Params (OCT 1-ch) |
|---|---|---|---|---:|
| **CNN** | `resnet50` | ImageNet-pretrained ResNet-50 (`backbone_mode: full`, dropout `0.3`) | 23,514,179 | 23,507,907 |
| **CNN** | `densenet121` | ImageNet-pretrained DenseNet-121 (`dropout: 0.3`) | 6,956,931 | 6,950,659 |
| **CNN** | `efficientnet_b0` | ImageNet-pretrained EfficientNet-B0 (`dropout: 0.3`) | 4,011,391 | 4,010,815 |
| **Standalone RNN** | `lstm` | 2-layer LSTM (`hidden_dim=128`); image mode splits `224×224` into `7×7 = 49` ordered `32×32` spatial patches (`patch_proj`); sequence mode encodes each B-scan via lightweight CNN stem (`slice_stem`) | 720,899 | 457,155 |
| **Standalone RNN** | `bilstm` | 2-layer Bidirectional LSTM (`hidden_dim=128`, `patch_size=32`, `dropout=0.3`) | 1,116,803 | 853,059 |
| **Hybrid CNN–RNN** | `resnet50_lstm` | ResNet-50 spatial grid/slice features + 1-layer LSTM (`hidden_dim=128`) | 24,247,747 | 24,241,475 |
| **Hybrid CNN–RNN** | `resnet50_bilstm` | ResNet-50 features + 1-layer BiLSTM (`hidden_dim=128`) with masked temporal pooling | 24,446,019 | 24,439,747 |
| **Hybrid CNN–RNN** | `efficientnet_b0_bilstm` | EfficientNet-B0 features + 1-layer BiLSTM (`hidden_dim=128`) with masked temporal pooling | 4,748,927 | 4,748,351 |
| **Baseline** | `mlp` | Adaptive `16×16` average pool + MLP `[256, 128]` (`dropout=0.3`) | 233,231 | 100,103 |
| **Baseline** | `autoencoder_classifier` | Convolutional encoder-decoder (`latent_dim=128`, `recon_size=64`) pretrained for `5` epochs, then joint classification + `0.1×MSE` reconstruction | 718,118 | 717,252 |

* **1-channel adaptation:** Pretrained RGB stems adapt first-layer conv weights cleanly for `in_channels=1` on OCT data without distorting parameter structures.
* **Sequence padding isolation:** Valid sequence masks exclude padded frames from BatchNorm statistics and temporal mean pooling (`tests/test_models_and_training.py` verifies zero padded-frame leakage).

---

## 7. Training Configuration & Reproducibility

Configured in [`configs/base.yaml`](configs/base.yaml) and executed in [`src/training/`](src/training/):

* **Random Seeds:** `[42, 123, 2024, 3407, 5678]` (`5` seeds per model-track pair)
* **Optimization:** `AdamW` (`lr=1e-4`, `weight_decay=1e-4`), Cosine Annealing LR scheduler, Gradient Clipping (`max_norm=1.0`), Mixed Precision (`torch.amp` CUDA FP16)
* **Schedule & Selection:** `epochs=30`, `early_stopping_patience=7`, `batch_size=16` (`4` for `oct_sequence`), inverse-frequency class-weighted `CrossEntropyLoss`, best checkpoint selected strictly by `validation macro_f1` (never on test data)
* **Verified Execution Environment:** Python `3.12.10`, PyTorch `2.14.0+cu132`, CUDA `13.2`, NVIDIA GeForce RTX 5050 Laptop GPU (`7.96 GB` VRAM, `sm_120`), `23.64 GB` RAM, Windows 11
* **Total Compute Across 175 Non-Smoke Runs:** `15,635.67 seconds` (`4.34 hours` cumulative GPU training time)

---

## 8. Headline Experimental Results

| Track | Winning Model | Mean Macro F1 ± Std | Mean Accuracy | Mean ROC-AUC (OvR) | Runner-Up (Mean Macro F1 ± Std) |
|---|---|---:|---:|---:|---|
| **Fundus 3-Class** (`n=205`) | **`efficientnet_b0`** (CNN) | **`0.9835 ± 0.0055`** | **`0.9883`** | **`0.9998`** | `efficientnet_b0_bilstm` (`0.9806 ± 0.0100`) |
| **OCT Individual** (`n=183`) | **`densenet121`** (CNN) | **`0.9928 ± 0.0099`** | **`0.9978`** | **`1.0000`** | `efficientnet_b0` (`0.9903 ± 0.0000`) |
| **OCT Sequence** (`n=23` windows) | **`resnet50_bilstm`** (Hybrid) | **`0.9849 ± 0.0338`** | **`0.9913`** | **`1.0000`** | `efficientnet_b0_bilstm` (`0.9142 ± 0.1138`) |
| **OCT Severity (Exploratory)** (`n=176`) | **`resnet50_lstm`** (Hybrid) | **`0.5274 ± 0.0780`** | `0.7250` | `0.7749` | `efficientnet_b0` (`0.5042 ± 0.0941`) / `resnet50_bilstm` (`Acc 0.7477`) |

**Key Comparative Takeaways:**
* **On 2D single images (`fundus_3class`, `oct_individual`):** Pure CNNs (`EfficientNet-B0`, `DenseNet-121`) slightly edge out or tie Hybrid CNN–RNNs while requiring fewer parameters and no recurrent head. Both CNNs and Hybrids massively outperform standalone patch-based RNNs (`0.5585–0.7522` Macro F1) and non-convolutional/AE baselines (`0.7164–0.8691` Macro F1).
* **On ordered OCT sequences (`oct_sequence`):** Bidirectional temporal modeling over deep ResNet-50 slice representations (`ResNet50–BiLSTM`, `0.9849`) substantially outperforms standalone RNNs built from scratch (`BiLSTM 0.3385`, `LSTM 0.2760`), demonstrating the importance of pretrained spatial encoders before temporal aggregation when sequences are scarce (`101` train windows).
* **On exploratory severity grading (`oct_severity`):** Performance drops sharply (`0.5274` best Macro F1) due to severe class imbalance (`40` G1 vs. `598` G2), scarce patient diversity (`1` prefix each in G1 and G4), and subtle inter-grade boundaries (G2 vs. G3).

---

## 9. Complete Quantitative Benchmarks (5-Seed Mean ± Std)

Source CSVs: [`results/tables/`](results/tables/)

### Track 1: Fundus 3-Class (`fundus_3class`, 50 runs, Test `n=205`)

| Model | Category | Macro F1 (Mean ± Std) | Accuracy | Bal. Acc | ROC-AUC | PR-AUC | Params | Train Time (s) | Latency (ms/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`efficientnet_b0`** | CNN | **0.9835 ± 0.0055** | **0.9883** | 0.9818 | **0.9998** | **0.9994** | 4,011,391 | 154.06 | 2.54 |
| `efficientnet_b0_bilstm` | Hybrid | 0.9806 ± 0.0100 | 0.9854 | **0.9820** | 0.9986 | 0.9957 | 4,748,927 | 116.72 | 2.57 |
| `densenet121` | CNN | 0.9795 ± 0.0054 | 0.9844 | 0.9786 | 0.9995 | 0.9983 | 6,956,931 | 159.56 | 3.25 |
| `resnet50` | CNN | 0.9779 ± 0.0034 | 0.9834 | 0.9771 | 0.9968 | 0.9943 | 23,514,179 | 121.08 | 3.10 |
| `resnet50_bilstm` | Hybrid | 0.9779 ± 0.0085 | 0.9834 | 0.9761 | 0.9964 | 0.9940 | 24,446,019 | 138.30 | 3.09 |
| `resnet50_lstm` | Hybrid | 0.9684 ± 0.0084 | 0.9766 | 0.9684 | 0.9953 | 0.9910 | 24,247,747 | 124.51 | 3.02 |
| `autoencoder_classifier` | Baseline | 0.8691 ± 0.0107 | 0.8937 | 0.8660 | 0.9639 | 0.9233 | 718,118 | **77.49** | **1.72** |
| `mlp` | Baseline | 0.8018 ± 0.0176 | 0.8420 | 0.8056 | 0.9491 | 0.8856 | **233,231** | 81.46 | 2.02 |
| `lstm` | RNN | 0.7522 ± 0.0132 | 0.7776 | 0.7727 | 0.9073 | 0.7900 | 720,899 | 153.45 | 2.28 |
| `bilstm` | RNN | 0.7230 ± 0.0480 | 0.7522 | 0.7456 | 0.8841 | 0.7447 | 1,116,803 | 179.73 | 2.78 |

### Track 2: OCT 3-Class Individual B-Scan (`oct_individual`, 50 runs, Test `n=183`)

| Model | Category | Macro F1 (Mean ± Std) | Accuracy | Bal. Acc | ROC-AUC | PR-AUC | Params | Train Time (s) | Latency (ms/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`densenet121`** | CNN | **0.9928 ± 0.0099** | **0.9978** | **0.9933** | **1.0000** | **1.0000** | 6,950,659 | 60.30 | 3.89 |
| `efficientnet_b0` | CNN | 0.9903 ± 0.0000 | 0.9945 | 0.9833 | 0.9997 | 0.9985 | 4,010,815 | **53.65** | 3.25 |
| `resnet50_bilstm` | Hybrid | 0.9863 ± 0.0255 | 0.9934 | 0.9858 | 0.9997 | 0.9996 | 24,439,747 | 68.74 | 3.87 |
| `efficientnet_b0_bilstm` | Hybrid | 0.9807 ± 0.0154 | 0.9913 | 0.9820 | 0.9996 | 0.9984 | 4,748,351 | 57.72 | 3.29 |
| `resnet50` | CNN | 0.9781 ± 0.0294 | 0.9902 | 0.9700 | 1.0000 | 1.0000 | 23,507,907 | 66.51 | 3.83 |
| `resnet50_lstm` | Hybrid | 0.9737 ± 0.0185 | 0.9869 | 0.9658 | 1.0000 | 1.0000 | 24,241,475 | 62.98 | 3.77 |
| `autoencoder_classifier` | Baseline | 0.7317 ± 0.0551 | 0.8426 | 0.7933 | 0.9457 | 0.9305 | 717,252 | 55.32 | **2.33** |
| `mlp` | Baseline | 0.7164 ± 0.0492 | 0.8678 | 0.6994 | 0.8998 | 0.8080 | **100,103** | 70.57 | 2.43 |
| `bilstm` | RNN | 0.6517 ± 0.0461 | 0.7836 | 0.7226 | 0.9449 | 0.9004 | 853,059 | 169.67 | 3.40 |
| `lstm` | RNN | 0.5585 ± 0.1212 | 0.5934 | 0.7346 | 0.8998 | 0.7703 | 457,155 | 81.48 | 2.89 |

### Track 3: OCT 3-Class Ordered Sequence (`oct_sequence`, 25 runs, Test `n=23` windows across `4` series)

| Model | Category | Macro F1 (Mean ± Std) | Accuracy | Bal. Acc | ROC-AUC | PR-AUC | Params | Train Time (s) | Latency (ms/window) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`resnet50_bilstm`** | Hybrid | **0.9849 ± 0.0338** | **0.9913** | **0.9778** | **1.0000** | **1.0000** | 24,439,747 | 72.34 | 29.67 |
| `efficientnet_b0_bilstm` | Hybrid | 0.9142 ± 0.1138 | 0.9652 | 0.9111 | **1.0000** | **1.0000** | 4,748,351 | 61.09 | 24.80 |
| `resnet50_lstm` | Hybrid | 0.8608 ± 0.2046 | 0.9565 | 0.8889 | **1.0000** | **1.0000** | 24,241,475 | 70.74 | 28.83 |
| `bilstm` | RNN | 0.3385 ± 0.1024 | 0.7565 | 0.3815 | 0.5362 | 0.5706 | 853,059 | 31.69 | 18.58 |
| `lstm` | RNN | 0.2760 ± 0.0279 | 0.6870 | 0.3111 | 0.4799 | 0.4612 | **457,155** | **28.37** | **18.35** |

### Track 4: OCT Exploratory Severity Grading (`oct_severity`, 50 runs, Test `n=176`)

| Model | Category | Macro F1 (Mean ± Std) | Accuracy | Bal. Acc | ROC-AUC | PR-AUC | Params | Train Time (s) | Latency (ms/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`resnet50_lstm`** | Hybrid | **0.5274 ± 0.0780** | 0.7250 | 0.5157 | 0.7749 | 0.6542 | 24,241,604 | 79.32 | 3.80 |
| `efficientnet_b0` | CNN | 0.5042 ± 0.0941 | 0.6875 | **0.5387** | **0.8326** | **0.6639** | 4,012,096 | 72.21 | 3.24 |
| `resnet50_bilstm` | Hybrid | 0.4789 ± 0.0717 | **0.7477** | 0.4945 | 0.7247 | 0.5869 | 24,440,004 | 124.91 | 4.13 |
| `densenet121` | CNN | 0.4683 ± 0.0766 | 0.6625 | 0.4982 | 0.6589 | 0.5672 | 6,951,684 | 114.48 | 3.92 |
| `resnet50` | CNN | 0.4495 ± 0.0774 | 0.6591 | 0.4270 | 0.6269 | 0.5235 | 23,509,956 | 77.49 | 3.87 |
| `efficientnet_b0_bilstm` | Hybrid | 0.3986 ± 0.0274 | 0.6773 | 0.4894 | 0.7986 | 0.6310 | 4,748,608 | 76.45 | 3.46 |
| `autoencoder_classifier` | Baseline | 0.3643 ± 0.0193 | 0.5125 | 0.4427 | 0.6732 | 0.4432 | 717,381 | **48.91** | 2.54 |
| `mlp` | Baseline | 0.2486 ± 0.0185 | 0.5170 | 0.3268 | 0.6865 | 0.3113 | **100,232** | 58.66 | **2.51** |
| `lstm` | RNN | 0.1808 ± 0.0478 | 0.4761 | 0.2099 | 0.4556 | 0.2519 | 457,284 | 49.06 | 2.93 |
| `bilstm` | RNN | 0.1476 ± 0.0330 | 0.3386 | 0.1888 | 0.4132 | 0.2222 | 853,316 | 108.08 | 3.41 |

---

## 10. Statistical Analysis & Methodological Transparency

Implemented in [`src/statistics/`](src/statistics/), producing outputs under [`results/statistical_analysis/`](results/statistical_analysis/):

1. **Omnibus Friedman Tests (Macro F1 across `n=5` seeds):** Significant overall differences across architectures in all 4 tracks:
   * `fundus_3class`: `χ² = 38.2527`, `p = 1.56e-5`, Kendall's `W = 0.8501`
   * `oct_individual`: `χ² = 36.0996`, `p = 3.80e-5`, Kendall's `W = 0.8022`
   * `oct_sequence`: `χ² = 16.6383`, `p = 2.27e-3`, Kendall's `W = 0.8319`
   * `oct_severity`: `χ² = 36.8836`, `p = 2.76e-5`, Kendall's `W = 0.8196`
2. **Hypothesis & Pairwise Comparison Scope:**
   * [`hypothesis_tests.csv`](results/statistical_analysis/hypothesis_tests.csv): **`149`** tests recorded (`4` Friedman omnibus + `145` sample-level `seed42` McNemar tests across tracks), **`87`** significant at Holm-Bonferroni `α = 0.05`.
   * [`pairwise_comparisons.csv`](results/statistical_analysis/pairwise_comparisons.csv): **`435`** pairwise rows (`145` paired t-tests, `145` Wilcoxon signed-rank tests, and `145` McNemar tests), **`152`** significant after within-track/family Holm-Bonferroni adjustment (`69` paired t-tests, `83` McNemar tests, `0` Wilcoxon tests).
3. **Important Statistical Caveats Documented by the Pipeline:**
   * **Seeds ≠ Independent Patient Cohorts:** Multi-seed Friedman, paired t-tests, and Wilcoxon tests evaluate optimization variance on fixed splits rather than generalization to independent clinical cohorts.
   * **Small-Sample Wilcoxon Floor (`n=5`):** A two-sided Wilcoxon signed-rank test on `5` paired seeds has a mathematical minimum `p = 0.0625` (> `0.05` even before Holm correction), yielding `0/145` significant Wilcoxon comparisons despite large effect sizes.
   * **Shapiro-Wilk Normality Violations:** **`13`** of the `145` paired t-tests flagged `Shapiro-Wilk p < 0.05` on seed-level differences.
   * **Clustered OCT Test Errors & Grouped Bootstrap CIs (`n=2000`):** Sample-level McNemar assumes independent test errors (violated on correlated OCT B-scans from few sequences); [`confidence_intervals.csv`](results/statistical_analysis/confidence_intervals.csv) uses grouped cluster/sequence resampling and explicitly warns when OCT test classes have `< 2` independent groups (`min_groups = 1`).

---

## 11. Error Analysis & Clinical Failure Modes

Across all `5` random seeds pooled on the untouched test splits:

* **Fundus 3-Class (`efficientnet_b0`, `1,025` pooled predictions = `205 × 5`):** Only **`12` errors (`1.17%`)** occurred across all 5 seeds — **`100%` of errors occurred in `Pseudopapilledema`** (`4` predicted `Normal`, `8` predicted `Papilledema`), while `Normal` (`585/585`) and `Papilledema` (`220/220`) achieved `100%` recall. Confusing benign pseudopapilledema with papilledema reflects clinically realistic disc-elevation ambiguity.
* **OCT Individual (`densenet121`, `915` pooled predictions = `183 × 5`):** Only **`2` errors (`0.22%`)** across all 5 seeds (`2` `Normal` slices predicted as `Pseudopapilledema`; `Papilledema 730/730` and `Pseudopapilledema 85/85`). Note that high accuracy partly reflects limited subject/series diversity in Normal (`1` prefix) and Pseudopapilledema (`1` prefix).
* **OCT Sequence (`resnet50_bilstm`, `115` pooled window predictions = `23 × 5`):** Only **`1` error (`0.87%`)** (`1` `Normal` window predicted as `Papilledema`); however, the test set contains only `23` windows from `4` sequences (`18` Papilledema, `3` Normal, `2` Pseudopapilledema).
* **OCT Severity Grading (`oct_severity`, `880` pooled predictions = `176 × 5`):** Severe **G3 class collapse** occurs despite inverse-frequency weighting: **`4` out of `10` models (`densenet121`, `resnet50`, `resnet50_bilstm`, `lstm`) scored `0/65` G3 recall** across all 5 seeds pooled (every G3 test image misclassified into G2), and **`2` models (`efficientnet_b0_bilstm`, `mlp`) scored `0/55` G4 recall**. Even the best Macro F1 model (`resnet50_lstm`, `0.5274`) recovered only `2/65` G3 and `12/55` G4 pooled samples while achieving `242/880` (`27.50%`) total errors.

---

## 12. Figures & Visual Artifacts

[`results/figures/`](results/figures/) contains **57 PNG figures** covering all tracks (`fundus_3class_*`, `oct_individual_*`, `oct_sequence_*`, `oct_severity_*`) plus track-summary visualizations:

* `dataset_class_distribution.png` — raw/retained dataset counts across tracks
* `*_comparison.png` / `cnn_comparison.png` / `rnn_comparison.png` / `hybrid_comparison.png` / `severity_comparison.png` — Macro F1 & Accuracy bar charts with error bars
* `*_training_loss_curves.png`, `*_validation_loss_curves.png`, `*_accuracy_curves.png`, `*_macro_f1_curves.png` — `30`-epoch learning trajectories
* `*_confusion_matrices.png`, `*_per_class_performance.png`, `*_roc_curves.png`, `*_pr_curves.png` — representative-seed (`seed42`) diagnostic curves and CMs
* `*_statistical_significance_heatmap.png`, `confidence_intervals.png` — Holm-adjusted significance matrices and grouped bootstrap `95%` CIs

> **Note:** In [`results/figures/`](results/figures/), `10` unprefixed filenames (e.g., `roc_curves.png`, `confusion_matrices.png`) duplicate the `fundus_3class_*` default plots, and `parameter_comparison.png`, `training_time_comparison.png`, and `inference_time_comparison.png` visualize the `fundus_3class` track (`computational_comparison.csv` contains full runtime metrics for all tracks).

---

## 13. Repository Structure

```text
Papilledema_analysis/
├── README.md                                  # Complete project documentation
├── requirements.txt                           # Python package dependencies
├── .gitignore                                 # Excludes raw data, checkpoints, logs, predictions, caches
├── main.py                                    # Unified CLI entrypoint
├── configs/
│   ├── base.yaml                              # Seeds, paths, training hyperparameters, evaluation settings
│   ├── fundus.yaml                            # Fundus dataset settings & near-duplicate thresholds
│   ├── oct.yaml                               # OCT 3-class, sequence window, and severity configs
│   ├── models.yaml                            # 10 model architecture hyperparameters
│   └── experiments.yaml                       # Track-to-model matrices for all 4 tracks
├── src/
│   ├── data/                                  # Validation, duplicate clustering, grouped splits, datasets, transforms
│   ├── models/                                # CNN, RNN, Hybrid CNN–RNN, MLP/AE baselines, model factory
│   ├── training/                              # Losses, trainer (AMP + early stopping), checkpointing, suite runner
│   ├── evaluation/                            # Metrics, test evaluator, error flagging
│   ├── experiments/                           # Experiment registry, state manager, automated report/plot export
│   ├── statistics/                            # Descriptive stats, grouped bootstrap CIs, Friedman, McNemar, Holm
│   ├── visualization/                         # Learning curves, CMs, ROC/PR, CI, and significance heatmap plots
│   └── utils/                                 # Config loader, CUDA/RTX 5050 check, logging, deterministic seeding
├── scripts/                                   # Standalone CLI runners (verify, split, train, suite, evaluate, stats, report, GPU check)
├── tests/                                     # Unit tests for data/leakage, models/training, metrics/stats/viz/notebooks
├── notebooks/
│   ├── Papilledema_Complete_Research_Analysis.ipynb       # Central consolidated analysis & results notebook (36 cells)
│   ├── Papilledema_Complete_Research_Experiments.ipynb    # Experiment orchestration notebook (32 cells)
│   └── Papilledema_Data_Exploration.ipynb                 # Dataset & CLAHE/sequence inspection notebook (10 cells)
├── results/
│   ├── tables/                                # Model comparison, per-track CSVs, computational comparison, dataset summary
│   ├── statistical_analysis/                  # Descriptive stats, CIs, hypothesis tests, pairwise comparisons, effect sizes
│   ├── reports/                               # Markdown summaries + notebook export manifest
│   └── figures/                               # 57 PNG charts and diagnostic plots
└── docs/
    └── README.md                              # Placeholder & pointer to research outputs / future paper PDF
```

*(Local-only gitignored folders generated when running locally: `data/raw/`, `data/manifests/*.csv`, `data/splits/*.csv`, `checkpoints/`, `experiments/`, `logs/`, `results/predictions/`.)*

---

## 14. Environment & Installation

```bash
# 1. Clone the repository
git clone https://github.com/shuvendu-kumar-mohapatra/papilledema-analysis.git
cd papilledema-analysis

# 2. Create and activate a Python 3.12 virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# 3. Install PyTorch + Torchvision matching your CUDA version (see https://pytorch.org/get-started/locally/)
# Example (CUDA 12.x / 13.x or CPU):
pip install torch torchvision

# 4. Install remaining research dependencies
pip install -r requirements.txt

# 5. Verify environment and GPU compatibility
python main.py verify-env
```

---

## 15. Usage & CLI Commands

All pipeline stages can be invoked via [`main.py`](main.py) or [`scripts/`](scripts/):

```bash
# 1. Verify hardware, CUDA, and installed packages
python main.py verify-env
python scripts/verify_environment.py

# 2. Verify raw Fundus & OCT datasets, detect duplicates & IR/SLO artifacts (requires raw data in data/raw/)
python main.py verify-data
python scripts/verify_data.py

# 3. Build and validate leakage-aware grouped splits
python main.py splits
python scripts/create_splits.py --dataset all --force-verify

# 4. Run real-data GPU forward/backward/checkpoint check across all track-model pairs
python scripts/verify_models_gpu.py

# 5. Run a 1-epoch smoke test (tags outputs with *__smoke)
python main.py smoke --track fundus_3class
python scripts/run_experiments.py --dataset fundus --smoke --epochs 1

# 6. Train a single model on a specific track
python scripts/train.py --dataset fundus --model efficientnet_b0 --seed 42
python scripts/train.py --dataset oct --task sequence --model resnet50_bilstm --seed 42 --batch-size 4

# 7. Execute multi-seed benchmark suites
python main.py run --track fundus_3class
python scripts/run_experiments.py --dataset all --multi-seed

# 8. Evaluate a completed checkpoint or re-run statistical analysis & report export
python scripts/evaluate.py --experiment_id fundus_3class__efficientnet_b0__seed42
python scripts/statistical_analysis.py --dataset all
python main.py report
python scripts/generate_report.py
```

---

## 16. Notebook Workflow Guide

| Notebook | Purpose | Notes |
|---|---|---|
| [`notebooks/Papilledema_Complete_Research_Analysis.ipynb`](notebooks/Papilledema_Complete_Research_Analysis.ipynb) | **Central consolidated notebook** (`36` cells) | Presents the complete dataset verification, grouped split audit, model architectures, 175-run multi-seed results, statistical tests, error analysis, and embedded visualizations. |
| [`notebooks/Papilledema_Complete_Research_Experiments.ipynb`](notebooks/Papilledema_Complete_Research_Experiments.ipynb) | End-to-end execution notebook (`32` cells) | Interactive orchestration of data auditing, split generation, GPU checks, training, and report export. |
| [`notebooks/Papilledema_Data_Exploration.ipynb`](notebooks/Papilledema_Data_Exploration.ipynb) | Dataset exploratory view (`10` cells) | Inspects Fundus CLAHE vs. RGB contrast, near-duplicate clusters, and ordered OCT B-scan series. |

---

## 17. Testing & Verification

Run the unit test suite (requires no external GPU hardware and cleanly skips local raw-split assertions if `data/splits/*.csv` are absent):

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Tests verify ([`tests/`](tests/)):
* [`test_data_and_splits.py`](tests/test_data_and_splits.py): RGB/CLAHE and grayscale preprocessing, sequence dataset windowing & RNG progression, IR/SLO detection, grouped split generation, synthetic duplicate/augmented-path leakage exceptions, and local manifest integrity.
* [`test_models_and_training.py`](tests/test_models_and_training.py): Forward/backward passes for all `10` architectures in `1`-channel and `3`-channel modes, masked padding isolation for sequence RNN/Hybrid models, `frozen`/`partial`/`full` backbone modes, class weights, autoencoder pretraining, and atomic checkpoint resume.
* [`test_eval_stats_viz_nb.py`](tests/test_eval_stats_viz_nb.py): Classification metric formulas, grouped bootstrap CIs, Holm-Bonferroni, Cohen's *d*, rank-biserial correlation, McNemar's test, track statistical pipeline, figure plotting functions, non-overwrite smoke isolation in `ExperimentManager`, and notebook validity without duplicated core definitions.

---

## 18. Limitations & Clinical Caveats

1. **Lack of Verified Patient IDs:** Fundus evaluation groups near-duplicate image clusters (`1,270` clusters), not clinically verified patient identities. Original OCT lacks verified patient IDs (`OCT Data with Patient ID` is explicitly excluded); sequences (`21` inferred series across `12` filename prefixes) were grouped heuristically.
2. **Single-Prefix OCT Classes:** In `oct_individual` / `oct_sequence`, `Normal` (`1` subject prefix) and `Pseudopapilledema` (`1` subject prefix) use contiguous spatial-block splits across train/val/test rather than subject-independent splits; the same holds for `G1` and `G4` in `oct_severity`.
3. **Small Sequence Test Set:** `oct_sequence` evaluates `23` windows derived from only `4` inferred test sequences (`18` Papilledema, `3` Normal, `2` Pseudopapilledema windows), producing wide confidence intervals despite `0.9849` mean Macro F1.
4. **Unvalidated Exploratory Severity Grading:** Frisén grades (`G1–G4`) lack clinical inter-grader documentation in the dataset, suffer extreme imbalance (`40` G1 vs. `598` G2), and exhibit G3/G4 recall collapse in multiple models.
5. **Statistical Scope:** Multi-seed tests (`n=5`) quantify training/seed variance on fixed splits, not independent clinical population generalization; `n=5` limits two-sided Wilcoxon power (`min p = 0.0625`), and `13/145` paired t-tests showed Shapiro-Wilk non-normality.
6. **Non-Diagnostic Research Use Only:** Results cannot be interpreted as clinically validated screening or diagnostic performance.

---

## 19. Future Work

* Validate on **multi-center, prospectively curated cohorts** with verified anonymized patient IDs and multi-grader clinical Frisén annotations.
* Address G1/G3/G4 severity imbalance via ordinal classification losses, focal/logit-adjusted objectives, and multi-patient sampling rather than single-slice cross-entropy.
* Evaluate multimodal fusion (paired Fundus + OCT B-scan volumes per eye) and calibration/uncertainty estimation under external distribution shift.
* Add external test-set cross-institutional generalization and Grad-CAM / saliency interpretability verification against optic disc margins and RNFL thickness regions.

---

## 20. Author, License & Citation

* **Author:** Shuvendu
* **Repository:** [github.com/shuvendu-kumar-mohapatra/papilledema-analysis](https://github.com/shuvendu-kumar-mohapatra/papilledema-analysis)
* **Code License:** No open-source `LICENSE` file has been selected yet; all rights remain reserved unless a `LICENSE` file is added prior to public release. External medical imaging datasets are **not redistributed** and remain subject to their original providers' terms.

### Suggested Citation

```bibtex
@misc{shuvendu2026papilledema,
  author       = {Shuvendu},
  title        = {Comparative Analysis of CNN, RNN, and Hybrid CNN-RNN Architectures for Papilledema Detection and Severity Grading},
  year         = {2026},
  howpublished = {\url{https://github.com/shuvendu-kumar-mohapatra/papilledema-analysis}}
}
```
