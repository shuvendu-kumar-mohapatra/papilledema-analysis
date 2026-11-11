# Final Research Summary

Fundus and OCT experimental tracks are evaluated and ranked strictly independently.

# Kim Fundus Classification Results

| Model | Seeds | Mean Macro F1 | Mean Accuracy | Mean Macro Recall | Params | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| efficientnet_b0 | 5 | 0.9835 | 0.9883 | 0.9818 | 4,011,391 | 2.54 |
| efficientnet_b0_bilstm | 5 | 0.9806 | 0.9854 | 0.9820 | 4,748,927 | 2.57 |
| densenet121 | 5 | 0.9795 | 0.9844 | 0.9786 | 6,956,931 | 3.25 |
| resnet50 | 5 | 0.9779 | 0.9834 | 0.9771 | 23,514,179 | 3.10 |
| resnet50_bilstm | 5 | 0.9779 | 0.9834 | 0.9761 | 24,446,019 | 3.09 |
| resnet50_lstm | 5 | 0.9684 | 0.9766 | 0.9684 | 24,247,747 | 3.02 |
| autoencoder_classifier | 5 | 0.8691 | 0.8937 | 0.8660 | 718,118 | 1.72 |
| mlp | 5 | 0.8018 | 0.8420 | 0.8056 | 233,231 | 2.02 |
| lstm | 5 | 0.7522 | 0.7776 | 0.7727 | 720,899 | 2.28 |
| bilstm | 5 | 0.7230 | 0.7522 | 0.7456 | 1,116,803 | 2.78 |

## Highlights
- Best by Macro F1: **efficientnet_b0** (0.9835)
- Best by Accuracy: **efficientnet_b0** (0.9883)
- Best by Macro Recall: **efficientnet_b0_bilstm** (0.9820)
- Fastest Inference: **autoencoder_classifier** (1.72 ms/sample)

## Caveats & Limitations
- Verified patient IDs are unavailable.
- Repeated and near-duplicate eye images are grouped into clusters kept intact across splits.
- Exact cross-class duplicates with conflicting labels are excluded from training/evaluation.
- Cannot be described as verified patient-independent evaluation.

---

# Original OCT 3-Class Results — Individual B-Scan (image mode)

| Model | Seeds | Mean Macro F1 | Mean Accuracy | Mean Macro Recall | Params | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| densenet121 | 5 | 0.9928 | 0.9978 | 0.9933 | 6,950,659 | 3.89 |
| efficientnet_b0 | 5 | 0.9903 | 0.9945 | 0.9833 | 4,010,815 | 3.25 |
| resnet50_bilstm | 5 | 0.9863 | 0.9934 | 0.9858 | 24,439,747 | 3.87 |
| efficientnet_b0_bilstm | 5 | 0.9807 | 0.9913 | 0.9820 | 4,748,351 | 3.29 |
| resnet50 | 5 | 0.9781 | 0.9902 | 0.9700 | 23,507,907 | 3.83 |
| resnet50_lstm | 5 | 0.9737 | 0.9869 | 0.9658 | 24,241,475 | 3.77 |
| autoencoder_classifier | 5 | 0.7317 | 0.8426 | 0.7933 | 717,252 | 2.33 |
| mlp | 5 | 0.7164 | 0.8678 | 0.6994 | 100,103 | 2.43 |
| bilstm | 5 | 0.6517 | 0.7836 | 0.7226 | 853,059 | 3.40 |
| lstm | 5 | 0.5585 | 0.5934 | 0.7346 | 457,155 | 2.89 |

## Highlights
- Best by Macro F1: **densenet121** (0.9928)
- Best by Accuracy: **densenet121** (0.9978)
- Best by Macro Recall: **densenet121** (0.9933)
- Fastest Inference: **autoencoder_classifier** (2.33 ms/sample)

# Original OCT 3-Class Results — Ordered Sequence (sequence mode)

| Model | Seeds | Mean Macro F1 | Mean Accuracy | Mean Macro Recall | Params | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| resnet50_bilstm | 5 | 0.9849 | 0.9913 | 0.9778 | 24,439,747 | 29.67 |
| efficientnet_b0_bilstm | 5 | 0.9142 | 0.9652 | 0.9111 | 4,748,351 | 24.80 |
| resnet50_lstm | 5 | 0.8608 | 0.9565 | 0.8889 | 24,241,475 | 28.83 |
| bilstm | 5 | 0.3385 | 0.7565 | 0.3815 | 853,059 | 18.58 |
| lstm | 5 | 0.2760 | 0.6870 | 0.3111 | 457,155 | 18.35 |

## Highlights
- Best by Macro F1: **resnet50_bilstm** (0.9849)
- Best by Accuracy: **resnet50_bilstm** (0.9913)
- Best by Macro Recall: **resnet50_bilstm** (0.9778)
- Fastest Inference: **lstm** (18.35 ms/sample)

## OCT Caveats & Limitations
- 1 IR/SLO non-B-scan image (heshama_001.jpg) is excluded, leaving 1,140 B-scans.
- Sequence IDs are inferred from filenames and visual continuity; never treated as verified patients.
- Normal (sabrenr) and Pseudopapilledema (heshama) each have only 1 inferred subject prefix (2 series); G1 and G4 each have 1 inferred prefix.
- No reliable clinical documentation validates Frisén G1-G4 severity grades; severity grading is strictly exploratory.
- Fabricated patient IDs and augmented folders are strictly excluded.

---

# Original OCT Exploratory Severity Grading (G1-G4)

| Model | Seeds | Mean Macro F1 | Mean Accuracy | Mean Macro Recall | Params | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| resnet50_lstm | 5 | 0.5274 | 0.7250 | 0.5157 | 24,241,604 | 3.80 |
| efficientnet_b0 | 5 | 0.5042 | 0.6875 | 0.5387 | 4,012,096 | 3.24 |
| resnet50_bilstm | 5 | 0.4789 | 0.7477 | 0.4945 | 24,440,004 | 4.13 |
| densenet121 | 5 | 0.4683 | 0.6625 | 0.4982 | 6,951,684 | 3.92 |
| resnet50 | 5 | 0.4495 | 0.6591 | 0.4270 | 23,509,956 | 3.87 |
| efficientnet_b0_bilstm | 5 | 0.3986 | 0.6773 | 0.4894 | 4,748,608 | 3.46 |
| autoencoder_classifier | 5 | 0.3643 | 0.5125 | 0.4427 | 717,381 | 2.54 |
| mlp | 5 | 0.2486 | 0.5170 | 0.3268 | 100,232 | 2.51 |
| lstm | 5 | 0.1808 | 0.4761 | 0.2099 | 457,284 | 2.93 |
| bilstm | 5 | 0.1476 | 0.3386 | 0.1888 | 853,316 | 3.41 |

## Highlights
- Best by Macro F1: **resnet50_lstm** (0.5274)
- Best by Accuracy: **resnet50_bilstm** (0.7477)
- Best by Macro Recall: **efficientnet_b0** (0.5387)
- Fastest Inference: **mlp** (2.51 ms/sample)

## Caveats & Limitations
- Strictly exploratory: no verified clinical documentation validating Frisén severity grades.
- G1 (hagerh) and G4 (mostafaa) each have only 1 inferred subject prefix; evaluation is NOT patient-independent.

---
## Statistical Findings
- Hypothesis tests recorded: 149
- Significant after Holm-Bonferroni (alpha=0.05): 87
- Reminder: Multi-seed tests assess training stochasticity on fixed splits, not patient-cohort generalization.