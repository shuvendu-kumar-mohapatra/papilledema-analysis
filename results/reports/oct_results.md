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
