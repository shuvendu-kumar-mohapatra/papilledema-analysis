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
