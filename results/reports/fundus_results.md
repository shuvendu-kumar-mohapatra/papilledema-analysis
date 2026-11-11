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
