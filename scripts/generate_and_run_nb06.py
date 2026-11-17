"""Generate and execute notebooks/06_Ablation_Study.ipynb."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nbformat as nbf
from src.utils.notebook_runner import execute_notebook


def build_nb06() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    }
    cells = []

    # Title & Executive Summary
    cells.append(nbf.v4.new_markdown_cell("""# Notebook 06: Controlled Systematic Ablation Studies

## Experimental Scope & Objectives
This notebook presents five controlled ablation investigations across the four experimental tracks, isolating the causal impact of architectural components, feature representation, trainability, and temporal context under identical 5-fold cross-validation splits:

1. **Ablation 1: CNN Backbone Architectures** — Comparing ResNet50 (residual identity shortcuts), DenseNet121 (dense feature concatenation), and EfficientNet-B0 (compound scaling with inverted bottlenecks).
2. **Ablation 2: RNN Directionality** — Comparing unidirectional LSTM with bidirectional BiLSTM under identical spatial patch/slice features.
3. **Ablation 3: Hybrid CNN–RNN Coupling vs. Unimodal CNNs** — Evaluating whether sequential feature aggregation improves performance over static spatial representations (ResNet50 vs ResNet50+LSTM vs ResNet50+BiLSTM; EfficientNet-B0 vs EfficientNet-B0+BiLSTM).
4. **Ablation 4: Backbone Feature Trainability** — Evaluating transfer learning depth under identical folds:
   - `frozen`: Feature extraction only; backbone weights locked, only the classification head trained.
   - `partial`: Top two convolutional blocks fine-tuned; bottom features frozen.
   - `full`: Complete end-to-end backpropagation through all layers.
   - Evaluated on Fundus (EfficientNet-B0) and Individual OCT (DenseNet121).
5. **Ablation 5: OCT Temporal Window Length** — Evaluating the effect of volumetric slice context on OCT sequence classification using ResNet50+BiLSTM:
   - Window lengths: $L=4$, $L=8$ (baseline), and $L=12$ slices per window.
   - Evaluated at both the **window level** and the **patient sequence level** ($N=21$) via out-of-fold probability pooling.
6. **Efficiency & Computational Tradeoffs** — Synthesizing parameter count, GPU training runtime, and classification Macro-F1 across all architectures.
"""))

    # Imports & Setup
    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
from IPython.display import display, HTML, Image

PROJECT_ROOT = Path("..").resolve()
ABL_DIR = PROJECT_ROOT / "results" / "ablation_study"
FIG_DIR = ABL_DIR / "figures"
print(f"Loading ablation artifacts from: {ABL_DIR}")
"""))

    # Section 1: Ablation 1 - CNN Backbone
    cells.append(nbf.v4.new_markdown_cell("""## 1. Ablation 1: CNN Backbone Comparison Across Experimental Tracks

Comparing the three core convolutional backbones (ResNet50, DenseNet121, EfficientNet-B0) across all 4 experimental tracks under identical 5-fold CV splits.
"""))

    cells.append(nbf.v4.new_code_cell("""df_abl1 = pd.read_csv(ABL_DIR / "abl01_cnn_backbone_comparison.csv")
display(HTML("<h4>Ablation 1: CNN Backbone Comparison (5-Fold Mean ± SD)</h4>"))
cols_abl = ["track", "model", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "accuracy_mean", "mcc_mean", "total_params", "train_time_mean_sec"]
display(HTML(df_abl1[cols_abl].to_html(classes="table table-striped table-hover", index=False)))

display(HTML("<h5>Ablation 1 Figure: CNN Backbone Comparison</h5>"))
display(Image(filename=str(FIG_DIR / "abl01_cnn_backbone_comparison.png")))
"""))

    # Section 2: Ablation 2 - LSTM vs BiLSTM
    cells.append(nbf.v4.new_markdown_cell("""## 2. Ablation 2: RNN Directionality (Unidirectional LSTM vs. Bidirectional BiLSTM)

Evaluating the benefit of bidirectional context propagation over unidirectional recurrent processing.
"""))

    cells.append(nbf.v4.new_code_cell("""df_abl2 = pd.read_csv(ABL_DIR / "abl02_rnn_lstm_vs_bilstm.csv")
display(HTML("<h4>Ablation 2: LSTM vs BiLSTM Comparison (5-Fold Mean ± SD)</h4>"))
display(HTML(df_abl2[cols_abl].to_html(classes="table table-striped table-hover", index=False)))

display(HTML("<h5>Ablation 2 Figure: LSTM vs BiLSTM Comparison</h5>"))
display(Image(filename=str(FIG_DIR / "abl02_rnn_lstm_vs_bilstm.png")))
"""))

    # Section 3: Ablation 3 - Hybrid CNN-RNN vs CNN-Only
    cells.append(nbf.v4.new_markdown_cell("""## 3. Ablation 3: Hybrid CNN–RNN Architectures vs. Pure CNNs

Analyzing whether recurrent temporal/patch pooling atop convolutional feature extractors outperforms spatial average pooling alone on 2D images.
"""))

    cells.append(nbf.v4.new_code_cell("""df_abl3 = pd.read_csv(ABL_DIR / "abl03_hybrid_cnn_rnn_comparison.csv")
display(HTML("<h4>Ablation 3: Hybrid vs CNN-Only Architectures (5-Fold Mean ± SD)</h4>"))
display(HTML(df_abl3[cols_abl].to_html(classes="table table-striped table-hover", index=False)))

display(HTML("<h5>Ablation 3 Figure: Hybrid vs CNN Comparison</h5>"))
display(Image(filename=str(FIG_DIR / "abl03_hybrid_cnn_rnn_comparison.png")))
"""))

    # Section 4: Ablation 4 - Backbone Trainability
    cells.append(nbf.v4.new_markdown_cell("""## 4. Ablation 4: Controlled CNN Backbone Trainability (Frozen vs. Partial vs. Full)

Quantifying the exact performance gain achieved by fine-tuning the deep backbone weights compared to linear probing on frozen ImageNet features and partial block tuning.
"""))

    cells.append(nbf.v4.new_code_cell("""df_abl4 = pd.read_csv(ABL_DIR / "abl04_backbone_trainability_folds.csv")
agg_abl4 = df_abl4.groupby(["track", "model", "backbone_mode"]).agg(
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_std=("macro_f1", "std"),
    bal_acc_mean=("balanced_accuracy", "mean"),
    acc_mean=("accuracy", "mean"),
    trainable_params=("trainable_params", "first"),
    total_params=("total_params", "first"),
    mean_duration_sec=("train_duration_sec", "mean")
).reset_index()

agg_abl4["trainable_pct"] = (100.0 * agg_abl4["trainable_params"] / agg_abl4["total_params"]).round(1)
display(HTML("<h4>Ablation 4: Backbone Trainability Summary (5-Fold Mean ± SD)</h4>"))
display(HTML(agg_abl4.to_html(classes="table table-striped table-hover", index=False)))

display(HTML("<h5>Ablation 4 Figure: Backbone Trainability Impact</h5>"))
display(Image(filename=str(FIG_DIR / "abl04_backbone_trainability_comparison.png")))
"""))

    # Section 5: Ablation 5 - OCT Sequence Length
    cells.append(nbf.v4.new_markdown_cell("""## 5. Ablation 5: Controlled OCT Sequence Window Length ($L=4$ vs $L=8$ vs $L=12$)

Evaluating how the volumetric temporal window span affects both window-level and patient sequence-level classification performance with ResNet50+BiLSTM.
"""))

    cells.append(nbf.v4.new_code_cell("""df_abl5_folds = pd.read_csv(ABL_DIR / "abl05_oct_sequence_length_folds.csv")
agg_abl5 = df_abl5_folds.groupby("seq_len").agg(
    n_windows_oof=("n_windows", "sum"),
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_std=("macro_f1", "std"),
    acc_mean=("accuracy", "mean"),
    mcc_mean=("mcc", "mean"),
    mean_duration_sec=("train_duration_sec", "mean")
).reset_index()

display(HTML("<h4>Ablation 5: Window-Level Performance by Sequence Length (5-Fold Mean ± SD)</h4>"))
display(HTML(agg_abl5.to_html(classes="table table-striped table-hover", index=False)))

df_abl5_lvl = pd.read_csv(ABL_DIR / "abl05_oct_sequence_length_seq_level.csv")
display(HTML("<h4>Ablation 5: Patient Sequence-Level Out-of-Fold Performance (N=21 Sequences)</h4>"))
display(HTML(df_abl5_lvl.to_html(classes="table table-striped table-hover", index=False)))

display(HTML("<h5>Ablation 5 Figure: Window & Sequence-Level Length Comparison</h5>"))
display(Image(filename=str(FIG_DIR / "abl05_oct_sequence_length_comparison.png")))
"""))

    # Section 6: Efficiency Tradeoff
    cells.append(nbf.v4.new_markdown_cell("""## 6. Synthesis: Parameter Complexity vs Training Duration vs Performance

Tradeoff scatter visualization illustrating the efficiency frontiers across architecture paradigms and modalities.
"""))

    cells.append(nbf.v4.new_code_cell("""display(HTML("<h5>Ablation 6: Efficiency & Compute Tradeoff Overview</h5>"))
display(Image(filename=str(FIG_DIR / "abl06_ablation_efficiency_tradeoff.png")))
"""))

    nb.cells = cells
    nb_path = PROJECT_ROOT / "notebooks" / "06_Ablation_Study.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {nb_path}")
    return nb_path


if __name__ == "__main__":
    p = build_nb06()
    execute_notebook(p, timeout_sec=180)
    print("Notebook 06 execution completed successfully!")

