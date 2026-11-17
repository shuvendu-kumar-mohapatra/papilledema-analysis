"""Generate and execute notebooks/04_Model_Evaluation_and_Visualization.ipynb."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nbformat as nbf
from src.utils.notebook_runner import execute_notebook


def build_nb04() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    }
    cells = []

    # Title & Executive Summary
    cells.append(nbf.v4.new_markdown_cell("""# Notebook 04: Comprehensive Model Evaluation and Visualization

## Executive Summary & Experimental Tracks
This notebook presents the complete cross-validation results, calibration diagnostics, confusion matrices, and comparative visualizations across all four experimental tracks:
1. **EXP-01: Fundus 3-Class Classification** (10 models, 50 CV runs)
2. **EXP-02: Individual OCT 3-Class Classification** (10 models, 50 CV runs)
3. **EXP-03: OCT Sequence 3-Class Classification** (5 models, 25 CV runs, window-level and sequence-level)
4. **EXP-04: OCT Severity Grading** (10 models, 50 CV runs, exploratory 4-grade ordinal/nominal)

All evaluations are conducted out-of-fold (OOF) across the 5 outer test splits to ensure completely unbiased generalization estimates.

### Methodological Notes & Limitations
- **Fundus (`StratifiedGroupKFold`)**: Strict grouping by duplicate cluster ID prevents near-duplicate leakage.
- **Individual OCT & Severity**: Papilledema / Grade 2 grouped by subject prefix; Normal / Pseudopapilledema / G1 / G3 / G4 split via deterministic slice blocks. *Explicit Limitation: Non-patient-independent validation due to single-source sequence prefixes.*
- **OCT Sequence**: Evaluated both at the window level ($N=146$) and at the full sequence level ($N=21$ sequences) via out-of-fold probability averaging.
"""))

    # Imports & Setup
    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
from IPython.display import display, HTML, Image

PROJECT_ROOT = Path("..").resolve()
EVAL_DIR = PROJECT_ROOT / "results" / "evaluation"
FIG_DIR = EVAL_DIR / "figures"
print(f"Loading evaluation artifacts from: {EVAL_DIR}")
"""))

    # Track 1: Fundus 3-Class
    cells.append(nbf.v4.new_markdown_cell("""## 1. EXP-01: Fundus 3-Class Classification

Evaluating 10 architectures (CNNs, RNNs, Hybrids, MLP, Autoencoder) under 5-fold `StratifiedGroupKFold`.
"""))

    cells.append(nbf.v4.new_code_cell("""df_fundus = pd.read_csv(EVAL_DIR / "cv_summary_fundus_3class.csv")
display(HTML("<h4>EXP-01 Fundus Performance Summary (5-Fold Mean ± SD)</h4>"))
cols_show = ["model", "category", "macro_f1_mean", "macro_f1_std", "accuracy_mean", "balanced_accuracy_mean", "cohen_kappa_mean", "mcc_mean", "train_time_mean_sec", "peak_gpu_mem_mb_max"]
display(HTML(df_fundus[cols_show].to_html(classes="table table-striped table-hover", index=False)))
"""))

    cells.append(nbf.v4.new_code_cell("""display(HTML("<h5>Model Performance & Multi-Metric Comparison</h5>"))
display(Image(filename=str(FIG_DIR / "fig_fundus_3class_01_model_comparison.png")))
display(HTML("<h5>Confusion Matrices (All 10 Models, Pooled Out-of-Fold)</h5>"))
display(Image(filename=str(FIG_DIR / "fig_fundus_3class_05_confusion_matrices_all.png")))
display(HTML("<h5>ROC, Precision-Recall, and Calibration Diagnostics</h5>"))
display(Image(filename=str(FIG_DIR / "fig_fundus_3class_06_roc_pr_calibration.png")))
display(HTML("<h5>Fold-to-Fold Metric Stability & Variance</h5>"))
display(Image(filename=str(FIG_DIR / "fig_fundus_3class_02_foldwise_stability.png")))
"""))

    # Track 2: Individual OCT
    cells.append(nbf.v4.new_markdown_cell("""## 2. EXP-02: Individual OCT Classification

Evaluating 10 architectures on 2D OCT B-scans.
> [!WARNING]
> Normal ($N=132$) and Pseudopapilledema ($N=113$) scans originate from single sequence prefixes in the underlying dataset. Consequently, patient-independent validation is not possible for these classes; evaluation is performed with contiguous slice-block partitioning.
"""))

    cells.append(nbf.v4.new_code_cell("""df_oct = pd.read_csv(EVAL_DIR / "cv_summary_oct_individual.csv")
display(HTML("<h4>EXP-02 Individual OCT Performance Summary (5-Fold Mean ± SD)</h4>"))
display(HTML(df_oct[cols_show].to_html(classes="table table-striped table-hover", index=False)))
"""))

    cells.append(nbf.v4.new_code_cell("""display(HTML("<h5>Model Performance & Multi-Metric Comparison</h5>"))
display(Image(filename=str(FIG_DIR / "fig_oct_individual_01_model_comparison.png")))
display(HTML("<h5>Confusion Matrices (All 10 Models, Pooled Out-of-Fold)</h5>"))
display(Image(filename=str(FIG_DIR / "fig_oct_individual_05_confusion_matrices_all.png")))
display(HTML("<h5>ROC, Precision-Recall, and Reliability Calibration</h5>"))
display(Image(filename=str(FIG_DIR / "fig_oct_individual_06_roc_pr_calibration.png")))
"""))

    # Track 3: OCT Sequence
    cells.append(nbf.v4.new_markdown_cell("""## 3. EXP-03: OCT Sequence Classification (Window-Level & Sequence-Level)

Evaluating sequence architectures (LSTM, BiLSTM, ResNet50+LSTM, ResNet50+BiLSTM, EfficientNet-B0+BiLSTM) across sequence windows ($L=8$, stride=8) and aggregated patient sequence volumes ($N=21$).
"""))

    cells.append(nbf.v4.new_code_cell("""df_seq = pd.read_csv(EVAL_DIR / "cv_summary_oct_sequence.csv")
display(HTML("<h4>EXP-03 OCT Sequence Performance Summary (Window-Level 5-Fold Mean ± SD)</h4>"))
display(HTML(df_seq[cols_show].to_html(classes="table table-striped table-hover", index=False)))

df_seq_lvl = pd.read_csv(EVAL_DIR / "oct_sequence_level_summary.csv")
display(HTML("<h4>Sequence-Level Out-of-Fold Evaluation (N=21 Sequences)</h4>"))
display(HTML(df_seq_lvl.to_html(classes="table table-striped table-hover", index=False)))
"""))

    cells.append(nbf.v4.new_code_cell("""display(HTML("<h5>Window-Level Performance & Metric Comparison</h5>"))
display(Image(filename=str(FIG_DIR / "fig_oct_sequence_01_model_comparison.png")))
display(HTML("<h5>Window-Level vs Sequence-Level Aggregation Comparison</h5>"))
display(Image(filename=str(FIG_DIR / "fig_oct_sequence_09_window_vs_sequence.png")))
display(HTML("<h5>ROC & Precision-Recall Trajectories</h5>"))
display(Image(filename=str(FIG_DIR / "fig_oct_sequence_06_roc_pr_calibration.png")))
"""))

    # Track 4: OCT Severity Grading
    cells.append(nbf.v4.new_markdown_cell("""## 4. EXP-04: OCT Severity Grading (Exploratory Ordinal / Multi-Class)

Evaluating 10 architectures on exploratory folder-derived severity grades (G1–G4).
Includes Quadratic Weighted Kappa (QWK), Mean Absolute Error (MAE), and Mean Squared Error (MSE).
"""))

    cells.append(nbf.v4.new_code_cell("""df_sev = pd.read_csv(EVAL_DIR / "cv_summary_oct_severity.csv")
display(HTML("<h4>EXP-04 OCT Severity Performance Summary (5-Fold Mean ± SD)</h4>"))
sev_cols = ["model", "category", "macro_f1_mean", "macro_f1_std", "accuracy_mean", "qwk_mean", "mae_mean", "mcc_mean", "cohen_kappa_mean"]
display(HTML(df_sev[sev_cols].to_html(classes="table table-striped table-hover", index=False)))
"""))

    cells.append(nbf.v4.new_code_cell("""display(HTML("<h5>Model Performance & Ordinal Agreement (QWK)</h5>"))
display(Image(filename=str(FIG_DIR / "fig_oct_severity_01_model_comparison.png")))
display(HTML("<h5>Severity Confusion Matrices (All 10 Models, Pooled Out-of-Fold)</h5>"))
display(Image(filename=str(FIG_DIR / "fig_oct_severity_05_confusion_matrices_all.png")))
display(HTML("<h5>Calibration & Probability Reliability Curves</h5>"))
display(Image(filename=str(FIG_DIR / "fig_oct_severity_06_roc_pr_calibration.png")))
"""))

    # Section 5: Cross-Track Synthesis
    cells.append(nbf.v4.new_markdown_cell("""## 5. Cross-Track Architecture Synthesis & Efficiency Pareto Analysis

Comparing architecture families (CNN, RNN, Hybrid, Baseline) across modalities and evaluating parameter complexity vs. training runtime.
"""))

    cells.append(nbf.v4.new_code_cell("""display(HTML("<h5>Architecture Families & Cross-Validation Stability Heatmaps</h5>"))
display(Image(filename=str(FIG_DIR / "fig_cross_01_architecture_heatmaps_and_families.png")))
display(HTML("<h5>Complexity & Runtime Pareto Curves (Tracks Evaluated Independently)</h5>"))
display(Image(filename=str(FIG_DIR / "fig_cross_02_complexity_and_runtime_pareto.png")))
"""))

    nb.cells = cells
    nb_path = PROJECT_ROOT / "notebooks" / "04_Model_Evaluation_and_Visualization.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {nb_path}")
    return nb_path


if __name__ == "__main__":
    p = build_nb04()
    execute_notebook(p, timeout_sec=180)
    print("Notebook 04 execution completed successfully!")

