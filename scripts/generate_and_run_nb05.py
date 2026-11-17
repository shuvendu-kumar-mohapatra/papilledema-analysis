"""Generate and execute notebooks/05_Statistical_Analysis.ipynb."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nbformat as nbf
from src.utils.notebook_runner import execute_notebook


def build_nb05() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    }
    cells = []

    # Title & Executive Summary
    cells.append(nbf.v4.new_markdown_cell("""# Notebook 05: Cross-Validation Statistical Methodology and Hypothesis Testing

## Rigorous Statistical Protocol for 5-Fold Cross-Validation
Standard independent hypothesis tests (such as Student's paired $t$-test or naive Wilcoxon signed-rank tests) violate the independence assumption when applied to $K$-fold cross-validation because training sets across folds overlap ($K-1$ folds share data), leading to artificially deflated variance and inflated Type I error rates.

To ensure statistical rigor, this notebook presents hypothesis testing using five complementary paradigms:
1. **Omnibus Non-Parametric Friedman Test**: Tests whether all architectures within a track perform identically across the 5 folds.
2. **Kendall's $W$ (Coefficient of Concordance)**: Quantifies inter-fold ranking agreement ($0 = \\text{no agreement}$, $1 = \\text{unanimous ranking}$).
3. **Nadeau–Bengio Corrected Paired $t$-Test**:
   $$\\hat{\\sigma}_{\\text{corrected}}^2 = \\left(\\frac{1}{K} + \\frac{n_{\\text{test}}}{n_{\\text{train}}}\\right) s^2, \\quad t = \\frac{\\bar{d}}{\\hat{\\sigma}_{\\text{corrected}}}$$
   with $K=5$, correcting degrees of freedom and variance for cross-validation overlap.
4. **Wilcoxon Signed-Rank Test with Sample Size Limitation**:
   > [!NOTE]
   > For $n=5$ folds, the minimum possible two-sided non-parametric Wilcoxon $p$-value is $2 \\times (1/2)^5 = 0.0625 > 0.05$. Therefore, exact Wilcoxon tests cannot detect significance at $\\alpha=0.05$ with 5 folds alone; reported for transparency alongside asymptotic values.
5. **Pooled Out-of-Fold (OOF) McNemar's Test**: Sample-level paired contingency test with Edward's continuity correction on independent outer predictions ($N_{\\text{fundus}}=1,367$, $N_{\\text{oct}}=1,140$, $N_{\\text{seq}}=146$, $N_{\\text{sev}}=895$).
6. **Holm–Bonferroni Step-Down Correction**: Controls Family-Wise Error Rate (FWER) across all pairwise hypothesis comparisons.
"""))

    # Imports & Setup
    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
from IPython.display import display, HTML, Image

PROJECT_ROOT = Path("..").resolve()
STATS_DIR = PROJECT_ROOT / "results" / "cross_validation" / "statistical_analysis"
FIG_DIR = STATS_DIR / "figures"
print(f"Loading statistical artifacts from: {STATS_DIR}")
"""))

    # Section 1: Friedman & Kendall's W
    cells.append(nbf.v4.new_markdown_cell("""## 1. Omnibus Hypothesis Testing: Friedman Test & Kendall's $W$

Evaluating whether architectural performance differences across the 5 folds are statistically significant across four primary evaluation metrics.
"""))

    cells.append(nbf.v4.new_code_cell("""df_friedman = pd.read_csv(STATS_DIR / "cv_friedman_tests.csv")
display(HTML("<h4>Omnibus Friedman & Kendall's W Results Across Tracks</h4>"))
display(HTML(df_friedman.to_html(classes="table table-striped table-hover", index=False)))

display(HTML("<h5>Kendall's Concordance and Omnibus Significance Figure</h5>"))
display(Image(filename=str(FIG_DIR / "cv_stats_01_friedman_kendall.png")))
"""))

    # Section 2: Pairwise Tests
    cells.append(nbf.v4.new_markdown_cell("""## 2. Pairwise Model Comparisons: Nadeau–Bengio vs. Naive vs. Wilcoxon

Evaluating pairwise differences against baseline and top-performing models using the Nadeau–Bengio correction and Holm FWER control.
"""))

    cells.append(nbf.v4.new_code_cell("""df_pairwise = pd.read_csv(STATS_DIR / "cv_pairwise_comparisons.csv")
display(HTML("<h4>Pairwise Comparisons Sample (Nadeau–Bengio Corrected paired t-test)</h4>"))
cols_pw = ["track", "metric", "model_a", "model_b", "mean_diff", "nb_t_stat", "nb_p_raw", "nb_p_holm", "nb_significant_holm", "wilcoxon_stat", "wilcoxon_p_raw"]
display(HTML(df_pairwise[cols_pw].head(20).to_html(classes="table table-striped table-hover", index=False)))

display(HTML("<h5>Pairwise Statistical Significance Heatmaps</h5>"))
display(Image(filename=str(FIG_DIR / "cv_stats_02_pairwise_heatmaps.png")))
"""))

    # Section 3: OOF McNemar
    cells.append(nbf.v4.new_markdown_cell("""## 3. Sample-Level Out-of-Fold (OOF) McNemar Contingency Tests

Testing sample-level classification disagreement on the complete pooled out-of-fold predictions.
"""))

    cells.append(nbf.v4.new_code_cell("""df_mcnemar = pd.read_csv(STATS_DIR / "cv_oof_mcnemar_tests.csv")
display(HTML("<h4>Pooled OOF McNemar Tests Sample</h4>"))
cols_mcn = ["track", "model_a", "model_b", "n_oof_items", "b_a_only_correct", "c_b_only_correct", "statistic", "p_value", "p_adjusted_holm", "significant_holm"]
display(HTML(df_mcnemar[cols_mcn].head(15).to_html(classes="table table-striped table-hover", index=False)))
"""))

    # Section 4: Original 5-Seed vs New 5-Fold CV
    cells.append(nbf.v4.new_markdown_cell("""## 4. Methodological Shift: Original 5-Seed Fixed-Split vs. 5-Fold Cross-Validation

Comparing the original 5-seed fixed test-set results with the leak-free 5-fold cross-validation estimates.
- Original fixed test sets suffered from duplicate cluster leakage (Fundus) and evaluated only a single random partition of patients/slices.
- The 5-fold cross-validation evaluates 100% of the dataset out-of-fold with grouping constraints, providing realistic clinical generalization bounds.
"""))

    cells.append(nbf.v4.new_code_cell("""df_comp = pd.read_csv(STATS_DIR / "original_vs_cv_comparison.csv")
display(HTML("<h4>Original 5-Seed Fixed-Split vs. 5-Fold Cross-Validation Comparison</h4>"))
display(HTML(df_comp.to_html(classes="table table-striped table-hover", index=False)))

display(HTML("<h5>Original vs. 5-Fold CV Comparison Plot</h5>"))
display(Image(filename=str(FIG_DIR / "cv_stats_03_original_vs_cv_comparison.png")))
"""))

    nb.cells = cells
    nb_path = PROJECT_ROOT / "notebooks" / "05_Statistical_Analysis.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {nb_path}")
    return nb_path


if __name__ == "__main__":
    p = build_nb05()
    execute_notebook(p, timeout_sec=180)
    print("Notebook 05 execution completed successfully!")
