# Research Figure Validation Report

Confirms all figures in `results/research_paper_figures/` were generated end-to-end by executing `notebooks/generate_research_figures_sections_9_to_12.ipynb` strictly from actual experimental outputs.

## 1. Input Source Integrity Audit

- **Track comparison CSVs loaded:** `model_comparison.csv` (10 Fundus models), `oct_individual_comparison.csv` (10 Individual OCT models), `oct_sequence_comparison.csv` (5 Sequence models only), `oct_severity_comparison.csv` (10 Exploratory Severity models).
- **Computational metrics CSV:** `computational_comparison.csv` (35 complete track–model rows; 0 missing latency or parameter values).
- **Statistical CSVs:** `hypothesis_tests.csv` (149 rows, including 4 Friedman omnibus rows), `pairwise_comparisons.csv` (435 rows: 152 Holm-significant).
- **Per-sample prediction CSVs:** **175** real evaluation CSVs loaded cleanly (Fundus: 50 files / 10,250 rows; Individual OCT: 50 files / 9,150 rows; OCT Sequence: 25 files / 575 rows; Exploratory OCT Severity: 50 files / 8,800 rows). All 8 `*__smoke*` files were excluded.

## 2. File-Existence & Render Checks (14 Figures = 28 Files)

| Figure | Section | PNG Size (KB) | PDF Size (KB) | Verified |
|---|---|---:|---:|---|
| `section_09_roc_illustration.png` | Section 9 | 177.5 | 38.0 | PASS |
| `section_09_confusion_matrix_structure.png` | Section 9 | 223.8 | 25.6 | PASS |
| `section_10_cross_track_performance_heatmap.png` | Section 10 | 492.9 | 59.0 | PASS |
| `section_10_best_model_confusion_matrices.png` | Section 10 | 516.5 | 55.2 | PASS |
| `section_10_classwise_performance_comparison.png` | Section 10 | 454.0 | 41.7 | PASS |
| `section_10_performance_vs_computational_cost.png` | Section 10 | 747.1 | 66.7 | PASS |
| `section_11_friedman_test_results.png` | Section 11 | 270.7 | 50.4 | PASS |
| `section_11_statistical_comparison.png` | Section 11 | 332.6 | 68.8 | PASS |
| `section_11_fundus_error_analysis.png` | Section 11 | 292.2 | 36.4 | PASS |
| `section_11_oct_error_analysis.png` | Section 11 | 308.0 | 50.9 | PASS |
| `section_11_oct_sequence_error_analysis.png` | Section 11 | 251.7 | 37.0 | PASS |
| `section_11_severity_error_analysis.png` | Section 11 | 401.2 | 45.3 | PASS |
| `section_11_cross_track_error_comparison.png` | Section 11 | 302.0 | 41.6 | PASS |
| `section_12_overall_research_summary.png` | Section 12 | 518.6 | 54.1 | PASS |

## 3. Research Integrity Confirmations

- **Independent Tracks Preserved:** Zero multimodal combination; four independent tracks retained cleanly (`fundus_3class`, `oct_individual`, `oct_sequence`, `oct_severity`).
- **Un-Evaluated Sequence Cells Masked:** Only the 5 evaluated sequence configurations appear in `oct_sequence` plots and heatmap cells (`Not Evaluated` hatched, never zero-filled).
- **Zero Fabricated Metrics:** Section 9 conceptual diagrams contain zero numerical AUCs or experimental counts; all Section 10–12 metrics match verified CSV tables and `175` real prediction files.
- **Non-Destructive Execution:** Pre-existing CSVs, checkpoints, JSON manifests, and `57` charts in `results/figures/` remain unaltered.
- **No Repetitive Legacy Bar Charts Retained:** Obsolete individual Macro-F1 bar plots and redundant best-model bar chart removed during cleanup.
