"""Classification metrics, one-vs-rest ROC-AUC, sensitivity, and specificity."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: np.ndarray | None,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute overall and per-class evaluation metrics safely."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    acc = float(accuracy_score(yt, yp)) if len(yt) else 0.0
    bal_acc = float(balanced_accuracy_score(yt, yp)) if len(yt) else 0.0
    cm = confusion_matrix(yt, yp, labels=labels)

    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        yt, yp, labels=labels, zero_division=0
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, average="weighted", zero_division=0
    )
    macro_p = float(np.mean(p_cls))
    macro_r = float(np.mean(r_cls))
    macro_f1 = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(w_f1)

    total = int(cm.sum())
    per_class: dict[str, dict[str, float | int]] = {}
    sens_list = []
    spec_list = []
    for idx, cname in enumerate(class_names):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens_list.append(sens)
        spec_list.append(spec)
        per_class[cname] = {
            "precision": round(float(p_cls[idx]), 6),
            "recall": round(float(r_cls[idx]), 6),
            "f1": round(float(f1_cls[idx]), 6),
            "sensitivity": round(float(sens), 6),
            "specificity": round(float(spec), 6),
            "support": int(sup_cls[idx]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    roc_auc = float("nan")
    pr_auc = float("nan")
    per_class_auc: dict[str, float | None] = {}
    per_class_pr_auc: dict[str, float | None] = {}
    if y_prob is not None and len(np.unique(yt)) >= 2:
        prob = np.asarray(y_prob, dtype=np.float64)
        # Ensure probabilities are normalized rows for sklearn
        row_sum = prob.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        prob = prob / row_sum
        present = sorted(np.unique(yt).tolist())
        if len(present) == n_classes:
            try:
                roc_auc = float(roc_auc_score(yt, prob, multi_class="ovr", average="macro"))
            except ValueError:
                roc_auc = float("nan")
        valid_prs = []
        for idx, cname in enumerate(class_names):
            y_bin = (yt == idx).astype(int)
            if 0 < y_bin.sum() < len(y_bin):
                per_class_auc[cname] = round(float(roc_auc_score(y_bin, prob[:, idx])), 6)
                ap = float(average_precision_score(y_bin, prob[:, idx]))
                per_class_pr_auc[cname] = round(ap, 6)
                valid_prs.append(ap)
            else:
                per_class_auc[cname] = None
                per_class_pr_auc[cname] = None
        if len(valid_prs) == n_classes:
            pr_auc = float(np.mean(valid_prs))

    return {
        "accuracy": round(acc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_precision": round(float(w_p), 6),
        "weighted_recall": round(float(w_r), 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_sensitivity": round(float(np.mean(sens_list)), 6),
        "macro_specificity": round(float(np.mean(spec_list)), 6),
        "roc_auc_ovr": round(roc_auc, 6) if not np.isnan(roc_auc) else None,
        "pr_auc_ovr": round(pr_auc, 6) if not np.isnan(pr_auc) else None,
        "per_class_auc": per_class_auc,
        "per_class_pr_auc": per_class_pr_auc,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "num_samples": total,
    }

