"""
evaluate.py
============
STEP 5: Performance Evaluation & Benchmarking.

WHAT THIS STEP DOES
----------------------
Takes every trained model (classical baselines + hybrid VQC + QSVM) and computes the exact
metrics the problem statement asks for: accuracy, sensitivity (recall), specificity, plus the
supporting metrics a reviewer will expect (precision, F1, AUC-ROC, confusion matrix), on the
held-out TEST set only (never train or val - that would leak information into your reported
numbers). Also records wall-clock training time per model for the "computational efficiency"
comparison, and runs a McNemar's test to check whether the quantum model's difference from the
best classical baseline is statistically significant (rather than noise from a small test set).

MODELS/ALGORITHMS USED IN THIS STEP
--------------------------------------
- scikit-learn.metrics: accuracy_score, recall_score (=sensitivity), precision_score, f1_score,
  roc_auc_score, confusion_matrix.
- Specificity is computed manually from the confusion matrix (scikit-learn has no built-in
  specificity function) as TN / (TN + FP).
- statsmodels / scipy McNemar's test: appropriate for comparing two classifiers' predictions on
  the SAME test set (paired samples), unlike a standard t-test which assumes independence.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Computes the full metric suite for one model's predictions on one dataset split."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
    return metrics


def compute_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, max_points: int = 60) -> dict:
    """
    Computes the ROC curve (FPR/TPR at every threshold) for one model, downsampled to
    `max_points` for a clean frontend chart (sklearn's roc_curve can return hundreds of points
    on datasets this size - a chart doesn't need every single one, just enough to draw a smooth
    curve). Points are always kept at even index spacing, including the first and last, so the
    curve's start (0,0) and end (1,1) are never dropped.
    """
    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "auc": float("nan")}

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]

    return {"fpr": [round(float(x), 4) for x in fpr], "tpr": [round(float(x), 4) for x in tpr], "auc": round(float(auc), 4)}


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, threshold: float = 0.5) -> dict:
    """
    McNemar's test: is there a statistically significant difference between two classifiers'
    predictions on the SAME test set? Appropriate here because both models are scored on
    identical samples (paired data), unlike an independent-samples t-test.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    pa = (pred_a >= threshold).astype(int)
    pb = (pred_b >= threshold).astype(int)
    correct_a = (pa == y_true)
    correct_b = (pb == y_true)

    # 2x2 table: [both correct, a correct b wrong], [a wrong b correct, both wrong]
    n00 = int(np.sum(correct_a & correct_b))
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n11 = int(np.sum(~correct_a & ~correct_b))
    table = [[n00, n01], [n10, n11]]

    result = mcnemar(table, exact=(n01 + n10) < 25)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "table": table}


def build_comparison_table(results: dict, y_test: np.ndarray) -> pd.DataFrame:
    """
    results: {model_name: {"y_pred_proba": np.ndarray, "train_time_sec": float}}
    Returns a single dataframe ready to drop into your report/paper as the main results table.
    """
    rows = []
    for name, r in results.items():
        m = compute_metrics(y_test, r["y_pred_proba"])
        rows.append({
            "Model": name,
            "Accuracy": round(m["accuracy"], 4),
            "Sensitivity": round(m["sensitivity_recall"], 4),
            "Specificity": round(m["specificity"], 4),
            "Precision": round(m["precision"], 4),
            "F1": round(m["f1_score"], 4),
            "AUC-ROC": round(m["auc_roc"], 4) if not np.isnan(m["auc_roc"]) else "N/A",
            "Train Time (s)": round(r["train_time_sec"], 3),
        })
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    logger.info("\n" + df.to_string(index=False))
    return df
