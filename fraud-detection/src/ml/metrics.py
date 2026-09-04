"""Shared evaluation metrics and threshold utilities."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Max-F1 threshold (can be unstable on flat PR curves — prefer FPR budget)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx])


def find_threshold_at_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    target_fpr: float = 0.01,
) -> float:
    """
    Threshold for a fixed false-positive rate on the negative class.

    Matches how fraud ops often think: "we can only review ~1% of legit volume."
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    neg_scores = scores[y_true == 0]
    if len(neg_scores) == 0:
        return find_best_threshold(y_true, scores)
    # Flag the top target_fpr fraction of legitimate scores as FP budget
    cutoff = float(np.quantile(neg_scores, 1.0 - target_fpr))
    return cutoff


def errors_to_calibrated_scores(
    errors: np.ndarray,
    reference_errors: np.ndarray,
) -> np.ndarray:
    """
    Map reconstruction errors to [0, 1] via the empirical CDF of reference errors.

    Unlike batch percentile-rank, this works for a single API request and is
    stable across batch sizes.
    """
    errors = np.asarray(errors, dtype=np.float64).ravel()
    reference_errors = np.asarray(reference_errors, dtype=np.float64).ravel()
    if len(reference_errors) == 0:
        return np.zeros_like(errors, dtype=np.float64)
    ref_sorted = np.sort(reference_errors)
    # Fraction of reference errors strictly below each error (+ mid-rank tie break)
    scores = np.searchsorted(ref_sorted, errors, side="right") / len(ref_sorted)
    return np.clip(scores, 0.0, 1.0)


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float | None = None,
    threshold_policy: str = "fpr",
    target_fpr: float = 0.01,
    model_version: str = "",
    model_type: str = "",
) -> dict:
    if threshold is None:
        if threshold_policy == "f1":
            threshold = find_best_threshold(y_true, scores)
        else:
            threshold = find_threshold_at_fpr(y_true, scores, target_fpr=target_fpr)

    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n_neg = max(int((y_true == 0).sum()), 1)
    n_pos = max(int((y_true == 1).sum()), 1)

    metrics = {
        "model_version": model_version,
        "model_type": model_type,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "threshold_policy": threshold_policy if threshold is not None else "provided",
        "target_fpr": float(target_fpr) if threshold_policy == "fpr" else None,
        "false_positive_rate": float(fp / n_neg),
        "true_positive_rate": float(tp / n_pos),
        "n_test": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "fraud_rate_pct": float(100.0 * y_true.mean()),
        "baseline_pr_auc": float(y_true.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"=== {metrics['model_version']} ({metrics['model_type']}) ===")
    print(f"Fraud rate in test set: {metrics['fraud_rate_pct']:.3f}%")
    print(f"Baseline PR-AUC (prevalence): {metrics['baseline_pr_auc']:.5f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1:         {metrics['f1']:.4f}")
    print(f"PR-AUC:     {metrics['pr_auc']:.4f}")
    baseline = metrics.get("baseline_pr_auc") or 0.0
    lift = (metrics["pr_auc"] / baseline) if baseline > 0 else float("nan")
    print(f"Lift vs baseline PR-AUC: {lift:.1f}×")
    print(f"FPR:        {metrics['false_positive_rate']:.4f}")
    print(
        f"Threshold:  {metrics['threshold']:.4f} "
        f"(policy={metrics.get('threshold_policy')}, "
        f"target_fpr={metrics.get('target_fpr')})"
    )


def save_metrics(metrics: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
