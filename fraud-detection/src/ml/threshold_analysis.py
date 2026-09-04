"""Threshold analysis for production tradeoffs (fixed FPR + top-K)."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from src.config import ARTIFACTS_DIR
from src.db.connection import get_session
from src.ml.dataset import load_transactions_df
from src.ml.metrics import find_best_threshold, find_threshold_at_fpr
from src.ml.registry import MODELS, load_model, score_batch
from src.ml.statement import featurize_dataframe


def threshold_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> dict:
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    valid = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid) == 0:
        return {"target_recall": target_recall, "threshold": None, "precision": 0.0, "recall": 0.0}
    idx = valid[np.argmax(precisions[valid])]
    return {
        "target_recall": target_recall,
        "threshold": float(thresholds[idx]),
        "precision": float(precisions[idx]),
        "recall": float(recalls[idx]),
    }


def threshold_for_top_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    k = min(k, len(scores))
    sorted_scores = np.sort(scores)[::-1]
    threshold = float(sorted_scores[k - 1])
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return {
        "top_k": k,
        "threshold": threshold,
        "alerts": int(y_pred.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "precision": tp / max(tp + fp, 1),
    }


def threshold_for_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict:
    threshold = find_threshold_at_fpr(y_true, scores, target_fpr=target_fpr)
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n_neg = max(tn + fp, 1)
    n_pos = max(tp + fn, 1)
    return {
        "target_fpr": target_fpr,
        "threshold": threshold,
        "false_positive_rate": fp / n_neg,
        "recall": tp / n_pos,
        "precision": tp / max(tp + fp, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def analyze_model(model_key: str) -> dict:
    session = get_session()
    try:
        _, test_df = load_transactions_df(session)
    finally:
        session.close()

    loaded = load_model(model_key)
    X = featurize_dataframe(test_df)
    y_true = test_df["actual_class"].values
    scores = score_batch(loaded, X)

    report = {
        "model_version": loaded.spec.version,
        "model_type": loaded.spec.model_type,
        "f1_optimal_threshold": find_best_threshold(y_true, scores),
        "fpr_budgets": [
            threshold_for_fpr(y_true, scores, target)
            for target in (0.001, 0.01, 0.05)
        ],
        "recall_targets": [
            threshold_at_recall(y_true, scores, target)
            for target in (0.5, 0.7, 0.8, 0.9)
        ],
        "top_k_review": [
            threshold_for_top_k(y_true, scores, k)
            for k in (50, 100, 200)
        ],
    }

    precisions, recalls, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(8, 5))
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve — {loaded.spec.version} (amount/time features only)")
    plt.grid(True, alpha=0.3)
    chart_path = ARTIFACTS_DIR / f"pr_curve_{model_key}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    report["pr_curve_chart"] = str(chart_path)

    return report


def analyze_all() -> dict:
    reports = {key: analyze_model(key) for key in MODELS}
    out_path = ARTIFACTS_DIR / "threshold_report.json"
    out_path.write_text(json.dumps(reports, indent=2))
    print(f"Saved threshold report to {out_path}")

    # Also write a README-ready markdown summary (full metrics at each FPR budget).
    comparison_path = ARTIFACTS_DIR / "model_comparison.json"
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else {}
    lines = [
        "# Metrics summary (amount + datetime features only)",
        "",
        "V1–V28 excluded on purpose. Description/merchant/MCC accepted by the API but **not scored**.",
        "",
        "Autoencoder ≈ random under this constraint (negative finding). Classifier shows lift",
        "vs prevalence but precision at a 1% FPR budget remains too low for deployment.",
        "",
        "## Headline metrics at 1% FPR budget",
        "",
        "| Model | PR-AUC | Baseline | Lift | Precision | Recall | F1 | Achieved FPR | Threshold |",
        "|-------|--------|----------|------|-----------|--------|----|--------------|-----------|",
    ]
    for key, m in comparison.items():
        baseline = m.get("baseline_pr_auc") or 0.0
        lift = m.get("pr_auc_lift")
        if lift is None:
            lift = (m["pr_auc"] / baseline) if baseline else float("nan")
        lines.append(
            f"| {m.get('model_version', key)} | {m.get('pr_auc', float('nan')):.4f} | "
            f"{baseline:.5f} | {lift:.1f}× | {m.get('precision', float('nan')):.4f} | "
            f"{m.get('recall', float('nan')):.4f} | {m.get('f1', float('nan')):.4f} | "
            f"{m.get('false_positive_rate', float('nan')):.4f} | {m.get('threshold', float('nan')):.4f} |"
        )

    lines.extend(["", "## FPR budget trade-offs (business decision surface)", ""])
    for key, report in reports.items():
        lines.append(f"### {report['model_version']}")
        lines.append("")
        lines.append("| Target FPR | Achieved FPR | Recall | Precision | Threshold | TP | FP |")
        lines.append("|------------|--------------|--------|-----------|-----------|----|----|")
        for row in report["fpr_budgets"]:
            lines.append(
                f"| {row['target_fpr']:.1%} | {row['false_positive_rate']:.4f} | "
                f"{row['recall']:.4f} | {row['precision']:.4f} | {row['threshold']:.4f} | "
                f"{row['tp']} | {row['fp']} |"
            )
        lines.append("")

    # Feature-space collision diagnostic (identical amount/time vectors → identical scores)
    session = get_session()
    try:
        _, test_df = load_transactions_df(session)
    finally:
        session.close()
    X = featurize_dataframe(test_df)
    X_key = np.round(X, 6)
    unique = np.unique(X_key, axis=0).shape[0]
    uniq_pct = 100.0 * unique / len(X)
    collision_pct = 100.0 - uniq_pct
    lines.extend(
        [
            "## Feature-space collisions → low precision",
            "",
            f"Unique amount/time feature vectors in test set: **{unique:,}** / {len(X):,} "
            f"({uniq_pct:.1f}% unique; ~{collision_pct:.0f}% share a signature with another row).",
            "",
            "That collision rate is a **direct cause** of low precision: the model cannot separate "
            "fraud from legitimate traffic that shares the same amount/time vector, so many "
            "alerts at a fixed FPR budget are structural false positives—not just a tunable threshold.",
            "",
            "## Limitations (short)",
            "",
            "- Autoencoder lift ≈ 1× baseline → not useful for ranking under this feature set.",
            "- Classifier precision at 1% FPR remains far too low for a real review queue without "
            "merchant/MCC/description signal.",
            "",
        ]
    )
    summary_path = ARTIFACTS_DIR / "metrics_summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"Saved README-ready summary to {summary_path}")

    for key, report in reports.items():
        print(f"\n=== {report['model_version']} ===")
        print(f"F1-optimal threshold: {report['f1_optimal_threshold']:.4f}")
        for row in report["fpr_budgets"]:
            print(
                f"  FPR≈{row['target_fpr']:.1%}: threshold={row['threshold']:.4f}, "
                f"actual_fpr={row['false_positive_rate']:.4f}, recall={row['recall']:.3f}, "
                f"precision={row['precision']:.3f}"
            )
        for row in report["top_k_review"]:
            print(
                f"  Top {row['top_k']} alerts: {row['true_positives']} TP, "
                f"{row['false_positives']} FP, precision={row['precision']:.3f}"
            )
    print(f"\nFeature collisions: {unique:,} unique / {len(X):,} test rows")
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["autoencoder", "classifier", "all"], default="all")
    args = parser.parse_args()
    if args.model == "all":
        analyze_all()
    else:
        report = analyze_model(args.model)
        path = ARTIFACTS_DIR / f"threshold_report_{args.model}.json"
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
