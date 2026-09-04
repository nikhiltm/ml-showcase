"""Evaluate models with precision, recall, F1, PR-AUC, and fixed-FPR thresholds."""

import argparse
import json

from src.config import ARTIFACTS_DIR
from src.db.connection import get_session
from src.ml.dataset import load_transactions_df
from src.ml.metrics import compute_metrics, print_metrics, save_metrics
from src.ml.registry import MODELS, evaluation_path, load_model, score_batch
from src.ml.statement import featurize_dataframe


def evaluate_model(
    model_key: str,
    *,
    threshold: float | None = None,
    target_fpr: float = 0.01,
) -> dict:
    session = get_session()
    try:
        _, test_df = load_transactions_df(session)
    finally:
        session.close()

    loaded = load_model(model_key)
    X = featurize_dataframe(test_df)
    y_true = test_df["actual_class"].values
    scores = score_batch(loaded, X)

    metrics = compute_metrics(
        y_true,
        scores,
        threshold=threshold,
        threshold_policy="fpr",
        target_fpr=target_fpr,
        model_version=loaded.spec.version,
        model_type=loaded.spec.model_type,
    )
    print_metrics(metrics)
    print(
        "Note: V1–V28 PCA features are intentionally excluded so the model can "
        "score real bank-statement JSON; PR-AUC will be much lower than ULB "
        "benchmarks that use V1–V28."
    )
    save_metrics(metrics, evaluation_path(model_key))
    return metrics


def evaluate_all(*, threshold: float | None = None, target_fpr: float = 0.01) -> dict[str, dict]:
    results = {}
    for model_key in MODELS:
        print()
        results[model_key] = evaluate_model(
            model_key, threshold=threshold, target_fpr=target_fpr
        )

    comparison_path = ARTIFACTS_DIR / "model_comparison.json"
    comparison = {
        key: {
            "model_version": m["model_version"],
            "model_type": m["model_type"],
            "pr_auc": m["pr_auc"],
            "baseline_pr_auc": m["baseline_pr_auc"],
            "pr_auc_lift": (
                m["pr_auc"] / m["baseline_pr_auc"] if m["baseline_pr_auc"] else None
            ),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "false_positive_rate": m["false_positive_rate"],
            "threshold": m["threshold"],
            "threshold_policy": m["threshold_policy"],
            "target_fpr": m["target_fpr"],
        }
        for key, m in results.items()
    }
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nSaved comparison to {comparison_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["autoencoder", "classifier", "all"], default="all")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()
    if args.model == "all":
        evaluate_all(threshold=args.threshold, target_fpr=args.target_fpr)
    else:
        evaluate_model(args.model, threshold=args.threshold, target_fpr=args.target_fpr)


if __name__ == "__main__":
    main()
