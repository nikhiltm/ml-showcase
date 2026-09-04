"""Score transactions and write predictions back to Postgres."""

import argparse
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import delete
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.schema import Prediction
from src.ml.dataset import load_all_transactions_df, load_transactions_df
from src.ml.metrics import find_threshold_at_fpr
from src.ml.registry import MODELS, load_model, score_batch
from src.ml.statement import featurize_dataframe


def write_predictions(
    session: Session,
    transaction_ids: list[int],
    fraud_scores: np.ndarray,
    predicted_classes: np.ndarray,
    *,
    model_version: str,
    batch_size: int = 1000,
):
    now = datetime.now(timezone.utc)
    records = [
        {
            "transaction_id": int(tid),
            "model_version": model_version,
            "fraud_score": float(score),
            "predicted_class": int(pred),
            "predicted_at": now,
        }
        for tid, score, pred in zip(transaction_ids, fraud_scores, predicted_classes)
    ]

    for i in range(0, len(records), batch_size):
        session.bulk_insert_mappings(Prediction, records[i : i + batch_size])
        session.commit()
        print(f"  Wrote {min(i + batch_size, len(records)):,} / {len(records):,}")


def run_predictions_for_model(
    session: Session,
    df,
    model_key: str,
    *,
    threshold: float | None = None,
    target_fpr: float = 0.01,
):
    loaded = load_model(model_key)
    X = featurize_dataframe(df)
    y_true = df["actual_class"].values
    fraud_scores = score_batch(loaded, X)

    if threshold is None:
        threshold = find_threshold_at_fpr(y_true, fraud_scores, target_fpr=target_fpr)

    predicted_classes = (fraud_scores >= threshold).astype(int)
    print(f"\n{loaded.spec.version} (threshold={threshold:.4f}, target_fpr={target_fpr})")
    write_predictions(
        session,
        df["id"].tolist(),
        fraud_scores,
        predicted_classes,
        model_version=loaded.spec.version,
    )


def run_predictions(
    *,
    model: str = "all",
    threshold: float | None = None,
    use_test_set: bool = True,
    target_fpr: float = 0.01,
):
    session = get_session()
    try:
        if use_test_set:
            _, df = load_transactions_df(session)
        else:
            df = load_all_transactions_df(session)

        model_keys = list(MODELS) if model == "all" else [model]
        versions = [MODELS[k].version for k in model_keys]
        session.execute(delete(Prediction).where(Prediction.model_version.in_(versions)))
        session.commit()

        for model_key in model_keys:
            run_predictions_for_model(
                session,
                df,
                model_key,
                threshold=threshold,
                target_fpr=target_fpr,
            )
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["autoencoder", "classifier", "all"], default="all")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--all-data", action="store_true")
    args = parser.parse_args()
    run_predictions(
        model=args.model,
        threshold=args.threshold,
        use_test_set=not args.all_data,
        target_fpr=args.target_fpr,
    )


if __name__ == "__main__":
    main()
