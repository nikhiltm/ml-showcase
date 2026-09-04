#!/usr/bin/env python3
"""Run the full fraud detection pipeline end-to-end."""

import argparse
import json
from pathlib import Path

from src.config import ARTIFACTS_DIR
from src.db.connection import get_engine, get_session
from src.db.load_data import download_dataset, load_csv_to_db
from src.db.queries import (
    compare_models,
    daily_alert_volume,
    false_positive_rate_by_bucket,
    fraud_capture_rate,
    fraud_rate_by_amount_bucket,
    top_risk_transactions,
)
from src.db.schema import create_tables
from src.ml.evaluate import evaluate_all
from src.ml.predict import run_predictions
from src.ml.threshold_analysis import analyze_all
from src.ml.train import train_autoencoder
from src.ml.train_classifier import train_classifier


def run_pipeline(
    *,
    csv_path: Path | None = None,
    skip_load: bool = False,
    epochs: int = 30,
    skip_train: bool = False,
):
    if not skip_load:
        csv = csv_path or download_dataset()
        print("=== 1/7 Load CSV into Postgres ===")
        load_csv_to_db(csv, drop_existing=True)
    else:
        print("=== 1/7 Skipping data load ===")
        create_tables(get_engine())

    if not skip_train:
        print("\n=== 2/7 Train autoencoder ===")
        train_autoencoder(epochs=epochs)
        print("\n=== 3/7 Train classifier ===")
        train_classifier(epochs=epochs)
    else:
        print("\n=== 2-3/7 Skipping training ===")

    print("\n=== 4/7 Evaluate both models ===")
    evaluate_all()

    print("\n=== 5/7 Threshold analysis ===")
    analyze_all()

    print("\n=== 6/7 Write predictions (both models) ===")
    run_predictions(model="all")

    print("\n=== 7/7 Analytics ===")
    session = get_session()
    try:
        print("\nModel comparison:")
        for row in compare_models(session):
            print(row)

        for model_key in ("v1.0.0-autoencoder", "v1.1.0-classifier"):
            print(f"\nTop 5 risky ({model_key}):")
            for row in top_risk_transactions(session, limit=5, model_version=model_key):
                print(row)

        print("\nFraud rate by amount bucket:")
        for row in fraud_rate_by_amount_bucket(session):
            print(row)

        print("\nFalse positive rate by bucket (autoencoder):")
        for row in false_positive_rate_by_bucket(session, model_version="v1.0.0-autoencoder"):
            print(row)

        print("\nFraud capture rate by bucket (classifier):")
        for row in fraud_capture_rate(session, model_version="v1.1.0-classifier"):
            print(row)

        print("\nDaily alert volume:")
        for row in daily_alert_volume(session):
            print(row)
    finally:
        session.close()

    comparison_path = ARTIFACTS_DIR / "model_comparison.json"
    if comparison_path.exists():
        print(f"\nPipeline complete. Results: {comparison_path}")
        print(json.dumps(json.loads(comparison_path.read_text()), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run full fraud detection pipeline")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    run_pipeline(
        csv_path=args.csv,
        skip_load=args.skip_load,
        skip_train=args.skip_train,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
