"""Bulk-load the Kaggle credit card fraud CSV into Postgres."""

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from src.config import FEATURE_COLUMNS
from src.db.connection import get_engine, get_session
from src.db.schema import Base, Transaction, create_tables


def download_dataset() -> Path:
    """Download via kagglehub (requires Kaggle API credentials)."""
    import kagglehub

    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = Path(path) / "creditcard.csv"
    if not csv_path.exists():
        # kagglehub may return the directory containing the file directly
        candidates = list(Path(path).glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV found in {path}")
        csv_path = candidates[0]
    return csv_path


def load_csv_to_db(csv_path: Path, batch_size: int = 5000, drop_existing: bool = False):
    engine = get_engine()
    if drop_existing:
        Base.metadata.drop_all(engine)
    create_tables(engine)

    df = pd.read_csv(csv_path)
    # Kaggle columns: Time, V1..V28, Amount, Class
    df = df.rename(
        columns={
            "Time": "time",
            "Amount": "amount",
            "Class": "actual_class",
            **{f"V{i}": f"v{i}" for i in range(1, 29)},
        }
    )

    records = df[["time", "amount", *FEATURE_COLUMNS, "actual_class"]].to_dict("records")

    session: Session = get_session()
    try:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            session.bulk_insert_mappings(Transaction, batch)
            session.commit()
            print(f"Inserted {min(i + batch_size, len(records)):,} / {len(records):,}")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Load fraud dataset CSV into Postgres")
    parser.add_argument("--csv", type=Path, help="Path to creditcard.csv (skip download)")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv or download_dataset()
    print(f"Loading from {csv_path}")
    load_csv_to_db(csv_path, batch_size=args.batch_size, drop_existing=args.drop_existing)


if __name__ == "__main__":
    main()
