"""Pull transactions from Postgres into a DataFrame for training."""

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import FEATURE_COLUMNS
from src.db.schema import Transaction


def load_all_transactions_df(session: Session) -> pd.DataFrame:
    """Load every transaction from Postgres."""
    stmt = select(
        Transaction.id,
        Transaction.time,
        Transaction.amount,
        *[getattr(Transaction, c) for c in FEATURE_COLUMNS],
        Transaction.actual_class,
    )
    return pd.read_sql(stmt, session.bind)


def load_transactions_df(
    session: Session,
    *,
    holdout_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Round-trip: Postgres → DataFrame.
    Returns (train_df, test_df) with a stratified split on actual_class.
    """
    stmt = select(
        Transaction.id,
        Transaction.time,
        Transaction.amount,
        *[getattr(Transaction, c) for c in FEATURE_COLUMNS],
        Transaction.actual_class,
    )
    df = pd.read_sql(stmt, session.bind)

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=holdout_fraction,
        stratify=df["actual_class"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def feature_matrix(df: pd.DataFrame, include_amount: bool = True) -> pd.DataFrame:
    cols = list(FEATURE_COLUMNS)
    if include_amount:
        cols = ["amount", *cols]
    return df[cols]
