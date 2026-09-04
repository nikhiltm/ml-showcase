"""
Bank-statement featurization (train/serve aligned).

Intentional constraint
----------------------
The ULB Kaggle dataset's predictive power lives mostly in V1–V28 (PCA of
withheld bank features). A real statement API cannot supply those columns, so
this project **deliberately excludes V1–V28** and scores only fields a statement
line actually has: amount + datetime.

Description / merchant / MCC are accepted on the API for a realistic request
shape and are echoed in the response, but they are **not** model features —
the training set has no real memos, and hashing synthetic memos only re-encoded
amount/time (noise, not signal).

Trade-off: PR-AUC will be far below published ULB benchmarks (~0.7–0.85) that
use V1–V28. That is expected under this production-realism constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

FEATURE_NAMES = [
    "amount",
    "log_amount",
    "hour_norm",
    "dow_norm",
    "is_night",
    "is_round_dollar",
    "amount_scale",
]
STATEMENT_FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class BankStatement:
    """One line from a credit card / bank statement."""

    amount: float
    date: str
    description: str
    merchant: str = ""
    mcc: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": self.amount,
            "date": self.date,
            "description": self.description,
            "merchant": self.merchant,
            "mcc": self.mcc,
        }


def _parse_date(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _numeric_features(amount: float, hour: int, dow: int) -> np.ndarray:
    amount = float(max(amount, 0.0))
    return np.array(
        [
            amount,
            float(np.log1p(amount)),
            hour / 23.0,
            dow / 6.0,
            1.0 if (hour >= 22 or hour < 6) else 0.0,
            1.0 if abs(amount - round(amount)) < 1e-6 else 0.0,
            min(amount / 1000.0, 5.0),
        ],
        dtype=np.float32,
    )


def featurize_bank_statement(
    amount: float,
    date: str | datetime,
    description: str = "",
    merchant: str = "",
    mcc: int | None = None,
) -> np.ndarray:
    """
    Featurize an API bank-statement line.

    description / merchant / mcc are accepted for API compatibility but do not
    enter the feature vector (no train-time equivalents in ULB).
    """
    del description, merchant, mcc  # explicit: not used in model features
    dt = _parse_date(date)
    return _numeric_features(float(amount), dt.hour, dt.weekday())


def featurize_dataset_row(time_elapsed: float, amount: float) -> np.ndarray:
    """Featurize a ULB row from Time (seconds elapsed) + Amount only."""
    hour = int((float(time_elapsed) % 86400) // 3600)
    dow = int(float(time_elapsed) // 86400) % 7
    return _numeric_features(float(amount), hour, dow)


def featurize_dataframe(df) -> np.ndarray:
    """Vectorize a transactions DataFrame with columns time, amount."""
    rows = [
        featurize_dataset_row(float(t), float(a))
        for t, a in zip(df["time"].tolist(), df["amount"].tolist())
    ]
    return np.vstack(rows).astype(np.float32)


# Sample API payloads. "expect" matches what amount/time features can do
# (description/merchant/MCC are not scored — see README Limitations).
SAMPLE_BANK_STATEMENTS: list[dict[str, Any]] = [
    {
        "name": "routine_grocery",
        "expect": "low score — typical evening mid-amount (memo not scored)",
        "statement": {
            "amount": 54.23,
            "date": "2024-06-12T18:42:00",
            "description": "WHOLEFOODS MARKET #1024 SAN FRANCISCO CA",
            "merchant": "Whole Foods",
            "mcc": 5411,
        },
    },
    {
        "name": "morning_coffee",
        "expect": "low score — small daytime amount (memo not scored)",
        "statement": {
            "amount": 6.75,
            "date": "2024-06-13T08:15:00",
            "description": "STARBUCKS STORE 20491 SEATTLE WA",
            "merchant": "Starbucks",
            "mcc": 5814,
        },
    },
    {
        "name": "overnight_foreign_wire_like",
        "expect": "elevated but often below threshold — reacts to odd hour + large amount, not wire-transfer text",
        "statement": {
            "amount": 2499.99,
            "date": "2024-06-14T03:17:00",
            "description": "INTL WIRE TRANSFER BENEFICIARY UNKNOWN CITY",
            "merchant": "WIRE*INTL",
            "mcc": 4829,
        },
    },
    {
        "name": "micro_card_test",
        "expect": "may elevate on tiny amount + overnight hour only — CARD VERIFICATION text is ignored",
        "statement": {
            "amount": 1.00,
            "date": "2024-06-14T02:05:00",
            "description": "CARD VERIFICATION AUTH TEMP HOLD",
            "merchant": "VERIFY*AUTH",
            "mcc": 5999,
        },
    },
    {
        "name": "electronics_big_ticket",
        "expect": "elevated on large evening amount — product/merchant text is ignored",
        "statement": {
            "amount": 1299.00,
            "date": "2024-06-15T21:50:00",
            "description": "BEST BUY #473 ONLINE DIGITAL DOWNLOAD",
            "merchant": "Best Buy",
            "mcc": 5732,
        },
    },
]
