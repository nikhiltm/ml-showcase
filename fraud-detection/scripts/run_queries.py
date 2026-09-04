"""Run analytics queries and print results."""

from src.db.connection import get_session
from src.db.queries import (
    compare_models,
    daily_alert_volume,
    false_positive_rate_by_bucket,
    fraud_capture_rate,
    fraud_rate_by_amount_bucket,
    top_risk_transactions,
)


def main():
    session = get_session()
    try:
        print("=== Model comparison ===")
        for row in compare_models(session):
            print(row)

        print("\n=== Top 20 highest-risk transactions (last 7 days) ===")
        for row in top_risk_transactions(session, limit=20):
            print(row)

        print("\n=== Fraud rate by amount bucket ===")
        for row in fraud_rate_by_amount_bucket(session):
            print(row)

        print("\n=== False positive rate by bucket ===")
        for row in false_positive_rate_by_bucket(session):
            print(row)

        print("\n=== Fraud capture rate by bucket ===")
        for row in fraud_capture_rate(session):
            print(row)

        print("\n=== Daily alert volume ===")
        for row in daily_alert_volume(session):
            print(row)
    finally:
        session.close()


if __name__ == "__main__":
    main()
