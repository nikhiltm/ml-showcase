"""Analytics queries that make Postgres useful beyond a data dump."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session


def _fetch_dicts(session: Session, sql: str, params: dict | None = None) -> list[dict]:
    """Run SQL via the connection (avoids ORM result loading issues on Colab)."""
    result = session.connection().execute(text(sql), params or {})
    return [dict(row._mapping) for row in result]


def top_risk_transactions(
    session: Session,
    *,
    limit: int = 20,
    since: datetime | None = None,
    model_version: str | None = None,
) -> list[dict]:
    """Top N highest-risk transactions in a time window (default: last 7 days)."""
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(days=7)

    sql = """
        SELECT t.id, t.time, t.amount, p.fraud_score, p.predicted_class,
               p.model_version, p.predicted_at, t.actual_class
        FROM predictions p
        JOIN transactions t ON t.id = p.transaction_id
        WHERE p.predicted_at >= :since
    """
    params: dict = {"since": since, "limit": limit}
    if model_version:
        sql += " AND p.model_version = :model_version"
        params["model_version"] = model_version
    sql += " ORDER BY p.fraud_score DESC LIMIT :limit"
    return _fetch_dicts(session, sql, params)


def fraud_rate_by_amount_bucket(session: Session) -> list[dict]:
    """Fraud rate grouped by transaction amount bucket."""
    sql = """
        SELECT
            CASE
                WHEN amount < 10 THEN '< $10'
                WHEN amount < 50 THEN '$10–$50'
                WHEN amount < 100 THEN '$50–$100'
                WHEN amount < 500 THEN '$100–$500'
                ELSE '$500+'
            END AS amount_bucket,
            COUNT(*) AS total,
            SUM(actual_class) AS fraud_count,
            SUM(actual_class) * 100.0 / COUNT(*) AS fraud_rate_pct
        FROM transactions
        GROUP BY 1
        ORDER BY 1
    """
    return _fetch_dicts(session, sql)


def _confusion_counts(session: Session, model_version: str | None = None) -> dict:
    sql = """
        SELECT p.predicted_class, t.actual_class, COUNT(*) AS count
        FROM predictions p
        JOIN transactions t ON t.id = p.transaction_id
    """
    params: dict = {}
    if model_version:
        sql += " WHERE p.model_version = :model_version"
        params["model_version"] = model_version
    sql += " GROUP BY p.predicted_class, t.actual_class"

    rows = _fetch_dicts(session, sql, params)
    tp = fp = fn = tn = 0
    for row in rows:
        pred, actual, count = row["predicted_class"], row["actual_class"], row["count"]
        if pred == 1 and actual == 1:
            tp += count
        elif pred == 1 and actual == 0:
            fp += count
        elif pred == 0 and actual == 1:
            fn += count
        else:
            tn += count

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def model_performance_summary(session: Session, model_version: str | None = None) -> dict:
    """Precision-style summary comparing predictions to actual labels."""
    return _confusion_counts(session, model_version)


def compare_models(session: Session) -> list[dict]:
    """Side-by-side performance for each model_version in predictions."""
    versions = [
        row["model_version"]
        for row in _fetch_dicts(
            session,
            "SELECT DISTINCT model_version FROM predictions ORDER BY model_version",
        )
    ]
    results = []
    for version in versions:
        stats = _confusion_counts(session, model_version=version)
        stats["model_version"] = version
        results.append(stats)
    return results


def false_positive_rate_by_bucket(
    session: Session,
    *,
    model_version: str | None = None,
) -> list[dict]:
    """False positive rate among flagged transactions, by amount bucket."""
    sql = """
        SELECT
            CASE
                WHEN t.amount < 10 THEN '< $10'
                WHEN t.amount < 50 THEN '$10–$50'
                WHEN t.amount < 100 THEN '$50–$100'
                WHEN t.amount < 500 THEN '$100–$500'
                ELSE '$500+'
            END AS amount_bucket,
            COUNT(*) AS flagged,
            SUM(CASE WHEN t.actual_class = 0 THEN 1 ELSE 0 END) AS false_positives,
            SUM(CASE WHEN t.actual_class = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
                AS false_positive_rate_pct
        FROM predictions p
        JOIN transactions t ON t.id = p.transaction_id
        WHERE p.predicted_class = 1
    """
    params: dict = {}
    if model_version:
        sql += " AND p.model_version = :model_version"
        params["model_version"] = model_version
    sql += " GROUP BY 1 ORDER BY 1"
    return _fetch_dicts(session, sql, params)


def daily_alert_volume(
    session: Session,
    *,
    model_version: str | None = None,
) -> list[dict]:
    """Number of flagged transactions per day."""
    bind = session.get_bind()
    dialect = bind.dialect.name if bind is not None else "postgresql"

    if dialect == "sqlite":
        day_expr = "strftime('%Y-%m-%d', p.predicted_at)"
    else:
        day_expr = "date_trunc('day', p.predicted_at)"

    sql = f"""
        SELECT
            {day_expr} AS day,
            COUNT(*) AS total_scored,
            SUM(p.predicted_class) AS alerts,
            SUM(CASE WHEN t.actual_class = 1 THEN 1 ELSE 0 END) AS actual_fraud
        FROM predictions p
        JOIN transactions t ON t.id = p.transaction_id
    """
    params: dict = {}
    if model_version:
        sql += " WHERE p.model_version = :model_version"
        params["model_version"] = model_version
    sql += f" GROUP BY {day_expr} ORDER BY 1"
    return _fetch_dicts(session, sql, params)


def fraud_capture_rate(
    session: Session,
    *,
    model_version: str | None = None,
) -> list[dict]:
    """Recall (fraud capture rate) by amount bucket."""
    sql = """
        SELECT
            CASE
                WHEN t.amount < 10 THEN '< $10'
                WHEN t.amount < 50 THEN '$10–$50'
                WHEN t.amount < 100 THEN '$50–$100'
                WHEN t.amount < 500 THEN '$100–$500'
                ELSE '$500+'
            END AS amount_bucket,
            SUM(t.actual_class) AS total_fraud,
            SUM(CASE WHEN p.predicted_class = 1 THEN 1 ELSE 0 END) AS caught,
            SUM(CASE WHEN p.predicted_class = 1 THEN 1 ELSE 0 END) * 100.0
                / NULLIF(SUM(t.actual_class), 0) AS capture_rate_pct
        FROM predictions p
        JOIN transactions t ON t.id = p.transaction_id
        WHERE t.actual_class = 1
    """
    params: dict = {}
    if model_version:
        sql += " AND p.model_version = :model_version"
        params["model_version"] = model_version
    sql += " GROUP BY 1 ORDER BY 1"
    return _fetch_dicts(session, sql, params)
