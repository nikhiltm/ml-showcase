"""FastAPI: submit a bank statement line → fraud verdict."""

import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import ARTIFACTS_DIR
from src.ml.registry import MODELS, LoadedModel, load_model, score_statements

_models: dict[str, LoadedModel] = {}
_thresholds: dict[str, float] = {}


class BankStatementRequest(BaseModel):
    """One line from a credit card / bank statement (hand-typed JSON OK)."""

    amount: float = Field(..., ge=0, description="Charge amount in USD")
    date: str = Field(..., description="ISO datetime, e.g. 2024-06-12T18:42:00")
    description: str = Field(..., description="Statement memo / description")
    merchant: str = Field("", description="Merchant name if known")
    mcc: int | None = Field(None, description="Optional merchant category code")


class FraudPrediction(BaseModel):
    is_fraudulent: bool
    label: Literal["fraud", "legitimate"]
    fraud_probability: float = Field(..., ge=0, le=1)
    threshold: float
    model_version: str
    model_type: str
    statement: BankStatementRequest


def _load_threshold(model_key: str) -> float:
    eval_path = ARTIFACTS_DIR / f"evaluation_{model_key}.json"
    if eval_path.exists():
        return json.loads(eval_path.read_text()).get("threshold", 0.5)
    return 0.5


def _predict(statement: BankStatementRequest, model_key: str) -> FraudPrediction:
    if model_key not in _models:
        raise HTTPException(
            503,
            f"Model '{model_key}' not loaded. Train first, then restart the API.",
        )

    loaded = _models[model_key]
    fraud_prob = float(score_statements(loaded, [statement.model_dump()])[0])
    threshold = _thresholds.get(model_key, 0.5)
    is_fraud = fraud_prob >= threshold

    return FraudPrediction(
        is_fraudulent=is_fraud,
        label="fraud" if is_fraud else "legitimate",
        fraud_probability=fraud_prob,
        threshold=threshold,
        model_version=loaded.spec.version,
        model_type=loaded.spec.model_type,
        statement=statement,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models, _thresholds
    for model_key in MODELS:
        try:
            _models[model_key] = load_model(model_key)
            _thresholds[model_key] = _load_threshold(model_key)
        except FileNotFoundError:
            pass
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Submit a bank/credit-card statement line and get is_fraudulent back. "
        "The model uses amount + datetime features only: ULB V1–V28 PCA columns "
        "are intentionally excluded so inference matches a real statement API. "
        "description/merchant/mcc are accepted and echoed but not scored. "
        "Expect lower PR-AUC than V1–V28 benchmarks — that is the realism trade-off."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.post(
    "/predict",
    response_model=FraudPrediction,
    summary="Predict if a bank statement line is fraudulent",
)
def predict(
    statement: BankStatementRequest,
    model: str = Query("classifier", enum=list(MODELS.keys())),
):
    return _predict(statement, model)


@app.post("/score", response_model=FraudPrediction, include_in_schema=False)
def score(
    statement: BankStatementRequest,
    model: str = Query("classifier", enum=list(MODELS.keys())),
):
    return _predict(statement, model)
