# Credit Card Fraud Detection

End-to-end fraud pipeline (Postgres → PyTorch → FastAPI) under an intentional constraint: **exclude ULB `V1`–`V28`** so `/predict` can score realistic bank-statement JSON (`amount` + `date`). Description/merchant/MCC are accepted for API realism but **not scored**. I compared an unsupervised autoencoder to a class-weighted classifier on that minimal feature set: the classifier beats prevalence by a wide margin (**12.0×** lift); the autoencoder does not (**1.4×** ≈ random) — a useful negative finding, not a second production model. Precision at a fixed 1% FPR budget is **3.7%** — too low to deploy without the merchant/MCC signal we left out.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikhiltm/ml-showcase/blob/main/fraud-detection/notebooks/colab_fraud_detection.ipynb)

Numbers below are from the latest Colab run (20 epochs, test fraud rate 0.172%). Regenerated locally as `artifacts/metrics_summary.md`.

## Headline lift

Raw PR-AUC is hard to read alone; **lift = PR-AUC / prevalence baseline (0.00172)** is the interpretable number.

| Model | PR-AUC | Baseline | Lift |
|-------|--------|----------|------|
| Autoencoder | 0.0024 | 0.00172 | **1.4×** (≈ random) |
| Classifier | 0.0206 | 0.00172 | **12.0×** |

**Takeaway:** under amount/datetime-only features, unsupervised reconstruction does not separate fraud from noise; supervised learning still recovers meaningful lift over random.

## FPR trade-off tables

Thresholds use a **fixed false-positive budget** (ops framing: “we can only review ~X% of legit volume”), not max-F1 on a flat PR curve. Default serving/eval threshold targets **1% FPR**.

### Autoencoder (`v1.0.0-autoencoder`)

| Target FPR | Achieved FPR | Recall | Precision | Threshold |
|------------|--------------|--------|-----------|-----------|
| 0.1% | 0.0010 | 0.000 | 0.000 | 0.9991 |
| 1.0% | 0.0100 | 0.010 | 0.002 | 0.9901 |
| 5.0% | 0.0501 | 0.082 | 0.003 | 0.9523 |

Top-k alerts: top 50 / 100 → 0 TP; top 200 → 1 TP (precision 0.005).

### Classifier (`v1.1.0-classifier`)

| Target FPR | Achieved FPR | Recall | Precision | Threshold |
|------------|--------------|--------|-----------|-----------|
| 0.1% | 0.0010 | 0.051 | 0.081 | 0.9244 |
| 1.0% | 0.0100 | 0.224 | 0.037 | 0.8661 |
| 5.0% | 0.0500 | 0.327 | 0.011 | 0.7522 |

At **1% FPR**, precision is **3.7%** (~27 alerts per true fraud) — usable as a ranking signal for demos, not as a deployable alert queue. Top 50 alerts: 5 TP / 49 FP (precision 0.093).

## Why precision stays low: feature collisions

**34,922 unique / 56,962 test rows** → **61.3%** unique vectors (**38.7%** share an amount/time signature with at least one other row).

That collision rate is a **direct cause** of low precision: the model cannot tell apart fraud and legitimate traffic that land on the same amount/time vector, so many alerts at a fixed FPR budget are structural false positives — not just a poorly chosen threshold.

## Limitations

1. **Autoencoder is non-functional for fraud ranking here** (1.4× baseline). Treat it as a compared unsupervised baseline / negative result, not a working detector.
2. **Precision at 1% FPR (3.7%) is far too low for production review queues** without merchant/MCC/description features explicitly excluded in v1.
3. **Feature collisions** make many fraud/legit pairs indistinguishable under the current feature set.
4. **Demo memos are not text tests** — “WIRE TRANSFER” / “CARD VERIFICATION” strings do not enter the score; sample `expect` labels describe amount/time behavior only. (Overnight wire-like sample scored 0.78 — elevated, still below the 0.87 threshold.)

## Future work

- Merchant / MCC / description embeddings once matched to a **labeled real-world statement dataset** (ULB does not provide that).
- Optional dual path: V1–V28 research model (benchmark ceiling) vs statement-compatible production model.

---

## How to run

**Colab:** [`notebooks/colab_fraud_detection.ipynb`](notebooks/colab_fraud_detection.ipynb) → set Neon URL + Kaggle key → `run_pipeline(epochs=20)`.

**Local:**

```bash
cd fraud-detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && cp .env.example .env
export PYTHONPATH=.
python scripts/run_pipeline.py --epochs 20
uvicorn src.api.main:app --reload
```

## API

```http
POST /predict?model=classifier
```

```json
{
  "amount": 2499.99,
  "date": "2024-06-14T03:17:00",
  "description": "INTL WIRE TRANSFER BENEFICIARY UNKNOWN CITY",
  "merchant": "WIRE*INTL",
  "mcc": 4829
}
```

Only `amount` + `date` affect the score.

## License

MIT
