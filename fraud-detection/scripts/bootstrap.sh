#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

if ! docker compose ps --status running 2>/dev/null | grep -q postgres; then
  echo "Starting local Postgres (docker compose up -d)..."
  docker compose up -d
  for i in {1..30}; do
    if docker compose exec -T postgres pg_isready -U fraud -d fraud >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

CSV_PATH="${CSV_PATH:-$(find ~/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud -name 'creditcard.csv' 2>/dev/null | head -1)}"
ARGS=()
if [[ -n "$CSV_PATH" && -f "$CSV_PATH" ]]; then
  ARGS+=(--csv "$CSV_PATH")
else
  echo "Note: CSV not in kagglehub cache; pipeline will download via kagglehub."
fi

.venv/bin/python scripts/run_pipeline.py "${ARGS[@]}" --epochs 20
