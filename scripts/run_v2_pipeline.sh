#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${XAI_API_KEY:-}" ]]; then
  set +o pipefail
  XAI_API_KEY="$(textutil -convert txt -stdout "$HOME/Downloads/send.rtf" 2>/dev/null | perl -ne 'if(/(xai-[A-Za-z0-9_-]+)/){print $1; exit}' || true)"
  set -o pipefail
fi
if [[ -z "${XAI_API_KEY:-}" ]]; then
  echo "XAI_API_KEY not set and not found in Downloads/send.rtf" >&2
  exit 2
fi
export XAI_API_KEY

LOCAL_RUN_ID="${LOCAL_RUN_ID:-adfe_v2_clean_local_grok}"
FRONTIER_RUN_ID="${FRONTIER_RUN_ID:-adfe_v2_frontier_grok_exploratory}"
WORKERS="${WORKERS:-4}"

uv run python -m adfe_runner iterate-v2 \
  --config configs/v2_clean_local_grok.yml \
  --cycles 1 \
  --batch-size all \
  --run-id "$LOCAL_RUN_ID" \
  --workers "$WORKERS"

uv run python -m adfe_runner audit-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id "$LOCAL_RUN_ID" \
  --expect-full

uv run python -m adfe_runner judge-sensitivity-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id "$LOCAL_RUN_ID" \
  --judge qwen3:8b \
  --workers "$WORKERS"

uv run python -m adfe_runner export-ratings-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id "$LOCAL_RUN_ID" \
  --max-items 120

uv run python -m adfe_runner build-site --run-id "$LOCAL_RUN_ID"

uv run python -m adfe_runner doctor --config configs/v2_frontier_grok_exploratory.yml

uv run python -m adfe_runner iterate-v2 \
  --config configs/v2_frontier_grok_exploratory.yml \
  --cycles 1 \
  --batch-size all \
  --run-id "$FRONTIER_RUN_ID" \
  --workers "$WORKERS"

uv run python -m adfe_runner audit-v2 \
  --config configs/v2_frontier_grok_exploratory.yml \
  --run-id "$FRONTIER_RUN_ID" \
  --expect-full

uv run python -m adfe_runner build-site --run-id "$LOCAL_RUN_ID"
