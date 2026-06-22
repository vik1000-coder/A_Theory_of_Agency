#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f "$HOME/.config/adfe/xai.env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.config/adfe/xai.env"
fi

if [[ -z "${XAI_API_KEY:-}" ]]; then
  echo "XAI_API_KEY must be set or available in ~/.config/adfe/xai.env" >&2
  exit 2
fi
export XAI_API_KEY

BASELINE_RUN_ID="${BASELINE_RUN_ID:-adfe_v2_clean_local_grok}"
REMEDIATION_RUN_ID="${REMEDIATION_RUN_ID:-adfe_role_policy_remediation_grok}"
FRONTIER_RUN_ID="${FRONTIER_RUN_ID:-adfe_v2_frontier_grok_exploratory}"
WORKERS="${WORKERS:-4}"

uv run python -m adfe_runner iterate-v2 \
  --config configs/v2_clean_local_grok.yml \
  --cycles 1 \
  --batch-size all \
  --run-id "$BASELINE_RUN_ID" \
  --workers "$WORKERS"

uv run python -m adfe_runner audit-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id "$BASELINE_RUN_ID" \
  --expect-full

uv run python -m adfe_runner judge-sensitivity-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id "$BASELINE_RUN_ID" \
  --judge qwen3:8b \
  --sample-strategy stratified \
  --sample-size 300 \
  --sample-seed 20260620 \
  --artifact-name qwen3_8b_stratified_300 \
  --workers "$WORKERS" \
  --score-json-retry 2

uv run python -m adfe_runner export-ratings-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id "$BASELINE_RUN_ID" \
  --max-items 120

uv run python -m adfe_runner iterate-v2 \
  --config configs/role_policy_remediation_grok.yml \
  --cycles 1 \
  --batch-size all \
  --run-id "$REMEDIATION_RUN_ID" \
  --workers "$WORKERS"

uv run python -m adfe_runner audit-v2 \
  --config configs/role_policy_remediation_grok.yml \
  --run-id "$REMEDIATION_RUN_ID" \
  --expect-full

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

uv run python -m adfe_runner build-paper-artifacts \
  --baseline-run-id "$BASELINE_RUN_ID" \
  --remediation-run-id "$REMEDIATION_RUN_ID" \
  --frontier-run-id "$FRONTIER_RUN_ID"

uv run python -m adfe_runner build-site \
  --config configs/v2_clean_local_grok.yml \
  --run-id "$BASELINE_RUN_ID"
