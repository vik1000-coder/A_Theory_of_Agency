#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$HOME/.config/adfe/xai.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOME/.config/adfe/xai.env"
  set +a
fi

if [[ -z "${XAI_API_KEY:-}" ]]; then
  echo "XAI_API_KEY is required. Set it in the environment or ~/.config/adfe/xai.env." >&2
  exit 2
fi

GENERATION_WORKERS="${GENERATION_WORKERS:-1}"
SCORE_WORKERS="${SCORE_WORKERS:-4}"
SENSITIVITY_WORKERS="${SENSITIVITY_WORKERS:-8}"

echo "== Preflight =="
uv run pytest
uv run python -m adfe_runner doctor --config configs/v2_clean_local_grok.yml
uv run python -m adfe_runner audit-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok \
  --expect-full

echo "== Sample manifests =="
uv run python -m adfe_runner export-v2-experiment-sample \
  --config configs/v2_clean_local_grok.yml \
  --source-run-id adfe_v2_clean_local_grok \
  --sample-size 300 \
  --out data/experiment_samples/policy_ablation_300_keys.json
uv run python -m adfe_runner export-regression-gate \
  --config configs/v2_clean_local_grok.yml \
  --source-run-id adfe_v2_clean_local_grok \
  --out data/regression_gates/civic_role_gate.json

echo "== Full remediation =="
uv run python -m adfe_runner iterate-v2 \
  --config configs/role_policy_remediation_grok.yml \
  --cycles 1 \
  --batch-size all \
  --run-id adfe_role_policy_remediation_grok \
  --generation-workers "$GENERATION_WORKERS" \
  --workers "$SCORE_WORKERS"
uv run python -m adfe_runner audit-v2 \
  --config configs/role_policy_remediation_grok.yml \
  --run-id adfe_role_policy_remediation_grok \
  --expect-full

echo "== Remediation judge sensitivity =="
uv run python -m adfe_runner judge-sensitivity-v2 \
  --config configs/role_policy_remediation_grok.yml \
  --run-id adfe_role_policy_remediation_grok \
  --judge qwen3:8b \
  --sample-strategy stratified \
  --sample-size 300 \
  --sample-seed 20260620 \
  --artifact-name qwen3_8b_stratified_300 \
  --workers "$SENSITIVITY_WORKERS" \
  --score-json-retry 2

echo "== Human review packet =="
uv run python -m adfe_runner export-ratings-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok \
  --max-items 120

echo "== Policy ablations =="
for cfg in configs/ablations/*.yml; do
  run_id="$(basename "$cfg" .yml)"
  uv run python -m adfe_runner iterate-v2 \
    --config "$cfg" \
    --cycles 1 \
    --batch-size all \
    --run-id "$run_id" \
    --generation-workers "$GENERATION_WORKERS" \
    --workers "$SCORE_WORKERS"
  uv run python -m adfe_runner audit-v2 \
    --config "$cfg" \
    --run-id "$run_id" \
    --expect-count 300
done

echo "== Stress mini-set =="
uv run python -m adfe_runner iterate-v2 \
  --config configs/stress_baseline_grok.yml \
  --cycles 1 \
  --batch-size all \
  --run-id adfe_stress_baseline_grok \
  --generation-workers "$GENERATION_WORKERS" \
  --workers "$SCORE_WORKERS"
uv run python -m adfe_runner audit-v2 \
  --config configs/stress_baseline_grok.yml \
  --run-id adfe_stress_baseline_grok \
  --expect-count 840
uv run python -m adfe_runner iterate-v2 \
  --config configs/stress_role_policy_grok.yml \
  --cycles 1 \
  --batch-size all \
  --run-id adfe_stress_role_policy_grok \
  --generation-workers "$GENERATION_WORKERS" \
  --workers "$SCORE_WORKERS"
uv run python -m adfe_runner audit-v2 \
  --config configs/stress_role_policy_grok.yml \
  --run-id adfe_stress_role_policy_grok \
  --expect-count 840

echo "== Regression gate and paper artifacts =="
set +e
uv run python -m adfe_runner check-regression-gate \
  --baseline-run-id adfe_v2_clean_local_grok \
  --candidate-run-id adfe_role_policy_remediation_grok \
  --gate-path data/regression_gates/civic_role_gate.json \
  --out paper/neurips_workshop/generated/regression_gate_summary.json
gate_status=$?
set -e

uv run python -m adfe_runner build-paper-artifacts
python3 /Users/vik/.codex/plugins/cache/openai-bundled/latex/0.2.3/scripts/compile_latex.py \
  /Users/vik/Developer/A_Theory_of_Agency/paper/neurips_workshop/paper.tex

exit "$gate_status"
