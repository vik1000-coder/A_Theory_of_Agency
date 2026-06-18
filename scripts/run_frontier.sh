#!/bin/bash
# Unattended, resumable frontier (Grok) audit + V judge gate. Launched by the launchd agent
# com.adfe.frontier (wrapped in `caffeinate -is`). Both steps resume from checkpoints, so a
# kill just continues. Exits 0 only when both finish (tells launchd to stop relaunching).
#
# Reads the xAI key from ~/.config/adfe/xai.env (never committed). Runs the V validation first
# (local judge, ~30 min) then the frontier study, so they don't contend for Ollama.

set -u
REPO="$HOME/Developer/A_Theory_of_Agency"
cd "$REPO" || exit 1
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
UV="/opt/homebrew/bin/uv"
[ -f "$HOME/.config/adfe/xai.env" ] && source "$HOME/.config/adfe/xai.env"

V_JSON="runs/judge_validation_neutrality_qwen3_8b/validation.json"
F_JSON="runs/adfe_frontier_grok/analysis.json"
log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*"; }

if [ -f runs/FRONTIER_DONE ]; then
  log "FRONTIER_DONE present; exiting 0"
  exit 0
fi

log "start (V: $(wc -l < runs/judge_validation_neutrality_qwen3_8b/results.jsonl 2>/dev/null || echo 0)/909, frontier gens: $(wc -l < runs/adfe_frontier_grok/generations.jsonl 2>/dev/null || echo 0)/630)"

# 1) V (viewpoint-symmetry) judge gate — local judge, resumes from checkpoint.
"$UV" run python -m adfe_runner validate-judge --task neutrality --per-type 40 || log "validate-judge exited $?"

# 2) Frontier study — Grok via API + local judge scoring, resumes by run-id.
"$UV" run python -m adfe_runner iterate --config configs/frontier_xai.yml \
  --cycles 1 --batch-size all --run-id adfe_frontier_grok || log "iterate exited $?"

if [ -f "$V_JSON" ] && [ -f "$F_JSON" ]; then
  log "FRONTIER + V COMPLETE"
  touch runs/FRONTIER_DONE
  exit 0
fi
log "incomplete (V=$( [ -f "$V_JSON" ] && echo y || echo n ) frontier=$( [ -f "$F_JSON" ] && echo y || echo n )); launchd will retry"
exit 1
