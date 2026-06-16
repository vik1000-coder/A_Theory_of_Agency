#!/bin/bash
# Unattended, resumable ADFE pipeline. Launched by the launchd agent
# com.adfe.pipeline (wrapped in `caffeinate -is` to prevent sleep on AC power).
#
# Every step is resumable: validate-judge checkpoints per item, iterate skips
# already-generated/scored items. So if the OS kills this mid-run, launchd
# relaunches it (KeepAlive on non-zero exit) and it continues where it stopped.
# It exits 0 only when both the factuality validation and the study have
# completed, which tells launchd to stop relaunching.

set -u
REPO="$HOME/Developer/A_Theory_of_Agency"
cd "$REPO" || exit 1
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
UV="/opt/homebrew/bin/uv"
JUDGE="qwen3:8b"
RUN_ID="adfe_clean_local_main"
FACT_JSON="runs/judge_validation_factuality_qwen3_8b/validation.json"
STUDY_JSON="runs/$RUN_ID/analysis.json"

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*"; }

# Already finished a previous time -> nothing to do, let launchd stop.
if [ -f runs/PIPELINE_DONE ]; then
  log "PIPELINE_DONE present; exiting 0"
  exit 0
fi

log "pipeline start (factuality checkpoint: $(wc -l < runs/judge_validation_factuality_${JUDGE/:/_}/results.jsonl 2>/dev/null || echo 0) items)"

# 1) Judge validation on factuality (E). Resumes from checkpoint.
"$UV" run python -m adfe_runner validate-judge --task factuality --judge "$JUDGE" || log "validate-judge exited $?"

# 2) The clean study (frozen, held-out judge, full factorial). Resumes by run-id.
"$UV" run python -m adfe_runner iterate --config configs/clean_local.yml \
  --cycles 1 --batch-size all --run-id "$RUN_ID" || log "iterate exited $?"

# 3) Regenerate the site data from the completed study (push is left to a human).
"$UV" run python -m adfe_runner build-site --run-id "$RUN_ID" || log "build-site exited $?"

if [ -f "$FACT_JSON" ] && [ -f "$STUDY_JSON" ]; then
  log "pipeline COMPLETE"
  touch runs/PIPELINE_DONE
  exit 0
fi
log "pipeline incomplete (fact=$( [ -f "$FACT_JSON" ] && echo y || echo n ) study=$( [ -f "$STUDY_JSON" ] && echo y || echo n )); launchd will retry"
exit 1
