# Unattended multi-day run (launchd)

Runs the full pipeline — judge factuality validation, then the clean study, then
`build-site` — as a per-user background service that survives closing the terminal/editor
and is resumable, so an OS kill or reboot just continues where it left off.

- `run_pipeline.sh` — the resumable pipeline (cd → validate → study → build-site). Exits 0
  only when both finish, which tells launchd to stop relaunching.
- `com.adfe.pipeline.plist` — the launchd agent. `caffeinate -is` keeps the Mac awake on AC;
  `KeepAlive {SuccessfulExit:false}` relaunches it if it's killed mid-run; `Nice 10` keeps it
  out of the way of interactive use.

## Install / start

```bash
cp scripts/com.adfe.pipeline.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.adfe.pipeline.plist
```

It starts immediately and on every login. Keep the Mac **plugged in** and **logged in**
(a per-user agent stops at logout; `caffeinate` only blocks sleep on AC power).

## Monitor

```bash
launchctl print gui/$(id -u)/com.adfe.pipeline | grep -iE 'state|pid'
tail -f runs/clean_study.log              # study progress
tail -n 3 runs/factuality_validation.log  # validation progress
```

Progress is also visible on disk: `runs/judge_validation_factuality_qwen3_8b/results.jsonl`
(validation checkpoint) and `runs/adfe_clean_local_main/generations.jsonl` (study).

## Stop / uninstall

```bash
launchctl bootout gui/$(id -u)/com.adfe.pipeline
rm ~/Library/LaunchAgents/com.adfe.pipeline.plist
```

## When it finishes

It writes `runs/PIPELINE_DONE` and regenerates `docs/data/summary.js`. Publish the updated
site with:

```bash
git add docs && git commit -m "site: clean study results" && git push
```
