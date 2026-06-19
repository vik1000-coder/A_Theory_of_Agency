# Updating the public site after a run

The site is two pieces:

- **`docs/index.html`** — the hand-designed page (problem, approach, data, judge, sensitivity, findings).
  Edit this only to change wording or design.
- **`docs/data/summary.js`** — the data, **generated** from run artifacts. Never hand-edit it.

GitHub Pages serves `docs/` on every push to `main`, so updating the site = regenerating the
data file and pushing.

## After every major run

1. Make sure the run finished analysis (`iterate` writes `analysis.json`; if you only have
   generations/scores, run `analyze --run-id <id>` first).

2. (Recommended) refresh the judge-validation gate so the page's κ is current:
   ```bash
   uv run python -m adfe_runner validate-judge --judge qwen3:8b
   ```

3. Audit the run artifacts. This fails on duplicate rows, unresolved generation failures,
   missing scores, or incomplete full-factorial runs:
   ```bash
   uv run python -m adfe_runner audit-run --run-id <run_id> --expect-full
   ```

4. If an alternate judge is available, refresh the judge-sensitivity artifact. The site will
   pick up the latest `runs/<run_id>/judge_sensitivity/*/comparison.json`:
   ```bash
   XAI_API_KEY=... uv run python -m adfe_runner judge-sensitivity \
     --config configs/clean_local.yml --run-id <run_id> \
     --judge xai:grok-4.3 --score-json-retry 2 --workers 4
   ```

5. Regenerate the page data. Pin the run with `--run-id`:
   ```bash
   uv run python -m adfe_runner build-site --run-id <run_id>
   ```
   The command prints the run id, its `contaminated` flag, and the judge κ it baked in.

6. Preview locally before pushing:
   ```bash
   python3 -m http.server 8099 --directory docs   # then open http://localhost:8099
   ```

7. Commit and push — Pages redeploys automatically (usually live within a minute):
   ```bash
   git add docs && git commit -m "site: update from <run_id>" && git push
   ```

## Citable vs. preliminary

`build-site` reads `run_meta.contaminated`. A run produced with `--calibrate` (or any pre-fix
run) is flagged contaminated and the page shows a **Preliminary** banner. For a headline you
intend to stand behind, point `--run-id` at a **frozen** run from `configs/clean_local.yml`
(no `--calibrate`), so the banner disappears and the numbers are citable.

## What the page shows

The judge-validation block (κ, accuracy, blind spots) is independent of the audited run and is
always the clean gate result. The judge-sensitivity block comes from the latest alternate-judge
comparison for the chosen run. The findings block (agency gradient, interval tests, refusal
asymmetry) comes from the chosen run.
