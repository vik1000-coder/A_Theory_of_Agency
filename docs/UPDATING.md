# Updating the Public Site

The GitHub Pages site has two pieces:

- `docs/index.html`: hand-edited page structure and wording.
- `docs/data/summary.js`: generated data from run artifacts. Do not hand-edit it.

GitHub Pages serves `docs/` after pushes to `main`.

## Baseline Site Refresh

```bash
uv run python -m adfe_runner audit-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok \
  --expect-full

uv run python -m adfe_runner build-site \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok
```

Preview locally:

```bash
python3 -m http.server 8099 --directory docs
```

Open `http://localhost:8099`.

## Judge Sensitivity Refresh

The site uses the latest v2 comparison artifact under `runs/<run_id>/v2/comparison_*.json`.
Regenerate the canonical stratified sample with:

```bash
uv run python -m adfe_runner judge-sensitivity-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok \
  --judge qwen3:8b \
  --sample-strategy stratified \
  --sample-size 300 \
  --sample-seed 20260620 \
  --artifact-name qwen3_8b_stratified_300 \
  --workers 8 \
  --score-json-retry 2
```

Then rerun `build-site`.

## Remediation Results

The public site currently reports the baseline evaluation. After the matched remediation run is
complete, first regenerate paper tables:

```bash
uv run python -m adfe_runner build-paper-artifacts \
  --baseline-run-id adfe_v2_clean_local_grok \
  --remediation-run-id adfe_role_policy_remediation_grok \
  --frontier-run-id adfe_v2_frontier_grok_exploratory
```

Only add remediation results to `docs/index.html` after the remediation run passes:

```bash
uv run python -m adfe_runner audit-v2 \
  --config configs/role_policy_remediation_grok.yml \
  --run-id adfe_role_policy_remediation_grok \
  --expect-full
```

## Publish

```bash
git add docs paper/neurips_workshop/generated
git commit -m "site: update role-conditioned fairness results"
git push
```

Before publishing, check that visible text does not refer to internal draft labels as if a reader
already knows the project history.
