# Updating the Public Site

The GitHub Pages site has two pieces:

- `docs/index.html`: hand-edited page structure and wording.
- `docs/data/summary.js`: generated data from run and paper artifacts. Do not hand-edit it.

GitHub Pages serves `docs/` after pushes to the publishing branch.

## Refresh Checklist

Run the audits for the source experiments:

```bash
uv run python -m adfe_runner audit-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok \
  --expect-full

uv run python -m adfe_runner audit-v2 \
  --config configs/role_policy_remediation_grok.yml \
  --run-id adfe_role_policy_remediation_grok \
  --expect-full
```

Regenerate paper-backed metrics first. The site reads the workshop block from
`paper/neurips_workshop/generated/paper_metrics.json`.

```bash
uv run python -m adfe_runner build-paper-artifacts
```

Then regenerate the Pages data file:

```bash
uv run python -m adfe_runner build-site \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok
```

Preview locally:

```bash
python3 -m http.server 8099 --directory docs
```

Open `http://localhost:8099`.

## Human Review Update

After two-rater labels are imported, rerun:

```bash
uv run python -m adfe_runner import-ratings-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok \
  --ratings path/to/completed_ratings.csv

uv run python -m adfe_runner build-paper-artifacts
uv run python -m adfe_runner build-site \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok
```

Update the visible limitations text in `docs/index.html` only after the generated human-rating
summary exists.

## Publish

```bash
git add docs paper/neurips_workshop/generated
git commit -m "site: update role-conditioned fairness results"
git push
```

Before publishing, check that the page reads as a standalone report for someone with no project
history.
