# Project Status

_Last updated: 2026-06-25._

This branch frames the repository as a role-counterfactual civic AI evaluation for a
ResponsibleFM-style workshop paper.

## Current State

| Area | Status |
| --- | --- |
| Submission branch | Active: `codex/neurips-workshop-submission` |
| Baseline local evaluation | Complete: `runs/adfe_v2_clean_local_grok` with 2,100 judged rows |
| Matched remediation | Complete: `runs/adfe_role_policy_remediation_grok` with 2,100 judged rows |
| Baseline judge sensitivity | Complete for 300-row stratified Qwen sample |
| Remediation judge sensitivity | Complete for 300-row stratified Qwen sample |
| Policy ablations | Complete: four 300-key targeted ablations |
| Stress mini-set | Complete: baseline and role-policy arms, 840 rows each |
| Regression gate | Complete and passing for remediation |
| Exploratory frontier arm | Complete; not pooled with local evidence |
| Human review packet | Export path implemented; completed two-rater labels not yet imported |
| Workshop paper package | Active under `paper/neurips_workshop/` |
| GitHub Pages report | Regenerated from run artifacts |

## Current Takeaway

The baseline found a real product problem: role prompts can change whether lawful civic prompts are
answered or refused, and mirrored lawful viewpoints can receive asymmetric treatment. The strongest
signal is one-sided refusal across mirrored pairs, not a single ideological bias score.

The role-policy remediation helps the clearest failure mode. It reduces one-sided refusals from
72/420 mirrored comparisons to 37/420 and improves the targeted failure sample. It does not solve
everything: aggregate refusal only moves modestly, non-refusal quality falls slightly on the full
matched grid, and model/role effects are uneven.

## Evidence Snapshot

| Result | Baseline | Role-policy remediation |
| --- | ---: | ---: |
| Full-grid refusal rate | 14.3% | 13.4% |
| Full-grid over-refusal rate | 13.1% | 12.4% |
| One-sided mirrored-pair refusals | 72 / 420 | 37 / 420 |
| Targeted-sample refusal rate | 81.0% | 51.0% |
| Regression-gate one-sided refusal | 80.0% | 21.1% |

## What Remains

- Import two-rater human labels and report agreement.
- Decide how strongly the paper should claim role-profile improvements, given judge sensitivity.
- Tighten examples in the paper appendix after human review identifies representative failures.
- Recompile the blind PDF and rescan for author-identifying content before submission.

## Regeneration Commands

```bash
uv run pytest
uv run python -m adfe_runner build-paper-artifacts
uv run python -m adfe_runner build-site \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok
```

Compile the draft:

```bash
python3 /Users/vik/.codex/plugins/cache/openai-bundled/latex/0.2.3/scripts/compile_latex.py \
  /Users/vik/Developer/A_Theory_of_Agency/paper/neurips_workshop/paper.tex
```
