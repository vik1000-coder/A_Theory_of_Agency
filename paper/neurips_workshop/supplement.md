# Supplement And Artifact Notes

This supplement describes the artifact package that accompanies the workshop draft. For a full
repository handoff, see [`../../ANALYSIS_HANDOFF.md`](../../ANALYSIS_HANDOFF.md).

## Artifact Contents

- Source code for generation, scoring, analysis, paper-artifact export, site export, regression
  gates, and human-rating import.
- Static prompt set, stress prompt set, source packets, role cards, remediation policy, ablation
  policies, targeted sample manifest, and regression-gate manifest.
- Canonical raw run artifacts under `runs/`.
- Generated CSV/JSON/TeX paper artifacts under `paper/neurips_workshop/generated/`.
- Compiled draft PDF at `paper/neurips_workshop/paper.pdf`.

## Canonical Runs

| Run ID | Purpose | Primary artifacts |
| --- | --- | --- |
| `adfe_v2_clean_local_grok` | Main local baseline, 2,100 judged rows | generations, primary scores, analysis, Qwen sensitivity sample, human-rating packet |
| `adfe_role_policy_remediation_grok` | Matched role-policy remediation, 2,100 judged rows | generations, primary scores, analysis, Qwen sensitivity sample |
| `adfe_v2_frontier_grok_exploratory` | Exploratory frontier arm | generations, scores, analysis |
| `adfe_stress_baseline_grok` | Stress-prompt baseline, 840 rows | generations, scores, analysis |
| `adfe_stress_role_policy_grok` | Stress-prompt role-policy arm, 840 rows | generations, scores, analysis |
| `no_viewpoint_parity` | Targeted ablation | generations, scores, analysis |
| `no_refusal_criteria` | Targeted ablation | generations, scores, analysis |
| `no_source_uncertainty` | Targeted ablation | generations, scores, analysis |
| `no_role_specific_rules` | Targeted ablation | generations, scores, analysis |

## Method Summary

The main baseline evaluates five local models, seven civic roles, thirty prompts, and two
role-presentation modes. Each row is one model output for one model, role, prompt, and mode.
The primary judge labels refusal, whether refusal was warranted, six answer-quality scores, and six
role-profile scores.

Matched-pair fairness uses six mirrored prompt pairs. A one-sided refusal occurs when the same
model, role, and mode refuses one side of a pair while answering the counterpart. Remediation
deltas are matched by `(model, role, agency_mode, prompt_id)`.

## Generated Tables

The paper uses generated artifacts rather than hand-entered results:

- `generated/numbers.tex`: headline number macros.
- `generated/paper_metrics.json`: machine-readable summary.
- `generated/baseline_role_effects_table.tex`: role-effect table included in the paper.
- `generated/tables/baseline_*.csv`: baseline aggregate tables.
- `generated/tables/remediation_matched_deltas.csv`: matched remediation deltas.
- `generated/tables/policy_ablation_deltas.csv`: ablation deltas.
- `generated/tables/stress_matched_deltas.csv`: stress-set deltas.
- `generated/tables/regression_gate_summary.csv`: regression-gate summary.
- `generated/tables/human_rating_summary.csv`: empty until completed human labels are imported.

## Reproducibility Path

```bash
uv sync --extra dev
uv run pytest
uv run python -m adfe_runner audit-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok \
  --expect-full
uv run python -m adfe_runner build-paper-artifacts
cd paper/neurips_workshop
latexmk -pdf paper.tex
```

## Human Review Status

The two-rater packet has been exported at `runs/adfe_v2_clean_local_grok/v2/rating_packet.csv`.
Completed human labels have not yet been imported, so the current claims should be read as
LLM-judge-calibrated. Refusal labels are more robust across judges than role-profile scores.

## Blind Submission Notes

Do not include public repository URLs, author names, local filesystem paths, acknowledgements, or
non-anonymous hosted links in the submitted PDF or supplement. If code is shared for review, package
it as an anonymous archive.
