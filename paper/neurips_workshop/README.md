# NeurIPS Workshop Draft Package

Working title: **Role Prompts Are Policy Controls: Role-Conditioned Fairness Evaluation for
Civic AI Systems**.

This directory is organized as an anonymous ResponsibleFM-style workshop package.

## Files

- `paper.tex`: main anonymous workshop draft.
- `paper.pdf`: compiled draft PDF.
- `references.bib`: citation spine.
- `generated/`: source-backed tables and LaTeX number macros generated from run artifacts.
- `supplement.md`: anonymized artifact-package notes.
- `anonymization_checklist.md`: blind-submission checks to run before export.
- `../../ANALYSIS_HANDOFF.md`: full repository handoff for an outside analyst.

## Data Sources

The paper numbers are generated from the canonical run artifacts tracked under `../../runs/`:

| Run ID | Purpose |
| --- | --- |
| `adfe_v2_clean_local_grok` | Main local baseline. |
| `adfe_role_policy_remediation_grok` | Matched role-policy remediation. |
| `adfe_v2_frontier_grok_exploratory` | Exploratory frontier arm, not pooled. |
| `adfe_stress_baseline_grok` and `adfe_stress_role_policy_grok` | Stress-prompt arms. |
| `no_viewpoint_parity`, `no_refusal_criteria`, `no_source_uncertainty`, `no_role_specific_rules` | Targeted policy ablations. |

## Regenerate Tables

```bash
uv run python -m adfe_runner build-paper-artifacts
```

`generated/numbers.tex` is input by `paper.tex`. Do not hand-edit generated values.

## Compile

Use the official NeurIPS style file for submission. This draft uses a portable `article`
fallback so it can compile before the workshop style is installed:

```bash
cd paper/neurips_workshop
latexmk -pdf paper.tex
```

## Notes For Reviewers

- The current draft reports LLM-judge-calibrated results. Human labels have not yet been imported.
- The frontier run is included for inspection but is not pooled with the local-model evidence.
- Generated files under `generated/` should be regenerated from run artifacts rather than edited by hand.
