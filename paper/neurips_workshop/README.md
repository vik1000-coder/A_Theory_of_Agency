# NeurIPS Workshop Draft Package

Working title: **Role Prompts Are Policy Controls: Role-Conditioned Fairness Evaluation for
Civic AI Systems**.

This directory is organized as an anonymous ResponsibleFM-style workshop package.

## Files

- `paper.tex`: main anonymous workshop draft.
- `references.bib`: citation spine.
- `generated/`: source-backed tables and LaTeX number macros generated from run artifacts.
- `supplement.md`: anonymized artifact-package notes.
- `anonymization_checklist.md`: blind-submission checks to run before export.

## Regenerate Tables

```bash
uv run python -m adfe_runner build-paper-artifacts \
  --baseline-run-id adfe_v2_clean_local_grok \
  --remediation-run-id adfe_role_policy_remediation_grok \
  --frontier-run-id adfe_v2_frontier_grok_exploratory \
  --out-dir paper/neurips_workshop/generated
```

`generated/numbers.tex` is input by `paper.tex`. Do not hand-edit generated values.

## Compile

Use the official NeurIPS style file for submission. This draft uses a portable `article`
fallback so it can compile before the workshop style is installed:

```bash
cd paper/neurips_workshop
latexmk -pdf paper.tex
```

or use the bundled Codex LaTeX compiler.
