# Anonymous Supplement Notes

This supplement describes the artifact package that should accompany a blind workshop submission.

## Contents

- Source code for generation, v2 scoring, v2 analysis, paper-artifact export, and human-rating import.
- Static prompt set, source packets, and role cards.
- Frozen configs for the baseline, remediation, and exploratory frontier runs.
- Generated CSV/JSON paper tables under `paper/neurips_workshop/generated/`.
- Human-review packet template and importer.

## Reproducibility Path

1. Install dependencies with `uv sync --extra dev`.
2. Run `uv run pytest`.
3. Audit the baseline with `audit-v2 --config configs/v2_clean_local_grok.yml --run-id adfe_v2_clean_local_grok --expect-full`.
4. Run the remediation arm with `configs/role_policy_remediation_grok.yml`.
5. Rebuild paper artifacts with `build-paper-artifacts`.
6. Import human labels with `import-ratings-v2` after two raters complete the packet.

## Blind Submission Notes

Do not include public repository URLs, author names, local filesystem paths, acknowledgements, or
non-anonymous hosted links in the submitted PDF or supplement. If code is shared for review, package
it as an anonymous archive.
