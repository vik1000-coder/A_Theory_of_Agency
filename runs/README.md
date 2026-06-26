# Canonical Run Artifacts

This directory tracks only the canonical workshop runs needed to inspect or reproduce the current
analysis. Older scratch and legacy runs remain ignored by Git.

## Included Runs

| Run ID | Purpose | Expected scored rows |
| --- | --- | ---: |
| `adfe_v2_clean_local_grok` | Main local baseline | 2,100 |
| `adfe_role_policy_remediation_grok` | Matched role-policy remediation | 2,100 |
| `adfe_v2_frontier_grok_exploratory` | Exploratory frontier arm, not pooled | 630 |
| `adfe_stress_baseline_grok` | Stress-prompt baseline | 840 |
| `adfe_stress_role_policy_grok` | Stress-prompt role-policy arm | 840 |
| `no_viewpoint_parity` | Targeted policy ablation | 300 |
| `no_refusal_criteria` | Targeted policy ablation | 300 |
| `no_source_uncertainty` | Targeted policy ablation | 300 |
| `no_role_specific_rules` | Targeted policy ablation | 300 |

## Common File Layout

- `frozen_config.yml`: config frozen at run time.
- `run_meta.json`: run metadata and model set.
- `generations.jsonl`: generated model outputs, one JSON object per evaluated item.
- `v2/xai_grok-4.3/scores.jsonl`: primary judge scores.
- `v2/analysis.json`: aggregate analysis for the run.
- `v2/observations.md`: generated human-readable observations.
- `v2/qwen3_8b_stratified_300/`: alternate-judge sensitivity sample when available.

The main baseline also includes `v2/rating_packet.csv`, the 120-item packet prepared for two-rater
human calibration. Completed human labels have not yet been imported.

See [`../ANALYSIS_HANDOFF.md`](../ANALYSIS_HANDOFF.md) for methods, metric definitions, and the
full data inventory.
