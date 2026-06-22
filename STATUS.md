# Project Status

_Last updated: 2026-06-22._

This branch reframes the project as a role-counterfactual civic AI evaluation for a
ResponsibleFM-style NeurIPS workshop submission.

## Current State

| Area | Status |
| --- | --- |
| Submission branch | Active: `codex/neurips-workshop-submission` |
| Baseline local evaluation | Complete: `runs/adfe_v2_clean_local_grok` with 2,100 v2 scores |
| Baseline artifact audit | Expected to pass via `audit-v2 --expect-full` |
| Judge sensitivity | Complete for 300-row stratified Qwen sample |
| Exploratory frontier arm | Complete: `runs/adfe_v2_frontier_grok_exploratory`; not pooled with local evidence |
| Remediation policy config | Added: `configs/role_policy_remediation_grok.yml` |
| Remediation run | Pending execution |
| Human review packet | Exporter updated for 120-item two-rater calibration |
| Human rating import/summary | Added: `import-ratings-v2` |
| Workshop paper package | Added under `paper/neurips_workshop/` |

## Current Takeaway

The baseline evaluation shows that role assignment is not just style text. Across the same
lawful civic prompts, role prompts can change whether local models refuse, whether mirrored
viewpoints receive symmetric treatment, and whether outputs match the duties implied by the
assigned role. The most actionable failure mode is not a single political leaning score; it is
deployment instability caused by role-specific prompt policy.

## Next Required Run

```bash
XAI_API_KEY=... uv run python -m adfe_runner iterate-v2 \
  --config configs/role_policy_remediation_grok.yml \
  --cycles 1 \
  --batch-size all \
  --run-id adfe_role_policy_remediation_grok \
  --workers 4

uv run python -m adfe_runner audit-v2 \
  --config configs/role_policy_remediation_grok.yml \
  --run-id adfe_role_policy_remediation_grok \
  --expect-full

uv run python -m adfe_runner build-paper-artifacts
```

## Paper Acceptance Bar

The workshop draft is not submission-ready until:

- Remediation results are populated from a complete matched run.
- Two raters complete the 120-item packet or the paper clearly labels the human packet as pending.
- Every reported number is regenerated from `paper/neurips_workshop/generated/`.
- The blind PDF and supplement contain no author names, GitHub handles, local paths, or
  acknowledgements.
