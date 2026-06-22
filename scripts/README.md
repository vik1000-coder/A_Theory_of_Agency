# Scripts

The active script is `run_v2_pipeline.sh`, which runs the workshop evaluation path:

1. Baseline role-counterfactual evaluation.
2. Baseline v2 audit.
3. Stratified alternate-judge sensitivity.
4. 120-item human-review packet export.
5. Matched role-policy remediation evaluation.
6. Remediation v2 audit.
7. Exploratory frontier arm.
8. Paper table generation.
9. Public site data regeneration.

Run it with:

```bash
XAI_API_KEY=... scripts/run_v2_pipeline.sh
```

or place the key in `~/.config/adfe/xai.env`:

```bash
export XAI_API_KEY=...
```

Optional environment variables:

- `BASELINE_RUN_ID`, default `adfe_v2_clean_local_grok`
- `REMEDIATION_RUN_ID`, default `adfe_role_policy_remediation_grok`
- `FRONTIER_RUN_ID`, default `adfe_v2_frontier_grok_exploratory`
- `WORKERS`, default `4`

The old launchd scripts for the pre-workshop narrative are archived under
`archives/workshop_legacy_20260622/scripts/`.
