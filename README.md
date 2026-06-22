# Role-Conditioned Fairness Evaluation for Civic AI

This repository supports a role-counterfactual evaluation of civic AI systems. The central
question is: **does changing the assigned civic role change refusal behavior, viewpoint
symmetry, and role-fit behavior on the same lawful political tasks?**

The project treats role prompts as policy-bearing deployment configuration, not harmless
wording. A civic assistant, advocate, mediator, government-information bot, or campaign aide can
face the same prompt and source packet, but each role carries different obligations. The harness
therefore evaluates role as an experimental variable.

**Public site:** https://vik1000-coder.github.io/A_Theory_of_Agency/

The active submission package is under [`paper/neurips_workshop/`](paper/neurips_workshop/).
Legacy narrative reports and pre-workshop configs are archived under
[`archives/workshop_legacy_20260622/`](archives/workshop_legacy_20260622/).

## Canonical Story

The workshop version reports one baseline evaluation and one directly matched remediation
evaluation:

1. **Baseline evaluation:** `adfe_v2_clean_local_grok`
   - 2,100 judged rows.
   - Five local generator models.
   - Seven civic roles.
   - Thirty civic prompts across six U.S. policy topics.
   - Explicit and implicit role-prompt conditions.
   - `xai:grok-4.3` as primary judge.

2. **Remediation evaluation:** `adfe_role_policy_remediation_grok`
   - Same models, prompts, roles, source packets, agency modes, and judge.
   - Adds an executable role-policy addendum with allowed help, refusal criteria, source
     requirements, uncertainty language, escalation triggers, and viewpoint parity.
   - Compared to baseline by matched `(model, role, agency_mode, prompt_id)` keys.

3. **Calibration and robustness:**
   - Alternate-judge sensitivity on a stratified 300-row sample.
   - A 120-item two-rater human review packet:
     40 refusal-asymmetry examples, 40 role-profile misses, 20 judge-disagreement examples,
     and 20 low-disagreement controls.

The paper claim is intentionally narrow: role prompts can change civic safety and fairness
outcomes, so deployed civic AI systems should version, test, and monitor role policies.

## Setup

```bash
uv sync --extra dev
uv run pytest
uv run python -m adfe_runner doctor --config configs/v2_clean_local_grok.yml
```

`doctor` checks config integrity, prompt/source cross-references, prompt-pair analogy, and model
availability.

## Baseline Run

```bash
XAI_API_KEY=... uv run python -m adfe_runner iterate-v2 \
  --config configs/v2_clean_local_grok.yml \
  --cycles 1 \
  --batch-size all \
  --run-id adfe_v2_clean_local_grok \
  --workers 4

uv run python -m adfe_runner audit-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok \
  --expect-full
```

The baseline artifacts live in `runs/adfe_v2_clean_local_grok/v2/`.

## Remediation Run

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
```

The only intended experimental difference from baseline is
[`data/remediation_role_policy_addendum.md`](data/remediation_role_policy_addendum.md).

## Judge Sensitivity

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

The public baseline already has this artifact at
`runs/adfe_v2_clean_local_grok/v2/comparison_qwen3_8b_stratified_300.json`.

## Human Review

Export the v2 packet:

```bash
uv run python -m adfe_runner export-ratings-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok \
  --max-items 120
```

Import completed two-rater labels:

```bash
uv run python -m adfe_runner import-ratings-v2 \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok \
  --ratings path/to/completed_ratings.csv
```

The importer writes `runs/<run_id>/v2/human_ratings.jsonl` and
`runs/<run_id>/v2/human_rating_summary.json`.

## Paper Artifacts

Regenerate source-backed paper tables:

```bash
uv run python -m adfe_runner build-paper-artifacts \
  --baseline-run-id adfe_v2_clean_local_grok \
  --remediation-run-id adfe_role_policy_remediation_grok \
  --frontier-run-id adfe_v2_frontier_grok_exploratory \
  --out-dir paper/neurips_workshop/generated
```

The command never hand-edits paper numbers. It exports baseline tables immediately and writes
remediation deltas once the remediation run has complete v2 scores.

Compile the draft:

```bash
python3 /Users/vik/.codex/plugins/cache/openai-bundled/latex/0.2.3/scripts/compile_latex.py \
  /Users/vik/Developer/A_Theory_of_Agency/paper/neurips_workshop/paper.tex
```

## Public Site

The GitHub Pages site in [`docs/`](docs/) is generated from run artifacts:

```bash
uv run python -m adfe_runner build-site \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok
```

See [`docs/UPDATING.md`](docs/UPDATING.md).

## Active Files

- `configs/v2_clean_local_grok.yml`: baseline role-counterfactual evaluation.
- `configs/role_policy_remediation_grok.yml`: matched remediation evaluation.
- `configs/v2_frontier_grok_exploratory.yml`: exploratory frontier arm, not pooled with the
  local baseline.
- `data/prompts.jsonl`: civic prompts and mirrored viewpoint pairs.
- `data/role_cards.yml`: assigned civic role definitions and expected role profiles.
- `data/source_packets/`: dated static source packets.
- `adfe_runner/v2_analysis.py`: refusal, non-refusal quality, role-profile, and judge-sensitivity
  analysis.
- `adfe_runner/paper.py`: generated tables and matched remediation deltas for the workshop paper.

## Limits

The current evidence is about small local models, U.S. civic topics, a static prompt set, and an
LLM-judge workflow. The exploratory frontier arm is useful for stress testing but not independent
evidence because it includes same-provider judging. Human review is a calibration packet, not a
replacement for the full evaluation.
