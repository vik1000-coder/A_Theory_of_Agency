# Analysis Handoff

This repository is a complete analysis package for the role-conditioned civic AI evaluation.
It contains the code, configs, prompts, source packets, role cards, canonical run artifacts,
generated tables, public report, and workshop-paper draft.

## One-Sentence Project Summary

The project tests whether assigning the same civic AI system a different role changes refusal,
viewpoint symmetry, usefulness, and role fit, and whether a clearer executable role policy reduces
unjustified asymmetries.

## What To Read First

1. [`README.md`](README.md): project overview, headline results, reproduction commands.
2. [`docs/index.html`](docs/index.html): public-facing report used for GitHub Pages.
3. [`paper/neurips_workshop/paper.pdf`](paper/neurips_workshop/paper.pdf): current workshop-paper PDF.
4. [`paper/neurips_workshop/paper.tex`](paper/neurips_workshop/paper.tex): source for the paper.
5. This handoff file: data inventory, method details, and analysis entry points.

## Current Main Takeaway

Roles affect civic behavior, but not as a simple improvement. The baseline shows role sensitivity
without reliable role calibration:

| Role | Baseline refusal | Baseline role fit | Interpretation |
| --- | ---: | ---: | --- |
| User advocate / steelman | 22.3% | 0.490 | Highest refusal and lowest fit; counterintuitive for a role that should usually help with the requested side. |
| Campaign aide | 16.7% | 0.558 | Persuasive context appears constrained rather than consistently enabled. |
| Personal assistant | 15.3% | 0.618 | Mid-level refusal with relatively high role fit. |
| Government information service | 14.0% | 0.629 | Highest role fit; the institutional role is partly recognized. |
| Deliberative mediator | 13.7% | 0.538 | Equal-standing and deliberation behavior remain hard. |
| Research librarian | 10.3% | 0.596 | Lower refusal with decent fit, but not a clean source-discipline win. |
| Civic news provider | 7.7% | 0.582 | Lowest refusal, but answering often is not the same as meeting news-role obligations. |

The remediation prompt policy reduces the clearest fairness failure:

| Result | Baseline | Role-policy remediation |
| --- | ---: | ---: |
| Full-grid refusal rate | 14.3% | 13.4% |
| Full-grid over-refusal rate | 13.1% | 12.4% |
| One-sided mirrored-pair refusals | 72 / 420 | 37 / 420 |
| Targeted-sample refusal rate | 81.0% | 51.0% |
| Regression-gate one-sided refusal | 80.0% | 21.1% |

The interpretation is not "roles made the models fair." The interpretation is that role prompts are
policy controls: they change civic access, usefulness, and role behavior, and therefore need testing
and release gates.

## Experimental Design

The main baseline grid is:

```text
5 local models x 7 civic roles x 30 prompts x 2 role-presentation modes = 2,100 judged rows
```

The matched-pair fairness grid is:

```text
6 mirrored prompt pairs x 5 models x 7 roles x 2 role-presentation modes = 420 pair comparisons
```

Design axes:

| Axis | Values |
| --- | --- |
| Local generator models | `llama3.2:3b`, `llama3.2:1b`, `phi3:mini`, `gemma3:1b`, `deepseek-r1:1.5b` |
| Roles | assistant, advocate, campaign aide, researcher, government information service, mediator, news provider |
| Topics | abortion policy, climate policy, immigration, policing/public safety, taxation, voting/election administration |
| Role-presentation modes | `explicit`, `implicit` |
| Primary judge | `xai:grok-4.3` |
| Sensitivity judge | `qwen3:8b` on stratified samples |

Role-presentation modes:

- `explicit`: the model sees the role card directly.
- `implicit`: the model sees a softer agency context without the role identity being named as directly.

## Empirical Definitions

- **Context effect:** a change in refusal, answer quality, or role fit when role, topic, prompt
  framing, or role-presentation mode changes.
- **Behavioral bias / fairness failure:** unjustified asymmetric treatment of comparable civic
  requests. The cleanest metric is one-sided refusal: same model, same role, same mode, matched
  prompt pair, one side refused and the counterpart answered.
- **Opinionatedness:** not measured here as a left/right ideology score. The project measures
  opinionated behavior indirectly through refusal asymmetry, viewpoint-symmetry scores, and
  role-inappropriate persuasion.
- **Usefulness:** access plus answer quality. Access means answering rather than refusing. Quality
  is scored only among non-refusals.
- **Role fit:** mean of six 0-1 role-profile scores checking whether the answer behaves like the
  assigned civic role.

The six quality and role-profile dimensions are:

| Code | Dimension |
| --- | --- |
| U | User-agency fidelity |
| E | Epistemic integrity |
| V | Viewpoint-symmetric treatment |
| C | Curation accountability |
| D | Deliberative equal standing |
| M | Non-manipulation/refusal integrity |

## Canonical Data Inventory

Inputs:

| Path | Contents |
| --- | --- |
| [`configs/v2_clean_local_grok.yml`](configs/v2_clean_local_grok.yml) | Main baseline config. |
| [`configs/role_policy_remediation_grok.yml`](configs/role_policy_remediation_grok.yml) | Matched remediation config. |
| [`configs/ablations/`](configs/ablations/) | Four targeted policy-ablation configs. |
| [`configs/stress_baseline_grok.yml`](configs/stress_baseline_grok.yml) | Stress baseline config. |
| [`configs/stress_role_policy_grok.yml`](configs/stress_role_policy_grok.yml) | Stress remediation config. |
| [`data/prompts.jsonl`](data/prompts.jsonl) | Main 30-prompt set and mirrored pairs. |
| [`data/stress_prompts.jsonl`](data/stress_prompts.jsonl) | 24-prompt stress set. |
| [`data/role_cards.yml`](data/role_cards.yml) | Seven role definitions and expected score intervals. |
| [`data/source_packets/`](data/source_packets/) | Static source packets for each civic topic. |
| [`data/remediation_role_policy_addendum.md`](data/remediation_role_policy_addendum.md) | Full policy intervention. |
| [`data/policy_ablation_addenda/`](data/policy_ablation_addenda/) | Ablated policy interventions. |
| [`data/experiment_samples/policy_ablation_300_keys.json`](data/experiment_samples/policy_ablation_300_keys.json) | Targeted 300-key ablation sample. |
| [`data/regression_gates/civic_role_gate.json`](data/regression_gates/civic_role_gate.json) | Regression-gate sample. |

Raw canonical run artifacts:

| Run ID | Purpose | Key files |
| --- | --- | --- |
| `adfe_v2_clean_local_grok` | Main 2,100-row baseline | `runs/adfe_v2_clean_local_grok/generations.jsonl`, `runs/adfe_v2_clean_local_grok/v2/xai_grok-4.3/scores.jsonl`, `runs/adfe_v2_clean_local_grok/v2/analysis.json` |
| `adfe_role_policy_remediation_grok` | Matched 2,100-row remediation | `runs/adfe_role_policy_remediation_grok/generations.jsonl`, `runs/adfe_role_policy_remediation_grok/v2/xai_grok-4.3/scores.jsonl`, `runs/adfe_role_policy_remediation_grok/v2/analysis.json` |
| `adfe_v2_frontier_grok_exploratory` | Exploratory frontier arm, not pooled | `runs/adfe_v2_frontier_grok_exploratory/` |
| `adfe_stress_baseline_grok` | 840-row stress baseline | `runs/adfe_stress_baseline_grok/` |
| `adfe_stress_role_policy_grok` | 840-row stress remediation | `runs/adfe_stress_role_policy_grok/` |
| `no_viewpoint_parity` | Targeted ablation | `runs/no_viewpoint_parity/` |
| `no_refusal_criteria` | Targeted ablation | `runs/no_refusal_criteria/` |
| `no_source_uncertainty` | Targeted ablation | `runs/no_source_uncertainty/` |
| `no_role_specific_rules` | Targeted ablation | `runs/no_role_specific_rules/` |

Generated analysis products:

| Path | Contents |
| --- | --- |
| [`paper/neurips_workshop/generated/paper_metrics.json`](paper/neurips_workshop/generated/paper_metrics.json) | JSON summary used by the paper and page. |
| [`paper/neurips_workshop/generated/numbers.tex`](paper/neurips_workshop/generated/numbers.tex) | Source-backed LaTeX macros. |
| [`paper/neurips_workshop/generated/tables/`](paper/neurips_workshop/generated/tables/) | CSV tables for baseline, remediation, ablations, stress, gate, and judge robustness. |
| [`paper/neurips_workshop/generated/baseline_role_effects_table.tex`](paper/neurips_workshop/generated/baseline_role_effects_table.tex) | Generated role-effect table used by the paper. |
| [`docs/data/summary.js`](docs/data/summary.js) | Run-backed data used by the public report. |

## JSONL Schemas At A Glance

Generation rows contain:

- `run_id`, `item_id`, `model`, `role`, `agency_mode`, `prompt_id`, `source_packet_id`
- `generation_prompt`
- `output`
- `created_at`, `cycle`, `error`

Score rows contain:

- identifying keys: `run_id`, `item_id`, `model`, `role`, `agency_mode`, `prompt_id`
- judge fields: `judge_model`, `json_valid`, `rationale`, `issues`
- refusal labels: `refusal`, `refusal_warranted`
- normalized scores: `quality_scores`, `role_profile_scores`
- raw 0-4 scores: `quality_scores_raw`, `role_profile_scores_raw`
- heuristic checks: `checks`

## Reproduction And Integrity Commands

Install and test:

```bash
uv sync --extra dev
uv run pytest
```

Audit canonical runs:

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

Rebuild paper/page artifacts from included runs:

```bash
uv run python -m adfe_runner build-paper-artifacts
uv run python -m adfe_runner build-site \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok
```

Compile the paper:

```bash
cd paper/neurips_workshop
latexmk -pdf paper.tex
```

## What Is Not Done Yet

- The 120-item two-rater human review packet exists at
  `runs/adfe_v2_clean_local_grok/v2/rating_packet.csv`, but completed human labels have not been
  imported.
- Role-profile scores are useful but more judge-sensitive than refusal labels.
- The frontier arm is exploratory and should not be pooled with the local-model evidence.
- The prompt/source set is U.S.-civic and static; current-law claims require external verification.
