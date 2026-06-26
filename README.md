# Role-Conditioned Fairness Evaluation for Civic AI

This repository evaluates a concrete question:

**When the same civic AI system is assigned a different role, does its refusal behavior,
viewpoint symmetry, and role-fit behavior change?**

The project treats role prompts as policy-bearing deployment configuration. A civic assistant,
advocate, mediator, government-information service, campaign aide, researcher, and news provider
can all face the same civic prompt and the same source packet, but each role carries different
obligations. The evaluation therefore varies assigned role as an experimental variable.

**Public report:** https://vik1000-coder.github.io/A_Theory_of_Agency/

The workshop paper package lives in [`paper/neurips_workshop/`](paper/neurips_workshop/). Legacy
pre-workshop reports and configs live in [`archives/workshop_legacy_20260622/`](archives/workshop_legacy_20260622/).

## What This Project Shows

The completed local experiments show three things.

1. **Role prompts matter, but not as a uniform improvement.** In the baseline run, the primary
   judge marks a 14.3% refusal rate, a 13.1% over-refusal rate, and 72 one-sided refusals across
   420 mirrored viewpoint comparisons. The role effects are uneven: the advocate role refuses most
   often and fits its role worst, while the news-provider role answers most often but is only
   middling on role fit.

2. **The clearest failure is asymmetric civic service.** The strongest problem is not a single
   left-right political score. It is that one side of a matched civic prompt pair can be refused
   while the counterpart is answered under the same model, role, and role-presentation mode.

3. **Explicit role policy helps, but does not solve everything.** A matched role-policy remediation
   run reduces one-sided refusals from 72 to 37 and improves the targeted failure sample. On the
   full grid, the aggregate refusal improvement is modest and non-refusal quality drops slightly,
   so role policy should be versioned and regression-tested rather than treated as a one-time fix.

The fairness interpretation is contextual. Role differences are not automatically bias: a
government-information service should behave differently from a campaign aide, and a mediator should
not sound like an advocate. The issue is unjustified context sensitivity: comparable civic requests
receive different refusal treatment, or the assigned role silently changes evidentiary burden across
political positions.

Human review has not yet been imported. Current findings are LLM-judge-calibrated, with a two-rater
packet ready for calibration.

## Experimental Design Axes

The main grid varies four things and keeps the source packets and judging procedure fixed.

| Axis | Values | Why it matters |
| --- | --- | --- |
| Local generator models | `llama3.2:3b`, `llama3.2:1b`, `phi3:mini`, `gemma3:1b`, `deepseek-r1:1.5b` | Tests whether the pattern is model-family or size-specific among compact local models. |
| Civic roles | assistant, advocate, campaign aide, researcher, government information service, mediator, news provider | Tests whether deployment role changes access, usefulness, and role-fit behavior. |
| Prompts | 30 civic prompts across six U.S. policy topics | Includes mirrored prompt pairs for viewpoint-symmetry checks. |
| Role-presentation modes | `explicit`, `implicit` | Tests direct role cards versus softer agency context. There are two modes; the “five” count is the number of local models. |

The seven roles are not supposed to collapse to one generic answer style. A calibrated system should
let role affect obligations: a campaign aide may write persuasive material within truthfulness
limits; a government-information service should avoid persuasion; a mediator should map
disagreement; a news provider should justify source selection and salience. The fairness problem is
not role variation itself. The problem is role variation that changes access or evidentiary burden
without a role-based reason.

Baseline role effects show that the models are role-sensitive but not reliably role-calibrated:

| Role | Baseline refusal | Baseline role fit | Interpretation |
| --- | ---: | ---: | --- |
| User advocate / steelman | 22.3% | 0.490 | Highest refusal and lowest fit; counterintuitive because this role should usually help with the requested side. |
| Campaign aide | 16.7% | 0.558 | Persuasive context appears constrained rather than consistently enabled. |
| Personal assistant | 15.3% | 0.618 | Mid-level refusal with relatively high role fit. |
| Government information service | 14.0% | 0.629 | Highest role fit; the institutional role is partly recognized. |
| Deliberative mediator | 13.7% | 0.538 | Equal-standing and deliberation behavior remain hard. |
| Research librarian | 10.3% | 0.596 | Lower refusal with decent fit, but not a clean source-discipline win. |
| Civic news provider | 7.7% | 0.582 | Lowest refusal, but answering often is not the same as meeting news-role obligations. |

## How the Numbers Are Computed

The baseline grid is:

```text
5 local models x 7 civic roles x 30 prompts x 2 role-presentation modes = 2,100 judged rows
```

The matched-pair grid is:

```text
6 matched prompt pairs x 5 models x 7 roles x 2 role-presentation modes = 420 pair comparisons
```

Source artifacts:

- Generations: `runs/adfe_v2_clean_local_grok/generations.jsonl`
- Primary judge scores: `runs/adfe_v2_clean_local_grok/v2/xai_grok-4.3/scores.jsonl`
- Aggregate analysis: `runs/adfe_v2_clean_local_grok/v2/analysis.json`
- Paper tables and macros: `paper/neurips_workshop/generated/`
- Public page data: `docs/data/summary.js`

The empirical definitions are:

- **Context effect:** a change in refusal, quality, or role fit when role, topic, prompt framing, or
  role-presentation mode changes.
- **Behavioral bias / fairness failure:** unjustified asymmetric treatment of comparable civic
  requests. The cleanest metric is one-sided refusal: same model, same role, same mode, matched
  prompt pair, one side refused and the counterpart answered.
- **Opinionatedness:** not measured as a left/right ideology score here. This study measures
  opinionated behavior indirectly through refusal asymmetry, viewpoint-symmetry scores, and
  role-inappropriate persuasion.
- **Usefulness:** access plus quality. Access means the model answers rather than refuses. Quality
  is scored only among non-refusals on six 0-1 dimensions.
- **Role fit:** mean of six 0-1 role-profile scores checking whether the output behaves like the
  assigned civic role.

For remediation, deltas are matched by `(model, role, agency_mode, prompt_id)`. The paper reports
mean paired deltas and approximate 95% confidence intervals using mean delta ± 1.96 standard errors.

## Canonical Experimental Package

The active story is:

1. **Baseline:** `adfe_v2_clean_local_grok`
   - 2,100 judged rows.
   - Five local generator models.
   - Seven civic roles.
   - Thirty prompts across six U.S. policy topics.
   - Explicit and implicit role-prompt conditions.
   - `xai:grok-4.3` as primary judge.

2. **Matched remediation:** `adfe_role_policy_remediation_grok`
   - Same models, prompts, roles, source packets, role-presentation modes, and judge.
   - Adds [`data/remediation_role_policy_addendum.md`](data/remediation_role_policy_addendum.md).
   - Compared by matched `(model, role, agency_mode, prompt_id)` keys.

3. **Robustness and stress checks:**
   - Alternate-judge sensitivity on stratified 300-row samples.
   - Four targeted policy-component ablations on the same 300-key failure sample.
   - A 24-prompt mirrored stress set in explicit role mode.
   - A failure-regression gate built from the worst baseline failures.
   - A 120-item two-rater human packet, pending completed labels.

## Setup

```bash
uv sync --extra dev
uv run pytest
uv run python -m adfe_runner doctor --config configs/v2_clean_local_grok.yml
```

`doctor` checks config integrity, prompt/source references, mirrored prompt structure, and model
availability.

## Reproduce the Main Runs

Baseline:

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

Role-policy remediation:

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

Full workshop package:

```bash
scripts/run_workshop_experiments.sh
```

That script runs the remediation, judge sensitivity, ablations, stress arms, regression gate, paper
artifact build, and site build using the canonical run IDs.

## Human Review

Export the 120-item packet:

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

## Paper and Public Site

Regenerate source-backed paper tables and TeX numbers:

```bash
uv run python -m adfe_runner build-paper-artifacts
```

Compile the paper:

```bash
python3 /Users/vik/.codex/plugins/cache/openai-bundled/latex/0.2.3/scripts/compile_latex.py \
  /Users/vik/Developer/A_Theory_of_Agency/paper/neurips_workshop/paper.tex
```

Regenerate GitHub Pages data:

```bash
uv run python -m adfe_runner build-site \
  --config configs/v2_clean_local_grok.yml \
  --run-id adfe_v2_clean_local_grok
```

The page source is [`docs/index.html`](docs/index.html), and its run-backed data file is
[`docs/data/summary.js`](docs/data/summary.js). See [`docs/UPDATING.md`](docs/UPDATING.md).

## Important Files

- [`configs/v2_clean_local_grok.yml`](configs/v2_clean_local_grok.yml): baseline evaluation.
- [`configs/role_policy_remediation_grok.yml`](configs/role_policy_remediation_grok.yml): matched remediation.
- [`configs/ablations/`](configs/ablations/): targeted policy ablations.
- [`configs/stress_baseline_grok.yml`](configs/stress_baseline_grok.yml): stress baseline.
- [`configs/stress_role_policy_grok.yml`](configs/stress_role_policy_grok.yml): stress remediation.
- [`data/prompts.jsonl`](data/prompts.jsonl): civic prompts and mirrored pairs.
- [`data/stress_prompts.jsonl`](data/stress_prompts.jsonl): harder mirrored stress prompts.
- [`data/role_cards.yml`](data/role_cards.yml): role definitions and expected profiles.
- [`data/source_packets/`](data/source_packets/): static source packets.
- [`adfe_runner/v2_analysis.py`](adfe_runner/v2_analysis.py): refusal, quality, role-fit, and judge-sensitivity analysis.
- [`adfe_runner/paper.py`](adfe_runner/paper.py): generated paper tables and matched deltas.
- [`adfe_runner/site.py`](adfe_runner/site.py): generated public-site data.

## Limits

The evidence is about small local models, U.S. civic topics, a static prompt set, and an
LLM-judge workflow. The exploratory frontier arm is useful for stress testing but is not pooled
with the main evidence because same-provider judging makes it less independent. Human review is the
next calibration step.
