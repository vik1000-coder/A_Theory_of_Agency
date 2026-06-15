# ADFE Runner

Local research harness for Agency-Dependent Fairness Evaluation (ADFE).

The harness runs agency-treatment political AI evaluations against local Ollama models, scores outputs on the six Agency-Dependent Fairness dimensions, analyzes role-fit and paired viewpoint asymmetries, and exports/imports human rating packets for publication-grade calibration.

## Setup

```bash
cd ~/Developer/A_Theory_of_Agency
uv sync --extra dev
uv run python -m adfe_runner doctor
```

## Public site

A GitHub Pages site under [`docs/`](docs/) presents the problem, data, judge validation, and
analysis. It renders from `docs/data/summary.js`, which is generated from run artifacts:

```bash
uv run python -m adfe_runner build-site --run-id <run_id>   # or no --run-id for the latest run
```

Commit and push `docs/` and Pages redeploys. Full workflow: [`docs/UPDATING.md`](docs/UPDATING.md).

## Methodology guardrails

The harness was hardened after an audit found the original pipeline could manufacture its
own effect. The current invariants:

- **Held-out judge.** The judge (`qwen3:8b`) must not be in the audited set. `configs/clean_local.yml`
  audits the non-qwen local models so the judge never scores itself or a same-family model.
- **Frozen by default.** `iterate` no longer tunes prompts. The auto-calibration loop is a
  dev tool, opt-in via `--calibrate`, and any run that uses it is flagged `contaminated` in
  `run_meta.json`. Never cite numbers from a contaminated run.
- **Blinded role inference.** Role inference is a separate judge pass with the assigned role
  hidden (`blind_role_inference`), so it is not circular.
- **Non-circular headline.** The primary outcome is the mixed-effects agency gradient
  (`score ~ agency_level + (1|model)`) plus raw dimension means; role-fit (distance from
  hand-written bands) is reported as secondary only.
- **Validated judge.** Run `validate-judge` (below) before trusting judge scores.
- **Analogous pairs.** `doctor` audits that viewpoint pairs match on everything but viewpoint.

## Validate the judge (Tier-0 gate)

Validate the judge against human safe/unsafe labels (XSTest) — no hand-rating needed. First
fetch the dataset (CC-BY-4.0, not redistributed in this repo):

```bash
mkdir -p data/validation
curl -sSL -o data/validation/xstest_prompts.csv \
  https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv
```

Then run the gate:

```bash
uv run python -m adfe_runner validate-judge --judge qwen3:8b            # full 450
uv run python -m adfe_runner validate-judge --judge qwen3:8b --per-type 3   # fast sample
```

Look at `runs/judge_validation_*/validation.md`: kappa below ~0.4 means the judge cannot be
trusted; fix it before running a study.

Current result for `qwen3:8b` (n=450): **kappa 0.78, accuracy 89%, safe-overflag 2.4%** —
reliable on lawful/benign and physical-harm requests (96–100%), with a known blind spot on
discrimination/hateful-opinion solicitation (`contrast_discr`, 8%). Scope M-dimension claims
accordingly.

## Smoke Run

```bash
uv run python -m adfe_runner iterate \
  --config configs/publication_pilot.yml \
  --cycles 2 \
  --batch-size 40 \
  --models qwen3:8b,llama3.2:3b
```

## Publication Pilot

```bash
uv run python -m adfe_runner iterate \
  --config configs/publication_pilot.yml \
  --cycles 5 \
  --batch-size 120 \
  --models qwen3:8b,llama3.2:3b \
  --export-rating-packet
```

## Human Rating Workflow

```bash
uv run python -m adfe_runner export-ratings --run-id <run_id> --strategy targeted-agency --max-items 120
uv run python -m adfe_runner import-ratings --run-id <run_id> --ratings path/to/ratings.csv
uv run python -m adfe_runner analyze --run-id <run_id> --with-human-calibration
```

## Agency Experiments

Explicit agency treatment:

```bash
uv run python -m adfe_runner iterate \
  --config configs/agency_effect_explicit.yml \
  --cycles 3 \
  --batch-size all \
  --models qwen3:8b,llama3.2:3b \
  --export-rating-packet
```

Explicit vs implicit vs neutral agency:

```bash
uv run python -m adfe_runner iterate \
  --config configs/agency_mode_contrast.yml \
  --cycles 2 \
  --batch-size all \
  --models qwen3:8b,llama3.2:3b
```

Focused refusal-asymmetry replication:

```bash
uv run python -m adfe_runner iterate \
  --config configs/refusal_asymmetry_replication.yml \
  --cycles 5 \
  --batch-size all \
  --models qwen3:8b,llama3.2:3b
```

Rescore existing generations after rubric changes:

```bash
uv run python -m adfe_runner rescore --run-id <run_id>
```

## Clean Final Run (frozen, held-out judge, full prompt bank)

Runs are frozen by default — no flag needed. `clean_local.yml` keeps the qwen judge out of
the audited set and uses all 30 prompts:

```bash
uv run python -m adfe_runner iterate \
  --config configs/clean_local.yml \
  --cycles 1 \
  --batch-size all
```

Population scope: small local models only. Audit frontier models later by prefixing a model
spec with `anthropic:` (e.g. `--models anthropic:claude-opus-4-8`) once `ANTHROPIC_API_KEY`
is set; the Anthropic backend is experimental and should be verified before a real audit.

Artifacts are written under `runs/<run_id>/`.
