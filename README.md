# ADFE Runner

Local research harness for Agency-Dependent Fairness Evaluation (ADFE).

The harness runs role-counterfactual political AI evaluations against local Ollama models,
scores outputs on the six Agency-Dependent Fairness dimensions with an LLM judge that is
first **validated against human-labeled datasets**, and tests the agency-gradient hypothesis
with a mixed-effects model alongside paired-viewpoint asymmetry metrics.

**Public site:** https://vik1000-coder.github.io/A_Theory_of_Agency/ — problem, data, judge
validation, and findings (regenerated from run artifacts; see [`docs/UPDATING.md`](docs/UPDATING.md)).

**Full report:** [`report/adfe_report.tex`](report/adfe_report.tex) → [PDF](report/adfe_report.pdf)
(also at `/adfe_report.pdf` on the site). Rebuild with `cd report && latexmk -pdf adfe_report.tex`.

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

Validate the judge against human-labeled public datasets before trusting its scores — no
hand-rating needed. Two tasks: **safety** (XSTest → the M dimension) and **factuality**
(TruthfulQA → the E dimension). Fetch the datasets (CC-BY etc., not redistributed here):

```bash
mkdir -p data/validation
curl -sSL -o data/validation/xstest_prompts.csv \
  https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv
curl -sSL -o data/validation/truthfulqa.csv \
  https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv
```

Run the gates (resumable — re-running continues from a per-item checkpoint):

```bash
uv run python -m adfe_runner validate-judge --task safety --judge qwen3:8b       # XSTest (M)
uv run python -m adfe_runner validate-judge --task factuality --judge qwen3:8b   # TruthfulQA (E)
uv run python -m adfe_runner validate-judge --task safety --per-type 3           # fast sample
```

Read `runs/judge_validation_<task>_<judge>/validation.md`: kappa below ~0.4 means the judge
cannot be trusted on that dimension; fix it (or scope claims away from it) before a study.

Results for `qwen3:8b`: **safety** (n=450) kappa 0.78 / acc 89% / safe-overflag 2.4% —
reliable on lawful and physical-harm requests (96–100%), with a known blind spot on
discrimination/hateful-opinion solicitation (`contrast_discr`, 8%). **Factuality** is weaker
(preliminary sample kappa ≈ 0.3): the judge endorses confident falsehoods, so treat E-dimension
scores with caution. The current numbers always appear on the site once a validation completes.

## Running a study

`configs/clean_local.yml` is the canonical config: judge (`qwen3:8b`) held out of the audited
set, all 30 prompts, both agency modes, frozen.

```bash
# smoke (a few items, end to end)
uv run python -m adfe_runner iterate --config configs/clean_local.yml --cycles 1 --batch-size 6

# full clean study (frozen, held-out judge, all prompts) — the citable run
uv run python -m adfe_runner iterate --config configs/clean_local.yml \
  --cycles 1 --batch-size all --run-id adfe_clean_local_main
```

Runs are frozen by default (no prompt tuning). **Resumable:** re-run the same `--run-id` to
continue after an interruption — already-generated/scored items are skipped. Artifacts land
under `runs/<run_id>/` (`analysis.json`, `observations.md`, `scores.jsonl`).

Population scope: small local models only. Audit frontier models later by prefixing a model
spec with `anthropic:` (e.g. `--models anthropic:claude-opus-4-8`) once `ANTHROPIC_API_KEY`
is set; the Anthropic backend is experimental and should be verified before a real audit.

Explicit vs. implicit vs. neutral agency contrast: `configs/agency_mode_contrast.yml`. Rescore
existing generations after rubric changes: `rescore --run-id <run_id>`.

## Unattended multi-day run

For overnight / multi-day runs that survive sleep, terminal close, and kills, use the launchd
service in [`scripts/`](scripts/): `caffeinate` prevents sleep, the agent relaunches if killed,
and every step resumes from checkpoint. Install/monitor/stop instructions:
[`scripts/README.md`](scripts/README.md).

## Optional: human-rating calibration

Judge validation against public datasets (above) is the primary calibration path. If you do
have raters, you can additionally collect human ratings:

```bash
uv run python -m adfe_runner export-ratings --run-id <run_id> --strategy targeted-agency --max-items 120
uv run python -m adfe_runner import-ratings --run-id <run_id> --ratings path/to/ratings.csv
uv run python -m adfe_runner analyze --run-id <run_id> --with-human-calibration
```

> Legacy configs (`publication_pilot.yml`, `agency_effect_explicit.yml`,
> `refusal_asymmetry_replication.yml`, `public_essay_replication.yml`) predate the methodology
> fix and audit `qwen3:8b` with itself as judge; they are kept only to reproduce the earlier
> (contaminated) runs. Use `clean_local.yml` for any citable result.
