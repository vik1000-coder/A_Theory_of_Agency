# Project status

_Last updated: 2026-06-16. Update this after each major run (alongside the site)._

ADFE asks whether an AI's political-fairness behavior depends on the civic role it is assigned
(the "agency gradient"). The conceptual framework is sound; the empirical harness was hardened
after an audit found the original pipeline could manufacture its own effect.

- **Repo:** https://github.com/vik1000-coder/A_Theory_of_Agency (public)
- **Site:** https://vik1000-coder.github.io/A_Theory_of_Agency/

## Where we are

| Area | Status |
|---|---|
| Methodology fixes (held-out judge, frozen runs, blinded inference, non-circular mixed-effects outcome, pair audit) | ✅ done (28 tests) |
| Judge validation — safety / M (XSTest, n=450) | ✅ κ=0.78, acc 89%, safe-overflag 2.4% |
| Judge validation — factuality / E (TruthfulQA, n=1580) | ✅ κ=0.50, catches 80% of false answers (E scores: moderate trust) |
| Clean study (`clean_local.yml`, n=2100, frozen, held-out judge) | ✅ **complete & citable** (`contaminated=false`) |
| Public site | ✅ live with **citable** results |

## Result (clean run `adfe_clean_local_main`, n=2100)

The **agency gradient is not supported** on small local models: curation accountability is flat
with agency (C: p≈0.50), only deliberative standing is marginal (p≈0.04, fails multiple-comparison
correction), 22/42 role-interval predictions violated. The one robust signal is viewpoint-asymmetric
**refusal** in specific models (llama3.2:3b answers one side of a lawful pair, refuses the mirror) —
a safety-tuning artifact, not an agency effect. A clean falsification on this population.

## What is running now

A launchd agent (`com.adfe.pipeline`) runs the pipeline unattended and resumably:
**factuality validation → clean study → `build-site`**. `caffeinate` prevents sleep; it
relaunches if killed and resumes from per-item checkpoints. Population: small local models
only (frontier audited later via the `anthropic:` backend).

Check / control:
```bash
launchctl print gui/$(id -u)/com.adfe.pipeline | grep -iE 'state|pid'
tail -f runs/clean_study.log                                   # study progress
wc -l runs/judge_validation_factuality_qwen3_8b/results.jsonl  # validation checkpoint (/1580)
launchctl bootout gui/$(id -u)/com.adfe.pipeline               # stop
```

## What happens when it finishes

1. The agent writes `runs/PIPELINE_DONE` and regenerates `docs/data/summary.js` from the
   completed study (`runs/adfe_clean_local_main/`) + both judge-validation reports.
2. Publish (one command; pushing is left to a human):
   ```bash
   git add docs && git commit -m "site: clean study results" && git push
   ```
3. The site then swaps from **Preliminary** to **citable**: the page's findings come from a
   frozen, held-out-judge, blinded run (`contaminated=false`), showing the real
   mixed-effects **agency gradient** plus both judge gates (safety κ + factuality κ).

## What it will (and won't) tell us

- **Will:** whether the agency gradient survives a clean test on small local models; the
  clean refusal-asymmetry signal; how far to trust the judge per dimension.
- **Won't (yet):** anything about frontier models (deferred), or the **V / viewpoint-symmetry**
  judge gate — the remaining validation dataset to wrangle (candidate: Anthropic's
  political-neutrality-eval).
