# Agency-Dependent Fairness: Empirical and Reference Audit

Date: June 13, 2026

## Bottom line

The program is empirically implementable. Its strongest version is not a generic "is the model biased?" benchmark, but a role-counterfactual audit: hold topic/task constant, vary the assigned civic role, and test whether model outputs move along the predicted fairness dimensions.

The manuscript's conceptual core is good and falsifiable:

- Roles are experimental treatments: assistant, advocate, researcher, news provider, mediator, government information service, campaign aide, recommender.
- Outputs are scored on a six-part fairness signature: user fidelity, epistemic integrity, viewpoint symmetry, curation accountability, deliberative standing, and non-manipulation/refusal integrity.
- The main empirical hypothesis is an agency gradient: as role-assigned salience/framing discretion rises, curation accountability and source discipline should rise, while invariant floors for factuality and non-manipulation remain stable.

The hard parts are solvable but need careful design: analogous viewpoint pairing, source-packet construction, human calibration, current-news snapshots, and LLM-judge validation.

## Reference audit

Mechanical checks:

- All 56 cited keys have matching `\bibitem`s.
- There are 82 bibliography items total.
- There are 26 unused bibliography items. They are not harmful, but the final article should either cite them in the legal/theory sections or prune them.
- No duplicate `\bibitem`s found.

Major verified empirical anchors:

- OpenAI's 2025 political-bias post exists and supports the manuscript's description: approximately 500 prompts, 100 topics, five slants, and axes including user invalidation, user escalation, personal political expression, asymmetric coverage, and political refusals. Source: https://openai.com/index/defining-and-evaluating-political-bias-in-llms/
- Anthropic's 2025 political even-handedness work exists and reports a paired-prompt method across 1,350 pairs, 9 task types, and 150 topics; the public repo exists. Sources: https://www.anthropic.com/news/political-even-handedness and https://github.com/anthropics/political-neutrality-eval
- Pew's October 1, 2025 chatbot-news item exists and supports the 2% often + 7% sometimes figure, and the fewer-than-1% preference figure. Source: https://www.pewresearch.org/short-reads/2025/10/01/relatively-few-americans-are-getting-news-from-ai-chatbots-like-chatgpt/
- Reuters Institute's 2025 Digital News Report and Generative AI and News Report pages exist. Sources: https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2025 and https://reutersinstitute.politics.ox.ac.uk/generative-ai-and-news-report-2025-how-people-think-about-ais-role-journalism-and-society
- EBU/BBC News Integrity in AI Assistants exists and supports the claim that almost half of AI answers had at least one significant issue, one third had serious sourcing problems, and one fifth had major accuracy issues. Source: https://www.ebu.ch/research/open/report/news-integrity-in-ai-assistants
- PARETO / balanced approval exists as arXiv:2605.28911 and supports 7,434 participants and 208,152 evaluations. Source: https://arxiv.org/abs/2605.28911
- Suzgun et al. news intermediaries exists as arXiv:2605.22785 and supports the 14-day, 2,100-question, six-chatbot, BBC-derived design and the reported free-response/retrieval/regional inequity findings. Source: https://arxiv.org/abs/2605.22785
- Polar exists as arXiv:2606.12922 and supports 4,026 multiple-choice instances, two axes, eight issue categories, U.S./South Korea comparison, and 38 LLMs. Source: https://arxiv.org/abs/2606.12922
- PoliticsBench exists as arXiv:2603.23841, revised June 2, 2026, and accepted to the ICML 2026 Trustworthy AI for Good Workshop. Source: https://arxiv.org/abs/2603.23841

Corrections recommended before publication:

1. `abrams-press-clause-report`: the current URL in the TeX returns 404. The likely correct Yale page is https://law.yale.edu/isp/initiatives/floyd-abrams-institute-freedom-expression/press-clause and the report title appears to be "The Press Clause: The Forgotten First Amendment," not "A Press Clause for the 21st Century."
2. `gabison-xian-principal-agent`: the SSRN page lists Garry Gabison and R. Patrick Xian, not "Ruoli Xian"; the title is "Inherent and emergent liability issues in LLM-based agentic systems: a principal-agent perspective." Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5210666
3. `qiu-representative-social-choice`: the author appears to be Tianyi Qiu, not Qian Qiu, and the full title is "Representative Social Choice: From Learning Theory to AI Alignment." Source: https://arxiv.org/abs/2410.23953
4. `su-teo-ai-rights`: the paper appears to be by Anna Su and Sue Anne Teo, not Yi-Ling Su and Marcus Teo. Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6660120
5. `moody`: update from `603 U.S. ___` to `603 U.S. 707 (2024)`.
6. `murthy`: update from `603 U.S. ___` to `603 U.S. 43 (2024)`.
7. `truthfulqa`: prefer ACL Anthology or arXiv over the OpenAI blog URL. Source: https://aclanthology.org/2022.acl-long.229/ or https://arxiv.org/abs/2109.07958

## Empirical design

Minimum viable study:

1. Select 6 high-salience topics: immigration, policing/public safety, climate, abortion/reproductive policy, taxation, voting/election administration.
2. Create 30 base prompts covering explain, compare, steelman, rebut, daily briefing, official-information, campaign-message, and mediation tasks.
3. Convert each base prompt into role-card conditions: assistant, advocate, researcher, news, mediator, government, campaign.
4. Build viewpoint pairs only where legality, factual status, audience, risk, and specificity are structurally analogous.
5. Generate outputs from local Ollama models plus any external models being audited.
6. Score each output on U/E/V/C/D/M with a 0-4 rubric, then normalize to 0-1.
7. Add automated checks: refusal detection, citation validity, source coverage, length/depth parity, stance markers, and false-premise handling.
8. Validate LLM graders against human/expert ratings before using them at scale.

Data needed:

- Imported prompt/data layer: Anthropic political-neutrality eval, PARETO, PoliticsBench, Polar, OpinionsQA, GlobalOpinionQA, FEVER/FEVEROUS, TruthfulQA, LIAR.
- Constructed role layer: role cards, role-counterfactual prompts, viewpoint-counterfactual pairs, refusal-parity pairs.
- Source-packet layer: official government pages, court filings, statutory/regulatory text, trusted datasets, mainstream news, local news, advocacy materials from multiple sides, and dated correction packets.
- Human-rating layer: politically diverse lay raters for perceived respect/role fit; expert raters for election/legal/news-source questions; reliability metadata.
- Longitudinal layer: daily source snapshots over 7-14 days, correction events, story-importance weights, source-type labels, frame labels, and user-profile conditions.

Primary statistics:

- Role manifestation score by role/model/topic.
- Underperformance and role-intrusion indices by dimension.
- Refusal parity gap across lawful analogous viewpoints.
- Viewpoint quality gap across depth, respect, evidence, and helpfulness.
- Role confusion matrix from blinded role inference.
- Mixed-effects regressions with random effects for model, prompt, topic, rater, and possibly source packet.
- Longitudinal curation metrics: omission risk, source concentration, update lag, frame entropy, profile-agenda gap, personalization capture.

## Ollama feasibility

Local environment:

- Ollama version: 0.30.8.
- Hardware: Apple M4, 16 GB memory.
- Available disk on the main data volume: about 20 GB.
- Installed models: `qwen3:8b`, `llama3.2:3b`, `llama3.2:1b`, `phi3:mini`, `qwen3:1.7b`, `deepseek-r1:1.5b`, `gemma3:1b`, `llava:7b`, `llama-guard3:1b`.

Local smoke test:

- `qwen3:8b` can separate assistant/news role behavior when called through the Ollama API with `"think": false`; the output was clean but generic. It is the best installed local text model for pilot generation and rubric grading.
- `llama3.2:3b` followed the assistant role but made an unsupported current-news-style claim in the news role. It should be treated as a weak baseline, not a reliable fact-sensitive news model.
- The CLI `ollama run qwen3:8b` leaked "Thinking..." content; use the HTTP API with `stream:false` and `think:false` for clean benchmark runs.

How to use local models:

- Use `qwen3:8b` for first-pass generation, role-legibility classification, rubric scoring, prompt expansion, and adjudication drafts.
- Use `llama-guard3:1b` for a separate safety/refusal classifier where applicable.
- Use `llama3.2:3b` and `phi3:mini` as low-cost baselines to show that the benchmark distinguishes weaker role behavior.
- Do not rely on local models for current political facts without source packets. The benchmark should feed dated source packets into the prompt and require source-grounded answers.

Model additions:

- With only about 20 GB free disk and 16 GB memory, do not pull 70B-class models locally.
- The most plausible near-term addition is `mistral-nemo`, a 12B model with a large context window, if disk space is freed. Ollama library: https://ollama.com/library/mistral-nemo
- `qwen3:14b` is also a plausible upgrade, but it may be tight on a 16 GB machine once other apps are running. Ollama library: https://ollama.com/library/qwen3%3A14b
- Larger `qwen3:30b` or 70B models should be remote/cloud or run on a larger machine.

## Recommendation

Build a small ADFE harness before expanding the paper:

- `data/role_cards.yml`
- `data/prompts.jsonl`
- `data/source_packets/`
- `scripts/generate_ollama.py`
- `scripts/score_outputs.py`
- `scripts/analyze_role_fit.R` or Python equivalent
- `outputs/generations.jsonl`
- `outputs/scores.parquet`

Start with a pilot of about 30 base prompts x 5 roles x 2 local models x 2 samples = 600 outputs. Human-score a stratified 100-output subset, calibrate the local judge, then scale.

The manuscript's strongest empirical contribution will be showing role-conditioned movement, not merely measuring ideology. If the outputs collapse across role cards, the theory has a clean falsification path. If they shift in the expected directions while preserving factual and non-manipulation floors, the paper has a strong empirical core.

## Implementation status (June 15, 2026)

The `adfe_runner` harness is built and was hardened after a methodology audit found the original pilot could manufacture its own effect. Current state:

- Judge validated against human labels (XSTest, n=450): qwen3:8b reaches kappa 0.78 / 89% accuracy, reliable on lawful and physical-harm requests, with a quantified blind spot on hateful-opinion solicitation. The judge is held out of the audited set.
- Confounds removed: runs are frozen by default (the prompt-tuning loop is opt-in and flags contamination), role inference is a separate blinded judge pass, and refusal labels are not silently fused.
- Outcome de-circularized: the primary test is now a mixed-effects agency gradient (`score ~ agency_level + (1|model)`); the hand-written role intervals are reported as falsifiable predictions; role-fit is secondary.
- Frontier models can be audited later via an `anthropic:`-prefixed backend; the current population is small local models only.

On the existing (pre-fix, contaminated) data the proper mixed-effects test shows the agency gradient is largely absent (only deliberative standing is significant). A clean run via `configs/clean_local.yml` is the next step; see `README.md`.
