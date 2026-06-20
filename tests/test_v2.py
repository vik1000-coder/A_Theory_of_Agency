from pathlib import Path

import pytest

from adfe_runner.io import load_config, load_prompts, load_role_cards, load_source_packets
from adfe_runner.schemas import DIMENSIONS, GenerationRecord, V2ScoreRecord, now_iso
from adfe_runner.scoring import parse_v2_judge_payload, score_generation_v2
from adfe_runner.v2_analysis import analyze_v2_scores, compare_v2_judges, stratified_v2_sample_manifest


ROOT = Path(__file__).resolve().parents[1]


class V2JudgeClient:
    def generate(self, model, prompt, options=None, think=False):
        return """
        {
          "refusal": false,
          "refusal_warranted": null,
          "quality_scores": {"U": 3, "E": 4, "V": 3, "C": 4, "D": 2, "M": 4},
          "role_profile_scores": {"U": 3, "E": 3, "V": 3, "C": 3, "D": 3, "M": 3},
          "inferred_role": "researcher",
          "issues": ["compact"],
          "rationale": "Grounded and role-aware."
        }
        """


class MalformedClient:
    def generate(self, model, prompt, options=None, think=False):
        return "not json"


def load_fixture_context(role_id="researcher"):
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)
    prompt = prompts[0]
    role = roles.by_id[role_id]
    packet = packets[prompt.source_packet_id]
    return prompt, role, roles, packet


def generation(role, prompt, packet, output="A useful sourced answer with enough detail to be judged."):
    return GenerationRecord(
        run_id="run",
        cycle=0,
        item_id=f"{role.id}:{prompt.id}",
        model="m1",
        role=role.id,
        prompt_id=prompt.id,
        source_packet_id=packet.id,
        generation_prompt="prompt",
        output=output,
        created_at=now_iso(),
    )


def make_v2_score(
    item_id,
    prompt_id,
    role,
    scores,
    *,
    refusal=False,
    refusal_warranted=None,
    profile=None,
    judge="xai:grok-4.3",
):
    profile = profile or scores
    return V2ScoreRecord(
        run_id="run",
        cycle=0,
        item_id=item_id,
        model="m1",
        role=role,
        prompt_id=prompt_id,
        source_packet_id="immigration_2026_static",
        judge_model=judge,
        refusal=refusal,
        refusal_warranted=refusal_warranted,
        quality_scores_raw={dim: scores[dim] * 4 for dim in DIMENSIONS},
        quality_scores=scores,
        role_profile_scores_raw={dim: profile[dim] * 4 for dim in DIMENSIONS},
        role_profile_scores=profile,
        inferred_role=role,
        json_valid=True,
        checks={"word_count": 100},
        created_at=now_iso(),
    )


def test_v2_parser_rejects_incomplete_payload():
    with pytest.raises(ValueError, match="missing dimensions"):
        parse_v2_judge_payload(
            {
                "refusal": False,
                "quality_scores": {"U": 4},
                "role_profile_scores": {dim: 4 for dim in DIMENSIONS},
                "issues": [],
                "rationale": "x",
            }
        )


def test_score_generation_v2_fails_closed_on_malformed_json():
    prompt, role, roles, packet = load_fixture_context()
    record = generation(role, prompt, packet)

    with pytest.raises(ValueError, match="malformed JSON"):
        score_generation_v2(
            record,
            prompt=prompt,
            assigned_role=role,
            role_ids=list(roles.by_id),
            packet=packet,
            client=MalformedClient(),
            judge_model="xai:grok-4.3",
            options={},
            score_json_retry=1,
        )


def test_score_generation_v2_separates_refusal_quality_and_profile_scores():
    prompt, role, roles, packet = load_fixture_context()
    record = generation(role, prompt, packet)

    score = score_generation_v2(
        record,
        prompt=prompt,
        assigned_role=role,
        role_ids=list(roles.by_id),
        packet=packet,
        client=V2JudgeClient(),
        judge_model="xai:grok-4.3",
        options={},
        score_json_retry=0,
    )

    assert score.refusal is False
    assert score.refusal_warranted is None
    assert score.quality_scores["E"] == 1.0
    assert score.role_profile_scores["D"] == 0.75
    assert score.checks["judge_json_attempts"] == 1


def test_v2_analysis_excludes_refusals_from_quality_means_and_reports_profile():
    prompts = load_prompts(ROOT / "data/prompts.jsonl")
    roles = load_role_cards(ROOT / "data/role_cards.yml").by_id
    good = {dim: 0.8 for dim in DIMENSIONS}
    low = {dim: 0.1 for dim in DIMENSIONS}
    scores = [
        make_v2_score("a", "immigration_briefing", "assistant", good),
        make_v2_score("b", "immigration_briefing", "assistant", low, refusal=True, refusal_warranted=False),
    ]

    out = analyze_v2_scores(scores, prompts, roles)

    assert out["overall"]["n_scores"] == 2
    assert out["overall"]["n_non_refusal"] == 1
    assert out["quality_non_refusal"]["overall"]["U"] == 0.8
    assert out["refusal"]["by_role"]["assistant"]["over_refusal_rate"] == 0.5
    assert out["role_profile"]["by_role"]["assistant"]["quality_rmse_to_expected_midpoint"] is not None


def test_v2_judge_comparison_reports_refusal_and_profile_disagreement():
    base_scores = {dim: 0.9 for dim in DIMENSIONS}
    alt_scores = {dim: 0.3 for dim in DIMENSIONS}
    primary = [make_v2_score("item", "immigration_briefing", "assistant", base_scores)]
    sensitivity = [
        make_v2_score(
            "item",
            "immigration_briefing",
            "assistant",
            alt_scores,
            refusal=True,
            refusal_warranted=False,
            judge="qwen3:8b",
        )
    ]

    out = compare_v2_judges(primary, sensitivity)

    assert out["available"] is True
    assert out["refusal_mismatch_count"] == 1
    assert out["quality_mean_abs_delta_non_refusal_both"] is None
    assert out["role_profile_mean_abs_delta"] == 0.6
    assert out["top_disagreements"][0]["refusal_mismatch"] is True


def test_stratified_v2_sample_balances_refusal_and_axes():
    prompts = load_prompts(ROOT / "data/prompts.jsonl")
    prompt_ids = [prompt.id for prompt in prompts[:4]]
    prompt_by_id = {prompt.id: prompt for prompt in prompts}
    scores = {dim: 0.8 for dim in DIMENSIONS}
    records = []
    primary = []
    idx = 0
    for model in ("m1", "m2"):
        for role in ("assistant", "researcher"):
            for mode in ("explicit", "implicit"):
                for prompt_id in prompt_ids:
                    prompt = prompt_by_id[prompt_id]
                    record = GenerationRecord(
                        run_id="run",
                        cycle=0,
                        item_id=f"{model}:{role}:{mode}:{prompt_id}",
                        model=model,
                        role=role,
                        agency_mode=mode,
                        prompt_id=prompt_id,
                        source_packet_id=prompt.source_packet_id,
                        generation_prompt="prompt",
                        output="A useful answer with enough text to judge.",
                        created_at=now_iso(),
                    )
                    records.append(record)
                    primary.append(
                        make_v2_score(
                            record.item_id,
                            prompt_id,
                            role,
                            scores,
                            refusal=idx % 3 == 0,
                        ).model_copy(update={"model": model, "agency_mode": mode, "source_packet_id": prompt.source_packet_id})
                    )
                    idx += 1

    manifest = stratified_v2_sample_manifest(records, primary, prompts, sample_size=12, seed=7)

    assert manifest["sample_size"] == 12
    assert manifest["sample_counts"]["primary_refusal"]["true"] == 6
    assert manifest["sample_counts"]["primary_refusal"]["false"] == 6
    assert set(manifest["sample_counts"]["model"]) == {"m1", "m2"}
    assert set(manifest["sample_counts"]["role"]) == {"assistant", "researcher"}
    assert set(manifest["sample_counts"]["agency_mode"]) == {"explicit", "implicit"}


def test_v2_sampled_comparison_reports_poststratified_metrics():
    base_scores = {dim: 0.9 for dim in DIMENSIONS}
    alt_scores = {dim: 0.6 for dim in DIMENSIONS}
    primary = [
        make_v2_score("a", "immigration_briefing", "assistant", base_scores, refusal=False),
        make_v2_score("b", "immigration_briefing", "assistant", base_scores, refusal=True),
    ]
    sensitivity = [
        make_v2_score("a", "immigration_briefing", "assistant", alt_scores, refusal=False, judge="qwen3:8b"),
        make_v2_score("b", "immigration_briefing", "assistant", alt_scores, refusal=False, judge="qwen3:8b"),
    ]
    manifest = {
        "schema": "adfe_v2_stratified_sensitivity_sample.v1",
        "strategy": "stratified_balanced_refusal_marginals",
        "seed": 1,
        "requested_sample_size": 2,
        "population_size": 10,
        "sample_size": 2,
        "balance_axes": ["primary_refusal"],
        "population_counts": {"primary_refusal": {"true": 2, "false": 8}},
        "sample_counts": {"primary_refusal": {"true": 1, "false": 1}},
        "items": [
            {"item_id": "a", "primary_refusal": False, "pair_key": "p"},
            {"item_id": "b", "primary_refusal": True, "pair_key": "p"},
        ],
    }

    out = compare_v2_judges(primary, sensitivity, sample_manifest=manifest)

    assert out["sample"]["available"] is True
    assert out["poststratified"]["available"] is True
    assert out["refusal_mismatch_count"] == 1
    assert out["poststratified"]["refusal_mismatch_rate"] == 0.2
    assert out["bootstrap_intervals"]["role_profile_mean_abs_delta"]["n"] == 2
