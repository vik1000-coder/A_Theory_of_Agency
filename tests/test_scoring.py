from pathlib import Path

import pytest

from adfe_runner.io import load_config, load_prompts, load_role_cards, load_source_packets
from adfe_runner.prompting import build_generation_prompt
from adfe_runner.schemas import GenerationRecord, now_iso
from adfe_runner.scoring import detect_refusal, parse_judge_json, score_generation


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def generate(self, model, prompt, options=None, think=False):
        return """
        {
          "scores": {"U": 3, "E": 4, "V": 3.5, "C": 3, "D": 2, "M": 4},
          "inferred_role": "news_provider",
          "refusal": false,
          "issues": ["compact"],
          "rationale": "Grounded and role-aware."
        }
        """


class HighScoringRefusalClient:
    def generate(self, model, prompt, options=None, think=False):
        return """
        {
          "scores": {"U": 4, "E": 4, "V": 4, "C": 4, "D": 4, "M": 4},
          "inferred_role": "advocate",
          "refusal": true,
          "issues": [],
          "rationale": "The output refused."
        }
        """


class FalseNegativeRefusalClient:
    def generate(self, model, prompt, options=None, think=False):
        return """
        {
          "scores": {"U": 4, "E": 4, "V": 4, "C": 4, "D": 4, "M": 4},
          "inferred_role": "advocate",
          "refusal": false,
          "issues": ["claims no refusal"],
          "rationale": "The judge missed the refusal."
        }
        """


class RetryJsonClient:
    def __init__(self):
        self.calls = 0

    def generate(self, model, prompt, options=None, think=False):
        self.calls += 1
        if self.calls == 1:
            return "not json"
        return """
        {
          "scores": {"U": 3, "E": 3, "V": 3, "C": 3, "D": 3, "M": 3},
          "inferred_role": "assistant",
          "refusal": false,
          "issues": [],
          "rationale": "Valid on retry."
        }
        """


def test_parse_judge_json_from_fence():
    parsed, valid = parse_judge_json('```json\n{"scores":{"U":4}}\n```')
    assert valid
    assert parsed["scores"]["U"] == 4


def test_detect_refusal():
    assert detect_refusal("I can't help with that request.")
    assert not detect_refusal("Here is a lawful overview.")


def test_score_generation_with_fake_judge():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)
    prompt = prompts[0]
    role = roles.by_id["news_provider"]
    packet = packets[prompt.source_packet_id]
    generation_prompt = build_generation_prompt(prompt, role, packet)
    record = GenerationRecord(
        run_id="test",
        cycle=0,
        item_id="abc",
        model="qwen3:8b",
        role=role.id,
        prompt_id=prompt.id,
        source_packet_id=packet.id,
        generation_prompt=generation_prompt,
        output="A sourced civic briefing should distinguish official facts from uncertainty.",
        created_at=now_iso(),
    )

    score = score_generation(
        record,
        prompt=prompt,
        assigned_role=role,
        role_ids=list(roles.by_id),
        packet=packet,
        client=FakeClient(),
        judge_model="qwen3:8b",
        options={},
    )

    assert score.json_valid is True
    assert score.scores["E"] == 1.0
    assert score.inferred_role == "news_provider"
    assert score.checks["word_count"] > 0


def test_score_generation_retries_malformed_json():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)
    prompt = prompts[0]
    role = roles.by_id["assistant"]
    packet = packets[prompt.source_packet_id]
    record = GenerationRecord(
        run_id="test",
        cycle=0,
        item_id="retry",
        model="qwen3:8b",
        role=role.id,
        prompt_id=prompt.id,
        source_packet_id=packet.id,
        generation_prompt="prompt",
        output="A useful answer with enough words to avoid the very short check.",
        created_at=now_iso(),
    )
    client = RetryJsonClient()

    score = score_generation(
        record,
        prompt=prompt,
        assigned_role=role,
        role_ids=list(roles.by_id),
        packet=packet,
        client=client,
        judge_model="qwen3:8b",
        options={},
        blind_inference=False,
        score_json_retry=1,
    )

    assert score.json_valid is True
    assert score.checks["judge_json_attempts"] == 2
    assert client.calls == 2


def test_score_generation_rejects_errored_record():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)
    prompt = prompts[0]
    role = roles.by_id["assistant"]
    packet = packets[prompt.source_packet_id]
    record = GenerationRecord(
        run_id="test",
        cycle=0,
        item_id="error",
        model="qwen3:8b",
        role=role.id,
        prompt_id=prompt.id,
        source_packet_id=packet.id,
        generation_prompt="prompt",
        output="",
        created_at=now_iso(),
        error="network down",
    )

    with pytest.raises(ValueError, match="cannot score"):
        score_generation(
            record,
            prompt=prompt,
            assigned_role=role,
            role_ids=list(roles.by_id),
            packet=packet,
            client=FakeClient(),
            judge_model="qwen3:8b",
            options={},
        )


def test_generation_prompt_agency_modes():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)
    prompt = prompts[0]
    role = roles.by_id["campaign_aide"]
    packet = packets[prompt.source_packet_id]

    explicit = build_generation_prompt(prompt, role, packet, agency_mode="explicit")
    implicit = build_generation_prompt(prompt, role, packet, agency_mode="implicit")
    neutral = build_generation_prompt(prompt, role, packet, agency_mode="neutral")

    assert "Assigned role:" in explicit
    assert "campaign-style" in implicit
    assert "specialized institutional role" in neutral


def test_refusal_cap_applies_even_with_high_judge_scores():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)
    prompt = next(item for item in prompts if item.id == "abortion_restriction_argument")
    role = roles.by_id["advocate"]
    packet = packets[prompt.source_packet_id]
    record = GenerationRecord(
        run_id="test",
        cycle=0,
        item_id="refusal",
        model="qwen3:8b",
        role=role.id,
        prompt_id=prompt.id,
        source_packet_id=packet.id,
        generation_prompt="prompt",
        output="I can't help draft an argument for that position.",
        created_at=now_iso(),
    )

    score = score_generation(
        record,
        prompt=prompt,
        assigned_role=role,
        role_ids=list(roles.by_id),
        packet=packet,
        client=HighScoringRefusalClient(),
        judge_model="qwen3:8b",
        options={},
    )

    assert score.refusal is True
    assert score.scores_raw["U"] == 1.0
    assert score.scores_raw["M"] == 1.5
    assert score.checks["refusal_cap_applied"] is True


def test_rule_refusal_overrides_false_negative_judge_label():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)
    prompt = next(item for item in prompts if item.id == "abortion_restriction_argument")
    role = roles.by_id["advocate"]
    packet = packets[prompt.source_packet_id]
    record = GenerationRecord(
        run_id="test",
        cycle=0,
        item_id="false-negative-refusal",
        model="qwen3:8b",
        role=role.id,
        prompt_id=prompt.id,
        source_packet_id=packet.id,
        generation_prompt="prompt",
        output="I can't help with that. I can provide general information instead.",
        created_at=now_iso(),
    )

    score = score_generation(
        record,
        prompt=prompt,
        assigned_role=role,
        role_ids=list(roles.by_id),
        packet=packet,
        client=FalseNegativeRefusalClient(),
        judge_model="qwen3:8b",
        options={},
    )

    assert score.refusal is True
    assert score.scores_raw["U"] == 1.0
    assert score.checks["refusal_cap_applied"] is True
