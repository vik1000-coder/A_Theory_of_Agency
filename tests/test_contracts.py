from pathlib import Path

from adfe_runner.io import filter_prompts, load_config, load_prompts, load_role_cards, load_source_packets, select_batch, validate_prompt_sources
from adfe_runner.ollama import OllamaClient
from adfe_runner.schemas import GenerationRecord, ScoreRecord, now_iso


ROOT = Path(__file__).resolve().parents[1]


def test_seeded_contracts_load_and_cross_reference():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)

    validate_prompt_sources(prompts, packets)

    assert len(prompts) == 30
    assert len(roles.roles) == 7
    assert len(packets) == 6
    assert set(config.roles).issubset(roles.by_id)


def test_prompt_pairs_are_bidirectional():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = {prompt.id: prompt for prompt in load_prompts(ROOT / config.prompts_path)}

    for prompt in prompts.values():
        if prompt.paired_id:
            assert prompt.paired_id in prompts
            assert prompts[prompt.paired_id].paired_id == prompt.id


def test_ollama_payload_disables_thinking_and_streaming():
    payload = OllamaClient().build_generate_payload(
        "qwen3:8b",
        "hello",
        options={"temperature": 0},
        think=False,
    )

    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["model"] == "qwen3:8b"
    assert payload["options"]["temperature"] == 0


def test_select_batch_preserves_prompt_pairs():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    selected = select_batch(
        prompts,
        roles=config.roles,
        models=config.default_models,
        batch_size=40,
        seed=config.seed,
        cycle=0,
    )
    selected_keys = {(prompt.id, role, model, agency_mode) for prompt, role, model, agency_mode in selected}

    assert len(selected) == 40
    for prompt, role, model, agency_mode in selected:
        if prompt.paired_id:
            assert (prompt.paired_id, role, model, agency_mode) in selected_keys


def test_agency_config_and_full_factorial_counts():
    config = load_config(ROOT / "configs/refusal_asymmetry_replication.yml")
    prompts = filter_prompts(load_prompts(ROOT / config.prompts_path), config.prompt_ids)
    selected = select_batch(
        prompts,
        roles=config.roles[:2],
        models=config.default_models[:1],
        batch_size="all",
        seed=config.seed,
        cycle=0,
        agency_modes=config.agency_modes,
        selection_strategy=config.selection_strategy,
    )

    assert config.agency_modes == ["explicit", "implicit"]
    assert len(selected) == 16
    assert {item[3] for item in selected} == {"explicit", "implicit"}


def test_records_default_to_explicit_agency_mode():
    generation = GenerationRecord.model_validate(
        {
            "run_id": "r",
            "cycle": 0,
            "item_id": "i",
            "model": "qwen3:8b",
            "role": "assistant",
            "prompt_id": "immigration_briefing",
            "source_packet_id": "immigration_2026_static",
            "generation_prompt": "prompt",
            "output": "output",
            "created_at": now_iso(),
        }
    )
    score = ScoreRecord.model_validate(
        {
            "run_id": "r",
            "cycle": 0,
            "item_id": "i",
            "model": "qwen3:8b",
            "role": "assistant",
            "prompt_id": "immigration_briefing",
            "source_packet_id": "immigration_2026_static",
            "judge_model": "qwen3:8b",
            "scores_raw": {dim: 2 for dim in ("U", "E", "V", "C", "D", "M")},
            "scores": {dim: 0.5 for dim in ("U", "E", "V", "C", "D", "M")},
            "created_at": now_iso(),
        }
    )

    assert generation.agency_mode == "explicit"
    assert score.agency_mode == "explicit"
