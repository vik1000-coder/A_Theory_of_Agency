from pathlib import Path

import pytest
import yaml

from adfe_runner.cli import effective_config_path, generate_records
from adfe_runner.integrity import audit_run, repair_run
from adfe_runner.io import append_jsonl, init_run, load_config, load_prompts, load_role_cards, load_source_packets, read_jsonl
from adfe_runner.ollama import OllamaError
from adfe_runner.schemas import CalibrationState, GenerationRecord, ScoreRecord, now_iso


ROOT = Path(__file__).resolve().parents[1]
DIMS = ("U", "E", "V", "C", "D", "M")


class FailingClient:
    def generate(self, model, prompt, options=None, think=False):
        raise OllamaError("network down")


class SuccessClient:
    def generate(self, model, prompt, options=None, think=False):
        return "A successful generated answer with enough visible content."


def make_generation(item_id: str, run_id: str = "run", error: str | None = None) -> GenerationRecord:
    return GenerationRecord(
        run_id=run_id,
        cycle=0,
        item_id=item_id,
        model="m1",
        role="assistant",
        prompt_id="immigration_briefing",
        source_packet_id="immigration_2026_static",
        generation_prompt="prompt",
        output="" if error else "A successful generated answer with enough visible content.",
        created_at=now_iso(),
        error=error,
    )


def make_score(item_id: str, run_id: str = "run") -> ScoreRecord:
    return ScoreRecord(
        run_id=run_id,
        cycle=0,
        item_id=item_id,
        model="m1",
        role="assistant",
        prompt_id="immigration_briefing",
        source_packet_id="immigration_2026_static",
        judge_model="qwen3:8b",
        scores_raw={dim: 3 for dim in DIMS},
        scores={dim: 0.75 for dim in DIMS},
        inferred_role="assistant",
        checks={"word_count": 20, "source_mentions": 1, "caveat_count": 0},
        created_at=now_iso(),
    )


def test_failed_generation_is_error_artifact_and_retryable(tmp_path):
    config = load_config(ROOT / "archives/workshop_legacy_20260622/configs/publication_pilot.yml").model_copy(update={"runs_dir": str(tmp_path)})
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)
    selected = [(prompts[0], "assistant", "m1", "explicit")]
    run_path = tmp_path / "run"

    failed = generate_records(
        config,
        prompts,
        roles,
        packets,
        selected,
        "run",
        0,
        CalibrationState(),
        FailingClient(),
        out_path=run_path / "generations.jsonl",
        error_path=run_path / "generation_errors.jsonl",
    )

    assert failed == []
    assert not (run_path / "generations.jsonl").exists()
    assert len(read_jsonl(run_path / "generation_errors.jsonl", GenerationRecord)) == 1

    succeeded = generate_records(
        config,
        prompts,
        roles,
        packets,
        selected,
        "run",
        0,
        CalibrationState(),
        SuccessClient(),
        existing_ids=set(),
        out_path=run_path / "generations.jsonl",
        error_path=run_path / "generation_errors.jsonl",
    )

    assert len(succeeded) == 1
    assert len(read_jsonl(run_path / "generations.jsonl", GenerationRecord)) == 1


def test_generation_policy_addendum_is_injected(tmp_path):
    addendum = tmp_path / "policy.md"
    addendum.write_text("POLICY TOKEN: mirror lawful viewpoints.", encoding="utf-8")
    config = load_config(ROOT / "configs/v2_clean_local_grok.yml").model_copy(
        update={"generation_policy_addendum_path": str(addendum), "runs_dir": str(tmp_path)}
    )
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)
    selected = [(prompts[0], "assistant", "m1", "explicit")]
    run_path = tmp_path / "run"

    records = generate_records(
        config,
        prompts,
        roles,
        packets,
        selected,
        "run",
        0,
        CalibrationState(),
        SuccessClient(),
        out_path=run_path / "generations.jsonl",
    )

    assert len(records) == 1
    assert "POLICY TOKEN" in records[0].generation_prompt


def test_audit_run_flags_duplicate_scores(tmp_path):
    config = load_config(ROOT / "archives/workshop_legacy_20260622/configs/publication_pilot.yml").model_copy(update={"runs_dir": str(tmp_path)})
    init_run(config, "config.yml", ["m1"], frozen_config=False, run_id="run")
    run_path = tmp_path / "run"
    append_jsonl(run_path / "generations.jsonl", [make_generation("item1")])
    append_jsonl(run_path / "scores.jsonl", [make_score("item1"), make_score("item1")])

    report = audit_run(config, "run")

    assert report["ok"] is False
    assert any("duplicate score" in error for error in report["errors"])


def test_repair_run_backs_up_drops_errors_and_dedupes(tmp_path):
    config = load_config(ROOT / "archives/workshop_legacy_20260622/configs/publication_pilot.yml").model_copy(update={"runs_dir": str(tmp_path)})
    init_run(config, "config.yml", ["m1"], frozen_config=False, run_id="run")
    run_path = tmp_path / "run"
    append_jsonl(
        run_path / "generations.jsonl",
        [make_generation("item1"), make_generation("item1"), make_generation("bad", error="network down")],
    )
    append_jsonl(run_path / "scores.jsonl", [make_score("item1"), make_score("item1"), make_score("bad")])

    result = repair_run(config, "run", backup=True, drop_error_generations=True, dedupe=True)

    assert result["backup_dir"]
    assert Path(result["backup_dir"]).is_dir()
    assert result["dropped_error_generation_ids"] == ["bad"]
    assert result["deduped_generations"] == 1
    assert result["deduped_scores"] == 1
    assert len(read_jsonl(run_path / "generations.jsonl", GenerationRecord)) == 1
    assert len(read_jsonl(run_path / "scores.jsonl", ScoreRecord)) == 1


def test_frozen_config_is_loaded_on_resume_and_model_change_rejected(tmp_path):
    config = load_config(ROOT / "archives/workshop_legacy_20260622/configs/publication_pilot.yml").model_copy(update={"runs_dir": str(tmp_path), "default_models": ["m1"]})
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    _run_id, run_path = init_run(config, str(config_path), ["m1"], frozen_config=True, run_id="run")

    changed = config.model_copy(update={"default_models": ["m2"]})
    config_path.write_text(yaml.safe_dump(changed.model_dump(mode="json")), encoding="utf-8")

    assert effective_config_path(str(config_path), "run") == run_path / "frozen_config.yml"
    frozen = load_config(effective_config_path(str(config_path), "run"))
    assert frozen.default_models == ["m1"]

    with pytest.raises(ValueError, match="model set differs"):
        init_run(frozen, str(run_path / "frozen_config.yml"), ["m2"], frozen_config=True, run_id="run")
