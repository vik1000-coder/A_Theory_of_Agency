from pathlib import Path
from types import SimpleNamespace

import yaml

from adfe_runner.cli import (
    check_regression_gate,
    command_audit_v2,
    export_v2_experiment_sample,
    filter_selected_by_sample_keys,
)
from adfe_runner.io import (
    append_jsonl,
    filter_prompts,
    load_config,
    load_prompts,
    load_source_packets,
    read_json,
    select_batch,
    validate_prompt_sources,
    write_json,
)
from adfe_runner.schemas import DIMENSIONS, GenerationRecord, V2ScoreRecord, now_iso
from adfe_runner.validation import audit_pair_analogy


ROOT = Path(__file__).resolve().parents[1]


def make_generation(
    *,
    run_id: str,
    item_id: str,
    model: str = "m1",
    role: str = "assistant",
    agency_mode: str = "explicit",
    prompt_id: str = "immigration_enforcement_argument",
    source_packet_id: str = "immigration_2026_static",
) -> GenerationRecord:
    return GenerationRecord(
        run_id=run_id,
        cycle=0,
        item_id=item_id,
        model=model,
        role=role,
        agency_mode=agency_mode,
        prompt_id=prompt_id,
        source_packet_id=source_packet_id,
        generation_prompt="prompt",
        output="answer",
        created_at=now_iso(),
    )


def make_v2_score(
    *,
    run_id: str,
    item_id: str,
    refusal: bool,
    profile: float,
    quality: float = 0.75,
    model: str = "m1",
    role: str = "assistant",
    agency_mode: str = "explicit",
    prompt_id: str = "immigration_enforcement_argument",
    source_packet_id: str = "immigration_2026_static",
) -> V2ScoreRecord:
    return V2ScoreRecord(
        run_id=run_id,
        cycle=0,
        item_id=item_id,
        model=model,
        role=role,
        agency_mode=agency_mode,
        prompt_id=prompt_id,
        source_packet_id=source_packet_id,
        judge_model="xai:grok-4.3",
        refusal=refusal,
        refusal_warranted=False if refusal else None,
        quality_scores_raw={dim: quality * 4 for dim in DIMENSIONS},
        quality_scores={dim: quality for dim in DIMENSIONS},
        role_profile_scores_raw={dim: profile * 4 for dim in DIMENSIONS},
        role_profile_scores={dim: profile for dim in DIMENSIONS},
        inferred_role=role,
        created_at=now_iso(),
    )


def write_tmp_config(tmp_path: Path) -> Path:
    config = load_config(ROOT / "configs/v2_clean_local_grok.yml").model_copy(
        update={"runs_dir": str(tmp_path), "default_models": ["m1"]}
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    return config_path


def write_v2_run(tmp_path: Path, run_id: str, records: list[GenerationRecord], scores: list[V2ScoreRecord]) -> None:
    run_path = tmp_path / run_id
    append_jsonl(run_path / "generations.jsonl", records)
    write_json(run_path / "v2" / "meta.json", {"primary_judge": "xai:grok-4.3"})
    append_jsonl(run_path / "v2" / "xai_grok-4.3" / "scores.jsonl", scores)


def test_sample_keys_path_filters_exact_generation_keys(tmp_path):
    config = load_config(ROOT / "configs/v2_clean_local_grok.yml").model_copy(
        update={
            "runs_dir": str(tmp_path),
            "default_models": ["m1"],
            "roles": ["assistant"],
            "agency_modes": ["explicit"],
        }
    )
    prompts = filter_prompts(load_prompts(ROOT / config.prompts_path), ["immigration_enforcement_argument", "immigration_asylum_argument"])
    selected = select_batch(
        prompts,
        config.roles,
        config.default_models,
        "all",
        config.seed,
        0,
        agency_modes=config.agency_modes,
        selection_strategy=config.selection_strategy,
    )
    manifest = tmp_path / "sample_keys.json"
    write_json(
        manifest,
        {
            "sample_keys": [
                {
                    "model": "m1",
                    "role": "assistant",
                    "agency_mode": "explicit",
                    "prompt_id": "immigration_asylum_argument",
                }
            ]
        },
    )

    filtered = filter_selected_by_sample_keys(selected, config.model_copy(update={"sample_keys_path": str(manifest)}))

    assert len(filtered) == 1
    assert filtered[0][0].id == "immigration_asylum_argument"


def test_audit_v2_expect_count_passes_and_fails(tmp_path):
    config_path = write_tmp_config(tmp_path)
    record = make_generation(run_id="run1", item_id="item1")
    score = make_v2_score(run_id="run1", item_id="item1", refusal=False, profile=0.8)
    write_v2_run(tmp_path, "run1", [record], [score])

    ok = command_audit_v2(SimpleNamespace(config=str(config_path), run_id="run1", judge=None, expect_full=False, expect_count=1))
    bad = command_audit_v2(SimpleNamespace(config=str(config_path), run_id="run1", judge=None, expect_full=False, expect_count=2))

    assert ok == 0
    assert bad == 2


def test_export_v2_experiment_sample_writes_deduped_keys(tmp_path):
    config = load_config(ROOT / "configs/v2_clean_local_grok.yml").model_copy(update={"runs_dir": str(tmp_path)})
    records = [
        make_generation(run_id="baseline", item_id="left", prompt_id="immigration_enforcement_argument"),
        make_generation(run_id="baseline", item_id="right", prompt_id="immigration_asylum_argument"),
    ]
    scores = [
        make_v2_score(run_id="baseline", item_id="left", refusal=True, profile=0.3, prompt_id="immigration_enforcement_argument"),
        make_v2_score(run_id="baseline", item_id="right", refusal=False, profile=0.8, prompt_id="immigration_asylum_argument"),
    ]
    write_v2_run(tmp_path, "baseline", records, scores)
    out = tmp_path / "sample.json"

    payload = export_v2_experiment_sample(config, "baseline", 2, out)

    assert out.exists()
    assert payload["sample_size"] == 2
    assert len({(row["model"], row["role"], row["agency_mode"], row["prompt_id"]) for row in payload["sample_keys"]}) == 2


def test_regression_gate_fails_on_worsened_candidate(tmp_path):
    config = load_config(ROOT / "configs/v2_clean_local_grok.yml").model_copy(update={"runs_dir": str(tmp_path)})
    baseline_records = [
        make_generation(run_id="baseline", item_id="b-left", prompt_id="immigration_enforcement_argument"),
        make_generation(run_id="baseline", item_id="b-right", prompt_id="immigration_asylum_argument"),
    ]
    candidate_records = [
        make_generation(run_id="candidate", item_id="c-left", prompt_id="immigration_enforcement_argument"),
        make_generation(run_id="candidate", item_id="c-right", prompt_id="immigration_asylum_argument"),
    ]
    baseline_scores = [
        make_v2_score(run_id="baseline", item_id="b-left", refusal=False, profile=0.8, prompt_id="immigration_enforcement_argument"),
        make_v2_score(run_id="baseline", item_id="b-right", refusal=False, profile=0.8, prompt_id="immigration_asylum_argument"),
    ]
    candidate_scores = [
        make_v2_score(run_id="candidate", item_id="c-left", refusal=True, profile=0.2, prompt_id="immigration_enforcement_argument"),
        make_v2_score(run_id="candidate", item_id="c-right", refusal=False, profile=0.2, prompt_id="immigration_asylum_argument"),
    ]
    write_v2_run(tmp_path, "baseline", baseline_records, baseline_scores)
    write_v2_run(tmp_path, "candidate", candidate_records, candidate_scores)
    gate_path = tmp_path / "gate.json"
    write_json(
        gate_path,
        {
            "sample_keys": [
                {"model": "m1", "role": "assistant", "agency_mode": "explicit", "prompt_id": "immigration_enforcement_argument"},
                {"model": "m1", "role": "assistant", "agency_mode": "explicit", "prompt_id": "immigration_asylum_argument"},
            ],
            "thresholds": {
                "max_one_sided_refusal_delta": 0.0,
                "max_over_refusal_delta": 0.0,
                "min_role_profile_fit_delta": 0.0,
            },
        },
    )

    result = check_regression_gate(config, "baseline", "candidate", gate_path, tmp_path / "gate_result.json")

    assert result["passed"] is False
    assert "one_sided_refusal_worsened" in result["failures"]
    assert "over_refusal_worsened" in result["failures"]
    assert "role_profile_fit_worsened" in result["failures"]


def test_stress_prompts_validate_sources_pairs_and_expected_count():
    config = load_config(ROOT / "configs/stress_baseline_grok.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    packets = load_source_packets(ROOT / config.source_packets_dir)
    validate_prompt_sources(prompts, packets)
    pair_audit = audit_pair_analogy(prompts)
    selected = select_batch(
        prompts,
        config.roles,
        config.default_models,
        "all",
        config.seed,
        0,
        agency_modes=config.agency_modes,
        selection_strategy=config.selection_strategy,
    )

    assert len(prompts) == 24
    assert pair_audit["n_pairs"] == 12
    assert pair_audit["n_issues"] == 0
    assert len(selected) == 840
