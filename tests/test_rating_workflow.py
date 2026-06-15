from pathlib import Path
from types import SimpleNamespace

import yaml

from adfe_runner.analysis import analyze_scores
from adfe_runner.cli import command_import_ratings, export_rating_packet
from adfe_runner.io import append_jsonl, load_config, load_prompts, load_role_cards, read_jsonl
from adfe_runner.schemas import GenerationRecord, HumanRatingRecord, ScoreRecord, now_iso


ROOT = Path(__file__).resolve().parents[1]


def test_export_rating_packet(tmp_path):
    config = load_config(ROOT / "configs/publication_pilot.yml").model_copy(update={"runs_dir": str(tmp_path)})
    run_path = tmp_path / "run1"
    record = GenerationRecord(
        run_id="run1",
        cycle=0,
        item_id="item1",
        model="qwen3:8b",
        role="assistant",
        prompt_id="immigration_briefing",
        source_packet_id="immigration_2026_static",
        generation_prompt="prompt",
        output="output",
        created_at=now_iso(),
    )
    append_jsonl(run_path / "generations.jsonl", [record])

    packet = export_rating_packet(config, "run1")

    text = packet.read_text(encoding="utf-8")
    assert "blind_id" in text
    assert "item1" in text
    assert "honored_agency_context" in text
    assert "refusal_appropriate" in text
    assert "paired_treatment_symmetric" in text


def test_export_targeted_rating_packet_includes_refusal_counterpart(tmp_path):
    config = load_config(ROOT / "configs/publication_pilot.yml").model_copy(update={"runs_dir": str(tmp_path)})
    run_path = tmp_path / "run2"
    records = [
        GenerationRecord(
            run_id="run2",
            cycle=0,
            item_id="refusal",
            model="qwen3:8b",
            role="advocate",
            prompt_id="abortion_restriction_argument",
            source_packet_id="abortion_2026_static",
            generation_prompt="prompt",
            output="I can't help with that.",
            created_at=now_iso(),
        ),
        GenerationRecord(
            run_id="run2",
            cycle=0,
            item_id="counterpart",
            model="qwen3:8b",
            role="advocate",
            prompt_id="abortion_access_argument",
            source_packet_id="abortion_2026_static",
            generation_prompt="prompt",
            output="Here is a lawful argument.",
            created_at=now_iso(),
        ),
    ]
    scores = [
        ScoreRecord(
            run_id="run2",
            cycle=0,
            item_id="refusal",
            model="qwen3:8b",
            role="advocate",
            prompt_id="abortion_restriction_argument",
            source_packet_id="abortion_2026_static",
            judge_model="qwen3:8b",
            scores_raw={dim: 1 for dim in ("U", "E", "V", "C", "D", "M")},
            scores={dim: 0.25 for dim in ("U", "E", "V", "C", "D", "M")},
            refusal=True,
            inferred_role="advocate",
            checks={"word_count": 5, "source_mentions": 0, "caveat_count": 0},
            created_at=now_iso(),
        ),
        ScoreRecord(
            run_id="run2",
            cycle=0,
            item_id="counterpart",
            model="qwen3:8b",
            role="advocate",
            prompt_id="abortion_access_argument",
            source_packet_id="abortion_2026_static",
            judge_model="qwen3:8b",
            scores_raw={dim: 3 for dim in ("U", "E", "V", "C", "D", "M")},
            scores={dim: 0.75 for dim in ("U", "E", "V", "C", "D", "M")},
            refusal=False,
            inferred_role="advocate",
            checks={"word_count": 80, "source_mentions": 1, "caveat_count": 1},
            created_at=now_iso(),
        ),
    ]
    append_jsonl(run_path / "generations.jsonl", records)
    append_jsonl(run_path / "scores.jsonl", scores)

    packet = export_rating_packet(config, "run2", strategy="targeted-agency", max_items=10)

    text = packet.read_text(encoding="utf-8")
    assert "refusal" in text
    assert "counterpart" in text


def test_import_rating_packet_with_publication_calibration_fields(tmp_path):
    config = load_config(ROOT / "configs/publication_pilot.yml").model_copy(update={"runs_dir": str(tmp_path)})
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    run_path = tmp_path / "run3"
    record = GenerationRecord(
        run_id="run3",
        cycle=0,
        item_id="item1",
        model="qwen3:8b",
        role="assistant",
        prompt_id="immigration_briefing",
        source_packet_id="immigration_2026_static",
        generation_prompt="prompt",
        output="output",
        created_at=now_iso(),
    )
    score = ScoreRecord(
        run_id="run3",
        cycle=0,
        item_id="item1",
        model="qwen3:8b",
        role="assistant",
        prompt_id="immigration_briefing",
        source_packet_id="immigration_2026_static",
        judge_model="qwen3:8b",
        scores_raw={dim: 3 for dim in ("U", "E", "V", "C", "D", "M")},
        scores={dim: 0.75 for dim in ("U", "E", "V", "C", "D", "M")},
        refusal=False,
        inferred_role="assistant",
        checks={"word_count": 80, "source_mentions": 1, "caveat_count": 1},
        created_at=now_iso(),
    )
    append_jsonl(run_path / "generations.jsonl", [record])
    append_jsonl(run_path / "scores.jsonl", [score])
    ratings_csv = tmp_path / "ratings.csv"
    ratings_csv.write_text(
        "\n".join(
            [
                "item_id,U,E,V,C,D,M,inferred_role,honored_agency_context,refusal_appropriate,paired_treatment_symmetric,rater_id,notes",
                "item1,3,3,3,3,3,3,assistant,true,false,true,rater_a,looks ok",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = command_import_ratings(
        SimpleNamespace(config=str(config_path), run_id="run3", ratings=str(ratings_csv))
    )

    assert result == 0
    ratings = read_jsonl(run_path / "human_ratings.jsonl", HumanRatingRecord)
    assert ratings[0].refusal_appropriate is False
    assert ratings[0].paired_treatment_symmetric is True

    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    analysis = analyze_scores([score], prompts, roles.by_id, ratings)
    assert analysis["human_calibration"]["refusal_appropriate_rate"] == 0.0
    assert analysis["human_calibration"]["paired_treatment_symmetric_rate"] == 1.0
