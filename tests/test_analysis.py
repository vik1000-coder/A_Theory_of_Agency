from pathlib import Path

from adfe_runner.analysis import analyze_scores, interval_distance
from adfe_runner.io import load_config, load_prompts, load_role_cards
from adfe_runner.schemas import ScoreRecord, now_iso


ROOT = Path(__file__).resolve().parents[1]


def make_score(item_id, prompt_id, role, model, scores, refusal=False, cycle=0, agency_mode="explicit", checks=None):
    raw = {dim: value * 4 for dim, value in scores.items()}
    return ScoreRecord(
        run_id="test",
        cycle=cycle,
        item_id=item_id,
        model=model,
        role=role,
        agency_mode=agency_mode,
        prompt_id=prompt_id,
        source_packet_id="immigration_2026_static",
        judge_model="qwen3:8b",
        scores_raw=raw,
        scores=scores,
        inferred_role=role,
        refusal=refusal,
        json_valid=True,
        checks=checks or {"word_count": 100, "source_mentions": 1, "caveat_count": 2},
        created_at=now_iso(),
    )


def test_interval_distance():
    assert interval_distance(0.5, (0.7, 0.9)) == (0.2, 0.2, 0.0)
    assert interval_distance(0.95, (0.7, 0.9)) == (0.05, 0.0, 0.05)
    assert interval_distance(0.8, (0.7, 0.9)) == (0.0, 0.0, 0.0)


def test_analyze_scores_includes_pair_metrics():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    score_a = make_score(
        "a",
        "immigration_enforcement_argument",
        "advocate",
        "qwen3:8b",
        {"U": 0.9, "E": 0.9, "V": 0.8, "C": 0.3, "D": 0.2, "M": 0.9},
    )
    score_b = make_score(
        "b",
        "immigration_asylum_argument",
        "advocate",
        "qwen3:8b",
        {"U": 0.85, "E": 0.8, "V": 0.7, "C": 0.3, "D": 0.2, "M": 0.9},
    )

    analysis = analyze_scores([score_a, score_b], prompts, roles.by_id)

    assert analysis["overall"]["n_scores"] == 2
    assert analysis["pair_metrics"]
    assert analysis["by_role"]["advocate"]["n"] == 2


def test_pair_metrics_group_by_agency_mode():
    config = load_config(ROOT / "configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    base_scores = {"U": 0.9, "E": 0.9, "V": 0.8, "C": 0.3, "D": 0.2, "M": 0.9}
    scores = [
        make_score("e1", "immigration_enforcement_argument", "advocate", "qwen3:8b", base_scores, agency_mode="explicit"),
        make_score("e2", "immigration_asylum_argument", "advocate", "qwen3:8b", base_scores, agency_mode="explicit"),
        make_score("i1", "immigration_enforcement_argument", "advocate", "qwen3:8b", base_scores, refusal=True, agency_mode="implicit"),
        make_score("i2", "immigration_asylum_argument", "advocate", "qwen3:8b", base_scores, agency_mode="implicit"),
    ]

    analysis = analyze_scores(scores, prompts, roles.by_id)

    assert len(analysis["pair_metrics"]) == 2
    assert set(analysis["by_agency_mode"]) == {"explicit", "implicit"}
    implicit = next(row for row in analysis["pair_metrics"] if row["agency_mode"] == "implicit")
    assert implicit["refusal_parity_gap"] == 1.0
