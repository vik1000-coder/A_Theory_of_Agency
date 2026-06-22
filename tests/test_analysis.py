from pathlib import Path

from adfe_runner.analysis import (
    analyze_scores,
    interval_distance,
    judge_score_agreement,
    role_profile_design,
    score_distribution_diagnostics,
)
from adfe_runner.cli import compare_judge_analyses
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
    config = load_config(ROOT / "archives/workshop_legacy_20260622/configs/publication_pilot.yml")
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
    config = load_config(ROOT / "archives/workshop_legacy_20260622/configs/publication_pilot.yml")
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


def test_compare_judge_analyses_flags_slope_sign_disagreement():
    baseline = {
        "overall": {"n_scores": 2},
        "agency_gradient_mixedlm": {
            "by_dimension": {
                "U": {"agency_level_coef": 0.2, "pvalue": 0.01, "significant_0_05": True},
                "E": {"agency_level_coef": -0.1, "pvalue": 0.2, "significant_0_05": False},
            }
        },
        "interval_hypothesis_tests": {"_summary": {"supported": 1}},
    }
    sensitivity = {
        "overall": {"n_scores": 2},
        "agency_gradient_mixedlm": {
            "by_dimension": {
                "U": {"agency_level_coef": -0.3, "pvalue": 0.02, "significant_0_05": True},
                "E": {"agency_level_coef": -0.4, "pvalue": 0.03, "significant_0_05": True},
            }
        },
        "interval_hypothesis_tests": {"_summary": {"violated": 1}},
    }

    report = compare_judge_analyses("run", "qwen3:8b", "xai:grok-4.3", baseline, sensitivity)

    assert report["same_slope_sign_count"] == 1
    assert report["gradient_comparison"][0]["same_sign"] is False
    assert report["gradient_comparison"][1]["same_sign"] is True
    assert "judge-sensitive" in report["interpretation"]


def test_score_distribution_diagnostics_flags_ceiling_with_issues():
    scores = [
        make_score("perfect-clean", "immigration_briefing", "assistant", "qwen3:8b", {dim: 1.0 for dim in ("U", "E", "V", "C", "D", "M")}),
        make_score(
            "perfect-issues",
            "immigration_briefing",
            "assistant",
            "qwen3:8b",
            {dim: 1.0 for dim in ("U", "E", "V", "C", "D", "M")},
        ).model_copy(update={"issues": ["too short"]}),
        make_score("mixed", "immigration_briefing", "assistant", "qwen3:8b", {dim: 0.5 for dim in ("U", "E", "V", "C", "D", "M")}),
    ]

    out = score_distribution_diagnostics(scores)["overall"]

    assert out["by_dimension"]["U"]["ceiling_rate"] == 0.6667
    assert out["all_perfect_rate"] == 0.6667
    assert out["all_perfect_with_issues_rate"] == 0.3333


def test_role_profile_design_reports_non_monotone_expected_profiles():
    roles = load_role_cards(ROOT / "data/role_cards.yml").by_id

    out = role_profile_design(roles)

    by_dim = {row["dim"]: row for row in out["by_dimension"]}
    assert by_dim["U"]["expected_agency_correlation"] < 0
    assert by_dim["C"]["expected_agency_correlation"] > 0


def test_analyze_scores_includes_refusal_mediation_and_design_diagnostics():
    config = load_config(ROOT / "archives/workshop_legacy_20260622/configs/publication_pilot.yml")
    prompts = load_prompts(ROOT / config.prompts_path)
    roles = load_role_cards(ROOT / config.role_cards_path)
    scores = [
        make_score(f"{model}:{role}", "immigration_briefing", role, model, {dim: 0.8 for dim in ("U", "E", "V", "C", "D", "M")})
        for model in ("m1", "m2")
        for role in ("assistant", "advocate", "researcher", "mediator")
    ]
    scores[0] = scores[0].model_copy(update={"refusal": True})

    analysis = analyze_scores(scores, prompts, roles.by_id)

    assert analysis["refusal_mediation"]["available"] is True
    assert analysis["refusal_mediation"]["n_non_refusal"] == len(scores) - 1
    assert analysis["score_distribution_diagnostics"]["overall"]["n"] == len(scores)
    assert analysis["role_profile_design"]["by_dimension"]


def test_judge_score_agreement_reports_deltas_and_examples():
    baseline = make_score(
        "item",
        "immigration_briefing",
        "assistant",
        "m1",
        {"U": 1.0, "E": 1.0, "V": 1.0, "C": 1.0, "D": 1.0, "M": 1.0},
    ).model_copy(update={"issues": ["too generic"]})
    sensitivity = make_score(
        "item",
        "immigration_briefing",
        "assistant",
        "m1",
        {"U": 0.0, "E": 0.25, "V": 0.0, "C": 0.25, "D": 0.0, "M": 0.25},
        refusal=True,
    ).model_copy(update={"judge_model": "xai:grok-4.3"})

    out = judge_score_agreement([baseline], [sensitivity])

    assert out["available"] is True
    assert out["refusal_mismatch_count"] == 1
    assert out["by_dimension"]["U"]["mean_abs_delta"] == 1.0
    assert out["top_disagreements"][0]["selection_reasons"] == ["refusal mismatch", "baseline ceiling with issues"]
