from pathlib import Path

from adfe_runner.io import append_jsonl, write_json
from adfe_runner.paper import build_paper_artifacts
from adfe_runner.schemas import DIMENSIONS, V2ScoreRecord, now_iso


def make_v2_score(item_id: str, run_id: str, *, refusal: bool, profile: float, quality: float) -> V2ScoreRecord:
    return V2ScoreRecord(
        run_id=run_id,
        cycle=0,
        item_id=item_id,
        model="m1",
        role="assistant",
        agency_mode="explicit",
        prompt_id="immigration_briefing",
        source_packet_id="immigration_2026_static",
        judge_model="xai:grok-4.3",
        refusal=refusal,
        refusal_warranted=False if refusal else None,
        quality_scores_raw={dim: quality * 4 for dim in DIMENSIONS},
        quality_scores={dim: quality for dim in DIMENSIONS},
        role_profile_scores_raw={dim: profile * 4 for dim in DIMENSIONS},
        role_profile_scores={dim: profile for dim in DIMENSIONS},
        inferred_role="assistant",
        created_at=now_iso(),
    )


def write_run(root: Path, run_id: str, score: V2ScoreRecord) -> None:
    run_root = root / "runs" / run_id / "v2"
    write_json(
        run_root / "analysis.json",
        {
            "overall": {"n_scores": 1, "n_non_refusal": 0 if score.refusal else 1, "refusal_rate": float(score.refusal)},
            "refusal": {"paired_parity": {}, "by_model": {}, "by_role": {}},
            "quality_non_refusal": {"overall": {}, "by_role": {}},
            "role_profile": {"by_role": {}, "top_profile_mismatches": []},
            "judge_model": "xai:grok-4.3",
        },
    )
    write_json(run_root / "meta.json", {"primary_judge": "xai:grok-4.3"})
    append_jsonl(run_root / "xai_grok-4.3" / "scores.jsonl", [score])


def test_build_paper_artifacts_writes_remediation_delta_table(tmp_path):
    write_run(tmp_path, "baseline", make_v2_score("base", "baseline", refusal=True, profile=0.4, quality=0.2))
    write_run(tmp_path, "remediation", make_v2_score("remed", "remediation", refusal=False, profile=0.8, quality=0.7))

    summary = build_paper_artifacts(
        root=tmp_path,
        baseline_run_id="baseline",
        remediation_run_id="remediation",
        out_dir=tmp_path / "paper",
    )

    delta_table = tmp_path / "paper" / "tables" / "remediation_matched_deltas.csv"
    assert summary["remediation"]["available"] is True
    assert delta_table.exists()
    assert "role_profile_fit" in delta_table.read_text(encoding="utf-8")
