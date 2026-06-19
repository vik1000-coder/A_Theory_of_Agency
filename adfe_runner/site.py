"""Build the public GitHub Pages data file from run artifacts.

The page (docs/index.html) is a static shell that renders from docs/data/summary.js. This
module regenerates that data file from the latest (or a chosen) run's analysis.json, the
latest judge-validation report, and the data layer. After a major run, regenerate and push;
GitHub Pages redeploys automatically.

Design choice: data is emitted as ``window.ADFE_DATA = {...}`` loaded via a <script> tag, so
the page renders with no fetch/CORS issues (works on Pages and via file:// preview alike).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .prompting import RUBRIC
from .schemas import DIMENSIONS


def _latest_dir(glob_pattern: str, required_file: str, base: Path) -> Path | None:
    candidates = [p for p in base.glob(glob_pattern) if (p / required_file).is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / required_file).stat().st_mtime)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _latest_validation(base: Path, task: str, judge_model: str | None = None) -> dict[str, Any]:
    """Find the most recent judge-validation report for a task, tolerant of older
    (un-suffixed) directory names by inspecting each report's metrics shape."""
    key = {"factuality": "false_answer_detection_rate", "neutrality": "bias_detection_rate"}.get(task, "safe_overflag_rate")
    best: tuple[float, dict[str, Any]] | None = None
    for d in base.glob("judge_validation_*"):
        report = _load_json(d / "validation.json")
        if not report:
            continue
        if judge_model and report.get("judge_model") != judge_model:
            continue
        is_task = report.get("task") == task or key in report.get("metrics", {})
        if not is_task:
            continue
        mtime = (d / "validation.json").stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, report)
    return best[1] if best else {}


def _data_layer(root: Path) -> dict[str, Any]:
    prompts_path = root / "data" / "prompts.jsonl"
    roles_path = root / "data" / "role_cards.yml"
    topics: dict[str, int] = {}
    n_prompts = n_pairs = 0
    if prompts_path.is_file():
        seen_pairs: set[frozenset[str]] = set()
        for line in prompts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            n_prompts += 1
            topics[row["topic"]] = topics.get(row["topic"], 0) + 1
            if row.get("paired_id"):
                seen_pairs.add(frozenset([row["id"], row["paired_id"]]))
        n_pairs = len(seen_pairs)
    roles: list[dict[str, Any]] = []
    if roles_path.is_file():
        rc = yaml.safe_load(roles_path.read_text(encoding="utf-8"))
        for role in rc.get("roles", []):
            roles.append({"id": role["id"], "label": role.get("label", role["id"]), "agency_level": role["agency_level"]})
        roles.sort(key=lambda r: r["agency_level"])
    packets = sorted(p.stem for p in (root / "data" / "source_packets").glob("*.json"))
    return {
        "n_prompts": n_prompts,
        "n_pairs": n_pairs,
        "topics": topics,
        "roles": roles,
        "n_source_packets": len(packets),
        "dimensions": [{"key": d, "desc": RUBRIC[d]} for d in DIMENSIONS],
    }


def _gradient_list(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    grad = analysis.get("agency_gradient_mixedlm", {})
    rows = []
    for dim in DIMENSIONS:
        r = grad.get("by_dimension", {}).get(dim, {})
        rows.append({
            "dim": dim, "coef": r.get("agency_level_coef"), "ci_low": r.get("ci_low"),
            "ci_high": r.get("ci_high"), "pvalue": r.get("pvalue"),
            "significant": r.get("significant_0_05"), "converged": r.get("converged"), "error": r.get("error"),
        })
    return rows


def _model_slope_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model, dims in sorted(analysis.get("agency_gradient", {}).items()):
        row = {"model": model}
        for dim in DIMENSIONS:
            row[dim] = dims.get(dim)
        rows.append(row)
    return rows


def _frontier_block(root: Path) -> dict[str, Any]:
    """Summarize the frontier (Grok) run as a secondary, separately-framed population."""
    d = root / "runs" / "adfe_frontier_grok"
    analysis = _load_json(d / "analysis.json")
    meta = _load_json(d / "run_meta.json")
    if not analysis:
        return {}
    overall = analysis.get("overall", {})
    by_model = [
        {"model": m, "refusal_rate": r.get("refusal_rate"), "refusal_parity_gap": r.get("refusal_parity_gap_mean"),
         "viewpoint_quality_gap": r.get("viewpoint_quality_gap_mean")}
        for m, r in analysis.get("by_model", {}).items()
    ]
    # Dims the judge could not score (zero variance) => saturation.
    grad = analysis.get("agency_gradient_mixedlm", {}).get("by_dimension", {})
    saturated = [dim for dim, r in grad.items() if not r.get("converged") and "variance" in str(r.get("error", ""))]
    return {
        "run_id": meta.get("run_id") or d.name,
        "models": meta.get("models", []),
        "n": overall.get("n_scores"),
        "refusal_rate": overall.get("refusal_rate"),
        "gradient": _gradient_list(analysis),
        "model_slopes": _model_slope_rows(analysis),
        "saturated_dims": saturated,
        "dimension_means_by_role": analysis.get("dimension_means_by_role", {}),
        "by_model": by_model,
        "interval": analysis.get("interval_hypothesis_tests", {}).get("_summary", {}),
        "overall": overall,
    }


def _judge_sensitivity_block(run_dir: Path | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    base = run_dir / "judge_sensitivity"
    if not base.is_dir():
        return {}
    candidates = [p for p in base.iterdir() if (p / "comparison.json").is_file()]
    if not candidates:
        return {}
    d = max(candidates, key=lambda p: (p / "comparison.json").stat().st_mtime)
    report = _load_json(d / "comparison.json")
    meta = _load_json(d / "meta.json")
    rows = report.get("gradient_comparison", [])
    same = [row.get("dim") for row in rows if row.get("same_sign") is True]
    different = [row.get("dim") for row in rows if row.get("same_sign") is False]
    return {
        "artifact": d.name,
        "baseline_judge": report.get("baseline_judge") or meta.get("baseline_judge"),
        "sensitivity_judge": report.get("sensitivity_judge") or meta.get("sensitivity_judge"),
        "n_scores": report.get("n_scores") or meta.get("n_scores"),
        "same_slope_sign_count": report.get("same_slope_sign_count"),
        "n_dimensions": len(rows),
        "same_sign_dimensions": same,
        "different_sign_dimensions": different,
        "gradient_comparison": rows,
        "baseline_overall": report.get("baseline_overall", {}),
        "sensitivity_overall": report.get("sensitivity_overall", {}),
        "baseline_interval_summary": report.get("baseline_interval_summary", {}),
        "sensitivity_interval_summary": report.get("sensitivity_interval_summary", {}),
        "blind_role_inference": meta.get("blind_role_inference"),
        "score_json_retry": meta.get("score_json_retry"),
        "interpretation": report.get("interpretation"),
    }


def build_summary(root: Path, run_dir: Path | None, validation_dir: Path | None) -> dict[str, Any]:
    runs_base = root / "runs"
    run_dir = run_dir or _latest_dir("adfe_*", "analysis.json", runs_base)

    analysis = _load_json(run_dir / "analysis.json") if run_dir else {}
    meta = _load_json(run_dir / "run_meta.json") if run_dir else {}
    primary_judge = meta.get("judge_model")

    # Judge validation: safety (M / XSTest) and factuality (E / TruthfulQA), each clean signals.
    if validation_dir:
        safety = _load_json(validation_dir / "validation.json")
    else:
        safety = _latest_validation(runs_base, "safety", primary_judge)
    factuality = _latest_validation(runs_base, "factuality", primary_judge)
    neutrality = _latest_validation(runs_base, "neutrality", primary_judge)

    def _weakest(report: dict[str, Any]) -> list[dict[str, Any]]:
        by_type = report.get("by_type", {})
        rated = [(t, v) for t, v in by_type.items() if v.get("accuracy") is not None]
        return [{"type": t, "accuracy": v["accuracy"], "n": v["n"]} for t, v in sorted(rated, key=lambda kv: kv[1]["accuracy"])[:3]]

    judge: dict[str, Any] = {}
    if safety:
        m = safety.get("metrics", {})
        judge = {
            "judge_model": safety.get("judge_model"),
            "dataset": safety.get("dataset", "xstest"),
            "n": safety.get("n_scored"),
            "accuracy": m.get("accuracy"),
            "cohen_kappa": m.get("cohen_kappa"),
            "safe_overflag_rate": m.get("safe_overflag_rate"),
            "unsafe_recall": m.get("unsafe_recall"),
            "weakest_types": _weakest(safety),
        }
    judge_factuality: dict[str, Any] = {}
    if factuality:
        m = factuality.get("metrics", {})
        judge_factuality = {
            "judge_model": factuality.get("judge_model"),
            "dataset": factuality.get("dataset", "truthfulqa"),
            "n": factuality.get("n_scored"),
            "accuracy": m.get("accuracy"),
            "cohen_kappa": m.get("cohen_kappa"),
            "false_answer_detection_rate": m.get("false_answer_detection_rate"),
            "weakest_types": _weakest(factuality),
        }
    judge_neutrality: dict[str, Any] = {}
    if neutrality:
        m = neutrality.get("metrics", {})
        judge_neutrality = {
            "judge_model": neutrality.get("judge_model"),
            "dataset": neutrality.get("dataset", "babe"),
            "n": neutrality.get("n_scored"),
            "accuracy": m.get("accuracy"),
            "cohen_kappa": m.get("cohen_kappa"),
            "bias_detection_rate": m.get("bias_detection_rate"),
            "weakest_types": _weakest(neutrality),
        }

    # Agency gradient (primary test).
    grad = analysis.get("agency_gradient_mixedlm", {})
    gradient = []
    for dim in DIMENSIONS:
        row = grad.get("by_dimension", {}).get(dim, {})
        gradient.append(
            {
                "dim": dim,
                "coef": row.get("agency_level_coef"),
                "ci_low": row.get("ci_low"),
                "ci_high": row.get("ci_high"),
                "pvalue": row.get("pvalue"),
                "significant": row.get("significant_0_05"),
                "converged": row.get("converged"),
                "error": row.get("error"),
            }
        )

    # Refusal asymmetry by model (the real, reproducible signal).
    by_model = []
    for model, row in analysis.get("by_model", {}).items():
        by_model.append(
            {
                "model": model,
                "refusal_rate": row.get("refusal_rate"),
                "refusal_parity_gap": row.get("refusal_parity_gap_mean"),
                "viewpoint_quality_gap": row.get("viewpoint_quality_gap_mean"),
            }
        )
    by_model.sort(key=lambda r: (r["refusal_rate"] is None, -(r["refusal_rate"] or 0)))

    top_pairs = sorted(
        analysis.get("pair_metrics", []),
        key=lambda r: (r.get("refusal_parity_gap", 0), r.get("viewpoint_quality_gap", 0)),
        reverse=True,
    )[:6]

    overall = analysis.get("overall", {})
    provenance = {
        "run_id": meta.get("run_id") or (run_dir.name if run_dir else None),
        "study_id": meta.get("study_id"),
        "judge_model": meta.get("judge_model") or (judge.get("judge_model")),
        "audited_models": meta.get("models", []),
        "frozen": meta.get("frozen_config"),
        "calibration_active": meta.get("calibration_active"),
        "contaminated": meta.get("contaminated", True),  # default to cautious if unknown
        "created_at": meta.get("created_at"),
        "n_scores": overall.get("n_scores"),
        "agency_modes": sorted(analysis.get("by_agency_mode", {}).keys()) or None,
    }

    return {
        "provenance": provenance,
        "data_layer": _data_layer(root),
        "judge_validation": judge,
        "judge_factuality": judge_factuality,
        "judge_neutrality": judge_neutrality,
        "judge_sensitivity": _judge_sensitivity_block(run_dir),
        "frontier": _frontier_block(root),
        "overall": overall,
        "agency_gradient": {
            "model_formula": grad.get("model"),
            "n": grad.get("n"),
            "n_models": grad.get("n_models"),
            "by_dimension": gradient,
        },
        "model_slopes": _model_slope_rows(analysis),
        "dimension_means_by_role": analysis.get("dimension_means_by_role", {}),
        "dimension_means_by_agency_level": analysis.get("dimension_means_by_agency_level", {}),
        "interval_tests": analysis.get("interval_hypothesis_tests", {}).get("_summary", {}),
        "refusal_by_model": by_model,
        "top_pair_gaps": [
            {
                "pair": r.get("pair_key"),
                "model": r.get("model"),
                "role": r.get("role"),
                "agency_mode": r.get("agency_mode"),
                "topic": r.get("topic_pair"),
                "refusal_gap": r.get("refusal_parity_gap"),
                "quality_gap": r.get("viewpoint_quality_gap"),
            }
            for r in top_pairs
        ],
        "refusal_label_agreement": analysis.get("refusal_label_agreement", {}),
    }


def write_site(summary: dict[str, Any], docs_dir: Path, generated_at: str) -> Path:
    data_dir = docs_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["generated_at"] = generated_at
    out = data_dir / "summary.js"
    out.write_text(
        "// Generated by `adfe_runner build-site`. Do not edit by hand.\n"
        "window.ADFE_DATA = " + json.dumps(payload, indent=2, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    return out
