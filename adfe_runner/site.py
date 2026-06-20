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


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> dict[str, Any]:
    if n <= 0:
        return {"successes": successes, "n": n, "rate": None, "ci_low": None, "ci_high": None}
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return {
        "successes": successes,
        "n": n,
        "rate": round(p, 4),
        "ci_low": round(max(0.0, center - margin), 4),
        "ci_high": round(min(1.0, center + margin), 4),
    }


def _rate_row(label: str, successes: int | None, n: int | None, note: str) -> dict[str, Any]:
    stats = _wilson_interval(int(successes or 0), int(n or 0))
    return {"label": label, "note": note, **stats}


def _v2_statistical_checks(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    if not analysis:
        return []
    overall = analysis.get("overall", {})
    refusal = analysis.get("refusal", {})
    paired = refusal.get("paired_parity", {})
    profile = analysis.get("role_profile", {}).get("interval_support", {})
    n_scores = int(overall.get("n_scores") or 0)
    n_refusal = round((overall.get("refusal_rate") or 0) * n_scores)
    n_over = round((overall.get("over_refusal_rate") or 0) * n_scores)
    n_pairs = int(paired.get("n_pairs") or 0)
    n_one_sided = int(paired.get("one_sided_refusal_count") or 0)
    n_role_dims = int(profile.get("n_role_dimensions") or 0)
    n_supported = int(profile.get("supported") or 0)
    return [
        _rate_row("Refusal rate", n_refusal, n_scores, "All generated outputs, refusal as its own outcome."),
        _rate_row("Over-refusal rate", n_over, n_scores, "Refusals the primary judge marked unwarranted."),
        _rate_row("One-sided paired refusal", n_one_sided, n_pairs, "Mirrored prompt pairs where only one side was refused."),
        _rate_row("Role-profile interval support", n_supported, n_role_dims, "Role x dimension expectations supported by non-refusal means."),
    ]


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
    adjusted_rows = report.get("adjusted_gradient_comparison", [])
    same = [row.get("dim") for row in rows if row.get("same_sign") is True]
    different = [row.get("dim") for row in rows if row.get("same_sign") is False]
    adjusted_same = [row.get("dim") for row in adjusted_rows if row.get("same_sign") is True]
    adjusted_different = [row.get("dim") for row in adjusted_rows if row.get("same_sign") is False]
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
        "adjusted_same_slope_sign_count": report.get("adjusted_same_slope_sign_count"),
        "adjusted_same_sign_dimensions": adjusted_same,
        "adjusted_different_sign_dimensions": adjusted_different,
        "adjusted_gradient_comparison": adjusted_rows,
        "baseline_overall": report.get("baseline_overall", {}),
        "sensitivity_overall": report.get("sensitivity_overall", {}),
        "baseline_distribution": report.get("baseline_distribution", {}),
        "sensitivity_distribution": report.get("sensitivity_distribution", {}),
        "judge_score_agreement": report.get("judge_score_agreement", {}),
        "baseline_interval_summary": report.get("baseline_interval_summary", {}),
        "sensitivity_interval_summary": report.get("sensitivity_interval_summary", {}),
        "blind_role_inference": meta.get("blind_role_inference"),
        "score_json_retry": meta.get("score_json_retry"),
        "interpretation": report.get("interpretation"),
    }


def _v2_block(run_dir: Path | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    v2_dir = run_dir / "v2"
    analysis = _load_json(v2_dir / "analysis.json")
    meta = _load_json(v2_dir / "meta.json")
    if not analysis:
        return {}
    comparisons = sorted(v2_dir.glob("comparison_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    comparison = _load_json(comparisons[0]) if comparisons else {}
    refusal = analysis.get("refusal", {})
    role_profile = analysis.get("role_profile", {})
    quality = analysis.get("quality_non_refusal", {})
    return {
        "available": True,
        "meta": meta,
        "overall": analysis.get("overall", {}),
        "judge_model": analysis.get("judge_model") or meta.get("primary_judge"),
        "exploratory_same_provider": analysis.get("exploratory_same_provider") or meta.get("exploratory_same_provider"),
        "refusal_by_role": [
            {"role": role, **row}
            for role, row in sorted(refusal.get("by_role", {}).items())
        ],
        "refusal_by_model": [
            {"model": model, **row}
            for model, row in sorted(refusal.get("by_model", {}).items())
        ],
        "paired_refusal": refusal.get("paired_parity", {}),
        "quality_non_refusal_by_role": [
            {"role": role, **row}
            for role, row in sorted(quality.get("by_role", {}).items())
        ],
        "quality_non_refusal_overall": quality.get("overall", {}),
        "role_profile_by_role": [
            {"role": role, **row}
            for role, row in sorted(role_profile.get("by_role", {}).items())
        ],
        "role_profile_interval_support": role_profile.get("interval_support", {}),
        "role_confusion": role_profile.get("role_confusion", {}),
        "top_profile_mismatches": role_profile.get("top_profile_mismatches", [])[:8],
        "role_profile_design": analysis.get("role_profile_design", {}),
        "quality_agency_correlations": analysis.get("role_profile_quality_correlations", {}),
        "judge_robustness": comparison,
        "statistical_checks": _v2_statistical_checks(analysis),
    }


def _v2_frontier_block(root: Path) -> dict[str, Any]:
    """Summarize the v2 frontier arm as exploratory, not pooled evidence."""
    d = root / "runs" / "adfe_v2_frontier_grok_exploratory"
    v2_dir = d / "v2"
    analysis = _load_json(v2_dir / "analysis.json")
    meta = _load_json(v2_dir / "meta.json")
    run_meta = _load_json(d / "run_meta.json")
    if not analysis:
        return {"available": False}
    refusal = analysis.get("refusal", {})
    quality = analysis.get("quality_non_refusal", {})
    role_profile = analysis.get("role_profile", {})
    return {
        "available": True,
        "run_id": meta.get("run_id") or run_meta.get("run_id") or d.name,
        "meta": meta,
        "models": run_meta.get("models", []),
        "judge_model": analysis.get("judge_model") or meta.get("primary_judge"),
        "exploratory_same_provider": True,
        "overall": analysis.get("overall", {}),
        "paired_refusal": refusal.get("paired_parity", {}),
        "refusal_by_model": [
            {"model": model, **row}
            for model, row in sorted(refusal.get("by_model", {}).items())
        ],
        "quality_non_refusal_overall": quality.get("overall", {}),
        "quality_non_refusal_by_model": [
            {"model": model, **row}
            for model, row in sorted(quality.get("by_model", {}).items())
        ],
        "role_profile_by_role": [
            {"role": role, **row}
            for role, row in sorted(role_profile.get("by_role", {}).items())
        ],
        "role_profile_interval_support": role_profile.get("interval_support", {}),
        "statistical_checks": _v2_statistical_checks(analysis),
    }


def _compact_validation(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    metrics = report.get("metrics", {})
    return {
        "judge_model": report.get("judge_model"),
        "dataset": report.get("dataset"),
        "task": report.get("task"),
        "n": report.get("n_scored"),
        "accuracy": metrics.get("accuracy"),
        "cohen_kappa": metrics.get("cohen_kappa"),
        "safe_overflag_rate": metrics.get("safe_overflag_rate"),
        "false_answer_detection_rate": metrics.get("false_answer_detection_rate"),
        "bias_detection_rate": metrics.get("bias_detection_rate"),
        "neutral_specificity": metrics.get("neutral_specificity"),
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
    adjusted_grad = analysis.get("agency_gradient_adjusted", {})
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

    v2 = _v2_block(run_dir)
    overall = analysis.get("overall", {})
    provenance = {
        "run_id": meta.get("run_id") or (run_dir.name if run_dir else None),
        "study_id": meta.get("study_id"),
        "judge_model": meta.get("judge_model") or v2.get("judge_model") or (judge.get("judge_model")),
        "audited_models": meta.get("models", []),
        "frozen": meta.get("frozen_config"),
        "calibration_active": meta.get("calibration_active"),
        "contaminated": meta.get("contaminated", True),  # default to cautious if unknown
        "created_at": meta.get("created_at"),
        "n_scores": overall.get("n_scores") or v2.get("overall", {}).get("n_scores"),
        "agency_modes": sorted(analysis.get("by_agency_mode", {}).keys()) or None,
    }

    sensitivity = _judge_sensitivity_block(run_dir)
    if sensitivity.get("sensitivity_judge"):
        sensitivity["validation"] = {
            "safety": _compact_validation(_latest_validation(runs_base, "safety", sensitivity["sensitivity_judge"])),
            "factuality": _compact_validation(_latest_validation(runs_base, "factuality", sensitivity["sensitivity_judge"])),
            "neutrality": _compact_validation(_latest_validation(runs_base, "neutrality", sensitivity["sensitivity_judge"])),
        }

    return {
        "provenance": provenance,
        "data_layer": _data_layer(root),
        "judge_validation": judge,
        "judge_factuality": judge_factuality,
        "judge_neutrality": judge_neutrality,
        "judge_sensitivity": sensitivity,
        "v2": v2,
        "v2_frontier": _v2_frontier_block(root),
        "frontier": _frontier_block(root),
        "overall": overall,
        "refusal_mediation": analysis.get("refusal_mediation", {}),
        "score_distribution_diagnostics": analysis.get("score_distribution_diagnostics", {}),
        "role_profile_design": analysis.get("role_profile_design", {}),
        "agency_gradient": {
            "model_formula": grad.get("model"),
            "n": grad.get("n"),
            "n_models": grad.get("n_models"),
            "by_dimension": gradient,
        },
        "agency_gradient_adjusted": {
            "model_formula": adjusted_grad.get("model"),
            "n": adjusted_grad.get("n"),
            "n_models": adjusted_grad.get("n_models"),
            "refusal_rate": adjusted_grad.get("refusal_rate"),
            "by_dimension": [
                {
                    "dim": dim,
                    "coef": adjusted_grad.get("by_dimension", {}).get(dim, {}).get("agency_level_coef"),
                    "pvalue": adjusted_grad.get("by_dimension", {}).get(dim, {}).get("agency_level_pvalue"),
                    "significant": adjusted_grad.get("by_dimension", {}).get(dim, {}).get("agency_level_significant_0_05"),
                    "refusal_coef": adjusted_grad.get("by_dimension", {}).get(dim, {}).get("refusal_coef"),
                    "refusal_pvalue": adjusted_grad.get("by_dimension", {}).get(dim, {}).get("refusal_pvalue"),
                    "converged": adjusted_grad.get("by_dimension", {}).get(dim, {}).get("converged"),
                    "error": adjusted_grad.get("by_dimension", {}).get(dim, {}).get("error"),
                }
                for dim in DIMENSIONS
            ],
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
    from .v2_figures import generate_v2_figures

    data_dir = docs_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["figures"] = {"v2": generate_v2_figures(payload, docs_dir)}
    payload["generated_at"] = generated_at
    out = data_dir / "summary.js"
    out.write_text(
        "// Generated by `adfe_runner build-site`. Do not edit by hand.\n"
        "window.ADFE_DATA = " + json.dumps(payload, indent=2, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    return out
