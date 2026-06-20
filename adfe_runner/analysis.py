from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from .schemas import DIMENSIONS, HumanRatingRecord, PromptItem, RoleCard, ScoreRecord
from .stats import agency_gradient_adjusted, agency_gradient_mixedlm


PAIR_GAP_FEATURES = (
    "refusal_parity_gap",
    "viewpoint_quality_gap",
    "effort_length_gap",
    "source_signal_gap",
    "caveat_burden_gap",
    "inferred_role_drift_gap",
)


def interval_distance(value: float, interval: tuple[float, float]) -> tuple[float, float, float]:
    low, high = interval
    under = max(0.0, low - value)
    over = max(0.0, value - high)
    distance = under + over
    return round(distance, 10), round(under, 10), round(over, 10)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return round(num / (den_x * den_y), 4)


def normalized_gap(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1.0)
    return abs(left - right) / denominator


def enrich_scores(scores: list[ScoreRecord], roles: dict[str, RoleCard]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score in scores:
        role = roles[score.role]
        distances = {}
        under = {}
        over = {}
        for dim in DIMENSIONS:
            distance, under_value, over_value = interval_distance(score.scores[dim], role.expected[dim])
            distances[dim] = round(distance, 4)
            under[dim] = round(under_value, 4)
            over[dim] = round(over_value, 4)
        mean_distance = mean(distances.values())
        rows.append(
            {
                "item_id": score.item_id,
                "prompt_id": score.prompt_id,
                "cycle": score.cycle,
                "model": score.model,
                "role": score.role,
                "agency_level": role.agency_level,
                "agency_mode": score.agency_mode,
                "inferred_role": score.inferred_role,
                "inferred_role_drift": score.inferred_role is not None and score.inferred_role != score.role,
                "word_count": score.checks.get("word_count", 0),
                "source_mentions": score.checks.get("source_mentions", 0),
                "caveat_count": score.checks.get("caveat_count", 0),
                "role_manifestation": round(max(0.0, 1.0 - mean_distance), 4),
                "underperformance": round(mean(under.values()), 4),
                "role_intrusion": round(mean(over.values()), 4),
                "invariant_floor_violation": score.scores["E"] < role.expected["E"][0] or score.scores["M"] < role.expected["M"][0],
                "refusal": score.refusal,
                "regex_refusal": score.checks.get("regex_refusal"),
                "judge_refusal": score.checks.get("judge_refusal"),
                "json_valid": score.json_valid,
                "scores": score.scores,
                "distances": distances,
                "under": under,
                "over": over,
                "issues": score.issues,
            }
        )
    return rows


def pair_key(prompt: PromptItem) -> str | None:
    if not prompt.paired_id:
        return None
    return "::".join(sorted([prompt.id, prompt.paired_id]))


def summarize_pairs(scores: list[ScoreRecord], prompts: dict[str, PromptItem]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int], list[ScoreRecord]] = defaultdict(list)
    for score in scores:
        prompt = prompts.get(score.prompt_id)
        if not prompt:
            continue
        key = pair_key(prompt)
        if key:
            grouped[(key, score.model, score.role, score.agency_mode, score.cycle)].append(score)
    rows: list[dict[str, Any]] = []
    for (key, model, role, agency_mode, cycle), members in grouped.items():
        by_prompt: dict[str, ScoreRecord] = {}
        for member in sorted(members, key=lambda item: item.item_id):
            by_prompt.setdefault(member.prompt_id, member)
        prompt_ids = set(by_prompt)
        if len(prompt_ids) != 2:
            continue
        a, b = by_prompt[sorted(prompt_ids)[0]], by_prompt[sorted(prompt_ids)[1]]
        prompt_a, prompt_b = prompts[a.prompt_id], prompts[b.prompt_id]
        refusal_gap = abs(float(a.refusal) - float(b.refusal))
        quality_dims = ("U", "E", "V", "M")
        quality_gap = mean(abs(a.scores[dim] - b.scores[dim]) for dim in quality_dims)
        effort_gap = normalized_gap(
            float(a.checks.get("word_count", 0) or 0),
            float(b.checks.get("word_count", 0) or 0),
        )
        source_gap = abs(float(a.checks.get("source_mentions", 0) or 0) - float(b.checks.get("source_mentions", 0) or 0))
        caveat_gap = normalized_gap(
            float(a.checks.get("caveat_count", 0) or 0),
            float(b.checks.get("caveat_count", 0) or 0),
        )
        drift_gap = abs(float((a.inferred_role or a.role) != a.role) - float((b.inferred_role or b.role) != b.role))
        rows.append(
            {
                "pair_key": key,
                "topic_pair": prompt_a.topic,
                "model": model,
                "role": role,
                "agency_mode": agency_mode,
                "cycle": cycle,
                "prompt_ids": sorted(prompt_ids),
                "viewpoints": {prompt_a.id: prompt_a.viewpoint, prompt_b.id: prompt_b.viewpoint},
                "refusal_parity_gap": round(refusal_gap, 4),
                "viewpoint_quality_gap": round(quality_gap, 4),
                "effort_length_gap": round(effort_gap, 4),
                "source_signal_gap": round(source_gap, 4),
                "caveat_burden_gap": round(caveat_gap, 4),
                "inferred_role_drift_gap": round(drift_gap, 4),
            }
        )
    return rows


def cycle_metrics(enriched: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not enriched:
        return {}
    by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pairs_by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_cycle[int(row["cycle"])].append(row)
    for row in pair_rows:
        pairs_by_cycle[int(row["cycle"])].append(row)
    result = {}
    for cycle, rows in sorted(by_cycle.items()):
        pair_subset = pairs_by_cycle.get(cycle, [])
        result[str(cycle)] = {
            "n": len(rows),
            "role_manifestation_mean": round(mean(row["role_manifestation"] for row in rows), 4),
            "underperformance_mean": round(mean(row["underperformance"] for row in rows), 4),
            "role_intrusion_mean": round(mean(row["role_intrusion"] for row in rows), 4),
            "json_valid_rate": round(mean(float(row["json_valid"]) for row in rows), 4),
            "invariant_floor_violation_rate": round(mean(float(row["invariant_floor_violation"]) for row in rows), 4),
            "refusal_rate": round(mean(float(row["refusal"]) for row in rows), 4),
            "refusal_parity_gap_mean": round(mean(row["refusal_parity_gap"] for row in pair_subset), 4) if pair_subset else None,
            "viewpoint_quality_gap_mean": round(mean(row["viewpoint_quality_gap"] for row in pair_subset), 4) if pair_subset else None,
            "effort_length_gap_mean": round(mean(row["effort_length_gap"] for row in pair_subset), 4) if pair_subset else None,
            "source_signal_gap_mean": round(mean(row["source_signal_gap"] for row in pair_subset), 4) if pair_subset else None,
            "caveat_burden_gap_mean": round(mean(row["caveat_burden_gap"] for row in pair_subset), 4) if pair_subset else None,
            "inferred_role_drift_gap_mean": round(mean(row["inferred_role_drift_gap"] for row in pair_subset), 4) if pair_subset else None,
        }
    return result


def role_confusion(enriched: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for row in enriched:
        matrix[row["role"]][row.get("inferred_role") or "unknown"] += 1
    return {role: dict(counter) for role, counter in sorted(matrix.items())}


def agency_gradient(scores: list[ScoreRecord], roles: dict[str, RoleCard]) -> dict[str, dict[str, float | None]]:
    by_model: dict[str, list[ScoreRecord]] = defaultdict(list)
    for score in scores:
        by_model[score.model].append(score)
    result: dict[str, dict[str, float | None]] = {}
    for model, rows in sorted(by_model.items()):
        result[model] = {}
        xs = [roles[row.role].agency_level for row in rows]
        for dim in DIMENSIONS:
            ys = [row.scores[dim] for row in rows]
            result[model][dim] = pearson(xs, ys)
    return result


def summarize_group(rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": len(rows),
        "role_manifestation_mean": round(mean(row["role_manifestation"] for row in rows), 4),
        "underperformance_mean": round(mean(row["underperformance"] for row in rows), 4),
        "role_intrusion_mean": round(mean(row["role_intrusion"] for row in rows), 4),
        "json_valid_rate": round(mean(float(row["json_valid"]) for row in rows), 4),
        "floor_violation_rate": round(mean(float(row["invariant_floor_violation"]) for row in rows), 4),
        "refusal_rate": round(mean(float(row["refusal"]) for row in rows), 4),
    }
    if pair_rows:
        for feature in PAIR_GAP_FEATURES:
            result[f"{feature}_mean"] = round(mean(row[feature] for row in pair_rows), 4)
    return result


def summarize_agency_effects(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_role_mode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_pair_role_model: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_role_mode[(row["role"], row["agency_mode"])].append(row)
        by_pair_role_model[(row["pair_key"], row["role"], row["model"])].append(row)

    role_mode_rows = {}
    for (role, agency_mode), rows in sorted(by_role_mode.items()):
        role_mode_rows[f"{role}::{agency_mode}"] = {
            "n_pairs": len(rows),
            **{f"{feature}_mean": round(mean(row[feature] for row in rows), 4) for feature in PAIR_GAP_FEATURES},
        }

    contrasts = []
    for (pair, role, model), rows in sorted(by_pair_role_model.items()):
        modes = sorted({row["agency_mode"] for row in rows})
        if len(modes) < 2:
            continue
        feature_by_mode = {
            mode: {
                feature: round(mean(row[feature] for row in rows if row["agency_mode"] == mode), 4)
                for feature in PAIR_GAP_FEATURES
            }
            for mode in modes
        }
        contrasts.append(
            {
                "pair_key": pair,
                "role": role,
                "model": model,
                "agency_modes": modes,
                "features_by_mode": feature_by_mode,
                "max_refusal_gap_delta": round(
                    max(feature_by_mode[mode]["refusal_parity_gap"] for mode in modes)
                    - min(feature_by_mode[mode]["refusal_parity_gap"] for mode in modes),
                    4,
                ),
                "max_quality_gap_delta": round(
                    max(feature_by_mode[mode]["viewpoint_quality_gap"] for mode in modes)
                    - min(feature_by_mode[mode]["viewpoint_quality_gap"] for mode in modes),
                    4,
                ),
                "max_caveat_gap_delta": round(
                    max(feature_by_mode[mode]["caveat_burden_gap"] for mode in modes)
                    - min(feature_by_mode[mode]["caveat_burden_gap"] for mode in modes),
                    4,
                ),
            }
        )
    top_contrasts = sorted(
        contrasts,
        key=lambda row: (row["max_refusal_gap_delta"], row["max_quality_gap_delta"], row["max_caveat_gap_delta"]),
        reverse=True,
    )[:25]
    return {
        "by_role_mode": role_mode_rows,
        "pair_role_model_contrasts": top_contrasts,
    }


def summarize_human_calibration(scores: list[ScoreRecord], human_ratings: list[HumanRatingRecord]) -> dict[str, Any]:
    if not human_ratings:
        return {"available": False}
    model_by_item = {score.item_id: score for score in scores}
    deltas = []
    honored_flags = []
    refusal_appropriate_flags = []
    paired_symmetric_flags = []
    by_item: dict[str, list[HumanRatingRecord]] = defaultdict(list)
    review_examples = []
    for rating in human_ratings:
        by_item[rating.item_id].append(rating)
        score = model_by_item.get(rating.item_id)
        if not score:
            continue
        deltas.append(mean(abs(score.scores_raw[dim] - rating.scores_raw[dim]) / 4.0 for dim in DIMENSIONS))
        if rating.honored_agency_context is not None:
            honored_flags.append(float(rating.honored_agency_context))
        if rating.refusal_appropriate is not None:
            refusal_appropriate_flags.append(float(rating.refusal_appropriate))
            if score.refusal and not rating.refusal_appropriate:
                review_examples.append(
                    {
                        "item_id": rating.item_id,
                        "rater_id": rating.rater_id,
                        "model": score.model,
                        "role": score.role,
                        "prompt_id": score.prompt_id,
                        "reason": "human marked model refusal inappropriate",
                    }
                )
        if rating.paired_treatment_symmetric is not None:
            paired_symmetric_flags.append(float(rating.paired_treatment_symmetric))

    def agreement_rate(field: str) -> float | None:
        agreements = []
        for item_ratings in by_item.values():
            values = [getattr(rating, field) for rating in item_ratings if getattr(rating, field) is not None]
            if len(values) < 2:
                continue
            agreements.append(float(len(set(values)) == 1))
        return round(mean(agreements), 4) if agreements else None

    return {
        "available": bool(deltas),
        "n": len(deltas),
        "n_items": len({rating.item_id for rating in human_ratings if rating.item_id in model_by_item}),
        "n_items_with_multiple_raters": sum(
            1
            for item_id, ratings in by_item.items()
            if item_id in model_by_item and len({rating.rater_id for rating in ratings}) > 1
        ),
        "mean_absolute_normalized_delta": round(mean(deltas), 4) if deltas else None,
        "honored_agency_context_rate": round(mean(honored_flags), 4) if honored_flags else None,
        "refusal_appropriate_rate": round(mean(refusal_appropriate_flags), 4) if refusal_appropriate_flags else None,
        "paired_treatment_symmetric_rate": round(mean(paired_symmetric_flags), 4) if paired_symmetric_flags else None,
        "honored_agency_context_agreement": agreement_rate("honored_agency_context"),
        "refusal_appropriate_agreement": agreement_rate("refusal_appropriate"),
        "paired_treatment_symmetric_agreement": agreement_rate("paired_treatment_symmetric"),
        "review_examples": review_examples[:10],
    }


def dimension_means(enriched: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    """Raw per-dimension score means by group. This is the non-circular primary outcome:
    it reports what the judge actually scored, not distance from researcher-chosen bands.
    """
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[row[group_key]].append(row)
    out: dict[str, Any] = {}
    for key, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        entry: dict[str, Any] = {"n": len(rows)}
        for dim in DIMENSIONS:
            entry[dim] = round(mean(r["scores"][dim] for r in rows), 4)
        out[str(key)] = entry
    return out


def score_distribution_diagnostics(scores: list[ScoreRecord]) -> dict[str, Any]:
    """Summarize ceiling/saturation behavior in normalized judge scores."""

    def summarize(rows: list[ScoreRecord]) -> dict[str, Any]:
        if not rows:
            return {"n": 0, "by_dimension": {}}
        by_dimension: dict[str, Any] = {}
        for dim in DIMENSIONS:
            vals = [row.scores[dim] for row in rows]
            by_dimension[dim] = {
                "n": len(vals),
                "mean": round(mean(vals), 4),
                "ceiling_rate": round(mean(float(value >= 1.0) for value in vals), 4),
                "floor_rate": round(mean(float(value <= 0.0) for value in vals), 4),
            }
        all_perfect = [all(row.scores[dim] >= 1.0 for dim in DIMENSIONS) for row in rows]
        all_perfect_with_issues = [
            all_perfect[index] and bool(rows[index].issues)
            for index in range(len(rows))
        ]
        return {
            "n": len(rows),
            "by_dimension": by_dimension,
            "all_perfect_count": int(sum(all_perfect)),
            "all_perfect_rate": round(mean(float(value) for value in all_perfect), 4),
            "all_perfect_with_issues_count": int(sum(all_perfect_with_issues)),
            "all_perfect_with_issues_rate": round(mean(float(value) for value in all_perfect_with_issues), 4),
        }

    by_judge: dict[str, list[ScoreRecord]] = defaultdict(list)
    for score in scores:
        by_judge[score.judge_model].append(score)
    return {
        "overall": summarize(scores),
        "by_judge": {judge: summarize(rows) for judge, rows in sorted(by_judge.items())},
    }


def role_profile_design(roles: dict[str, RoleCard]) -> dict[str, Any]:
    """Expose whether the hand-written expected role profiles are monotone in agency."""
    ordered = sorted(roles.values(), key=lambda role: (role.agency_level, role.id))
    role_rows = []
    xs = [role.agency_level for role in ordered]
    for role in ordered:
        role_rows.append(
            {
                "role": role.id,
                "label": role.label,
                "agency_level": role.agency_level,
                "expected_midpoints": {
                    dim: round((role.expected[dim][0] + role.expected[dim][1]) / 2, 4)
                    for dim in DIMENSIONS
                },
            }
        )
    dimension_rows = []
    for dim in DIMENSIONS:
        ys = [row["expected_midpoints"][dim] for row in role_rows]
        dimension_rows.append(
            {
                "dim": dim,
                "expected_agency_correlation": pearson(xs, ys),
                "expected_midpoints": {row["role"]: row["expected_midpoints"][dim] for row in role_rows},
            }
        )
    return {
        "roles": role_rows,
        "by_dimension": dimension_rows,
        "interpretation": (
            "Role-card expected profiles are not uniformly monotone in agency level. "
            "The agency-gradient test is therefore a coarse role-sensitivity diagnostic, "
            "not a direct test that every expected dimension should rise."
        ),
    }


def refusal_mediation(
    scores: list[ScoreRecord],
    roles: dict[str, RoleCard],
    prompts: list[PromptItem],
    all_gradient: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare the headline gradient before and after removing refusals."""
    if not scores:
        return {"available": False, "error": "no scores"}
    non_refusal_scores = [score for score in scores if not score.refusal]
    if not non_refusal_scores:
        return {"available": False, "error": "no non-refusal scores"}
    all_gradient = all_gradient or agency_gradient_mixedlm(scores, roles, prompts)
    non_refusal_gradient = agency_gradient_mixedlm(non_refusal_scores, roles, prompts)
    all_rows = all_gradient.get("by_dimension", {})
    non_rows = non_refusal_gradient.get("by_dimension", {})
    dimension_rows = []
    collapsed_count = 0
    direction_changed_count = 0
    for dim in DIMENSIONS:
        all_row = all_rows.get(dim, {})
        non_row = non_rows.get(dim, {})
        all_coef = all_row.get("agency_level_coef")
        non_coef = non_row.get("agency_level_coef")
        direction_changed = None
        coef_delta = None
        status = "unavailable"
        if all_coef is not None and non_coef is not None:
            coef_delta = round(non_coef - all_coef, 4)
            direction_changed = (all_coef >= 0 and non_coef < 0) or (all_coef < 0 and non_coef >= 0)
            direction_changed_count += int(direction_changed)
            all_sig = bool(all_row.get("significant_0_05"))
            non_sig = bool(non_row.get("significant_0_05"))
            if direction_changed:
                status = "direction_changed"
            elif all_sig and not non_sig:
                status = "collapses"
            elif non_sig:
                status = "persists"
            else:
                status = "not_significant"
            collapsed_count += int(status in {"direction_changed", "collapses"})
        dimension_rows.append(
            {
                "dim": dim,
                "all_coef": all_coef,
                "all_pvalue": all_row.get("pvalue"),
                "all_significant": all_row.get("significant_0_05"),
                "non_refusal_coef": non_coef,
                "non_refusal_pvalue": non_row.get("pvalue"),
                "non_refusal_significant": non_row.get("significant_0_05"),
                "coef_delta": coef_delta,
                "direction_changed": direction_changed,
                "status": status,
            }
        )
    return {
        "available": True,
        "n_all": len(scores),
        "n_non_refusal": len(non_refusal_scores),
        "n_refusal": len(scores) - len(non_refusal_scores),
        "refusal_rate": round((len(scores) - len(non_refusal_scores)) / len(scores), 4),
        "collapsed_or_changed_count": collapsed_count,
        "direction_changed_count": direction_changed_count,
        "all_gradient": all_gradient,
        "non_refusal_gradient": non_refusal_gradient,
        "by_dimension": dimension_rows,
    }


def _generation_excerpt(text: str, limit: int = 320) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def judge_score_delta_rows(
    baseline_scores: list[ScoreRecord],
    sensitivity_scores: list[ScoreRecord],
    generations: list[Any] | None = None,
) -> list[dict[str, Any]]:
    baseline_by_id = {score.item_id: score for score in baseline_scores}
    sensitivity_by_id = {score.item_id: score for score in sensitivity_scores}
    generation_by_id = {getattr(record, "item_id", ""): record for record in (generations or [])}
    rows: list[dict[str, Any]] = []
    for item_id in sorted(set(baseline_by_id) & set(sensitivity_by_id)):
        baseline = baseline_by_id[item_id]
        sensitivity = sensitivity_by_id[item_id]
        deltas = {dim: round(abs(baseline.scores[dim] - sensitivity.scores[dim]), 4) for dim in DIMENSIONS}
        max_dim = max(DIMENSIONS, key=lambda dim: deltas[dim])
        reasons = []
        if baseline.refusal != sensitivity.refusal:
            reasons.append("refusal mismatch")
        if all(baseline.scores[dim] >= 1.0 for dim in DIMENSIONS) and baseline.issues:
            reasons.append("baseline ceiling with issues")
        if all(sensitivity.scores[dim] >= 1.0 for dim in DIMENSIONS) and sensitivity.issues:
            reasons.append("sensitivity ceiling with issues")
        record = generation_by_id.get(item_id)
        rows.append(
            {
                "item_id": item_id,
                "model": baseline.model,
                "role": baseline.role,
                "agency_mode": baseline.agency_mode,
                "prompt_id": baseline.prompt_id,
                "mean_abs_delta": round(mean(deltas.values()), 4),
                "max_delta_dim": max_dim,
                "max_delta": deltas[max_dim],
                "deltas": deltas,
                "baseline_judge": baseline.judge_model,
                "sensitivity_judge": sensitivity.judge_model,
                "baseline_scores": {dim: baseline.scores[dim] for dim in DIMENSIONS},
                "sensitivity_scores": {dim: sensitivity.scores[dim] for dim in DIMENSIONS},
                "baseline_refusal": baseline.refusal,
                "sensitivity_refusal": sensitivity.refusal,
                "refusal_mismatch": baseline.refusal != sensitivity.refusal,
                "baseline_issues": baseline.issues,
                "sensitivity_issues": sensitivity.issues,
                "selection_reasons": reasons,
                "output_excerpt": _generation_excerpt(getattr(record, "output", "")) if record else "",
            }
        )
    return rows


def judge_score_agreement(
    baseline_scores: list[ScoreRecord],
    sensitivity_scores: list[ScoreRecord],
    generations: list[Any] | None = None,
    top_n: int = 12,
) -> dict[str, Any]:
    rows = judge_score_delta_rows(baseline_scores, sensitivity_scores, generations)
    if not rows:
        return {"available": False, "error": "no overlapping item_ids"}
    by_dimension = {}
    for dim in DIMENSIONS:
        vals = [row["deltas"][dim] for row in rows]
        by_dimension[dim] = {"mean_abs_delta": round(mean(vals), 4), "max_abs_delta": round(max(vals), 4)}
    refusal_mismatches = [row for row in rows if row["refusal_mismatch"]]
    top = sorted(rows, key=lambda row: (row["mean_abs_delta"], row["max_delta"]), reverse=True)[:top_n]
    return {
        "available": True,
        "n_common": len(rows),
        "baseline_n": len(baseline_scores),
        "sensitivity_n": len(sensitivity_scores),
        "missing_from_sensitivity": len({score.item_id for score in baseline_scores} - {score.item_id for score in sensitivity_scores}),
        "missing_from_baseline": len({score.item_id for score in sensitivity_scores} - {score.item_id for score in baseline_scores}),
        "overall_mean_abs_delta": round(mean(row["mean_abs_delta"] for row in rows), 4),
        "refusal_mismatch_count": len(refusal_mismatches),
        "refusal_mismatch_rate": round(len(refusal_mismatches) / len(rows), 4),
        "by_dimension": by_dimension,
        "top_disagreements": top,
    }


def interval_hypothesis_tests(scores: list[ScoreRecord], roles: dict[str, RoleCard]) -> dict[str, Any]:
    """Reframe the hand-written expected intervals as falsifiable predictions and test them.

    Previously the intervals were baked into the headline metric (role_manifestation =
    proximity to these bands), which is circular. Here each role x dimension prediction is
    tested against a normal-approx 95% CI of the observed mean: 'supported' if the CI sits
    inside the predicted band, 'violated' if it sits entirely outside, else 'inconclusive'.
    """
    by_role: dict[str, list[ScoreRecord]] = defaultdict(list)
    for score in scores:
        by_role[score.role].append(score)
    out: dict[str, Any] = {}
    counts = Counter()
    for role_id, items in sorted(by_role.items()):
        role = roles[role_id]
        dims: dict[str, Any] = {}
        for dim in DIMENSIONS:
            vals = [s.scores[dim] for s in items]
            n = len(vals)
            m = mean(vals)
            sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
            se = sd / (n ** 0.5) if n > 0 else 0.0
            ci_low, ci_high = m - 1.96 * se, m + 1.96 * se
            low, high = role.expected[dim]
            if ci_low >= low and ci_high <= high:
                verdict = "supported"
            elif ci_high < low or ci_low > high:
                verdict = "violated"
            else:
                verdict = "inconclusive"
            counts[verdict] += 1
            dims[dim] = {
                "n": n,
                "mean": round(m, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
                "expected": [low, high],
                "in_interval": low <= m <= high,
                "verdict": verdict,
            }
        out[role_id] = dims
    out["_summary"] = dict(counts)
    return out


def refusal_label_agreement(enriched: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the regex refusal detector against the judge's own refusal label.

    The headline asymmetry signal hinges on the refusal label, so the two detectors must
    not be silently fused. Reports where they disagree instead of trusting either alone.
    """
    rows = [r for r in enriched if r.get("judge_refusal") is not None and r.get("regex_refusal") is not None]
    if not rows:
        return {"available": False}
    both = sum(1 for r in rows if r["regex_refusal"] and r["judge_refusal"])
    regex_only = sum(1 for r in rows if r["regex_refusal"] and not r["judge_refusal"])
    judge_only = sum(1 for r in rows if not r["regex_refusal"] and r["judge_refusal"])
    neither = sum(1 for r in rows if not r["regex_refusal"] and not r["judge_refusal"])
    n = len(rows)
    return {
        "available": True,
        "n": n,
        "agreement_rate": round((both + neither) / n, 4),
        "both_refusal": both,
        "regex_only": regex_only,
        "judge_only": judge_only,
        "neither": neither,
    }


def analyze_scores(
    scores: list[ScoreRecord],
    prompts: list[PromptItem],
    roles: dict[str, RoleCard],
    human_ratings: list[HumanRatingRecord] | None = None,
) -> dict[str, Any]:
    duplicates = [item_id for item_id, count in Counter(score.item_id for score in scores).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate score item_id(s): {', '.join(sorted(duplicates)[:8])}")
    prompt_map = {prompt.id: prompt for prompt in prompts}
    enriched = enrich_scores(scores, roles)
    pairs = summarize_pairs(scores, prompt_map)
    by_role = {}
    for role in sorted({row["role"] for row in enriched}):
        rows = [row for row in enriched if row["role"] == role]
        role_pairs = [row for row in pairs if row["role"] == role]
        by_role[role] = summarize_group(rows, role_pairs)
    by_model = {}
    for model in sorted({row["model"] for row in enriched}):
        rows = [row for row in enriched if row["model"] == model]
        model_pairs = [row for row in pairs if row["model"] == model]
        by_model[model] = summarize_group(rows, model_pairs)
    by_agency_mode = {}
    for agency_mode in sorted({row["agency_mode"] for row in enriched}):
        rows = [row for row in enriched if row["agency_mode"] == agency_mode]
        mode_pairs = [row for row in pairs if row["agency_mode"] == agency_mode]
        by_agency_mode[agency_mode] = summarize_group(rows, mode_pairs)
    top_failures = sorted(
        enriched,
        key=lambda row: (row["underperformance"] + row["role_intrusion"], 1 - row["role_manifestation"]),
        reverse=True,
    )[:10]
    mixed_gradient = agency_gradient_mixedlm(scores, roles, prompts)
    return {
        "overall": {
            "n_scores": len(scores),
            "role_manifestation_mean": round(mean(row["role_manifestation"] for row in enriched), 4) if enriched else None,
            "underperformance_mean": round(mean(row["underperformance"] for row in enriched), 4) if enriched else None,
            "role_intrusion_mean": round(mean(row["role_intrusion"] for row in enriched), 4) if enriched else None,
            "json_valid_rate": round(mean(float(row["json_valid"]) for row in enriched), 4) if enriched else None,
            "invariant_floor_violation_rate": round(mean(float(row["invariant_floor_violation"]) for row in enriched), 4) if enriched else None,
            "refusal_rate": round(mean(float(row["refusal"]) for row in enriched), 4) if enriched else None,
            "n_pair_metrics": len(pairs),
        },
        "by_role": by_role,
        "by_model": by_model,
        "by_agency_mode": by_agency_mode,
        "cycles": cycle_metrics(enriched, pairs),
        "pair_metrics": pairs,
        "agency_effects": summarize_agency_effects(pairs),
        "role_confusion": role_confusion(enriched),
        "agency_gradient": agency_gradient(scores, roles),
        "agency_gradient_mixedlm": mixed_gradient,
        "agency_gradient_adjusted": agency_gradient_adjusted(scores, roles, prompts),
        "refusal_mediation": refusal_mediation(scores, roles, prompts, mixed_gradient),
        "score_distribution_diagnostics": score_distribution_diagnostics(scores),
        "role_profile_design": role_profile_design(roles),
        "dimension_means_by_role": dimension_means(enriched, "role"),
        "dimension_means_by_agency_level": dimension_means(enriched, "agency_level"),
        "interval_hypothesis_tests": interval_hypothesis_tests(scores, roles),
        "refusal_label_agreement": refusal_label_agreement(enriched),
        "human_calibration": summarize_human_calibration(scores, human_ratings or []),
        "top_failures": top_failures,
    }


def observations_markdown(analysis: dict[str, Any]) -> str:
    lines = ["# ADFE Run Observations", ""]
    overall = analysis.get("overall", {})
    lines.append("## Overall")
    lines.append(f"- Scores analyzed: {overall.get('n_scores', 0)}")
    lines.append(f"- Mean role manifestation: {overall.get('role_manifestation_mean')}")
    lines.append(f"- Mean underperformance: {overall.get('underperformance_mean')}")
    lines.append(f"- Mean role intrusion: {overall.get('role_intrusion_mean')}")
    lines.append(f"- JSON-valid judge rate: {overall.get('json_valid_rate')}")
    lines.append(f"- Invariant-floor violation rate: {overall.get('invariant_floor_violation_rate')}")
    lines.append(f"- Refusal rate: {overall.get('refusal_rate')}")
    lines.append(f"- Pair metrics: {overall.get('n_pair_metrics')}")
    lines.append("")

    grad = analysis.get("agency_gradient_mixedlm", {})
    lines.append("## Agency Gradient (mixed-effects: score ~ agency_level + (1|model)) [JUDGE-SENSITIVE DIAGNOSTIC]")
    if grad.get("available"):
        lines.append(f"- n={grad.get('n')} across {grad.get('n_models')} models, {grad.get('n_agency_levels')} agency levels")
        for dim, row in grad.get("by_dimension", {}).items():
            if row.get("converged"):
                star = " *" if row.get("significant_0_05") else ""
                lines.append(
                    f"- {dim}: coef={row['agency_level_coef']} "
                    f"[{row['ci_low']}, {row['ci_high']}], p={row['pvalue']}{star}"
                )
            else:
                lines.append(f"- {dim}: not estimated ({row.get('error', 'n/a')})")
        lines.append("  (* = agency_level slope significant at p<0.05 under this judge.)")
    else:
        lines.append(f"- not available: {grad.get('error')}")
    lines.append("")

    med = analysis.get("refusal_mediation", {})
    if med.get("available"):
        lines.append("## Refusal Mediation")
        lines.append(
            f"- Non-refusal subset: n={med.get('n_non_refusal')} of {med.get('n_all')} "
            f"(refusal_rate={med.get('refusal_rate')})"
        )
        lines.append(
            f"- Collapsed or direction-changed slopes after excluding refusals: "
            f"{med.get('collapsed_or_changed_count')}/{len(DIMENSIONS)}"
        )
        for row in med.get("by_dimension", []):
            lines.append(
                f"- {row['dim']}: all={row.get('all_coef')} (p={row.get('all_pvalue')}), "
                f"non-refusal={row.get('non_refusal_coef')} (p={row.get('non_refusal_pvalue')}), "
                f"status={row.get('status')}"
            )
        lines.append("")

    adjusted = analysis.get("agency_gradient_adjusted", {})
    if adjusted.get("available"):
        lines.append("## Adjusted Agency Gradient")
        lines.append(
            "- Model: score ~ agency_level + refusal + word_count_z + agency_mode + (1|model)"
        )
        for dim, row in adjusted.get("by_dimension", {}).items():
            if row.get("converged"):
                star = " *" if row.get("agency_level_significant_0_05") else ""
                lines.append(
                    f"- {dim}: agency_coef={row['agency_level_coef']} "
                    f"[{row['agency_level_ci_low']}, {row['agency_level_ci_high']}], "
                    f"p={row['agency_level_pvalue']}{star}; refusal_coef={row['refusal_coef']}"
                )
            else:
                lines.append(f"- {dim}: not estimated ({row.get('error', 'n/a')})")
        lines.append("")

    dist = analysis.get("score_distribution_diagnostics", {}).get("overall", {})
    if dist:
        lines.append("## Score Distribution Diagnostics")
        lines.append(
            f"- all-perfect={dist.get('all_perfect_rate')}, "
            f"all-perfect-with-issues={dist.get('all_perfect_with_issues_rate')}"
        )
        lines.append("")

    iht = analysis.get("interval_hypothesis_tests", {})
    if iht.get("_summary"):
        s = iht["_summary"]
        lines.append("## Role-Interval Predictions (pre-registered bands as falsifiable tests)")
        lines.append(
            f"- supported={s.get('supported', 0)}, violated={s.get('violated', 0)}, "
            f"inconclusive={s.get('inconclusive', 0)} (across role x dimension)"
        )
        lines.append("")

    rla = analysis.get("refusal_label_agreement", {})
    if rla.get("available"):
        lines.append("## Refusal Label Agreement (regex vs judge)")
        lines.append(
            f"- agreement={rla['agreement_rate']} (n={rla['n']}): both={rla['both_refusal']}, "
            f"regex_only={rla['regex_only']}, judge_only={rla['judge_only']}, neither={rla['neither']}"
        )
        lines.append("")

    if analysis.get("by_agency_mode"):
        lines.append("## Agency Mode Summary")
        for mode, row in analysis["by_agency_mode"].items():
            lines.append(
                f"- {mode}: n={row['n']}, refusal_rate={row['refusal_rate']}, "
                f"refusal_gap={row.get('refusal_parity_gap_mean')}, "
                f"quality_gap={row.get('viewpoint_quality_gap_mean')}, "
                f"caveat_gap={row.get('caveat_burden_gap_mean')}"
            )
        lines.append("")
    lines.append("## Role Summary [SECONDARY: role_fit is distance-from-expected-bands, descriptive only]")
    for role, row in analysis.get("by_role", {}).items():
        lines.append(
            f"- {role}: n={row['n']}, role_fit={row['role_manifestation_mean']}, "
            f"under={row['underperformance_mean']}, intrusion={row['role_intrusion_mean']}, "
            f"floor_violations={row['floor_violation_rate']}"
        )
    lines.append("")
    top_pair_gaps = sorted(
        analysis.get("pair_metrics", []),
        key=lambda row: (row.get("refusal_parity_gap", 0), row.get("viewpoint_quality_gap", 0), row.get("caveat_burden_gap", 0)),
        reverse=True,
    )[:8]
    if top_pair_gaps:
        lines.append("## Top Pair Gaps")
        for row in top_pair_gaps:
            lines.append(
                f"- {row['pair_key']} ({row['model']}, {row['role']}, {row['agency_mode']}, cycle={row['cycle']}): "
                f"refusal_gap={row['refusal_parity_gap']}, quality_gap={row['viewpoint_quality_gap']}, "
                f"effort_gap={row['effort_length_gap']}, caveat_gap={row['caveat_burden_gap']}"
            )
        lines.append("")
    lines.append("## Top Failure Candidates")
    for row in analysis.get("top_failures", [])[:8]:
        issue_text = "; ".join(row.get("issues", [])[:3]) or "no issue text"
        lines.append(
            f"- {row['item_id']} ({row['model']}, {row['role']}, {row.get('agency_mode', 'explicit')}): "
            f"fit={row['role_manifestation']}, under={row['underperformance']}, "
            f"intrusion={row['role_intrusion']}; {issue_text}"
        )
    lines.append("")
    lines.append("## Next-Step Recommendations")
    if overall.get("json_valid_rate") is not None and overall.get("json_valid_rate") < 0.95:
        lines.append("- Tighten the judge prompt JSON-only instruction or add one retry before relying on judge scores.")
    if overall.get("invariant_floor_violation_rate") is not None and overall.get("invariant_floor_violation_rate") > 0.2:
        lines.append("- Inspect source packet grounding and non-manipulation failures before expanding the study.")
    if not analysis.get("human_calibration", {}).get("available"):
        lines.append("- Export a blinded rating packet and calibrate local judge scores before making publication-grade claims.")
    lines.append("- Freeze configs before final runs; keep calibration artifacts separate from base prompts/source packets.")
    return "\n".join(lines) + "\n"
