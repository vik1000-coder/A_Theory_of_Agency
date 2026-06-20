"""Proper inferential test of the agency gradient.

The agency gradient is ADFE's central novel claim: as a role's assigned agency_level rises,
certain dimensions (curation accountability, source discipline) should rise while factuality
and non-manipulation floors stay flat. agency_level is a fixed property of each role, so role
and agency_level are collinear -- the gradient is therefore a *continuous* test of agency_level
ACROSS roles, not a role fixed effect.

We replace the previous per-model Pearson r (underpowered, confounded by model) with a mixed
model: score ~ agency_level, with a random intercept per model so the slope is estimated while
accounting for model-to-model differences. Degrades gracefully (returns reasons, never raises)
so it can run inside analyze without statsmodels guarantees at call time.
"""

from __future__ import annotations

from typing import Any

from .schemas import DIMENSIONS, PromptItem, RoleCard, ScoreRecord


def agency_gradient_mixedlm(
    scores: list[ScoreRecord],
    roles: dict[str, RoleCard],
    prompts: list[PromptItem] | None = None,
) -> dict[str, Any]:
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except Exception as exc:  # pragma: no cover - import guard
        return {"available": False, "error": f"statsmodels/pandas unavailable: {exc}"}

    topic_by_prompt = {p.id: p.topic for p in (prompts or [])}
    rows: list[dict[str, Any]] = []
    for score in scores:
        role = roles.get(score.role)
        if role is None:
            continue
        row: dict[str, Any] = {
            "model": score.model,
            "role": score.role,
            "agency_level": role.agency_level,
            "prompt_id": score.prompt_id,
            "topic": topic_by_prompt.get(score.prompt_id, "na"),
        }
        for dim in DIMENSIONS:
            row[dim] = score.scores[dim]
        rows.append(row)

    if len(rows) < 12:
        return {"available": False, "error": "too few scores for a mixed model (<12)"}

    df = pd.DataFrame(rows)
    n_models = int(df["model"].nunique())
    result: dict[str, Any] = {
        "available": True,
        "model": "score ~ agency_level + (1 | model)",
        "n": int(len(df)),
        "n_models": n_models,
        "n_agency_levels": int(df["agency_level"].nunique()),
        "by_dimension": {},
    }

    for dim in DIMENSIONS:
        entry: dict[str, Any] = {"converged": False}
        try:
            if n_models < 2:
                entry["error"] = "need >=2 models for a model random effect"
            elif df[dim].std() == 0 or df["agency_level"].nunique() < 2:
                entry["error"] = "insufficient variance in outcome or agency_level"
            else:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # singular RE covariance is benign here
                    fit = smf.mixedlm(f"{dim} ~ agency_level", df, groups=df["model"]).fit(
                        reml=False, method="lbfgs", disp=False
                    )
                ci = fit.conf_int().loc["agency_level"]
                pval = float(fit.pvalues.get("agency_level"))
                entry = {
                    "converged": bool(getattr(fit, "converged", True)),
                    "agency_level_coef": round(float(fit.params.get("agency_level")), 4),
                    "se": round(float(fit.bse.get("agency_level")), 4),
                    "pvalue": round(pval, 4),
                    "ci_low": round(float(ci.iloc[0]), 4),
                    "ci_high": round(float(ci.iloc[1]), 4),
                    "significant_0_05": bool(pval < 0.05),
                }
        except Exception as exc:  # pragma: no cover - numerical edge cases
            entry = {"converged": False, "error": str(exc)[:200]}
        result["by_dimension"][dim] = entry

    return result


def agency_gradient_adjusted(
    scores: list[ScoreRecord],
    roles: dict[str, RoleCard],
    prompts: list[PromptItem] | None = None,
) -> dict[str, Any]:
    """Estimate agency slope while controlling for refusal and output-length artifacts."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except Exception as exc:  # pragma: no cover - import guard
        return {"available": False, "error": f"statsmodels/pandas unavailable: {exc}"}

    topic_by_prompt = {p.id: p.topic for p in (prompts or [])}
    rows: list[dict[str, Any]] = []
    for score in scores:
        role = roles.get(score.role)
        if role is None:
            continue
        row: dict[str, Any] = {
            "model": score.model,
            "role": score.role,
            "agency_level": role.agency_level,
            "prompt_id": score.prompt_id,
            "topic": topic_by_prompt.get(score.prompt_id, "na"),
            "agency_mode": score.agency_mode,
            "refusal": float(score.refusal),
            "word_count": float(score.checks.get("word_count") or 0.0),
        }
        for dim in DIMENSIONS:
            row[dim] = score.scores[dim]
        rows.append(row)

    if len(rows) < 12:
        return {"available": False, "error": "too few scores for an adjusted mixed model (<12)"}

    df = pd.DataFrame(rows)
    n_models = int(df["model"].nunique())
    word_sd = float(df["word_count"].std() or 0.0)
    df["word_count_z"] = 0.0 if word_sd == 0.0 else (df["word_count"] - df["word_count"].mean()) / word_sd
    result: dict[str, Any] = {
        "available": True,
        "model": "score ~ agency_level + refusal + word_count_z + agency_mode + (1 | model)",
        "n": int(len(df)),
        "n_models": n_models,
        "n_agency_levels": int(df["agency_level"].nunique()),
        "refusal_rate": round(float(df["refusal"].mean()), 4),
        "by_dimension": {},
    }

    for dim in DIMENSIONS:
        entry: dict[str, Any] = {"converged": False}
        try:
            if n_models < 2:
                entry["error"] = "need >=2 models for a model random effect"
            elif df[dim].std() == 0 or df["agency_level"].nunique() < 2:
                entry["error"] = "insufficient variance in outcome or agency_level"
            else:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.mixedlm(
                        f"{dim} ~ agency_level + refusal + word_count_z + agency_mode",
                        df,
                        groups=df["model"],
                    ).fit(reml=False, method="lbfgs", disp=False)
                ci = fit.conf_int().loc["agency_level"]
                pval = float(fit.pvalues.get("agency_level"))
                refusal_p = float(fit.pvalues.get("refusal"))
                entry = {
                    "converged": bool(getattr(fit, "converged", True)),
                    "agency_level_coef": round(float(fit.params.get("agency_level")), 4),
                    "agency_level_se": round(float(fit.bse.get("agency_level")), 4),
                    "agency_level_pvalue": round(pval, 4),
                    "agency_level_ci_low": round(float(ci.iloc[0]), 4),
                    "agency_level_ci_high": round(float(ci.iloc[1]), 4),
                    "agency_level_significant_0_05": bool(pval < 0.05),
                    "refusal_coef": round(float(fit.params.get("refusal")), 4),
                    "refusal_pvalue": round(refusal_p, 4),
                    "refusal_significant_0_05": bool(refusal_p < 0.05),
                    "word_count_z_coef": round(float(fit.params.get("word_count_z")), 4),
                    "word_count_z_pvalue": round(float(fit.pvalues.get("word_count_z")), 4),
                }
        except Exception as exc:  # pragma: no cover - numerical edge cases
            entry = {"converged": False, "error": str(exc)[:200]}
        result["by_dimension"][dim] = entry

    return result
