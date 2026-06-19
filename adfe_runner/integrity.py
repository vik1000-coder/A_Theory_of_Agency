from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import (
    filter_prompts,
    load_prompts,
    read_json,
    read_jsonl,
    run_dir,
    select_batch,
    write_jsonl,
)
from .schemas import GenerationRecord, RunMeta, ScoreRecord, StudyConfig, now_iso


ARTIFACT_FILES = (
    "generations.jsonl",
    "generation_errors.jsonl",
    "scores.jsonl",
    "analysis.json",
    "observations.md",
)


def duplicate_item_ids(rows: Iterable[Any]) -> list[str]:
    counts = Counter(getattr(row, "item_id", None) for row in rows)
    return sorted(item_id for item_id, count in counts.items() if item_id and count > 1)


def require_unique_item_ids(rows: list[Any], label: str) -> None:
    duplicates = duplicate_item_ids(rows)
    if duplicates:
        preview = ", ".join(duplicates[:8])
        suffix = "" if len(duplicates) <= 8 else f" (+{len(duplicates) - 8} more)"
        raise ValueError(f"duplicate {label} item_id(s): {preview}{suffix}")


def dedupe_by_item_id(rows: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for row in rows:
        item_id = getattr(row, "item_id", None)
        if item_id in seen:
            continue
        seen.add(item_id)
        out.append(row)
    return out


def backup_run_artifacts(path: Path) -> Path:
    stamp = now_iso().replace(":", "").replace("-", "")
    backup_dir = path / "backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in ARTIFACT_FILES:
        src = path / name
        if src.exists():
            shutil.copyfile(src, backup_dir / name)
    return backup_dir


def _successful_generations(generations: list[GenerationRecord]) -> list[GenerationRecord]:
    return [record for record in generations if not record.error and record.output.strip()]


def _expected_full_count(config: StudyConfig, models: list[str]) -> int:
    prompts = filter_prompts(load_prompts(config.prompts_path), config.prompt_ids)
    selected = select_batch(
        prompts,
        config.roles,
        models,
        "all",
        config.seed,
        0,
        agency_modes=config.agency_modes,
        selection_strategy=config.selection_strategy,
    )
    return len(selected)


def audit_run(
    config: StudyConfig,
    run_id: str,
    expect_full: bool = False,
    allow_contaminated: bool = False,
) -> dict[str, Any]:
    path = run_dir(config, run_id)
    errors: list[str] = []
    warnings: list[str] = []
    meta_data = read_json(path / "run_meta.json")
    meta = RunMeta.model_validate(meta_data) if meta_data else None
    generations = read_jsonl(path / "generations.jsonl", GenerationRecord)
    scores = read_jsonl(path / "scores.jsonl", ScoreRecord)
    generation_errors = read_jsonl(path / "generation_errors.jsonl", GenerationRecord)

    if not path.exists():
        errors.append(f"run directory does not exist: {path}")
    if meta is None:
        errors.append("missing run_meta.json")
    elif meta.contaminated and not allow_contaminated:
        errors.append("run is contaminated; pass --allow-contaminated to audit anyway")
    if meta and meta.frozen_config and not (path / "frozen_config.yml").is_file():
        errors.append("frozen run is missing frozen_config.yml")

    gen_dupes = duplicate_item_ids(generations)
    score_dupes = duplicate_item_ids(scores)
    if gen_dupes:
        errors.append(f"duplicate generation item_id(s): {', '.join(gen_dupes[:8])}")
    if score_dupes:
        errors.append(f"duplicate score item_id(s): {', '.join(score_dupes[:8])}")

    errored_generations = [record for record in generations if record.error or not record.output.strip()]
    if errored_generations:
        errors.append(f"generations.jsonl contains {len(errored_generations)} errored/empty row(s)")

    successful = _successful_generations(generations)
    generation_ids = {record.item_id for record in successful}
    score_ids = {record.item_id for record in scores}
    error_ids = {record.item_id for record in generation_errors}
    unresolved_error_ids = sorted(error_ids - generation_ids)
    if unresolved_error_ids:
        errors.append(f"generation_errors.jsonl contains {len(unresolved_error_ids)} unresolved failure(s)")

    missing_scores = sorted(generation_ids - score_ids)
    orphan_scores = sorted(score_ids - generation_ids)
    if missing_scores:
        errors.append(f"{len(missing_scores)} successful generation(s) have no score")
    if orphan_scores:
        errors.append(f"{len(orphan_scores)} score row(s) have no successful generation")

    expected = None
    models = meta.models if meta else config.default_models
    if expect_full:
        expected = _expected_full_count(config, models)
        if len(generation_ids) != expected:
            errors.append(f"expected {expected} successful generation(s), found {len(generation_ids)}")
        if len(score_ids) != expected:
            errors.append(f"expected {expected} score row(s), found {len(score_ids)}")

    if meta and sorted(meta.models) != sorted(models):
        warnings.append("metadata model order differs from audit model order")

    return {
        "ok": not errors,
        "run_id": run_id,
        "path": str(path),
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "n_generations": len(generations),
            "n_successful_generations": len(generation_ids),
            "n_generation_errors": len(generation_errors),
            "n_unresolved_generation_errors": len(unresolved_error_ids),
            "n_scores": len(scores),
            "expected_full_count": expected,
            "duplicate_generation_ids": len(gen_dupes),
            "duplicate_score_ids": len(score_dupes),
        },
    }


def repair_run(
    config: StudyConfig,
    run_id: str,
    backup: bool,
    drop_error_generations: bool,
    dedupe: bool,
) -> dict[str, Any]:
    path = run_dir(config, run_id)
    if not path.exists():
        raise ValueError(f"run directory does not exist: {path}")
    backup_dir = backup_run_artifacts(path) if backup else None
    generations = read_jsonl(path / "generations.jsonl", GenerationRecord)
    scores = read_jsonl(path / "scores.jsonl", ScoreRecord)

    dropped_ids: set[str] = set()
    if drop_error_generations:
        kept_generations: list[GenerationRecord] = []
        for record in generations:
            if record.error or not record.output.strip():
                dropped_ids.add(record.item_id)
            else:
                kept_generations.append(record)
        generations = kept_generations
        scores = [record for record in scores if record.item_id not in dropped_ids]

    before_gen = len(generations)
    before_scores = len(scores)
    if dedupe:
        generations = dedupe_by_item_id(generations)
        scores = dedupe_by_item_id(scores)

    write_jsonl(path / "generations.jsonl", generations)
    write_jsonl(path / "scores.jsonl", scores)

    return {
        "backup_dir": str(backup_dir) if backup_dir else None,
        "dropped_error_generation_ids": sorted(dropped_ids),
        "deduped_generations": before_gen - len(generations),
        "deduped_scores": before_scores - len(scores),
        "n_generations": len(generations),
        "n_scores": len(scores),
    }
