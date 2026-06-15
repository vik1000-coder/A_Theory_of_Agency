from __future__ import annotations

import csv
import json
import random
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, TypeVar

import yaml
from pydantic import BaseModel

from .schemas import (
    AgencyMode,
    CalibrationState,
    HumanRatingRecord,
    PromptItem,
    RoleCardsFile,
    RunMeta,
    SelectionStrategy,
    SourcePacket,
    StudyConfig,
    now_iso,
)

T = TypeVar("T", bound=BaseModel)


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (base or Path.cwd()) / value


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return data


def load_config(path: str | Path) -> StudyConfig:
    config_path = resolve_path(path)
    return StudyConfig.model_validate(read_yaml(config_path))


def load_role_cards(path: str | Path) -> RoleCardsFile:
    return RoleCardsFile.model_validate(read_yaml(resolve_path(path)))


def load_prompts(path: str | Path) -> list[PromptItem]:
    items: list[PromptItem] = []
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(PromptItem.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid prompt JSONL at line {line_no}: {exc}") from exc
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt ids must be unique")
    return items


def load_source_packets(directory: str | Path) -> dict[str, SourcePacket]:
    packets: dict[str, SourcePacket] = {}
    for path in sorted(resolve_path(directory).glob("*.json")):
        packet = SourcePacket.model_validate_json(path.read_text(encoding="utf-8"))
        if packet.id in packets:
            raise ValueError(f"duplicate source packet id {packet.id}")
        packets[packet.id] = packet
    if not packets:
        raise ValueError(f"no source packets found in {directory}")
    return packets


def validate_prompt_sources(prompts: list[PromptItem], packets: dict[str, SourcePacket]) -> None:
    missing = sorted({prompt.source_packet_id for prompt in prompts} - set(packets))
    if missing:
        raise ValueError(f"prompts reference missing source packets: {missing}")


def filter_prompts(prompts: list[PromptItem], prompt_ids: list[str] | None) -> list[PromptItem]:
    if prompt_ids is None:
        return prompts
    prompt_by_id = {prompt.id: prompt for prompt in prompts}
    missing = sorted(set(prompt_ids) - set(prompt_by_id))
    if missing:
        raise ValueError(f"config references missing prompt_ids: {missing}")
    selected = [prompt_by_id[prompt_id] for prompt_id in prompt_ids]
    selected_ids = {prompt.id for prompt in selected}
    broken_pairs = sorted(
        prompt.id
        for prompt in selected
        if prompt.paired_id and prompt.paired_id in prompt_by_id and prompt.paired_id not in selected_ids
    )
    if broken_pairs:
        raise ValueError(f"prompt_ids must include paired counterparts for: {broken_pairs}")
    return selected


def append_jsonl(path: Path, rows: Iterable[BaseModel | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, BaseModel):
                payload = row.model_dump(mode="json")
            else:
                payload = row
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, BaseModel):
                payload = row.model_dump(mode="json")
            else:
                payload = row
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path, model: type[T] | None = None) -> list[T] | list[dict]:
    if not path.exists():
        return []
    rows: list = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            rows.append(model.model_validate(data) if model else data)
    return rows


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def new_run_id(study_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{study_id}_{stamp}_{uuid.uuid4().hex[:8]}"


def run_dir(config: StudyConfig, run_id: str) -> Path:
    return resolve_path(config.runs_dir) / run_id


def init_run(
    config: StudyConfig,
    config_path: str,
    models: list[str],
    frozen_config: bool,
    run_id: str | None = None,
    calibration_active: bool = False,
) -> tuple[str, Path]:
    selected = run_id or new_run_id(config.study_id)
    path = run_dir(config, selected)
    path.mkdir(parents=True, exist_ok=True)
    meta_path = path / "run_meta.json"
    if meta_path.exists():
        meta = RunMeta.model_validate(read_json(meta_path))
        meta.updated_at = now_iso()
        meta.calibration_active = calibration_active
        # Contamination is sticky: once the tuning loop ran in this run dir it stays flagged.
        meta.contaminated = meta.contaminated or calibration_active
    else:
        meta = RunMeta(
            run_id=selected,
            study_id=config.study_id,
            config_path=str(config_path),
            models=models,
            frozen_config=frozen_config,
            created_at=now_iso(),
            updated_at=now_iso(),
            judge_model=config.judge_model,
            calibration_active=calibration_active,
            blind_role_inference=config.blind_role_inference,
            contaminated=calibration_active,
        )
    write_json(meta_path, meta.model_dump(mode="json"))
    if frozen_config:
        frozen_path = path / "frozen_config.yml"
        if not frozen_path.exists():
            shutil.copyfile(resolve_path(config_path), frozen_path)
    calibration_path = path / "calibration" / "state.json"
    if not calibration_path.exists():
        write_json(calibration_path, CalibrationState().model_dump(mode="json"))
    return selected, path


def load_calibration_state(path: Path) -> CalibrationState:
    return CalibrationState.model_validate(read_json(path / "calibration" / "state.json", CalibrationState().model_dump()))


def save_calibration_state(path: Path, state: CalibrationState) -> None:
    write_json(path / "calibration" / "state.json", state.model_dump(mode="json"))


def select_batch(
    prompts: list[PromptItem],
    roles: list[str],
    models: list[str],
    batch_size: int | str,
    seed: int,
    cycle: int,
    agency_modes: list[AgencyMode] | None = None,
    selection_strategy: SelectionStrategy = "paired_balanced",
) -> list[tuple[PromptItem, str, str, AgencyMode]]:
    modes = agency_modes or ["explicit"]
    prompt_by_id = {prompt.id: prompt for prompt in prompts}
    units: list[list[tuple[PromptItem, str, str, AgencyMode]]] = []
    seen_pairs: set[tuple[str, str, str, str, str]] = set()
    for mode in modes:
        for prompt in prompts:
            for role in roles:
                for model in models:
                    if prompt.paired_id and prompt.paired_id in prompt_by_id:
                        left, right = sorted([prompt.id, prompt.paired_id])
                        pair_key = (left, right, role, model, mode)
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)
                        paired = prompt_by_id[prompt.paired_id]
                        units.append([(prompt, role, model, mode), (paired, role, model, mode)])
                    else:
                        units.append([(prompt, role, model, mode)])
    rng = random.Random(seed + cycle)
    if selection_strategy == "paired_balanced":
        rng.shuffle(units)
    if batch_size == "all":
        selected = [item for unit in units for item in unit]
    else:
        if selection_strategy == "full_factorial_pairs":
            rng.shuffle(units)
        selected = []
        target = int(batch_size)
        for unit in units:
            if len(selected) + len(unit) > target:
                continue
            selected.extend(unit)
            if len(selected) == target:
                break
    model_order = {model: idx for idx, model in enumerate(models)}
    role_order = {role: idx for idx, role in enumerate(roles)}
    mode_order = {mode: idx for idx, mode in enumerate(modes)}
    return sorted(selected, key=lambda item: (model_order[item[2]], role_order[item[1]], mode_order[item[3]], item[0].id))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_human_ratings(path: Path, ratings: list[HumanRatingRecord]) -> None:
    append_jsonl(path, ratings)
