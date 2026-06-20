from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


DIMENSIONS = ("U", "E", "V", "C", "D", "M")
AgencyMode = Literal["explicit", "implicit", "neutral"]
SelectionStrategy = Literal["paired_balanced", "full_factorial_pairs"]


class PromptItem(BaseModel):
    id: str
    topic: str
    task: str
    viewpoint: str
    paired_id: str | None = None
    risk: str
    audience: str
    source_packet_id: str
    prompt: str

    @field_validator("id", "topic", "task", "risk", "audience", "source_packet_id", "prompt")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class SourceEntry(BaseModel):
    title: str
    url_or_path: str
    source_class: str
    date: str
    claims: list[str] = Field(default_factory=list)
    notes: str = ""


class SourcePacket(BaseModel):
    id: str
    topic: str
    date: str
    summary: str
    sources: list[SourceEntry]

    @model_validator(mode="after")
    def require_sources(self) -> "SourcePacket":
        if not self.sources:
            raise ValueError("source packet must include at least one source")
        return self


class RoleCard(BaseModel):
    id: str
    label: str
    agency_level: float = Field(ge=0.0, le=1.0)
    card: str
    expected: dict[str, tuple[float, float]]

    @model_validator(mode="after")
    def validate_expected(self) -> "RoleCard":
        missing = set(DIMENSIONS) - set(self.expected)
        if missing:
            raise ValueError(f"missing expected intervals for {sorted(missing)}")
        for dim, interval in self.expected.items():
            if dim not in DIMENSIONS:
                raise ValueError(f"unknown dimension {dim}")
            low, high = interval
            if not 0 <= low <= high <= 1:
                raise ValueError(f"invalid interval for {dim}: {interval}")
        return self


class RoleCardsFile(BaseModel):
    dimensions: dict[str, str]
    roles: list[RoleCard]

    @property
    def by_id(self) -> dict[str, RoleCard]:
        return {role.id: role for role in self.roles}


class GenerationOptions(BaseModel):
    temperature: float = 0.2
    top_p: float = 0.9
    num_predict: int = 700


class StudyConfig(BaseModel):
    study_id: str
    seed: int = 20260613
    ollama_url: str = "http://localhost:11434"
    prompts_path: str = "data/prompts.jsonl"
    role_cards_path: str = "data/role_cards.yml"
    source_packets_dir: str = "data/source_packets"
    runs_dir: str = "runs"
    default_models: list[str] = Field(default_factory=lambda: ["qwen3:8b", "llama3.2:3b"])
    judge_model: str = "qwen3:8b"
    roles: list[str]
    prompt_ids: list[str] | None = None
    agency_modes: list[AgencyMode] = Field(default_factory=lambda: ["explicit"])
    selection_strategy: SelectionStrategy = "paired_balanced"
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    score_json_retry: int = 1
    allow_heuristic_fallback: bool = True
    blind_role_inference: bool = True

    @model_validator(mode="after")
    def validate_experiment_axes(self) -> "StudyConfig":
        if not self.roles:
            raise ValueError("config must include at least one role")
        if not self.default_models:
            raise ValueError("config must include at least one default model")
        if not self.agency_modes:
            raise ValueError("config must include at least one agency mode")
        if self.prompt_ids is not None and not self.prompt_ids:
            raise ValueError("prompt_ids must be omitted or non-empty")
        return self


class GenerationRecord(BaseModel):
    run_id: str
    cycle: int
    item_id: str
    model: str
    role: str
    agency_mode: AgencyMode = "explicit"
    prompt_id: str
    source_packet_id: str
    generation_prompt: str
    output: str
    created_at: str
    calibration_id: str | None = None
    error: str | None = None


class ScoreRecord(BaseModel):
    run_id: str
    cycle: int
    item_id: str
    model: str
    role: str
    agency_mode: AgencyMode = "explicit"
    prompt_id: str
    source_packet_id: str
    judge_model: str
    scores_raw: dict[str, float]
    scores: dict[str, float]
    inferred_role: str | None = None
    refusal: bool = False
    json_valid: bool = True
    checks: dict[str, Any] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    rationale: str = ""
    created_at: str

    @model_validator(mode="after")
    def validate_scores(self) -> "ScoreRecord":
        missing = set(DIMENSIONS) - set(self.scores)
        if missing:
            raise ValueError(f"missing normalized scores for {sorted(missing)}")
        for dim in DIMENSIONS:
            raw = self.scores_raw.get(dim)
            norm = self.scores.get(dim)
            if raw is None or not 0 <= raw <= 4:
                raise ValueError(f"invalid raw score for {dim}: {raw}")
            if norm is None or not 0 <= norm <= 1:
                raise ValueError(f"invalid normalized score for {dim}: {norm}")
        return self


class V2ScoreRecord(BaseModel):
    run_id: str
    cycle: int
    item_id: str
    model: str
    role: str
    agency_mode: AgencyMode = "explicit"
    prompt_id: str
    source_packet_id: str
    judge_model: str
    refusal: bool
    refusal_warranted: bool | None = None
    quality_scores_raw: dict[str, float]
    quality_scores: dict[str, float]
    role_profile_scores_raw: dict[str, float]
    role_profile_scores: dict[str, float]
    inferred_role: str | None = None
    json_valid: bool = True
    checks: dict[str, Any] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    rationale: str = ""
    created_at: str

    @model_validator(mode="after")
    def validate_v2_scores(self) -> "V2ScoreRecord":
        for label, raw_scores, normalized_scores in (
            ("quality", self.quality_scores_raw, self.quality_scores),
            ("role_profile", self.role_profile_scores_raw, self.role_profile_scores),
        ):
            missing = set(DIMENSIONS) - set(normalized_scores)
            if missing:
                raise ValueError(f"missing {label} normalized scores for {sorted(missing)}")
            for dim in DIMENSIONS:
                raw = raw_scores.get(dim)
                norm = normalized_scores.get(dim)
                if raw is None or not 0 <= raw <= 4:
                    raise ValueError(f"invalid {label} raw score for {dim}: {raw}")
                if norm is None or not 0 <= norm <= 1:
                    raise ValueError(f"invalid {label} normalized score for {dim}: {norm}")
        return self


class HumanRatingRecord(BaseModel):
    run_id: str
    item_id: str
    rater_id: str
    scores_raw: dict[str, float]
    inferred_role: str | None = None
    honored_agency_context: bool | None = None
    refusal_appropriate: bool | None = None
    paired_treatment_symmetric: bool | None = None
    notes: str = ""
    imported_at: str


class CalibrationState(BaseModel):
    active_generation_addendum: str = ""
    active_judge_addendum: str = ""
    pending_generation_addendum: str = ""
    pending_judge_addendum: str = ""
    accepted: list[dict[str, Any]] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)


class RunMeta(BaseModel):
    run_id: str
    study_id: str
    config_path: str
    models: list[str]
    frozen_config: bool
    created_at: str
    updated_at: str
    judge_model: str = ""
    calibration_active: bool = False
    blind_role_inference: bool = True
    contaminated: bool = False  # True if the prompt-tuning loop was ever enabled for this run
    config_hash: str = ""
    design_hash: str = ""


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


BatchSize = int | Literal["all"]
