"""Tests that lock in the Tier 1-4 methodology fixes as invariants.

These guard against the failure modes the original pipeline had: a self-tuned judge, a
role-aware (circular) role inference, a circular headline metric, and unverified pairs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from adfe_runner.analysis import analyze_scores, interval_hypothesis_tests, refusal_label_agreement
from adfe_runner.backends import RoutedClient, parse_model_spec
from adfe_runner import cli as cli_module
from adfe_runner.cli import command_doctor
from adfe_runner.io import init_run, load_config, load_prompts, load_role_cards, read_json
from adfe_runner.ollama import OllamaError
from adfe_runner.prompting import build_role_inference_prompt
from adfe_runner.schemas import DIMENSIONS, RunMeta, ScoreRecord, now_iso
from adfe_runner.stats import agency_gradient_mixedlm
from adfe_runner.validation import audit_pair_analogy

ROOT = Path(__file__).resolve().parents[1]


def make_score(model: str, role: str, scores: dict[str, float], **checks) -> ScoreRecord:
    norm = {dim: scores.get(dim, 0.5) for dim in DIMENSIONS}
    raw = {dim: round(norm[dim] * 4, 4) for dim in DIMENSIONS}
    return ScoreRecord(
        run_id="t",
        cycle=0,
        item_id=f"{model}:{role}:{checks.get('prompt_id','p')}",
        model=model,
        role=role,
        prompt_id=checks.get("prompt_id", "immigration_briefing"),
        source_packet_id="immigration_2026_static",
        judge_model="qwen3:8b",
        scores_raw=raw,
        scores=norm,
        inferred_role=role,
        refusal=bool(checks.get("refusal", False)),
        json_valid=True,
        checks={k: v for k, v in checks.items() if k in ("regex_refusal", "judge_refusal")},
        issues=[],
        rationale="",
        created_at=now_iso(),
    )


# ---------- Tier 1: de-confounding ----------

def test_frozen_is_default_and_calibration_flagged(tmp_path):
    config = load_config(ROOT / "archives/workshop_legacy_20260622/configs/clean_local.yml")
    object.__setattr__(config, "runs_dir", str(tmp_path))
    # Default iterate path: calibration not active -> not contaminated, judge recorded.
    _id, path = init_run(config, "archives/workshop_legacy_20260622/configs/clean_local.yml", ["llama3.2:3b"], frozen_config=True, calibration_active=False)
    meta = RunMeta.model_validate(read_json(path / "run_meta.json"))
    assert meta.frozen_config is True
    assert meta.calibration_active is False
    assert meta.contaminated is False
    assert meta.judge_model == "qwen3:8b"
    # Opt-in calibration must mark the run contaminated, stickily.
    _id2, path2 = init_run(config, "archives/workshop_legacy_20260622/configs/clean_local.yml", ["llama3.2:3b"], frozen_config=False, calibration_active=True)
    meta2 = RunMeta.model_validate(read_json(path2 / "run_meta.json"))
    assert meta2.calibration_active is True
    assert meta2.contaminated is True


def test_blinded_role_inference_hides_assigned_role():
    roles = load_role_cards(ROOT / "data/role_cards.yml")
    prompts = load_prompts(ROOT / "data/prompts.jsonl")
    prompt = prompts[0]
    assigned = roles.by_id["campaign_aide"]
    blinded = build_role_inference_prompt("Some output text.", list(roles.by_id), packet=None, prompt=prompt)  # type: ignore[arg-type]
    # The judge must not be told which role was assigned.
    assert assigned.card.strip()[:40] not in blinded
    assert "Assigned role" not in blinded
    assert "Agency context" not in blinded
    # It still must offer the role vocabulary to choose from.
    assert "campaign_aide" in blinded


def test_refusal_label_agreement_separates_detectors():
    enriched = [
        {"regex_refusal": True, "judge_refusal": True},
        {"regex_refusal": False, "judge_refusal": True},
        {"regex_refusal": True, "judge_refusal": False},
        {"regex_refusal": False, "judge_refusal": False},
    ]
    out = refusal_label_agreement(enriched)
    assert out["available"] is True
    assert out["both_refusal"] == 1
    assert out["judge_only"] == 1
    assert out["regex_only"] == 1
    assert out["agreement_rate"] == 0.5


# ---------- Tier 2: non-circular metrics ----------

def test_mixedlm_recovers_planted_gradient():
    roles = load_role_cards(ROOT / "data/role_cards.yml").by_id
    prompts = load_prompts(ROOT / "data/prompts.jsonl")
    scores = []
    # C tracks agency_level (planted gradient) with small deterministic residual noise so the
    # model is estimable; M is flat (a floor that should not move with agency).
    for mi, model in enumerate(("m1", "m2", "m3")):
        for role_id, role in roles.items():
            for rep in range(4):
                level = role.agency_level
                jitter = (rep - 1.5) * 0.03 + (mi - 1) * 0.02
                c_val = min(1.0, max(0.0, 0.2 + 0.6 * level + jitter))
                m_val = min(1.0, max(0.0, 0.9 + jitter))
                scores.append(
                    make_score(
                        model,
                        role_id,
                        {"C": c_val, "M": m_val, "U": 0.5, "E": 0.7, "V": 0.6, "D": 0.5},
                        prompt_id=f"p{rep}",
                    )
                )
    result = agency_gradient_mixedlm(scores, roles, prompts)
    assert result["available"] is True
    c = result["by_dimension"]["C"]
    assert c["converged"] and c["agency_level_coef"] > 0.2 and c["significant_0_05"] is True
    m = result["by_dimension"]["M"]
    assert abs(m["agency_level_coef"]) < 0.1 and m["significant_0_05"] is False


def test_interval_hypothesis_verdicts():
    roles = load_role_cards(ROOT / "data/role_cards.yml").by_id
    # assistant expected C band is [0.20, 0.70]; put all C inside -> supported, far outside -> violated.
    inside = [make_score("m", "assistant", {"C": 0.45}, prompt_id=f"p{i}") for i in range(5)]
    outside = [make_score("m", "assistant", {"C": 0.95}, prompt_id=f"p{i}") for i in range(5)]
    assert interval_hypothesis_tests(inside, roles)["assistant"]["C"]["verdict"] == "supported"
    assert interval_hypothesis_tests(outside, roles)["assistant"]["C"]["verdict"] == "violated"


def test_analyze_scores_emits_tier2_sections():
    roles = load_role_cards(ROOT / "data/role_cards.yml").by_id
    prompts = load_prompts(ROOT / "data/prompts.jsonl")
    scores = [make_score(m, r, {}, prompt_id="immigration_briefing") for m in ("m1", "m2") for r in roles]
    analysis = analyze_scores(scores, prompts, roles)
    assert "agency_gradient_mixedlm" in analysis
    assert "dimension_means_by_role" in analysis
    assert "dimension_means_by_agency_level" in analysis
    assert "interval_hypothesis_tests" in analysis


# ---------- Tier 3: backend abstraction ----------

def test_parse_model_spec_routes_only_known_api_prefixes():
    assert parse_model_spec("qwen3:8b") == ("ollama", "qwen3:8b")
    assert parse_model_spec("llama3.2:1b") == ("ollama", "llama3.2:1b")
    assert parse_model_spec("anthropic:claude-opus-4-8") == ("anthropic", "claude-opus-4-8")


def test_routed_client_requires_key_for_api_models(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = RoutedClient("http://localhost:11434")
    with pytest.raises(OllamaError):
        client.ensure_models(["anthropic:claude-opus-4-8"])
    with pytest.raises(OllamaError):
        client.generate("anthropic:claude-opus-4-8", "hi")


def test_doctor_uses_routed_client_for_remote_models(monkeypatch, tmp_path):
    config = load_config(ROOT / "archives/workshop_legacy_20260622/configs/clean_local.yml").model_copy(update={"default_models": ["xai:grok-test"]})
    config_path = tmp_path / "frontier.yml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    seen: list[list[str]] = []

    class FakeRoutedClient:
        def __init__(self, ollama_url):
            self.ollama_url = ollama_url

        def ensure_models(self, models):
            seen.append(models)

    monkeypatch.setattr(cli_module, "RoutedClient", FakeRoutedClient)

    result = command_doctor(SimpleNamespace(config=str(config_path), models=None))

    assert result == 0
    assert seen == [["qwen3:8b", "xai:grok-test"]]


# ---------- Tier 4: pair analogy ----------

def test_real_prompt_pairs_are_analogous():
    prompts = load_prompts(ROOT / "data/prompts.jsonl")
    audit = audit_pair_analogy(prompts)
    assert audit["n_pairs"] > 0
    assert audit["n_issues"] == 0, audit["issues"]


def test_pair_analogy_flags_mismatched_pair():
    prompts = load_prompts(ROOT / "data/prompts.jsonl")
    pair = [p for p in prompts if p.paired_id][:2]
    a, b = pair[0], pair[1]
    object.__setattr__(b, "paired_id", a.id)
    object.__setattr__(a, "paired_id", b.id)
    object.__setattr__(b, "risk", "totally-different-risk")  # break analogy
    audit = audit_pair_analogy([a, b])
    assert audit["n_issues"] >= 1
    assert any("risk" in issue["problem"] for issue in audit["issues"])
