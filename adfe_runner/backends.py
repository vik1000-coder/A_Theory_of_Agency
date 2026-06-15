"""Generation backends.

Phase 1 of ADFE is local-only (Ollama). This module adds the seam for auditing frontier
models later without touching the pipeline: a model spec may be prefixed with a backend,
e.g. ``anthropic:claude-opus-4-8``; bare specs route to Ollama. ``RoutedClient`` is a
drop-in for ``OllamaClient`` (same ``generate``/``tags``/``ensure_models`` surface) so the
generation path is unchanged when no API specs are used.

The AnthropicBackend is EXPERIMENTAL and untested without a key; verify the model id and
api version against the claude-api reference before enabling it for a real audit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

from .ollama import OllamaClient, OllamaError

API_PREFIXES = ("anthropic",)


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Return (backend, model). 'anthropic:claude-opus-4-8' -> ('anthropic', '...'),
    bare 'qwen3:8b' -> ('ollama', 'qwen3:8b'). Only known API prefixes are split, so local
    Ollama tags that contain ':' (every one of them) are left intact."""
    head, sep, tail = spec.partition(":")
    if sep and head in API_PREFIXES:
        return head, tail
    return "ollama", spec


@dataclass
class AnthropicBackend:
    api_key: str | None = None
    base_url: str = "https://api.anthropic.com"
    version: str = "2023-06-01"
    timeout: int = 180

    def generate(self, model: str, prompt: str, options: dict[str, Any] | None = None, think: bool | None = None) -> str:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise OllamaError("ANTHROPIC_API_KEY not set; cannot use anthropic backend")
        options = options or {}
        body = {
            "model": model,
            "max_tokens": int(options.get("num_predict", 1024)),
            "messages": [{"role": "user", "content": prompt}],
        }
        if "temperature" in options:
            body["temperature"] = options["temperature"]
        if "top_p" in options:
            body["top_p"] = options["top_p"]
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/v1/messages",
                json=body,
                headers={"x-api-key": key, "anthropic-version": self.version, "content-type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Anthropic generation failed for {model}: {exc}") from exc
        data = response.json()
        parts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
        return "".join(parts).strip()


@dataclass
class RoutedClient:
    """Drop-in for OllamaClient that dispatches by model spec prefix."""

    ollama_url: str = "http://localhost:11434"
    timeout: int = 180
    ollama: OllamaClient = field(init=False)
    _anthropic: AnthropicBackend | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.ollama = OllamaClient(self.ollama_url, timeout=self.timeout)

    def _anthropic_backend(self) -> AnthropicBackend:
        if self._anthropic is None:
            self._anthropic = AnthropicBackend()
        return self._anthropic

    def tags(self) -> list[str]:
        return self.ollama.tags()

    def ensure_models(self, models: list[str]) -> None:
        local = [parse_model_spec(m)[1] for m in models if parse_model_spec(m)[0] == "ollama"]
        if local:
            self.ollama.ensure_models(local)
        api = [m for m in models if parse_model_spec(m)[0] != "ollama"]
        if api and not os.environ.get("ANTHROPIC_API_KEY"):
            raise OllamaError(f"API models requested but ANTHROPIC_API_KEY not set: {api}")

    def generate(self, model: str, prompt: str, options: dict[str, Any] | None = None, think: bool | None = False) -> str:
        backend, name = parse_model_spec(model)
        if backend == "anthropic":
            return self._anthropic_backend().generate(name, prompt, options=options, think=think)
        return self.ollama.generate(name, prompt, options=options, think=think)
