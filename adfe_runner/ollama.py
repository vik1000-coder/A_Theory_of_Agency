from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import requests

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks that reasoning models (e.g. deepseek-r1)
    emit even with think disabled. We score the user-facing answer, not the scratchpad."""
    return _THINK_BLOCK.sub("", text).strip()


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    timeout: int = 180

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def tags(self) -> list[str]:
        try:
            response = requests.get(self._url("/api/tags"), timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"could not reach Ollama at {self.base_url}: {exc}") from exc
        payload = response.json()
        return sorted(model["name"] for model in payload.get("models", []))

    def ensure_models(self, models: list[str]) -> None:
        installed = set(self.tags())
        missing = sorted(set(models) - installed)
        if missing:
            raise OllamaError(f"missing Ollama models: {', '.join(missing)}")

    def build_generate_payload(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any] | None = None,
        think: bool | None = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        }
        if think is not None:
            payload["think"] = think
        return payload

    def generate(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any] | None = None,
        think: bool | None = False,
    ) -> str:
        payload = self.build_generate_payload(model, prompt, options=options, think=think)
        try:
            response = requests.post(self._url("/api/generate"), json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama generation failed for {model}: {exc}") from exc
        data = response.json()
        if "response" not in data:
            raise OllamaError(f"Ollama returned no response for {model}: {data}")
        return strip_think(str(data["response"]))
