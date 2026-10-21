"""Client Azure OpenAI avec cache, retries et sortie structurée."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings
from .schemas import Classification, json_schema

# Exceptions API à retenter (importées paresseusement pour permettre les tests hors ligne)
try:
    from openai import (  # type: ignore[import-not-found]
        APIConnectionError,
        APIError,
        APITimeoutError,
        RateLimitError,
    )

    _RETRYABLE = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        APIError,
        ConnectionError,
        TimeoutError,
    )
except ImportError:  # pragma: no cover - openai non installé
    _RETRYABLE = (Exception,)


class LLMClientError(Exception):
    """Erreur non-retentable côté client LLM."""


@dataclass
class ClassificationCall:
    classification: Classification
    raw_json: str
    cached: bool
    model: str
    prompt_version: str
    latency_ms: int
    usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    error: str | None = None


_LIST_LIMITS = {"evidence": 6}
_DEFAULT_LIST_LIMIT = 10
_TEXT_LIMITS = {"justification": 600}


def _clip_to_limits(parsed: dict[str, Any]) -> dict[str, Any]:
    """Tronque les champs trop longs (Ollama n'applique pas maxItems/maxLength du schéma)."""
    clipped = dict(parsed)
    for name, value in parsed.items():
        if isinstance(value, list):
            clipped[name] = value[: _LIST_LIMITS.get(name, _DEFAULT_LIST_LIMIT)]
        elif name in _TEXT_LIMITS and isinstance(value, str):
            clipped[name] = value[: _TEXT_LIMITS[name]]
    if isinstance(clipped.get("confidence"), (int, float)):
        clipped["confidence"] = min(1.0, max(0.0, float(clipped["confidence"])))
    return clipped


class LLMBackend(Protocol):
    """Interface minimale pour permettre le mocking en tests."""

    def complete(self, *, system: str, user: str, deployment: str) -> tuple[str, dict[str, int]]:
        ...


class AzureOpenAIBackend:
    """Backend Azure OpenAI utilisant l'endpoint Chat Completions."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1000,
        themes: list[str] | None = None,
    ) -> None:
        try:
            from openai import AzureOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise LLMClientError("Le paquet `openai` n'est pas installé.") from exc

        endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
        )

        if not endpoint or not api_key:
            raise LLMClientError(
                "AZURE_OPENAI_ENDPOINT et AZURE_OPENAI_API_KEY sont requis (voir .env.example)."
            )

        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._schema = json_schema(themes)

    def complete(self, *, system: str, user: str, deployment: str) -> tuple[str, dict[str, int]]:
        response = self._client.chat.completions.create(
            model=deployment,
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
            response_format={"type": "json_schema", "json_schema": self._schema},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            "completion_tokens": (
                getattr(response.usage, "completion_tokens", 0) if response.usage else 0
            ),
            "total_tokens": getattr(response.usage, "total_tokens", 0) if response.usage else 0,
        }
        return content, usage


class OllamaUnavailableError(ConnectionError):
    """Serveur Ollama injoignable (retentable)."""


class OllamaBackend:
    """Backend local Ollama (`/api/chat`) avec sortie contrainte par schéma JSON."""

    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_output_tokens: int = 1000,
        num_ctx: int = 8192,
        themes: list[str] | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._url = host.rstrip("/") + "/api/chat"
        self._options = {
            "temperature": temperature,
            "num_predict": max_output_tokens,
            "num_ctx": num_ctx,
        }
        self._schema = json_schema(themes)["schema"]
        self._timeout = timeout

    def complete(self, *, system: str, user: str, deployment: str) -> tuple[str, dict[str, int]]:
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": deployment,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": self._schema,
                "stream": False,
                "options": self._options,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LLMClientError(f"Ollama HTTP {exc.code} : {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaUnavailableError(f"Ollama injoignable ({self._url}) : {exc}") from exc

        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return data.get("message", {}).get("content", ""), usage


class LLMClient:
    """Wrapper qui gère cache disque, retries et validation Pydantic."""

    def __init__(
        self,
        settings: Settings,
        prompt: str,
        backend: LLMBackend,
        *,
        deployment: str,
    ) -> None:
        self._settings = settings
        self._prompt = prompt
        self._backend = backend
        self._deployment = deployment
        self._prompt_version = settings.classification.prompt_version
        self._cache_dir = settings.cache_root
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, rule_id: str, text_hash: str) -> Path:
        # Indépendant de `rule_id` : les sous-règles au texte identique partagent la réponse.
        key = hashlib.sha256(
            f"{text_hash}|{self._deployment}|{self._prompt_version}".encode()
        ).hexdigest()
        # Regroupe les fichiers pour éviter d'exploser un seul dossier
        return self._cache_dir / key[:2] / f"{key}.json"

    def classify(
        self,
        *,
        rule_id: str,
        source_file: str,
        text: str,
        text_hash: str,
    ) -> ClassificationCall:
        cache_path = self._cache_path(rule_id, text_hash)
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return ClassificationCall(
                classification=Classification.model_validate(data["classification"]),
                raw_json=data["raw_json"],
                cached=True,
                model=data.get("model", self._deployment),
                prompt_version=data.get("prompt_version", self._prompt_version),
                latency_ms=0,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        user_payload = json.dumps(
            {"rule_id": rule_id, "source_file": source_file, "text": text},
            ensure_ascii=False,
        )

        @retry(
            reraise=True,
            retry=retry_if_exception_type(_RETRYABLE),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(self._settings.classification.max_retries),
        )
        def _call() -> tuple[str, dict[str, int], int]:
            import time

            t0 = time.perf_counter()
            content, usage = self._backend.complete(
                system=self._prompt,
                user=user_payload,
                deployment=self._deployment,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return content, usage, latency_ms

        try:
            raw_json, usage, latency_ms = _call()
        except RetryError as exc:  # pragma: no cover - dépendant du backend
            raise LLMClientError(f"Toutes les tentatives ont échoué : {exc}") from exc

        try:
            parsed: dict[str, Any] = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Réponse LLM non-JSON : {raw_json[:200]}...") from exc

        classification = Classification.model_validate(_clip_to_limits(parsed))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "classification": classification.model_dump(mode="json"),
                    "raw_json": raw_json,
                    "model": self._deployment,
                    "prompt_version": self._prompt_version,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return ClassificationCall(
            classification=classification,
            raw_json=raw_json,
            cached=False,
            model=self._deployment,
            prompt_version=self._prompt_version,
            latency_ms=latency_ms,
            usage=usage,
        )
