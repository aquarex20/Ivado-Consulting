"""Chargement et validation de la configuration du pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class CorpusConfig(BaseModel):
    roots: list[str] = Field(default_factory=list)
    include_globs: list[str] = Field(
        default_factory=lambda: ["*.doc", "*.docx", "*.DOC", "*.DOCX"]
    )
    exclude_prefixes: list[str] = Field(default_factory=lambda: ["~$"])


class ExtractionConfig(BaseModel):
    strategies: list[str] = Field(
        default_factory=lambda: ["textutil", "python-docx", "wpd2text"]
    )
    timeout_seconds: int = 30

    @field_validator("strategies")
    @classmethod
    def _validate_strategies(cls, v: list[str]) -> list[str]:
        allowed = {"textutil", "python-docx", "wpd2text"}
        for s in v:
            if s not in allowed:
                raise ValueError(f"Stratégie inconnue: {s}. Autorisées: {sorted(allowed)}")
        return v


class SegmentationConfig(BaseModel):
    min_segment_chars: int = 40
    max_subrules_per_file: int = 200


class ClassificationConfig(BaseModel):
    backend: Literal["azure", "ollama"] = "azure"
    # Nom du deployment Azure, ou du modèle Ollama (ex. `llama3.2`).
    deployment: str | None = None
    ollama_host: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192
    temperature: float = 0.0
    max_output_tokens: int = 1000
    max_concurrency: int = 4
    max_retries: int = 5
    low_confidence_threshold: float = 0.6
    prompt_version: str = "v2"
    prompt_file: str = "config/classification_prompt.md"
    themes: list[str] | None = None


class OutputConfig(BaseModel):
    root: str = "data/output"
    cache_dir: str = "data/cache"
    checkpoint_every: int = 20


class PilotConfig(BaseModel):
    sample_size_g: int = 25
    sample_size_p: int = 25
    random_seed: int = 42


class Settings(BaseModel):
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    pilot: PilotConfig = Field(default_factory=PilotConfig)

    # Racine du projet (résolue au chargement, pas depuis YAML)
    project_root: Path = Field(default_factory=Path.cwd, exclude=True)

    @property
    def output_root(self) -> Path:
        return self.project_root / self.output.root

    @property
    def cache_root(self) -> Path:
        return self.project_root / self.output.cache_dir

    @property
    def prompt_path(self) -> Path:
        return self.project_root / self.classification.prompt_file

    def corpus_paths(self) -> list[Path]:
        return [self.project_root / root for root in self.corpus.roots]


def _default_settings_path(project_root: Path) -> Path:
    env_path = os.environ.get("ICARE_SETTINGS_FILE")
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else project_root / p
    candidate = project_root / "config" / "settings.yaml"
    if candidate.exists():
        return candidate
    return project_root / "config" / "settings.example.yaml"


def load_settings(
    settings_path: Path | None = None,
    project_root: Path | None = None,
) -> Settings:
    """Charge la configuration YAML et retourne un objet `Settings` validé."""

    root = (project_root or Path.cwd()).resolve()
    path = settings_path or _default_settings_path(root)

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {path}. "
            "Copiez config/settings.example.yaml vers config/settings.yaml."
        )

    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    settings = Settings.model_validate(data)
    settings.project_root = root
    return settings
