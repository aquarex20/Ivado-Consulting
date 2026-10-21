"""Fixtures partagées : mini-corpus, settings, backend LLM factice."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from icare_rules.config import Settings, load_settings
from icare_rules.llm_client import LLMBackend

MINI_SETTINGS: dict[str, Any] = {
    "corpus": {
        "roots": ["corpus/G", "corpus/P"],
        "include_globs": ["*.docx"],
        "exclude_prefixes": ["~$"],
    },
    "extraction": {"strategies": ["python-docx"], "timeout_seconds": 10},
    "segmentation": {"min_segment_chars": 20, "max_subrules_per_file": 50},
    "classification": {
        "deployment": "test-deployment",
        "temperature": 0.0,
        "max_output_tokens": 300,
        "max_concurrency": 1,
        "max_retries": 1,
        "low_confidence_threshold": 0.6,
        "prompt_version": "test",
        "prompt_file": "config/classification_prompt.md",
    },
    "output": {
        "root": "data/output",
        "cache_dir": "data/cache",
        "checkpoint_every": 5,
    },
    "pilot": {"sample_size_g": 1, "sample_size_p": 1, "random_seed": 1},
}


def _make_docx(path: Path, paragraphs: list[str]) -> None:
    from docx import Document

    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """Crée un mini-projet complet dans un tmp_path."""
    (tmp_path / "corpus" / "G").mkdir(parents=True)
    (tmp_path / "corpus" / "P").mkdir(parents=True)

    # G0001.docx : règle générale avec un unique segment (aggregat file)
    _make_docx(
        tmp_path / "corpus" / "G" / "G0001.docx",
        [
            "G0001",
            "Le candidat doit avoir cumulé au moins 45 crédits universitaires "
            "pour être considéré éligible.",
        ],
    )
    # G0001.doc : doublon d'extension à ignorer
    _make_docx(
        tmp_path / "corpus" / "G" / "G0001.doc",
        ["G0001 ancien format legacy"],
    )
    # P0100.docx : règle particulière avec 3 sous-règles numérotées
    _make_docx(
        tmp_path / "corpus" / "P" / "P0100.docx",
        [
            "P0100",
            "1  Certificat X précédé du Cours secondaire : on accorde onze (11) années de scolarité.",
            "2  Certificat Y accompagné du Diplôme Z : on accorde treize (13) années de scolarité.",
            "3- Diplôme W équivalent au domaine de l'éducation ; totaliser les crédits pour dépasser 60.",
        ],
    )

    # Config + prompt + schema
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.yaml").write_text(
        yaml.safe_dump(MINI_SETTINGS, allow_unicode=True), encoding="utf-8"
    )
    project_root_source = Path(__file__).resolve().parent.parent
    shutil.copy(
        project_root_source / "config" / "classification_prompt.md",
        tmp_path / "config" / "classification_prompt.md",
    )
    (tmp_path / "schemas").mkdir()
    shutil.copy(
        project_root_source / "schemas" / "classification.schema.json",
        tmp_path / "schemas" / "classification.schema.json",
    )
    return tmp_path


@pytest.fixture()
def settings(project: Path) -> Settings:
    return load_settings(
        settings_path=project / "config" / "settings.yaml",
        project_root=project,
    )


class FakeLLMBackend(LLMBackend):
    """Backend factice : classe selon des mots-clés simples pour les tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, *, system: str, user: str, deployment: str
    ) -> tuple[str, dict[str, int]]:
        import json

        payload = json.loads(user)
        text = payload["text"].lower()
        self.calls.append({"rule_id": payload["rule_id"], "deployment": deployment})
        if "totaliser" in text or "dépasser" in text or "60" in text:
            category = "Hybride"
            confidence = 0.85
        elif "équivalent" in text or "on accorde" in text:
            category = "Qualitative"
            confidence = 0.75
        elif "crédits" in text or "45" in text:
            category = "Déterministe"
            confidence = 0.9
        else:
            category = "Qualitative"
            confidence = 0.5
        response = {
            "category": category,
            "justification": "Test.",
            "evidence": [payload["text"][:60]],
            "confidence": confidence,
            "manual_review": confidence < 0.6,
            "theme": "test " + category.lower(),
            "keywords": ["scolarité", category.lower()],
            "countries": [],
            "granted_value": None,
            "dossier_variables": ["diplome_obtenu"],
        }
        return json.dumps(response, ensure_ascii=False), {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
        }


@pytest.fixture()
def fake_backend() -> FakeLLMBackend:
    return FakeLLMBackend()
