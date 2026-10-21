"""Filesystem layout for the Phase 1 PoC corpus."""

from __future__ import annotations

from pathlib import Path

COLLECTION = "icare_rules_poc"
MAX_RULES_DEFAULT = 25


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def corpus_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "corpus"


def sources_pdf(root: Path | None = None, *, category: str) -> Path:
    return corpus_root(root) / "sources" / "pdf" / category


def reference_dir(root: Path | None = None, *, category: str) -> Path:
    return corpus_root(root) / "reference" / category


def derived_dir(root: Path | None = None) -> Path:
    return corpus_root(root) / "derived"


def qdrant_dir(root: Path | None = None) -> Path:
    return derived_dir(root) / "qdrant"


def catalog_path(root: Path | None = None) -> Path:
    return derived_dir(root) / "catalog.jsonl"


def font_path() -> Path:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No Unicode TTF found for PDF generation.")
