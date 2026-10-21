"""Locate a Qdrant excerpt inside the immutable PDF (Ctrl+F equivalent)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import PdfLocation


def _fold(text: str) -> str:
    nfc = unicodedata.normalize("NFKD", text)
    ascii_ = nfc.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", ascii_).strip().lower()


def _pdf_lines(path: Path) -> list[tuple[int, int, str]]:
    reader = PdfReader(str(path))
    found: list[tuple[int, int, str]] = []
    for page_i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        line_no = 0
        for line in raw.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            line_no += 1
            found.append((page_i, line_no, cleaned))
    return found


def locate_excerpt(pdf_path: Path | str, excerpt: str, *, payload: dict) -> PdfLocation:
    """Find the first PDF line that contains the retrieved excerpt.

    Falls back to the longest overlapping token window so wrapped lines still hit.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF introuvable: {path}")

    lines = _pdf_lines(path)
    needle = _fold(excerpt)
    if not needle:
        raise ValueError("extrait vide, impossible de localiser le PDF")

    ranked: list[tuple[int, PdfLocation]] = []
    for page, line, text in lines:
        folded = _fold(text)
        if not folded:
            continue
        if needle in folded:
            score = 1000 + len(folded)
        elif folded in needle and len(folded) >= 24:
            score = len(folded)
        else:
            continue
        ranked.append(
            (
                score,
                PdfLocation(
                    source_filename=payload.get("source_filename", path.name),
                    source_path=str(path),
                    source_hash=payload.get("source_hash", ""),
                    page=page,
                    line=line,
                    line_text=text,
                    excerpt=excerpt,
                ),
            )
        )
    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]

    tokens = [t for t in re.split(r"\W+", needle) if len(t) >= 4]
    if tokens:
        window = " ".join(tokens[:8])
        for page, line, text in lines:
            if window and window in _fold(text):
                return PdfLocation(
                    source_filename=payload.get("source_filename", path.name),
                    source_path=str(path),
                    source_hash=payload.get("source_hash", ""),
                    page=page,
                    line=line,
                    line_text=text,
                    excerpt=excerpt,
                    method="pdf-text-search-window",
                )

    # Last resort: stored coordinates from ingest (still verified against PDF existence).
    return PdfLocation(
        source_filename=payload.get("source_filename", path.name),
        source_path=str(path),
        source_hash=payload.get("source_hash", ""),
        page=int(payload.get("page") or 1),
        line=int(payload.get("line_start") or 1),
        line_text=excerpt,
        excerpt=excerpt,
        method="provenance-fallback",
    )
