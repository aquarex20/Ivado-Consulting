"""Write the immutable PDF, reviewed Markdown, and provenance map."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas

from icare_rules.utils import sha256_file

from .paths import font_path

_FONT = "IcareUni"
_WRAP = 92
_LINE_H = 14
_MARGIN = 56


def drop_placeholder(path: Path) -> None:
    """Remove an iCloud placeholder so the next write creates a local file."""
    if not path.exists():
        return
    if getattr(path.stat(), "st_flags", 0) & 0x40000000:
        path.unlink()


def _register_font() -> None:
    if _FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(_FONT, str(font_path())))


def write_pdf(text: str, dest: Path, *, title: str) -> list[dict]:
    """Render `text` as a searchable PDF and return page/line coordinates."""
    _register_font()
    dest.parent.mkdir(parents=True, exist_ok=True)
    drop_placeholder(dest)
    page_w, page_h = letter
    c = pdf_canvas.Canvas(str(dest), pagesize=letter)
    c.setTitle(title)
    y = page_h - _MARGIN
    page = 1
    line_on_page = 0
    coords: list[dict] = []

    c.setFont(_FONT, 10)
    for raw in text.splitlines() or [""]:
        wrapped = textwrap.wrap(raw, width=_WRAP) or [""]
        for piece in wrapped:
            if y < _MARGIN:
                c.showPage()
                c.setFont(_FONT, 10)
                page += 1
                line_on_page = 0
                y = page_h - _MARGIN
            line_on_page += 1
            c.drawString(_MARGIN, y, piece)
            coords.append({"page": page, "line": line_on_page, "text": piece})
            y -= _LINE_H
    c.save()
    return coords


def write_markdown(
    dest: Path,
    *,
    document_id: str,
    category: str,
    title: str,
    text: str,
    source_filename: str,
    source_hash: str,
    source_id: str,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    drop_placeholder(dest)
    front = "\n".join(
        [
            "---",
            f"document_id: {document_id}",
            f"rule_codes: [{document_id}]",
            f"title: {json.dumps(title, ensure_ascii=False)}",
            "language: fr",
            f"category: {category}",
            f"source_id: {source_id}",
            f"source_filename: {source_filename}",
            f"source_hash: {source_hash}",
            "fidelity_status: prepared",
            "authority_status: unknown",
            "release_status: poc",
            "---",
            "",
            f"# {document_id}",
            "",
            text.strip(),
            "",
        ]
    )
    dest.write_text(front, encoding="utf-8")


def write_provenance(
    dest: Path,
    *,
    document_id: str,
    source_id: str,
    source_filename: str,
    source_hash: str,
    evidence: list[dict],
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    drop_placeholder(dest)
    dest.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "source_id": source_id,
                "source_filename": source_filename,
                "source_hash": source_hash,
                "evidence": evidence,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def source_hash_of(path: Path) -> str:
    return sha256_file(path)
