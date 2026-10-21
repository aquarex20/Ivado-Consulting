"""Build a 25-rule Phase 1 subset: PDF + Markdown + provenance + Qdrant."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from icare_rules.config import load_settings
from icare_rules.extraction import ExtractionError, extract_text
from icare_rules.inventory import InventoryEntry, build_inventory
from icare_rules.segmentation import segment_rule
from icare_rules.utils import write_jsonl

from .documents import source_hash_of, write_markdown, write_pdf, write_provenance
from .models import EvidenceChunk
from .paths import (
    MAX_RULES_DEFAULT,
    catalog_path,
    project_root,
    reference_dir,
    sources_pdf,
)
from .search import upsert_chunks

logger = logging.getLogger(__name__)

MIN_CHARS = 180
MAX_CHARS = 3500


def _inventory(root: Path) -> list[InventoryEntry]:
    settings = load_settings(project_root=root)
    settings.project_root = root
    inventory_file = settings.output_root / "inventory.jsonl"
    if inventory_file.exists():
        entries: list[InventoryEntry] = []
        for raw in inventory_file.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                entries.append(InventoryEntry(**json.loads(raw)))
        return [e for e in entries if e.kept]
    return [e for e in build_inventory(settings, hash_files=False, progress=False) if e.kept]


def _candidates(entries: list[InventoryEntry]) -> list[InventoryEntry]:
    ranked = sorted(
        entries,
        key=lambda e: (
            0 if e.extension == ".docx" else 1,
            0 if e.kind == "G" else 1,
            e.rule_id,
        ),
    )
    return ranked


def _first_line(text: str, fallback: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned and cleaned.upper() != fallback:
            return cleaned[:120]
    return fallback


def select_and_materialize(
    root: Path | None = None,
    *,
    max_rules: int = MAX_RULES_DEFAULT,
) -> list[EvidenceChunk]:
    root = root or project_root()
    settings = load_settings(project_root=root)
    settings.project_root = root
    picked: list[InventoryEntry] = []
    want_g = max_rules // 2
    want_p = max_rules - want_g

    for entry in _candidates(_inventory(root)):
        if entry.kind == "G" and sum(1 for p in picked if p.kind == "G") >= want_g:
            continue
        if entry.kind == "P" and sum(1 for p in picked if p.kind == "P") >= want_p:
            continue
        path = Path(entry.absolute_path)
        if not path.exists():
            continue
        try:
            extracted = extract_text(path, settings)
        except ExtractionError:
            continue
        if not (MIN_CHARS <= len(extracted.text) <= MAX_CHARS):
            continue
        picked.append(entry)
        if len(picked) >= max_rules:
            break

    if len(picked) < max_rules:
        raise RuntimeError(
            f"Seulement {len(picked)} règles propres trouvées (cible {max_rules})."
        )

    chunks: list[EvidenceChunk] = []
    catalog: list[dict] = []

    for entry in picked:
        path = Path(entry.absolute_path)
        extracted = extract_text(path, settings)
        category = "general" if entry.kind == "G" else "particular"
        source_name = f"{entry.rule_id}.pdf"
        pdf_path = sources_pdf(root, category=category) / source_name
        md_path = reference_dir(root, category=category) / f"{entry.rule_id}.md"
        prov_path = reference_dir(root, category=category) / f"{entry.rule_id}.provenance.json"
        source_id = f"src-{entry.rule_id}"
        title = _first_line(extracted.text, entry.rule_id)

        coords = write_pdf(extracted.text, pdf_path, title=entry.rule_id)
        source_hash = source_hash_of(pdf_path)
        write_markdown(
            md_path,
            document_id=entry.rule_id,
            category=category,
            title=title,
            text=extracted.text,
            source_filename=source_name,
            source_hash=source_hash,
            source_id=source_id,
        )

        segments = segment_rule(
            parent_rule_id=entry.rule_id,
            source_file=entry.source_file,
            source_hash=source_hash,
            text=extracted.text,
            settings=settings,
        )
        usable = [s for s in segments if s.kind == "subrule"] or [
            s for s in segments if s.kind == "file"
        ]

        evidence_meta: list[dict] = []
        for seg in usable:
            excerpt = " ".join(seg.text_normalized.split())[:240]
            line_start, line_end, page = _span_for_excerpt(coords, excerpt)
            chunk = EvidenceChunk(
                chunk_id=seg.rule_id,
                document_id=entry.rule_id,
                rule_codes=[entry.rule_id],
                category=category,
                title=title,
                text=seg.text_normalized,
                source_id=source_id,
                source_filename=source_name,
                source_hash=source_hash,
                source_path=str(pdf_path.relative_to(root)),
                markdown_path=str(md_path.relative_to(root)),
                section_id=seg.rule_id,
                evidence_unit_id=seg.rule_id,
                page=page,
                line_start=line_start,
                line_end=line_end,
                excerpt=excerpt,
            )
            chunks.append(chunk)
            evidence_meta.append(
                {
                    "evidence_unit_id": seg.rule_id,
                    "chunk_id": seg.rule_id,
                    "page": page,
                    "line_start": line_start,
                    "line_end": line_end,
                    "excerpt": excerpt,
                }
            )

        write_provenance(
            prov_path,
            document_id=entry.rule_id,
            source_id=source_id,
            source_filename=source_name,
            source_hash=source_hash,
            evidence=evidence_meta,
        )
        catalog.append(
            {
                "document_id": entry.rule_id,
                "kind": entry.kind,
                "source_file": entry.source_file,
                "pdf": str(pdf_path.relative_to(root)),
                "markdown": str(md_path.relative_to(root)),
                "chunks": len(evidence_meta),
            }
        )
        logger.info("Préparé %s (%s, %d chunks)", entry.rule_id, category, len(evidence_meta))

    write_jsonl(catalog_path(root), catalog)
    n = upsert_chunks(chunks, root=root)
    logger.info("Index Qdrant: %d chunks pour %d règles", n, len(picked))
    return chunks


def _span_for_excerpt(coords: list[dict], excerpt: str) -> tuple[int, int, int]:
    if not coords:
        return 1, 1, 1
    needle = excerpt[:80].lower()
    for row in coords:
        hay = row["text"].lower()
        if hay and (hay[:40] in needle or needle[:40] in hay):
            return row["line"], row["line"], row["page"]
    return coords[0]["line"], coords[0]["line"], coords[0]["page"]


def run_ingest(root: Path | None = None, *, max_rules: int = MAX_RULES_DEFAULT) -> list[EvidenceChunk]:
    return select_and_materialize(root, max_rules=max_rules)
