from __future__ import annotations

from pathlib import Path

from icare_poc.agents import run_agents
from icare_poc.documents import write_pdf
from icare_poc.locate import locate_excerpt
from icare_poc.models import EvidenceChunk
from icare_poc.search import first_match, upsert_chunks


def _chunk(root: Path, *, chunk_id: str, document_id: str, text: str, pdf: Path) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        rule_codes=[document_id],
        category="general",
        title=document_id,
        text=text,
        source_id=f"src-{document_id}",
        source_filename=pdf.name,
        source_hash="abc",
        source_path=str(pdf.relative_to(root)),
        markdown_path=f"reference/{document_id}.md",
        section_id=chunk_id,
        evidence_unit_id=chunk_id,
        page=1,
        line_start=1,
        line_end=2,
        excerpt=text[:80],
    )


def test_locate_finds_pdf_line(tmp_path: Path):
    pdf = tmp_path / "G9999.pdf"
    write_pdf(
        "G9999\nOn accorde deux (2) années de scolarité.\nN.B. Exiger le relevé de notes.",
        pdf,
        title="G9999",
    )
    loc = locate_excerpt(pdf, "On accorde deux (2) années de scolarité.", payload={"source_filename": "G9999.pdf"})
    assert loc.page == 1
    assert "accorde" in loc.line_text.lower()
    assert loc.method.startswith("pdf-text-search")


def test_hybrid_prefers_mentioned_rule_code(tmp_path: Path):
    pdf_a = tmp_path / "G0001.pdf"
    pdf_b = tmp_path / "P1001.pdf"
    write_pdf("Le candidat doit totaliser 45 crédits.", pdf_a, title="G0001")
    write_pdf("Bachelor of Arts Hebrew University: on accorde deux années.", pdf_b, title="P1001")
    upsert_chunks(
        [
            _chunk(tmp_path, chunk_id="G0001", document_id="G0001", text="45 crédits universitaires", pdf=pdf_a),
            _chunk(
                tmp_path,
                chunk_id="P1001#001",
                document_id="P1001",
                text="Bachelor of Arts Hebrew University: on accorde deux années.",
                pdf=pdf_b,
            ),
        ],
        root=tmp_path,
    )
    hit = first_match("Que dit la règle P1001 pour le Bachelor of Arts ?", root=tmp_path)
    assert hit is not None
    assert hit.payload["document_id"] == "P1001"


def test_agents_return_qdrant_id_and_pdf_line(tmp_path: Path):
    pdf = tmp_path / "sources" / "pdf" / "general" / "G2553.pdf"
    pdf.parent.mkdir(parents=True)
    text = "1 unité = 1/30 d'année de scolarité qui s'ajoute à la scolarité déjà acquise."
    write_pdf(f"G2553\n{text}", pdf, title="G2553")
    upsert_chunks(
        [_chunk(tmp_path, chunk_id="G2553#002", document_id="G2553", text=text, pdf=pdf)],
        root=tmp_path,
    )
    events = list(run_agents("combien vaut une unité de scolarité ?", root=tmp_path))
    kinds = [(e.agent, e.status) for e in events]
    assert ("retriever", "done") in kinds
    assert ("citer", "done") in kinds
    assert ("writer", "done") in kinds
    retriever = next(e for e in events if e.agent == "retriever" and e.status == "done")
    citer = next(e for e in events if e.agent == "citer" and e.status == "done")
    writer = next(e for e in events if e.agent == "writer" and e.status == "done")
    assert retriever.detail == "G2553#002"
    assert citer.data["page"] >= 1
    assert "G2553" in writer.message
