from __future__ import annotations

import json
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


class _OllamaResponse:
    def __init__(self, answer: str):
        self._raw = json.dumps({"response": answer}).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _OllamaResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _patch_ollama(monkeypatch, answer: str, captured: dict) -> None:
    def urlopen(req, timeout=0):
        captured["prompt"] = json.loads(req.data.decode())["prompt"]
        return _OllamaResponse(answer)

    monkeypatch.setattr("icare_poc.agents.urllib.request.urlopen", urlopen)


def test_agents_return_qdrant_id_and_pdf_line(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "sources" / "pdf" / "general" / "G2553.pdf"
    pdf.parent.mkdir(parents=True)
    text = "1 unité = 1/30 d'année de scolarité qui s'ajoute à la scolarité déjà acquise."
    write_pdf(f"G2553\n{text}", pdf, title="G2553")
    upsert_chunks(
        [_chunk(tmp_path, chunk_id="G2553#002", document_id="G2553", text=text, pdf=pdf)],
        root=tmp_path,
    )
    answer = "Selon G2553, une unité vaut un trentième d'année de scolarité."
    captured: dict = {}
    _patch_ollama(monkeypatch, answer, captured)
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
    assert writer.message == answer
    assert writer.detail == "ollama"
    assert writer.data["reference"]["text"] == text
    assert text in captured["prompt"]
    assert "G2553" in captured["prompt"]


def test_writer_reformulates_instead_of_pasting_the_rule(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "sources" / "pdf" / "particular" / "P9000.pdf"
    pdf.parent.mkdir(parents=True)
    filler = "Ce diplôme est décerné par un établissement énuméré ci-après. " * 6
    answer_clause = (
        "On accorde, pour ce Bachelor of Architecture, deux (2) années de scolarité "
        "qui s'ajoutent à la scolarité déjà acquise."
    )
    text = filler + answer_clause + " " + filler
    write_pdf(f"P9000\n{text}", pdf, title="P9000")
    upsert_chunks(
        [_chunk(tmp_path, chunk_id="P9000#001", document_id="P9000", text=text, pdf=pdf)],
        root=tmp_path,
    )
    answer = "La règle P9000 accorde deux années pour ce bachelor d'architecture."
    captured: dict = {}
    _patch_ollama(monkeypatch, answer, captured)
    writer = next(
        event
        for event in run_agents("Combien d'années pour un Bachelor of Architecture ?", root=tmp_path)
        if event.agent == "writer" and event.status == "done"
    )
    assert writer.message == answer
    assert filler.strip() not in writer.message
    assert text.strip() in captured["prompt"]
    assert writer.data["reference"]["text"] == text.strip()


def test_writer_says_when_the_reference_is_about_something_else(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "sources" / "pdf" / "particular" / "P2175.pdf"
    pdf.parent.mkdir(parents=True)
    text = (
        '"Credit" = 1 unité\n'
        "Lorsque les études précédant le diplôme valent douze (12) années de scolarité ou plus "
        "et comportent de la formation générale, on ne devra pas dépasser deux (2) années de scolarité.\n"
        '"Bachelor of Arts" ou "Bachelor of Science", obtenu à la suite de la réussite du programme; '
        "on accorde, pour ce bachelor, la portion du programme qui ne fait pas l'objet d'un recouvrement "
        "avec les études antérieures."
    )
    write_pdf(text, pdf, title="P2175")
    upsert_chunks(
        [_chunk(tmp_path, chunk_id="P2175#003", document_id="P2175", text=text, pdf=pdf)],
        root=tmp_path,
    )
    answer = "Je ne sais pas. Le passage ne parle pas d'un Bachelor of Architecture au Québec."
    captured: dict = {}
    _patch_ollama(monkeypatch, answer, captured)
    question = "Combien d'années pour un Bachelor of Architecture au Québec ?"
    writer = next(
        event
        for event in run_agents(question, root=tmp_path)
        if event.agent == "writer" and event.status == "done"
    )
    assert writer.message == answer
    assert question in captured["prompt"]
    assert "Je ne sais pas" in captured["prompt"]
    assert "Bachelor of Arts" not in writer.message
