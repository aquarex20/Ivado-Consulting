from __future__ import annotations

import sqlite3
from pathlib import Path

from icare_poc.agents import run_agents
from icare_poc.chat_store import (
    add_message,
    conversation_title,
    create_conversation,
    delete_conversation,
    import_legacy_runs,
    list_conversations,
    list_messages,
    turn_from_events,
)
from icare_poc.documents import write_pdf
from icare_poc.metrics_store import db_path
from icare_poc.models import AgentEvent, EvidenceChunk
from icare_poc.search import upsert_chunks


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


def test_conversation_keeps_messages_and_search(tmp_path: Path):
    first = create_conversation(tmp_path, conversation_title("  Combien   d'années ?  "))
    add_message(tmp_path, first, role="user", content="Combien d'années ?")
    add_message(
        tmp_path,
        first,
        role="assistant",
        content="Deux années.",
        status="done",
        payload={"reference": {"pdf": "P.pdf", "page": 2, "line": 4, "text": "On accorde deux années."}},
        created_at="2026-10-01T12:00:00+00:00",
    )
    second = create_conversation(tmp_path, "Autre question")
    add_message(tmp_path, second, role="user", content="Que dit G0013 ?", created_at="2026-10-02T12:00:00+00:00")

    listed = list_conversations(tmp_path)
    assert [item.id for item in listed] == [second, first]
    messages = list_messages(tmp_path, first)
    assert [item.role for item in messages] == ["user", "assistant"]
    assert messages[1].reference["text"] == "On accorde deux années."
    assert list_conversations(tmp_path, query="deux années")[0].id == first
    assert list_conversations(tmp_path, query="100%") == []

    delete_conversation(tmp_path, first)
    assert list_messages(tmp_path, first) == []
    with sqlite3.connect(db_path(tmp_path)) as conn:
        left = conn.execute("SELECT count(*) FROM messages WHERE conversation_id = ?", (first,)).fetchone()[0]
    assert left == 0


def test_legacy_runs_import_once(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "sources" / "pdf" / "general" / "G2553.pdf"
    pdf.parent.mkdir(parents=True)
    text = "1 unité = 1/30 d'année de scolarité."
    write_pdf(f"G2553\n{text}", pdf, title="G2553")
    upsert_chunks(
        [_chunk(tmp_path, chunk_id="G2553#002", document_id="G2553", text=text, pdf=pdf)],
        root=tmp_path,
    )
    answer = "Selon G2553, une unité vaut un trentième d'année."
    monkeypatch.setattr("icare_poc.agents._chat", lambda prompt, model: (answer, 3, 2))
    question = "combien vaut une unité ?"
    events = list(run_agents(question, root=tmp_path))
    turn = turn_from_events(events)

    assert import_legacy_runs(tmp_path) == 1
    assert import_legacy_runs(tmp_path) == 0
    conversation = list_conversations(tmp_path)[0]
    messages = list_messages(tmp_path, conversation.id)
    assert messages[0].content == question
    assert messages[1].content == answer
    assert messages[1].status == "done"
    assert messages[1].source_run_id == turn["run_id"]
    assert messages[1].citation["rule"] == "G2553"
    assert turn["payload"]["reference"]["text"] == text


def test_saved_turn_is_not_imported_again(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "sources" / "pdf" / "general" / "G2553.pdf"
    pdf.parent.mkdir(parents=True)
    text = "1 unité = 1/30 d'année de scolarité."
    write_pdf(f"G2553\n{text}", pdf, title="G2553")
    upsert_chunks(
        [_chunk(tmp_path, chunk_id="G2553#002", document_id="G2553", text=text, pdf=pdf)],
        root=tmp_path,
    )
    monkeypatch.setattr("icare_poc.agents._chat", lambda prompt, model: ("Deux années.", 3, 2))
    events = list(run_agents("combien vaut une unité ?", root=tmp_path))
    turn = turn_from_events(events)
    conversation_id = create_conversation(tmp_path, "combien vaut une unité ?")
    add_message(tmp_path, conversation_id, role="user", content="combien vaut une unité ?")
    add_message(
        tmp_path,
        conversation_id,
        role="assistant",
        content=turn["content"],
        status=turn["status"],
        payload=turn["payload"],
        source_run_id=turn["run_id"],
    )

    assert import_legacy_runs(tmp_path) == 0
    assert len(list_conversations(tmp_path)) == 1


def test_turn_from_events_keeps_the_pdf_reference():
    events = [
        AgentEvent("retriever", "done", "ok", data={"chunk_id": "G1", "document_id": "G1"}),
        AgentEvent(
            "writer",
            "done",
            "Réponse.",
            data={"reference": {"pdf": "G1.pdf", "text": "passage"}, "citation": {"rule": "G1"}},
        ),
        AgentEvent("log", "done", "enregistré", data={"run_id": 7}),
    ]
    turn = turn_from_events(events)
    assert turn["content"] == "Réponse."
    assert turn["status"] == "done"
    assert turn["run_id"] == 7
    assert turn["payload"]["reference"]["text"] == "passage"
