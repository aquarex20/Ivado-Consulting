"""Conversations du PoC, dans le même SQLite que les métriques de requêtes.

Chaque discussion est une suite de messages. Le passage cité et la citation
sont gardés dans le message, pour réafficher la discussion sans relancer
les agents. ``source_run_id`` relie la réponse à la ligne de ``query_runs``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icare_rules.utils import utc_now_iso

from .metrics_store import db_path
from .models import AgentEvent

_LEGACY_STATUS_TEXT = {
    "retriever_error": "Aucun enregistrement pour cette question.",
    "writer_error": "Le modèle n'a pas répondu.",
    "retrieved": "Recherche interrompue avant la réponse.",
}


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ChatMessage:
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: str
    status: str | None
    payload: dict[str, Any]
    source_run_id: int | None

    @property
    def reference(self) -> dict[str, Any] | None:
        value = self.payload.get("reference")
        return value if isinstance(value, dict) else None

    @property
    def citation(self) -> dict[str, Any] | None:
        value = self.payload.get("citation")
        return value if isinstance(value, dict) else None


def conversation_title(question: str, limit: int = 80) -> str:
    compact = " ".join(question.split()) or "Discussion"
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def create_conversation(root: Path, title: str, *, created_at: str | None = None) -> int:
    stamp = created_at or utc_now_iso()

    def write(conn: sqlite3.Connection) -> int:
        cursor = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, stamp, stamp),
        )
        return int(cursor.lastrowid)

    return _run(root, write)


def get_conversation(root: Path, conversation_id: int) -> Conversation | None:
    def read(conn: sqlite3.Connection) -> Conversation | None:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return _conversation(row) if row else None

    return _run(root, read)


def list_conversations(root: Path, *, query: str = "") -> list[Conversation]:
    needle = " ".join(query.split())
    like = f"%{_escape_like(needle)}%" if needle else "%"

    def read(conn: sqlite3.Connection) -> list[Conversation]:
        rows = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE ? = ''
               OR title LIKE ? ESCAPE '\\'
               OR id IN (
                    SELECT conversation_id FROM messages
                    WHERE content LIKE ? ESCAPE '\\'
               )
            ORDER BY updated_at DESC, id DESC
            """,
            (needle, like, like),
        ).fetchall()
        return [_conversation(row) for row in rows]

    return _run(root, read)


def list_messages(root: Path, conversation_id: int) -> list[ChatMessage]:
    def read(conn: sqlite3.Connection) -> list[ChatMessage]:
        rows = conn.execute(
            """
            SELECT id, conversation_id, role, content, created_at, status, payload_json, source_run_id
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
        return [_message(row) for row in rows]

    return _run(root, read)


def add_message(
    root: Path,
    conversation_id: int,
    *,
    role: str,
    content: str,
    status: str | None = None,
    payload: dict[str, Any] | None = None,
    source_run_id: int | None = None,
    created_at: str | None = None,
) -> int:
    if role not in {"user", "assistant"}:
        raise ValueError(f"rôle inconnu: {role}")
    stamp = created_at or utc_now_iso()
    encoded = json.dumps(payload or {}, ensure_ascii=False)

    def write(conn: sqlite3.Connection) -> int:
        cursor = conn.execute(
            """
            INSERT INTO messages (
              conversation_id, role, content, created_at, status, payload_json, source_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, stamp, status, encoded, source_run_id),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (stamp, conversation_id),
        )
        return int(cursor.lastrowid)

    return _run(root, write)


def delete_conversation(root: Path, conversation_id: int) -> None:
    def write(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    _run(root, write)


def import_legacy_runs(root: Path) -> int:
    """Crée une discussion par requête déjà mesurée et pas encore importée."""

    def write(conn: sqlite3.Connection) -> int:
        if not _table_exists(conn, "query_runs"):
            return 0
        rows = conn.execute(
            """
            SELECT id, at, status, question, answer, document_id, chunk_id
            FROM query_runs
            WHERE question IS NOT NULL
              AND trim(question) != ''
              AND id NOT IN (
                    SELECT source_run_id FROM messages WHERE source_run_id IS NOT NULL
              )
            ORDER BY id
            """
        ).fetchall()
        imported = 0
        for row in rows:
            question = str(row["question"]).strip()
            stamp = row["at"] or utc_now_iso()
            cursor = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (conversation_title(question), stamp, stamp),
            )
            conversation_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO messages (
                  conversation_id, role, content, created_at, status, payload_json, source_run_id
                ) VALUES (?, 'user', ?, ?, NULL, '{}', NULL)
                """,
                (conversation_id, question, stamp),
            )
            citation: dict[str, Any] = {}
            if row["document_id"] or row["chunk_id"]:
                citation = {"rule": row["document_id"], "chunk_id": row["chunk_id"]}
            payload = json.dumps({"citation": citation} if citation else {}, ensure_ascii=False)
            content = (row["answer"] or "").strip() or _LEGACY_STATUS_TEXT.get(
                row["status"] or "",
                "Requête enregistrée sans réponse.",
            )
            conn.execute(
                """
                INSERT INTO messages (
                  conversation_id, role, content, created_at, status, payload_json, source_run_id
                ) VALUES (?, 'assistant', ?, ?, ?, ?, ?)
                """,
                (conversation_id, content, stamp, row["status"], payload, row["id"]),
            )
            imported += 1
        return imported

    return _run(root, write)


def turn_from_events(events: list[AgentEvent]) -> dict[str, Any]:
    """Réduit le flux d'agents au message à enregistrer."""
    retrieval = None
    citation = None
    reference = None
    content = ""
    status = "error"
    run_id = None
    for event in events:
        if event.agent == "log" and event.status == "done":
            run_id = event.data.get("run_id")
            continue
        if event.agent == "retriever" and event.status == "done":
            retrieval = event.data
        elif event.agent == "retriever" and event.status == "error":
            content = event.message
            status = "retriever_error"
        elif event.agent == "citer" and event.status == "error":
            content = event.message
            status = "error"
        elif event.agent == "writer" and event.status == "error":
            content = event.message
            status = "writer_error"
        elif event.agent == "writer" and event.status == "done":
            content = event.message
            status = "done"
            reference = event.data.get("reference")
            citation = event.data.get("citation") or citation
    return {
        "content": content or "Aucune réponse.",
        "status": status,
        "payload": {
            "reference": reference,
            "citation": citation,
            "retrieval": retrieval,
        },
        "run_id": run_id,
    }


def _conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=int(row["id"]),
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _message(row: sqlite3.Row) -> ChatMessage:
    raw = row["payload_json"] or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return ChatMessage(
        id=int(row["id"]),
        conversation_id=int(row["conversation_id"]),
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        status=row["status"],
        payload=payload,
        source_run_id=row["source_run_id"],
    )


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
          id INTEGER PRIMARY KEY,
          title TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY,
          conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
          role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
          content TEXT NOT NULL,
          created_at TEXT NOT NULL,
          status TEXT,
          payload_json TEXT,
          source_run_id INTEGER UNIQUE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS conversations_updated ON conversations(updated_at)")


def _run(root: Path, fn):
    path = db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        _ensure(conn)
        result = fn(conn)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
