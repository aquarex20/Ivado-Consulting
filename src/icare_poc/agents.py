"""Three-agent routing: retrieve → cite PDF line → reformulate with Ollama."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from .locate import locate_excerpt
from .models import AgentEvent, PdfLocation, RetrievalHit
from .paths import project_root
from .search import first_match


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "llama3.2")


def reformulate(question: str, hit: RetrievalHit, location: PdfLocation) -> str:
    payload = hit.payload
    body = (payload.get("text") or "").strip()
    excerpt = (location.line_text or payload.get("excerpt") or "").strip()
    passage = body or excerpt
    prompt = (
        "Tu réponds à la question en français clair. "
        "Tu t'appuies seulement sur le passage. Tu n'inventes aucun chiffre. "
        "Si le passage ne permet pas de répondre, parce qu'il ne parle pas du diplôme, "
        "du pays ou du lieu demandé, dis « Je ne sais pas » et nomme ce qui manque. "
        "N'explique pas une autre règle à la place. "
        "Si le passage répond, dis ce qu'il décide, avec tes mots, en deux ou trois phrases. "
        "Ne recopie pas les phrases du passage. "
        "Ne cite pas le fichier, la page ni la ligne. Nomme le code de règle une fois.\n\n"
        f"Question: {question}\n"
        f"Règle: {payload.get('document_id')}\n"
        f"Passage:\n{passage}\n"
    )
    req = urllib.request.Request(
        f"{_ollama_host()}/api/generate",
        data=json.dumps(
            {
                "model": _ollama_model(),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data.get("response") or "").strip()
    if not text:
        raise RuntimeError("réponse Ollama vide")
    return text


def run_agents(question: str, *, root: Path | None = None) -> Iterator[AgentEvent]:
    root = root or project_root()
    q = question.strip()
    if not q:
        yield AgentEvent("retriever", "error", "Question vide.")
        return

    yield AgentEvent(
        "retriever",
        "running",
        "Recherche hybride dans Qdrant (dense + BM25)…",
    )
    hit = first_match(q, root=root)
    if hit is None:
        yield AgentEvent("retriever", "error", "Aucun enregistrement Qdrant pour cette question.")
        return

    yield AgentEvent(
        "retriever",
        "done",
        f"Enregistrement retenu: {hit.chunk_id}",
        detail=hit.chunk_id,
        data={
            "chunk_id": hit.chunk_id,
            "document_id": hit.payload.get("document_id"),
            "score": round(hit.score, 4),
            "dense_score": round(hit.dense_score, 4),
            "bm25_score": round(hit.bm25_score, 4),
            "preview": (hit.payload.get("excerpt") or hit.payload.get("text") or "")[:280],
            "source_filename": hit.payload.get("source_filename"),
        },
    )

    yield AgentEvent(
        "citer",
        "running",
        f"Recherche de l'extrait dans {hit.payload.get('source_filename')}…",
    )
    pdf_path = root / str(hit.payload.get("source_path"))
    location = locate_excerpt(pdf_path, hit.payload.get("excerpt") or hit.payload.get("text") or "", payload=hit.payload)
    yield AgentEvent(
        "citer",
        "done",
        f"{location.source_filename} · p.{location.page} l.{location.line}",
        detail=location.line_text,
        data={
            "source_filename": location.source_filename,
            "source_hash": location.source_hash,
            "page": location.page,
            "line": location.line,
            "line_text": location.line_text,
            "method": location.method,
        },
    )

    yield AgentEvent("writer", "running", "Reformulation de la réponse…")
    try:
        answer = reformulate(q, hit, location)
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        yield AgentEvent("writer", "error", f"Ollama n'a pas répondu : {exc}")
        return

    passage = (hit.payload.get("text") or "").strip()
    yield AgentEvent(
        "writer",
        "done",
        answer,
        detail="ollama",
        data={
            "answer": answer,
            "reformulation": "ollama",
            "reference": {
                "rule": hit.payload.get("document_id"),
                "text": passage,
                "line_text": location.line_text,
                "pdf": location.source_filename,
                "page": location.page,
                "line": location.line,
            },
            "citation": {
                "rule": hit.payload.get("document_id"),
                "chunk_id": hit.chunk_id,
                "pdf": location.source_filename,
                "page": location.page,
                "line": location.line,
            },
        },
    )
