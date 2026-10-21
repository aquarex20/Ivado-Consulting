"""Three-agent routing: retrieve → cite PDF line → reformulate."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from .locate import locate_excerpt
from .models import AgentEvent, PdfLocation, RetrievalHit
from .paths import project_root
from .search import first_match, tokenize


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "llama3.2")


_CLAUSE_SPLIT = re.compile(r"[\n\r]+|•|;|(?<=[.!?])\s+")
_DECISION = re.compile(
    r"\b(on accorde|il est accordé|ne devra pas dépasser|ne doit pas dépasser)\b",
    re.IGNORECASE,
)


def _clauses(text: str) -> list[str]:
    clauses: list[str] = []
    for raw in _CLAUSE_SPLIT.split(text or ""):
        clause = re.sub(r"\s+", " ", raw).strip(" \t•-*–—")
        clause = re.sub(r"^\d+\s*[-.)]\s*", "", clause)
        if len(clause) >= 24:
            clauses.append(clause)
    return clauses


def _focus(clause: str) -> str:
    """Keep the decision in a long clause, and leave the setup in the reference."""
    match = _DECISION.search(clause)
    if match and len(clause) - match.start() >= 24:
        return clause[match.start() :]
    return clause


def _best_clause(question: str, text: str, excerpt: str) -> str:
    clauses = _clauses(text) or _clauses(excerpt)
    if not clauses:
        fallback = re.sub(r"\s+", " ", excerpt or text).strip()
        return _focus(fallback)
    query_tokens = set(tokenize(question))

    def score(clause: str) -> float:
        tokens = tokenize(clause)
        if not tokens:
            return 0.0
        overlap = len(set(tokens) & query_tokens)
        return overlap / (len(set(tokens)) ** 0.5)

    ranked = sorted(clauses, key=score, reverse=True)
    positive = [clause for clause in ranked if score(clause) > 0]
    return _focus(positive[0] if positive else ranked[0])


def _as_answer(clause: str) -> str:
    """Turn a source clause into the agent's own sentence."""
    text = re.sub(r"\s+", " ", clause).strip(" .;")
    text = re.sub(r"\bon accorde\b", "il est accordé", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"\s*=\s*", " correspond à ", text, count=1)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text and text[0].isupper() and not text.startswith('"'):
        text = text[0].lower() + text[1:]
    return f"{text}."


def _reformulate_local(question: str, hit: RetrievalHit, location: PdfLocation) -> str:
    payload = hit.payload
    rule = payload.get("document_id") or hit.chunk_id
    body = (payload.get("text") or "").strip()
    excerpt = (location.line_text or payload.get("excerpt") or "").strip()
    clause = _best_clause(question, body, excerpt)
    if not clause:
        return f"La règle {rule} ne contient pas de passage répondant à cette question."
    return f"Selon la règle {rule}, {_as_answer(clause)}"


def _reformulate_ollama(question: str, hit: RetrievalHit, location: PdfLocation) -> str:
    payload = hit.payload
    body = (payload.get("text") or "").strip()
    excerpt = (location.line_text or payload.get("excerpt") or "").strip()
    passage = _best_clause(question, body, excerpt) or body
    prompt = (
        "Tu es l'agent qui répond à l'usager. Reformule la réponse avec tes propres mots, "
        "en français, en deux ou trois phrases maximum. "
        "Appuie-toi uniquement sur le passage. N'invente aucun chiffre, diplôme ou condition. "
        "Ne recopie pas le passage tel quel. "
        "N'inclus pas le nom du fichier, la page ni la ligne. "
        "Mentionne le code de règle une fois.\n\n"
        f"Question: {question}\n"
        f"Règle: {payload.get('document_id')}\n"
        f"Passage: {passage}\n"
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
    source = f"{body}\n{passage}"
    folded_answer = re.sub(r"\s+", " ", text)
    folded_source = re.sub(r"\s+", " ", source)
    if len(folded_source) > 180 and folded_source[:180] in folded_answer:
        raise RuntimeError("réponse Ollama recopiée")
    if len(folded_answer) > 400 and len(folded_answer) > 0.8 * max(len(folded_source), 1):
        raise RuntimeError("réponse Ollama trop proche du passage")
    return text


def reformulate(question: str, hit: RetrievalHit, location: PdfLocation) -> tuple[str, str]:
    try:
        return _reformulate_ollama(question, hit, location), "ollama"
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError, OSError):
        return _reformulate_local(question, hit, location), "local"


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
    answer, how = reformulate(q, hit, location)
    passage = (hit.payload.get("text") or "").strip()
    yield AgentEvent(
        "writer",
        "done",
        answer,
        detail=how,
        data={
            "answer": answer,
            "reformulation": how,
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
