"""Three-agent routing: retrieve → cite PDF line → reformulate with OpenAI."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from icare_rules.utils import utc_now_iso

from .locate import locate_excerpt
from .metrics_store import record_run
from .models import AgentEvent, PdfLocation, RetrievalHit
from .paths import project_root
from .search import retrieve, search_metrics


def _openai_model() -> str:
    load_dotenv(project_root() / ".env", override=False)
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _api_key() -> str:
    load_dotenv(project_root() / ".env", override=False)
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        raise RuntimeError("API_KEY manquante dans .env")
    return key


def _chat(prompt: str, model: str) -> tuple[str, int | None, int | None]:
    from openai import OpenAI

    response = OpenAI(api_key=_api_key(), timeout=20).chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (response.choices[0].message.content or "").strip()
    usage = response.usage
    return (
        text,
        getattr(usage, "prompt_tokens", None) if usage else None,
        getattr(usage, "completion_tokens", None) if usage else None,
    )


@dataclass
class ModelReply:
    text: str
    prompt: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ms: int | None = None
    load_duration_ms: int | None = None
    prompt_eval_duration_ms: int | None = None
    eval_duration_ms: int | None = None


def build_prompt(question: str, hit: RetrievalHit, location: PdfLocation) -> str:
    payload = hit.payload
    body = (payload.get("text") or "").strip()
    excerpt = (location.line_text or payload.get("excerpt") or "").strip()
    passage = body or excerpt
    return (
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


def reformulate(question: str, hit: RetrievalHit, location: PdfLocation) -> ModelReply:
    prompt = build_prompt(question, hit, location)
    model = _openai_model()
    started = time.perf_counter()
    text, prompt_tokens, completion_tokens = _chat(prompt, model)
    if not text:
        raise RuntimeError("réponse OpenAI vide")
    return ModelReply(
        text=text,
        prompt=prompt,
        model=model,
        latency_ms=int((time.perf_counter() - started) * 1000),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _blank_query_log() -> dict:
    return {
        "at": utc_now_iso(),
        "pipeline": "qualitative",
        "status": "retriever_error",
        "question": None,
        "model_prompt": None,
        "answer": None,
        "latency_ms": None,
        "model_latency_ms": None,
        "model": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_duration_ms": None,
        "load_duration_ms": None,
        "prompt_eval_duration_ms": None,
        "eval_duration_ms": None,
        "chunk_id": None,
        "document_id": None,
        "score": None,
        "second_score": None,
        "dense_score": None,
        "bm25_score": None,
        "citation_method": None,
    }


def run_agents(question: str, *, root: Path | None = None) -> Iterator[AgentEvent]:
    root = root or project_root()
    record = _blank_query_log()
    started = time.perf_counter()
    try:
        yield from _run_agents(question, root=root, record=record)
    finally:
        record["latency_ms"] = int((time.perf_counter() - started) * 1000)
        record_run(root, record)


def _apply_reply(record: dict, reply: ModelReply) -> None:
    record.update(
        model_prompt=reply.prompt,
        answer=reply.text,
        model=reply.model,
        model_latency_ms=reply.latency_ms,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        total_duration_ms=reply.total_duration_ms,
        load_duration_ms=reply.load_duration_ms,
        prompt_eval_duration_ms=reply.prompt_eval_duration_ms,
        eval_duration_ms=reply.eval_duration_ms,
    )


def _run_agents(question: str, *, root: Path, record: dict) -> Iterator[AgentEvent]:
    q = question.strip()
    record["question"] = q
    if not q:
        yield AgentEvent("retriever", "error", "Question vide.")
        return

    yield AgentEvent(
        "retriever",
        "running",
        "Recherche hybride dans Qdrant (dense + BM25)…",
    )
    hits = retrieve(q, root=root, limit=5)
    metrics = search_metrics(hits)
    record["score_delta"] = metrics["score_delta"]
    record["zero_lexical"] = metrics["zero_lexical"]
    hit = hits[0] if hits else None
    if hit is None:
        yield AgentEvent("retriever", "error", "Aucun enregistrement Qdrant pour cette question.")
        return

    score = round(hit.score, 4)
    dense_score = round(hit.dense_score, 4)
    bm25_score = round(hit.bm25_score, 4)
    record.update(
        status="retrieved",
        chunk_id=hit.chunk_id,
        document_id=hit.payload.get("document_id"),
        score=metrics["score"],
        second_score=metrics["second_score"],
        dense_score=metrics["dense_score"],
        bm25_score=metrics["bm25_score"],
    )
    yield AgentEvent(
        "retriever",
        "done",
        f"Enregistrement retenu: {hit.chunk_id}",
        detail=hit.chunk_id,
        data={
            "chunk_id": hit.chunk_id,
            "document_id": hit.payload.get("document_id"),
            "score": score,
            "dense_score": dense_score,
            "bm25_score": bm25_score,
            "second_score": metrics["second_score"],
            "score_delta": metrics["score_delta"],
            "zero_lexical": metrics["zero_lexical"],
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
    record["citation_method"] = location.method
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
    model_started = time.perf_counter()
    try:
        reply = reformulate(q, hit, location)
    except Exception as exc:
        record["model"] = _openai_model()
        record["model_prompt"] = build_prompt(q, hit, location)
        record["model_latency_ms"] = int((time.perf_counter() - model_started) * 1000)
        record["status"] = "writer_error"
        yield AgentEvent("writer", "error", f"OpenAI n'a pas répondu : {exc}")
        return

    _apply_reply(record, reply)
    record["status"] = "done"
    answer = reply.text

    passage = (hit.payload.get("text") or "").strip()
    yield AgentEvent(
        "writer",
        "done",
        answer,
        detail="openai",
        data={
            "answer": answer,
            "reformulation": "openai",
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
