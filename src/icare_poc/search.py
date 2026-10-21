"""Local Qdrant index with hybrid dense + BM25 retrieval."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .models import EvidenceChunk, RetrievalHit
from .paths import COLLECTION, project_root, qdrant_dir

VECTOR_SIZE = 256
RRF_K = 60
RULE_ID_RE = re.compile(r"\b([GP])\s?-?(\d{3,5})\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9à-öø-ÿœæ]+", re.IGNORECASE)


def _client(root: Path | None = None) -> QdrantClient:
    path = qdrant_dir(root)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"icare:{chunk_id}"))


def tokenize(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return [t.lower() for t in _TOKEN_RE.findall(folded) if len(t) >= 2]


def embed(text: str, dim: int = VECTOR_SIZE) -> list[float]:
    """Hashed n-gram embedding — no model download, stable for a 25-rule index."""
    vec = [0.0] * dim
    tokens = tokenize(text)
    grams: list[str] = list(tokens)
    grams.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False))
    for gram in grams:
        digest = hashlib.md5(gram.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % dim
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return vec


def ensure_collection(client: QdrantClient | None = None, root: Path | None = None) -> QdrantClient:
    client = client or _client(root)
    names = {c.name for c in client.get_collections().collections}
    if COLLECTION in names:
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE),
    )
    return client


def upsert_chunks(chunks: Iterable[EvidenceChunk], root: Path | None = None) -> int:
    client = ensure_collection(root=root)
    points = [
        qm.PointStruct(
            id=point_id(chunk.chunk_id),
            vector=embed(chunk.text),
            payload=chunk.payload(),
        )
        for chunk in chunks
    ]
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def _scroll_all(client: QdrantClient) -> list[qm.Record]:
    records: list[qm.Record] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(batch)
        if offset is None:
            break
    return records


def _bm25_scores(query: str, docs: list[str]) -> list[float]:
    q_tokens = tokenize(query)
    tokenized = [tokenize(doc) for doc in docs]
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    n = len(tokenized) or 1
    avgdl = sum(len(t) for t in tokenized) / n
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for tokens in tokenized:
        tf = Counter(tokens)
        dl = len(tokens) or 1
        score = 0.0
        for term in q_tokens:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf[term] * (k1 + 1) / denom)
        scores.append(score)
    return scores


def _rrf(rankings: list[list[str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def _mentioned_rule(query: str) -> str | None:
    m = RULE_ID_RE.search(query)
    if not m:
        return None
    return f"{m.group(1).upper()}{int(m.group(2)):04d}"


def retrieve(query: str, *, root: Path | None = None, limit: int = 5) -> list[RetrievalHit]:
    """Hybrid search: exact rule-code match, then dense + BM25 fused with RRF."""
    root = root or project_root()
    client = _client(root)
    records = _scroll_all(client)
    if not records:
        return []

    mentioned = _mentioned_rule(query)
    dense = client.query_points(
        collection_name=COLLECTION,
        query=embed(query),
        limit=min(limit * 4, len(records)),
        with_payload=True,
    ).points
    dense_ids = [str(p.payload.get("chunk_id")) for p in dense if p.payload]
    dense_by_id = {str(p.payload.get("chunk_id")): float(p.score or 0.0) for p in dense if p.payload}

    texts = [str((r.payload or {}).get("text") or "") for r in records]
    bm25 = _bm25_scores(query, texts)
    bm25_pairs = sorted(
        (
            (str((r.payload or {}).get("chunk_id")), score)
            for r, score in zip(records, bm25, strict=True)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    bm25_ids = [chunk_id for chunk_id, score in bm25_pairs if score > 0]
    bm25_by_id = dict(bm25_pairs)

    fused = _rrf([dense_ids, bm25_ids or dense_ids])
    payload_by_id = {str((r.payload or {}).get("chunk_id")): (r.payload or {}) for r in records}

    if mentioned:
        for chunk_id, payload in payload_by_id.items():
            codes = [str(c) for c in payload.get("rule_codes") or []]
            if mentioned in codes or payload.get("document_id") == mentioned:
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 2.0

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    hits: list[RetrievalHit] = []
    for chunk_id, score in ranked[:limit]:
        payload = payload_by_id.get(chunk_id)
        if not payload:
            continue
        hits.append(
            RetrievalHit(
                chunk_id=chunk_id,
                score=score,
                dense_score=dense_by_id.get(chunk_id, 0.0),
                bm25_score=bm25_by_id.get(chunk_id, 0.0),
                payload=payload,
            )
        )
    return hits


def first_match(query: str, *, root: Path | None = None) -> RetrievalHit | None:
    hits = retrieve(query, root=root, limit=5)
    return hits[0] if hits else None


def record_count(root: Path | None = None) -> int:
    try:
        return int(_client(root).count(COLLECTION, exact=True).count)
    except Exception:  # noqa: BLE001 — empty or unreadable index
        return 0
