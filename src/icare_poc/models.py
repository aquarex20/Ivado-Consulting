"""Shared records for ingest, retrieval, and the Streamlit flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceChunk:
    chunk_id: str
    document_id: str
    rule_codes: list[str]
    category: str
    title: str
    text: str
    source_id: str
    source_filename: str
    source_hash: str
    source_path: str
    markdown_path: str
    section_id: str
    evidence_unit_id: str
    page: int
    line_start: int
    line_end: int
    excerpt: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    dense_score: float
    bm25_score: float
    payload: dict[str, Any]


@dataclass
class PdfLocation:
    source_filename: str
    source_path: str
    source_hash: str
    page: int
    line: int
    line_text: str
    excerpt: str
    method: str = "pdf-text-search"


@dataclass
class AgentEvent:
    agent: str
    status: str
    message: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
