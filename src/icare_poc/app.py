"""Streamlit UI: watch each agent as it works."""

from __future__ import annotations

import html
import time

import streamlit as st

from .agents import run_agents
from .paths import catalog_path, project_root
from .search import record_count

st.set_page_config(page_title="ICARE · lookup", page_icon="📄", layout="wide")

# The reference passage wraps in place. It is not a fixed-height scroll box.
_REFERENCE_CSS = """
<style>
[data-testid="stExpander"] details,
[data-testid="stExpanderDetails"],
[data-testid="stExpander"] [data-testid="stCode"],
[data-testid="stExpander"] pre {
  overflow: visible !important;
  overflow-x: visible !important;
  max-height: none !important;
  height: auto !important;
  white-space: pre-wrap !important;
}
.icare-reference {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  overflow: visible;
  max-height: none;
  margin: 0;
}
</style>
"""

ROOT = project_root()
AGENTS = [
    ("retriever", "1 · Retrieval", "Qdrant hybrid search (dense + BM25)"),
    ("citer", "2 · PDF reference", "Exact line in the immutable PDF"),
    ("writer", "3 · Answer", "Short reformulation for the user"),
]
# Each of these is answered by a rule present in the PoC index.
INDEXED_QUESTIONS = [
    "Quelle base de scolarité accorde la règle G0013 ?",
    "Combien d'années maximum pour des études de droit comparables à une licence, règle G0015 ?",
    "Combien d'années de scolarité maximum pour le collégial professionnel au Québec ?",
    "Combien d'années pour un programme d'un an après un premier diplôme universitaire, règle G0595 ?",
    "Combien d'années pour une Licence en gestion des entreprises au Liban ?",
    "Combien d'années de scolarité pour un Bachelor of Science option Mathematics du Chu Hai College ?",
]


def _index_ready() -> tuple[bool, str]:
    if not catalog_path(ROOT).exists():
        return False, "No catalog yet. Run `icare-poc ingest` first."
    n = record_count(ROOT)
    if not n:
        return False, "Qdrant collection is empty. Run `icare-poc ingest`."
    return True, f"{n} Qdrant records indexed"


def _card(title: str, subtitle: str) -> dict:
    box = st.container(border=True)
    with box:
        st.markdown(f"**{title}**")
        st.caption(subtitle)
        status = st.empty()
        detail = st.empty()
    return {"status": status, "detail": detail}


def _reference_block(reference: dict) -> None:
    pdf = reference.get("pdf") or "PDF"
    page = reference.get("page") or "—"
    line = reference.get("line") or "—"
    passage = (reference.get("text") or reference.get("line_text") or "").strip()
    with st.expander("Reference", expanded=False):
        st.caption(f"{pdf} · page {page} · line {line}")
        if passage:
            # One HTML line, so Markdown does not turn indented rule text into a scrolling code block.
            safe = html.escape(passage).replace("\t", " ").replace("\n", "<br>")
            st.markdown(
                f"<div class='icare-reference'>{safe}</div>",
                unsafe_allow_html=True,
            )


def main() -> None:
    st.markdown(_REFERENCE_CSS, unsafe_allow_html=True)
    st.title("ICARE rule lookup")
    st.caption(
        "Three agents, in order: retrieve a Qdrant record, lock the PDF line, "
        "then reformulate. The record always traces back to the immutable PDF."
    )

    ready, info = _index_ready()
    st.caption(info)
    if not ready:
        st.stop()

    typed = st.chat_input("Ask about a rule, diploma, country, or schooling value…")
    question = typed or st.session_state.pop("pending_question", None)
    if not question:
        st.caption("Questions sur des règles indexées")
        for index, sample in enumerate(INDEXED_QUESTIONS):
            if st.button(sample, key=f"indexed-question-{index}"):
                st.session_state.pending_question = sample
                st.rerun()
        return

    st.markdown(f"**Question.** {question}")
    cols = st.columns(3)
    cards = {}
    for col, (key, title, subtitle) in zip(cols, AGENTS, strict=True):
        with col:
            cards[key] = _card(title, subtitle)
            cards[key]["status"].markdown("Waiting…")

    answer_box = st.empty()
    for event in run_agents(question, root=ROOT):
        card = cards.get(event.agent)
        if card is None:
            continue
        if event.status == "running":
            card["status"].markdown(f":blue[{event.message}]")
            card["detail"].empty()
        elif event.status == "error":
            card["status"].markdown(f":red[{event.message}]")
        elif event.agent == "retriever":
            card["status"].markdown(f"Record **`{event.detail}`**")
            extra = event.data
            card["detail"].markdown(
                f"<p style='color:#6b7280;font-size:12px;margin:0'>"
                f"score {extra.get('score')} · dense {extra.get('dense_score')} · "
                f"bm25 {extra.get('bm25_score')}<br>{extra.get('preview')}</p>",
                unsafe_allow_html=True,
            )
        elif event.agent == "citer":
            card["status"].markdown(event.message)
            card["detail"].caption("Passage kept for Reference")
        elif event.agent == "writer":
            card["status"].markdown("Done")
            card["detail"].caption(f"via {event.detail}")
            answer_box.success(event.message)
            _reference_block(event.data.get("reference") or {})
        time.sleep(0.15)


if __name__ == "__main__":
    main()
