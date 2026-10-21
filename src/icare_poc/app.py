"""Streamlit UI: watch each agent as it works."""

from __future__ import annotations

import time

import streamlit as st

from .agents import run_agents
from .paths import catalog_path, project_root
from .search import record_count

st.set_page_config(page_title="ICARE · lookup", page_icon="📄", layout="wide")

ROOT = project_root()
AGENTS = [
    ("retriever", "1 · Retrieval", "Qdrant hybrid search (dense + BM25)"),
    ("citer", "2 · PDF reference", "Exact line in the immutable PDF"),
    ("writer", "3 · Answer", "Short reformulation for the user"),
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


def main() -> None:
    st.title("ICARE rule lookup")
    st.caption(
        "Three agents, in order: retrieve a Qdrant record, lock the PDF line, "
        "then reformulate. The record always traces back to the immutable PDF."
    )

    ready, info = _index_ready()
    st.caption(info)
    if not ready:
        st.stop()

    question = st.chat_input("Ask about a rule, diploma, country, or schooling value…")
    if not question:
        st.info("Example: « Combien d'années pour un Bachelor of Architecture au Québec ? »")
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
            card["detail"].markdown(
                f"<p style='color:#9ca3af;font-size:12px;margin:0'>"
                f"{event.detail}</p>",
                unsafe_allow_html=True,
            )
        elif event.agent == "writer":
            card["status"].markdown("Done")
            card["detail"].caption(f"via {event.detail}")
            answer_box.success(event.message)
        time.sleep(0.15)


if __name__ == "__main__":
    main()
