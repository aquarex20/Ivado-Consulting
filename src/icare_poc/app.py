"""Streamlit UI: a short answer, and the PDF line that supports it."""

from __future__ import annotations

import html
import time
from datetime import datetime

import streamlit as st

from .agents import run_agents
from .chat_store import (
    ChatMessage,
    Conversation,
    add_message,
    conversation_title,
    create_conversation,
    delete_conversation,
    get_conversation,
    import_legacy_runs,
    list_conversations,
    list_messages,
    turn_from_events,
)
from .paths import catalog_path, project_root
from .search import record_count
from .theme import DEFAULT_THEME, css, normalize_theme

_MONTHS = ("jan", "fév", "mar", "avr", "mai", "jun", "jul", "aoû", "sep", "oct", "nov", "déc")

st.set_page_config(
    page_title="ICARE",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

ROOT = project_root()
# Visible trail, Claude-style. Keys are display only; agents still drive the real work.
THINKING = (
    ("think", "Réflexion"),
    ("analyse", "Analyse"),
    ("lecture", "Lecture"),
    ("test", "Exécution de test"),
    ("write", "Rédaction"),
)
_ICON_PATHS = {
    "think": '<circle cx="8" cy="8" r="2.2"/><path d="M8 2.4v1.4M8 12.2v1.4M2.4 8h1.4M12.2 8h1.4M4 4l1 1M11 11l1 1M12 4l-1 1M5 11l-1 1"/>',
    "analyse": '<circle cx="7.2" cy="7.2" r="3.3"/><path d="M12.4 12.4 9.6 9.6"/>',
    "lecture": '<path d="M3 4.2h6.2a2 2 0 0 1 2 2v6.4H5a2 2 0 0 0-2 2V4.2z"/><path d="M13 4.2H6.8a2 2 0 0 0-2 2v6.4H11a2 2 0 0 1 2 2V4.2z"/>',
    "test": '<path d="M6.2 2.6h3.6M8 2.6v4.4L12.2 13a1.1 1.1 0 0 1-.9 1.8H4.7a1.1 1.1 0 0 1-.9-1.8L8 7"/>',
    "write": '<path d="M10.8 3.2 12.8 5.2 6 12H4v-2z"/><path d="M9.4 4.6 11.4 6.6"/>',
}
DOSSIERS = (
    {
        "id": "HAD-2026-014",
        "name": "Lina Haddad",
        "diploma": "Licence en gestion des entreprises",
        "country": "Liban",
        "institution": "Université libanaise",
        "object": "Base de scolarité",
        "status": "En analyse",
        "pieces": ("Diplôme", "Relevé de notes", "Pièce d'identité"),
        "question": "Combien d'années pour une Licence en gestion des entreprises au Liban ?",
    },
    {
        "id": "CHE-2026-008",
        "name": "James Chen",
        "diploma": "Bachelor of Science, Mathematics",
        "country": "Hong Kong",
        "institution": "Chu Hai College",
        "object": "Années à accorder",
        "status": "Reçu",
        "pieces": ("Diplôme", "Relevé", "Descriptif de programme"),
        "question": "Combien d'années de scolarité pour un Bachelor of Science option Mathematics du Chu Hai College ?",
    },
    {
        "id": "BER-2026-021",
        "name": "Camille Bergeron",
        "diploma": "DEC, techniques administratives",
        "country": "Québec",
        "institution": "Cégep",
        "object": "Scolarité maximale",
        "status": "En attente de pièce",
        "pieces": ("Diplôme", "Relevé de notes"),
        "question": "Combien d'années de scolarité maximum pour le collégial professionnel au Québec ?",
    },
)
def _index_ready() -> tuple[bool, str]:
    if not catalog_path(ROOT).exists():
        return False, "Le catalogue n'est pas encore prêt. Lancez `icare-poc ingest`."
    n = record_count(ROOT)
    if not n:
        return False, "L'index est vide. Lancez `icare-poc ingest`."
    return True, f"{n} extraits indexés"


def _format_when(iso: str) -> str:
    try:
        stamp = datetime.fromisoformat(iso)
    except ValueError:
        return iso[:10]
    return f"{stamp.day} {_MONTHS[stamp.month - 1]} {stamp.year}"


def _sidebar_label(conversation: Conversation) -> str:
    title = conversation.title
    if len(title) > 42:
        title = title[:41].rstrip() + "…"
    return f"{title}  ·  {_format_when(conversation.updated_at)}"


def _apply_theme() -> str:
    theme = normalize_theme(st.session_state.get("theme"))
    st.session_state.theme = theme
    st.markdown(css(theme), unsafe_allow_html=True)
    return theme


def _render_header(theme: str) -> None:
    brand, tools = st.columns([4.2, 1.3])
    with brand:
        st.markdown(
            "<div class='icare-brand'>"
            "<p class='icare-name'>ICARE</p>"
            "<p class='icare-tagline'>Évaluation des dossiers</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with tools:
        with st.container(key="theme_switch", horizontal=True, horizontal_alignment="right", gap="small"):
            if st.button(
                "☀",
                help="Thème clair",
                key="theme-clair",
                type="primary" if theme == "clair" else "tertiary",
            ):
                st.session_state.theme = "clair"
                st.rerun()
            if st.button(
                "☾",
                help="Thème sombre",
                key="theme-sombre",
                type="primary" if theme == "sombre" else "tertiary",
            ):
                st.session_state.theme = "sombre"
                st.rerun()


def _dossier(dossier_id: str | None) -> dict | None:
    return next((item for item in DOSSIERS if item["id"] == dossier_id), None)


def _open_dossier(dossier_id: str) -> None:
    st.session_state.dossier_id = dossier_id
    st.session_state.conversation_id = None
    st.rerun()


def _render_sidebar() -> None:
    current = st.session_state.get("conversation_id")
    active = st.session_state.get("dossier_id")
    with st.sidebar:
        st.markdown("<p class='icare-side-title'>Dossiers</p>", unsafe_allow_html=True)
        for dossier in DOSSIERS:
            selected = dossier["id"] == active and current is None
            rail, body = st.columns([0.06, 0.94], gap="small")
            with rail:
                st.markdown(
                    "<div class='icare-rail'></div>" if selected else "<div class='icare-rail-off'></div>",
                    unsafe_allow_html=True,
                )
            with body:
                if st.button(
                    f"{dossier['name']}\n{dossier['id']}",
                    key=f"open-dossier-{dossier['id']}",
                    use_container_width=True,
                ):
                    _open_dossier(dossier["id"])
        st.markdown("<p class='icare-side-title icare-side-next'>Analyses</p>", unsafe_allow_html=True)
        query = st.text_input(
            "Rechercher",
            placeholder="Rechercher une analyse",
            label_visibility="collapsed",
        )
        conversations = list_conversations(ROOT, query=query)
        if not conversations:
            st.caption("Aucune analyse ne correspond." if query.strip() else "Aucune analyse enregistrée.")
        for conversation in conversations:
            selected = conversation.id == current
            rail, body = st.columns([0.06, 0.94], gap="small")
            with rail:
                st.markdown(
                    "<div class='icare-rail'></div>" if selected else "<div class='icare-rail-off'></div>",
                    unsafe_allow_html=True,
                )
            with body:
                if st.button(
                    _sidebar_label(conversation),
                    key=f"open-conversation-{conversation.id}",
                    use_container_width=True,
                ):
                    st.session_state.conversation_id = conversation.id
                    st.session_state.dossier_id = None
                    st.rerun()
        if current is not None:
            st.divider()
            if st.button("Retirer cette analyse", use_container_width=True):
                delete_conversation(ROOT, current)
                st.session_state.conversation_id = None
                st.rerun()


def _source_card(reference: dict | None, citation: dict | None = None) -> None:
    ref = reference or {}
    cite = citation or {}
    rule = ref.get("rule") or cite.get("rule")
    pdf = ref.get("pdf") or cite.get("pdf")
    page = ref.get("page") or cite.get("page")
    line = ref.get("line") or cite.get("line")
    passage = (ref.get("text") or ref.get("line_text") or "").strip()
    bits = [str(rule)] if rule else []
    if pdf:
        bits.append(str(pdf))
    if page not in (None, "", "—"):
        bits.append(f"page {page}")
    if line not in (None, "", "—"):
        bits.append(f"ligne {line}")
    meta = "  ·  ".join(bits) if bits else "Source du Manuel"
    if passage:
        body = (
            f"<div class='icare-reference'>"
            f"{html.escape(passage).replace(chr(9), ' ').replace(chr(10), '<br>')}"
            f"</div>"
        )
    else:
        body = "<p class='icare-source-empty'>La règle citée n'a pas encore de passage joint.</p>"
    st.markdown(
        "<div class='icare-source'>"
        "<p class='icare-source-kicker'>Source</p>"
        f"<p class='icare-source-meta'>{html.escape(meta)}</p>"
        f"{body}"
        "</div>",
        unsafe_allow_html=True,
    )


def _icon(name: str) -> str:
    path = _ICON_PATHS.get(name, "")
    return f"<span class='icare-think-ico'><svg viewBox='0 0 16 16' aria-hidden='true'>{path}</svg></span>"


def _thinking_html(states: dict[str, str]) -> str:
    current = next((label for key, label in THINKING if states.get(key) == "running"), None)
    items = []
    for key, label in THINKING:
        state = states.get(key)
        if state is None:
            continue
        kind = {"running": "is-run", "error": "is-err", "done": "is-done"}.get(state, "")
        items.append(
            f"<li class='{kind}'>{_icon(key)}<span>{html.escape(label)}</span></li>"
        )
    now = (
        f"<div class='icare-think-now'><span class='icare-orb'></span>"
        f"<span class='icare-think-label'>{html.escape(current)}</span></div>"
        if current
        else ""
    )
    return f"<div class='icare-think'>{now}<ul class='icare-think-trail'>{''.join(items)}</ul></div>"


def _finished_thinking(states: dict[str, str]) -> list[dict[str, str]]:
    return [{"key": key, "label": label} for key, label in THINKING if states.get(key) == "done"]


def _thinking_block(payload: dict) -> None:
    steps = payload.get("thinking")
    if not isinstance(steps, list) or not steps:
        return
    ms = payload.get("thinking_ms")
    seconds = max(1, int(round((ms or 0) / 1000)))
    items = "".join(
        f"<li class='is-done'>{_icon(str(step.get('key')))}<span>{html.escape(str(step.get('label') or ''))}</span></li>"
        for step in steps
        if isinstance(step, dict)
    )
    with st.expander(f"Réflexion · {seconds} s", expanded=False):
        st.markdown(f"<ul class='icare-think-trail'>{items}</ul>", unsafe_allow_html=True)


def _technical_block(retrieval: dict | None) -> None:
    if not retrieval:
        return
    with st.expander("Détail technique", expanded=False):
        st.caption(
            f"{retrieval.get('chunk_id') or '—'}  ·  "
            f"score {retrieval.get('score')}  ·  "
            f"dense {retrieval.get('dense_score')}  ·  "
            f"bm25 {retrieval.get('bm25_score')}"
        )


def _render_user(text: str) -> None:
    st.markdown(f"<div class='icare-user'>{html.escape(text)}</div>", unsafe_allow_html=True)


def _render_assistant(message: ChatMessage) -> None:
    if message.status != "done":
        st.markdown(f"<p class='icare-error'>{html.escape(message.content)}</p>", unsafe_allow_html=True)
    else:
        _thinking_block(message.payload)
        st.markdown(f"<p class='icare-answer'>{html.escape(message.content)}</p>", unsafe_allow_html=True)
    _source_card(message.reference, message.citation)
    retrieval = message.payload.get("retrieval")
    _technical_block(retrieval if isinstance(retrieval, dict) else None)


def _render_transcript(messages: list[ChatMessage]) -> None:
    for message in messages:
        if message.role == "user":
            _render_user(message.content)
        else:
            _render_assistant(message)


def _field(label: str, value: str) -> str:
    return (
        "<div class='icare-field'>"
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        "</div>"
    )


def _render_register() -> None:
    st.markdown("<p class='icare-section'>File d'évaluation</p>", unsafe_allow_html=True)
    for dossier in DOSSIERS:
        if st.button(
            f"{dossier['id']}   {dossier['name']}   ·   {dossier['diploma']}   ·   {dossier['country']}",
            key=f"register-{dossier['id']}",
            use_container_width=True,
        ):
            _open_dossier(dossier["id"])


def _render_case(dossier: dict) -> None:
    pieces = "".join(f"<li>{html.escape(piece)}</li>" for piece in dossier["pieces"])
    st.markdown(
        "<div class='icare-case'>"
        "<div class='icare-case-top'>"
        f"<p class='icare-case-id'>{html.escape(dossier['id'])}</p>"
        f"<span class='icare-status'>{html.escape(dossier['status'])}</span>"
        "</div>"
        f"<h2 class='icare-case-name'>{html.escape(dossier['name'])}</h2>"
        "<div class='icare-fields'>"
        + _field("Diplôme", dossier["diploma"])
        + _field("Pays", dossier["country"])
        + _field("Établissement", dossier["institution"])
        + _field("Objet", dossier["object"])
        + "</div>"
        f"<p class='icare-pieces-label'>Pièces au dossier</p><ul class='icare-pieces'>{pieces}</ul>"
        "</div>",
        unsafe_allow_html=True,
    )


def _remember_turn(conversation_id: int, question: str, progress, answer_box) -> None:
    add_message(ROOT, conversation_id, role="user", content=question)
    events = []
    states: dict[str, str] = {}
    started = time.perf_counter()

    def show(key: str, state: str, *, linger: float = 0.0) -> None:
        states[key] = state
        progress.markdown(_thinking_html(states), unsafe_allow_html=True)
        if linger:
            time.sleep(linger)

    show("think", "running", linger=0.45)
    try:
        for event in run_agents(question, root=ROOT):
            events.append(event)
            if event.agent == "retriever" and event.status == "running":
                show("think", "done")
                show("analyse", "running")
            elif event.agent == "retriever" and event.status == "done":
                show("analyse", "done")
            elif event.agent == "retriever" and event.status == "error":
                show("analyse", "error")
            elif event.agent == "citer" and event.status == "running":
                show("lecture", "running")
            elif event.agent == "citer" and event.status == "done":
                show("lecture", "done")
                show("test", "running", linger=0.55)
                show("test", "done")
            elif event.agent == "citer" and event.status == "error":
                show("lecture", "error")
            elif event.agent == "writer" and event.status == "running":
                show("write", "running")
            elif event.agent == "writer" and event.status == "done":
                show("write", "done")
                answer_box.markdown(
                    f"<p class='icare-answer'>{html.escape(event.message)}</p>",
                    unsafe_allow_html=True,
                )
                _source_card(event.data.get("reference") or {}, event.data.get("citation") or {})
            elif event.agent == "writer" and event.status == "error":
                show("write", "error")
            if event.status == "error" and event.agent in {"retriever", "citer", "writer"}:
                answer_box.markdown(
                    f"<p class='icare-error'>{html.escape(event.message)}</p>",
                    unsafe_allow_html=True,
                )
            time.sleep(0.1)
        turn = turn_from_events(events)
    except Exception as exc:
        turn = {"content": str(exc), "status": "error", "payload": {}, "run_id": None}
        answer_box.markdown(f"<p class='icare-error'>{html.escape(str(exc))}</p>", unsafe_allow_html=True)
    turn["payload"]["thinking"] = _finished_thinking(states)
    turn["payload"]["thinking_ms"] = int((time.perf_counter() - started) * 1000)
    add_message(
        ROOT,
        conversation_id,
        role="assistant",
        content=turn["content"],
        status=turn["status"],
        payload=turn["payload"],
        source_run_id=turn["run_id"],
    )


def main() -> None:
    theme = _apply_theme()
    import_legacy_runs(ROOT)
    _render_sidebar()
    _render_header(theme)

    dossier = _dossier(st.session_state.get("dossier_id"))
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id is not None and get_conversation(ROOT, conversation_id) is None:
        st.session_state.conversation_id = None
        conversation_id = None
    messages = list_messages(ROOT, conversation_id) if conversation_id is not None else []
    if dossier is not None:
        _render_case(dossier)
    if messages:
        st.markdown("<p class='icare-section'>Décision</p>", unsafe_allow_html=True)
        _render_transcript(messages)

    ready, info = _index_ready()
    if not ready:
        st.caption(info)
        st.stop()

    typed = st.chat_input("Question sur ce dossier…")
    question = typed or st.session_state.pop("pending_question", None)
    if not question:
        if dossier is not None and not messages:
            if st.button("Établir la décision", key="decide"):
                st.session_state.pending_question = dossier["question"]
                st.rerun()
        elif not messages:
            _render_register()
        return

    if conversation_id is None:
        title = f"{dossier['name']} · {dossier['id']}" if dossier else conversation_title(question)
        conversation_id = create_conversation(ROOT, title)
        st.session_state.conversation_id = conversation_id

    _render_user(question)
    progress = st.empty()
    answer_box = st.empty()
    _remember_turn(conversation_id, question, progress, answer_box)
    st.rerun()


if __name__ == "__main__":
    main()
