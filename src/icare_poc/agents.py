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


_STOP = {
    "combien", "pour", "dans", "avec", "sans", "quelle", "quelles", "quel", "quels",
    "vaut", "cette", "sont", "une", "des", "les", "par", "sur", "plus", "tout",
    "comment", "quand", "quoi", "est", "que", "qui", "fait", "entre", "depuis",
    "apres", "avant", "peut", "selon", "règle", "regle",
}
_YEAR_PHRASE = re.compile(
    r"\b((?:une|un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|"
    r"treize|quatorze|quinze|seize|dix-huit|dix-sept|\d+)"
    r"(?:\s+et\s+demie)?(?:\s*\([^)]*\))?)\s+années?(?:\s+de\s+scolarité)?",
    re.IGNORECASE,
)
_VERBATIM_WINDOW = 48


def _missing_terms(question: str, passage: str) -> list[str]:
    known = set(tokenize(passage))
    missing: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'-]{3,}", question):
        folded = tokenize(word)
        if not folded:
            continue
        token = folded[0]
        if token in _STOP or token in known or token in seen:
            continue
        seen.add(token)
        missing.append(word)
    return missing


def _mention(words: list[str]) -> str:
    quoted = [f"« {word} »" for word in words]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " ni " + quoted[-1]


def _program_names(text: str) -> list[str]:
    names: list[str] = []
    for name in re.findall(r'"([^"]{1,80})"', text):
        name = re.sub(r"\s+", " ", name).strip(" ,")
        if name.endswith("...") or not re.search(r"bachelor|maitrise|maîtrise|doctor|licence", name, re.I):
            continue
        base = re.split(r",", name)[0].strip()
        if base and base not in names:
            names.append(base)
    return names


def _year_amount(fragment: str) -> str | None:
    match = _YEAR_PHRASE.search(fragment)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(0)).strip()


def _grants(text: str) -> list[str]:
    found: list[str] = []
    if re.search(r"portion du programme qui ne fait pas l'objet d'un recouvrement", text, re.I):
        names = _program_names(text)
        if names:
            label = " ou ".join(f"un {name}" for name in names[:3])
            found.append(
                "Elle vise "
                f"{label} : la scolarité accordée est seulement la partie du programme "
                "qui ne recouvre pas des études déjà reconnues."
            )
        else:
            found.append(
                "La scolarité accordée est seulement la partie du programme "
                "qui ne recouvre pas des études déjà reconnues."
            )
    for match in re.finditer(r"on accorde(.{0,280}?)(?:[.;]|$)", text, re.I):
        body = match.group(1)
        if "portion du programme" in body.lower():
            continue
        if re.search(r"aucune scolarité", body, re.I):
            found.append("La règle n'accorde aucune scolarité additionnelle.")
            continue
        amount = _year_amount(body)
        if not amount:
            continue
        pour = ""
        pour_match = re.search(r"pour\s+([^,]{3,80})", body, re.I)
        if pour_match:
            pour = f" pour {pour_match.group(1).strip()}"
        extra = ""
        if re.search(r"s'ajoute|s'ajoutent|déjà acquise", body, re.I):
            extra = ", en plus de la scolarité déjà acquise"
        found.append(f"La règle accorde {amount}{pour}{extra}.")
    return found


def _cap(text: str) -> str | None:
    match = re.search(r"ne (?:devra|doit) pas dépasser\s+(.{0,120})", text, re.I)
    if match is None:
        match = re.search(r"sans toutefois dépasser\s+(.{0,120})", text, re.I)
    if match is None:
        return None
    amount = _year_amount(match.group(0))
    if not amount:
        return None
    if re.search(r"formation générale", text, re.I) and re.search(r"douze\s*\(\s*12\s*\)", text, re.I):
        return (
            "Si les études qui précèdent ou accompagnent ce diplôme valent déjà douze (12) années ou plus "
            "et comprennent de la formation générale, cette formation générale ne peut pas dépasser "
            f"{amount}."
        )
    return f"Cette scolarité ne peut pas dépasser {amount}."


def _equivalences(text: str) -> list[str]:
    found: list[str] = []
    for raw in re.split(r"[\n•]+", text):
        line = re.sub(r"\s+", " ", raw.replace("\t", " ")).strip(" •-\t\"")
        match = re.match(
            r"(.{1,40}?)\s*=\s*(.{1,80}?)(?:\s+qui s'ajoute(?:nt)? à la scolarité déjà acquise)?\.?$",
            line,
        )
        if not match:
            continue
        left = match.group(1).strip(" \"")
        right = match.group(2).strip(" \"")
        if not left or not right:
            continue
        extra = ", en plus de la scolarité déjà acquise" if "déjà acquise" in line.lower() else ""
        if left[:1].isdigit():
            sentence = f"{left} correspond à {right}{extra}."
        else:
            sentence = f"« {left} » correspond à {right}{extra}."
        if sentence not in found:
            found.append(sentence)
    return found


def _compose(question: str, rule: str, passage: str) -> str:
    text = passage.strip()
    if not text:
        return f"La règle {rule} ne contient pas de passage répondant à cette question."
    facts = [*_grants(text)]
    cap = _cap(text)
    if cap:
        facts.append(cap)
    facts.extend(_equivalences(text))
    missing = _missing_terms(question, text)
    if missing:
        return (
            f"Je ne sais pas. La règle {rule} ne mentionne pas {_mention(missing)}, "
            "donc ce passage ne permet pas de répondre."
        )
    if not facts:
        return (
            f"Je ne sais pas. La règle {rule} ne contient pas de décision "
            "qui réponde à cette question."
        )
    first = facts[0]
    grant_lead = "La règle accorde"
    none_lead = "La règle n'accorde"
    aim_lead = "Elle vise"
    if first.startswith(grant_lead):
        first = f"La règle {rule} accorde{first[len(grant_lead):]}"
    elif first.startswith(aim_lead):
        first = f"La règle {rule} vise{first[len(aim_lead):]}"
    elif first.startswith(none_lead):
        first = f"La règle {rule} n'accorde{first[len(none_lead):]}"
    else:
        rest = first[1:] if first[:1].isalpha() else first
        head = first[:1].lower() if first[:1].isalpha() else ""
        first = f"Selon la règle {rule}, {head}{rest}"
    return " ".join([first, *facts[1:]])


def _copies_source(answer: str, source: str) -> bool:
    folded_answer = re.sub(r"\s+", " ", answer).lower()
    folded_source = re.sub(r"\s+", " ", source).lower()
    if len(folded_source) <= _VERBATIM_WINDOW:
        return False
    for start in range(0, len(folded_source) - _VERBATIM_WINDOW, 12):
        if folded_source[start : start + _VERBATIM_WINDOW] in folded_answer:
            return True
    return False


def _reformulate_local(question: str, hit: RetrievalHit, location: PdfLocation) -> str:
    payload = hit.payload
    rule = str(payload.get("document_id") or hit.chunk_id)
    body = (payload.get("text") or "").strip()
    excerpt = (location.line_text or payload.get("excerpt") or "").strip()
    return _compose(question, rule, body or excerpt)


def _reformulate_ollama(question: str, hit: RetrievalHit, location: PdfLocation) -> str:
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
    if _copies_source(text, passage):
        raise RuntimeError("réponse Ollama trop proche du passage")
    return text


def _admits_ignorance(answer: str) -> bool:
    folded = answer.lower()
    return "je ne sais pas" in folded or "je ne peux pas" in folded


def reformulate(question: str, hit: RetrievalHit, location: PdfLocation) -> tuple[str, str]:
    passage = (hit.payload.get("text") or location.line_text or "").strip()
    try:
        answer = _reformulate_ollama(question, hit, location)
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError, OSError):
        return _reformulate_local(question, hit, location), "local"
    if _missing_terms(question, passage) and not _admits_ignorance(answer):
        return _reformulate_local(question, hit, location), "local"
    return answer, "ollama"


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
