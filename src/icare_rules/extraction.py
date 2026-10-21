"""Extraction du texte des fichiers Word (.doc / .docx)."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


class ExtractionError(Exception):
    """Levée quand toutes les stratégies d'extraction échouent."""


@dataclass
class ExtractionResult:
    text: str
    strategy: str
    warnings: list[str]


def _run_textutil(path: Path, timeout: int) -> str:
    if not shutil.which("textutil"):
        raise ExtractionError("`textutil` introuvable (macOS uniquement).")
    try:
        completed = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExtractionError(f"textutil timeout ({timeout}s) sur {path.name}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ExtractionError(f"textutil a échoué : {stderr}")
    return completed.stdout.decode("utf-8", errors="replace")


WORDPERFECT_MAGIC = b"\xffWPC"
# Au-delà de cette proportion de caractères de contrôle, la sortie est du binaire mal décodé.
MAX_CONTROL_CHAR_RATIO = 0.01
_PAGE_MARKER_RE = re.compile(r"^\s*(?:Page\s+)?\d+\s+de\s+\d+\s*$|\s{5,}(?:Page\s+)?\d+\s+de\s+\d+\s*$")


def is_wordperfect(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(4) == WORDPERFECT_MAGIC


def control_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(1 for c in text if (ord(c) < 32 and c not in "\n\t\r") or c == "\ufffd")
    return bad / len(text)


def _run_wpd2text(path: Path, timeout: int) -> str:
    if not shutil.which("wpd2text"):
        raise ExtractionError("`wpd2text` introuvable (installer avec `brew install libwpd`).")
    try:
        completed = subprocess.run(
            ["wpd2text", str(path)],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExtractionError(f"wpd2text timeout ({timeout}s) sur {path.name}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ExtractionError(f"wpd2text a échoué : {stderr}")
    text = completed.stdout.decode("utf-8", errors="replace")
    # Les en-têtes « Page 1 de 3 » seraient sinon pris pour des marqueurs de sous-règle.
    return "\n".join(_PAGE_MARKER_RE.sub("", line) for line in text.split("\n"))


def _run_python_docx(path: Path) -> str:
    if path.suffix.lower() != ".docx":
        raise ExtractionError("python-docx ne supporte que .docx")
    try:
        from docx import Document  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("python-docx non installé") from exc

    doc = Document(str(path))
    parts: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append("\t".join(cells))
    return "\n".join(parts)


def normalize_text(raw: str) -> str:
    """Normalise le texte : Unicode NFC, espaces, retours à la ligne."""
    text = unicodedata.normalize("NFC", raw)
    # Remplacer tabs par espaces mais préserver les listes numérotées
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Supprimer les espaces en fin de ligne et compacter les triples \n
    lines = [line.rstrip() for line in text.split("\n")]
    normalized_lines: list[str] = []
    empty_streak = 0
    for line in lines:
        if not line.strip():
            empty_streak += 1
            if empty_streak <= 1:
                normalized_lines.append("")
            continue
        empty_streak = 0
        normalized_lines.append(line)
    while normalized_lines and not normalized_lines[0]:
        normalized_lines.pop(0)
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    return "\n".join(normalized_lines)


def extract_text(path: Path, settings: Settings) -> ExtractionResult:
    """Essaie chaque stratégie configurée jusqu'à en trouver une qui fonctionne."""
    warnings: list[str] = []
    last_error: str | None = None
    timeout = settings.extraction.timeout_seconds
    # Les .doc WordPerfect sont « lus » par textutil mais produisent du binaire :
    # ils ne passent que par wpd2text, les autres formats jamais.
    wordperfect = is_wordperfect(path)
    strategies = [
        s for s in settings.extraction.strategies if (s == "wpd2text") == wordperfect
    ]
    if not strategies:
        kind = "WordPerfect" if wordperfect else "Word"
        raise ExtractionError(f"aucune stratégie configurée pour un fichier {kind}")

    for strategy in strategies:
        try:
            if strategy == "textutil":
                raw = _run_textutil(path, timeout=timeout)
            elif strategy == "python-docx":
                raw = _run_python_docx(path)
            elif strategy == "wpd2text":
                raw = _run_wpd2text(path, timeout=timeout)
            else:
                warnings.append(f"stratégie inconnue ignorée : {strategy}")
                continue
        except ExtractionError as exc:
            warnings.append(f"{strategy}: {exc}")
            last_error = str(exc)
            continue

        text = normalize_text(raw)
        if len(text.strip()) < 5:
            warnings.append(f"{strategy}: sortie trop courte ({len(text)} caractères)")
            last_error = "sortie vide"
            continue
        ratio = control_char_ratio(text)
        if ratio > MAX_CONTROL_CHAR_RATIO:
            warnings.append(f"{strategy}: sortie illisible ({ratio:.1%} de caractères de contrôle)")
            last_error = "sortie illisible"
            continue
        return ExtractionResult(text=text, strategy=strategy, warnings=warnings)

    raise ExtractionError(f"toutes les stratégies ont échoué : {last_error}")


def extract_to_tempfile(path: Path, settings: Settings) -> tuple[ExtractionResult, Path]:
    """Extrait le texte et l'écrit dans un fichier temporaire (utile pour cache)."""
    result = extract_text(path, settings)
    fd = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8")
    fd.write(result.text)
    fd.close()
    return result, Path(fd.name)
