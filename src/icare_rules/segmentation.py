"""Segmentation du texte extrait en sous-règles stables."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from .config import Settings
from .utils import sha256_text

# Marqueurs de sous-règle observés dans le corpus :
#   "1", "1-", "1‑", "1)", "1.", "13-", "A)", "B)", "a)", "III-"
# La ponctuation est optionnelle uniquement pour les marqueurs numériques
# (`1 <tab>`) car sinon des lettres isolées seraient prises comme entêtes.
_NUM_HEADER = re.compile(
    r"^(?P<marker>[0-9]{1,3})"
    r"(?P<sep>[)\.\-\u2011\u2013\u2012\u2014]?)"
    r"(?:[\t ]+|$)"
)
_ALPHA_HEADER = re.compile(
    r"^(?P<marker>[A-Za-z]|[IVXLCDM]{2,4})"
    r"(?P<sep>[)\.\-\u2011\u2013\u2012\u2014])"
    r"(?:[\t ]+|$)"
)


@dataclass
class Segment:
    rule_id: str  # G0539, ou G0539#001 pour sous-règle
    parent_rule_id: str  # G0539
    kind: str  # "file" | "subrule"
    index: int  # 0 pour l'agrégat fichier, 1..N pour les sous-règles
    marker: str  # préfixe original ("1", "13", "A", ...)
    source_file: str
    source_hash: str
    text_normalized: str
    text_hash: str
    char_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_header_line(line: str) -> tuple[str, str] | None:
    """Retourne (marker, sep) si la ligne débute par un marqueur de sous-règle."""
    stripped = line.strip()
    if not stripped:
        return None
    m = _NUM_HEADER.match(stripped)
    if m:
        return m.group("marker"), m.group("sep")
    m = _ALPHA_HEADER.match(stripped)
    if m:
        marker = m.group("marker")
        sep = m.group("sep")
        # Une lettre isolée ne doit être acceptée que devant `)` (A) B) ...)
        if len(marker) == 1 and sep != ")":
            return None
        return marker, sep
    return None


def _header_prefix_length(line: str, marker: str, sep: str) -> int:
    """Nombre de caractères à retirer en tête de `line` pour obtenir le contenu."""
    return len(marker) + len(sep)


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Découpe le texte en (marker, contenu) selon les entêtes trouvées.

    Retourne une liste vide si aucun entête n'est détecté.
    """
    lines = text.split("\n")
    segments: list[tuple[str, list[str]]] = []
    current_marker: str | None = None
    current_lines: list[str] = []
    seen_header = False

    for line in lines:
        header = _is_header_line(line)
        if header:
            seen_header = True
            if current_marker is not None or current_lines:
                segments.append((current_marker or "", current_lines))
            marker, sep = header
            current_marker = marker
            # Retirer l'entête de la ligne pour garder le contenu
            stripped = line.strip()
            prefix_len = _header_prefix_length(stripped, marker, sep)
            content = stripped[prefix_len:].lstrip("\t ")
            current_lines = [content] if content else []
        else:
            current_lines.append(line)

    if current_marker is not None or current_lines:
        segments.append((current_marker or "", current_lines))

    if not seen_header:
        return []

    result: list[tuple[str, str]] = []
    for marker, chunk in segments:
        content = "\n".join(chunk).strip()
        if content or marker:
            result.append((marker, content))
    return result


def _drop_preamble(segments: list[tuple[str, str]], parent_rule_id: str) -> list[tuple[str, str]]:
    """Le texte avant le premier entête contient souvent juste l'ID de la règle.

    On l'écarte s'il est vide ou trivial ; sinon on le garde comme préambule.
    """
    if not segments:
        return segments
    first_marker, first_content = segments[0]
    if first_marker == "":
        stripped = first_content.strip()
        # Enlève l'ID répété (ex. "P0048" seul)
        if stripped in {parent_rule_id, ""} or len(stripped) < 20:
            return segments[1:]
    return segments


def segment_rule(
    parent_rule_id: str,
    source_file: str,
    source_hash: str,
    text: str,
    settings: Settings,
) -> list[Segment]:
    """Retourne toujours au moins un segment (l'agrégat fichier).

    Si des sous-règles sont détectées, elles sont ajoutées avec un identifiant
    stable de la forme `P0048#001`.
    """
    warnings: list[str] = []
    aggregate = Segment(
        rule_id=parent_rule_id,
        parent_rule_id=parent_rule_id,
        kind="file",
        index=0,
        marker="",
        source_file=source_file,
        source_hash=source_hash,
        text_normalized=text,
        text_hash=sha256_text(text),
        char_count=len(text),
        warnings=[],
    )

    header_segments = _split_by_headers(text)
    header_segments = _drop_preamble(header_segments, parent_rule_id)

    subrules: list[Segment] = []
    for i, (marker, content) in enumerate(header_segments, start=1):
        if len(content) < settings.segmentation.min_segment_chars:
            warnings.append(f"segment #{i:03d} ignoré (trop court: {len(content)} chars)")
            continue
        subrules.append(
            Segment(
                rule_id=f"{parent_rule_id}#{i:03d}",
                parent_rule_id=parent_rule_id,
                kind="subrule",
                index=i,
                marker=marker,
                source_file=source_file,
                source_hash=source_hash,
                text_normalized=content,
                text_hash=sha256_text(content),
                char_count=len(content),
                warnings=[],
            )
        )

    if len(subrules) > settings.segmentation.max_subrules_per_file:
        warnings.append(
            f"nombre de sous-règles ({len(subrules)}) > max "
            f"({settings.segmentation.max_subrules_per_file}); tronqué"
        )
        subrules = subrules[: settings.segmentation.max_subrules_per_file]

    aggregate.warnings = warnings
    return [aggregate, *subrules]


def iter_segments_for_classification(segments: Iterable[Segment]) -> Iterable[Segment]:
    """Sélectionne les segments à envoyer au LLM.

    Règle : si le fichier a des sous-règles, on classe chaque sous-règle ; sinon
    on classe l'agrégat fichier.
    """
    by_parent: dict[str, list[Segment]] = {}
    for seg in segments:
        by_parent.setdefault(seg.parent_rule_id, []).append(seg)

    for _parent, group in by_parent.items():
        subs = [s for s in group if s.kind == "subrule"]
        if subs:
            yield from subs
        else:
            aggregates = [s for s in group if s.kind == "file"]
            if aggregates:
                yield aggregates[0]
