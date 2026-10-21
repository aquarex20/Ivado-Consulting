"""Profilage du corpus sans LLM : structure, quantités, renvois, gabarits, mots-clés TF-IDF."""

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Settings
from .segmentation import Segment, iter_segments_for_classification
from .utils import sha256_text, write_jsonl

TOP_KEYWORDS = 5
TOP_N_SUMMARY = 30
MAX_DF_RATIO = 0.02

_NUMBER_WORDS = (
    "zéro|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|"
    "quatorze|quinze|seize|dix-sept|dix-huit|dix-neuf|vingt|trente|quarante|"
    "cinquante|soixante|cent|deux cents?|trois cents?|quatre cents?|mille"
)

# « deux (2) années », « 45 crédits », « deux cent quarante (240) heures »
_QUANTITY_RE = re.compile(
    r"(?:\((?P<paren>\d{1,4})\)|\b(?P<bare>\d{1,4}))\s+"
    r"(?:premi[èe]res?\s+|derni[èe]res?\s+)?"
    r"(?P<unit>ann[ée]es?|ans\b|cr[ée]dits?|heures?|unit[ée]s?|mois|semaines?|sessions?|trimestres?|semestres?)",
    re.IGNORECASE,
)
# « (16/30) », « 16/30 d'année »
_FRACTION_RE = re.compile(r"\(?\b(?P<num>\d{1,3})\s*/\s*(?P<den>\d{1,3})\b\)?")
_GRANT_CLAUSE_RE = re.compile(
    r"(?:on\s+accorde|accord[ée]e?s?|scolarit[ée]\s+maximum)[^.;\n]*",
    re.IGNORECASE,
)
_RULE_REF_RE = re.compile(r"\b(?P<kind>[GP])\s?-?(?P<num>\d{3,5})\b")
_SECTION_P_RE = re.compile(r"r[èe]gle\s+particuli[èe]re|section\s+P\b", re.IGNORECASE)
_NB_RE = re.compile(r"\bN\.\s?B\.", re.IGNORECASE)
_REQUIRED_DOC_RE = re.compile(
    r"\bexiger\s+(?:le|la|les|l['’]|un|une|des)?\s*(?P<doc>[^.;:\n]{3,80})",
    re.IGNORECASE,
)
_ADDITIVE_RE = re.compile(r"s['’]ajout\w*\s+[àa]\s+la\s+scolarit", re.IGNORECASE)
_CAP_RE = re.compile(r"\bmaxim(?:um|al|ale)\b|\bne\s+(?:peut|doit)\s+d[ée]passer\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"“«]\s*(?P<lib>[^\"”»]{3,150}?)\s*[\"”»]")

_TOKEN_RE = re.compile(r"[a-zà-öø-ÿœæ]+(?:['’-][a-zà-öø-ÿœæ]+)*", re.IGNORECASE)
_ELISION_RE = re.compile(r"^(?:[cdjlmnst]|qu|jusqu|lorsqu|puisqu)['’]")
_STOPWORDS_TEXT = """
    a à au aux avec ce ces cet cette dans de des du elle en et être eux il ils je la le les leur
    leurs lui ma mais me même mes moi mon ne nos notre nous on ou où par pas pour qu que qui sa se
    ses son sur ta te tes toi ton tu un une vos votre vous est sont été sera ont avait sans soit
    selon lorsque lorsqu si ainsi alors tout toute tous toutes chaque autre autres afin dont cas
    celui celle ceux celles leur plus moins entre après avant sous vers chez puis donc comme
    peut doit être avoir fait faire non oui ci là l d s n c qu j m t y
    the of and in for to at by on with from
"""
_STOPWORDS = set(_STOPWORDS_TEXT.split()) | set(re.split(r"[| ]", _NUMBER_WORDS.replace("?", "")))


@dataclass
class Quantity:
    value: float
    unit: str
    raw: str


@dataclass
class RuleProfile:
    rule_id: str
    parent_rule_id: str
    kind: str
    char_count: int
    word_count: int
    subrule_count: int
    has_nb: bool
    quantities: list[dict] = field(default_factory=list)
    granted: list[dict] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    mentions_particular_rule: bool = False
    required_documents: list[str] = field(default_factory=list)
    additive: bool = False
    capped: bool = False
    quoted_labels: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    template_id: str = ""
    template_size: int = 1
    exact_duplicate_count: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_unit(unit: str) -> str:
    u = unicodedata.normalize("NFKD", unit.lower()).encode("ascii", "ignore").decode()
    if u.startswith("an"):
        return "année"
    if u.startswith("credit"):
        return "crédit"
    if u.startswith("unit"):
        return "unité"
    return u.rstrip("s") if u != "mois" else u


def extract_quantities(text: str) -> list[Quantity]:
    found: list[Quantity] = []
    for m in _QUANTITY_RE.finditer(text):
        value = float(m.group("paren") or m.group("bare"))
        found.append(Quantity(value, _normalize_unit(m.group("unit")), m.group(0)))
    for m in _FRACTION_RE.finditer(text):
        num, den = int(m.group("num")), int(m.group("den"))
        if den == 0 or num > den:
            continue
        found.append(Quantity(round(num / den, 4), "fraction d'année", f"{num}/{den}"))
    return found


def extract_granted(text: str) -> list[Quantity]:
    """Quantités situées dans une clause d'attribution (« on accorde … »)."""
    granted: list[Quantity] = []
    for clause in _GRANT_CLAUSE_RE.finditer(text):
        granted.extend(extract_quantities(clause.group(0)))
    return granted


def extract_references(text: str, self_id: str) -> list[str]:
    refs = {f"{m.group('kind')}{int(m.group('num')):04d}" for m in _RULE_REF_RE.finditer(text)}
    refs.discard(self_id)
    return sorted(refs)


def extract_required_documents(text: str) -> list[str]:
    return [m.group("doc").strip().lower() for m in _REQUIRED_DOC_RE.finditer(text)]


def template_signature(text: str) -> str:
    """Masque libellés, nombres et noms propres pour révéler le gabarit d'une règle."""
    t = _QUOTED_RE.sub(" <LIB> ", text)
    t = re.sub(r"\d+(?:\s*/\s*\d+)?", "<N>", t)
    t = re.sub(rf"\b(?:{_NUMBER_WORDS})\b", "<N>", t, flags=re.IGNORECASE)
    t = re.sub(r"\b[A-ZÀ-ÖØ-Þ][\w'’.-]*", "<C>", t)
    t = re.sub(r"(?:<C>[\s,()-]*)+", "<C> ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return sha256_text(t)[:12]


def tokenize(text: str) -> list[str]:
    tokens = [_ELISION_RE.sub("", t.lower()) for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def tfidf_keywords(
    docs: list[list[str]],
    top_k: int = TOP_KEYWORDS,
    min_df: int = 2,
    max_df_ratio: float = MAX_DF_RATIO,
) -> list[list[str]]:
    """Mots-clés TF-IDF par document.

    Ignore les termes présents dans moins de `min_df` documents (coquilles) ou dans
    plus de `max_df_ratio` des documents (formules types du Manuel).
    """
    n = len(docs)
    max_df = max(min_df, max_df_ratio * n)
    df: Counter[str] = Counter()
    for terms in docs:
        df.update(set(terms))
    result: list[list[str]] = []
    for terms in docs:
        tf = Counter(terms)
        scored = [
            (count * math.log(n / df[term]), term)
            for term, count in tf.items()
            if min_df <= df[term] <= max_df
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        result.append([term for _score, term in scored[:top_k]])
    return result


def build_profiles(segments: list[Segment]) -> list[RuleProfile]:
    subrule_counts: Counter[str] = Counter(s.parent_rule_id for s in segments if s.kind == "subrule")
    targets = list(iter_segments_for_classification(segments))

    profiles: list[RuleProfile] = []
    for seg in targets:
        text = seg.text_normalized
        profiles.append(
            RuleProfile(
                rule_id=seg.rule_id,
                parent_rule_id=seg.parent_rule_id,
                kind=seg.kind,
                char_count=seg.char_count,
                word_count=len(text.split()),
                subrule_count=subrule_counts.get(seg.parent_rule_id, 0),
                has_nb=bool(_NB_RE.search(text)),
                quantities=[asdict(q) for q in extract_quantities(text)],
                granted=[asdict(q) for q in extract_granted(text)],
                references=extract_references(text, seg.parent_rule_id),
                mentions_particular_rule=bool(_SECTION_P_RE.search(text)),
                required_documents=extract_required_documents(text),
                additive=bool(_ADDITIVE_RE.search(text)),
                capped=bool(_CAP_RE.search(text)),
                quoted_labels=[m.group("lib").strip() for m in _QUOTED_RE.finditer(text)],
                template_id=template_signature(text),
            )
        )

    keywords = tfidf_keywords([tokenize(s.text_normalized) for s in targets])
    template_sizes = Counter(p.template_id for p in profiles)
    hash_counts = Counter(s.text_hash for s in targets)
    for p, seg, kw in zip(profiles, targets, keywords, strict=True):
        p.keywords = kw
        p.template_size = template_sizes[p.template_id]
        p.exact_duplicate_count = hash_counts[seg.text_hash]
    return profiles


def _quantity_label(q: dict) -> str:
    if q["unit"] == "fraction d'année":
        return f"{q['raw']} d'année"
    value = int(q["value"]) if float(q["value"]).is_integer() else q["value"]
    unit = q["unit"]
    if value > 1 and unit not in ("mois",):
        unit += "s"
    return f"{value} {unit}"


def _distribution(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    s = sorted(values)

    def pct(p: float) -> int:
        return s[min(len(s) - 1, int(p * len(s)))]

    return {
        "count": len(s),
        "min": s[0],
        "p25": pct(0.25),
        "median": pct(0.5),
        "p75": pct(0.75),
        "p95": pct(0.95),
        "max": s[-1],
        "mean": round(sum(s) / len(s), 1),
    }


def summarize_profiles(profiles: list[RuleProfile]) -> dict:
    n = len(profiles)
    by_parent: dict[str, list[RuleProfile]] = defaultdict(list)
    for p in profiles:
        by_parent[p.parent_rule_id].append(p)

    keyword_counts = Counter(k for p in profiles for k in p.keywords)
    granted_counts = Counter(_quantity_label(q) for p in profiles for q in p.granted)
    doc_counts = Counter(d for p in profiles for d in p.required_documents)
    label_counts = Counter(lbl for p in profiles for lbl in p.quoted_labels)
    inbound = Counter(ref for p in profiles for ref in p.references)
    templates = Counter(p.template_id for p in profiles)
    template_examples = {p.template_id: p.rule_id for p in reversed(profiles)}

    def share(pred) -> float:
        return round(sum(1 for p in profiles if pred(p)) / n, 4) if n else 0.0

    return {
        "totals": {
            "files": len(by_parent),
            "files_G": sum(1 for k in by_parent if k.startswith("G")),
            "files_P": sum(1 for k in by_parent if k.startswith("P")),
            "rules_profiled": n,
            "distinct_templates": len(templates),
            "rules_in_shared_templates": sum(c for c in templates.values() if c > 1),
            "exact_duplicate_rules": sum(1 for p in profiles if p.exact_duplicate_count > 1),
        },
        "length": {
            "chars": _distribution([p.char_count for p in profiles]),
            "words": _distribution([p.word_count for p in profiles]),
            "subrules_per_file": _distribution(
                [len(v) if v[0].kind == "subrule" else 0 for v in by_parent.values()]
            ),
        },
        "features_share": {
            "has_nb": share(lambda p: p.has_nb),
            "has_quantity": share(lambda p: p.quantities),
            "has_granted_value": share(lambda p: p.granted),
            "additive": share(lambda p: p.additive),
            "capped": share(lambda p: p.capped),
            "references_other_rule": share(lambda p: p.references),
            "mentions_particular_rule": share(lambda p: p.mentions_particular_rule),
            "requires_document": share(lambda p: p.required_documents),
            "quotes_label": share(lambda p: p.quoted_labels),
        },
        "top_keywords": keyword_counts.most_common(TOP_N_SUMMARY),
        "top_granted_values": granted_counts.most_common(TOP_N_SUMMARY),
        "top_required_documents": doc_counts.most_common(TOP_N_SUMMARY),
        "top_quoted_labels": label_counts.most_common(TOP_N_SUMMARY),
        "most_referenced_rules": inbound.most_common(TOP_N_SUMMARY),
        "largest_templates": [
            {"template_id": t, "size": c, "example_rule_id": template_examples[t]}
            for t, c in templates.most_common(TOP_N_SUMMARY)
            if c > 1
        ],
    }


# ---------------------------------------------------------------------------
# Écriture des artefacts
# ---------------------------------------------------------------------------


def profile_jsonl_path(settings: Settings) -> Path:
    return settings.output_root / "rule_profile.jsonl"


def profile_csv_path(settings: Settings) -> Path:
    return settings.output_root / "rule_profile.csv"


def references_csv_path(settings: Settings) -> Path:
    return settings.output_root / "rule_references.csv"


def corpus_profile_path(settings: Settings) -> Path:
    return settings.output_root / "corpus_profile.json"


def write_profiles(settings: Settings, profiles: list[RuleProfile]) -> dict[str, Path]:
    write_jsonl(profile_jsonl_path(settings), (p.to_dict() for p in profiles))

    csv_path = profile_csv_path(settings)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rule_id", "parent_rule_id", "kind", "char_count", "word_count",
                "subrule_count", "keywords", "granted", "quantities", "additive",
                "capped", "has_nb", "required_documents", "references",
                "mentions_particular_rule", "quoted_labels", "template_id",
                "template_size", "exact_duplicate_count",
            ]
        )
        for p in profiles:
            w.writerow(
                [
                    p.rule_id, p.parent_rule_id, p.kind, p.char_count, p.word_count,
                    p.subrule_count, " | ".join(p.keywords),
                    " | ".join(_quantity_label(q) for q in p.granted),
                    " | ".join(_quantity_label(q) for q in p.quantities),
                    p.additive, p.capped, p.has_nb, " | ".join(p.required_documents),
                    " | ".join(p.references), p.mentions_particular_rule,
                    " | ".join(p.quoted_labels), p.template_id, p.template_size,
                    p.exact_duplicate_count,
                ]
            )

    refs_path = references_csv_path(settings)
    with refs_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_rule_id", "source_parent_rule_id", "target_rule_id"])
        for p in profiles:
            for ref in p.references:
                w.writerow([p.rule_id, p.parent_rule_id, ref])

    summary_path = corpus_profile_path(settings)
    summary_path.write_text(
        json.dumps(summarize_profiles(profiles), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "profile_jsonl": profile_jsonl_path(settings),
        "profile_csv": csv_path,
        "references": refs_path,
        "corpus_profile": summary_path,
    }
