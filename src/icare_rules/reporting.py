"""Génération des rapports (CSV détaillé, synthèse, révision, métriques)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .config import Settings
from .pipeline import (
    ClassificationRecord,
    classification_errors_path,
    extraction_errors_path,
    load_classifications,
)
from .utils import read_jsonl

CATEGORIES = ("Déterministe", "Qualitative", "Hybride")


def detailed_csv_path(settings: Settings) -> Path:
    return settings.output_root / "classifications_detail.csv"


def summary_csv_path(settings: Settings) -> Path:
    return settings.output_root / "classifications_summary_by_file.csv"


def review_csv_path(settings: Settings) -> Path:
    return settings.output_root / "manual_review_queue.csv"


def metrics_json_path(settings: Settings) -> Path:
    return settings.output_root / "metrics.json"


def vocabulary_csv_path(settings: Settings) -> Path:
    return settings.output_root / "attribute_vocabulary.csv"


VOCABULARY_FIELDS = (
    "theme",
    "keywords",
    "diplomas",
    "institutions",
    "countries",
    "prerequisites",
    "granted_value",
    "dossier_variables",
)
TOP_N_ATTRIBUTES = 30


def _attribute_values(record: ClassificationRecord, name: str) -> list[str]:
    raw = getattr(record, name)
    values = raw if isinstance(raw, list) else [raw]
    return sorted({v.strip().lower() for v in values if v and v.strip()})


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_detailed_csv(settings: Settings, records: list[ClassificationRecord]) -> Path:
    header = [
        "rule_id",
        "parent_rule_id",
        "kind",
        "source_file",
        "source_hash",
        "char_count",
        "category",
        "justification",
        "evidence",
        "confidence",
        "manual_review",
        "low_confidence",
        "model",
        "prompt_version",
        "cached",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "warnings",
        "at",
        "theme",
        "keywords",
        "diplomas",
        "institutions",
        "countries",
        "prerequisites",
        "granted_value",
        "dossier_variables",
    ]
    rows = [
        [
            r.rule_id,
            r.parent_rule_id,
            r.kind,
            r.source_file,
            r.source_hash,
            r.char_count,
            r.category,
            r.justification,
            " | ".join(r.evidence),
            f"{r.confidence:.3f}",
            r.manual_review,
            r.low_confidence,
            r.model,
            r.prompt_version,
            r.cached,
            r.latency_ms,
            r.prompt_tokens,
            r.completion_tokens,
            r.total_tokens,
            " | ".join(r.warnings),
            r.at,
            r.theme,
            " | ".join(r.keywords),
            " | ".join(r.diplomas),
            " | ".join(r.institutions),
            " | ".join(r.countries),
            " | ".join(r.prerequisites),
            r.granted_value or "",
            " | ".join(r.dossier_variables),
        ]
        for r in records
    ]
    path = detailed_csv_path(settings)
    _write_csv(path, header, rows)
    return path


def write_summary_by_file(settings: Settings, records: list[ClassificationRecord]) -> Path:
    by_parent: dict[str, list[ClassificationRecord]] = defaultdict(list)
    for r in records:
        by_parent[r.parent_rule_id].append(r)

    header = [
        "parent_rule_id",
        "source_file",
        "subrule_count",
        *[f"n_{c}" for c in CATEGORIES],
        *[f"pct_{c}" for c in CATEGORIES],
        "dominant_category",
        "n_manual_review",
        "n_low_confidence",
        "avg_confidence",
    ]
    rows = []
    for parent, group in sorted(by_parent.items()):
        counts = Counter(r.category for r in group)
        total = len(group)
        pcts = {c: (counts.get(c, 0) / total if total else 0.0) for c in CATEGORIES}
        dominant = max(CATEGORIES, key=lambda c: counts.get(c, 0)) if total else ""
        avg_conf = sum(r.confidence for r in group) / total if total else 0.0
        rows.append(
            [
                parent,
                group[0].source_file,
                total,
                *[counts.get(c, 0) for c in CATEGORIES],
                *[f"{pcts[c]:.3f}" for c in CATEGORIES],
                dominant,
                sum(1 for r in group if r.manual_review),
                sum(1 for r in group if r.low_confidence),
                f"{avg_conf:.3f}",
            ]
        )

    path = summary_csv_path(settings)
    _write_csv(path, header, rows)
    return path


def write_review_queue(settings: Settings, records: list[ClassificationRecord]) -> Path:
    review = [r for r in records if r.manual_review or r.low_confidence]
    header = [
        "rule_id",
        "source_file",
        "category",
        "confidence",
        "manual_review",
        "low_confidence",
        "justification",
        "evidence",
    ]
    rows = [
        [
            r.rule_id,
            r.source_file,
            r.category,
            f"{r.confidence:.3f}",
            r.manual_review,
            r.low_confidence,
            r.justification,
            " | ".join(r.evidence),
        ]
        for r in review
    ]
    path = review_csv_path(settings)
    _write_csv(path, header, rows)
    return path


def write_attribute_vocabulary(settings: Settings, records: list[ClassificationRecord]) -> Path:
    """Une ligne par (attribut, valeur) avec sa fréquence et sa ventilation par catégorie.

    Sert de base pour consolider les thèmes/mots-clés libres en taxonomie fermée.
    """
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for r in records:
        for name in VOCABULARY_FIELDS:
            for value in _attribute_values(r, name):
                counts[(name, value)][r.category] += 1

    header = ["attribute", "value", "count", *[f"n_{c}" for c in CATEGORIES]]
    rows = [
        [name, value, sum(c.values()), *[c.get(cat, 0) for cat in CATEGORIES]]
        for (name, value), c in counts.items()
    ]
    rows.sort(key=lambda row: (VOCABULARY_FIELDS.index(row[0]), -row[2], row[1]))
    path = vocabulary_csv_path(settings)
    _write_csv(path, header, rows)
    return path


def _attribute_metrics(records: list[ClassificationRecord]) -> dict:
    top = {
        name: Counter(v for r in records for v in _attribute_values(r, name)).most_common(
            TOP_N_ATTRIBUTES
        )
        for name in VOCABULARY_FIELDS
    }
    theme_by_category: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(CATEGORIES, 0))
    for r in records:
        theme = r.theme.strip().lower() or "(vide)"
        theme_by_category[theme][r.category] += 1
    return {
        "distinct_themes": len(theme_by_category),
        "top": top,
        "theme_by_category": dict(
            sorted(theme_by_category.items(), key=lambda kv: -sum(kv[1].values()))
        ),
    }


def write_metrics(settings: Settings, records: list[ClassificationRecord]) -> Path:
    total = len(records)
    counts = Counter(r.category for r in records)
    manual = sum(1 for r in records if r.manual_review)
    low_conf = sum(1 for r in records if r.low_confidence)
    cached = sum(1 for r in records if r.cached)
    latencies = [r.latency_ms for r in records if not r.cached and r.latency_ms > 0]
    tokens_prompt = sum(r.prompt_tokens for r in records)
    tokens_completion = sum(r.completion_tokens for r in records)

    extraction_errors = sum(1 for _ in read_jsonl(extraction_errors_path(settings)))
    classification_errors = sum(1 for _ in read_jsonl(classification_errors_path(settings)))

    payload = {
        "totals": {
            "records": total,
            "cached": cached,
            "extraction_errors": extraction_errors,
            "classification_errors": classification_errors,
        },
        "distribution": {
            "counts": {c: counts.get(c, 0) for c in CATEGORIES},
            "percent": {
                c: (counts.get(c, 0) / total if total else 0.0) for c in CATEGORIES
            },
        },
        "quality": {
            "manual_review": manual,
            "low_confidence": low_conf,
            "avg_confidence": (
                sum(r.confidence for r in records) / total if total else 0.0
            ),
        },
        "latency_ms": {
            "count": len(latencies),
            "avg": (sum(latencies) / len(latencies) if latencies else 0.0),
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
        },
        "tokens": {
            "prompt": tokens_prompt,
            "completion": tokens_completion,
            "total": tokens_prompt + tokens_completion,
        },
        "attributes": _attribute_metrics(records),
    }
    path = metrics_json_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_all_reports(
    settings: Settings, records: list[ClassificationRecord] | None = None
) -> dict[str, Path]:
    records = records or load_classifications(settings)
    return {
        "detail": write_detailed_csv(settings, records),
        "summary_by_file": write_summary_by_file(settings, records),
        "review_queue": write_review_queue(settings, records),
        "metrics": write_metrics(settings, records),
        "attribute_vocabulary": write_attribute_vocabulary(settings, records),
    }
