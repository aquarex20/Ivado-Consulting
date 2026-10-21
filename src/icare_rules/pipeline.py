"""Orchestration du pipeline : ingestion, classification, checkpointing."""

from __future__ import annotations

import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Settings, load_settings
from .extraction import ExtractionError, extract_text
from .inventory import InventoryEntry, build_inventory, summarize, write_inventory
from .llm_client import (
    AzureOpenAIBackend,
    ClassificationCall,
    LLMBackend,
    LLMClient,
    OllamaBackend,
)
from .segmentation import Segment, iter_segments_for_classification, segment_rule
from .utils import read_jsonl, utc_now_iso, write_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chemins d'artefacts
# ---------------------------------------------------------------------------


def inventory_path(settings: Settings) -> Path:
    return settings.output_root / "inventory.jsonl"


def segments_path(settings: Settings) -> Path:
    return settings.output_root / "segments.jsonl"


def classifications_path(settings: Settings) -> Path:
    return settings.output_root / "classifications.jsonl"


def extraction_errors_path(settings: Settings) -> Path:
    return settings.output_root / "extraction_errors.jsonl"


def classification_errors_path(settings: Settings) -> Path:
    return settings.output_root / "classification_errors.jsonl"


def run_metadata_path(settings: Settings) -> Path:
    return settings.output_root / "run_metadata.json"


# ---------------------------------------------------------------------------
# Étape 1 : inventaire
# ---------------------------------------------------------------------------


def run_inventory(
    settings: Settings,
    *,
    workers: int | None = None,
    progress: bool = True,
    hash_files: bool = True,
) -> list[InventoryEntry]:
    entries = build_inventory(
        settings, workers=workers, progress=progress, hash_files=hash_files
    )
    out = inventory_path(settings)
    write_inventory(entries, out)
    logger.info("Inventaire écrit dans %s (%d entrées)", out, len(entries))
    logger.info("Résumé: %s", summarize(entries))
    return entries


def load_inventory(settings: Settings) -> list[InventoryEntry]:
    path = inventory_path(settings)
    entries: list[InventoryEntry] = []
    for data in read_jsonl(path):
        entries.append(InventoryEntry(**data))
    return entries


# ---------------------------------------------------------------------------
# Étape 2 : extraction + segmentation
# ---------------------------------------------------------------------------


@dataclass
class ExtractionFailure:
    rule_id: str
    source_file: str
    error: str
    warnings: list[str] = field(default_factory=list)
    at: str = field(default_factory=utc_now_iso)


def run_extract_and_segment(
    settings: Settings,
    entries: list[InventoryEntry] | None = None,
    *,
    limit: int | None = None,
    progress: bool = True,
) -> tuple[list[Segment], list[ExtractionFailure]]:
    entries = entries or load_inventory(settings)
    kept = [e for e in entries if e.kept]
    if limit is not None:
        kept = kept[:limit]

    from tqdm import tqdm as _tqdm

    segments: list[Segment] = []
    failures: list[ExtractionFailure] = []
    iterator = _tqdm(kept, desc="Extraction", unit="fichier") if progress else kept

    for entry in iterator:
        path = Path(entry.absolute_path)
        try:
            result = extract_text(path, settings)
        except ExtractionError as exc:
            failures.append(
                ExtractionFailure(
                    rule_id=entry.rule_id,
                    source_file=entry.source_file,
                    error=str(exc),
                )
            )
            continue

        file_segments = segment_rule(
            parent_rule_id=entry.rule_id,
            source_file=entry.source_file,
            source_hash=entry.sha256,
            text=result.text,
            settings=settings,
        )
        # Attacher les warnings d'extraction au segment agrégat
        if file_segments and result.warnings:
            file_segments[0].warnings.extend(
                f"extraction[{result.strategy}]: {w}" for w in result.warnings
            )
        segments.extend(file_segments)

    write_jsonl(segments_path(settings), (s.to_dict() for s in segments))
    if failures:
        write_jsonl(
            extraction_errors_path(settings), (asdict(f) for f in failures)
        )
    logger.info(
        "Extraction: %d segments, %d échecs (voir %s)",
        len(segments),
        len(failures),
        extraction_errors_path(settings) if failures else "",
    )
    return segments, failures


def load_segments(settings: Settings) -> list[Segment]:
    path = segments_path(settings)
    return [Segment(**data) for data in read_jsonl(path)]


# ---------------------------------------------------------------------------
# Étape 3 : classification LLM
# ---------------------------------------------------------------------------


@dataclass
class ClassificationRecord:
    rule_id: str
    parent_rule_id: str
    kind: str
    source_file: str
    source_hash: str
    text_hash: str
    char_count: int
    category: str
    justification: str
    evidence: list[str]
    confidence: float
    manual_review: bool
    low_confidence: bool
    model: str
    prompt_version: str
    cached: bool
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    warnings: list[str]
    at: str = field(default_factory=utc_now_iso)
    theme: str = ""
    keywords: list[str] = field(default_factory=list)
    diplomas: list[str] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    granted_value: str | None = None
    dossier_variables: list[str] = field(default_factory=list)


@dataclass
class ClassificationFailure:
    rule_id: str
    source_file: str
    error: str
    at: str = field(default_factory=utc_now_iso)


def _build_llm_client(settings: Settings, backend: LLMBackend | None = None) -> LLMClient:
    prompt_path = settings.prompt_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt introuvable: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")

    cfg = settings.classification
    if cfg.backend == "ollama":
        deployment = cfg.deployment or os.environ.get("OLLAMA_MODEL")
        if not deployment:
            raise RuntimeError(
                "Modèle Ollama non défini (config.classification.deployment ou OLLAMA_MODEL)."
            )
        backend = backend or OllamaBackend(
            host=cfg.ollama_host,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            num_ctx=cfg.ollama_num_ctx,
            themes=cfg.themes,
        )
        return LLMClient(settings, prompt=prompt, backend=backend, deployment=deployment)

    deployment = cfg.deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise RuntimeError(
            "Deployment Azure OpenAI non défini "
            "(config.classification.deployment ou AZURE_OPENAI_DEPLOYMENT)."
        )

    backend = backend or AzureOpenAIBackend(
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        themes=cfg.themes,
    )
    return LLMClient(settings, prompt=prompt, backend=backend, deployment=deployment)


def _completed_rule_ids(path: Path) -> set[str]:
    return {rec["rule_id"] for rec in read_jsonl(path)}


def _record_from_call(seg: Segment, call: ClassificationCall, low_thr: float) -> ClassificationRecord:
    c = call.classification
    return ClassificationRecord(
        rule_id=seg.rule_id,
        parent_rule_id=seg.parent_rule_id,
        kind=seg.kind,
        source_file=seg.source_file,
        source_hash=seg.source_hash,
        text_hash=seg.text_hash,
        char_count=seg.char_count,
        category=call.classification.category.value,
        justification=call.classification.justification,
        evidence=list(call.classification.evidence),
        confidence=call.classification.confidence,
        manual_review=call.classification.manual_review,
        low_confidence=call.classification.confidence < low_thr,
        model=call.model,
        prompt_version=call.prompt_version,
        cached=call.cached,
        latency_ms=call.latency_ms,
        prompt_tokens=call.usage.get("prompt_tokens", 0),
        completion_tokens=call.usage.get("completion_tokens", 0),
        total_tokens=call.usage.get("total_tokens", 0),
        warnings=list(seg.warnings),
        theme=c.theme,
        keywords=list(c.keywords),
        diplomas=list(c.diplomas),
        institutions=list(c.institutions),
        countries=list(c.countries),
        prerequisites=list(c.prerequisites),
        granted_value=c.granted_value,
        dossier_variables=list(c.dossier_variables),
    )


def run_classify(
    settings: Settings,
    *,
    segments: list[Segment] | None = None,
    backend: LLMBackend | None = None,
    resume: bool = True,
    limit: int | None = None,
    shuffle_seed: int | None = None,
) -> tuple[list[ClassificationRecord], list[ClassificationFailure]]:
    segments = segments or load_segments(settings)
    to_classify = list(iter_segments_for_classification(segments))
    if shuffle_seed is not None:
        # Ordre aléatoire reproductible : tout résultat partiel reste représentatif du corpus.
        random.Random(shuffle_seed).shuffle(to_classify)
    if limit is not None:
        to_classify = to_classify[:limit]

    out_path = classifications_path(settings)
    err_path = classification_errors_path(settings)

    already: set[str] = _completed_rule_ids(out_path) if resume and out_path.exists() else set()
    if already:
        to_classify = [s for s in to_classify if s.rule_id not in already]
        logger.info("Reprise: %d déjà classées, %d à traiter", len(already), len(to_classify))

    llm = _build_llm_client(settings, backend=backend)

    records: list[ClassificationRecord] = []
    failures: list[ClassificationFailure] = []
    low_thr = settings.classification.low_confidence_threshold
    checkpoint_every = max(1, settings.output.checkpoint_every)
    max_concurrency = max(1, settings.classification.max_concurrency)

    pending_records: list[ClassificationRecord] = []
    pending_failures: list[ClassificationFailure] = []

    def _flush() -> None:
        nonlocal pending_records, pending_failures
        if pending_records:
            write_jsonl(out_path, (asdict(r) for r in pending_records), append=True)
            records.extend(pending_records)
            pending_records = []
        if pending_failures:
            write_jsonl(err_path, (asdict(f) for f in pending_failures), append=True)
            failures.extend(pending_failures)
            pending_failures = []
        done = len(records) + len(failures)
        logger.info(
            "Progression: %d/%d (%.1f%%), %d échec(s)",
            done,
            len(to_classify),
            100 * done / max(1, len(to_classify)),
            len(failures),
        )

    def _process(seg: Segment) -> ClassificationRecord | ClassificationFailure:
        try:
            call = llm.classify(
                rule_id=seg.rule_id,
                source_file=seg.source_file,
                text=seg.text_normalized,
                text_hash=seg.text_hash,
            )
            return _record_from_call(seg, call, low_thr)
        except Exception as exc:  # noqa: BLE001 - toutes les erreurs deviennent des échecs traçables
            return ClassificationFailure(
                rule_id=seg.rule_id,
                source_file=seg.source_file,
                error=f"{type(exc).__name__}: {exc}",
            )

    if max_concurrency == 1:
        for i, seg in enumerate(to_classify, start=1):
            outcome = _process(seg)
            if isinstance(outcome, ClassificationRecord):
                pending_records.append(outcome)
            else:
                pending_failures.append(outcome)
            if i % checkpoint_every == 0:
                _flush()
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = {pool.submit(_process, seg): seg for seg in to_classify}
            for i, fut in enumerate(as_completed(futures), start=1):
                outcome = fut.result()
                if isinstance(outcome, ClassificationRecord):
                    pending_records.append(outcome)
                else:
                    pending_failures.append(outcome)
                if i % checkpoint_every == 0:
                    _flush()

    _flush()

    logger.info(
        "Classification terminée: %d succès, %d échecs (voir %s)",
        len(records),
        len(failures),
        err_path if failures else "",
    )
    return records, failures


def load_classifications(settings: Settings) -> list[ClassificationRecord]:
    path = classifications_path(settings)
    return [ClassificationRecord(**rec) for rec in read_jsonl(path)]


# ---------------------------------------------------------------------------
# Pilote stratifié
# ---------------------------------------------------------------------------


def sample_pilot(
    segments: list[Segment],
    *,
    sample_g: int,
    sample_p: int,
    seed: int,
) -> list[Segment]:
    to_classify = list(iter_segments_for_classification(segments))
    g = [s for s in to_classify if s.parent_rule_id.startswith("G")]
    p = [s for s in to_classify if s.parent_rule_id.startswith("P")]
    rng = random.Random(seed)
    rng.shuffle(g)
    rng.shuffle(p)
    return g[:sample_g] + p[:sample_p]


# ---------------------------------------------------------------------------
# Écriture des métadonnées d'exécution
# ---------------------------------------------------------------------------


def write_run_metadata(settings: Settings, extra: dict | None = None) -> Path:
    import json

    payload = {
        "at": utc_now_iso(),
        "backend": settings.classification.backend,
        "prompt_version": settings.classification.prompt_version,
        "deployment": settings.classification.deployment
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        "temperature": settings.classification.temperature,
        "low_confidence_threshold": settings.classification.low_confidence_threshold,
    }
    if extra:
        payload.update(extra)
    path = run_metadata_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def get_settings(settings_path: Path | None = None) -> Settings:
    return load_settings(settings_path=settings_path)
