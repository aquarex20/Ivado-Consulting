"""Inventaire du corpus : découverte, hash, dédoublonnage des .doc/.docx."""

from __future__ import annotations

import fnmatch
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from .config import Settings
from .utils import sha256_file, utc_now_iso, write_jsonl

logger = logging.getLogger(__name__)

# Parallélisme d'accès disque pour l'inventaire ; utile quand les fichiers
# sont sur OneDrive/iCloud avec fetch à la demande. Réglable via variable
# d'environnement `ICARE_INVENTORY_WORKERS`.
_DEFAULT_INVENTORY_WORKERS = 16

RULE_ID_RE = re.compile(r"^([GP])(\d{2,5})", re.IGNORECASE)


@dataclass
class InventoryEntry:
    rule_id: str  # ex. G0539, P0048
    kind: str  # "G" ou "P"
    source_file: str  # chemin relatif à project_root
    absolute_path: str  # chemin absolu
    extension: str  # .doc, .docx, ...
    file_size: int
    sha256: str
    kept: bool  # True si retenu pour la suite du pipeline
    duplicate_of: str | None = None  # rule_id doublon si `kept=False`
    reason: str = ""  # motif de dédoublonnage / rejet
    discovered_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_rule_id(name: str) -> tuple[str, str] | None:
    """Retourne (rule_id, kind) depuis un nom de fichier tel que `G0539.docx`."""
    stem = Path(name).stem
    m = RULE_ID_RE.match(stem)
    if not m:
        return None
    kind = m.group(1).upper()
    number = m.group(2)
    # Normalise sur 4 chiffres (le corpus utilise 4 chiffres partout)
    if len(number) < 4:
        number = number.zfill(4)
    return f"{kind}{number}", kind


def _matches_globs(name: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in globs)


def _has_excluded_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def _discover_files(settings: Settings) -> list[Path]:
    files: list[Path] = []
    for root in settings.corpus_paths():
        if not root.exists():
            continue
        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            if not entry.is_file():
                continue
            if _has_excluded_prefix(entry.name, settings.corpus.exclude_prefixes):
                continue
            if not _matches_globs(entry.name, settings.corpus.include_globs):
                continue
            files.append(entry)
    return files


def _extension_rank(ext: str) -> int:
    """Priorité pour choisir la version canonique : .docx > .doc."""
    ext = ext.lower()
    if ext == ".docx":
        return 0
    if ext == ".doc":
        return 1
    return 2


def _make_entry(
    path: Path, settings: Settings, *, do_hash: bool = True
) -> InventoryEntry | None:
    rule = _extract_rule_id(path.name)
    if not rule:
        return None
    rule_id, kind = rule
    stat = path.stat()
    sha = sha256_file(path) if do_hash else ""
    return InventoryEntry(
        rule_id=rule_id,
        kind=kind,
        source_file=str(path.relative_to(settings.project_root)),
        absolute_path=str(path),
        extension=path.suffix.lower(),
        file_size=stat.st_size,
        sha256=sha,
        kept=True,
    )


def build_inventory(
    settings: Settings,
    *,
    workers: int | None = None,
    progress: bool = True,
    hash_files: bool = True,
) -> list[InventoryEntry]:
    """Construit l'inventaire, dédoublonne les .doc/.docx par `rule_id`.

    Le hachage est parallélisé (I/O bound) pour supporter les dossiers
    OneDrive/iCloud avec fetch à la demande.
    """
    import os

    paths = _discover_files(settings)
    logger.info("Découverte: %d fichiers candidats", len(paths))

    workers = workers or int(os.environ.get("ICARE_INVENTORY_WORKERS", _DEFAULT_INVENTORY_WORKERS))
    entries: list[InventoryEntry] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_make_entry, p, settings, do_hash=hash_files) for p in paths]
        iterator = as_completed(futures)
        if progress:
            desc = "Hachage fichiers" if hash_files else "Inventaire fichiers"
            iterator = tqdm(iterator, total=len(futures), desc=desc, unit="fichier")
        for fut in iterator:
            entry = fut.result()
            if entry is not None:
                entries.append(entry)

    # Dédoublonnage par rule_id : garder l'extension prioritaire (.docx > .doc).
    by_rule: dict[str, list[InventoryEntry]] = {}
    for e in entries:
        by_rule.setdefault(e.rule_id, []).append(e)

    for rule_id, group in by_rule.items():
        if len(group) == 1:
            continue
        group.sort(key=lambda e: (_extension_rank(e.extension), -e.file_size, e.source_file))
        winner = group[0]
        for loser in group[1:]:
            loser.kept = False
            loser.duplicate_of = rule_id
            loser.reason = f"doublon de {winner.source_file}"

    return entries


def write_inventory(entries: list[InventoryEntry], output_path: Path) -> int:
    return write_jsonl(output_path, (e.to_dict() for e in entries))


def summarize(entries: list[InventoryEntry]) -> dict:
    kept = [e for e in entries if e.kept]
    return {
        "total": len(entries),
        "kept": len(kept),
        "duplicates": len(entries) - len(kept),
        "by_kind": {
            "G": sum(1 for e in kept if e.kind == "G"),
            "P": sum(1 for e in kept if e.kind == "P"),
        },
        "by_extension": {
            ext: sum(1 for e in kept if e.extension == ext)
            for ext in sorted({e.extension for e in kept})
        },
    }
