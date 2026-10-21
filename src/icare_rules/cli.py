"""Interface en ligne de commande `icare-rules`."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import pipeline as pl
from . import profiling, reporting

app = typer.Typer(
    add_completion=False,
    help="Pipeline de classification des règles ICARE.",
    no_args_is_help=True,
)
console = Console()


def _setup(settings_file: Path | None, verbose: bool) -> pl.Settings:
    load_dotenv(override=False)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    return pl.load_settings(settings_path=settings_file)


@app.command()
def inventory(
    settings_file: Path = typer.Option(None, "--settings", help="Fichier YAML de config."),
    workers: int = typer.Option(
        None, help="Threads pour hachage (utile sur OneDrive/iCloud lents)."
    ),
    no_hash: bool = typer.Option(
        False,
        "--no-hash",
        help="Ne pas calculer les SHA-256 (utile pour un premier passage rapide sur OneDrive).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Inventorie le corpus et écrit `data/output/inventory.jsonl`."""
    settings = _setup(settings_file, verbose)
    entries = pl.run_inventory(settings, workers=workers, hash_files=not no_hash)
    summary = pl.summarize(entries)

    table = Table(title="Inventaire du corpus", show_lines=False)
    table.add_column("Métrique")
    table.add_column("Valeur", justify="right")
    for key, value in [
        ("Fichiers découverts", summary["total"]),
        ("Fichiers retenus", summary["kept"]),
        ("Doublons ignorés", summary["duplicates"]),
        ("Règles G", summary["by_kind"]["G"]),
        ("Règles P", summary["by_kind"]["P"]),
    ]:
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def extract(
    settings_file: Path = typer.Option(None, "--settings", help="Fichier YAML de config."),
    limit: int = typer.Option(None, help="Limiter le nombre de fichiers extraits."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Extrait le texte et segmente en sous-règles."""
    settings = _setup(settings_file, verbose)
    if not pl.inventory_path(settings).exists():
        console.print("[yellow]Aucun inventaire trouvé, lancement de `inventory`...[/]")
        pl.run_inventory(settings)
    segments, failures = pl.run_extract_and_segment(settings, limit=limit)
    console.print(
        f"[green]Extraction terminée[/]: {len(segments)} segments, "
        f"{len(failures)} échec(s)."
    )
    if failures:
        console.print(
            f"[yellow]Voir[/] {pl.extraction_errors_path(settings)}"
        )


@app.command()
def profile(
    settings_file: Path = typer.Option(None, "--settings", help="Fichier YAML de config."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Profile les règles sans LLM (structure, quantités, renvois, gabarits, mots-clés TF-IDF)."""
    settings = _setup(settings_file, verbose)
    if not pl.segments_path(settings).exists():
        console.print("[yellow]Aucun segment trouvé, lancement de `extract`...[/]")
        pl.run_extract_and_segment(settings)
    profiles = profiling.build_profiles(pl.load_segments(settings))
    paths = profiling.write_profiles(settings, profiles)
    summary = profiling.summarize_profiles(profiles)

    table = Table(title="Profil du corpus")
    table.add_column("Métrique")
    table.add_column("Valeur", justify="right")
    for key, value in summary["totals"].items():
        table.add_row(key, str(value))
    for key, value in summary["features_share"].items():
        table.add_row(f"part {key}", f"{value:.1%}")
    console.print(table)
    for name, path in paths.items():
        console.print(f"  {name}: {path}")


def _resolve_backend():
    from .llm_client import AzureOpenAIBackend

    return AzureOpenAIBackend


BACKEND_OPTION = typer.Option(None, "--backend", help="`azure` ou `ollama` (surcharge la config).")
MODEL_OPTION = typer.Option(None, "--model", help="Deployment Azure ou modèle Ollama (ex. llama3.2).")


def _apply_llm_overrides(settings: pl.Settings, backend: str | None, model: str | None) -> None:
    if backend:
        if backend not in ("azure", "ollama"):
            raise typer.BadParameter("--backend doit valoir `azure` ou `ollama`.")
        settings.classification.backend = backend
    if model:
        settings.classification.deployment = model


@app.command()
def classify(
    settings_file: Path = typer.Option(None, "--settings", help="Fichier YAML de config."),
    limit: int = typer.Option(None, help="Limiter le nombre de sous-règles."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignorer les résultats existants."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Utiliser un backend heuristique local (aucun appel Azure).",
    ),
    llm_backend: str = BACKEND_OPTION,
    model: str = MODEL_OPTION,
    shuffle: bool = typer.Option(
        False,
        "--shuffle",
        help="Ordre aléatoire reproductible (graine `pilot.random_seed`) : résultats partiels représentatifs.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Classe chaque sous-règle avec Azure OpenAI, Ollama (local) ou l'heuristique de dry-run."""
    settings = _setup(settings_file, verbose)
    _apply_llm_overrides(settings, llm_backend, model)
    backend = None
    if dry_run:
        from .dry_run_backend import DryRunBackend

        backend = DryRunBackend()
        # Force un deployment neutre pour que _build_llm_client ne râle pas
        settings.classification.deployment = (
            settings.classification.deployment or "dry-run"
        )
    elif settings.classification.backend == "azure":
        _resolve_backend()  # vérifie que le paquet openai est disponible
    if not pl.segments_path(settings).exists():
        console.print("[yellow]Aucun segment trouvé, lancement de `extract`...[/]")
        pl.run_extract_and_segment(settings)
    records, failures = pl.run_classify(
        settings,
        backend=backend,
        resume=not no_resume,
        limit=limit,
        shuffle_seed=settings.pilot.random_seed if shuffle else None,
    )
    pl.write_run_metadata(
        settings,
        extra={"limit": limit, "resume": not no_resume, "dry_run": dry_run, "shuffle": shuffle},
    )
    console.print(
        f"[green]Classification terminée[/]: {len(records)} succès, "
        f"{len(failures)} échec(s)."
    )


@app.command()
def pilot(
    settings_file: Path = typer.Option(None, "--settings", help="Fichier YAML de config."),
    sample_g: int = typer.Option(None, "--sample-g", help="Taille de l'échantillon G."),
    sample_p: int = typer.Option(None, "--sample-p", help="Taille de l'échantillon P."),
    llm_backend: str = BACKEND_OPTION,
    model: str = MODEL_OPTION,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Exécute un pilote stratifié sur un échantillon G/P."""
    settings = _setup(settings_file, verbose)
    _apply_llm_overrides(settings, llm_backend, model)
    if sample_g is not None:
        settings.pilot.sample_size_g = sample_g
    if sample_p is not None:
        settings.pilot.sample_size_p = sample_p
    if not pl.segments_path(settings).exists():
        pl.run_inventory(settings)
        pl.run_extract_and_segment(settings)
    segments = pl.load_segments(settings)
    sample = pl.sample_pilot(
        segments,
        sample_g=settings.pilot.sample_size_g,
        sample_p=settings.pilot.sample_size_p,
        seed=settings.pilot.random_seed,
    )
    console.print(f"Pilote: {len(sample)} sous-règles échantillonnées.")
    records, failures = pl.run_classify(settings, segments=sample, resume=True)
    pl.write_run_metadata(
        settings, extra={"mode": "pilot", "sample_size": len(sample)}
    )
    console.print(
        f"[green]Pilote terminé[/]: {len(records)} succès, "
        f"{len(failures)} échec(s)."
    )


@app.command()
def report(
    settings_file: Path = typer.Option(None, "--settings", help="Fichier YAML de config."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Génère CSV détaillé, synthèse par fichier, file de révision et métriques."""
    settings = _setup(settings_file, verbose)
    paths = reporting.build_all_reports(settings)
    table = Table(title="Rapports générés")
    table.add_column("Artefact")
    table.add_column("Chemin")
    for name, path in paths.items():
        table.add_row(name, str(path.relative_to(settings.project_root)))
    console.print(table)


@app.command()
def run(
    settings_file: Path = typer.Option(None, "--settings", help="Fichier YAML de config."),
    limit: int = typer.Option(None, help="Limiter le nombre de sous-règles."),
    llm_backend: str = BACKEND_OPTION,
    model: str = MODEL_OPTION,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Exécute la totalité du pipeline : inventaire → extraction → profil → classification → rapport."""
    settings = _setup(settings_file, verbose)
    _apply_llm_overrides(settings, llm_backend, model)
    pl.run_inventory(settings)
    segments, _failures = pl.run_extract_and_segment(settings)
    profiling.write_profiles(settings, profiling.build_profiles(segments))
    pl.run_classify(settings, limit=limit)
    pl.write_run_metadata(settings, extra={"limit": limit, "mode": "run"})
    reporting.build_all_reports(settings)
    console.print("[green]Pipeline complet terminé.[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
