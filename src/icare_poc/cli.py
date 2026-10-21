"""CLI for the Phase 1 retrieval PoC."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from .ingest import run_ingest
from .paths import MAX_RULES_DEFAULT, catalog_path, project_root
from .search import retrieve

app = typer.Typer(add_completion=False, help="PoC ICARE: ingest 25 rules into Qdrant and query them.", no_args_is_help=True)
console = Console()


@app.command()
def ingest(
    max_rules: int = typer.Option(MAX_RULES_DEFAULT, min=2, max=30),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Extract a small clean subset, write PDF/Markdown, and index Qdrant."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s")
    chunks = run_ingest(project_root(), max_rules=max_rules)
    docs = {c.document_id for c in chunks}
    table = Table(title="Sous-ensemble indexé")
    table.add_column("Règles")
    table.add_column("Chunks Qdrant")
    table.add_column("Catalogue")
    table.add_row(str(len(docs)), str(len(chunks)), str(catalog_path(project_root()).relative_to(project_root())))
    console.print(table)


@app.command()
def query(question: str) -> None:
    """Print the first hybrid hit (for a quick terminal check)."""
    hits = retrieve(question, root=project_root(), limit=3)
    if not hits:
        console.print("[red]Aucun résultat[/]")
        raise typer.Exit(1)
    for hit in hits:
        console.print(
            f"[bold]{hit.chunk_id}[/]  rrf={hit.score:.3f}  "
            f"{hit.payload.get('source_filename')} p.{hit.payload.get('page')}"
        )
        console.print(hit.payload.get("excerpt") or "")
        console.print()


@app.command()
def app_cmd() -> None:
    """Launch the Streamlit agent view."""
    import subprocess
    import sys

    target = project_root() / "app_poc.py"
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(target)]))


# Typer reserves `app`; expose a friendlier name.
app.command("ui")(app_cmd)
