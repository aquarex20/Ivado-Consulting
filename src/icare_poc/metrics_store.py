"""Local SQLite log of each PoC query.

Raw measurements live in ``query_runs``. Scores are not stored: the views below
recompute them. To change a formula, edit ``COMPUTED`` or ``DAILY`` and the next
insert recreates the views. Past rows stay as measured.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Computed from raw columns. ``sql`` is the expression used in query_scores.
COMPUTED: tuple[dict[str, str], ...] = (
    {
        "name": "score_delta",
        "sql": "CASE WHEN score IS NULL OR second_score IS NULL THEN NULL ELSE round(score - second_score, 6) END",
        "description": "Écart de score RRF entre le 1er hit et le 2e. Null s'il n'y a qu'un hit.",
    },
    {
        "name": "zero_lexical",
        "sql": "CASE WHEN bm25_score IS NULL THEN NULL WHEN bm25_score = 0 THEN 1 ELSE 0 END",
        "description": "1 quand le BM25 du hit retenu vaut exactement 0.",
    },
    {
        "name": "refusal",
        "sql": "CASE WHEN answer LIKE 'Je ne sais pas%' THEN 1 ELSE 0 END",
        "description": "1 quand le modèle dit qu'il ne sait pas.",
    },
    {
        "name": "completion_tokens_per_s",
        "sql": "CASE WHEN model_latency_ms > 0 AND completion_tokens IS NOT NULL THEN completion_tokens * 1000.0 / model_latency_ms END",
        "description": "Tokens générés par seconde, à partir du temps de l'appel modèle.",
    },
)

# One row per day, built on top of query_scores.
DAILY: tuple[dict[str, str], ...] = (
    {"name": "runs", "sql": "count(*)", "description": "Nombre de requêtes du jour."},
    {"name": "n_done", "sql": "sum(status = 'done')", "description": "Requêtes terminées avec une réponse."},
    {"name": "n_writer_error", "sql": "sum(status = 'writer_error')", "description": "Appels modèle en échec."},
    {"name": "n_retriever_error", "sql": "sum(status = 'retriever_error')", "description": "Recherches sans hit."},
    {"name": "avg_score_delta", "sql": "avg(score_delta)", "description": "Écart RRF moyen."},
    {"name": "zero_lexical_rate", "sql": "avg(zero_lexical)", "description": "Part des hits sans aucun mot de la question."},
    {"name": "refusal_rate", "sql": "avg(refusal)", "description": "Part des réponses « Je ne sais pas »."},
    {"name": "avg_latency_ms", "sql": "avg(latency_ms)", "description": "Temps moyen de la requête complète."},
    {"name": "avg_model_latency_ms", "sql": "avg(model_latency_ms)", "description": "Temps moyen de l'appel modèle."},
    {"name": "avg_score", "sql": "avg(score)", "description": "Score RRF moyen du hit retenu."},
    {"name": "avg_bm25", "sql": "avg(bm25_score)", "description": "BM25 moyen du hit retenu."},
    {"name": "avg_dense", "sql": "avg(dense_score)", "description": "Similarité dense moyenne du hit retenu."},
)

RAW_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("at", "TEXT", "Horodatage UTC du début de la requête."),
    ("pipeline", "TEXT", "Voie de traitement. Aujourd'hui : qualitative."),
    ("status", "TEXT", "retriever_error, retrieved, writer_error ou done."),
    ("question", "TEXT", "Prompt de l'utilisateur."),
    ("model_prompt", "TEXT", "Prompt envoyé au modèle."),
    ("answer", "TEXT", "Réponse du modèle."),
    ("latency_ms", "INTEGER", "Temps total de la requête, en millisecondes."),
    ("model_latency_ms", "INTEGER", "Temps de l'appel modèle, en millisecondes."),
    ("model", "TEXT", "Nom du modèle."),
    ("prompt_tokens", "INTEGER", "Tokens du prompt, si le modèle les renvoie."),
    ("completion_tokens", "INTEGER", "Tokens générés, si le modèle les renvoie."),
    ("total_duration_ms", "INTEGER", "total_duration Ollama, converti en millisecondes."),
    ("load_duration_ms", "INTEGER", "load_duration Ollama, converti en millisecondes."),
    ("prompt_eval_duration_ms", "INTEGER", "prompt_eval_duration Ollama, converti en millisecondes."),
    ("eval_duration_ms", "INTEGER", "eval_duration Ollama, converti en millisecondes."),
    ("chunk_id", "TEXT", "Chunk retenu."),
    ("document_id", "TEXT", "Code de règle du chunk retenu."),
    ("score", "REAL", "Score RRF du 1er hit, avant arrondi d'affichage."),
    ("second_score", "REAL", "Score RRF du 2e hit. Null s'il n'y en a pas."),
    ("dense_score", "REAL", "Similarité cosinus du hit retenu."),
    ("bm25_score", "REAL", "Score BM25 du hit retenu."),
    ("citation_method", "TEXT", "Méthode de localisation dans le PDF."),
)

_INSERT_COLUMNS = tuple(name for name, _, _ in RAW_COLUMNS)


def score_delta(score: float | None, second_score: float | None) -> float | None:
    """Same formula as the ``score_delta`` view expression."""
    if score is None or second_score is None:
        return None
    return round(score - second_score, 6)


def zero_lexical(bm25_score: float | None) -> bool | None:
    """Same formula as the ``zero_lexical`` view expression."""
    if bm25_score is None:
        return None
    return bm25_score == 0.0


def db_path(root: Path) -> Path:
    return root / "data" / "output" / "poc_metrics.sqlite"


def record_run(root: Path, row: dict[str, Any]) -> None:
    path = db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _ensure(conn)
        values = [row.get(name) for name in _INSERT_COLUMNS]
        placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
        columns = ", ".join(_INSERT_COLUMNS)
        conn.execute(f"INSERT INTO query_runs ({columns}) VALUES ({placeholders})", values)


def _ensure(conn: sqlite3.Connection) -> None:
    column_sql = ",\n  ".join(f"{name} {kind}" for name, kind, _ in RAW_COLUMNS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS query_runs (\n  id INTEGER PRIMARY KEY,\n  {column_sql}\n)")
    conn.execute("CREATE INDEX IF NOT EXISTS query_runs_at ON query_runs(at)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metric_definitions (
          name TEXT PRIMARY KEY,
          kind TEXT NOT NULL,
          expression TEXT NOT NULL,
          description TEXT NOT NULL
        )
        """
    )
    conn.execute("DELETE FROM metric_definitions")
    conn.executemany(
        "INSERT INTO metric_definitions (name, kind, expression, description) VALUES (?, ?, ?, ?)",
        [("raw." + name, "raw", name, description) for name, _, description in RAW_COLUMNS],
    )
    conn.executemany(
        "INSERT INTO metric_definitions (name, kind, expression, description) VALUES (?, ?, ?, ?)",
        [(item["name"], "computed", item["sql"], item["description"]) for item in (*COMPUTED, *DAILY)],
    )
    score_columns = ",\n  ".join(f"{item['sql']} AS {item['name']}" for item in COMPUTED)
    conn.execute("DROP VIEW IF EXISTS query_scores_daily")
    conn.execute("DROP VIEW IF EXISTS query_scores")
    conn.execute(
        f"""
        CREATE VIEW query_scores AS
        SELECT
          id, at, pipeline, status, question, model_prompt, answer,
          latency_ms, model_latency_ms, model,
          prompt_tokens, completion_tokens,
          total_duration_ms, load_duration_ms, prompt_eval_duration_ms, eval_duration_ms,
          chunk_id, document_id, score, second_score, dense_score, bm25_score, citation_method,
          {score_columns}
        FROM query_runs
        """
    )
    daily_columns = ",\n  ".join(f"{item['sql']} AS {item['name']}" for item in DAILY)
    conn.execute(
        f"""
        CREATE VIEW query_scores_daily AS
        SELECT
          substr(at, 1, 10) AS day,
          {daily_columns}
        FROM query_scores
        GROUP BY substr(at, 1, 10)
        """
    )
