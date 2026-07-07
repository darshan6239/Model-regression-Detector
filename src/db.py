"""
SQLite persistence for eval run history. Every run's metadata, aggregate
stats, and per-case results get stored here so the comparator and dashboard
can query history.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent.parent / "runs" / "run_history.db"


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            prompt_version TEXT NOT NULL,
            model TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            aggregate_stats_json TEXT NOT NULL,
            results_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_run(run_id: str, prompt_version: str, model: str, dataset_version: str,
             aggregate_stats: dict, results: list[dict]) -> None:
    init_db()
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO runs
           (run_id, prompt_version, model, timestamp, dataset_version, aggregate_stats_json, results_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            prompt_version,
            model,
            datetime.now(timezone.utc).isoformat(),
            dataset_version,
            json.dumps(aggregate_stats),
            json.dumps(results),
        ),
    )
    conn.commit()
    conn.close()


def get_latest_run(exclude_run_id: str = None) -> dict | None:
    """Fetches the most recent run, optionally excluding a specific run_id
    (used to fetch the 'baseline' when comparing against the current run)."""
    init_db()
    conn = get_connection()
    if exclude_run_id:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id != ? ORDER BY timestamp DESC LIMIT 1",
            (exclude_run_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_run_by_id(run_id: str) -> dict | None:
    init_db()
    conn = get_connection()
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_recent_runs(limit: int = 10) -> list[dict]:
    init_db()
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["aggregate_stats"] = json.loads(d.pop("aggregate_stats_json"))
    d["results"] = json.loads(d.pop("results_json"))
    return d
