"""Load the raw CSVs into SQLite and run the SQL feature-engineering scripts.

The SQL in ``sql/`` is standard (CASE, NULLIF, COALESCE, LEFT JOIN, GROUP BY)
and runs unchanged on PostgreSQL; SQLite is used so the project needs no server.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"

RAW_TABLES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
}


def connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode = OFF")
    con.execute("PRAGMA synchronous = OFF")
    return con


def load_raw(con: sqlite3.Connection, data_dir: str | Path, chunksize: int = 200_000) -> dict[str, int]:
    """Load each raw CSV into its own table, in chunks to keep memory flat."""
    data_dir = Path(data_dir)
    counts: dict[str, int] = {}
    for table, filename in RAW_TABLES.items():
        con.execute(f"DROP TABLE IF EXISTS {table}")
        n = 0
        for chunk in pd.read_csv(data_dir / filename, chunksize=chunksize):
            chunk.to_sql(table, con, if_exists="append", index=False)
            n += len(chunk)
        counts[table] = n
    con.execute("CREATE INDEX IF NOT EXISTS ix_bureau_curr ON bureau (SK_ID_CURR)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_app_curr ON application (SK_ID_CURR)")
    con.commit()
    return counts


def run_script(con: sqlite3.Connection, name: str) -> None:
    con.executescript((SQL_DIR / name).read_text())
    con.commit()


def build_features(con: sqlite3.Connection) -> pd.DataFrame:
    run_script(con, "01_bureau_features.sql")
    run_script(con, "02_model_features.sql")
    return pd.read_sql_query("SELECT * FROM model_features ORDER BY SK_ID_CURR", con)


def named_queries(name: str = "03_portfolio_eda.sql") -> dict[str, str]:
    """Split a SQL file into queries keyed by their ``-- name: xyz`` header."""
    text = (SQL_DIR / name).read_text()
    parts = re.split(r"^--\s*name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def portfolio_views(con: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    return {k: pd.read_sql_query(q, con) for k, q in named_queries().items()}
