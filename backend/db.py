"""The platform database (PostgreSQL + PostGIS), FarmScan's own schema ("farm").

    with db.tx() as conn:            # one transaction: commits on success, rolls back on any error
        conn.execute("SELECT …", (…,)).fetchone()     # rows are dicts
"""
import logging
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config import settings

log = logging.getLogger("farmscan.db")
MIGRATIONS = Path(__file__).resolve().parent / "migrations"


def options(extra: str = "") -> str:
    path = ",".join(dict.fromkeys(x for x in (settings.db_schema, "public", extra) if x))
    return f"-c search_path={path} -c statement_timeout=60000"


pool = ConnectionPool(settings.database_url, open=False, min_size=1, max_size=settings.db_pool_max,
                      kwargs={"row_factory": dict_row, "options": options(), "prepare_threshold": None})


def postgis_schema(conn) -> str:
    """The schema PostGIS is installed in (installing it in public when the database does not have it yet)."""
    row = conn.execute("SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace "
                       "WHERE e.extname = 'postgis'").fetchone()
    if row:
        return row[0]
    conn.execute("CREATE EXTENSION IF NOT EXISTS postgis SCHEMA public")
    return "public"


def migrate() -> list[str]:
    """Apply any migrations/*.sql not yet recorded. Safe to run from several processes at once."""
    applied: list[str] = []
    with psycopg.connect(settings.database_url, autocommit=True, options=options()) as conn:
        conn.execute("SELECT pg_advisory_lock(717171)")
        try:
            gis = postgis_schema(conn)
            if gis != "public":
                pool.kwargs["options"] = options(gis)
                conn.execute(f'SET search_path = "{settings.db_schema}", public, "{gis}"')
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{settings.db_schema}"')
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{settings.db_schema}".schema_migrations '
                         "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
            done = {r[0] for r in conn.execute(f'SELECT name FROM "{settings.db_schema}".schema_migrations')}
            for f in sorted(MIGRATIONS.glob("*.sql")):
                if f.name in done:
                    continue
                with conn.transaction():
                    conn.execute(f.read_text())
                    conn.execute(f'INSERT INTO "{settings.db_schema}".schema_migrations (name) VALUES (%s)', (f.name,))
                applied.append(f.name)
                log.info("applied migration %s", f.name)
        finally:
            conn.execute("SELECT pg_advisory_unlock(717171)")
    return applied


def start() -> None:
    migrate()
    if pool.closed:
        pool.open(wait=True)


@contextmanager
def tx():
    with pool.connection() as conn:
        yield conn
