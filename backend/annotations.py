"""
Share & Collaborate: generates read-only share tokens for a boundary+result,
and stores annotations (point comments) on a farm — matching Solvi's
"Annotations and comments" + "Sharing via public web links" features.
"""
import secrets
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "farmscan.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_sharing_tables():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS share_links (
            token TEXT PRIMARY KEY,
            boundary_id TEXT NOT NULL,
            index_name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id TEXT PRIMARY KEY,
            boundary_id TEXT NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            comment TEXT NOT NULL,
            author TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def create_share_link(boundary_id: str, index_name: str = None) -> str:
    token = secrets.token_urlsafe(12)
    conn = get_db()
    conn.execute(
        "INSERT INTO share_links (token, boundary_id, index_name) VALUES (?, ?, ?)",
        (token, boundary_id, index_name),
    )
    conn.commit()
    conn.close()
    return token


def resolve_share_link(token: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM share_links WHERE token = ?", (token,)).fetchone()
    conn.close()
    return dict(row) if row else None


def add_annotation(boundary_id: str, lat: float, lng: float, comment: str, author: str = None) -> dict:
    import uuid
    ann_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO annotations (id, boundary_id, lat, lng, comment, author) VALUES (?, ?, ?, ?, ?, ?)",
        (ann_id, boundary_id, lat, lng, comment, author),
    )
    conn.commit()
    conn.close()
    return {"id": ann_id, "boundary_id": boundary_id, "lat": lat, "lng": lng,
            "comment": comment, "author": author}


def list_annotations(boundary_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM annotations WHERE boundary_id = ? ORDER BY created_at DESC", (boundary_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
