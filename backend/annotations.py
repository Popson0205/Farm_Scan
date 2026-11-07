"""
Share & Collaborate: read-only share links for a farm (+ result), and annotations (point comments) on a farm —
Solvi's "Annotations and comments" + "Sharing via public web links". Farms are checked to belong to the person's
organization before these are called.
"""
import secrets
import uuid

import db


def create_share_link(boundary_id: str, index_name: str = None, created_by: str = None) -> str:
    token = secrets.token_urlsafe(12)
    with db.tx() as conn:
        conn.execute("INSERT INTO share_links (token, boundary_id, index_name, created_by) VALUES (%s, %s, %s, %s)",
                     (token, boundary_id, index_name, created_by))
    return token


def resolve_share_link(token: str):
    with db.tx() as conn:
        row = conn.execute("SELECT token, boundary_id::text AS boundary_id, index_name, created_at FROM share_links "
                           "WHERE token = %s", (token,)).fetchone()
    return row


def add_annotation(boundary_id: str, lat: float, lng: float, comment: str, author: str = None, conn=None) -> dict:
    ann_id = str(uuid.uuid4())
    sql = "INSERT INTO annotations (id, boundary_id, lat, lng, comment, author) VALUES (%s, %s, %s, %s, %s, %s)"
    if conn is not None:
        conn.execute(sql, (ann_id, boundary_id, lat, lng, comment, author))
    else:
        with db.tx() as c:
            c.execute(sql, (ann_id, boundary_id, lat, lng, comment, author))
    return {"id": ann_id, "boundary_id": boundary_id, "lat": lat, "lng": lng, "comment": comment, "author": author}


def list_annotations(boundary_id: str) -> list[dict]:
    with db.tx() as conn:
        rows = conn.execute("SELECT id::text AS id, boundary_id::text AS boundary_id, lat, lng, comment, author, "
                            "created_at::text AS created_at FROM annotations WHERE boundary_id = %s "
                            "ORDER BY created_at DESC", (boundary_id,)).fetchall()
    return rows


AUTO_SCOUT_AUTHOR = "FarmScan scout"


def replace_auto_scouting_notes(boundary_id: str, notes: list[dict]) -> list[dict]:
    """Replaces this farm's machine-generated scouting notes with a fresh set (so re-running problem-area detection
    doesn't pile up duplicates). Notes written by people are never touched. notes: [{lat, lng, comment}]."""
    with db.tx() as conn:
        conn.execute("DELETE FROM annotations WHERE boundary_id = %s AND author = %s", (boundary_id, AUTO_SCOUT_AUTHOR))
        return [add_annotation(boundary_id, n["lat"], n["lng"], n["comment"], AUTO_SCOUT_AUTHOR, conn=conn) for n in notes]
