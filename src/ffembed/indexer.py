"""Turns a file into rows in the database: read -> chunk -> embed -> store."""

from __future__ import annotations

import fnmatch
import hashlib
import sqlite3
from pathlib import Path

from . import db
from .chunk import chunk_text
from .embed import embed_texts
from .vision import DEFAULT_VISION_MODEL, embed_image, is_image_path

TEXT_READ_ERRORS = (UnicodeDecodeError, OSError)


def matches(path: Path, pattern: str) -> bool:
    return fnmatch.fnmatch(path.name, pattern)


def iter_target_files(root: Path, pattern: str):
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and matches(p, pattern):
            yield p


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _vision_model_for(target_row: sqlite3.Row) -> str:
    vision = target_row["vision_model"]
    return vision if vision else DEFAULT_VISION_MODEL


def index_file(conn: sqlite3.Connection, target_row: sqlite3.Row, path: Path, *, force: bool = False) -> bool:
    """Index a single file against its target. Returns True if (re)indexed."""
    try:
        data = path.read_bytes()
    except OSError:
        return False
    h = file_hash(data)
    existing = db.get_file(conn, str(path))
    if not force and existing is not None and existing["hash"] == h:
        return False

    file_id = db.upsert_file(conn, target_row["id"], str(path), path.stat().st_mtime, h)

    if is_image_path(path):
        vec = embed_image(_vision_model_for(target_row), path)
        db.insert_chunk(conn, file_id, 0, "", vec, kind="image")
        return True

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False

    pieces = chunk_text(text)
    if pieces:
        vectors = embed_texts(target_row["model"], pieces)
        for i, (piece, vec) in enumerate(zip(pieces, vectors)):
            db.insert_chunk(conn, file_id, i, piece, vec)
    return True


def index_target(conn: sqlite3.Connection, target_row: sqlite3.Row, *, force: bool = False, on_file=None):
    root = Path(target_row["path"])
    count = 0
    for path in iter_target_files(root, target_row["pattern"]):
        if index_file(conn, target_row, path, force=force):
            count += 1
            if on_file:
                on_file(path)
    return count


def remove_missing_files(conn: sqlite3.Connection, target_row: sqlite3.Row):
    root = Path(target_row["path"])
    rows = conn.execute("SELECT path FROM files WHERE target_id = ?", (target_row["id"],)).fetchall()
    removed = 0
    for row in rows:
        p = Path(row["path"])
        if not p.exists() or not matches(p, target_row["pattern"]):
            db.remove_file(conn, row["path"])
            removed += 1
    return removed
