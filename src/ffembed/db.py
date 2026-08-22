"""SQLite storage: targets, files, chunks (with embeddings as float32 blobs)."""

from __future__ import annotations

import sqlite3
import struct
import time
from contextlib import contextmanager
from pathlib import Path

from .paths import DB_PATH, ensure_root

SCHEMA = """
CREATE TABLE IF NOT EXISTS targets (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    pattern TEXT NOT NULL DEFAULT '*',
    model TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    target_id INTEGER NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
    path TEXT NOT NULL UNIQUE,
    mtime REAL NOT NULL,
    hash TEXT NOT NULL,
    indexed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_target ON files(target_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
"""


def connect() -> sqlite3.Connection:
    ensure_root()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def cursor():
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def pack_vector(values) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


# --- targets -----------------------------------------------------------

def add_target(conn: sqlite3.Connection, path: str, pattern: str, model: str) -> int:
    conn.execute(
        "INSERT INTO targets (path, pattern, model, created_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(path) DO UPDATE SET pattern=excluded.pattern, model=excluded.model",
        (path, pattern, model, time.time()),
    )
    return conn.execute("SELECT id FROM targets WHERE path = ?", (path,)).fetchone()["id"]


def remove_target(conn: sqlite3.Connection, path: str) -> bool:
    cur = conn.execute("DELETE FROM targets WHERE path = ?", (path,))
    return cur.rowcount > 0


def list_targets(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM targets ORDER BY path").fetchall()


def get_target_for_path(conn: sqlite3.Connection, file_path: Path) -> sqlite3.Row | None:
    """Find the most specific watched target that contains file_path."""
    best = None
    for row in list_targets(conn):
        root = Path(row["path"])
        try:
            file_path.relative_to(root)
        except ValueError:
            continue
        if best is None or len(str(root)) > len(str(best["path"])):
            best = row
    return best


# --- files / chunks ------------------------------------------------------

def upsert_file(conn: sqlite3.Connection, target_id: int, path: str, mtime: float, file_hash: str) -> int:
    conn.execute(
        "INSERT INTO files (target_id, path, mtime, hash, indexed_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, hash=excluded.hash, indexed_at=excluded.indexed_at",
        (target_id, path, mtime, file_hash, time.time()),
    )
    file_id = conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()["id"]
    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    return file_id


def get_file(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()


def remove_file(conn: sqlite3.Connection, path: str) -> None:
    conn.execute("DELETE FROM files WHERE path = ?", (path,))


def insert_chunk(conn: sqlite3.Connection, file_id: int, index: int, text: str, embedding) -> None:
    conn.execute(
        "INSERT INTO chunks (file_id, chunk_index, text, dim, embedding) VALUES (?, ?, ?, ?, ?)",
        (file_id, index, text, len(embedding), pack_vector(embedding)),
    )


def stats(conn: sqlite3.Connection) -> dict:
    return {
        "targets": conn.execute("SELECT COUNT(*) c FROM targets").fetchone()["c"],
        "files": conn.execute("SELECT COUNT(*) c FROM files").fetchone()["c"],
        "chunks": conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"],
    }


def all_chunks_for_search(conn: sqlite3.Connection, target_path: str | None = None):
    query = """
        SELECT chunks.id, chunks.text, chunks.embedding, chunks.dim,
               files.path AS file_path, targets.model AS model, targets.path AS target_path
        FROM chunks
        JOIN files ON files.id = chunks.file_id
        JOIN targets ON targets.id = files.target_id
    """
    params = ()
    if target_path:
        query += " WHERE targets.path = ?"
        params = (target_path,)
    return conn.execute(query, params).fetchall()
