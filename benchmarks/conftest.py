"""Shared fixtures for ffembed benchmarks.

All benchmarks use an isolated temporary home directory so they do not touch
``~/.ffembed``.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

from ffembed import db as ffembed_db
from ffembed import indexer, paths as ffembed_paths

from .corpus import generate_corpus

BENCH_MODEL = "minilm"


@pytest.fixture(scope="session", autouse=True)
def _isolated_ffembed_home():
    """Redirect ffembed's ROOT/DB_PATH/CACHE_DIR to a temp dir for the session."""
    tmp = Path(tempfile.mkdtemp(prefix="ffembed-bench-"))
    original_root = ffembed_paths.ROOT
    original_db = ffembed_paths.DB_PATH
    original_cache = ffembed_paths.CACHE_DIR
    ffembed_paths.ROOT = tmp
    ffembed_paths.DB_PATH = tmp / "db.sqlite"
    ffembed_paths.CACHE_DIR = tmp / "models"
    ffembed_paths.ensure_root()
    yield tmp
    # Restore originals before cleanup so later code is not confused.
    ffembed_paths.ROOT = original_root
    ffembed_paths.DB_PATH = original_db
    ffembed_paths.CACHE_DIR = original_cache
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _warmup_model():
    """Load the embedding model once so measured times exclude model load."""
    from ffembed import embed

    embed.embed_query(BENCH_MODEL, "warmup")


@pytest.fixture()
def corpus(tmp_path: Path):
    """Return a temporary corpus directory pre-filled with files."""
    return tmp_path / "corpus"


@pytest.fixture()
def make_corpus(corpus: Path):
    """Factory fixture to create a corpus of N files."""
    def _make(files: int, paragraphs_per_file: int = 5):
        return generate_corpus(corpus, files, paragraphs_per_file)
    return _make


@pytest.fixture()
def conn(_isolated_ffembed_home: Path):
    """Yield a fresh ffembed sqlite connection backed by the isolated home."""
    connection = ffembed_db.connect()
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture()
def index_target(conn: sqlite3.Connection):
    """Factory fixture to index a directory and return target metadata."""
    def _index(path: Path, model: str = "bge-small", pattern: str = "*.md"):
        target_id = ffembed_db.add_target(conn, str(path), pattern, model)
        target_row = conn.execute("SELECT * FROM targets WHERE id = ?", (target_id,)).fetchone()
        indexer.index_target(conn, target_row)
        conn.commit()
        return target_row
    return _index


@pytest.fixture()
def indexed_implicit(_isolated_ffembed_home):
    """Create + index an implicit corpus of N notes; yield (conn, root, mentions)."""
    def _make(files: int):
        from .corpus import generate_implicit_corpus

        root = Path(_isolated_ffembed_home) / f"implicit_{files}"
        _, mentions = generate_implicit_corpus(root, files)
        connection = ffembed_db.connect()
        target_id = ffembed_db.add_target(connection, str(root), "*.md", BENCH_MODEL)
        target_row = connection.execute("SELECT * FROM targets WHERE id = ?", (target_id,)).fetchone()
        indexer.index_target(connection, target_row)
        connection.commit()
        return connection, root, mentions
    return _make
