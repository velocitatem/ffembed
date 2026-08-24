"""Benchmark ffembed semantic search against plain grep filename search.

Raw query latency only — for the agent token-cost numbers (the interesting
ones) see test_agent_tokens.py / test_agent_live.py. Intentionally small so
it finishes in seconds.

Run with:
    uv run --group dev pytest benchmarks/test_latency.py --benchmark-only
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from ffembed import db as ffembed_db
from ffembed import embed, search


# Tiny, fast model for benchmarks.  Same API as the default bge-small.
BENCH_MODEL = "minilm"

QUERIES = [
    ("debounce", "filesystem watcher"),
    ("vector database", "sqlite index"),
    ("asyncio", "background daemon"),
]


@pytest.fixture(scope="session", autouse=True)
def _warmup_model():
    """Load the embedding model once so benchmark times measure search, not load."""
    embed.embed_query(BENCH_MODEL, "warmup")


@pytest.mark.benchmark(min_rounds=5, max_time=3.0)
@pytest.mark.parametrize("files", [10, 50, 100])
@pytest.mark.parametrize("query,keyword", QUERIES)
def test_ffembed_search(benchmark, index_target, make_corpus, conn: sqlite3.Connection, files: int, query: str, keyword: str):
    """Measure ffembed semantic search latency on a pre-indexed corpus."""
    corpus = make_corpus(files)
    target = index_target(corpus[0].parent, model=BENCH_MODEL)

    def _search():
        return search.search(conn, query, target_path=target["path"], k=5, model=BENCH_MODEL)

    result = benchmark(_search)
    assert len(result) <= 5


@pytest.mark.benchmark(min_rounds=10, max_time=3.0)
@pytest.mark.parametrize("files", [10, 50, 100])
@pytest.mark.parametrize("query,keyword", QUERIES)
def test_grep_filename_search(benchmark, make_corpus, files: int, query: str, keyword: str):
    """Measure grep -rl filename search latency on the same corpus."""
    corpus = make_corpus(files)
    root = str(corpus[0].parent)

    def _grep():
        # Use the injected keyword as a proxy for a grep query.
        return subprocess.run(
            ["grep", "-rl", "--include=*.md", keyword, root],
            capture_output=True,
            text=True,
            check=False,
        )

    result = benchmark(_grep)
    assert result.returncode in (0, 1)


@pytest.mark.benchmark(min_rounds=3, max_time=5.0)
@pytest.mark.parametrize("files", [10, 50])
def test_ffembed_cold_first_search(benchmark, make_corpus, conn: sqlite3.Connection, files: int):
    """Measure end-to-end time: index a corpus then run one semantic query."""
    from ffembed import indexer

    corpus = make_corpus(files)
    root = corpus[0].parent

    def _cold_search():
        existing = conn.execute("SELECT id FROM targets WHERE path = ?", (str(root),)).fetchone()
        if existing is None:
            target_id = ffembed_db.add_target(conn, str(root), "*.md", BENCH_MODEL)
        else:
            target_id = existing["id"]
        target_row = conn.execute("SELECT * FROM targets WHERE id = ?", (target_id,)).fetchone()
        indexer.index_target(conn, target_row)
        conn.commit()
        return search.search(conn, "debounce", target_path=str(root), k=5, model=BENCH_MODEL)

    result = benchmark(_cold_search)
    assert len(result) <= 5


@pytest.mark.benchmark(min_rounds=10, max_time=3.0)
@pytest.mark.parametrize("files", [10, 50])
def test_grep_cold_first_search(benchmark, make_corpus, files: int):
    """Measure cold grep time: no index, just one search."""
    corpus = make_corpus(files)
    root = str(corpus[0].parent)

    def _cold_grep():
        return subprocess.run(
            ["grep", "-rl", "--include=*.md", "debounce", root],
            capture_output=True,
            text=True,
            check=False,
        )

    result = benchmark(_cold_grep)
    assert result.returncode in (0, 1)
