"""Daemon debounce behaviour under write bursts.

A flurry of editor saves must collapse into ONE re-index per file after the
debounce window of quiet. Drives the real DebouncedIndexer against a real
indexed target (no LLM, no external daemon).

Run:
    uv run --group dev pytest benchmarks/test_debounce.py -q
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

BENCH_MODEL = "minilm"
DEBOUNCE_S = 1.0


@pytest.fixture(scope="module", autouse=True)
def _warm_model():
    from ffembed import embed

    embed.embed_query(BENCH_MODEL, "warmup")


@pytest.fixture()
def watched(tmp_path):
    """Real corpus, indexed target, instrumented DebouncedIndexer."""
    from ffembed import db as ffembed_db
    from ffembed.indexer import index_file
    from ffembed.watcher import DebouncedIndexer

    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(10):
        (root / f"note_{i}.md").write_text(f"initial {i}\n\nalpha passage\n")

    conn = ffembed_db.connect()
    target_id = ffembed_db.add_target(conn, str(root), "*.md", BENCH_MODEL)
    target_row = next(t for t in ffembed_db.list_targets(conn)
                      if t["id"] == target_id)

    # Initial full indexing through the normal path so 'files' exist.
    for p in sorted(root.glob("*.md")):
        index_file(conn, target_row, p)
    conn.commit()

    calls = {"index_file": 0, "lock": threading.Lock()}

    def process(path_str: str):
        # Mirrors daemon.run_forever: a fresh short-lived connection, because
        # DebouncedIndexer invokes this from its own worker thread.
        path = Path(path_str)
        work_conn = ffembed_db.connect()
        try:
            target = ffembed_db.get_target_for_path(work_conn, path)
            if target is None:
                return
            with calls["lock"]:
                calls["index_file"] += 1
            if not path.exists():
                return
            index_file(work_conn, target, path)
            work_conn.commit()
        finally:
            work_conn.close()

    def get_target(path: Path):
        with ffembed_db.cursor() as c:
            return ffembed_db.get_target_for_path(c, path)

    handler = DebouncedIndexer(get_target, process,
                               debounce_seconds=DEBOUNCE_S)
    handler.start()
    yield root, handler, calls
    handler.stop()
    conn.close()


def _send_burst(handler, root: Path, n_events: int, n_files: int):
    from watchdog.events import FileModifiedEvent

    for i in range(n_events):
        handler.on_modified(
            FileModifiedEvent(str(root / f"note_{i % n_files}.md")))


def test_burst_collapses_to_one_reindex_per_file(watched):
    """40 rapid events on 10 files => exactly 10 re-indexes once quiet."""
    root, handler, calls = watched
    _send_burst(handler, root, n_events=40, n_files=10)

    deadline = time.monotonic() + DEBOUNCE_S * 2 + 15
    while time.monotonic() < deadline and calls["index_file"] < 10:
        time.sleep(0.1)
    time.sleep(0.8)  # let any debounce leakage land too

    assert calls["index_file"] == 10, (
        f"debounce failed: expected exactly 10 re-indexes after a burst of "
        f"40 events on 10 files, saw {calls['index_file']}"
    )


def test_two_bursts_settle_separately(watched):
    """Two distinct bursts separated by quiet windows => 5 then 5 more."""
    root, handler, calls = watched
    for i in range(5):  # only first five are burst-touched
        (root / f"note_{i}.md").write_text(f"initial {i} v1\n")
    _send_burst(handler, root, n_events=30, n_files=5)

    deadline = time.monotonic() + DEBOUNCE_S * 2 + 12
    while time.monotonic() < deadline and calls["index_file"] < 5:
        time.sleep(0.05)
    first = calls["index_file"]

    time.sleep(DEBOUNCE_S + 0.5)  # ensure a clean quiet window
    for i in range(5):
        (root / f"note_{i}.md").write_text(f"initial {i} v2\n")
    _send_burst(handler, root, n_events=30, n_files=5)
    deadline = time.monotonic() + DEBOUNCE_S * 2 + 12
    while time.monotonic() < deadline and calls["index_file"] < first + 5:
        time.sleep(0.05)
    second_extra = calls["index_file"] - first

    assert first == 5, f"first burst produced {first} re-indexes, want 5"
    assert second_extra == 5, (
        f"second burst produced {second_extra} re-indexes, want 5"
    )
