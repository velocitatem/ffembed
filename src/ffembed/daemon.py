"""Background daemon lifecycle: start/stop/status + the actual watch loop."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import db
from .indexer import index_file, index_target, remove_missing_files
from .paths import LOG_PATH, PID_PATH, ensure_root
from .watcher import DebouncedIndexer, watch_forever


def _read_pid() -> int | None:
    if not PID_PATH.exists():
        return None
    try:
        return int(PID_PATH.read_text().strip())
    except ValueError:
        return None


def is_running() -> int | None:
    pid = _read_pid()
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        PID_PATH.unlink(missing_ok=True)
        return None
    except PermissionError:
        return pid
    return pid


def start(debounce_seconds: float = 2.0) -> int:
    ensure_root()
    existing = is_running()
    if existing:
        return existing
    log = open(LOG_PATH, "a")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "ffembed.daemon", "--debounce", str(debounce_seconds)],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    PID_PATH.write_text(str(proc.pid))
    return proc.pid


def stop() -> bool:
    pid = is_running()
    if not pid:
        return False
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if is_running() is None:
            return True
        time.sleep(0.25)
    return is_running() is None


def run_foreground(debounce_seconds: float = 2.0):
    """The actual watch loop. Blocks forever, reacting to file events."""
    ensure_root()
    PID_PATH.write_text(str(os.getpid()))

    def get_target_for_path(path: Path):
        with db.cursor() as conn:
            return db.get_target_for_path(conn, path)

    def process(path_str: str):
        path = Path(path_str)
        with db.cursor() as conn:
            target = db.get_target_for_path(conn, path)
            if target is None:
                return
            if not path.exists():
                db.remove_file(conn, path_str)
                print(f"[ffembed] removed {path}")
                return
            if index_file(conn, target, path):
                print(f"[ffembed] indexed {path}")

    with db.cursor() as conn:
        targets = db.list_targets(conn)
        for t in targets:
            index_target(conn, t)
            remove_missing_files(conn, t)

    roots = [t["path"] for t in targets]
    handler = DebouncedIndexer(get_target_for_path, process, debounce_seconds=debounce_seconds)
    observer = watch_forever(roots, handler)

    stop_flag = {"stop": False}

    def _sigterm(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    print(f"[ffembed] watching {len(roots)} target(s), debounce={debounce_seconds}s")
    try:
        while not stop_flag["stop"]:
            time.sleep(0.5)
    finally:
        observer.stop()
        observer.join(timeout=5)
        handler.stop()
        PID_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debounce", type=float, default=2.0)
    args = parser.parse_args()
    run_foreground(args.debounce)
