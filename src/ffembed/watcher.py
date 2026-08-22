"""Debounced filesystem watching: coalesces bursts of change events per file
so a quick sequence of writes triggers one re-embed, not N of them."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .indexer import matches


class DebouncedIndexer(FileSystemEventHandler):
    def __init__(self, get_target_for_path, process, debounce_seconds: float = 2.0):
        self.get_target_for_path = get_target_for_path
        self.process = process
        self.debounce_seconds = debounce_seconds
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _touch(self, path: str):
        target = self.get_target_for_path(Path(path))
        if target is None:
            return
        if not matches(Path(path), target["pattern"]):
            return
        with self._lock:
            self._pending[path] = time.monotonic()

    def on_created(self, event):
        if not event.is_directory:
            self._touch(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._touch(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._touch(event.dest_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._touch(event.src_path)

    def _loop(self):
        while not self._stop.is_set():
            now = time.monotonic()
            ready = []
            with self._lock:
                for path, ts in list(self._pending.items()):
                    if now - ts >= self.debounce_seconds:
                        ready.append(path)
                        del self._pending[path]
            for path in ready:
                try:
                    self.process(path)
                except Exception as exc:  # keep the daemon alive on a bad file
                    print(f"[ffembed] error processing {path}: {exc}")
            time.sleep(0.5)


def watch_forever(roots: list[str], handler: DebouncedIndexer):
    observer = Observer()
    for root in roots:
        p = Path(root)
        if p.exists():
            observer.schedule(handler, root, recursive=True)
    observer.start()
    handler.start()
    return observer
