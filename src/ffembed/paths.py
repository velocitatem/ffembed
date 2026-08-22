"""Filesystem layout for ffembed's home directory (~/.ffembed)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path.home() / ".ffembed"
DB_PATH = ROOT / "db.sqlite"
PID_PATH = ROOT / "daemon.pid"
LOG_PATH = ROOT / "daemon.log"
CACHE_DIR = ROOT / "models"


def ensure_root() -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ROOT
