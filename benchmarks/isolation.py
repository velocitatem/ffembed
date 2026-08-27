"""Isolate ffembed's HOME-dependent module state into a temp directory.

``ffembed.paths`` and the modules that imported constants from it at
load time (db, embed, daemon) must ALL be repointed, otherwise an index
write touches the user's real ``~/.ffembed``.
"""

from __future__ import annotations

from pathlib import Path


def isolate(home: Path) -> tuple:
    """Redirect all ffembed path constants under ``home``; return saved values."""
    from ffembed import db as fdb
    from ffembed import paths as fp

    root = home / ".ffembed"
    root.mkdir(parents=True, exist_ok=True)
    real_models = Path.home() / ".ffembed" / "models"
    link = root / "models"
    if real_models.is_dir() and not link.exists():
        try:
            link.symlink_to(real_models)
        except OSError:
            pass  # cache miss tolerated: fastembed will download if needed

    saved = {
        "paths": (fp.ROOT, fp.DB_PATH, fp.CACHE_DIR),
        "db": fdb.DB_PATH,
    }
    fp.ROOT = root
    fp.DB_PATH = root / "db.sqlite"
    fp.CACHE_DIR = root / "models"
    # Modules that bound DB_PATH/CACHE_DIR at import time must be updated too.
    fdb.DB_PATH = fp.DB_PATH
    try:
        from ffembed import embed as fembed
        fembed.CACHE_DIR = fp.CACHE_DIR
    except ImportError:
        pass
    return saved


def restore(saved: dict) -> None:
    from ffembed import db as fdb
    from ffembed import paths as fp

    fp.ROOT, fp.DB_PATH, fp.CACHE_DIR = saved["paths"]
    fdb.DB_PATH = saved["paths"][1]
