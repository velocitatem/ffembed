"""Result-file infrastructure shared by all suites.

Every suite writes ``benchmarks/results/<suite>/<stamp>.json``::

    {
      "schema": 1,
      "suite": "...",
      "timestamp": "...",
      "config": {...},        # exact invocation knobs
      "env": {...},           # machine + package manifest
      "metrics": {...},       # aggregates, including CIs where applicable
      "records": [...]        # raw per-item measurements (episodes, queries...)
    }

Markdown summaries are rendered FROM these files so published numbers can
never drift from the raw data.
"""

from __future__ import annotations

import json
import multiprocessing
import platform
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def env_manifest() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": multiprocessing.cpu_count(),
        "ffembed_commit": git_commit(),
    }
    for pkg in ("fastembed", "bm25s", "pytrec_eval"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return info


def write_results(suite: str, config: dict, metrics: dict,
                  records: list[dict]) -> Path:
    out_dir = RESULTS_DIR / suite
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "schema": 1,
        "suite": suite,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": config,
        "env": env_manifest(),
        "metrics": metrics,
        "records": records,
    }
    path = out_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
