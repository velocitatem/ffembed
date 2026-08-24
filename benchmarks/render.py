"""Render a pytest-benchmark JSON file into a Markdown table.

Usage:
    uv run --group dev pytest benchmarks/ --benchmark-only --benchmark-json=benchmarks/results.json
    uv run --group dev python benchmarks/render.py benchmarks/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _mean_us(bench: dict) -> float:
    return bench["stats"]["mean"] * 1_000_000


def render(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for bench in sorted(data["benchmarks"], key=lambda b: b["name"]):
        name = bench["name"]
        mean = _mean_us(bench)
        rounds = bench["stats"]["rounds"]
        rows.append(f"| `{name}` | {mean:,.0f} | {rounds} |")

    header = "| Benchmark | Mean (µs) | Rounds |\n|---|---|---|\n"
    return header + "\n".join(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results.json>", file=sys.stderr)
        sys.exit(1)
    print(render(Path(sys.argv[1])))
