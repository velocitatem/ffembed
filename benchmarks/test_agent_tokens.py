"""Token-cost benchmark: how many tokens does an agent burn finding the
right file with each search strategy?

This is not a latency benchmark — it simulates agent policies (no LLM calls)
and counts the tokens each policy would ingest: tool outputs plus any full
file reads needed to verify a hit. See agent_sim.py for the policies.

Run:
    uv run --group dev pytest benchmarks/test_agent_tokens.py -q -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .agent_sim import Report, run_all_policies
from .corpus import TASKS


CORPUS_SIZES = [50, 200]


@pytest.mark.parametrize("files", CORPUS_SIZES)
def test_agent_token_resolution(indexed_implicit, files: int):
    """Simulate three file-finding policies and report tokens-to-resolution."""
    conn, root, mentions = indexed_implicit(files)

    report = Report()
    for attempt in run_all_policies(root, conn, str(root), TASKS, mentions):
        report.add(attempt)

    table = report.markdown()
    print(f"\n### tokens to find the right file ({files} notes, implicit prose)\n")
    print(table)

    out = Path(__file__).parent / "results_agents.md"
    section = (
        f"### {files} notes\n\n"
        f"{len(TASKS)} natural-language tasks over neutral-named notes whose "
        f"prose never names its topic.\n\n{table}\n"
    )
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    marker = f"### {files} notes"
    if marker in existing:
        pre = existing.split(marker)[0]
        rest = "\n### ".join(existing.split(marker)[1].split("\n### ")[1:])
        out.write_text(pre + section + rest, encoding="utf-8")
    else:
        out.write_text(existing + section, encoding="utf-8")

    summary = report.summary()
    assert summary["ffembed"]["success_pct"] >= 75, "semantic policy should resolve most tasks"
