"""Live agent benchmark: real LLM tool-use loops finding the right file.

Requires OPENAI_API_KEY (skipped otherwise). Model via FFEMPLE_BENCH_MODEL /
FFEMBED_BENCH_MODEL env var, default gpt-4o-mini. Tokens come from the API's
usage numbers — no estimation. See agent_live.py.

Run:
    uv run --group dev pytest benchmarks/test_agent_live.py -q -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .agent_live import MODEL, live_markdown, run_policy
from .corpus import TASKS


CORPUS_SIZES = [50]

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@pytest.mark.parametrize("files", CORPUS_SIZES)
def test_live_agent_token_resolution(indexed_implicit, files: int):
    """Run a real LLM agent per task with each toolbox; report real tokens."""
    from openai import OpenAI

    conn, root, _mentions = indexed_implicit(files)
    client = OpenAI()

    attempts = []
    for policy in ("list", "grep", "ffembed"):
        attempts += run_policy(client, policy, TASKS, root, conn, str(root))

    table = live_markdown(attempts)
    print(f"\n### live agents ({files} notes, model {MODEL})\n")
    print(table)

    out = Path(__file__).parent / "results_agents_live.md"
    section = (
        f"### {files} notes ({MODEL})\n\n"
        f"Real LLM tool-use loop, temperature 0, max {10} turns. Tokens are "
        f"prompt + completion from the API.\n\n{table}\n"
    )
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    marker = f"### {files} notes"
    if marker in existing:
        pre = existing.split(marker)[0]
        rest = "\n### ".join(existing.split(marker)[1].split("\n### ")[1:])
        out.write_text(pre + section + rest, encoding="utf-8")
    else:
        out.write_text(existing + section, encoding="utf-8")

    by_policy = {}
    for a in attempts:
        by_policy.setdefault(a.policy, []).append(a)
    for policy, rs in by_policy.items():
        assert any(r.found for r in rs), f"{policy} agent never found the right file"
