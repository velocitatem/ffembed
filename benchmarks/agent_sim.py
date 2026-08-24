"""Simulated agent policies for file-finding, with token accounting.

No LLM is called. Each policy is a deterministic stand-in for how an agent
behaves, and we count the tokens it would have ingested on the way to the
right file. Token counts use ~4 chars per token (documented heuristic).

Policies:
- list_all:    run `ls`, then read files one by one until the right one.
- grep_loop:   try keywords with `grep -rl`; read every hit until resolved.
               Failed keyword guesses still cost a round trip.
- ffembed:     one semantic query; inspect ranked snippets until resolved.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path


# Framing cost of one tool round trip (command text + model reasoning around
# an empty or tiny result), in tokens.
TOOL_CALL_OVERHEAD = 30

# Snippets the semantic agent inspects before giving up / opening a file.
SNIPPETS_INSPECTED = 3


def count_tokens(text: str) -> int:
    """~4 chars per token. Good enough for prose comparisons."""
    return max(1, len(text) // 4)


@dataclass
class Attempt:
    policy: str
    question: str
    found: bool
    calls: int
    tokens: int
    reads: int = 0  # full-file reads (the expensive part for grep agents)


def _resolve_list_all(root: Path, paths: list[Path], good_paths: set[str]) -> Attempt:
    calls, tokens, reads = 0, TOOL_CALL_OVERHEAD, 0
    listing = "\n".join(p.name for p in sorted(paths))
    calls += 1
    tokens += count_tokens(listing)
    for p in sorted(paths):
        calls += 1
        reads += 1
        tokens += count_tokens(p.read_text(encoding="utf-8"))
        if str(p) in good_paths:
            return Attempt("list_all", "", True, calls, tokens, reads)
    return Attempt("list_all", "", False, calls, tokens, reads)


def _resolve_grep_loop(root: Path, keywords: list[str], good_paths: set[str]) -> Attempt:
    calls, tokens, reads = 0, 0, 0
    for kw in keywords:
        proc = subprocess.run(
            ["grep", "-rl", "--include=*.md", kw, str(root)],
            capture_output=True, text=True, check=False,
        )
        calls += 1
        tokens += TOOL_CALL_OVERHEAD + count_tokens(proc.stdout)
        if not proc.stdout.strip():
            continue  # guessed wrong; try the next synonym
        for line in sorted(proc.stdout.splitlines()):
            p = Path(line)
            if not p.is_file():
                continue
            calls += 1
            reads += 1
            tokens += count_tokens(p.read_text(encoding="utf-8"))
            if str(p) in good_paths:
                return Attempt("grep_loop", "", True, calls, tokens, reads)
    return Attempt("grep_loop", "", False, calls, tokens, reads)


def _resolve_ffembed(conn, target_path: str, question: str, good_paths: set[str]) -> Attempt:
    from ffembed import search

    results = search.search(conn, question, target_path=target_path, k=10)
    calls, tokens = 1, TOOL_CALL_OVERHEAD
    # The agent inspects top snippets (cheap), then opens at most one file.
    for score, row in results[:SNIPPETS_INSPECTED]:
        tokens += count_tokens(row["text"] or "")
        if row["file_path"] in good_paths:
            return Attempt("ffembed", question, True, calls, tokens, 0)
    if results:
        # Best-guess full read before giving up.
        best = Path(results[0][1]["file_path"])
        tokens += count_tokens(best.read_text(encoding="utf-8"))
        return Attempt(
            "ffembed", question, best and str(best) in good_paths,
            calls + 1, tokens, 1,
        )
    return Attempt("ffembed", question, False, calls, tokens, 0)


@dataclass
class Report:
    rows: list[Attempt] = field(default_factory=list)

    def add(self, attempt: Attempt) -> None:
        self.rows.append(attempt)

    def summary(self) -> dict[str, dict]:
        out: dict[str, list[Attempt]] = {}
        for r in self.rows:
            out.setdefault(r.policy, []).append(r)
        summary = {}
        for policy, rs in out.items():
            toks = sorted(r.tokens for r in rs)
            n = len(toks)
            summary[policy] = {
                "tasks": n,
                "success_pct": 100.0 * sum(r.found for r in rs) / n,
                "median_tokens": toks[n // 2],
                "p90_tokens": toks[min(n - 1, int(n * 0.9))],
                "median_calls": sorted(r.calls for r in rs)[n // 2],
                "median_reads": sorted(r.reads for r in rs)[n // 2],
            }
        return summary

    def markdown(self) -> str:
        header = (
            "| policy | success | median tokens | p90 tokens | median tool calls | median file reads |\n"
            "|---|---|---|---|---|---|\n"
        )
        lines = []
        for policy, s in self.summary().items():
            lines.append(
                f"| {policy} | {s['success_pct']:.0f}% | {s['median_tokens']:,} "
                f"| {s['p90_tokens']:,} | {s['median_calls']} | {s['median_reads']} |"
            )
        return header + "\n".join(lines) + "\n"


def run_all_policies(root: Path, conn, target_path: str, tasks: list[dict],
                     dominant: dict[str, list[str]]) -> list[Attempt]:
    """Run every policy over every task and return raw attempts."""
    paths = sorted(root.glob("*.md"))
    attempts: list[Attempt] = []
    for task in tasks:
        good = set(dominant.get(task["topic"], []))
        attempts.append(replace(_resolve_list_all(root, paths, good), question=task["question"]))
        attempts.append(replace(_resolve_grep_loop(root, task["keywords"], good), question=task["question"]))
        attempts.append(_resolve_ffembed(conn, target_path, task["question"], good))
    return attempts
