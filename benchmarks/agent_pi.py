"""pi coding-agent harness: does ffembed reduce what a real agent burns?

For each task we spawn a fresh `pi -p` session inside a corpus of
neutral-named, implicitly-written notes, and ask it to find the note that
answers the question. Two arms, same agent, same model:

- shell:   pi's default tools (read/bash/edit/write)
- ffembed: identical, plus a `semantic_search` tool (ffembed_extension.ts)

Tokens and cost come from pi's own usage accounting in its JSON event stream;
success is graded deterministically (the finished-on file must contain the
answer passage). Every episode is appended to an episodes.jsonl for audit.

Usage:
    uv run --group dev python -m benchmarks.agent_pi --size 100 --per-style 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

from .agent_live import MODEL  # default model name, overridable via flag/env
from .corpus import build_tasks, generate_implicit_corpus

PI = shutil.which("pi")
FFEMBED = shutil.which("ffembed")
EXTENSION = Path(__file__).parent.parent / "extensions" / "ffembed.ts"

ARMS = {
    "shell": [],
    "ffembed": ["--extension", str(EXTENSION)],
}

FILE_RE = re.compile(r"^\s*FILE:\s*(.+?)\s*$", re.MULTILINE)


@dataclass
class Episode:
    arm: str
    style: str
    topic: str
    question: str
    found: bool
    chosen: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_s: float
    tool_calls: dict[str, int]
    error: str = ""


def _setup_workspace(base: Path, size: int, fillers: int) -> tuple[Path, Path]:
    """Create corpus + isolated HOME with ffembed indexed; return (corpus, home)."""
    corpus = base / "notes"
    home = base / "home"
    (home / ".ffembed").mkdir(parents=True)
    # Reuse the real model cache so no ONNX weights are re-downloaded.
    real_models = Path.home() / ".ffembed" / "models"
    if real_models.is_dir():
        (home / ".ffembed" / "models").symlink_to(real_models)
    env = {**os.environ, "HOME": str(home)}

    generate_implicit_corpus(corpus, size, fillers_per_note=fillers)
    subprocess.run(
        [FFEMBED, "watch", str(corpus), "--filter", "*.md"],
        check=True, capture_output=True, text=True, env=env,
    )
    return corpus, home


def run_episode(arm: str, task: dict, corpus: Path, home: Path,
                model: str, timeout_s: int = 300) -> Episode:
    prompt = (
        f'Find the note in this directory that answers: "{task["question"]}"\n'
        "Verify by reading it. Then reply with exactly one line:\n"
        "FILE: <filename>"
    )
    cmd = [
        PI, "-p", "--mode", "json", "--no-session",
        "--provider", "openai", "--model", model,
        *ARMS[arm], prompt,
    ]
    env = {**os.environ, "HOME": str(home)}
    start = time.time()
    proc = subprocess.run(
        cmd, cwd=str(corpus), env=env, capture_output=True, text=True,
        timeout=timeout_s,
    )
    duration = time.time() - start

    ep = Episode(arm=arm, style=task["style"], topic=task["topic"],
                 question=task["question"], found=False, chosen="",
                 input_tokens=0, output_tokens=0, cost_usd=0.0,
                 duration_s=duration, tool_calls={})
    if proc.returncode != 0:
        ep.error = f"exit {proc.returncode}: {proc.stderr[-300:]}"
        return ep

    final_text = ""
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "message_end":
            msg = ev.get("message", {})
            if msg.get("role") != "assistant":
                continue
            usage = msg.get("usage") or {}
            ep.input_tokens += usage.get("input", 0)
            ep.output_tokens += usage.get("output", 0)
            ep.cost_usd += (usage.get("cost") or {}).get("total", 0.0)
            texts = [c.get("text", "") for c in msg.get("content", [])
                     if isinstance(c, dict) and c.get("type") == "text"]
            if texts:
                final_text = "\n".join(texts)
        elif ev.get("type") == "tool_execution_start":
            name = ev.get("toolName", "?")
            ep.tool_calls[name] = ep.tool_calls.get(name, 0) + 1

    m = FILE_RE.search(final_text)
    if not m:
        ep.error = "no FILE: line in final message"
        return ep
    chosen = corpus / Path(m.group(1).strip().lstrip("`")).name
    ep.chosen = chosen.name
    passage = _passage_for(corpus, home, task["topic"])
    ep.found = chosen.is_file() and passage in chosen.read_text(encoding="utf-8")
    return ep


def _passage_for(corpus: Path, home: Path, topic: str) -> str:
    # Ground truth comes from the deterministic generator (seed 7).
    from .corpus import IMPLICIT_PASSAGES

    return IMPLICIT_PASSAGES[topic][:80]


def run_harness(size: int, per_style: int | None, arms: list[str],
                model: str, workers: int, fillers: int,
                out_dir: Path) -> list[Episode]:
    tasks = build_tasks(limit_per_style=per_style)
    base = Path(tempfile.mkdtemp(prefix="ffembed-pi-"))
    corpus, home = _setup_workspace(base, size, fillers)
    print(f"workspace: {base}\ncorpus: {size} notes, {len(tasks)} tasks, "
          f"arms: {arms}, model: {model}\n")

    episodes: list[Episode] = []
    jobs = [(arm, t) for arm in arms for t in tasks]
    jsonl = out_dir / "episodes.jsonl"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_episode, arm, t, corpus, home, model): (arm, t)
                   for arm, t in jobs}
        done = 0
        for fut in as_completed(futures):
            try:
                ep = fut.result()
            except Exception as e:  # timeout etc.
                arm, t = futures[fut]
                ep = Episode(arm=arm, style=t["style"], topic=t["topic"],
                             question=t["question"], found=False, chosen="",
                             input_tokens=0, output_tokens=0, cost_usd=0.0,
                             duration_s=0.0, tool_calls={}, error=str(e))
            episodes.append(ep)
            done += 1
            with jsonl.open("a") as f:
                f.write(json.dumps(asdict(ep)) + "\n")
            mark = "✓" if ep.found else "✗"
            print(f"[{done}/{len(jobs)}] {mark} {ep.arm:8} {ep.style:7} "
                  f"{ep.topic[:20]:20} in={ep.input_tokens:>6} "
                  f"out={ep.output_tokens:>5} ${ep.cost_usd:.4f} "
                  f"{ep.duration_s:5.1f}s {('ERR ' + ep.error[:60]) if ep.error else ''}")

    shutil.rmtree(base, ignore_errors=True)
    return episodes


def summarize(episodes: list[Episode]) -> str:
    def stats(rs: list[Episode]) -> dict:
        totals = sorted(r.input_tokens + r.output_tokens for r in rs)
        n = len(totals)
        return {
            "n": n,
            "success": 100.0 * sum(r.found for r in rs) / n if n else 0.0,
            "p10": totals[max(0, int(n * 0.1) - 1)] if n else 0,
            "median": totals[n // 2] if n else 0,
            "p90": totals[min(n - 1, int(n * 0.9))] if n else 0,
            "cost": sum(r.cost_usd for r in rs),
        }

    lines = ["## By arm\n",
             "| arm | n | success | finished | p10 tok | median tok | p90 tok | cost |\n"
             "|---|---|---|---|---|---|---|---|\n"]
    for arm in ARMS:
        rs = [e for e in episodes if e.arm == arm]
        if not rs:
            continue
        s = stats(rs)
        finished = 100.0 * sum(1 for r in rs if not r.error) / s["n"]
        lines.append(f"| {arm} | {s['n']} | {s['success']:.0f}% | "
                     f"{finished:.0f}% | {s['p10']:,} | {s['median']:,} | "
                     f"{s['p90']:,} | ${s['cost']:.3f} |\n")

    lines.append("\n## Median tokens by question style\n")
    styles = sorted({e.style for e in episodes})
    lines.append("| arm | " + " | ".join(styles) + " |\n"
                 "|---|" + "---|" * len(styles) + "\n")
    for arm in ARMS:
        cells = []
        for st in styles:
            rs = [e for e in episodes if e.arm == arm and e.style == st]
            if rs:
                med = sorted(r.input_tokens + r.output_tokens for r in rs)[len(rs) // 2]
                ok = sum(r.found for r in rs)
                cells.append(f"{med:,} ({ok}/{len(rs)})")
            else:
                cells.append("-")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--fillers", type=int, default=4,
                    help="off-topic passages per note (drives note length)")
    ap.add_argument("--per-style", type=int, default=None,
                    help="cap topics per question style (default: all 8)")
    ap.add_argument("--arms", default="shell,ffembed")
    ap.add_argument("--model", default=os.environ.get("FFEMBED_BENCH_MODEL", MODEL))
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    assert PI, "pi CLI not found on PATH"
    assert FFEMBED, "ffembed CLI not found on PATH"

    out_dir = Path(__file__).parent
    episodes = run_harness(args.size, args.per_style, args.arms.split(","),
                           args.model, args.workers, args.fillers, out_dir)
    report = summarize(episodes)
    print("\n" + report)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    md = out_dir / "results_pi.md"
    md.write_text(
        f"# pi harness — {args.size} notes, model {args.model}\n\n"
        f"Run {stamp}. `finished` = emitted a FILE: verdict at all; "
        f"`success` = verdict was a file containing the answer passage.\n\n{report}",
        encoding="utf-8",
    )
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
