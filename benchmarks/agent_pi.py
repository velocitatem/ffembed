"""pi coding-agent harness: does ffembed reduce what a real agent burns?

For each task we spawn a fresh `pi -p` session inside a corpus of
neutral-named, implicitly-written notes, and ask it to find the note that
answers the question. Two arms, same agent, same model:

- shell:   pi's default tools (read/bash/edit/write)
- ffembed: identical, plus a `semantic_search` tool registered by
           ../extensions/ffembed.ts

Methodology:
- PAIRED DESIGN: tasks are queued alternating arms so every task runs in
  both arms before other work starts — differences reflect the tool, not
  sampling drift between runs.
- Deterministic grading: the finished-on file must contain the answer passage.
- Success gets Wilson 95% CIs; token deltas across paired tasks get bootstrap
  CIs and a two-sided paired permutation test.
- --resume skips episode pairs already recorded for this exact config
  (episodes.jsonl holds one JSON record per past run), so re-running after a
  crash only pays for missing episodes.

Usage:
    uv run --group dev python -m benchmarks.agent_pi --size 100 --per-style 4
    uv run --group dev python -m benchmarks.agent_pi --bank v2   # 43-topic bank
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

from .corpus import build_tasks, generate_implicit_corpus
from .question_bank_v2 import BANK as BANK_V2, build_tasks_v2
from .results import write_results
from .stats import bootstrap_ci, mean, paired_permutation_test, wilson_interval

PI = shutil.which("pi")
FFEMBED = shutil.which("ffembed")
EXTENSION = Path(__file__).parent.parent / "extensions" / "ffembed.ts"

ARMS = ("shell", "ffembed")
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


# --- workspace / task construction -------------------------------------------

def _setup_workspace(base: Path, size: int, fillers: int,
                     bank: str) -> tuple[Path, Path]:
    """Create corpus + isolated HOME with ffembed indexed; return (corpus, home)."""
    corpus = base / "notes"
    home = base / "home"
    (home / ".ffembed").mkdir(parents=True)
    real_models = Path.home() / ".ffembed" / "models"
    if real_models.is_dir():  # reuse cache so no ONNX weights are downloaded
        (home / ".ffembed" / "models").symlink_to(real_models)
    env = {**os.environ, "HOME": str(home)}

    if bank == "v2":
        from .corpus import generate_implicit_corpus_v2
        generate_implicit_corpus_v2(corpus, size, fillers_per_note=fillers)
    else:
        generate_implicit_corpus(corpus, size, fillers_per_note=fillers)
    subprocess.run(
        [FFEMBED, "watch", str(corpus), "--filter", "*.md"],
        check=True, capture_output=True, text=True, env=env,
    )
    return corpus, home


def _build_tasks(bank: str, per_style: int | None) -> list[dict]:
    """Tasks carry their own ground-truth passage so grading stays
    self-contained no matter which bank generated the corpus."""
    if bank == "v2":
        tasks = build_tasks_v2(limit_per_style=per_style)
        for t in tasks:
            t["passage"] = BANK_V2[t["topic"]]["passage"]
    else:
        from .corpus import IMPLICIT_PASSAGES
        tasks = build_tasks(limit_per_style=per_style)
        for t in tasks:
            t["passage"] = IMPLICIT_PASSAGES[t["topic"]]
    return tasks


# --- single episode ------------------------------------------------------------

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
        *(["--extension", str(EXTENSION)] if arm == "ffembed" else []),
        prompt,
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
    ep.found = chosen.is_file() and task["passage"][:80] in \
        chosen.read_text(encoding="utf-8")
    return ep


def _episode_key(arm: str, task: dict) -> str:
    # One question per (style, topic) in both banks, so this identifies a task.
    return json.dumps([arm, task["style"], task["topic"]])


# --- statistics & reporting -----------------------------------------------------

def summarize(episodes: list[Episode]) -> tuple[str, dict]:
    def arm_stats(rs: list[Episode]) -> dict | None:
        if not rs:
            return None
        good = [r for r in rs if not r.error]
        n = len(good)
        n_solved = sum(r.found for r in good)
        lo, hi = wilson_interval(n_solved, n)
        tokens = sorted(r.input_tokens + r.output_tokens for r in good)
        costs = [r.cost_usd for r in good]
        cost_lo, cost_hi = bootstrap_ci(costs, confidence=0.95)
        solved_cost = sum(costs) / n_solved if n_solved else float("inf")
        return {
            "n": len(rs), "finished": n,
            "success": n_solved / n if n else 0.0,
            "success_ci95": [round(lo, 4), round(hi, 4)],
            "p50_tok": tokens[len(tokens) // 2] if tokens else 0,
            "p90_tok": tokens[min(n - 1, int(n * 0.9))] if n else 0,
            "mean_cost_usd": mean(costs),
            "mean_cost_ci95": [round(cost_lo, 5), round(cost_hi, 5)],
            "cost_per_solved_usd": round(solved_cost, 5),
        }

    arms_stats = {arm: arm_stats([e for e in episodes if e.arm == arm])
                  for arm in ARMS}

    pairs_token, wins, losses = [], 0, 0
    by_task: dict[tuple, dict[str, Episode]] = {}
    for e in episodes:
        by_task.setdefault((e.style, e.topic), {})[e.arm] = e
    for pair in by_task.values():
        if set(pair) == set(ARMS) and not pair["shell"].error \
                and not pair["ffembed"].error:
            s, f = pair["shell"], pair["ffembed"]
            st, ft = s.input_tokens + s.output_tokens, f.input_tokens + f.output_tokens
            pairs_token.append((st, ft))
            wins += f.found > s.found
            losses += f.found < s.found
    pval = (paired_permutation_test([x for x, _ in pairs_token],
                                    [y for _, y in pairs_token])
            if len(pairs_token) >= 5 else float("nan"))

    metrics: dict = {"arms": arms_stats}
    if pairs_token:
        deltas = [y - x for x, y in pairs_token]
        d_lo, d_hi = bootstrap_ci(deltas, confidence=0.95)
        metrics["paired"] = {
            "n_pairs": len(pairs_token),
            "token_delta_mean": round(mean(deltas), 1),
            "token_delta_ci95": [round(d_lo, 1), round(d_hi, 1)],
            "ffembed_wins": wins, "shell_wins": losses,
            "permutation_p_value": round(pval, 4),
        }

    lines = ["## By arm\n",
             "| arm | n | success (95% CI) | p50 tok | p90 tok "
             "| cost/task | cost/solved |\n|---|---|---|---|---|---|---|\n"]
    for arm in ARMS:
        s = arms_stats[arm]
        if not s:
            continue
        lines.append(
            f"| {arm} | {s['finished']} | {100 * s['success']:.0f}% "
            f"[{100 * s['success_ci95'][0]:.0f}, {100 * s['success_ci95'][1]:.0f}] "
            f"| {s['p50_tok']:,} | {s['p90_tok']:,} "
            f"| ${s['mean_cost_usd']:.4f} | ${s['cost_per_solved_usd']:.4f} |\n")

    styles = sorted({e.style for e in episodes})
    lines.append("\n## Median tokens by question style\n"
                 "| arm | " + " | ".join(styles) + " |\n"
                 "|---|" + "---|" * len(styles) + "\n")
    for arm in ARMS:
        cells = []
        for st in styles:
            rs = [e for e in episodes if e.arm == arm and e.style == st
                  and not e.error]
            if rs:
                med = sorted(r.input_tokens + r.output_tokens
                             for r in rs)[len(rs) // 2]
                ok = sum(r.found for r in rs)
                cells.append(f"{med:,} ({ok}/{len(rs)})")
            else:
                cells.append("-")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |\n")

    if "paired" in metrics:
        p = metrics["paired"]
        lines.append(
            f"\nPaired over {p['n_pairs']} complete tasks: ffembed wins "
            f"{p['ffembed_wins']}, shell wins {p['shell_wins']}; mean token "
            f"delta {p['token_delta_mean']:,.0f} per task "
            f"(95% CI [{p['token_delta_ci95'][0]:,.0f}, "
            f"{p['token_delta_ci95'][1]:,.0f}]), permutation p="
            f"{p['permutation_p_value']}.\n")
    return "".join(lines), metrics


def load_prior(out_dir: Path, cfg: dict) -> set[str]:
    """Episode keys already recorded for an identical past configuration."""
    done: set[str] = set()
    jsonl = out_dir / "episodes.jsonl"
    if not jsonl.exists():
        return done
    wanted = {k: cfg[k] for k in ("size", "fillers", "bank", "model")}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rcfg = rec.get("config", {})
        if any(rcfg.get(k) != v for k, v in wanted.items()):
            continue
        arm, style, topic = rec.get("arm"), rec.get("style"), rec.get("topic")
        if arm and style and topic:
            done.add(_episode_key(arm, {"style": style, "topic": topic}))
    return done


# --- orchestration -----------------------------------------------------------------

def run_harness(size: int, per_style: int | None, arms: list[str],
                model: str, workers: int, fillers: int, bank: str,
                out_dir: Path) -> list[Episode]:
    tasks = _build_tasks(bank, per_style)
    cfg = {"size": size, "fillers": fillers, "bank": bank, "model": model}

    # Alternate arms so a task's two episodes start back to back.
    jobs = [(arm, t) for t in tasks for arm in arms]
    if len(arms) > 1:
        prior = load_prior(out_dir, cfg)
        jobs = [(arm, t) for arm, t in jobs
                if _episode_key(arm, t) not in prior]
        if prior:
            print(f"resuming: {len(prior)} episode(s) already recorded "
                  f"for this config\n")

    print(f"corpus: {size} notes ({bank}), {len(tasks)} tasks x {len(arms)} "
          f"arms, model: {model}\n")

    base = Path(tempfile.mkdtemp(prefix="ffembed-pi-"))
    episodes: list[Episode] = []
    jsonl = out_dir / "episodes.jsonl"
    try:
        corpus, home = _setup_workspace(base, size, fillers, bank)
        total = len(jobs)
        done = 0
        # Queue tasks in order so paired neighbours run close together, but
        # allow enough parallelism that slow episodes don't stall the pipeline.
        queue = list(jobs)
        with ThreadPoolExecutor(max_workers=max(workers, len(arms))) as pool:
            futures = {
                pool.submit(run_episode, arm, t, corpus, home, model): (arm, t)
                for arm, t in queue
            }
            for fut in as_completed(futures):
                arm, t = futures[fut]
                try:
                    ep = fut.result()
                except Exception as e:  # timeout etc.
                    ep = Episode(arm=arm, style=t["style"], topic=t["topic"],
                                 question=t["question"], found=False,
                                 chosen="", input_tokens=0, output_tokens=0,
                                 cost_usd=0.0, duration_s=0.0,
                                 tool_calls={}, error=str(e))
                done += 1
                mark = "✓" if ep.found else "✗"
                print(f"[{done}/{total}] {mark} {ep.arm:8} {ep.style:7} "
                      f"{ep.topic[:20]:20} in={ep.input_tokens:>6} "
                      f"out={ep.output_tokens:>5} ${ep.cost_usd:.4f} "
                      f"{ep.duration_s:5.1f}s "
                      f"{('ERR ' + ep.error[:60]) if ep.error else ''}",
                      flush=True)
                # Append immediately: interruptions never lose paid work.
                with jsonl.open("a") as fh:
                    fh.write(json.dumps({"config": cfg, **asdict(ep)}) + "\n")
                episodes.append(ep)
        return episodes
    finally:
        shutil.rmtree(base, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--fillers", type=int, default=4)
    ap.add_argument("--bank", choices=["v1", "v2"], default="v1")
    ap.add_argument("--per-style", type=int, default=None)
    ap.add_argument("--arms", default="shell,ffembed")
    ap.add_argument("--model",
                    default=os.environ.get("FFEMBED_BENCH_MODEL", "gpt-4o-mini"))
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    assert PI, "pi CLI not found on PATH"
    assert FFEMBED, "ffembed CLI not found on PATH"

    out_dir = Path(__file__).parent
    episodes = run_harness(args.size, args.per_style, args.arms.split(","),
                           args.model, args.workers, args.fillers, args.bank,
                           out_dir)

    report, metrics = summarize(episodes)
    print("\n" + report)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = write_results("agent_pi", vars(args), metrics,
                         [asdict(e) for e in episodes])
    md = out_dir / "results_pi.md"
    md.write_text(
        f"# pi harness — {args.size} notes, bank {args.bank}, "
        f"model {args.model}\n\nRun {stamp}. `success` = verdict was a file "
        f"containing the answer passage. CIs at 95%; Wilson for proportions, "
        f"bootstrap for costs/tokens. Every task ran in both arms.\n\n{report}",
        encoding="utf-8")
    print(f"wrote {md}\nwrote {path}")


if __name__ == "__main__":
    main()
