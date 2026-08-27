"""Systems benchmark at scale: what does ffembed cost to run, not just how
smart is it?

Per corpus scale (small / medium / large), measures:
- query latency distribution: p50 / p95 / p99 over a mixed query workload
- indexing throughput (docs/s)
- peak resident memory (whole process; coarse but honest)
- on-disk index size

Baselines at identical scales: bm25s (sparse, indexed) and ripgrep (no
index, scanned per query).

Usage:
    uv run --group dev python -m benchmarks.run_systems \
        --sizes 1000 5000 20000 --model bge-small

(results also written as JSON under benchmarks/results/systems/)
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .corpus import generate_implicit_corpus_v2
from .question_bank_v2 import build_tasks_v2
from .retrievers import BM25Retriever
from .results import write_results


def percentile(sorted_vals: list[float], p: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, int(round(p / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def run_scale(n_notes: int, model: str, n_queries: int,
              tmp: Path, fillers: int = 12) -> dict | None:
    from ffembed import db as ffembed_db
    from ffembed import indexer

    from .isolation import isolate, restore

    root = tmp / f"corpus_{n_notes}"
    paths, _qrels = generate_implicit_corpus_v2(
        root, n_notes, fillers_per_note=fillers
    )

    home = tmp / "home"
    saved = isolate(home)

    result: dict = {"notes": n_notes}
    conn = None

    try:
        conn = ffembed_db.connect()
        target_id = ffembed_db.add_target(conn, str(root), "*.md", model)
        row = conn.execute("SELECT * FROM targets WHERE id=?",
                           (target_id,)).fetchone()

        # --- indexing throughput ---
        t0 = time.perf_counter()
        indexer.index_target(conn, row)
        conn.commit()
        index_s = time.perf_counter() - t0
        stats = ffembed_db.stats(conn)
        result["chunks"] = stats["chunks"]
        result["index_seconds"] = round(index_s, 2)
        result["index_docs_per_s"] = round(n_notes / index_s, 1)
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
        result["index_db_mb"] = round(Path(db_path).stat().st_size / 1e6, 1)

        queries = [t["question"] for t in build_tasks_v2()]

        # --- ffembed dense latencies ---
        from ffembed import search as ff_search

        latencies = []
        for i in range(n_queries):
            q = queries[i % len(queries)]
            t0 = time.perf_counter()
            ff_search.search(conn, q, target_path=str(root), k=5, model=model)
            latencies.append((time.perf_counter() - t0) * 1000)
        srt = sorted(latencies)
        result["ffembed_latency_ms_p50"] = round(percentile(srt, 50), 2)
        result["ffembed_latency_ms_p95"] = round(percentile(srt, 95), 2)
        result["ffembed_latency_ms_p99"] = round(percentile(srt, 99), 2)
        conn.close()

        # --- bm25 at same scale ---
        sys_bm = BM25Retriever()
        docs = {p.stem: p.read_text(encoding="utf-8") for p in paths}
        t0 = time.perf_counter()
        sys_bm.index(docs)
        bm_index_s = time.perf_counter() - t0
        lat = []
        for i in range(n_queries):
            q = queries[i % len(queries)]
            t0 = time.perf_counter()
            sys_bm.search(q, k=5)
            lat.append((time.perf_counter() - t0) * 1000)
        srt = sorted(lat)
        result["bm25_index_seconds"] = round(bm_index_s, 2)
        result["bm25_latency_ms_p50"] = round(percentile(srt, 50), 3)
        result["bm25_latency_ms_p95"] = round(percentile(srt, 95), 3)
        result["bm25_latency_ms_p99"] = round(percentile(srt, 99), 3)

        # --- cold keyword scan ---
        from .question_bank_v2 import BANK

        cold = []
        topics_cycle = sorted(BANK)[:n_queries]
        for w in topics_cycle:
            kw = w.split()[0]
            t0 = time.perf_counter()
            subprocess.run(["grep", "-rl", "--include=*.md", kw, str(root)],
                           capture_output=True, text=True, check=False)
            cold.append((time.perf_counter() - t0) * 1000)
        srt = sorted(cold)
        result["grep_cold_latency_ms_p50"] = round(percentile(srt, 50), 1)
        result["grep_cold_latency_ms_p95"] = round(percentile(srt, 95), 1)
        result["grep_cold_latency_ms_p99"] = round(percentile(srt, 99), 1)

        result["peak_rss_mb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        return result
    finally:
        from .isolation import restore

        restore(saved)
        if conn is not None:
            conn.close()


def render_markdown(records: list[dict]) -> str:
    lines = [
        "| notes | chunks | idx s | idx docs/s | db MB | ff p50/p95/p99 ms "
        "| bm25 p50/p95/p99 ms | grep cold p50/p95 ms |\n"
        "|---|---|---|---|---|---|---|---|\n",
    ]
    for r in records:
        lines.append(
            f"| {r['notes']:,} | {r['chunks']:,} | {r['index_seconds']} "
            f"| {r['index_docs_per_s']} | {r['index_db_mb']} "
            f"| {r['ffembed_latency_ms_p50']}/{r['ffembed_latency_ms_p95']}"
            f"/{r['ffembed_latency_ms_p99']} "
            f"| {r['bm25_latency_ms_p50']}/{r['bm25_latency_ms_p95']}"
            f"/{r['bm25_latency_ms_p99']} "
            f"| {r['grep_cold_latency_ms_p50']}/{r['grep_cold_latency_ms_p95']} |\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000])
    ap.add_argument("--model", default="bge-small")
    ap.add_argument("--queries", type=int, default=60,
                    help="latency samples per system per size")
    ap.add_argument("--fillers", type=int, default=12,
                    help="off-topic passages per note (drives chunks/note)")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="ffembed-systems-"))
    records = []
    try:
        for n in args.sizes:
            print(f"scale {n} notes...", flush=True)
            rec = run_scale(n, args.model, args.queries, tmp,
                            fillers=args.fillers)
            if rec:
                records.append(rec)
                print(json.dumps(rec, indent=None))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    metrics = {"records": records,
               "peak_rss_process_mb": round(
                   resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    path = write_results("systems", vars(args), metrics, records)
    md = Path(__file__).parent / "results_systems.md"
    md.write_text("# Systems scaling\n\n" + render_markdown(records),
                  encoding="utf-8")
    print(render_markdown(records))
    print(f"wrote {md}\nwrote {path}")


if __name__ == "__main__":
    main()
