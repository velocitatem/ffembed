"""Retrieval-quality suite: BEIR-style graded evaluation of search systems.

Loads a dataset (corpus / queries / qrels), runs each requested system,
scores nDCG@10, MRR@10, Recall@100 with pytrec_eval, and writes a JSON
result file under benchmarks/results/retrieval/.

Usage:
    uv run --group dev python -m benchmarks.run_retrieval \
        --dataset synthetic --size 100 \
        --systems ffembed:bge-small ffembed:minilm bm25 fts5 hybrid:bge-small

    uv run --group dev python -m benchmarks.run_retrieval \
        --dataset scifact --max-queries 100 --systems bm25 ffembed:bge-small

    uv run --group dev python -m benchmarks.run_retrieval \
        --dataset repo --repo-url https://github.com/pallets/flask.git \
        --repo-ref 3.0.3

Every run also emits TREC-format run files next to the JSON for inspection
with external tools.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import datasets_io
from .qrels import evaluate, topk_run, write_trec_run
from .retrievers import SYSTEMS_HELP, make_system
from .retrievers import BM25Retriever, FFEmbedRetriever  # noqa: F401 (sharing API)
from .results import RESULTS_DIR, write_results


def load_dataset(args) -> tuple[dict, dict, dict]:
    if args.dataset == "synthetic":
        return datasets_io.load_synthetic(files=args.size)
    if args.dataset == "repo":
        return datasets_io.load_repo(
            url=args.repo_url, ref=args.repo_ref, max_commits=args.max_commits
        )
    if args.dataset == "csn":
        return datasets_io.load_csn(max_queries=args.max_queries or 300)
    return datasets_io.load_beir(args.dataset, max_queries=args.max_queries)


def run_system(spec: str, docs: dict, queries: dict, args) -> dict:
    """Index + query one system; returns its scored run and timing info."""
    retriever = make_system(spec)
    t0 = time.perf_counter()
    retriever.index(docs)
    index_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    run: dict[str, dict[str, float]] = {}
    depth = min(100, max(1, len(docs)))
    for qid, query in queries.items():
        run[qid] = {docid: score for docid, score in retriever.search(query, k=depth)}
    query_ms = (time.perf_counter() - t0) * 1000 / max(1, len(queries))
    retriever.close()
    return {"run": topk_run(run, 100), "index_s": index_s, "query_ms": query_ms}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=["synthetic", "scifact", "nfcorpus", "touche2020", "repo", "csn"])
    ap.add_argument("--size", type=int, default=100,
                    help="synthetic corpus size in notes")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--systems", nargs="+", required=True, help=SYSTEMS_HELP)
    ap.add_argument("--k", type=int, default=5,
                    help="top-k handed to ffembed at query time")
    ap.add_argument("--repo-url", default="https://github.com/pallets/flask.git")
    ap.add_argument("--repo-ref", default="3.0.3")
    ap.add_argument("--max-commits", type=int, default=120)
    args = ap.parse_args()

    docs, queries, qrels = load_dataset(args)
    print(f"dataset={args.dataset}: {len(docs)} docs, {len(queries)} queries")

    records = []
    metrics = {}
    stamp_dir = RESULTS_DIR / "retrieval"
    stamp_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.systems:
        label = spec.replace(":", "_").replace("/", "-")
        t0 = time.time()
        result = run_system(spec, docs, queries, args)
        scores = evaluate(qrels, result["run"])
        elapsed = time.time() - t0
        metrics[spec] = {
            **{k: round(v, 4) for k, v in scores.items()},
            "index_seconds": round(result["index_s"], 2),
            "mean_query_ms": round(result["query_ms"], 2),
        }
        write_trec_run(result["run"], stamp_dir / f"run_{label}.trec", tag=label)
        print(f"{spec:24} nDCG@10={metrics[spec]['ndcg@10']:.3f} "
              f"MRR@10={metrics[spec]['mrr@10']:.3f} "
              f"R@100={metrics[spec]['recall@100']:.3f} "
              f"(idx {result['index_s']:.1f}s, q {result['query_ms']:.1f}ms)")
        records.append({
            "system": spec,
            "elapsed_s": elapsed,
            "per_query": {qid: sorted(runv.items(), key=lambda kv: -kv[1])[:10]
                          for qid, runv in result["run"].items()},
        })

    config = vars(args)
    path = write_results("retrieval", config, metrics, [
        {**r, "per_query": "<written to TREC run files>"} for r in records
    ])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
