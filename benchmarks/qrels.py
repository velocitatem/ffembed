"""Relevance-graded retrieval evaluation, TREC style.

qrels / runs use the standard nested dicts:

    qrels[qid][docid] -> int relevance grade
    run[qid][docid]   -> float system score

Metrics follow BEIR conventions: nDCG@10, MRR@10, Recall@100.
Uses ``pytrec_eval`` (the library behind BEIR and MTEB) when importable and
falls back to an equivalent pure-Python implementation otherwise.
"""

from __future__ import annotations

import math
from typing import Callable

try:
    import pytrec_eval

    HAS_PYTREC = True
except ImportError:  # pragma: no cover - fallback path
    pytrec_eval = None
    HAS_PYTREC = False


def topk_run(run: dict[str, dict[str, float]], k: int) -> dict[str, dict[str, float]]:
    """Truncate every query's ranked list to its top-k docs."""
    out = {}
    for qid, scores in run.items():
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        out[qid] = dict(ranked)
    return out


# --- pure-python reference implementations ---------------------------------

def _dcg(grants: list[float]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grants))


def ndcg_at_k(qrel: dict[str, int], scored: dict[str, float], k: int) -> float:
    gains = [qrel.get(d, 0) for d in sorted(scored, key=lambda x: (-scored[x], x))[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    return _dcg(gains) / _dcg(ideal) if ideal else 0.0


def rr_at_k(qrel: dict[str, int], scored: dict[str, float], k: int) -> float:
    ranked = sorted(scored, key=lambda x: (-scored[x], x))[:k]
    for i, d in enumerate(ranked):
        if qrel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(qrel: dict[str, int], scored: dict[str, float], k: int,
                min_grade: int = 1) -> float:
    relevant = {d for d, g in qrel.items() if g >= min_grade}
    if not relevant:
        return 0.0
    ranked = set(sorted(scored, key=lambda x: (-scored[x], x))[:k])
    return len(relevant & ranked) / len(relevant)


def evaluate(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]],
             k_ndcg: int = 10, k_rr: int = 10, k_recall: int = 100) -> dict[str, float]:
    """Return mean nDCG@k_ndcg, MRR@k_rr, Recall@k_recall over all queries."""
    run_trunc = topk_run(run, max(k_ndcg, k_rr, k_recall))
    queries = [qid for qid in qrels if qrels[qid] and qid in run_trunc]
    if not queries:
        raise ValueError("no overlapping queries between qrels and run")

    if HAS_PYTREC:
        judge = pytrec_eval.RelevanceEvaluator(
            {q: {d: int(g) for d, g in qrels[q].items()} for q in queries},
            {"ndcg_cut", "recall"},
        )
        per_query = judge.evaluate({q: run_trunc[q] for q in queries})
        ndcgs = [per_query[q]["ndcg_cut_" + str(k_ndcg)] for q in queries]
        recalls = [per_query[q]["recall_" + str(k_recall)] for q in queries]
    else:
        ndcgs = [ndcg_at_k(qrels[q], run_trunc[q], k_ndcg) for q in queries]
        recalls = [recall_at_k(qrels[q], run_trunc[q], k_recall) for q in queries]

    rrs = [rr_at_k(qrels[q], run_trunc[q], k_rr) for q in queries]

    def m(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    return {
        f"ndcg@{k_ndcg}": m(ndcgs),
        f"mrr@{k_rr}": m(rrs),
        f"recall@{k_recall}": m(recalls),
        "_n_queries": float(len(queries)),
    }


# --- TREC file formats ------------------------------------------------------

def write_trec_run(run: dict[str, dict[str, float]], path, tag: str = "system") -> None:
    """Write a run in standard TREC format: qid Q0 docid rank score tag."""
    lines = []
    for qid, scores in sorted(run.items()):
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        for rank, (docid, score) in enumerate(ranked, start=1):
            lines.append(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{tag}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_qrels_tsv(path) -> dict[str, dict[str, int]]:
    """Read qrels TSV in either layout:

    - BEIR zip format:      qid<TAB>0<TAB>docid<TAB>grade   (4 cols)
    - HF BeIR/*-qrels mirror: qid<TAB>docid<TAB>grade       (3 cols)
    """
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                qid, _, docid, grade = parts[:4]
            elif len(parts) == 3:
                qid, docid, grade = parts
            else:
                continue
            qrels.setdefault(qid, {})[docid] = int(grade)
    return qrels
