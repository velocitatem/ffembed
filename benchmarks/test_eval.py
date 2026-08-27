"""Unit checks for the shared evaluation/statistics machinery."""

from __future__ import annotations

import random

import pytest

from .qrels import evaluate, ndcg_at_k, recall_at_k, rr_at_k
from .stats import bootstrap_ci, paired_permutation_test, wilson_interval


def _random_qrels_run(seed=0, n_q=20, n_docs=50, seed_ratio=0.1):
    rng = random.Random(seed)
    qrels, run = {}, {}
    for q in range(n_q):
        qid = f"q{q}"
        golds = rng.sample(range(n_docs), k=rng.randint(1, 5))
        qrels[qid] = {f"d{g}": rng.choice([1, 2]) for g in golds}
        scores = {}
        for d in range(n_docs):
            if rng.random() < 0.8:
                scores[f"d{d}"] = rng.random()
        # boost a random gold sometimes to create signal
        if rng.random() < 0.7:
            gid = next(iter(qrels[qid]))
            scores[gid] = 10.0
        run[qid] = scores
    return qrels, run


def test_pure_python_matches_pytrec_eval():
    pytest.importorskip("pytrec_eval")
    from benchmarks.qrels import HAS_PYTREC

    assert HAS_PYTREC
    from pytrec_eval import RelevanceEvaluator

    qrels, run = _random_qrels_run()
    ours = evaluate(qrels, run)
    judge = RelevanceEvaluator(
        {q: {d: int(g) for d, g in qs.items()} for q, qs in qrels.items()},
        {"ndcg_cut", "recall"},
    )
    theirs = judge.evaluate(run)
    n = len(qrels)
    their_ndcg = sum(v["ndcg_cut_10"] for v in theirs.values()) / n
    their_recall = sum(v["recall_100"] for v in theirs.values()) / n
    assert abs(ours["ndcg@10"] - their_ndcg) < 1e-9
    assert abs(ours["recall@100"] - their_recall) < 1e-9


def test_metric_edge_cases():
    assert ndcg_at_k({"a": 2}, {"b": 1.0}, 10) == 0.0
    assert ndcg_at_k({"a": 2}, {"a": 1.0}, 10) == 1.0
    assert rr_at_k({"a": 1}, {"x": 9, "y": 8, "a": 7}, 10) == 1 / 3
    assert recall_at_k({"a": 1, "b": 1}, {"a": 1}, 10) == 0.5
    assert recall_at_k({}, {"a": 1}, 10) == 0.0


def test_wilson_interval_sane():
    lo, hi = wilson_interval(18, 24)
    assert 0.55 <= lo <= 0.95 and hi <= 1.0 and lo < hi
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_bootstrap_ci_and_permutation():
    xs_a = [100.0] * 20 + [200.0] * 10
    xs_b = [80.0] * 30
    lo, hi = bootstrap_ci(xs_b, iters=500, seed=1)
    assert lo == hi == 80.0  # constant list -> degenerate CI

    diffs_strong = ([0.0] * 3 + [10.0] * 27, [-10.0] * 27 + [0.0] * 3)
    p = paired_permutation_test(*diffs_strong, iters=2000, seed=2)
    assert p < 0.01

    # identical lists should never be "significant"
    same = [float(i % 7) for i in range(40)]
    p_same = paired_permutation_test(same, same[:], iters=2000, seed=3)
    assert p_same > 0.5
