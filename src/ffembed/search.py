"""Brute-force cosine similarity search over stored chunks. Fine at personal
scale (thousands to low tens-of-thousands of chunks); no ANN index needed."""

from __future__ import annotations

import math
import sqlite3

from . import db
from .embed import embed_query


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search(conn: sqlite3.Connection, query: str, *, target_path: str | None = None, k: int = 5, model: str | None = None):
    rows = db.all_chunks_for_search(conn, target_path)
    if not rows:
        return []

    by_model: dict[str, list] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    results = []
    for model_name, model_rows in by_model.items():
        if model and model_name != model:
            continue
        q_vec = embed_query(model_name, query)
        for r in model_rows:
            vec = db.unpack_vector(r["embedding"])
            score = cosine(q_vec, vec)
            results.append((score, r))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]
