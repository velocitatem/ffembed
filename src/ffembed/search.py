"""Brute-force cosine similarity search over stored chunks. Fine at personal
scale (thousands to low tens-of-thousands of chunks); no ANN index needed."""

from __future__ import annotations

import math
import sqlite3

from pathlib import Path

from . import db
from .embed import embed_query
from .vision import DEFAULT_VISION_MODEL, embed_image, is_image_path


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _group_by_model(rows, kind: str):
    """Group rows by the model that produced their embeddings."""
    key = "vision_model" if kind == "image" else "model"
    groups: dict[str, list] = {}
    for r in rows:
        model_name = r[key]
        if kind == "image" and not model_name:
            model_name = DEFAULT_VISION_MODEL
        groups.setdefault(model_name, []).append(r)
    return groups


def search(conn: sqlite3.Connection, query: str, *, target_path: str | None = None, k: int = 5, model: str | None = None):
    kind = "image" if is_image_path(query) and Path(query).exists() else "text"
    rows = db.all_chunks_for_search(conn, target_path, kind=kind)
    if not rows:
        return []

    by_model = _group_by_model(rows, kind)

    results = []
    for model_name, model_rows in by_model.items():
        if model and model_name != model:
            continue
        if kind == "image":
            q_vec = embed_image(model_name, query)
        else:
            q_vec = embed_query(model_name, query)
        for r in model_rows:
            vec = db.unpack_vector(r["embedding"])
            score = cosine(q_vec, vec)
            results.append((score, r))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]
