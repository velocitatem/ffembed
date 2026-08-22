"""The embedding garden: short aliases over local, ONNX-runtime embedding
models served via fastembed. Nothing leaves the machine."""

from __future__ import annotations

from functools import lru_cache

from .paths import CACHE_DIR, ensure_root

GARDEN = {
    "bge-small": "BAAI/bge-small-en-v1.5",       # 384d, default: fast + solid
    "bge-base": "BAAI/bge-base-en-v1.5",         # 768d, more accurate, slower
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",  # 384d, tiny/fast
    "gte-small": "thenlper/gte-small",           # 384d
    "multilingual": "intfloat/multilingual-e5-small",  # 384d, many languages
}

DEFAULT_MODEL = "bge-small"


def resolve_model_name(alias: str) -> str:
    return GARDEN.get(alias, alias)


@lru_cache(maxsize=4)
def _loaded(model_name: str):
    from fastembed import TextEmbedding

    ensure_root()
    return TextEmbedding(model_name=resolve_model_name(model_name), cache_dir=str(CACHE_DIR))


def embed_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _loaded(model_name)
    return [vec.tolist() for vec in model.embed(texts)]


def embed_query(model_name: str, text: str) -> list[float]:
    model = _loaded(model_name)
    return next(iter(model.query_embed([text]))).tolist()
