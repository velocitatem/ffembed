"""Pluggable retrieval systems behind one interface, so every suite scores
them identically.

    docs: dict[docid, text]
    search(query, k) -> list[(docid, score)]

Systems:
- ffembed:<model>   the real thing (indexer + cosine scan), via an isolated home
- bm25              bm25s Okapi BM25 — the canonical sparse baseline
- fts5              SQLite's built-in full-text search
- hybrid            reciprocal-rank fusion of ffembed:<model> + bm25
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

SYSTEMS_HELP = """\
system names understood by the retrieval suites:
  ffembed:bge-small | ffembed:minilm | ...   (any alias from `ffembed models`)
  bm25
  fts5
  hybrid:bge-small        (RRF fusion of ffembed:<model> and bm25)
"""


class Retriever:
    def __init__(self, name: str):
        self.name = name

    def index(self, docs: dict[str, str]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def make_system(spec: str) -> Retriever:
    spec = spec.strip()
    if spec.startswith("ffembed:"):
        return FFEmbedRetriever(spec.split(":", 1)[1])
    if spec.startswith("hybrid:"):
        return HybridRetriever(spec.split(":", 1)[1])
    if spec == "bm25":
        return BM25Retriever()
    if spec == "fts5":
        return FTS5Retriever()
    raise ValueError(f"unknown system '{spec}'\n{SYSTEMS_HELP}")


# --- ffembed -----------------------------------------------------------------

class FFEmbedRetriever(Retriever):
    """Index docs through ffembed's real indexer into an isolated HOME."""

    def __init__(self, model: str):
        super().__init__(f"ffembed:{model}")
        self.model = model
        self._tmp: Path | None = None
        self._conn: sqlite3.Connection | None = None
        self._target_path: str | None = None
        self._file_of: dict[str, str] = {}
        self._flushed = 0

    def index(self, docs: dict[str, str], *, batched: bool = True) -> None:
        """Index through ffembed's own store/chunker. Batched mode embeds all
        chunks in large batches (same vectors, ~50x faster than per-file
        ONNX round trips for corpora in the thousands)."""
        from ffembed import db as ffembed_db

        from .isolation import isolate, restore

        # Isolate HOME so indexing never touches the user's real ~/.ffembed.
        self._tmp = Path(tempfile.mkdtemp(prefix="ffembed-retriever-"))
        home = self._tmp / "home"
        saved = isolate(home)

        root = self._tmp / "corpus"
        root.mkdir(parents=True)
        # Filenames must be flat; keep the mapping back to real docids.
        self._file_of: dict[str, str] = {}
        for docid, text in docs.items():
            fname = docid.replace("/", "__")
            assert fname not in self._file_of.values(), f"collision for {docid}"
            self._file_of[fname] = docid
            (root / f"{fname}.txt").write_text(text, encoding="utf-8")

        try:
            self._conn = ffembed_db.connect()
            target_id = ffembed_db.add_target(
                self._conn, str(root), "*.txt", self.model
            )
            row = self._conn.execute(
                "SELECT * FROM targets WHERE id = ?", (target_id,)
            ).fetchone()
            if batched:
                self._index_batched(row)
            else:
                from ffembed import indexer

                indexer.index_target(self._conn, row)
            self._conn.commit()
        finally:
            restore(saved)
        self._target_path = str(root)

    def _index_batched(self, target_row) -> None:
        """Chunk every doc with ffembed's own chunker, embed all chunks in
        batches, write directly through ffembed's DB layer."""
        import hashlib
        import os
        from ffembed.chunk import chunk_text
        from ffembed.db import insert_chunk, upsert_file
        from ffembed.embed import embed_texts

        batch: list[tuple[int, int, str]] = []  # (file_id, idx, text)

        def flush():
            if not batch:
                return
            vecs = embed_texts(self.model, [t for _, _, t in batch])
            for (fid, idx, text), vec in zip(batch, vecs):
                insert_chunk(self._conn, fid, idx, text,
                             vec.tolist() if hasattr(vec, "tolist") else list(vec))
            batch.clear()
            self._flushed += len(vecs)
            if os.environ.get("FFEMBED_BENCH_VERBOSE"):
                print(f"[index {self.name}] {self._flushed} chunks", flush=True)

        root_path = Path(target_row["path"])
        all_txt = sorted(root_path.glob("*.txt"))
        for path in all_txt:
            data = path.read_bytes()
            file_id = upsert_file(
                self._conn, target_row["id"], str(path),
                os.path.getmtime(path), hashlib.sha256(data).hexdigest(),
            )
            for i, chunk in enumerate(chunk_text(data.decode("utf-8"))):
                batch.append((file_id, i, chunk))
                if len(batch) >= 256:
                    flush()
        flush()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        from ffembed import search as ff_search

        results = ff_search.search(
            self._conn, query, target_path=self._target_path, k=k, model=self.model
        )
        out = []
        for score, row in results:
            stem = Path(row["file_path"]).stem
            # stem == docid.replace('/', '__'); invert deterministically.
            docid = self._file_of.get(stem) or stem.replace("__", "/")
            out.append((docid, float(score)))
        return out

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


# --- sparse baselines ----------------------------------------------------------

class BM25Retriever(Retriever):
    def __init__(self):
        super().__init__("bm25")
        self._retriever = None
        self._ids: list[str] = []

    def index(self, docs: dict[str, str]) -> None:
        import bm25s

        self._ids = list(docs)
        corpus_tokens = bm25s.tokenize([docs[i] for i in self._ids],
                                       stopwords="en", show_progress=False)
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        import bm25s

        q_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results, scores = self._retriever.retrieve(q_tokens, k=k,
                                                   show_progress=False)
        # Drop zero-score padding so unhit queries yield genuinely empty runs.
        return [(self._ids[i], float(s)) for i, s in zip(results[0], scores[0])
                if s > 0]


class FTS5Retriever(Retriever):
    # Small hard-coded stopword list: bm25s ships one, SQLite does not.
    _STOPS = frozenset(
        "a an and are as at be but by for from had has have he her his i in "
        "is it its of on or that the their them then there these they this to "
        "was we were what when where which who will with you your we our so no"
        "not do did does how why".split()
    )

    def __init__(self):
        super().__init__("fts5")
        import re

        self._word_re = re.compile(r"[a-z0-9]+")
        self._conn = sqlite3.connect(":memory:")

    def index(self, docs: dict[str, str]) -> None:
        c = self._conn
        c.execute("CREATE VIRTUAL TABLE docs USING fts5(docid UNINDEXED, body)")
        c.executemany("INSERT INTO docs VALUES (?, ?)", list(docs.items()))
        c.commit()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        # Mirror what a user actually gets from SQLite search: strip
        # punctuation/stopwords and OR the remaining content words.
        tokens = [t for t in self._word_re.findall(query.lower())
                  if t not in self._STOPS]
        safe = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
        if not safe:
            return []
        rows = self._conn.execute(
            "SELECT docid, bm25(docs) AS rank FROM docs WHERE docs MATCH ? "
            "ORDER BY rank LIMIT ?", (safe, k),
        ).fetchall()
        # bm25() returns lower-is-better; flip so higher score == better.
        return [(docid, -float(rank)) for docid, rank in rows]

    def close(self) -> None:
        self._conn.close()


class HybridRetriever(Retriever):
    """Reciprocal-rank fusion of dense (ffembed) and sparse (bm25) runs."""

    RRF_K = 60

    def __init__(self, model: str, dense: FFEmbedRetriever | None = None,
                 sparse: BM25Retriever | None = None):
        super().__init__(f"hybrid:{model}")
        # Share already-indexed sub-retrievers so hybrid does not re-embed.
        self.dense = dense if dense is not None else FFEmbedRetriever(model)
        self.sparse = sparse if sparse is not None else BM25Retriever()
        self.n_docs = 0
        self._owns_dense = dense is None
        self._owns_sparse = sparse is None

    def index(self, docs: dict[str, str]) -> None:
        self.docs = docs
        self.n_docs = len(docs)
        self.dense.index(docs)
        self.sparse.index(docs)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        depth = min(max(k * 5, 50), self.n_docs)
        runs = [self.dense.search(query, depth), self.sparse.search(query, depth)]
        fused: dict[str, float] = {}
        for run in runs:
            for rank, (docid, _) in enumerate(run):
                fused[docid] = fused.get(docid, 0.0) + 1.0 / (self.RRF_K + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return ranked

    def close(self) -> None:
        if self._owns_dense:
            self.dense.close()
        if self._owns_sparse:
            self.sparse.close()
