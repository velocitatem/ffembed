"""Dataset loaders producing BEIR-style triples:

    docs    dict[docid, text]
    queries dict[qid, query text]
    qrels   dict[qid, {docid: graded relevance}]

Available datasets:

- ``synthetic``      frozen question_bank_v2 topics rendered as neutral notes
- ``scifact``, ``nfcorpus`` + other BeIR sets   downloaded from HF hub mirrors
- ``repo``           real repo clone; queries from commit subjects, positives
                     from the files each commit touched
- ``csn``            CodeSearchNet (python) reformatted to file-level retrieval

All downloads land in ``benchmarks/.data/`` (gitignored).
"""

from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / ".data"

HF_BASE = "https://huggingface.co/datasets/BeIR"


# --- shared helpers ----------------------------------------------------------

def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ffembed-bench"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _read_parquet(data: bytes):
    import io

    import pandas as pd

    return pd.read_parquet(io.BytesIO(data))


# --- synthetic ---------------------------------------------------------------

def load_synthetic(files: int = 100, fillers_per_note: int = 2, seed: int = 7):
    """Render v2-question-bank notes; evaluate on questions for every topic
    whose main-subject note belongs to the query-visible half (deterministic
    per topic name)."""
    import zlib

    from .corpus import generate_implicit_corpus_v2
    from .question_bank_v2 import BANK, QUESTION_STYLES

    root = DATA_DIR / f"synthetic_{files}"
    paths, doc_qrels = generate_implicit_corpus_v2(
        root, files, fillers_per_note=fillers_per_note, seed=seed
    )
    docs = {p.stem: p.read_text(encoding="utf-8") for p in paths}

    topics = sorted(BANK)
    vis = {t: zlib.crc32(t.encode("utf-8")) % 2 == 0 for t in topics}

    queries, qrels = {}, {}
    for t in topics:
        grades = {d: g for d, g in doc_qrels[t].items() if d in docs}
        mains = [d for d, g in grades.items() if g == 2]
        if not vis[t] or not mains:
            continue
        # Each topic yields three distinct query ids, one per style.
        for i, style in enumerate(QUESTION_STYLES):
            qid = f"{t.replace(' ', '_')}#{i}"
            queries[qid] = BANK[t]["questions"][style]
            qrels[qid] = dict(grades)
    return docs, queries, qrels


# --- BeIR --------------------------------------------------------------------

BEIR_SETS = {"scifact", "nfcorpus", "touche2020"}


def load_beir(name: str, max_queries: int | None = None):
    """Zero-shot BEIR subset from the official HF mirrors (parquet + TSV)."""
    assert name in BEIR_SETS, f"dataset '{name}' not registered ({sorted(BEIR_SETS)})"

    corpus_dir = DATA_DIR / name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_pq = corpus_dir / "corpus.parquet"
    queries_pq = corpus_dir / "queries.parquet"
    qrels_tsv = corpus_dir / "test.tsv"
    if not corpus_pq.exists():
        corpus_pq.write_bytes(
            _http_get(f"{HF_BASE}/{name}/resolve/main/corpus/corpus-00000-of-00001.parquet")
        )
    if not queries_pq.exists():
        queries_pq.write_bytes(
            _http_get(f"{HF_BASE}/{name}/resolve/main/queries/queries-00000-of-00001.parquet")
        )
    if not qrels_tsv.exists():
        raw = _http_get(f"{HF_BASE}/{name}-qrels/resolve/main/test.tsv").decode("utf-8")
        qrels_tsv.write_text(raw, encoding="utf-8")

    corpus_df = _read_parquet(corpus_pq.read_bytes())
    queries_df = _read_parquet(queries_pq.read_bytes())

    from .qrels import read_qrels_tsv

    all_qrels = read_qrels_tsv(qrels_tsv)
    docs = {}
    for _, row in corpus_df.iterrows():
        did = str(row["_id"])
        title = row["title"] if isinstance(row["title"], str) else ""
        body = row["text"] if isinstance(row["text"], str) else ""
        text = f"{title}\n\n{body}".strip()
        if text:
            docs[did] = text
    queries, qrels = {}, {}
    ids = list(all_qrels)
    if max_queries:
        ids = ids[:max_queries]
    for qid in ids:
        queries[qid] = str(queries_df.loc[queries_df["_id"] == qid, "text"].iloc[0])
        qrels[qid] = {d: g for d, g in all_qrels[qid].items() if d in docs}
        qrels[qid] = {d: g for d, g in qrels[qid].items() if g > 0}
        if not qrels[qid]:
            del queries[qid], qrels[qid]
    return docs, queries, qrels


# --- real-repo tasks ---------------------------------------------------------

def load_repo(url: str = "https://github.com/pallets/flask.git",
              ref: str = "3.0.3", globs: tuple[str, ...] = ("*.py",),
              max_commits: int = 120, min_msg_words: int = 5):
    """Index a cloned open-source repo; derive tasks from commit history.

    Query  = commit subject (what a teammate would type to find that change)
    Rel 2  = files touched by that commit
    Rel 1  = files carrying identifiers mentioned in the message
    """
    clone = DATA_DIR / ("repo_" + url.rstrip("/").split("/")[-1].removesuffix(".git"))
    if not (clone / ".git").exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", url, str(clone)],
                       check=True)
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    stamp = clone / ".ffembed-ref"
    stamped = stamp.read_text().strip() if stamp.exists() else ""
    if stamped != f"{ref}:{head[:12]}":
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", ref], check=True)
        head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        stamp.write_text(f"{ref}:{head[:12]}")

    docs = {}
    for glob in globs:
        out = subprocess.run(["git", "-C", str(clone), "ls-files", glob],
                             capture_output=True, text=True, check=True).stdout
        for rel in out.splitlines():
            path = clone / rel
            try:
                docs[rel] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pass

    log = subprocess.run(
        ["git", "-C", str(clone), "log", "-n", str(max_commits),
         "--pretty=format:%H%x00%s"],
        capture_output=True, text=True, check=True).stdout.splitlines()

    queries, qrels = {}, {}
    for line in log:
        sha, _, subject = line.partition("\x00")
        if len(subject.split()) < min_msg_words or subject.lower().startswith(("bump", "release")):
            continue
        changed = subprocess.run(
            ["git", "-C", str(clone), "diff-tree", "--no-commit-id", "--name-only",
             "-r", sha],
            capture_output=True, text=True, check=True).stdout.splitlines()
        changed = [c for c in changed if c in docs and "test" not in c]
        if not 1 <= len(changed) <= 4:
            continue
        qid = sha[:10]
        queries[qid] = subject
        qrels[qid] = {c: 2 for c in changed}
        # Cheap negative-adjacent relevance: same directory neighbours often
        # share vocabulary; scoring them 1 lets nDCG distinguish quality.
        dirs = {str(Path(c).parent) for c in changed}
        qrels[qid].update({
            d: 1 for d in docs
            if str(Path(d).parent) in dirs and d not in changed
        })
    return docs, queries, qrels


# --- CodeSearchNet -----------------------------------------------------------

CSN_PARQUET_URL = (
    "https://huggingface.co/datasets/code-search-net/code_search_net/"
    "resolve/main/python/test-00000-of-00001.parquet"
)


def load_csn(max_queries: int = 300, n_distractors: int = 2000):
    """CodeSearchNet python test split -> file-level retrieval.

    Query = a function's docstring; gold = the file containing it (rel 2),
    same-repo other files (rel 1). The corpus is the union of gold files plus
    ``n_distractors`` random non-gold files from any repo, so results stay
    comparable run-to-run without paying to embed all ~10k files.
    """
    import random

    target = DATA_DIR / "csn_python_test.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(_http_get(CSN_PARQUET_URL))
    df = _read_parquet(target.read_bytes())

    rng = random.Random(42)
    urls = df["func_code_url"].tolist()
    # Deduplicate file ids deterministically.
    order = list(dict.fromkeys(u.split("#")[0] for u in urls))

    selected_qids = []
    queries: dict[str, str] = {}
    qrels_gold: dict[str, set[str]] = {}
    for i in range(len(df)):
        if len(selected_qids) >= max_queries:
            break
        doc = str(df.iloc[i]["func_documentation_string"] or "").strip()
        fid = urls[i].split("#")[0]
        if len(doc) < 20:
            continue
        qid = f"csn-{i}"
        selected_qids.append(i)
        queries[qid] = doc
        qrels_gold[qid] = {fid}

    repo_of = {f: "/".join(f.split("/")[:-1]) for f in set().union(*qrels_gold.values())}

    pool = [f for f in order if f not in set().union(*qrels_gold.values())]
    distract = [pool[i] for i in rng.sample(range(len(pool)),
                                           min(n_distractors, len(pool)))]

    docs = {}
    keep = set().union(*qrels_gold.values()) | set(distract)
    text_by_file: dict[str, list[str]] = {}
    for i in range(len(df)):
        fid = urls[i].split("#")[0]
        if fid in keep:
            text_by_file.setdefault(fid, []).append(df.iloc[i]["func_code_string"])
    for fid in keep:
        docs[fid] = "\n\n".join(text_by_file.get(fid, [""]))

    qrels = {}
    for qid, golds in qrels_gold.items():
        repo = repo_of[next(iter(golds))]
        gr = {f: 2 for f in golds}
        # rel 1: same-repo neighbours (vocab overlap, not the answer)
        gr.update({d: 1 for d in docs
                   if d not in golds and repo_of.get(d) == repo})
        qrels[qid] = gr
    return docs, queries, qrels
