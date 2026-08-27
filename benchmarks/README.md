# ffembed benchmarks

Four suites, one entrypoint (`python -m benchmarks.run <suite>`), all writing
JSON results with environment manifests under `benchmarks/results/`:

1. **Retrieval quality** (`run_retrieval.py`) — the headline. Graded BEIR-style
   evaluation: nDCG@10, MRR@10, Recall@100 across multiple systems and
   datasets.
2. **Agent token cost** (`agent_pi.py`, `agent_live.py`) — real agent
   sessions, paired design, confidence intervals and significance tests.
3. **Simulated agent policies** (`test_agent_tokens.py`,
   `test_agent_sim.py`-style) — deterministic, no LLM.
4. **Systems scaling** (`run_systems.py`) — latency percentiles, indexing
   throughput, memory and index size at 1k–20k-note scale.

Raw outputs (`results_*.md`, `results/*.json`, `episodes.jsonl`, `.data/`)
are generated; `.data/` holds downloaded datasets (gitignored).

## Results (2026-08-27 runs)

nDCG@10 / MRR@10 / Recall@100, pytrec_eval. Hardware: 24-core desktop CPU,
all models local ONNX. Selected systems; every run is reproducible with the
commands under each section below.

### Synthetic implicit notes (100 notes, frozen 43-topic bank)

| system | nDCG@10 | MRR@10 | Recall@100 | index |
|---|---|---|---|---|
| bm25 | .483 | .589 | .744 | 0.1 s |
| fts5 | .489 | .595 | .716 | 0.0 s |
| ffembed:minilm | .541 | **.745** | **1.00** | 1.5 s |
| ffembed:bge-small | .531 | .725 | 1.00 | 18.3 s |
| hybrid:bge-small | .534 | .718 | 1.00 | 17.6 s |

Query latency (mean): minilm 44 ms, bge-small 93 ms, hybrid 108 ms, bm25
0.08 ms. Prose never names its topic and filenames are neutral, so keyword
systems cannot shortcut; ffembed retrieves *every* relevant note and ranks
the answer first far more often (MRR .745 vs .59 — typically your first
result).

### SciFact (public BEIR set, 5,183 docs, first 40 queries)

| system | nDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| bm25 | .797 | .785 | .920 |
| fts5 | .800 | .793 | .945 |
| ffembed:minilm | .650 | .603 | .938 |

External validation: published BEIR leaderboards report ≈0.65 nDCG@10 for
MiniLM on SciFact; this harness reproduces it. Caveat: our bm25s
configuration is not Anserini-tuned, so its figure is not
leaderboard-comparable, and 40 queries leaves wide confidence intervals —
raise `--max-queries` for publishable numbers.

### Real-repo code (flask @3.0.3, commit-message queries, 82 .py files)

| system | nDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| bm25 | .568 | .950 | .706 |
| fts5 | .602 | 1.00 | .723 |
| ffembed:minilm | .274 | .567 | .705 |
| hybrid:minilm | .562 | .825 | **.840** |

Commit messages share vocabulary with code identifiers, so keyword search
wins precision here; dense recall lifts hybrid to the best overall. If you
index a repo, prefer `hybrid:` over pure `ffembed:`. (bge-small/minilm
native-code embedding models would likely close the dense gap — untested.)

### CodeSearchNet → file-level (python test split, sampled)

| system | nDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| bm25 | .627 | 1.00 | .515 |
| fts5 | .628 | 1.00 | .571 |

Same pattern as repo code: docstrings share vocabulary with the code they
describe, so sparse baselines are strong; dense models without code
pretraining lag.

## 1. Retrieval quality (BEIR-style)

Datasets load as `(docs, queries, qrels)` — graded relevance included — from
[`datasets_io.py`](datasets_io.py):

| dataset | what it is | provenance |
|---|---|---|
| `synthetic` | neutral-named notes built from the frozen 43-topic question bank (`question_bank_v2.BANK`, do-not-edit) with rel-2 main subjects and rel-1 filler asides | generated, seeds fixed |
| `scifact`, `nfcorpus`, `touche2020` | standard zero-shot retrieval sets | HF mirrors `BeIR/<name>` + `BeIR/<name>-qrels` |
| `repo` | a real repo clone (default: flask @3.0.3); queries are commit subjects, rel-2 = files touched by that commit, rel-1 = same-directory neighbours | derived from git log |
| `csn` | CodeSearchNet python test split restructured to file-level retrieval; query = docstring, rel-2 = containing file, rel-1 = same-repo neighbours, plus sampled distractor files | HF `code-search-net/code_search_net` |

Systems ([`retrievers.py`](retrievers.py)) behind one interface:

- `ffembed:<model>` — ffembed's real chunker + store + brute-force cosine
  (any alias from `ffembed models`; indexing uses ffembed's own DB layer in
  batched mode)
- `bm25` — bm25s Okapi BM25, the canonical sparse baseline
- `fts5` — SQLite FTS5 with stopword-filtered OR queries
- `hybrid:<model>` — reciprocal-rank fusion of dense + sparse

Scoring ([`qrels.py`](qrels.py)) uses `pytrec_eval` (the engine behind BEIR
and MTEB) for nDCG@10 and Recall@100, computing MRR@10 over top-10 runs. A
pure-Python fallback matches pytrec_eval semantics if unavailable.

```bash
uv run --group dev python -m benchmarks.run retrieval \
    --dataset synthetic --size 500 \
    --systems ffembed:bge-small bm25 fts5 hybrid:bge-small

# published-benchmark territory:
uv run --group dev python -m benchmarks.run retrieval \
    --dataset scifact --max-queries 100 --systems bm25 ffembed:bge-small

# code files, real repo:
uv run --group dev python -m benchmarks.run retrieval \
    --dataset repo --repo-ref 3.0.3

# code files, CodeSearchNet:
uv run --group dev python -m benchmarks.run retrieval \
    --dataset csn --max-queries 300
```

Each run also writes TREC-format run files to
`benchmarks/results/retrieval/run_<system>.trec` so results can be checked
with external tooling (trec_eval, ir_measures).

Methodology notes:

- The synthetic corpus prose never names its topic and filenames are neutral,
  so keyword systems cannot shortcut; question styles split into direct /
  symptom / vague phrasings per topic.
- `question_bank_v2.BANK_VERSION` must never change between compared runs;
  bump it only with a full rerun of every synthetic number you publish.
- Datasets download into `benchmarks/.data/` on first use (~30 MB for scifact).

## 2. pi agent harness (paired, budgeted)

For each task we spawn fresh `pi -p` sessions in a corpus of neutral-named,
implicitly-written notes; both arms run identical agents + model, one with an
extra `semantic_search` tool ([extension](../extensions/ffembed.ts)). Every
task runs in BOTH arms before any new task starts (paired design), grading is
deterministic (finished-on file contains the answer passage), and reporting
includes Wilson 95% CIs on success, bootstrap CIs on token/cost deltas, and a
two-sided paired permutation test.

Sample result (legacy single run, 100 notes, gpt-4o-mini — the paired suite
reports the same shape plus intervals):

| arm | success | p50 tok | p90 tok | cost/task | cost/solved |
|---|---|---|---|---|---|
| shell | 71% | 3,910 | 12,520 | $0.0053 | $0.0074 |
| + ffembed | 92% | 2,080 | 2,114 | $0.00054 | $0.00059 |

Cost control:

```bash
uv run --group dev python -m benchmarks.run agent-pi --bank v2 --per-style 4
```

- gpt-4o-mini at default settings: worst case ≈ $0.02/episode → ~$2 for a
  full 43-task × 2-arm pass. Print/save costs come from pi's own usage
  accounting, not estimates.
- `episodes.jsonl` records every finished episode tagged with its config;
  re-running the same command skips completed episodes, so interrupted runs
  only pay for missing pairs.

`--bank v1` reproduces the legacy 8-topic corpus; prefer v2 for anything you
publish.

## 3. Simulated agent policies

No API calls: three deterministic policies (`list_all`, `grep_loop`,
ffembed) over the implicit corpus count tokens ingested en route to the right
file (`~4 chars/token`). Run:

```bash
uv run --group dev pytest benchmarks/test_agent_tokens.py -q -s
```

## 4. Systems scaling

```bash
uv run --group dev python -m benchmarks.run systems --sizes 1000 5000 20000
```

Per scale: ffembed query p50/p95/p99 vs indexed BM25 and cold `grep`;
indexing docs/s; SQLite size on disk; peak RSS. Micro-latency variants live
in [`test_latency.py`](test_latency.py) via pytest-benchmark:

```bash
uv run --group dev python -m benchmarks.run latency
```

Daemon debounce correctness under write bursts is asserted (not just timed)
in [`test_debounce.py`](test_debounce.py): 40 rapid events on 10 files must
collapse into exactly one re-index per file after the quiet window.
## Reproducibility rules

- Every suite result is JSON (`schema`, `config`, `env`, `metrics`,
  `records`) written timestamped under `benchmarks/results/<suite>/`;
  markdown summaries render from those files (`render.py`).
- Isolated HOME: all suites redirect ffembed's state to temp dirs
  ([isolation.py](isolation.py)); nothing touches your real `~/.ffembed`
  (and thanks to repointing modules that re-export path constants, a running
  daemon can't corrupt benchmark numbers either).
- Model cache is shared read-only via symlink; no ONNX downloads during
  measured phases (set `HF_HUB_OFFLINE=1` to guarantee).
- Statistics helpers ([stats.py](stats.py)): percentile-bootstrap CIs,
  Wilson intervals, paired permutation tests — no hand-wavy "92% vs 71%"
  without intervals.

## Typical runtime

Synthetic retrieval suite: seconds–minutes locally (no network needed for
`synthetic`/`repo`; `.data` downloads cached). Each pi episode ≈ 5–30 s
wall time, parallelized. Systems suite scales linearly in corpus size.
