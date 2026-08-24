# ffembed benchmarks

Three suites:

1. **pi coding-agent harness** (`agent_pi.py`) — the headline. Real agent
   sessions, real tokens and cost from pi's usage accounting.
2. **Agent token cost, simulated** (`test_agent_tokens.py`) — deterministic
   policies, no LLM needed.
3. **Raw latency** (`test_latency.py`) — query latency of pre-indexed
   semantic search vs `grep -rl`. Included for completeness; latency is not
   the point.

Result files (`results_*`, `episodes.jsonl`) are generated output and
gitignored — regenerable with the commands below.

## pi harness

For each task we spawn a fresh `pi -p --mode json` session inside a corpus of
neutral-named notes whose prose *never names its topic* (the realistic hard
case for keyword search). Both arms are the same agent + model; one gets
plain shell tools, the other an extra `semantic_search` tool registered by
[`ffembed_extension.ts`](ffembed_extension.ts).

Grading is deterministic: the file the agent finishes on must contain the
answer passage. Tokens and cost come from pi's per-message `usage` — no
estimation. Every episode is appended to `episodes.jsonl` for audit.

```bash
uv run --group dev python -m benchmarks.agent_pi --size 100                  # 24 tasks x 2 arms
uv run --group dev python -m benchmarks.agent_pi --size 100 --arms ffembed   # one arm
```

Flags: `--per-style N` caps topics per question style (direct / symptom /
vague), `--fillers N` sets off-topic passages per note (note length),
`--model` picks any model the OpenAI provider serves.

Sample result (100 notes, 24 tasks, gpt-4o-mini):

| arm | success | finished | median tok | p90 tok | cost/task |
|---|---|---|---|---|---|
| shell tools | 71% | 71% | 3,910 | 12,520 | $0.0053 |
| + ffembed | **92%** | 100% | 2,080 | **2,114** | **$0.00054** |

- The semantic arm's spend is flat (p90 ≈ median): ranked snippets arrive in
  the first tool call. Shell agents' cost is task-dependent (p90 ≈ 6× their
  median) because they read notes to verify keyword hits.
- Every shell-arm failure was format non-compliance (no verdict emitted);
  every verdict a shell agent did emit was correct. The ffembed arm always
  finished and picked a wrong file once.

There is also `test_agent_live.py`, an older raw-API tool-use loop with the
same grading idea; it needs only `OPENAI_API_KEY` and no pi install.

## Simulated agent token cost

No LLM is called. Three deterministic agent policies are simulated over a
corpus of neutral-named notes whose prose *never names its topic* (implicit
descriptions — the realistic hard case for keyword tools), and we count what
each policy ingests on the way to a file that actually answers the question:

- `list_all` — `ls`, then read files until one fits.
- `grep_loop` — try synonym keywords with `grep -rl`, read every hit.
  Wrong guesses still cost a round trip.
- `ffembed` — one semantic query, inspect top-ranked snippets.

Token counts use ~4 chars/token; each tool round trip carries a fixed 30-token
framing overhead.

```bash
uv run --group dev pytest benchmarks/test_agent_tokens.py -q -s
```

Writes `benchmarks/results_agents.md`.

## Raw latency

```bash
uv run --group dev pytest benchmarks/test_latency.py --benchmark-only
# optional JSON + markdown rendering:
uv run --group dev pytest benchmarks/test_latency.py --benchmark-only \
    --benchmark-json=benchmarks/results.json
uv run --group dev python benchmarks/render.py benchmarks/results.json
```

Typical runtime: simulated + latency suites ~2 minutes; each pi episode adds
roughly 5–30 s of wall time (parallelized across workers).

## Notes

- All benchmarks use an isolated temporary `~/.ffembed` directory (the pi
  harness runs everything under a fake `$HOME`), so they do not pollute your
  normal index or daemon.
- The embedding model (`minilm`) is warmed up once so reported times measure
  search, not model loading.
