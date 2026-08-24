![ffembed](banner.png)

no server, no cloud calls, no vector DB to run.

**~10× cheaper per find, and finds more often**: real pi coding-agent sessions
resolve tasks at $0.00054 vs $0.0053 per task with 92% vs 71% success
(see [benchmarks](benchmarks/README.md)).

A tiny local semantic index for your files. Point it at a directory, give it
a glob filter, and it keeps an embedding index in sync as files change.

Your agents will thank you and so will your wallet. Teach them to use it
with the [ffembed skill](.claude/skills/ffembed/SKILL.md) ... drop it in
your project's `.claude/skills/` and any Claude Code session will know
how to drive it.

## Use with pi

One command gives every [pi](https://github.com/badlogic/pi-mono) session a
`semantic_search` tool backed by your ffembed index:

```bash
pi install git:github.com/velocitatem/ffembed
```

Then, with a directory indexed (`ffembed watch <dir>`), the agent can search
by meaning out of the box — this is exactly what the [benchmark
harness](benchmarks/) measures: agents resolve find-the-note tasks ~10×
cheaper with the tool than with shell alone. The repo also ships an
[ffembed skill](.claude/skills/ffembed/SKILL.md) that pi discovers on
install, so it knows when to reach for it.

## Install

```
# text-only (small, no torch)
uv tool install git+https://github.com/velocitatem/ffembed

# with image embedding support via DINOv3
uv tool install 'git+https://github.com/velocitatem/ffembed[image]'
```

```
ffembed watch ~/notes --filter "*.md"   # index once, register for watching
ffembed start                           # background daemon, debounced
ffembed search "that thing about debounce"
ffembed stop
```

## How it works

- Everything lives under `~/.ffembed/`: one SQLite database (`db.sqlite`),
  the daemon's pidfile and log, and a cache of downloaded models.
- Embeddings run locally via [fastembed](https://github.com/qdrant/fastembed)
  (ONNX runtime, no torch). Pick a model with `--model`, or see the whole
  garden with `ffembed models` (default: `bge-small`).
- Files are read, chunked on paragraph/sentence boundaries (~1800 chars,
  200 char overlap), embedded, and stored as chunk rows.
- Image files are embedded whole with a local DINOv3 model (requires the
  `image` extra).
- The daemon watches registered directories with `watchdog` and debounces
  bursts of filesystem events (default 2s of quiet) before re-indexing a
  file, so a flurry of editor saves triggers one re-embed, not several.
- Search embeds the query and does a brute-force cosine scan over stored
  chunks — simple, and plenty fast at personal-library scale. Text queries
  search text chunks; image paths search image chunks.

## Commands

| command | what it does |
|---|---|
| `ffembed watch DIR [--filter GLOB] [--model NAME] [--vision-model NAME]` | register + index a directory |
| `ffembed unwatch DIR` | stop watching and drop its data |
| `ffembed list` | show watched directories and stats |
| `ffembed reindex [DIR]` | force a full reindex |
| `ffembed search QUERY [--dir DIR] [-k N]` | semantic search (text or image path) |
| `ffembed start [--debounce SECS]` | start the background daemon |
| `ffembed stop` / `restart` | stop / restart the daemon |
| `ffembed status` | daemon + index status |
| `ffembed models` | list available text and vision embedding models |

Adding a new `watch` target while the daemon is running requires a
`ffembed restart` to pick it up.

## Benchmarks

Real coding-agent harness: for each of 24 natural-language tasks we spawn a
fresh [pi](https://github.com/badlogic/pi-mono) session in a corpus of
100 neutral-named notes whose prose never names its topic. Same agent, same
model (gpt-4o-mini); one arm gets plain shell tools, the other gets an extra
`semantic_search` tool backed by ffembed.

| arm | success | median tokens | p90 tokens | cost/task |
|---|---|---|---|---|
| shell tools | 71% | 3,910 | 12,520 | $0.0053 
| + ffembed search | **92%** | 2,080 | **2,114** | **$0.00054** |

The shell agents' cost is wildly task-dependent (p90 nearly 6× the median);
the semantic agent's first call returns ranked snippets, so its spend is flat
regardless of question style or corpus size — and it finishes more tasks at
about a tenth of the cost.

Run it yourself:

```bash
uv run --group dev python -m benchmarks.agent_pi --size 100
```

See [benchmarks/README.md](benchmarks/README.md) for methodology, plus a
simulated no-API variant and raw latency benchmarks.
