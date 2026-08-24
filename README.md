![ffembed](banner.png)

no server, no cloud calls, no vector DB to run.

**~4.5× fewer tokens to get the right file into your agent's hand** than grep

A tiny local semantic index for your files. Point it at a directory, give it
a glob filter, and it keeps an embedding index in sync as files change.

Your agents will thank you and so will your wallet. Teach them to use it
with the [ffembed skill](.claude/skills/ffembed/SKILL.md) ... drop it in
your project's `.claude/skills/` and any Claude Code session will know
how to drive it.

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

Live result (50 notes, 8 tasks, gpt-4o-mini, temperature 0):

| policy | success | median total tokens | median tool calls |
|---|---|---|---|
| list + read | 100% | 7,234 | 52 |
| grep + read | 100% | 7,679 | 53 |
| ffembed (ours) | 100% | **1,690** | **3** |

Grep agents read dozens of notes to verify keyword hits; the semantic agent's first call returns ranked snippets, so it resolves in a fraction of the tokens, and its cost stays flat as the corpus grows while keyword agents read more.
