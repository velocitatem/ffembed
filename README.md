# ffembed

A tiny local semantic index for your files. Point it at a directory, give it
a glob filter, and it keeps an embedding index in sync as files change —
no server, no cloud calls, no vector DB to run.

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
- The daemon watches registered directories with `watchdog` and debounces
  bursts of filesystem events (default 2s of quiet) before re-indexing a
  file, so a flurry of editor saves triggers one re-embed, not several.
- Search embeds the query and does a brute-force cosine scan over stored
  chunks — simple, and plenty fast at personal-library scale.

## Commands

| command | what it does |
|---|---|
| `ffembed watch DIR [--filter GLOB] [--model NAME]` | register + index a directory |
| `ffembed unwatch DIR` | stop watching and drop its data |
| `ffembed list` | show watched directories and stats |
| `ffembed reindex [DIR]` | force a full reindex |
| `ffembed search QUERY [--dir DIR] [-k N]` | semantic search |
| `ffembed start [--debounce SECS]` | start the background daemon |
| `ffembed stop` / `restart` | stop / restart the daemon |
| `ffembed status` | daemon + index status |
| `ffembed models` | list available embedding models |

Adding a new `watch` target while the daemon is running requires a
`ffembed restart` to pick it up.
