---
name: ffembed
description: Search a directory of local files (notes, docs, code comments) by meaning instead of grep, using ffembed's local semantic index. Use when the user asks to find, search, or recall something from their files by topic/meaning rather than exact text, or asks to set up semantic search over a directory.
---

# ffembed

A local-only semantic file index. No server, no cloud calls, no vector DB
to run — embeddings happen on-device via `fastembed`, storage is one
SQLite file under `~/.ffembed/`. Prefer this over `grep`/`ripgrep` when
the user's query is conceptual ("that thing about debounce", "notes on
sourdough") rather than an exact string.

If the `ffembed` command is not found, it isn't installed — tell the user
rather than trying to install it yourself.

## Quick start

```
ffembed watch ~/notes --filter "*.md"   # register + index once
ffembed start                            # background daemon, keeps it in sync
ffembed search "that thing about debounce"
```

## Workflow

1. **Check what's already indexed**: `ffembed list` shows watched
   directories, their glob filter, and file/chunk counts. `ffembed status`
   shows whether the daemon is running.
2. **If the target directory isn't watched yet**, register it:
   `ffembed watch <dir> --filter "<glob>"` (default filter is `*.md`).
   This indexes once immediately, synchronously — no need to wait for the
   daemon for a first answer.
3. **Search**: `ffembed search "<query>" [--dir <dir>] [-k N]`. Results
   are printed as `score  file_path` followed by a snippet, sorted best
   first. Read the file at the printed path for full context before
   answering the user — the snippet is truncated.
4. **Keep it fresh**: if the daemon isn't running (`ffembed status`) and
   the user will keep editing files, suggest `ffembed start` so future
   edits get picked up automatically (debounced, default 2s of quiet).
   A one-off search doesn't need the daemon; `watch` already indexed
   current content.
5. **New watch target while daemon is running**: `ffembed watch` on a new
   directory won't be picked up by an already-running daemon. Run
   `ffembed restart` after adding a target if the daemon is up.

## Other commands

- `ffembed reindex [dir]` — force a full reindex (e.g. after changing
  `--model` or `--filter` for a target you re-`watch`ed).
- `ffembed unwatch <dir>` — stop watching and drop its indexed data.
- `ffembed models` — list the embedding model garden (default
  `bge-small`); pass `--model <alias>` to `watch` to pick another.
- `ffembed stop` — stop the background daemon.

## Notes

- Search is brute-force cosine similarity — fine up to tens of thousands
  of chunks; don't worry about scale for a personal notes/docs directory.
- Only files matching the target's glob filter are indexed. If a search
  turns up nothing, check `ffembed list` — the directory may not be
  watched, or the filter may exclude the file type in question.
