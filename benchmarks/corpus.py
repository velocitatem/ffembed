"""Helpers to build synthetic text corpora for benchmarking."""

from __future__ import annotations

import random
from pathlib import Path


TOPICS = [
    "debounce",
    "asyncio",
    "vector database",
    "embedding model",
    "filesystem watcher",
    "chunking strategy",
    "cosine similarity",
    "sqlite index",
    "background daemon",
    "huggingface token",
]


def _paragraph(topic: str, idx: int) -> str:
    """Generate a deterministic paragraph about a topic."""
    return (
        f"## {topic} notes ({idx})\n\n"
        f"This document discusses {topic}. "
        f"When working with {topic}, it is important to consider latency, "
        f"throughput, and correctness. {topic.capitalize()} is often used in "
        f"systems that need to react quickly to changing inputs. "
        f"Developers should measure {topic} carefully before deploying to production.\n"
    )


def generate_corpus(root: Path, files: int, paragraphs_per_file: int = 5) -> list[Path]:
    """Create a directory of markdown files with injected topics.

    Returns the list of created file paths.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    created: list[Path] = []
    for i in range(files):
        topic = rng.choice(TOPICS)
        path = root / f"doc_{i:05d}_{topic.replace(' ', '_')}.md"
        paragraphs = [_paragraph(topic, j) for j in range(paragraphs_per_file)]
        # Sprinkle a few unrelated topics to make search less trivial.
        for _ in range(rng.randint(0, 2)):
            paragraphs.append(_paragraph(rng.choice(TOPICS), -1))
        path.write_text("\n".join(paragraphs), encoding="utf-8")
        created.append(path)
    return created


# --- implicit corpus ------------------------------------------------------
#
# The realistic hard case for keyword search: notes describe concepts without
# ever using their name ("wait until things go quiet" instead of "debounce"),
# and filenames are neutral (note_0007.md). An agent cannot win by grepping
# for the obvious word; it must either guess synonyms or understand meaning.

IMPLICIT_PASSAGES = {
    "debounce": (
        "When events arrive in a flurry, do nothing immediately. Wait for a "
        "stretch of quiet before acting once, so a burst of saves triggers a "
        "single pass instead of many."
    ),
    "asyncio": (
        "Network calls should not freeze everything else. Hand them to the "
        "scheduler and let other work continue while the socket finishes."
    ),
    "vector database": (
        "Keep numeric summaries of what each document means. Similar items "
        "end up close together, so finding related notes is just a distance "
        "scan over stored points."
    ),
    "filesystem watcher": (
        "Ask the operating system to tell us when a file changes instead of "
        "polling every second; it wakes us only when something happened."
    ),
    "chunking strategy": (
        "Long documents are split at paragraph boundaries with a little "
        "overlap carried across cuts, so no idea loses its surrounding "
        "context."
    ),
    "cosine similarity": (
        "To compare two meaning vectors, divide the dot product by the "
        "product of the lengths; this scores angle, not magnitude."
    ),
    "sqlite index": (
        "Everything persists in one embedded database file on disk: plain "
        "tables, blobs for the heavy payloads, no server process."
    ),
    "background daemon": (
        "A helper process detaches from the terminal and keeps running, "
        "holding state between commands and reacting to events as they come."
    ),
}

# --- question bank ---------------------------------------------------------
#
# Three phrasings per topic so the harness measures robustness to how the
# question is asked, not just whether one sentence embeds well:
#
#   direct  — plainly describes the mechanism
#   symptom — describes the problem the mechanism solves
#   vague   — half-remembered, low-information phrasing

QUESTION_STYLES = ("direct", "symptom", "vague")

QUESTION_BANK: dict[str, dict[str, str]] = {
    "debounce": {
        "direct": "which note describes waiting for things to go quiet before acting on grouped events?",
        "symptom": "we were doing expensive work over and over when many changes landed together — find where we wrote the fix",
        "vague": "there was something about bursts of activity, where was that?",
    },
    "asyncio": {
        "direct": "which note covers handing slow network calls to a scheduler so other work continues?",
        "symptom": "find the note about the whole program freezing whenever one request hangs",
        "vague": "didn't we write something about things waiting on each other?",
    },
    "vector database": {
        "direct": "which note explains keeping numeric summaries of documents so related ones sit close together?",
        "symptom": "we needed to find similar notes without reading them all — where did we sketch that?",
        "vague": "something about points and distance, which note was it?",
    },
    "filesystem watcher": {
        "direct": "which note covers being told about file changes instead of checking in a loop?",
        "symptom": "polling every second was wasteful — find where we wrote the alternative",
        "vague": "the note about reacting when files change, where is it?",
    },
    "chunking strategy": {
        "direct": "which note describes cutting long text into pieces while carrying overlap across cuts?",
        "symptom": "ideas were losing context when documents got split — find where we solved that",
        "vague": "something about splitting documents sensibly, where was it?",
    },
    "cosine similarity": {
        "direct": "which note shows comparing vectors by angle, dividing the dot product by the lengths?",
        "symptom": "we needed a score that ignores vector magnitude — find where we worked it out",
        "vague": "the maths for how alike two embeddings are, which note?",
    },
    "sqlite index": {
        "direct": "which note says state lives in a single embedded database file with blobs for payloads?",
        "symptom": "we wanted persistence with zero server administration — where did we decide that?",
        "vague": "the note about where everything is stored, where is it?",
    },
    "background daemon": {
        "direct": "which note covers a detached helper process holding state between commands?",
        "symptom": "commands kept losing state when the terminal closed — find where we fixed that",
        "vague": "something about the helper that keeps running, which note?",
    },
}


def build_tasks(limit_per_style: int | None = None, seed: int = 0) -> list[dict]:
    """Flatten the question bank into tasks, optionally capped per style."""
    rng = random.Random(seed)
    topics = sorted(QUESTION_BANK)
    tasks = []
    for style in QUESTION_STYLES:
        entries = [(t, QUESTION_BANK[t][style]) for t in topics]
        if limit_per_style is not None:
            entries = rng.sample(entries, k=min(limit_per_style, len(entries)))
        for topic, question in entries:
            tasks.append({"topic": topic, "question": question, "style": style})
    return tasks



# Natural-language questions an agent might be asked, with the keywords a
# literal-minded grep agent would plausibly try first (often absent from the
# text because the corpus is written implicitly). Used by the simulated
# policies; the pi harness draws from QUESTION_BANK instead.
TASKS = [
    {"topic": "debounce",
     "question": "where did we write down how to stop rapid-fire saves causing repeated work?",
     "keywords": ["debounce", "burst", "saves"]},
    {"topic": "asyncio",
     "question": "which note talks about network requests blocking the rest of the program?",
     "keywords": ["blocking", "network", "async"]},
    {"topic": "vector database",
     "question": "where is the bit about storing summaries of documents so similar ones sit near each other?",
     "keywords": ["vector", "embedding store", "similarity"]},
    {"topic": "filesystem watcher",
     "question": "did we ever note how to hear about file changes without polling?",
     "keywords": ["watcher", "inotify", "polling"]},
    {"topic": "chunking strategy",
     "question": "which note covers splitting long documents without losing context at the cut?",
     "keywords": ["chunking", "splitting documents", "overlap"]},
    {"topic": "cosine similarity",
     "question": "where did we explain scoring two vectors by angle rather than size?",
     "keywords": ["cosine", "angle", "dot product"]},
    {"topic": "sqlite index",
     "question": "which note says where all the data lives when there is no server?",
     "keywords": ["sqlite", "database file", "storage"]},
    {"topic": "background daemon",
     "question": "did we note how the helper keeps running after the terminal closes?",
     "keywords": ["daemon", "background process", "detach"]},
]


def generate_implicit_corpus(root: Path, files: int, *, fillers_per_note: int = 2,
                             seed: int = 7) -> tuple[list[Path], dict[str, list[str]]]:
    """Neutral-named files whose prose never names its dominant topic.

    Each note carries one main passage plus ``fillers_per_note`` off-topic
    passages, so notes grow (and full-file reads get expensive) as fillers
    increases.

    Returns (paths, mentions) where mentions maps topic -> paths containing
    that topic's passage (as main subject or as an aside). A file counts as
    "the right file" for a task if it contains the relevant passage.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    topics = list(IMPLICIT_PASSAGES)
    created: list[Path] = []
    mentions: dict[str, list[str]] = {}
    for i in range(files):
        main = topics[i % len(topics)]
        others = [t for t in topics if t != main]
        filler = rng.sample(others, k=min(fillers_per_note, len(others)))
        passages = [IMPLICIT_PASSAGES[main]] + [IMPLICIT_PASSAGES[t] for t in filler]
        rng.shuffle(passages)
        body = "\n\n".join(f"- {p}" for p in passages)
        path = root / f"note_{i:04d}.md"
        path.write_text(body, encoding="utf-8")
        created.append(path)
        for t in [main] + filler:
            mentions.setdefault(t, []).append(str(path))
    return created, mentions
