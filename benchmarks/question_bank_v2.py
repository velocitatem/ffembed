"""Frozen expansion bank for the synthetic-notes retrieval benchmark.

One implicit passage plus three question styles per topic, in the same spirit
as ``corpus.IMPLICIT_PASSAGES`` but 5× wider so no run can memorise eight
examples:

    direct  — plainly describes the mechanism
    symptom — describes the problem the mechanism solves
    vague   — half-remembered, low-information phrasing

*** DO NOT EDIT between benchmark runs. ***
Every published number is only comparable within a fixed bank revision. Bump
``BANK_VERSION`` whenever anything here changes.

Filenames never name their topic and passages never use the obvious keyword,
so keyword tools cannot win by matching on the topic name.
"""

from __future__ import annotations

BANK_VERSION = "v2-2026-08-27"

BANK: dict[str, dict[str, str]] = {
    "debounce": {
        "passage": (
            "When events arrive in a flurry, do nothing immediately. Wait for a "
            "stretch of quiet before acting once, so a burst of saves triggers a "
            "single pass instead of many."
        ),
        "questions": {
            "direct": "which note describes waiting for things to go quiet before acting on grouped events?",
            "symptom": "we were doing expensive work over and over when many changes landed together — find where we wrote the fix",
            "vague": "there was something about bursts of activity, where was that?",
        },
    },
    "asyncio": {
        "passage": (
            "Network calls should not freeze everything else. Hand them to the "
            "scheduler and let other work continue while the socket finishes."
        ),
        "questions": {
            "direct": "which note covers handing slow network calls to a scheduler so other work continues?",
            "symptom": "find the note about the whole program freezing whenever one request hangs",
            "vague": "didn't we write something about things waiting on each other?",
        },
    },
    "vector database": {
        "passage": (
            "Keep numeric summaries of what each document means. Similar items "
            "end up close together, so finding related notes is just a distance "
            "scan over stored points."
        ),
        "questions": {
            "direct": "which note explains keeping numeric summaries of documents so related ones sit close together?",
            "symptom": "we needed to find similar notes without reading them all — where did we sketch that?",
            "vague": "something about points and distance, which note was it?",
        },
    },
    "filesystem watcher": {
        "passage": (
            "Ask the operating system to tell us when a file changes instead of "
            "polling every second; it wakes us only when something happened."
        ),
        "questions": {
            "direct": "which note covers being told about file changes instead of checking in a loop?",
            "symptom": "polling every second was wasteful — find where we wrote the alternative",
            "vague": "the note about reacting when files change, where is it?",
        },
    },
    "chunking strategy": {
        "passage": (
            "Long documents are split at paragraph boundaries with a little "
            "overlap carried across cuts, so no idea loses its surrounding "
            "context."
        ),
        "questions": {
            "direct": "which note describes cutting long text into pieces while carrying overlap across cuts?",
            "symptom": "ideas were losing context when documents got split — find where we solved that",
            "vague": "something about splitting documents sensibly, where was it?",
        },
    },
    "cosine similarity": {
        "passage": (
            "To compare two meaning vectors, divide the dot product by the "
            "product of the lengths; this scores angle, not magnitude."
        ),
        "questions": {
            "direct": "which note shows comparing vectors by angle, dividing the dot product by the lengths?",
            "symptom": "we needed a score that ignores vector magnitude — find where we worked it out",
            "vague": "the maths for how alike two embeddings are, which note?",
        },
    },
    "sqlite index": {
        "passage": (
            "Everything persists in one embedded database file on disk: plain "
            "tables, blobs for the heavy payloads, no server process."
        ),
        "questions": {
            "direct": "which note says state lives in a single embedded database file with blobs for payloads?",
            "symptom": "we wanted persistence with zero server administration — where did we decide that?",
            "vague": "the note about where everything is stored, where is it?",
        },
    },
    "background daemon": {
        "passage": (
            "A helper process detaches from the terminal and keeps running, "
            "holding state between commands and reacting to events as they come."
        ),
        "questions": {
            "direct": "which note covers a detached helper process holding state between commands?",
            "symptom": "commands kept losing state when the terminal closed — find where we fixed that",
            "vague": "something about the helper that keeps running, which note?",
        },
    },
    "git rebase": {
        "passage": (
            "Before opening a pull request, replay each of your commits onto the "
            "tip of main one at a time and drop any that turned out to be noise, "
            "so history reads like it was written in one careful pass."
        ),
        "questions": {
            "direct": "which note describes replaying commits onto the tip of main to clean up history?",
            "symptom": "reviewers complained our branch history looked like spaghetti — find where we wrote down the cleanup",
            "vague": "something about tidying up old commits before review, which note?",
        },
    },
    "merge conflicts": {
        "passage": (
            "When two branches change the same lines, git pauses and asks you to "
            "pick a side or blend both edits by hand before continuing."
        ),
        "questions": {
            "direct": "which note covers what happens when two branches edit the same lines?",
            "symptom": "a teammate hit that pause mid-pull and lost work — find where we wrote the recovery steps",
            "vague": "the bit about picking sides when edits collide, where is it?",
        },
    },
    "containers": {
        "passage": (
            "Ship the app with its own minimal filesystem and dependencies "
            "bundled in a sealed image, so what runs here also runs there."
        ),
        "questions": {
            "direct": "which note describes bundling an app into a sealed image with its own filesystem?",
            "symptom": "'works on my machine' struck again — find where we wrote how we fixed it",
            "vague": "that thing about shipping the whole environment along, which note?",
        },
    },
    "environment configuration": {
        "passage": (
            "Values that differ per deployment live outside the code, injected "
            "into the process at startup rather than baked into source files."
        ),
        "questions": {
            "direct": "which note says deployment-specific values get injected at startup instead of living in source?",
            "symptom": "a staging password leaked through a commit — find where we wrote the rule that prevents that",
            "vague": "something about settings not belonging in the repo, where was it?",
        },
    },
    "secrets management": {
        "passage": (
            "Credentials never go in version control. They ride along from a "
            "vault or local-only file at launch time, and rotate on a schedule."
        ),
        "questions": {
            "direct": "which note covers keeping credentials out of git and loading them from a vault instead?",
            "symptom": "someone pasted an API key into chat — find where we wrote the policy after that",
            "vague": "the notes about where passwords should live, which note?",
        },
    },
    "ssh key auth": {
        "passage": (
            "Instead of typing passwords for remote machines, prove identity "
            "with a keypair: the public half sits on the server, the private "
            "half never leaves your laptop."
        ),
        "questions": {
            "direct": "which note describes proving server identity with a keypair whose private half stays local?",
            "symptom": "password prompts were blocked in scripts — find where we wrote the workaround",
            "vague": "something about not leaving your laptop half anywhere, which note?",
        },
    },
    "dns resolution": {
        "passage": (
            "Names are resolved top-down: ask the resolver, walk root "
            "nameservers toward the authoritative answer, then cache what you "
            "learned for a while."
        ),
        "questions": {
            "direct": "which note walks the path from resolver to authoritative nameserver for a name lookup?",
            "symptom": "a domain move didn't propagate for hours — find where we wrote why",
            "vague": "the note about names turning into addresses, where is it?",
        },
    },
    "http caching": {
        "passage": (
            "Let responses carry freshness stamps so repeat requests skip the "
            "network entirely, and revalidate cheaply once the stamp expires."
        ),
        "questions": {
            "direct": "which note covers response freshness stamps and cheap revalidation?",
            "symptom": "our API bill spiked because every client refetched everything — find where we wrote the fix",
            "vague": "something about not hitting the network again for unchanged data, which note?",
        },
    },
    "api pagination": {
        "passage": (
            "Never return everything at once. Hand back one page keyed by a "
            "cursor pointing at where you stopped, so deep listings stay cheap "
            "on the server."
        ),
        "questions": {
            "direct": "which note describes returning listings page by page using a cursor?",
            "symptom": "one endpoint dumped ten thousand rows on every call — find where we designed the fix",
            "vague": "the note about a pointer saying where you left off, where is it?",
        },
    },
    "jwt auth": {
        "passage": (
            "Sign a small claim object and hand it to the client; the server "
            "just checks the signature on the way back in, no session store "
            "needed."
        ),
        "questions": {
            "direct": "which note explains handing clients signed claims so the server stores nothing?",
            "symptom": "sessions vanished whenever we restarted the backend — find where we moved away from storing them",
            "vague": "something about carrying proof of who you are in the token itself, which note?",
        },
    },
    "oauth flows": {
        "passage": (
            "Delegate sign-in to a provider: redirect the user there, receive "
            "an authorization code back, trade it for tokens server-side."
        ),
        "questions": {
            "direct": "which note covers redirecting users to a provider and trading the returned code for tokens?",
            "symptom": "we kept getting asked to build login from scratch — find where we wrote the delegated approach",
            "vague": "that bit about bouncing off to someone else's login page, where is it?",
        },
    },
    "rate limiting": {
        "passage": (
            "Track how many requests each client makes in a sliding window and "
            "reject the ones over budget, so one noisy consumer can't starve "
            "everyone else."
        ),
        "questions": {
            "direct": "which note describes rejecting requests above a per-client budget measured over a window?",
            "symptom": "one customer's runaway script brought the service down — find where we wrote the guardrail",
            "vague": "the note about capping how often people can knock on the door, where is it?",
        },
    },
    "idempotency keys": {
        "passage": (
            "Clients attach a unique marker to each operation; replays of the "
            "same marker return the original result instead of doing the work "
            "twice."
        ),
        "questions": {
            "direct": "which note says repeated operations with the same marker return the first result?",
            "symptom": "double-clicked checkout buttons charged cards twice — find where we wrote the protection",
            "vague": "something about making repeats safe, which note?",
        },
    },
    "retry backoff": {
        "passage": (
            "On transient failures, wait a little longer before each new "
            "attempt — doubling gaps with some jitter — instead of hammering a "
            "struggling service."
        ),
        "questions": {
            "direct": "which note covers waiting progressively longer between reattempts with jitter added?",
            "symptom": "our retries were pile-ons during an outage — find where we wrote the spacing rule",
            "vague": "the note about trying again but slower each time, where is it?",
        },
    },
    "circuit breaker": {
        "passage": (
            "After enough consecutive failures, stop calling the sick "
            "dependency altogether for a while and fail fast locally; probe it "
            "occasionally to see if it recovered."
        ),
        "questions": {
            "direct": "which note describes tripping a switch that stops calls to a failing dependency until it recovers?",
            "symptom": "timeouts cascaded because we kept calling a dead upstream — find where we designed the stopper",
            "vague": "something about cutting ties for a bit then probing, which note?",
        },
    },
    "message queues": {
        "passage": (
            "Put work onto a durable line and let separate workers drain it at "
            "their own pace, so spikes buffer instead of dropping tasks."
        ),
        "questions": {
            "direct": "which note covers draining buffered work with dedicated consumers at their own pace?",
            "symptom": "tasks vanished during traffic spikes — find where we wrote the buffering design",
            "vague": "that thing about a line of jobs waiting its turn, where is it?",
        },
    },
    "database migrations": {
        "passage": (
            "Schema changes ship as ordered, reversible steps applied by a "
            "tool, so every environment reaches the same shape without anyone "
            "typing SQL by hand."
        ),
        "questions": {
            "direct": "which note describes applying ordered, reversible schema steps via a tool everywhere?",
            "symptom": "staging and prod had drifted into different table shapes — find where we wrote the discipline",
            "vague": "the note about changing table shapes safely across environments, which note?",
        },
    },
    "query tuning": {
        "passage": (
            "Read the planner's chosen execution plan first; most slow reads "
            "are full scans of tables that needed a sorted lookup structure "
            "built ahead of time."
        ),
        "questions": {
            "direct": "which note says inspect the execution plan before touching slow queries?",
            "symptom": "a dashboard took thirty seconds to load — find where we diagnosed it",
            "vague": "something about why lookups crawled, where was it?",
        },
    },
    "n+1 queries": {
        "passage": (
            "Looping over parent rows and fetching each child separately turns "
            "one listing into hundreds of round trips; batch the children in a "
            "second grouped fetch instead."
        ),
        "questions": {
            "direct": "which note explains batching child fetches to avoid one round trip per row?",
            "symptom": "a list page fired four hundred tiny selects — find where we wrote the diagnosis",
            "vague": "the bit about fetching everything one item at a time being wrong, where is it?",
        },
    },
    "transaction isolation": {
        "passage": (
            "Concurrent writers stepping on each other's rows caused phantom "
            "reads; wrap the read-modify-write in one atomic unit so nobody "
            "sees half-finished state."
        ),
        "questions": {
            "direct": "which note covers wrapping read-modify-write sequences so partial state is never visible?",
            "symptom": "balances went negative under concurrent writes — find where we wrote the boundary rules",
            "vague": "something about nobody seeing half-done changes, which note?",
        },
    },
    "connection pooling": {
        "passage": (
            "Opening a fresh database socket per request wastes milliseconds "
            "and exhausts ports; keep a warm pool of connections and lend them "
            "out briefly."
        ),
        "questions": {
            "direct": "which note describes lending out warm database sockets instead of opening fresh ones?",
            "symptom": "the service exhausted ephemeral ports under load — find where we wrote the remedy",
            "vague": "the note about reusing already-open wires to the database, where is it?",
        },
    },
    "memoization": {
        "passage": (
            "Pure functions recomputing the same inputs over and over should "
            "remember previous answers in a bounded map keyed by the arguments."
        ),
        "questions": {
            "direct": "which note says cache prior answers of pure functions keyed by their arguments?",
            "symptom": "rendering revalidated the same expensive computation thousands of times — find where we fixed it",
            "vague": "something about the function remembering what it told you last time, which note?",
        },
    },
    "structured logging": {
        "passage": (
            "Emit log events as key-value records, not freeform sentences, so "
            "they can be filtered and joined later; include a correlation id on "
            "every entry."
        ),
        "questions": {
            "direct": "which note prescribes machine-readable log records joined by a shared correlation id?",
            "symptom": "debugging meant grepping prose for hours — find where we changed how events are emitted",
            "vague": "that decision about log lines having fields, where is it?",
        },
    },
    "distributed tracing": {
        "passage": (
            "Follow one request's journey across services as a single trace: "
            "each hop records timing and tags so you can see where latency "
            "accumulates."
        ),
        "questions": {
            "direct": "which note covers stitching one request's journey across many services into a single view?",
            "symptom": "p99 latency grew but no single service owned it — find where we wrote how to attribute it",
            "vague": "the note about following a request around like a passport stamp, which note?",
        },
    },
    "health checks": {
        "passage": (
            "Expose liveness and readiness endpoints so the orchestrator knows "
            "whether to restart the process or merely keep sending it traffic."
        ),
        "questions": {
            "direct": "which note distinguishes endpoints that say restart-me from those that say send-traffic?",
            "symptom": "the orchestrator restarted instances mid-deploy and dropped requests — find where we tuned the signals",
            "vague": "something about the orchestrator knowing if we're okay yet, where is it?",
        },
    },
    "test isolation": {
        "passage": (
            "Each test gets a freshly constructed world — temp directories, "
            "seeded databases, stubbed clocks — so order never matters and "
            "failures reproduce alone."
        ),
        "questions": {
            "direct": "which note says every test builds its own fresh world so ordering never matters?",
            "symptom": "the suite passed solo but failed in CI — find where we wrote down why",
            "vague": "the bit about tests stepping on each other's leftovers, which note?",
        },
    },
    "flaky tests": {
        "passage": (
            "Quarantine intermittently red tests rather than rerunning them "
            "forever; hunt the nondeterminism down with seeds recorded from "
            "each failure."
        ),
        "questions": {
            "direct": "which note recommends quarantining intermittent failures and capturing seeds to hunt them?",
            "symptom": "CI reruns hid real breakage behind retries — find where we wrote the policy",
            "vague": "something about the test that fails only sometimes, where is it?",
        },
    },
    "memory profiling": {
        "passage": (
            "Resident size crept upward with every processed batch. Snapshot "
            "allocations over time and compare heap dumps before chasing the "
            "retainer graph by hand."
        ),
        "questions": {
            "direct": "which note compares allocation snapshots to find what holds objects forever?",
            "symptom": "the worker OOM-killed after six hours — find where we investigated",
            "vague": "the note about RAM only ever going up, which note?",
        },
    },
    "unicode handling": {
        "passage": (
            "Normalize all incoming text to one canonical form at the boundary "
            "and decode bytes explicitly; naive byte slicing mangles multi-byte "
            "characters."
        ),
        "questions": {
            "direct": "which note prescribes normalizing text to one canonical form at the boundary?",
            "symptom": "usernames with accents broke search — find where we wrote the encoding rules",
            "vague": "that thing about accented letters behaving weirdly, where is it?",
        },
    },
    "timezone handling": {
        "passage": (
            "Store instants as UTC, attach regions only at the edges for "
            "display, and never do arithmetic on wall-clock times directly."
        ),
        "questions": {
            "direct": "which note says store instants as UTC and convert to local zones only for display?",
            "symptom": "reminders fired an hour early after daylight saving shifted — find where we wrote the rule",
            "vague": "the note about clock arithmetic going wrong twice a year, which note?",
        },
    },
    "cron scheduling": {
        "passage": (
            "Recurring jobs register in one declarative table of schedules and "
            "entrypoints, guarded against overlapping runs by advisory locks."
        ),
        "questions": {
            "direct": "which note describes a declarative table of recurring job schedules guarded against overlap?",
            "symptom": "two nightly copies of the same job ran at once — find where we wrote the fix",
            "vague": "something about the thing that fires jobs on a clock, where is it?",
        },
    },
    "graceful shutdown": {
        "passage": (
            "Catch termination signals, stop accepting new work, finish or "
            "requeue in-flight items, then exit cleanly inside a deadline."
        ),
        "questions": {
            "direct": "which note covers stopping intake, draining in-flight work, and exiting within a deadline?",
            "symptom": "deploys killed jobs midway and corrupted files — find where we wrote the drain procedure",
            "vague": "the bit about finishing up politely before quitting, which note?",
        },
    },
    "feature flags": {
        "passage": (
            "Wrap unfinished behavior behind runtime toggles so releases ship "
            "dark and exposure ramps by cohort without redeploying."
        ),
        "questions": {
            "direct": "which note says ship dark behind runtime toggles ramped up by cohort?",
            "symptom": "every release gated on finished-perfect-or-revert — find where we changed that",
            "vague": "the note about switches deciding who sees the new thing, where is it?",
        },
    },
    "canary release": {
        "passage": (
            "Route a small slice of live traffic to the new version first; if "
            "its error rates hold steady, widen the slice step by step."
        ),
        "questions": {
            "direct": "which note covers feeding new versions increasing slices of traffic while watching errors?",
            "symptom": "a bad deploy hit everyone simultaneously — find where we wrote the gradual rollout",
            "vague": "something about dipping a toe in before swimming, which note?",
        },
    },
    "dependency lockfiles": {
        "passage": (
            "Pin the exact transitively resolved package set in a committed "
            "manifest so builds are byte-identical months later on any machine."
        ),
        "questions": {
            "direct": "which note prescribes committing a manifest pinning the full transitive package set?",
            "symptom": "fresh installs picked different library versions than prod — find where we wrote the policy",
            "vague": "the note about everyone building exactly the same bits, where is it?",
        },
    },
}


QUESTION_STYLES = ("direct", "symptom", "vague")


def build_tasks_v2(limit_per_style: int | None = None, seed: int = 0) -> list[dict]:
    """Flatten BANK into tasks, optionally capping topics per style."""
    import random

    rng = random.Random(seed)
    topics = sorted(BANK)
    tasks = []
    for style in QUESTION_STYLES:
        entries = [(t, BANK[t]["questions"][style]) for t in topics]
        if limit_per_style is not None:
            entries = rng.sample(entries, k=min(limit_per_style, len(entries)))
        for topic, question in entries:
            tasks.append({"topic": topic, "question": question, "style": style})
    return tasks
