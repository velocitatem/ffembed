"""Live agent benchmark: a real LLM tool-use loop finds the right file.

For each task, an LLM agent gets one of three toolboxes and must locate the
note that answers a natural-language question, then call `finish(path)`:

- list:   `list_files` + `read_file`          (brute force)
- grep:   `grep` + `list_files` + `read_file` (keyword search)
- ffembed:`semantic_search` + `read_file`     (semantic search)

Tokens are read from the API's `usage` — real prompt + completion tokens,
no estimation. Success means the file the agent finished with actually
contains the answer passage (ground truth from the corpus generator).

Requires OPENAI_API_KEY. Model via FFEMBED_BENCH_MODEL (default gpt-4o-mini),
temperature 0.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .corpus import IMPLICIT_PASSAGES


MODEL = os.environ.get("FFEMBED_BENCH_MODEL", "gpt-4o-mini")
MAX_TURNS = 10
SEARCH_K = 5


# --- toolboxes -------------------------------------------------------------

LIST_FILES = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List all note filenames in the directory.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

GREP = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search note contents for a regex pattern. Returns the paths of "
            "matching notes (one per line), or nothing if no match."
        ),
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
}

SEMANTIC_SEARCH = {
    "type": "function",
    "function": {
        "name": "semantic_search",
        "description": (
            "Search notes by meaning. Returns the top matches with a snippet "
            "of each note's text and a relevance score. Use a natural-language "
            "description of what you are looking for."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the full text of one note.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

FINISH = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Declare the file that answers the question and stop.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

TOOLBOXES = {
    "list": [LIST_FILES, READ_FILE, FINISH],
    "grep": [GREP, LIST_FILES, READ_FILE, FINISH],
    "ffembed": [SEMANTIC_SEARCH, READ_FILE, FINISH],
}


SYSTEM_PROMPT = (
    "You are helping find a note in a directory of markdown files. "
    "Use your tools to locate the single file that best answers the user's "
    "question, then call finish with that file's path. Read a file to verify "
    "before finishing. Work efficiently."
)


@dataclass
class LiveAttempt:
    policy: str
    question: str
    found: bool
    turns: int
    input_tokens: int
    output_tokens: int
    chosen: str
    tool_calls: int


class Episode:
    """One agent run: dispatches tools, counts tokens."""

    def __init__(self, client: OpenAI, policy: str, root: Path,
                 conn=None, target_path: str | None = None):
        self.client = client
        self.policy = policy
        self.root = root
        self.conn = conn
        self.target_path = target_path
        self.input_tokens = 0
        self.output_tokens = 0
        self.turns = 0
        self.tool_calls = 0
        self.messages: list[dict] = []
        self.chosen: str | None = None

    def run(self, question: str) -> None:
        self.messages = [{"role": "user", "content": f"Question: {question}"}]
        for _ in range(MAX_TURNS):
            self.turns += 1
            resp = self.client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=1024,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.messages,
                tools=TOOLBOXES[self.policy],
            )
            self.input_tokens += resp.usage.prompt_tokens
            self.output_tokens += resp.usage.completion_tokens
            msg = resp.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))

            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                return  # gave up without finish
            results = []
            for tc in tool_calls:
                self.tool_calls += 1
                out = self._dispatch(tc.function.name, tc.function.arguments)
                results.append({"role": "tool", "tool_call_id": tc.id, "content": out})
                if tc.function.name == "finish":
                    try:
                        self.chosen = json.loads(tc.function.arguments)["path"]
                    except (json.JSONDecodeError, KeyError):
                        self.chosen = ""
                    return
            self.messages.extend(results)

    # -- tool dispatch ------------------------------------------------------

    def _dispatch(self, name: str, args_json: str) -> str:
        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError:
            return "error: invalid arguments"
        if name == "list_files":
            names = sorted(p.name for p in self.root.glob("*.md"))
            return "\n".join(names)
        if name == "grep":
            proc = subprocess.run(
                ["grep", "-rEl", "--include=*.md", args["pattern"], str(self.root)],
                capture_output=True, text=True, check=False,
            )
            return "\n".join(str(Path(p).name) for p in proc.stdout.splitlines()) or "(no matches)"
        if name == "semantic_search":
            from ffembed import search

            results = search.search(self.conn, args["query"], target_path=self.target_path, k=SEARCH_K)
            lines = []
            for score, row in results:
                snippet = (row["text"] or "")[:400]
                lines.append(f"{Path(row['file_path']).name} (score {score:.3f}): {snippet}")
            return "\n\n".join(lines) or "(no matches)"
        if name == "read_file":
            p = self.root / Path(args["path"]).name
            if not p.is_file():
                return f"error: {args['path']} not found"
            return p.read_text(encoding="utf-8")
        if name == "finish":
            return "ok"
        return f"error: unknown tool {name}"


def is_correct(root: Path, chosen: str | None, topic: str) -> bool:
    """The picked file must actually contain the topic's answer passage."""
    if not chosen:
        return False
    p = root / Path(chosen).name
    if not p.is_file():
        return False
    return IMPLICIT_PASSAGES[topic][:80] in p.read_text(encoding="utf-8")


def run_policy(client: OpenAI, policy: str, tasks: list[dict], root: Path,
               conn=None, target_path: str | None = None) -> list[LiveAttempt]:
    attempts = []
    for task in tasks:
        ep = Episode(client, policy, root, conn, target_path)
        ep.run(task["question"])
        attempts.append(LiveAttempt(
            policy=policy,
            question=task["question"],
            found=is_correct(root, ep.chosen, task["topic"]),
            turns=ep.turns,
            input_tokens=ep.input_tokens,
            output_tokens=ep.output_tokens,
            chosen=ep.chosen or "",
            tool_calls=ep.tool_calls,
        ))
    return attempts


def live_markdown(attempts: list[LiveAttempt]) -> str:
    header = (
        "| policy | success | median total tokens | median turns | median tool calls |\n"
        "|---|---|---|---|---|\n"
    )
    by_policy: dict[str, list[LiveAttempt]] = {}
    for a in attempts:
        by_policy.setdefault(a.policy, []).append(a)
    lines = []
    for policy in ("list", "grep", "ffembed"):
        rs = by_policy.get(policy)
        if not rs:
            continue
        totals = sorted(a.input_tokens + a.output_tokens for a in rs)
        n = len(totals)
        lines.append(
            f"| {policy} | {100 * sum(r.found for r in rs) / n:.0f}% "
            f"| {totals[n // 2]:,} "
            f"| {sorted(r.turns for r in rs)[n // 2]} "
            f"| {sorted(r.tool_calls for r in rs)[n // 2]} |"
        )
    return header + "\n".join(lines) + "\n"
