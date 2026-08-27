"""Single entrypoint for every benchmark suite.

    uv run --group dev python -m benchmarks.run <suite> [suite options...]

Suites:
    retrieval   graded retrieval quality (synthetic / BeIR / repo / csn)
    systems     scaling: latency percentiles, throughput, memory, db size
    agent-pi    real pi coding-agent runs (needs OPENAI_API_KEY; prints cost)
    latency     pytest-benchmark micro-latency suite

All suites write JSON results under benchmarks/results/<suite>/ plus a
rendered markdown summary next to the code.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


SUITES = {
    "retrieval": "benchmarks.run_retrieval",
    "systems": "benchmarks.run_systems",
    "agent-pi": "benchmarks.agent_pi",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", choices=sorted(SUITES) + ["latency"])
    ns, rest = ap.parse_known_args()

    if ns.suite == "latency":
        cmd = [sys.executable, "-m", "pytest", "benchmarks/test_latency.py",
               "--benchmark-only", *rest]
        raise SystemExit(subprocess.run(cmd).returncode)

    module = SUITES[ns.suite]
    cmd = [sys.executable, "-m", module, *rest]
    if ns.suite == "agent-pi":
        print("[agent-pi] spawns real LLM sessions and spends API credit.")
        print("[agent-pi] estimated worst case is a few cents per episode;\n"
              "[agent-pi] interrupt (Ctrl-C) any time — completed episodes\n"
              "[agent-pi] are kept in episodes.jsonl and reused on rerun.\n")
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
