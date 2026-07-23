#!/usr/bin/env python3
"""
run_relevance_evals.py — exercise the relevance gate independently.

Loads scripts/kb/_relevance_evals.yaml (positive / negative / known_edge
suites), classifies each prompt with `is_kb_relevant()`, reports per-
suite pass/fail counts. Exit non-zero if positive-suite recall drops
below 95% or negative-suite precision drops below 95%.

This is the gate-only complement to run_evals.py (which scores the
post-gate skill routing). A regression in is_kb_relevant() shows here
first; a regression in subcorpus boosting / per-card overrides shows in
run_evals.py.

Usage:
  python3 scripts/kb/run_relevance_evals.py [--root REPO_ROOT] [--quiet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _relevance import is_kb_relevant  # noqa: E402


def _load_yaml_lite(p: Path) -> dict:
    """Tiny YAML reader for our suite schema (3 top-level lists, each of
    `- prompt: ...\\n  expect: ...\\n  note: ...`).
    """
    text = p.read_text(encoding="utf-8")
    suites: dict[str, list[dict]] = {"positive": [], "negative": [], "known_edge": []}
    cur_suite: str | None = None
    cur: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line in ("positive:", "negative:", "known_edge:"):
            # Flush the pending entry into the OLD suite before switching.
            if cur is not None and cur_suite is not None:
                suites[cur_suite].append(cur)
                cur = None
            cur_suite = line[:-1]
            continue
        if cur_suite is None:
            continue
        if line.startswith("  - prompt:"):
            if cur is not None:
                suites[cur_suite].append(cur)
            cur = {"prompt": _strip_quotes(line.split(":", 1)[1].strip()),
                   "expect": True, "note": ""}
        elif line.startswith("    expect:"):
            v = line.split(":", 1)[1].strip().lower()
            if cur is not None:
                cur["expect"] = v in ("true", "yes", "1")
        elif line.startswith("    note:"):
            if cur is not None:
                cur["note"] = _strip_quotes(line.split(":", 1)[1].strip())
    if cur is not None and cur_suite is not None:
        suites[cur_suite].append(cur)
    return suites


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--positive-threshold", type=float, default=0.95,
        help="Minimum positive recall (default 0.95)")
    parser.add_argument(
        "--negative-threshold", type=float, default=0.95,
        help="Minimum negative precision (default 0.95)")
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    suite_path = repo_root / "scripts" / "kb" / "_relevance_evals.yaml"
    if not suite_path.is_file():
        print(f"error: suite file missing at {suite_path}", file=sys.stderr)
        return 1

    suites = _load_yaml_lite(suite_path)

    pos_total = len(suites["positive"])
    pos_hits = 0
    pos_misses: list[str] = []
    for q in suites["positive"]:
        actual = is_kb_relevant(q["prompt"])
        if actual == q["expect"]:
            pos_hits += 1
        else:
            pos_misses.append(q["prompt"])

    neg_total = len(suites["negative"])
    neg_hits = 0
    neg_misses: list[str] = []
    for q in suites["negative"]:
        actual = is_kb_relevant(q["prompt"])
        if actual == q["expect"]:
            neg_hits += 1
        else:
            neg_misses.append(q["prompt"])

    edge_results: list[tuple[str, bool, bool, str]] = []
    for q in suites["known_edge"]:
        actual = is_kb_relevant(q["prompt"])
        edge_results.append((q["prompt"], q["expect"], actual, q.get("note", "")))

    pos_recall = pos_hits / pos_total if pos_total else 0.0
    neg_precision = neg_hits / neg_total if neg_total else 0.0

    print(f"Positive suite (gate should accept): "
          f"{pos_hits}/{pos_total} = {pos_recall:.1%}")
    if pos_misses and not args.quiet:
        print("  False negatives (gate should accept but rejected):")
        for p in pos_misses:
            print(f"    - {p!r}")
    print(f"Negative suite (gate should reject): "
          f"{neg_hits}/{neg_total} = {neg_precision:.1%}")
    if neg_misses and not args.quiet:
        print("  False positives (gate should reject but accepted):")
        for p in neg_misses:
            print(f"    - {p!r}")
    print(f"Known-edge suite (informational, not scored): "
          f"{len(suites['known_edge'])} entries")
    if edge_results and not args.quiet:
        for prompt, expect, actual, note in edge_results:
            label = "behaves as documented" if expect == actual else "DRIFTED"
            print(f"  [{label}] expect={expect} actual={actual} | {prompt!r}")
            if note:
                print(f"            note: {note}")

    fail = False
    if pos_recall < args.positive_threshold:
        print(f"FAIL: positive recall {pos_recall:.1%} < threshold "
              f"{args.positive_threshold:.1%}", file=sys.stderr)
        fail = True
    if neg_precision < args.negative_threshold:
        print(f"FAIL: negative precision {neg_precision:.1%} < threshold "
              f"{args.negative_threshold:.1%}", file=sys.stderr)
        fail = True

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
