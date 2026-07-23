#!/usr/bin/env python3
"""
run_evals.py — run all _eval/queries.yaml suites against route_query.py.

For every Skill at .claude/skills/<name>/_eval/queries.yaml, runs each query
through route_query.py and scores it on:

  - hit_top1            : did the recommended Skill match expected_skill?
  - hit_top3            : was expected_skill in the top-3 unique skills?
  - cards_recall        : fraction of expected_cards that appeared in the
                          top-3 returned cards.
  - cards_precision     : fraction of returned top-3 cards that were
                          expected (per the expected_cards list).

forbidden_behavior is recorded but not auto-checked (would need an LLM
judge for those — out of scope for this script).

Usage:
  python3 scripts/kb/run_evals.py [--root REPO_ROOT] [--quiet]

Exit code:
  0 — all suites loaded and ran
  1 — script error
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def _load_yaml_lite(p: Path) -> dict:
    """Tiny purpose-built YAML reader for the queries.yaml schema we use.

    Schema (Codex-review revision: split required vs helpful cards):
        queries:
          - prompt: "..."
            expected_skill: skill_name
            required_cards:        # MUST be returned for full credit
              - path1
            helpful_cards:         # bonus / stretch goal — does not penalize
              - path2
              - path3
            expected_cards:        # legacy field; treated as required_cards
              - path
            forbidden_behavior:
              - phrase1
            accept_top3: true|false  # meta-router relaxation

    Avoids a PyYAML dependency.
    """
    text = p.read_text(encoding="utf-8")
    queries: list[dict] = []
    cur: dict | None = None
    cur_list_key: str | None = None

    def _flush() -> None:
        # Backwards-compat: if neither required_cards nor helpful_cards is
        # set but expected_cards is, treat expected_cards as required.
        if cur is None:
            return
        if not cur["required_cards"] and not cur["helpful_cards"] \
                and cur["expected_cards"]:
            cur["required_cards"] = list(cur["expected_cards"])
        queries.append(cur)

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("queries:"):
            continue
        if line.startswith("  - prompt:"):
            _flush()
            cur = {
                "prompt": _strip_quotes(line.split(":", 1)[1].strip()),
                "expected_skill": "",
                "expected_cards": [],
                "required_cards": [],
                "helpful_cards": [],
                "forbidden_behavior": [],
            }
            cur_list_key = None
            continue
        if cur is None:
            continue
        if line.startswith("    expected_skill:"):
            cur["expected_skill"] = _strip_quotes(line.split(":", 1)[1].strip())
            cur_list_key = None
        elif line.startswith("    expected_cards:"):
            cur_list_key = "expected_cards"
        elif line.startswith("    required_cards:"):
            cur_list_key = "required_cards"
        elif line.startswith("    helpful_cards:"):
            cur_list_key = "helpful_cards"
        elif line.startswith("    forbidden_behavior:"):
            cur_list_key = "forbidden_behavior"
        elif line.startswith("    accept_top3:"):
            v = _strip_quotes(line.split(":", 1)[1].strip()).lower()
            cur["accept_top3"] = v in ("true", "yes", "1")
            cur_list_key = None
        elif line.startswith("      - "):
            if cur_list_key:
                cur[cur_list_key].append(_strip_quotes(line[len("      - "):].strip()))
        elif line.startswith("    prompt:"):
            cur["prompt"] = _strip_quotes(line.split(":", 1)[1].strip())
    _flush()
    return {"queries": queries}


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _route(prompt: str, repo_root: Path) -> str:
    """Run route_query.py on a prompt; return stdout text."""
    cmd = [
        "python3",
        str(repo_root / "scripts" / "kb" / "route_query.py"),
        "--root",
        str(repo_root),
        "--top",
        "3",
    ]
    payload = json.dumps({"prompt": prompt})
    try:
        proc = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.stdout
    except subprocess.TimeoutExpired:
        return ""


_PATH_RE = re.compile(r"`(\.claude/knowledge/[^`]+\.md)`")
# Skill identifiers in router output appear in two forms:
#   - `skill-name`  (backtick-wrapped — used for the primary skill)
#   - (`skill-name`) or (skill-name)  (parenthesized — used for non-primary
#     per-card annotations and "Other potentially relevant skills" lists)
# Match either form, deduped against the canonical skill list later.
_SKILL_RE = re.compile(r"`([a-z][\w-]*)`|\(([\w-]+)\)")

# phase 2-rev fix: restrict accepted skill names to the canonical 9. Without
# this, any hyphenated parenthesized token in a card title (e.g., a section
# header or product name) could leak into the skills list and pollute scoring.
_KNOWN_SKILLS = {
    "navigating-project-kb",
    "validating-nonroot-gpu-measurement",
    "interpreting-ptx-isa",
    "tuning-blackwell-sm120-kernels",
    "analyzing-sass-nvbit-traces",
    "compiling-cuda-artifacts",
    "optimizing-cuda-kernels",
    "reasoning-about-nvidia-architecture-history",
    "designing-gpu-simulator",
    "implementing-sol-baselines",
}


def _parse_router_output(text: str) -> tuple[list[str], list[str]]:
    paths = _PATH_RE.findall(text)
    # _SKILL_RE has two alternation groups; one will be empty per match.
    matches = _SKILL_RE.findall(text)
    skill_seen: list[str] = []
    for backtick, paren in matches:
        s = backtick or paren
        if s and s not in skill_seen and s in _KNOWN_SKILLS:
            skill_seen.append(s)
    return paths, skill_seen


def _score_query(q: dict, returned_paths: list[str], returned_skills: list[str]) -> dict:
    required_cards = set(q.get("required_cards") or [])
    helpful_cards = set(q.get("helpful_cards") or [])
    # Legacy: if neither is set but expected_cards is, treat as required.
    if not required_cards and not helpful_cards:
        required_cards = set(q.get("expected_cards") or [])
    all_expected = required_cards | helpful_cards

    returned_set = set(returned_paths)
    expected_skill = q.get("expected_skill", "")
    accept_top3 = bool(q.get("accept_top3"))

    # Required recall: fraction of REQUIRED cards that the router returned.
    # This is the load-bearing card-quality signal — a routing change that
    # drops a required-card hit is a regression.
    if required_cards:
        req_hit = required_cards & returned_set
        required_recall = len(req_hit) / len(required_cards)
    else:
        required_recall = float("nan")

    # Helpful recall: fraction of HELPFUL cards that the router returned.
    # Stretch-goal signal — informational only, not part of the headline
    # score. Helps prioritize future precision-tuning without penalizing
    # routes that already cover the must-haves.
    if helpful_cards:
        help_hit = helpful_cards & returned_set
        helpful_recall = len(help_hit) / len(helpful_cards)
    else:
        helpful_recall = float("nan")

    # Combined recall (legacy `cards_recall` field retained for backwards
    # compat). Computed over `all_expected` so existing dashboards still
    # see a number, but new code should prefer required_recall.
    if all_expected:
        all_hit = all_expected & returned_set
        cards_recall = len(all_hit) / len(all_expected)
        unique_returned = len(set(returned_paths))
        cards_precision = (len(all_hit) / unique_returned
                           if unique_returned else 0.0)
    else:
        cards_recall = float("nan")
        cards_precision = float("nan")

    in_top3 = expected_skill in returned_skills
    rank_one = bool(returned_skills) and returned_skills[0] == expected_skill
    hit_top1 = rank_one or (accept_top3 and in_top3)
    hit_top3 = in_top3

    return {
        "prompt": q["prompt"],
        "expected_skill": expected_skill,
        "required_cards": sorted(required_cards),
        "helpful_cards": sorted(helpful_cards),
        "expected_cards": sorted(all_expected),  # legacy union
        "returned_paths": returned_paths,
        "returned_skills": returned_skills,
        "hit_top1": hit_top1,
        "hit_top3": hit_top3,
        "required_recall": required_recall,
        "helpful_recall": helpful_recall,
        "cards_recall": cards_recall,
        "cards_precision": cards_precision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    skills_dir = repo_root / ".claude" / "skills"
    if not skills_dir.is_dir():
        print(f"error: {skills_dir} not found", file=sys.stderr)
        return 1

    suites: list[tuple[str, Path]] = []
    for d in sorted(skills_dir.iterdir()):
        if not d.is_dir():
            continue
        ev = d / "_eval" / "queries.yaml"
        if ev.is_file():
            suites.append((d.name, ev))

    if not suites:
        print("no eval suites found", file=sys.stderr)
        return 1

    all_results: list[dict] = []
    for name, ev_path in suites:
        data = _load_yaml_lite(ev_path)
        for q in data.get("queries", []):
            text = _route(q["prompt"], repo_root)
            paths, skills = _parse_router_output(text)
            r = _score_query(q, paths, skills)
            r["suite"] = name
            all_results.append(r)

    n = len(all_results)
    if n == 0:
        print("no queries scored", file=sys.stderr)
        return 1

    n_top1 = sum(1 for r in all_results if r["hit_top1"])
    n_top3 = sum(1 for r in all_results if r["hit_top3"])
    req_recalls = [r["required_recall"] for r in all_results
                   if r["required_recall"] == r["required_recall"]]
    help_recalls = [r["helpful_recall"] for r in all_results
                    if r["helpful_recall"] == r["helpful_recall"]]
    all_recalls = [r["cards_recall"] for r in all_results
                   if r["cards_recall"] == r["cards_recall"]]
    precs = [r["cards_precision"] for r in all_results
             if r["cards_precision"] == r["cards_precision"]]
    avg_req_recall = sum(req_recalls) / len(req_recalls) if req_recalls else 0.0
    avg_help_recall = sum(help_recalls) / len(help_recalls) if help_recalls else float("nan")
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
    avg_precision = sum(precs) / len(precs) if precs else 0.0

    print(f"Eval suites: {len(suites)}")
    print(f"Total queries: {n}")
    print(f"Skill match @top1:  {n_top1:>3} / {n}  ({100*n_top1/n:5.1f}%)")
    print(f"Skill match @top3:  {n_top3:>3} / {n}  ({100*n_top3/n:5.1f}%)")
    print(f"Required-card recall (must-return cards):        {avg_req_recall:.3f}  "
          f"({len(req_recalls)} queries)")
    if help_recalls:
        print(f"Helpful-card recall (stretch / bonus cards):     "
              f"{avg_help_recall:.3f}  ({len(help_recalls)} queries)")
    print(f"All-card recall (legacy union):                  {avg_recall:.3f}")
    print(f"Card precision (avg over returned cards):        {avg_precision:.3f}")
    print()

    if not args.quiet:
        print("Per-suite results:")
        by_suite: dict[str, list[dict]] = {}
        for r in all_results:
            by_suite.setdefault(r["suite"], []).append(r)
        for s, rs in sorted(by_suite.items()):
            sn = len(rs)
            st1 = sum(1 for r in rs if r["hit_top1"])
            st3 = sum(1 for r in rs if r["hit_top3"])
            sreq = [r["required_recall"] for r in rs
                    if r["required_recall"] == r["required_recall"]]
            avgreq = sum(sreq) / len(sreq) if sreq else 0.0
            print(f"  {s:<46s}  top1={st1}/{sn}  top3={st3}/{sn}  "
                  f"req_recall={avgreq:.3f}")

        print()
        print("Misses (top1 != expected_skill):")
        for r in all_results:
            if r["hit_top1"]:
                continue
            print(f"  [{r['suite']}] '{r['prompt'][:60]}...'")
            print(f"    expected_skill={r['expected_skill']}, returned_skills={r['returned_skills']}")
            missing = [c for c in r["expected_cards"] if c not in r["returned_paths"]]
            if missing:
                print(f"    missing expected cards: {missing}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
