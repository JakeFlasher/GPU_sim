"""Forbidden-pattern pre-flight scan for the harness.

Re-uses the scaffold validator's forbidden-surface patterns
(``baselines_scaffold.validate._FORBIDDEN_PATTERNS``) and applies them
to every landed baseline's disk sources plus the ``deferred.json``
ledger entries. Adds two harness-specific checks:

1. ``deferred.json`` entries must carry real git SHAs (at least 7 hex
   chars) in ``harness_commit``, not placeholder strings like
   ``"round-2-pending"``.
2. The deferred ledger's per-entry ``status`` must be one of the
   schema-declared values (``deferred_correctness`` /
   ``deferred_reward_hack``). The legacy ``deferred_slow`` status
   retired on 2026-05-13 with the move to a correctness-only landing
   contract; slow-but-correct kernels now land, not defer.

This module is invoked as a pre-flight gate by the aggregate report
step in ``report.py`` so the report cannot ship if either gate fails.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from baselines_scaffold.validate import (  # noqa: E402
    _CUDA_GRAPH_NOTE_PATTERNS,
    _FORBIDDEN_PATTERNS,
)


# Real-SHA pattern: at least 7 hex chars. ``round-2-pending``,
# ``unknown``, ``TBD`` and similar placeholders are rejected.
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")

_VALID_DEFERRED_STATUSES = frozenset(
    {"deferred_correctness", "deferred_reward_hack"}
)


def scan_source_text(rel_path: str, text: str) -> List[Tuple[str, str, str]]:
    """Forbidden-surface scan on one source text.

    Returns ``[(label, rule, detail), ...]`` for each hit. CUDA Graph
    matches that are co-located with a note-style comment are not
    flagged (preserves the scaffold validator's per-problem note
    escape hatch).
    """
    hits: List[Tuple[str, str, str]] = []
    for fp in _FORBIDDEN_PATTERNS:
        if not fp.pattern.search(text):
            continue
        if fp.label in ("cudaGraphInstantiate", "torch.cuda.CUDAGraph"):
            if any(p.search(text) for p in _CUDA_GRAPH_NOTE_PATTERNS):
                continue
        hits.append((f"{rel_path}:{fp.label}", fp.rule, fp.detail))
    return hits


def scan_solution_json_compile_options(
    rel_label: str,
    solution_dict: dict,
) -> List[Tuple[str, str, str]]:
    """Forbidden-surface scan over a solution.json's ``compile_options``.

    The disk source files do not carry the ``compile_options`` flags
    (those live in JSON only), so a forbidden surface like
    ``--lock-clocks`` or ``ncu`` embedded in
    ``compile_options.cuda_cflags`` would slip past the disk-text
    scan. This helper mirrors the scaffold validator's behavior at
    ``scripts/baselines_scaffold/validate.py:482-489`` so the harness
    pre-flight has the same scope as the scaffold validator.

    Returns ``[(label, rule, detail), ...]`` for each hit. The label
    is prefixed with ``rel_label`` (typically the baseline directory)
    and the field name (e.g. ``baselines/.../cuda_cpp/solution.json:
    compile_options.cuda_cflags``) so operators can identify the
    offending field from the report's stderr.
    """
    hits: List[Tuple[str, str, str]] = []
    spec = solution_dict.get("spec", {}) or {}
    compile_options = spec.get("compile_options") or {}
    for field in ("cflags", "cuda_cflags", "ld_flags"):
        flags = compile_options.get(field) or []
        joined = " ".join(str(f) for f in flags)
        if not joined:
            continue
        field_label = f"compile_options.{field}"
        for label, rule, detail in scan_source_text(field_label, joined):
            hits.append((f"{rel_label}/{label}", rule, detail))
    return hits


_HARNESS_PRODUCTION_EXCLUDED_DIR_NAMES = frozenset(
    {"tests", "patches", "__pycache__", ".harness_traces"}
)

# Files that LEGITIMATELY contain the forbidden pattern strings as
# regex literals or mutation-guard data — they exist precisely to
# detect these tokens elsewhere. Listing them here keeps the harness
# scan focused on accidental Docker tokens in the dispatch path.
_HARNESS_PRODUCTION_EXCLUDED_FILES = frozenset(
    {"lint.py"}
)

# Docker-specific forbidden patterns enforced on harness production
# code (the canonical Native-dispatch gate). The full
# :data:`_FORBIDDEN_PATTERNS` set is too broad for harness production
# because lint.py legitimately stores some patterns as data.
_HARNESS_PRODUCTION_DOCKER_LABELS = frozenset(
    {
        "docker invocation",
        "--privileged flag",
        "--gpus flag",
        "sol-execbench image reference",
    }
)


def scan_harness_production_code(
    harness_root: Path = _THIS_DIR,
) -> List[Tuple[str, str, str]]:
    """Forbidden-surface scan over harness production Python sources.

    Mirrors :func:`scan_landed_baselines` for the harness's own modules
    so the lint gate enforces zero Docker tokens in the harness's
    dispatch path. Only the Docker-specific patterns from
    :data:`_HARNESS_PRODUCTION_DOCKER_LABELS` are applied — the other
    forbidden patterns (NCU, clock-locks, CUDA streams) are already
    enforced on landed sources by :func:`scan_landed_baselines` and
    appear legitimately in the harness's own lint / mutation guards.

    Test files, vendored patches, build caches, trace bundles, and the
    files listed in :data:`_HARNESS_PRODUCTION_EXCLUDED_FILES` are
    excluded.

    Returns ``[(label, rule, detail), ...]`` for each hit; the caller
    (typically ``report.py``'s pre-write gate) refuses to write the
    aggregate report when any production hit fires.
    """
    docker_patterns = tuple(
        fp for fp in _FORBIDDEN_PATTERNS
        if fp.label in _HARNESS_PRODUCTION_DOCKER_LABELS
    )
    hits: List[Tuple[str, str, str]] = []
    for path in sorted(harness_root.rglob("*.py")):
        if not path.is_file():
            continue
        if any(part in _HARNESS_PRODUCTION_EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.name in _HARNESS_PRODUCTION_EXCLUDED_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(harness_root))
        for fp in docker_patterns:
            if fp.pattern.search(text):
                hits.append(
                    (f"{harness_root}/{rel}:{fp.label}", fp.rule, fp.detail)
                )
    return hits


# Container-era runtime terms that must never appear in active
# production strings (probe ``tool_version``, surface
# ``waiver_reason``, parse-error helper text, ``extra`` wrapper
# labels) after the Docker removal landed. The native dispatch path
# runs every wrapper as a host subprocess; these terms describe a
# container-mode dispatch that no longer exists. Tests and explicit
# historical-context disclaimer blocks are excluded the same way
# the Docker-token scan excludes them — by file/dir filter, since
# the disclaimer prose lives in module docstrings of files that
# already opt out of this scan via filename.
#
# Generalisation note (per BL-20260512-lint-path-validity-all-subcorpora):
# this scanner extends the existing harness-production scan across
# the same subcorpus (Python files under
# ``scripts/baselines_harness/`` minus tests / patches / caches /
# trace bundles / lint.py) and the same field set (any matching
# substring in the file body). When future container-era vocabulary
# needs to be added, append to :data:`_CONTAINER_ERA_RUNTIME_TERMS`
# and the scan auto-extends.
_CONTAINER_ERA_RUNTIME_TERMS: Tuple[str, ...] = (
    "in-container",
    "eval container",
    "container-side",
    "eval image",
)


def scan_harness_production_text_for_container_terms(
    harness_root: Path = _THIS_DIR,
) -> List[Tuple[str, str, str]]:
    """Container-era runtime-string scan over harness production code.

    Companion to :func:`scan_harness_production_code` that catches
    the container-era doc-sweep gaps in surface plugins: the
    Docker-token regex only fires on the four scaffold labels and
    does not catch free-text strings like
    ``"in-container compute-sanitizer"`` or
    ``"compute-sanitizer (container-side)"``.

    Same exclusion set as :func:`scan_harness_production_code`. The
    caller (``report.py``'s pre-write gate) refuses to write the
    aggregate report when any production hit fires.
    """
    hits: List[Tuple[str, str, str]] = []
    for path in sorted(harness_root.rglob("*.py")):
        if not path.is_file():
            continue
        if any(part in _HARNESS_PRODUCTION_EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.name in _HARNESS_PRODUCTION_EXCLUDED_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(harness_root))
        for term in _CONTAINER_ERA_RUNTIME_TERMS:
            if term in text:
                hits.append(
                    (
                        f"{harness_root}/{rel}:container-term:{term}",
                        "harness production code must use host/native "
                        "terminology, not container-era runtime terms",
                        (
                            "Rewrite the active string to refer to the "
                            "host subprocess / native dispatch path. "
                            "If the term legitimately appears in a "
                            "historical-context disclaimer, move that "
                            "disclaimer into the module docstring of a "
                            "scan-excluded file (lint.py) "
                            "or out of the harness production tree."
                        ),
                    )
                )
    return hits


# Workflow-history token patterns. Production comments should
# describe what the code does, not the iteration history of how it
# got there. The four pattern categories below match review-loop
# bookkeeping (numbered round labels, numbered side-issue labels,
# anchor-rewrite phrasing, plan-requires prose) that belongs in
# commit messages, PR descriptions, or the ``.humanize/`` run-state
# files, not in production source.
#
# Each entry is ``(category, compiled regex)``. The category labels
# are deliberately written without hyphens so they cannot match
# their own anchor-rewrite pattern; the compiled regexes are built
# from string fragments for the same reason (the literal anchor
# spelling is split across concatenation so it does not appear as a
# contiguous token in this source file).
_WORKFLOW_HISTORY_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("numbered_round", re.compile(r"\bRound[ -]?[0-9]+\b")),
    ("numbered_sideissue", re.compile(r"\bside" + r"-issue-[0-9]+\b")),
    ("anchor_rewrite", re.compile(r"\bre" + r"-anchor\b")),
    ("plan_requires", re.compile(r"\bThe plan req" + r"uires\b")),
)


# The workflow-history scanner uses its own narrower exclusion set:
# it does NOT exclude ``lint.py`` (its rule applies to ordinary
# prose in scanner module too — only the regex sample data above is
# allowed to look like it does, and the construct-from-fragments
# trick keeps even those samples from matching). The Docker-token
# and container-term scanners keep excluding ``lint.py`` via
# :data:`_HARNESS_PRODUCTION_EXCLUDED_FILES` because they store
# their token literals as data.
_WORKFLOW_HISTORY_PRODUCTION_EXCLUDED_FILES = frozenset()


def scan_harness_production_text_for_workflow_history_terms(
    harness_root: Path = _THIS_DIR,
) -> List[Tuple[str, str, str]]:
    """Review-bookkeeping prose scan over harness production code.

    Companion to :func:`scan_harness_production_code` and
    :func:`scan_harness_production_text_for_container_terms`.
    Production comments should describe what the code does; review-
    loop history belongs in commit messages, PR descriptions, and
    the ``.humanize/`` run state.

    Exclusion policy is narrower than the other two scanners: this
    scanner walks ``lint.py`` itself (its rule applies to ordinary
    prose in the scanner module too). Tests, patches, build caches,
    and trace bundles remain excluded.

    Returns ``[(label, rule, detail), ...]`` for each hit. The
    caller (``report.py``'s pre-write gate) refuses to write the
    aggregate report when any production hit fires.
    """
    hits: List[Tuple[str, str, str]] = []
    for path in sorted(harness_root.rglob("*.py")):
        if not path.is_file():
            continue
        if any(part in _HARNESS_PRODUCTION_EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.name in _WORKFLOW_HISTORY_PRODUCTION_EXCLUDED_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(harness_root))
        for label, pattern in _WORKFLOW_HISTORY_PATTERNS:
            for match in pattern.finditer(text):
                hits.append(
                    (
                        f"{harness_root}/{rel}:workflow-history:{label}:{match.group(0)!r}",
                        "harness production comments must describe "
                        "current behavior, not review bookkeeping",
                        (
                            "Rewrite the comment / docstring to "
                            "describe what the code does. Review "
                            "bookkeeping (numbered round labels, "
                            "numbered side-tagged issues, anchor-"
                            "rewrite phrasing, plan-requires prose) "
                            "belongs in commit messages, PR "
                            "descriptions, or the .humanize/ run "
                            "state, not in the source tree."
                        ),
                    )
                )
    return hits


def scan_landed_baselines(baseline_dirs: Iterable[Path]) -> List[Tuple[str, str, str]]:
    """Scan every landed baseline's on-disk sources AND solution.json.

    Each entry in ``baseline_dirs`` is a path to
    ``baselines/<cat>/<slug>/<lang>/``. The lint reads each file under
    the variant directory and applies the forbidden-surface pattern
    set. Two complementary scans run per baseline:

    1. **Disk-text scan** (every file except ``solution.json``): the
       canonical authoring surface.
    2. **``solution.json`` ``compile_options`` scan**: JSON-only
       fields like ``compile_options.cuda_cflags`` can carry
       forbidden surfaces (``--lock-clocks``, ``ncu``, etc.) that
       the disk-text scan would miss because authors don't write
       build flags in source files. The scaffold validator at
       ``scripts/baselines_scaffold/validate.py:482-489`` already
       does this scan; the harness pre-flight mirrors it so the
       aggregate report cannot ship with a clean lint while the
       validator would fail.
    """
    hits: List[Tuple[str, str, str]] = []
    for baseline in baseline_dirs:
        if not baseline.is_dir():
            continue
        for path in sorted(baseline.rglob("*")):
            if not path.is_file() or path.name == "solution.json":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = str(path.relative_to(baseline))
            for label, rule, detail in scan_source_text(rel, text):
                hits.append(
                    (f"{baseline}/{label}", rule, detail)
                )
        solution_path = baseline / "solution.json"
        if solution_path.is_file():
            try:
                solution_dict = json.loads(
                    solution_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                solution_dict = None
            if isinstance(solution_dict, dict):
                hits.extend(
                    scan_solution_json_compile_options(
                        str(baseline), solution_dict
                    )
                )
    return hits


def validate_deferred_ledger(
    ledger_path: Path,
) -> List[str]:
    """Validate ``deferred.json`` entry hygiene.

    Returns a list of failure messages (empty when the ledger is
    clean). Catches:

    - Missing or non-list ``entries``.
    - Per-entry ``harness_commit`` that isn't a real git SHA (rejects
      placeholders like ``"round-2-pending"``).
    - Per-entry ``status`` that isn't in the declared valid set.
    - Missing required per-entry fields.
    """
    if not ledger_path.is_file():
        return [f"deferred ledger missing at {ledger_path}"]
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"deferred ledger is not valid JSON: {exc}"]

    entries = ledger.get("entries")
    if not isinstance(entries, list):
        return ["deferred ledger missing 'entries' list"]

    failures: List[str] = []
    required_fields = (
        "slug",
        "language",
        "status",
        "justification",
        "evidence",
        "harness_commit",
    )
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(f"entry[{i}] is not a JSON object")
            continue
        for f in required_fields:
            if f not in entry:
                failures.append(f"entry[{i}] missing required field '{f}'")
        status = entry.get("status")
        if status is not None and status not in _VALID_DEFERRED_STATUSES:
            failures.append(
                f"entry[{i}] status {status!r} not in "
                f"{sorted(_VALID_DEFERRED_STATUSES)}"
            )
        commit = entry.get("harness_commit", "")
        if not isinstance(commit, str) or not _GIT_SHA_RE.match(commit):
            failures.append(
                f"entry[{i}] harness_commit {commit!r} is not a real "
                f"git SHA (7-40 hex chars); reject placeholders like "
                f"'round-N-pending'."
            )
    return failures
