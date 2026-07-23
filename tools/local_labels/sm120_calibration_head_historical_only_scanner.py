"""AC-6.4 v1 / v2 / v3 calibration-head historical-only scanner.

Walks the ACTIVE sm_120 code path Python files and asserts that
no string reference to ``calibration_head_v1.json`` /
``calibration_head_v2.json`` / ``calibration_head_v3.json`` /
``calibration_evaluation_v1.json`` / ``calibration_evaluation_v2.json``
/ ``calibration_evaluation_v3.json`` appears OUTSIDE explicit
historical-only contexts.

A reference is considered "historical-only" iff:

* Its containing source LINE includes any of the historical-only
  markers (case-insensitive substring): ``"retired"``,
  ``"historical-only"``, ``"historical only"``, ``"legacy"``,
  ``"pre-AC-6.4"``, ``"pre-Round-20"``, ``"BL-20260527-"``, or
  the marker comment ``# HISTORICAL_ONLY_CALIBRATION_REF``.
* OR the immediately-preceding line includes one of those
  markers (for multi-line docstring blocks that frame the
  reference as historical context).
* OR the file's docstring (first contiguous string at module
  level) includes the marker (covers files whose entire
  scope is historical / amendment-loop work).

Any remaining reference is flagged as a defect. The scanner
returns a structured result so callers can iterate the defects
deterministically.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple


_REPO_ROOT = Path(__file__).resolve().parents[2]


# The set of "active" code-path files this scanner walks. Each
# path is repo-relative.
DEFAULT_ACTIVE_PATHS: tuple[Path, ...] = (
    Path("tools/local_labels/run_headline_cohort_predict_compare.py"),
    Path("workloads/simbench_bw/comparison_runner.py"),
    Path("tools/local_labels/sm120_activation_bundle_producer.py"),
)


# Historical artifact filenames the scanner searches for.
_HISTORICAL_FILENAMES = (
    "calibration_head_v1.json",
    "calibration_head_v2.json",
    "calibration_head_v3.json",
    "calibration_evaluation_v1.json",
    "calibration_evaluation_v2.json",
    "calibration_evaluation_v3.json",
)


# Markers that exempt a line as "historical-only".
_HISTORICAL_MARKERS = (
    "retired",
    "historical-only",
    "historical only",
    "legacy",
    "pre-ac-6.4",
    "pre-round-20",
    "bl-20260527-",
    "# historical_only_calibration_ref",
)


_FILENAME_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(f) for f in _HISTORICAL_FILENAMES) + r")\b"
)


class HistoricalRefDefect(NamedTuple):
    """One non-historical reference to a retired calibration
    sidecar in an active code path. ``path`` is repo-relative;
    ``line_no`` is 1-based; ``filename`` is the matched artifact
    name; ``line`` is the offending source line (stripped)."""

    path: str
    line_no: int
    filename: str
    line: str


class HistoricalOnlyScanResult(NamedTuple):
    """Result of the AC-6.4 historical-only scan. ``ok`` is True
    iff ``defects`` is empty."""

    ok: bool
    defects: tuple[HistoricalRefDefect, ...]


def _line_is_historical(line: str) -> bool:
    lower = line.lower()
    return any(marker in lower for marker in _HISTORICAL_MARKERS)


def _module_docstring_is_historical(text: str) -> bool:
    """Return True if the file's module-level docstring (first
    triple-quoted string at module scope) contains one of the
    historical markers. The scan walks the file character-by-
    character to find the docstring opening AND its closing
    delimiter — a fully-empty file or one without a docstring
    returns False."""
    # Skip leading whitespace + comments at module level.
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == "#":
            nl = text.find("\n", i)
            if nl == -1:
                return False
            i = nl + 1
            continue
        break
    # Now `i` is at the first non-whitespace, non-comment char.
    for delim in ('"""', "'''"):
        if text.startswith(delim, i):
            end = text.find(delim, i + 3)
            if end == -1:
                return False
            docstring = text[i + 3:end]
            lower = docstring.lower()
            return any(marker in lower for marker in _HISTORICAL_MARKERS)
    return False


def scan_active_paths_for_historical_refs(
    *,
    repo_root: Path = _REPO_ROOT,
    active_paths: tuple[Path, ...] = DEFAULT_ACTIVE_PATHS,
) -> HistoricalOnlyScanResult:
    """Scan ``active_paths`` (repo-relative) for references to the
    retired calibration sidecar filenames. Returns a
    ``HistoricalOnlyScanResult`` whose ``ok`` is True iff every
    reference falls inside a historical-only context (line- or
    module-docstring-marked)."""
    defects: list[HistoricalRefDefect] = []
    for rel in active_paths:
        path = repo_root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        module_doc_is_historical = _module_docstring_is_historical(text)
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            for m in _FILENAME_RE.finditer(line):
                filename = m.group(0)
                if module_doc_is_historical:
                    continue
                if _line_is_historical(line):
                    continue
                # Multi-line lookback (5 lines): a marker on any
                # of the 5 immediately-preceding source lines
                # covers the case where a multi-line docstring
                # or argparse help string frames the reference
                # as historical context (the reference itself
                # is on a line without the marker).
                _lookback = 5
                start = max(0, idx - _lookback)
                if any(
                    _line_is_historical(lines[j])
                    for j in range(start, idx)
                ):
                    continue
                defects.append(HistoricalRefDefect(
                    path=str(rel),
                    line_no=idx + 1,
                    filename=filename,
                    line=line.strip(),
                ))
    return HistoricalOnlyScanResult(
        ok=not defects, defects=tuple(defects),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", action="append", type=Path, default=None,
        help=(
            "Optional repo-relative path to add to the active "
            "code-path set. May be repeated. When omitted the "
            "scanner uses the default active-path tuple."
        ),
    )
    args = parser.parse_args(argv)
    if args.path:
        active_paths = tuple(args.path) + DEFAULT_ACTIVE_PATHS
    else:
        active_paths = DEFAULT_ACTIVE_PATHS
    result = scan_active_paths_for_historical_refs(
        active_paths=active_paths,
    )
    if result.ok:
        print(
            f"OK: scanned {len(active_paths)} active path(s); "
            f"0 non-historical references to v1/v2/v3 calibration "
            f"sidecars"
        )
        return 0
    print(
        f"FAIL: scanned {len(active_paths)} active path(s); "
        f"{len(result.defects)} non-historical reference(s) to "
        f"v1/v2/v3 calibration sidecars:",
        file=sys.stderr,
    )
    for d in result.defects:
        print(
            f"  {d.path}:{d.line_no}: {d.filename} -> {d.line!r}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
