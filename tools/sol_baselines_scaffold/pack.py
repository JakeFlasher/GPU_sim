#!/usr/bin/env python3
"""Re-pack ``solution.json`` from editable disk files.

The scaffolder emits a ``solution.json`` whose ``sources[].content``
matches the just-copied disk files. As the author hand-edits
``kernel.cu`` / ``main.cpp`` / etc., that JSON content goes stale.
``pack.py`` rewrites ``solution.json.sources[]`` from the disk files
so the JSON tracks what's actually on disk.

The pack is **deterministic** and **idempotent**:

- Entry source (the file matching ``spec.entry_point``'s prefix)
  always appears first in ``sources[]``.
- Remaining sources are sorted lexicographically by ``path``.
- Running pack twice on an unchanged tree produces a byte-identical
  ``solution.json``.

Disk wins on staleness: if the user hand-edits
``solution.json.sources[].content`` directly without re-packing, the
downstream validate step will detect the drift via Pydantic source-
path checks and fail.

Usage:
    python tools/sol_baselines_scaffold/pack.py \\
        --baseline baselines/FIB/problem_001_fused_add_rmsnorm_h2048/cuda_cpp

The tool fails closed (exit 2, nothing written) when:

- The baseline directory does not contain ``solution.json``.
- A file referenced by the prior ``solution.json.sources[]`` is
  missing from disk.
- The entry source named by ``spec.entry_point`` is not on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PackError(Exception):
    """Fail-closed packer failure with an actionable message."""


def _entry_path_from_spec(solution_dict: Dict[str, Any]) -> str:
    spec = solution_dict.get("spec", {})
    entry_point = spec.get("entry_point", "")
    if "::" not in entry_point:
        raise PackError(
            f"solution.json spec.entry_point {entry_point!r} is missing the "
            f"'<file>::<function>' separator."
        )
    return entry_point.split("::", 1)[0]


def _walk_disk_sources(baseline_dir: Path) -> Dict[str, str]:
    """Return ``{rel_path: text}`` for every readable file in ``baseline_dir``.

    Skips ``solution.json`` (output of this tool) and any other
    auxiliary files (those whose paths the prior solution did not
    reference are filtered out by the caller — this pass just reads
    everything so a precise diff is possible).
    """
    disk: Dict[str, str] = {}
    for path in sorted(baseline_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "solution.json":
            continue
        rel = path.relative_to(baseline_dir).as_posix()
        try:
            disk[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise PackError(
                f"File {rel} under {baseline_dir} is not UTF-8 readable: "
                f"{exc}. Source files must be text."
            ) from exc
    return disk


def _select_tracked_sources(
    prior_sources: List[Dict[str, Any]],
    disk: Dict[str, str],
    entry_path: str,
) -> Dict[str, str]:
    """Return only the disk files the prior solution.json tracked.

    The packer regenerates content for the paths the previous
    ``solution.json`` already named. Any auxiliary files dropped into
    the baseline directory by the author (e.g. ``notes.md``,
    intermediate ``.o`` artefacts) are deliberately NOT picked up — the
    set of tracked source paths is owned by ``scaffold.py`` (initial
    write) and any subsequent explicit re-scaffolds.

    Fails closed if a previously tracked path has gone missing on disk.
    """
    tracked: Dict[str, str] = {}
    missing: List[str] = []
    for src in prior_sources:
        rel = src.get("path")
        if not isinstance(rel, str):
            raise PackError(
                f"solution.json sources[] entry missing 'path' string: {src!r}."
            )
        if rel not in disk:
            missing.append(rel)
            continue
        tracked[rel] = disk[rel]
    if missing:
        raise PackError(
            f"Disk files referenced by solution.json.sources[] are missing: "
            f"{missing}. Restore them or re-scaffold the baseline."
        )
    if entry_path not in tracked:
        raise PackError(
            f"Entry source {entry_path!r} (from spec.entry_point) is not "
            f"present in the tracked sources set. Either re-scaffold or "
            f"adjust the entry_point."
        )
    return tracked


def _canonical_source_order(
    tracked: Dict[str, str], entry_path: str
) -> List[Tuple[str, str]]:
    """Entry source first; remaining sources sorted lex by ``path``."""
    remaining = sorted(p for p in tracked if p != entry_path)
    ordered: List[Tuple[str, str]] = [(entry_path, tracked[entry_path])]
    for path in remaining:
        ordered.append((path, tracked[path]))
    return ordered


def pack(baseline_dir: Path) -> Path:
    """Rewrite ``baseline_dir/solution.json`` from disk; return its path."""
    solution_path = baseline_dir / "solution.json"
    if not solution_path.is_file():
        raise PackError(
            f"solution.json not found at {solution_path}. Run scaffold.py "
            f"first to materialise the baseline tree."
        )
    solution_dict = json.loads(solution_path.read_text(encoding="utf-8"))
    entry_path = _entry_path_from_spec(solution_dict)
    prior_sources = solution_dict.get("sources", [])
    if not isinstance(prior_sources, list) or not prior_sources:
        raise PackError(
            "solution.json sources[] must be a non-empty list (Pydantic "
            "Solution requires min_length=1)."
        )
    disk = _walk_disk_sources(baseline_dir)
    tracked = _select_tracked_sources(prior_sources, disk, entry_path)
    ordered = _canonical_source_order(tracked, entry_path)
    solution_dict["sources"] = [
        {"path": path, "content": content} for path, content in ordered
    ]
    # Stable serialisation: indent=2, preserve dict insertion order
    # (the live Solution model emits fields in declaration order).
    solution_path.write_text(
        json.dumps(solution_dict, indent=2) + "\n",
        encoding="utf-8",
    )
    return solution_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/sol_baselines_scaffold/pack.py",
        description=(
            "Regenerate solution.json.sources[].content from disk in "
            "deterministic order. Idempotent."
        ),
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help=(
            "Path to baselines/<cat>/<slug>/<lang>/ — the directory "
            "containing solution.json and the editable source files."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        out = pack(args.baseline)
    except PackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"packed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
