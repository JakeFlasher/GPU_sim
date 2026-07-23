"""Buildability / collection-readiness preflight for the JSON tri-label roster.

The earlier `sm120_cohort_buildability_preflight.py` reads the legacy TOML
roster (`sm120_cohort_expansion_roster.toml`). This preflight reads the new
frozen JSON roster (`data/local_labels/sm120_tri_label_roster.json`, `cells[]`)
and reports, per cell, whether the live collection (NVBit trace + NCU L1/L2 +
CUPTI duration) can run:

1. the cell's `binary_path` exists on disk (or builds via the per-slug
   `make` when ``--build-missing`` is set),
2. the NVBit tracer `.so` is built (so the trace can be produced at sweep time),
3. the slug source directory (where the per-slug Makefile lives) exists,
4. the cell carries the trace/manifest paths the sweep will populate.

A cell with no failure tokens is ``collection_ready``. The witness pins the
roster SHA-256 and the roster's own ``source_commit`` so a stale roster is
visible. Binaries themselves are gitignored host-local build artifacts; only the
witness JSON is committed.

Usage:

    python tools/local_labels/sm120_tri_label_buildability_preflight.py \\
        --roster data/local_labels/sm120_tri_label_roster.json \\
        --output data/local_labels/sm120_tri_label_buildability_witness.json \\
        [--build-missing]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
NVBIT_TRACER_SO = (
    REPO_ROOT / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def _is_unsafe_rel_path(rel: str, repo_root: Path) -> bool:
    """A roster artifact path must be a safe repo-relative path. Reject
    absolute paths, parent-traversal (``..``) components, AND any path that
    — after resolving symlinks under ``repo_root`` — escapes the repo root.
    The lexical checks alone are insufficient: a repo-relative path through
    a symlinked component can still resolve outside the repo."""
    candidate = Path(rel)
    if candidate.is_absolute() or ".." in candidate.parts:
        return True
    try:
        resolved = (repo_root / candidate).resolve(strict=False)
        resolved.relative_to(repo_root.resolve())
    except (ValueError, OSError):
        return True
    return False


def _run_make_for_slug(slug_dir: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["make"], cwd=slug_dir, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return False, "make timed out after 300s"
    except (FileNotFoundError, OSError) as exc:
        return False, f"make invocation failed: {exc}"
    transcript = (result.stdout + result.stderr).strip()
    tail = transcript.splitlines()[-1] if transcript else "no_output"
    return result.returncode == 0, tail


def check_cell(
    cell: dict,
    *,
    build_missing: bool,
    repo_root: Path,
    tracer_so: Path,
) -> dict:
    """Return a witness row for one roster cell."""
    out: dict = {
        "slug": cell.get("slug"),
        "shape_point": cell.get("shape_point"),
        "family_bucket": cell.get("family_bucket"),
        "split": cell.get("split"),
        "roster_source": cell.get("roster_source"),
        "manifest_path": cell.get("manifest_path"),
        "expected_trace_json_path": cell.get("trace_json_path"),
        "resolved_binary_path": None,
        "binary_sha256_if_built": None,
        "failure_tokens": [],
    }

    binary_rel = cell.get("binary_path") or ""
    manifest_rel = cell.get("manifest_path") or ""
    trace_rel = cell.get("trace_json_path") or ""

    # Path contract: the live sweep needs each cell to name where its
    # trace + manifest land, and every roster path must be a safe
    # repo-relative path. A missing or unsafe path is not collection-ready.
    if not binary_rel:
        out["failure_tokens"].append("binary_path_absent_in_roster")
    if not manifest_rel:
        out["failure_tokens"].append("manifest_path_absent_in_roster")
    if not trace_rel:
        out["failure_tokens"].append("trace_json_path_absent_in_roster")
    for field, rel in (
        ("binary_path", binary_rel),
        ("manifest_path", manifest_rel),
        ("trace_json_path", trace_rel),
    ):
        if rel and _is_unsafe_rel_path(rel, repo_root):
            out["failure_tokens"].append(f"unsafe_path:{field}")

    # The binary presence/build check only runs when binary_path is a
    # present, safe, repo-relative path.
    if binary_rel and not _is_unsafe_rel_path(binary_rel, repo_root):
        binary_abs = repo_root / binary_rel
        slug_dir = binary_abs.parent
        if not binary_abs.is_file() and build_missing and slug_dir.is_dir():
            success, tail = _run_make_for_slug(slug_dir)
            if not success:
                out["failure_tokens"].append(f"make_failed: {tail}")
        if binary_abs.is_file():
            out["resolved_binary_path"] = binary_rel
            out["binary_sha256_if_built"] = _sha256_file(binary_abs)
        else:
            out["failure_tokens"].append("binary_path_missing")
        if not slug_dir.is_dir():
            out["failure_tokens"].append("slug_source_dir_missing")

    # The manifest + trace must actually EXIST on disk for the cell to be
    # collection-ready -- a present, safe path STRING is not enough. The live
    # sweep CONSUMES these files (the measurement manifest pins
    # kernel_signature_sha256; the NVBit trace is the cell's trace leg), so a
    # missing input fails the sweep per-cell. Existence is checked only when the
    # path is present AND safe (an absent/unsafe path already emitted its own
    # token above; checking is.file() on an unsafe path would itself be unsafe).
    for field, rel, missing_token in (
        ("manifest_path", manifest_rel, "manifest_path_missing"),
        ("trace_json_path", trace_rel, "trace_json_path_missing"),
    ):
        if rel and not _is_unsafe_rel_path(rel, repo_root):
            if not (repo_root / rel).is_file():
                out["failure_tokens"].append(missing_token)

    if not tracer_so.is_file():
        out["failure_tokens"].append("nvbit_tracer_so_missing")

    return out


def preflight(
    roster_path: Path,
    *,
    build_missing: bool = False,
    repo_root: Path = REPO_ROOT,
    tracer_so: Path = NVBIT_TRACER_SO,
) -> dict:
    """Return the witness dict for the roster at ``roster_path``."""
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    cells = roster.get("cells", [])
    rows = [
        check_cell(
            cell,
            build_missing=build_missing,
            repo_root=repo_root,
            tracer_so=tracer_so,
        )
        for cell in cells
    ]
    for row in rows:
        row["collection_ready"] = not row["failure_tokens"]
    ready = sum(1 for r in rows if r["collection_ready"])
    binaries_built = sum(1 for r in rows if r["resolved_binary_path"] is not None)
    return {
        "preflight_tool_version": __version__,
        "source_commit": _git_head_sha(repo_root),
        "roster_source_commit": roster.get("source_commit"),
        "roster_sha256": _sha256_file(roster_path),
        "total_cells": len(rows),
        "collection_ready_cells": ready,
        "binaries_built": binaries_built,
        "binaries_missing": len(rows) - binaries_built,
        "all_collection_ready": ready == len(rows) and len(rows) > 0,
        "cells": rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path("data/local_labels/sm120_tri_label_roster.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/local_labels/sm120_tri_label_buildability_witness.json"),
    )
    parser.add_argument("--build-missing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    witness = preflight(args.roster, build_missing=args.build_missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n")
    print(
        f"preflight: {witness['collection_ready_cells']}/{witness['total_cells']} "
        f"cells collection-ready; binaries {witness['binaries_built']} built / "
        f"{witness['binaries_missing']} missing; witness -> {args.output}"
    )
    if not witness["all_collection_ready"]:
        for row in witness["cells"]:
            if row["failure_tokens"]:
                print(
                    f"  NOT READY: {row['slug']}/{row['shape_point']}: {row['failure_tokens']}",
                    file=sys.stderr,
                )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
