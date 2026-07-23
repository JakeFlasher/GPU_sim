"""Backfill trace-byproduct `manifest_pairing_index.json` sidecars
for every extant host-local trace directory.

Per Codex's round-22 review, the round-22 emitter helper in
`workloads/simbench_bw/run_suite.py::_emit_byproduct_manifest_pairing_index`
fires on every future live trace capture, but the existing
host-local trace directories under
`workloads/simbench_bw/results/*/rtx_3090_ti/sm_86/trace_trace_l2_accuracy_xl/`
only had the carveout cell backfilled. This script discovers every
extant trace directory + parent trace JSON, reads the parent JSON's
5-field identity tuple (`slug`, `shape_point`, `binary_sha256`,
`source_commit`, `kernel_signature_sha256`), and writes
`manifest_pairing_index.json` to each byproduct subtree
(`sector_stream/`, `extra_info/`).

The output sidecars are gitignored host-local artifacts (the
trace results tree is gitignored); this is a one-shot operator
script for closing Codex's round-22 mainline gap.

Usage:
    python3 tools/local_labels/backfill_trace_byproduct_mpi_sidecars.py

Exits 0 on success, non-zero if any parent trace JSON is missing
required identity fields.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOST_LOCAL_RESULTS_ROOT = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "results"
)
_TRACE_DIR_NAME = "trace_trace_l2_accuracy_xl"
_TRACE_JSON_NAME = f"{_TRACE_DIR_NAME}.json"
_BYPRODUCT_DIRS = ("sector_stream", "extra_info")
_REQUIRED_KEYS = (
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
)


def _discover() -> list[tuple[Path, Path]]:
    """Return list of (trace_dir, parent_trace_json) pairs."""
    out: list[tuple[Path, Path]] = []
    if not _HOST_LOCAL_RESULTS_ROOT.is_dir():
        return out
    for cell_dir in sorted(_HOST_LOCAL_RESULTS_ROOT.iterdir()):
        if not cell_dir.is_dir():
            continue
        td = cell_dir / "rtx_3090_ti" / "sm_86" / _TRACE_DIR_NAME
        pj = cell_dir / "rtx_3090_ti" / "sm_86" / _TRACE_JSON_NAME
        if td.is_dir() and pj.is_file():
            out.append((td, pj))
    return out


def _extract_mpi(parent_json: dict) -> tuple[dict | None, str | None]:
    """Extract the 5-field MPI tuple from the parent trace JSON.

    Returns (mpi_dict, error_reason). `error_reason` is non-None when
    a required field is missing or non-string.
    """
    mpi: dict = {}
    for k in _REQUIRED_KEYS:
        v = parent_json.get(k)
        if v is None:
            return None, f"missing required field {k!r} in parent JSON"
        if not isinstance(v, str):
            return None, (
                f"field {k!r} in parent JSON is not a string: {v!r}"
            )
        mpi[k] = v
    return mpi, None


def main() -> int:
    discovered = _discover()
    if not discovered:
        print(
            "backfill_trace_byproduct_mpi_sidecars: no host-local "
            "trace directories under "
            f"{_HOST_LOCAL_RESULTS_ROOT}; nothing to do",
            file=sys.stderr,
        )
        return 0
    written = 0
    skipped_no_byproduct = 0
    errors: list[str] = []
    for trace_dir, parent_json_path in discovered:
        with parent_json_path.open("r", encoding="utf-8") as fh:
            parent = json.load(fh)
        mpi, err = _extract_mpi(parent)
        if err is not None:
            errors.append(
                f"{trace_dir.parts[-4]}: {err} "
                f"(parent {parent_json_path})"
            )
            continue
        for byproduct_dir in _BYPRODUCT_DIRS:
            target_dir = trace_dir / byproduct_dir
            if not target_dir.is_dir():
                skipped_no_byproduct += 1
                continue
            target = target_dir / "manifest_pairing_index.json"
            with target.open("w", encoding="utf-8") as fh:
                json.dump(mpi, fh, indent=2, sort_keys=True)
                fh.write("\n")
            written += 1
            print(
                f"wrote {target.relative_to(_REPO_ROOT)}",
            )
    print(
        f"\nbackfill summary: {written} sidecars written, "
        f"{skipped_no_byproduct} byproduct subtrees skipped (missing), "
        f"{len(errors)} errors"
    )
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
