"""Operator-facing cross-host admitted-cell merge tool.

The operator runs this CLI on the receiving host (typically the
RTX 5060 laptop after a sm_120 sweep) to merge a new
`label_admitted.json` produced by a sweep with the existing
operator-facing `data/local_labels/label_admitted.json`. The
tool:

1. Reads `--existing` (the current operator-facing admitted file).
2. Reads `--new-from-sweep` (the sweep run's emitted admitted
   file).
3. Detects duplicate (slug, shape_point, arch) tuples across the
   two inputs; rejects with a typed diagnostic before writing.
4. Concatenates `cells` lists (existing first, then new),
   preserving order.
5. Validates the merged result against
   `workloads/simbench_bw/schemas/local_label_admitted.schema.json`.
6. Writes the merged file to `--out` (defaults to
   `data/local_labels/label_admitted.json`).
7. Emits a one-line `merge_admitted_cells: merged sm_86=N
   sm_120=M total=K (schema-valid)` verdict on stdout.

The tool replaces two inline `python3 - <<'PY' ... PY` heredoc
snippets in `tools/local_labels/SWEEP_RUNBOOK.md`'s cross-host
sm_120 pivot workflow (the merge step + the schema-validate step)
with a single bounded CLI surface so the operator's reading-path
is one command, not a paste-able recipe.

CLI:

  python3 tools/local_labels/merge_admitted_cells.py \
      --existing <path> --new-from-sweep <path> \
      [--out <path>] \
      [--schema-path <path>] \
      [--quiet]

Exit codes:
- 0 on successful merge + schema-validation.
- 2 on any schema/io error (missing input, malformed input JSON,
  schema validation failure, duplicate-cell rejection).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import jsonschema


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
# Shared mixed-arity corpus-cell identity helper (sibling import) so the merge
# key reasons about base-vs-variant identically to the aggregator / selector /
# staging / validator / ledger — through the ONE corpus_cell_key, no re-inlined
# copy of the (slug, shape, arch[, variant_id]) tuple logic.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from corpus_cell_identity import corpus_cell_key  # noqa: E402
_DEFAULT_OUT_PATH = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
_DEFAULT_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "local_label_admitted.schema.json"
)


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"merge_admitted_cells: {label} file not found at {path!s}",
        )
    with path.open("r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"merge_admitted_cells: {label} file at {path!s} did not "
                f"parse as JSON: {exc}",
            ) from exc


def _cell_key(cell: dict) -> tuple:
    """Mixed-arity duplicate-detection key via the shared ``corpus_cell_key``
    helper (no re-inlined identity logic). A base cell keys on the 3-tuple
    ``(arch, slug, shape_point)``; a register-cap build variant widens to
    ``(arch, slug, shape_point, variant_id)`` so sibling caps + a base/variant
    pair of the same (arch, slug, shape) coexist and only a TRUE same-identity
    duplicate is rejected. ``corpus_cell_key`` raises ``ValueError`` on a
    noncanonical / inconsistent register_cap/variant_id pair (the inputs are
    schema-valid per-field, but the schema does not enforce cross-field
    agreement, so this is the consistency gate); the merge re-wraps it into
    its typed diagnostic with the cell's (slug, shape, arch) context."""
    try:
        return corpus_cell_key(cell, with_arch=True)
    except ValueError as err:
        raise ValueError(
            f"merge_admitted_cells: cell "
            f"(slug={cell.get('slug')!r}, shape_point={cell.get('shape_point')!r}, "
            f"arch={cell.get('arch')!r}) has an inconsistent build-variant "
            f"identity: {err}"
        ) from err


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a sweep-produced admitted-cell file with the "
            "operator-facing data/local_labels/label_admitted.json "
            "and validate the result against the strict schema."
        ),
    )
    parser.add_argument(
        "--existing",
        type=Path,
        required=True,
        help=(
            "Path to the existing operator-facing admitted file "
            "(typically data/local_labels/label_admitted.json)."
        ),
    )
    parser.add_argument(
        "--new-from-sweep",
        type=Path,
        required=True,
        help=(
            "Path to the sweep run's emitted label_admitted.json "
            "(under data/local_labels/sweeps/<sweep_dir>/...)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT_PATH,
        help=(
            f"Path to write the merged admitted file (default: "
            f"{_DEFAULT_OUT_PATH!s})."
        ),
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=_DEFAULT_SCHEMA_PATH,
        help=(
            f"Path to the strict admitted-cell schema (default: "
            f"{_DEFAULT_SCHEMA_PATH!s})."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Emit only the verdict token on stdout (suppress "
            "verbose detail)."
        ),
    )
    return parser


def _validate_input(payload: dict, label: str, path: Path,
                    schema: dict) -> None:
    """Validate an input admitted-file payload against the strict
    schema. Raises ``ValueError`` with a typed
    ``merge_admitted_cells:`` prefix on failure so the caller can
    convert it into rc=2 + stderr message.

    The strict schema rejects non-object roots, ``cells`` that is
    not an array, and cell entries that are not objects matching
    ``$defs.admitted_cell``, so the schema catches every
    structurally invalid shape the duplicate-cell scan and
    ``_cell_key()`` helper assumed away."""
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(
            f"merge_admitted_cells: {label} file at {path!s} failed "
            f"strict-schema validation: {exc.message}",
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        existing = _load_json(args.existing, "existing")
        new = _load_json(args.new_from_sweep, "new-from-sweep")
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    try:
        schema = _load_json(args.schema_path, "schema")
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    try:
        _validate_input(existing, "existing", args.existing, schema)
        _validate_input(new, "new-from-sweep", args.new_from_sweep, schema)
    except jsonschema.SchemaError as exc:
        sys.stderr.write(
            f"merge_admitted_cells: schema at {args.schema_path!s} is "
            f"not a valid JSON Schema: {exc.message}\n",
        )
        return 2
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    existing_cells = existing.get("cells", []) or []
    new_cells = new.get("cells", []) or []

    try:
        existing_keys = {_cell_key(c) for c in existing_cells}
        new_keys = [_cell_key(c) for c in new_cells]
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    # Reject a duplicate WITHIN --new-from-sweep (same mixed-arity identity
    # appearing twice in the sweep payload) BEFORE writing — otherwise two
    # same-variant rows in the new file would be silently concatenated into the
    # corpus. (The cross-existing/new check below catches existing-vs-new dups.)
    intra_new_dups = sorted({k for k, n in Counter(new_keys).items() if n > 1})
    if intra_new_dups:
        # corpus_cell_key(with_arch=True) is (arch, slug, shape_point[, vid]);
        # slug is at index 1.
        slugs = sorted({k[1] for k in intra_new_dups})
        sys.stderr.write(
            f"merge_admitted_cells: duplicate cell(s) WITHIN "
            f"--new-from-sweep (slug={slugs}): refusing to merge\n",
        )
        return 2
    duplicates = [k for k in new_keys if k in existing_keys]
    if duplicates:
        slugs = sorted({k[1] for k in duplicates})
        sys.stderr.write(
            f"merge_admitted_cells: duplicate cell(s) across "
            f"--existing and --new-from-sweep "
            f"(slug={slugs}): refusing to merge\n",
        )
        return 2

    merged_cells = list(existing_cells) + list(new_cells)
    merged = {"schema_version": 5, "cells": merged_cells}

    try:
        jsonschema.validate(instance=merged, schema=schema)
    except jsonschema.SchemaError as exc:
        sys.stderr.write(
            f"merge_admitted_cells: schema at {args.schema_path!s} is "
            f"not a valid JSON Schema: {exc.message}\n",
        )
        return 2
    except jsonschema.ValidationError as exc:
        sys.stderr.write(
            f"merge_admitted_cells: merged admitted file failed "
            f"strict-schema validation: {exc.message}\n",
        )
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    counts = Counter(c.get("arch", "") for c in merged_cells)
    sm86 = counts.get("sm_86", 0)
    sm120 = counts.get("sm_120", 0)
    total = len(merged_cells)
    verdict = (
        f"merge_admitted_cells: merged sm_86={sm86} sm_120={sm120} "
        f"total={total} (schema-valid)"
    )
    if args.quiet:
        sys.stdout.write(verdict + "\n")
    else:
        sys.stdout.write(verdict + "\n")
        sys.stdout.write(f"merge_admitted_cells: wrote {args.out!s}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
