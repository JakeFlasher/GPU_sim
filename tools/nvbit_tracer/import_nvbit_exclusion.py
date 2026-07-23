"""Importer for NVBit tracer exclusion sidecars.

Reads a JSON sidecar emitted by the NVBit tracer when it cannot
truthfully represent an instruction's memory-operation class
(``unsupported_memory_path``), cannot resolve the access target to
an allocator-tracked memory range (``unmapped_memory_range``), or
observes a single warp instruction whose active lanes span multiple
distinct non-zero allocations (``cross_allocation_lane_span``). All
three shapes share the same set of structural fields and are
validated against per-typed-reason JSON Schema documents under
``workloads/simbench_bw/schemas/``:

- ``nvbit_unsupported_memory_path.schema.json``
- ``nvbit_unmapped_memory_range.schema.json``
- ``nvbit_cross_allocation_lane_span.schema.json``

The importer dispatches on the ``typed_reason`` field so the same
entry point validates any of the three sidecar shapes. A wrong
literal in ``typed_reason`` (or a sidecar that omits it) is rejected
with a typed reason rather than silently misclassified.

The producer (`tools/nvbit_tracer/tracer_tool/exclusion_aggregator.cc`)
emits each typed-reason sidecar as a JSON array of entries, one entry
per (kernel, instruction_pc, op_class) key. ``validate(entry)`` is
the per-entry public API for callers that already have one entry in
hand; ``validate_path(path)`` validates a producer-shaped file: it
admits an empty array (the producer's "no entries this run"
contract), validates each entry through ``validate(entry)`` with its
index injected into the diagnostic, and rejects mixed typed reasons
within a single file (each typed-reason file must be homogeneous).

Public entry points:

- ``load_schema(typed_reason)`` returns the bundled JSON Schema.
- ``validate(entry)`` runs all checks against a single in-memory
  entry dict.
- ``validate_path(path)`` opens the file and validates the array
  shape + each entry.
- ``main(argv)`` is the CLI entry point; exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCHEMA_DIR = _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"

_SCHEMA_FILES: dict[str, str] = {
    "unsupported_memory_path": "nvbit_unsupported_memory_path.schema.json",
    "unmapped_memory_range": "nvbit_unmapped_memory_range.schema.json",
    "cross_allocation_lane_span":
        "nvbit_cross_allocation_lane_span.schema.json",
}

VALID_OP_CLASSES: tuple[str, ...] = (
    "GLOBAL",
    "SHARED",
    "LOCAL",
    "TEXTURE",
    "SURFACE",
    "ATOMIC",
    "TMA",
    "ASYNC_COPY",
    "CLUSTER_DSMEM",
    # Blackwell (sm_100) tensor-memory storage class. Producer-
    # side classifier in
    # `tools/nvbit_tracer/tracer_tool/memory_op_class.h` routes
    # the `tcgen05.{ld,st}` family (mnemonics `LDT` / `STT`)
    # to MemoryOpClass::kTmem; consumer-side widening here
    # admits the corresponding string label per
    # BL-20260507-multi-surface-lockstep-changes (producer +
    # consumer extension land in the same commit).
    "TMEM",
    # `UNSPECIFIED` is the sentinel for STANDARD_MEM opcodes the
    # classifier could not match against any known prefix. The
    # producer-side typed exclusion routes those through this
    # sidecar so the validator does not reject them as
    # `memory_op_class_missing_on_memory_instruction`.
    "UNSPECIFIED",
)


@dataclass
class ValidationResult:
    """Outcome of validating an exclusion sidecar.

    On success ``ok`` is True and ``typed_reason`` carries the
    sidecar's typed reason value (one of the keys of ``_SCHEMA_FILES``,
    currently ``unsupported_memory_path``, ``unmapped_memory_range``,
    or ``cross_allocation_lane_span``).

    ``entries`` carries the number of entries the validator
    iterated when called via :func:`validate_path` (zero for empty
    `[]` arrays, the entry count for non-empty homogeneous files,
    and zero on any failure path). Single-entry callers using
    :func:`validate` directly leave ``entries`` at its default of
    zero.

    On failure ``typed_reason`` is one of:

    - ``sidecar_unreadable`` — file does not exist or is not valid
      JSON.
    - ``sidecar_not_a_list`` — the file's top-level JSON value is
      not an array. The producer aggregator emits each typed-reason
      sidecar as a JSON list (possibly empty); a top-level object
      is the legacy single-entry shape that pre-Round-6 callers
      used and is no longer accepted by ``validate_path``.
    - ``sidecar_typed_reason_mismatch`` — entries within a single
      file carry different ``typed_reason`` values. Each per-typed-
      reason file (`unsupported_memory_path.json`,
      `unmapped_memory_range.json`,
      `cross_allocation_lane_span.json`) must be homogeneous; the
      producer partitions entries by typed_reason and writes each
      partition to its own file.
    - ``schema_validation_failed`` — JSON Schema validation failed
      against the dispatched schema. ``detail`` carries the entry
      index when the failure was raised through ``validate_path``.
    - ``typed_reason_missing`` — entry does not carry a
      ``typed_reason`` field, so dispatch is impossible.
    - ``typed_reason_unknown`` — entry's ``typed_reason`` does not
      name a known schema.
    """

    ok: bool
    typed_reason: str = ""
    detail: str = ""
    entries: int = 0


def load_schema(typed_reason: str) -> dict[str, Any]:
    """Return the bundled JSON Schema document for ``typed_reason``.

    Raises ``KeyError`` if ``typed_reason`` is not one of the known
    exclusion reasons (see ``_SCHEMA_FILES``). Most callers should
    go through ``validate`` instead, which translates an unknown
    reason into a typed ``ValidationResult`` rather than raising.
    """
    schema_filename = _SCHEMA_FILES[typed_reason]
    with (_SCHEMA_DIR / schema_filename).open(encoding="utf-8") as fh:
        return json.load(fh)


def _typed_reason(entry: dict[str, Any]) -> str | None:
    value = entry.get("typed_reason")
    if isinstance(value, str):
        return value
    return None


def validate(entry: dict[str, Any]) -> ValidationResult:
    """Validate a single in-memory ``entry`` dict against the schema
    dispatched by its ``typed_reason`` field. Returns a
    ``ValidationResult``. Callers with a producer-shaped file should
    use :func:`validate_path` instead.
    """
    typed_reason = _typed_reason(entry)
    if typed_reason is None:
        return ValidationResult(
            ok=False,
            typed_reason="typed_reason_missing",
            detail="entry does not carry a 'typed_reason' string field",
        )
    if typed_reason not in _SCHEMA_FILES:
        return ValidationResult(
            ok=False,
            typed_reason="typed_reason_unknown",
            detail=(
                f"entry typed_reason={typed_reason!r} is not one of "
                f"{tuple(_SCHEMA_FILES)}"
            ),
        )
    schema = load_schema(typed_reason)
    try:
        jsonschema.validate(entry, schema)
    except jsonschema.ValidationError as exc:
        return ValidationResult(
            ok=False,
            typed_reason="schema_validation_failed",
            detail=str(exc.message),
        )
    return ValidationResult(ok=True, typed_reason=typed_reason)


def validate_path(path: Path) -> ValidationResult:
    """Read ``path`` as JSON and validate the producer-emitted file
    shape: a top-level array of per-entry dicts, all sharing the
    same ``typed_reason``.

    Returns ``ValidationResult(ok=True, typed_reason=<reason>)`` on
    success. For an empty array, ``typed_reason`` is empty and
    ``detail`` is ``"sidecar_empty"`` so the success path stays
    distinguishable from a non-empty homogeneous file.

    Failure shapes:

    - ``sidecar_unreadable`` if the file is missing or not valid
      JSON.
    - ``sidecar_not_a_list`` if the top-level JSON value is not an
      array.
    - Any per-entry typed reason from :func:`validate` (with
      ``detail`` prefixed by ``entry[i]:``) if a specific entry
      fails.
    - ``sidecar_typed_reason_mismatch`` if entries carry different
      ``typed_reason`` values within the same file.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationResult(
            ok=False,
            typed_reason="sidecar_unreadable",
            detail=f"could not read {path}: {exc}",
        )
    if not isinstance(payload, list):
        return ValidationResult(
            ok=False,
            typed_reason="sidecar_not_a_list",
            detail=(
                f"{path}: top-level JSON value is "
                f"{type(payload).__name__}, expected array; the "
                f"producer aggregator emits each typed-reason "
                f"sidecar as a JSON list (possibly empty)"
            ),
        )
    if len(payload) == 0:
        return ValidationResult(
            ok=True, typed_reason="", detail="sidecar_empty",
            entries=0,
        )
    file_typed_reason: str | None = None
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            return ValidationResult(
                ok=False,
                typed_reason="schema_validation_failed",
                detail=(
                    f"{path}: entry[{index}] is "
                    f"{type(entry).__name__}, expected object"
                ),
            )
        result = validate(entry)
        if not result.ok:
            return ValidationResult(
                ok=False,
                typed_reason=result.typed_reason,
                detail=f"{path}: entry[{index}]: {result.detail}",
            )
        if file_typed_reason is None:
            file_typed_reason = result.typed_reason
        elif result.typed_reason != file_typed_reason:
            return ValidationResult(
                ok=False,
                typed_reason="sidecar_typed_reason_mismatch",
                detail=(
                    f"{path}: entry[{index}] typed_reason="
                    f"{result.typed_reason!r} does not match the "
                    f"file's first entry typed_reason="
                    f"{file_typed_reason!r}; each per-typed-reason "
                    f"sidecar file must be homogeneous"
                ),
            )
    assert file_typed_reason is not None
    return ValidationResult(
        ok=True,
        typed_reason=file_typed_reason,
        entries=len(payload),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an NVBit tracer exclusion sidecar file "
            "(unsupported_memory_path.json, "
            "unmapped_memory_range.json, or "
            "cross_allocation_lane_span.json) — a JSON array of "
            "per-entry dicts."
        )
    )
    parser.add_argument("sidecar", type=Path)
    args = parser.parse_args(argv)

    result = validate_path(args.sidecar)
    if not result.ok:
        sys.stderr.write(
            f"import_nvbit_exclusion: {result.typed_reason}: "
            f"{result.detail}\n"
        )
        return 1
    if result.detail == "sidecar_empty":
        sys.stdout.write(
            f"ok: {args.sidecar} is empty (entries=0)\n"
        )
    else:
        sys.stdout.write(
            f"ok: {args.sidecar} typed_reason="
            f"{result.typed_reason} entries={result.entries}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
