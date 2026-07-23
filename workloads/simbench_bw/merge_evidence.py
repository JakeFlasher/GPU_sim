#!/usr/bin/env python3
"""Historical CUPTI/NVBit two-run evidence replay join.

This tool joins frozen sidecars from the retired counter-free workflow. That
workflow ran the same kernel twice because CUPTI Activity and NVBit injection
were intentionally kept in separate producer processes. This module performs
offline replay only and never invokes either producer. It joins the archived
runs by the canonical launch key:

    (slug, shape_point, repeat_index, launch_index, kernel_signature_sha256)

It validates both historical sidecars, checks binary identity, rejects
unmatched or duplicated launches, and writes joined records only when
the entire input set is conflict-free.

Usage:
    python3 merge_evidence.py join-cupti-nvbit \
        --cupti-dir <dir> --nvbit-dir <dir> --output-dir <dir> \
        --platform {h100|rtx_pro_6000_blackwell|rtx_5060_laptop|rtx_3090_ti}

Exit codes:
    0  every CUPTI/NVBit launch joined successfully.
    1  conflict requires manual resolution; no joined files were written.
    2  CLI usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parent
PLATFORM_KEYS: frozenset[str] = frozenset({
    "h100",
    "rtx_pro_6000_blackwell",
    "rtx_5060_laptop",
    "rtx_3090_ti",
})


# ---------------------------------------------------------------------------
# join-cupti-nvbit subcommand: NVBit + CUPTI two-run offline join.
# ---------------------------------------------------------------------------

JOINED_EVIDENCE_SCHEMA_VERSION: int = 1

# Canonical 5-tuple join key
# (`(slug, shape_point, repeat_index, launch_index, kernel_signature_sha256)`).
# The first 4 components form the "natural key" (the same kernel launch
# in two identical runs); the 5th component is the load-bearing
# integrity check — if the same natural key appears on both sides with
# different signature hashes, the two runs DID NOT produce bit-
# identical kernel shapes and the join is invalid.
_JOIN_NATURAL_KEY_FIELDS: tuple[str, ...] = (
    "slug", "shape_point", "repeat_index", "launch_index",
)
_JOIN_INTEGRITY_KEY_FIELD: str = "kernel_signature_sha256"

_CUPTI_SIDECAR_SCHEMA_PATH = (
    SUITE_ROOT
    / "schemas"
    / "historical"
    / "cupti_activity_sidecar.v1.schema.json"
)
_NVBIT_TRACE_SUMMARY_SCHEMA_PATH = SUITE_ROOT / "schemas" / "nvbit_trace_summary.schema.json"


def _load_join_schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the historical CUPTI and retained NVBit replay schemas."""
    with open(_CUPTI_SIDECAR_SCHEMA_PATH) as f:
        cupti_schema = json.load(f)
    with open(_NVBIT_TRACE_SUMMARY_SCHEMA_PATH) as f:
        nvbit_schema = json.load(f)
    return cupti_schema, nvbit_schema


def _allowed_platforms_for_join() -> frozenset[str]:
    """Canonical platform set for joined output paths."""
    return PLATFORM_KEYS


def _discover_json_files(root: Path, glob_pattern: str) -> list[Path]:
    """Recursively discover files matching glob_pattern under root."""
    return sorted(root.rglob(glob_pattern))


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Stdlib fallback validators for the fields the join consumes
# directly. These guarantee schema-invalid conflicts fire even when
# `jsonschema` is not installed in the runtime environment. The
# fallback does NOT replace jsonschema for full-schema enforcement;
# it covers the join-relevant subset and runs FIRST so a deterministic
# token is emitted regardless of jsonschema availability. When
# jsonschema is available it runs as a stricter superset.

_NVBIT_SUMMARY_REQUIRED_FIELDS: dict[str, type] = {
    "schema_version": int,
    "slug": str,
    "shape_point": str,
    "repeat_index": int,
    "launch_index": int,
    "kernel_name": str,
    "kernel_signature_sha256": str,
    "inst_count": int,
    "binary_sha256": str,
    "nvbit_trace_pb_path": str,
    "nvbit_trace_pb_sha256": str,
    "nvbit_tracer_build_hash": str,
}
_NVBIT_SUMMARY_SHA256_FIELDS: tuple[str, ...] = (
    "kernel_signature_sha256", "binary_sha256", "nvbit_trace_pb_sha256",
)
_NVBIT_SUMMARY_NONNEGATIVE_INTS: tuple[str, ...] = (
    "schema_version", "repeat_index", "launch_index", "inst_count",
)
_NVBIT_SUMMARY_MIN_LENGTH_STRINGS: tuple[str, ...] = (
    "slug", "shape_point", "kernel_name",
    "nvbit_trace_pb_path", "nvbit_tracer_build_hash",
)

_CUPTI_JOIN_RELEVANT_REQUIRED_FIELDS: dict[str, type] = {
    "slug": str,
    "shape_point": str,
    "repeat_index": int,
    "launch_index": int,
    "kernel_signature_sha256": str,
    "binary_sha256": str,
}
_CUPTI_JOIN_RELEVANT_SHA256_FIELDS: tuple[str, ...] = (
    "kernel_signature_sha256", "binary_sha256",
)


def _validate_nvbit_summary_stdlib(record: dict[str, Any]) -> str | None:
    """Stdlib fallback validator for the NVBit summary schema's
    required fields, types, SHA-256 patterns, and additionalProperties:
    false. Returns None if valid, or a one-line error message if
    invalid (the dispatcher prepends `joined_evidence_schema_invalid_`).
    """
    # Required-field presence + type.
    for field, expected_type in _NVBIT_SUMMARY_REQUIRED_FIELDS.items():
        if field not in record:
            return f"missing required field {field!r}"
        value = record[field]
        # Booleans are a subtype of int in Python — reject them where
        # the schema expects an int (matches jsonschema's draft 2020-12
        # semantics where {"type":"integer"} rejects booleans).
        if expected_type is int and isinstance(value, bool):
            return f"field {field!r} is bool, expected integer"
        if not isinstance(value, expected_type):
            return (
                f"field {field!r} has type {type(value).__name__}, "
                f"expected {expected_type.__name__}"
            )
    # Non-negative integer constraints.
    for field in _NVBIT_SUMMARY_NONNEGATIVE_INTS:
        if record[field] < 0:
            return f"field {field!r} is negative ({record[field]}); minimum 0"
    if record["schema_version"] < 1:
        return f"field 'schema_version' is {record['schema_version']}; minimum 1"
    # Min-length string constraints.
    for field in _NVBIT_SUMMARY_MIN_LENGTH_STRINGS:
        if len(record[field]) < 1:
            return f"field {field!r} is empty; minLength 1"
    # SHA-256 hex pattern (64 lowercase hex chars).
    for field in _NVBIT_SUMMARY_SHA256_FIELDS:
        if not _SHA256_RE.match(record[field]):
            return (
                f"field {field!r} value {record[field]!r} does not match "
                f"^[0-9a-f]{{64}}$"
            )
    # additionalProperties: false.
    extras = set(record) - set(_NVBIT_SUMMARY_REQUIRED_FIELDS)
    if extras:
        return f"additionalProperties not allowed: {sorted(extras)}"
    return None


def _validate_cupti_sidecar_stdlib(record: dict[str, Any]) -> str | None:
    """Stdlib fallback validator for the CUPTI sidecar fields the
    join directly consumes: 5-tuple join key + binary_sha256. Other
    CUPTI fields fall through to jsonschema's full-schema enforcement
    when available. Returns None if valid, or a one-line error
    message if invalid.
    """
    for field, expected_type in _CUPTI_JOIN_RELEVANT_REQUIRED_FIELDS.items():
        if field not in record:
            return f"missing required field {field!r}"
        value = record[field]
        if expected_type is int and isinstance(value, bool):
            return f"field {field!r} is bool, expected integer"
        if not isinstance(value, expected_type):
            return (
                f"field {field!r} has type {type(value).__name__}, "
                f"expected {expected_type.__name__}"
            )
    if record["repeat_index"] < 0 or record["launch_index"] < 0:
        return "field 'repeat_index' or 'launch_index' is negative; minimum 0"
    for field in _CUPTI_JOIN_RELEVANT_SHA256_FIELDS:
        if not _SHA256_RE.match(record[field]):
            return (
                f"field {field!r} value {record[field]!r} does not match "
                f"^[0-9a-f]{{64}}$"
            )
    return None


def _validate_against_schema(
    record: dict[str, Any], schema: dict[str, Any], source_path: Path,
    *, side_label: str,
) -> str | None:
    """Validate a record. Returns None if valid, or a one-line error
    message if invalid. Runs the stdlib fallback FIRST (so the
    join-relevant fields are gated even without jsonschema), then
    runs jsonschema as a stricter superset when importable.
    """
    if side_label == "nvbit":
        stdlib_err = _validate_nvbit_summary_stdlib(record)
    elif side_label == "cupti":
        stdlib_err = _validate_cupti_sidecar_stdlib(record)
    else:
        stdlib_err = None
    if stdlib_err is not None:
        return f"{source_path}: {stdlib_err}"
    try:
        import jsonschema  # local import so the dep stays optional
    except ImportError:
        return None
    try:
        jsonschema.validate(record, schema)
    except jsonschema.ValidationError as e:
        return f"{source_path}: {e.message}"
    return None


def _index_records_by_natural_key_with_counts(
    paths: list[Path],
    schema: dict[str, Any],
    side_label: str,
) -> tuple[dict[tuple, list[dict[str, Any]]], list[str]]:
    """Load each path, schema-validate, and group by the natural-key
    4-tuple `(slug, shape_point, repeat_index, launch_index)`. All
    records on the same natural key are preserved (no deduplication)
    so duplicate 5-tuples within one side are observable downstream
    as launch-count mismatches. Returns (natural_key_to_records,
    schema_invalid_tokens).
    """
    schema_invalid: list[str] = []
    natural_to_records: dict[tuple, list[dict[str, Any]]] = {}
    for path in paths:
        try:
            with open(path) as f:
                record = json.load(f)
        except json.JSONDecodeError as exc:
            schema_invalid.append(
                f"joined_evidence_schema_invalid_{side_label}_{path}: not valid JSON: {exc}"
            )
            continue
        validation_err = _validate_against_schema(
            record, schema, path, side_label=side_label,
        )
        if validation_err is not None:
            schema_invalid.append(
                f"joined_evidence_schema_invalid_{side_label}_{validation_err}"
            )
            continue
        # Build the natural-key 4-tuple from the validated record.
        try:
            natural = tuple(record[f] for f in _JOIN_NATURAL_KEY_FIELDS)
        except KeyError as e:
            schema_invalid.append(
                f"joined_evidence_schema_invalid_{side_label}_{path}: missing required join field {e}"
            )
            continue
        natural_to_records.setdefault(natural, []).append(record)
    return natural_to_records, schema_invalid


def _joined_output_path(
    output_dir: Path, platform: str, record: dict[str, Any],
) -> Path:
    """Output path: <output-dir>/<platform>/<slug>/<shape_point>/joined_<rep>_<launch>.json"""
    return (
        output_dir / platform / record["slug"] / record["shape_point"]
        / f"joined_{record['repeat_index']}_{record['launch_index']}.json"
    )


def _run_join_cupti_nvbit(args: argparse.Namespace) -> int:
    """NVBit + CUPTI two-run offline join."""
    # Hard prerequisite: jsonschema MUST be importable so the full
    # CUPTI sidecar schema (required fields / dim3 sub-objects /
    # activity_kind enum / memcpy/memset arrays / etc.) is enforced
    # before any record is admitted. The stdlib subset validators
    # `_validate_cupti_sidecar_stdlib` + `_validate_nvbit_summary_stdlib`
    # cover the join-relevant fields (5-tuple + binary_sha256) as
    # defense-in-depth but cannot enforce the full schema. Refusing
    # to run without jsonschema is simpler and safer than maintaining
    # a stdlib mirror of the full CUPTI schema.
    try:
        import jsonschema  # noqa: F401  # pyright: ignore[reportMissingModuleSource, reportUnusedImport]
    except ImportError:
        print(
            "joined_evidence_jsonschema_unavailable: the join-cupti-nvbit "
            "subcommand requires the `jsonschema` Python runtime to "
            "enforce the full CUPTI sidecar + NVBit summary schemas. "
            "Install with `pip install jsonschema` and re-run.",
            file=sys.stderr,
        )
        return 2

    cupti_dir = Path(args.cupti_dir)
    nvbit_dir = Path(args.nvbit_dir)
    output_dir = Path(args.output_dir)

    if not cupti_dir.is_dir():
        print(f"joined_evidence_cupti_dir_missing: {cupti_dir}", file=sys.stderr)
        return 2
    if not nvbit_dir.is_dir():
        print(f"joined_evidence_nvbit_dir_missing: {nvbit_dir}", file=sys.stderr)
        return 2

    allowed_platforms = _allowed_platforms_for_join()
    if args.platform not in allowed_platforms:
        print(
            f"joined_evidence_unknown_platform: {args.platform!r} not in {sorted(allowed_platforms)}",
            file=sys.stderr,
        )
        return 2

    cupti_schema, nvbit_schema = _load_join_schemas()
    cupti_paths = _discover_json_files(cupti_dir, "cupti_activity_*.json")
    nvbit_paths = _discover_json_files(nvbit_dir, "nvbit_trace_summary_*.json")

    cupti_natural, cupti_schema_errs = _index_records_by_natural_key_with_counts(
        cupti_paths, cupti_schema, "cupti",
    )
    nvbit_natural, nvbit_schema_errs = _index_records_by_natural_key_with_counts(
        nvbit_paths, nvbit_schema, "nvbit",
    )

    conflicts: list[str] = []
    conflicts.extend(cupti_schema_errs)
    conflicts.extend(nvbit_schema_errs)

    all_natural_keys = sorted(set(cupti_natural) | set(nvbit_natural))

    joined_records: list[tuple[Path, dict[str, Any]]] = []
    for natural in all_natural_keys:
        cupti_records_here = cupti_natural.get(natural, [])
        nvbit_records_here = nvbit_natural.get(natural, [])
        if cupti_records_here and not nvbit_records_here:
            token = "/".join(str(c) for c in natural)
            conflicts.append(f"joined_evidence_unmatched_cupti_{token}")
            continue
        if nvbit_records_here and not cupti_records_here:
            token = "/".join(str(c) for c in natural)
            conflicts.append(f"joined_evidence_unmatched_nvbit_{token}")
            continue
        # Both sides present. Per-side full-key uniqueness check:
        # within ONE side, every full 5-tuple
        # `(slug, shape_point, repeat_index, launch_index, sig)` MUST
        # appear at most once. A historical producer that reused launch_index
        # or emitted duplicate summaries on the same side trips this
        # check even when the cross-side count + signature multisets
        # both match.
        cupti_full_key_counts = Counter(
            r[_JOIN_INTEGRITY_KEY_FIELD] for r in cupti_records_here
        )
        nvbit_full_key_counts = Counter(
            r[_JOIN_INTEGRITY_KEY_FIELD] for r in nvbit_records_here
        )
        per_side_dups = {
            "cupti": {sig: c for sig, c in cupti_full_key_counts.items() if c > 1},
            "nvbit": {sig: c for sig, c in nvbit_full_key_counts.items() if c > 1},
        }
        if per_side_dups["cupti"] or per_side_dups["nvbit"]:
            token = "/".join(str(c) for c in natural)
            conflicts.append(
                f"joined_evidence_nvbit_cupti_launch_count_mismatch_{token}: "
                f"per_side_duplicate_5tuple cupti_dups={per_side_dups['cupti']} "
                f"nvbit_dups={per_side_dups['nvbit']}"
            )
            continue
        # Cross-side count check: same natural key MUST appear with
        # equal counts on both sides.
        cupti_count = len(cupti_records_here)
        nvbit_count = len(nvbit_records_here)
        if cupti_count != nvbit_count:
            token = "/".join(str(c) for c in natural)
            conflicts.append(
                f"joined_evidence_nvbit_cupti_launch_count_mismatch_{token}: "
                f"cupti_count={cupti_count} nvbit_count={nvbit_count}"
            )
            continue
        # Same natural key, same count on both sides. Verify
        # signatures match (multiset equality). A signature mismatch
        # here means the two runs DID NOT produce bit-identical
        # kernel shapes; the join is invalid.
        cupti_sigs = sorted(r[_JOIN_INTEGRITY_KEY_FIELD] for r in cupti_records_here)
        nvbit_sigs = sorted(r[_JOIN_INTEGRITY_KEY_FIELD] for r in nvbit_records_here)
        if cupti_sigs != nvbit_sigs:
            token = "/".join(str(c) for c in natural)
            conflicts.append(
                f"joined_evidence_signature_mismatch_{token}: "
                f"cupti={cupti_sigs} nvbit={nvbit_sigs}"
            )
            continue
        # Same natural key, same count, same signatures. Pair each
        # CUPTI record to the NVBit record with the same signature
        # (both lists are sorted by signature for determinism). For
        # each pair, verify binary_sha256 matches; emit joined record
        # on success.
        cupti_records_sorted = sorted(
            cupti_records_here, key=lambda r: r[_JOIN_INTEGRITY_KEY_FIELD],
        )
        nvbit_records_sorted = sorted(
            nvbit_records_here, key=lambda r: r[_JOIN_INTEGRITY_KEY_FIELD],
        )
        for cupti_record, nvbit_record in zip(cupti_records_sorted, nvbit_records_sorted):
            sig = cupti_record[_JOIN_INTEGRITY_KEY_FIELD]
            cupti_binary = cupti_record["binary_sha256"]
            nvbit_binary = nvbit_record["binary_sha256"]
            if cupti_binary != nvbit_binary:
                token = "/".join(str(c) for c in natural)
                conflicts.append(
                    f"joined_evidence_binary_sha256_mismatch_{token}: "
                    f"cupti={cupti_binary} nvbit={nvbit_binary}"
                )
                continue
            joined = {
                "merge_evidence_schema_version": JOINED_EVIDENCE_SCHEMA_VERSION,
                "joined_at_utc": _utc_now_iso8601(),
                "natural_key": {f: cupti_record[f] for f in _JOIN_NATURAL_KEY_FIELDS},
                "kernel_signature_sha256": sig,
                "binary_sha256": cupti_binary,
                "cupti_sidecar": cupti_record,
                "nvbit_trace_summary": nvbit_record,
            }
            out_path = _joined_output_path(output_dir, args.platform, cupti_record)
            joined_records.append((out_path, joined))

    # Defense-in-depth final guard: every joined record's output path
    # MUST be unique across the queued writes. The upstream per-side
    # uniqueness + cross-side count + signature-multiset checks should
    # already have caught any case that produces a path collision, but
    # this guard is the load-bearing safety net so a future regression
    # in those upstream checks cannot silently overwrite a joined file.
    output_path_counts = Counter(p for p, _ in joined_records)
    duplicate_paths = sorted(
        p for p, count in output_path_counts.items() if count > 1
    )
    for dup_path in duplicate_paths:
        conflicts.append(
            f"joined_evidence_output_path_collision_{dup_path}: "
            f"{output_path_counts[dup_path]} joined records target the same "
            f"output path; refusing to write (fail-closed)."
        )

    for c in conflicts:
        print(f"conflict: {c}", file=sys.stderr)

    if conflicts:
        print(
            f"merge_evidence join-cupti-nvbit: {len(conflicts)} conflict(s); "
            f"{len(joined_records)} clean join(s) NOT written (fail-closed).",
            file=sys.stderr,
        )
        return 1

    if args.check:
        for path, _ in joined_records:
            print(f"would_write: {path}")
        print(
            f"merge_evidence join-cupti-nvbit: --check: {len(joined_records)} "
            f"join(s) would be written; no files created."
        )
        return 0

    for path, record in joined_records:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, sort_keys=False)
            f.write("\n")
    print(
        f"merge_evidence join-cupti-nvbit: ok: {len(joined_records)} joined "
        f"file(s) written under {output_dir}/{args.platform}/."
    )
    return 0


def _utc_now_iso8601() -> str:
    """Current UTC time as ISO-8601 with `Z` suffix."""
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SimBench-BW CUPTI/NVBit two-run offline join tool.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    jcn = sub.add_parser(
        "join-cupti-nvbit",
        help="Join NVBit and CUPTI two-run evidence by canonical 5-tuple.",
    )
    jcn.add_argument("--cupti-dir", required=True,
                     help="Directory tree containing cupti_activity_*.json sidecars.")
    jcn.add_argument("--nvbit-dir", required=True,
                     help="Directory tree containing nvbit_trace_summary_*.json files.")
    jcn.add_argument("--output-dir", required=True,
                     help="Where joined files are written: "
                          "<output-dir>/<platform>/<slug>/<shape_point>/"
                          "joined_<rep>_<launch>.json")
    jcn.add_argument("--platform", required=True,
                     help="Target platform key (one of the schema's allowlist: "
                          "h100, rtx_pro_6000_blackwell, rtx_3090_ti).")
    jcn.add_argument("--check", action="store_true",
                     help="Report-only: do NOT write joined files.")
    jcn.add_argument("--strict", action="store_true",
                     help="Treat warnings as conflicts.")

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "join-cupti-nvbit":
        return _run_join_cupti_nvbit(args)
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
