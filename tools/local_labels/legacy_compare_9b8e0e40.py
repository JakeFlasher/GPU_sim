#!/usr/bin/env python3
"""Pure-CPU reproduction of the frozen pre-V1 SOL definition comparator.

This module exists only to reconstruct the problem-070 preparation refusal at
HEAD 9b8e0e40.  It is not an equivalence-certification rule and must not be
imported by capture code.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import sys


AMENDMENT_ID = "ANNEX_C_AMEND_R28_PREP070_RECON_V2"
RULE_ID = "SOL_DEF_LEGACY_COMPARE_9B8E0E40"
REFUSAL_CODE = "sol_stage_source_mismatch"
TOOL_PATH = "tools/local_labels/legacy_compare_9b8e0e40.py"
SOURCE_SHA256 = (
    "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa"
)
STAGED_SHA256 = (
    "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387"
)
SOURCE_SCRATCH_PATH = (
    ".humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/"
    "scratch/source_definition.json"
)
STAGED_SCRATCH_PATH = (
    ".humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/"
    "scratch/staged_definition.json"
)
PREPARE_MANIFEST_JOURNAL_RELATIVE_PATH = (
    "cells/sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_"
    "computation/sol_stage/capture_prepare_manifest.json"
)
PREPARE_MANIFEST_SHA256 = (
    "d9c2fd5e55a9edf860ff23c2b2b018e4c63800adb47d26174ff664c0bf20c7e7"
)
SOURCE_DEFINITION_PATH = (
    "/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/tools/"
    "sol_execbench/data/benchmark/L1/070_mamba2_fused_intra_chunk_diagonal_"
    "computation/definition.json"
)
STAGED_JOURNAL_RELATIVE_PATH = (
    "cells/sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_"
    "computation/sol_stage/definition.json"
)
_OPTIONAL_FIELDS = ("hf_id", "op_type", "custom_inputs_entrypoint")
_LOWER_HEX = frozenset("0123456789abcdef")


class LegacyCompareRefusal(Exception):
    """Typed, expected refusal from the reproduction command."""

    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--source", required=True)
    parser.add_argument("--staged", required=True)
    parser.add_argument("--expect-source-sha", required=True)
    parser.add_argument("--expect-staged-sha", required=True)
    parser.add_argument("--tool-head", required=True)
    parser.add_argument("--emit", required=True)
    return parser


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _read_verified(path_text, expected_sha256, side):
    try:
        data = pathlib.Path(path_text).read_bytes()
    except OSError as exc:
        raise LegacyCompareRefusal(
            "SOL_LEGACY_REPRO_INPUT_DRIFT",
            f"cannot read {side} input {path_text!r}: {exc}",
        ) from exc
    observed = _sha256(data)
    if observed != expected_sha256:
        raise LegacyCompareRefusal(
            "SOL_LEGACY_REPRO_INPUT_DRIFT",
            f"{side} sha256 expected {expected_sha256}, observed {observed}",
        )
    return data, observed


def _parse_object(data, side):
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise LegacyCompareRefusal(
            "SOL_LEGACY_REPRO_PARSE_ERROR",
            f"cannot parse {side} as a JSON object: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise LegacyCompareRefusal(
            "SOL_LEGACY_REPRO_PARSE_ERROR",
            f"{side} top-level value is not a JSON object",
        )
    return value


def normalized_definition(document):
    """Apply the exact normalization used by the runner at 9b8e0e40."""

    normalized = dict(document)
    for field in _OPTIONAL_FIELDS:
        if normalized.get(field) == "":
            normalized[field] = None
    return normalized


def _pointer(key):
    return "/" + key.replace("~", "~0").replace("/", "~1")


def compare_documents(source, staged):
    """Return the historical equality result and changed top-level keys."""

    normalized_source = normalized_definition(source)
    normalized_staged = normalized_definition(staged)
    all_keys = set(normalized_source) | set(normalized_staged)
    differing_keys = sorted(
        key
        for key in all_keys
        if key not in normalized_source
        or key not in normalized_staged
        or normalized_source[key] != normalized_staged[key]
    )
    equal = normalized_source == normalized_staged
    return {
        "equal": equal,
        "differing_top_level_keys": differing_keys,
        "difference_pointers": [_pointer(key) for key in differing_keys],
        "maps_to_refusal_code": None if equal else REFUSAL_CODE,
    }


def _valid_head(value):
    return len(value) == 40 and all(character in _LOWER_HEX for character in value)


def _record(args, executed_at, source_observed, staged_observed, verdict):
    return {
        "record_type": "legacy_comparator_reproduction",
        "amendment_id": AMENDMENT_ID,
        "rule_id": RULE_ID,
        "executed_at": executed_at,
        "no_gpu": True,
        "pure_cpu": True,
        "capture_code_imported": False,
        "tool": {
            "path": TOOL_PATH,
            "sha256": _sha256(pathlib.Path(__file__).resolve().read_bytes()),
            "head": args.tool_head,
        },
        "inputs": {
            "source_scratch_path": args.source,
            "source_sha256_expected": args.expect_source_sha,
            "source_sha256_observed": source_observed,
            "staged_scratch_path": args.staged,
            "staged_sha256_expected": args.expect_staged_sha,
            "staged_sha256_observed": staged_observed,
            "source_origin": {
                "kind": "external_prepare_manifest_pinned",
                "prepare_manifest_journal_relative_path": (
                    PREPARE_MANIFEST_JOURNAL_RELATIVE_PATH
                ),
                "prepare_manifest_sha256": PREPARE_MANIFEST_SHA256,
                "source_path": SOURCE_DEFINITION_PATH,
            },
            "staged_origin": {
                "kind": "historical_journal_member",
                "journal_relative_path": STAGED_JOURNAL_RELATIVE_PATH,
            },
        },
        "verdict": verdict,
        "python_version": (
            f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
        ),
        "command_argv": list(sys.argv),
    }


def _serialize(record):
    return (
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _emit(path_text, payload):
    try:
        with pathlib.Path(path_text).open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise LegacyCompareRefusal(
            "SOL_LEGACY_REPRO_EMIT_EXISTS",
            f"emit target already exists: {path_text}",
        ) from exc


def main():
    args = _parser().parse_args()
    if not _valid_head(args.tool_head):
        raise LegacyCompareRefusal(
            "SOL_LEGACY_REPRO_HEAD_INVALID",
            "--tool-head must be exactly 40 lowercase hexadecimal characters",
        )

    executed_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    source_bytes, source_observed = _read_verified(
        args.source, args.expect_source_sha, "source"
    )
    staged_bytes, staged_observed = _read_verified(
        args.staged, args.expect_staged_sha, "staged"
    )
    source = _parse_object(source_bytes, "source")
    staged = _parse_object(staged_bytes, "staged")
    verdict = compare_documents(source, staged)
    record = _record(
        args, executed_at, source_observed, staged_observed, verdict
    )
    _emit(args.emit, _serialize(record))
    print(f"legacy_compare_9b8e0e40: EMITTED {args.emit}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LegacyCompareRefusal as exc:
        print(
            f"legacy_compare_9b8e0e40: REFUSED [{exc.code}] {exc.detail}",
            file=sys.stderr,
        )
        raise SystemExit(2)
