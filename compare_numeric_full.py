#!/usr/bin/env python3
"""Compare complete-corpus replay rows against retained baseline JSONL.

Replay schemas intentionally grow observer-only fields.  This checker
compares common finite numeric fields while enforcing row identity, status,
architecture, mode, launch-count, and return-code contracts.  It accepts paths
so a fresh checkout is not tied to the original workstation layout.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "tools")
from jsonl_io import open_jsonl_reader

COMPARISON_SCHEMA_VERSION = 2
DEFAULT_BASELINE_ROOT = Path("/home/jakeshea/5090_branch/final_results")
DEFAULT_CASES = (
    (
        "sm120_disabled",
        DEFAULT_BASELINE_ROOT / "dram_disabled/full.jsonl.zst",
        Path("full_replay_runs/review3_sm120_dram_disabled/full.jsonl.zst"),
    ),
    (
        "sm120_enabled",
        DEFAULT_BASELINE_ROOT / "dram_enabled/full.jsonl.zst",
        Path("full_replay_runs/review3_sm120_dram_enabled/full.jsonl.zst"),
    ),
    (
        "sm86",
        DEFAULT_BASELINE_ROOT / "3090ti/full.jsonl.zst",
        Path("full_replay_runs/review3_sm86_full/full.jsonl.zst"),
    ),
)
REPO_ROOT = Path(__file__).resolve().parent

REQUIRED_FIELDS = (
    "status",
    "architecture",
    "memory_mode",
    "cache_scope",
    "launch_count",
    "returncode",
)
MISSING = object()


def canonical_capture(value: Any) -> str:
    """Bridge the historical SM120 slash-vs-double-underscore spelling."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"canonical_capture must be a non-empty string: {value!r}")
    return value.replace("/", "__")


def _finite_issues(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    issues: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        issues.append(".".join(path) or "<root>")
    elif isinstance(value, dict):
        for key, child in value.items():
            issues.extend(_finite_issues(child, path + (str(key),)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_finite_issues(child, path + (str(index),)))
    return issues


def load_rows(path: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    raw_ids: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    nonfinite: list[dict[str, Any]] = []
    with open_jsonl_reader(path) as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            key = canonical_capture(row.get("canonical_capture"))
            raw = row["canonical_capture"]
            if key in rows:
                duplicates.append(
                    {"canonical_capture": key, "first": raw_ids[key], "duplicate": raw}
                )
                continue
            rows[key] = row
            raw_ids[key] = raw
            bad = _finite_issues(row)
            if bad:
                nonfinite.append({"canonical_capture": key, "paths": bad[:16], "count": len(bad)})
    return {"rows": rows, "duplicates": duplicates, "nonfinite": nonfinite}


def _launch_count(row: dict[str, Any]) -> int | None:
    launches = row.get("launches")
    return len(launches) if isinstance(launches, list) else None


def required_value(row: dict[str, Any], field: str) -> Any:
    if field == "launch_count":
        return _launch_count(row) if "launches" in row else MISSING
    return row[field] if field in row else MISSING


def contract_presence(row: dict[str, Any]) -> list[str]:
    """Return required fields absent from a row's status-specific schema."""

    status = row.get("status", MISSING)
    required = {"status"}
    if status == "ok":
        required.update({"architecture", "memory_mode", "cache_scope", "launches", "returncode"})
    return sorted(
        field
        for field in required
        if field not in row or (field == "launches" and not isinstance(row[field], list))
    )


def _contains_finite_number(value: Any) -> bool:
    if finite_number(value):
        return True
    if isinstance(value, dict):
        return any(_contains_finite_number(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_finite_number(child) for child in value)
    return False


def record_missing_baseline_numeric_fields(
    baseline: Any,
    unified: Any,
    path: tuple[str, ...],
    out: list[str],
) -> None:
    """Detect baseline numeric fields silently dropped by the unified row.

    Extra unified fields are expected observer additions.  A baseline-only
    numeric field is different: it can indicate a dropped required result and
    must be visible even when all surviving common fields match.
    """

    if isinstance(baseline, dict) and isinstance(unified, dict):
        for key in set(baseline) - set(unified):
            child_path = path + (str(key),)
            if not ignore(child_path) and _contains_finite_number(baseline[key]):
                out.append(".".join(child_path))
        for key in set(baseline) & set(unified):
            record_missing_baseline_numeric_fields(
                baseline[key], unified[key], path + (str(key),), out
            )
    elif isinstance(baseline, list) and isinstance(unified, list):
        if all(isinstance(item, dict) and "key" in item for item in baseline + unified):
            left = {str(item["key"]): item for item in baseline}
            right = {str(item["key"]): item for item in unified}
            for key in set(left) - set(right):
                child_path = path + (key,)
                if not ignore(child_path) and _contains_finite_number(left[key]):
                    out.append(".".join(child_path))
            for key in set(left) & set(right):
                record_missing_baseline_numeric_fields(left[key], right[key], path + (key,), out)
        else:
            for index, (left, right) in enumerate(zip(baseline, unified)):
                record_missing_baseline_numeric_fields(left, right, path + (str(index),), out)


def display_path(path: Path) -> str:
    """Keep reports portable without hiding which side was compared."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        parts = path.as_posix().split("/final_results/", 1)
        if len(parts) == 2:
            return "<source-root>/final_results/" + parts[1]
        return str(path)


def ignore(path: tuple[str, ...]) -> bool:
    if not path:
        return False
    name = path[-1]
    if name in {
        "schema_version",
        "reference_clock_hz",
        "service_rate_bytes_per_sec",
        "sim_total_wall_ns",
        "sim_replay_wall_ns",
        "driver_wall_ms",
    }:
        return True
    if name.endswith("_wall_ns") or name.endswith("_wall_ms"):
        return True
    if name in {
        "capture_dir",
        "replay_environment_removed",
        "semantics",
        "detail",
        "reason",
        "role",
        "status",
        "unit",
        "ncu_metric",
        "key",
        "arch",
        "source_key",
        "model_status",
        "model_unit",
        "comparison_target_unit",
        "profile_id",
        "scope",
        "observation_scope",
        "configuration_source",
        "non_claim",
        "timing_scope",
        "geometry_status",
        "partition_mapping_status",
        "mshr_count",
        "queue_policy",
        "residency",
        "outstanding_demand",
        "dram_service",
        "issue",
        "transaction_complete",
        "barrier_arrival",
        "wait_or_test",
        "destination_visibility",
        "payload_observation",
        "cache_scope",
        "memory_mode",
        "architecture",
        "carded_dram_bandwidth",
        "sm120_alpha_profile_id",
        "sm120_alpha",
        "simulator_version",
        "canonical_capture",
    }:
        return True
    if any(
        token in name
        for token in (
            "request_mask_audit",
            "youngest_priority_order_mismatches",
            "pending_pressure",
            "store_packet_candidate",
            "pending_merge_sectors",
            "new_fill_sectors",
            "l1_unrequested_subline",
            "wavefront_mask_lookup",
            "constant_memory_",
            "local_canonical",
            "local_frame256",
            "local_payload",
            "atomic_conflict",
            "shared_ideal",
            "shared_legacy",
            "tma_shadow",
            "sync_sidecar",
            "global_store_adjacent",
            "slice_accesses",
        )
    ):
        return True
    return False


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def record_numeric_differences(
    left: Any, right: Any, path: tuple[str, ...], out: list[tuple[tuple[str, ...], Any, Any]]
) -> None:
    if ignore(path):
        return
    if isinstance(left, dict) and isinstance(right, dict):
        for key in set(left) & set(right):
            record_numeric_differences(left[key], right[key], path + (str(key),), out)
    elif isinstance(left, list) and isinstance(right, list):
        if all(isinstance(item, dict) and "key" in item for item in left + right):
            left_by_key = {str(item["key"]): item for item in left}
            right_by_key = {str(item["key"]): item for item in right}
            for key in set(left_by_key) & set(right_by_key):
                record_numeric_differences(
                    left_by_key[key], right_by_key[key], path + (key,), out
                )
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                record_numeric_differences(a, b, path + (str(index),), out)
    elif finite_number(left) and finite_number(right) and left != right:
        out.append((path, left, right))


def compare_case(label: str, baseline_path: Path, unified_path: Path) -> dict[str, Any]:
    baseline = load_rows(baseline_path)
    unified = load_rows(unified_path)
    base_rows = baseline["rows"]
    unified_rows = unified["rows"]
    base_keys = set(base_rows)
    unified_keys = set(unified_rows)
    common = sorted(base_keys & unified_keys)
    missing = sorted(base_keys - unified_keys)
    extra = sorted(unified_keys - base_keys)
    required_diffs: list[dict[str, Any]] = []
    missing_baseline_numeric_fields: list[dict[str, Any]] = []
    numeric_counts: collections.Counter[str] = collections.Counter()
    numeric_examples: dict[str, tuple[str, Any, Any]] = {}
    rows_with_numeric_diffs = 0
    numeric_diff_count = 0
    for key in common:
        left = base_rows[key]
        right = unified_rows[key]
        for field in contract_presence(left):
            unified_value = required_value(right, field)
            required_diffs.append(
                {
                    "canonical_capture": key,
                    "field": field,
                    "baseline": "<missing>",
                    "unified": "<missing>" if unified_value is MISSING else unified_value,
                }
            )
        for field in contract_presence(right):
            baseline_value = required_value(left, field)
            required_diffs.append(
                {
                    "canonical_capture": key,
                    "field": field,
                    "baseline": "<missing>" if baseline_value is MISSING else baseline_value,
                    "unified": "<missing>",
                }
            )
        for field in REQUIRED_FIELDS:
            a = required_value(left, field)
            b = required_value(right, field)
            if (a is MISSING) != (b is MISSING) or (a is not MISSING and b is not MISSING and a != b):
                required_diffs.append(
                    {
                        "canonical_capture": key,
                        "field": field,
                        "baseline": "<missing>" if a is MISSING else a,
                        "unified": "<missing>" if b is MISSING else b,
                    }
                )
        missing_fields: list[str] = []
        record_missing_baseline_numeric_fields(left, right, (), missing_fields)
        if missing_fields:
            missing_baseline_numeric_fields.append(
                {"canonical_capture": key, "paths": sorted(set(missing_fields))[:100]}
            )
        diffs: list[tuple[tuple[str, ...], Any, Any]] = []
        record_numeric_differences(left, right, (), diffs)
        if diffs:
            rows_with_numeric_diffs += 1
        numeric_diff_count += len(diffs)
        for path, a, b in diffs:
            name = ".".join(path)
            numeric_counts[name] += 1
            numeric_examples.setdefault(name, (key, a, b))
    status_baseline = collections.Counter(row.get("status") for row in base_rows.values())
    status_unified = collections.Counter(row.get("status") for row in unified_rows.values())
    result = {
        "label": label,
        "baseline_path": display_path(baseline_path),
        "unified_path": display_path(unified_path),
        "rows": len(base_rows),
        "unified_rows": len(unified_rows),
        "common_rows": len(common),
        "missing_rows": missing,
        "extra_rows": extra,
        "duplicate_baseline_rows": baseline["duplicates"],
        "duplicate_unified_rows": unified["duplicates"],
        "nonfinite_baseline_rows": baseline["nonfinite"],
        "nonfinite_unified_rows": unified["nonfinite"],
        "status_counts_baseline": dict(sorted(status_baseline.items(), key=lambda item: str(item[0]))),
        "status_counts_unified": dict(sorted(status_unified.items(), key=lambda item: str(item[0]))),
        "required_metadata_diffs": required_diffs,
        "missing_baseline_numeric_fields": missing_baseline_numeric_fields,
        "rows_with_numeric_diffs": rows_with_numeric_diffs,
        "numeric_diff_count": numeric_diff_count,
        "top_numeric_diffs": [
            {"path": name, "count": count, "example": numeric_examples[name]}
            for name, count in numeric_counts.most_common(100)
        ],
    }
    result["ok"] = not any(
        (
            missing,
            extra,
            baseline["duplicates"],
            unified["duplicates"],
            baseline["nonfinite"],
            unified["nonfinite"],
            required_diffs,
            missing_baseline_numeric_fields,
            numeric_diff_count,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("LABEL", "BASELINE", "UNIFIED"),
        help="compare one case; may be repeated (defaults to retained local campaigns)",
    )
    parser.add_argument("--output", type=Path, default=Path("FULL_NUMERIC_COMPARISON.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = args.case or [(label, str(base), str(unified)) for label, base, unified in DEFAULT_CASES]
    results = [compare_case(label, Path(base), Path(unified)) for label, base, unified in cases]
    document = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "normalization": {
            "canonical_capture": "replace('/', '__') for the historical SM120 capture-id separator only; collisions fail",
            "numeric_fields": "all common finite numeric fields, excluding documented wall-clock and observer-only additive fields",
            "required_metadata": list(REQUIRED_FIELDS),
        },
        "cases": results,
        "ok": all(result["ok"] for result in results),
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "label",
                        "rows",
                        "unified_rows",
                        "common_rows",
                        "missing_rows",
                        "extra_rows",
                        "required_metadata_diffs",
                        "rows_with_numeric_diffs",
                        "numeric_diff_count",
                        "ok",
                    )
                },
                indent=2,
            )
        )
        print(json.dumps(result["top_numeric_diffs"][:30], indent=2))
    return 0 if document["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
