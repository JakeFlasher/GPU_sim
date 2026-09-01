#!/usr/bin/env python3
"""Combine every repeat from a prospective NCU follow-up into one campaign.

The utility is intentionally hardware-only: it never reads simulator output or
accuracy values. It admits a follow-up only when the base seven-run batch has a
metric whose relative range exceeds the declared trigger, and it retains every
repeat from both batches.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


class CombineError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CombineError(f"{path}: root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _ncu_record(campaign: Mapping[str, Any], workload: str) -> Mapping[str, Any]:
    rows = [
        row
        for row in campaign.get("records", [])
        if isinstance(row, Mapping)
        and row.get("workload_id") == workload
        and row.get("stage") == "ncu"
    ]
    if len(rows) != 1:
        raise CombineError(f"{workload}: expected exactly one NCU record, got {len(rows)}")
    return rows[0]


def _hardware_identity(snapshot: Mapping[str, Any]) -> tuple[object, ...]:
    fields = (
        "index",
        "name",
        "uuid",
        "compute_capability",
        "driver_version",
        "graphics_clock_mhz",
        "sm_clock_mhz",
        "memory_clock_mhz",
    )
    return tuple(snapshot.get(field) for field in fields)


def _repeats(record: Mapping[str, Any], workload: str) -> list[Mapping[str, Any]]:
    result = record.get("result")
    if not isinstance(result, Mapping):
        raise CombineError(f"{workload}: NCU result is missing")
    repeats = result.get("repeats")
    if not isinstance(repeats, list) or not repeats:
        raise CombineError(f"{workload}: NCU repeats are missing")
    if result.get("repeat_count") != len(repeats):
        raise CombineError(f"{workload}: repeat_count differs from repeats")
    if not all(isinstance(row, Mapping) for row in repeats):
        raise CombineError(f"{workload}: NCU repeat is not an object")
    return repeats


def _metric_values(repeats: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    identities: list[object] = []
    for repeat in repeats:
        metrics = repeat.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise CombineError("NCU repeat metrics are missing")
        identities.append(repeat.get("identity"))
        for name, observation in metrics.items():
            if not isinstance(observation, Mapping):
                raise CombineError(f"NCU metric {name}: observation is malformed")
            value = observation.get("value")
            if not isinstance(value, (int, float)):
                raise CombineError(f"NCU metric {name}: value is not numeric")
            values.setdefault(str(name), []).append(float(value))
    if any(identity != identities[0] for identity in identities[1:]):
        raise CombineError("NCU launch identity changed within a repeat batch")
    if any(len(row) != len(repeats) for row in values.values()):
        raise CombineError("NCU metric coverage changed within a repeat batch")
    return values


def _maximum_relative_range(
    repeats: Sequence[Mapping[str, Any]],
) -> tuple[str, float]:
    worst_name = ""
    worst_value = -1.0
    for name, values in _metric_values(repeats).items():
        mean = statistics.fmean(values)
        relative_range = 0.0 if mean == 0.0 else (max(values) - min(values)) / mean * 100.0
        if relative_range > worst_value:
            worst_name = name
            worst_value = relative_range
    return worst_name, worst_value


def combine(args: argparse.Namespace) -> dict[str, Any]:
    base_path = args.base.resolve()
    followup_path = args.followup.resolve()
    output = args.output.resolve()
    base = _load(base_path)
    followup = _load(followup_path)
    if base.get("schema_version") != 1 or followup.get("schema_version") != 1:
        raise CombineError("campaign schema_version must be 1")
    for field in ("benchmark_id", "repository", "inputs", "scientific_boundary"):
        if base.get(field) != followup.get(field):
            raise CombineError(f"base/follow-up campaign {field} differs")

    base_record = _ncu_record(base, args.workload)
    followup_record = _ncu_record(followup, args.workload)
    for field in ("workload_id", "stage"):
        if base_record.get(field) != followup_record.get(field):
            raise CombineError(f"base/follow-up NCU record {field} differs")
    host_snapshots = [
        record[boundary]
        for record in (base_record, followup_record)
        for boundary in ("host_pre", "host_post")
    ]
    identities = {
        _hardware_identity(snapshot)
        for snapshot in host_snapshots
        if isinstance(snapshot, Mapping)
    }
    if len(identities) != 1 or len(host_snapshots) != 4:
        raise CombineError("base/follow-up NCU hardware identity or locked clocks differ")

    base_repeats = _repeats(base_record, args.workload)
    followup_repeats = _repeats(followup_record, args.workload)
    base_metrics = _metric_values(base_repeats)
    followup_metrics = _metric_values(followup_repeats)
    if set(base_metrics) != set(followup_metrics):
        raise CombineError("base/follow-up NCU metric identities differ")
    base_identity = base_repeats[0].get("identity")
    if any(row.get("identity") != base_identity for row in followup_repeats):
        raise CombineError("base/follow-up NCU launch identity differs")

    worst_metric, worst_range = _maximum_relative_range(base_repeats)
    if worst_range <= args.trigger_relative_range_pct:
        raise CombineError(
            f"base maximum relative range {worst_range:.3f}% does not exceed "
            f"the {args.trigger_relative_range_pct:.3f}% follow-up trigger"
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise CombineError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    combined_repeats: list[dict[str, Any]] = []
    for source_path, source_rows in (
        (base_path, base_repeats),
        (followup_path, followup_repeats),
    ):
        for source_row in source_rows:
            row = copy.deepcopy(dict(source_row))
            source_repeat = row.get("repeat")
            row["repeat"] = len(combined_repeats) + 1
            row["source_repeat"] = source_repeat
            row["source_campaign_manifest"] = _portable(source_path)
            combined_repeats.append(row)

    result = copy.deepcopy(base)
    result["output_root"] = _portable(output)
    records = result.get("records")
    if not isinstance(records, list):
        raise CombineError("base campaign records are missing")
    replaced = False
    for index, record in enumerate(records):
        if (
            isinstance(record, Mapping)
            and record.get("workload_id") == args.workload
            and record.get("stage") == "ncu"
        ):
            combined_record = copy.deepcopy(dict(base_record))
            combined_record["finished_at"] = followup_record["finished_at"]
            combined_record["host_post"] = copy.deepcopy(followup_record["host_post"])
            combined_record["result"] = {
                "repeat_count": len(combined_repeats),
                "repeats": combined_repeats,
            }
            records[index] = combined_record
            replaced = True
            break
    if not replaced:
        raise CombineError("base NCU record escaped replacement")
    result["ncu_repeat_extension"] = {
        "policy": "retain_all_repeats_from_one_prospective_followup_batch_after_base_range_trigger",
        "workload_id": args.workload,
        "trigger_relative_range_pct": args.trigger_relative_range_pct,
        "base_maximum_relative_range_metric": worst_metric,
        "base_maximum_relative_range_pct": worst_range,
        "base_repeat_count": len(base_repeats),
        "followup_repeat_count": len(followup_repeats),
        "combined_repeat_count": len(combined_repeats),
        "source_campaigns": [
            {"path": _portable(path), "sha256": _sha256(path)}
            for path in (base_path, followup_path)
        ],
        "base_host_pre": copy.deepcopy(base_record["host_pre"]),
        "base_host_post": copy.deepcopy(base_record["host_post"]),
        "followup_host_pre": copy.deepcopy(followup_record["host_pre"]),
        "followup_host_post": copy.deepcopy(followup_record["host_post"]),
        "simulator_outputs_consulted": False,
        "accuracy_values_consulted": False,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trigger-relative-range-pct", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = combine(args)
        destination = args.output.resolve() / "campaign_manifest.json"
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        print(destination)
        return 0
    except (CombineError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"combine NCU follow-up: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
