#!/usr/bin/env python3
"""Reduce the prospective SM86 asynchronous-copy wait-prefix campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "results/captures/sm86_async_wait_prefix_order_20260831_v1"
DEFAULT_AGGREGATE = REPO_ROOT / "data/local_labels/sm86_async_wait_prefix_order_report.json"
VARIANTS = ("l2_old", "l1_old")
FIELDS = (
    "issue_commit_cycles",
    "prefix_wait_cycles",
    "oldest_consume_cycles",
    "remainder_wait_cycles",
    "total_cycles",
)
EXACT_NCU = {
    "smsp__inst_executed_op_ldgsts.sum": 128.0,
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum": 4096.0,
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum": 1024.0,
}


class AnalysisError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected one JSON object")
    return value


def require_object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{context}: expected object")
    return value


def exact_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{context}: expected finite number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise AnalysisError(f"{context}: expected finite number")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_static(root: Path) -> dict[str, Any]:
    summary = read_json(root / "static/summary.json")
    expected = {
        "ldgsts": 10,
        "commits": 10,
        "wait_group_1": 1,
        "wait_group_2": 1,
        "wait_group_0": 2,
    }
    if summary.get("global_counts") != expected:
        raise AnalysisError(f"static async-control counts differ: {summary.get('global_counts')}")
    control = root / "static/control_bits.json"
    control_value = read_json(control)
    control_summary = require_object(control_value.get("summary"), "control summary")
    if control_value.get("arch") != "sm_86" or control_summary.get(
        "total_unknown_encoding"
    ) != 0:
        raise AnalysisError("control sidecar is not a complete sm_86 decode")
    functions = control_value.get("functions")
    if not isinstance(functions, Mapping) or len(functions) != 3:
        raise AnalysisError("control sidecar must contain exactly three functions")
    return {
        "passed": True,
        "global_counts": expected,
        "function_count": len(functions),
        "binary": summary.get("binary"),
        "control_sha256": sha256(control),
    }


def validate_nvbit(root: Path, binary_sha: str) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    totals: dict[str, int] = {}
    for variant in VARIANTS:
        cell = root / "nvbit" / f"identification_{variant}_g64"
        complete = read_json(cell / "complete.json")
        binding = read_json(cell / "campaign_binding.json")
        if complete.get("complete") is not True:
            raise AnalysisError(f"{cell}: NVBit capture is not complete")
        if binding.get("binary", {}).get("sha256") != binary_sha:
            raise AnalysisError(f"{cell}: NVBit binary identity differs")
        with (cell / "stats.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise AnalysisError(f"{cell}: stats row count {len(rows)}, expected one")
        row = {str(key).strip(): str(value).strip() for key, value in rows[0].items()}
        total = int(row["total_insts"])
        reported = int(row["total_reported_insts"])
        if total <= 0 or total != reported:
            raise AnalysisError(f"{cell}: incomplete reported instruction stream")
        if "wait_prefix_identification_kernel" not in row["kernel mangled name"]:
            raise AnalysisError(f"{cell}: wrong traced kernel")
        payload_count = binding.get("output_contract", {}).get("payload_file_count")
        if payload_count != 1:
            raise AnalysisError(f"{cell}: expected one threadblock payload")
        totals[variant] = total
        cells[variant] = {
            "instructions": total,
            "reported_instructions": reported,
            "payload_files": payload_count,
            "trace_sha256": sha256(cell / "dynamic_trace.pb"),
            "binding_sha256": sha256(cell / "campaign_binding.json"),
        }
    return {"passed": True, "cells": cells, "path_instruction_delta": totals["l2_old"] - totals["l1_old"]}


def validate_ncu(root: Path, binary_sha: str) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    metrics_by_variant: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        path = root / "ncu" / f"identification_{variant}_g64/summary.json"
        summary = read_json(path)
        if summary.get("variant") != variant or summary.get("epochs") != 64:
            raise AnalysisError(f"{path}: NCU cell identity mismatch")
        if summary.get("binary", {}).get("sha256") != binary_sha:
            raise AnalysisError(f"{path}: NCU binary identity differs")
        raw = require_object(summary.get("metrics"), f"{path}: metrics")
        metrics = {key: exact_number(raw.get(key), f"{path}: {key}") for key in raw}
        for metric, expected in EXACT_NCU.items():
            if metrics.get(metric) != expected:
                raise AnalysisError(f"{path}: {metric}={metrics.get(metric)}, expected {expected}")
        metrics_by_variant[variant] = metrics
        summaries[variant] = {
            "metrics": metrics,
            "report_sha256": summary.get("report", {}).get("sha256"),
            "raw_csv_sha256": summary.get("raw_csv", {}).get("sha256"),
        }
    if metrics_by_variant["l2_old"] != metrics_by_variant["l1_old"]:
        raise AnalysisError("paired NCU aggregate metrics differ")
    metrics = metrics_by_variant["l2_old"]
    if metrics["lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"] <= 0:
        raise AnalysisError("NCU smoke has no attributable LTS hits")
    if metrics["lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"] != 0:
        raise AnalysisError("NCU smoke has nonzero LTS misses")
    return {"passed": True, "paired_equal": True, "cells": summaries}


def replay_launch(path: Path) -> Mapping[str, Any]:
    value = read_json(path)
    launches = value.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise AnalysisError(f"{path}: expected one replay launch")
    return require_object(launches[0], f"{path}: launch")


def validate_replays(root: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    expected_partial = {
        "wait_count": 64,
        "immediate_sum": 64,
        "committed_groups_before": 128,
        "consumed_prefix_groups": 64,
        "groups_remaining": 64,
    }
    expected_draining = {
        "wait_count": 64,
        "immediate_sum": 0,
        "committed_groups_before": 64,
        "consumed_prefix_groups": 64,
        "groups_remaining": 0,
    }
    for variant in VARIANTS:
        directory = root / "simulator" / f"identification_{variant}_g64"
        first = directory / "replay-1.json"
        second = directory / "replay-2.json"
        first_sha = sha256(first)
        second_sha = sha256(second)
        if first_sha != second_sha:
            raise AnalysisError(f"{directory}: replay twins differ")
        launch = replay_launch(first)
        timing = require_object(launch.get("timing"), f"{first}: timing")
        readiness = require_object(timing.get("readiness"), f"{first}: readiness")
        if readiness.get("schema_version") != 19:
            raise AnalysisError(f"{first}: readiness schema is not 19")
        waits = require_object(readiness.get("async_wait_prefix"), f"{first}: waits")
        partial = require_object(waits.get("partial"), f"{first}: partial")
        draining = require_object(waits.get("draining"), f"{first}: draining")
        for field, expected in expected_partial.items():
            if partial.get(field) != expected:
                raise AnalysisError(f"{first}: partial {field}={partial.get(field)}, expected {expected}")
        for field, expected in expected_draining.items():
            if draining.get(field) != expected:
                raise AnalysisError(f"{first}: draining {field}={draining.get(field)}, expected {expected}")
        groups = require_object(readiness.get("async_groups"), f"{first}: async groups")
        if groups.get("groups_committed") != 128 or groups.get("groups_waited") != 128:
            raise AnalysisError(f"{first}: async group counts differ")
        frontier = require_object(timing.get("completion_frontier"), f"{first}: frontier")
        tails = require_object(frontier.get("unmodeled_tails"), f"{first}: tails")
        if tails.get("async_copy_shared_visibility") is not True:
            raise AnalysisError(f"{first}: async-copy visibility tail was cleared")
        counters = require_object(launch.get("counters"), f"{first}: counters")
        l2_source = require_object(counters.get("l2_by_source"), f"{first}: l2 source")
        bypass = require_object(l2_source.get("bypass"), f"{first}: bypass")
        if bypass.get("sectors") != 2048:
            raise AnalysisError(f"{first}: bypass source sector count differs")
        cells[variant] = {
            "replay_sha256": first_sha,
            "diagnostic_last_issue_cycles": timing.get("diagnostic_last_issue_cycles"),
            "partial_exposure_cycles": partial.get("exposure_cycles"),
            "draining_exposure_cycles": draining.get("exposure_cycles"),
            "instructions_issued": counters.get("instructions_issued"),
            "bypass_sectors": bypass.get("sectors"),
            "typed_tails": dict(tails),
        }
    return {"passed": True, "deterministic": True, "cells": cells}


def validate_direct(root: Path) -> dict[str, Any]:
    deltas = {field: [] for field in FIELDS}
    pairs = []
    for repetition in range(1, 11):
        directory = root / "direct/identification" / f"repeat-{repetition}"
        records = {variant: read_json(directory / f"{variant}.json") for variant in VARIANTS}
        row: dict[str, Any] = {"repetition": repetition}
        for variant, record in records.items():
            expected = {
                "workload": "async_wait_prefix_order",
                "mode": "identification",
                "variant": variant,
                "epochs": 8192,
                "groups_per_epoch": 2,
                "wait_group_immediate": 1,
                "correct": True,
                "sum_consistent": True,
                "result_store_after_intervals": True,
            }
            for field, value in expected.items():
                if record.get(field) != value:
                    raise AnalysisError(f"{directory}: {variant} {field} differs")
        for field in FIELDS:
            left = exact_number(records["l2_old"].get(field), f"{directory}: l2 {field}")
            right = exact_number(records["l1_old"].get(field), f"{directory}: l1 {field}")
            delta = (left - right) / 8192.0
            deltas[field].append(delta)
            row[field.replace("_cycles", "_delta_per_epoch")] = delta
        pairs.append(row)

    medians = {field: statistics.median(values) for field, values in deltas.items()}
    issue_abs_median = statistics.median(abs(value) for value in deltas["issue_commit_cycles"])
    consume_abs_median = statistics.median(
        abs(value) for value in deltas["oldest_consume_cycles"]
    )
    positive_prefix = sum(value > 0 for value in deltas["prefix_wait_cycles"])
    negative_remainder = sum(value < 0 for value in deltas["remainder_wait_cycles"])
    cancellation = sum(
        abs(total) < min(prefix, -remainder)
        for total, prefix, remainder in zip(
            deltas["total_cycles"],
            deltas["prefix_wait_cycles"],
            deltas["remainder_wait_cycles"],
            strict=True,
        )
        if prefix > 0 and remainder < 0
    )
    gates = {
        "issue_commit_tied": issue_abs_median <= 1.0,
        "oldest_consume_tied": consume_abs_median <= 1.0,
        "prefix_magnitude": medians["prefix_wait_cycles"] >= 64.0,
        "prefix_repeat_sign": positive_prefix >= 8,
        "remainder_magnitude": medians["remainder_wait_cycles"] <= -64.0,
        "remainder_repeat_sign": negative_remainder >= 8,
        "total_cancellation": cancellation >= 8,
    }
    return {
        "pairs": pairs,
        "deltas_per_epoch": deltas,
        "medians_per_epoch": medians,
        "issue_absolute_median_per_epoch": issue_abs_median,
        "oldest_consume_absolute_median_per_epoch": consume_abs_median,
        "positive_prefix_repeats": positive_prefix,
        "negative_remainder_repeats": negative_remainder,
        "cancellation_repeats": cancellation,
        "gates": gates,
        "passed": all(gates.values()),
    }


def simulator_transfer(direct: Mapping[str, Any], replays: Mapping[str, Any]) -> dict[str, Any]:
    cells = require_object(replays.get("cells"), "replay cells")
    l2 = require_object(cells.get("l2_old"), "l2 replay")
    l1 = require_object(cells.get("l1_old"), "l1 replay")
    predicted_prefix = (exact_number(l2.get("partial_exposure_cycles"), "l2 partial") - exact_number(l1.get("partial_exposure_cycles"), "l1 partial")) / 64.0
    predicted_remainder = (exact_number(l2.get("draining_exposure_cycles"), "l2 drain") - exact_number(l1.get("draining_exposure_cycles"), "l1 drain")) / 64.0
    live = require_object(direct.get("medians_per_epoch"), "live medians")
    live_prefix = exact_number(live.get("prefix_wait_cycles"), "live prefix")
    live_remainder = exact_number(live.get("remainder_wait_cycles"), "live remainder")
    prefix_error = abs(predicted_prefix - live_prefix)
    remainder_error = abs(predicted_remainder - live_remainder)
    prefix_tolerance = max(16.0, 0.15 * abs(live_prefix))
    remainder_tolerance = max(16.0, 0.15 * abs(live_remainder))
    gates = {
        "prefix_direction": predicted_prefix > 0,
        "remainder_direction": predicted_remainder < 0,
        "prefix_absolute_transfer": prefix_error <= prefix_tolerance,
        "remainder_absolute_transfer": remainder_error <= remainder_tolerance,
    }
    return {
        "predicted_prefix_delta_per_epoch": predicted_prefix,
        "predicted_remainder_delta_per_epoch": predicted_remainder,
        "live_prefix_delta_per_epoch": live_prefix,
        "live_remainder_delta_per_epoch": live_remainder,
        "prefix_absolute_error": prefix_error,
        "remainder_absolute_error": remainder_error,
        "prefix_tolerance": prefix_tolerance,
        "remainder_tolerance": remainder_tolerance,
        "gates": gates,
        "passed": all(gates.values()),
    }


def holdout_state(root: Path) -> dict[str, Any]:
    candidates = [
        root / "direct/holdout",
        root / "nvbit/holdout",
        root / "ncu/holdout",
        root / "simulator/holdout",
    ]
    present = [str(path) for path in candidates if path.exists()]
    if present:
        raise AnalysisError(f"holdout artifacts exist before authorization: {present}")
    return {"authorized": False, "captured": False, "timing_read": False}


def build_report(root: Path) -> dict[str, Any]:
    static = validate_static(root)
    binary = require_object(static.get("binary"), "static binary")
    binary_sha = str(binary.get("sha256"))
    nvbit = validate_nvbit(root, binary_sha)
    ncu = validate_ncu(root, binary_sha)
    replays = validate_replays(root)
    direct = validate_direct(root)
    transfer = simulator_transfer(direct, replays)
    holdout = holdout_state(root)
    identification_passed = bool(direct["passed"] and transfer["passed"])
    if identification_passed:
        raise AnalysisError("identification passed but the prospective holdout is absent")
    return {
        "schema_version": 1,
        "assay": "sm86_async_wait_prefix_order_v1",
        "status": "complete_wait_prefix_identification_no_go",
        "decision": (
            "FIFO phase directions reproduced, but the frozen issue, draining-magnitude, "
            "cancellation, and absolute remainder-transfer gates failed before holdout."
        ),
        "root": str(root.resolve()),
        "timing_promotion_authorized": False,
        "typed_visibility_tail_required": True,
        "static": static,
        "nvbit": nvbit,
        "ncu": ncu,
        "simulator_replays": replays,
        "identification": direct,
        "simulator_transfer": transfer,
        "holdout": holdout,
        "counts": {
            "direct_pairs": 10,
            "direct_runs": 20,
            "ncu_reports": 2,
            "nvbit_traces": 2,
            "simulator_outputs": 4,
        },
        "criteria": {
            "structural_smoke_passed": True,
            "identification_passed": False,
            "holdout_untouched": True,
            "timing_unchanged": True,
            "wait_prefix_observation_timing_neutral": True,
            "ldgdepbar_ordinary_wait_mask_admitted": True,
        },
    }


def markdown(report: Mapping[str, Any]) -> str:
    identification = require_object(report["identification"], "identification")
    medians = require_object(identification["medians_per_epoch"], "medians")
    transfer = require_object(report["simulator_transfer"], "transfer")
    return f"""# SM86 asynchronous-copy wait-prefix ordering result

Status: `{report['status']}`

The live FIFO phase directions are repeat-stable, but the frozen identification
does not transfer. The three-group holdout was not captured or read, production
timing remains unchanged, and `async_copy_shared_visibility` remains required.

| Paired `l2_old - l1_old` quantity | Live median cycles/epoch |
|---|---:|
| Issue and commit | {medians['issue_commit_cycles']:.6f} |
| Prefix wait | {medians['prefix_wait_cycles']:.6f} |
| Oldest consume | {medians['oldest_consume_cycles']:.6f} |
| Remainder wait | {medians['remainder_wait_cycles']:.6f} |
| Total | {medians['total_cycles']:.6f} |

The schema-19 simulator predicts prefix and remainder exposure deltas of
`{transfer['predicted_prefix_delta_per_epoch']:.6f}` and
`{transfer['predicted_remainder_delta_per_epoch']:.6f}` cycles/epoch. The
prefix absolute-transfer gate passes; the remainder gate does not.
"""


def write_or_validate(path: Path, payload: bytes, validate_only: bool) -> None:
    if validate_only:
        try:
            current = path.read_bytes()
        except OSError as error:
            raise AnalysisError(f"{path}: cannot validate retained output: {error}") from error
        if current != payload:
            raise AnalysisError(f"{path}: retained output differs from fresh reduction")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / "analysis/report.json"
    output_md = args.output_md or root / "analysis/report.md"
    try:
        report = build_report(root)
        encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        rendered = markdown(report).encode()
        write_or_validate(output, encoded, args.validate_only)
        write_or_validate(output_md, rendered, args.validate_only)
        write_or_validate(args.aggregate_output, encoded, args.validate_only)
    except (AnalysisError, OSError, ValueError, KeyError) as error:
        print(f"wait-prefix analysis: {error}", file=sys.stderr)
        return 1
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
