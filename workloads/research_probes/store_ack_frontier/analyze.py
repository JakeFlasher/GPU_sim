#!/usr/bin/env python3
"""Validate and reduce the sm_86 STORE acknowledgement-frontier assay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import campaign


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def median(values: Iterable[int | float]) -> int | float:
    material = list(values)
    require(bool(material), "cannot take median of an empty sequence")
    return statistics.median(material)


def summary(values: Iterable[int | float]) -> dict[str, int | float]:
    material = list(values)
    require(bool(material), "cannot summarize an empty sequence")
    return {
        "minimum": min(material),
        "median": median(material),
        "maximum": max(material),
        "positive_count": sum(value > 0 for value in material),
        "zero_count": sum(value == 0 for value in material),
        "negative_count": sum(value < 0 for value in material),
        "count": len(material),
    }


def native_runs(root: Path, cell: campaign.Cell) -> list[Path]:
    directory = root / "direct/native" / cell.key
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.glob("repeat-*")
        if (path / "records.json").is_file() and (path / "run.json").is_file()
    )


def parse_gpu_clock(run: Mapping[str, Any]) -> int:
    clocks: set[int] = set()
    for name in ("gpu_before", "gpu_after"):
        record = run[name]
        require(record["return_code"] == 0, f"{name}: nvidia-smi failed")
        fields = [field.strip() for field in record["output"].split(",")]
        require(len(fields) == 11, f"{name}: unexpected nvidia-smi shape")
        require(fields[1] == "NVIDIA GeForce RTX 3090 Ti", f"{name}: GPU name drift")
        require(
            fields[2] == "GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c",
            f"{name}: GPU UUID drift",
        )
        require(fields[3] == "8.6", f"{name}: compute capability drift")
        clocks.add(int(fields[8]))
    require(len(clocks) == 1, "SM clock changed across one process run")
    return next(iter(clocks))


def reduce_native_run(
    path: Path, cell: campaign.Cell
) -> tuple[dict[str, int | float], list[int], int, str]:
    records = load_json(path / "records.json")
    run = load_json(path / "run.json")
    require(isinstance(records, list), f"{path}: records are not a list")
    trials = [record for record in records if record.get("record") == "trial"]
    ctas = [record for record in records if record.get("record") == "cta"]
    summaries = [record for record in records if record.get("record") == "summary"]
    require(len(summaries) == 3, f"{path}: expected three summaries")
    trial_ids = sorted({int(record["trial"]) for record in trials})
    require(trial_ids == list(range(len(trial_ids))), f"{path}: trial ids are not dense")
    require(len(trial_ids) % 2 == 1, f"{path}: trial count is not odd")
    require(len(trials) == len(trial_ids) * 3, f"{path}: trial arm cardinality")
    require(
        len(ctas) == len(trial_ids) * 3 * cell.blocks,
        f"{path}: raw CTA cardinality",
    )
    by_trial: dict[tuple[int, str], Mapping[str, Any]] = {}
    timer_deltas: list[int] = []
    for record in trials:
        require(record["correct"] is True, f"{path}: incorrect trial")
        require(record["queue_preloaded"] is True, f"{path}: queue gate failed")
        require(record["trace_only"] is False, f"{path}: native run is trace-only")
        expected_unique_sms = 84 if cell.blocks == 84 else 1
        require(
            record["unique_sms"] == expected_unique_sms,
            f"{path}: SM placement cardinality drift",
        )
        key = (int(record["trial"]), str(record["mode"]))
        require(key not in by_trial, f"{path}: duplicate trial arm")
        by_trial[key] = record
        for field in (
            "issue_ns_max",
            "fence_ns_max",
            "grid_span_ns",
            "handoff_ns",
        ):
            value = int(record[field])
            if value > 0:
                timer_deltas.append(value)
    for trial in trial_ids:
        expected_order = [campaign.ARMS[(trial + index) % 3] for index in range(3)]
        observed_order = [
            next(
                str(record["mode"])
                for record in trials
                if int(record["trial"]) == trial
                and int(record["order_index"]) == index
            )
            for index in range(3)
        ]
        require(observed_order == expected_order, f"{path}: arm order drift")
    for record in ctas:
        require(
            int(record["after_issue_ns"]) >= int(record["before_issue_ns"]),
            f"{path}: CTA issue timer inversion",
        )
        require(
            int(record["after_fence_ns"]) >= int(record["after_issue_ns"]),
            f"{path}: CTA fence timer inversion",
        )
        for left, right in (
            ("before_issue_ns", "after_issue_ns"),
            ("after_issue_ns", "after_fence_ns"),
        ):
            delta = int(record[right]) - int(record[left])
            if delta > 0:
                timer_deltas.append(delta)

    def differences(field: str, left: str, right: str) -> list[int | float]:
        return [
            by_trial[(trial, left)][field] - by_trial[(trial, right)][field]
            for trial in trial_ids
        ]

    reduced: dict[str, int | float] = {
        "issue_store_none_minus_control_ns": median(
            differences("issue_ns_max", "store_none", "control")
        ),
        "issue_store_device_minus_store_none_ns": median(
            differences("issue_ns_max", "store_device", "store_none")
        ),
        "device_fence_slot_ns": median(
            by_trial[(trial, "store_device")]["fence_ns_max"]
            for trial in trial_ids
        ),
        "none_handoff_minus_control_ns": median(
            differences("handoff_ns", "store_none", "control")
        ),
        "device_handoff_minus_control_ns": median(
            differences("handoff_ns", "store_device", "control")
        ),
        "handoff_contraction_none_minus_device_ns": median(
            differences("handoff_ns", "store_none", "store_device")
        ),
        "none_pair_minus_control_ns": median(
            differences("pair_event_ns", "store_none", "control")
        ),
        "device_pair_minus_control_ns": median(
            differences("pair_event_ns", "store_device", "control")
        ),
        "device_pair_minus_none_ns": median(
            differences("pair_event_ns", "store_device", "store_none")
        ),
        "maximum_host_enqueue_ns": max(
            int(record["host_enqueue_ns"]) for record in trials
        ),
        "minimum_gate_observed_ns": min(
            int(record["gate_observed_ns"]) for record in trials
        ),
    }
    return reduced, timer_deltas, parse_gpu_clock(run), run["binary"]["sha256"]


def static_contract(root: Path) -> dict[str, Any]:
    sass_path = root / "static/sass.txt"
    resources_path = root / "static/resources.txt"
    sidecar_path = root / "static/store_ack_frontier_trace.control_bits.json"
    for path in (sass_path, resources_path, sidecar_path):
        require(path.is_file() and path.stat().st_size > 0, f"missing static input {path}")
    sass = sass_path.read_text(encoding="utf-8")
    start = sass.index("Function : _Z15store_ack_probe")
    end = sass.find("\n\t\tFunction :", start + 1)
    if end < 0:
        end = len(sass)
    probe = sass[start:end]
    target_store = probe.index("STG.E.STRONG.SM")
    issue_timer = probe.index("SR_GLOBALTIMERLO", target_store)
    membar = probe.index("MEMBAR.SC.GPU", issue_timer)
    fence_timer = probe.index("SR_GLOBALTIMERLO", membar)
    reporting_store = probe.index("STG.E.64", fence_timer)
    require(
        target_store < issue_timer < membar < fence_timer < reporting_store,
        "probe SASS order mismatch",
    )
    resources = resources_path.read_text(encoding="utf-8")
    probe_resource = next(
        line
        for line in resources.splitlines()
        if line.strip().startswith("REG:28 STACK:")
    )
    require("STACK:0" in probe_resource, "probe has a stack frame")
    require("LOCAL:0" in probe_resource, "probe has local/spill storage")
    return {
        "sass_sha256": digest(sass_path),
        "resources_sha256": digest(resources_path),
        "control_sidecar_sha256": digest(sidecar_path),
        "target_store_static_count": probe.count("STG.E.STRONG.SM"),
        "device_membar_static_count": probe.count("MEMBAR.SC.GPU"),
        "globaltimer_static_count": probe.count("SR_GLOBALTIMERLO"),
        "spill_stores": 0,
        "spill_loads": 0,
        "ordered": True,
    }


def trace_stats(path: Path) -> dict[str, Any]:
    with (path / "stats.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"{path}: expected one trace stats row")
    normalized = {key.strip(): value.strip() for key, value in rows[0].items()}
    require(
        normalized["total_insts"] == normalized["total_reported_insts"],
        f"{path}: trace instruction cardinality mismatch",
    )
    for name in (
        "unsupported_memory_path.json",
        "unmapped_memory_range.json",
        "cross_allocation_lane_span.json",
    ):
        require(load_json(path / "extra_info" / name) == [], f"{path}: {name} nonempty")
    return {
        "total_insts": int(normalized["total_insts"]),
        "kernel": normalized["kernel mangled name"],
        "dynamic_trace_sha256": digest(path / "dynamic_trace.pb"),
    }


def replay_contract(
    root: Path, cells: Mapping[str, campaign.Cell], sm_clock_mhz: int
) -> dict[str, Any]:
    simulator = root / "simulator"
    if not simulator.is_dir():
        return {"available": False, "cells": {}}
    reports: dict[str, Any] = {}
    for cell_dir in sorted(path for path in simulator.iterdir() if path.is_dir()):
        cell = cells[cell_dir.name]
        for repeat_dir in sorted(cell_dir.glob("repeat-*")):
            repeat = repeat_dir.name
            arms: dict[str, Any] = {}
            for arm in campaign.ARMS:
                arm_dir = repeat_dir / arm
                if not (arm_dir / "identity.json").is_file():
                    continue
                identity = load_json(arm_dir / "identity.json")
                require(
                    all(mode["deterministic_twins"] for mode in identity["modes"].values()),
                    f"{arm_dir}: replay twins failed",
                )
                trace = root / "nvbit" / cell.key / repeat / arm
                stats = trace_stats(trace)
                default = load_json(arm_dir / "default-1.json")
                active = load_json(arm_dir / "write_ack-1.json")
                default_launch = default["launches"][0]
                active_launch = active["launches"][0]
                timing = active_launch["timing"]
                readiness = timing["readiness"]
                issue = readiness["global_memory_issue"]
                ack = readiness["memory_write_ack"]
                tails = timing["completion_frontier"]["unmodeled_tails"]
                require(readiness["schema_version"] == 22, f"{arm_dir}: schema 22 absent")
                require(ack["acks_scheduled"] == ack["acks_processed"], f"{arm_dir}: ack loss")
                require(tails["dirty_writeback"] is True, f"{arm_dir}: dirty tail cleared")
                require(tails["coherence"] is True, f"{arm_dir}: coherence tail cleared")
                arms[arm] = {
                    "trace": stats,
                    "instructions_issued": active_launch["counters"]["instructions_issued"],
                    "store_instructions": issue["store_instructions"],
                    "store_sectors": issue["store_sectors"],
                    "acks": ack["acks_scheduled"],
                    "maximum_pending_acks": ack["maximum_pending_acks"],
                    "diagnostic_last_issue_cycles": timing["diagnostic_last_issue_cycles"],
                    "terminal_event_cycles": timing["last_processed_modeled_system_event_time"],
                    "memory_write_completion_limiting_cycles": readiness[
                        "limiting_cycles"
                    ].get("memory_write_completion", 0),
                    "default_completion_status": default_launch["timing"][
                        "completion_frontier"
                    ]["status"],
                    "active_completion_status": timing["completion_frontier"]["status"],
                }
            if set(arms) != set(campaign.ARMS):
                continue
            expected_instructions = cell.blocks * cell.depth
            expected_sectors = expected_instructions * cell.active_lanes
            for arm in ("store_none", "store_device"):
                require(
                    arms[arm]["store_instructions"] - arms["control"]["store_instructions"]
                    == expected_instructions,
                    f"{cell.key}/{repeat}/{arm}: target STORE instruction mismatch",
                )
                require(
                    arms[arm]["store_sectors"] - arms["control"]["store_sectors"]
                    == expected_sectors,
                    f"{cell.key}/{repeat}/{arm}: target STORE sector mismatch",
                )
                require(
                    arms[arm]["acks"] - arms["control"]["acks"]
                    == expected_instructions,
                    f"{cell.key}/{repeat}/{arm}: target ack mismatch",
                )
            require(
                arms["store_none"]["store_instructions"]
                == arms["store_device"]["store_instructions"],
                f"{cell.key}/{repeat}: STORE arm instruction mismatch",
            )
            require(
                arms["store_none"]["store_sectors"]
                == arms["store_device"]["store_sectors"],
                f"{cell.key}/{repeat}: STORE arm sector mismatch",
            )
            none_tail_cycles = max(
                0,
                arms["store_none"]["terminal_event_cycles"]
                - arms["store_none"]["diagnostic_last_issue_cycles"],
            )
            control_tail_cycles = max(
                0,
                arms["control"]["terminal_event_cycles"]
                - arms["control"]["diagnostic_last_issue_cycles"],
            )
            matched_tail_cycles = none_tail_cycles - control_tail_cycles
            device_shift_cycles = (
                arms["store_device"]["diagnostic_last_issue_cycles"]
                - arms["store_none"]["diagnostic_last_issue_cycles"]
            )
            reports[f"{cell.key}/{repeat}"] = {
                "arms": arms,
                "expected_payload_store_instructions": expected_instructions,
                "expected_payload_store_sectors": expected_sectors,
                "matched_no_fence_tail_cycles": matched_tail_cycles,
                "matched_no_fence_tail_ns_at_observed_sm_clock": matched_tail_cycles
                * 1000.0
                / sm_clock_mhz,
                "device_issue_shift_cycles": device_shift_cycles,
                "device_issue_shift_ns_at_observed_sm_clock": device_shift_cycles
                * 1000.0
                / sm_clock_mhz,
            }
    return {"available": bool(reports), "cells": reports}


def analyze(root: Path, stage: str) -> dict[str, Any]:
    matrix = load_json(root / "metadata/matrix.json")
    cells = {
        cell.key: cell
        for cell in campaign.configured_cells()
        if stage == "complete" or cell.stage in {"identification", "calibration"}
    }
    require(matrix["probe"] == "sm86_store_ack_frontier_assay", "matrix probe drift")
    static = static_contract(root)
    timer_deltas: list[int] = []
    clocks: set[int] = set()
    binaries: set[str] = set()
    per_cell: dict[str, Any] = {}
    for key, cell in cells.items():
        runs = native_runs(root, cell)
        require(runs, f"{key}: no native process repeats")
        reduced: list[dict[str, int | float]] = []
        for path in runs:
            row, deltas, clock, binary = reduce_native_run(path, cell)
            reduced.append(row)
            timer_deltas.extend(deltas)
            clocks.add(clock)
            binaries.add(binary)
        fields = sorted(reduced[0])
        per_cell[key] = {
            "stage": cell.stage,
            "process_repeats": len(runs),
            "per_repeat": reduced,
            "metrics": {field: summary(row[field] for row in reduced) for field in fields},
        }
    require(len(clocks) == 1, f"campaign SM clock set is not singular: {sorted(clocks)}")
    require(len(binaries) == 1, "native evidence mixes binary identities")
    binary_manifest = load_json(root / "static/binaries.json")
    require(
        binary_manifest["binaries"]["store_ack_frontier"]["sha256"]
        == next(iter(binaries)),
        "native evidence does not match the retained static binary",
    )
    quantum = 0
    for delta in timer_deltas:
        quantum = math.gcd(quantum, delta)
    require(quantum > 0, "no positive globaltimer delta")
    calibration_key = "b1_d16_l0_line_unique_g0"
    empty_fence = per_cell[calibration_key]["metrics"]["device_fence_slot_ns"]["median"]
    for key, value in per_cell.items():
        value["loaded_device_fence_ns"] = (
            value["metrics"]["device_fence_slot_ns"]["median"] - empty_fence
        )
        value["timer_resolution_class"] = (
            "resolved"
            if max(
                abs(value["metrics"]["none_handoff_minus_control_ns"]["median"]),
                abs(value["loaded_device_fence_ns"]),
            )
            >= quantum
            else "below_globaltimer_resolution"
        )

    b84_keys = ("b84_d4_l32_line_unique_g0", "b84_d16_l32_line_unique_g0")
    qualitative = all(
        per_cell[key]["metrics"]["none_handoff_minus_control_ns"]["median"] >= quantum
        and per_cell[key]["loaded_device_fence_ns"] >= quantum
        and per_cell[key]["metrics"][
            "handoff_contraction_none_minus_device_ns"
        ]["median"]
        >= quantum
        for key in b84_keys
    )
    identification = {
        "b84_fence_terminal_shift": qualitative,
        "b1_resolution_control": all(
            per_cell[key]["timer_resolution_class"] == "below_globaltimer_resolution"
            for key in ("b1_d4_l32_line_unique_g0", "b1_d16_l32_line_unique_g0")
        ),
        "holdout_authorized": qualitative,
    }
    replay = replay_contract(root, {cell.key: cell for cell in campaign.configured_cells()}, next(iter(clocks)))
    if stage == "identification":
        status = (
            "qualified_multicta_fence_terminal_shift_b1_below_resolution"
            if qualitative
            else "no_go_identification"
        )
    else:
        holdout_keys = [key for key, cell in cells.items() if cell.stage == "holdout"]
        holdout_resolved = [
            key
            for key in holdout_keys
            if per_cell[key]["timer_resolution_class"] == "resolved"
        ]
        holdout_positive = all(
            per_cell[key]["metrics"]["none_handoff_minus_control_ns"]["median"] >= quantum
            and per_cell[key]["metrics"][
                "handoff_contraction_none_minus_device_ns"
            ]["median"]
            >= 0
            for key in holdout_resolved
        )
        status = (
            "complete_multicta_shift_transfer_no_per_store_timing"
            if qualitative and holdout_positive and holdout_resolved
            else "complete_store_ack_assay_no_go"
        )
        identification["holdout_resolved_cells"] = holdout_resolved
        identification["holdout_transfer_positive"] = holdout_positive
    return {
        "schema_version": 1,
        "milestone": "sm86_store_ack_frontier_assay",
        "stage": stage,
        "status": status,
        "host": {
            "gpu": "NVIDIA GeForce RTX 3090 Ti",
            "uuid": "GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c",
            "compute_capability": "8.6",
            "sm_clock_mhz": next(iter(clocks)),
            "globaltimer_quantum_ns": quantum,
        },
        "binary_sha256": next(iter(binaries)),
        "static": static,
        "empty_device_fence_calibration_ns": empty_fence,
        "cells": per_cell,
        "identification": identification,
        "simulator_bridge": replay,
        "claim_boundary": {
            "ncu_used_for_selection": False,
            "dirty_writeback_cleared": False,
            "coherence_cleared": False,
            "globaltimer_cross_arch_claim": False,
            "interpretation": "matched qualitative movement between implicit handoff and explicit GPU fence; not pure writeback latency",
        },
    }


def write_readme(root: Path, report: Mapping[str, Any]) -> None:
    host = report["host"]
    lines = [
        "# SM86 STORE acknowledgement-frontier assay",
        "",
        f"Status: `{report['status']}`",
        "",
        f"Pinned host: {host['gpu']} / {host['uuid']} / {host['sm_clock_mhz']} MHz SM.",
        f"Observed `%globaltimer` quantum: {host['globaltimer_quantum_ns']} ns.",
        "",
        "The live assay uses a prequeued ordinary same-stream producer/sentinel pair.",
        "B1 effects at or below one timer quantum are retained as unresolved controls.",
        "B84 results identify whether STORE work moves from implicit handoff into an",
        "explicit device fence. NCU did not select or parameterize the result.",
        "",
        "Dirty-writeback and coherence tails remain asserted; the sentinel proves only",
        "same-device downstream visibility plus a composite task-boundary handoff.",
        "",
    ]
    (root / "analysis/README.md").write_text("\n".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--root", type=Path, default=campaign.DEFAULT_OUTPUT
    )
    result.add_argument("--stage", choices=("identification", "complete"), required=True)
    result.add_argument("--validate-only", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = analyze(args.root.resolve(), args.stage)
        if not args.validate_only:
            write_json = campaign.write_json
            write_json(args.root.resolve() / "analysis/report.json", report)
            write_readme(args.root.resolve(), report)
        print(json.dumps({
            "status": report["status"],
            "globaltimer_quantum_ns": report["host"]["globaltimer_quantum_ns"],
            "holdout_authorized": report["identification"]["holdout_authorized"],
        }, sort_keys=True))
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"store-ack analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
