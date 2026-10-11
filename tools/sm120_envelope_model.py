#!/usr/bin/env python3
"""Fast hardware-in-the-loop SM120 stage-envelope evaluator.

This intentionally replaces the permanently banned external cycle-simulator
comparison.  It consumes the alpha replay/evidence JSONL, optional CUPTI PC
sampling stall rows, raw NCU projections, and uninstrumented timing metadata.
It emits feature vectors and interval frontiers; it never invents cache
geometry, issue ports, or TMA completion timestamps.

PC sampling input may be a lightweight CSV or a CUPTI decoder JSON summary;
rows without a complete launch identity remain explicitly unassigned.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _optional_int(row: dict[str, Any], names: tuple[str, ...]) -> int | None:
    """Read one optional integer without collapsing missing identity to zero."""

    for name in names:
        value = row.get(name)
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None
    return None


def _first_value(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def load_pc_sampling(path: Path | None) -> dict[str, Any]:
    """Load optional CUPTI PC-sampling rows without inventing launch identity.

    The previous parser defaulted a missing launch column to ``0``.  That made
    a process-level row silently attach to launch zero and destroyed the
    distinction between an absent sample and a zero-valued sample.  The new
    document retains each source row (including PC, stall reason, correlation,
    sample and drop counts), indexes only rows with an explicit capture and
    launch identity, and reports unmatched rows separately.
    """

    document: dict[str, Any] = {
        "path": str(path) if path is not None else None,
        "status": "unavailable" if path is None or not path.is_file() else "observed",
        "rows": [],
        "index": {},
        "unmatched_rows": [],
        "row_count": 0,
        "sample_count": 0,
        "declared_sample_count": None,
        "drop_count": 0,
        "declared_drop_count": None,
        "identity_rows": 0,
        "errors": [],
        "collection_mode": None,
    }
    if path is None or not path.is_file():
        return document

    index: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def append_row(parsed: dict[str, Any]) -> None:
        """Retain one source row and index only complete launch identity."""

        document["rows"].append(parsed)
        document["row_count"] += 1
        if parsed.get("samples") is not None:
            document["sample_count"] += parsed["samples"]
        if parsed.get("dropped") is not None:
            document["drop_count"] += parsed["dropped"]
        if parsed.get("errors"):
            document["errors"].extend(
                {"source_line": parsed.get("source_line"), "code": error}
                for error in parsed["errors"]
            )
        capture_name = parsed.get("capture_name")
        launch_index = parsed.get("launch_index")
        if capture_name is not None and launch_index is not None:
            index.setdefault((capture_name, launch_index), []).append(parsed)
            document["identity_rows"] += 1
        else:
            document["unmatched_rows"].append(parsed)

    def parse_row(
        row: dict[str, Any],
        *,
        line_number: int | None,
        reason_override: str | None = None,
        sample_override: int | None = None,
        dropped_override: int | None = None,
        collection_mode: Any = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        capture = _first_value(row, ("capture_name", "capture"))
        launch_raw = _first_value(row, ("launch_index", "launch", "kernel_launch"))
        launch = _optional_int(row, ("launch_index", "launch", "kernel_launch"))
        if launch_raw is not None and launch is None:
            errors.append("invalid_launch_identity")
        sample_raw = _first_value(row, ("samples", "sample_count", "count"))
        sample_count = sample_override if sample_override is not None else _optional_int(
            row, ("samples", "sample_count", "count")
        )
        if sample_raw is not None and sample_count is None:
            errors.append("invalid_sample_count")
        drop_raw = _first_value(
            row, ("dropped", "drop_count", "samples_dropped", "records_dropped")
        )
        drop_count = dropped_override if dropped_override is not None else _optional_int(
            row, ("dropped", "drop_count", "samples_dropped", "records_dropped")
        )
        if drop_raw is not None and drop_count is None:
            errors.append("invalid_drop_count")
        capture_name = str(capture) if capture is not None else None
        if capture_name is None:
            errors.append("missing_capture_identity")
        if launch is None:
            errors.append("missing_launch_identity")
        if _first_value(row, ("pc", "program_counter", "pc_offset")) is None:
            errors.append("missing_pc")
        if _first_value(row, ("function", "function_name", "function_index")) is None:
            errors.append("missing_function_identity")
        return {
            "source_line": line_number,
            "capture_name": capture_name,
            "launch_index": launch,
            "device_id": _optional_int(row, ("device_id", "device")),
            "stream_id": _optional_int(row, ("stream_id", "stream")),
            "kernel_id": _optional_int(row, ("kernel_id", "kernel")),
            "module": _first_value(row, ("module", "module_name")),
            "function": _first_value(row, ("function", "function_name")),
            "function_index": _optional_int(row, ("function_index", "function_id")),
            "correlation_id": _first_value(
                row, ("correlation_id", "correlation", "corr_id")
            ),
            "pc": _first_value(row, ("pc", "program_counter", "pc_offset")),
            "stall_reason": reason_override
            or _first_value(row, ("stall_reason", "reason"))
            or "unknown",
            "samples": max(0, sample_count) if sample_count is not None else None,
            "dropped": max(0, drop_count) if drop_count is not None else None,
            "sample_period": _optional_int(row, ("sample_period", "period")),
            "collection_mode": collection_mode
            if collection_mode is not None
            else _first_value(row, ("collection_mode", "mode", "sampling_mode")),
            "errors": errors,
            "raw": dict(row),
        }

    # CUPTI's native decoder emits a JSON summary (pc_records + buffers), while
    # lightweight local collectors often emit CSV.  Accept both without
    # reducing a missing launch to zero.
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            document["status"] = "unavailable"
            document["errors"].append({"code": "unparseable_pc_sampling_json", "detail": str(exc)})
            return document
        if isinstance(payload, dict):
            document["collection_mode"] = _first_value(
                payload, ("collection_mode", "mode", "sampling_mode")
            )
            records = payload.get("pc_records") or payload.get("records") or []
            if isinstance(records, list):
                for record_number, raw_record in enumerate(records, start=1):
                    if not isinstance(raw_record, dict):
                        document["errors"].append(
                            {"source_line": record_number, "code": "unparseable_pc_record"}
                        )
                        continue
                    reasons = raw_record.get("stall_reason_samples") or {}
                    if isinstance(reasons, dict) and reasons:
                        for reason, count in reasons.items():
                            parsed_samples = _optional_int({"count": count}, ("count",))
                            parsed = parse_row(
                                raw_record,
                                line_number=record_number,
                                reason_override=str(reason),
                                sample_override=parsed_samples,
                                collection_mode=document["collection_mode"],
                            )
                            if parsed_samples is None:
                                parsed["errors"].append("invalid_sample_count")
                            append_row(parsed)
                    else:
                        append_row(
                            parse_row(
                                raw_record,
                                line_number=record_number,
                                collection_mode=document["collection_mode"],
                            )
                        )
            declared_drops = _optional_int(
                payload, ("dropped_samples", "drop_count", "samples_dropped")
            )
            if declared_drops is not None:
                document["declared_drop_count"] = declared_drops
                document["drop_count"] = max(document["drop_count"], declared_drops)
            declared_samples = _optional_int(
                payload, ("total_samples", "sample_count", "samples")
            )
            if declared_samples is not None:
                document["declared_sample_count"] = declared_samples
            document["metadata"] = {
                key: payload.get(key)
                for key in ("schema", "buffer_count", "pc_record_count", "total_samples", "buffers")
                if key in payload
            }
        else:
            document["errors"].append({"code": "unparseable_pc_sampling_shape"})
        document["status"] = (
            "observed_empty"
            if document["row_count"] == 0
            else "observed_partial_identity"
            if document["unmatched_rows"]
            else "observed"
        )
        document["index"] = index
        return document

    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            append_row(parse_row(row, line_number=line_number))
    if document["row_count"] == 0:
        document["status"] = "observed_empty"
    elif document["unmatched_rows"]:
        document["status"] = "observed_partial_identity"
    document["index"] = index
    return document


def _pc_document(pc: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the document and tolerate the pre-F0.2 empty-map caller."""

    if not pc:
        return {
            "status": "unavailable",
            "rows": [],
            "index": {},
            "unmatched_rows": [],
            "row_count": 0,
            "sample_count": 0,
            "drop_count": 0,
            "identity_rows": 0,
        }
    if "index" in pc:
        return pc
    # Legacy in-memory shape: preserve its aggregate counts but do not infer
    # a launch identity from the old key.
    rows: list[dict[str, Any]] = []
    for key, reasons in pc.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        capture, launch = key
        for reason, count in (reasons or {}).items():
            rows.append(
                {
                    "capture_name": capture,
                    "launch_index": launch,
                    "correlation_id": None,
                    "pc": None,
                    "stall_reason": reason,
                    "samples": count,
                    "dropped": None,
                    "raw": {},
                }
            )
    return {
        "status": "legacy_aggregate",
        "rows": rows,
        "index": {
            key: [row for row in rows if (row.get("capture_name"), row.get("launch_index")) == key]
            for key in {(row.get("capture_name"), row.get("launch_index")) for row in rows}
        },
        "unmatched_rows": [],
        "row_count": len(rows),
        "sample_count": sum(int(row.get("samples") or 0) for row in rows),
        "drop_count": 0,
        "identity_rows": len(rows),
    }


def launch_features(
    record: dict[str, Any],
    launch_index: int,
    launch: dict[str, Any],
    pc: dict[str, Any],
    attribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one launch row using launch-local attribution when available.

    F0.1 adds ``simulator.per_launch[].feature_attribution``.  Prefer that
    surface over capture-wide sidecar totals; older alpha JSONL remains
    readable through the by-kernel fallback below.
    """

    attribution = attribution or {}
    trace = attribution.get("trace") or launch.get("trace_metrics") or {}
    memory = trace.get("memory") or {}
    readiness = (launch.get("timing") or {}).get("readiness") or {}
    async_groups = readiness.get("async_groups") or {}
    sidecar = attribution.get("sidecar") or {}
    sidecar_available = bool(attribution.get("sidecar"))
    if not sidecar_available:
        sidecar_all = record.get("async_sidecar_observation") or {}
        sidecar_by_kernel = sidecar_all.get("by_kernel_id") or {}
        sidecar = sidecar_by_kernel.get(str(launch.get("kernel_id")), {})
        sidecar_available = bool(sidecar)
    attr_request_count = _optional_int(trace, ("memory_requests",))
    request_count = (
        max(0, attr_request_count)
        if attr_request_count is not None
        else sum(
            int(value.get("requests", 0) or 0)
            for value in memory.values()
            if isinstance(value, dict)
        )
    )
    issue_cycles = (launch.get("timing") or {}).get("diagnostic_last_issue_cycles")
    pc_doc = _pc_document(pc)
    capture_name = str(record.get("capture_name", ""))
    pc_rows = list(pc_doc.get("index", {}).get((capture_name, launch_index), []))
    stall: dict[str, int] = {}
    for row in pc_rows:
        reason = str(row.get("stall_reason") or "unknown")
        samples = row.get("samples")
        if samples is not None:
            stall[reason] = stall.get(reason, 0) + max(0, int(samples))
    instructions = int(
        (launch.get("counters") or {}).get("instructions_issued")
        or trace.get("warp_instructions")
        or 0
    )
    instruction_trace = (launch.get("trace_metrics") or {}).get("instructions") or {}
    predicated_off = int(instruction_trace.get("predicated_off_warp_instructions") or 0)
    memory_opcodes = (
        "LD",
        "ST",
        "ATOM",
        "RED",
        "LDS",
        "STS",
        "LDGSTS",
        "UTMA",
        "TMA",
    )
    by_opcode = (launch.get("trace_metrics") or {}).get("by_opcode") or {}
    predicated_off_memory = sum(
        int((activity or {}).get("predicated_off_warp_instructions") or 0)
        for opcode, activity in by_opcode.items()
        if str(opcode).upper().startswith(memory_opcodes)
    )
    barrier_pipeline = readiness.get("barrier_pipeline") or {}
    barrier_records = int(barrier_pipeline.get("arrivals", 0) or 0)
    barrier_warp_instructions = sum(
        int((activity or {}).get("warp_instructions") or 0)
        for opcode, activity in by_opcode.items()
        if str(opcode).upper().startswith(("BAR", "ARRIVES", "DEPBAR", "UTCBAR"))
    )
    barrier_predicated_off = sum(
        int((activity or {}).get("predicated_off_warp_instructions") or 0)
        for opcode, activity in by_opcode.items()
        if str(opcode).upper().startswith(("BAR", "ARRIVES", "DEPBAR", "UTCBAR"))
    )
    issue_status = (
        "unavailable"
        if issue_cycles is None
        else "predicated_off"
        if instructions and predicated_off >= instructions
        else "predicted_zero"
        if issue_cycles == 0
        else "modeled"
    )
    request_status = (
        "predicated_off"
        if request_count == 0 and predicated_off_memory > 0
        else "predicted_zero"
        if request_count == 0 and predicated_off_memory == 0
        else "observed_trace"
    )
    barrier_status = (
        "unavailable"
        if not barrier_pipeline and barrier_warp_instructions == 0
        else "predicated_off"
        if barrier_warp_instructions and barrier_predicated_off >= barrier_warp_instructions
        else "predicted_zero"
        if barrier_records == 0 and barrier_warp_instructions == 0
        else "modeled"
    )
    tma_records = int(sidecar.get("tma_records", 0) or 0)
    async_status = (
        "unavailable"
        if not sidecar_available
        else "predicted_zero"
        if tma_records == 0
        else "not_modelled"
    )
    visibility_status = async_status
    join_status = (
        "identity"
        if pc_rows
        else "unavailable"
        if pc_doc.get("status") in {"unavailable", "observed_empty"}
        else "no_matching_launch"
    )
    return {
        "capture_name": record.get("capture_name"),
        "launch_index": launch_index,
        "kernel_name": launch.get("kernel_name"),
        "features": {
            "issue_cycles": issue_cycles,
            "instructions": instructions,
            "memory_requests": request_count,
            "decoded_sectors": trace.get("sectors_decoded"),
            "async_source_cycles": async_groups.get("source_cycles", 0),
            "async_wait_groups": async_groups.get("groups_waited", 0),
            "tma_records": sidecar.get("tma_records", 0),
            "tma_bytes": sidecar.get("tma_byte_count_sum", 0),
            "barrier_records": barrier_records,
        },
        "feature_status": {
            "issue_cycles": issue_status,
            "instructions": "observed_trace" if instructions else "unavailable",
            "memory_requests": request_status,
            "async_source_cycles": async_status,
            "async_wait_groups": async_status,
            "tma_records": "unavailable"
            if not sidecar_available
            else "observed_sidecar"
            if tma_records
            else "predicted_zero",
            "barrier_records": barrier_status,
        },
        "pc_stall_samples": stall,
        "pc_stall_records": pc_rows,
        "pc_sampling": {
            "status": join_status,
            "source_rows": len(pc_rows),
            "sample_count": sum(int(row.get("samples") or 0) for row in pc_rows),
            "drop_count": sum(int(row.get("dropped") or 0) for row in pc_rows),
            "identity": {
                "capture_name": capture_name,
                "launch_index": launch_index,
                "device_id": launch.get("device_id"),
                "stream_id": launch.get("stream_id"),
                "kernel_id": launch.get("kernel_id"),
            },
        },
        "frontier": {
            "issue": {"lower": issue_cycles, "upper": issue_cycles, "status": issue_status},
            "memory_service": {
                "lower": None,
                "upper": None,
                "status": "predicated_off"
                if request_status == "predicated_off"
                else "not_modelled",
            },
            "barrier": {
                "lower": None,
                "upper": None,
                "status": barrier_status,
            },
            "async_completion": {
                "lower": None,
                "upper": None,
                "status": async_status,
            },
            "visibility": {
                "lower": None,
                "upper": None,
                "status": visibility_status,
            },
        },
        "status": "pc_sampling_augmented" if pc_rows else "feature_vector_only",
    }


def evaluate(records: list[dict[str, Any]], pc: dict[str, Any] | None) -> dict[str, Any]:
    pc_doc = _pc_document(pc)
    rows: list[dict[str, Any]] = []
    for record in records:
        result = (record.get("simulator") or {}).get("result") or {}
        per_launch = (record.get("simulator") or {}).get("per_launch") or []
        attribution_by_index: dict[int, dict[str, Any]] = {}
        for item in per_launch:
            if not isinstance(item, dict) or not item.get("feature_attribution"):
                continue
            try:
                item_index = int(item.get("launch_index"))
            except (TypeError, ValueError):
                continue
            attribution_by_index[item_index] = item.get("feature_attribution") or {}
        for index, launch in enumerate(result.get("launches", []) if isinstance(result, dict) else []):
            rows.append(
                launch_features(
                    record,
                    index,
                    launch,
                    pc_doc,
                    attribution_by_index.get(index),
                )
            )
    return {
        "schema_version": 1,
        "model": "sm120_fast_hardware_in_loop_stage_envelope_v1",
        "external_cycle_simulator_comparison": "permanently_banned",
        "status": "pc_sampling_augmented"
        if pc_doc.get("status") in {"observed", "observed_partial_identity", "legacy_aggregate"}
        else "feature_vector_only_until_pc_sampling",
        # Keep every source row, including rows that could not be joined to a
        # launch.  Consumers may aggregate `pc_stall_samples`, but they must
        # not lose the original PC/correlation/drop identity in the process.
        "pc_sampling": {
            "status": pc_doc.get("status", "unavailable"),
            "path": pc_doc.get("path"),
            "collection_mode": pc_doc.get("collection_mode"),
            "row_count": pc_doc.get("row_count", 0),
            "identity_rows": pc_doc.get("identity_rows", 0),
            "sample_count": pc_doc.get("sample_count", 0),
            "declared_sample_count": pc_doc.get("declared_sample_count"),
            "drop_count": pc_doc.get("drop_count", 0),
            "declared_drop_count": pc_doc.get("declared_drop_count"),
            "errors": pc_doc.get("errors", []),
            "metadata": pc_doc.get("metadata", {}),
            "rows": pc_doc.get("rows", []),
            "unmatched_rows": pc_doc.get("unmatched_rows", []),
        },
        "feature_rows": rows,
        "aggregate": {
            "capture_count": len(records),
            "launch_count": len(rows),
            "pc_sampling_launches": sum(bool(row["pc_stall_samples"]) for row in rows),
            "pc_sampling_rows": pc_doc.get("row_count", 0),
            "pc_sampling_sample_count": pc_doc.get("sample_count", 0),
            "pc_sampling_drop_count": pc_doc.get("drop_count", 0),
            "pc_sampling_unmatched_rows": len(pc_doc.get("unmatched_rows", [])),
            "pc_sampling_identity_rows": pc_doc.get("identity_rows", 0),
            "pc_sampling_status": pc_doc.get("status", "unavailable"),
            "timing_scope": "not_emitted_by_feature_evaluator",
            "timing_source": "sm120_replay_bench.py simulator.replay_speedup",
        },
        "stage_rules": {
            "issue": "PC-stall and trace issue frontier; no port topology",
            "memory_service": "fit quantiles from route/stride/working-set interventions",
            "async_completion": "fit only after payload visibility witness",
            "frontier_composition": "partial_order/interval, not a scalar hardware cycle claim",
        },
        "next_experiment_selector": {
            "method": "candidate-entropy intervention selection",
            "priority": ["CUPTI_PC_sampling", "TMA_payload_litmus", "route_differential", "occupancy_grid_sweep"],
            "reason": "select the smallest probe that separates the most remaining hypotheses",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pc-sampling", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = evaluate(records, load_pc_sampling(args.pc_sampling))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

