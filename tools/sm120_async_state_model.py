#!/usr/bin/env python3
"""Build a causal, logical TMA/UTMA/mbarrier event ledger from sidecars.

The ledger separates issue, source-service, barrier generation, wait and
destination visibility.  It is an executable model input for SM120 alpha
replay; it is not a hardware completion timestamp and never treats descriptor
bytes as a payload-visibility witness.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any


def rows(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def model_capture(capture: Path, source_rate: int, destination_rate: int, payload_log: Path | None = None) -> dict[str, Any]:
    events: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    barriers: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {"addresses": set(), "operations": collections.Counter(), "generations": 0})
    for row in rows(capture / "extra_info/tma_trace.jsonl"):
        key = str(row.get("kernel_id", "unknown"))
        bytes_count = max(0, int(row.get("byte_count", 0) or 0))
        opcode = str(row.get("opcode", "UNKNOWN"))
        event = {
            "kind": "tma_issue",
            "kernel_id": row.get("kernel_id"),
            "function_id": row.get("function_id"),
            "warp_id": row.get("warp_id"),
            "warp_instruction_index": row.get("warp_instruction_index"),
            "pc": row.get("pc"),
            "opcode": opcode,
            "bytes": bytes_count,
            "src_memspace": row.get("src_memspace"),
            "dst_memspace": row.get("dst_memspace"),
            "barrier_address": row.get("dst_mbarrier_address") if row.get("dst_mbar_valid") else None,
            "phase": "unknown_without_runtime_phase",
        }
        events[key].append(event)
        if event["barrier_address"] is not None:
            barriers[key]["addresses"].add(int(event["barrier_address"]))
        barriers[key]["operations"][opcode] += 1
    for row in rows(capture / "extra_info/sync_trace.jsonl"):
        key = str(row.get("kernel_id", "unknown"))
        opcode = str(row.get("opcode", "UNKNOWN"))
        address = row.get("barrier_address_u32") if row.get("barrier_address_present") else None
        events[key].append(
            {
                "kind": "barrier_or_wait",
                "kernel_id": row.get("kernel_id"),
                "function_id": row.get("function_id"),
                "warp_id": row.get("warp_id"),
                "warp_instruction_index": row.get("warp_instruction_index"),
                "pc": row.get("pc"),
                "opcode": opcode,
                "sync_role": row.get("sync_role"),
                "barrier_address": address,
                "phase": row.get("phase", "unknown"),
                "value_complete": row.get("value_complete"),
            }
        )
        if address is not None:
            barriers[key]["addresses"].add(int(address))
        barriers[key]["operations"][opcode] += 1
        if "INIT" in opcode.upper():
            barriers[key]["generations"] += 1
    stdout = payload_log if payload_log is not None else capture / "program.stdout"
    payload_matches = []
    if stdout.is_file():
        text = stdout.read_text(encoding="utf-8", errors="replace")
        payload_matches = [
            {"label": match.group(1), "bytes": int(match.group(2)), "first_u16": int(match.group(3)), "last_u16": int(match.group(4))}
            for match in re.finditer(
                r"SM120_PAYLOAD_OBSERVED\s+label=([^\s]+)\s+bytes=(\d+)\s+first_u16=(\d+)\s+last_u16=(\d+)",
                text,
            )
        ]
    launches = []
    for key, owned in sorted(events.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        owned.sort(key=lambda value: (value.get("warp_id", -1), value.get("warp_instruction_index", -1), value.get("pc", -1)))
        tma = [event for event in owned if event["kind"] == "tma_issue"]
        bytes_total = sum(int(event.get("bytes", 0) or 0) for event in tma)
        loads = [event for event in tma if "LDG" in str(event.get("opcode", "")).upper()]
        stores = [event for event in tma if "STG" in str(event.get("opcode", "")).upper()]
        source_cycles = math.ceil(bytes_total / max(1, source_rate)) if bytes_total else 0
        destination_cycles = math.ceil(sum(int(event.get("bytes", 0) or 0) for event in stores) / max(1, destination_rate)) if stores else 0
        op_counts = barriers[key]["operations"]
        phase_checks = sum(count for opcode, count in op_counts.items() if "PHASECHK" in opcode.upper() or "TRYWAIT" in opcode.upper())
        arrivals = sum(count for opcode, count in op_counts.items() if "ARRIVE" in opcode.upper() or "ARVCNT" in opcode.upper())
        launches.append(
            {
                "kernel_id": int(key) if key.isdigit() else key,
                "event_count": len(owned),
                "tma_records": len(tma),
                "tma_bytes": bytes_total,
                "load_records": len(loads),
                "store_records": len(stores),
                "logical_source_service_cycles": source_cycles,
                "logical_destination_service_cycles": destination_cycles,
                "committed_groups": 0,
                "waited_groups": 0,
                "barrier_generations_observed": max(barriers[key]["generations"], phase_checks),
                "barrier_counts": {"arrivals": arrivals, "phase_checks_or_waits": phase_checks, "init_markers": barriers[key]["generations"]},
                "barrier_addresses": sorted(barriers[key]["addresses"]),
                "opcode_counts": dict(barriers[key]["operations"]),
                "visibility": "unobserved_without_payload_litmus",
                "payload_observation": payload_matches,
                "events": owned[:2048],
                "events_truncated": len(owned) > 2048,
            }
        )
    return {
        "capture_name": capture.name,
        "model": "sm120_logical_async_state_v1",
        "source_rate_bytes_per_cycle": source_rate,
        "destination_rate_bytes_per_cycle": destination_rate,
        "launches": launches,
        "completion_timestamps": "not present in NVBit sidecars",
        "payload_observation_status": "observed_after_kernel_sync" if payload_matches else "unavailable",
        "physical_non_claim": ["TMA_engine_queue", "mbarrier_visibility_timestamp", "cluster_interconnect_order"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/dev/shm/sm120_rtx_5090_work"))
    parser.add_argument("--capture", action="append")
    parser.add_argument("--source-rate", type=int, default=64)
    parser.add_argument("--destination-rate", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--payload-log", type=Path)
    args = parser.parse_args()
    wanted = set(args.capture or [])
    output = []
    for capture in sorted((args.work_root / "captures").iterdir() if (args.work_root / "captures").is_dir() else []):
        if capture.is_dir() and (not wanted or capture.name in wanted):
            output.append(model_capture(capture, max(1, args.source_rate), max(1, args.destination_rate), args.payload_log if len(output) == 0 else None))
    result = {"schema_version": 1, "model": "sm120_logical_async_campaign_v1", "captures": output, "capture_count": len(output)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"capture_count": len(output), "launch_count": sum(len(row["launches"]) for row in output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
