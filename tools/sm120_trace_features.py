#!/usr/bin/env python3
"""Extract compact SM120 trace/locality features without timing replay.

The native protobuf binding is read directly from the retained NVBit trace.
Only structural addresses, masks, static PC labels and TMA descriptor bytes are
used; no NCU counter or measured runtime is consumed by this feature path.
The cache curves are logical fully-associative LRU scenarios, not an SM120 set
hash, associativity, MSHR or replacement claim.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.sm120_trace_addresses import load_bindings as load_dynamic_trace_bindings, operand_sectors


def static_labels(capture: Path) -> dict[int, str]:
    path = capture / "extra_info/native_static_metadata.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    labels: dict[int, str] = {}
    for kernel in value.get("kernels", []) if isinstance(value, dict) else []:
        for inst in kernel.get("instructions", []) if isinstance(kernel, dict) else []:
            try:
                labels[int(inst.get("pc_num_dec"))] = str(inst.get("opcode") or inst.get("sass") or "UNKNOWN")
            except (TypeError, ValueError):
                continue
    return labels


def sidecar_summary(capture: Path) -> dict[str, Any]:
    per_kernel: dict[str, dict[str, Any]] = {}
    path = capture / "extra_info/tma_trace.jsonl"
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                key = str(row.get("kernel_id", "unknown"))
                bucket = per_kernel.setdefault(key, {"records": 0, "bytes": 0, "opcodes": set()})
                bucket["records"] += 1
                bucket["bytes"] += max(0, int(row.get("byte_count", 0) or 0))
                if row.get("opcode"):
                    bucket["opcodes"].add(str(row["opcode"]))
    for bucket in per_kernel.values():
        bucket["opcodes"] = sorted(bucket["opcodes"])
    return per_kernel


def lru_curve(sectors: list[int], capacities: list[int], sample_limit: int) -> dict[str, Any]:
    sequence = sectors[:sample_limit]
    result: dict[str, Any] = {"sampled_references": len(sequence), "capacity_sectors": {}}
    for capacity in capacities:
        cache: collections.OrderedDict[int, None] = collections.OrderedDict()
        hits = 0
        for sector in sequence:
            if sector in cache:
                hits += 1
                cache.move_to_end(sector)
            else:
                cache[sector] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
        result["capacity_sectors"][str(capacity)] = {
            "hits": hits,
            "misses": len(sequence) - hits,
            "hit_rate": hits / len(sequence) if sequence else None,
        }
    # A bounded exact stack-distance sample for the first 20k references keeps
    # this path useful on tiny toys while avoiding an O(N^2) scan on CUTLASS.
    distances: list[int | None] = []
    recent: list[int] = []
    for sector in sequence[:20_000]:
        try:
            index = recent.index(sector)
        except ValueError:
            distances.append(None)
        else:
            distances.append(index)
            recent.pop(index)
        recent.insert(0, sector)
        if len(recent) > 100_000:
            del recent[100_000:]
    numeric = [value for value in distances if value is not None]
    result["reuse_distance_sample"] = {
        "references": len(distances),
        "hits": len(numeric),
        "cold_references": len(distances) - len(numeric),
        "median": sorted(numeric)[len(numeric) // 2] if numeric else None,
        "max": max(numeric) if numeric else None,
        "ordering": "instrumented_receive_order_logical_scenario",
    }
    return result


def extract_capture(capture: Path, sample_limit: int) -> dict[str, Any]:
    bindings = load_dynamic_trace_bindings()
    trace = bindings.trace_pb2.Trace()
    trace.ParseFromString((capture / "dynamic_trace.pb").read_bytes())
    labels = static_labels(capture)
    tma = sidecar_summary(capture)
    launch_rows: list[dict[str, Any]] = []
    for device_id, device in sorted(trace.gpu_device.items()):
        for stream_id, stream in sorted(device.streams.items()):
            for kernel in stream.kernels:
                key = str(kernel.id)
                sectors: list[int] = []
                address_records = 0
                instruction_records = 0
                opcode_counts: collections.Counter[str] = collections.Counter()
                warp_count = 0
                root = capture / "threadblocks" / f"device_{device_id}" / f"stream_{stream_id}" / f"kernel_{kernel.id}"
                for path in sorted(root.glob("*.pb")):
                    tb = bindings.threadblock_pb2.threadblock()
                    tb.ParseFromString(path.read_bytes())
                    warp_count += len(tb.warps)
                    for warp in tb.warps.values():
                        for instruction in warp.instructions:
                            instruction_records += 1
                            pc = int(instruction.pc)
                            opcode_counts[labels.get(pc, "UNKNOWN").split(" ", 1)[0]] += 1
                            for address in instruction.addresses:
                                decoded, lane_count = operand_sectors(address, instruction)
                                address_records += lane_count
                                sectors.extend(decoded)
                launch_rows.append(
                    {
                        "device_id": int(device_id),
                        "stream_id": int(stream_id),
                        "kernel_id": int(kernel.id),
                        "kernel_name": kernel.name,
                        "grid": [kernel.grid_dim.x, kernel.grid_dim.y, kernel.grid_dim.z],
                        "block": [kernel.block_dim.x, kernel.block_dim.y, kernel.block_dim.z],
                        "warps": warp_count,
                        "instruction_records": instruction_records,
                        "address_records": address_records,
                        "unique_sectors": len(set(sectors)),
                        "opcode_counts": dict(opcode_counts),
                        "tma": tma.get(key, {"records": 0, "bytes": 0, "opcodes": []}),
                        "locality": lru_curve(sectors, [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], sample_limit),
                        "locality_scenarios": {
                            "instrumented_receive_order": lru_curve(sectors, [32, 128, 512, 2048, 8192], sample_limit),
                            "reverse_warp_cta_order": lru_curve(list(reversed(sectors)), [32, 128, 512, 2048, 8192], sample_limit),
                            "warm_second_pass": lru_curve((sectors + sectors)[: max(1, sample_limit)], [32, 128, 512, 2048, 8192], sample_limit),
                        },
                        "address_order": "trace_threadblock_file_order; hardware interleaving unknown",
                    }
                )
    return {
        "capture_name": capture.name,
        "binary_version": int(trace.binary_version),
        "trace_contract_id": trace.trace_contract_id,
        "launches": launch_rows,
        "static_pc_label_count": len(labels),
        "input_domains": ["dynamic_trace.pb", "native_static_metadata.json", "tma_trace.jsonl"],
        "ncu_used_as_feature": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/dev/shm/sm120_rtx_5090_work"))
    parser.add_argument("--capture", action="append")
    parser.add_argument("--sample-limit", type=int, default=100_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    captures_root = args.work_root / "captures"
    wanted = set(args.capture or [])
    rows = []
    for capture in sorted(captures_root.iterdir() if captures_root.is_dir() else []):
        if capture.is_dir() and (not wanted or capture.name in wanted) and (capture / "dynamic_trace.pb").is_file():
            rows.append(extract_capture(capture, max(1, args.sample_limit)))
    result = {"schema_version": 1, "model": "sm120_direct_trace_locality_v1", "capture_count": len(rows), "captures": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"capture_count": len(rows), "launch_count": sum(len(row["launches"]) for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
