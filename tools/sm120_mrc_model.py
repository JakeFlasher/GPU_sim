#!/usr/bin/env python3
"""Bounded trace-driven reuse-distance histograms and logical MRCs.

The model consumes dynamic NVBit protobuf traces and reports *logical* fully
associative LRU scenarios.  It deliberately does not infer SM120 set hashes,
associativity, MSHR counts, slice mapping, replacement policy, or DRAM policy.
Each launch is capped at ``max_references``.  References beyond that cap are
reported in an explicit ``sample`` histogram bin and are excluded from the
modeled MRC denominator; this prevents a truncated prefix from being presented
as a complete launch.

The reuse distance is computed with a Fenwick tree over the bounded sampled
prefix.  Active tree entries are the last occurrence of each sector.  For a
re-reference, the number of active entries after its previous occurrence is
the LRU stack distance.  First references are ``cold`` and distances beyond
the configured exact histogram range are ``overflow``.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.sm120_trace_addresses import load_bindings as load_dynamic_trace_bindings, lane_addresses


DEFAULT_CAPACITIES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Values are from dynamic_trace.proto.  Keeping these local makes the parser
# usable with the authenticated generated binding without depending on a
# generated enum wrapper API.
SPACE_GLOBAL = 1
SPACE_SHARED = 2
SPACE_LOCAL = 3
SPACE_GENERIC = 4
CLASS_GLOBAL = 1
CLASS_SHARED = 2
CLASS_LOCAL = 3
CLASS_TEXTURE = 4
CLASS_SURFACE = 5
CLASS_ATOMIC = 6
CLASS_TMA = 7
CLASS_ASYNC_COPY = 8
CLASS_CLUSTER_DSMEM = 9
CLASS_TMEM = 10


class Fenwick:
    """Fenwick tree for active last-occurrence positions."""

    __slots__ = ("tree",)

    def __init__(self, size: int) -> None:
        self.tree = [0] * (size + 1)

    def add(self, index: int, value: int) -> None:
        n = len(self.tree)
        while index < n:
            self.tree[index] += value
            index += index & -index

    def prefix(self, index: int) -> int:
        total = 0
        while index > 0:
            total += self.tree[index]
            index -= index & -index
        return total


def _empty_histogram(max_distance_bin: int) -> collections.Counter[str]:
    # Keep sparse distance bins, but always expose the semantic boundary bins.
    histogram: collections.Counter[str] = collections.Counter()
    histogram["cold"] = 0
    histogram["overflow"] = 0
    histogram["sample"] = 0
    return histogram


def analyze_sequence(
    sequence: list[int],
    capacities: list[int],
    max_distance_bin: int,
    sample_bin: int = 0,
    scenario_semantics: str = "sampled_sequence",
    warmup: list | None = None,
) -> dict[str, Any]:
    """Return a bounded exact reuse histogram and its logical LRU MRC."""

    # Every requested capacity must fit in the exact portion of the
    # histogram.  Otherwise an overflow distance could be below a larger
    # capacity and the MRC would conservatively misclassify a possible hit as
    # a miss.  The CLI normally supplies 8192 for capacities up to 8192; this
    # guard also makes the small direct microbenchmarks and future callers
    # self-consistent.
    max_distance_bin = max(1, int(max_distance_bin), max([int(x) for x in capacities] or [1]))
    warmup = warmup or []
    n = len(warmup) + len(sequence)
    fenwick = Fenwick(n)
    last_seen: dict[int, int] = {}
    histogram = _empty_histogram(max_distance_bin)
    references = 0
    for position, sector in enumerate(warmup + sequence, 1):
        scored = position > len(warmup)
        references += int(scored)
        old_position = last_seen.get(sector)
        if old_position is None:
            if scored:
                histogram["cold"] += 1
        else:
            # Entries after old_position are unique sectors whose latest
            # occurrence intervened since the previous reference.
            distance = fenwick.prefix(position - 1) - fenwick.prefix(old_position)
            if scored and distance >= max_distance_bin:
                histogram["overflow"] += 1
            elif scored:
                histogram[f"distance_{distance}"] += 1
            fenwick.add(old_position, -1)
        fenwick.add(position, 1)
        last_seen[sector] = position

    histogram["sample"] = max(0, int(sample_bin))
    cold = int(histogram["cold"])
    overflow = int(histogram["overflow"])
    modeled = references
    mrc: dict[str, dict[str, Any]] = {}
    for capacity in capacities:
        capacity = max(1, int(capacity))
        misses = cold + overflow
        for key, count in histogram.items():
            if not key.startswith("distance_"):
                continue
            try:
                distance = int(key.split("_", 1)[1])
            except ValueError:
                continue
            if distance >= capacity:
                misses += int(count)
        hits = max(0, modeled - misses)
        mrc[str(capacity)] = {
            "capacity_sectors": capacity,
            "modeled_references": modeled,
            "hits": hits,
            "misses": misses,
            "miss_ratio": misses / modeled if modeled else None,
            "hit_ratio": hits / modeled if modeled else None,
        }
    return {
        "references": modeled,
        "cold_references": cold,
        "overflow_references": overflow,
        "sample_bin_references": int(histogram["sample"]),
        "histogram": {key: int(value) for key, value in sorted(histogram.items())},
        "mrc": mrc,
        "max_distance_bin": max_distance_bin,
        "scenario_semantics": scenario_semantics,
    }


def _field_present(message: Any, field: str) -> bool:
    try:
        return bool(message.HasField(field))
    except (AttributeError, ValueError):
        return False


def _space_and_class(address: Any, instruction: Any) -> tuple[int, int, bool]:
    has_space = _field_present(address, "address_space")
    space = int(address.address_space) if has_space else 0
    # Treat an explicitly serialized UNSPECIFIED enum as absent so the
    # instruction-level class can still provide a coarse route.
    has_space = bool(has_space and space != 0)
    try:
        op_class = int(instruction.memory_op_class)
    except (AttributeError, TypeError, ValueError):
        op_class = 0
    return space, op_class, has_space


def route_accepts(route: str, space: int, op_class: int, has_space: bool = False) -> tuple[bool, str]:
    """Classify an address for a declared logical cache route."""

    if route == "all":
        return True, "included_all"
    if op_class in (CLASS_TMA, CLASS_ASYNC_COPY):
        return False, "excluded_async_or_tma"
    # An explicit address-space field is stronger than a broad/unspecified
    # memory-op class.  This prevents a shared address tagged by a generic
    # class from leaking into a global/local cache route.
    if has_space:
        if route == "shared":
            return (space == SPACE_SHARED), "shared"
        if route == "global":
            return (space == SPACE_GLOBAL), "global"
        if route == "global_shared":
            return (space in (SPACE_GLOBAL, SPACE_SHARED)), "global_or_shared"
        return (space in (SPACE_GLOBAL, SPACE_LOCAL, SPACE_GENERIC)), "global_local_generic"
    if route == "shared":
        return (space == SPACE_SHARED or op_class == CLASS_SHARED), "shared"
    if route == "global":
        return (space == SPACE_GLOBAL or op_class == CLASS_GLOBAL), "global"
    if route == "global_shared":
        return (
            space in (SPACE_GLOBAL, SPACE_SHARED) or op_class in (CLASS_GLOBAL, CLASS_SHARED)
        ), "global_or_shared"
    # The default cache-like route includes global, local and generic memory.
    accepted = space in (SPACE_GLOBAL, SPACE_LOCAL, SPACE_GENERIC) or op_class in (
        CLASS_GLOBAL,
        CLASS_LOCAL,
        CLASS_ATOMIC,
        CLASS_TEXTURE,
        CLASS_SURFACE,
        CLASS_CLUSTER_DSMEM,
        CLASS_TMEM,
    )
    return accepted, "global_local_generic"


def _lane_mask(address: Any, value_count: int) -> int:
    raw_mask = int(getattr(address, "lane_mask", 0) or 0)
    if raw_mask:
        return raw_mask
    return (1 << min(value_count, 32)) - 1 if value_count else 0


def _address_width(address: Any, instruction: Any) -> int:
    width = int(getattr(address, "transfer_width", 0) or 0)
    if not width:
        width = int(getattr(instruction, "data_width", 0) or 0)
    return max(1, width)


def collect_launch(
    capture: Path,
    device_id: int,
    stream_id: int,
    kernel: Any,
    bindings: Any,
    route: str,
    max_references: int,
) -> tuple[list[int], dict[str, int], int, int]:
    """Collect a bounded sector prefix and route counters for one launch."""

    sequence: list[int] = []
    counts: collections.Counter[str] = collections.Counter()
    raw_sector_references = 0
    address_records = 0
    root = capture / "threadblocks" / f"device_{device_id}" / f"stream_{stream_id}" / f"kernel_{kernel.id}"
    for path in sorted(root.glob("*.pb")):
        tb = bindings.threadblock_pb2.threadblock()
        try:
            tb.ParseFromString(path.read_bytes())
        except (OSError, ValueError):
            counts["decode_error"] += 1
            continue
        for warp_id, warp in sorted(tb.warps.items()):
            for instruction in warp.instructions:
                for address in instruction.addresses:
                    space, op_class, has_space = _space_and_class(address, instruction)
                    accepted, reason = route_accepts(route, space, op_class, has_space)
                    counts[reason] += 1
                    if not has_space:
                        counts["missing_address_space"] += 1
                    if not accepted:
                        continue
                    lanes, width = lane_addresses(address, instruction)
                    address_records += len(lanes)
                    counts[f"compression_{address.compression_format}"] += 1
                    operand_sectors = set()
                    resolved_space = space if has_space else (
                        SPACE_SHARED if op_class == CLASS_SHARED else
                        SPACE_LOCAL if op_class == CLASS_LOCAL else SPACE_GLOBAL)
                    # Shared addresses are CTA-private. Local addresses are
                    # thread-private; neither may alias a global virtual address.
                    for lane, value in lanes:
                        start = int(value)
                        end = start + width - 1
                        for sector in range(start >> 5, (end >> 5) + 1):
                            private = (path.name, warp_id, lane) if resolved_space == SPACE_LOCAL else (
                                (path.name,) if resolved_space == SPACE_SHARED else ())
                            operand_sectors.add((resolved_space, private, sector))
                    raw_sector_references += len(operand_sectors)
                    remaining = max(0, max_references - len(sequence))
                    sequence.extend(sorted(operand_sectors)[:remaining])
    counts["raw_sector_references"] = raw_sector_references
    counts["sampled_sector_references"] = len(sequence)
    counts["sample_bin_references"] = max(0, raw_sector_references - len(sequence))
    return sequence, dict(counts), raw_sector_references, address_records


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
                labels[int(inst.get("pc_num_dec"))] = str(inst.get("opcode") or "UNKNOWN")
            except (TypeError, ValueError):
                continue
    return labels


def sidecar_summary(capture: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    path = capture / "extra_info/tma_trace.jsonl"
    if not path.is_file():
        return result
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            key = str(row.get("kernel_id", "unknown"))
            bucket = result.setdefault(key, {"records": 0, "bytes": 0, "opcodes": set()})
            bucket["records"] += 1
            bucket["bytes"] += max(0, int(row.get("byte_count", 0) or 0))
            if row.get("opcode"):
                bucket["opcodes"].add(str(row["opcode"]))
    for bucket in result.values():
        bucket["opcodes"] = sorted(bucket["opcodes"])
    return result


def _scenario_rows(
    sequence: list[int],
    raw_references: int,
    capacities: list[int],
    max_distance_bin: int,
) -> dict[str, dict[str, Any]]:
    sampled_tail = max(0, raw_references - len(sequence))
    rows = {
        "instrumented_receive_order": analyze_sequence(
            sequence,
            capacities,
            max_distance_bin,
            sampled_tail,
            "instrumented_receive_order_sampled_prefix",
        ),
        "reverse_warp_cta_order": analyze_sequence(
            list(reversed(sequence)),
            capacities,
            max_distance_bin,
            sampled_tail,
            "reverse_of_sampled_prefix;full_hardware_interleaving_unknown",
        ),
    }
    # This is a warm logical pass over the sampled prefix, not a claim that a
    # GPU cache survives a launch or that the second pass has identical timing.
    rows["warm_second_pass"] = analyze_sequence(
        sequence,
        capacities,
        max_distance_bin,
        sampled_tail,
        "second_pass_only_after_sampled_prefix_warmup;logical_warm_pass",
        warmup=sequence,
    )
    return rows


def extract_capture(
    capture: Path,
    route: str,
    max_references: int,
    capacities: list[int],
    max_distance_bin: int,
) -> dict[str, Any]:
    bindings = load_dynamic_trace_bindings()
    trace = bindings.trace_pb2.Trace()
    trace.ParseFromString((capture / "dynamic_trace.pb").read_bytes())
    tma = sidecar_summary(capture)
    launches: list[dict[str, Any]] = []
    for device_id, device in sorted(trace.gpu_device.items()):
        for stream_id, stream in sorted(device.streams.items()):
            for kernel in stream.kernels:
                sequence, route_counts, raw_refs, address_records = collect_launch(
                    capture,
                    int(device_id),
                    int(stream_id),
                    kernel,
                    bindings,
                    route,
                    max_references,
                )
                launches.append(
                    {
                        "device_id": int(device_id),
                        "stream_id": int(stream_id),
                        "kernel_id": int(kernel.id),
                        "kernel_name": kernel.name,
                        "grid": [kernel.grid_dim.x, kernel.grid_dim.y, kernel.grid_dim.z],
                        "block": [kernel.block_dim.x, kernel.block_dim.y, kernel.block_dim.z],
                        "address_records": address_records,
                        "raw_sector_references": raw_refs,
                        "sampled_sector_references": len(sequence),
                        "sample_truncated": raw_refs > len(sequence),
                        "route_counts": route_counts,
                        "tma": tma.get(str(kernel.id), {"records": 0, "bytes": 0, "opcodes": []}),
                        "reuse_distance_scenarios": _scenario_rows(
                            sequence, raw_refs, capacities, max_distance_bin
                        ),
                        "ordering_note": "threadblock file order is an explicit logical scenario; hardware interleaving is unknown",
                    }
                )
    return {
        "capture_name": capture.name,
        "binary_version": int(trace.binary_version),
        "trace_contract_id": trace.trace_contract_id,
        "launches": launches,
        "route": route,
        "sector_bytes": 32,
        "max_references_per_launch": max_references,
        "max_distance_bin": max_distance_bin,
        "capacities_sectors": capacities,
        "input_domains": ["dynamic_trace.pb", "threadblocks/*.pb", "tma_trace.jsonl"],
        "ncu_used_as_feature": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/dev/shm/sm120_rtx_5090_work"))
    parser.add_argument("--capture", action="append")
    parser.add_argument("--route", choices=["all", "global", "shared", "global_shared", "global_local"], default="global_local")
    parser.add_argument("--max-references", type=int, default=250_000)
    parser.add_argument("--max-distance-bin", type=int, default=8192)
    parser.add_argument("--capacity-sectors", type=int, action="append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    capacities = sorted(set(args.capacity_sectors or DEFAULT_CAPACITIES))
    max_references = max(1, int(args.max_references))
    max_distance_bin = max(1, int(args.max_distance_bin), max(capacities or [1]))
    captures_root = args.work_root / "captures"
    wanted = set(args.capture or [])
    rows: list[dict[str, Any]] = []
    if captures_root.is_dir():
        for capture in sorted(captures_root.iterdir()):
            if not capture.is_dir() or (wanted and capture.name not in wanted):
                continue
            if not (capture / "dynamic_trace.pb").is_file():
                continue
            rows.append(extract_capture(capture, args.route, max_references, capacities, max_distance_bin))
    result = {
        "schema_version": 1,
        "model": "sm120_reuse_distance_mrc_v1",
        "route": args.route,
        "sector_bytes": 32,
        "capacities_sectors": capacities,
        "max_references_per_launch": max_references,
        "max_distance_bin": max_distance_bin,
        "capture_count": len(rows),
        "launch_count": sum(len(row["launches"]) for row in rows),
        "physical_non_claim": [
            "set_hash",
            "associativity",
            "MSHR_count",
            "slice_mapping",
            "replacement_policy",
            "DRAM_arbitration_policy",
        ],
        "captures": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"capture_count": len(rows), "launch_count": result["launch_count"], "route": args.route}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
