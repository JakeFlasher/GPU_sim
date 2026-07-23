"""Schedule-marginalized order projections producer (task13 / the
order-projections clause).

GPU execution order is not uniquely observable under non-root
tracing and can vary by scheduling, residency, and latency. The
original-plan order-projections clause names six plausible
orderings of the access stream, computes a per-projection
hit-rate proxy via LRU stack distance over per-touched-sector
access events against the cache-model L2 capacity, and reports
a `disagreement_variance` scalar across the admitted
projections.

The six projections:

- ``lane``: lane index → sector access ordering (each lane's
  access stream in lane-index order).
- ``warp_local``: per-warp instruction-order ordering (each
  warp's accesses interleaved within the warp's
  instruction sequence).
- ``cta_local``: per-CTA ordering (all warps interleaved by
  warp-index, then per-warp instruction order).
- ``sm_local``: per-SM ordering. Requires `sm_id` to be
  present on at least one instruction; otherwise the
  projection lands typed-unavailable
  (`projection_unavailable_sm_local`, the original-plan
  exclusion-taxonomy enumerated reason).
- ``kernel_l2``: kernel-launch ordering (all CTAs / warps
  interleaved by lexicographic CTA path, then warp order,
  then instruction order). The "L2-as-stream" baseline.
- ``sequential_allocation``: order grouped by
  `allocation_id`. Requires the
  `extra_info/allocations.json` sidecar; otherwise the
  projection lands typed-unavailable
  (`sequential_allocation_projection_unobservable_no_allocations_sidecar`).

Per-projection hit-rate proxy:

For each access event in the projection's order, compute the
LRU-stack distance for the access's 32B sector over the
projection's prior accesses (within this kernel-launch's
ordering, irrespective of CTA scope). An access is modeled as
a hit iff its reuse distance is strictly less than
``capacity_bytes // 32``. Cold accesses are misses. The
hit-rate proxy is the fraction of accesses that are hits.

`disagreement_variance` is the population variance of the
admitted projections' hit-rate proxies. When fewer than two
projections admit, the scalar is replaced by a
typed-unavailable record carrying
`insufficient_admitted_projections`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_TOOLS_NVBIT = _HERE.parent / "nvbit_tracer"
_TOOLS_HRD_PROFILER = _HERE.parent / "hrd_profiler"
if str(_TOOLS_NVBIT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_NVBIT))
if str(_TOOLS_HRD_PROFILER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_HRD_PROFILER))

import import_trace_contract  # type: ignore[import-not-found]  # noqa: E402

from reuse_distance_tracker import (  # type: ignore[import-not-found]  # noqa: E402
    ReuseDistanceTracker,
)


PROJECTION_NAMES: tuple[str, ...] = (
    "lane",
    "warp_local",
    "cta_local",
    "sm_local",
    "kernel_l2",
    "sequential_allocation",
)

# Producer-side typed-reason set. Lockstep at exactly two
# surfaces per the C-AC-C inventory: this constant +
# `workloads/simbench_bw/schemas/order_projections.schema.json`
# `$defs.typed_reason.enum`. The SM-local unavailable arm uses
# the original-plan exclusion-taxonomy enumerated reason
# (``projection_unavailable_sm_local``) per Codex's R25 review;
# the other six are round-added and live in
# ``ROUND_ADDED_TYPED_REASONS``.
ORDER_PROJECTIONS_TYPED_REASONS: frozenset[str] = frozenset({
    "lane_projection_unobservable",
    "warp_local_projection_unobservable",
    "cta_local_projection_unobservable",
    "projection_unavailable_sm_local",
    "kernel_l2_projection_unobservable",
    "sequential_allocation_projection_unobservable_no_allocations_sidecar",
    "insufficient_admitted_projections",
})


_PROVENANCE: dict[str, str] = {
    "lane": (
        "lane index then per-warp instruction order; each lane's "
        "touched-sector stream sequenced before the next lane's"
    ),
    "warp_local": (
        "per-warp instruction order; accesses within each warp "
        "interleaved by lane index per instruction"
    ),
    "cta_local": (
        "per-CTA: warps in numeric order, then per-warp instruction "
        "order, then per-instruction lane index"
    ),
    "sm_local": (
        "per-SM (sm_id) ordering; CTAs grouped by SM residency, "
        "then CTA-local order"
    ),
    "kernel_l2": (
        "kernel-launch L2 view: lexicographic CTA path, then per-CTA "
        "order"
    ),
    "sequential_allocation": (
        "grouped by allocation_id (allocations.json sidecar); within "
        "each allocation, lexicographic CTA path then per-CTA order"
    ),
}


def _access_events(
    trace_dir: Path,
    *,
    threadblock_pb2_mod: Any,
    allocations: dict[int, Any] | None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Walk every threadblock proto and emit per-touched-sector
    access events. Returns ``(events, total_threadblocks, any_sm_id)``
    where ``any_sm_id`` is True when at least one event carried a
    ``sm_id`` (so the SM-local projection is observable)."""
    events: list[dict[str, Any]] = []
    threadblocks_root = trace_dir / "threadblocks"
    any_sm_id = False
    total_tbs = 0
    for tb_path in sorted(threadblocks_root.rglob("*.pb")):
        cta_key = tb_path.relative_to(threadblocks_root).as_posix()
        tb = threadblock_pb2_mod.threadblock()
        tb.ParseFromString(tb_path.read_bytes())
        total_tbs += 1
        for warp_index_in_tb, warp_id in enumerate(sorted(tb.warps.keys())):
            warp = tb.warps[warp_id]
            for inst_index, inst in enumerate(warp.instructions):
                if not inst.HasField("memory_op_class"):
                    continue
                if not inst.addresses:
                    continue
                and_mask = inst.active_mask & inst.predicate_mask
                if and_mask == 0:
                    continue
                sm_id = (
                    inst.sm_id if inst.HasField("sm_id") else None
                )
                if sm_id is not None:
                    any_sm_id = True
                for addr_record in inst.addresses:
                    recon = import_trace_contract._reconstruct_lane_vector(
                        addr_record, and_mask
                    )
                    lane_vector = recon[0]
                    if lane_vector is None:
                        continue
                    data_width = (
                        addr_record.data_width
                        if addr_record.data_width
                        else inst.data_width
                    )
                    if data_width <= 0:
                        continue
                    addr_alloc_id = (
                        addr_record.allocation_id
                        if addr_record.HasField("allocation_id")
                        else 0
                    )
                    for lane_index, lane_addr in enumerate(lane_vector):
                        if not ((and_mask >> lane_index) & 1):
                            continue
                        if lane_addr == import_trace_contract.LANE_INACTIVE_SENTINEL:
                            continue
                        access_start = int(lane_addr)
                        access_end = access_start + int(data_width)
                        first_sector = access_start // 32
                        last_sector = (access_end - 1) // 32
                        for sector_n in range(first_sector, last_sector + 1):
                            sector_byte = sector_n * 32
                            allocation_key = _resolve_allocation_key(
                                sector_byte, allocations, addr_alloc_id,
                            )
                            events.append({
                                "cta_key": cta_key,
                                "warp_id": warp_id,
                                "warp_index_in_tb": warp_index_in_tb,
                                "inst_index": inst_index,
                                "lane_index": lane_index,
                                "sector_id": sector_n,
                                "sm_id": sm_id,
                                "allocation_key": allocation_key,
                            })
    return events, total_tbs, any_sm_id


def _resolve_allocation_key(
    sector_byte_addr: int,
    allocations: dict[int, Any] | None,
    wire_allocation_id: int,
) -> str | None:
    """Resolve a stable allocation key for an access event.
    Returns the allocations-sidecar entry's id when the address
    falls in a known range; otherwise the wire-side
    `allocation_id` (when non-zero); otherwise None (the access
    cannot be sequenced under the sequential-allocation
    projection)."""
    if allocations:
        for alloc_id, entry in allocations.items():
            base = entry.base_address
            if base <= sector_byte_addr < base + entry.size_bytes:
                return f"alloc:{alloc_id}"
    if wire_allocation_id:
        return f"wire:{wire_allocation_id}"
    return None


def _hit_rate_proxy_under_order(
    ordered_events: list[dict[str, Any]],
    *,
    capacity_bytes: int,
) -> tuple[float, int]:
    """Compute the LRU-stack-distance hit-rate proxy under the
    given ordering. Returns ``(hit_rate, access_count)``.

    A sector access is modeled as a hit iff its LRU stack
    distance < ``capacity_bytes // 32``. Cold accesses are
    misses. The proxy is the fraction of hit accesses over the
    total events. When ``access_count == 0``, returns
    ``(0.0, 0)``."""
    if not ordered_events:
        return 0.0, 0
    sectors_per_capacity = max(1, capacity_bytes // 32)
    # R53: replaced the O(N) `lru.index/remove/append` per access
    # (which made this loop O(N^2) and hung warm_cold_pair_relaunch
    # trace_00 alongside the parallel HRD-producer hang) with a
    # Fenwick-tree-backed reuse-distance tracker. Same 0-indexed
    # `rd` semantics; same hit/miss admission rule.
    n = len(ordered_events)
    tracker = ReuseDistanceTracker(max_accesses=n)
    hits = 0
    for t, ev in enumerate(ordered_events, start=1):
        rd, state = tracker.access(ev["sector_id"], t)
        if state == "finite" and rd is not None and rd < sectors_per_capacity:
            hits += 1
    return hits / n, n


def _order_lane(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda e: (e["lane_index"], e["cta_key"], e["warp_id"], e["inst_index"]),
    )


def _order_warp_local(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda e: (e["cta_key"], e["warp_id"], e["inst_index"], e["lane_index"]),
    )


def _order_cta_local(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda e: (e["cta_key"], e["warp_index_in_tb"], e["inst_index"], e["lane_index"]),
    )


def _order_sm_local(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda e: (
            e["sm_id"] if e["sm_id"] is not None else -1,
            e["cta_key"], e["warp_id"], e["inst_index"], e["lane_index"],
        ),
    )


def _order_kernel_l2(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Lexicographic CTA path order, then per-CTA ordering.
    return sorted(
        events,
        key=lambda e: (e["cta_key"], e["warp_index_in_tb"], e["inst_index"], e["lane_index"]),
    )


def _order_sequential_allocation(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if any(e["allocation_key"] is None for e in events):
        return None
    return sorted(
        events,
        key=lambda e: (
            e["allocation_key"], e["cta_key"], e["warp_index_in_tb"],
            e["inst_index"], e["lane_index"],
        ),
    )


def _typed_unavailable_projection(
    name: str, typed_reason: str,
) -> dict[str, Any]:
    return {
        "result_kind": "typed_unavailable",
        "name": name,
        "typed_reason": typed_reason,
        "provenance": _PROVENANCE[name],
    }


def _admitted_projection(
    *, name: str, hit_rate: float, access_count: int,
) -> dict[str, Any]:
    return {
        "result_kind": "ok",
        "name": name,
        "provenance": _PROVENANCE[name],
        "hit_rate_proxy": hit_rate,
        "access_count": access_count,
        "low_confidence": access_count == 0,
    }


def _population_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def compute_order_projections(
    *,
    trace_dir: Path,
    manifest_pairing_index: dict[str, str],
    device: dict[str, Any],
    capacity_bytes: int,
) -> dict[str, Any]:
    """Walk the trace directory's threadblock protos, build the
    six order projections, compute per-projection hit-rate
    proxies, and aggregate the disagreement-variance scalar.

    Raises RuntimeError on wire-contract failure (the producer
    cannot emit a meaningful bundle from broken evidence; the
    caller surfaces the failure into a typed
    `hrd_skip_reason`-equivalent if needed).
    """
    contract = import_trace_contract.validate_trace_dir(trace_dir)
    if not contract.ok:
        raise RuntimeError(
            f"order-projections producer rejects trace_dir "
            f"{trace_dir}: {contract.typed_reason}: "
            f"{contract.detail}"
        )
    threadblock_pb2_mod = import_trace_contract._import_pb_modules()[0]
    allocations, alloc_err = (
        import_trace_contract.load_allocations_sidecar(trace_dir)
    )
    if alloc_err is not None:
        raise RuntimeError(
            f"order-projections producer rejects allocations sidecar: "
            f"{alloc_err.typed_reason}: {alloc_err.detail}"
        )
    events, total_tbs, any_sm_id = _access_events(
        trace_dir,
        threadblock_pb2_mod=threadblock_pb2_mod,
        allocations=allocations,
    )
    return _build_bundle(
        events=events,
        total_tbs=total_tbs,
        any_sm_id=any_sm_id,
        manifest_pairing_index=manifest_pairing_index,
        device=device,
        capacity_bytes=capacity_bytes,
    )


def _build_bundle(
    *,
    events: list[dict[str, Any]],
    total_tbs: int,
    any_sm_id: bool,
    manifest_pairing_index: dict[str, str],
    device: dict[str, Any],
    capacity_bytes: int,
) -> dict[str, Any]:
    """Inner builder; takes a precomputed event list. Split out
    so test fixtures can exercise the projection logic without
    going through the proto walker."""
    projections: list[dict[str, Any]] = []
    admitted_hit_rates: list[float] = []

    def _emit_admitted(name: str, ordered: list[dict[str, Any]]) -> None:
        hit_rate, access_count = _hit_rate_proxy_under_order(
            ordered, capacity_bytes=capacity_bytes,
        )
        projections.append(_admitted_projection(
            name=name, hit_rate=hit_rate, access_count=access_count,
        ))
        admitted_hit_rates.append(hit_rate)

    _emit_admitted("lane", _order_lane(events))
    _emit_admitted("warp_local", _order_warp_local(events))
    _emit_admitted("cta_local", _order_cta_local(events))

    if any_sm_id:
        _emit_admitted("sm_local", _order_sm_local(events))
    else:
        projections.append(_typed_unavailable_projection(
            "sm_local",
            "projection_unavailable_sm_local",
        ))

    _emit_admitted("kernel_l2", _order_kernel_l2(events))

    seq_order = _order_sequential_allocation(events)
    if seq_order is None:
        projections.append(_typed_unavailable_projection(
            "sequential_allocation",
            "sequential_allocation_projection_unobservable_no_allocations_sidecar",
        ))
    else:
        _emit_admitted("sequential_allocation", seq_order)

    if len(admitted_hit_rates) >= 2:
        disagreement_variance = {
            "result_kind": "ok",
            "value": _population_variance(admitted_hit_rates),
            "admitted_projection_count": len(admitted_hit_rates),
        }
    else:
        disagreement_variance = {
            "result_kind": "typed_unavailable",
            "typed_reason": "insufficient_admitted_projections",
            "admitted_projection_count": len(admitted_hit_rates),
        }

    evidence = [
        {
            "result_kind": "info",
            "note_kind": "order_projections_access_summary",
            "total_access_events": len(events),
            "threadblocks_walked": total_tbs,
        }
    ]

    return {
        "schema_version": 1,
        "result_kind": "ok",
        "manifest_pairing_index": dict(manifest_pairing_index),
        "device": dict(device),
        "projections": projections,
        "disagreement_variance": disagreement_variance,
        "evidence": evidence,
    }


__all__ = [
    "compute_order_projections",
    "ORDER_PROJECTIONS_TYPED_REASONS",
    "PROJECTION_NAMES",
]
