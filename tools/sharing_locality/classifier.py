"""Sharing-locality classifier (task12 / the sharing-locality clause) — original-plan
GPU-Shalom sharing-locality features grounded vs projection.

Walks an NVBit threadblock proto trace via
`tools.nvbit_tracer.import_trace_contract` and emits a bundle
admitting `workloads/simbench_bw/schemas/sharing_locality.schema.json`.

The original-plan sharing-locality clause enumeration:

- **Three GROUNDED classes** computed directly from per-lane /
  per-warp / per-CTA address sets observed in the trace:
  - ``intra_lane`` — same lane re-touches the same sector
    within the launch.
  - ``intra_warp`` — different lanes within the same warp touch
    the same sector at different times within the launch.
  - ``inter_warp_same_cta`` — different warps within the same
    threadblock (CTA) touch the same sector at different times
    within the launch.

- **Four PROJECTED classes** that are NOT directly observable
  from a single threadblock trace and are emitted as
  typed-unavailable records with the appropriate
  ``*_projection_required`` typed reason; or, when projection
  inputs are supplied to the API, as admitted records with
  ``provenance="projection"`` and explicit ``projection_inputs``:
  - ``inter_cta_same_sm`` — pairs of CTAs that landed on the
    same SM share sectors. Requires an SM-residency map.
  - ``inter_cta_cross_sm`` — pairs of CTAs across different SMs
    share sectors. Requires multi-SM ordering.
  - ``sequential_kernel`` — cross-kernel reuse. Requires kernel
    ordering and a global view across launches.
  - ``private_cold`` — sectors unique to one lane / never reused.
    Requires a global view across all launches.

The original-plan sharing-locality clause negative invariant is enforced by the
schema: a projected-class record cannot carry
``provenance="grounded"``; class_name + provenance combinations
are bound by the per-record `oneOf` branches.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_TOOLS_NVBIT = _HERE.parent / "nvbit_tracer"
if str(_TOOLS_NVBIT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_NVBIT))

import import_trace_contract  # type: ignore[import-not-found]  # noqa: E402


GROUNDED_CLASSES: tuple[str, ...] = (
    "intra_lane",
    "intra_warp",
    "inter_warp_same_cta",
)

PROJECTED_CLASSES: tuple[str, ...] = (
    "inter_cta_same_sm",
    "inter_cta_cross_sm",
    "sequential_kernel",
    "private_cold",
)

# Producer-side typed-reason set. Lockstep at exactly two
# surfaces per the C-AC-C inventory: this constant +
# `workloads/simbench_bw/schemas/sharing_locality.schema.json`
# `$defs.typed_reason.enum`. The
# `SharingLocalityTypedReasonsParityTest` enforcer in
# `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`
# rejects drift.
SHARING_LOCALITY_TYPED_REASONS: frozenset[str] = frozenset({
    "intra_lane_unobservable",
    "intra_warp_unobservable",
    "inter_warp_same_cta_unobservable",
    "inter_cta_same_sm_projection_required",
    "inter_cta_cross_sm_projection_required",
    "sequential_kernel_projection_required",
    "private_cold_projection_required",
})


_PROJECTED_TYPED_REASON: dict[str, str] = {
    "inter_cta_same_sm": "inter_cta_same_sm_projection_required",
    "inter_cta_cross_sm": "inter_cta_cross_sm_projection_required",
    "sequential_kernel": "sequential_kernel_projection_required",
    "private_cold": "private_cold_projection_required",
}


def _walk_threadblock(
    tb: Any,
    *,
    cta_key: str,
    intra_lane_pairs: dict[tuple[str, int, int, int], int],
    intra_warp_pairs: dict[tuple[str, int, int], set[int]],
) -> tuple[int, dict[int, set[int]]]:
    """Walk one threadblock proto, accumulating CTA-keyed
    per-class pair counts in the caller's mutable dicts and
    returning a per-CTA inter-warp accumulator.

    ``cta_key`` is a stable string identity for THIS threadblock
    (the path-derived
    ``tb_path.relative_to(threadblocks_root).as_posix()`` per
    Codex's R23 review directive). Path-derived keying avoids
    relying on ``tb.block_id`` (which synthetic test fixtures may
    not set) and additionally encodes device/stream/kernel
    context so cross-launch caller code can stay correct.

    Returns ``(accesses, inter_warp_per_cta)`` where:

    - ``accesses``: count of (lane, sector) accesses contributed
      by this threadblock.
    - ``inter_warp_per_cta``: ``sector_id -> set[warp_id]`` for
      THIS threadblock only. Same-CTA sharing fires when
      ``len(set) >= 2``. Caller aggregates per-CTA reuse +
      distinct-sector counts.

    The mutable CTA-keyed structures are:

    - ``intra_lane_pairs[(cta_key, warp_id, lane_index, sector_id)] = count``:
      same local lane within the same CTA re-touches the same
      sector. ``count >= 2`` means reuse fired. The R24 fix for
      Codex's R23 review finding that ``warpid_tb`` (and
      therefore ``(warp_id, lane_index)``) is CTA-local — two
      different CTAs with matching local warp/lane IDs are
      distinct lanes, not the same lane.
    - ``intra_warp_pairs[(cta_key, warp_id, sector_id)] = set[lane_index]``:
      DISTINCT lanes within the same warp within the same CTA
      that touched the sector. ``len >= 2`` means warp-level
      sharing fired. Same R24 CTA-keying rationale.
    """
    accesses = 0
    inter_warp_per_cta: dict[int, set[int]] = {}
    for warp_id in sorted(tb.warps.keys()):
        warp = tb.warps[warp_id]
        for inst in warp.instructions:
            if not inst.HasField("memory_op_class"):
                continue
            if not inst.addresses:
                continue
            and_mask = inst.active_mask & inst.predicate_mask
            if and_mask == 0:
                continue
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
                        accesses += 1
                        lane_key = (
                            cta_key, warp_id, lane_index, sector_n,
                        )
                        intra_lane_pairs[lane_key] = (
                            intra_lane_pairs.get(lane_key, 0) + 1
                        )
                        warp_key = (cta_key, warp_id, sector_n)
                        intra_warp_pairs.setdefault(
                            warp_key, set(),
                        ).add(lane_index)
                        inter_warp_per_cta.setdefault(
                            sector_n, set(),
                        ).add(warp_id)
    return accesses, inter_warp_per_cta


def _grounded_record(
    *,
    class_name: str,
    reuse_count: int,
    distinct_pair_count: int,
) -> dict[str, Any]:
    fraction = (
        reuse_count / distinct_pair_count
        if distinct_pair_count > 0 else 0.0
    )
    return {
        "result_kind": "ok",
        "class_name": class_name,
        "provenance": "grounded",
        "reuse_count": reuse_count,
        "distinct_pair_count": distinct_pair_count,
        "fraction": fraction,
        "low_confidence": distinct_pair_count == 0,
    }


def _projected_typed_unavailable(class_name: str) -> dict[str, Any]:
    return {
        "result_kind": "typed_unavailable",
        "class_name": class_name,
        "typed_reason": _PROJECTED_TYPED_REASON[class_name],
        "provenance": "projection",
    }


def _projected_admitted_record(
    *,
    class_name: str,
    projection_inputs: dict[str, Any],
    reuse_count: int,
    distinct_pair_count: int,
) -> dict[str, Any]:
    fraction = (
        reuse_count / distinct_pair_count
        if distinct_pair_count > 0 else 0.0
    )
    return {
        "result_kind": "ok",
        "class_name": class_name,
        "provenance": "projection",
        "projection_inputs": dict(projection_inputs),
        "reuse_count": reuse_count,
        "distinct_pair_count": distinct_pair_count,
        "fraction": fraction,
        "low_confidence": distinct_pair_count == 0,
    }


def compute_sharing_locality(
    *,
    trace_dir: Path,
    manifest_pairing_index: dict[str, str],
    device: dict[str, Any],
    projected_inputs: dict[str, dict[str, Any]] | None = None,
    projected_counts: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Walk every threadblock proto under ``trace_dir/threadblocks``
    and compute the three grounded sharing-locality classes plus
    the four projected classes.

    ``projected_inputs`` (optional) maps a projected class name to
    the projection inputs the caller supplies (an SM-residency
    map, kernel ordering, etc.). When provided AND
    ``projected_counts`` carries the corresponding
    ``(reuse_count, distinct_pair_count)`` pair, the projected
    class lands as an admitted record with
    ``provenance="projection"``. Otherwise the projected class
    lands as a typed-unavailable record with the
    ``*_projection_required`` typed reason.

    The bundle is the schema-conformant dict; the caller writes
    it to disk.
    """
    contract = import_trace_contract.validate_trace_dir(trace_dir)
    if not contract.ok:
        raise RuntimeError(
            f"sharing-locality classifier rejects trace_dir "
            f"{trace_dir}: {contract.typed_reason}: "
            f"{contract.detail}"
        )
    threadblock_pb2 = import_trace_contract._import_pb_modules()[0]

    # Intra-lane and intra-warp accumulators are CTA-keyed
    # (Codex's R23 review fix): `warpid_tb` is CTA-local, so the
    # `(warp_id, lane_index)` tuple is NOT launch-unique. The
    # `cta_key` derived from the threadblock proto's relative
    # path under `threadblocks/` provides a stable per-CTA
    # identity that also encodes device/stream/kernel context.
    intra_lane_pairs: dict[tuple[str, int, int, int], int] = {}
    intra_warp_pairs: dict[tuple[str, int, int], set[int]] = {}
    # Per-CTA inter-warp counts. Each threadblock contributes its
    # own (distinct-sector-count, reuse-sector-count) pair; the
    # totals are summed across CTAs. This scoping is the R23 fix
    # for Codex's R22 review finding: a global accumulator
    # conflated cross-CTA sectors with same-CTA sharing.
    inter_warp_distinct = 0
    inter_warp_reuse = 0
    total_accesses = 0
    threadblocks_root = trace_dir / "threadblocks"
    for tb_path in sorted(threadblocks_root.rglob("*.pb")):
        tb = threadblock_pb2.threadblock()
        tb.ParseFromString(tb_path.read_bytes())
        cta_key = tb_path.relative_to(threadblocks_root).as_posix()
        accesses, inter_warp_per_cta = _walk_threadblock(
            tb,
            cta_key=cta_key,
            intra_lane_pairs=intra_lane_pairs,
            intra_warp_pairs=intra_warp_pairs,
        )
        total_accesses += accesses
        inter_warp_distinct += len(inter_warp_per_cta)
        inter_warp_reuse += sum(
            1 for warps in inter_warp_per_cta.values()
            if len(warps) >= 2
        )

    intra_lane_distinct = len(intra_lane_pairs)
    intra_lane_reuse = sum(
        1 for c in intra_lane_pairs.values() if c >= 2
    )
    intra_warp_distinct = len(intra_warp_pairs)
    # R23 fix: count distinct lanes per (warp_id, sector). Reuse
    # fires only when the warp has at least two distinct lanes
    # touching the sector (true intra-warp sharing). Repeated
    # touches by ONE lane are intra-lane events and must NOT
    # count as intra-warp reuse.
    intra_warp_reuse = sum(
        1 for lanes in intra_warp_pairs.values()
        if len(lanes) >= 2
    )

    grounded = [
        _grounded_record(
            class_name="intra_lane",
            reuse_count=intra_lane_reuse,
            distinct_pair_count=intra_lane_distinct,
        ),
        _grounded_record(
            class_name="intra_warp",
            reuse_count=intra_warp_reuse,
            distinct_pair_count=intra_warp_distinct,
        ),
        _grounded_record(
            class_name="inter_warp_same_cta",
            reuse_count=inter_warp_reuse,
            distinct_pair_count=inter_warp_distinct,
        ),
    ]

    projected = []
    for class_name in PROJECTED_CLASSES:
        if (
            projected_inputs is not None
            and class_name in projected_inputs
            and projected_counts is not None
            and class_name in projected_counts
        ):
            r, n = projected_counts[class_name]
            projected.append(_projected_admitted_record(
                class_name=class_name,
                projection_inputs=projected_inputs[class_name],
                reuse_count=r,
                distinct_pair_count=n,
            ))
        else:
            projected.append(_projected_typed_unavailable(class_name))

    evidence = [
        {
            "result_kind": "info",
            "note_kind": "sharing_locality_pair_count_summary",
            "total_pairs": (
                intra_lane_distinct
                + intra_warp_distinct
                + inter_warp_distinct
            ),
            "addresses_observed": total_accesses,
        }
    ]

    return {
        "schema_version": 1,
        "result_kind": "ok",
        "manifest_pairing_index": dict(manifest_pairing_index),
        "device": dict(device),
        "grounded": grounded,
        "projected": projected,
        "evidence": evidence,
    }


__all__ = [
    "compute_sharing_locality",
    "SHARING_LOCALITY_TYPED_REASONS",
    "GROUNDED_CLASSES",
    "PROJECTED_CLASSES",
]
