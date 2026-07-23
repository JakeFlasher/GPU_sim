"""Deterministic sector stream generator (C-AC-F).

The generator is deliberately pure: no I/O, no hidden state, no
randomness. All inputs and outputs are explicit; every typed
exclusion carries enough context to reproduce the rejection
without re-running the pipeline. Determinism matters because the
generator is the offline-checkable surface that downstream
consumers (the future divergence proxies module, the HRD
profiler) rely on for sector-counting equivalence between sm_86
and sm_120 traces.

Inputs
------
``lane_addresses``
    A length-32 sequence of byte addresses, one per lane in the
    warp. Inactive lanes still occupy a slot; the address value
    for inactive lanes is ignored. The 32-lane requirement is a
    hard contract (NVIDIA SIMT warp width); a different length
    is rejected via ``lane_address_length_invalid`` rather than
    silently truncated.
``active_mask``
    32-bit warp execution mask (bit ``i`` set iff lane ``i`` is
    architecturally active for this instruction).
``predicate_mask``
    32-bit warp predicate mask (bit ``i`` set iff lane ``i``'s
    predicate evaluates true).
``data_width``
    Width in bytes of a single lane's access. Restricted to
    ``{1, 2, 4, 8, 16}`` per the SASS LD/ST byte-size set
    (``.U8``, ``.U16``, ``.32``, ``.64``, ``.128``). A different
    value is rejected via ``data_width_invalid``.
``memory_op_class``
    Uppercase string from ``MemoryOpClass`` (``GLOBAL``,
    ``LOCAL``, ``SHARED``, ...). Only ``GLOBAL`` and ``LOCAL``
    admit; others reject with
    ``sector_stream_class_unsupported``.

Outputs
-------
On success a dict shaped::

    {
        "result_kind": "sector_stream",
        "touched_32B_sectors": int,
        "touched_128B_lines": int,
        "sectors_per_request": float,
        "classification": str,  # see SECTOR_STREAM_CLASSIFICATIONS
        "memory_op_class": str,
        "active_lane_count": int,
    }

On rejection a dict shaped::

    {
        "result_kind": "typed_exclusion",
        "typed_reason": str,  # see SECTOR_STREAM_TYPED_REASONS
        "memory_op_class": str,
        ...  # context fields specific to the typed reason
    }

The two top-level shapes correspond to the two branches in
``workloads/simbench_bw/schemas/sector_stream.schema.json``.
"""

from __future__ import annotations

from typing import Any, Sequence


# Memory-op classes that admit. The set is fixed by the plan's
# C-AC-F scope ("scoped to `global` and `local` memory classes;
# remaining classes typed-excluded with explicit reasons").
SUPPORTED_MEMORY_OP_CLASSES: frozenset[str] = frozenset({
    "GLOBAL",
    "LOCAL",
})

# Memory-op classes that reject with the typed
# `sector_stream_class_unsupported` reason. These are the
# remaining members of `MemoryOpClass` (excluding `UNSPECIFIED`,
# which routes through `unsupported_memory_path` upstream and
# therefore should never reach this generator).
UNSUPPORTED_MEMORY_OP_CLASSES: frozenset[str] = frozenset({
    "SHARED",
    "TEXTURE",
    "SURFACE",
    "ATOMIC",
    "TMA",
    "ASYNC_COPY",
    "CLUSTER_DSMEM",
})

# Per-address memory space (mirror of dynamic_trace.MemorySpace in
# address.proto). When a record carries an explicit space, GLOBAL/GENERIC is L2
# traffic (admit) and SHARED/LOCAL is not (typed-exclude) — this is how a global
# atomic / LDGSTS global source (op class ATOMIC / ASYNC_COPY) enters the sector
# stream while a shared atomic (ATOMS) / shared destination stays out.
_L2_ADDRESS_SPACES: frozenset[int] = frozenset({1, 4})  # GLOBAL, GENERIC
_NON_L2_ADDRESS_SPACES: frozenset[int] = frozenset({2, 3})  # SHARED, LOCAL

# Allowed values for the ``classification`` output field. The
# five-value vocabulary is the plan's literal text.
SECTOR_STREAM_CLASSIFICATIONS: frozenset[str] = frozenset({
    "uniform",
    "broadcast",
    "overlap",
    "strided",
    "scattered",
})

# Typed-exclusion vocabulary. Two of these (``no_lanes_active``,
# ``lane_address_length_invalid``) already appear in the plan's
# AC19 enumeration; the other two
# (``data_width_invalid``, ``sector_stream_class_unsupported``)
# are introduced here as members of the sector-stream taxonomy
# family per C-AC-C.
SECTOR_STREAM_TYPED_REASONS: frozenset[str] = frozenset({
    "no_lanes_active",
    "lane_address_length_invalid",
    "data_width_invalid",
    "sector_stream_class_unsupported",
})

_ALLOWED_DATA_WIDTHS: frozenset[int] = frozenset({1, 2, 4, 8, 16})

_WARP_LANE_COUNT = 32
_SECTOR_BYTES = 32
_LINE_BYTES = 128


def compute_sector_stream(
    *,
    lane_addresses: Sequence[int],
    active_mask: int,
    predicate_mask: int,
    data_width: int,
    memory_op_class: str,
    address_space: int | None = None,
) -> dict[str, Any]:
    """Compute the sector stream summary for one warp memory access.

    All keyword-only by design; positional ordering would invite
    silent argument-swap bugs in callers that mix
    ``active_mask`` and ``predicate_mask``.

    ``address_space`` is the per-address memory space (address-carrying
    contract). When present it is AUTHORITATIVE over ``memory_op_class``: a
    GLOBAL/GENERIC-space record is L2 traffic and admits (including a global
    atomic or the LDGSTS global source, whose ``memory_op_class`` is ATOMIC /
    ASYNC_COPY); a SHARED/LOCAL-space record is not L2 traffic and is typed-
    excluded. Absent (legacy traces) falls back to the per-class check.
    """
    # Effective-space admission first (cheaper than the lane walk).
    if address_space in _L2_ADDRESS_SPACES:
        pass  # global L2 traffic — admit regardless of op class
    elif address_space in _NON_L2_ADDRESS_SPACES:
        return {
            "result_kind": "typed_exclusion",
            "typed_reason": "sector_stream_class_unsupported",
            "memory_op_class": memory_op_class,
        }
    elif memory_op_class in UNSUPPORTED_MEMORY_OP_CLASSES:
        return {
            "result_kind": "typed_exclusion",
            "typed_reason": "sector_stream_class_unsupported",
            "memory_op_class": memory_op_class,
        }
    elif memory_op_class not in SUPPORTED_MEMORY_OP_CLASSES:
        # Unknown or sentinel class. The upstream NVBit producer
        # routes ``UNSPECIFIED`` through the unsupported_memory_path
        # sidecar before it ever reaches us, so the only way a
        # non-supported / non-unsupported value gets here is a
        # caller bug. Return the same typed reason so the bug is
        # visible at the schema level rather than via a Python
        # traceback.
        return {
            "result_kind": "typed_exclusion",
            "typed_reason": "sector_stream_class_unsupported",
            "memory_op_class": memory_op_class,
        }

    if len(lane_addresses) != _WARP_LANE_COUNT:
        return {
            "result_kind": "typed_exclusion",
            "typed_reason": "lane_address_length_invalid",
            "memory_op_class": memory_op_class,
            "observed_length": len(lane_addresses),
            "expected_length": _WARP_LANE_COUNT,
        }

    if data_width not in _ALLOWED_DATA_WIDTHS:
        return {
            "result_kind": "typed_exclusion",
            "typed_reason": "data_width_invalid",
            "memory_op_class": memory_op_class,
            "observed_data_width": data_width,
            "allowed_data_widths": sorted(_ALLOWED_DATA_WIDTHS),
        }

    effective_mask = active_mask & predicate_mask
    if effective_mask == 0:
        return {
            "result_kind": "typed_exclusion",
            "typed_reason": "no_lanes_active",
            "memory_op_class": memory_op_class,
            "active_mask": active_mask,
            "predicate_mask": predicate_mask,
        }

    return _compute_admitted(
        lane_addresses=lane_addresses,
        effective_mask=effective_mask,
        data_width=data_width,
        memory_op_class=memory_op_class,
    )


def _compute_admitted(
    *,
    lane_addresses: Sequence[int],
    effective_mask: int,
    data_width: int,
    memory_op_class: str,
) -> dict[str, Any]:
    """Compute outputs for a record that has cleared the gates."""
    # Collect the per-lane (start, end-exclusive) byte ranges so
    # vector loads (data_width > 32 / 16 / 8) that cross 32B and
    # 128B boundaries contribute additional sectors / lines to
    # the touched-set counts. The end is exclusive so a 16-byte
    # vector load aligned at offset 0 touches bytes [0, 16) and
    # therefore one 32B sector; a 16-byte vector load at offset
    # 24 touches [24, 40), spanning sectors [0, 32) and
    # [32, 64) — two sectors.
    touched_sectors: set[int] = set()
    touched_lines: set[int] = set()
    active_lane_count = 0
    active_addresses: list[int] = []
    for lane_idx in range(_WARP_LANE_COUNT):
        if not (effective_mask >> lane_idx) & 1:
            continue
        active_lane_count += 1
        addr = int(lane_addresses[lane_idx])
        active_addresses.append(addr)
        # Each byte the lane touches contributes to its
        # own (sector, line) bucket. Walk byte-by-byte by
        # ``min(remaining, _SECTOR_BYTES)`` so vector widths
        # that exceed 32B (only 128B is in scope, but the
        # generator should be size-agnostic for future
        # widths) are counted correctly.
        remaining = data_width
        cursor = addr
        while remaining > 0:
            sector_id = cursor // _SECTOR_BYTES
            line_id = cursor // _LINE_BYTES
            touched_sectors.add(sector_id)
            touched_lines.add(line_id)
            # Advance to the next sector boundary or to the
            # end of the lane access, whichever is closer.
            sector_end = (sector_id + 1) * _SECTOR_BYTES
            step = min(remaining, sector_end - cursor)
            cursor += step
            remaining -= step

    classification = _classify(
        active_addresses=active_addresses,
        data_width=data_width,
    )

    # ``sectors_per_request`` is a per-warp memory-request metric:
    # one warp instruction issues one request, and the request
    # touches `len(touched_sectors)` distinct 32B sectors. The
    # original-plan AC9 regression invariant pins
    # uniform/broadcast warps at exactly 1.0 here (one warp →
    # one sector). Normalizing by ``active_lane_count`` would
    # produce 1/32 for a broadcast and break that invariant.
    sectors_per_request = float(len(touched_sectors))

    return {
        "result_kind": "sector_stream",
        "touched_32B_sectors": len(touched_sectors),
        "touched_128B_lines": len(touched_lines),
        "sectors_per_request": sectors_per_request,
        "classification": classification,
        "memory_op_class": memory_op_class,
        "active_lane_count": active_lane_count,
    }


def _classify(
    *,
    active_addresses: list[int],
    data_width: int,
) -> str:
    """Pick one of the five classifications.

    Decision order is deliberate. The earlier predicates are
    more specific than the later ones so a uniform pattern is
    not misclassified as broadcast (uniform implies broadcast,
    but the inverse is not true — we want the strictly stronger
    label when both apply).
    """
    if not active_addresses:
        return "scattered"  # unreachable in the admitted path

    distinct_starts = set(active_addresses)
    if len(distinct_starts) == 1:
        # Every active lane has the same starting address.
        # If the active count is > 1 this is broadcast (one
        # value fanned out); if exactly 1 it is also
        # uniform-by-degenerate-singleton, but we prefer the
        # broadcast label to match the plan's
        # ``broadcast`` test pattern (single replicated
        # address).
        return "broadcast"

    # ``uniform`` per the plan's pattern enumeration is "all
    # lanes touch the same 32B sector" — the addresses can
    # differ within a sector but stay coalesced. This is a
    # weaker condition than literal-broadcast and is the
    # canonical good case for global/local accesses.
    sectors = {addr // _SECTOR_BYTES for addr in active_addresses}
    if len(sectors) == 1:
        return "uniform"

    # ``strided`` requires a constant nonzero stride between
    # consecutive active lanes' starts. Sort by lane address
    # and check pairwise differences.
    sorted_addrs = sorted(active_addresses)
    if len(sorted_addrs) >= 2:
        first_stride = sorted_addrs[1] - sorted_addrs[0]
        if first_stride != 0 and all(
            sorted_addrs[i + 1] - sorted_addrs[i] == first_stride
            for i in range(len(sorted_addrs) - 1)
        ):
            return "strided"

    # ``overlap`` is when distinct lanes touch overlapping byte
    # ranges (their per-lane [addr, addr+data_width) ranges
    # intersect at least one byte). We check this by walking
    # the sorted list and asking whether the next start is
    # strictly inside the prior range. We do NOT also classify
    # full duplicate-address cases here: those land in
    # ``broadcast`` above only if ALL active addresses match;
    # otherwise mixed duplicates fall here as overlap.
    sorted_addrs2 = sorted(active_addresses)
    overlap = False
    for i in range(len(sorted_addrs2) - 1):
        if sorted_addrs2[i + 1] < sorted_addrs2[i] + data_width:
            overlap = True
            break
    if overlap:
        return "overlap"

    # No simpler structure detected.
    return "scattered"


def main(argv: list[str] | None = None) -> int:
    """CLI surface: read one JSON record from stdin and print the
    sector-stream output (admitted or typed-exclusion) on stdout.

    Input shape::

        {
          "lane_addresses": [<32 ints>],
          "active_mask": <int>,
          "predicate_mask": <int>,
          "data_width": <int>,
          "memory_op_class": "<GLOBAL|LOCAL|...>"
        }

    Used for ad-hoc operator verification (e.g., piping a
    captured warp record from an offline trace into the
    generator to confirm the expected typed exclusion fires).
    The CLI is intentionally minimal — the canonical consumer
    is the in-process Python API.
    """
    import argparse
    import json as _json
    import sys

    parser = argparse.ArgumentParser(
        prog="python tools/sector_stream/generator.py",
        description=(
            "Map one warp memory access record (read from stdin "
            "as JSON) to a sector-stream summary or typed "
            "exclusion. The schema lives at "
            "workloads/simbench_bw/schemas/sector_stream.schema.json."
        ),
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        help=(
            "Optional path to a JSON file containing the input "
            "record. If omitted the record is read from stdin."
        ),
    )
    args = parser.parse_args(argv)

    if args.input is not None:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = _json.load(f)
    else:
        payload = _json.load(sys.stdin)

    result = compute_sector_stream(
        lane_addresses=payload["lane_addresses"],
        active_mask=payload["active_mask"],
        predicate_mask=payload["predicate_mask"],
        data_width=payload["data_width"],
        memory_op_class=payload["memory_op_class"],
    )
    sys.stdout.write(_json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
