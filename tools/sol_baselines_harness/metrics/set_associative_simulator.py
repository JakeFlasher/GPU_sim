"""Set-associative sectored L2 cache simulator.

Consumes a cache-geometry spec from
`tools/sol_baselines_harness/metrics/cache_geometry.py` and models
the L2 as per-slice per-set per-way slot arrays where each slot
holds one 128 B line with four 32 B sectors and per-sector valid
bits. Replacement applies the Zhang USENIX'24 two-phase rule:
linear insertion while the set is still populating, LRU eviction
once every way is occupied.

Routing applies the line-aligned hash
`(sector_id >> 7) % (num_sets_per_slice * num_slices)` and maps
the result to `(slice_idx, set_idx)`. The `>> 7` shift groups
every sector within a 4 KiB page to the same global slot — a
coarse spatial routing choice. Whether the production GPU's
set-hash matches this is a calibration question the downstream
NCU label sweep settles; the simulator honors the chosen formula
so its behavior is reproducible.

Modulo is used unconditionally even though a `& (n - 1)` bitmask
would suffice when `num_sets_per_slice * num_slices` is a power
of two. Common sm_86 L2 geometry (6 slices x 512 sets per slice
= 3072 global slots) is not a power of two, so the bitmask form
would alias; modulo is well-defined for all inputs and equivalent
to the bitmask when `n` is a power of two.

The simulator does not yet model L1/L2 policy separation
(`.cg` bypass, no-write-allocate store filtering, atomic policy
routing, local-memory exclusion); those live in the separate
`cache_access_policy` module that callers consult before invoking
this simulator. For the same reason this simulator returns only
`"hit"` or `"miss"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_LINE_ALIGNED_SECTOR_HASH_SHIFT: int = 7

# Set-index hash policy vocabulary. The simulator's geometry
# spec carries an optional ``set_index_policy`` field that
# selects between two routings:
#
# * ``page_aligned_shift7`` (default): the legacy L2-oriented
#   hash that collapses 128 sectors into a single global slot
#   via ``sector_id >> 7``. Preserved for byte-for-byte
#   compatibility with existing L2 tests + downstream consumers
#   that audit produced binary hashes against pre-shift-7
#   captures.
# * ``line_indexed``: the canonical line-indexed mapping used
#   for the per-SM L1 simulator instance. ``set_index =
#   (sector_id // sectors_per_line) % num_sets``. Adjacent
#   lines land in distinct sets, which is the textbook
#   generic set-associative cache routing (matches GPGPU-Sim's
#   L1 cache model and the simulator's existing per-sector
#   line tracking convention).
#
# Geometry specs that omit the field default to
# ``page_aligned_shift7``. Specs with an unknown policy raise
# a ``ValueError`` from the constructor; the caller is
# responsible for catching that and routing the launch to a
# ``cache_geometry_unresolved`` typed-unavailable bundle.
SET_INDEX_POLICY_PAGE_ALIGNED_SHIFT7: str = "page_aligned_shift7"
SET_INDEX_POLICY_LINE_INDEXED: str = "line_indexed"
_SET_INDEX_POLICY_VOCABULARY: frozenset[str] = frozenset({
    SET_INDEX_POLICY_PAGE_ALIGNED_SHIFT7,
    SET_INDEX_POLICY_LINE_INDEXED,
})


@dataclass
class LineSlot:
    """One way of one set. `is_valid=False` means the slot has
    never been populated (still in the linear-insertion phase
    when the containing set has at least one such slot). A line
    in cache tracks which of its four 32 B sectors have been
    brought in via the per-sector valid bits.
    """

    is_valid: bool = False
    line_id: int = 0
    sectors_valid_mask: int = 0
    lru_timestamp: int = 0


class SetAssociativeSimulator:
    """Per-slice per-set per-way sectored L2 model with Zhang
    two-phase replacement.

    Constructed from a `CacheGeometry` spec dict (the output of
    `resolve_cache_geometry`). Each access is a sector id; the
    simulator returns `"hit"` or `"miss"` based on the per-sector
    valid bits inside each cached line.
    """

    HIT: str = "hit"
    MISS: str = "miss"

    def __init__(self, geometry: dict[str, Any]) -> None:
        l2_bytes = geometry.get("l2_capacity_bytes")
        line_size = geometry.get("line_size_bytes")
        associativity = geometry.get("associativity_l2")
        num_slices = geometry.get("num_l2_slices")
        sector_size = geometry.get("sector_size_bytes")
        self._validate_geometry(
            l2_bytes=l2_bytes,
            line_size=line_size,
            associativity=associativity,
            num_slices=num_slices,
            sector_size=sector_size,
        )
        total_bytes_per_slice = l2_bytes // num_slices
        bytes_per_set = line_size * associativity
        if total_bytes_per_slice % bytes_per_set != 0:
            raise ValueError(
                f"l2_capacity_bytes={l2_bytes} / num_l2_slices="
                f"{num_slices} = {total_bytes_per_slice} is not "
                f"divisible by line_size_bytes * associativity_l2 = "
                f"{line_size} * {associativity} = {bytes_per_set}; "
                f"cannot derive an integer num_sets_per_slice"
            )
        num_sets_per_slice = total_bytes_per_slice // bytes_per_set
        sectors_per_line = line_size // sector_size
        if line_size % sector_size != 0:
            raise ValueError(
                f"line_size_bytes={line_size} is not a multiple "
                f"of sector_size_bytes={sector_size}"
            )
        set_index_policy = geometry.get(
            "set_index_policy",
            SET_INDEX_POLICY_PAGE_ALIGNED_SHIFT7,
        )
        if set_index_policy not in _SET_INDEX_POLICY_VOCABULARY:
            raise ValueError(
                f"set_index_policy={set_index_policy!r} is not in "
                f"the closed vocabulary "
                f"{sorted(_SET_INDEX_POLICY_VOCABULARY)!r}; the "
                f"caller must route this to "
                f"cache_geometry_unresolved with the unknown "
                f"policy in the provenance"
            )
        self.geometry = geometry
        self.l2_capacity_bytes = int(l2_bytes)
        self.line_size_bytes = int(line_size)
        self.sector_size_bytes = int(sector_size)
        self.associativity_l2 = int(associativity)
        self.num_l2_slices = int(num_slices)
        self.num_sets_per_slice = int(num_sets_per_slice)
        self.sectors_per_line = int(sectors_per_line)
        self.set_index_policy = set_index_policy
        # Sector count per line determines how many bits the
        # sectors_valid_mask uses; sm_86 (32 B sector + 128 B
        # line) = 4 sectors per line.
        self._sectors_valid_full_mask = (1 << sectors_per_line) - 1
        self._access_counter = 0
        # slots[slice_idx][set_idx][way_idx]
        self.slots: list[list[list[LineSlot]]] = [
            [
                [LineSlot() for _ in range(self.associativity_l2)]
                for _ in range(self.num_sets_per_slice)
            ]
            for _ in range(self.num_l2_slices)
        ]

    @staticmethod
    def _validate_geometry(
        *,
        l2_bytes: Any,
        line_size: Any,
        associativity: Any,
        num_slices: Any,
        sector_size: Any,
    ) -> None:
        for name, value in (
            ("l2_capacity_bytes", l2_bytes),
            ("line_size_bytes", line_size),
            ("associativity_l2", associativity),
            ("num_l2_slices", num_slices),
            ("sector_size_bytes", sector_size),
        ):
            if value is None:
                raise ValueError(
                    f"geometry missing required field {name!r}"
                )
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"geometry field {name!r} must be a positive "
                    f"int; got {value!r}"
                )

    def _route(self, sector_id: int) -> tuple[int, int, int]:
        """Resolve a sector_id to `(slice_idx, set_idx, line_id)`.

        Two routing policies are supported via the geometry
        spec's ``set_index_policy`` field:

        * ``page_aligned_shift7`` (default): the legacy
          L2-oriented hash that collapses a 4 KiB sector range
          (``sector_id >> 7``) into a single global slot.
        * ``line_indexed``: per-SM L1 routing where each
          successive line lands in a distinct global slot.
          ``global_slot = (sector_id // sectors_per_line) %
          (num_sets_per_slice * num_slices)``. Adjacent lines
          therefore distribute across sets and the L1
          simulator's per-SM working-set boundary takes
          effect.

        Both policies derive ``slice_idx`` / ``set_idx`` from
        ``global_slot`` using the same slice-first
        partitioning to keep the slice / set distribution
        deterministic across policies.

        The ``line_id`` is the byte-line-aligned id used as
        the tag inside the chosen way; for a 128 B line with
        32 B sectors this collapses to ``sector_id >> 2``
        regardless of the routing policy.
        """
        if sector_id < 0:
            raise ValueError(
                f"sector_id must be non-negative; got {sector_id}"
            )
        if self.set_index_policy == SET_INDEX_POLICY_LINE_INDEXED:
            global_slot = (
                (sector_id // self.sectors_per_line)
                % (self.num_sets_per_slice * self.num_l2_slices)
            )
        else:
            global_slot = (
                (sector_id >> _LINE_ALIGNED_SECTOR_HASH_SHIFT)
                % (self.num_sets_per_slice * self.num_l2_slices)
            )
        slice_idx = global_slot % self.num_l2_slices
        set_idx = global_slot // self.num_l2_slices
        line_id = sector_id // self.sectors_per_line
        return slice_idx, set_idx, line_id

    def access(self, sector_id: int) -> str:
        """One sector access. Returns `"hit"` if the line is in
        cache AND the sector's valid bit is set; returns
        `"miss"` otherwise. The access updates LRU on every
        call (hit or miss) and brings the missed sector in.
        """
        self._access_counter += 1
        slice_idx, set_idx, line_id = self._route(sector_id)
        ways = self.slots[slice_idx][set_idx]
        sector_offset_in_line = sector_id % self.sectors_per_line
        sector_bit = 1 << sector_offset_in_line

        for slot in ways:
            if slot.is_valid and slot.line_id == line_id:
                # Line is in cache. Check sector validity.
                slot.lru_timestamp = self._access_counter
                if slot.sectors_valid_mask & sector_bit:
                    return self.HIT
                # Line present but this sector wasn't filled
                # before. Bring it in.
                slot.sectors_valid_mask |= sector_bit
                return self.MISS

        # Line not present. Allocate using Zhang two-phase:
        # linear insertion during set population, LRU once
        # filled.
        target_slot = self._select_eviction_target(ways)
        target_slot.is_valid = True
        target_slot.line_id = line_id
        target_slot.sectors_valid_mask = sector_bit
        target_slot.lru_timestamp = self._access_counter
        return self.MISS

    def _select_eviction_target(
        self, ways: list[LineSlot]
    ) -> LineSlot:
        """Zhang two-phase target selection.

        Phase 1 (set still populating): return the first way
        with `is_valid == False`. This is linear insertion in
        way-index order — the population pattern Zhang USENIX'24
        empirically observed on Ampere/Ada L2.

        Phase 2 (every way valid): return the way with the
        smallest `lru_timestamp` (true LRU). The replacement
        target is the LRU way, NOT a round-robin slot.
        """
        for slot in ways:
            if not slot.is_valid:
                return slot
        oldest = ways[0]
        for slot in ways[1:]:
            if slot.lru_timestamp < oldest.lru_timestamp:
                oldest = slot
        return oldest

    def reset(self) -> None:
        """Clear every slot and reset the access counter. Used
        between launches when the cold/warm launch protocol
        chooses a cold-cache initial state.
        """
        self._access_counter = 0
        for slice_arr in self.slots:
            for set_arr in slice_arr:
                for slot in set_arr:
                    slot.is_valid = False
                    slot.line_id = 0
                    slot.sectors_valid_mask = 0
                    slot.lru_timestamp = 0


__all__ = [
    "LineSlot",
    "SET_INDEX_POLICY_LINE_INDEXED",
    "SET_INDEX_POLICY_PAGE_ALIGNED_SHIFT7",
    "SetAssociativeSimulator",
]
