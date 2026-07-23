"""Paired-control tests for the per-SM L1 fully-associative
LRU walker at ``tools/cache_predictor/l1_walker.py``.

Exercises:
  - block_to_sm_id under each scheduling policy.
  - walk_per_sm_l1 hit / miss / bypass accounting.
  - Per-SM isolation: blocks on different SMs do not evict
    each other's lines.
  - Capacity-induced eviction: a working set larger than
    capacity produces the expected miss count.
  - l1_eligible filtering: bypass accesses pass through.
  - apply_carveout_to_capacity arithmetic.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


import json  # noqa: E402

from l1_walker import (  # noqa: E402
    LINE_SIZE_BYTES,
    SECTOR_SIZE_BYTES,
    SECTORS_PER_LINE,
    L1CtaToSmScheduleAssignmentInvalid,
    L1CtaToSmScheduleMissingOccupancy,
    L1CtaToSmScheduleUnknown,
    L1WalkerAccessInvalid,
    L1WalkerCapacityInvalid,
    L1WalkerDefaultCapacityMissing,
    address_to_line_id,
    apply_carveout_to_capacity,
    block_to_sm_id,
    walk_per_sm_l1,
)


_TABLE_PATH = _HERE / "l1_static_card_table.json"


def _sm_86_table_entry() -> dict:
    """Read the sm_86 row from the published static-card table.

    Tests source L1 capacity expectations from this helper
    instead of hard-coding integers; the plan's discipline
    forbids hard-coded L1 capacity in implementation OR tests.
    """
    raw = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
    for entry in raw["entries"]:
        if entry["arch"] == "sm_86":
            return entry
    raise RuntimeError(
        "sm_86 row missing from l1_static_card_table.json"
    )


class BlockToSmIdTest(unittest.TestCase):
    """The CTA-to-SM mapping respects the scheduling policy."""

    def test_round_robin_modulo_sm_count(self) -> None:
        for i in range(20):
            self.assertEqual(
                block_to_sm_id(
                    i, sm_count=4, schedule_policy="round_robin_homogeneous",
                ),
                i % 4,
            )

    def test_occupancy_derived_uses_supplied_assignment(self) -> None:
        assignment = [3, 0, 1, 2, 3, 0]
        for i, expected in enumerate(assignment):
            self.assertEqual(
                block_to_sm_id(
                    i, sm_count=4,
                    schedule_policy="occupancy_derived",
                    occupancy_derived_assignment=assignment,
                ),
                expected,
            )

    def test_occupancy_derived_without_assignment_raises(self) -> None:
        with self.assertRaises(L1CtaToSmScheduleMissingOccupancy):
            block_to_sm_id(
                0, sm_count=4,
                schedule_policy="occupancy_derived",
            )

    def test_occupancy_derived_out_of_range_index_raises(self) -> None:
        with self.assertRaises(L1CtaToSmScheduleAssignmentInvalid):
            block_to_sm_id(
                5, sm_count=4,
                schedule_policy="occupancy_derived",
                occupancy_derived_assignment=[0, 1, 2, 3],
            )

    def test_occupancy_derived_out_of_range_sm_id_raises(self) -> None:
        with self.assertRaises(L1CtaToSmScheduleAssignmentInvalid):
            block_to_sm_id(
                0, sm_count=4,
                schedule_policy="occupancy_derived",
                occupancy_derived_assignment=[7],
            )

    def test_unknown_policy_raises(self) -> None:
        with self.assertRaises(L1CtaToSmScheduleUnknown):
            block_to_sm_id(
                0, sm_count=4, schedule_policy="unknown",
            )

    def test_zero_sm_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            block_to_sm_id(
                0, sm_count=0,
                schedule_policy="round_robin_homogeneous",
            )

    def test_unrecognized_policy_raises(self) -> None:
        with self.assertRaises(ValueError):
            block_to_sm_id(
                0, sm_count=4, schedule_policy="future_policy",
            )


class AddressToLineIdTest(unittest.TestCase):
    """Byte address -> 128B line id."""

    def test_address_zero_to_line_zero(self) -> None:
        self.assertEqual(address_to_line_id(0), 0)

    def test_address_within_first_line(self) -> None:
        for offset in (0, 31, 64, 127):
            self.assertEqual(address_to_line_id(offset), 0)

    def test_address_at_line_boundary(self) -> None:
        self.assertEqual(address_to_line_id(128), 1)

    def test_line_size_constant(self) -> None:
        self.assertEqual(LINE_SIZE_BYTES, 128)


class WalkPerSmL1Test(unittest.TestCase):
    """walk_per_sm_l1: hit / miss / bypass accounting + per-SM
    isolation."""

    def test_single_block_repeat_access_yields_one_miss_one_hit(
        self,
    ) -> None:
        accesses = [[(line_id := 42, True), (line_id, True)]]
        result = walk_per_sm_l1(
            accesses,
            sm_count=4,
            capacity_bytes=1024,
            schedule_policy="round_robin_homogeneous",
        )
        # First access misses (cold), second hits (line just
        # allocated).
        self.assertEqual(result.miss_count, 1)
        self.assertEqual(result.hit_count, 1)
        self.assertEqual(result.bypass_count, 0)
        self.assertEqual(result.total_eligible_accesses, 2)
        self.assertAlmostEqual(result.hit_rate, 0.5)

    def test_bypass_accesses_excluded_from_numerator_and_denominator(
        self,
    ) -> None:
        accesses = [[(10, True), (10, False), (20, False)]]
        result = walk_per_sm_l1(
            accesses,
            sm_count=4,
            capacity_bytes=1024,
            schedule_policy="round_robin_homogeneous",
        )
        # The 2 bypass accesses (l1_eligible=False) are counted
        # in bypass_count but not in hit/miss/total_eligible.
        self.assertEqual(result.miss_count, 1)
        self.assertEqual(result.hit_count, 0)
        self.assertEqual(result.bypass_count, 2)
        self.assertEqual(result.total_eligible_accesses, 1)

    def test_per_sm_isolation_blocks_on_different_sms_do_not_evict(
        self,
    ) -> None:
        # Block 0 -> sm 0, block 1 -> sm 1 (round_robin). Each
        # block touches a different line; the second access of
        # block 0 should still hit because block 1's line
        # didn't enter sm 0's stack.
        accesses = [
            [(100, True), (100, True)],  # block 0 -> sm 0
            [(200, True)],               # block 1 -> sm 1
        ]
        result = walk_per_sm_l1(
            accesses,
            sm_count=2,
            capacity_bytes=1024,
            schedule_policy="round_robin_homogeneous",
        )
        # block 0: miss(100) + hit(100); block 1: miss(200).
        # Total: 1 hit, 2 miss.
        self.assertEqual(result.hit_count, 1)
        self.assertEqual(result.miss_count, 2)
        self.assertEqual(result.per_sm_hit_count, {0: 1})
        self.assertEqual(result.per_sm_miss_count, {0: 1, 1: 1})

    def test_capacity_induced_eviction(self) -> None:
        # Capacity = 256 bytes => 2 lines per SM. Touch 3 distinct
        # lines, then revisit the first. The first revisit should
        # MISS because it was evicted (only the 2 most-recent lines
        # are in the stack).
        accesses = [[
            (1, True),   # miss line 1
            (2, True),   # miss line 2; stack = [2, 1]
            (3, True),   # miss line 3; stack = [3, 2]; line 1 evicted
            (1, True),   # miss line 1 (evicted); stack = [1, 3]
        ]]
        result = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=2 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
        )
        self.assertEqual(result.miss_count, 4)
        self.assertEqual(result.hit_count, 0)

    def test_capacity_below_line_size_raises(self) -> None:
        with self.assertRaises(L1WalkerCapacityInvalid):
            walk_per_sm_l1(
                [[(1, True)]],
                sm_count=1,
                capacity_bytes=64,  # below LINE_SIZE_BYTES=128
                schedule_policy="round_robin_homogeneous",
            )

    def test_occupancy_derived_policy_routes_blocks_correctly(
        self,
    ) -> None:
        # Block 0 and block 2 both map to SM 0; block 1 maps to SM 1.
        # Block 0 touches line 5; block 2 revisits line 5; on SM 0
        # the second touch should hit (line 5 already resident).
        accesses = [
            [(5, True)],  # block 0 -> sm 0; miss
            [(7, True)],  # block 1 -> sm 1; miss
            [(5, True)],  # block 2 -> sm 0; hit (line 5 still in sm 0's stack)
        ]
        result = walk_per_sm_l1(
            accesses,
            sm_count=2,
            capacity_bytes=1024,
            schedule_policy="occupancy_derived",
            occupancy_derived_assignment=[0, 1, 0],
        )
        self.assertEqual(result.hit_count, 1)
        self.assertEqual(result.miss_count, 2)

    def test_unknown_schedule_policy_fail_closes(self) -> None:
        with self.assertRaises(L1CtaToSmScheduleUnknown):
            walk_per_sm_l1(
                [[(1, True)]],
                sm_count=4,
                capacity_bytes=1024,
                schedule_policy="unknown",
            )

    def test_empty_input_yields_zero_counts(self) -> None:
        result = walk_per_sm_l1(
            [],
            sm_count=4,
            capacity_bytes=1024,
            schedule_policy="round_robin_homogeneous",
        )
        self.assertEqual(result.hit_count, 0)
        self.assertEqual(result.miss_count, 0)
        self.assertEqual(result.bypass_count, 0)
        self.assertEqual(result.total_eligible_accesses, 0)
        self.assertEqual(result.hit_rate, 0.0)

    def test_eight_blocks_round_robin_across_four_sms(self) -> None:
        # The plan's positive control: a kernel with grid=8 blocks
        # and round_robin_homogeneous scheduling distributes blocks
        # evenly across the SM count. Each block touches a unique
        # line; per-SM stacks reflect the partition (2 blocks per
        # SM for 8 blocks / 4 SMs).
        accesses = [
            [(line_id, True)] for line_id in range(8)
        ]
        result = walk_per_sm_l1(
            accesses,
            sm_count=4,
            capacity_bytes=1024,
            schedule_policy="round_robin_homogeneous",
        )
        # 8 unique lines, all cold misses on first touch.
        self.assertEqual(result.miss_count, 8)
        self.assertEqual(result.hit_count, 0)
        # Each SM saw exactly 2 misses (its 2 blocks each cold).
        for sm_id in range(4):
            self.assertEqual(result.per_sm_miss_count[sm_id], 2)


class ApplyCarveoutToCapacityTest(unittest.TestCase):
    """apply_carveout_to_capacity arithmetic.

    All sm_86 capacity expectations are sourced from
    ``l1_static_card_table.json`` via ``_sm_86_table_entry()``
    per the plan's no-hard-coded-capacity discipline."""

    def test_no_shared_returns_unified_budget(self) -> None:
        sm_86 = _sm_86_table_entry()
        cap = apply_carveout_to_capacity(
            unified_budget_bytes=sm_86["l1tex_capacity_bytes"],
            static_shared_bytes=0,
            dynamic_shared_bytes=0,
            carveout_source="device_properties_manifest",
            carveout_max_l1_bytes=sm_86["carveout_max_l1_bytes"],
        )
        self.assertEqual(cap, sm_86["l1tex_capacity_bytes"])

    def test_shared_reduces_capacity(self) -> None:
        # unified_budget - some shared bytes = expected L1 capacity
        sm_86 = _sm_86_table_entry()
        unified = sm_86["l1tex_capacity_bytes"]
        shared = 48 * 1024
        cap = apply_carveout_to_capacity(
            unified_budget_bytes=unified,
            static_shared_bytes=shared,
            dynamic_shared_bytes=0,
            carveout_source="device_properties_manifest",
            carveout_max_l1_bytes=sm_86["carveout_max_l1_bytes"],
        )
        self.assertEqual(cap, unified - shared)

    def test_static_and_dynamic_shared_sum(self) -> None:
        sm_86 = _sm_86_table_entry()
        unified = sm_86["l1tex_capacity_bytes"]
        cap = apply_carveout_to_capacity(
            unified_budget_bytes=unified,
            static_shared_bytes=16 * 1024,
            dynamic_shared_bytes=16 * 1024,
            carveout_source="device_properties_manifest",
            carveout_max_l1_bytes=sm_86["carveout_max_l1_bytes"],
        )
        self.assertEqual(cap, unified - 32 * 1024)

    def test_carveout_max_caps_effective_l1(self) -> None:
        # Even with 0 shared, an explicit non-default carveout
        # cap limits L1.
        sm_86 = _sm_86_table_entry()
        cap = apply_carveout_to_capacity(
            unified_budget_bytes=sm_86["l1tex_capacity_bytes"],
            static_shared_bytes=0,
            dynamic_shared_bytes=0,
            carveout_source="cuda_func_attribute",
            carveout_max_l1_bytes=32 * 1024,
        )
        self.assertEqual(cap, 32 * 1024)

    def test_default_carveout_returns_default_l1_bytes(self) -> None:
        # The plan's reproducer: default carveout on sm_86
        # must yield the table-derived default L1 capacity
        # (NOT the max-carveout cap).
        sm_86 = _sm_86_table_entry()
        cap = apply_carveout_to_capacity(
            unified_budget_bytes=sm_86["l1tex_capacity_bytes"],
            static_shared_bytes=0,
            dynamic_shared_bytes=0,
            carveout_source="default",
            carveout_max_l1_bytes=sm_86["carveout_max_l1_bytes"],
            default_l1_bytes=sm_86["default_l1_bytes"],
        )
        self.assertEqual(cap, sm_86["default_l1_bytes"])
        # And the default L1 is NOT the same as the
        # carveout-max L1 on sm_86 (the table records them as
        # distinct values).
        self.assertNotEqual(
            sm_86["default_l1_bytes"],
            sm_86["carveout_max_l1_bytes"],
        )

    def test_default_carveout_without_default_l1_bytes_raises(
        self,
    ) -> None:
        # default carveout source requires default_l1_bytes
        # provenance; missing -> typed exception.
        sm_86 = _sm_86_table_entry()
        with self.assertRaises(L1WalkerDefaultCapacityMissing):
            apply_carveout_to_capacity(
                unified_budget_bytes=sm_86["l1tex_capacity_bytes"],
                static_shared_bytes=0,
                dynamic_shared_bytes=0,
                carveout_source="default",
                carveout_max_l1_bytes=(
                    sm_86["carveout_max_l1_bytes"]
                ),
                # default_l1_bytes intentionally omitted
            )

    def test_default_carveout_ignores_shared_usage(self) -> None:
        # Per the carveout semantics: in default mode the L1
        # capacity is a fixed bucket property; the kernel's
        # shared request does NOT subtract from L1 (the bucket
        # caps both sides independently).
        sm_86 = _sm_86_table_entry()
        cap = apply_carveout_to_capacity(
            unified_budget_bytes=sm_86["l1tex_capacity_bytes"],
            static_shared_bytes=48 * 1024,
            dynamic_shared_bytes=0,
            carveout_source="default",
            carveout_max_l1_bytes=sm_86["carveout_max_l1_bytes"],
            default_l1_bytes=sm_86["default_l1_bytes"],
        )
        self.assertEqual(cap, sm_86["default_l1_bytes"])

    def test_shared_exceeds_budget_raises(self) -> None:
        sm_86 = _sm_86_table_entry()
        with self.assertRaises(L1WalkerCapacityInvalid):
            apply_carveout_to_capacity(
                unified_budget_bytes=sm_86["l1tex_capacity_bytes"],
                static_shared_bytes=200 * 1024,
                dynamic_shared_bytes=0,
                carveout_source="cuda_func_attribute",
                carveout_max_l1_bytes=(
                    sm_86["carveout_max_l1_bytes"]
                ),
            )

    def test_effective_below_line_size_raises(self) -> None:
        # Synthetic small budget for the line-size boundary; not
        # tied to any arch's real capacity, so a literal is
        # legitimate (this test checks the boundary, not the
        # arch's capacity).
        with self.assertRaises(L1WalkerCapacityInvalid):
            apply_carveout_to_capacity(
                unified_budget_bytes=LINE_SIZE_BYTES,
                static_shared_bytes=LINE_SIZE_BYTES // 2,
                dynamic_shared_bytes=0,
                carveout_source="cuda_func_attribute",
                carveout_max_l1_bytes=LINE_SIZE_BYTES,
            )

    def test_carveout_percent_zero_returns_max_l1(self) -> None:
        """R18 corrected semantics: per CUDA's
        `cudaFuncAttributePreferredSharedMemoryCarveout`, value=0
        means NO shared-memory carveout, i.e. prefer MAX L1.
        The walker's effective L1 capacity must equal
        `carveout_max_l1_bytes` at carveout_percent=0."""
        sm_86 = _sm_86_table_entry()
        cap = apply_carveout_to_capacity(
            unified_budget_bytes=sm_86["l1tex_capacity_bytes"],
            static_shared_bytes=0,
            dynamic_shared_bytes=0,
            carveout_source="cuda_func_attribute_carveout_percent",
            carveout_max_l1_bytes=sm_86["carveout_max_l1_bytes"],
            default_l1_bytes=sm_86["default_l1_bytes"],
            carveout_percent=0,
        )
        self.assertEqual(cap, sm_86["carveout_max_l1_bytes"])

    def test_carveout_percent_hundred_returns_default_l1(self) -> None:
        """R18 corrected semantics: per CUDA's
        `cudaFuncAttributePreferredSharedMemoryCarveout`, value=100
        means FULL shared-memory carveout, i.e. prefer MIN L1.
        The walker's effective L1 capacity must equal
        `default_l1_bytes` (the minimum bucket) at
        carveout_percent=100."""
        sm_86 = _sm_86_table_entry()
        cap = apply_carveout_to_capacity(
            unified_budget_bytes=sm_86["l1tex_capacity_bytes"],
            static_shared_bytes=0,
            dynamic_shared_bytes=0,
            carveout_source="cuda_func_attribute_carveout_percent",
            carveout_max_l1_bytes=sm_86["carveout_max_l1_bytes"],
            default_l1_bytes=sm_86["default_l1_bytes"],
            carveout_percent=100,
        )
        self.assertEqual(cap, sm_86["default_l1_bytes"])

    def test_carveout_percent_fifty_returns_midpoint(self) -> None:
        """R18: a carveout_percent of 50 returns the midpoint of
        the [default, max] L1 capacity range; the linear
        interpolation is symmetric so this test pins the
        midpoint regardless of inversion direction. Mirrors the
        carveout_sensitivity_probe kernel's default
        carveout_percent=50 (from its main.cu)."""
        sm_86 = _sm_86_table_entry()
        cap = apply_carveout_to_capacity(
            unified_budget_bytes=sm_86["l1tex_capacity_bytes"],
            static_shared_bytes=0,
            dynamic_shared_bytes=0,
            carveout_source="cuda_func_attribute_carveout_percent",
            carveout_max_l1_bytes=sm_86["carveout_max_l1_bytes"],
            default_l1_bytes=sm_86["default_l1_bytes"],
            carveout_percent=50,
        )
        expected = (
            sm_86["carveout_max_l1_bytes"]
            + sm_86["default_l1_bytes"]
        ) // 2
        self.assertEqual(cap, expected)


class SectorWithLineAllocationTest(unittest.TestCase):
    """The walker accepts both legacy 2-tuple (line_id,
    l1_eligible) inputs and new 3-tuple (line_id, sector_id,
    l1_eligible) inputs. With 3-tuples the hit decision is
    sector-with-line-allocation: a hit requires both parent
    line in LRU AND touched sector in that line's residency
    set."""

    def test_two_sectors_same_line_first_pass_two_misses(self) -> None:
        """Both sectors in the same line on the first iteration
        are misses (line is allocated on the first sector access;
        the second sector access finds the line resident but the
        sector NOT resident, so it also misses but does NOT
        re-allocate)."""
        # line_id=0 covers global sector_ids [0, 3]; the test
        # accesses sectors 0 and 1.
        accesses = [[
            (0, 0, True),  # sector 0 of line 0
            (0, 1, True),  # sector 1 of line 0
        ]]
        result = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=2 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
        )
        self.assertEqual(result.miss_count, 2)
        self.assertEqual(result.hit_count, 0)
        self.assertEqual(result.total_eligible_accesses, 2)

    def test_two_sectors_same_line_second_pass_two_hits(self) -> None:
        """Revisiting the same sectors after the first pass
        produces two hits (both line + sector are resident)."""
        accesses = [[
            (0, 0, True), (0, 1, True),
            (0, 0, True), (0, 1, True),
        ]]
        result = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=2 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
        )
        self.assertEqual(result.miss_count, 2)
        self.assertEqual(result.hit_count, 2)
        self.assertEqual(result.total_eligible_accesses, 4)

    def test_eviction_clears_sector_residency(self) -> None:
        """When a line is evicted (LRU past capacity), its
        sector-residency state is discarded. Re-accessing the
        evicted line's sectors produces a miss."""
        # Capacity = 2 lines. Access lines 1, 2, 3, then back
        # to line 1. Line 1's sectors should be gone.
        accesses = [[
            (1, 4, True),    # sector 4 of line 1; miss (cold)
            (2, 8, True),    # sector 8 of line 2; miss (cold)
            (3, 12, True),   # sector 12 of line 3; miss; evicts line 1
            (1, 4, True),    # sector 4 of line 1; miss (line 1 evicted)
        ]]
        result = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=2 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
        )
        self.assertEqual(result.miss_count, 4)
        self.assertEqual(result.hit_count, 0)

    def test_legacy_2tuple_input_back_compat(self) -> None:
        """Legacy ``(line_id, l1_eligible)`` 2-tuple inputs still
        work: the walker treats them as a single-sector access
        per line so the hit decision degrades to line-level
        LRU."""
        accesses = [[
            (1, True),   # line 1: cold miss
            (1, True),   # line 1: hit (line + sector both resident)
        ]]
        result = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=2 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
        )
        self.assertEqual(result.miss_count, 1)
        self.assertEqual(result.hit_count, 1)

    def test_invalid_access_tuple_arity_raises(self) -> None:
        """An access tuple with 4+ elements raises the typed
        L1WalkerAccessInvalid exception so the L1 head can
        fail-close with a documented diagnostic."""
        with self.assertRaises(L1WalkerAccessInvalid):
            walk_per_sm_l1(
                [[(1, 2, 3, True)]],  # 4-tuple, unsupported
                sm_count=1,
                capacity_bytes=2 * LINE_SIZE_BYTES,
                schedule_policy="round_robin_homogeneous",
            )

    def test_distinct_sectors_in_same_line_share_line_lru(self) -> None:
        """Accessing different sectors of the same line keeps
        the parent line at the top of the LRU; the second
        sector access does NOT cause a line eviction."""
        # Capacity = 1 line. Access sector 0 of line 0, then
        # sector 1 of line 0. Both should be in the same line;
        # the second access should NOT evict line 0.
        accesses = [[
            (0, 0, True),   # cold miss, line 0 in LRU, sector 0 resident
            (0, 1, True),   # miss (sector 1 not resident), but line 0 stays
            (0, 0, True),   # hit (line 0 + sector 0 still resident)
            (0, 1, True),   # hit (line 0 + sector 1 still resident)
        ]]
        result = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=LINE_SIZE_BYTES,  # only 1 line fits
            schedule_policy="round_robin_homogeneous",
        )
        self.assertEqual(result.miss_count, 2)
        self.assertEqual(result.hit_count, 2)


class WalkerWarmupLaunchesTest(unittest.TestCase):
    """Paired-control regression for the ``warmup_launches``
    parameter introduced to model NCU's ``--launch-skip``
    profiling methodology.

    NCU counts hit/miss only on the measured launch (the launch
    after N warmup launches have already gone through the
    kernel). For a kernel whose working set fits in L1, the
    warmup launches load the lines and the measured launch sees
    all hits. The walker matches this by replaying the per-block
    traversal N+1 times with persistent per-SM LRU + sector
    residency state and counting only the final replay.
    """

    def test_warmup_launches_default_zero_matches_legacy_cold_cache(
        self,
    ) -> None:
        """Default ``warmup_launches=0`` must match the legacy
        cold-cache single-launch behavior so the parameter is
        backward compatible."""
        accesses = [[
            (0, 0, True),
            (1, 4, True),
            (0, 0, True),
        ]]
        legacy_result = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=4 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
        )
        explicit_zero_result = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=4 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
            warmup_launches=0,
        )
        self.assertEqual(
            legacy_result.hit_count, explicit_zero_result.hit_count,
        )
        self.assertEqual(
            legacy_result.miss_count, explicit_zero_result.miss_count,
        )

    def test_warmup_launches_one_warms_cache_for_repeat_pattern(
        self,
    ) -> None:
        """With ``warmup_launches=1``, the first launch loads
        the lines (its counters are discarded) and the second
        (measured) launch sees ALL hits because the working set
        fits in L1."""
        # Per-block trace touches 2 distinct lines; capacity
        # holds 4 lines so the entire working set fits.
        accesses = [[
            (0, 0, True),
            (1, 4, True),
        ]]
        cold_result = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=4 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
            warmup_launches=0,
        )
        warm_result = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=4 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
            warmup_launches=1,
        )
        # Cold-cache: both accesses miss.
        self.assertEqual(cold_result.miss_count, 2)
        self.assertEqual(cold_result.hit_count, 0)
        # Warm-cache (after 1 warmup launch): both accesses hit.
        self.assertEqual(warm_result.miss_count, 0)
        self.assertEqual(warm_result.hit_count, 2)

    def test_warmup_launches_five_matches_one_for_capacity_fitting(
        self,
    ) -> None:
        """For a working-set-fits-in-capacity kernel, more
        warmup launches do not change the steady-state hit
        rate. The walker's behavior at N=1 and N=5 must
        match when the working set fits."""
        accesses = [[
            (0, 0, True),
            (1, 4, True),
            (2, 8, True),
            (3, 12, True),
        ]]
        result_n1 = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=8 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
            warmup_launches=1,
        )
        result_n5 = walk_per_sm_l1(
            accesses,
            sm_count=1,
            capacity_bytes=8 * LINE_SIZE_BYTES,
            schedule_policy="round_robin_homogeneous",
            warmup_launches=5,
        )
        self.assertEqual(result_n1.hit_count, result_n5.hit_count)
        self.assertEqual(result_n1.miss_count, result_n5.miss_count)
        # Sanity: working set fits, so all accesses hit on the
        # measured launch.
        self.assertEqual(result_n1.miss_count, 0)
        self.assertEqual(result_n1.hit_count, 4)

    def test_warmup_launches_negative_raises(self) -> None:
        """Negative warmup_launches is not meaningful and must
        fail-closed with ValueError."""
        accesses = [[(0, 0, True)]]
        with self.assertRaises(ValueError):
            walk_per_sm_l1(
                accesses,
                sm_count=1,
                capacity_bytes=LINE_SIZE_BYTES,
                schedule_policy="round_robin_homogeneous",
                warmup_launches=-1,
            )


if __name__ == "__main__":
    unittest.main()
