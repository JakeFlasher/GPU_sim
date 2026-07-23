"""Analytic + fail-closed tests for the L1 -> miss-stream -> L2 filter.

The positive test uses a synthetic mixed-.ca/.cg, multi-SM access set whose L1
pass/miss partition and L1-miss-fed L2 result are computed by hand, so the test
pins the exact normalization arithmetic (not just "it runs"). The negatives pin
the two fail-closed contracts: a full-stream L2 replay is rejected for an L1/L2
headline comparison, and a cell with no SM/CTA partition metadata fails closed
under headline-live operation unless the round-robin fallback is admitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_CACHE_PREDICTOR = _HERE.parent / "cache_predictor"
for _p in (_HERE, _CACHE_PREDICTOR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sector_cache_replay as scr  # noqa: E402
import l1_to_l2_miss_filter as flt  # noqa: E402
from l1_trace_adapter import (  # noqa: E402
    CtaToSmDerivation,
    L1CtaToSmConflictingObservation,
    L1CtaToSmPartialMetadata,
    TraceRecord,
    derive_cta_to_sm_assignment,
)


# .ca loads are L1-eligible; .cg loads bypass L1 (and reach L2 directly).
_CA = True
_CG = False

# A matched cache-op manifest: pc 0x10 is an L1-eligible .ca load, pc 0x20 a
# .cg bypass load (the shape build_per_block_l1_accesses joins against).
_MANIFEST = {
    "schema_version": 1,
    "binary_source": "canned_fixture",
    "arch_token": "sm_120",
    "kernel_mangled_name": "_Z11test_kernelPi",
    "sass_provenance_status": "matched",
    "per_pc_instruction_cache_op": [
        {"pc": "0x10", "instruction_cache_op": "LDG_E_CA", "l1_eligible": True},
        {"pc": "0x20", "instruction_cache_op": "LDG_E_CG", "l1_eligible": False},
    ],
}

# Synthetic launch: two CTAs, mixed .ca/.cg, sub-line sector reuse.
#   block 0: line 10 sector 40 (.ca) x2, line 10 sector 41 (.ca), line 20 (.cg)
#   block 1: line 10 sector 40 (.ca) x2, line 30 (.cg)
# sector_offset = global_sector_id % 4: 40->0, 41->1, 80->0, 120->0.
_ANALYTIC_BLOCKS = [
    [(10, 40, _CA), (10, 40, _CA), (10, 41, _CA), (20, 80, _CG)],
    [(10, 40, _CA), (10, 40, _CA), (30, 120, _CG)],
]

# Large L1 (no capacity eviction at this scale) so L1 hits are pure reuse.
_L1_CAPACITY_BYTES = 4096
# Large L2 so L2 hits are pure (line, sector) reuse (no conflict/capacity eviction).
_L2_CONFIG = scr.CacheReplayConfig(
    capacity_bytes=1 << 20, associativity=16, num_slices=1
)


def _full_stream_offsets(blocks):
    """The raw (line_id, sector_offset) stream with NO L1 filtering -- every
    access in walk order, the input the un-filtered L2 replay would see."""
    line_ids, offsets = [], []
    for accesses in blocks:
        for line_id, sector_id, _elig in accesses:
            line_ids.append(line_id)
            offsets.append(sector_id % 4)
    return line_ids, offsets


def test_analytic_two_sm_l1_partition_and_miss_fed_l2() -> None:
    # Core mechanics test: a KNOWN per-CTA SM map (block 0 -> SM0, block 1 -> SM1)
    # supplied directly to the core. A raw assignment is honestly tagged
    # caller_supplied_partition (NOT real_sm_partition -- the core cannot prove it
    # came from a trace; only the derived path claims real). Provenance does not
    # change the L1/L2 arithmetic, which is what this test pins.
    res = flt.l1_to_l2_filtered_replay(
        _ANALYTIC_BLOCKS,
        sm_count=2,
        l1_capacity_bytes=_L1_CAPACITY_BYTES,
        l2_config=_L2_CONFIG,
        cta_to_sm_assignment=[0, 1],
    )
    # L1 accounting (per-SM): SM0 sees (10/s40 miss, 10/s40 hit, 10/s41 miss);
    # SM1 sees (10/s40 miss, 10/s40 hit). Bypass: line20 + line30.
    assert res.l1_result.hit_count == 2
    assert res.l1_result.miss_count == 3
    assert res.l1_result.bypass_count == 2
    # The L1-miss stream fed to L2 = misses + bypass, in walk order.
    assert res.l2_input_stream == ((10, 0), (10, 1), (20, 0), (10, 0), (30, 0))
    assert res.l2_input_count == 5
    assert res.l2_input_count == res.l1_eligible_misses + res.l1_bypass_accesses
    assert res.l1_hits_absorbed == 2  # absorbed by L1, NOT in the L2 stream
    # L2 over the filtered stream: only the repeated (10, sector 0) hits.
    assert res.l2_result.hits == 1
    assert res.l2_result.total_accesses == 5
    assert res.l2_result.predicted_hit_ratio == pytest.approx(0.2)
    # Provenance: a raw caller-supplied map is NOT claimed as real; L2 input is
    # the L1-miss-filtered stream.
    assert res.partition_provenance == flt.PARTITION_CALLER_SUPPLIED
    assert res.used_real_partition_metadata is False
    assert res.l2_input_provenance == flt.L2_INPUT_L1_FILTERED


def test_full_stream_over_predicts_vs_l1_filtered() -> None:
    # The whole point of the filter: the un-filtered full stream over-predicts
    # the L2 hit ratio because it double-counts intra-line reuse the L1 absorbs.
    res = flt.l1_to_l2_filtered_replay(
        _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
        l2_config=_L2_CONFIG, cta_to_sm_assignment=[0, 1],
    )
    full_lines, full_offsets = _full_stream_offsets(_ANALYTIC_BLOCKS)
    full = scr.replay_access_stream(full_lines, full_offsets, _L2_CONFIG)
    assert full.total_accesses == 7
    assert full.predicted_hit_ratio == pytest.approx(3 / 7)
    assert res.l2_result.predicted_hit_ratio < full.predicted_hit_ratio


def test_per_sm_partition_changes_l1_accounting() -> None:
    # Collapsing both CTAs onto ONE SM warms line 10 across blocks, so block 1's
    # repeats become L1 hits -> a different L1-miss stream than the 2-SM case.
    res = flt.l1_to_l2_filtered_replay(
        _ANALYTIC_BLOCKS, sm_count=1, l1_capacity_bytes=_L1_CAPACITY_BYTES,
        l2_config=_L2_CONFIG, cta_to_sm_assignment=[0, 0],
    )
    assert res.l1_result.hit_count == 3
    assert res.l1_result.miss_count == 2
    assert res.l1_result.bypass_count == 2
    assert res.l2_input_stream == ((10, 0), (10, 1), (20, 0), (30, 0))
    assert res.l2_result.hits == 0
    assert res.l2_result.total_accesses == 4
    assert res.l2_result.predicted_hit_ratio == pytest.approx(0.0)


def test_round_robin_fallback_matches_explicit_when_layout_agrees() -> None:
    # With sm_count=2 the round-robin fallback (block i -> i % 2) yields the same
    # CTA->SM layout as the explicit [0, 1] map, so the L2 result must match --
    # but the provenance must record it was the admitted fallback.
    caller_supplied = flt.l1_to_l2_filtered_replay(
        _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
        l2_config=_L2_CONFIG, cta_to_sm_assignment=[0, 1],
    )
    fallback = flt.l1_to_l2_filtered_replay(
        _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
        l2_config=_L2_CONFIG, cta_to_sm_assignment=None,
        headline_live=True, admit_round_robin_fallback=True,
    )
    assert fallback.l2_input_stream == caller_supplied.l2_input_stream
    assert fallback.partition_provenance == flt.PARTITION_ROUND_ROBIN_FALLBACK
    assert fallback.used_real_partition_metadata is False
    assert caller_supplied.partition_provenance == flt.PARTITION_CALLER_SUPPLIED


def test_missing_partition_metadata_fails_closed_headline_live() -> None:
    with pytest.raises(flt.L1ToL2PartitionMetadataMissing):
        flt.l1_to_l2_filtered_replay(
            _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
            l2_config=_L2_CONFIG, cta_to_sm_assignment=None,
            headline_live=True, admit_round_robin_fallback=False,
        )


def test_offline_fallback_allowed_when_not_headline_live() -> None:
    # Outside headline-live operation the fallback is allowed without an explicit
    # admission (it is still tagged as the fallback).
    res = flt.l1_to_l2_filtered_replay(
        _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
        l2_config=_L2_CONFIG, cta_to_sm_assignment=None, headline_live=False,
    )
    assert res.partition_provenance == flt.PARTITION_ROUND_ROBIN_FALLBACK
    assert res.used_real_partition_metadata is False


def test_full_stream_l2_rejected_for_l1l2_headline() -> None:
    with pytest.raises(flt.FullStreamL2RejectedForL1L2Headline):
        flt.enforce_l1_miss_filtered_for_l1l2_headline(
            ["L1", "L2"], flt.L2_INPUT_FULL_STREAM
        )
    # The L1-miss-filtered stream passes the headline guard.
    flt.enforce_l1_miss_filtered_for_l1l2_headline(
        ["L1", "L2"], flt.L2_INPUT_L1_FILTERED
    )
    # An L2-only scope imposes no L1-filtering requirement.
    flt.enforce_l1_miss_filtered_for_l1l2_headline(
        ["L2"], flt.L2_INPUT_FULL_STREAM
    )


def test_filter_result_passes_headline_guard() -> None:
    res = flt.l1_to_l2_filtered_replay(
        _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
        l2_config=_L2_CONFIG, cta_to_sm_assignment=[0, 1],
    )
    # A filter output is always L1-miss-filtered, so it clears the headline guard.
    flt.enforce_l1_miss_filtered_for_l1l2_headline(
        ["L1", "L2"], res.l2_input_provenance
    )


def _records_with_sm(block0_sm, block1_sm):
    """3-record trace: block0 = .ca line0 x2 (addrs 0,32) + .cg line2 (256);
    block1 = .ca line0 (0). Each record's sm_id is supplied per block (None
    omits it)."""
    return [
        TraceRecord(0, "0x10", (0, 32), sm_id=block0_sm),
        TraceRecord(0, "0x20", (256,), sm_id=block0_sm),
        TraceRecord(1, "0x10", (0,), sm_id=block1_sm),
    ]


def test_from_trace_derives_real_sm_partition_from_sm_id() -> None:
    # The trace-fed entry point DERIVES the per-CTA partition from the records'
    # real sm_id (block0 -> SM0, block1 -> SM1) and only then claims
    # real_sm_partition. No caller-supplied assignment is involved.
    res = flt.filtered_replay_from_trace(
        _records_with_sm(0, 1), _MANIFEST, sm_count=2, grid_dim=2, block_dim=32,
        l1_capacity_bytes=_L1_CAPACITY_BYTES, l2_config=_L2_CONFIG,
        headline_live=True,
    )
    # block0(SM0): line0 miss, line0 hit, line2 bypass; block1(SM1): line0 miss.
    assert res.l1_result.hit_count == 1
    assert res.l1_result.miss_count == 2
    assert res.l1_result.bypass_count == 1
    assert res.l2_input_stream == ((0, 0), (2, 0), (0, 0))
    assert res.l2_input_count == 3
    assert res.partition_provenance == flt.PARTITION_REAL_SM
    assert res.used_real_partition_metadata is True
    assert res.l2_input_provenance == flt.L2_INPUT_L1_FILTERED
    flt.enforce_l1_miss_filtered_for_l1l2_headline(["L1", "L2"], res.l2_input_provenance)


def test_real_sm_partition_not_forgeable_from_public_core() -> None:
    # The PUBLIC raw-assignment path must NOT mint the trusted real_sm_partition
    # label -- only the trace-derived path earns it. Passing it raises.
    with pytest.raises(ValueError):
        flt.l1_to_l2_filtered_replay(
            _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
            l2_config=_L2_CONFIG, cta_to_sm_assignment=[0, 1],
            partition_provenance=flt.PARTITION_REAL_SM,
        )
    # round_robin_fallback is the no-assignment path's tag, also not mintable here.
    with pytest.raises(ValueError):
        flt.l1_to_l2_filtered_replay(
            _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
            l2_config=_L2_CONFIG, cta_to_sm_assignment=[0, 1],
            partition_provenance=flt.PARTITION_ROUND_ROBIN_FALLBACK,
        )
    # Omitted provenance with a raw map -> the honest caller_supplied tag.
    res = flt.l1_to_l2_filtered_replay(
        _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
        l2_config=_L2_CONFIG, cta_to_sm_assignment=[0, 1],
    )
    assert res.partition_provenance == flt.PARTITION_CALLER_SUPPLIED
    assert res.used_real_partition_metadata is False
    # Explicit caller_supplied is the only provenance the raw path accepts.
    res2 = flt.l1_to_l2_filtered_replay(
        _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
        l2_config=_L2_CONFIG, cta_to_sm_assignment=[0, 1],
        partition_provenance=flt.PARTITION_CALLER_SUPPLIED,
    )
    assert res2.partition_provenance == flt.PARTITION_CALLER_SUPPLIED


def test_internal_derivation_helper_requires_successful_derivation() -> None:
    # The internal real-minter rejects a non-successful derivation, so
    # real_sm_partition cannot be produced without proven trace metadata.
    bad = CtaToSmDerivation(assignment=None, metadata_present=False, observed_block_count=0)
    with pytest.raises(ValueError):
        flt._filtered_replay_for_derivation(
            _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
            l2_config=_L2_CONFIG, derivation=bad,
        )
    # A successful derivation does mint it.
    good = CtaToSmDerivation(assignment=(0, 1), metadata_present=True, observed_block_count=2)
    res = flt._filtered_replay_for_derivation(
        _ANALYTIC_BLOCKS, sm_count=2, l1_capacity_bytes=_L1_CAPACITY_BYTES,
        l2_config=_L2_CONFIG, derivation=good,
    )
    assert res.partition_provenance == flt.PARTITION_REAL_SM
    assert res.used_real_partition_metadata is True


def test_from_trace_conflicting_sm_id_rejected() -> None:
    # block0's two records disagree on the SM -> a CTA cannot be on two SMs.
    records = [
        TraceRecord(0, "0x10", (0,), sm_id=0),
        TraceRecord(0, "0x10", (32,), sm_id=1),
        TraceRecord(1, "0x10", (0,), sm_id=1),
    ]
    with pytest.raises(L1CtaToSmConflictingObservation):
        flt.filtered_replay_from_trace(
            records, _MANIFEST, sm_count=2, grid_dim=2, block_dim=32,
            l1_capacity_bytes=_L1_CAPACITY_BYTES, l2_config=_L2_CONFIG,
            headline_live=True,
        )


def test_from_trace_partial_sm_metadata_rejected() -> None:
    # block0 carries sm_id but block1 (which issued a record) does not -> a real
    # partition must be exact over every block that did memory work.
    with pytest.raises(L1CtaToSmPartialMetadata):
        flt.filtered_replay_from_trace(
            _records_with_sm(0, None), _MANIFEST, sm_count=2, grid_dim=2,
            block_dim=32, l1_capacity_bytes=_L1_CAPACITY_BYTES,
            l2_config=_L2_CONFIG, headline_live=True,
        )


def test_from_trace_no_sm_metadata_fails_closed_headline_live() -> None:
    # No record carries sm_id -> no real partition; under headline-live without
    # an admitted fallback this fails closed (does NOT silently round-robin).
    with pytest.raises(flt.L1ToL2PartitionMetadataMissing):
        flt.filtered_replay_from_trace(
            _records_with_sm(None, None), _MANIFEST, sm_count=2, grid_dim=2,
            block_dim=32, l1_capacity_bytes=_L1_CAPACITY_BYTES,
            l2_config=_L2_CONFIG, headline_live=True,
            admit_round_robin_fallback=False,
        )


def test_from_trace_no_sm_metadata_admitted_fallback() -> None:
    # No sm_id + admitted fallback -> round-robin, tagged as the fallback.
    res = flt.filtered_replay_from_trace(
        _records_with_sm(None, None), _MANIFEST, sm_count=2, grid_dim=2,
        block_dim=32, l1_capacity_bytes=_L1_CAPACITY_BYTES, l2_config=_L2_CONFIG,
        headline_live=True, admit_round_robin_fallback=True,
    )
    assert res.partition_provenance == flt.PARTITION_ROUND_ROBIN_FALLBACK
    assert res.used_real_partition_metadata is False


def test_derive_cta_to_sm_assignment_unit() -> None:
    # Direct unit: consistent sm_id per block -> exact assignment; an empty block
    # (no records) is filled with the round-robin placeholder; observed count is
    # the number of blocks that carried a real sm_id.
    records = [
        TraceRecord(0, "0x10", (0,), sm_id=1),
        TraceRecord(0, "0x10", (32,), sm_id=1),
        TraceRecord(2, "0x10", (0,), sm_id=0),
    ]
    d = derive_cta_to_sm_assignment(records, grid_dim=4, sm_count=2)
    assert isinstance(d, CtaToSmDerivation)
    assert d.metadata_present is True
    assert d.observed_block_count == 2
    # blocks 0->1, 2->0 observed; empty blocks 1,3 -> placeholder index % 2.
    assert d.assignment == (1, 1 % 2, 0, 3 % 2)
    # No sm_id anywhere -> no metadata.
    none_d = derive_cta_to_sm_assignment(
        [TraceRecord(0, "0x10", (0,)), TraceRecord(1, "0x10", (0,))],
        grid_dim=2, sm_count=2,
    )
    assert none_d == CtaToSmDerivation(None, False, 0)
    # Out-of-range sm_id is rejected.
    with pytest.raises(L1CtaToSmConflictingObservation):
        derive_cta_to_sm_assignment(
            [TraceRecord(0, "0x10", (0,), sm_id=9)], grid_dim=2, sm_count=2
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
