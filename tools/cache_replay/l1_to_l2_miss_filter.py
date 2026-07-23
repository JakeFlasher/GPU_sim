"""L1 -> miss-stream -> L2 trace-normalization filter (cache accounting).

NCU's L2 sector hit rate observes ONLY the L1-miss stream: an access that hits
the per-SM L1 is served there and never reaches L2. Feeding the FULL decoded
access stream to the L2 replay therefore double-counts the intra-line reuse the
per-SM L1 absorbs and systematically over-predicts the L2 hit ratio (the live
falsification recorded in ``data/local_labels/cache_replay_falsification.json``
measured a mean signed over-prediction of ~0.34 from exactly this trace-fidelity
defect). The headline L1/L2 comparison must instead be fed the L1-miss stream.

This module is the normalization stage that closes the gap documented in
``tools/cache_replay/l1_diagnostic.py`` (which intentionally replays the whole
cell stream through one diagnostic L1 with no per-SM partition and no .ca/.cg
handling). It:

  1. partitions accesses per-SM / per-CTA using the real ``sm_id`` /
     ``warpid_sm`` CTA->SM map when the trace carries it, and refuses
     headline-live operation when no partition metadata exists UNLESS the caller
     explicitly admits a round-robin fallback (the output is then tagged with the
     fallback provenance so a consumer can never mistake it for real metadata);
  2. applies .ca / .cg L1 eligibility (reusing the existing L1 cache-op
     classifier through the trace adapter; bypass loads skip L1 and reach L2
     directly);
  3. runs the existing per-SM L1 walker
     (``tools/cache_predictor/l1_walker.l1_miss_stream_per_sm``) and emits ONLY
     the L1-miss stream (L1-eligible misses + L1-bypass accesses) into the L2
     sector replay (``sector_cache_replay.replay_access_stream``);
  4. provides a guard so any L1/L2 headline comparison REJECTS a full-stream
     (un-filtered) L2 replay.

It is trace-accounting / input-normalization over already-decoded NVBit traces:
it models cache residency to decide which accesses reach L2, NOT kernel
scheduling or wall-clock timing. It is explicitly NOT a GPU simulator -- it reads
only already-collected traces and never invokes any privileged measurement,
driver-root mutation, or privileged profiling surface.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_HERE = Path(__file__).resolve().parent
_CACHE_PREDICTOR = _HERE.parent / "cache_predictor"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_CACHE_PREDICTOR) not in sys.path:
    sys.path.insert(0, str(_CACHE_PREDICTOR))

import sector_cache_replay as scr  # noqa: E402
from l1_walker import (  # noqa: E402
    PerSmL1Result,
    l1_miss_stream_per_sm,
)


# --------------------------------------------------------------------------
# Provenance vocabulary (closed). Every filter output is tagged with which
# CTA->SM partition source it used and that its L2 input is the L1-miss stream.
# --------------------------------------------------------------------------
# real_sm_partition is RESERVED for an assignment DERIVED from the trace's real
# sm_id metadata (only filtered_replay_from_trace -> derive_cta_to_sm_assignment
# sets it). A raw caller-supplied assignment to the core is tagged
# caller_supplied_partition -- honest: the core cannot prove it came from a trace.
PARTITION_REAL_SM = "real_sm_partition"
PARTITION_CALLER_SUPPLIED = "caller_supplied_partition"
PARTITION_ROUND_ROBIN_FALLBACK = "round_robin_fallback"
PARTITION_PROVENANCE_VALUES = frozenset(
    {PARTITION_REAL_SM, PARTITION_CALLER_SUPPLIED, PARTITION_ROUND_ROBIN_FALLBACK}
)

L2_INPUT_L1_FILTERED = "l1_miss_filtered"
L2_INPUT_FULL_STREAM = "full_stream"


class L1ToL2PartitionMetadataMissing(Exception):
    """Raised when a headline-live L1/L2 filter is requested but the trace
    carries no per-CTA ``sm_id`` / ``warpid_sm`` partition metadata and the
    caller did not explicitly admit the round-robin fallback. Failing closed
    here prevents a silent fallback from masquerading as a real per-SM
    partition in a headline comparison."""

    TOKEN = "l1_to_l2_partition_metadata_missing"


class FullStreamL2RejectedForL1L2Headline(Exception):
    """Raised when an L1/L2 headline comparison is handed a full-stream
    (un-filtered) L2 replay instead of the L1-miss-filtered stream. The
    full-stream L2 systematically over-predicts the L2 sector hit rate, so it is
    refused for the headline L1/L2 path."""

    TOKEN = "l1l2_headline_requires_l1_miss_filtered_stream"


@dataclass(frozen=True)
class L1ToL2FilterResult:
    """Result of the L1 -> miss-stream -> L2 normalization for one launch."""

    l2_input_stream: tuple[tuple[int, int], ...]
    l1_result: PerSmL1Result
    l2_result: scr.ReplayResult
    partition_provenance: str
    l2_input_provenance: str
    used_real_partition_metadata: bool
    l1_hits_absorbed: int
    l1_eligible_misses: int
    l1_bypass_accesses: int
    l2_input_count: int

    @property
    def l2_predicted_hit_ratio(self) -> float:
        return self.l2_result.predicted_hit_ratio

    def to_provenance(self) -> dict:
        return {
            "partition_provenance": self.partition_provenance,
            "used_real_partition_metadata": self.used_real_partition_metadata,
            "l2_input_provenance": self.l2_input_provenance,
            "l1_hits_absorbed": self.l1_hits_absorbed,
            "l1_eligible_misses": self.l1_eligible_misses,
            "l1_bypass_accesses": self.l1_bypass_accesses,
            "l2_input_count": self.l2_input_count,
            "l2_predicted_hit_ratio": self.l2_result.predicted_hit_ratio,
            "l2_total_accesses": self.l2_result.total_accesses,
        }


def l1_to_l2_filtered_replay(
    per_block_accesses: Sequence[Sequence[tuple]],
    *,
    sm_count: int,
    l1_capacity_bytes: int,
    l2_config: scr.CacheReplayConfig,
    cta_to_sm_assignment: Sequence[int] | None = None,
    partition_provenance: str | None = None,
    headline_live: bool = False,
    admit_round_robin_fallback: bool = False,
    warmup_launches: int = 0,
) -> L1ToL2FilterResult:
    """Partition per-SM, run the per-SM L1, and replay ONLY the L1-miss stream
    through the L2 model.

    ``per_block_accesses`` is the per-CTA walker access sequence (the shape the
    trace adapter ``build_per_block_l1_accesses`` produces); each access is a
    ``(line_id, l1_eligible)`` or ``(line_id, sector_id, l1_eligible)`` tuple.

    Partition source:
      - ``cta_to_sm_assignment`` not None: a per-CTA SM map. This PUBLIC path
        NEVER mints the trusted ``real_sm_partition`` label -- it cannot prove a
        raw assignment came from real trace metadata. ``partition_provenance``
        may only be omitted (-> ``caller_supplied_partition``,
        ``used_real_partition_metadata=False``) or explicitly
        ``caller_supplied_partition``; passing ``real_sm_partition`` or
        ``round_robin_fallback`` here RAISES. ``real_sm_partition`` is earned
        ONLY via ``filtered_replay_from_trace`` ->
        ``derive_cta_to_sm_assignment`` -> the internal
        ``_filtered_replay_for_derivation`` (which requires a successful
        derivation), so the trusted label is not forgeable through this API.
      - ``cta_to_sm_assignment`` None: no partition metadata. The only way to
        partition is the round-robin fallback. Under ``headline_live`` this is
        refused (``L1ToL2PartitionMetadataMissing``) UNLESS
        ``admit_round_robin_fallback`` is set; when admitted (or when not
        headline-live), provenance = ``round_robin_fallback``.

    Returns an :class:`L1ToL2FilterResult` whose ``l2_input_provenance`` is
    always ``l1_miss_filtered`` -- the L2 replay only ever sees the L1-miss
    stream.
    """
    if cta_to_sm_assignment is not None:
        provenance = partition_provenance or PARTITION_CALLER_SUPPLIED
        if provenance not in PARTITION_PROVENANCE_VALUES:
            raise ValueError(
                f"partition_provenance {provenance!r} not in "
                f"{sorted(PARTITION_PROVENANCE_VALUES)}"
            )
        if provenance != PARTITION_CALLER_SUPPLIED:
            # real_sm_partition and round_robin_fallback are NOT mintable on the
            # public raw-assignment path. real_sm_partition is reserved for the
            # trace-derived path (proves consistent per-CTA sm_id);
            # round_robin_fallback is the no-assignment path's tag.
            raise ValueError(
                f"partition_provenance={provenance!r} cannot be claimed on the "
                f"public raw-assignment path; only {PARTITION_CALLER_SUPPLIED!r} "
                f"is allowed here. real_sm_partition is earned only via "
                f"filtered_replay_from_trace + derive_cta_to_sm_assignment."
            )
        return _run_filter(
            per_block_accesses,
            sm_count=sm_count,
            l1_capacity_bytes=l1_capacity_bytes,
            l2_config=l2_config,
            schedule_policy="occupancy_derived",
            assignment=cta_to_sm_assignment,
            partition_provenance=PARTITION_CALLER_SUPPLIED,
            used_real=False,
            warmup_launches=warmup_launches,
        )
    if partition_provenance is not None:
        raise ValueError(
            "partition_provenance must not be set when cta_to_sm_assignment "
            "is None (the no-metadata path stamps round_robin_fallback)"
        )
    if headline_live and not admit_round_robin_fallback:
        raise L1ToL2PartitionMetadataMissing(
            "no per-CTA sm_id / warpid_sm partition metadata and "
            "admit_round_robin_fallback is False; refusing headline-live "
            "L1/L2 filter rather than silently round-robin-partitioning "
            "the access stream"
        )
    return _run_filter(
        per_block_accesses,
        sm_count=sm_count,
        l1_capacity_bytes=l1_capacity_bytes,
        l2_config=l2_config,
        schedule_policy="round_robin_homogeneous",
        assignment=None,
        partition_provenance=PARTITION_ROUND_ROBIN_FALLBACK,
        used_real=False,
        warmup_launches=warmup_launches,
    )


def _filtered_replay_for_derivation(
    per_block_accesses: Sequence[Sequence[tuple]],
    *,
    sm_count: int,
    l1_capacity_bytes: int,
    l2_config: scr.CacheReplayConfig,
    derivation,
    warmup_launches: int = 0,
) -> L1ToL2FilterResult:
    """INTERNAL: the ONLY path that stamps ``real_sm_partition``. It requires a
    SUCCESSFUL ``CtaToSmDerivation`` -- ``metadata_present=True`` with a non-None
    assignment built by ``derive_cta_to_sm_assignment`` from the trace's real
    ``sm_id``. Because the public ``l1_to_l2_filtered_replay`` refuses
    ``real_sm_partition``, the trusted label cannot be minted from a raw
    caller-supplied map."""
    if not (
        getattr(derivation, "metadata_present", False)
        and getattr(derivation, "assignment", None) is not None
    ):
        raise ValueError(
            "real_sm_partition requires a successful CtaToSmDerivation "
            "(metadata_present=True with a non-None assignment); got "
            f"{derivation!r}"
        )
    return _run_filter(
        per_block_accesses,
        sm_count=sm_count,
        l1_capacity_bytes=l1_capacity_bytes,
        l2_config=l2_config,
        schedule_policy="occupancy_derived",
        assignment=derivation.assignment,
        partition_provenance=PARTITION_REAL_SM,
        used_real=True,
        warmup_launches=warmup_launches,
    )


def _run_filter(
    per_block_accesses: Sequence[Sequence[tuple]],
    *,
    sm_count: int,
    l1_capacity_bytes: int,
    l2_config: scr.CacheReplayConfig,
    schedule_policy: str,
    assignment: Sequence[int] | None,
    partition_provenance: str,
    used_real: bool,
    warmup_launches: int,
) -> L1ToL2FilterResult:
    """INTERNAL mechanics: run the per-SM L1, emit the L1-miss stream, replay it
    through L2, and assemble the result with the RESOLVED provenance. The callers
    decide the provenance (the public API refuses ``real_sm_partition``); this
    function does not infer trust from its inputs."""
    l1_result, stream = l1_miss_stream_per_sm(
        per_block_accesses,
        sm_count=sm_count,
        capacity_bytes=l1_capacity_bytes,
        schedule_policy=schedule_policy,
        occupancy_derived_assignment=assignment,
        warmup_launches=warmup_launches,
    )

    # Internal consistency: the L2 input is exactly the accesses L1 did not
    # serve -- the L1-eligible misses plus the L1-bypass accesses.
    expected_l2_input = l1_result.miss_count + l1_result.bypass_count
    if len(stream) != expected_l2_input:
        raise AssertionError(
            f"L1-miss stream length {len(stream)} != miss_count "
            f"{l1_result.miss_count} + bypass_count {l1_result.bypass_count}"
        )

    line_ids = [line_id for line_id, _ in stream]
    sector_offsets = [offset for _, offset in stream]
    l2_result = scr.replay_access_stream(line_ids, sector_offsets, l2_config)

    return L1ToL2FilterResult(
        l2_input_stream=tuple(stream),
        l1_result=l1_result,
        l2_result=l2_result,
        partition_provenance=partition_provenance,
        l2_input_provenance=L2_INPUT_L1_FILTERED,
        used_real_partition_metadata=used_real,
        l1_hits_absorbed=l1_result.hit_count,
        l1_eligible_misses=l1_result.miss_count,
        l1_bypass_accesses=l1_result.bypass_count,
        l2_input_count=len(stream),
    )


def filtered_replay_from_trace(
    trace_records,
    classifier_manifest: dict,
    *,
    sm_count: int,
    grid_dim: int | None,
    block_dim: int | None,
    l1_capacity_bytes: int,
    l2_config: scr.CacheReplayConfig,
    headline_live: bool = False,
    admit_round_robin_fallback: bool = False,
    warmup_launches: int = 0,
    require_data_width: bool = False,
) -> L1ToL2FilterResult:
    """Decode an NVBit trace through the existing import-contract adapter
    (``build_per_block_l1_accesses``, which applies the .ca/.cg L1 classifier),
    DERIVE the per-CTA SM partition from the trace's REAL ``sm_id`` metadata
    (``derive_cta_to_sm_assignment``), and run the L1 -> miss-stream -> L2 filter.
    This is the trace-fed entry point; ``l1_to_l2_filtered_replay`` is the
    per-block-accesses core.

    The partition is NEVER caller-supplied here: it is derived from the records
    so the ``real_sm_partition`` provenance is only ever claimed when the trace
    actually carries a consistent per-CTA ``sm_id``. ``derive_cta_to_sm_assignment``
    raises on conflicting / partial ``sm_id`` metadata; when ``sm_id`` is absent
    everywhere the call falls through to the core's fail-closed (headline-live) /
    round-robin-fallback path."""
    from l1_trace_adapter import (  # lazy: heavy chain
        build_per_block_l1_accesses,
        derive_cta_to_sm_assignment,
    )

    # Materialize: the records are consumed by BOTH the access decode and the
    # sm_id derivation, so a single-pass iterator would be exhausted after the
    # first consumer.
    records = list(trace_records)
    per_block_accesses = build_per_block_l1_accesses(
        records,
        classifier_manifest,
        sm_count=sm_count,
        grid_dim=grid_dim,
        block_dim=block_dim,
        require_data_width=require_data_width,
    )
    derivation = derive_cta_to_sm_assignment(
        records, grid_dim=grid_dim, sm_count=sm_count
    )
    if derivation.metadata_present:
        # Route through the internal helper -- the ONLY minter of
        # real_sm_partition -- which requires this successful derivation. The
        # public core cannot stamp real_sm_partition, so the trusted label is
        # reachable only from a proven trace-derived assignment.
        return _filtered_replay_for_derivation(
            per_block_accesses,
            sm_count=sm_count,
            l1_capacity_bytes=l1_capacity_bytes,
            l2_config=l2_config,
            derivation=derivation,
            warmup_launches=warmup_launches,
        )
    # No sm_id metadata anywhere: fall through to the core's fail-closed /
    # round-robin-fallback path (no assignment, no provenance claim).
    return l1_to_l2_filtered_replay(
        per_block_accesses,
        sm_count=sm_count,
        l1_capacity_bytes=l1_capacity_bytes,
        l2_config=l2_config,
        cta_to_sm_assignment=None,
        headline_live=headline_live,
        admit_round_robin_fallback=admit_round_robin_fallback,
        warmup_launches=warmup_launches,
    )


def enforce_l1_miss_filtered_for_l1l2_headline(
    cache_layers: Sequence[str],
    l2_input_provenance: str,
) -> None:
    """Guard the cache comparison path: when the comparison scope is L1/L2
    headline mode (both ``L1`` and ``L2`` present in ``cache_layers``), the L2
    leg MUST be the L1-miss-filtered stream. A full-stream L2 replay is rejected
    (``FullStreamL2RejectedForL1L2Headline``). An L2-only scope imposes no such
    requirement."""
    layers = {str(layer).strip().upper() for layer in cache_layers}
    if {"L1", "L2"} <= layers and l2_input_provenance != L2_INPUT_L1_FILTERED:
        raise FullStreamL2RejectedForL1L2Headline(
            f"L1/L2 headline comparison (cache_layers={sorted(layers)}) requires "
            f"the L1-miss-filtered L2 stream "
            f"(l2_input_provenance={L2_INPUT_L1_FILTERED!r}); got "
            f"{l2_input_provenance!r}. A full-stream L2 replay over-predicts the "
            f"L2 sector hit rate by double-counting intra-line reuse the per-SM "
            f"L1 absorbs, so it is refused for the headline L1/L2 path."
        )
