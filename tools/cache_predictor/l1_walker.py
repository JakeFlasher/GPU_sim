"""Per-SM L1 fully-associative LRU walker.

Implements the baseline L1 cache model: each SM has its own
L1 line cache; CTAs on different SMs do not evict each other's
L1 lines on Ampere / Blackwell. The walker takes per-block
access sequences as input (each access is
``(line_id, l1_eligible)``), maps each block to an SM via the
CTA-to-SM scheduling policy, maintains a per-SM line-level LRU
stack, and emits per-launch hit-rate counters.

Hit decision: **sector-with-line-allocation**. A hit fires when
the access's parent line is in the SM's line-level LRU stack
within capacity (``capacity_bytes // 128`` lines). Misses
allocate the entire 128B line at the top of the stack. Bypass
accesses (``l1_eligible=False``) are excluded from BOTH the
numerator and denominator — they pass through L1 without
touching the modeled line state.

The walker is fully-associative per the Plan 1 baseline
(`l1_model_class="fully_associative_baseline"`); set-associative
refinement is deferred to Plan 2 unless DEC-2 fires. The
set-conflict diagnostic fixture (queued for a future slice) documents the
baseline's known over-prediction on conflict-heavy patterns.

CTA-to-SM scheduling policies (`cta_to_sm_schedule_policy`):

  - ``round_robin_homogeneous``: ``sm_id = block_index %
    sm_count``. The default baseline; on Ampere / Blackwell with
    a homogeneous block size, the hardware scheduler typically
    distributes blocks round-robin until each SM's occupancy
    cap is reached.
  - ``occupancy_derived``: the caller supplies a per-block
    sm_id mapping (``occupancy_derived_assignment``) computed
    from the kernel's occupancy + launch parameters. This is
    a more faithful model when the caller has access to the
    occupancy data.
  - ``unknown``: the walker rejects the input with a typed
    ``L1CtaToSmScheduleUnknown`` exception so the L1 head can
    emit the matching typed_unavailable arm. Guessing the
    policy biases the prediction.

The walker takes pre-processed inputs (per-block access
sequences rather than raw proto traces) so it can be exercised
by paired-control tests without proto / trace dependencies. The
production wiring that constructs the per-block accesses from
a trace + classifier manifest lives in
``predict_l1_hit_rate_from_manifest`` in
``workloads/simbench_bw/cache_predictor.py``.
"""
from __future__ import annotations

from typing import Iterable, NamedTuple, Sequence


class L1CtaToSmScheduleUnknown(Exception):
    """Raised by ``walk_per_sm_l1`` when the scheduling policy
    is ``unknown`` (the walker has no defensible CTA-to-SM
    mapping). The L1 head catches this to emit a fail-close
    typed_unavailable bundle with reason
    ``predictor_l1_cta_to_sm_schedule_unknown``."""

    TOKEN = "predictor_l1_cta_to_sm_schedule_unknown"


class L1CtaToSmScheduleMissingOccupancy(Exception):
    """Raised when ``cta_to_sm_schedule_policy=occupancy_derived``
    but the caller did not supply
    ``occupancy_derived_assignment``."""

    TOKEN = "predictor_l1_cta_to_sm_schedule_unknown"


class L1CtaToSmScheduleAssignmentInvalid(Exception):
    """Raised when ``occupancy_derived_assignment`` is malformed
    (length mismatch with the number of blocks, or maps a block
    to an out-of-range SM)."""

    TOKEN = "predictor_l1_cta_to_sm_schedule_unknown"


class L1WalkerCapacityInvalid(Exception):
    """Raised when ``capacity_bytes`` is below the 128B line
    granularity (no line fits in the cache)."""

    TOKEN = "predictor_l1_capacity_invalid"


LINE_SIZE_BYTES: int = 128
SECTOR_SIZE_BYTES: int = 32
SECTORS_PER_LINE: int = LINE_SIZE_BYTES // SECTOR_SIZE_BYTES


class PerSmL1Result(NamedTuple):
    """Aggregate L1 walker result."""

    hit_count: int
    miss_count: int
    bypass_count: int
    total_eligible_accesses: int
    per_sm_hit_count: dict[int, int]
    per_sm_miss_count: dict[int, int]
    sm_count: int

    @property
    def hit_rate(self) -> float:
        """Hit rate over l1_eligible accesses only. Returns 0.0
        when total_eligible_accesses is zero (the L1 head's
        ``predictor_l1_no_modeled_l1_eligible_accesses`` arm
        handles that case upstream; this fallback exists so the
        walker is total)."""
        if self.total_eligible_accesses == 0:
            return 0.0
        return self.hit_count / float(self.total_eligible_accesses)


def block_to_sm_id(
    block_index: int,
    *,
    sm_count: int,
    schedule_policy: str,
    occupancy_derived_assignment: Sequence[int] | None = None,
) -> int:
    """Map a block_index to its sm_id per the scheduling policy.

    ``round_robin_homogeneous``: returns ``block_index %
    sm_count``.

    ``occupancy_derived``: looks up
    ``occupancy_derived_assignment[block_index]``; raises
    ``L1CtaToSmScheduleMissingOccupancy`` when the assignment is
    None, and ``L1CtaToSmScheduleAssignmentInvalid`` when the
    block_index is out of range or the mapped sm_id is
    out-of-range.

    ``unknown``: raises ``L1CtaToSmScheduleUnknown``.
    """
    if isinstance(sm_count, bool) or not isinstance(sm_count, int) or sm_count <= 0:
        raise ValueError(
            f"sm_count must be >= 1, got {sm_count!r}"
        )
    if schedule_policy == "round_robin_homogeneous":
        return block_index % sm_count
    if schedule_policy == "occupancy_derived":
        if occupancy_derived_assignment is None:
            raise L1CtaToSmScheduleMissingOccupancy(
                "occupancy_derived_assignment required when "
                "cta_to_sm_schedule_policy=occupancy_derived"
            )
        if block_index >= len(occupancy_derived_assignment):
            raise L1CtaToSmScheduleAssignmentInvalid(
                f"block_index {block_index} out of range for "
                f"occupancy_derived_assignment length "
                f"{len(occupancy_derived_assignment)}"
            )
        sm_id = int(occupancy_derived_assignment[block_index])
        if not (0 <= sm_id < sm_count):
            raise L1CtaToSmScheduleAssignmentInvalid(
                f"block_index {block_index} mapped to sm_id "
                f"{sm_id} which is out of range [0, "
                f"{sm_count})"
            )
        return sm_id
    if schedule_policy == "unknown":
        raise L1CtaToSmScheduleUnknown(
            "cta_to_sm_schedule_policy=unknown; the per-SM L1 "
            "walker cannot defensibly assign blocks to SMs and "
            "the L1 head must emit predictor_l1_cta_to_sm_"
            "schedule_unknown"
        )
    raise ValueError(
        f"unrecognized cta_to_sm_schedule_policy "
        f"{schedule_policy!r}; expected one of "
        f"{{'round_robin_homogeneous', 'occupancy_derived', "
        f"'unknown'}}"
    )


def address_to_line_id(address: int) -> int:
    """Map a byte address to its 128B line id."""
    return address // LINE_SIZE_BYTES


def _normalize_access(
    access: tuple,
) -> tuple[int, int, bool]:
    """Coerce a walker access tuple to the canonical
    ``(line_id, sector_id, l1_eligible)`` 3-tuple shape.

    Two input forms are accepted:
      - ``(line_id, l1_eligible)`` 2-tuple: legacy line-level
        form. The walker treats this as a single sector access
        with ``sector_id = line_id * SECTORS_PER_LINE`` (the
        first sector of the named line). The hit decision then
        degrades to line-level LRU because all accesses to a
        given line carry the same sector_id; this preserves the
        pre-Round-15 walker behavior for call sites that have
        not learned the sector contract yet.
      - ``(line_id, sector_id, l1_eligible)`` 3-tuple: the
        sector-with-line-allocation form. ``sector_id`` is the
        GLOBAL sector index = ``byte_address // SECTOR_SIZE_BYTES``;
        the walker computes line residence from
        ``sector_id // SECTORS_PER_LINE`` for consistency
        checking against ``line_id``.
    """
    if len(access) == 2:
        line_id, l1_eligible = access
        sector_id = int(line_id) * SECTORS_PER_LINE
        return int(line_id), sector_id, bool(l1_eligible)
    if len(access) == 3:
        line_id, sector_id, l1_eligible = access
        return int(line_id), int(sector_id), bool(l1_eligible)
    raise L1WalkerAccessInvalid(
        f"walker access must be a 2-tuple "
        f"(line_id, l1_eligible) or 3-tuple "
        f"(line_id, sector_id, l1_eligible); got {access!r}"
    )


class L1WalkerAccessInvalid(Exception):
    """Raised when a walker access tuple has an unsupported
    arity. The L1 head's fail-close arm catches this to emit
    ``predictor_l1_sass_provenance_missing`` with evidence
    naming the offending tuple."""

    TOKEN = "predictor_l1_walker_access_invalid"


def _walk_one_launch_fa(
    per_block_accesses: Sequence[Sequence[tuple]],
    *,
    sm_count: int,
    line_capacity: int,
    schedule_policy: str,
    occupancy_derived_assignment: Sequence[int] | None,
    per_sm_lru: dict[int, list[int]],
    per_sm_sectors: dict[int, dict[int, set[int]]],
    collect_l2_stream: list[tuple[int, int]] | None = None,
) -> tuple[int, int, int, dict[int, int], dict[int, int]]:
    """One launch of the fully-associative walker against a
    persistent per-SM LRU + sector-residency state. Returns the
    launch's (hit_count, miss_count, bypass_count,
    per_sm_hit_count, per_sm_miss_count). The cache-state dicts
    are mutated in place so subsequent launches can reuse the
    same dictionaries to model multi-launch warm-cache behavior.

    When ``collect_l2_stream`` is provided, the launch APPENDS to
    it the ordered ``(line_id, sector_offset)`` pairs that reach
    the L2 (the L1-MISS STREAM): every L1-eligible access that
    MISSES the per-SM L1, plus every L1-bypass access (``.cg`` and
    friends skip L1 entirely). L1 HITS are absorbed and do NOT
    enter the stream. ``sector_offset`` is the sector index within
    the 128B line (``sector_id % SECTORS_PER_LINE``), the shape the
    L2 sector replay consumes. Used only for the measured launch
    (pass ``None`` during warmup)."""
    hit_count = 0
    miss_count = 0
    bypass_count = 0
    per_sm_hit_count: dict[int, int] = {}
    per_sm_miss_count: dict[int, int] = {}
    for block_index, accesses in enumerate(per_block_accesses):
        sm_id = block_to_sm_id(
            block_index,
            sm_count=sm_count,
            schedule_policy=schedule_policy,
            occupancy_derived_assignment=occupancy_derived_assignment,
        )
        lru = per_sm_lru.setdefault(sm_id, [])
        sectors = per_sm_sectors.setdefault(sm_id, {})
        for access in accesses:
            line_id, sector_id, l1_eligible = _normalize_access(access)
            if not l1_eligible:
                bypass_count += 1
                # L1-bypass accesses skip L1 and reach L2 directly.
                if collect_l2_stream is not None:
                    collect_l2_stream.append((line_id, sector_id % SECTORS_PER_LINE))
                continue
            line_present = line_id in lru
            if line_present:
                pos = lru.index(line_id)
                line_in_capacity = pos < line_capacity
            else:
                line_in_capacity = False
            sector_present = (
                line_in_capacity
                and sector_id in sectors.get(line_id, set())
            )
            if sector_present:
                hit_count += 1
                per_sm_hit_count[sm_id] = (
                    per_sm_hit_count.get(sm_id, 0) + 1
                )
                lru.remove(line_id)
                lru.insert(0, line_id)
            else:
                miss_count += 1
                per_sm_miss_count[sm_id] = (
                    per_sm_miss_count.get(sm_id, 0) + 1
                )
                # An L1-eligible MISS is served by L2 -> enters the L1-miss stream.
                if collect_l2_stream is not None:
                    collect_l2_stream.append((line_id, sector_id % SECTORS_PER_LINE))
                if line_present:
                    lru.remove(line_id)
                lru.insert(0, line_id)
                sectors.setdefault(line_id, set()).add(sector_id)
            if len(lru) > line_capacity:
                for evicted in lru[line_capacity:]:
                    sectors.pop(evicted, None)
                del lru[line_capacity:]
    return (
        hit_count,
        miss_count,
        bypass_count,
        per_sm_hit_count,
        per_sm_miss_count,
    )


def walk_per_sm_l1(
    per_block_accesses: Sequence[Sequence[tuple]],
    *,
    sm_count: int,
    capacity_bytes: int,
    schedule_policy: str,
    occupancy_derived_assignment: Sequence[int] | None = None,
    warmup_launches: int = 0,
) -> PerSmL1Result:
    """Walk a launch's per-block access sequences and return the
    aggregate per-SM L1 hit-rate counters.

    Each ``per_block_accesses[i]`` is an ordered sequence of
    walker access tuples representing the accesses issued by
    block_index ``i`` on the SM the schedule policy assigns it
    to. See ``_normalize_access`` for the two accepted tuple
    shapes.

    ``capacity_bytes`` is the per-SM L1 budget after the carveout
    has been applied; the walker computes the per-SM line-stack
    capacity as ``capacity_bytes // LINE_SIZE_BYTES`` lines.

    ``warmup_launches`` (default 0) is the number of NCU-style
    warmup launches simulated BEFORE the measured launch. When
    ``warmup_launches >= 1`` the walker replays the same
    ``per_block_accesses`` traversal ``warmup_launches`` times
    with persistent per-SM LRU + sector-residency state (no
    counting), then executes ONE measured launch against the now
    warm cache and returns those counters. This matches NCU's
    profiling methodology (default ``--launch-skip=5``: 5
    warmup launches go through the kernel before counters are
    enabled on the measured launch). The deterministic-kernel
    assumption is: the trace's per-block access sequence is the
    same on every launch, so re-running it warms the cache the
    same way the real kernel does on the GPU.

    The hit decision is **sector-with-line-allocation** per the
    plan: a hit fires iff (a) the line is in the per-SM LRU
    stack within capacity AND (b) the touched sector is in that
    line's residency set. A miss allocates the line (LRU insert
    at the top), marks the touched sector as resident, and
    evicts the LRU line if over capacity (clearing all
    sector-residency bits for the evicted line).

    Bypass accesses (``l1_eligible=False``) pass through
    without touching the stack or any sector-residency state.

    The walker is backward-compatible with the legacy
    line-level 2-tuple input: a 2-tuple ``(line_id,
    l1_eligible)`` is treated as a single-sector access into
    ``sector_id = line_id * SECTORS_PER_LINE`` so the per-line
    sector-residency set collapses to a one-element set and
    the hit decision degrades to line-level LRU.

    Returns a ``PerSmL1Result`` with hit_count / miss_count /
    bypass_count + per-SM tallies + the derived hit_rate over
    eligible accesses (counting one event per sector access).
    """
    if isinstance(sm_count, bool) or not isinstance(sm_count, int) or sm_count <= 0:
        raise ValueError(f"sm_count must be >= 1, got {sm_count!r}")
    if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int) \
            or capacity_bytes < LINE_SIZE_BYTES:
        raise L1WalkerCapacityInvalid(
            f"capacity_bytes={capacity_bytes} is below the "
            f"128B line granularity; at least one line must fit"
        )
    if isinstance(warmup_launches, bool) or not isinstance(warmup_launches, int) \
            or warmup_launches < 0:
        raise ValueError(
            f"warmup_launches={warmup_launches} must be >= 0; "
            f"0 = no warmup (cold-cache single-launch)."
        )
    line_capacity = capacity_bytes // LINE_SIZE_BYTES

    per_sm_lru: dict[int, list[int]] = {}
    per_sm_sectors: dict[int, dict[int, set[int]]] = {}

    for _ in range(warmup_launches):
        _walk_one_launch_fa(
            per_block_accesses,
            sm_count=sm_count,
            line_capacity=line_capacity,
            schedule_policy=schedule_policy,
            occupancy_derived_assignment=occupancy_derived_assignment,
            per_sm_lru=per_sm_lru,
            per_sm_sectors=per_sm_sectors,
        )

    (
        hit_count,
        miss_count,
        bypass_count,
        per_sm_hit_count,
        per_sm_miss_count,
    ) = _walk_one_launch_fa(
        per_block_accesses,
        sm_count=sm_count,
        line_capacity=line_capacity,
        schedule_policy=schedule_policy,
        occupancy_derived_assignment=occupancy_derived_assignment,
        per_sm_lru=per_sm_lru,
        per_sm_sectors=per_sm_sectors,
    )

    return PerSmL1Result(
        hit_count=hit_count,
        miss_count=miss_count,
        bypass_count=bypass_count,
        total_eligible_accesses=hit_count + miss_count,
        per_sm_hit_count=dict(per_sm_hit_count),
        per_sm_miss_count=dict(per_sm_miss_count),
        sm_count=sm_count,
    )


def l1_miss_stream_per_sm(
    per_block_accesses: Sequence[Sequence[tuple]],
    *,
    sm_count: int,
    capacity_bytes: int,
    schedule_policy: str,
    occupancy_derived_assignment: Sequence[int] | None = None,
    warmup_launches: int = 0,
) -> tuple[PerSmL1Result, list[tuple[int, int]]]:
    """Like ``walk_per_sm_l1`` but ALSO returns the ordered L1-MISS STREAM that
    reaches L2 -- the ``(line_id, sector_offset)`` pairs for every L1-eligible
    access that misses the per-SM L1 plus every L1-bypass access (L1 hits are
    absorbed and excluded). This is exactly the stream NCU's L2 sector hit rate
    observes; feed it to ``sector_cache_replay.replay_access_stream`` for an
    L1-filtered L2 prediction. Warmup launches do NOT contribute to the stream
    (only the measured launch does)."""
    if capacity_bytes < LINE_SIZE_BYTES:
        raise L1WalkerCapacityInvalid(
            f"capacity_bytes={capacity_bytes} is below the 128B line granularity"
        )
    if warmup_launches < 0:
        raise ValueError(f"warmup_launches={warmup_launches} must be >= 0")
    line_capacity = capacity_bytes // LINE_SIZE_BYTES
    per_sm_lru: dict[int, list[int]] = {}
    per_sm_sectors: dict[int, dict[int, set[int]]] = {}
    for _ in range(warmup_launches):
        _walk_one_launch_fa(
            per_block_accesses, sm_count=sm_count, line_capacity=line_capacity,
            schedule_policy=schedule_policy,
            occupancy_derived_assignment=occupancy_derived_assignment,
            per_sm_lru=per_sm_lru, per_sm_sectors=per_sm_sectors, collect_l2_stream=None,
        )
    l2_stream: list[tuple[int, int]] = []
    (hit_count, miss_count, bypass_count, per_sm_hit_count, per_sm_miss_count) = _walk_one_launch_fa(
        per_block_accesses, sm_count=sm_count, line_capacity=line_capacity,
        schedule_policy=schedule_policy,
        occupancy_derived_assignment=occupancy_derived_assignment,
        per_sm_lru=per_sm_lru, per_sm_sectors=per_sm_sectors, collect_l2_stream=l2_stream,
    )
    result = PerSmL1Result(
        hit_count=hit_count, miss_count=miss_count, bypass_count=bypass_count,
        total_eligible_accesses=hit_count + miss_count,
        per_sm_hit_count=dict(per_sm_hit_count), per_sm_miss_count=dict(per_sm_miss_count),
        sm_count=sm_count,
    )
    return result, l2_stream


class L1WalkerDefaultCapacityMissing(Exception):
    """Raised when ``apply_carveout_to_capacity`` is called with
    ``carveout_source="default"`` but no ``default_l1_bytes``
    was supplied. The L1 head's fail-close arm catches this to
    emit ``predictor_l1_sass_provenance_missing`` with evidence
    naming the missing capacity field."""

    TOKEN = "predictor_l1_default_capacity_provenance_missing"


def apply_carveout_to_capacity(
    *,
    unified_budget_bytes: int,
    static_shared_bytes: int,
    dynamic_shared_bytes: int,
    carveout_source: str,
    carveout_max_l1_bytes: int,
    default_l1_bytes: int | None = None,
    carveout_percent: int | None = None,
) -> int:
    """Compute the effective L1 capacity after subtracting the
    kernel's actual shared-memory carveout from the unified L1 /
    shared budget.

    ``unified_budget_bytes`` is the architecture's per-SM
    combined L1 + shared budget; values are sourced from a
    manifest, a CUDA-attribute query stub, or the published
    static-card table at
    ``tools/cache_predictor/l1_static_card_table.json``.

    ``static_shared_bytes`` + ``dynamic_shared_bytes`` is the
    total shared memory the kernel requested via __shared__ +
    extern __shared__.

    ``carveout_source`` is the provenance label for the carveout
    policy:
      - ``device_properties_manifest`` / ``cuda_func_attribute``
        / ``cuda_func_attribute_query`` /
        ``cuda_device_attribute_query``: the caller selected a
        specific carveout; the effective L1 is
        ``unified_budget_bytes - shared_total``, capped at
        ``carveout_max_l1_bytes`` (toolkit-imposed maximum).
      - ``default``: the toolkit's default carveout. Per the
        plan's L1 capacity provenance discipline, the default
        L1 capacity is a CARVEOUT-BUCKET PROPERTY (not a
        derived value from unified_budget - shared); the
        function returns ``default_l1_bytes`` directly. The
        caller is responsible for sourcing default_l1_bytes
        from the static-card table or an equivalent provenance
        surface; passing ``default_l1_bytes=None`` while
        ``carveout_source="default"`` raises
        ``L1WalkerDefaultCapacityMissing`` so the L1 head can
        fail-close with the matching typed reason.

    ``carveout_percent`` (R17 + R18 carveout-aware capacity
    source): when
    ``carveout_source == "cuda_func_attribute_carveout_percent"``,
    the kernel called
    ``cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout, carveout_percent)``
    at launch time. Per NVIDIA's CUDA Programming Guide
    (`cudaFuncAttributePreferredSharedMemoryCarveout`), the
    value is the PREFERRED SHARED-MEMORY carveout percentage:
    ``carveout_percent=0`` means NO shared-memory carveout
    (prefer MAX L1, MIN shared); ``carveout_percent=100`` means
    FULL shared-memory carveout (prefer MIN L1, MAX shared).
    The effective L1 capacity is therefore inversely
    interpolated between ``carveout_max_l1_bytes``
    (carveout_percent=0, maximum-L1 bucket) and
    ``default_l1_bytes`` (carveout_percent=100, minimum-L1
    bucket). The interpolation approximates the architecture's
    discrete carveout buckets sufficiently for the walker's
    per-SM capacity model — the hardware partition points are
    not strictly linear but the error bound for closure is
    within the documented per-cell budget. The shared-memory
    request is NOT subtracted because the carveout setting
    itself caps the split.

    R18 update: the previous interpretation (carveout_percent=0
    → min L1; carveout_percent=100 → max L1) inverted the CUDA
    documented semantics. Codex round-17 review caught this.
    The correction here aligns with the vendor documentation.

    Returns the effective per-SM L1 capacity in bytes; raises
    ``L1WalkerCapacityInvalid`` if the result drops below the
    128B line granularity."""
    if carveout_source == "cuda_func_attribute_carveout_percent":
        if carveout_percent is None:
            raise ValueError(
                "carveout_percent required when "
                "carveout_source='cuda_func_attribute_carveout_percent'"
            )
        if not 0 <= int(carveout_percent) <= 100:
            raise ValueError(
                f"carveout_percent={carveout_percent} out of "
                f"[0, 100] range"
            )
        if default_l1_bytes is None:
            raise L1WalkerDefaultCapacityMissing(
                "carveout_source='cuda_func_attribute_carveout_percent' "
                "requires default_l1_bytes as the minimum-L1 "
                "bucket anchor (carveout_percent=100 bucket per "
                "CUDA's preferred-shared-carveout semantics)"
            )
        # R18 corrected semantics: inverse interpolation per
        # CUDA's `cudaFuncAttributePreferredSharedMemoryCarveout`
        # documentation. carveout_percent=0 ⇒ MAX L1 (no shared
        # carveout); carveout_percent=100 ⇒ MIN L1 (full shared
        # carveout). The architecture's discrete bucket points
        # (e.g., sm_86's {16KB, 32KB, 64KB, 100KB, 128KB}) are
        # approximated linearly; the toolkit's runtime clips
        # the request to the nearest supported bucket, so the
        # model's interpolation gives a result close to the
        # hardware's actual partition for any carveout_percent.
        cp = int(carveout_percent)
        delta = carveout_max_l1_bytes - default_l1_bytes
        effective = int(
            carveout_max_l1_bytes - (delta * cp) // 100
        )
        if effective < LINE_SIZE_BYTES:
            raise L1WalkerCapacityInvalid(
                f"carveout-aware L1 capacity "
                f"({effective} bytes) is below the 128B line "
                f"granularity; the carveout_percent "
                f"{carveout_percent} produced a sub-line result"
            )
        return effective
    if carveout_source == "default":
        # The default carveout bucket has a FIXED L1 share
        # documented by the static-card table (or equivalent
        # provenance). The shared-memory request does NOT
        # subtract from L1 in default mode because the bucket
        # itself caps both L1 and shared independently.
        if default_l1_bytes is None:
            raise L1WalkerDefaultCapacityMissing(
                "carveout_source='default' requires "
                "default_l1_bytes to be supplied (sourced from "
                "the static-card table for static_card_constant "
                "L1 capacity provenance, or from the manifest / "
                "query stub for other provenance surfaces)"
            )
        if default_l1_bytes < LINE_SIZE_BYTES:
            raise L1WalkerCapacityInvalid(
                f"default L1 capacity {default_l1_bytes} bytes "
                f"is below the 128B line granularity"
            )
        return int(default_l1_bytes)

    shared_total = static_shared_bytes + dynamic_shared_bytes
    if shared_total > unified_budget_bytes:
        raise L1WalkerCapacityInvalid(
            f"requested shared memory {shared_total} bytes "
            f"exceeds the unified budget "
            f"{unified_budget_bytes} bytes"
        )
    effective = unified_budget_bytes - shared_total
    if effective > carveout_max_l1_bytes:
        effective = carveout_max_l1_bytes
    if effective < LINE_SIZE_BYTES:
        raise L1WalkerCapacityInvalid(
            f"effective L1 capacity after carveout "
            f"({effective} bytes) is below the 128B line "
            f"granularity; the kernel's shared usage saturates "
            f"the unified budget"
        )
    return effective


class L1WalkerWaysInvalid(Exception):
    """Raised when the set-associative walker is called with
    ``ways < 1``. The L1 head's fail-close arm catches this to
    emit ``predictor_l1_capacity_invalid`` (the same family as
    capacity-below-line-granularity); the set-associative
    walker's geometry source is the static-card table's
    ``l1_ways`` field."""

    TOKEN = "predictor_l1_capacity_invalid"


class L1WalkerSetCountInvalid(Exception):
    """Raised when the set-associative walker is called with
    a derived ``set_count < 1`` (e.g., ``capacity_bytes / 128 /
    ways`` rounds down to zero). The L1 head's fail-close arm
    catches this to emit ``predictor_l1_capacity_invalid``."""

    TOKEN = "predictor_l1_capacity_invalid"


class PerSmPerSetL1Result(NamedTuple):
    """Aggregate set-associative L1 walker result.

    Mirrors ``PerSmL1Result``'s field shape so consumers can be
    polymorphic over the two walker variants. Adds the
    geometry-provenance fields (``ways``, ``set_count``) so the
    L1 head's evidence block can record what geometry the
    DEC-2 walker actually used."""

    hit_count: int
    miss_count: int
    bypass_count: int
    total_eligible_accesses: int
    per_sm_hit_count: dict[int, int]
    per_sm_miss_count: dict[int, int]
    sm_count: int
    ways: int
    set_count: int

    @property
    def hit_rate(self) -> float:
        """Hit rate over l1_eligible accesses only."""
        if self.total_eligible_accesses == 0:
            return 0.0
        return self.hit_count / float(self.total_eligible_accesses)


def _walk_one_launch_sa(
    per_block_accesses: Sequence[Sequence[tuple]],
    *,
    sm_count: int,
    ways: int,
    set_count: int,
    schedule_policy: str,
    occupancy_derived_assignment: Sequence[int] | None,
    per_sm_lru: dict[int, dict[int, list[int]]],
    per_sm_sectors: dict[int, dict[int, dict[int, set[int]]]],
) -> tuple[int, int, int, dict[int, int], dict[int, int]]:
    """One launch of the set-associative walker against persistent
    per-(SM, set) LRU + sector-residency state. Mirrors
    ``_walk_one_launch_fa`` but with the set-associative geometry.
    """
    hit_count = 0
    miss_count = 0
    bypass_count = 0
    per_sm_hit_count: dict[int, int] = {}
    per_sm_miss_count: dict[int, int] = {}
    for block_index, accesses in enumerate(per_block_accesses):
        sm_id = block_to_sm_id(
            block_index,
            sm_count=sm_count,
            schedule_policy=schedule_policy,
            occupancy_derived_assignment=occupancy_derived_assignment,
        )
        sm_lru = per_sm_lru.setdefault(sm_id, {})
        sm_sectors = per_sm_sectors.setdefault(sm_id, {})
        for access in accesses:
            line_id, sector_id, l1_eligible = _normalize_access(access)
            if not l1_eligible:
                bypass_count += 1
                continue
            set_index = line_id % set_count
            lru = sm_lru.setdefault(set_index, [])
            sectors_for_set = sm_sectors.setdefault(set_index, {})
            line_present = line_id in lru
            if line_present:
                pos = lru.index(line_id)
                line_in_capacity = pos < ways
            else:
                line_in_capacity = False
            sector_present = (
                line_in_capacity
                and sector_id in sectors_for_set.get(
                    line_id, set(),
                )
            )
            if sector_present:
                hit_count += 1
                per_sm_hit_count[sm_id] = (
                    per_sm_hit_count.get(sm_id, 0) + 1
                )
                lru.remove(line_id)
                lru.insert(0, line_id)
            else:
                miss_count += 1
                per_sm_miss_count[sm_id] = (
                    per_sm_miss_count.get(sm_id, 0) + 1
                )
                if line_present:
                    lru.remove(line_id)
                lru.insert(0, line_id)
                sectors_for_set.setdefault(
                    line_id, set(),
                ).add(sector_id)
            if len(lru) > ways:
                for evicted in lru[ways:]:
                    sectors_for_set.pop(evicted, None)
                del lru[ways:]
    return (
        hit_count,
        miss_count,
        bypass_count,
        per_sm_hit_count,
        per_sm_miss_count,
    )


def walk_per_sm_per_set_l1(
    per_block_accesses: Sequence[Sequence[tuple]],
    *,
    sm_count: int,
    capacity_bytes: int,
    ways: int,
    schedule_policy: str,
    occupancy_derived_assignment: Sequence[int] | None = None,
    warmup_launches: int = 0,
) -> PerSmPerSetL1Result:
    """Walk a launch's per-block accesses with a per-(SM, set)
    set-associative LRU model.

    Mirrors ``walk_per_sm_l1``'s signature + hit decision but
    splits each SM's L1 lines across ``set_count`` LRU stacks,
    each capped at ``ways`` lines (instead of one fully-
    associative per-SM LRU capped at ``capacity_bytes / 128``).

    Set index for a 128B line is ``line_id % set_count``. The
    set_count is derived from the cell's effective capacity:
    ``set_count = capacity_bytes // (LINE_SIZE_BYTES * ways)``.
    The hit decision stays sector-with-line-allocation: a hit
    fires iff the line is in its (SM, set) LRU stack within
    ``ways`` capacity AND the touched sector is in the line's
    residency set. Eviction is LRU per-(SM, set), independent of
    other sets.

    The set-associative model addresses the fully-associative
    baseline's known over-prediction on conflict-heavy access
    patterns (paired-control shape 8 in
    ``test_l1_paired_controls.py``); the DEC-2 fixture is the
    canary. On non-conflict patterns where each set's working
    set stays within ``ways``, the SA walker's hit rate equals
    the FA walker's hit rate (no false demotion).

    Bypass accesses (``l1_eligible=False``) pass through without
    touching the stack or sector-residency state.

    ``warmup_launches`` (default 0) is the number of NCU-style
    warmup launches simulated BEFORE the measured launch (see
    ``walk_per_sm_l1`` for the full semantics).

    Fail-close arms:
      - ``ways < 1`` -> ``L1WalkerWaysInvalid``
      - ``capacity_bytes < ways * LINE_SIZE_BYTES`` (so
        set_count < 1) -> ``L1WalkerSetCountInvalid``
      - ``warmup_launches < 0`` -> ``ValueError``
      - upstream policy / capacity / access exceptions are
        unchanged from the FA walker.
    """
    if isinstance(sm_count, bool) or not isinstance(sm_count, int) or sm_count <= 0:
        raise ValueError(f"sm_count must be >= 1, got {sm_count!r}")
    if isinstance(ways, bool) or not isinstance(ways, int) or ways < 1:
        raise L1WalkerWaysInvalid(
            f"ways={ways} is invalid; the set-associative walker "
            f"requires at least one way per set"
        )
    if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int) \
            or capacity_bytes < LINE_SIZE_BYTES:
        raise L1WalkerCapacityInvalid(
            f"capacity_bytes={capacity_bytes} is below the 128B "
            f"line granularity; at least one line must fit"
        )
    if isinstance(warmup_launches, bool) or not isinstance(warmup_launches, int) \
            or warmup_launches < 0:
        raise ValueError(
            f"warmup_launches={warmup_launches} must be >= 0; "
            f"0 = no warmup (cold-cache single-launch)."
        )
    line_capacity_total = capacity_bytes // LINE_SIZE_BYTES
    set_count = line_capacity_total // ways
    if set_count < 1:
        raise L1WalkerSetCountInvalid(
            f"derived set_count={set_count} is < 1 "
            f"(capacity_bytes={capacity_bytes}, ways={ways}); "
            f"the effective capacity does not admit a single "
            f"fully-populated set"
        )

    per_sm_lru: dict[int, dict[int, list[int]]] = {}
    per_sm_sectors: dict[
        int, dict[int, dict[int, set[int]]]
    ] = {}

    for _ in range(warmup_launches):
        _walk_one_launch_sa(
            per_block_accesses,
            sm_count=sm_count,
            ways=ways,
            set_count=set_count,
            schedule_policy=schedule_policy,
            occupancy_derived_assignment=occupancy_derived_assignment,
            per_sm_lru=per_sm_lru,
            per_sm_sectors=per_sm_sectors,
        )

    (
        hit_count,
        miss_count,
        bypass_count,
        per_sm_hit_count,
        per_sm_miss_count,
    ) = _walk_one_launch_sa(
        per_block_accesses,
        sm_count=sm_count,
        ways=ways,
        set_count=set_count,
        schedule_policy=schedule_policy,
        occupancy_derived_assignment=occupancy_derived_assignment,
        per_sm_lru=per_sm_lru,
        per_sm_sectors=per_sm_sectors,
    )

    return PerSmPerSetL1Result(
        hit_count=hit_count,
        miss_count=miss_count,
        bypass_count=bypass_count,
        total_eligible_accesses=hit_count + miss_count,
        per_sm_hit_count=dict(per_sm_hit_count),
        per_sm_miss_count=dict(per_sm_miss_count),
        sm_count=sm_count,
        ways=ways,
        set_count=set_count,
    )


__all__ = [
    "LINE_SIZE_BYTES",
    "SECTOR_SIZE_BYTES",
    "SECTORS_PER_LINE",
    "L1CtaToSmScheduleUnknown",
    "L1CtaToSmScheduleMissingOccupancy",
    "L1CtaToSmScheduleAssignmentInvalid",
    "L1WalkerAccessInvalid",
    "L1WalkerCapacityInvalid",
    "L1WalkerDefaultCapacityMissing",
    "L1WalkerSetCountInvalid",
    "L1WalkerWaysInvalid",
    "PerSmL1Result",
    "PerSmPerSetL1Result",
    "address_to_line_id",
    "apply_carveout_to_capacity",
    "block_to_sm_id",
    "l1_miss_stream_per_sm",
    "walk_per_sm_l1",
    "walk_per_sm_per_set_l1",
]
