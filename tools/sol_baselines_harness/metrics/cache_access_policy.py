"""L1/L2 cache-access policy router.

Given a Tier-2 compact-record-like access description (op_kind,
address_space, cache_modifier), returns a routing decision that
names whether the access participates in L1 and/or L2, whether it
counts toward the L2 read-hit denominator, and a typed
`result_kind` enum. The router is a pure function over the
inputs; it does NOT mutate any cache state.

Policy is informed by NVIDIA Ampere documentation and Zhang
USENIX'24 empirical findings:

* **L1**: write-through, no-write-allocate. Loads enter L1 unless
  the `.cg` cache modifier requests L1 bypass; stores write
  through to L2 without allocating an L1 entry; atomics bypass
  L1 entirely so coherence is enforced at the L2 layer.
* **L2**: write-back, no-write-allocate, non-inclusive with L1.
  Loads count toward the read tally. Stores DO NOT count toward
  the read-hit denominator (the predictor's accuracy gate uses
  read-only sector hit rate). Atomic RMW counts toward both
  read and write.
* **`.cg` (cache-global) modifier**: skip L1, enter L2.
* **Shared / DSMEM**: scratchpad accesses; route to neither
  L1 nor L2 cache.
* **Local memory** (compiler-spilled per-thread arrays): served
  by L1 + private memory; the L2 read-scoring denominator
  excludes local-memory traffic.
* **Constant memory** (LDC / ULDC / LDCU): bypasses L1 via
  the parallel constant cache; constant-cache misses reach
  L2 architecturally. The L2 read-scoring denominator
  excludes constant-cache traffic because the calibration
  label's LSU-pipe source-unit denominator does not count
  constant-cache traffic on Ampere sm_86 (the constant
  cache is a separate source-unit class).

A record whose op_kind / address_space combination is not
recognized returns `result_kind="unknown"` with
`policy_uncalibrated=True`; downstream callers should treat
such records as typed-unavailable rather than attempting to
score them.

The router is consumed by the simulator integration layer; it
emits decisions, not cache-state side effects. The actual L2
hit/miss tally happens in `set_associative_simulator.access()`,
which calls this router only when the routing decision routes
to L2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


# v1.0 schema op_kind enum (mirrored from the Tier-2 sidecar
# schema). Keep in sync with the producer's emission paths.
OP_KIND_UNKNOWN: int = 0
OP_KIND_GLOBAL_LOAD: int = 1
OP_KIND_GLOBAL_STORE: int = 2
OP_KIND_ATOMIC_LOAD: int = 3
OP_KIND_ATOMIC_STORE: int = 4
OP_KIND_ATOMIC_RMW: int = 5
OP_KIND_CP_ASYNC: int = 6
OP_KIND_TMA: int = 7

# v1.0 schema address_space enum.
ADDRESS_SPACE_GLOBAL: int = 1
ADDRESS_SPACE_SHARED: int = 2
ADDRESS_SPACE_LOCAL: int = 3
ADDRESS_SPACE_DSMEM: int = 4
# Constant memory loads (LDC / LDCU). Used heavily for
# kernel parameter passing via the c[0x0] bank on every
# kernel launch. The constant cache sits parallel to L1
# between the SM and L2; constant-cache misses do reach
# L2 architecturally, but the calibration label's L2
# read denominator is the LSU-pipe source-unit
# denominator and does NOT count constant-cache
# traffic on Ampere sm_86 (the constant cache is a
# separate source-unit class). Constant loads therefore
# route to L2 architecturally but are excluded from the
# L2 read-score producer total so the producer
# denominator matches the label denominator
# class-for-class.
ADDRESS_SPACE_CONSTANT: int = 5

# v1.0 schema cache_modifier enum. `0` is the default (no
# explicit modifier), `2` is `.cg` (cache-global, L1 bypass),
# and `255` is the schema's explicit `unknown` sentinel.
# Unknown modifiers MUST fail closed via the unknown-policy
# helper — they are not silently mapped to default policy.
CACHE_MODIFIER_DEFAULT: int = 0
CACHE_MODIFIER_CG_BYPASS_L1: int = 2
CACHE_MODIFIER_UNKNOWN: int = 255

# ``l1_access_policy`` enum. Carved out of the legacy
# reserved-pad u16 low byte (compact-record wire offset 50)
# to disambiguate the folded Ampere `.cg` case (`STRONG.GPU`
# SASS form, ``cache_modifier=0`` indistinguishable from
# default policy). The routing layer consults this enum AHEAD
# of the legacy ``cache_modifier`` byte for L1 admit/bypass
# decisions on global loads.
L1_ACCESS_POLICY_UNKNOWN: int = 0
L1_ACCESS_POLICY_ADMITTED: int = 1
L1_ACCESS_POLICY_BYPASSED: int = 2

# Typed result_kind vocabulary the router emits. Downstream
# bundle composition keys decisions off these strings.
RESULT_KIND_L1_L2_READ: str = "l1_l2_read"
RESULT_KIND_L1_BYPASS_L2_READ: str = "l1_bypass_l2_read"
RESULT_KIND_L2_READ_ONLY: str = "l2_read_only"
RESULT_KIND_L2_WRITE_ONLY: str = "l2_write_only"
RESULT_KIND_L2_RMW: str = "l2_rmw"
RESULT_KIND_SHARED_MEMORY_ONLY: str = "shared_memory_only"
RESULT_KIND_LOCAL_PRIVATE: str = "local_private"
RESULT_KIND_DSMEM_ONLY: str = "dsmem_only"
RESULT_KIND_CP_ASYNC_SOURCE: str = "cp_async_source"
RESULT_KIND_TMA_GLOBAL: str = "tma_global"
RESULT_KIND_UNKNOWN: str = "unknown"

ALL_RESULT_KINDS: frozenset[str] = frozenset({
    RESULT_KIND_L1_L2_READ,
    RESULT_KIND_L1_BYPASS_L2_READ,
    RESULT_KIND_L2_READ_ONLY,
    RESULT_KIND_L2_WRITE_ONLY,
    RESULT_KIND_L2_RMW,
    RESULT_KIND_SHARED_MEMORY_ONLY,
    RESULT_KIND_LOCAL_PRIVATE,
    RESULT_KIND_DSMEM_ONLY,
    RESULT_KIND_CP_ASYNC_SOURCE,
    RESULT_KIND_TMA_GLOBAL,
    RESULT_KIND_UNKNOWN,
})


@dataclass(frozen=True)
class AccessRouting:
    """Routing decision for one access. `l1_*` and `l2_*` flags
    indicate cache-layer participation. `counts_for_l2_read_score`
    is the gate the L2 hit-rate tally consults — stores and
    local-memory traffic must be excluded per the read-only
    sector hit-rate metric the predictor calibrates against.
    `policy_uncalibrated` flags records the router does not
    recognize; callers should route those to typed-unavailable.
    """

    result_kind: str
    l1_read: bool
    l1_write: bool
    l2_read: bool
    l2_write: bool
    counts_for_l2_read_score: bool
    policy_uncalibrated: bool


def _unknown_policy_routing() -> AccessRouting:
    """Return the canonical fail-closed routing for any access
    the policy table does not recognize: `result_kind=unknown`,
    every L1/L2 participation flag False, excluded from the
    L2 read tally, `policy_uncalibrated=True`. Downstream
    bundle composition should treat the record as
    typed-unavailable.
    """
    return AccessRouting(
        result_kind=RESULT_KIND_UNKNOWN,
        l1_read=False, l1_write=False,
        l2_read=False, l2_write=False,
        counts_for_l2_read_score=False,
        policy_uncalibrated=True,
    )


def route_access(
    *,
    op_kind: int,
    address_space: int,
    cache_modifier: int = CACHE_MODIFIER_DEFAULT,
    l1_access_policy: int = L1_ACCESS_POLICY_UNKNOWN,
    constant_memory_counts_for_l2_read_score: bool = False,
) -> AccessRouting:
    """Resolve an access to its cache-layer routing decision.

    Inputs follow the v1.0 sidecar schema enums (see module
    constants). Returns an `AccessRouting` immutable record.

    Fail-closed policy:

    * `op_kind=OP_KIND_UNKNOWN` (the schema's explicit `0`
      unknown sentinel) returns the unknown-policy result
      regardless of address space.
    * `cache_modifier=CACHE_MODIFIER_UNKNOWN` (the schema's
      explicit `255` unknown sentinel) on a global access
      fails closed before op_kind dispatch.
    * `address_space=ADDRESS_SPACE_LOCAL` admits only
      `OP_KIND_GLOBAL_LOAD` and `OP_KIND_GLOBAL_STORE` (the
      load/store opcodes the compiler emits when spilling).
      Any other op kind on local memory routes to
      unknown-policy.
    * An unrecognized op_kind on a recognized address space
      returns unknown-policy.
    * An unrecognized address space returns unknown-policy.

    Recognized policy table:

      address_space=shared / dsmem: scratchpad; not modeled as
      L1/L2.

      address_space=local + op_kind in {global_load,
      global_store}: L1 + private; excluded from L2 read
      scoring.

      Within global address space:
        op_kind=global_load + cache_modifier=default:
          L1 + L2 read.
        op_kind=global_load + cache_modifier=.cg (bypass L1):
          L1 BYPASS + L2 read.
        op_kind=global_store:
          L2 write only; excluded from L2 read scoring
          (no-write-allocate write-back).
        op_kind=atomic_load:
          L2 read only (atomics bypass L1).
        op_kind=atomic_store:
          L2 write only (RED-family reduction; no return);
          excluded from L2 read scoring.
        op_kind=atomic_rmw:
          L2 read + write (ATOM-family RMW; bypasses L1).
        op_kind=cp_async:
          L1 + L2 read (source-side global access; LDGSTS
          implicitly caches).
        op_kind=tma:
          L2 read only (TMA bypasses L1).
    """
    # API-level fail-closed: ``l1_access_policy`` must lie in
    # the admitted enum range {0=l1_unknown, 1=l1_admitted,
    # 2=l1_bypassed}. Parser-side validation rejects sidecar
    # bytes outside this set, but non-parser callers (test
    # fixtures, programmatic invokers, downstream consumers
    # that build records by hand) could pass arbitrary
    # integer values. An unknown policy value must fail
    # closed at the routing API boundary — the policy-
    # separation discipline forbids silent fall-through to
    # the legacy ``cache_modifier`` default path on
    # unrecognized policy values.
    if l1_access_policy not in (
        L1_ACCESS_POLICY_UNKNOWN,
        L1_ACCESS_POLICY_ADMITTED,
        L1_ACCESS_POLICY_BYPASSED,
    ):
        return _unknown_policy_routing()
    # Schema-explicit `op_kind=0` is the unknown sentinel; it
    # fails closed on every address space.
    if op_kind == OP_KIND_UNKNOWN:
        return _unknown_policy_routing()

    if address_space == ADDRESS_SPACE_SHARED:
        return AccessRouting(
            result_kind=RESULT_KIND_SHARED_MEMORY_ONLY,
            l1_read=False, l1_write=False,
            l2_read=False, l2_write=False,
            counts_for_l2_read_score=False,
            policy_uncalibrated=False,
        )
    if address_space == ADDRESS_SPACE_DSMEM:
        return AccessRouting(
            result_kind=RESULT_KIND_DSMEM_ONLY,
            l1_read=False, l1_write=False,
            l2_read=False, l2_write=False,
            counts_for_l2_read_score=False,
            policy_uncalibrated=False,
        )
    if address_space == ADDRESS_SPACE_CONSTANT:
        # Constant memory loads bypass L1 (the constant
        # cache sits parallel to L1 between the SM and
        # L2). Constant cache misses do reach L2
        # architecturally. Whether they COUNT toward the
        # L2 read-score denominator depends on the arch's
        # label denominator class:
        #
        # * Ampere sm_86 (constant_memory_counts_for_l2_read_score=
        #   False): the LSU-pipe source-unit counter does NOT
        #   include constant-cache routed traffic; constant
        #   loads route to L2 architecturally but are excluded
        #   from the denominator so the simulator's class
        #   matches the label's class.
        #
        # * Blackwell-consumer sm_120 (flag=True): the
        #   texture-source-unit counter on the L2
        #   (lts__t_sectors_srcunit_tex_op_read) INCLUDES the
        #   constant-cache-routed traffic; the simulator must
        #   count it in the denominator to match.
        #
        # The classifier admits only load opcodes on constant
        # memory; stores to constant memory do not exist at
        # the SASS layer, so any non-load op_kind on constant
        # memory fails closed regardless of the flag.
        if op_kind == OP_KIND_GLOBAL_LOAD:
            return AccessRouting(
                result_kind=RESULT_KIND_L2_READ_ONLY,
                l1_read=False, l1_write=False,
                l2_read=True, l2_write=False,
                counts_for_l2_read_score=(
                    constant_memory_counts_for_l2_read_score
                ),
                policy_uncalibrated=False,
            )
        return _unknown_policy_routing()
    if address_space == ADDRESS_SPACE_LOCAL:
        # Local memory is the compiler's per-thread spill
        # surface. The classifier admits only LDL/STL-shaped
        # accesses (load/store op kinds). Atomics, cp.async,
        # TMA, and any unrecognized op kind on local memory
        # fail closed — those have no documented L1/local
        # policy mapping.
        if op_kind == OP_KIND_GLOBAL_LOAD:
            return AccessRouting(
                result_kind=RESULT_KIND_LOCAL_PRIVATE,
                l1_read=True, l1_write=False,
                l2_read=False, l2_write=False,
                counts_for_l2_read_score=False,
                policy_uncalibrated=False,
            )
        if op_kind == OP_KIND_GLOBAL_STORE:
            return AccessRouting(
                result_kind=RESULT_KIND_LOCAL_PRIVATE,
                l1_read=False, l1_write=True,
                l2_read=False, l2_write=False,
                counts_for_l2_read_score=False,
                policy_uncalibrated=False,
            )
        return _unknown_policy_routing()
    if address_space != ADDRESS_SPACE_GLOBAL:
        # Unknown address space.
        return _unknown_policy_routing()

    # Global address space. Fail closed on the unknown cache-
    # modifier sentinel BEFORE dispatching by op_kind — a value
    # the producer flagged as `unknown` must not be silently
    # mapped to default policy.
    if cache_modifier == CACHE_MODIFIER_UNKNOWN:
        return _unknown_policy_routing()

    if op_kind == OP_KIND_GLOBAL_LOAD:
        # Consult ``l1_access_policy`` FIRST. When the producer
        # classified the SASS form (``.CG`` suffix on sm_90+ OR
        # Ampere ``STRONG.GPU`` fold on sm_86), the routing
        # decision is taken from the explicit policy enum
        # regardless of the legacy ``cache_modifier`` byte.
        # This disambiguates the folded-Ampere `.cg` case where
        # ``cache_modifier=0`` indistinguishably encodes both
        # default and folded ``.cg``.
        if l1_access_policy == L1_ACCESS_POLICY_BYPASSED:
            return AccessRouting(
                result_kind=RESULT_KIND_L1_BYPASS_L2_READ,
                l1_read=False, l1_write=False,
                l2_read=True, l2_write=False,
                counts_for_l2_read_score=True,
                policy_uncalibrated=False,
            )
        if l1_access_policy == L1_ACCESS_POLICY_ADMITTED:
            return AccessRouting(
                result_kind=RESULT_KIND_L1_L2_READ,
                l1_read=True, l1_write=False,
                l2_read=True, l2_write=False,
                counts_for_l2_read_score=True,
                policy_uncalibrated=False,
            )
        # ``l1_access_policy=l1_unknown`` (the default for legacy
        # sidecars and for records the producer did not
        # classify): fall through to the legacy
        # ``cache_modifier`` dispatch. Old sidecars stamped
        # cache_modifier=2 for ``.CG`` continue to route to L1
        # bypass; cache_modifier=0 still maps to L1+L2 read
        # under the legacy semantics; any other cache_modifier
        # still fails closed.
        if cache_modifier == CACHE_MODIFIER_CG_BYPASS_L1:
            return AccessRouting(
                result_kind=RESULT_KIND_L1_BYPASS_L2_READ,
                l1_read=False, l1_write=False,
                l2_read=True, l2_write=False,
                counts_for_l2_read_score=True,
                policy_uncalibrated=False,
            )
        if cache_modifier == CACHE_MODIFIER_DEFAULT:
            return AccessRouting(
                result_kind=RESULT_KIND_L1_L2_READ,
                l1_read=True, l1_write=False,
                l2_read=True, l2_write=False,
                counts_for_l2_read_score=True,
                policy_uncalibrated=False,
            )
        # Any other cache_modifier on a global load is
        # unrecognized — fail closed rather than silently
        # mapping to default policy.
        return _unknown_policy_routing()
    if op_kind == OP_KIND_GLOBAL_STORE:
        # Write-through L1 (no allocate) -> write-back L2 (no
        # allocate). The L2 write traffic is recorded but the
        # store is excluded from the L2 read-score denominator.
        return AccessRouting(
            result_kind=RESULT_KIND_L2_WRITE_ONLY,
            l1_read=False, l1_write=True,
            l2_read=False, l2_write=True,
            counts_for_l2_read_score=False,
            policy_uncalibrated=False,
        )
    if op_kind == OP_KIND_ATOMIC_LOAD:
        return AccessRouting(
            result_kind=RESULT_KIND_L2_READ_ONLY,
            l1_read=False, l1_write=False,
            l2_read=True, l2_write=False,
            counts_for_l2_read_score=True,
            policy_uncalibrated=False,
        )
    if op_kind == OP_KIND_ATOMIC_STORE:
        # RED-family reduction: write-only with atomic ordering;
        # excluded from L2 read scoring.
        return AccessRouting(
            result_kind=RESULT_KIND_L2_WRITE_ONLY,
            l1_read=False, l1_write=False,
            l2_read=False, l2_write=True,
            counts_for_l2_read_score=False,
            policy_uncalibrated=False,
        )
    if op_kind == OP_KIND_ATOMIC_RMW:
        # ATOM-family RMW: reads prior value AND writes new
        # value, all under L2-line atomic ordering. Counts
        # toward the L2 read tally because the read is real
        # memory traffic.
        return AccessRouting(
            result_kind=RESULT_KIND_L2_RMW,
            l1_read=False, l1_write=False,
            l2_read=True, l2_write=True,
            counts_for_l2_read_score=True,
            policy_uncalibrated=False,
        )
    if op_kind == OP_KIND_CP_ASYNC:
        return AccessRouting(
            result_kind=RESULT_KIND_CP_ASYNC_SOURCE,
            l1_read=True, l1_write=False,
            l2_read=True, l2_write=False,
            counts_for_l2_read_score=True,
            policy_uncalibrated=False,
        )
    if op_kind == OP_KIND_TMA:
        return AccessRouting(
            result_kind=RESULT_KIND_TMA_GLOBAL,
            l1_read=False, l1_write=False,
            l2_read=True, l2_write=False,
            counts_for_l2_read_score=True,
            policy_uncalibrated=False,
        )
    # Unrecognized op_kind on global address space.
    return _unknown_policy_routing()


def route_record(record: dict[str, Any]) -> AccessRouting:
    """Convenience wrapper that extracts the four policy fields
    from a record-like dict (e.g., a parsed CompactAccessRecord
    or a hand-authored test fixture). Missing keys raise
    KeyError; out-of-range values flow through `route_access`
    and are routed to `result_kind=unknown`.

    When the record carries an ``l1_access_policy`` key it is
    forwarded to `route_access` ahead of ``cache_modifier``.
    Legacy callers that built records without the field default
    to ``L1_ACCESS_POLICY_UNKNOWN`` (0), preserving the
    pre-extension routing semantics.
    """
    return route_access(
        op_kind=int(record["op_kind"]),
        address_space=int(record["address_space"]),
        cache_modifier=int(record.get("cache_modifier", 0)),
        l1_access_policy=int(
            record.get("l1_access_policy", L1_ACCESS_POLICY_UNKNOWN)
        ),
    )


def audit_l2_read_denominator(
    records: Sequence[Any],
) -> dict[tuple[int, int, int], int]:
    """Group routed compact records by the classification
    triple ``(op_kind, address_space, l1_access_policy)`` and
    sum the ``counts_for_l2_read_score`` sector contribution
    per class.

    Returns a dict keyed by ``(op_kind, address_space,
    l1_access_policy)`` where the value is the total number
    of 32-byte sectors the producer routed as
    L2-read-denominator-eligible for that class. Records
    whose routed ``counts_for_l2_read_score`` is False
    contribute 0 to their class total but still appear in
    the result so the audit can document which classes the
    producer DID see (vs. classes that are entirely missing).

    The audit's per-class breakdown lets a downstream
    consumer compare the producer's totals against an
    external L2-read-denominator reference recorded by the
    calibration ledger (the ledger's
    `l2_read_sectors_total` field, exposed on
    ``SmokeCellSpec`` as ``ncu_l2_sectors_total``). A large
    gap means the producer is failing to instrument one or
    more SASS instruction classes that contribute to the
    external L2-read denominator — candidate missing
    classes include constant memory loads (LDC), texture
    loads, surface loads, and instruction-cache misses on
    cold function entry.

    The helper accepts any sequence of records exposing the
    routing-input fields as attributes (``op_kind``,
    ``address_space``, ``cache_modifier``,
    ``l1_access_policy``, and a sector contribution accessor:
    ``sector_count`` or ``sector_ids``). The
    ``CompactAccessRecord`` dataclass from
    ``tier2_sidecar_parser.py`` satisfies this contract.
    """
    totals: dict[tuple[int, int, int], int] = {}
    for record in records:
        op_kind = int(getattr(record, "op_kind", 0))
        address_space = int(getattr(record, "address_space", 0))
        cache_modifier = int(getattr(record, "cache_modifier", 0))
        l1_access_policy = int(
            getattr(record, "l1_access_policy", L1_ACCESS_POLICY_UNKNOWN)
        )
        routing = route_access(
            op_kind=op_kind,
            address_space=address_space,
            cache_modifier=cache_modifier,
            l1_access_policy=l1_access_policy,
        )
        key = (op_kind, address_space, l1_access_policy)
        # Sector contribution: prefer explicit ``sector_count``
        # (the host-emitted compact-record header) and fall
        # back to ``len(sector_ids)`` for the parsed
        # ``CompactAccessRecord`` dataclass.
        sector_count = getattr(record, "sector_count", None)
        if sector_count is None:
            sector_ids = getattr(record, "sector_ids", ())
            sector_count = len(sector_ids)
        sector_count = int(sector_count)
        # Only routes that count toward the L2 read score
        # contribute sectors to the audit total; the
        # remainder appear as zero-sector classes so the
        # audit documents what the producer saw at each
        # classification triple.
        contribution = (
            sector_count if routing.counts_for_l2_read_score else 0
        )
        totals[key] = totals.get(key, 0) + contribution
    return totals


__all__ = [
    "AccessRouting",
    "OP_KIND_UNKNOWN",
    "OP_KIND_GLOBAL_LOAD",
    "OP_KIND_GLOBAL_STORE",
    "OP_KIND_ATOMIC_LOAD",
    "OP_KIND_ATOMIC_STORE",
    "OP_KIND_ATOMIC_RMW",
    "OP_KIND_CP_ASYNC",
    "OP_KIND_TMA",
    "ADDRESS_SPACE_GLOBAL",
    "ADDRESS_SPACE_SHARED",
    "ADDRESS_SPACE_LOCAL",
    "ADDRESS_SPACE_DSMEM",
    "ADDRESS_SPACE_CONSTANT",
    "CACHE_MODIFIER_DEFAULT",
    "CACHE_MODIFIER_CG_BYPASS_L1",
    "CACHE_MODIFIER_UNKNOWN",
    "RESULT_KIND_L1_L2_READ",
    "RESULT_KIND_L1_BYPASS_L2_READ",
    "RESULT_KIND_L2_READ_ONLY",
    "RESULT_KIND_L2_WRITE_ONLY",
    "RESULT_KIND_L2_RMW",
    "RESULT_KIND_SHARED_MEMORY_ONLY",
    "RESULT_KIND_LOCAL_PRIVATE",
    "RESULT_KIND_DSMEM_ONLY",
    "RESULT_KIND_CP_ASYNC_SOURCE",
    "RESULT_KIND_TMA_GLOBAL",
    "RESULT_KIND_UNKNOWN",
    "ALL_RESULT_KINDS",
    "L1_ACCESS_POLICY_UNKNOWN",
    "L1_ACCESS_POLICY_ADMITTED",
    "L1_ACCESS_POLICY_BYPASSED",
    "route_access",
    "route_record",
    "audit_l2_read_denominator",
]
