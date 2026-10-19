# Conditional locality and MRC metrics

**Status:** design and validity contract; no MRC is promoted into scored cache
predictions yet.

## Why a new metric is needed

NCU does not emit a capacity-indexed miss-ratio curve, Mattson stack-distance
histogram, or reuse-interval distribution. Its L1/L2 counters are realized
single-operating-point endpoint measurements. Therefore a universal claim that
an inferred MRC has `<10%` APE against NCU is undefined unless the reference
curve is independently generated. The simulator must separate intrinsic
trace-locality accuracy from conditional hardware-endpoint accuracy.

## Metric classes

### Class A: exact trace-locality metrics

These are the primary MRC metrics and have an exact analytical reference that
does not use NCU:

1. **Sector-aware reuse-distance distribution** `RD(line, sector_mask)`:
   distinct intervening 128-byte lines, with four 32-byte sector-validity bits.
   A line reuse is not automatically a sector hit. The line tag is refreshed
   by a touch to any sector, while eviction invalidates all four sector bits;
   the sector curve therefore needs a line-LRU eviction-bottleneck state, not
   an independently tracked sector recency.
2. **Fully-associative LRU MRC** `M_LRU(C) = P(RD >= C)` over a declared
   capacity grid `C`. Cold references and right-censored final references are
   counted explicitly.
3. **Footprint/survival curve** `F(tau)` and `S(C)` derived from the same
   reference stream. Reuse interval/time is retained as a separate diagnostic;
   it is not mislabeled as reuse distance.

The exact reference is a deterministic Mattson/Fenwick order-statistics pass
over a declared stream. A bounded Ding-style histogram or SHARDS sample is
scored against this reference, not against NCU. Report sector-weighted WAPE,
MAE in percentage points, mean/median/P90 error, signed bias, coverage, and
monotonicity violations. The initial promotion bar is `<2%` WAPE/MAE on held-
out trace streams, not a fabricated hardware-accuracy claim.

#### Class-A stream semantics

Each logical tick is a non-empty batch of origin-qualified 128-byte line keys
and four-bit 32-byte sector masks. All queries for a tick occur before any
line is inserted into the recency structure (two-phase batch update); sorting
same-tick lines by address would create artificial distance. A first line or
sector is `cold`; final references are finite observations, while an
unobserved future reuse is right-censored rather than imputed. Capacity is
always expressed in `lines_128B` and sector volumes in `sectors_32B`.

The fully-associative curve is a zero-latency locality reference. It does not
silently include MSHR/pending-fill merges, fill latency, set/slice hashing, or
replacement policy details. Those are reported by the live cache model or by
separate conditional diagnostics. The existing record-gap survival metric is
an RI curve, not a footprint: a footprint maps a window length to expected
working-set size and requires explicit cold-prefix and right-censored terms.

### Class B: conditional effective hardware bounds

These metrics may be compared with NCU only where the endpoint and stream are
well posed. For an explicit observable stratum `S`, report:

* `M_cond(C_min | S)` and `M_cond(C_max | S)` as an interval, not a single
  guessed silicon curve;
* interval coverage and width;
* single-point hit-rate percentage-point error only when the NCU denominator is
  nonzero, the join is exact or explicitly accepted, and the cache client scope
  matches (`global`, `L1TEX`, source-TEX L2, operation-read L2, etc.).

Strata are trace/card properties only: architecture, client/form, full versus
partial coverage, cold versus persistent launch scope, and whether ordering
is exact. They must never be selected from NCU values. Zero-target rows are
reported as exact-zero classification and excluded from relative APE, not
silently discarded.

## Three stream scopes

* **Tier 1:** within-warp trace stream; exact and always available, but not a
  hardware arrival stream.
* **Tier 2:** per-SM cycle-interleaved L1 acceptance stream at the read-only
  `SmCore` issue seam; only valid when the replay exposes a deterministic order.
* **Tier 3:** timestamp/sequence-ordered transactional L2 stream at the
  `access_mask_event_coalesced_transactional` seam, partitioned by static
  `L2ClientScope`; TMA payloads remain separate unless completion linkage is
  present.

Tier-2 and Tier-3 streams are only promoted to hardware-proxy diagnostics when
their ordering contract is deterministic and recorded. Pending-fill merges
must be emitted separately from resident hits and capacity misses.

The live cycle-interleaved L1 and event-driven L2 models remain the source of
timing/cache state. MRC observers are shadow-only until promotion gates pass.

## Algorithm and gates

Phase 1 is a bounded exact/reference implementation: deterministic sector
coalescing, same-tick batch semantics, line-LRU-coupled sector validity,
Fenwick/Mattson RD, cold/censored counters, and a declared capacity grid.
Phase 2 evaluates Ding reuse-interval and SHARDS-style bounded sampling against
the exact reference with a memory/RSS budget. Phase 3 considers conditional
hardware bounds only on held-out, scope-compatible strata.

Every phase requires conservation (`hits + misses == requests` where defined),
coalescing idempotence, line/sector cardinality bounds, monotone MRC, cold and
finite-trace endpoint checks, deterministic replay, and bit-identical existing
IPC/L1/L2/DRAM/timing when the observer is enabled or disabled. No NCU value may
enter fitting, model selection, a lookup table, or a per-workload branch.

## Current validity boundary

The existing `trace_metrics.reuse_interval` observer is a valid Class-A
diagnostic for within-warp record-gap survival, with complete replay invariance
and measured low overhead. It does **not** currently emit exact RD/MRC and must
not be presented as a hardware MRC. Cross-warp/SM chronology, SM120
set/slice/hash/replacement, carveout, MSHR horizons, TMA barrier linkage, and
some cross-surface launch identities remain unidentifiable. If those fields are
not supplied, completion means a validated trace-local MRC plus a documented
conditional-boundary result, not an invented universal `<10%` NCU claim.
