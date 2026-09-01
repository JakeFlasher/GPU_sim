# SM86 L2 dirty-eviction topology milestone

## Goal and completion bar

Add exact, timing-neutral dirty-sector state to the simulator's shared L2 and
expose the victim topology needed by a future below-L2 write service. Complete
the slice when real retained STORE traces report deterministic dirty-eviction
lines/sectors plus terminal dirty residency, with unchanged issue timing,
lookup counters, DRAM-read traffic, event ordering, and typed tails.

This milestone does not add a write bandwidth/rate, schedule a DRAM write
completion, clear `dirty_writeback`/`coherence`, or promote NCU
`dram__sectors_write.sum` as an implementation input. The architecture card's
`write_allocate_non_inclusive` token establishes allocation behavior, not an
independently proven silicon write-back policy; the new state is therefore an
explicit simulator topology hypothesis.

## L2 state and transaction contract

Each private L2 way gains a four-bit `dirty_mask` satisfying:

```text
dirty_mask subset-of valid_mask
dirty_mask disjoint-from pending_mask
```

- A write-class sector access (STORE, RED, or ATOM) makes the sector valid and
  dirty and supersedes a pending read for that sector.
- A read allocation/fill remains clean.
- A fill completion never clears or creates dirty state.
- Snapshot/cold-session copying preserves the exact mask.

Replacement exposes one immutable `L2Eviction` payload containing the victim
line id plus valid, dirty, and pending masks. The existing transactional access
callback observes that payload before mutation. A callback/counter failure must
leave cache state, replacement state, counters, and observer state unchanged.

## Timing-neutral output

Publish a separate architecture-bound component, not lookup counters:

- evicted lines;
- evicted valid sectors;
- evicted pending-read lines/sectors;
- dirty-eviction lines/sectors; and
- terminal resident dirty lines/sectors.

Eviction counters are additive within one launch. Terminal residency is a
shared-cache gauge and must remain per launch/cache session; never sum it across
launches as if it were traffic. No `Ev`, response, horizon, DRAM-server work,
issue timestamp, or completion frontier changes in this slice.

## Retained-trace identification

Use the existing L1TEX admission corpus with exact full traces:

- B84/W8/8-lane unique-line D8, D16, D32, D64, and D128;
- B84/W8/8-lane line-reuse D32; and
- matched control traces.

The unique-line footprints cross the 6 MiB/49,152-line carded L2 boundary:

```text
D8    43,008 lines / 5.25 MiB
D16   86,016 lines / 10.5 MiB
D32  172,032 lines / 21 MiB
D64  344,064 lines / 42 MiB
D128 688,128 lines / 84 MiB
```

The reuse control has the same target STORE instruction/L1TEX sector/LTS
request counts as unique D32 but only 5,376 unique lines (0.656 MiB).

NCU DRAM-write sectors remain an independent comparison target only. They mix
dirty evictions, implicit task-boundary flushes, profiler replay/cache control,
and pre-sweep state, so they cannot identify a writeback latency or prove that
one simulated victim equals one completed DRAM transaction.

## Exact gates

- Cold and resident writes set exactly the requested dirty bits.
- Reads/fills remain clean; a write superseding pending data becomes
  valid-and-dirty without stale fill resurrection.
- Every replacement reports the actual LRU/SRRIP/BIP victim and exact masks.
- Clean replacement produces no dirty writeback work.
- Partially dirty lines report only dirty sectors, not all valid sectors.
- Failed callbacks and overflow checks are fully transactional.
- Default lookup hit/miss/write-tag counters and L2 state order remain equal
  after removing only the new observer.
- Retained replay twins are byte-identical.
- Unique-line cells show monotonic/nondecreasing dirty-eviction opportunity
  across the capacity crossing, while reuse remains structurally separated.
- Terminal resident dirty sectors plus dirty-evicted sectors do not exceed the
  exact write-class sector stream after accounting for repeated overwrites.
- `dirty_writeback=true` and `coherence=true` remain asserted on every STORE
  replay.

## No-go and next boundary

Return a bounded no-go if dirty state cannot remain transactional through
pending-fill replacement, if retained traffic conservation fails, or if the
new observer perturbs timing/lookup behavior. Do not repair a no-go with a
constant, target-derived rate, or tail clearing.

A passing result authorizes a later finite below-L2 write-service experiment.
It does not authorize that service's rate, queue width, terminal flush policy,
or coherence semantics.
