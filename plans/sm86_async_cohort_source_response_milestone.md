# SM86 asynchronous-copy cohort source-response milestone

## Context and current evidence

The schema-16 issue-spacing milestone is complete at commit `7b1241f2`.
Its 52-cell live matrix, 12 NCU endpoints, 22 NVBit traces, and deterministic
replays reject a per-copy destination pipeline driven only by each copy's
maximum source-completion time. All `Q={1,2,4,8,16}` candidates tie at 3.008
cycles/group calibration MAE, so no Q is identified and no timing is active.

The retained destination-bearing traces still contain the lane-paired global
source and shared destination operands. The trace reader currently collapses
those operands to copy-wide cohort counts and topology sums/maxima. The memory
response path already preserves exact `(instruction, sector_index)` identity,
so the missing join is trace-side source-sector membership to cohort-local
destination topology, not a new reply-token protocol.

The existing issue-spacing cells have uniform topology among the eight source
cohorts of each copy. A direct additive cohort penalty is already lower-bound
falsified: the severe calibration cells are about 13.93--14.05 cycles/group,
below their frozen 18-cycle local penalty. Implementing another Q sweep over
that recurrence would not add evidence. A fresh discriminator must instead
hold source and destination marginals fixed while changing which slow source
cohorts are paired with expensive destination cohorts.

## Goal

Retain exact SM86 LDGSTS source-sector/cohort response ordering in cold replay
state, evaluate a parameter-free topology/source-overlap candidate without
changing production timing, and run a matched-marginal live GPU pairing assay
that can causally accept or reject that mechanism. Promote a bypass timing rule
only if it passes the predeclared live and replay transfer gates; otherwise
retain schema-17 observation, the typed visibility tail, and a completed no-go.

## Non-goals

- No JADE/HETR or CPU-GPU co-simulation work.
- No `sm_120` behavior or card changes.
- No retuning schema-15 coefficients, residual correction, fitted intercept,
  layout-specific constant, or renewed schema-16 Q search.
- No use of NCU measurements as simulator inputs.
- No production timing or tail removal based only on the retained uniform-
  cohort spacing traces.
- No broad tracer/schema rewrite; the existing destination-bearing NVBit
  payload is sufficient.

## Interfaces and compatibility

### Exact trace projection

Extend the LDGSTS destination projection in `sim/trace/src/reader.rs` with a
sparse warp-local representation of each flexible 128-byte source cohort:

- exact active payload bytes;
- destination 128-byte line fanout;
- maximum 32-bank/4-byte-word multiplicity;
- cohort-local bypass wavefront projection; and
- one `u32` cohort-membership mask for each compact source sector, aligned to
  the record's exact `ValidatedSectorPlan::Sectors` order.

A sector may intersect two flexible cohorts, so membership is a bitmask rather
than a single ordinal. Aggregate `LdgstsDestinationTopology` values must be
recomposed exactly from the retained cohort records. Source-only legacy traces
remain an explicit no-cohort fallback; predicated-off records retain empty
spans. Do not place a variable cohort vector in the 32-bit
`GlobalAsyncCopy::completion_work` token.

Thread sparse spans/arenas through replay without enlarging the hot
`ReplayInst` unless evidence shows no simpler representation. `ResponseId` and
`ReplyTo` remain unchanged.

### Cold SM observation

For each bypass copy, retain copy-wide schema-16 compatibility fields plus
cohort-local cold state:

```text
pending source sectors
first source-response time
last source-response time
exact payload work
destination lines
bank wavefronts
bypass wavefronts
```

One source response decrements the production copy/group counter once and
updates every cohort selected by its membership mask. Validate masks,
cardinalities, aggregate topology, and pending counts transactionally before
mutation. Out-of-order replies and one sector completing two cohorts must be
well-defined and deterministic.

### Timing-neutral candidate

Freeze the existing topology coefficients and existing architecture-local
bypass payload conversion. For cohort `c`:

```text
payload_ready_c = ldgsts_completion_ready(last_response_c,
                                           exact_payload_work_c)
P_c = 4 * (bypass_wavefronts_c - 1)
    + 2 * (bank_wavefronts_c - 1)
topology_ready_c = first_response_c + P_c
finish_c = max(payload_ready_c, topology_ready_c)
group_candidate = max_c(finish_c)
```

This hypothesis allows cohort topology work to overlap the source-response
envelope. It has no fitted parameter and no shared-server/Q term. Schema 17
reports cohort counts, response spans, candidate completion/exposure relative
to the wait's `current_since`, and earlier/later comparison with existing
frontiers. Before transfer is proven, it must not mutate
`AsyncCopyGroup::completion_time`, destination-pipe readiness, scheduling,
traffic, cards, or the typed tail.

## Implementation slices

### Slice 1: Exact cohort/source ownership

Implement the trace projection, compact sparse arenas, and replay handoff.
Exercise a boundary-sector case with multi-cohort membership, heterogeneous
cohort topology, ordinary uniform cohorts, predicated-off input, and the
source-only fallback. Require per-cohort recomposition to equal the established
copy-wide topology exactly.

### Slice 2: Schema-17 response observation

Populate cold per-copy cohort state at bypass issue and update it from exact
response identities. Implement the parameter-free overlap candidate and
schema-17 diagnostics. Replay the retained 22 issue-spacing traces twice.
Require exact response accounting and deterministic output. Normalize away
only schema-17 observation fields and restore schema 16; the result must equal
the predecessor output for production timing, traffic, terminal modeled event
time, completion frontier, and tails.

Reduce the existing live comparisons with the frozen split:

- structural calibration: depth-2 conflict/severe, spacing 1 and 4;
- retained transfer: depth-2 conflict/severe at spacing 0/16, all depth-2
  skew at spacing 0/1/4/16, and depth-8 skew at spacing 0/1/4.

This reduction may establish a no-go but cannot authorize timing because the
cohort topology is uniform within each copy.

### Slice 3: Matched-marginal cohort-pairing probe

Add one store-free, barrier-free, one-warp probe beside the existing async
source probes. Each LDGSTS copy contains eight flexible 128-byte source
cohorts, four source sectors and 64 payload bytes per cohort:

- four cohorts use a small prewarmed L2-resident source footprint;
- four cohorts advance through a cold source ring larger than retained L2;
- four destination cohorts use compact topology;
- four destination cohorts use severe topology.

The two treatment variants preserve the exact source address vector,
destination address multiset, payload, static LDGSTS positions, resources, and
copy count. They differ only in pairing:

- `slow_expensive`: cold source cohorts receive severe destinations;
- `slow_cheap`: cold source cohorts receive compact destinations.

Use three clock frontiers (pre-issue, post-last-issue, post-wait) and place the
only result store after all measured intervals. Run both variants as a paired
AB/BA measurement with independent cold-ring regions and explicit prewarm so
process-level scheduler pauses cancel without cross-arm cache reuse.

Freeze depths `1/2/4/8`, spacing samples `0/4`, nine direct process repeats,
and 8,192 measured groups after a fixed warmup. This is 8 matched comparisons,
not 16 independently interpreted cells. Capture one NCU profile and one NVBit
trace per variant/depth/spacing endpoint at 64 groups. A smaller smoke may run
first, but the frozen matrix is the completion evidence.

Static and NCU admission requires:

- exact LDGSTS count and source-sector count between paired variants;
- exact destination-wavefront total between paired variants;
- both L2-hit and L2-miss/DRAM source populations are nonzero;
- for each L1/L2 hit, miss, and DRAM sector metric, the paired absolute
  difference is no more than
  `max(4, ceil(0.01 * expected_source_sectors))`;
- paired median issue time differs by no more than 0.5 cycle/group and at
  least eight of nine repeats differ by no more than 1.0 cycle/group; and
- no source-work, topology, store, barrier, or placement drift.

If these conditions cannot be achieved, record the first concrete invalidating
signal and finish as an invalid/no-go assay rather than relabeling routes.

### Slice 4: Transfer decision and optional activation

The primary live target is paired
`slow_expensive - slow_cheap` post-issue wait cycles per group. The existing
copy-wide and aggregate models predict zero because their marginals are equal.
Schema 17 predicts the delta from cohort response/topology pairing.

Use depth 1/2, spacing 0 only as the identification slice. Hold out depth 4/8
at spacing 0 and every depth at spacing 4 until the implementation and analysis
are frozen. No parameter is fitted. A material signal requires at least eight
of nine paired repeats with the predicted sign and an absolute median of at
least 1.0 cycle/group on both identification depths. Transfer requires:

- identification and holdout MAE no greater than 1.0 cycle/group;
- maximum absolute error no greater than 2.0 cycles/group in each split;
- correct sign on every admitted comparison;
- all structural/NCU invariants; and
- no regression after schema-17 normalization.

Only if every gate passes may the exact parameter-free cohort overlap frontier
be connected to bypass wait readiness and the typed visibility tail be
reconsidered. Activation must then replay the full retained spacing slice and
the predecessor async-source regression slice without traffic changes. If any
gate fails, leave production timing and the tail unchanged and document the
strongest falsified mechanism plus one next discriminating check.

### Slice 5: Retained evidence and handoff

Add one reducer that validates the static, direct, NCU, NVBit, replay,
normalization, and identification/holdout contracts and emits JSON plus a short
Markdown report. Update `docs/STATUS.md`, the live-baseline closure plan, the
probe README, and a dedicated milestone document. Commit coherent vertical
slices and finish with a clean worktree.

## Validation and evidence

The RLCR run must attach these same-commit deterministic checks:

- `rust-workspace-tests`: full Rust workspace library tests.
- `live-gpu-campaign`: the retained cohort-pairing reducer confirms the frozen
  direct/NCU/NVBit matrix and its structural admission gates.
- `replay-determinism`: every required schema-17 trace has byte-identical twin
  replays and normalizes to its schema-16 predecessor.
- `cohort-response-analysis`: the reducer reports the frozen
  identification/holdout decision and verifies timing/tail promotion state.

Also run focused reader/SM tests and a bounded live smoke as soon as their
paths are runnable. Retain commands, device identity, workload arguments,
cards, trace/control inputs, and salient metrics in the evidence root.

## Risks and rollback

- Flexible cohorts are not cache-line aligned and sectors may overlap cohorts;
  a one-sector/one-cohort assumption is invalid.
- The matched live source mix may be disturbed by NCU replay or cache control;
  counter admission, not intent, names the mix.
- A flat pairing result is informative and must not be converted into a fitted
  constant.
- Schema-17 state remains boxed/cold. If the candidate fails, keep exact
  observation if it is deterministic and compatible; do not retain an active
  timing path.
- Existing production timing is the rollback point. No destructive data or
  unrelated branch changes are in scope.

## Acceptance criteria

- **AC-1 — Exact cohort identity:** Trace and replay retain exact cohort
  descriptors plus multi-membership source-sector masks, and recomposition
  equals the established copy-wide topology for all supported inputs.
- **AC-2 — Timing-neutral schema 17:** Exact per-cohort response state and the
  parameter-free overlap candidate replay deterministically on all 22 retained
  traces without changing production output beyond schema-17 observations.
- **AC-3 — Valid live pairing assay:** The frozen 8-comparison, nine-repeat
  live matrix plus NCU and NVBit evidence completes on the expected SM86 GPU
  and passes matched-marginal admission, or records a concrete invalid/no-go
  result without relabeling.
- **AC-4 — Evidence-backed decision:** The reducer applies the frozen
  identification/holdout gates and either validates a transferable candidate
  or retains production timing and the typed tail with a completed no-go.
- **AC-5 — Reproducible handoff:** Tests, twin replays, reports, documentation,
  content identities, and commits are complete; the worktree is clean and no
  JADE/HETR, CPU-GPU, or SM120 surface changed.
