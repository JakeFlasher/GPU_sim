# SM86 asynchronous-copy multi-line response-envelope milestone

## Context and current evidence

The schema-18 cohort reservation-exposure milestone is complete and was
accepted by Humanize RLCR at commit `b935c2e7`. Schema 18 retains exact
source-sector/cohort membership, first and last modeled response times, and a
timing-neutral first-response-anchored topology-reservation score. It does not
change asynchronous-copy completion, `DEPBAR.LE SB0` readiness, destination
service, traffic, scheduler state, or `async_copy_shared_visibility`.

Observation: the retained cohort-pairing traces put every cohort in one
physical 128-byte L2 line. Their within-cohort response spans are therefore
zero. The schema-18 score has the correct live sign on its frozen
identification cells but fails the frozen magnitude bounds (9.828
cycles/group MAE and 12.259 maximum error), so the holdout remains unread and
production timing remains unchanged.

Observation: ordinary cold sectors issued by one LDGSTS instruction all see
the same modeled 423-cycle DRAM floor. Four or eight sectors cannot create a
response span through the carded bandwidth server alone.

Observation: the current shared-L2 event path tracks pending sectors by
allocation generation and sector identity. A later response-bearing LDGSTS
request that matches an older scalar `.cg` read reuses the older ready time;
fresh sectors on a second line retain the later LDGSTS ready time. With the
current SM86 cards, the available pending-response lead is bounded by roughly
`423 - 244 = 179` cycles.

Hypothesis: if all sectors of one-line cohorts and only the first line of
two-line cohorts are requested before LDGSTS, every cohort has the same early
first response while only the two-line cohorts retain a later response. Pairing
the two-line cohorts with severe destinations should then expose schema 18's
within-cohort stream-hiding term without a compensating prearrival term.

## Goal

Build and run a fresh SM86 live-GPU/NVBit/simulator assay that deliberately
creates independently completing source lines inside one flexible LDGSTS
cohort. Establish either:

1. a bounded identification and untouched-depth transfer of the existing
   schema-18 residual score; or
2. a reproducible structural, identification, or holdout no-go.

Both outcomes are complete only when the evidence is reproducible and the
simulator's production timing and typed visibility tail remain unchanged.

## Goals and non-goals

### Goals

- Exercise nonzero `cohort_response_envelope_cycles` and
  `cohort_topology_stream_hidden_cycles` through real pending-line merges.
- Preserve exact source work, preload work, destination marginals, kernel
  shape, and launch order between paired variants.
- Validate source work with static SASS, fresh NVBit traces, and
  source-attributed local-device LTS counters before reading live timing.
- Compare the parameter-free schema-18 paired score with live paired wait
  deltas on a frozen depth/lead split.
- Stop early when a static, response-envelope, or NCU attribution gate proves
  the assay structurally invalid.

### Non-goals

- No schema 19, fitted coefficient, intercept, scale, per-layout constant, or
  live-timing-derived tuning.
- No change to production completion, wait readiness, destination service,
  scheduler state, traffic, cards, or typed-tail selection.
- No claim that device-wide raw DRAM counters isolate SM traffic on this
  display-active host.
- No depth 4/8, inter-copy spacing sweep, multi-CTA occupancy sweep,
  JADE/HETR, CPU-GPU co-simulation, `sm_120`, or display reconfiguration.
- No process gate, approval hash, or provenance authority. Content identities
  are ordinary reproducibility metadata only.

## Interfaces and compatibility

### Probe geometry

Add `async_multiline_response` beside the retained async-source probes. One
warp issues depths `1` and `2`; every copy uses a separate 2 KiB source step.
Within each step, four aligned and four split source cohorts use these bases:

```text
aligned: 0, 256, 512, 768
split:   1088, 1344, 1600, 1856
lane offsets within each cohort: 0, 32, 64, 96
```

This produces exactly eight independent four-lane/64-byte flexible cohorts,
32 one-hot 32-byte source sectors, and 12 unique 128-byte L2 lines per copy.
The maximum exclusive address is 1968, leaving 80 bytes in the step.

Before each copy's LDGSTS, issue one predicated scalar `ld.global.cg.u32`:

- lanes 0..15 request all 16 sectors of the four aligned cohorts;
- lanes 16..23 request the first-line eight sectors of the four split cohorts;
- lanes 24..31 are inactive with a zero-initialized result.

Thus every copy has exactly one scalar LDG instruction, 24 preload sectors,
and eight preload lines. Retain every scalar result and make its first consumer
occur only after the post-`wait_group 0` clock sample. Static control evidence
must prove that no ordinary preload barrier is consumed before that clock.

After all scalar preloads for a group, execute a compile-time retained
clock-sample/XOR lead with sample counts `0/4/8/16`. Then sample
`before_issue`, issue contiguous LDGSTS copies, commit, wait, and sample
`after_wait`. The only global result store remains after every measured
interval.

The destination mapping retains four compact and four severe cohorts per
copy. `span_expensive` pairs split source cohorts with severe destinations;
`span_cheap` swaps source-to-destination ownership by XOR 4. Both variants use
the exact same relative LDGSTS source vector, scalar-preload vector, and
destination multiset.

Use two independently initialized, alignment-equivalent source arms inside
one allocation for paired AB/BA invocations. Initialize the first 1024 bytes
of every source step to `0x31` and the second 1024 bytes to `0x57`. Retain a
64-group unmeasured warmup and a no-wrap bound for 8192 measured groups.

### Exact work contract

For `G` groups at depth `D`:

```text
scalar LDG warp instructions = G * D
scalar LDG sectors           = 24 * G * D
scalar preload lines         = 8 * G * D
LDGSTS warp instructions     = G * D
LDGSTS sectors               = 32 * G * D
unique LDGSTS lines          = 12 * G * D
destination wavefronts       = 20 * G * D
local-device LTS lookups     = 56 * G * D
```

At the 64-group depth-1 smoke this is 64 scalar LDGs, 1536 scalar
sectors, 64 LDGSTS instructions, 2048 LDGSTS sectors, and 3584 attributable
LTS lookups.

### Schema-18 prediction

Let `S_i` be the replayed split-cohort response span for copy `i`. All cohort
first responses are equally early, so prearrival hiding must be zero. Compact
cohorts have topology penalty zero; severe cohorts have penalty 18. Therefore:

```text
span_expensive residual per copy = 18 - min(18, S_i)
span_cheap residual per copy     = 18
paired group score               = -min_i(min(18, S_i))
```

The score is read directly from the existing aggregate schema-18 residual
diagnostic; no parameter is fit. For the lead-16 smoke, every copy must span at
least 18 and no more than 179 modeled cycles, so the paired score must be
exactly -18 cycles/group. `span_expensive` must report four severe cohorts'
stream hiding per copy, `span_cheap` must report zero stream hiding, and both
must report zero prearrival hiding.

### Paired NCU evidence

Reuse the accepted same-application application-replay configuration:

```text
--devices 0
--target-processes application-only
--profile-from-start off
--launch-count 2
--replay-mode application
--app-replay-mode strict
--app-replay-match grid
--cache-control none
--clock-control none
```

Each bundle collects these six metrics separately so every report must contain
one replay pass and exactly two ordered target rows:

```text
smsp__inst_executed_op_global_ld.sum
smsp__inst_executed_op_ldgsts.sum
lts__t_sectors_srcunit_tex_aperture_device_op_read.sum
lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum
lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum
dram__sectors_read.sum
```

The exact LTS total is the attribution frontier. Separately replayed hit and
miss populations must both be nonzero, their sum may differ from total by at
most `max(4, ceil(0.01 * total))`, and paired hit/miss deltas use the same
tolerance. Raw DRAM and `raw_dram - attributed_lts_misses` remain recorded,
non-gating background sentinels.

## Implementation slices

### Slice 1: Probe and mandatory vertical smoke

Add the CUDA probe, Makefile target, static audit, capture/replay helper, and
reducer. Build all depth `1/2` by lead `0/4/8/16` specializations for SM86.
Reject spills, local-memory traffic, an early preload-result consumer, or any
ordinary dependency wait that consumes preload barriers before `after_wait`.
The async `DEPBAR.LE SB0, 0x0` remains expected and must not be confused with
an ordinary preload wait.

Before a campaign, run one depth-1/lead-16 vertical smoke:

1. one correct direct AB pair;
2. one endpoint NVBit trace for each variant plus a fresh complete static
   sidecar;
3. two byte-identical schema-18 replays per endpoint; and
4. one six-report paired NCU bundle at 64 groups.

Require exact work/topology, no unsupported or cross-allocation trace records,
zero prearrival hiding, a response envelope in `[18, 179]` cycles/copy,
saturated expensive stream hiding, residuals `0` versus `18` per group, paired
score -18, exact NCU instruction/LTS totals, and bounded hit/miss reconciliation.

If a scientific gate fails, emit a machine-readable
`complete_response_envelope_structural_no_go` and stop before the full matrix.
A compiler error, capture failure, timeout, malformed trace/report, stale
binary, or missing row is not a scientific no-go and must be fixed and rerun.

### Slice 2: Frozen live-GPU and trace campaign

Only after the smoke passes, capture the full matrix at depths `1/2` and lead
samples `0/4/8/16`:

- direct: ten paired 8192-group repeats per cell, odd AB and even BA;
- NCU: ten paired 64-group bundles per cell, odd AB and even BA, six one-pass
  metric reports per bundle (80 bundles, 480 reports, 960 ordered rows);
- NVBit: both endpoint variants for every cell (16 traces), each at 64 groups;
  and
- simulator: two independent schema-18 replays for every trace (32 outputs).

Retain commands, stdout/stderr, exit status, timeout, order, arguments, exact
device identity, binary/static-sidecar identity, raw profiler reports, traces,
replay outputs, and salient counts under
`results/captures/sm86_async_multiline_response_20260831_v1/`.

Every direct run must be correct and sum-consistent. Every cell's median paired
issue delta must be exactly zero before its wait delta is admitted. Every NCU
row and NVBit trace must satisfy the exact work contract. Lead-dependent
replayed response envelopes must be monotone nondecreasing, and lead 16 must
remain saturated for every group/copy.

### Slice 3: Parameter-free identification and holdout

Apply NCU/static/NVBit/replay admission before reading direct wait labels. The
frozen split is:

- identification: all four depth-1 lead cells;
- holdout: all four depth-2 lead cells.

For each cell compare the median live
`span_expensive - span_cheap` wait delta with the schema-18 paired residual
score. The existing accuracy bounds remain 1.0 cycle/group MAE, 2.0
cycle/group maximum absolute error, and correct sign for every nonzero
prediction. No coefficient or selection rule may be derived from the live
labels.

If identification fails, emit `complete_response_envelope_identification_no_go`
without reading holdout timing. If it passes, evaluate the untouched holdout
and emit either `complete_response_envelope_transfer` or
`complete_response_envelope_holdout_no_go`. None of these outcomes changes
production timing in this milestone.

### Slice 4: Documentation and handoff

Write one machine-readable report and concise milestone document. Update
`docs/STATUS.md`, the live-baseline closure plan, and the async-source README.
State observations, inferences, unresolved uncertainty, timing/tail status, and
the next simulator action only if one remains.

## Data or state migration

There is no schema or retained-data migration. Existing schema-17/18 traces,
campaigns, cards, and reports remain immutable. The new evidence root and its
new probe are additive. Rollback is deletion of the additive probe/campaign
code and return to accepted commit `b935c2e7`; no existing capture is rewritten.

## Validation and evidence

The RLCR candidate commit must attach these deterministic checks:

- `rust-workspace-tests`: full Rust workspace library tests.
- `response-envelope-static`: SM86 build/static/control audit for every
  depth/lead specialization, including exact opcode placement, no spills/local
  memory, and no pre-`after_wait` preload dependency consumption.
- `response-envelope-smoke`: mandatory depth-1/lead-16 direct, NVBit,
  schema-18 twin, and paired-NCU vertical gate; success may represent the
  explicitly completed structural no-go.
- `live-gpu-campaign`: validates either the completed structural no-go or the
  complete frozen direct/NCU/NVBit matrix without invoking hidden defaults.
- `replay-determinism`: validates exact topology and byte-identical replay
  twins for every admitted trace.
- `response-envelope-analysis`: validates attribution-first ordering, the
  frozen identification/holdout decision, and unchanged production timing and
  typed-tail state.

The analysis command exits successfully for a complete transfer or one of the
three complete no-go outcomes. Incomplete evidence, tooling failure, violated
identity, malformed output, or an unclassified result remains nonzero.

## Risks and rollback

- Ptxas may consume a scalar-load barrier early, especially at depth 2. The
  static gate treats that as a structural assay no-go rather than silently
  measuring a completed preload.
- A lead may complete the preload before LDGSTS. Lead 16 must remain below the
  modeled 179-cycle pending window; an L2 hit instead of a pending merge
  invalidates the mechanism.
- Separate NCU metric profiles do not share one cache history. The explicit LTS
  total is exact; hit/miss populations remain bounded, separately replayed
  attribution evidence.
- Raw DRAM is shared with asynchronous display/copy/video engines and remains
  non-gating.
- A numerical transfer supports the schema-18 mechanism only on this frozen
  assay. It does not by itself authorize production wait timing.
- Rollback leaves schema 18 and `async_copy_shared_visibility` exactly as they
  are at `b935c2e7`.

## Acceptance criteria

- **AC-1 — Pending-line probe integrity:** The depth `1/2`, lead
  `0/4/8/16` probe builds for SM86 with exact source/preload/destination work,
  no spills or local memory, and no preload-result dependency consumed before
  the post-wait clock.
- **AC-2 — Executable response-envelope gate:** The depth-1/lead-16 vertical
  smoke either proves exact nonzero pending-line stream hiding and paired score
  -18 with admitted source work, or records a reproducible structural no-go
  before the full matrix.
- **AC-3 — Fresh live and replay evidence:** If AC-2 passes, the frozen
  direct/NCU/NVBit/replay matrix is complete, correctly ordered, attributable,
  deterministic, and exact on all declared work/topology invariants.
- **AC-4 — Frozen transfer decision:** Attribution precedes live labels and
  the parameter-free depth-1 identification/depth-2 holdout produces exactly
  one complete transfer or no-go status under the frozen bounds.
- **AC-5 — Timing-neutral handoff:** Reports, commands, identities,
  documentation, commits, required RLCR evidence, and a clean worktree provide
  a reproducible GPU-simulator-only handoff; production timing and
  `async_copy_shared_visibility` remain unchanged.
