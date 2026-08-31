# SM86 asynchronous-copy source-line ownership discriminator

## Context and current evidence

The multi-line response-envelope milestone is complete and accepted by
Humanize RLCR at commit `5fa761fd`. Its scalar `.cg` preload creates a genuine
179-cycle response envelope inside a split flexible cohort, with zero
prearrival hiding and saturated schema-18 stream hiding. It nevertheless stops
as `complete_response_envelope_structural_no_go`: the frozen plan required 20
destination wavefront observations per copy, while the 32-lane probe produces
32 payload/bypass wavefront observations and 20 unique destination lines.

Observation: the raw NVBit trace already contains every lane's source and
destination addresses. Trace projection retains source-sector-to-cohort masks
and aggregate cohort topology, and replay retains each cohort's first/last
response. It discards which physical source line owns which subset of the
cohort's destination topology.

Hypothesis: if two variants keep source requests, preload requests, response
times, destination address multisets, and every existing aggregate cohort
coordinate exact but swap whether the early or late source line owns the
larger destination fanout, a repeat-stable live wait delta identifies this
missing line-to-destination ownership coordinate. Existing schema 18 must
predict an exact tie.

This is a fresh prospective assay. No wait delta, score, or fit from
`async_multiline_response` or its excluded round-1 full capture may be read or
imported.

## Goal

Build and run a bounded SM86 live-GPU/NVBit/replay discriminator that swaps
source-line ownership inside split LDGSTS cohorts while preserving the actual
32-payload-wavefront/20-destination-line interface. Produce one reproducible
structural, identification, holdout, or material-signal decision without
changing simulator timing or adding a line-local schema.

## Goals and non-goals

### Goals

- Freeze payload wavefronts, trace-projected bypass wavefronts, destination
  lines, cohort topology, source/preload work, and launch shape before timing.
- Compare `early_fanout - late_fanout` on a first-line-early identification
  orientation and a physically reversed second-line-early holdout.
- Prove that current schema-18 observations and production fingerprints tie
  exactly between paired variants.
- Admit live wait labels only after static, NCU, NVBit, replay, identity, and
  issue-control gates pass.
- Retain truthful per-invocation manifests and a tracked aggregate report from
  the initial smoke onward.

### Non-goals

- No schema 19, line-local timing candidate, fitted coefficient, intercept,
  scale, or production wait change.
- No reuse of the consumed/excluded `async_multiline_response` timing labels.
- No depth, lead, spacing, multi-warp, multi-CTA, or occupancy sweep.
- No raw device-wide DRAM equality claim on the display-active host.
- No JADE/HETR, CPU-GPU co-simulation, `sm_120`, or display reconfiguration.
- No workflow gate or cryptographic authority. Content hashes are ordinary
  reproducibility metadata.

## Interfaces and compatibility

### Source and preload geometry

Add a new `async_source_line_ownership` probe. One warp issues exactly one
LDGSTS copy per group with a fixed retained lead of 16 clock samples. The
source step is 2 KiB and contains eight four-lane/64-byte flexible cohorts:

```text
aligned cohort bases: 0, 256, 512, 768
split cohort bases:   1088, 1344, 1600, 1856
lane offsets:         0, 32, 64, 96
```

Aligned cohorts occupy one source line. Split cohorts occupy two source lines.
Initialize the first 128 bytes of every 256-byte source slot to `0x31` and the
second 128 bytes to `0x57`, so both LDGSTS and preload correctness remain
line-distinguishable.

Before LDGSTS, issue one predicated scalar `ld.global.cg.u32` per group:

- all 16 aligned-cohort lanes are active; and
- for split cohorts, either lane-in-cohort `0/1` is active in
  `first_line_early`, or lane-in-cohort `2/3` is active in
  `second_line_early`.

The scalar result is first consumed after the post-`wait_group 0` clock. Every
copy therefore has one scalar LDG instruction, 24 preload sectors on eight
lines, one LDGSTS instruction, 32 LDGSTS sectors on 12 unique source lines,
and 56 local-device LTS lookup sectors.

### Destination ownership swap

Each source cohort owns a separate 512-byte destination slot.

Aligned controls always use lane offsets:

```text
0, 32, 128, 160
```

This is two destination lines, two bank wavefronts, and four projected bypass
wavefronts per aligned cohort.

Split cohorts always use the fixed destination multiset:

```text
fanout pair: 0, 128
compact pair: 256, 288
```

This is three destination lines, three bank wavefronts, and four projected
bypass wavefronts per split cohort. `early_fanout` maps the currently preloaded
split source-line lanes to `0/128` and the late lanes to `256/288`.
`late_fanout` swaps those exact address pairs. For the second-line-early
orientation, the early predicate and semantic ownership both reverse; the
meaning of the variant names does not.

Both variants must have the same per-copy aggregate interface:

```text
active payload transfers / destination_wavefronts_observed = 32
trace-projected bypass wavefront total                     = 32
destination lines                                          = 20
source cohorts                                             = 8
source sectors / response memberships                      = 32
maximum cohort bank wavefronts                             = 3
maximum cohort bypass wavefronts                           = 4
cohort topology penalty total                              = 120 cycles
```

The source vector, scalar-preload vector, destination address multiset, static
LDGSTS positions, and all aggregate topology coordinates must be exactly equal
within each pair. The reducer must validate the per-lane ownership permutation
from the fresh trace rather than infer it from variant names.

### Current simulator prediction

Aligned controls are identical. Each split cohort has the same early/late
response times and the same aggregate topology in both variants. Schema 18
must therefore report exact paired equality for response envelopes,
prearrival hiding, stream hiding, residual exposure, completion/frontier
fields, counters, and typed tails. Its paired residual prediction is zero.

Any admitted nonzero live effect is evidence for missing source-line ownership,
not permission to alter timing in this milestone.

### Paired NCU contract

For each paired observation, run both variants in one application invocation
with alternating AB/BA order and the accepted strict application-replay
configuration. Collect these seven metrics separately; every report must have
one replay pass and exactly two ordered target rows:

```text
smsp__inst_executed_op_global_ld.sum
smsp__inst_executed_op_ldgsts.sum
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum
lts__t_sectors_srcunit_tex_aperture_device_op_read.sum
lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum
lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum
dram__sectors_read.sum
```

At 64 groups, each row must report 64 scalar LDGs, 64 LDGSTS instructions,
2,048 shared payload wavefronts, and 3,584 local-device LTS lookups. Hit and
miss populations must be nonzero; their separate-profile closure and paired
deltas must not exceed 36 sectors. Raw DRAM remains recorded and non-gating.

## Implementation slices

### Slice 1: New probe and smoke-only gate

Add the CUDA probe, Makefile target, focused capture path, reducer, and concise
documentation. Reuse the accepted execution-manifest and static-binding
helpers rather than duplicating a new provenance framework.

Build the two early-orientation specializations. Require zero spill/local
memory, exact opcode counts, no preload-barrier consumption before the
post-wait clock, and the only result store after measured intervals.

Run a `first_line_early` smoke containing one correct direct AB pair, one
seven-report paired NCU bundle, two endpoint NVBit traces, and two replay twins
per endpoint. The smoke reducer must emit an explicit pass or a complete
structural no-go. Missing, malformed, stale, mixed-tracer, multi-pass, or
manifest-incomplete evidence remains nonzero.

### Slice 2: Frozen two-orientation campaign

Only after the smoke passes, capture:

- direct: ten paired 8192-group repeats for each orientation, odd AB/even BA;
- NCU: ten paired 64-group, seven-report bundles for each orientation (20
  bundles, 140 reports, 280 ordered rows);
- NVBit: both variants for both orientations (four traces); and
- simulator: two schema-18 replays per trace (eight outputs).

Retain all commands, configured timeouts, start/finish timestamps, return and
timeout status, stream identities, binary/tool/input/output identities, GPU
identity, raw reports, traces, and replay outputs under:

```text
results/captures/sm86_async_line_ownership_20260831_v1/
```

Publish a compact aggregate at:

```text
data/local_labels/sm86_async_line_ownership_report.json
```

### Slice 3: Frozen mechanism decision

Admission order is static → NCU → NVBit → replay → direct issue → direct wait.
The identification cell is `first_line_early`. Its line-ownership signal passes
only if:

- median paired issue delta is exactly zero;
- median `early_fanout - late_fanout` wait delta is at most -1.0 cycle/group;
  and
- at least eight of ten repeat deltas are negative.

If identification fails, emit
`complete_source_line_ownership_identification_no_go` and do not read the
second-orientation timing fields. If identification passes, evaluate the
untouched `second_line_early` holdout under the same bounds. Emit either
`complete_source_line_ownership_signal` or
`complete_source_line_ownership_holdout_no_go`.

There are no fitted parameters. A signal authorizes only a later prospective
schema-19 observation plan; it cannot change readiness or tail state here.

### Slice 4: Documentation and handoff

Document the exact mechanism, admitted evidence, outcome, claim limits, and
next simulator seam. Update `docs/STATUS.md`, the live-baseline closure plan,
and the probe README. Keep the excluded multi-line round-1 matrix inert and
unread.

## Data or state migration

There is no trace schema, simulator output schema, or retained-data migration.
The new binary, evidence root, and tracked aggregate are additive. Existing
schema-18 captures and the accepted structural-no-go report remain unchanged.

## Validation and evidence

The RLCR candidate commit must attach:

- `rust-workspace-tests`: full Rust workspace library tests.
- `line-ownership-static`: source/tool-bound SM86 static audit and run-manifest
  validation for both early-orientation specializations.
- `line-ownership-smoke`: smoke-only structural/attribution/replay gate.
- `line-ownership-live-campaign`: exact paired NCU and direct campaign
  validation, or a completed smoke structural no-go.
- `line-ownership-replay`: exact per-lane ownership, aggregate topology,
  deterministic twins, and schema-18 paired-tie validation.
- `line-ownership-analysis`: attribution-first frozen identification/holdout
  decision and tracked aggregate report.

The analyzer exits zero only for an explicit smoke pass followed by a complete
decision, or a completed scientific no-go. Tooling, identity, manifest, row,
trace, or replay failures remain nonzero.

## Risks and rollback

- Ptxas may consume the preload result too early. Static evidence rejects the
  probe instead of interpreting a completed load as pending.
- Aggregate topology equality does not prove lane ownership. The reducer must
  inspect source/destination lane vectors in each fresh trace.
- The live effect may be absent or orientation-sensitive. Either is a useful
  no-go and does not justify fitting.
- Application replay collects metric families in separate executions; exact
  LTS total is authoritative while hit/miss closure stays bounded.
- Rollback removes only the additive probe/campaign/report and returns to
  accepted commit `5fa761fd`.

## Acceptance criteria

- **AC-1 — Exact prospective interface:** The new probe literally preserves
  32 payload wavefront observations, 32 projected bypass wavefronts, 20
  destination lines, exact source/preload work, and the per-lane ownership
  permutation with clean static scheduling.
- **AC-2 — Executable early gate:** The first-line-early smoke explicitly
  passes every structural/attribution/replay gate or records a complete
  scientific structural no-go before the full campaign.
- **AC-3 — Fresh two-orientation evidence:** If AC-2 passes, all 20 paired NCU
  bundles, 20 direct pairs, four traces, and eight replay outputs are complete,
  identity-bound, deterministic, and exact on declared invariants.
- **AC-4 — Frozen ownership decision:** The reducer evaluates first-line
  identification before second-line holdout and emits exactly one complete
  signal or no-go status without fitting or reading predecessor labels.
- **AC-5 — Timing-neutral handoff:** The tracked report, documentation,
  commits, evidence, and clean worktree reproduce the GPU-simulator-only
  result while production timing and `async_copy_shared_visibility` remain
  unchanged.
