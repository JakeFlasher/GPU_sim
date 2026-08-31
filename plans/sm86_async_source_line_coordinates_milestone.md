# SM86 source-line ownership coordinate-corrected campaign

## Context and current evidence

The source-line ownership smoke is complete and accepted by Humanize RLCR at
commit `57f6c5ff`. Its per-lane source, preload, and destination ownership
permutations are exact, and current schema 18 plus production fingerprints tie
between `early_fanout` and `late_fanout`. It stops before timing because its
prospective plan conflated four different coordinates.

The accepted smoke independently observes, per copy:

```text
NCU shared LDGSTS wavefronts          = 20
simulator destination_wavefronts      = 24
trace bypass-wavefront projection     = 32
unique destination lines              = 20
```

At 64 groups those are 1,280, 1,536, 2,048, and 1,280 respectively. These are
separate interfaces and must never be collapsed, relabeled, or compared as if
they were one count.

The ownership probe's smoke direct result was provenance-validated but its
issue and wait fields were not read. This milestone uses a newly named
executable, evidence root, assay ID, and tracked aggregate. No result or report
under either predecessor line-ownership root may be opened or imported.

## Goal

Run the complete two-orientation source-line ownership discriminator under the
correct four-coordinate contract and produce a frozen identification/holdout
signal or no-go. The result remains observational: no simulator schema or
production timing change is in scope.

## Goals and non-goals

### Goals

- Reuse the already accepted CUDA mechanism without changing its lane mapping.
- Compile it as a fresh executable named
  `async_source_line_ownership_coordinates` with a distinct static identity.
- Freeze NCU shared wavefronts 20, simulator destination wavefronts 24, bypass
  projection 32, and destination lines 20 as separately named invariants.
- Run a mandatory first-line-early smoke, then the complete two-orientation
  matrix only after an identity-matched smoke-pass aggregate authorizes it.
- Preserve strict admission order and explicit issue/wait label-read state.
- Decide the line-ownership signal without fitting magnitude or reading any
  predecessor timing result.

### Non-goals

- No CUDA geometry, source/preload vector, destination ownership, lead, depth,
  or launch-shape change.
- No schema 19, per-line replay observation, readiness change, tail removal,
  coefficient, intercept, scale, or label-derived threshold.
- No import, validation, regression check, or filesystem read beneath prior
  line-ownership result roots.
- No JADE/HETR, CPU-GPU co-simulation, `sm_120`, multi-warp, multi-CTA, or
  display reconfiguration.
- No procedural authority attached to hashes or manifests.

## Interfaces and compatibility

### Fresh generation identity

Compile the unchanged
`workloads/research_probes/async_source_only/async_source_line_ownership.cu`
through a new isolated `Makefile.line_ownership_coordinates` into:

```text
workloads/research_probes/async_source_only/async_source_line_ownership_coordinates
```

Use only these new evidence paths:

```text
results/captures/sm86_async_line_coordinates_20260831_v1/
data/local_labels/sm86_async_line_coordinates_report.json
```

The executable basename, static binary identity, evidence root, report assay
ID `sm86_async_source_line_ownership_coordinates_v1`, and aggregate root field
are the machine-checkable generation marker.

### Closed campaign contract

Extract the reusable hard-coded campaign identity into one frozen immutable
contract used by the execution producer, NCU producer, and analyzer. The
contract contains only identity and structural expectations:

- assay ID, binary, source, isolated Makefile, evidence root, aggregate path;
- NCU shared wavefronts per copy = 20;
- simulator destination wavefronts per copy = 24;
- bypass wavefronts per copy = 32;
- destination lines per copy = 20; and
- fixed source/cohort/topology counts already established below.

Signal thresholds remain source constants, not CLI/configuration fields. A
one-shot configuration function must reject mutation after campaign work
starts. Existing modules retain their prior-generation defaults, and focused
tests must prove default and coordinate-corrected contracts remain distinct.

Thin entrypoint scripts may configure the new immutable contract and delegate
to the existing implementation. They must not patch values opportunistically
after validation begins or fall back to predecessor paths.

### Frozen probe and topology

The CUDA mechanism remains exactly:

- one warp, depth 1, 16 retained clock samples;
- four one-line aligned source cohorts and four two-line split cohorts;
- one scalar `.cg` load preloading all aligned sectors plus the selected split
  source line;
- aligned destination offsets `0/32/128/160`;
- split destination multiset `0/128/256/288`; and
- paired `early_fanout`/`late_fanout` ownership permutation.

At 64 groups, exact trace/replay values are:

```text
copies_observed                         = 64
source_cohorts_observed                 = 512
cohort_response_memberships_observed    = 2,048
destination_wavefronts_observed         = 1,536
bypass_wavefronts_observed              = 2,048
destination_lines_observed              = 1,280
max_bank_wavefronts_sum                 = 192
max_bypass_wavefronts_sum               = 256
cohort_topology_penalty_cycles          = 7,680
cohort_overlap_groups_observed          = 64
```

The reducer must verify both the LDGSTS lane ownership permutation and the
scalar preload active-lane/source vector for both orientations. Within each
orientation, complete schema-18 observations and production fingerprints must
tie exactly between variants. Cross-orientation equality is not required.

### NCU contract

Each paired bundle retains seven independent one-pass reports with two ordered
rows:

```text
smsp__inst_executed_op_global_ld.sum                                  = 64
smsp__inst_executed_op_ldgsts.sum                                     = 64
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum      = 1,280
lts__t_sectors_srcunit_tex_aperture_device_op_read.sum                = 3,584
lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum     > 0
lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum    > 0
dram__sectors_read.sum                                                = recorded only
```

Hit/miss closure and paired hit/miss deltas must be at most 36 sectors. Raw
DRAM remains non-gating.

### Label isolation and decision

The analyzer admission order is:

```text
static identity/control
→ NCU bundle contract
→ NVBit payload/sidecar identities and per-lane mappings
→ replay twins, exact coordinates, schema-18/production ties
→ direct provenance for all retained repeats
→ identification issue fields
→ identification wait fields
→ holdout issue fields
→ holdout wait fields
```

JSON/result files may be hashed for provenance without parsing them. Direct
issue deltas are derived from arm counters and checked against retained
top-level values. Wait deltas are derived and checked only after the relevant
issue gate passes.

The tracked report records separately:

```text
identification_issue_read
identification_wait_read
holdout_issue_read
holdout_wait_read
```

The first-line-early identification passes only when median issue delta is
exactly zero, median `early_fanout - late_fanout` wait delta is at most -1.0
cycle/group, and at least eight of ten repeat deltas are negative. If it fails,
emit `complete_line_coordinates_identification_no_go` and keep both holdout
timing flags false.

If identification passes, apply the identical frozen rule to the
second-line-early holdout. Emit either
`complete_line_coordinates_signal` or
`complete_line_coordinates_holdout_no_go`. No magnitude fit or equality across
orientations is allowed.

## Implementation slices

### Slice 1: Contract parameterization and fresh smoke

Add the immutable contract and three thin coordinate-generation entrypoints
for execution capture, NCU capture, and analysis. Preserve existing defaults
and tracked predecessor reports byte-for-byte under synthetic/default-contract
tests that do not open predecessor result roots.

Build the fresh executable and run only the first-line-early smoke: one direct
AB artifact, one seven-report paired NCU bundle, two NVBit endpoints, and replay
twins. Publish a smoke-pass aggregate bound to the fresh root/binary before any
full command can run.

### Slice 2: Full fresh campaign

After the smoke authorization passes, capture:

- 10 paired 8192-group direct repeats per orientation, odd AB/even BA;
- 10 paired 64-group NCU bundles per orientation (20 bundles, 140 reports,
  280 ordered rows);
- four NVBit endpoint traces; and
- two replay outputs per trace (eight outputs).

Every command retains timeout/timestamp/status and exact executable/input/
output identities. NVBit payload manifests cover all threadblock protobufs and
consumed sidecars and are replay inputs.

### Slice 3: Frozen identification and holdout

Run the complete structural admission before opening identification issue
fields. Apply the decision above. Provenance-validate all holdout artifacts,
but do not parse holdout result JSON unless identification passes.

No outcome authorizes timing. A material two-orientation signal authorizes
only a later prospective schema-19 plan.

### Slice 4: Documentation and handoff

Add a tracked aggregate and concise milestone document. Update
`docs/STATUS.md`, the live-baseline closure plan, and the async-source README.
Report all four coordinate values separately, explicit label-read flags, and
the next simulator action only if one remains.

## Data or state migration

There is no simulator or trace schema migration. The fresh executable,
isolated Makefile, root, and report are additive. Prior accepted reports and
raw roots remain inert and unread.

## Validation and evidence

The RLCR candidate must attach:

- `rust-workspace-tests`: full Rust workspace library tests.
- `line-coordinates-contract`: immutable default/fresh contract tests,
  predecessor-root no-read checks, static build/control, and smoke
  authorization.
- `line-coordinates-smoke`: fresh smoke exact on all four coordinates.
- `line-coordinates-live-campaign`: 20 direct artifacts and 20 seven-family
  NCU bundles, or a completed smoke structural no-go.
- `line-coordinates-replay`: four per-lane traces, payload identities, replay
  twins, exact coordinate gates, and pairwise schema/production ties.
- `line-coordinates-analysis`: frozen identification/holdout decision, label-
  read flags, and tracked aggregate.

Missing, malformed, stale, predecessor-root, mixed-tracer, manifest-invalid,
multi-pass, coordinate-mismatched, or inconsistent-delta evidence is nonzero.

## Risks and rollback

- Contract parameterization can accidentally drift predecessor defaults.
  Synthetic contract tests and static source/build checks must catch this
  without reading predecessor results.
- A renamed executable is not sufficient alone; every manifest and tracked
  report must bind the new binary/root/assay identity.
- The live signal may fail identification. That is a complete no-go, not a
  reason to change the sign or magnitude threshold.
- Rollback removes only the additive coordinate-generation wrappers, Makefile,
  evidence, and report, returning to accepted commit `57f6c5ff`.

## Acceptance criteria

- **AC-1 — Exact four-coordinate smoke:** The fresh executable passes the
  mandatory smoke with NCU 20, simulator destination 24, bypass 32, and
  destination lines 20 per copy, plus exact per-lane/preload mappings.
- **AC-2 — Fresh identity and isolation:** New binary/root/assay/report
  identities are enforced, full capture is smoke-authorized, and no predecessor
  result path is opened.
- **AC-3 — Complete campaign:** If the smoke passes, all 20 NCU bundles, 20
  direct artifacts, four traces, and eight replay outputs are manifest-valid,
  deterministic, and exact on their separately named coordinates.
- **AC-4 — Frozen signal decision:** Identification precedes holdout; arm
  counters determine deltas; label-read flags are truthful; exactly one signal
  or no-go status is emitted without fitting.
- **AC-5 — Timing-neutral handoff:** Tracked report, documentation, commits,
  evidence, and a clean worktree reproduce the result while simulator timing,
  schemas, and `async_copy_shared_visibility` remain unchanged.
