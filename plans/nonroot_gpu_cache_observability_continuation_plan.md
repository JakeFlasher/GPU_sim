# Continuation Plan — Counter-Free GPU Cache Observability (sm_120 Loop) — v4 (post-DEC, converged)

## Goal Description

Continue the original
`plans/nonroot_gpu_cache_observability_plan.md` toward
completion via **cross-arch development on the RTX 5060
Laptop (sm_120)**. Loop 2 is the second-arch leg of the
cross-arch development model documented in CLAUDE.md
"Two-Device Local Development Model": the prior 3090 Ti
session earned the sm_86 evidence baseline; this loop earns
the matching sm_120 evidence and offline foundation work.

Loop 2 explicitly:
- Stays on the local-device cross-arch axis (sm_86 +
  sm_120) per CLAUDE.md Rules 1-5.
- Defers server work entirely (out of project scope until
  the codebase passes strict accuracy/coverage gates on
  BOTH local devices via cross-arch verification).
- Records 3090 Ti-blocked items (NCU profiling, NVBit/CUPTI
  rebuild, sm_86 live witnesses) in detail; closure of these
  items is deferred to a future 3090 Ti session, NOT this
  loop.
- Drops the RLCR Stop-hook resilience proposal (not in the
  original plan; out of scope per user decision).

CLAUDE.md Rules 1-5 remain binding. The Two-Device Local
Development Model evidence statuses (`Offline-verified`,
`Verified-live-sm_86`, `Verified-live-sm_120`,
`Cross-arch-verified`) gate every claim of completion. Loop
2 deliverables can earn at most `Verified-live-sm_120` in
this loop; `Cross-arch-verified` upgrades happen when the
sm_86 follow-up runs in a future 3090 Ti session.

Naming convention: continuation-specific criteria use
prefix `C-AC-` (e.g. `C-AC-A`); references to the original
plan's criteria keep their unprefixed `AC-N` form unchanged.

## Acceptance Criteria

### Track A — 3090 Ti issue ledger (record-only; closure deferred to a future sm_86 session)

- C-AC-A: 3090 Ti issue ledger.
  - Positive Tests: every sm_86-blocked item discovered or
    carried into Loop 2 is recorded with sufficient detail in
    `.humanize/rlcr/<session>/blocked-on-3090ti.md` (path
    chosen during execution) so a future 3090 Ti operator
    can resume work without ambiguity. The ledger entry for
    each item includes: (a) item name (e.g. task7,
    task11b-task11d, sm_86 NCU profiling rebuild, sm_86 NVBit
    tracer rebuild); (b) what is blocked; (c) the precise
    operator-facing artifact (runbook path, command shape,
    expected outcome); (d) any sm_120-side prerequisites
    completed in Loop 2 that the future sm_86 work depends
    on; (e) the most recent sm_120-side evidence captured
    that pairs with the deferred sm_86 work.
  - Negative Tests: an sm_86 issue discovered during a Loop
    2 round but not added to the ledger fails a
    goal-tracker hygiene check at round end. A ledger entry
    that names a runbook path that does not resolve at
    `5060_dev_loop2` HEAD fails the same check.

### Track B — Cross-cutting hardening (offline; sm_120-verifiable)

- C-AC-B: Operator-runbook source-of-truth + command-verification
  policy (closes BL-20260504-runbook-self-reference and
  BL-20260504-runbook-command-verification adoption).
  - Positive Tests: every operator runbook under `tools/`
    follows the hybrid pattern from
    `BL-20260504-runbook-self-reference` (read procedure
    from branch HEAD via `git show <branch>:<path>`;
    verification target SHA may be pinned and is explicitly
    distinguished from the source-of-truth SHA). Every
    documented repo CLI invocation shape (e.g. invocations
    of `tools/local_labels/label_driver.sh`,
    `make -C tools/nvbit_tracer/tracer_tool ...`,
    `pytest <path>`) is exercised by a fake-* integration
    test (positive + negative-control) per
    `BL-20260504-runbook-command-verification`. Pure
    environment probes (`nvidia-smi`, `nvcc --version`,
    `ncu --version`) are covered by lighter static checks
    (presence + expected stdout pattern), not full fake
    integration.
  - Negative Tests: an operator runbook that pins a
    source-of-truth `git checkout` SHA without explicit
    verification-target separation fails a runbook-policy
    pytest case. A documented repo CLI shape that fails the
    live argument guard fails its corresponding integration
    test.

### Track C — Toolchain hygiene (offline; sm_120-verifiable)

- C-AC-C: Per-taxonomy-family AC-19 lockstep enforcement
  (advances original-plan AC-19).
  - Positive Tests:
    `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`
    is extended so that every typed-reason taxonomy family
    has a defined surface inventory (the existing op-class
    enum chain has 5 surfaces: proto wire enum + producer
    C++ enum + consumer Python admit list + JSON schema
    enum + C++↔proto static_assert; the trace-budget family
    has 4 surfaces: constants + writer + discovery + schema;
    the local-labels family has 1 surface: producer-only).
    Every typed reason newly introduced in this loop
    declares its taxonomy family at introduction; lockstep
    is enforced WITHIN that family — the scanner asserts
    presence on every surface declared for the family. The
    scanner's surface-inventory map is a top-level constant
    in the test file.
  - Negative Tests: a typed reason added to the producer
    C++ enum but not to the proto wire enum + Python admit
    list + JSON schema + static_assert (op-class family)
    fails the parity test. A typed reason added to the
    producer-only local-labels family does NOT require the
    other surfaces; the scanner correctly does not require
    them.

- C-AC-D: Scoped plan-token scanner (closes original-plan
  AC-20).
  - Positive Tests: a pytest case under `tests/` greps
    tracked files (via `git ls-files`) restricted to
    implementation-language extensions
    (`*.py`/`*.sh`/`*.cu`/`*.cuh`/`*.cpp`/`*.h`/`*.proto`/`*.json`/`*.toml`/`*.mk`)
    plus all `*test*.py` for forbidden plan-token strings
    (`AC-`, `Milestone`, `Phase` (with surrounding word
    boundary), `Step` (with surrounding word boundary),
    `Round N`). The scanner uses generated temp fixtures or
    split strings in its test data so the test file itself
    does not contain forbidden tokens that would spuriously
    match. The exemption table is hard-coded in the test
    and excludes: `tools/**/*.md` (operator markdown),
    `plans/**` (plan documents), `references/**` (vendored
    reference material), `.humanize/**` (gitignored loop
    state), generated artifacts, and the test file itself.
  - Negative Tests: a `.py` source file containing
    `# AC-5 sector stream...` fails the scanner. A markdown
    file under `tools/cloud/` does NOT fail (correctly
    exempt). A new violation under `tools/local_labels/*.py`
    is caught. The scanner test file's own diagnostic
    strings (constructed via fixtures or split strings) do
    not spuriously fail it.

- C-AC-E: Architecture-capability metadata + portability
  layer (closes original-plan AC-17).
  - Positive Tests: a single
    `tools/arch_capabilities.json` enumerates EVERY
    project-owned collection script (`tools/local_labels/`,
    `tools/nvbit_tracer/scripts/`, future feature
    extractor directories), feature extractor module, and
    operator entry point
    (`tools/local_labels/label_driver.sh`, etc.). Each entry
    declares
    `{path: "...", supported_arches: [...], notes: "..."}`.
    The portability-coverage pytest case discovers entry
    points via repository scan (Python `def main()`, shell
    `#!/usr/bin/env bash` files NOT under exclusion paths,
    Makefile `.PHONY` collection targets) and asserts each
    discovered entry has a declaration. Scripts on
    unsupported architectures emit typed `arch_unsupported`.
    The metadata explicitly distinguishes `sm_86` (RTX 3090
    Ti Ampere desktop) from `sm_120-laptop` (RTX 5060 Laptop
    Blackwell); server architectures (sm_80, sm_89, sm_90a,
    server sm_120) are NOT included in any
    `supported_arches` list because server work is out of
    Loop 2 scope.
  - Negative Tests: a new collection / extraction script
    introduced WITHOUT a declaration in
    `arch_capabilities.json` fails the coverage test.
    Explicit exclusions (tests under `tests/`, generated
    artifacts under `data/`, vendored code under
    `references/`, docs-only scripts, `__pycache__/`) are
    NOT flagged. A script that returns a default value
    silently on an unsupported architecture (rather than
    emitting `arch_unsupported`) is caught by a
    typed-exclusion regression test. Adding a server
    architecture (`sm_80`, `sm_89`, `sm_90a`, server
    `sm_120`) to any `supported_arches` list during Loop 2
    fails a scope-guard pytest case.

### Track D — Offline foundation (sm_120 implementation; live evidence earnable on the 5060 laptop in this loop)

> **Cross-arch discipline** (per CLAUDE.md): every Track D
> deliverable in Loop 2 earns at most
> `Verified-live-sm_120` in this loop. The matching sm_86
> follow-up earns `Cross-arch-verified` in a future 3090 Ti
> session (see Track A ledger).

- C-AC-F: Sector stream generator covers original-plan AC-5,
  scoped to `global` and `local` memory classes; remaining
  classes typed-excluded with explicit reasons.
  - Positive Tests: deterministic generator under a new
    implementation directory (path chosen during execution,
    e.g. `tools/sector_stream/`) maps
    `(lane_addresses, active_mask, predicate_mask,
    data_width)` to `(touched_32B_sectors,
    touched_128B_lines, sectors_per_request, classification
    ∈ {uniform, broadcast, overlap, strided, scattered})`
    for `global` and `local` classes. Synthetic-trace unit
    tests cover the 8 patterns from the original AC-5
    positive tests PLUS vector-width sector-span fixtures
    (8B / 16B vector loads crossing 32B and 128B boundaries
    per lane). Output schema lives at
    `workloads/simbench_bw/schemas/sector_stream.schema.json`.
    Live evidence: at least one sm_120 live trace from
    `workloads/simbench_bw/` is consumed end-to-end and
    produces sector-stream output that admits via the schema.
  - Negative Tests: a record with effective participation
    zero (i.e., `(active_mask & predicate_mask) == 0`)
    produces typed `no_lanes_active`; a record with
    `lane_addresses` length != 32 produces typed
    `lane_address_length_invalid`; a record with
    `data_width` not in {1, 2, 4, 8, 16} is rejected; a
    record with `memory_op_class ∈ {shared, texture,
    surface, atomic, tma, async_copy, cluster_dsmem}`
    produces typed `sector_stream_class_unsupported` (these
    classes are deferred to a later loop's per-class
    handling).

- C-AC-G: Memory-divergence + MSHR-pressure proxies covers
  original-plan AC-9 with reviewer fixes applied.
  - Positive Tests: 8 proxies emitted per kernel per
    launch; each proxy declares its required trace fields
    in a manifest section of the output schema at
    `workloads/simbench_bw/schemas/divergence_proxies.schema.json`.
    Proxies that need evidence not present in the trace
    emit typed unavailable reasons (e.g.
    `outstanding_load_density_unavailable_no_phase_records`,
    `stride_entropy_unavailable_no_lane_table`) instead of
    zero defaults. `outstanding_load_density_by_phase` is
    annotated in the schema description as an
    approximation bound to phase markers, NOT true MSHR
    occupancy. Live evidence: at least one sm_120 live
    trace produces a proxy bundle that admits via the
    schema.
  - Negative Tests: as in original AC-9 positive/negative
    tests with the typed-unavailable axis added — a proxy
    whose computation requires evidence the trace did not
    capture must emit a typed unavailable reason, never a
    silent zero default.

- C-AC-H: NVBit admissibility budgets completion (closes
  original-plan AC-13).
  - Positive Tests: the two new typed reasons
    (`nvbit_slowdown_budget_exceeded`,
    `nvbit_drop_rate_exceeded`) are declared as members of
    the **trace-budget taxonomy family** in C-AC-C's
    surface-inventory map. Lockstep is enforced at exactly
    the trace-budget family's four surfaces (constants in
    `tools/cloud/trace_budgets.py` + writer literal in
    `workloads/simbench_bw/run_suite.py` + validator
    discovery + JSON schema). The verifier
    (`workloads/simbench_bw/verify_artifacts.py` or
    sister) reads the run summary and rejects artifacts
    violating the budgets. Producer-side drop counter
    source is explicitly enumerated in the schema
    description for `nvbit_drop_rate_exceeded`: NVBit
    channel overflow + ring-buffer loss + decoder gaps +
    expected-record mismatch. Live evidence: at least one
    sm_120 live sweep produces (or correctly does not
    produce, when budgets are not exceeded) the new typed
    sidecars.
  - Negative Tests: as in original AC-13 (a run that
    silently truncates without the typed-exclusion sidecar
    fails the verifier). A budget update that lands on
    only some of the trace-budget family's four surfaces
    fails C-AC-C's parity scanner. Producer drop counters
    that add a new source without schema update fail the
    parity scanner.

### Track E — Original-plan AC-5+ extensions (Upper Bound; sm_120 implementation only; cross-arch follow-up deferred)

> Track E items extend Loop 2's reach into the original
> plan's AC-5+ space. Each item earns at most
> `Verified-live-sm_120` in Loop 2; sm_86 cross-arch
> follow-up is recorded in Track A's ledger.

The Upper Bound carries the following original-plan ACs in
priority order:

- AC-6 HRD profiler + cache-model spec (sm_120
  implementation; CUDA attributes vs static device-card
  override distinction preserved).
- AC-7 sharing-locality features grounded vs projection
  (sm_120 implementation).
- AC-8 schedule-marginalized order projections (sm_120
  implementation; explicit ordering provenance per
  projection).
- AC-10 cache-tomography perturbation harness (sm_120
  implementation; SASS modifier confirmation + correctness-
  equivalence check per perturbation binary).
- AC-11 adaptive NVBit sampling (sm_120 implementation;
  estimator + MRC quantile set + distance metric specified
  before implementation).
- AC-14 microbenchmark suite (sm_120 implementation; per-
  arch SASS slices verified by existing
  `verify_kernel_contracts.py`-style discipline).
- AC-15 combined dual-workload validation (sm_120 only;
  manifest-pairing index by `(slug, shape_point,
  binary_sha256, source_commit, kernel_signature_sha256)`).
- AC-16 NVML covariate sampler **local-device portion only**
  (the sm_120 laptop reads NVML covariates per the
  read-only-NVML rule; server-architecture portions are
  out of Loop 2 scope per AC-12 deferral).
- AC-18 workload READMEs only (the cloud `RUNBOOK.md` is
  deferred to a post-server-readiness loop).

Each Track E item carried into Loop 2 is itself an
acceptance criterion; the original AC numbering stays
unchanged. The Lower Bound does NOT include Track E.

### Out of scope for Loop 2 (deferred; tracked but not closed)

- task7 sm_86 local-label hardware acceptance (3090 Ti
  session required) — recorded in Track A ledger.
- task11b-task11d sm_86 NVBit tracer cross-arch (3090 Ti
  session required) — recorded in Track A ledger.
- AC-12 server collection bootstrap — deferred until both
  local devices pass strict accuracy/coverage gates via
  cross-arch verification.
- AC-16 server-architecture NVML portions (local-device
  NVML is in Track E Upper Bound only).
- AC-18 cloud `tools/cloud/RUNBOOK.md` — deferred to a
  post-server-readiness loop.
- RLCR Stop-hook resilience (not in original plan; out of
  scope per DEC-3).
- Event-based async-free retirement for stream-ordered
  allocator hooks (Loop 1 R6 review queued this; Codex's
  directive #4 anti-displacement; recorded as queued
  non-blocking item, NOT closed in Loop 2).

## Path Boundaries

### Upper Bound
Loop 2 delivers Tracks A + B + C + D AND the Track E
Upper Bound carries (AC-6, AC-7, AC-8, AC-10, AC-11, AC-14,
AC-15, AC-16 local-device portion, AC-18 workload READMEs)
with both `Offline-verified` and `Verified-live-sm_120`
evidence. Every existing AND new operator runbook complies
with the C-AC-B runbook policy. The arch capability
metadata covers every project-owned collection /
extraction / operator entry point. Loop 2 produces
COMPLETE sm_120 evidence for every deliverable; the
matching sm_86 evidence is recorded in the Track A ledger
for future closure.

### Lower Bound
Loop 2 delivers Tracks A + B + C + D only. Track E items
are explicitly queued for a later loop. Track A records
sm_86-blocked items in detail. Track B lands the
cross-cutting runbook policy on every existing runbook.
Track C lands AC-19 lockstep + scoped AC-20 scanner + arch
capabilities metadata across project-owned entry points.
Track D lands C-AC-F (sector stream global+local) +
C-AC-G (divergence proxies with typed unavailable) +
C-AC-H (budget completion). All Track D items earn
`Offline-verified` after their offline tests pass,
independent of Track A; they additionally earn
`Verified-live-sm_120` after live sm_120 evidence is
captured.

### Allowed Choices
- Can use: NVBit dynamic SASS/address tracing; CUPTI
  Activity; CUDA events; Nsight Systems for user-space
  timeline; static SASS via `cuobjdump` / `nvdisasm`;
  read-only NVML; local NCU ONLY into `data/local_labels/`
  via `tools/local_labels/` on RTX 3090 Ti sm_86 + RTX
  5060 laptop sm_120.
- Can use for C-AC-E metadata: a single
  `tools/arch_capabilities.json` (chosen) plus
  language-specific test discovery; per-script
  `supported_arches: list[str]` constants ARE NOT
  preferred (a central manifest reduces drift).
- Cannot use: NCU on cloud / server GPUs (CLAUDE.md Rule
  1); `nvidia-smi` mutation flags (Rule 2); privileged
  CUPTI surfaces (Events / Metrics / Range Profiling / PM
  Sampling / PerfWorks) (Rule 3); kernel-module
  operations / driver install / DKMS / MOK /
  `modprobe nvidia` / reboot workflows (Rule 4); simulator
  revival (`nvasim`, Accel-Sim, simulator-validation
  plans) (Rule 5).
- Cannot use: any `supported_arches` declaration that
  includes server architectures (sm_80, sm_89, sm_90a,
  server sm_120) during Loop 2 — server work is out of
  scope until the codebase passes strict local
  accuracy/coverage gates via cross-arch verification.

## Feasibility Hints and Suggestions

### Conceptual Approach
- **Branch policy**: Loop 2 runs on a fresh branch forked
  from `5060_dev` HEAD (path chosen during execution, e.g.
  `5060_dev_loop2`). Hardware operators on the 3090 Ti
  pull the branch and run from HEAD per the source-of-truth
  lesson when they execute the deferred Track A items in
  a future session.
- **Cross-arch coordination**: Loop 2 captures sm_120 live
  evidence for Track D + Track E deliverables. Each
  deliverable's sm_86 follow-up is recorded as an entry in
  the Track A ledger so the future 3090 Ti session can
  resume without ambiguity.
- **AC-19 lockstep enforcement** (C-AC-C) extends the
  existing `_scrub_cpp_non_code_regions` + scanner pattern
  in `test_exclusion_taxonomy_parity.py`. The
  taxonomy-family surface map is added as a top-level
  constant; each new typed reason declares its family at
  introduction.
- **AC-20 scanner** (C-AC-D) uses a single Python pytest
  case that walks `git ls-files`, applies the extension
  allowlist, applies the exemption table, and checks for
  forbidden plan tokens. The scanner test file uses
  generated fixtures (e.g. `f"AC{'-'}5 violation"`) to
  avoid spurious self-matches.
- **C-AC-E metadata** is a flat JSON file mapping
  entry-point relative paths to
  `{supported_arches: [...], notes: "..."}`. The
  portability test discovers entry points by scanning for
  Python `def main()`, shell `#!/usr/bin/env bash` files,
  and Makefile `.PHONY` targets that produce collection /
  extraction / operator artifacts. Server architectures
  are forbidden in `supported_arches` during Loop 2
  (scope-guard test catches violations).

### Track C dependency graph
- C-AC-C (parity scanner) MUST land before any new typed
  reason is introduced in this loop, because new reasons
  need to register their taxonomy family in the scanner's
  surface map at introduction.
- C-AC-D (plan-token scanner) does NOT depend on C-AC-C.
- C-AC-E (arch capability metadata) does NOT depend on
  C-AC-C or C-AC-D.
- C-AC-H (budget completion) depends on C-AC-C (lockstep
  scanner must exist before new typed reasons land).

### Relevant References
- `tools/local_labels/TASK7_RUNBOOK.md` — repaired Loop 1
  R40, the template for C-AC-B runbook policy.
- `tools/local_labels/test_label_driver_integration.py
  ::Task7RunbookDocumentedCommandTest` — the verification
  template for C-AC-B.
- `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py` —
  the scanner template for C-AC-C lockstep extension.
- `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md` — the
  existing source-of-truth + verification-target hybrid
  runbook; C-AC-B clarifies its source-of-truth language
  and adds command-verification tests for its repo CLI
  invocation shapes.
- `data/local_labels/.gitignore` — quarantine + sidecar
  retention policy (`*.ncu-rep*` blocked; sidecars
  `label_*.json` tracked).
- `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`
  — the static guard that enforces Rules 1-5; remains
  binding.
- `workloads/simbench_bw/schemas/` — schema directory; new
  schemas for Loop 2 land here
  (`sector_stream.schema.json`,
  `divergence_proxies.schema.json`,
  `nvbit_slowdown_budget_exceeded.schema.json`,
  `nvbit_drop_rate_exceeded.schema.json`).
- `references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md`
  — sample device card; CUDA-attribute vs device-card
  distinction is preserved in any Track E AC-6 cache-model
  spec work.
- `.humanize/bitlesson.md` — the two adopted BitLessons
  (`BL-20260504-runbook-self-reference`,
  `BL-20260504-runbook-command-verification`) referenced by
  C-AC-B.

## Dependencies and Sequence

### Milestone 1: Cross-cutting hardenings (Track B; offline)
- Phase 1A: C-AC-B repair `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md`
  source-of-truth language to clarify the hybrid pattern
  (read procedure from HEAD; verification target SHA may
  remain pinned).
- Phase 1B: C-AC-B add command-verification integration
  tests for documented repo CLI invocation shapes in
  `TASK7_RUNBOOK.md` and `CROSS_ARCH_RUNBOOK.md` (positive
  + negative-control); environment probes use static
  checks.

### Milestone 2: Toolchain hygiene (Track C; offline)
- Phase 2A: C-AC-C lockstep extension (per-taxonomy-family
  surface inventory) — must land BEFORE any new typed
  reason in Track D.
- Phase 2B: C-AC-D scoped plan-token scanner + exemption
  table; clean current implementation-file violations.
- Phase 2C: C-AC-E arch capability metadata + portability
  test + scope-guard against server architectures.

### Milestone 3: Track A ledger (concurrent with Track B/C/D)
- Phase 3A: initialize
  `.humanize/rlcr/<session>/blocked-on-3090ti.md` at Loop 2
  Round 0 with the carry-over entries from Loop 1
  (task7, task11b-task11d).
- Phase 3B: every round that touches a Track D / Track E
  deliverable updates the ledger with the matching sm_86
  follow-up entry.

### Milestone 4: Offline foundation (Track D; offline +
  `Verified-live-sm_120`)
- Phase 4A: C-AC-F sector stream (global + local first;
  typed exclusions for other classes; vector-width tests).
- Phase 4B: C-AC-G divergence proxies (typed unavailable
  reasons; outstanding_load_density annotation).
- Phase 4C: C-AC-H budget completion (slowdown +
  drop-rate typed reasons declared in trace-budget family;
  producer counter source enumerated).

### Milestone 5 (Upper Bound only): Track E original-plan
  carries (sm_120 implementation; cross-arch follow-up via
  Track A ledger)
- Phase 5A: AC-6 HRD profiler + cache-model spec (sm_120).
- Phase 5B: AC-7 sharing-locality features.
- Phase 5C: AC-8 order projections.
- Phase 5D: AC-10 perturbation harness (SASS +
  correctness checks).
- Phase 5E: AC-11 adaptive sampler.
- Phase 5F: AC-14 microbenchmark suite.
- Phase 5G: AC-15 combined dual-workload validation +
  manifest pairing.
- Phase 5H: AC-16 local-device NVML covariate sampler.
- Phase 5I: AC-18 workload READMEs.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Initialize `.humanize/rlcr/<session>/blocked-on-3090ti.md` ledger at Loop 2 Round 0; populate carry-over entries from Loop 1 (task7, task11b-task11d) with full detail per C-AC-A positive-test fields | C-AC-A | coding | - |
| task2 | Repair `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md` source-of-truth language to clarify the hybrid pattern (read procedure from HEAD; verification target SHA may remain pinned) | C-AC-B | coding | - |
| task3 | Add command-verification integration tests for documented repo CLI invocation shapes in `TASK7_RUNBOOK.md` and `CROSS_ARCH_RUNBOOK.md` (positive + negative-control); environment probes use static checks | C-AC-B | coding | task2 |
| task4 | Extend `test_exclusion_taxonomy_parity.py` with per-taxonomy-family surface-inventory map and lockstep enforcement for new typed reasons | C-AC-C | coding | - |
| task5 | Author `tests/test_plan_token_scanner.py` (path chosen during execution) with extension allowlist + exemption table + generated test fixtures; clean current implementation-file violations | C-AC-D | coding | - |
| task6 | Author `tools/arch_capabilities.json` + `tests/test_arch_capabilities_coverage.py`; declare every project-owned collection / extraction / operator entry point; restrict portability scan to project-owned paths; add scope-guard pytest case rejecting server architectures during Loop 2 | C-AC-E | coding | - |
| task7 | Author sector stream generator under new implementation directory (path chosen during execution) for global + local memory classes; typed-exclude other classes; schema at `workloads/simbench_bw/schemas/sector_stream.schema.json`; vector-width fixtures; capture sm_120 live evidence and update Track A ledger with sm_86 follow-up entry | C-AC-F | coding | task4, task6 |
| task8 | Author divergence proxies module; declare required trace fields per proxy in `workloads/simbench_bw/schemas/divergence_proxies.schema.json`; emit typed unavailable reasons; annotate `outstanding_load_density_by_phase` as approximation; capture sm_120 live evidence; update Track A ledger | C-AC-G | coding | task7 |
| task9 | Land slowdown + drop-rate typed reasons declared as members of trace-budget taxonomy family in C-AC-C inventory; landed at exactly the four trace-budget surfaces (constants + writer + discovery + schema); capture sm_120 live evidence; update Track A ledger | C-AC-H | coding | task4 |
| task10 | (UPPER BOUND ONLY) Codex analysis of HRD profiler bootstrap CI binning, resampling unit, minimum sample count, infinity/cold bin handling for original AC-6; produce spec preserving CUDA attrs vs device-card distinction | AC-6 (original) | analyze | task4 |
| task11 | (UPPER BOUND ONLY) Author HRD profiler + cache-model spec implementation per task10 spec; capture sm_120 live evidence; update Track A ledger | AC-6 (original) | coding | task10 |
| task12 | (UPPER BOUND ONLY) Author sharing-locality features (grounded + projection) per original AC-7; capture sm_120 live evidence; update Track A ledger | AC-7 (original) | coding | task7, task11 |
| task13 | (UPPER BOUND ONLY) Author 6 order projections with explicit ordering provenance per original AC-8; capture sm_120 live evidence; update Track A ledger | AC-8 (original) | coding | task12 |
| task14 | (UPPER BOUND ONLY) Codex analysis of perturbation correctness-equivalence check shape | AC-10 (original) | analyze | - |
| task15 | (UPPER BOUND ONLY) Author perturbation harness with SASS + correctness checks; ledger; capture sm_120 live evidence; update Track A ledger | AC-10 (original) | coding | task14, task4 |
| task16 | (UPPER BOUND ONLY) Codex analysis of adaptive sampler estimator + MRC quantile set + distance metric spec | AC-11 (original) | analyze | task11 |
| task17 | (UPPER BOUND ONLY) Author adaptive sampler implementation per task16 spec; capture sm_120 live evidence; update Track A ledger | AC-11 (original) | coding | task16 |
| task18 | (UPPER BOUND ONLY) Author microbenchmark suite kernels and shape files per original AC-14; capture sm_120 live evidence; update Track A ledger | AC-14 (original) | coding | task7 |
| task19 | (UPPER BOUND ONLY) Author combined dual-workload integration test + manifest-pairing index by `(slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256)`; sm_120 only | AC-15 (original) | coding | task7, task18 |
| task20 | (UPPER BOUND ONLY) Author NVML covariate sampler local-device portion (sm_120 5060 laptop) per the existing schema | AC-16 (original) | coding | - |
| task21 | (UPPER BOUND ONLY) Update `workloads/simbench_bw/README.md` and `workloads/portable_lcd/README.md` with the data-collection invocation; verify per C-AC-B fake-* integration tests | AC-18 (original) | coding | task3 |

## Claude-Codex Deliberation

### Agreements (final after Phase 5 + Phase 6)
- Hybrid runbook policy (read procedure from HEAD;
  verification target SHA may remain pinned) is correct
  and matches existing `BL-20260504-runbook-self-reference`.
- CLAUDE.md Rules 1-5 + Two-Device Local Development
  Model + cross-arch development on sm_86 + sm_120 remain
  binding.
- task7 closure semantics (per C-AC-A ledger framing): the
  3090 Ti session that follows Loop 2 closes task7 with
  N>=5 individually admitted sidecars; admitted cell OR
  `label_repeatability_failed` is closure;
  `label_repeatability_missing` /
  `label_provenance_inconsistent` are failed collection
  requiring remediation.
- Adding executable command-shape tests for operator
  runbooks is justified by the R39/R40 failure.
- Typed unavailable reasons (not zero defaults) for
  divergence/MSHR proxies, perturbation correctness checks,
  adaptive-sampler spec, and budget drop-counter
  provenance are reasonable.
- Per-taxonomy-family AC-19 lockstep with surface-inventory
  map (5-layer op-class chain, 4-layer trace-budget,
  1-layer local-labels) is correct.
- Server architectures (sm_80, sm_89, sm_90a, server
  sm_120) are forbidden in `supported_arches` during Loop 2;
  scope-guard pytest case enforces this.

### Resolved Disagreements
- AC-ID collision with original plan: RESOLVED by using
  `C-AC-A` through `C-AC-H` for continuation-specific
  criteria + preserved unprefixed `AC-N` for original-plan
  references.
- Track D parallelism vs Codex directive #3: RESOLVED — Track
  D earns `Offline-verified` + `Verified-live-sm_120` in
  Loop 2; sm_86 cross-arch follow-up recorded in Track A
  ledger and deferred.
- C-AC-J → C-AC-H budget surfaces: RESOLVED — declared as
  trace-budget taxonomy family in C-AC-C inventory; lockstep
  at the family's four surfaces.
- Per-architecture metadata: RESOLVED — single
  `tools/arch_capabilities.json`; project-owned scope only;
  server architectures forbidden during Loop 2.
- task7 closure semantics: RESOLVED in Track A ledger
  framing (closure executed in a future 3090 Ti session
  with N>=5 admitted sidecars).
- Server scope: RESOLVED per DEC-2 — server work is out
  of project scope for Loop 2 (and until both local
  devices pass strict accuracy/coverage gates).
- RLCR Stop-hook resilience: RESOLVED per DEC-3 — out of
  scope for Loop 2.

### Convergence Status
- Final Status: `converged` (Phase 5 round 3 returned no
  REQUIRED_CHANGES and no DISAGREE; Phase 6 user decisions
  DEC-1, DEC-2, DEC-3 resolved with scope reductions
  applied to v4).

## Pending User Decisions

(All Phase 6 decisions resolved.)

- DEC-1 (sm_86 task7 closure): RESOLVED — task7 stays
  open in Loop 2; recorded in detail in Track A ledger;
  closure deferred to a future 3090 Ti session.
  - Final Decision: defer 3090 Ti hardware items to a
    future sm_86 session; record sufficient detail in
    Track A ledger.

- DEC-2 (server architecture coverage): RESOLVED — server
  work is out of project scope for Loop 2 and remains out
  of scope until the codebase passes strict
  accuracy/coverage gates on both local devices via
  cross-arch verification.
  - Final Decision: drop AC-12 server bootstrap, AC-16
    server-arch NVML, AC-18 cloud RUNBOOK from Loop 2.

- DEC-3 (RLCR Stop-hook resilience): RESOLVED — out of
  scope per user; not in original plan.
  - Final Decision: drop the C-AC-D RLCR resilience
    proposal entirely.

## Implementation Notes

### Code Style Requirements
- Implementation code, tests, and comments must NOT
  contain plan-token strings (`AC-`, `Milestone`, `Phase`,
  `Step`, `Round N`). Operator markdown under
  `tools/**/*.md` is exempt per C-AC-D. Plan documents
  under `plans/`, reference materials under `references/`,
  and gitignored loop state under `.humanize/` are exempt
  by extension.
- Domain-appropriate naming: `cache_hit_rate_l2_sector`
  not `metric_X`; `correctness_shape` not
  `shape_token_1`.

### Wire-extension policy (from CLAUDE.md)
Producer-side schema additions (new typed reasons, new
enum values, new compression formats) are admitted as
ADDITIVE extensions per CLAUDE.md "Wire-extension policy".
Bumping `schema_version` is reserved for breaking contract
changes.

### Lockstep discipline (C-AC-C)
Every new typed reason or schema field requires the
relevant taxonomy family's full surface set to land in
ONE round. The parity scanner's family-surface map is the
single source of truth for what "lockstep" means per
family.

### Two-Device Discipline (CLAUDE.md)
Every round summary uses one of the four explicit evidence
statuses (`Offline-verified` / `Verified-live-sm_86` /
`Verified-live-sm_120` / `Cross-arch-verified`). In Loop
2, deliverables can earn at most `Verified-live-sm_120`;
`Cross-arch-verified` is earned in a future 3090 Ti
session that consumes the Track A ledger.

### Branch and source-of-truth policy
Loop 2 runs on a fresh branch forked from `5060_dev` HEAD.
Hardware operators on the 3090 Ti pull the branch and run
from HEAD per `BL-20260504-runbook-self-reference` (read
procedure from HEAD; verification target SHA may be
pinned). Cross-device artifact transfer (e.g.
local-label sidecars produced on the 3090 Ti) follows the
existing `data/local_labels/.gitignore` policy:
`*.ncu-rep*` quarantined and not committed; `label_*.json`
sidecars tracked.

### Server scope guard (per DEC-2)
The `tools/arch_capabilities.json` schema explicitly
forbids server architectures (sm_80, sm_89, sm_90a, server
sm_120) in any `supported_arches` list during Loop 2. A
scope-guard pytest case in
`tests/test_arch_capabilities_coverage.py` (or sister)
fails if any entry includes a server architecture. This
guard is removed in a future loop after the codebase
passes strict local accuracy/coverage gates.

--- Original Design Draft Start ---

# Continuation Draft — Counter-Free GPU Cache Observability (RLCR Loop 2 / 2026-05-04)

> Input draft for `/humanize:gen-plan`. This document captures the
> state of the original plan
> `plans/nonroot_gpu_cache_observability_plan.md` as of 40 rounds
> into the 5060_dev RLCR loop (started 2026-05-04T03:41:11Z,
> session id `2026-05-04_11-41-11`, branch `5060_dev`, master
> baseline at commit `d16b375`, R40 HEAD at `6192e06`). Its
> purpose is to seed a follow-up plan that (a) closes the
> remaining work from the original plan, (b) carries forward the
> operational discoveries (BitLessons + a process anomaly)
> learned across the 40 rounds, and (c) keeps the Two-Device
> Local Development Model + CLAUDE.md Rules 1-5 binding.

## 1. Origin and Authoritative References

- **Original plan**:
  `plans/nonroot_gpu_cache_observability_plan.md` (full goal,
  acceptance criteria AC-1 through AC-20, path boundaries,
  feasibility hints, milestones, task breakdown). The
  continuation plan inherits the original plan's goal description
  verbatim and re-references its acceptance criteria by ID.
- **Project rules**: `CLAUDE.md` Rules 1-5 (no NCU on cloud /
  server GPUs; local-label exception for sm_86 + sm_120 via
  `tools/local_labels/` writing into `data/local_labels/`; no
  driver-root operations; no privileged CUPTI surfaces; no
  host-driver work; no simulator revival). The static guard at
  `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`
  enforces all five.
- **Two-Device Local Development Model**: `CLAUDE.md` Section
  "Two-Device Local Development Model" — sm_86 (RTX 3090 Ti
  Ampere) and sm_120 (RTX 5060 Laptop Blackwell) as equal-citizen
  local GPUs with explicit per-architecture evidence statuses
  (`Offline-verified`, `Verified-live-sm_86`,
  `Verified-live-sm_120`, `Cross-arch-verified`).
- **Goal tracker (R1-R40)**:
  `.humanize/rlcr/2026-05-04_11-41-11/goal-tracker.md` (gitignored,
  local-only) records every round's plan-evolution rows,
  blocking/queued side issues, and Completed-and-Verified entries.
- **BitLesson knowledge base**: `.humanize/bitlesson.md`
  (gitignored) carries two project-specific lessons learned in
  this loop (see Section 5 below).
- **Operator runbooks (committed under `tools/`, version-controlled)**:
  - `tools/local_labels/TASK7_RUNBOOK.md` (R39 authored, R40
    repaired against live driver) — sm_86 task7 acceptance.
  - `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md` (R7-R9 authored
    + drift-recovered) — sm_86 task11b-task11d cross-arch
    follow-up.

## 2. Acceptance Criteria Status (AC-1 through AC-20)

The continuation plan inherits the AC numbering from the original
plan unchanged. Status reflects the end of R40 of the 5060_dev
loop on 2026-05-04.

### AC-1 — Contract amendment + static guard scoped allowlist
**MET** (Rounds 0-2 of the prior 3090 Ti loop, preserved on
master). `CLAUDE.md` Rule 1 carries the local-label exception
language; `data/local_labels/` and `tools/local_labels/` are
excluded from the no-NCU scan; the scoped allowlist test inside
`workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`
keeps Rules 2-4 binding within the quarantine subtrees. No
follow-up work needed.

### AC-2 — Local label generation pipeline
**PARTIAL**.
- sm_120 hardware acceptance: **MET** (Round 10 of the current
  loop produced
  `data/local_labels/sm_120/fp32_fma_dep_chain/trace_00/label_0_0.json`
  with full provenance contract; `import_label.py` returned OK).
- sm_120 broader portable_lcd kernels: **prep MET** (Round 13
  produced the 15-active-row artifact set with hardware-accurate
  `gpu_sku=rtx_5060_laptop` provenance).
- sm_86 hardware acceptance (task7): **prep ready, hardware
  blocked**. The repaired `tools/local_labels/TASK7_RUNBOOK.md`
  (Round 40) walks the operator through the documented
  `--shape-point correctness_shape` + `-- --correctness_shape
  --reps 1 --output_dir <path>` command shape; verified end-to-end
  via `Task7RunbookDocumentedCommandTest` with fake-NCU shim plus
  fake portable_lcd workload; manually exercised on the 5060 with
  the real `gemm_hmma_bf16_sm80_run` binary (driver completes
  EXIT 0, sidecar lands at the documented path). **Closing AC-2
  on sm_86 requires a 3090 Ti session.**

### AC-3 — Label repeatability gate
**PARTIAL**.
- sm_120 N>=5 evidence: **MET** (Round 11 of the current loop
  produced 5 admitted sidecars and an aggregator verdict —
  `label_repeatability_failed` with L2 stddev 6.05pp on the
  noisy compute-bound microbench; the gate firing IS the AC-3
  evidence per the plan's typed-reason language).
- sm_86 N>=5 evidence: **pending task7** (3090 Ti session).

### AC-4 — NVBit per-instruction trace contract
**PARTIAL** (much landed, sm_86 cross-arch follow-up
hardware-blocked).
- task9 / 9b / 9c / 9d: **landed** (proto schema, STANDARD_MEM
  producer, classifier with cluster-DSMEM `.MMIO` heuristic,
  helper-aware live witness).
- task10 (synchronous allocator coverage): **landed Rounds 3-4 of
  current loop**. Stream-ordered async + pool + VMM hooks
  added; same-base-reuse retirement bug fixed; cuCtxSynchronize_v2
  + cuStreamQuery hooks; VMM lifecycle covered.
- task11b (`unsupported_memory_path` typed-exclusion sidecars):
  **landed** with 6 explicit excluded classes + `kUnspecified`
  → `UNSPECIFIED` additive widening for unknown opcodes.
- task11c (32-entry lane-address sentinel format end-to-end via
  reconstruction-aware validator): **landed** (prior loop).
- task11d (validator scalar-presence + 6 typed reasons +
  allocation/range identity): **landed Rounds 5-6 of current
  loop**.
- **task11b-task11d-cross-arch-followup**: **pending 3090 Ti
  session**. The current loop earned `Verified-live-sm_120` for
  Rounds 5-6 (live sweep `679 passed, 2 skipped`); sm_86 cross-arch
  validation is queued. The CROSS_ARCH_RUNBOOK.md (R9 drift-
  recovered) is the operator-side artifact; R7+R8 stalled on a
  self-reference trap that R9 broke with the read-from-HEAD
  source-of-truth pattern.
- **Event-based async-free retirement**: queued non-blocking
  hardening (Codex's directive #4 anti-displacement).

### AC-5 — Warp coalescer & sector stream generator
**NOT MET**. Gated on AC-4 cross-arch closure (Codex's
directive #3: AC-5 onset gated on sm_86 closure).

### AC-6 — Conditional HRD profiler with cache-model spec
**NOT MET**. Gated on AC-5.

### AC-7 — GPU-Shalom sharing-locality features (grounded vs projection)
**NOT MET**. Gated on AC-5/AC-6.

### AC-8 — Schedule-marginalized order projections
**NOT MET**. Gated on AC-5/AC-6/AC-7.

### AC-9 — Memory-divergence and MSHR-pressure proxies
**NOT MET**. Gated on AC-5.

### AC-10 — Cache-tomography perturbation harness with SASS confirmation
**NOT MET**. Gated on AC-5.

### AC-11 — Condens-style adaptive NVBit sampling
**NOT MET**. Builds on the existing `trace_volume_exceeded.json`
budget mechanism (already implemented).

### AC-12 — Server collection script package
**NOT MET**. Note: cloud / server GPU labeling is forbidden by
CLAUDE.md Rule 1 — this AC covers feature collection on the
four server architectures (sm_120 RTX PRO 6000 Blackwell, sm_90a
H100, sm_80 A100, sm_89 L40S) WITHOUT NCU labeling. Server-side
collection is downstream of the local-label work.

### AC-13 — NVBit admissibility budgets
**PARTIAL**. The `trace_volume_exceeded.json` sidecar mechanism
is in place (from earlier loops). Slowdown budget
(`nvbit_slowdown_budget_exceeded`) and dropped-records budget
(`nvbit_drop_rate_exceeded`) typed reasons remain to be wired.

### AC-14 — Microbenchmark suite
**NOT MET**. The portable_lcd suite contains 15 active rows
that exercise canonical compute / memory paths; the AC-14 suite
is broader (pointer chase, working-set sweep, stride sweep,
warm/cold pair, `.ca`/`.cg`/`.cs` variants, L2 prefetch hints,
L2 persisting-window sweep, carveout probe, bandwidth saturation
sweep, resident-warp pressure sweep). Implementation gated on
AC-5/AC-10.

### AC-15 — Validate on both workload directories
**PARTIAL**. portable_lcd sm_120 path is green
(`pytest workloads/portable_lcd/tests` → `84 passed`); simbench
live sweep is green (`679 passed, 2 skipped` on R2/R3/R4 of the
current loop). A combined integration test that exercises both
registries through the AC-5+ toolchain is pending.

### AC-16 — NVML covariate collection
**NOT MET**. Gated on AC-12 server collection.

### AC-17 — Architecture portability layer
**NOT MET**. Gated on AC-5+ implementation (each script must
declare its `supported_arches` at implementation time).

### AC-18 — Operator runbook & verification
**PARTIAL**. The two local-label / NVBit-tracer operator
runbooks are in place (`TASK7_RUNBOOK.md`,
`CROSS_ARCH_RUNBOOK.md`). The cloud `tools/cloud/RUNBOOK.md` is
pending AC-12 implementation.

### AC-19 — Reproducibility & exclusion taxonomy
**PARTIAL** (heavily advanced in the current loop). The AC-19
parity scanner at
`tools/nvbit_tracer/test_exclusion_taxonomy_parity.py` has been
hardened across 5 surfaces:
  - producer C++ enum (`memory_op_class_classifier::Value`)
  - proto wire enum (`dynamic_trace::MemoryOpClass`)
  - consumer Python admit list (`VALID_OP_CLASSES`)
  - JSON schema enum (`excluded_op_class.enum`)
  - C++↔proto compile-time numeric pin (`static_assert` block in
    `tracer_tool.cu`)

  Plus: trace-budget surface (4-way parity), local-labels
  surface, scanner robustness against arbitrary lexical hostility
  (raw strings, comments, ordinary strings, char literals — one
  unified `_scrub_cpp_non_code_regions` walker post-R36).

  Pending entries (gated on respective ACs implementing their
  producers): typed reasons named in AC-5 / AC-6 / AC-8 / AC-9 /
  AC-10 / AC-11 / AC-12 / AC-13 / AC-16 (full list in the AC-19
  positive-test enumeration of the original plan).

### AC-20 — Code style scanner / cleanup
**NOT MET**. The AC-19 parity test file itself contains
plan-token mentions (`AC-19`, plan-round prose) — Codex's R28
review noted this is queued until the AC-20 scanner work begins
(directive #4 anti-displacement).

## 3. Hardware-Blocked Items (require a 3090 Ti session)

These items cannot progress on the current 5060 laptop session
and are queued for the next 3090 Ti session.

### task7 — sm_86 local-label hardware acceptance
- **Status**: prep ready (R40 of current loop). Operator follows
  `tools/local_labels/TASK7_RUNBOOK.md` from `5060_dev` HEAD.
- **Outcome shape**: either an admitted cell at
  `data/local_labels/sm_86/<slug>/correctness_shape/` OR an
  excluded cell with a typed reason (both are valid AC-2/AC-3
  evidence; R10/R11 sm_120 precedent established the typed
  exclusion verdict IS valid AC-3 closure).
- **Recommended kernel**: `gemm_hmma_bf16_sm80` from the
  portable_lcd suite.
- **Closes**: AC-2 sm_86 + AC-3 sm_86. Required by Codex's
  directive #3 before AC-5 can begin.

### task11b-task11d-cross-arch-followup — sm_86 NVBit tracer cross-arch
- **Status**: queued. Operator follows
  `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md` from `5060_dev`
  HEAD.
- **Outcome shape**: Round 5/6 of the current loop earned
  `Verified-live-sm_120` for the per-lane resolution +
  `cross_allocation_lane_span` typed-exclusion taxonomy + the
  consumer-side `allocation_id_missing` validator gate. The
  3090 Ti session must rebuild `tracer_tool.so` against sm_86
  and rerun the live sweep (expected `679 passed, 2 skipped`
  baseline) to earn `Cross-arch-verified` status.
- **Closes**: AC-4 cross-arch. Required by Codex's directive #3
  before AC-5 can begin.

## 4. Plan-Forward Work (gated on sm_86 closure)

After task7 + task11b-task11d-cross-arch-followup close, the
continuation plan must drive AC-5 through AC-20 implementation.
The original plan's task breakdown
(`plans/nonroot_gpu_cache_observability_plan.md` Section "Task
Breakdown") enumerates the deliverables. The continuation plan
should preserve that breakdown verbatim where possible and
re-tag tasks (`coding` for Claude / `analyze` for Codex via
`/humanize:ask-codex`).

A reasonable execution order, per the original plan's milestones
and the AC dependency graph:

1. **AC-5 + AC-9 first** (sector stream + divergence proxies):
   the deterministic generator + 8 proxies are the foundation
   for AC-6/AC-7/AC-8.
2. **AC-6 + AC-7 + AC-8** (HRD profiler + sharing locality +
   order projections): require AC-5 sector output.
3. **AC-10** (perturbation harness with SASS confirmation):
   independent of AC-5-9 in implementation but consumes AC-5
   sectors and AC-6 cache-model specs at run time.
4. **AC-11** (adaptive sampler): wraps the existing trace-volume
   budget; can land after AC-5 + AC-9.
5. **AC-13** (admissibility budgets): completes the existing
   trace_volume_exceeded mechanism with slowdown / drop-rate
   typed reasons.
6. **AC-14** (microbenchmark suite): consumes the AC-5-9
   feature extractors.
7. **AC-12 + AC-16 + AC-17** (server bootstrap + NVML +
   portability layer): server-side execution.
8. **AC-15** (combined dual-workload validation): integration
   gate after AC-5-14.
9. **AC-18** (operator runbook): cloud-side runbook.
10. **AC-19** (taxonomy parity): incremental as each new typed
    reason lands.
11. **AC-20** (plan-token scanner cleanup): final pass.

## 5. Operational Discoveries to Carry Forward

### BL-20260504-runbook-self-reference (R7-R9)
**Lesson**: Long-lived operator runbooks under `tools/` that pin
a `git checkout` target SHA produce a self-referential staleness
trap when the runbook's TEXT at the pinned SHA is from a prior
round. Round 7 + Round 8 stalled on this pattern; Round 9 broke
the trap.

**Solution**: Runbooks must (a) include a "Source of Truth"
subsection at the top instructing operators to read the runbook
ONCE from `5060_dev` HEAD via `git show 5060_dev:<path>`, (b)
NOT pin a `git checkout` target SHA, (c) use whatever commit is
currently checked out at execution time.

**Already applied**: `TASK7_RUNBOOK.md` and
`CROSS_ARCH_RUNBOOK.md` both use the read-from-HEAD pattern.

### BL-20260504-runbook-command-verification (R39-R40)
**Lesson**: Operator runbooks that document concrete shell
commands authored from memory or by analogy can document
commands that fail before reaching their purpose. R39 introduced
TASK7_RUNBOOK.md with four interlocking defects (wrong binary
path, simbench-style shape token, missing `--` separator + app
args, false Makefile claim) that all failed the driver's
shape-binding guard before NCU.

**Solution**: Treat operator-runbook authoring as an executable
contract. When adding/modifying a runbook with concrete commands,
exercise the EXACT command shape against the live driver / CLI
in the same round, with fake-NCU shim + fake-binary stand-ins
where actual hardware is unavailable. Add positive (documented
shape reaches the boundary) AND negative-control (broken shape
fails at the expected diagnostic) integration tests to durably
guard against drift.

**Already applied**: R40 added
`Task7RunbookDocumentedCommandTest` with both methods (positive +
negative control) reusing the existing `_NCU_SHIM_TEMPLATE` and
`_PORTABLE_LCD_FAKE_WORKLOAD` fixtures.

**Generalization for the continuation plan**: every new
operator-facing runbook (e.g. `tools/cloud/RUNBOOK.md` for
AC-18) MUST land with a fake-* integration test exercising at
least one documented command shape end-to-end.

## 6. Process Anomaly Observed at R40

**Observation**: For Rounds 1 through 39 of the current loop,
the Claude Code Stop hook (configured by the humanize plugin at
`hooks/loop-codex-stop-hook.sh`) fired automatically when the
agent ended its turn, triggering `codex review` against the
round summary and producing
`.humanize/rlcr/2026-05-04_11-41-11/round-N-review-result.md`.
The user reports that for Round 40, the Stop hook did NOT fire
automatically, and Codex review did not run. (The user also
reports an earlier diagnosis attributing the failure to
`--plugin-dir` not loading hooks is incorrect — the hooks worked
for the prior 39 rounds with the same launch command.)

**Implications**:
1. The continuation plan needs a process-level remediation that
   does not rely solely on the Stop hook firing automatically.
2. The remediation should NOT change the humanize plugin
   internals (out of project scope).
3. The remediation SHOULD give the operator (and Claude) a
   reliable way to invoke Codex review at round boundaries even
   if the hook is silent.

**Possible remediation directions** (for the gen-plan to weigh):
- **Manual invocation fallback**: document a one-liner
  (`codex review --base <branch>` or
  `${CLAUDE_PLUGIN_ROOT}/scripts/rlcr-stop-gate.sh`) that the
  operator OR the agent can run to trigger review at round
  boundaries when the Stop hook is silent.
- **Hook diagnosis subtask**: add an early task that captures
  hook-firing telemetry (e.g., a small log-tap script wrapping
  `loop-codex-stop-hook.sh` to record every invocation, exit
  code, and stderr). If the hook stops firing again, the
  telemetry pinpoints whether the hook ran at all, whether it
  exited zero, or whether Codex itself stalled.
- **Out-of-band acceptance criterion**: the plan can require
  that every round summary be followed by a recorded review
  artifact (auto OR manual), rejecting an "in-flight" round
  without one.

## 7. Goals for the Continuation Plan

The continuation plan's high-level goals are:

1. **Close the hardware-blocked items**: complete task7 and
   task11b-task11d-cross-arch-followup on the 3090 Ti, capture
   `Verified-live-sm_86` evidence, earn `Cross-arch-verified`
   for the prior `Verified-live-sm_120` Round 5-6 work.
2. **Drive AC-5 through AC-20 implementation** in the order
   sketched in Section 4, starting with the AC-5 sector stream
   generator as the foundation.
3. **Make the RLCR loop resilient to Stop hook silence**:
   document the manual fallback, optionally capture telemetry,
   and include explicit per-round verification that a review
   artifact exists before the round is marked Completed.
4. **Carry forward the BitLessons**: every new runbook follows
   `BL-20260504-runbook-self-reference`
   (read-from-HEAD source-of-truth) AND
   `BL-20260504-runbook-command-verification` (verify documented
   commands against the live CLI in the same round).
5. **Preserve the Two-Device discipline**: every round summary
   uses one of the four explicit evidence statuses; cross-arch
   follow-ups are queued explicitly rather than buried.
6. **Preserve CLAUDE.md Rules 1-5**: no new code path may
   invoke NCU outside the local-label quarantine; the static
   guard at
   `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`
   remains binding.

## 8. Constraints and Anti-Goals

- **No simulator revival** (CLAUDE.md Rule 5): the plan does NOT
  reintroduce `nvasim`, Accel-Sim, or simulator-validation work
  as active project direction.
- **No NCU on cloud / server GPUs** (CLAUDE.md Rule 1): cloud
  collection (AC-12) explicitly excludes NCU; only the local
  3090 Ti and 5060 laptop may invoke NCU, and only into
  `data/local_labels/`.
- **No driver-root, no privileged CUPTI, no host-driver work**
  (CLAUDE.md Rules 2-4).
- **No model training / prediction / paper writing** (deferred
  to a follow-up plan that consumes the artifacts produced by
  this toolchain — DEC-1 of the original plan).
- **No DEC-4 validation thresholds** (deferred to a follow-up
  validation plan — DEC-4 of the original plan).
- **No DEC-10 OOD abstention rate target** (model-design
  parameter, not collection — DEC-10 of the original plan).
- **Plan-token cleanup** is itself an AC (AC-20); the plan's
  own use of `AC-*` / `Milestone` / `Phase` / `Step` terminology
  is for plan-document reference only and must not appear in
  implementation code/tests (CLAUDE.md cross-rules + AC-20).

## 9. Suggested Acceptance Criteria for the Continuation Plan

The continuation plan should propose its own AC list that is a
near-superset of the original AC list with status-aware
adjustments. A reasonable starting point:

- **AC-A** (continuation): close task7 sm_86 hardware
  acceptance, earn `Verified-live-sm_86` for the local-label
  pipeline, update goal-tracker. Closes original AC-2/AC-3 sm_86.
- **AC-B** (continuation): close
  task11b-task11d-cross-arch-followup, earn
  `Cross-arch-verified` for the AC-4 slices that previously
  earned `Verified-live-sm_120` only. Closes original AC-4
  cross-arch.
- **AC-C** (continuation): event-based async-free retirement
  hardening for stream-ordered allocator hooks (was queued in
  the original loop's R6 review).
- **AC-D** (continuation): RLCR loop resilience to Stop hook
  silence — a documented manual fallback procedure + optional
  telemetry tap + per-round verification that a review artifact
  exists.
- **AC-E** (continuation): every new operator runbook follows
  the two BitLessons; the plan's runbook deliverables (e.g. for
  AC-18) include fake-* integration tests by construction.
- **AC-original-5 through AC-original-20**: re-stated verbatim
  from the original plan, with status updated where appropriate
  (AC-19 partial-but-advanced, AC-13 partial, others as in
  Section 2 above).

## 10. Path Boundaries (Continuation)

- **Upper bound**: the continuation plan delivers the full
  AC-5 through AC-20 toolchain on top of the now-cross-arch-
  verified AC-4 base, plus the RLCR resilience deliverables and
  the runbook-pattern enforcement, with all server-side flows
  exercised on at least one representative architecture from
  each of {sm_80, sm_89, sm_90a, sm_120 server}.
- **Lower bound**: the continuation plan delivers task7 +
  task11b-task11d-cross-arch closure (sm_86), AC-5 sector stream
  generator + AC-9 divergence proxies (the AC-5+ foundation),
  AC-13 budget completion, AC-19 incremental taxonomy parity for
  whatever new typed reasons land, AC-20 cleanup pass, plus the
  RLCR resilience documentation deliverable. Server-side flows
  (AC-12 / AC-16 / AC-17 / AC-18 cloud runbook) deferred to a
  third loop.
- **Allowed choices**:
  - Can use: NVBit, CUPTI Activity, CUDA events, Nsight Systems,
    static SASS via CUDA binary utilities, read-only NVML.
  - Cannot use: NCU on cloud GPUs; `nvidia-smi` mutation flags;
    privileged CUPTI surfaces (Events, Metrics, Range Profiling,
    PM Sampling, PerfWorks); kernel-module operations; driver
    install / DKMS / MOK / `modprobe nvidia` / reboot workflows;
    simulator revival.

## 11. Feasibility Hints

- **Reuse existing infrastructure**: every continuation deliverable
  builds on what landed in the current loop. Specifically:
  - The AC-19 parity scanner (`test_exclusion_taxonomy_parity.py`,
    139 tests) is the template for adding new typed-reason
    parity guards as AC-5+ producers land.
  - The fake-NCU integration test infrastructure
    (`_NCU_SHIM_TEMPLATE`, `_PORTABLE_LCD_FAKE_WORKLOAD` in
    `tools/local_labels/test_label_driver_integration.py`) is
    the template for runbook command verification tests.
  - The local-label driver (`label_driver.sh`) is the only
    NCU-invoking path in the repo and stays that way; no AC-5+
    deliverable adds a second NCU caller.
  - The trace-volume-budget mechanism is the model for AC-13's
    slowdown / drop-rate typed reasons.

- **Architecture portability discipline** (AC-17): every new
  script MUST declare a `supported_arches: list[str]` constant
  at implementation time. The portability-coverage test in
  AC-17's positive test enumerates them. Adding the test FIRST
  (before the AC-17 implementation) catches missed declarations
  during AC-5 through AC-16 development.

- **Cross-arch validation timing**: by the Two-Device discipline,
  every AC-5+ slice that lands on one device queues a
  follow-up on the other. Plan to alternate sessions: sm_120
  laptop sessions develop the slice, sm_86 desktop sessions
  earn the second-arch evidence (or vice versa).

## 12. Dependencies and Sequence

### Milestone 1: Hardware-blocked closure (3090 Ti session)
- Phase 1A: task7 — local-label driver on portable_lcd kernel,
  N>=5 repeatability, admit/exclude verdict captured. Earns
  `Verified-live-sm_86` for AC-2/AC-3.
- Phase 1B: task11b-task11d-cross-arch-followup — rebuild
  `tracer_tool.so` for sm_86, rerun live sweep, capture
  `cross_allocation_lane_span` evidence on sm_86. Earns
  `Cross-arch-verified` for AC-4 slices Round 5/6.

### Milestone 2: AC-5 + AC-9 foundation
- Phase 2A: AC-5 deterministic sector stream generator with
  schema, fixtures across all 8 access patterns enumerated in
  the original AC-5 positive tests.
- Phase 2B: AC-9 eight divergence/MSHR proxies with
  per-proxy synthetic-kernel unit tests pinned to numeric
  tolerance.

### Milestone 3: AC-6 + AC-7 + AC-8
- Phase 3A: AC-6 conditional HRD profiler + cache-model spec
  schema + bootstrap CI.
- Phase 3B: AC-7 grounded vs projection sharing-locality
  features.
- Phase 3C: AC-8 six order projections + disagreement-variance
  scalar.

### Milestone 4: AC-10 + AC-11 + AC-13 + AC-14
- Phase 4A: AC-10 perturbation harness with SASS confirmation
  ledger.
- Phase 4B: AC-11 adaptive sampler.
- Phase 4C: AC-13 budget completion (slowdown + drop-rate
  typed reasons).
- Phase 4D: AC-14 microbenchmark suite + per-arch SASS slices.

### Milestone 5: AC-12 + AC-16 + AC-17 + AC-18
- Phase 5A: AC-12 cloud bootstrap + per-arch graceful
  degradation.
- Phase 5B: AC-16 NVML covariate sampler.
- Phase 5C: AC-17 portability layer (`supported_arches` in
  every script + portability-coverage test).
- Phase 5D: AC-18 cloud RUNBOOK.md + workload READMEs.

### Milestone 6: AC-15 + AC-19 + AC-20
- Phase 6A: AC-15 combined dual-workload integration test.
- Phase 6B: AC-19 final taxonomy parity audit (every new typed
  reason from AC-5-14 covered).
- Phase 6C: AC-20 code-style scanner + plan-token cleanup
  across implementation tree.

### Milestone 7 (continuation-specific): RLCR resilience
- Phase 7A: document the Stop-hook-silent fallback procedure in
  `tools/cloud/RUNBOOK.md` (or a new `tools/RLCR_RESILIENCE.md`).
- Phase 7B: optional hook-firing telemetry tap.

## 13. Implementation Notes

- **Code style**: implementation code, tests, and comments must
  not contain plan-token terminology (`AC-*`, `Milestone`,
  `Phase`, `Step`). The plan document itself uses these tokens
  for reference; AC-20 enforces the boundary.
- **Schema versioning**: producer-side schema additions
  (e.g. new typed reasons, new enum values) are admitted as
  ADDITIVE extensions per CLAUDE.md "Wire-extension policy"
  (proto3 `optional`, new enum value at the end of the list,
  new compression_format value). Bumping `schema_version` is
  reserved for breaking contract changes.
- **Producer-consumer lockstep**: every new typed reason or
  schema field requires producer + JSON Schema +
  `import_*.py` `VALID_*` set + AC-19 parity scanner update in
  one round to avoid the consumer-side rejection that the
  asymmetric wire policy explicitly anticipates.
- **Live-witness discipline**: every AC-5+ deliverable that
  consumes trace data should be exercised under
  `SIMBENCH_REQUIRE_LIVE_GPU=1` on at least one local
  architecture before being marked Completed; the matching
  cross-arch follow-up is queued explicitly.
- **No forbidden surfaces**: the AC-12 cloud bootstrap must
  detect and refuse forbidden surfaces (privileged CUPTI,
  clock locking, root-required mutation) with typed
  `forbidden_surface_attempted` exit before any collection
  begins.

--- Original Design Draft End ---
