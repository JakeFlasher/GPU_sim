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
