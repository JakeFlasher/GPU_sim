# Continuation Plan — Counter-Free GPU Cache Observability (sm_86 Cross-Arch Loop) — v5

## Goal Description

Resume the original
`plans/nonroot_gpu_cache_observability_plan.md` /
`plans/nonroot_gpu_cache_observability_continuation_plan.md`
work toward completion via **cross-arch development on the
RTX 3090 Ti (sm_86)**. The receiving session is the second
device pivot in the project's two-device development model
(documented in CLAUDE.md "Two-Device Local Development
Model"). The prior Loop 2 (sm_120, RTX 5060 laptop) earned
the sm_120 evidence baseline for AC-1 through AC-14 and
landed the host-only verifier infrastructure for AC-15 and
AC-16; this loop earns the matching sm_86 live evidence and
finishes AC-18 (workload READMEs).

The new loop explicitly:

- **Stays on the local-device cross-arch axis** (sm_86 +
  sm_120) per CLAUDE.md Rules 1-5.
- **Defers server work entirely** (out of project scope
  until the codebase passes strict accuracy/coverage gates
  on BOTH local devices via `Cross-arch-verified`).
- **Records sm_120-blocked items as a record-only ledger**
  inside `tools/BLOCKED_ON_3090TI.md` (the file's name is
  historical; the ledger is now bidirectional —
  sm_86-blocked-during-Loop-2 entries persist; new
  sm_120-blocked-during-this-loop entries should land
  alongside).
- **Drops the RLCR Stop-hook resilience proposal** (not
  in the original plan; out of scope per user decision).
- **Re-tags AC-15 and AC-16** from the prior loop's
  `Verified-live-sm_120` to `Offline-verified` per the
  CLAUDE.md tightened evidence-status discipline:
  `Verified-live-*` is reserved for sweeps that actually
  exercise GPU runtime paths.

CLAUDE.md Rules 1-5 remain binding. The Two-Device Local
Development Model evidence statuses
(`Offline-verified`, `Verified-live-sm_86`,
`Verified-live-sm_120`, `Cross-arch-verified`) gate every
claim of completion.

Deliverables of this loop can earn:

- `Verified-live-sm_86` for AC-2 / AC-3 / AC-4 — the Loop 1
  carry-overs (`task7` local-label hardware acceptance and
  `task11b–task11d` NVBit tracer cross-arch in
  `tools/BLOCKED_ON_3090TI.md`) had no sm_120 evidence and
  remained sm_86-blocked through Loop 2. This loop's
  closure of those ledger entries earns
  `Verified-live-sm_86` directly. The matching
  `Cross-arch-verified` promotion for AC-2 / AC-3 / AC-4
  requires a future sm_120 session because the Loop 2
  scope explicitly deferred them.
- `Verified-live-sm_86` for AC-5 through AC-14 — the
  receiving session captures the matching live evidence on
  the 3090 Ti against the same producer surfaces Loop 2
  sm_120 already exercised.
- `Cross-arch-verified` for AC-5 through AC-14 — when the
  sm_86 live evidence pairs with the sm_120 evidence
  already on disk and the round summary records both
  device/driver/clocks lines.
- `Verified-live-sm_86` for AC-15 / AC-16 — only after a
  real producer-side live sweep runs (not a host-only
  re-run of the verifier).
- `Verified-live-sm_120` *promotion* for AC-15 / AC-16 —
  requires a future sm_120 session capturing the
  producer-side live evidence; the host-only verifier
  on its own is `Offline-verified`.
- `Verified-live-sm_86` for AC-18 (workload READMEs +
  fake-* command verification) — first mainline of this
  loop.
- `Cross-arch-verified` for AC-18 — when the operator
  runs the documented commands on sm_120 in a future
  session and they parse cleanly.

Naming convention preserved from v1: continuation-specific
criteria use prefix `C-AC-`; references to the original
plan's criteria keep their unprefixed `AC-N` form.

## Acceptance Criteria

### Track A — sm_120-blocked / sm_86-blocked dual ledger

- C-AC-A2: Bidirectional Track A ledger.
  - Positive Tests: `tools/BLOCKED_ON_3090TI.md` is
    extended to also accept sm_120-blocked-during-this-
    loop entries (e.g., a new feature implemented on
    sm_86 whose cross-arch verification requires a
    future sm_120 session). Every entry — whether
    sm_86-blocked or sm_120-blocked — carries the
    same five sections: (a) item name, (b) what is
    blocked, (c) precise operator artifact, (d)
    completed prerequisites on the originating device,
    (e) most recent originating-device evidence.
  - Negative Tests: an entry that omits the blocking-
    device label or any of the five sections fails a
    goal-tracker hygiene check at round end.

### Track B — Live sm_86 evidence sweep for AC-2 / AC-3 / AC-4 / AC-5 through AC-14

The receiving session converts the existing
`tools/BLOCKED_ON_3090TI.md` cross-arch entries into real
sm_86 live evidence. Track B closes the Loop 1 carry-overs
(AC-2 / AC-3 local-label hardware acceptance per `task7`,
AC-4 NVBit tracer cross-arch per `task11b–task11d`) AND
promotes the Loop 2 sm_120 ACs (AC-5 through AC-14) by
re-capturing live evidence on the 3090 Ti.

The two Loop 1 carry-over criteria below are NOT
`Verified-live-sm_120` — they have no sm_120 evidence at
all (both were sm_86-blocked when Loop 2 ran on the 5060
laptop, and Loop 2 explicitly deferred them). Their
closure on this loop earns `Verified-live-sm_86`. The
matching `Cross-arch-verified` upgrade requires a future
sm_120 session.

The Loop 2 sm_120 criteria (`C-AC-F2`..`AC-14`) are
`Verified-live-sm_120` on entry; closure of this track
promotes each to `Cross-arch-verified`.

- C-AC-2A3A: AC-2 / AC-3 sm_86 local-label hardware
  acceptance (closes Loop 1 carry-over `task7` ledger
  entry).
  - Positive Tests: follow
    `tools/local_labels/TASK7_RUNBOOK.md` to invoke
    `tools/local_labels/label_driver.sh` on the 3090 Ti;
    produce sm_86 NCU-derived label sidecars under
    `data/local_labels/<slug>/sm_86/`; the local-label
    NCU exception in CLAUDE.md Rule 1 covers this work.
    `tools/local_labels/repeatability_aggregator.py`
    returns an admit verdict for N>=5 repeats per
    (slug, shape_point); the round summary names
    device/driver/CUDA/NVBit build/clocks. Earns
    `Verified-live-sm_86` for AC-2 / AC-3.
  - Negative Tests: a sm_86 label sidecar that fails
    schema validation or a repeatability aggregate
    that falls below the registered admit threshold
    fails the round.

- C-AC-4A: AC-4 NVBit trace contract sm_86 live
  evidence (closes Loop 1 carry-over `task11b–task11d`
  ledger entry).
  - Positive Tests: follow
    `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md` to
    rebuild `tracer_tool.so` on this host (the Loop 2
    `NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool`
    incantation may need a different host gcc). Run
    the live trace contract sweep against
    `dram_stream_stride_mix` on sm_86; capture an
    admitted `dynamic_trace.pb` + matching
    `manifest.json`; run
    `python -m pytest workloads/simbench_bw/tests/test_nvbit_trace_contract_live.py`
    with `SIMBENCH_REQUIRE_LIVE_GPU=1`; confirm the
    helper-section resolver admits both Ampere and
    Blackwell layouts. Earns `Verified-live-sm_86`
    for AC-4. Many later Track B criteria depend on
    a working sm_86 tracer.
  - Negative Tests: a live trace whose helper-section
    resolver hits the ambiguity-failing path (per
    `BL-20260505-cross-arch-sass-mnemonic-divergence`)
    or whose `manifest.json` fails strict validation
    fails the round.

- C-AC-F2: Sector stream sm_86 live evidence
  (cross-arch-promotes original-plan AC-5 / Loop 2 C-AC-F).
  - Positive Tests: a real `dram_stream_stride_mix` NVBit
    capture runs on the 3090 Ti producing a
    schema-valid `sector_stream.jsonl` admitting through
    `tools/sector_stream/consume_trace.py`; the round
    summary names device/driver/CUDA/NVBit/clocks; the
    record count + class distribution is recorded as
    sm_86 baseline (independent of the sm_120 baseline
    of 41 admitted records).
  - Negative Tests: a sm_86 capture whose
    `sector_stream.jsonl` fails strict schema validation
    fails the round.

- C-AC-G2: Divergence proxies sm_86 live evidence.
  - Positive Tests: a real divergence-proxies bundle
    capture on sm_86 admits through the schema; the
    round summary records the bundle's
    `manifest_pairing_index.source_commit` and pins the
    arch-coupled `stride_entropy` distribution + MSHR
    pressure proxy as the sm_86 baseline.
  - Negative Tests: a sm_86 bundle whose typed-
    unavailable branch fails strict validation fails.

- C-AC-H2: Trace-budget sm_86 live verifier.
  - Positive Tests: `verify_artifacts.py --results` runs
    against a real sm_86 capture for at least one
    canonical shape and exits `ok`; slowdown ratio +
    drop rate are recorded as the sm_86 baseline.
  - Negative Tests: stale wall-time metadata or
    expected-record-count mismatch is rejected.

- AC-6 sm_86: HRD profiler live evidence on sm_86.
  - Positive Tests: `hrd_profile.json` is emitted with
    `manifest_pairing_index.source_commit` matching the
    round HEAD; bin distribution recorded as sm_86
    baseline (different from sm_120 because of L1/L2
    sizing differences); `verify_artifacts.py` admits
    cleanly; the budget-exceeded typed-unavailable arm
    behaves correctly when triggered.
  - Negative Tests: stale cached `cache_model_underconfigured_typed_reason`
    or `bundle_underconfigured` is rejected.

- AC-7 sm_86: Sharing locality live evidence on sm_86.
  - Positive Tests: `sharing_locality.json` admits;
    the cross-CTA negative controls remain valid;
    `intra_lane_pairs` and `intra_warp_pairs` aggregate
    over the sm_86 capture's CTA-keyed records.
  - Negative Tests: same as Loop 2's negative controls;
    the sm_86 bundle exercises them on the new arch.

- AC-8 sm_86: Order projections live evidence on sm_86.
  - Positive Tests: 6 admitted projections in
    `order_projections.json`; cached
    `order_projections_admitted_count == 6`; verifier
    `ok`.
  - Negative Tests: a `projection_unavailable_sm_local`
    fail-closed path correctly fires when the producer
    can't compute one of the six on sm_86.

- AC-10 sm_86: Perturbation harness live evidence on
  sm_86.
  - Positive Tests: `perturbation_run_ledger.json`
    admits; `working_set_sweep` per-MiB element counts
    correct; cache-modifier SASS regex admits the
    sm_86 Ampere mnemonics
    (`LDG.E.CA` / `LDG.E.CG`) per
    `BL-20260505-cross-arch-sass-mnemonic-divergence`;
    `cache_modifier_cs` admits.
  - Negative Tests: the modifier-optimized-away typed-
    unavailable arm correctly fires for fixtures missing
    the SASS opcode.

- AC-11 sm_86: Adaptive sampler live evidence on sm_86.
  - Positive Tests: the
    `_SUPPORTED_ARCHES = ("sm_86", TARGET_ARCH)` admit
    set continues to include sm_86; the live sweep
    produces a `trace_convergence_windows_digest` that
    round-trips through `verify_artifacts.py`.
  - Negative Tests: stale cached digest or stale cached
    manifest tuple is rejected.

- AC-14 sm_86: Microbenchmark suite live evidence on
  sm_86.
  - Positive Tests: 9 R51-contract shape points × the
    appropriate canonical sm_86 result roots admit via
    `verify_artifacts.py --results`; per-shape feature
    sidecars (HRD, order projections, divergence
    proxies, sharing locality) emit and bind. 4 R12-
    style fail-closed verifiers admit. The HRD
    budget-exceeded typed-unavailable arm admits when
    triggered.
  - Negative Tests: a structural-only family that
    fails to produce a live sm_86 capture correctly
    surfaces a typed `not_implemented` blocker on its
    sm_86 path while remaining `live_verified_sm_120`
    in the inventory.

### Track C — Loop 2 carryover (host-only re-run + queued cleanup)

- C-AC-I2: Re-run the Loop 2 host-only suites on sm_86
  and re-record `Offline-verified`.
  - Positive Tests: `pytest tools/ workloads/simbench_bw/tests/ -q`
    passes 1944+ on the 3090 Ti host; the 4 validators
    (`trace_size_estimator.py --validate-all`,
    `validate.py --all`, `verify_coverage.py --strict`,
    AC-14 + plan-token gates) all return `ok`; the
    R56-R60 host-only test counts hold (26 dual-workload
    pairing index tests, 10 read-only-NVML guard tests,
    8 NVML metadata-pin tests). The round summary
    records `Offline-verified` for AC-15 and AC-16
    explicitly; no automatic promotion.
  - Negative Tests: any host-only test that regressed
    during scp fails the round.

### Track D — AC-15 / AC-16 producer-side live evidence (optional, gates true cross-arch closure)

- AC-15 producer-side live evidence on sm_86 (optional in
  this loop; required for `Verified-live-sm_86`
  promotion).
  - Positive Tests: build at least one active portable_lcd
    binary on the 3090 Ti, run the simbench timing capture
    that produces a `trace_<shape>.json` with the full
    5-tuple identity, then run
    `build_combined_pairing_index` against the live
    rosters + the actual capture tree. Promotion
    requires ALL of:
    1. The combined index is non-empty.
    2. BOTH workload registries
       (`simbench_bw` AND `portable_lcd`) contribute at
       least one identity each — `len(index) > 0` alone
       admits a single-registry result and is not
       sufficient.
    3. Every identity's `source_commit` equals the
       receiving session's `git rev-parse HEAD` AND is
       NOT the helper's `"0" * 64` zero-fill fallback
       (the fallback fires silently when
       `_resolve_source_commit` cannot read `.git/HEAD`
       — for example a fetched clone with
       packed refs). Operator must verify the helper
       handles packed refs correctly before claiming
       promotion; see
       `tools/BLOCKED_ON_3090TI.md::loop2-task19` for
       the latent-foot-gun details.
    4. The live `LiveRosterSmokeTest` admits without
       firing any typed-exception path.
  - Negative Tests: same fail-closed arms as the host-only
    suite (missing binary, missing manifest, slug
    mismatch, schema-invalid manifest, binary digest
    mismatch).

- AC-16 producer-side live evidence on sm_86 (optional in
  this loop; required for `Verified-live-sm_86`
  promotion).
  - Positive Tests: produce a preflight manifest
    (`bash tools/cloud/preflight.sh --out
    /tmp/simbench_preflight.json`) and run
    `python3 workloads/simbench_bw/run_suite.py
    --mode timing --kernel <slug> --shape-point timing_00
    --enable-cupti-activity --preflight-manifest
    /tmp/simbench_preflight.json` on sm_86 (the
    `--preflight-manifest` flag is a hard requirement
    of `--enable-cupti-activity` per `run_suite.py`'s
    EXIT_CUPTI_ACTIVITY_PREFLIGHT_MANIFEST_REQUIRED
    guard). Confirm `nvml_covariate_<repeat>.json`
    sidecars are emitted alongside
    `cupti_activity_<repeat>.json`; the
    `measurement_manifest.json` carries the 4 NVML
    provenance fields; `verify_nvml_metadata_pin`
    admits the live pair; collector status is `ok` for
    at least one repeat.
  - Negative Tests: `verify_nvml_metadata_pin` rejects a
    manifest whose cached fields disagree with the live
    sidecars (e.g., manually mutate the cached
    `polling_interval_ns` and confirm
    `NvmlMetadataDriftError` fires).

### Track E — AC-18 closure (workload READMEs + fake-* command verification)

- AC-18 (workload portion):
  - Positive Tests: `workloads/simbench_bw/README.md`
    documents the exact local-device command sequence
    using live CLI shapes:
    `make -C workloads/simbench_bw cupti-collector-lib`,
    `bash tools/cloud/preflight.sh --out /tmp/simbench_preflight.json`,
    `python3 workloads/simbench_bw/run_suite.py --mode timing
    --kernel dram_stream_stride_mix --shape-point timing_00
    --enable-cupti-activity --preflight-manifest
    /tmp/simbench_preflight.json`,
    `python3 workloads/simbench_bw/run_suite.py --mode trace
    --kernel dram_stream_stride_mix --shape-point trace_00
    --phase-inject-ab`,
    `python3 workloads/simbench_bw/run_suite.py --mode
    emit-dual-run-manifest ...`.
    `workloads/portable_lcd/README.md` documents
    `make -C workloads/portable_lcd/<active-slug>`,
    `python3 workloads/portable_lcd/verify_kernel_contracts.py`,
    or the narrower verified contract command. Both
    READMEs follow the runbook source-of-truth pattern
    (`BL-20260504-runbook-self-reference`).
  - Negative Tests: `tools/test_runbook_command_verification.py`
    (or a workload-README-specific sibling) asserts the
    README prose contains the exact argv shapes; drives
    parser-safe commands through the live CLIs without
    GPU dependency; rejects misspelled flags / missing
    required values via `BL-20260504-runbook-command-verification`-
    pattern fake-* tests.

## Carried Status (entry to this loop)

The original-plan AC-1 (preflight manifest) was earned in
the very first project loop and is not in scope for any
subsequent re-validation. The Loop 1 carry-overs AC-2 /
AC-3 / AC-4 are sm_86-blocked work that was deferred from
Loop 2; they enter this loop without sm_120 evidence and
must be closed by this loop. The Loop 2 sm_120 ACs enter
this loop with `Verified-live-sm_120` (the C-AC-* +
AC-5..AC-14 set) or `Offline-verified` (AC-15 and AC-16
after the R59 / R60 re-tagging).

| AC | Loop status (entry to this loop) |
|----|----------------------------------|
| AC-1 | Closed in original plan; not in scope for re-validation. |
| AC-2 | **No sm_120 evidence** — Loop 1 carry-over via `task7`; this loop earns `Verified-live-sm_86`. |
| AC-3 | **No sm_120 evidence** — Loop 1 carry-over via `task7`; this loop earns `Verified-live-sm_86`. |
| AC-4 | **No sm_120 evidence** — Loop 1 carry-over via `task11b–task11d`; this loop earns `Verified-live-sm_86`. |
| C-AC-A | Offline-verified |
| C-AC-B | Offline-verified |
| C-AC-C | Offline-verified |
| C-AC-D | Offline-verified |
| C-AC-E | Offline-verified |
| C-AC-F | Verified-live-sm_120 |
| C-AC-G | Verified-live-sm_120 |
| C-AC-H | Verified-live-sm_120 |
| AC-6 | Verified-live-sm_120 |
| AC-7 | Verified-live-sm_120 |
| AC-8 | Verified-live-sm_120 |
| AC-10 | Verified-live-sm_120 |
| AC-11 | Verified-live-sm_120 |
| AC-14 | Verified-live-sm_120 |
| AC-15 | **Offline-verified** (re-tagged from Verified-live-sm_120) |
| AC-16 (local-device) | **Offline-verified** (re-tagged from Verified-live-sm_120) |

The promotion to `Cross-arch-verified` happens incrementally
during this loop as Track B + Track D + Track E evidence
land.

## Explicitly Deferred

| Item | Rationale | When to reconsider |
|------|-----------|--------------------|
| AC-12 server collection bootstrap | Server work out of project scope until BOTH local devices pass `Cross-arch-verified`. | After this loop closes Track B. |
| AC-16 server-architecture NVML | Same scope rationale as AC-12. | Same trigger as AC-12. |
| AC-18 cloud `tools/cloud/RUNBOOK.md` | Same scope rationale as AC-12. | Same trigger as AC-12. |
| RLCR Stop-hook resilience | Not in original plan; out of scope per user decision. | A future plan that explicitly scopes RLCR-process resilience. |

## Receiving-Session Discipline (mandatory)

- **Re-read CLAUDE.md and `HANDOFF.md` at session start.**
  CLAUDE.md is the durable contract; HANDOFF.md is the
  machine-to-machine transfer state. Both are required
  reading before the first round contract.
- **Do NOT promote AC-15 / AC-16 by re-running host-only
  tests.** Promotion requires a real producer-side
  live sweep on this arch.
- **Each round summary names device/driver/CUDA/NVBit/
  clocks** when claiming `Verified-live-sm_86`.
- **Mutable section of `goal-tracker.md` is the only
  evolving state.** The immutable section is fixed at
  Round 0.
- **The plan-token scanner is binding** — implementation
  files cannot reference `AC-N`, `Milestone`, `Phase`,
  `Step`, or `Round N`. Re-prose violations in
  domain-appropriate language.

## Out-of-scope for the new loop (do not displace mainline)

- Server-architecture NVML / cloud bootstrap.
- Simulator revival (CLAUDE.md Rule 5 binding).
- Driver-root operations (CLAUDE.md Rules 2-4 binding).
- NCU on cloud/server GPUs (CLAUDE.md Rule 1; local
  exception remains for `tools/local_labels/`).
- Any reorganization of `.humanize/rlcr/` from a previous
  loop (already deleted before Round 0).

## Risks for the new loop

- **Arch-coupling surfaces during sm_86 live sweeps.**
  Per CLAUDE.md Rule 1 of the Two-Device Local Development
  Model, opcode sets, register counts, SASS layouts,
  helper-function inlining policies, and compression-
  format distributions are NOT constant across arches.
  The cache-modifier SASS regex
  (`BL-20260505-cross-arch-sass-mnemonic-divergence`)
  already admits both Ampere and Blackwell forms; new
  arch-coupled findings should land as BitLesson entries
  + producer/verifier lockstep changes per
  `BL-20260505-tools-subdir-module-name-collision`-style
  discipline.
- **Tracer rebuild on sm_86.** The 3090 Ti session may
  need to rebuild `tracer_tool.so` for sm_86 if the
  receiving host's CUDA toolkit / gcc version differs
  from the sm_120 build. The Loop 2 `gcc14` recipe
  (`NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool`)
  remains the documented build incantation.
- **AC-18 fake-* command verification has no GPU
  dependency**, so it should land first as the lowest-
  risk new mainline. Track B's live sweeps run after
  AC-18 closes the host-only contract.
- **A future sm_120 session** will be needed to promote
  AC-18 to `Cross-arch-verified` and to capture the
  producer-side live evidence for AC-15 / AC-16 on sm_120.
  This loop's job is to land the sm_86 leg cleanly.
