# AC-16 Cross-Arch Replay Plan For dram_stream_stride_mix On sm_120 (RTX 5060 Laptop)

## Goal Description

Capture sm_120 timing-mode live evidence for `dram_stream_stride_mix/timing_00` on the RTX 5060 laptop (Blackwell sm_120), admit via the existing strict admission helper, and promote AC-16 second-workload-eligibility from `Verified-live-sm_86` to `Cross-arch-verified` by combining the new sm_120 evidence with the sm_86 baseline already at HEAD.

This is the second of two single-arch RLCR loops for AC-16 second-workload-eligibility. The first loop (v7, session id `2026-05-08_08-57-02` on the RTX 3090 Ti host) verified the sm_86 leg with all 10 plan ACs closed at HEAD `5ea52f3` (commit chain: `9f5a2ca` hook + `14e37bc` TOML + `eaf5eee` regression tests + `30ae840` text-contract cleanup + `5ea52f3` AC-7 hardening). This second loop verifies the sm_120 leg ONLY — no new substantive code is needed; the host-side glue (dlsym hook + cudaEvent bracket in `main.cu`, `-ldl` in `Makefile`, `timing_00.launch_config` in TOML, regression tests) is already at HEAD and is architecture-independent.

The structural-stall trap from the v7 loop is **pre-empted by design**: this plan's anti-goals do NOT include "no sm_120 pivot" — sm_120 is the loop's mainline. The plan's anti-goals are the inverse: no sm_86 work (the sm_86 baseline is regression-locked, not in scope to re-verify on the laptop). The auto-memory entry `~/.claude/projects/.../memory/project_rlcr_structural_stall_pattern.md` from v7 R3 documents the lesson; this plan applies it.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Reference and target paths in plan, contract, and round summary match the actual filesystem (carry-forward from v7 plan AC-1).
  - Positive Tests:
    - `test -f workloads/simbench_bw/l2/rmsnorm_fwd_bf16/main.cu` rc=0 (reference)
    - `test -f workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu` rc=0 (target)
    - The round contract / plan / summary cite both paths verbatim with `l2/` and `l1/` respectively
  - Negative Tests:
    - `test -f workloads/simbench_bw/l1/rmsnorm_fwd_bf16/main.cu` rc!=0 (legacy probe path absent)

- AC-2: Source-level instrumentation present at HEAD (verification-only carry-forward from v7 sm_86 plan AC-2).
  - Positive Tests:
    - File contains `#include <dlfcn.h>` and `#include <math.h>`
    - `typedef void (*simbench_cupti_record_fn)(double);` typedef + docstring block
    - `cudaEventCreate`/`cudaEventRecord`/`cudaEventSynchronize`/`cudaEventElapsedTime`/`cudaEventDestroy` chain bracket only the kernel launch
    - Finite-positive guard `if (!isfinite(wall_time_ms) || wall_time_ms <= 0.0) fail(...)` present
    - `dlsym(RTLD_DEFAULT, "simbench_cupti_record_cuda_event_wall_time_ms")` with null-check
    - `printf("  \"wall_time_ms\": %.9f,\n", wall_time_ms);` literal appears EXACTLY TWICE (one per stdout JSON output mode)
  - Negative Tests:
    - The dlsym call site does NOT pass an unguarded function-pointer
    - The cudaEvent bracket does NOT enclose any non-launch operations
    - The wall_time_ms field is NOT emitted before the `cudaEventElapsedTime` call

- AC-3: Build produces a working multi-arch binary on the RTX 5060 laptop.
  - Positive Tests:
    - `Makefile` link line includes `-lcudart -ldl` (carry-forward; verify, do not modify)
    - `make -C workloads/simbench_bw/l1/dram_stream_stride_mix clean all` exits rc=0 on the laptop
    - `cuobjdump --list-elf dram_stream_stride_mix_run` lists cubins for sm_86, sm_90a, AND sm_120 (the existing ARCH_GATE token set; the binary is multi-arch, so all three cubins must be present even though only the sm_120 cubin executes on the laptop)
    - Native gcc-14 toolchain used per the project's feedback memory (NVCC_CCBIN=g++-14 if needed)
  - Negative Tests:
    - Build does NOT fail with undefined-reference linker errors
    - `cuobjdump` does NOT report missing cubins outside the ARCH_GATE token set

- AC-4: Backward compatibility on sm_120 (no LD_PRELOAD execution preserves baseline) + cross-arch correctness invariance.
  - Positive Tests:
    - Standalone `dram_stream_stride_mix_run --shape-point trace_00` (no LD_PRELOAD) on the laptop exits rc=0 and emits the expected JSON envelope including `wall_time_ms` and all pre-existing fields (`schema_version`, `slug`, `shape_set`, `shape_point`, `output`)
    - Standalone `dram_stream_stride_mix_run --shape-point correctness_00 --output-mode sha256_f32le` (no LD_PRELOAD) on the laptop exits rc=0 and emits an `output_sha256` digest equal to the v7 sm_86 baseline `786c10f82133609842e2be4226ff90154a5efde3c2ff001a504dd2367828e782` (cross-arch correctness invariance: Codex review verified the kernel uses identity output `out[i] = in[i]` with the only FP accumulation writing to `sink[0]` which is NOT included in the SHA-256 hash; therefore the digest is deterministic across arches independently of FP rounding mode, NaN propagation, or transcendental approximation differences). **This equality is strictly required**; if it fails, AC-4 BLOCKS, AC-11 BLOCKS, and the round logs a side issue documenting cross-arch correctness drift before any further AC-5/AC-6 work.
  - Negative Tests:
    - Standalone run does NOT emit a different digest from the sm_86 baseline (cross-arch correctness invariance is strict; failure blocks the loop, not just AC-4)
    - Standalone run does NOT crash, hang, or emit stderr error markers when dlsym lookup returns nullptr

  **Optional non-blocking sanity** (NOT a positive test for AC-4 closure): the existing trace-mode capture `python3 workloads/simbench_bw/run_suite.py --mode trace --kernel dram_stream_stride_mix --shape-point trace_00 --phase-inject-ab` may be exercised on sm_120 to confirm the NVBit tracer + phase-injection harness work on Blackwell. Failures on sm_120 trace-mode are NOT blocking for AC-4 / AC-11 / cross-arch promotion (they would indicate NVBit/SASS classifier issues that are out-of-scope for the AC-16 timing-mode CUPTI dual-source contract; log as a side issue and proceed). The AC-16 second-workload-eligibility contract is timing-mode + CUPTI Activity dual-source, NOT trace-mode + NVBit.

- AC-5: Fresh sm_120 timing-mode capture.
  - Positive Tests:
    - Fresh sm_120 preflight manifest at `<round_root>/preflight_manifest.json` admits via `tools/cloud/preflight_manifest.py --validate`
    - `python3 workloads/simbench_bw/run_suite.py --mode timing --kernel dram_stream_stride_mix --shape-point timing_00 --enable-cupti-activity --preflight-manifest <round_root>/preflight_manifest.json` exits rc=0 on the laptop
    - Capture root layout (paths shown relative to `workloads/simbench_bw/results/dram_stream_stride_mix/<laptop_gpu_sku>/sm_120/`):
      - Parent timing artifact: `timing_timing_00.json`
      - Per-repeat sidecars: `timing_00/cupti_activity_<i>.json` for i in 0..29 (30 sidecars)
      - Per-repeat NVML covariates: `timing_00/nvml_covariate_<i>.json` for i in 0..29 (30 sidecars)
      - **Shape-level manifest at `timing_00/measurement_manifest.json`** (NOT at the parent results root; the strict-admission helper consumes the manifest at the shape-level path per `cupti_dual_source_live_admission.py` + `run_suite.py` orchestration)
    - Timing artifact's `dual_source_reconciliation_failures` array is empty
    - Shape-level `measurement_manifest.json` records `enable_cupti_activity=True`, `compiler_flags` includes `arch=compute_120,code=sm_120`, `nvml_collector_status=ok`, `gpu_sku=<laptop_gpu_sku>`, `compute_capability_major=12`, `compute_capability_minor=0`
    - **Freshness proof**: round summary records SHA-256 of `dram_stream_stride_mix_run` + `libsimbench_cupti_activity_collector.so` + capture timestamp + `git rev-parse HEAD`, all consistent with a single rebuild+capture session on the laptop (no carried 3090 Ti binaries)
  - Negative Tests:
    - Capture does NOT produce 0 sidecars or fewer than `repeat_iters=30`
    - Capture does NOT proceed if preflight manifest's NVML metadata pin verifier rejects
    - Timing artifact does NOT contain non-empty `dual_source_reconciliation_failures`
    - The shape-level `measurement_manifest.json` does NOT report a `binary_sha256` matching a carried 3090 Ti binary (the freshness proof would catch a stale binary that was scp'd from the originating host and not rebuilt)

- AC-6: Strict admission helper accepts sm_120 evidence.
  - Positive Tests:
    - `admit_cupti_dual_source_live_capture(timing_artifact_path=<sm120>/timing_timing_00.json, sidecar_dir=<sm120>/timing_00, expected_slug="dram_stream_stride_mix", expected_shape_point="timing_00", collect_defects=True)` returns `result.ok=True`
    - `result.cupti_sidecar_pass_count == 30 and result.cupti_sidecar_total == 30`
    - `result.defects` is the empty list (zero typed defects)
    - `result.timing_artifact_failure_count == 0`
  - Negative Tests:
    - Wrong-slug sentinel (e.g., `expected_slug="rmsnorm_fwd_bf16"` against the dram capture) returns `ok=False` with all 4 surface defect tokens (`timing_artifact_slug_mismatch`, `cupti_sidecar_slug_mismatch`, `nvml_sidecar_slug_mismatch`, `measurement_manifest_slug_mismatch`); the strict R22-widened identity contract applies identically on sm_120
    - **Forced-failure sentinel**: a synthetic-fixture probe that mutates one CUPTI sidecar's `dual_source_reconciliation_passed` to `false` (or equivalently makes one sample's `cupti_activity_duration_ns` violate the dual-source tolerance against the cuda-event measurement) returns `ok=False` with `timing_artifact_failure_count >= 1` AND the appropriate per-repeat defect tokens; the helper does NOT silently pass when value-axis correctness fails (this catches the R20→R21 regression class that BL-20260509-strict-admission-contract-first was extracted to prevent)

- AC-7: Regression coverage (verification-only carry-forward).
  - Positive Tests:
    - `pytest workloads/simbench_bw/tests/test_dram_stream_stride_mix_hook_static.py workloads/simbench_bw/tests/test_cupti_dual_source_live_admission.py::CuptiDualSourceAdmissionSecondWorkloadParameterizedTest -q` exits rc=0 with 7 passed on the laptop (4 static + 3 parameterized)
    - The exact-literal-count assertion from v7 R1 (`self.text.count(r'printf("  \"wall_time_ms\": %.9f,\n", wall_time_ms);') == 2`) passes against the carried main.cu
  - Negative Tests:
    - The static test does NOT pass when any one of the four required substrings is absent from `dram_stream_stride_mix/main.cu` (regression-locks the v7 R1 hardened printf-format-and-cardinality assertion)
    - The parameterized helper test does NOT pass when its synthetic fixture's `dual_source_reconciliation_passed` field is forced to false (the helper must reject; carries forward the R22 widened contract verification)
    - The targeted-test invocation does NOT silently change result counts (e.g., a regression that adds or removes a static-test method is caught by the explicit "7 passed" expectation in the positive test)

- AC-8: Existing v7 sm_86 evidence remains coherent (regression baseline carry-forward, transport-aware, branch-aware).
  - Positive Tests (transport-conditional):
    - **If repo synced via raw scp** (capture artifacts survive): the carried v7 sm_86 capture under `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/` admits `ok=True 30/30 0-defect` when invoked with the same helper signature on the laptop
    - **If repo synced via git fetch** (capture artifacts do NOT survive): the AC-8 first positive test is satisfied by citing the v7 R0/R1 helper-return result as the documented sm_86 baseline (`ok=True, cupti_sidecar_pass_count=30, cupti_sidecar_total=30, defects=[]`), recorded in v7 round-0-summary.md AC-6 section + Codex Round-1 review's verification-performed section. **Recommendation**: before the v8 loop launches, copy a compact v7 evidence manifest (e.g., a JSON file containing the v7 helper-return values + the sm_86 capture directory's SHA-256 hashes) outside `.humanize/rlcr/` (which gets deleted on the receiving host) so the documented baseline is on-disk on the laptop. This makes git-fetch transport viable without losing the AC-8 anchor.
    - Broad sweep `pytest tools/ workloads/simbench_bw/tests/ -q` on the laptop produces no NEW failures relative to the v7 baseline. **Branch-aware wording**: at HEAD `5ea52f3`, the carried baseline was 4 failures (`tools/test_runbook_command_verification.py::SimbenchReadmeRunSuiteTimingDocumentedCommandTest::test_documented_in_simbench_readme` + 3 `workloads/simbench_bw/tests/test_handle_timing_dual_source_hard_gate.py` classification cases). If the receiving session pulls a successor HEAD that has changed the carried baseline (e.g., a hygiene round closed any of these), the AC-8 expectation is "no NEW failures BEYOND THE BASELINE OBSERVED ON SUCCESSOR HEAD AT v8 LOOP START", not the literal 4-failure count. The round summary records the observed baseline at v8 R0 entry as the regression anchor.
    - AC-15 combined pairing index helper still reports the durable invariants `total=25` entries and `zero_fill_source_commits=0`; the `at_head` reading is observational and not pinned (per v7 plan AC-8 discipline). **Branch-aware wording**: if the successor HEAD has added or removed pairing-index entries, the durable invariants `zero_fill_source_commits=0` is still binding but `total=25` may shift; record the observed `total` at v8 R0 entry as the regression anchor.
  - Negative Tests:
    - The broad test sweep does NOT introduce new test count regressions BEYOND the v8-R0-entry-recorded baseline (regardless of whether that baseline matches v7's literal 4 failures)
    - The carried R23 rmsnorm helper invocation does NOT degrade
    - The pairing index does NOT introduce zero-fill source_commits (the `zero_fill_source_commits=0` invariant is durable across HEAD advances)

- AC-9: Status wording reflects sm_120-only promotion FIRST, then cross-arch combination.
  - Positive Tests (in this order):
    - Round summary names device (RTX 5060 laptop), driver version, CUDA toolkit version, NVBit build hash, native gcc version, observed preflight clocks/probes per CLAUDE.md "Two-Device Local Development Model"
    - Round summary records `cuobjdump --list-elf` output for the rebuilt dram_stream_stride_mix_run binary on the laptop
    - Goal-tracker mutable section FIRST records this as `Verified-live-sm_120` for AC-16-second-workload-eligibility (independent of cross-arch claim)
    - Goal-tracker THEN combines v7 `Verified-live-sm_86` (carried) + this loop's `Verified-live-sm_120` into `Cross-arch-verified` for AC-16 second-workload-eligibility (this transition is AC-11)
  - Negative Tests:
    - Goal-tracker does NOT mark this AC closure as `Cross-arch-verified` UNTIL both `Verified-live-sm_86` (from v7) AND `Verified-live-sm_120` (this loop) are independently established with named live-evidence artifacts
    - Round summary does NOT use absolute-completion-gate wording (no "the only", "sole", "completes" without scope-bounding) per the v7 R25/R27 retraction discipline
    - HANDOFF.md, top-level README.md, tools/cloud/README.md, simbench_bw/README.md, nvbit_tracer/README.md NOT modified (carry-forward anti-goal from v7 plan)

- AC-10: TOML launch_config still correct (verification-only carry-forward).
  - Positive Tests:
    - `workloads/simbench_bw/shapes/dram_stream_stride_mix.toml` `timing_00.launch_config` matches the v7 final state: grid `[1, 1, 1]` and block `[32, 1, 1]`
    - `python3 workloads/simbench_bw/validate.py --all` exits rc=0 on the laptop
    - Trace-budget verifier and any pairing/estimation consumers admit the `[1,1,1] / [32,1,1]` launch_config without errors on sm_120
  - Negative Tests:
    - `validate.py --all` does NOT silently pass if the TOML has been mutated to the prior incorrect `[256,1,1] / [128,1,1]` values (the contract surface is meaningful; mutation must be detected)
    - The TOML's `timing_00.launch_config` block does NOT silently change between transport (raw scp) and local checkout (git fetch); both must yield the same `grid=[1,1,1]` `block=[32,1,1]` values per the v7 commit `14e37bc`

- AC-11: AC-16 second-workload-eligibility promotes to `Cross-arch-verified`.
  - Positive Tests:
    - Goal-tracker MUTABLE Completed and Verified table records `AC-16 second-workload-eligibility (dram_stream_stride_mix/timing_00)` as `Cross-arch-verified` with the combined evidence anchor: v7 sm_86 capture (HEAD `5ea52f3`) + this loop's sm_120 capture (HEAD ≥ `5ea52f3`)
    - Round summary explicitly states the cross-arch closure rationale: the same dlsym hook + cudaEvent bracket + helper admission contract works identically on both arches; behavioral evidence is consistent across sm_86 and sm_120; only timing-side evidence (clocks, throttle state, %clock64 phase counts) is per-architecture and is recorded as observational covariate
  - Negative Tests:
    - The promotion does NOT happen unless BOTH legs have independent live witnesses with named device + driver + CUDA + NVBit + gcc + observed-clocks provenance
    - The promotion does NOT claim arches beyond sm_86 + sm_120 (e.g., cloud sm_100 / B200 datacenter remains explicitly out-of-scope; this loop covers two-device local development model only)
    - The promotion does NOT cause silent edits to HANDOFF.md / top-level README.md / tools/cloud/README.md / nvbit_tracer/README.md / simbench_bw/README.md (carry-forward anti-goal)

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

All 11 ACs closed in a single round on the laptop: bootstrap (AC-1), source-instrumentation verify (AC-2), rebuild + cuobjdump (AC-3), backward-compat with cross-arch digest equality (AC-4), fresh sm_120 timing-mode live capture with 30 admitted sidecars (AC-5), strict-admission helper invocation returning `ok=True 30/30 0-defect` (AC-6), targeted regression tests pass (AC-7), broad sweep with no new regressions vs v7 baseline (AC-8), round summary with full Two-Device-Model provenance (AC-9), TOML still correct (AC-10), AC-11 cross-arch promotion to `Cross-arch-verified`. The round ends with the goal-tracker MUTABLE Completed and Verified table recording the cross-arch closure.

### Lower Bound (Minimum Acceptable Scope)

AC-1, AC-2, AC-3, AC-4 (the cross-arch digest equality is **strictly required** — Codex review verified that the kernel's digest output is `out[i] = in[i]` identity with no transcendentals or FP rounding hazards, so digest equality must hold; if it fails, AC-4 + AC-11 BLOCK and the round logs a side issue documenting cross-arch correctness drift), and an attempted AC-5 capture. If AC-6 surfaces typed defects (e.g., dram-specific tolerance mismatch on sm_120 driven by a different per-launch duration distribution under Blackwell boost-clock policies vs Ampere), the round counts as ADVANCED if the defects are typed and traceable; the helper-internal cause is documented for follow-up. The cross-arch promotion (AC-11) defers to a follow-up round if AC-6 doesn't admit cleanly. AC-9 status wording must still be honored.

### Allowed Choices

- Can use: the existing v7 main.cu / Makefile / TOML / tests as-is (no edits); the existing strict-admission helper (no internal modifications); fresh sm_120 preflight manifest generation; the v7 R0 capture procedure as the canonical pattern; native gcc-14 toolchain per the feedback memory; carry-forward of v7 R0 round summary structure for the Two-Device-Model provenance recording.
- Cannot use: any sm_86-host work (this loop is sm_120-only); cloud or datacenter GPU evidence (this loop is laptop-only per CLAUDE.md "Two-Device Local Development Model"); NCU profiling on cloud or server GPUs (per CLAUDE.md Rule 1); driver-state changes (per Rule 2); privileged CUPTI surfaces (per Rule 3 — Activity is allowed); host-driver work (per Rule 4); simulator revival (per Rule 5); conda/mamba/miniforge envs (per the feedback memory `feedback_native_toolchain.md`); modifying any tracked code surface — main.cu / Makefile / TOML / tests / helper / orchestrator (this loop is verification + capture, not implementation); widening the dram_stream_stride_mix ARCH_GATE token set from `sm_120` to `sm_120a` or `sm_120f` (the kernel uses ordinary global loads + identity FP32 output stream, not FP4/NVFP4/MX/block-scaled MMA or any sm_120a-only feature; portable `sm_120` is the correct target per the project's tuning-blackwell-sm120-kernels skill card discipline).

> **Note on Deterministic Designs**: The hook code pattern is fully deterministic (mirrors v7 sm_86 exactly). The round's optional flexibility lives only in the path-boundary band between "rebuild+attempted-capture" (lower) and "rebuild+admitted-capture+cross-arch-promotion" (upper).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. **Bootstrap** — On the laptop:
   - **Read `HANDOFF.md` at the repo root FIRST** before any other action (per CLAUDE.md "Session Handoff" rule: this is the machine-to-machine transfer document and is the only place that pins the project's current state for the receiving session).
   - scp the repo from the originating 3090 Ti host **OR** `git fetch && git checkout 5060_3090ti_cross_dev` (or whatever branch carries `5ea52f3` or its successor)
   - `rm -rf .humanize/rlcr/` per CLAUDE.md cross-host handoff semantics (host-local loop state is not portable; deletion is mandatory before starting a fresh RLCR loop on the receiving host)
   - **Note on auto-memory**: `~/.claude/projects/.../memory/` is **machine-local** and is NOT carried by either repo scp OR git fetch (per CLAUDE.md "Memory at `~/.claude/projects/<project>/memory/` is **machine-local** and is NOT carried by scp or git"). The receiving 5060-laptop session starts with empty auto-memory; do not assume any prior-session memory is available. If the v7 R3 entry `project_rlcr_structural_stall_pattern.md` is desired as planning context, the operator must manually copy it via raw file scp (independent of repo transport).
   - If raw repo scp was used, confirm `.humanize/bitlesson.md` is present (it survives raw repo scp but is gitignored, so a git fetch drops it; if absent and you want it, raw-scp the file from the originating host)
   - If raw repo scp was used, the v7 capture artifacts at `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/` are present; if git fetch, they are absent and AC-8's first positive test reads the v7 documented baseline instead
   - **Do NOT reuse the v7 round-0 preflight manifest** at `.humanize/rlcr/2026-05-08_08-57-02/preflight_manifest.json` — that manifest pins the 3090 Ti host's GPU UUID, driver state, NVBit build hash, and host hostname. A fresh sm_120 preflight is mandatory regardless of transport (and the v7 directory is deleted in step above anyway).

2. **Stale-Artifact Cleanup + Rebuild** — On the laptop, before any rebuild:
   - **Delete any stale build products that may have survived raw scp from the 3090 Ti host but target sm_86** (per CLAUDE.md: "State that does NOT survive any cross-host transport cleanly: the built tracer at `tools/nvbit_tracer/tracer_tool/tracer_tool.so`, the built CUPTI Activity collector at `workloads/simbench_bw/libsimbench_cupti_activity_collector.so`, per-slug compiled binaries"). Run:
     - `rm -f tools/nvbit_tracer/tracer_tool/tracer_tool.so`
     - `rm -f workloads/simbench_bw/libsimbench_cupti_activity_collector.so`
     - `rm -f workloads/simbench_bw/l1/dram_stream_stride_mix/dram_stream_stride_mix_run`
     - **Clear any stale sm_120 timing artifacts** that might mask a partial new run: `rm -rf workloads/simbench_bw/results/dram_stream_stride_mix/<laptop_gpu_sku>/sm_120/timing_00 workloads/simbench_bw/results/dram_stream_stride_mix/<laptop_gpu_sku>/sm_120/timing_timing_00.json` (target the workload's sm_120 directory only; do NOT touch the carried sm_86 captures or other workloads)
   - Now rebuild on the laptop with explicit gcc-14 toolchain:
     - **Tracer** (Makefile honors `NVCC_CCBIN`): `NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool`
     - **Collector** (`workloads/simbench_bw/Makefile` has `NVCC ?= nvcc` and does NOT honor `NVCC_CCBIN`; override `NVCC` directly to inject `-ccbin`): `make -C workloads/simbench_bw cupti-collector-lib COLLECTOR_ARCH=sm_120 NVCC='nvcc -ccbin=/usr/bin/g++-14'`
     - **Dram binary** (per-kernel Makefile includes `common.mk` which has `NVCC ?= nvcc`; same override pattern): `make -C workloads/simbench_bw/l1/dram_stream_stride_mix clean all NVCC='nvcc -ccbin=/usr/bin/g++-14'`
     - **Toolchain sanity-check alternative**: if the laptop's host-default nvcc already resolves ccbin to `/usr/bin/g++-14` (verify via `nvcc -v -dryrun /tmp/empty.cu 2>&1 | grep -i 'compiler-bindir\|cudafe1.gen.c\|gnu_compiler'`), the bare `make ... COLLECTOR_ARCH=sm_120` and `make clean all` invocations work without the `NVCC=...` override. Default to the explicit override unless the operator has confirmed the host default.
   - Verify `cuobjdump --list-elf workloads/simbench_bw/l1/dram_stream_stride_mix/dram_stream_stride_mix_run` reports sm_86 + sm_90a + sm_120 cubins (all three present; the binary is fat per the existing ARCH_GATE token set; sm_120 cubin is the one that runs at execute-time on the laptop, but sm_86 + sm_90a remain present for portability and their absence would indicate a Makefile regression)
   - **Record freshness proof**: SHA-256 each rebuilt artifact (`sha256sum tools/nvbit_tracer/tracer_tool/tracer_tool.so workloads/simbench_bw/libsimbench_cupti_activity_collector.so workloads/simbench_bw/l1/dram_stream_stride_mix/dram_stream_stride_mix_run`) and capture mtimes; record in the round summary alongside `git rev-parse HEAD` so the round contract's "fresh build, fresh capture, against current HEAD" claim is auditable.

3. **Backward-compat smoketest** — On the laptop:
   - Run the binary standalone on `--shape-point trace_00` (no LD_PRELOAD); confirm JSON envelope intact and includes `wall_time_ms` field
   - Run on `--shape-point correctness_00 --output-mode sha256_f32le`; confirm `output_sha256` digest equals the v7 sm_86 baseline `786c10f82133609842e2be4226ff90154a5efde3c2ff001a504dd2367828e782`
   - Re-run the standalone twice on each shape to confirm digest determinism on the laptop

4. **Live capture** — On the laptop:
   - Generate fresh preflight manifest via `tools/cloud/preflight.sh --out <round_root>/preflight_manifest.json`
   - Validate with `tools/cloud/preflight_manifest.py --validate <round_root>/preflight_manifest.json`
   - Run timing-mode capture with `--enable-cupti-activity --preflight-manifest <fresh>`
   - Verify 30 cupti + 30 nvml sidecars + measurement_manifest at the canonical sm_120 results root

5. **Helper admission** — On the laptop:
   - Invoke `admit_cupti_dual_source_live_capture` with the corrected two-argument signature (timing_artifact_path + sidecar_dir, NOT a single root, per the v7 plan's resolved disagreements)
   - Capture the return tuple. If `ok=True` and `defects=[]` and `cupti_sidecar_pass_count=30`, AC-6 closes
   - Run the wrong-slug negative sentinel (e.g., `expected_slug="rmsnorm_fwd_bf16"` against the dram capture) to confirm the helper rejects with all 4 surface defect tokens

6. **Regression sanity** — On the laptop:
   - Run targeted tests: `pytest workloads/simbench_bw/tests/test_dram_stream_stride_mix_hook_static.py workloads/simbench_bw/tests/test_cupti_dual_source_live_admission.py::CuptiDualSourceAdmissionSecondWorkloadParameterizedTest -q` → expect 7 passed
   - Run broad sweep: `pytest tools/ workloads/simbench_bw/tests/ -q` → expect 4 carried baseline failures + no new regressions

7. **Round summary** — On the laptop:
   - Record device + driver + CUDA + NVBit + gcc + observed clocks per Two-Device-Model
   - Record `cuobjdump --list-elf` output
   - Record helper return values
   - First promote `dram_stream_stride_mix/timing_00` to `Verified-live-sm_120` (AC-9)
   - Then combine with carried v7 `Verified-live-sm_86` evidence to declare AC-16 second-workload-eligibility as `Cross-arch-verified` (AC-11)
   - Cite the v7 loop session id (`2026-05-08_08-57-02`) and HEAD (`5ea52f3` or its successor) as the sm_86 evidence anchor

### Relevant References

- `workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu` — target (already-instrumented at HEAD; no edits needed)
- `workloads/simbench_bw/l2/rmsnorm_fwd_bf16/main.cu` — reference pattern (cross-arch proven in earlier work)
- `workloads/simbench_bw/cupti_dual_source_live_admission.py` — strict admission helper (no edits needed)
- `workloads/simbench_bw/cupti_activity_collector.cu` — collector source (rebuild for sm_120)
- `workloads/simbench_bw/tests/test_dram_stream_stride_mix_hook_static.py` — static regression test (no edits needed)
- `workloads/simbench_bw/tests/test_cupti_dual_source_live_admission.py::CuptiDualSourceAdmissionSecondWorkloadParameterizedTest` — parameterized helper test (no edits needed)
- `workloads/simbench_bw/shapes/dram_stream_stride_mix.toml` — shape definition (no edits needed)
- `tools/cloud/preflight.sh` + `tools/cloud/preflight_manifest.py` — preflight pipeline
- `tools/nvbit_tracer/tracer_tool/` — tracer source (rebuild for sm_120)
- `~/.claude/projects/.../memory/project_rlcr_structural_stall_pattern.md` (machine-local; only present if the operator separately raw-scp'd the memory directory from the 3090 Ti host — repo scp/git fetch do NOT carry auto-memory) — documents v7 loop's structural-stall lesson; informs this loop's narrow-scope discipline
- `~/.claude/projects/.../memory/feedback_native_toolchain.md` — native gcc-14 toolchain requirement
- v7 round artifacts: `.humanize/rlcr/2026-05-08_08-57-02/round-0-summary.md` + Codex review results (cited via this plan's prose; not on disk on the laptop unless raw-scp'd)

## Dependencies and Sequence

### Milestones

1. **Milestone 1 — Bootstrap (closes AC-1, AC-2, AC-3)**:
   - Phase A: Repo sync to laptop (scp or git fetch).
   - Phase B: `rm -rf .humanize/rlcr/`; confirm `.humanize/bitlesson.md` state.
   - Phase C: Rebuild tracer + collector + dram binary on laptop.
   - Phase D: Verify cubin set via cuobjdump.
   - No commits in this milestone (all build artifacts are gitignored).

2. **Milestone 2 — Backward-compat (closes AC-4, AC-10)**:
   - Phase A: Standalone smoketest in default json mode; record JSON envelope shape.
   - Phase B: Standalone smoketest in sha256_f32le mode; confirm digest equals sm_86 baseline.
   - Phase C: Trace-mode regression check.
   - Phase D: validate.py --all on the laptop.
   - No commits in this milestone (verification only).

3. **Milestone 3 — Live capture + helper admit (closes AC-5, AC-6)**:
   - Phase A: Fresh sm_120 preflight manifest.
   - Phase B: Timing-mode capture with `--enable-cupti-activity`.
   - Phase C: Helper admission positive + wrong-slug negative sentinel.
   - No commits in this milestone (capture artifacts gitignored).

4. **Milestone 4 — Regression + cross-arch closure (closes AC-7, AC-8, AC-9, AC-11)**:
   - Phase A: Targeted tests.
   - Phase B: Broad sweep.
   - Phase C: Round summary with full Two-Device-Model provenance.
   - Phase D: Goal-tracker promotion to `Verified-live-sm_120` first, then `Cross-arch-verified` after combining with carried v7 sm_86 evidence.
   - No commits unless a follow-up tracked-file change is required (none anticipated in this verification loop).

### Dependency Graph

- M2 depends on M1 (need built binary).
- M3 depends on M1+M2.
- M4 depends on M3.
- AC-11 (cross-arch promotion) depends on AC-9 (`Verified-live-sm_120` recorded first) AND the carried v7 `Verified-live-sm_86` (via raw scp evidence OR git fetch documented baseline).

## Task Breakdown

Each task carries exactly one routing tag.

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Read `HANDOFF.md` first per CLAUDE.md "Session Handoff" rule; scp or `git fetch` + `git checkout 5060_3090ti_cross_dev` (HEAD `5ea52f3` or successor) onto the 5060 laptop; `rm -rf .humanize/rlcr/`; confirm `.humanize/bitlesson.md` and v7 sm_86 capture artifact survival policy per transport (raw scp preserves both; git fetch drops both); auto-memory at `~/.claude/projects/.../memory/` is machine-local and is NOT carried by either transport | AC-1, AC-2 | coding | - |
| task1b | **Stale-artifact cleanup**: `rm -f tools/nvbit_tracer/tracer_tool/tracer_tool.so workloads/simbench_bw/libsimbench_cupti_activity_collector.so workloads/simbench_bw/l1/dram_stream_stride_mix/dram_stream_stride_mix_run`; `rm -rf workloads/simbench_bw/results/dram_stream_stride_mix/<laptop_gpu_sku>/sm_120/timing_00 workloads/simbench_bw/results/dram_stream_stride_mix/<laptop_gpu_sku>/sm_120/timing_timing_00.json` (target sm_120 only; preserve carried sm_86 captures and other workloads) | AC-3, AC-5 | coding | task1 |
| task2 | Rebuild tracer (`NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool`) + collector (`make -C workloads/simbench_bw cupti-collector-lib COLLECTOR_ARCH=sm_120 NVCC='nvcc -ccbin=/usr/bin/g++-14'`) + dram binary (`make -C workloads/simbench_bw/l1/dram_stream_stride_mix clean all NVCC='nvcc -ccbin=/usr/bin/g++-14'`) on laptop. **NOTE**: `NVCC_CCBIN=g++-14` is honored ONLY by the tracer Makefile; the simbench_bw Makefile + per-kernel common.mk use `NVCC ?= nvcc` and require the explicit `NVCC='nvcc -ccbin=...'` override. Toolchain-sanity-check alternative documented in Conceptual Approach step 2 | AC-3 | coding | task1b |
| task3 | Run `cuobjdump --list-elf workloads/simbench_bw/l1/dram_stream_stride_mix/dram_stream_stride_mix_run`; verify sm_86 + sm_90a + sm_120 cubins all present. Record SHA-256 of all rebuilt artifacts (tracer .so, collector .so, dram binary) for AC-5 freshness proof | AC-3 | coding | task2 |
| task4 | Standalone smoketest: `dram_stream_stride_mix_run --shape-point trace_00` (verify JSON envelope + wall_time_ms field) AND `--shape-point correctness_00 --output-mode sha256_f32le` (verify `output_sha256` STRICTLY equals the v7 sm_86 baseline `786c10f82133609842e2be4226ff90154a5efde3c2ff001a504dd2367828e782`). If digest does NOT match, AC-4 BLOCKS, AC-11 BLOCKS, log a cross-arch correctness drift side issue and stop the round | AC-4 | coding | task2 |
| task5 | `python3 workloads/simbench_bw/validate.py --all` on laptop | AC-10 | coding | task1 |
| task6 | Generate fresh sm_120 preflight: `tools/cloud/preflight.sh --out <round_root>/preflight_manifest.json`; validate with `tools/cloud/preflight_manifest.py --validate <fresh>`. Do NOT reuse the v7 R0 preflight (it pins 3090 Ti host state; the v7 round directory is deleted by task1 anyway) | AC-5 | coding | task1 |
| task7 | Run timing-mode live capture: `python3 workloads/simbench_bw/run_suite.py --mode timing --kernel dram_stream_stride_mix --shape-point timing_00 --enable-cupti-activity --preflight-manifest <fresh>`; verify 30 cupti + 30 nvml sidecars + shape-level `timing_00/measurement_manifest.json` emit | AC-5 | coding | task2, task6 |
| task8 | Invoke `admit_cupti_dual_source_live_capture(timing_artifact_path, sidecar_dir, expected_slug="dram_stream_stride_mix", expected_shape_point="timing_00", collect_defects=True)`; verify positive admission `ok=True 30/30 0-defect`; run wrong-slug negative sentinel; run synthetic-fixture forced-failure sentinel (mutate one `dual_source_reconciliation_passed` to false; verify rejection) | AC-6 | coding | task7 |
| task9 | If AC-6 surfaces typed defects on sm_120 (e.g., dram-specific tolerance mismatch driven by Blackwell boost-clock policy differences), decide tolerance treatment: prefer reusing rmsnorm-calibrated tolerance; only run `calibrate_dual_source_tolerance.py` against dram if a Blackwell-specific recalibration is justified by named typed defects (deferred follow-up round if recalibration is needed) | AC-6 | analyze | task8 |
| task10 | Run targeted tests: `pytest test_dram_stream_stride_mix_hook_static.py + ::CuptiDualSourceAdmissionSecondWorkloadParameterizedTest -q`; verify 7 passed (4 static + 3 parameterized) | AC-7 | coding | task1 |
| task11 | Run broad sweep `pytest tools/ workloads/simbench_bw/tests/ -q`; verify no NEW failures BEYOND the v8-R0-entry-recorded baseline (which may differ from the v7 4-failure baseline if the receiving session pulls a successor HEAD with hygiene fixes) | AC-8 | coding | task10 |
| task12 | Verify carried v7 sm_86 evidence per transport: raw-scp invokes helper against on-disk sm_86 capture (must still admit `ok=True 30/30 0-defect`); git-fetch cites v7 R0/R1 documented baseline (operator should pre-position a compact v7 evidence manifest outside `.humanize/rlcr/` before deletion if git-fetch transport is chosen) | AC-8 | analyze | task1 |
| task13 | Write round summary recording laptop device + driver + CUDA + NVBit + gcc + clocks + cuobjdump output + freshness-proof SHA-256s + helper return values; FIRST promote `dram_stream_stride_mix/timing_00` to `Verified-live-sm_120` (AC-9); THEN combine with carried v7 `Verified-live-sm_86` to record AC-16 second-workload-eligibility as `Cross-arch-verified` (AC-11). The two-step promotion ensures Codex review can verify each leg independently | AC-9, AC-11 | coding | task8, task9, task11, task12 |

## Claude-Codex Deliberation

### Agreements

- The narrow-scope sm_120-only framing pre-empts the v7 structural-stall trap. The plan body explicitly inverts the trap: this loop has no sm_86 deferral baked in.
- No tracked code surfaces are modified in this loop; the v7 `5ea52f3` chain already carries all the substantive code.
- The strict admission helper's contract (DEFECT_TOKENS = 20 entries, 4-surface identity from v7 R22 widening) is regression-locked and applies identically on sm_120.
- Cross-arch correctness invariance is a strict AC-4 requirement: same kernel input set must produce the same `output_sha256` digest on sm_86 and sm_120 (the digest hashes only the output byte stream, not timing data).
- Status wording must distinguish "Verified-live-sm_120" (AC-9, this loop's evidence alone) from "Cross-arch-verified" (AC-11, combining with carried v7 sm_86 evidence). The promotion is two-step.
- Transport-aware AC-8: raw scp vs git fetch produce different evidence states; the AC accommodates both and explicitly documents which test paths apply per transport.
- The receiving 5060-laptop session is responsible for documenting which device produced the live evidence (per CLAUDE.md item 5: "Each session is responsible for documenting which device produced the live evidence in the round summary").
- BL-20260508-post-commit-recapture-discipline applies trivially: HEAD is stable across the loop (no new substantive commits planned) so source_commit pinning is degenerate.
- BL-20260509-strict-admission-contract-first applies: the helper is regression-locked and exercised positive + negative on sm_120 as it was on sm_86.

### Resolved Disagreements

- **Loop exit primary path**: Claude's draft v1 said the operator should immediately run `/humanize:cancel-rlcr-loop` after Codex returns ADVANCED with cross-arch promotion landed. Codex's first-pass review pointed out this contradicted the canonical loop semantics — the loop SHOULD reach `COMPLETE` naturally if all plan-scoped ACs are closed AND no mainline deferral remains. Resolution: Loop Exit Discipline rewritten to make `COMPLETE` the primary path and `/humanize:cancel-rlcr-loop` the operator-only fallback for stop-hook misclassification edge cases. Claude/Codex must NOT execute cancel itself. Rationale: the v8 plan's structural-stall pre-emption (sm_120-only mainline, no host-constrained deferrals) makes natural `COMPLETE` achievable; cancel is only needed if the stop-hook misfires.

- **AC-4 cross-arch digest equality strictness**: Claude's draft v1 made digest equality optional in the lower bound. Codex's first-pass review verified the kernel code (`workloads/simbench_bw/l1/dram_stream_stride_mix/dram_stream_stride_mix.cu` lines 25, 35) — output is identity `out[i] = in[i]` and the only FP accumulation writes `sink[0]` which is NOT included in the SHA-256 hash. Therefore digest equality is mathematically required; failure indicates real cross-arch correctness drift, not legitimate Blackwell-vs-Ampere variance. Resolution: AC-4 + lower bound updated to make digest equality strictly required. If it fails, AC-4 + AC-11 BLOCK and the round logs a side issue.

- **Task7 dependency on Task2**: Claude's draft v1 had `task7` (live capture) depending only on `task6` (preflight). Codex flagged that live capture cannot run against possibly-stale raw-scp binaries; rebuild (task2) must complete first. Resolution: `task7 Depends On` updated to `task2, task6`.

- **Task13 dependency on AC-6 admission**: Claude's draft v1 had `task13` (round summary + cross-arch promotion) depending only on tests (task11) + sm_86 verification (task12). Codex flagged that the cross-arch promotion (AC-11) cannot land if the helper admission (task8) didn't succeed. Resolution: `task13 Depends On` updated to `task8, task9, task11, task12` so the round summary cannot record `Verified-live-sm_120` without confirmed AC-6 closure.

- **Stale-artifact cleanup**: Claude's draft v1 had no explicit cleanup task before rebuild. Codex flagged that CLAUDE.md "Session Handoff" semantics make tracer .so / collector .so / per-slug binaries stale after raw scp from a different-arch host, and stale sm_120 timing artifacts could mask a partial new run. Resolution: added `task1b` for stale-artifact cleanup with explicit `rm -f` commands and a sm_120-only timing-artifact directory clear (preserving carried sm_86 captures and other workloads).

- **gcc-14 build commands**: Claude's draft v1 used `NVCC_CCBIN=g++-14` for both tracer and collector + dram binary. Codex inspected `workloads/simbench_bw/Makefile` line 30 + `workloads/simbench_bw/common.mk` line 25 and found that only the tracer Makefile honors `NVCC_CCBIN`; the simbench_bw + per-kernel Makefiles use `NVCC ?= nvcc` which doesn't pick up `NVCC_CCBIN`. Resolution: build commands updated to use `NVCC='nvcc -ccbin=/usr/bin/g++-14'` override for collector + dram binary; tracer keeps `NVCC_CCBIN=g++-14`. A toolchain-sanity-check alternative is documented for cases where the host-default nvcc already resolves ccbin to g++-14.

- **AC-5 manifest path location**: Claude's draft v1 listed `measurement_manifest.json` ambiguously at the parent results root. Codex inspected the helper + orchestrator and confirmed the canonical path is shape-level (`<sm120>/timing_00/measurement_manifest.json`). Resolution: AC-5 positive-test layout updated to make the shape-level path explicit.

- **AC-7 + AC-10 missing negative tests**: Claude's draft v1 omitted negative-test subsections for AC-7 and AC-10 (carry-forward implied but not explicit). Codex flagged the gap. Resolution: explicit negative tests added for both ACs (AC-7: missing-substring + forced-failure-rejection regression locks; AC-10: contract-mutation rejection + transport-invariance).

- **AC-6 negative-test breadth**: Claude's draft v1 had only the wrong-slug negative sentinel. Codex flagged the missing forced-failure (value-axis) negative — the R20→R21 regression that BL-20260509 was extracted to prevent. Resolution: AC-6 negative now includes a synthetic-fixture forced-failure sentinel that mutates `dual_source_reconciliation_passed` to false and verifies the helper rejects.

- **Bootstrap ordering + auto-memory wording**: Claude's draft v1 didn't explicitly call out `HANDOFF.md` first-read; the auto-memory wording incorrectly suggested the entry survives raw scp. Codex flagged both. Resolution: bootstrap step 1 now says "Read `HANDOFF.md` FIRST" per CLAUDE.md "Session Handoff" rule; auto-memory note clarifies that `~/.claude/projects/.../memory/` is machine-local independent of transport, requiring separate raw file scp if the operator wants the v7 R3 entry.

- **v7 preflight reuse**: Claude's draft v1 was ambiguous about whether the v7 round-0 preflight manifest could be reused. Codex flagged that it pins 3090 Ti host state and must NOT be reused. Resolution: bootstrap explicitly says "Do NOT reuse the v7 round-0 preflight manifest" + rationale; the v7 directory is deleted in step 1 anyway.

- **AC-4 trace-mode demotion**: Claude's draft v1 listed `--phase-inject-ab` trace-mode admission as a positive test for AC-4 closure. Codex flagged that this risks turning the CUPTI timing loop into a broader Blackwell NVBit trace loop if the trace-mode harness has sm_120-specific issues. Resolution: trace-mode is moved to "Optional non-blocking sanity" outside the AC-4 positive tests; failures there are side issues, not AC-4 blockers, because the AC-16 contract is timing-mode + CUPTI Activity, not trace-mode + NVBit.

- **sm_120a/sm_120f anti-goal**: Codex's first-pass review suggested adding an explicit anti-goal that the workload's ARCH_GATE token set must NOT widen to sm_120a/sm_120f because the kernel uses ordinary global loads + identity FP32 output (no FP4/NVFP4/MX/block-scaled MMA or sm_120a-only features). Resolution: anti-goal added to "Cannot use" in Allowed Choices.

- **AC-5 freshness proof**: Codex's first-pass review suggested recording SHA-256 + mtime + capture timestamp + source_commit for rebuilt artifacts so the round contract's "fresh build, fresh capture, against current HEAD" claim is auditable. Resolution: AC-5 freshness-proof positive test + negative test added; task3 records the SHA-256s; task13 includes them in the round summary.

- **AC-8 branch-aware wording**: Codex's first-pass review suggested making AC-8's broad-sweep failure-count gate branch-aware ("4 failures at HEAD `5ea52f3`, unless successor HEAD has changed the baseline"). Resolution: AC-8 wording updated to record the observed baseline at v8 R0 entry as the regression anchor, not pin the literal 4-failure count.

### Convergence Status

- Final Status: `converged after one Codex review pass`
- Convergence Rounds Executed: 1 (Codex first-pass comprehensive review against the v8 draft v1; all 9 REQUIRED CHANGES + 3 SUGGESTED IMPROVEMENTS folded into v8 draft v2; the 3 OPEN QUESTIONS are addressed in `## Pending User Decisions`).
- All Codex `REQUIRED CHANGES` from the first-pass review have been incorporated into the v8 plan v2. The 3 SUGGESTED IMPROVEMENTS (sm_120a/sm_120f anti-goal, branch-aware AC-8, AC-5 freshness proof) are also incorporated.
- The 3 OPEN QUESTIONS are documented in `## Pending User Decisions` items 1, 2, 3, 4 (with the v7-evidence-manifest pre-position as item 4 conditional on transport choice).
- All 5 AGREEMENTS from Codex's review remain unmodified in the v8 plan v2.

## Pending User Decisions

1. **Repo transport**: raw scp vs git fetch. Either is acceptable; the plan accommodates both. Raw scp preserves the v7 sm_86 capture artifacts and `.humanize/bitlesson.md`; git fetch loses both. Auto-memory at `~/.claude/projects/.../memory/` is **machine-local independent of transport** and is NOT carried by either; the v7 R3 entry `project_rlcr_structural_stall_pattern.md` must be raw-scp'd separately if desired. **Recommendation**: prefer raw scp from the 3090 Ti host so the v7 sm_86 capture is on-disk for direct AC-8 helper re-invocation; if git fetch is chosen, pre-position a compact v7 evidence manifest (JSON with helper-return values + capture-directory SHA-256) outside `.humanize/rlcr/` before the v7 directory is deleted, so AC-8's documented-baseline anchor survives.

2. **Branch successor strategy**: if new commits land on the originating host between v7 loop close and 5060 pivot, should the laptop pick up `5ea52f3` exactly, or the latest tip of `5060_3090ti_cross_dev`? **Recommendation**: latest tip if the new commits are all non-substantive (e.g., tracked-doc hygiene from a different loop, plan file edits, or test fixes) AND the v7 sm_86 work surfaces (main.cu, Makefile, TOML, regression tests, helper) are byte-identical at that tip; pin to `5ea52f3` if there is any uncertainty about whether the successor commits could change the AC-2/AC-7/AC-10 verification results. The receiving session must verify the v7 sm_86 work surfaces at the chosen HEAD before launching the v8 loop. Note that this loop's anti-goals exclude tracked-doc hygiene work in the v8 loop itself, but accepting pre-existing successor-doc commits made on the originating host outside any RLCR loop is fine if they don't disturb the AC-2/AC-7/AC-10 surfaces.

3. **Laptop toolchain pre-flight**: confirm the 5060 laptop host has CUDA Toolkit ≥13.2 (or whatever release contains `nvcc` with `sm_120` codegen support; CUDA 13.2 is the originating-host version that this plan was tested against), `/usr/bin/g++-14` is installed and is the chosen `-ccbin` target per the `feedback_native_toolchain.md` memory, and the laptop's `nvidia-smi` reports the RTX 5060 device with compute capability `12.0` (sm_120). If any of these prerequisites are missing, the v8 loop will fail at bootstrap or build time, NOT at the AC-logic time. The operator should run a manual pre-flight check before launching the loop:
   ```
   nvcc --version | grep -E '13\.[2-9]|1[4-9]\.'
   nvidia-smi --query-gpu=name,compute_capability,driver_version --format=csv
   /usr/bin/g++-14 --version
   ls /opt/cuda/extras/CUPTI/{include,lib64} 2>&1 | head
   ```
   and confirm sm_120 is in `nvcc --help | grep -A 5 'gpu-architecture'` output.

4. **v7 evidence manifest pre-position** (only relevant if git-fetch transport is chosen, dependent on decision 1): if the operator picks git fetch, before running `rm -rf .humanize/rlcr/` on the laptop, copy a compact JSON capturing the v7 R0 helper-return values + sm_86 capture directory SHA-256s to a non-gitignored / non-rlcr location (e.g., `~/v7_sm86_evidence_manifest.json` on the laptop), so AC-8's "documented baseline" anchor is on-disk and citable in the v8 round summary.

## Implementation Notes

### Code Style Requirements (carry-forward from v7 plan)

- Implementation code (`.cu`, `.py`, `.sh`) and comments must NOT contain plan-specific terminology such as "AC-N", "Milestone N", "Step N", "Phase N", "Round N", "task1..N", or similar workflow markers. These belong in plan documentation only.
- Test docstrings and assertion messages follow the same rule (per the existing `tools/test_plan_token_scanner.py` discipline).

### Two-Device Local Development Model Reminders

- This loop is sm_120-only on the laptop. The cross-arch promotion (AC-11) requires the sm_86 evidence to be either present on disk (raw scp) OR documented from the v7 loop (git fetch); either is acceptable per the transport-aware AC-8.
- The hook code is arch-independent host-side glue. The sm_120 capture exercises the same dlsym + cudaEvent contract that the sm_86 capture exercised; observable differences would be timing-only (different boost clocks, different SASS encodings, different throttle behaviors), not behavioral. Per CLAUDE.md item 1: "Never assume an opcode set, register count, SASS layout, helper-function inlining policy, or compression-format distribution is constant across architectures."
- The session must record on which device the live evidence was produced per CLAUDE.md item 5; this plan's AC-9 enforces the discipline.

### Critical Rules Reminders (carry-forward from CLAUDE.md)

- No NCU on cloud/server (this loop is laptop-local; NCU is admissible per Rule 1 local-label exception ONLY through `tools/local_labels/` and `data/local_labels/`, which this plan does NOT invoke; Activity is the canonical timing surface here).
- No driver-state changes (no `nvidia-smi` clock locks, persistence, MIG, compute-mode toggles, or kernel-module parameter edits).
- No privileged CUPTI surfaces (Events, Metrics, Range Profiling, PM Sampling, PerfWorks). CUPTI Activity is allowed.
- No host-driver work (no driver install, DKMS, MOK, modprobe, or reboot workflows).
- No simulator revival (no `nvasim`, Accel-Sim, or simulator-validation plans).

### Pre-empting the v7 Structural-Stall Trap

Per the auto-memory entry `project_rlcr_structural_stall_pattern.md` from v7 R3 (preserved on the originating host; survives raw scp; absent on git fetch), the v7 loop entered a structural stall because its plan baked in a host-constrained deferral against a single-arch host. THIS plan inverts the trap: the v8 loop is sm_120-only and does NOT carry any sm_86 deferral baked in. The sm_86 evidence is either preserved on disk or documented as a regression baseline; in neither case is sm_86 a deferred goal. The loop should reach `COMPLETE` after AC-11 promotion without structural stall.

If the goal-tracker stop-hook nevertheless treats some sm_86 reverification as a deferred item, that would be the v8 analog of the v7 trap and the round summary should diagnose it the same way and request operator cancellation per the v7 R4 Plan Evolution Log entry pattern.

### Loop Exit Discipline (NEW for this loop, learned from v7)

The canonical exit for the v8 loop is to **let the stop-hook emit `COMPLETE`** once Codex returns `ADVANCED` with the cross-arch promotion (`Cross-arch-verified`) landed AND no mainline deferral remains. This is the primary path; the v8 plan is designed to be free of host-constrained deferrals (sm_120 IS the mainline; sm_86 is regression-locked baseline, not deferred work) so this should succeed.

`/humanize:cancel-rlcr-loop` is the **fallback only**, invoked manually by the operator if the stop-hook still returns `ADVANCED`/`STALLED` despite all plan-scoped ACs being closed (e.g., due to carried side issues like the 4 baseline test failures that the stop-hook misclassifies as deferred work). Claude/Codex must **not** execute the cancel command itself — that is operator-only per the system-prompt rule against unilateral exit attempts.

The v7 loop's structural-stall pattern arose because the plan baked in an explicitly-deferred cross-arch leg AGAINST a single-arch host (the loop could not reach `COMPLETE` from the originating host because the deferral target was unreachable). The v8 plan inverts this by being sm_120-only with no deferrals against the laptop host; the loop should reach `COMPLETE` cleanly. If it does not, that's a v8 analog of the v7 trap and the round summary should diagnose it as such (the carried sm_86 evidence-only side issues OR the 4 carried baseline test failures may be misclassified by the stop-hook as deferred), and the operator's fallback cancel is the canonical resolution.

## Output File Convention

This plan is the input to `/humanize:gen-plan` for refinement. The intended output filename is `plans/ac16_dram_stream_cupti_hook_sm120_plan.md` (this file). No translated variant is configured.
