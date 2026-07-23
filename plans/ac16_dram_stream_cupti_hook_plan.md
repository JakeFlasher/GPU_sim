# AC-16 Producer-Side dlsym Hook Round For dram_stream_stride_mix On sm_86

## Goal Description

Add a `simbench_cupti_record_cuda_event_wall_time_ms` dlsym hook plus full cudaEvent wall-clock instrumentation to `workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu`, so the workload becomes AC-16-eligible for dual-source CUPTI/CUDA-event reconciliation on RTX 3090 Ti sm_86.

This is the recovered mainline objective for the next fresh RLCR loop after the v3 loop's R31-R32-R33 STALLED chain (Mainline Drift Circuit Breaker fired). The work mirrors the proven reference pattern at `workloads/simbench_bw/l2/rmsnorm_fwd_bf16/main.cu` lines 117-139 + the `-ldl` link in that file's Makefile, then captures fresh sm_86 live evidence and admits via the existing strict-admission helper. The round is scoped tightly to one AC seam — no batching of queued side issues, no doc-hygiene work, no operator-sanction work, no 5060-laptop pivot. Cross-arch promotion to `Cross-arch-verified` is explicitly deferred to a future 5060-laptop session per the Two-Device Local Development Model.

At original plan inception (before the v6 RLCR loop's Round 0 work landed at HEAD `eaf5eee`/`30ae840`/`1050428`): dram_stream_stride_mix's main.cu launched and synchronized the kernel without any cudaEvent timing or dlsym hook, so captures via `--mode timing --enable-cupti-activity` produced sidecars with `cuda_event_wall_time_ms=0` and `dual_source_reconciliation_passed=false` deterministically. The v6 R0 implementation reverses that deterministic failure and unlocks dram_stream_stride_mix as a second AC-16 producer alongside `rmsnorm_fwd_bf16/timing_00`. A fresh loop reading this plan should treat the AC-2 through AC-10 substantive evidence (hook + Makefile + TOML + tests + capture + helper admission) as ALREADY LANDED at HEAD; the new loop's Round 0 should re-validate the existing evidence rather than re-create it.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Reference and target paths in plan, contract, and round summary match the actual filesystem.
  - Positive Tests (expected to PASS):
    - `test -f workloads/simbench_bw/l2/rmsnorm_fwd_bf16/main.cu` returns rc=0 (reference)
    - `test -f workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu` returns rc=0 (target)
    - The round contract / plan / summary cite both paths verbatim with `l2/` and `l1/` respectively
  - Negative Tests (expected to FAIL):
    - `test -f workloads/simbench_bw/l1/rmsnorm_fwd_bf16/main.cu` returns non-zero (the legacy gen-idea-draft probe path; the file does not exist on the filesystem)
    - No reference-path or target-path citation in the round contract / plan / summary uses the legacy wrong `l1/rmsnorm_fwd_bf16/...` prefix; that path appears in tracked artifacts only as the explicit negative-control probe descriptor (the actual reference for the rmsnorm hook pattern lives at `l2/rmsnorm_fwd_bf16/main.cu`)

- AC-2: Source-level instrumentation lands in `workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu`.
  - Positive Tests (expected to PASS):
    - File contains `#include <dlfcn.h>` (currently absent)
    - File contains `#include <math.h>` (or `<cmath>`) for the `isfinite` guard
    - File contains a typedef matching the reference: `typedef void (*simbench_cupti_record_fn)(double);` with a docstring block describing the LD_PRELOAD contract and the no-op fallback
    - File contains `cudaEventCreate(&kernel_start)` and `cudaEventCreate(&kernel_stop)` calls before the kernel launch
    - File contains `cudaEventRecord(kernel_start, 0)` BEFORE `dram_stream_stride_mix_launch(...)` and `cudaEventRecord(kernel_stop, 0)` AFTER
    - File contains `cudaEventSynchronize(kernel_stop)` and `cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop)` calls
    - File contains a finite-positive guard: equivalent to `if (!isfinite(wall_time_ms) || wall_time_ms <= 0.0) fail("non-finite-or-non-positive wall_time_ms", 0);`
    - File contains `dlsym(RTLD_DEFAULT, "simbench_cupti_record_cuda_event_wall_time_ms")` with a null-check before the call
    - File contains `cudaEventDestroy(kernel_start)` and `cudaEventDestroy(kernel_stop)` cleanup calls
    - Both stdout JSON paths (default `json` mode AND `sha256_f32le` mode) emit `"wall_time_ms": <printf-double>` with sufficient precision (matching rmsnorm's `%.9f`)
  - Negative Tests (expected to FAIL):
    - The dlsym call site does NOT pass an unguarded function-pointer (no missing null-check)
    - The cudaEvent bracket does NOT enclose any non-launch operations (e.g., D2H copy is OUTSIDE the bracket; only the kernel launch is timed)
    - The wall_time_ms field is NOT emitted before the `cudaEventElapsedTime` call (no use-before-init pattern)

- AC-3: Build produces a working binary.
  - Positive Tests (expected to PASS):
    - `workloads/simbench_bw/l1/dram_stream_stride_mix/Makefile` link line includes `-lcudart -ldl` (was `-lcudart` only)
    - `make -C workloads/simbench_bw/l1/dram_stream_stride_mix clean all` exits rc=0
    - `cuobjdump --list-elf workloads/simbench_bw/l1/dram_stream_stride_mix/dram_stream_stride_mix_run` lists cubins for sm_86, sm_90a, and sm_120 (the existing ARCH_GATE token set)
  - Negative Tests (expected to FAIL):
    - Build with the original `-lcudart`-only link line FAILS with an undefined-reference linker error for `dlsym`
    - `cuobjdump` does NOT report missing or unexpected cubins outside the ARCH_GATE token set

- AC-4: Backward compatibility — non-CUPTI execution and sha256_f32le digest are preserved.
  - Positive Tests (expected to PASS):
    - Standalone run `dram_stream_stride_mix_run --shape-point trace_00` (no LD_PRELOAD) exits rc=0 and emits a JSON object that includes the new `wall_time_ms` field plus all pre-existing fields (`schema_version`, `slug`, `shape_set`, `shape_point`, `output`)
    - Standalone run `dram_stream_stride_mix_run --shape-point correctness_00 --output-mode sha256_f32le` (no LD_PRELOAD) exits rc=0 and emits the same `output_sha256` hex digest as a pre-hook reference run on the same shape (digest depends only on `output` byte stream; new wall_time_ms field is informational and ignored by the SHA-256 hash)
    - Existing trace-mode capture `python3 workloads/simbench_bw/run_suite.py --mode trace --kernel dram_stream_stride_mix --shape-point trace_00 --phase-inject-ab` admits unchanged (AC-15 / AC-4 contract preserved)
  - Negative Tests (expected to FAIL):
    - Standalone run does NOT emit a different `output_sha256` digest from a pre-hook reference run on the same shape (the digest is byte-identical; only the JSON envelope grew)
    - Standalone run does NOT crash, hang, or emit stderr error markers when the dlsym lookup returns nullptr

- AC-5: Fresh sm_86 timing-mode capture.
  - Positive Tests (expected to PASS):
    - A fresh preflight manifest is generated in this round at `<round_root>/preflight_manifest.json` (NOT a stale `/tmp/simbench_preflight.json`); the manifest admits via `tools/cloud/preflight_manifest.py` validation
    - `python3 workloads/simbench_bw/run_suite.py --mode timing --kernel dram_stream_stride_mix --shape-point timing_00 --enable-cupti-activity --preflight-manifest <round_root>/preflight_manifest.json` exits rc=0
    - Capture root `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/` contains `timing_timing_00.json` (timing artifact at parent level), `timing_00/cupti_activity_<i>.json` for i in 0..29 (30 sidecars), `timing_00/nvml_covariate_<i>.json` for i in 0..29 (30 sidecars), and `measurement_manifest.json` at the shape-level path
    - The timing artifact's `dual_source_reconciliation_failures` array is empty (no failed repeats)
  - Negative Tests (expected to FAIL):
    - Capture does NOT produce 0 sidecars or fewer than `repeat_iters=30` sidecars
    - Capture does NOT proceed if the preflight manifest's NVML metadata pin verifier reports rejection
    - The timing artifact does NOT contain non-empty `dual_source_reconciliation_failures`

- AC-6: Strict admission helper accepts the fresh evidence.
  - Positive Tests (expected to PASS):
    - Helper invocation `from workloads.simbench_bw.cupti_dual_source_live_admission import admit_cupti_dual_source_live_capture; result = admit_cupti_dual_source_live_capture(timing_artifact_path=<root>/timing_timing_00.json, sidecar_dir=<root>/timing_00, expected_slug="dram_stream_stride_mix", expected_shape_point="timing_00", collect_defects=True)` returns `result.ok=True`
    - `result.cupti_sidecar_pass_count == 30 and result.cupti_sidecar_total == 30`
    - `result.defects` is the empty list (zero typed defects)
    - `result.timing_failure_count == 0`
  - Negative Tests (expected to FAIL):
    - Helper does NOT return `ok=True` when invoked with `expected_slug="rmsnorm_fwd_bf16"` against the dram capture (slug mismatch must surface as a typed defect)
    - Helper does NOT silently pass when any sidecar has `dual_source_reconciliation_passed=false` (the strict contract is preserved)

- AC-7: Targeted regression coverage.
  - Positive Tests (expected to PASS):
    - New test `workloads/simbench_bw/tests/test_dram_stream_stride_mix_hook_static.py` exists and exercises a Python content-scan of `workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu`, asserting presence of: `#include <dlfcn.h>`, `cudaEventElapsedTime`, `wall_time_ms` printf format, and `dlsym(RTLD_DEFAULT, "simbench_cupti_record_cuda_event_wall_time_ms"` literal substring
    - `python3 -m pytest workloads/simbench_bw/tests/test_dram_stream_stride_mix_hook_static.py -q` exits rc=0 with at least 4 passing assertions (one per substring above)
    - New test in `workloads/simbench_bw/tests/test_cupti_dual_source_live_admission.py` parameterizes the helper to also admit dram_stream_stride_mix/timing_00 against a synthetic fixture (preferred) OR against a committed live-evidence fixture (acceptable fallback); existing `_DEFAULT_SLUG="rmsnorm_fwd_bf16"` rmsnorm test cases continue to pass unchanged
  - Negative Tests (expected to FAIL):
    - The static test does NOT pass when any one of the four required substrings is absent from the source file
    - The parameterized helper test does NOT pass when the fixture's `dual_source_reconciliation_passed` field is forced to false (the helper must reject)

- AC-8: Existing evidence remains coherent (no regression on prior AC closures).
  - Positive Tests (expected to PASS):
    - `python3 -m pytest tools/ workloads/simbench_bw/tests/ -q` exits with no unexpected NEW failures or skips relative to the round-entry baseline. The round-entry baseline at the v6 RLCR cancelled-loop HEAD `1050428` carries four pre-existing failures inherited from earlier work (`tools/test_runbook_command_verification.py::SimbenchReadmeRunSuiteTimingDocumentedCommandTest::test_documented_in_simbench_readme` and three `workloads/simbench_bw/tests/test_handle_timing_dual_source_hard_gate.py` classification cases); these are queued side issues and do NOT count as new regressions if they remain failing in the new loop's broad sweep
    - Helper invocation against the carried R23 rmsnorm/timing_00 artifact still returns `ok=True, cupti_sidecar_pass_count=30, cupti_sidecar_total=30, defects=[]`
    - AC-15 combined pairing index helper (`workloads/simbench_bw/dual_workload_pairing_index.py`) still reports the durable invariants `total=25` entries and `zero_fill_source_commits=0`. The `at_head` reading reads `at_head=15/25` at v6 cancelled-loop HEAD `1050428` (drift documented as the post-commit-recapture-discipline effect of bumping HEAD past v3-era source_commit pins on simbench-side captures); a new loop must NOT pin a literal `at_head=N/25` ratio because HEAD advances bump it
  - Negative Tests (expected to FAIL):
    - The broad test sweep does NOT introduce new test count regressions (specific test_count integers are not pinned; the gate is "no unexpected failures or skips")
    - The carried R23 rmsnorm helper invocation does NOT degrade to `ok=False` after this round's commits land

- AC-9: Status wording reflects sm_86-only promotion (not Cross-arch-verified).
  - Positive Tests (expected to PASS):
    - Round summary names device (RTX 3090 Ti), driver version, CUDA toolkit version, NVBit build hash, native gcc version, observed preflight clocks/probes (graphics MHz, memory MHz, temperature, power) per CLAUDE.md "Two-Device Local Development Model"
    - Round summary records `cuobjdump --list-elf` output for the rebuilt dram_stream_stride_mix_run binary
    - Goal-tracker mutable section records this as `Verified-live-sm_86` for AC-16-second-workload-eligibility, NOT `Cross-arch-verified`
    - Goal-tracker keeps the active 5060-laptop replay row open and unchanged in scope
  - Negative Tests (expected to FAIL):
    - Goal-tracker does NOT mark this AC closure as `Cross-arch-verified`
    - Round summary does NOT use absolute-completion-gate wording (no "the only", "sole", "completes" without scope-bounding) per the R25/R27 retraction discipline
    - HANDOFF.md does NOT have its "Three-Fold Completion Gate" or AC closure table modified to imply path-2 (5060 replay) is unblocked

- AC-10: TOML launch_config contract repair (small in-scope fix).
  - Positive Tests (expected to PASS):
    - `workloads/simbench_bw/shapes/dram_stream_stride_mix.toml` `timing_00.launch_config` matches the actual launcher: grid `[1, 1, 1]` and block `[32, 1, 1]` (the launcher uses these per the `dram_stream_stride_mix_launch` implementation, not the previously-listed `[256,1,1]` / `[128,1,1]` values)
    - Trace-budget verifier and any pairing/estimation consumers admit the corrected launch_config without errors
    - `python3 workloads/simbench_bw/validate.py --all` exits rc=0
  - Negative Tests (expected to FAIL):
    - Validator does NOT silently accept the prior incorrect `[256,1,1]` / `[128,1,1]` values when they conflict with the launcher's actual grid/block (the contract surface is meaningful)

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes all 10 acceptance criteria closed in a single round: source-level instrumentation in dram_stream_stride_mix/main.cu (AC-2), Makefile -ldl link (AC-3), backward-compat smoketest (AC-4), fresh sm_86 timing-mode live capture with 30 admitted sidecars (AC-5), strict-admission helper invocation returning `ok=True 30/30 0-defect` (AC-6), targeted regression test coverage including a static-content-scan test (AC-7), broad sweep with no regression on existing rmsnorm AC-16 evidence (AC-8), round summary with full Two-Device-Model provenance (AC-9), and TOML launch_config repair (AC-10). The round ends with goal-tracker promoting dram_stream_stride_mix to `Verified-live-sm_86` for AC-16-second-workload-eligibility and queueing the 5060-laptop sm_120 leg for cross-arch closure.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes AC-1 (path correctness), AC-2 (source-level instrumentation), AC-3 (Makefile link), and AC-4 (backward compat) all closed AND a fresh sm_86 timing-mode live capture is attempted (AC-5). If AC-6 strict-admission helper invocation surfaces typed defects (e.g., dram-specific tolerance mismatch driven by a different per-launch duration distribution), the round still counts as ADVANCED if the defects are typed/traceable and the helper-internal cause is documented in the round summary as the next round's recovery target; the round contract calls this out explicitly so Codex can classify accordingly. AC-9 status wording must still be honored; AC-10 TOML repair may be deferred if scope feels tight.

### Allowed Choices

- Can use: the rmsnorm_fwd_bf16 reference pattern verbatim (typedef name, dlsym call site, JSON field name `wall_time_ms`); the existing `cupti_dual_source_live_admission.py` helper as-is (no internal modifications); Python content-scan tests (regex/substring only, no AST parsing of .cu sources); the existing rmsnorm-calibrated dual-source tolerance fixture (no recalibration unless AC-6 surfaces a tolerance defect).
- Cannot use: a different dlsym mechanism or hook signature (must mirror `simbench_cupti_record_cuda_event_wall_time_ms` exactly); test parameterization that changes existing rmsnorm test default behavior; tools/cloud/README.md edits (Codex R31 prohibition); batched closure of queued side issues 1, 2, 6, 7 (Drift Circuit Breaker risk); any 5060-laptop sm_120 capture or `Cross-arch-verified` promotion claim.

> **Note on Deterministic Designs**: The hook code pattern is fully deterministic (mirrors rmsnorm exactly). The round's optional flexibility lives only in the path-boundary band between "code+attempted-capture" (lower) and "code+admitted-capture+TOML+test" (upper).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. **Source instrumentation** — open `workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu`. Add headers `#include <dlfcn.h>` and `#include <math.h>` near the existing CUDA headers (after line 3). Add the typedef + docstring block before `main()` (mirroring the reference's lines 31-40). In `main()`, immediately before the `dram_stream_stride_mix_launch(...)` call (current line 172), insert cudaEvent creation + record(start). Immediately after the launch and before the existing `cudaDeviceSynchronize` (current line 173), insert record(stop) + Synchronize(stop) + ElapsedTime + finite-positive guard + Destroy. Compute `double wall_time_ms = (double)elapsed_ms`. After the existing `cudaDeviceSynchronize`, insert the dlsym + null-check + call block (mirroring the reference's lines 132-139). In both stdout JSON paths (lines 193-196 sha256 mode + lines 199-217 default json mode), add a printf line emitting `"wall_time_ms": %.9f` with proper JSON comma placement.

2. **Build** — edit `workloads/simbench_bw/l1/dram_stream_stride_mix/Makefile` line 13 to extend the link line from `-lcudart` to `-lcudart -ldl`. Run `make -C workloads/simbench_bw/l1/dram_stream_stride_mix clean all`; verify rc=0 and binary exists.

3. **Backward-compat smoketest** — run `./dram_stream_stride_mix_run --shape-point trace_00` and confirm the JSON envelope is intact + new `wall_time_ms` field present. Run with `--output-mode sha256_f32le` and confirm `output_sha256` digest matches a pre-hook reference run captured in this round's preflight (record both digests in the round summary for trail).

4. **Live capture** — generate fresh preflight manifest via `tools/cloud/preflight.sh` (committed-script invocation, NOT a hand-rolled stale path). Run the capture command with `--enable-cupti-activity --preflight-manifest <fresh-path>`. Verify 30 cupti + 30 nvml sidecars + measurement_manifest at the canonical results root.

5. **Helper admission** — invoke `admit_cupti_dual_source_live_capture` via Python with the corrected signature (timing_artifact_path + sidecar_dir, NOT a single root). Capture the return tuple. If `ok=True` and `defects=[]` and `cupti_sidecar_pass_count=30`, AC-6 closes. If defects appear, document them as the recovery target.

6. **Static test** — write `workloads/simbench_bw/tests/test_dram_stream_stride_mix_hook_static.py` with 4 substring assertions over the file contents. Use Python's `pathlib` + `read_text()` + `assert "..." in text` pattern (no AST).

7. **TOML repair** — edit `workloads/simbench_bw/shapes/dram_stream_stride_mix.toml` `timing_00.launch_config` to grid `[1, 1, 1]` and block `[32, 1, 1]`. Re-run `python3 workloads/simbench_bw/validate.py --all`.

8. **Round summary** — record device + driver + CUDA + NVBit + gcc + observed clocks; record `cuobjdump --list-elf` output; record helper return values; promote dram_stream_stride_mix to `Verified-live-sm_86` for AC-16-second-workload-eligibility; queue 5060-laptop leg.

### Relevant References

- `workloads/simbench_bw/l2/rmsnorm_fwd_bf16/main.cu` — reference implementation; copy pattern from lines 17, 31-40, 117-139, 153
- `workloads/simbench_bw/l2/rmsnorm_fwd_bf16/Makefile` line 24 — reference link line `-lcudart -ldl`
- `workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu` — target file; insertion point around lines 165-174 (between H2D copy and D2H copy)
- `workloads/simbench_bw/l1/dram_stream_stride_mix/Makefile` line 13 — Makefile link line to extend
- `workloads/simbench_bw/cupti_activity_collector.cu` line 808 — receiver symbol `extern "C" void simbench_cupti_record_cuda_event_wall_time_ms(double ms)`
- `workloads/simbench_bw/cupti_dual_source_live_admission.py` — strict admission helper; arguments `timing_artifact_path` + `sidecar_dir` + `expected_slug` + `expected_shape_point`
- `workloads/simbench_bw/tests/test_cupti_dual_source_live_admission.py` — existing helper test harness; `_DEFAULT_SLUG="rmsnorm_fwd_bf16"` defaults are at lines 38-40
- `workloads/simbench_bw/shapes/dram_stream_stride_mix.toml` — timing_00 shape definition (repeat_iters=30) + launch_config entry to repair
- `tools/cloud/preflight.sh` + `tools/cloud/preflight_manifest.py` — preflight manifest generation pipeline
- `.humanize/rlcr/2026-05-07_02-01-13/round-23-summary.md` — R23 baseline (rmsnorm AC-16 admit `ok=True 30/30 0-defect`)
- `HANDOFF.md` — Three-Fold Completion Gate framing; do NOT modify in this round

## Dependencies and Sequence

### Milestones

1. **Milestone 1 — Source + Build (closes AC-1, AC-2, AC-3)**:
   - Phase A: Add headers, typedef, cudaEvent bracket, finite-positive guard, dlsym hook to `dram_stream_stride_mix/main.cu`.
   - Phase B: Add `wall_time_ms` JSON emission in both output modes.
   - Phase C: Update Makefile link line; rebuild binary; verify cuobjdump cubin set.
   - Commit: single `coding` commit covering all of Milestone 1.

2. **Milestone 2 — Backward-Compat + TOML (closes AC-4, AC-10)**:
   - Phase A: Standalone smoketest in default `json` mode; record JSON envelope shape.
   - Phase B: Standalone smoketest in `sha256_f32le` mode; record `output_sha256` digest; compare against pre-hook reference (capture pre-hook reference BEFORE Milestone 1 commit, or recompute by checking out parent commit).
   - Phase C: Trace-mode regression check (existing `--mode trace` admits unchanged).
   - Phase D: TOML launch_config repair + `validate.py --all`.
   - Commit: separate `coding` commit for the TOML repair (so Codex review can spot-check independently); the smoketest evidence lives in the round logs.

3. **Milestone 3 — Live Capture + Helper Admit (closes AC-5, AC-6)**:
   - Phase A: Fresh preflight manifest generation via `tools/cloud/preflight.sh`.
   - Phase B: Timing-mode capture with `--enable-cupti-activity` (post-commit per BL-20260508 post-commit-recapture-discipline).
   - Phase C: Helper admission invocation against the new evidence.
   - Phase D: Verify rmsnorm AC-16 baseline still admits (AC-8 partial).
   - Output: capture log + helper-return dump; no commit (live-capture artifacts under `workloads/simbench_bw/results/` are gitignored).

4. **Milestone 4 — Test Coverage + Round Closure (closes AC-7, AC-8 fully, AC-9)**:
   - Phase A: Add static-content-scan test `test_dram_stream_stride_mix_hook_static.py`.
   - Phase B: Add parameterized helper test (synthetic fixture preferred).
   - Phase C: Run broad sweep `pytest tools/ workloads/simbench_bw/tests/ -q`.
   - Phase D: Update goal-tracker mutable section; write round summary with full Two-Device-Model provenance.
   - Commit: `coding` commit for the test additions; round-state artifacts in `.humanize/rlcr/` (gitignored).

### Dependency Graph

- Milestone 2 depends on Milestone 1 (need built binary before running smoketests).
- Milestone 3 depends on Milestone 1+2 (need code committed AND backward compat verified before live capture; per BL-20260508 capture must be POST-commit so source-commit equality holds at HEAD).
- Milestone 4 depends on Milestone 3 (test parameterization references the helper return shape from Milestone 3; the round summary names live-evidence outputs).
- AC-10 (TOML repair) is in Milestone 2 specifically because it's an independent contract repair that doesn't gate the hook's correctness; if scope feels tight, it can defer to a follow-up round per the lower-bound clause.

## Task Breakdown

Each task carries exactly one routing tag.

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Add `#include <dlfcn.h>` + `#include <math.h>` + typedef + cudaEvent bracket + finite-positive guard + dlsym hook to `workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu` mirroring `l2/rmsnorm_fwd_bf16/main.cu` lines 17, 31-40, 117-139 | AC-2 | coding | - |
| task2 | Update `workloads/simbench_bw/l1/dram_stream_stride_mix/Makefile` line 13 link list from `-lcudart` to `-lcudart -ldl`; verify `make clean all` passes | AC-3 | coding | task1 |
| task3 | Emit `"wall_time_ms": %.9f` JSON field in both default `json` mode and `sha256_f32le` mode of dram_stream_stride_mix/main.cu | AC-2, AC-4 | coding | task1 |
| task4 | Capture pre-hook `output_sha256` reference digest for `--shape-point correctness_00 --output-mode sha256_f32le`, then re-verify post-hook digest equals pre-hook digest (backward compat) | AC-4 | coding | task2 |
| task5 | Repair `workloads/simbench_bw/shapes/dram_stream_stride_mix.toml` `timing_00.launch_config` to grid `[1, 1, 1]` and block `[32, 1, 1]`; re-run `validate.py --all` | AC-10 | coding | - |
| task6 | Generate fresh preflight manifest via `tools/cloud/preflight.sh`; verify NVML metadata pin admits | AC-5 | coding | task1, task2 |
| task7 | Run timing-mode live capture `python3 workloads/simbench_bw/run_suite.py --mode timing --kernel dram_stream_stride_mix --shape-point timing_00 --enable-cupti-activity --preflight-manifest <fresh>` POST-commit per BL-20260508; verify 30 cupti + 30 nvml sidecars + manifest emit | AC-5 | coding | task6 |
| task8 | Invoke `admit_cupti_dual_source_live_capture(timing_artifact_path, sidecar_dir, expected_slug="dram_stream_stride_mix", expected_shape_point="timing_00", collect_defects=True)`; verify `ok=True, cupti_sidecar_pass_count=30, defects=[]` | AC-6 | coding | task7 |
| task9 | Decide tolerance treatment: reuse existing rmsnorm-calibrated tolerance fixture; only run `calibrate_dual_source_tolerance.py` against dram if AC-6 surfaces tolerance defects (deferred follow-up) | AC-6 | analyze | task8 |
| task10 | Add `workloads/simbench_bw/tests/test_dram_stream_stride_mix_hook_static.py` with 4 substring assertions: `#include <dlfcn.h>`, `cudaEventElapsedTime`, `wall_time_ms`, `dlsym(RTLD_DEFAULT, "simbench_cupti_record_cuda_event_wall_time_ms"` | AC-7 | coding | task1 |
| task11 | Add parameterized helper test in `test_cupti_dual_source_live_admission.py` admitting dram_stream_stride_mix/timing_00 (synthetic fixture preferred) without breaking existing rmsnorm defaults | AC-7 | coding | task8 |
| task12 | Verify rmsnorm AC-16 baseline at carried R23 evidence still returns `ok=True 30/30 0-defect`; verify AC-15 combined pairing index unchanged | AC-8 | analyze | task1, task5 |
| task13 | Run broad sweep `pytest tools/ workloads/simbench_bw/tests/ -q`; verify no unexpected failures or skips relative to round-entry baseline | AC-8 | coding | task10, task11 |
| task14 | Write round summary recording device+driver+CUDA+NVBit+gcc+clocks per Two-Device-Model + cuobjdump output + helper return values; update goal-tracker mutable section; queue 5060-laptop sm_120 row | AC-9 | coding | task13 |

## Claude-Codex Deliberation

### Agreements

- The core approach (mirror rmsnorm pattern) is correct.
- `-ldl` Makefile change is appropriate and necessary (current dram Makefile only links `-lcudart`).
- Fresh sm_86 live capture + strict-admission helper invocation is the right live proof for AC-16 promotion.
- Status wording must remain `Verified-live-sm_86`; cross-arch promotion is gated on a future 5060-laptop session.
- The reference path is `l2/rmsnorm_fwd_bf16/main.cu` (NOT `l1/`), and the target is `l1/dram_stream_stride_mix/main.cu`.
- The cudaEvent bracket must enclose ONLY the kernel launch, not the D2H copy or any other operation; the dlsym call comes after sync, before D2H.
- The dlsym hook must guard on null function pointer (no unconditional call).
- The helper admission contract is `(timing_artifact_path, sidecar_dir, expected_slug, expected_shape_point)`, not a single artifact root.
- Reuse the existing rmsnorm-calibrated dual-source tolerance fixture for this round; recalibrate only if AC-6 surfaces a tolerance defect.
- Add `wall_time_ms` to BOTH `json` and `sha256_f32le` output modes (consumers ignore unknown fields; sha256 digest is unaffected because it hashes only the `output` byte stream).
- Static regression test belongs under `workloads/simbench_bw/tests/`, not `tools/`.
- TOML launch_config mismatch is in scope as a small contract repair (fix to grid `[1,1,1]` block `[32,1,1]` matching actual launcher).
- The round must NOT pin specific test-count integers (e.g., R23's "238 passed, 120 subtests") as non-regression gates; use general "no unexpected failures or skips" language.

### Resolved Disagreements

- **Helper command shape**: Claude's candidate plan v1 used `<artifact_root>` as a single positional argument. Codex's R1 review pointed out the helper takes `timing_artifact_path` and `sidecar_dir` separately. Resolution: AC-6 + task8 use the correct two-argument signature with `collect_defects=True` for diagnostic visibility. Rationale: the helper's actual API was the authoritative source.
- **Finite-positive guard**: Claude's candidate plan v1 said "finite-positive guard"; Codex's R1 review specified `isfinite(wall_time_ms) && wall_time_ms > 0.0` plus `<math.h>` include. Resolution: AC-2 explicitly names both the function name and the `> 0.0` bound. Rationale: a guard that admitted zero would still produce a downstream defect.
- **Preflight manifest freshness**: Claude's candidate plan v1 cited `/tmp/simbench_preflight.json`. Codex's R1 review flagged this as a stale-artifact risk. Resolution: AC-5 + task6 require the round to generate a fresh manifest via `tools/cloud/preflight.sh`. Rationale: stale preflight evidence drift was a recurring v3 issue.
- **Test count gate**: Claude's candidate plan v1 cited R23's "238 passed, 120 subtests" as a non-regression gate. Codex's R1 review flagged this as stale (the broad suite has grown since R23). Resolution: AC-8 uses general "no unexpected failures or skips" language with no specific integer pin. Rationale: pinning stale counts creates false-failure noise.
- **Static test location**: Claude's candidate plan v1 placed the static test under `tools/`. Codex's R1 review noted convention places tests under `workloads/simbench_bw/tests/`. Resolution: task10 places the test there. Rationale: matches existing repo convention.
- **TOML launch_config mismatch handling**: Claude's candidate plan v1 listed it as an open question (in scope, queue, or non-issue). Codex's R1 review said it should be in scope as a small repair (the orchestrator does not use the TOML grid/block, but the repo treats it as meaningful contract metadata). Resolution: AC-10 + task5 add it to the round. Rationale: contract surfaces should not be left silently misaligned.
- **wall_time_ms in sha256_f32le mode**: Claude's candidate plan v1 was undecided. Codex's R1 review said add it (consumers ignore unknown fields; perturbation harness keys off `output_mode`/`output_sha256`/`element_count` only). Resolution: AC-2 + task3 emit in both modes. Rationale: consistency simplifies future consumers and the sha256 digest is unaffected because it hashes only `output` bytes.

### Convergence Status

- Final Status: `converged`
- Convergence Rounds Executed: 2 (Codex first-pass + Codex second-pass reasonability review). All REQUIRED_CHANGES from R1 were incorporated into the plan; no DISAGREE items at the substantive level remain. Codex's `OPEN QUESTIONS` were all answered within R1 with concrete resolutions that the plan adopts.

## Pending User Decisions

(none — all open questions resolved by Codex R1 review; the path-boundary mechanism handles the upper/lower-bound success-criterion flexibility without requiring a binary user decision.)

## Implementation Notes

### Code Style Requirements

- Implementation code (`.cu`, `.py`, `.sh`) and comments must NOT contain plan-specific terminology such as "AC-N", "Milestone N", "Step N", "Phase N", "Round N", "task1..14", or similar workflow markers. These belong in plan documentation only.
- Use descriptive, domain-appropriate naming and prose in code: comments may describe "the dual-source reconciliation contract", "the LD_PRELOAD'd CUPTI collector", "the cudaEventElapsedTime-derived wall-clock measurement", but must not say "AC-2 instrumentation" or "Round 0 hook".
- Test docstrings and assertion messages must follow the same rule (per BL-20260507 plan-token-scanner discipline; the existing `tools/test_plan_token_scanner.py` will fail-fast if violated).

### Two-Device Local Development Model Reminders

- This round is sm_86-only. The round summary must use `Verified-live-sm_86` for any promotion claim. Cross-arch promotion to `Cross-arch-verified` requires a future 5060-laptop session; this round queues but does NOT close that work.
- The hook code is arch-independent host-side glue. The 5060-laptop session inherits the hook without re-implementation, but the live admission must be replayed there (the receiving session captures fresh sm_120 evidence, runs the same helper, and records `Verified-live-sm_120`; only then does the AC promote to `Cross-arch-verified`).
- Per CLAUDE.md, an arch-coupled live result on sm_86 does not extend to sm_120 without explicit live witness on the second arch.

### Critical Rules Reminders

- CLAUDE.md Rules 1-5 are binding throughout the round. No NCU on cloud/server (the local-label exception in `tools/local_labels/` does not apply to the dram_stream_stride_mix workload). No driver-state changes. No privileged CUPTI surfaces (Activity is allowed). No host-driver work. No simulator revival.
- The loop-edit-validator hook may block direct Edit/Write on prior round summary files; if a legitimate amendment is needed, use the operator-sanctioned Python file-write path (per the just-sanctioned R18/R19 amendment policy) and document the amendment in the round summary's `Goal Tracker Update Request` section.

## Output File Convention

This template is used to produce the main output file (`plans/ac16_dram_stream_cupti_hook_plan.md`).

### Translated Language Variant

`alternative_plan_language` is empty in the merged config; no translated variant is generated.

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# Tightly Scoped Single-AC Round For dram_stream_stride_mix CUPTI Hook

## Original Idea

Brainstorm and verify with Codex two candidate follow-up RLCR loop scopes for the NVIDIA_Sims_Predict_NCU project. Repo at /home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU. Active host: RTX 3090 Ti sm_86 Ampere (CUDA 13.2, NVBit 1.8, native g++-14). Just exited a 33-round v3 RLCR loop after the Mainline Drift Circuit Breaker fired at R31 STALLED + R32 STALLED + R33 STALLED. Path 1 (R18/R19 amendment-method bypass) was just operator-sanctioned. Path 2 (5060-laptop sm_120 replay) requires session pivot — out of scope here. Path 3 (server/cloud) explicitly deferred per IMMUTABLE plan.

The two candidate scopes are:

**Option A — Close 7 queued side issues** (~3-5 rounds estimated; or 1-2 if batched). Issues identified in `.humanize/rlcr/2026-05-07_02-01-13/goal-tracker.md` Queued Side Issues section (lines ~222-237):
1. Preflight `12.*` policy/prose drift in `tools/cloud/preflight.sh` lines 51, 377 + `tools/cloud/preflight_manifest.py` lines 131-134 — code admits all CC 12.* but prose says "server sm_120" rejected. Pure prose/code alignment. Host-only. Low complexity (2 files, ~20-30 lines).
2. `measurement_manifest.json` `driver_version` ambiguity — `tools/cloud/preflight_manifest.py` line ~117 reports CUDA driver API "13.2" while `preflight.sh` line 844 reports NVIDIA driver "595.71.05" under same field name. Need rename to `cuda_driver_version` or `nvidia_driver_version` + schema/consumer update. Host-only. Medium complexity (3 files, ~15-20 tests).
3. Per-source NVBit drop counters beyond residual expected-record mismatch — `tools/nvbit_tracer/` producer-side instrumentation. Not yet exercised by live evidence (R23 trace had zero drops). Host-only. Medium-high complexity (1-2 files, ~100-200 lines + 5-10 tests).
4. Event-based async-free retirement for stream-ordered allocator hooks — `tools/nvbit_tracer/tracer_tool/allocation_registry.h`. Synchronization refactoring. Host-only. Medium complexity (~50-150 lines + 3-5 tests).
5. Workload-side `wall_time_ms` for `dram_stream_stride_mix` — same as Option B; this is the dlsym hook addition. Very low complexity.
6. Perturbation-harness arch_capabilities prose drift — `tools/arch_capabilities.json` line 103 says "nine perturbation binaries" (actually 10); line 121 ledger.py note misses `cache_modifier_cs`. Pure prose. Very low complexity (1 file, 3 prose edits).
7. R21 retroactive-helper wording caution — `workloads/simbench_bw/cupti_dual_source_live_admission.py` "retroactive helper applied to R20 evidence" wording overclaims. Docstring update. Very low complexity.

Issues are independent of each other; can be batched in any order. None require sm_120 hardware.

**Option B — Add `simbench_cupti_record_cuda_event_wall_time_ms` dlsym hook to `dram_stream_stride_mix/main.cu`** (~1 round). Reference implementation is at `workloads/simbench_bw/l2/rmsnorm_fwd_bf16/main.cu` lines 117-139 (Round 0 of the v6 RLCR loop confirmed the actual reference lives under `l2/`, not `l1/`). Pattern is `dlsym(RTLD_DEFAULT, "simbench_cupti_record_cuda_event_wall_time_ms")` after `cudaEventElapsedTime`; calls record(ms) into the CUPTI Activity collector's per-process global. Currently `dram_stream_stride_mix/main.cu` lines 172-174 launches+syncs+memcpy without any timing or hook. Adding the hook makes the workload AC-16-eligible (currently fails dual-source reconciliation deterministically because no cuda-event measurement to reconcile against). Single file modification ~10-20 lines. No new tests needed; existing `cupti_dual_source_live_admission.py` validator + run_suite.py orchestration would automatically begin admitting dram_stream_stride_mix `--mode timing --enable-cupti-activity` captures. Host-only on sm_86. Cross-arch implication: code is arch-independent; future 5060-laptop session inherits the hook without re-implementation.

Note: Option A item 5 IS Option B. So a hybrid approach is natural: do Option B first (closes A item 5), then either continue with Option A items 1, 2, 6, 7 (low-risk doc/prose batch) and defer items 3, 4 (producer-side instrumentation needing live regression evidence to motivate), or stop after B and pivot to 5060.

Please brainstorm:
1. Which option (A, B, or hybrid B-then-A-subset) is the better next-loop scope given the v3 loop just exited via the Mainline Drift Circuit Breaker — i.e., what scope avoids re-triggering the displacement classification Codex applied to R28-R31?
2. What are the strongest arguments for and against each option as a fresh-RLCR-loop mainline?
3. Are there hidden risks (e.g., would item 3's per-source drop counters require fresh live capture? would item 4 require behavioral change that could regress R23's trace evidence?) that would make those items individually unsuitable as a single-round mainline?
4. Get Codex's verification of the recommendation.

Focus the analysis on what would land an ADVANCED verdict from Codex review, given the v3 loop's drift history.

## Primary Direction: Plan-Text Framing Discipline

### Rationale

Frame the next-loop plan text so Codex review reads it as forward progress on the original v3 plan, not as a continuation of the R28-R31 displacement chain — the wording and scope-bounding choice matters as much as the work itself.

### Approach Summary

The next fresh RLCR loop (following v3's 33-round R28-R31 stalled sequence) should be scoped as a **single-AC mainline focused on AC-16 producer-side second-workload eligibility on sm_86**, specifically: add the `simbench_cupti_record_cuda_event_wall_time_ms` dlsym hook to `dram_stream_stride_mix/main.cu` to unblock the workload from dual-source reconciliation failure and promote it from AC-16-ineligible to AC-16-eligible without requiring a host device pivot.

**Core mechanism**: Insert the reference dlsym pattern from `l2/rmsnorm_fwd_bf16/main.cu` (lines 132-139) after the `cudaEventElapsedTime` measurement in `l1/dram_stream_stride_mix/main.cu` (lines 172-174). The hook resolves only when the CUPTI Activity collector .so is LD_PRELOAD'd; otherwise it is a no-op, preserving baseline behavior. The hook enables the existing `cupti_dual_source_live_admission.py` validator to admit the workload's dual-source reconciliation (currently deterministically fails because no cuda-event measurement is available to reconcile against the kernel timestamp).

**Affected components**: single file modification (~10-20 lines), no new tests (existing `cupti_dual_source_live_admission.py` validators + `run_suite.py` orchestration automatically begin admitting `--mode timing --enable-cupti-activity` captures for `dram_stream_stride_mix`), no production-code behavior change to existing AC-15 or AC-16 evidence on `rmsnorm_fwd_bf16`.

**Plan-text framing for Codex review**: Name the mainline objective as **"AC-16 producer-side dual-source measurement hook for `dram_stream_stride_mix` on sm_86"** rather than "add dlsym hook" or "close issue 5". This frames it as forward progress on the original plan's AC-16 producer-side closure goal, directly addressing the v3 loop's R28-R31 drift pattern (accidental scope creep into documentation hygiene rather than AC advancement). The contract includes explicit negative-test definition: the round fails if `dram_stream_stride_mix --mode timing --enable-cupti-activity` still fails dual-source reconciliation post-hook, or if the hook changes behavior when LD_PRELOAD is not set.

**Specific contract language (ready for `/humanize:gen-plan` consumption):**

```markdown
## Mainline Objective

Add the `simbench_cupti_record_cuda_event_wall_time_ms` dlsym hook to
`workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu` to unblock
AC-16 producer-side dual-source measurement eligibility for the
dram_stream_stride_mix workload on sm_86.

Current state (v3 R0-R31): AC-16 producer-side dual-source reconciliation
succeeds for rmsnorm_fwd_bf16 (hook present, proven R20-R23 Verified-live-sm_86)
but deterministically fails for dram_stream_stride_mix (hook absent; CUPTI
Activity collector has no cuda-event measurement to reconcile against kernel
timestamp). The workload is therefore AC-16-ineligible per the dual-source
contract.

Scope: Insert the reference pattern from rmsnorm_fwd_bf16/main.cu
(dlsym + function-pointer call) after cudaEventElapsedTime in
dram_stream_stride_mix/main.cu lines 172-174. No new tests.
Existing cupti_dual_source_live_admission.py validators + run_suite.py
orchestration automatically begin admitting --mode timing --enable-cupti-activity
captures for the workload post-hook.

Expected outcome: dram_stream_stride_mix moves from AC-16-ineligible
to AC-16-eligible and can be promoted to Verified-live-sm_86 for AC-16
producer-side in a subsequent round after fresh capture post-hook, provided
dual-source reconciliation reaches 30/30 pass rate (same threshold as
rmsnorm_fwd_bf16 R23).

This round advances the original plan's AC-16 closure goal and addresses the
v3 loop's recent displacement cycle (R28-R31: docs-only work) by restoring
focus to AC-level production-code advancement.

## Negative-Test Definition

Round succeeds if and only if ALL of the following hold:

1. dram_stream_stride_mix/main.cu lines 172-174 (kernel launch, sync, memcpy)
   are bracketed by cudaEvent measurement with a dlsym hook call that resolves
   to simbench_cupti_record_cuda_event_wall_time_ms.

2. Hook call is guarded on non-null function pointer (dlsym result);
   hook is no-op when LD_PRELOAD is absent (baseline behavior preserved).

3. Post-hook measurement, `python3 workloads/simbench_bw/run_suite.py
   --mode timing --kernel dram_stream_stride_mix --shape-point timing_00
   --enable-cupti-activity --preflight-manifest /tmp/manifest.json`
   produces CUPTI Activity sidecars (cupti_activity_*.json) that pass
   cupti_dual_source_live_admission.py validation (no JSONDecodeError,
   no defect_token).

4. Baseline rmsnorm_fwd_bf16 AC-16 evidence (Verified-live-sm_86 from R20-R23)
   remains coherent and passes re-validation at round exit.

5. Broad sweep (tools/ + workloads/simbench_bw/tests/) passes at round exit
   with no regression in test count from round entry.

Round fails if dram_stream_stride_mix post-hook still fails dual-source
reconciliation (deterministically zero passes out of repeat attempts),
or if the hook changes behavior when LD_PRELOAD is absent.

## Out of Scope — Explicit Anti-Goals

Do **NOT**:
- Attempt R18/R19 amendment-method operator sanction. The blocker
  remains open and operator-only (note: just sanctioned by operator
  outside this loop; record closure but do not work it inside the round).
- Pursue the three mainline gaps (5060-laptop sm_120 replay,
  server/cloud AC closure, Codex sandbox resilience).
- Add queued-side-issue work (preflight 12.* drift, manifest
  driver_version ambiguity, NVBit drop counters, async-free
  retirement, perturbation-harness arch_capabilities prose,
  R21 retroactive-helper wording). These remain queued and must
  NOT displace the AC-16 hook mainline.
- Introduce wording overclaims about completion gates (R24/R25/R27
  pattern). This round closes one AC-level seam; the 5060 session
  and server work remain open.

This round has a single mainline objective (AC-16 hook for
dram_stream_stride_mix) and explicitly does not attempt to close
other surfaces.
```

### Objective Evidence

- `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/l2/rmsnorm_fwd_bf16/main.cu` lines 31-39 (typedef for hook function pointer) + 132-139 (dlsym + call pattern). Reference implementation demonstrates exact pattern: `dlsym(RTLD_DEFAULT, "simbench_cupti_record_cuda_event_wall_time_ms")` resolves to a function that records the wall_time_ms measurement into the LD_PRELOAD'd collector library's per-process global.
- `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu` lines 1-8 (headers; no dlfcn.h present), lines 172-174 (launch + sync + memcpy without timing or hook), lines 142-175 (main function structure). Target insertion point is immediately after `cudaEventSynchronize` at line 173 and before `cudaMemcpy` at line 174 (same temporal window as rmsnorm_fwd_bf16).
- `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/tests/test_cupti_dual_source_live_admission.py` (lines 1-50 of implementation; full test harness exists and exercises `admit_cupti_dual_source_live_capture` helper across positive and negative scenarios). Tests for both `measurement_manifest_invalid_json` defect recovery (round 23) and full dual-source reconciliation pass/fail coverage (rounds 20-23) are already in place and regression-locked.
- **Plan-precedent for ADVANCED-verdict framing:**
  - R23 contract lines 21-29: Mainline Objective frames a contract-completeness round in AC terms ("AC-16 producer-side live evidence on sm_86 — strict-helper-admitted with the FULL contract"). Not: "fix the manifest-invalid-json path and add 7 tests."
  - R28 contract lines 12-16: Mainline Objective frames a hygiene round tightly to one AC impact ("no AC promotion is asserted; AC-15 + AC-16 producer-side `Verified-live-sm_86` evidence at substantive parent remains coherent"). Explicitly names what is NOT being closed.
  - R31 contract lines 12-25: Mainline Objective frames a second hygiene round using the same BL pattern but stays focused on one RC (`tools/nvbit_tracer/README.md`). Lines 19-22 explicitly say "NOT close mainline gap #1, gap #2, or gap #3".
  - Codex verdicts (R28-R31 ADVANCED on substantive parts, STALLED only on round-classification): The pattern is that when a round has a single, tightly scoped, AC-aligned mainline objective and explicitly names what it is NOT closing, Codex issues ADVANCED verdicts.
- **Negative precedent for STALLED:** R24 review result lines 1-15 shows the three mainline gaps (cross-arch replay, server/cloud, operator-sanction) are all that block genuine loop completion. R28-R31 worked around this by finding autonomously-closeable hygiene items, each with explicit anti-goals. R32-R33 tried to repeat the hygiene pattern after Codex R31 review explicitly said "further 3090 Ti-only README hygiene would now be process drift" (round-31-review-result.md lines 85-92). The STALLED verdict in R32-R33 reflects pattern-breaking: hygiene work without a concrete production-code AC anchor.

### Known Risks

- **False positive dual-source pass**: The hook adds 10-20 LOC calling a dlsym'd function. If the hook function pointer resolves but the collector .so's internal registration fails silently, the dual-source reconciliation could appear to pass when the Activity collector never received the measurement. Mitigation: the existing `test_cupti_dual_source_live_admission.py` validator checks that both `cuda_event_wall_time_ms` AND `cupti_activity_duration_ns` are present and reconcile; if the hook call succeeds but the sidecar shows no collector response, the validator still fails. Test is regression-locked from R20-R23.
- **Behavior change when LD_PRELOAD is not set**: If dlsym fails to resolve (LD_PRELOAD absent), the hook returns nullptr and execution continues. However, if any subsequent code path assumes the record was called, baseline behavior changes. Mitigation: reference implementation (rmsnorm_fwd_bf16) has been exercised for 4 rounds in both modes. The pattern is proven safe. No baseline regression observed in R20's broad sweep (212 passed, 80 subtests).
- **Arch-independence claim**: The hook is in .cu source, not device code. The same hook works on sm_86 (proven in rmsnorm_fwd_bf16, promoted R20 AC-16), and will work on sm_120 without re-implementation. Hidden risk: if the CUPTI Activity collector .so's symbol-resolution behavior differs on sm_120 (different CUDA 13.2 runtime library availability), the hook could silently fail to resolve on sm_120. This is out-of-scope for the sm_86 loop but should be verified on the 5060 laptop session.
- **Hook integration into run_suite.py orchestration**: The `run_suite.py --mode timing --enable-cupti-activity` path already passes `LD_PRELOAD` via the collector .so to the workload subprocess. The hook will auto-resolve when run under that orchestration path. However, if the collector .so is built on a different GCC version or with incompatible -fvisibility flags, the symbol may not export. Mitigation: R20 confirmed the symbol is correctly exported from `libsimbench_cupti_activity_collector.so` (built in the same session). No symbol-rename risk for sm_86.

## Alternative Directions Considered

### Alt-1: Codex-Verdict Engineering

- Gist: Engineer the next-loop scope explicitly for ADVANCED verdict acceptance under Codex's known evaluation criteria (single mainline, 1-round completability, test/commit/validator proof, no displacement signal). Recommends Hybrid B + A items 1, 2, 6, 7 in 1 round, framed with substantive AC anchor (Option B) primary and hygiene secondary, leveraging R23/R28/R30 ADVANCED precedent and avoiding the R31 STALLED displacement pattern.
- Objective Evidence:
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/l2/rmsnorm_fwd_bf16/main.cu` lines 15-40, 117-139 (reference dlsym hook implementation including #include <dlfcn.h>, typedef simbench_cupti_record_fn at line 40, dlsym at line 135, conditional call at lines 136-139)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/.humanize/rlcr/2026-05-07_02-01-13/round-30-review-result.md` lines 12-18 (R30 ADVANCED verdict precedent: narrow documentation fix accepted because scope-bounded, correct, did not displace next mainline)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/.humanize/rlcr/2026-05-07_02-01-13/round-23-review-result.md` lines 3-43 (R23 ADVANCED on substantive helper fix: correct code, passing tests, passing validator, no regression)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/.humanize/rlcr/2026-05-07_02-01-13/round-31-review-result.md` lines 18-24 (R31 STALLED naming the displacement chain pattern; tracker correctly keeps cross-arch replay active)
  - Validator infrastructure exists at `workloads/simbench_bw/cupti_dual_source_live_admission.py` (R23-accepted helper); test harness at `workloads/simbench_bw/tests/test_cupti_dual_source_live_admission.py` (26+ tests)
- Why not primary: Direction 4's framing emphasis subsumes Codex-verdict engineering (good framing IS what produces ADVANCED verdicts) and provides ready-to-paste contract language; this alternative's strength is its meta-process grounding in actual review-result texts but is less directly actionable for plan generation.

### Alt-2: Cross-Arch Path-2 Readiness

- Gist: Optimize for the upcoming 5060-laptop session pivot — pick scope that leaves the receiving session with the cleanest inherit-and-replay surface. Recommends Hybrid B-then-A-subset (items 1, 2, 6, 7) because Option B's hook is arch-independent and unlocks multi-workload AC-16 capture (rmsnorm + dram_stream) in a single 5060 round vs. sequential rounds. Quantified inheritance benefit: ~1-2 rounds saved on 5060 hardware.
- Objective Evidence:
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/HANDOFF.md` lines 550-572 (AC-16 receiving-session contract prescribes rmsnorm_fwd_bf16/timing_00 as initial proof-of-concept; dram_stream_stride_mix is a secondary candidate)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/shapes/dram_stream_stride_mix.toml` (timing_00: warps=1024, accesses=4096, stride_classes=4, repeat_iters=30 — same 30-repeat shape as rmsnorm timing_00)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/cupti_dual_source_live_admission.py` (schema-agnostic dual-source contract already admits any workload/shape pair; no producer changes needed)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/timing_timing_00.json` (sm_86 baseline artifact exists with no dual-source reconciliation metadata, confirming current hook absence)
- Why not primary: This direction's recommendation (Hybrid B + A1/2/6/7) overlaps with primary; the forward-looking 5060-session inherit-cost framing is supporting evidence for choosing Option B but is one of several arguments rather than the principal axis. The framing-discipline angle (primary) absorbs this benefit while better addressing the immediate v3 drift pattern.

### Alt-3: Regression Blast-Radius Minimization

- Gist: Treat R23 evidence as load-bearing; minimize touch to NVBit tracer producer (AC-15 dependency) and AC-16 helper internals. Audit each Option A item for blast radius: A1, A6, A7 ISOLATED; A2 ADJACENT (test rename only); A3, A4 COUPLED (touches NVBit producer + allocator hooks; could regress R23 trace sequencing). Recommends Option B + A1/A2/A6/A7 split across 2 rounds (B in round N, A subset in N+1) to prevent any single-round failure from compromising both.
- Objective Evidence:
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/tools/nvbit_tracer/tracer_tool/allocation_registry.h` lines 65-114 (sync-point hook driven, cuStreamSynchronize/cuCtxSynchronize; refactoring to event-based would alter retirement timing)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/schemas/nvbit_drop_rate_exceeded.schema.json` lines 1-78 (4-field counter set; R23 baseline has zero drops across all traces, no live evidence to validate new instrumentation)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/run_suite.py` line 269 (driver_version producer via cuDriverGetVersion → CUDA driver API)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/tools/cloud/preflight.sh` line 844 (driver_version producer via nvidia-smi → NVIDIA driver) — confirms naming conflation across producers
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/tools/cloud/test_preflight_manifest.py` (111 tests covering manifest fields; rename would cascade through all of them)
- Why not primary: The blast-radius angle correctly identifies items 3, 4 as risky (alignment with primary's anti-goals), but the primary direction's tight single-AC framing provides equivalent regression protection by simply not attempting items 3, 4 at all — and additionally addresses the framing failure that drove v3's drift. Splitting B and A across 2 rounds is also less ROI-efficient than the primary's "B alone" lean recommendation (matches Alt-5).

### Alt-4: Existing-BL Coverage Application

- Gist: Pick the scope that exercises the 9 existing BL entries (especially BL-20260507 tracked-doc-hygiene + BL-20260509 strict-admission-contract-first) cleanly without surfacing new failure-mode patterns. Recommends Hybrid B + A items 1/2/6/7 batched: each item maps to an existing BL (Item 1 → BL-20260507 lockstep; Item 2 → BL-20260506 multi-clause schema; Items 6, 7 → BL-20260507 prose hygiene; Option B → BL-20260509 admission-contract). Items 3, 4 don't cleanly map to existing BL (would require fresh discovery rounds).
- Objective Evidence:
  - `.humanize/bitlesson.md` 9 BL entries (BL-20260504 through BL-20260509); each specific-problem-and-specific-solution patterns
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/tools/test_blocked_on_3090ti_hygiene.py` lines 94-150 (multi-clause schema validator pattern proving BL-20260506 application)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/cupti_dual_source_live_admission.py` lines 99-120 (20 DEFECT_TOKENS covering identity + value + schema + consistency across 4 surfaces — BL-20260509 application; adding hook makes dram_stream_stride_mix admit without `timing_artifact_dual_source_failures_nonempty`)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/tools/perturbation_harness/ledger.py` lines 48-51 (authoritative producer-side enum already correct for arch_capabilities prose fix in item 6)
- Why not primary: The BL-coverage angle is excellent supporting analysis (proves the recommended scope doesn't surface uncovered failure modes), but it's a verification axis rather than a generative principle. The primary's framing-discipline angle generates the round contract directly; BL coverage validates that the contract's actions are well-precedented.

### Alt-5: Round-ROI Maximization

- Gist: Maximize information-per-round; prefer highest-leverage single-round work over batched cleanup. Quantitative ROI matrix recommends Option B alone as primary (1 round, AC-16 unlock + cross-arch prep; HIGH ROI), Hybrid B + A1/6/7 sequenced in 2 rounds as secondary (MEDIUM-HIGH if B finishes fast). Option A batched in 1 round flagged as MEDIUM with batching-defect-discovery risk.
- Objective Evidence:
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/cupti_activity_collector.cu` line 808 (`extern "C" void simbench_cupti_record_cuda_event_wall_time_ms(double ms)` already exported as dlsym target)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/run_suite.py` line 758 (already consumes `cuda_event_wall_time_ms` field; no orchestration changes needed)
  - `/home/jakeshea/3090ti_5060laptop_cross_study/NVIDIA_Sims_Predict_NCU/workloads/simbench_bw/cupti_dual_source_live_admission.py` lines 1-402 (validator wired for any timing_00-shape workload; admits on hook presence)
  - R23 + R20 + R28 precedents for single-round implementation+capture+validation with deferred cleanup
- Why not primary: This direction recommends "B alone" as primary (vs. primary's "B with explicit anti-goals against A items"). The ROI-quantitative framing is excellent quantitative confirmation that B alone is highest-leverage, but provides less actionable contract language than the framing-discipline direction. Effectively the same recommendation; the primary's framing emphasis is more directly useful for `/humanize:gen-plan`.

## Synthesis Notes

All five alternatives converge on Option B (the dlsym hook addition) as the right primary mainline; the disagreement is entirely about whether and how to batch Option A items 1/2/6/7 alongside it. Three alternatives (Alt-1, Alt-2, Alt-4) recommend Hybrid B+A1/2/6/7 in one round; Alt-3 recommends splitting across two rounds; Alt-5 recommends B alone with A subset deferred. The primary direction (Plan-Text Framing Discipline) sides with Alt-5's "B alone" recommendation because tight scope is exactly what Codex's R28-R31 displacement classification was punishing — adding ANY non-AC-anchored work (even low-risk prose) reintroduces the displacement vector. If the operator wants to close A items 1/2/6/7 quickly, the cleanest path is a separate single-round R+1 framed as **"queued-side-issue prose batch — explicitly NOT mainline"**, which honors the v3 R28-R31 displacement lesson by NOT pretending hygiene is mainline.

The most valuable elements from the alternatives that fold into the primary if desired:
- From **Alt-1** (Codex-Verdict Engineering): The verbatim ADVANCED-verdict precedents (R23/R28/R30) with line numbers — these belong in the round contract's "Why This Is the Disciplined R0" justification section.
- From **Alt-2** (Cross-Arch Path-2 Readiness): The cross-arch inheritance argument — the hook code is arch-independent and the 5060 session inherits it without re-implementation. This is the strongest "this isn't displacement" argument for an autonomous-host round.
- From **Alt-3** (Regression Blast-Radius Minimization): The COUPLED classification of items 3, 4 (NVBit producer + allocator hook) — these belong in the contract's anti-goals section as "explicitly excluded because they touch R23 evidence dependencies."
- From **Alt-4** (Existing-BL Coverage Application): The BL-20260509 application argument — adding the hook is a clean extension of the strict-admission-contract-first lesson, not a new pattern.
- From **Alt-5** (Round-ROI Maximization): The quantitative ROI matrix — useful as evidence that B alone is highest-ROI, refuting any review-side suggestion that "while you're at it, also fix items 1/6/7."

If the operator chooses to deviate toward the Hybrid path (B + A1/2/6/7 in one round), the primary direction's Negative-Test Definition section needs only minor extension — add 4 additional success criteria (one per A item), each requiring a structural lockstep test (per BL-20260507 pattern) so Codex can mechanically verify each item is closed. The displacement risk increases mildly but remains low because items 1/2/6/7 all map to existing BL patterns (per Alt-4) and are ISOLATED-blast-radius (per Alt-3).

--- Original Design Draft End ---
