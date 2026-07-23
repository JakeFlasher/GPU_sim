# Cross-Arch Bounded First sm_120 Label Admission Loop

## Goal Description

Land a single bounded RLCR loop on the RTX 5060 laptop (sm_120 Blackwell) that closes at evidence status `Verified-live-sm_120` by combining four pieces in one loop:

1. **Schema-drift remediation**: admit the field `source_commit_override_from` (carried by 43 of 44 existing sm_86 admitted cells) into the v4 `local_label_admitted.schema.json` as an optional audit-annotation field, since the field is currently rejected by the schema's strict `additionalProperties: false` discipline and would block any cross-host merge.
2. **Minimum hardware-config decoupling** sufficient to admit sm_120 cells through the load-bearing label-sweep + merge + comparison-runner path: platform/SKU registry extension for `rtx_5060_laptop` across all lockstep surfaces; arch-token policy clarification (`sm_120` for label/comparison surfaces; `sm_120-laptop` for device/capability/sidecar-directory surfaces; `rtx_5060_laptop/sm_120` for trace/results paths); sm_120 launch-parameter sidecar directory + `extend_launch_parameters.py` arch-filter lift; gridDim alpha cross-arch metadata field.
3. **Operator-driven sm_120 NCU label sweep** against the 5-cell PlattCalibrator activation training cohort (`layernorm_fwd_bf16_sm80/timing_shape`, `rmsnorm_fwd_bf16_sm80/timing_shape`, `silu_fwd_bf16_sm80/correctness_shape`, `silu_fwd_bf16_sm80/timing_shape`, `softmax_safe_bf16_sm80/timing_shape`) under the CLAUDE.md Rule 1 local-label exception, plus 1 optional stretch cell (`dram_stream_stride_mix/timing_00`) from the memory-stress family. Closure requires at least 5 sm_120 cells admitted at the unchanged `stddev_threshold = 0.03` gate; the stretch cell is non-blocking.
4. **Direct cross-arch admission audit** via `workloads/simbench_bw/comparison_runner.py` (NOT through the headline-cohort orchestrator, which carries additional sm_86 hardcodes deferred to a follow-on loop) against the merged `label_admitted.json` (44 sm_86 + ≥5 sm_120) using synthetic predictor bundles for both arches, verifying the already-wired `require_both_arches_for_cross_arch` cross-arch admission gate emits `top_level_admission_scope = "cross_arch_require_both_arches"` and the per-arch `coverage_gate_verdict.meets_floor` is true for both arches.

Closure is `Verified-live-sm_120` only. `Cross-arch-verified` is explicitly deferred per CLAUDE.md "Two-Device Local Development Model" (Codex review is one-arch-at-a-time; a fresh sm_86 re-validation on the 3090 Ti in a separate loop is the precondition for the combined status). The loop's scope is bounded BELOW the historical M1 (5060 laptop, 8 rounds → Mainline Drift Circuit Breaker at scope-closure stalemate) and v7 (3090 Ti, 4 rounds → circuit-breaker at structural stall after 3 STALLED verdicts on deferred-but-incomplete sm_120 leg) failure patterns by making sm_120 the mainline on the laptop with no host-constrained deferrals: every AC closes locally on this device, and explicitly-deferred items (full activation-trace-orchestrator decoupling, leave-arch-out calibration evaluation, sm_100 zero-shot prediction, headline-cohort orchestrator sm_120 parameterization) are recorded as non-blocking with named follow-on plans.

Two empirically-verified facts shrink the engineering surface vs. an unbounded cross-arch loop:
- The `require_both_arches_for_cross_arch` admission gate is ALREADY WIRED in `comparison_runner.py` (lines 2780-2793) with 9 passing tests in `test_require_both_arches_for_cross_arch.py` (Codex Round 1 ran the file and observed 9 passes, correcting the gen-idea draft's 11-test estimate).
- The NVBit Blackwell TMEM opcode tracer extension is ALREADY SHIPPED: `MemoryOpClass::kTmem = 10` in `tools/nvbit_tracer/tracer_tool/memory_op_class.h`; full LDT/LDTM/STT/STTM/UTCCP classifier coverage; proto enum widened additively (`instruction.proto`); consumer-side schema enum extended (`nvbit_unsupported_memory_path.schema.json`); 216 passed + 93 subtests in the NVBit test suite per Codex Round 1's verification run. Note: TMEM still routes as an unsupported-memory-path sidecar (`path_resolved=true`) rather than a modeled cache path; that modeling is out of scope for this loop.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. The order matches the commit slice order in `## Dependencies and Sequence`.

- AC-1: Schema-drift remediation for `source_commit_override_from`. The v4 `workloads/simbench_bw/schemas/local_label_admitted.schema.json` admits `source_commit_override_from` as an OPTIONAL string field on cells (additive widening per CLAUDE.md wire-extension policy; no `schema_version` bump because the addition is purely additive and existing readers continue to validate). The 44 existing sm_86 admitted cells strict-validate against the patched v4 schema; `tools/local_labels/merge_admitted_cells.py` no longer rejects the file as input.
  - Positive Tests (expected to PASS):
    - `jsonschema.validate(json.load(open('data/local_labels/label_admitted.json')), json.load(open('workloads/simbench_bw/schemas/local_label_admitted.schema.json')))` returns without raising.
    - A v4 cell omitting `source_commit_override_from` validates against the patched schema (the field is optional, not required).
    - A v4 cell carrying `source_commit_override_from = "<40-hex-char SHA>"` validates.
    - `tools/local_labels/merge_admitted_cells.py` accepts the current `label_admitted.json` as input without `additional properties` validation errors.
  - Negative Tests (expected to FAIL/REJECT):
    - A cell carrying an unknown additional field (e.g., `not_a_real_field`) is still rejected by `additionalProperties: false`.
    - A cell carrying `source_commit_override_from` with a non-string value (e.g., integer 42) is rejected by the field's type constraint.
    - A static-source-scan test that the schema patch is purely additive: the diff for `schemas/local_label_admitted.schema.json` adds exactly one entry to `cells.items.properties` and does not alter `required`, `additionalProperties`, or the `schema_version` pin (still `const: 4`).

- AC-2: Platform/SKU registry extension for `rtx_5060_laptop` across all lockstep surfaces (per `BL-20260507-arch-token-multi-surface-lockstep`). The five surfaces below all admit `rtx_5060_laptop` mapped to compute capability `(12, 0)`:
  - `workloads/simbench_bw/run_suite.py::_infer_gpu_sku` adds a branch for CC `(12, 0)` + device-name substring match on `"5060"` returning `"rtx_5060_laptop"` (placed before the existing `"blackwell_consumer_unknown"` fallback).
  - `workloads/simbench_bw/update_goldens.py` `--platform` choices list includes `"rtx_5060_laptop"`.
  - `workloads/simbench_bw/arch_resolver.py::PLATFORM_TO_CC_FAMILY` includes `"rtx_5060_laptop": (12, 0)`.
  - `workloads/simbench_bw/schemas/suite_results.schema.json` platform enum (if present) includes `"rtx_5060_laptop"`.
  - `workloads/simbench_bw/merge_evidence.py` platform allowlist (if present) includes `"rtx_5060_laptop"`.
  - Positive Tests:
    - `_infer_gpu_sku(props)` with `compute_capability_major=12, compute_capability_minor=0, device_name="NVIDIA GeForce RTX 5060 Laptop GPU"` returns `"rtx_5060_laptop"`.
    - `PLATFORM_TO_CC_FAMILY["rtx_5060_laptop"]` returns `(12, 0)`.
    - A cross-surface lockstep regression test that iterates the five surfaces and asserts `"rtx_5060_laptop"` is present in every one.
  - Negative Tests:
    - A drift between surfaces (e.g., adding `"rtx_5060_laptop"` to `_infer_gpu_sku` but not to `PLATFORM_TO_CC_FAMILY`) is detected by the cross-surface lockstep test.
    - `_infer_gpu_sku` on CC `(12, 0)` with `device_name="NVIDIA RTX PRO 6000 Blackwell"` returns `"rtx_pro_6000_blackwell"` (existing branch behavior preserved; the `"5060"` branch must not regress the `"pro 6000"` branch).

- AC-3: Activation-trace-orchestrator decoupling is DEFERRED to a separate follow-on loop. Per Codex Round 1: `tools/local_labels/activation_trace_orchestrator.py` hardcodes more than `_RTX_3090_TI_PROPS` (output path `rtx_3090_ti/<arch_dir>`, sidecar `device_name`, `gpu_sku`, manifest/compiler provenance); a partial lift would leave several hardcodes uncovered and risk reopening the M1/v7 scope-closure stall pattern. The label-sweep load-bearing path for AC-6 runs through `label_driver.sh` (already arch-agnostic per `label_driver.sh:213-237` capability gate), not `activation_trace_orchestrator.py`, so this deferral does NOT block any other AC in this loop.
  - Positive Tests (recording the deferral):
    - The plan-evolution / round-summary section explicitly records `activation_trace_orchestrator.py` decoupling as a deferred non-blocking item (NOT incomplete) and names the follow-on plan slug that will close it.
    - A static-source-scan test that `activation_trace_orchestrator.py` is NOT modified by this loop's commits (git-log audit assertion).
  - Negative Tests:
    - An attempted edit to `activation_trace_orchestrator.py` in any commit of this loop is flagged by the round-summary structural scanner.

- AC-4: Arch-token policy + sm_120 launch-parameter sidecars + `extend_launch_parameters.py` arch-filter lift.
  - AC-4.1: Arch-token policy is documented and enforced via test fixtures. The three sm_120 path tokens are:
    - **Label/comparison token**: `sm_120` (used in `label_admitted.json`'s `arch` field, `comparison_contract.toml`'s `arch_set`, `comparison_runner.py`'s 7-field join key).
    - **Device/capability/sidecar-directory token**: `sm_120-laptop` (used in `arch_capabilities.json`'s `supported_arches` list, `tools/hrd_profiler/device_properties.json` keys, `data/local_labels/launch_parameters_sidecars/<arch>/` subdirectory name).
    - **Trace/results platform-path token**: `rtx_5060_laptop/sm_120` (used in trace/results path templates that follow the existing `rtx_3090_ti/sm_86` pattern).
    - Boundary mapping is deterministic: `label_driver.sh:213-237` already maps `sm_120 → sm_120-laptop` for the capability gate; the convention extends without code change to other label-emitting surfaces.
    - Positive: a fixture cell with `arch: "sm_120"` schema-validates against `local_label_admitted.schema.json`; a sidecar directory at `data/local_labels/launch_parameters_sidecars/sm_120-laptop/` is recognized by the launch-parameter loader.
    - Negative: a cell with `arch: "sm_120-laptop"` accidentally appearing in `label_admitted.json` is rejected (the v4 schema's arch enum admits only `"sm_86"` and `"sm_120"`); a static-source-scan test rejects any `label_admitted.json` cell whose `arch` field is `"sm_120-laptop"`.
  - AC-4.2: Create `data/local_labels/launch_parameters_sidecars/sm_120-laptop/` directory parallel to the existing `sm_86/`. The directory may be initially empty; if a `.gitkeep` is needed to commit an empty directory, add it. Otherwise the operator-driven sweep in AC-6 populates the directory.
    - Positive: directory exists at `data/local_labels/launch_parameters_sidecars/sm_120-laptop/`; after AC-6 lands, sidecars for the 5 activation cells appear there.
    - Negative: the launch-parameter loader's lookup for `arch=sm_120-laptop` returns a typed exception when the directory is missing, NOT a silent fall-through to no-correction (preserves the fail-closed discipline).
  - AC-4.3: `tools/local_labels/extend_launch_parameters.py` admits `arch=sm_120` rows alongside the existing `arch=sm_86` filter (per Codex Round 1 — currently the script filters to sm_86 only despite `arch_capabilities.json` declaring `sm_120-laptop` support). The arch filter widens to `{"sm_86", "sm_120"}`.
    - Positive: a row with `arch=sm_120` is admitted by `extend_launch_parameters.py` and emits a sidecar under `data/local_labels/launch_parameters_sidecars/sm_120-laptop/`.
    - Negative: a row with an unrecognized arch (e.g., `sm_999`) is still rejected by the filter.
  - AC-4.4: Predictor-side code that currently reads from `data/local_labels/launch_parameters_sidecars/sm_86/` is parameterized on arch. On missing arch directory, the lookup raises a typed exception (e.g., `LaunchParameterSidecarDirectoryMissing`) rather than silently skipping the gridDim correction.
    - Positive: lookup for `arch=sm_86` returns existing sidecars; lookup for `arch=sm_120-laptop` returns the new sidecars from AC-6.
    - Negative: lookup for `arch=sm_120-laptop` BEFORE AC-6 lands raises the typed exception (loud failure mode; no silent miscalibration).

- AC-5: gridDim alpha cross-arch extrapolation metadata. Add a `grid_dim_alpha_source` field to the predictor-bundle schema `workloads/simbench_bw/schemas/cache_predictor.schema.json` as an optional field with enum `{"sm_86_calibrated", "sm_86_extrapolated", "uncalibrated"}`. Backward compatibility is preserved: legacy sm_86 predictor bundles emitted before this AC continue to schema-validate without the field. Newly emitted predictor bundles carry the field per these rules:
  - sm_86 predictor bundles (newly emitted): `grid_dim_alpha_source = "sm_86_calibrated"`; alpha value unchanged at `0.110`.
  - sm_120 predictor bundles (newly emitted): `grid_dim_alpha_source = "sm_86_extrapolated"`; alpha value REUSED at `0.110` (NOT scaled by SM-count ratio; not zeroed). This is the explicit cross-arch extrapolation tag.
  - The runner reads the field for diagnostic emission in the comparison bundle (per-cell `grid_dim_alpha_source` recorded in the included-cell diagnostic block, not gated on admission) but does NOT gate cell admission on the field's value.
  - Positive Tests:
    - A predictor bundle with `grid_dim_alpha_source = "sm_86_extrapolated"` validates against the patched schema.
    - A predictor bundle without the field (legacy shape) validates against the patched schema.
    - A predictor bundle on the 5060 laptop with `device = "NVIDIA GeForce RTX 5060 Laptop GPU"` emits `grid_dim_alpha_source = "sm_86_extrapolated"` with alpha = `0.110`.
    - The comparison bundle's per-cell included-cell diagnostic block surfaces the `grid_dim_alpha_source` value for any included cell whose predictor bundle carried the field.
  - Negative Tests:
    - A predictor bundle with `grid_dim_alpha_source = "made_up_value"` is schema-rejected.
    - A static-source-scan test that the runner does NOT call any admission-gate function on `grid_dim_alpha_source` (the field is metadata-only).
    - A predictor bundle on sm_120 that emits `alpha != 0.110` is rejected by a regression test (the alpha is REUSED, not scaled).

- AC-6: sm_120 NCU label sweep (operator stage; CLAUDE.md Rule 1 local-label exception). Operator runs `tools/local_labels/sweep_repeatability_matrix.py --live` against the candidate cell set on the RTX 5060 laptop. Required floor: at least 5 sm_120 cells admit at the unchanged `stddev_threshold = 0.03` with `evidence_kind = "live"`. Candidate cells:
  - 5 required activation cells (matching the existing sm_86 PlattCalibrator training cohort for direct cross-arch comparability):
    - `layernorm_fwd_bf16_sm80/timing_shape`
    - `rmsnorm_fwd_bf16_sm80/timing_shape`
    - `silu_fwd_bf16_sm80/correctness_shape`
    - `silu_fwd_bf16_sm80/timing_shape`
    - `softmax_safe_bf16_sm80/timing_shape`
  - 1 optional stretch cell (memory-stress family, non-blocking):
    - `dram_stream_stride_mix/timing_00` (carries prior sm_86 evidence; useful for cohort diversity but not required for floor satisfaction).
  - The pre-existing `data/local_labels/sm_120/fp32_fma_dep_chain/` artifacts (from prior M1 work) are NON-COUNTING for this loop's floor (per Codex Round 1 recommendation; that cell previously failed repeatability and is not part of the activation cohort).
  - Positive Tests:
    - After the operator sweep, `len([c for c in cells if c["arch"] == "sm_120" and c["evidence_kind"] == "live"]) >= 5` in `data/local_labels/label_admitted.json`.
    - Each new sm_120 cell strict-validates against the patched v4 schema from AC-1.
    - The 5 cells have `slug` matching the activation cohort list above (exact-string match for at least the 5 required slugs).
  - Negative Tests:
    - If fewer than 5 sm_120 cells admit, the round STALLS at AC-6; closure is NOT softened by relaxing the stddev threshold or by lowering the floor (DEC-6 hard-binary). The round summary records the stall, the per-cell stddev distribution, and the candidate follow-on (e.g., sweep additional cells; investigate per-cell repeatability variance source).
    - Synthetic-fixture cells with `evidence_kind = "synthetic"` do NOT count toward the floor (the live-only floor discipline from `cache_predictor_per_arch_diagnostics_plan.md` is preserved).
    - A sweep that loosens `stddev_threshold` below 0.03 in `repeatability_aggregator.py` is detected by the existing static-source-scan test in that file.

- AC-7: Cross-host merge audit. After AC-6 lands the new sm_120 cells, the operator (or a single audit script) invokes `tools/local_labels/merge_admitted_cells.py` to consolidate the new sm_120 sweep output into `data/local_labels/label_admitted.json`. Post-merge state: 44 sm_86 cells (carried) + ≥5 sm_120 cells (new) = ≥49 cells total. Strict v4 schema validation passes for the entire merged file (relying on AC-1's `source_commit_override_from` field admission).
  - Positive Tests:
    - `len([c for c in cells if c["arch"] == "sm_86"]) == 44` AND `len([c for c in cells if c["arch"] == "sm_120"]) >= 5` AND `len(cells) >= 49`.
    - `jsonschema.validate(merged_file, v4_schema)` passes for the entire merged file.
    - The 44 sm_86 cells are byte-identical to their pre-merge state (no silent re-aggregation or migration drift; verified via per-cell SHA-256 hash equality of canonical-JSON serialization).
  - Negative Tests:
    - Any cell that fails strict v4 schema validation blocks the merge (the existing `merge_admitted_cells.py` strict-validation gate is preserved).
    - A sm_86 cell whose value changes during the merge (e.g., a re-aggregation that recomputes `stddev_l2_sector`) is detected by the per-cell hash comparison.
    - A sm_120 cell whose `evidence_kind` is `synthetic` is rejected by the live-only floor check in AC-6 BEFORE reaching the merge.

- AC-8: Direct cross-arch admission audit via `comparison_runner.py`. Invoke `workloads/simbench_bw/comparison_runner.py` directly (NOT through `run_headline_cohort_predict_compare.py`, which is deferred per AC-3) with `--label-admitted-path data/local_labels/label_admitted.json` and a synthetic predictor-bundle fixture providing predictor bundles for BOTH arches: at least 5 bundles matching existing sm_86 admitted cells' 7-field join keys at `cache_layer = "L2"`, AND at least 5 bundles matching the new sm_120 admitted cells' 7-field join keys at `cache_layer = "L2"`. Floor satisfaction in `comparison_runner.py` depends on JOINED cells (the intersection of admitted labels with predictor bundles on the 7-field key), NOT on raw label counts (per Codex Round 1 mechanical correction).
  - Critical fixture constraint (per Codex Round 2): synthetic predictor bundles MUST use runner-recognized `device` strings, because `comparison_runner.py` derives `arch` from the `device` field, NOT from a predictor-bundle `arch` field. The runner's recognized strings are:
    - `"NVIDIA GeForce RTX 3090 Ti"` → arch `sm_86`
    - `"NVIDIA GeForce RTX 5060 Laptop GPU"` → arch `sm_120`
  - Floor satisfaction is computed over cells whose label-side `evidence_kind == "live"` (synthetic predictor bundles are fine because the predictor side is always model output; synthetic LABEL cells, by contrast, do not satisfy the live floor).
  - Assertions:
    - `top_level_admission_scope == "cross_arch_require_both_arches"` (because both arches have at least 5 joined live cells).
    - `per_arch.sm_86.coverage_gate_verdict.meets_floor == true` AND `per_arch.sm_120.coverage_gate_verdict.meets_floor == true`.
    - `result_kind == "admitted"` with a combined geomean reported (not `"typed_unavailable"`).
    - The bundle's `included_cells` carry per-cell `grid_dim_alpha_source` values from AC-5 (`sm_86_calibrated` for sm_86 cells; `sm_86_extrapolated` for sm_120 cells).
  - Positive Tests:
    - The fixture above produces `result_kind == "admitted"` with `top_level_admission_scope == "cross_arch_require_both_arches"`.
    - The bundle's `per_arch.sm_86.n_headline_cells >= 5` AND `per_arch.sm_120.n_headline_cells >= 5`.
    - A fresh re-run of the audit with the same fixture produces a byte-identical bundle (deterministic serialization).
  - Negative Tests:
    - A fixture that omits all sm_120 predictor bundles drops `result_kind` to `"typed_unavailable"` with `typed_reason = "headline_geomean_admitted_cell_count_below_threshold"` for sm_120 (single-arch floor fails; cross-arch gate cannot be satisfied).
    - A fixture that omits all sm_86 predictor bundles drops `result_kind` similarly for sm_86 (mirror failure mode).
    - A fixture that uses an unrecognized `device` string (e.g., `"NVIDIA RTX 9999"`) for sm_120 bundles routes those bundles to `typed_unavailable` because the runner cannot resolve arch from the device.
    - A fixture that uses synthetic LABEL cells (not just synthetic predictor bundles) with `evidence_kind = "synthetic"` does NOT satisfy the live-only floor; result_kind drops to `"typed_unavailable"`.

- AC-9: Test-first commit ordering + plan-token scanner + hygiene-grep gate (per `BL-20260509-strict-admission-contract-first`).
  - For each AC slice (AC-1, AC-2, AC-4, AC-5, AC-7, AC-8), the failing-test commit precedes the implementation commit; `git log` shows the test commit's SHA before the implementation commit's SHA for each slice. AC-3 (deferral) has no implementation commit; AC-6 (operator stage) has an evidence-only commit. AC-10 (round summary) has a documentation commit.
  - `tools/test_plan_token_scanner.py` returns the project's expected pass count (currently `23 passed` per recent commits) after every commit in this round.
  - Multi-file widened workflow-marker grep returns no matches in `tools/local_labels/` + `workloads/simbench_bw/`:
    - `grep -RInE 'DEC-|[Pp]lan line|M1 plan|M2 plan|upper-bound|lower-bound scope' tools/local_labels/ workloads/simbench_bw/` returns zero hits (subject to the genuinely-domain `[hit_rate_lower, hit_rate_upper]` schema-field exception).
  - Positive Tests:
    - `git log --oneline` for this loop's commits shows test-commit SHA before implementation-commit SHA for each AC slice.
    - `python tools/test_plan_token_scanner.py` returns `23 passed` after every commit.
    - The widened-grep command returns zero matches after every commit.
  - Negative Tests:
    - An implementation commit landing without a preceding test commit for the same AC slice is detected by a `git log` audit.
    - A plan-token marker introduced into any code or test file is detected by either the scanner or the widened grep.

- AC-10: Round summary records architecture-coupling metadata; explicit `Verified-live-sm_120` closure (DEC-4 RESOLVED). The round summary records the receiving session's full hardware-coupling discipline metadata per `BL-20260508-post-commit-recapture-discipline`:
  - GPU model: `NVIDIA GeForce RTX 5060 Laptop GPU` (verified via `nvidia-smi`).
  - Driver version: e.g., `595.71.05` (verified via `nvidia-smi`).
  - CUDA toolkit version: verified via `nvcc --version`.
  - NVBit build hash: verified via the tracer build artifact's recorded build hash.
  - Observed clocks / throttle state during the sweep: captured via `tools/local_labels/capture_observed_state.py` and pinned to the round.
  - Evidence status: `Verified-live-sm_120` only (NOT `Cross-arch-verified`).
  - The summary explicitly enumerates the deferred non-blocking items (AC-3 activation orchestrator decoupling; leave-arch-out calibration evaluation; sm_100 zero-shot; headline-cohort orchestrator parameterization) with named follow-on plan slugs queued for each.
  - Positive Tests:
    - The round summary file contains all 5 hardware metadata fields plus the explicit `Verified-live-sm_120` evidence-status string.
    - The round summary's `Plan Evolution Log` records AC-3 as `deferred non-blocking` (NOT `incomplete`).
  - Negative Tests:
    - A round summary that claims `Cross-arch-verified` without fresh sm_86 re-validation is rejected by a round-summary structural scanner (one-arch-at-a-time discipline lock).
    - A round summary that omits any of the 5 hardware metadata fields is flagged by the same scanner.
    - A round summary that frames a deferred item as `incomplete` (rather than `deferred non-blocking`) is flagged by the scanner (avoids re-triggering the v7/M1 structural-stall trap).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes AC-1 through AC-10 closed (AC-3 explicitly deferred and recorded as non-blocking; the remaining ACs all closed), AC-6's stretch cell `dram_stream_stride_mix/timing_00` admits in addition to the 5 required activation cells, and the optional preflight script (per Codex Round 1) asserts counts by `(arch, slug, shape_point)` before invoking the AC-8 comparison audit. The audit's comparison bundle reports per-cell `grid_dim_alpha_source` diagnostic values for all included cells. Evidence status closes at `Verified-live-sm_120` for the new sm_120 evidence; the existing 44 sm_86 cells carry forward at `Verified-live-sm_86` (no fresh sm_86 re-validation in this loop); the combined `Cross-arch-verified` status is documented as queued for a separate follow-on loop on the 3090 Ti.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes AC-1, AC-2, AC-4 (all sub-criteria), AC-5, AC-6 (exactly 5 required activation cells admitted; stretch cell may stall without blocking closure), AC-7, AC-8, AC-9, AC-10 closed. AC-3 is explicitly DEFERRED out of lower bound per Codex Round 1 recommendation (the activation-trace-orchestrator decoupling has more hardcodes than a partial lift would cover; deferring it preserves the bounded-MVP framing). Evidence status closes at `Verified-live-sm_120` only. Explicitly deferred items (recorded as non-blocking, not incomplete): AC-3 activation-trace-orchestrator decoupling; full headline-cohort orchestrator (`run_headline_cohort_predict_compare.py`) sm_120 parameterization; leave-arch-out cross-arch calibration evaluation (`tier2_rd_calibration_cross_arch_plan.md` AC-T4.3); sm_100 zero-shot prediction (`tier2_rd_calibration_cross_arch_plan.md` AC-T6); `Cross-arch-verified` promotion (requires fresh sm_86 re-validation on the 3090 Ti per CLAUDE.md "one-arch-at-a-time").

### Allowed Choices

- Can use:
  - The direct sweep + merge + comparison-runner path: `tools/local_labels/sweep_repeatability_matrix.py` (arch-agnostic per the existing implementation) + `tools/local_labels/merge_admitted_cells.py` (strict-validation append-only merge) + `workloads/simbench_bw/comparison_runner.py` (already-wired `require_both_arches_for_cross_arch` gate).
  - Synthetic predictor bundles for the AC-8 audit (matching real label cells' 7-field join keys with `device` strings recognized by the runner).
  - The existing `BL-20260507-arch-token-multi-surface-lockstep` regression-test pattern as the model for AC-2's cross-surface lockstep test.
  - `tools/local_labels/capture_observed_state.py` for AC-10's hardware-coupling metadata capture.
  - Standard library + `jsonschema` (already a project dependency) for schema validation.
  - The existing 5-cell PlattCalibrator activation cohort identifiers from `data/local_labels/calibration_head_v1.json` as the canonical AC-6 candidate set.
- Cannot use:
  - NCU invocation outside the `data/local_labels/` quarantine + `tools/local_labels/` driver (CLAUDE.md Rule 1; the local-label exception bounds NCU to this path).
  - GPU-driver-root operations: nvidia-smi clock locks, persistence mode changes, MIG mode changes, power-limit changes (CLAUDE.md Rule 2).
  - Privileged CUPTI surfaces: Events, Metrics, Range Profiling, PM Sampling, PerfWorks (CLAUDE.md Rule 3).
  - Host-driver work: NVIDIA driver install, DKMS, MOK, `modprobe nvidia`, reboot workflows (CLAUDE.md Rule 4).
  - Simulator revival: `nvasim`, Accel-Sim, simulator-validation plans (CLAUDE.md Rule 5).
  - Routing the AC-8 audit through `tools/local_labels/run_headline_cohort_predict_compare.py` (deferred per AC-3 rationale; that orchestrator carries additional sm_86 hardcodes — trace paths, launch-parameter sidecar paths, device geometry, expected identity, default device, output filenames — that this loop does NOT lift).
  - Editing `tools/local_labels/activation_trace_orchestrator.py` in any commit of this loop (per AC-3 deferral; protected by the round-summary structural scanner).
  - Loosening `stddev_threshold = 0.03` in `tools/local_labels/repeatability_aggregator.py` (the existing static-source-scan test enforces this).
  - Claiming `Cross-arch-verified` evidence status for this loop (the loop's closure is `Verified-live-sm_120` only per CLAUDE.md "Codex review is one-arch-at-a-time").
  - Plan-token markers (`AC-N`, `Milestone`, `Phase`, `Step`, `Round N`, `M2 plan`, `upper-bound` / `lower-bound scope`, `DEC-N`, `Plan line N`) in implementation code or comments (per `BL-20260509-hygiene-validation-grep-must-be-case-insensitive`).
  - Touching the 7-tuple `manifest_pairing_index` join key at `comparison_runner.py:99-107` (the prior plan's invariant carries forward).
  - Bumping `schema_version` on `local_label_admitted.schema.json` (AC-1's `source_commit_override_from` addition is purely additive and does not require a version bump per CLAUDE.md wire-extension policy).

> **Note on Deterministic Designs**: This plan is largely deterministic in its load-bearing pieces (the 5-cell activation cohort, the 5+ live-cell floor, the unchanged stddev_threshold = 0.03, the direct comparison-runner audit path, the `Verified-live-sm_120`-only closure, the AC-3 deferral). Choice axes are limited to the AC-6 stretch cell's inclusion (`dram_stream_stride_mix/timing_00` optional) and the optional preflight script in the upper bound. The DEC resolutions (DEC-1 through DEC-8) are all autonomously resolved per operator authorization granted prior to this plan's authoring.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The natural per-slice ladder, derived from the test-first discipline (`BL-20260509-strict-admission-contract-first`) + commit slice order from Codex Round 1:

1. **AC-1 slice (schema-drift remediation)**: failing test asserting `jsonschema.validate(label_admitted.json, v4 schema)` passes (currently fails on `source_commit_override_from`) → implementation patches the schema to add the optional field. After this commit, the merge path is unblocked.

2. **AC-2 slice (platform/SKU lockstep)**: failing cross-surface lockstep test asserting `rtx_5060_laptop` is present in all 5 surfaces → implementation edits to each surface. The five-surface change is atomic (per `BL-20260507-arch-token-multi-surface-lockstep`); one commit pair.

3. **AC-4 slice (arch-token policy + sidecar parameterization + extend_launch_parameters lift)**: failing tests for AC-4.1 (arch-token boundary mapping) + AC-4.2 (sidecar directory existence) + AC-4.3 (extend_launch_parameters arch filter widening) + AC-4.4 (predictor-side typed-error on missing arch directory) → implementation lands the directory, widens the filter, parameterizes the lookup. The arch-token policy doc lives in `tools/local_labels/SWEEP_RUNBOOK.md` or a sibling reference; the test fixtures pin the boundary mapping.

4. **AC-5 slice (gridDim alpha metadata)**: failing tests for schema admission of the optional enum field + emission rules per arch + diagnostic emission in the comparison bundle → implementation extends the schema, edits `cache_predictor.py` to emit the field on newly-generated bundles, edits `comparison_runner.py` to surface the diagnostic.

5. **AC-6 + AC-7 slice (operator stage + merge audit)**: the operator runs `sweep_repeatability_matrix.py --live` against the 5 (+1 stretch) cells; output lands in `data/local_labels/sweeps/<timestamp>/`; operator (or audit script) invokes `merge_admitted_cells.py` to consolidate; round summary records the device/driver/CUDA/clock metadata per AC-10. This stage produces evidence commits (one for the sweep output + merged file; one for the round summary).

6. **AC-8 slice (direct comparison audit)**: failing test fixturing 5+5 synthetic predictor bundles with correct device strings → implementation invokes `comparison_runner.py` with the merged labels + bundles; asserts the cross-arch admission scope + per-arch meets_floor + result_kind=admitted. Optional preflight script asserting counts by `(arch, slug, shape_point)` lands here.

7. **AC-9 / AC-10 (continuous)**: scanner + summary closure validated after every commit. AC-10's round summary is the final commit of the loop.

### Relevant References

- `workloads/simbench_bw/schemas/local_label_admitted.schema.json` — v4 admitted-cell schema; AC-1 patches `cells.items.properties` to add `source_commit_override_from`.
- `workloads/simbench_bw/comparison_runner.py:2780-2793` — admission gate already conditional on `contract.coverage_gate.require_both_arches_for_cross_arch`; AC-8 exercises this end-to-end.
- `workloads/simbench_bw/comparison_runner.py:1024-1029` — `CoverageGate` dataclass field.
- `workloads/simbench_bw/comparison_runner.py:1276-1303` — contract loader validation.
- `workloads/simbench_bw/comparison_runner.py:540-584` + `301-332` — `check_unsupported_blackwell_memory_path()` and `check_calibration_loocv_p90_residual()` fire-paths (verification-only this loop).
- `workloads/simbench_bw/comparison_contract.toml` lines 119, 122 — `smoke_test_floor_per_arch = 5` and `require_both_arches_for_cross_arch = true`.
- `workloads/simbench_bw/tests/test_require_both_arches_for_cross_arch.py` — 9 passing tests covering both gate modes with synthetic fixtures (the gen-idea draft incorrectly estimated 11; Codex Round 1 verified 9).
- `workloads/simbench_bw/run_suite.py::_infer_gpu_sku` — AC-2 SKU branch; existing branches handle CC (8, 6), (9, 0), (12, 0) + "pro 6000" name match.
- `workloads/simbench_bw/arch_resolver.py::PLATFORM_TO_CC_FAMILY` — AC-2 registry; current entries: `h100`, `rtx_pro_6000_blackwell`, `rtx_3090_ti`.
- `workloads/simbench_bw/update_goldens.py` — AC-2 `--platform` choices.
- `workloads/simbench_bw/merge_evidence.py` — AC-2 platform allowlist (path: `workloads/simbench_bw/`, NOT `tools/local_labels/`; per Codex Round 1 correction).
- `workloads/simbench_bw/cache_predictor.py` (around line 2601) — `_GRIDDIM_PENALTY_ALPHA = 0.110`; AC-5 emits `grid_dim_alpha_source` field on predictor bundles.
- `workloads/simbench_bw/schemas/cache_predictor.schema.json` — AC-5 adds optional enum field.
- `tools/local_labels/label_driver.sh` lines 213-237 — arch-capability gate; already maps `sm_120 → sm_120-laptop`; load-bearing for AC-6.
- `tools/local_labels/sweep_repeatability_matrix.py` line 873 — arch-token mapping; arch-agnostic sweep harness used in AC-6.
- `tools/local_labels/extend_launch_parameters.py` — AC-4.3 arch filter widening.
- `tools/local_labels/merge_admitted_cells.py` — AC-7 append-only merge with strict v4 validation.
- `tools/local_labels/SWEEP_RUNBOOK.md` lines 333-373 — wall-clock budget reference for the operator sweep.
- `tools/local_labels/check_admission_floor.py` — `--cross-arch --floor 5 --all-source-commits` ready-to-use audit verifier.
- `tools/local_labels/capture_observed_state.py` — AC-10 hardware-coupling metadata capture.
- `tools/arch_capabilities.json` lines 379-383, 417, 422-425 — `RESTRICTED TO sm_86` notes that this loop's AC-2 + AC-4 partially lift (the full lift remains queued).
- `tools/nvbit_tracer/tracer_tool/memory_op_class.h:176-189` and sibling files — Blackwell TMEM opcode classification (already shipped; verification-only this loop).
- `data/local_labels/label_admitted.json` — 44 sm_86 cells; 43 carry `source_commit_override_from` (the AC-1 blocker).
- `data/local_labels/calibration_head_v1.json` — PlattCalibrator + 5 training cells (the AC-6 candidate set).
- `data/local_labels/sm_120/fp32_fma_dep_chain/` — prior M1 cell artifacts; NON-COUNTING for AC-6 (per Codex Round 1).
- `plans/tier2_rd_calibration_cross_arch_plan.md` lines 62-78 — explicit DEFERRED-AC list; this loop closes AC-T3 + the residual AC-T7 wiring; defers AC-T4 + AC-T6.
- `plans/cache_predictor_m2_label_stabilization_plan.md` — M2 AC-5 per-arch split discipline that this loop respects (sm_120 leg only; sm_86 leg deferred to follow-on).
- `plans/cache_predictor_per_arch_diagnostics_plan.md` — DEC-7 history showing `require_both_arches_for_cross_arch` was deferred but later wired (confirmed in this loop's verification).
- `CLAUDE.md` — Two-Device Local Development Model + wire-extension policy + Rules 1-5 + Codex review one-arch-at-a-time discipline.
- `HANDOFF.md` (current state is STALE per draft; should be re-derived from plan files + git log this loop) — historical v8 plan precedent at lines 1070-1102 ("sm_120 IS the mainline on the laptop").

## Dependencies and Sequence

### Milestones

This plan advances a single milestone (Cross-Arch Bounded First sm_120 Label Admission Loop) within one RLCR loop on the RTX 5060 laptop.

1. **Cross-Arch Bounded First sm_120 Label Admission Loop**: closes within one RLCR loop on the laptop, slice-by-AC.
   - Phase A: schema-drift remediation (AC-1 test commit + impl commit). Architecture-independent; unblocks merge.
   - Phase B: platform/SKU lockstep (AC-2 test commit + impl commit). Architecture-independent.
   - Phase C: arch-token policy + sidecar + extend_launch_parameters (AC-4 test commits + impl commits per sub-criterion). Architecture-coupled to sm_120 paths; load-bearing for AC-6 sidecar emission + AC-8 audit.
   - Phase D: gridDim alpha metadata (AC-5 test commit + impl commit). Architecture-coupled; load-bearing for AC-8 diagnostic emission.
   - Phase E: AC-3 deferral recording (single documentation commit; no code change). Operator action: queue follow-on plan slug.
   - Phase F: live sweep (AC-6 operator stage; one evidence commit per sweep verdict). Architecture-coupled to sm_120 hardware.
   - Phase G: cross-host merge audit (AC-7 evidence commit + assertion script). Architecture-independent but architecture-conditioned (asserts per-arch counts).
   - Phase H: direct comparison audit (AC-8 test commit + impl commit + optional preflight script). Architecture-conditioned; load-bearing for the closure claim.
   - Phase I: round summary + hygiene closure (AC-9 / AC-10 continuous + final round-summary commit). Records evidence status `Verified-live-sm_120` and queued follow-on items.

### Dependency Graph

- AC-1 has no in-loop dependencies; lands first. Architecture-independent.
- AC-2 depends on AC-1 conceptually (the merge path must be unblocked before testing platform parity end-to-end), but the registry edits themselves are independent; lands second per commit slice order.
- AC-3 is a deferral record; no in-loop dependencies; lands as a single doc commit anytime in Phase E.
- AC-4 depends on AC-2 (the platform/SKU registry must admit `rtx_5060_laptop` before AC-4's arch-token policy can be enforced); lands third per commit slice order.
- AC-5 depends on AC-4 (the arch token must be locked before predictor bundles can carry per-arch metadata reliably); lands fourth.
- AC-6 depends on AC-1, AC-2, AC-4, AC-5 all green (the schema must accept the labels; the platform registry must recognize the SKU; the arch-token boundary must be locked; the predictor bundle metadata schema must accept the new field). AC-6 is the operator stage; lands in Phase F.
- AC-7 depends on AC-6 (the sweep output must exist before merging) AND AC-1 (the schema must accept the carried 44 sm_86 cells); lands in Phase G.
- AC-8 depends on AC-5 (predictor bundles must carry the metadata field for diagnostic surfacing) AND AC-7 (the merged labels must exist for the runner to load) AND AC-2 (the runner's device-to-arch mapping must be in lockstep with the SKU registry); lands in Phase H.
- AC-9 is continuous (every commit must pass the scanner + grep gate).
- AC-10 depends on AC-6 + AC-7 + AC-8 all green (the round summary records the closure evidence); lands as the final commit in Phase I.

There are no out-of-loop tooling dependencies. The cross-host transfer is a project-discipline dependency (CLAUDE.md "Session Handoff" + the project's `HANDOFF.md`), NOT a tooling dependency for this plan; this loop runs entirely on the RTX 5060 laptop with the existing artifacts carried via prior transports.

## Task Breakdown

Each task carries exactly one routing tag.

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1a | Failing test asserting `jsonschema.validate(label_admitted.json, v4 schema)` passes (currently fails on `source_commit_override_from`). Tests fail at this commit. | AC-1, AC-9 | coding | - |
| task1b | Implementation: patch `workloads/simbench_bw/schemas/local_label_admitted.schema.json` to add `source_commit_override_from` as optional string field on `cells.items.properties`; preserve `additionalProperties: false`; do NOT bump `schema_version`. Tests from task1a turn green. | AC-1 | coding | task1a |
| task2a | Failing cross-surface lockstep test asserting `"rtx_5060_laptop"` is present in `_infer_gpu_sku` branch logic, `update_goldens.py --platform` choices, `PLATFORM_TO_CC_FAMILY`, `suite_results.schema.json` platform enum (if present), `merge_evidence.py` platform allowlist (if present). Tests fail. | AC-2, AC-9 | coding | task1b |
| task2b | Implementation: extend `_infer_gpu_sku` with CC `(12, 0)` + `"5060"` device-name match → `"rtx_5060_laptop"`; add `"rtx_5060_laptop"` to `update_goldens.py --platform` choices, `PLATFORM_TO_CC_FAMILY`, `suite_results.schema.json` platform enum, `merge_evidence.py` platform allowlist. Tests from task2a turn green. | AC-2 | coding | task2a |
| task3 | Documentation commit: record AC-3 (activation-trace-orchestrator decoupling) as deferred non-blocking in the round contract / round summary; name the follow-on plan slug (e.g., `activation_trace_orchestrator_sm120_decoupling_plan`); add a static-source-scan test asserting `activation_trace_orchestrator.py` is NOT modified by this loop's commits. | AC-3, AC-9 | coding | task2b |
| task4a | Failing tests for AC-4: arch-token policy boundary mapping (`sm_120` for labels; `sm_120-laptop` for devices/sidecars; `rtx_5060_laptop/sm_120` for trace/results); sidecar directory lookup for `sm_120-laptop` returns sidecars when present; `extend_launch_parameters.py` admits `arch=sm_120` rows; predictor-side launch-parameter lookup raises typed exception on missing arch directory. Tests fail. | AC-4, AC-9 | coding | task3 |
| task4b | Implementation: create `data/local_labels/launch_parameters_sidecars/sm_120-laptop/` (with `.gitkeep` if needed); widen `extend_launch_parameters.py` arch filter to `{"sm_86", "sm_120"}`; parameterize predictor-side launch-parameter lookup on arch; add static-source-scan test rejecting `arch="sm_120-laptop"` literal in `label_admitted.json` cells. Tests from task4a turn green. | AC-4 | coding | task4a |
| task5a | Failing tests for AC-5: schema admits `grid_dim_alpha_source` as optional enum field on predictor bundles; sm_86 bundles emit `sm_86_calibrated`; sm_120 bundles emit `sm_86_extrapolated`; alpha value REUSED at 0.110 on sm_120 (not scaled); runner surfaces the field in per-cell diagnostic block; runner does NOT gate admission on the field. Tests fail. | AC-5, AC-9 | coding | task4b |
| task5b | Implementation: extend `workloads/simbench_bw/schemas/cache_predictor.schema.json` with optional `grid_dim_alpha_source` enum field; edit `cache_predictor.py` to emit the field on newly-generated bundles per arch; edit `comparison_runner.py` to surface the field in the per-cell included-cell diagnostic block. Tests from task5a turn green. | AC-5 | coding | task5a |
| task6 | Operator stage: run `tools/local_labels/sweep_repeatability_matrix.py --live` on the RTX 5060 laptop against the 5-cell activation cohort (layernorm/rmsnorm/silu_correctness/silu_timing/softmax_safe) + 1 stretch cell (dram_stream_stride_mix/timing_00); ensure ≥5 cells admit at `stddev_threshold = 0.03` with `evidence_kind = "live"`; sidecar artifacts land under `data/local_labels/sweeps/<timestamp>/`. Evidence commit. | AC-6 | coding | task5b |
| task7 | Run `tools/local_labels/merge_admitted_cells.py` against existing `label_admitted.json` + the new sm_120 sweep output; verify post-merge state (44 sm_86 + ≥5 sm_120 cells; strict v4 schema validation passes; per-cell SHA-256 equality for the 44 carried sm_86 cells). Evidence commit. | AC-7 | coding | task6 |
| task8a | Failing test fixturing synthetic predictor bundles for BOTH arches (5 sm_86 with `device = "NVIDIA GeForce RTX 3090 Ti"` + 5 sm_120 with `device = "NVIDIA GeForce RTX 5060 Laptop GPU"`); asserting `comparison_runner.py` produces `result_kind=admitted` with `top_level_admission_scope=cross_arch_require_both_arches` and per-arch meets_floor=true for both. Tests fail. | AC-8, AC-9 | coding | task7 |
| task8b | Implementation: optional preflight script asserting counts by `(arch, slug, shape_point)` before the audit; invoke `comparison_runner.py` with the merged labels + fixture bundles; verify the assertions. Tests from task8a turn green. | AC-8 | coding | task8a |
| task9 | Round summary commit: record GPU model, driver version, CUDA toolkit version, NVBit build hash, observed clocks/throttle state via `capture_observed_state.py`; record evidence status `Verified-live-sm_120` only; enumerate deferred non-blocking follow-on items with named plan slugs. | AC-10 | coding | task8b |
| task10 | Independent Codex review of the closure summary via `/humanize:ask-codex`. Analyze whether the round closure shape correctly avoids the v7/M1 stop-hook structural-stall trap; verify no `Cross-arch-verified` overclaim. | AC-10 | analyze | task9 |

## Claude-Codex Deliberation

### Agreements

- The bounded scope (sm_120 IS the mainline on the laptop; no host-constrained deferrals) is directionally right.
- The 5-cell sm_120 live floor matches the existing PlattCalibrator activation training cohort, making the cross-arch comparison directly defensible without expanding the cohort beyond what the calibration head already covers.
- No `stddev_threshold` softening; no `Cross-arch-verified` overclaim; direct `comparison_runner.py` audit path (NOT through the headline-cohort orchestrator).
- The pre-existing schema-drift on `source_commit_override_from` is a real blocker that must be remediated BEFORE any merge (verified: 43 of 44 cells carry the field; current jsonschema.validate fails).
- The `require_both_arches_for_cross_arch` admission gate is already wired in `comparison_runner.py` (lines 2780-2793) with 9 passing tests in `test_require_both_arches_for_cross_arch.py` (correction: the gen-idea draft estimated 11; Codex Round 1 verified 9).
- The NVBit Blackwell TMEM opcode tracer extension is already shipped (`MemoryOpClass::kTmem = 10`; classifier coverage complete; proto + consumer schema widened; 216 NVBit tests + 93 subtests pass per Codex Round 1 verification). TMEM routes as unsupported-memory-path sidecar (`path_resolved=true`); modeling TMEM as a cache path is out of scope.
- DEC-1 admit-as-optional resolution is sound: 43/44 cells already carry the field; additive widening preserves provenance without bumping schema_version.
- DEC-2 5-cell floor + 1-stretch resolution is sound: matches the Platt training cohort; stretch optional doesn't block closure.
- DEC-3 reuse-0.110-with-metadata-tag resolution is sound: smallest surface; preserves predictor function; metadata surfaces the cross-arch extrapolation explicitly.
- DEC-4 Verified-live-sm_120-only resolution is required per CLAUDE.md "Codex review is one-arch-at-a-time" discipline.
- DEC-5 direct-comparison_runner-not-headline-orchestrator resolution is sound: avoids lifting more hardcodes than the bounded MVP scope warrants; the headline orchestrator has trace path, launch-parameter sidecar path, device geometry, expected identity, default device, output filename hardcodes that would balloon the scope.
- DEC-6 hard-binary-at-5 resolution is consistent with the original draft's binary closure framing; the stretch cell provides slack.
- DEC-7 AC-3-fully-deferred resolution is sound (Codex Round 1 recommendation): activation_trace_orchestrator.py has more hardcodes than a partial lift would cover; deferring preserves the bounded-MVP framing.
- DEC-8 arch-token-policy-three-tokens resolution makes the boundary mapping explicit: `sm_120` for labels; `sm_120-laptop` for devices/capabilities/sidecars; `rtx_5060_laptop/sm_120` for trace/results paths.

### Resolved Disagreements

- **AC-8 floor satisfaction mechanics** (Codex Round 1 REQUIRED_CHANGE): the gen-idea draft framed the AC-8 audit as if 5 sm_120 predictor bundles alone would let sm_86=44 meet floor. Codex pointed out that `comparison_runner.py` counts floor satisfaction from JOINED cells (the intersection of admitted labels with predictor bundles on the 7-field key), not from raw labels. **Resolution**: AC-8 fixture requires synthetic predictor bundles for BOTH arches (≥5 sm_86 + ≥5 sm_120), each matching real admitted labels' 7-field join keys at `cache_layer = "L2"`. Without this, the audit always drops to `typed_unavailable` for the arch with no joined cells.
- **AC-8 device-string fixture constraint** (Codex Round 2 REQUIRED_CHANGE): synthetic predictor bundles must use runner-recognized `device` strings because the runner derives arch from `device`, not from a predictor-bundle `arch` field. **Resolution**: the fixture pins `device = "NVIDIA GeForce RTX 3090 Ti"` for sm_86 bundles and `device = "NVIDIA GeForce RTX 5060 Laptop GPU"` for sm_120 bundles. The AC-8 negative test surface includes a fixture with an unrecognized device string to verify the typed_unavailable arm.
- **AC-2 path correction** (Codex Round 1 REQUIRED_CHANGE): the gen-idea draft placed `merge_evidence.py` under `tools/local_labels/`; Codex verified the actual path is `workloads/simbench_bw/merge_evidence.py`. **Resolution**: AC-2 surface list corrected to the proper path.
- **AC-3 deferral vs partial lift** (Codex Round 1 RECOMMENDED): the gen-idea draft proposed a partial lift of `_RTX_3090_TI_PROPS` only; Codex pointed out that the activation orchestrator hardcodes more (output path, sidecar device_name, gpu_sku, manifest/compiler provenance), so a partial lift would leave hardcodes uncovered. **Resolution**: AC-3 fully deferred out of lower bound. The load-bearing path for AC-6 runs through `label_driver.sh` (already arch-agnostic), not the activation orchestrator, so the deferral does not block any other AC.
- **AC-4 arch-token ambiguity** (Codex Round 1 REQUIRED_CHANGE): the gen-idea draft proposed `data/local_labels/launch_parameters_sidecars/sm_120-laptop/` but didn't lock the boundary policy preventing `sm_120-laptop` from accidentally appearing in `label_admitted.json`'s `arch` field. **Resolution**: AC-4.1 documents the three sm_120 path tokens explicitly + a negative static-source-scan test rejecting `arch="sm_120-laptop"` in label cells.
- **AC-4 extend_launch_parameters arch filter** (Codex Round 1 REQUIRED_CHANGE): the gen-idea draft did not mention `extend_launch_parameters.py`'s sm_86-only arch filter. Codex pointed out that the script filters arch=sm_86 only despite `arch_capabilities.json` declaring sm_120-laptop support, so the sidecar emission for sm_120 would silently produce nothing. **Resolution**: AC-4.3 widens the arch filter to `{"sm_86", "sm_120"}`.
- **AC-5 gridDim alpha semantics** (Codex Round 1 ALTERNATIVE_DIRECTIONS): the gen-idea draft proposed `_GRIDDIM_PENALTY_ALPHA → null` with a new typed-unavailable reason `gridDim_penalty_alpha_uncalibrated_for_arch`. Codex preferred reusing `0.110` with an explicit metadata tag, avoiding the schema-bump-to-add-a-new-typed-reason surface and the predictor-side gating risk. **Resolution**: DEC-3 reuse-0.110-with-`grid_dim_alpha_source`-metadata-tag; field is optional (backward-compat for legacy bundles); runner reads but does not gate.
- **AC-5 diagnostic emission location** (Codex Round 2 OPTIONAL_IMPROVEMENT): clarification on where the runner surfaces the `grid_dim_alpha_source` value. **Resolution**: AC-5 emits the field in the per-cell included-cell diagnostic block of the comparison bundle (not the bundle-level summary), preserving the per-cell traceability.
- **Test count correction** (Codex Round 1 TECHNICAL_GAP): the gen-idea draft stated 11 tests pass in `test_require_both_arches_for_cross_arch.py`; Codex ran the file and observed 9 passes. **Resolution**: plan now states 9 tests (the test count is informational; the existing test coverage is sufficient regardless of exact count).
- **Non-counting sm_120 prior cell** (Codex Round 1 OPTIONAL_IMPROVEMENT): the gen-idea draft did not explicitly mark `data/local_labels/sm_120/fp32_fma_dep_chain/` as non-counting. **Resolution**: AC-6 explicitly notes the prior cell is non-counting for this loop's floor.

### Convergence Status

- Final Status: `converged`. Codex Round 2 returned no `REQUIRED_CHANGES` against the v2 candidate; the single Round 2 REQUIRED_CHANGE (AC-8 device-string fixture constraint) plus the two Round 2 OPTIONAL_IMPROVEMENTS (AC-4 three-token spell-out + AC-5 diagnostic-emission location) are all folded into the final plan above. No UNRESOLVED items.

## Pending User Decisions

All 8 DECs were autonomously resolved during plan authoring per user authorization granted prior to plan generation (operator-stated direction: "From now on, whenever there are any pending user DECs, discuss with codex and make the best decision yourself"). DECs are recorded here for traceability; no items remain `PENDING`.

- DEC-1: Should `source_commit_override_from` be admitted into the v4 label schema as an optional audit annotation, or moved out of `label_admitted.json` before merge?
  - Claude Position: admit as optional (additive widening; preserves the audit-trail provenance carried by 43 of 44 cells; no schema_version bump needed).
  - Codex Position: open question with both options viable (no strong preference).
  - Tradeoff Summary: admit-as-optional preserves audit provenance and avoids destructive removal of carried evidence; remove-before-merge would lose the provenance and risk re-introduction. Per CLAUDE.md wire-extension policy, additive widening is the preferred path for new fields that consumers must accept without breaking existing emission.
  - Decision Status: `RESOLVED: admit as optional`. Implementation: AC-1 patches `cells.items.properties` additively; no schema_version bump; existing 44 cells strict-validate post-patch.

- DEC-2: Should the sm_120 cohort be exactly the 5 activation cells matching the Platt training cohort, or should one memory-stress / GEMM cell be mandatory?
  - Claude Position: 5 activation cells as strict floor; 1 stretch cell (`dram_stream_stride_mix/timing_00`) optional.
  - Codex Position: agrees the activation cohort matches the calibration head's training distribution and is the smallest defensible floor; recommends adding at least one memory-stress cell to the stretch set.
  - Tradeoff Summary: activation-only floor matches the existing calibration training distribution and produces directly comparable cross-arch evidence; mandating a memory-stress cell would expand wall-clock budget + binary-build risk on the laptop; optional stretch addresses the cohort-diversity risk without expanding the floor.
  - Decision Status: `RESOLVED: 5-floor + 1-stretch`. Implementation: AC-6 lists the 5 required activation cells + the 1 optional stretch cell; stretch is non-blocking; cohort diversity is recorded as a known scope limitation in the round summary.

- DEC-3: Should the MVP reuse sm_86 gridDim alpha `0.110` with an explicit extrapolation tag, disable the correction (`alpha = 0`) and tag as uncalibrated, or emit a new typed-unavailable reason?
  - Claude Position (initial): emit new typed-unavailable reason (`gridDim_penalty_alpha_uncalibrated_for_arch`).
  - Codex Position: prefer reuse-0.110-with-metadata-tag; do NOT scale alpha by SM-count ratio (the formula already uses per-arch SM count).
  - Tradeoff Summary: reuse-0.110-with-tag preserves predictor function and surfaces the cross-arch extrapolation uncertainty via metadata; disable-and-tag-uncalibrated is more pessimistic than necessary (the gridDim penalty has the same FORM across arches; only magnitude may be miscalibrated); new typed-unavailable reason requires a schema change + admission-gate widening that the bounded MVP scope doesn't warrant.
  - Decision Status: `RESOLVED: reuse 0.110 with grid_dim_alpha_source metadata tag`. Implementation: AC-5 adds optional enum field to predictor bundle schema; sm_120 bundles emit `sm_86_extrapolated`; runner reads for diagnostic surfacing but does not gate admission.

- DEC-4: Is the closure target `Verified-live-sm_120` only, with `Cross-arch-verified` deferred until a later leave-arch-out / cross-arch evaluation?
  - Claude Position: `Verified-live-sm_120` only; `Cross-arch-verified` deferred per CLAUDE.md "one-arch-at-a-time".
  - Codex Position: agrees; do not claim `Cross-arch-verified` unless both arches are live and admitted under the same cross-arch audit.
  - Tradeoff Summary: claiming `Cross-arch-verified` from this laptop alone would violate CLAUDE.md's discipline (the receiving host cannot independently verify the other arch); explicit deferral preserves the discipline and avoids re-triggering the v7/M1 structural-stall trap.
  - Decision Status: `RESOLVED: Verified-live-sm_120 only`. Implementation: AC-10 round summary records the closure status explicitly; structural scanner rejects `Cross-arch-verified` claim without fresh sm_86 re-validation; the combined `Cross-arch-verified` follow-on is queued as a named plan.

- DEC-5: Should the AC-8 audit route through `comparison_runner.py` directly (with synthetic predictor bundles), or through `run_headline_cohort_predict_compare.py` (the headline orchestrator)?
  - Claude Position (initial): route through `run_headline_cohort_predict_compare.py` (the gen-idea draft's audit step).
  - Codex Position: direct `comparison_runner.py` path; the headline orchestrator is NOT sm_120-ready (trace paths, launch-parameter sidecars, device geometry, expected identity, default device, output filenames all sm_86/RTX 3090 Ti).
  - Tradeoff Summary: direct path avoids lifting more hardcodes than the bounded MVP scope warrants; headline orchestrator path would require AC-3-equivalent work + headline orchestrator parameterization to make the audit viable, ballooning the loop scope and likely re-triggering the M1/v7 stall pattern.
  - Decision Status: `RESOLVED: direct comparison_runner.py path`. Implementation: AC-8 invokes `comparison_runner.py` with synthetic predictor bundles for both arches; headline orchestrator sm_120 parameterization is queued as a separate follow-on plan.

- DEC-6: What is the closure shape if fewer than 5 sm_120 cells admit (e.g., 3 admit, 2 fail repeatability)? Hard-binary or stretch goal?
  - Claude Position: hard-binary at 5; round STALLS if floor not met; closure NOT softened.
  - Codex Position: agrees; the stretch cell provides slack but does not lower the floor.
  - Tradeoff Summary: hard-binary preserves the contract floor's integrity; soft-floor (typed_unavailable when <5) would erode the cross-arch closure claim; the stretch cell hedges against per-cell failure without lowering the floor.
  - Decision Status: `RESOLVED: hard-binary at 5`. Implementation: AC-6 hard-binary; if <5 cells admit, the round summary records the stall + per-cell stddev distribution + the candidate follow-on (e.g., expand candidate set; investigate per-cell variance source).

- DEC-7: Should AC-3 (activation-trace-orchestrator decoupling) be in lower bound, upper bound, or fully deferred?
  - Claude Position (initial): lower bound (partial lift of `_RTX_3090_TI_PROPS` only).
  - Codex Position: fully defer; activation orchestrator hardcodes more than `_RTX_3090_TI_PROPS` (output path, sidecar device_name, gpu_sku, manifest/compiler provenance); a partial lift would leave hardcodes uncovered.
  - Tradeoff Summary: partial lift in this loop would risk re-triggering the M1/v7 scope-closure stall pattern by leaving the activation orchestrator in a half-parameterized state; full deferral preserves the bounded-MVP framing; the load-bearing path for AC-6 (label sweep) runs through `label_driver.sh` (already arch-agnostic), not the activation orchestrator, so the deferral does NOT block AC-6 or any other AC.
  - Decision Status: `RESOLVED: fully defer`. Implementation: AC-3 is a deferral record; named follow-on plan slug enqueued; round summary records as `deferred non-blocking` (NOT `incomplete`); static-source-scan test rejects edits to `activation_trace_orchestrator.py` in this loop.

- DEC-8: What is the arch-token policy for the sm_120 paths?
  - Claude Position: the three sm_120 path tokens are `sm_120` (label/comparison), `sm_120-laptop` (device/capability/sidecar), `rtx_5060_laptop/sm_120` (trace/results); deterministic boundary mapping per `label_driver.sh:213-237` precedent.
  - Codex Position: agrees; recommends explicit spell-out to prevent implementation drift.
  - Tradeoff Summary: explicit three-token policy prevents the arch-token ambiguity that would otherwise let `sm_120-laptop` accidentally appear in `label_admitted.json` cells and break the cross-arch join; the existing `label_driver.sh:213-237` boundary mapping is the precedent.
  - Decision Status: `RESOLVED: three-token policy locked`. Implementation: AC-4.1 documents the three tokens + the boundary mapping; AC-4.1 negative test rejects `arch="sm_120-laptop"` in `label_admitted.json` cells.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Phase`, `Step`, `Round N`, `M2 plan`, `upper-bound`, `lower-bound scope`, `DEC-N`, `Plan line N`, or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead. Examples:
  - Reference the contract terms by their field names (`require_both_arches_for_cross_arch`, `smoke_test_floor_per_arch`, `grid_dim_alpha_source`), not by `AC-N` labels.
  - Reference the typed exclusion tokens by their literal names (`headline_geomean_admitted_cell_count_below_threshold`, `cross_arch_require_both_arches`, `target_arch_unlabeled_extrapolation`), not by `AC-N` labels.
  - The `tools/test_plan_token_scanner.py` gate enforces this; it must return its expected pass count (currently `23 passed` per recent commits) after every commit in this round.
  - The multi-file widened workflow-marker grep (`grep -RInE 'DEC-|[Pp]lan line|M1 plan|M2 plan|upper-bound|lower-bound scope' tools/local_labels/ workloads/simbench_bw/`) must return no matches except for genuinely-domain `[hit_rate_lower, hit_rate_upper]` schema-field references.

### Cross-Host Transfer Context

This plan was authored on the **RTX 5060 laptop (sm_120 Blackwell)**, which is also the execution host for this RLCR loop. There is no cross-host transfer associated with this loop's execution; the existing 44 sm_86 admitted cells in `data/local_labels/label_admitted.json` were transported to this host via prior cross-host transports (git fetch or raw scp). Per CLAUDE.md "Session Handoff", `.humanize/rlcr/` was confirmed absent on this host before the loop started (the receiving host's clean state). The current `HANDOFF.md` describes a prior M1→M2 transfer back to the 3090 Ti and is STALE relative to actual HEAD; this loop's round contract reads from `plans/tier2_rd_calibration_cross_arch_plan.md` AC-T3/T7 + the current `label_admitted.json` + git log, NOT from the stale `HANDOFF.md` framing.

### Follow-On Plans Queued

The following items are explicitly deferred from this loop and queued as named follow-on plans:

1. **Activation-trace-orchestrator decoupling**: full parameterization of `tools/local_labels/activation_trace_orchestrator.py` (output path, sidecar device_name, gpu_sku, manifest/compiler provenance). Tentative slug: `activation_trace_orchestrator_sm120_decoupling_plan.md`. Closes AC-3 of this loop.

2. **Headline-cohort orchestrator sm_120 parameterization**: full lift of `tools/local_labels/run_headline_cohort_predict_compare.py` and `run_headline_cohort_sweep.py` sm_120 readiness (trace paths, launch-parameter sidecars, device geometry, expected identity, default device, output filenames). Tentative slug: `headline_cohort_orchestrator_sm120_decoupling_plan.md`.

3. **Leave-arch-out cross-arch calibration evaluation**: extend `fit_calibration_head.py` + `calibration_evaluation_v1.schema.json` to compute leave-arch-out residuals (train sm_86 score sm_120; train sm_120 score sm_86) for the PlattCalibrator using the real sm_120 labels produced by this loop. Tentative slug: `tier2_cross_arch_calibration_leave_arch_out_plan.md`. Closes `tier2_rd_calibration_cross_arch_plan.md` AC-T4.3.

4. **sm_100 zero-shot prediction**: NVBit trace capture on a cloud B200 + zero-shot prediction emission under the non-root contract. Tentative slug: `tier2_cross_arch_sm100_zero_shot_plan.md`. Closes `tier2_rd_calibration_cross_arch_plan.md` AC-T6.

5. **Cross-arch-verified promotion**: fresh sm_86 re-validation on the RTX 3090 Ti + combined audit producing `Cross-arch-verified` evidence status. Tentative slug: `cross_arch_verified_promotion_sm86_revalidation_plan.md`. Closes the long-deferred cross-arch evidence claim per CLAUDE.md "one-arch-at-a-time".

--- Original Design Draft Start ---

# Bounded First Cross-Arch Closure With sm_120 Label Evidence

## Original Idea

cross-arch accuracy + hardware-config decoupling for the cache-hit-rate predictor on the RTX 5060 laptop (sm_120 Blackwell); land the long-deferred Tier-2-cross-arch follow-up loop — produce sm_120 NCU ground-truth labels under the local-label exception, empirically evaluate the existing sm_86-trained PlattCalibrator's leave-arch-out residuals against sm_120 labels, wire `require_both_arches_for_cross_arch` from silent TOML into the runner's admission gate, surface the two cross-arch typed reasons (`unsupported_blackwell_memory_path` + `target_arch_unlabeled_extrapolation`) + the AC-T7 LOOCV-residual reason, and lift sm_86 hardcodes that prevent honest cross-arch serving (the "Submission-track scope (sm_86 + RTX 3090 Ti)" framing in cache_predictor.py; `_RTX_3090_TI_PROPS` in activation_trace_orchestrator.py; the `rtx_3090_ti/sm_86` path hardcodes in 3 headline-cohort orchestrators; the missing `rtx_5060_laptop` entry in update_goldens.py --platform choices; the sm_86-tuned gridDim α=0.110 penalty). Scope is bounded BELOW the historical M1 + v7 scope-closure stalls — start with a narrow first cross-arch loop that can close on this laptop alone. Treat the sm_120 NCU label sweep as the load-bearing evidence (CLAUDE.md Rule 1 local-label exception covers it). Per CLAUDE.md "Two-Device Local Development Model", lift to `Verified-live-sm_120` only; combined with prior sm_86 evidence eventually becomes `Cross-arch-verified`. Critical context: 44 sm_86 cells already admitted at schema_version=4; PlattCalibrator chosen on 5 sm_86 activation training pairs; sm_120 admitted set is currently empty; HANDOFF.md is stale (talks about M1→M2 pivot back to 3090 Ti; reality is we're now back on the laptop for the deferred cross-arch frontier); .humanize/bitlesson.md absent on this host so BL discipline must be re-derived from plan files

## Primary Direction: Bounded MVP Cross-Arch Loop

### Rationale

Narrowest combination of label evidence + gate wiring that produces a complete first cross-arch closure on the laptop in one loop, deferring calibration eval, decoupling refactor, and tracer extension as follow-ons (mirrors BL-20260509-rlcr-recovery-round-bounded-scope).

### Approach Summary

Land a single bounded RLCR loop on the RTX 5060 laptop that closes at `Verified-live-sm_120` by combining (a) the minimum hardware-config decoupling needed to admit sm_120 evidence into the existing headline-cohort orchestrators, (b) a 5-cell sm_120 NCU label sweep matching the existing sm_86 activation training cohort (layernorm/rmsnorm/silu_correctness/silu_timing/softmax_safe) so the cohorts are directly comparable, and (c) a post-sweep admission audit that exercises the already-wired `require_both_arches_for_cross_arch` gate end-to-end across the combined sm_86 (44 cells) + sm_120 (≥5 cells) admitted set. The loop EXPLICITLY DEFERS: cross-arch calibration evaluation with leave-arch-out residuals (Alt-2; needs the new labels first), the full hardcode-lift refactor sweep across all queued surfaces (Alt-3; this loop ships only the minimum subset), and any sm_100 zero-shot work (out of all alts). Two empirically informed surprises from the parallel exploration reshape the scope: the cross-arch admission gate (Alt-4) is ALREADY WIRED in `comparison_runner.py:2780-2793` with passing tests in `test_require_both_arches_for_cross_arch.py`, and the NVBit Blackwell opcode tracer extension (Alt-5) is ALREADY SHIPPED (TMEM family classified to `MemoryOpClass::kTmem` value 10, proto schema extended, consumer schema widened, ~16 unit-test assertions green). Those two surprises shrink the loop's engineering surface to: (i) parameterize `_RTX_3090_TI_PROPS` into a `(arch, sku)`-keyed device-props factory and lift the three headline orchestrators' `rtx_3090_ti/sm_86` path templates per `tools/arch_capabilities.json` entries 379-383/417/422-425; (ii) add `"rtx_5060_laptop"` to `update_goldens.py --platform` choices and the corresponding entry in `_infer_gpu_sku()`; (iii) extend `_GRIDDIM_PENALTY_ALPHA` from a module constant to a per-arch dict with `sm_120` initially set to `null` (typed fallback) to avoid silently reusing the sm_86-tuned 0.110; (iv) operator-driven 5-cell sm_120 NCU sweep via existing `sweep_repeatability_matrix.py` (already arch-agnostic per Alt-1); (v) merge into `label_admitted.json` via existing `merge_admitted_cells.py`; (vi) audit run of `run_headline_cohort_predict_compare.py` confirming the gate emits `top_level_admission_scope = "cross_arch_require_both_arches"` and the new sm_120 leg admits at `evidence_kind = "live"`. Sequencing follows BL-20260509-strict-admission-contract-first (test-first per slice) + BL-20260508-post-commit-recapture-discipline (live sweep follows code commits; round summary records device/driver/CUDA/clock state). The cross-arch typed reason `unsupported_blackwell_memory_path` already has a production fire-path; only `target_arch_unlabeled_extrapolation` lacks a fire-path (deferred to a future zero-shot loop, NOT this loop). The AC-T7 `calibration_loocv_p90_residual_exceeded` reason is already wired (`comparison_runner.py:301-332`); recomputing the threshold check on the post-merge calibration_evaluation_v1.json is a verification step, not new code.

### Synthesis Notes (from gen-idea)

The Phase 3 exploration produced two narrative-altering surprises: the cross-arch admission gate (Alt-4) and the Blackwell TMEM opcode tracer extension (Alt-5) are both substantially complete in the current HEAD, materially shrinking the engineering surface needed to close a first cross-arch loop on the laptop. The primary (Alt-6 / Bounded MVP) absorbs those findings. If the user prefers a narrower primary, Alt-1 (Evidence-First) is the natural fallback. If the user prefers a broader primary, Alt-3 (Hardware-Config Decoupling Refactor) folds in. Alt-2 (Calibration Transfer Evaluation) is the natural NEXT loop after the primary closes. The sequencing implied by these findings is: this loop (primary) → next loop (Alt-2 with real labels + Alt-3 full refactor in parallel) → loop after (Cross-arch-verified promotion via fresh sm_86 re-validation on the 3090 Ti per CLAUDE.md "one-arch-at-a-time"). The BL-20260509-rlcr-recovery-round-bounded-scope discipline is the load-bearing structural principle.

--- Original Design Draft End ---
