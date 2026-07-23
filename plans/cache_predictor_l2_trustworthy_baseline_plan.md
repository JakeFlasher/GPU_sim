# Trustworthy L2 Sector Hit Rate Baseline on sm_86

## Goal Description

Replace the current "first-artifact" headline `comparison_bundle.json` (geomean 0.3289, CI [0.0945, 0.7533], 5 admitted activation cells all predicting `hit_rate=1.0000`) with a TRUSTWORTHY `Verified-live-sm_86` L2-sector-hit-rate predictor-vs-NCU geomean baseline produced by six dependency-ordered rounds: A0 (offline regeneration of stale bundles under current producer code), A1 (live re-collection of the 5 activation NCU labels with the R28 `--launch-skip` warmup-offset bug fix), B (predictor-saturation fixes, in strict order: diagnostics → model refinements → calibration head), C (microbench rehabilitation without lowering `MINIMUM_BIN_SAMPLE_COUNT=50`), Polluter (root-cause of the `dram_stream_stride_mix` trace-sidecar re-emission), and portable_lcd (restore traces or live-gate the test). A cross-cutting headline-validity guard lands in the LAST round, fail-closed against near-zero prediction variance, zero microbench contribution, or wide CI. A 1-line cleanup drops the stale pre-R12 host-absolute-paths queue entry. Cross-arch sm_120 work is EXPLICITLY DEFERRED and out of scope for this plan.

The work runs entirely under the CLAUDE.md `tools/local_labels/` NCU exception (Rule 1) on the local 3090 Ti device; no driver-root operations (Rule 2), no privileged CUPTI surfaces (Rule 3), no host-driver work (Rule 4), and no simulator revival (Rule 5).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Stale bundles regenerated from current producer code (Round A0). Scope per DEC-3 RESOLVED: the per-source R16/R17/R24/R28 ablation applies to the 5 ACTIVATION cells only; the 8 microbench cells get fresh bundles under new larger shapes in Round C (DEC-3 RESOLVED to "larger shapes for all 8") and are NOT represented in the Round A0 per-source ablation.
  - Positive Tests (expected to PASS):
    - Re-running `tools/hrd_profiler/profiler.py` + `workloads/simbench_bw/cache_predictor.py` on the existing 13-cell sweep traces emits per-cell predictor bundles whose `source_commit` field equals the current HEAD.
    - The regenerated `comparison_bundle.json` shows per-cell `predicted_hit_rate` values that vary across the 5 admitted activation cells (no longer all `1.0000`); per-cell variance is recorded in a side-by-side `regeneration_delta.json` artifact alongside the new bundle.
    - `regeneration_delta.json` enumerates PER-ACTIVATION-CELL deltas (5 cells: layernorm/rmsnorm/silu_correctness/silu_timing/softmax_safe) attributable to each of the four staleness sources (R16 cold-as-miss denominator, R17 all-cold-launch admission, R24 comparable_cells_per_arch widening, R28 NCU launch-skip offset). Attribution method (per Codex Round 1 OPTIONAL_IMPROVEMENT): produce four ablation re-runs per activation cell — one with each producer-code change reverted in turn — and record the delta `(value_with_change) - (value_with_change_reverted)` per cell per source. The ablation harness reuses existing `git checkout` of the relevant pre-Rxx HEAD into a worktree under `EnterWorktree`-style isolation; no production-tree edits required.
    - `regeneration_delta.json` carries an explicit `microbench_cells_deferred_to_round_c` block listing the 8 microbench slugs and noting they are excluded from the per-source ablation because Round C will replace their binaries (per DEC-3 RESOLVED).
  - Negative Tests (expected to FAIL):
    - A regenerated bundle whose `source_commit` does not match the predictor-bundle build commit MUST be rejected by `tools/local_labels/evidence_manifest.py`'s pairing check (`_check_comparison_semantic_binding`).
    - A regeneration attempt that admits any activation cell predicting exactly `1.0000` after sub-3 + sub-4 land MUST trigger the per-round shadow-diagnostic warning (`predictor_saturation_residual`).
    - A `regeneration_delta.json` missing the per-source ablation columns (R16 / R17 / R24 / R28) for the 5 activation cells MUST cause the Round A0 summary scanner to flag incomplete attribution.
    - A `regeneration_delta.json` that ATTEMPTS to compute per-source ablation for any of the 8 microbench cells MUST be flagged — those cells are deferred to Round C per DEC-3 RESOLVED and have no stable binary to ablate against.

- AC-2: Activation NCU labels re-collected with the R28 launch-skip fix; pre-R28 labels typed-superseded (Round A1).
  - Positive Tests:
    - `tools/local_labels/run_headline_cohort_sweep.py` (or the appropriate sweep entrypoint) is invoked on the 3090 Ti and produces 5 new label rows for `layernorm_fwd_bf16_sm80(timing)`, `rmsnorm_fwd_bf16_sm80(timing)`, `silu_fwd_bf16_sm80(correctness)`, `silu_fwd_bf16_sm80(timing)`, `softmax_safe_bf16_sm80(timing)`; each row carries the corrected `--launch-skip` value in its label-side metadata.
    - The pre-R28 label rows for those 5 cells are moved (or copied with a typed marker) into `data/local_labels/label_excluded.json` with a NEW typed reason `superseded_by_launch_skip_fix` (analogous to the existing `label_repeatability_failed` exclusion pattern in `tools/local_labels/repeatability_aggregator.py`).
    - The new sweep directory's round summary names the GPU model (RTX 3090 Ti), driver version, CUDA toolkit, NVBit build, and observed clocks per the `Verified-live-sm_86` evidence requirement in CLAUDE.md.
  - Negative Tests:
    - A re-join attempt where any of the 5 cells silently mixes pre-R28 and post-R28 labels MUST fail the `tools/local_labels/select_best_config_labels.py` rogue-stale filter (which already rejects mismatched-identity payloads).
    - A round summary that omits device + driver + CUDA toolkit + NVBit build + observed-clocks block MUST be flagged by the round-summary structural scanner.

- AC-3: Per-launch diagnostic fields emitted from `tools/hrd_profiler/observation_builder.py::_walk_launch` (Round B sub-slice 1, MUST land first within Round B).
  - Positive Tests:
    - `_walk_launch` (lines 132-260 in `observation_builder.py`) returns per-launch records that include aggregated `cold_count`, `finite_miss_count`, `unique_sector_count`, reuse-distance percentiles (p50/p90/p99), `working_set_sector_count`, and per-op load/store class counts. The existing per-access `sector_state`/`line_state`/`allocation_state`/`region_state` "cold"/"finite" tags are aggregated into per-launch sums; no new per-access state is invented.
    - The diagnostic fields propagate through `build_launches_from_trace_dir` (line 263) and `_BuildResult.launches` into the existing per-launch JSON shape consumed by `cache_predictor.py::_aggregate_launches_from_observations`.
    - A failing-test-first witness asserts: on a synthetic launch with all-cold accesses, `cold_count == access_count` and `finite_miss_count == 0`; on a synthetic launch with all-warm accesses, the inverse.
  - Negative Tests:
    - A predictor-side change (sub-slice 2 model refinement, sub-slice 3 calibration head) committed BEFORE sub-slice 1's diagnostic fields exist MUST be rejected by the round-contract structural scanner (commit ordering enforced by the round contract's slice list).
    - A diagnostic field whose value disagrees with a recomputation from the existing per-access state machine MUST cause the witness test to fail.

> **Tier-2 annotation (added 2026-05-16, Tier-2 RLCR loop Round 1, AC-T8 cleanup-with-round-open invariant)**: The follow-up plan at `plans/tier2_rd_calibration_cross_arch_plan.md` adds a SEPARATE Maeda 2017 second-layer HRD experiment under its AC-T1 AFTER this active plan's AC-4.1 closes; the annotation MUST NOT redefine, narrow, or widen the active AC-4.1 closure semantics. Active loop owns its own AC-4.1 closure shape.

- AC-4: Predictor model refinements (Round B sub-slice 2, MUST land after sub-slice 1, MUST land before sub-slice 3). Each refinement is its own AC sub-criterion with failing-test-first ordering.
  - AC-4.1: Set/slice-aware effective capacity replaces the current `capacity_bytes // 32` scalar in `_walk_launch` (line 244-248) with a slice-geometry-aware effective capacity probed from the device-properties manifest.
    - Positive: per-launch hit rate on a synthetic single-slice working-set kernel decreases vs. the fully-associative baseline by a documented margin (>5%) when the working set exceeds slice capacity.
    - Negative: a refinement that changes per-launch hit rate on a kernel whose working set fits within ANY single slice MUST fail the witness (no false negatives on small kernels).
  - AC-4.2: Sector-edge accounting at 32B granularity correctly classifies sub-32B accesses that span sector boundaries.
    - Positive: a synthetic kernel with 16B-aligned 32B loads spanning sector boundaries reports 2× the sector-access count of a 32B-aligned variant.
    - Negative: a sector-aligned access MUST NOT be double-counted.
  - AC-4.3: Write-policy classification distinguishes write-allocate vs. no-write-allocate per-op classes in the diagnostic stream.
    - Positive: a store-only synthetic kernel reports `store_count > 0`, `load_count == 0`; the predictor's per-launch hit rate accounts for write-allocate misses.
    - Negative: classifying a load as a store MUST cause the witness to fail.
  - AC-4.4: Coarse CTA/SM interleaving model approximates concurrent-CTA L2 contention.
    - Positive: a synthetic kernel with N concurrent CTAs all touching the same sectors reports a per-launch hit rate accounting for sector reuse across CTAs.
    - Negative: a kernel with disjoint per-CTA working sets MUST report unchanged per-launch hit rate vs. a single-CTA baseline.

- AC-5: Constrained calibration head (Round B sub-slice 3, MUST land LAST within Round B; MUST NOT land if AC-4 sub-slices have not demonstrably improved per-cell prediction variance).
  - Positive Tests:
    - A monotone (or logistic) calibration head fit on the per-cell `(predicted, label)` pairs from the post-AC-4 regenerated bundle reduces per-cell relative error vs. the un-calibrated baseline by a documented margin.
    - The training-vs-evaluation split is recorded in the round summary and matches the user-pinned split (DEC-4).
  - Negative Tests:
    - A calibration-head commit landing BEFORE AC-3 + AC-4 are committed MUST be rejected by the round-contract slice ordering.
    - A calibration head that produces non-monotone behavior (e.g., a higher predicted value mapping to a lower calibrated value) MUST fail a monotonicity witness test.

- AC-6: Microbench rehabilitation via LARGER SHAPES for ALL 8 microbench cells (per DEC-3 RESOLVED); `MINIMUM_BIN_SAMPLE_COUNT` and `CI_HALF_WIDTH_TARGET` (the second bin-admission predicate at `tools/hrd_profiler/profiler.py:463`) BOTH unchanged. Round C summary records the IDENTITIES of admitted cells (not just count N), so AC-9's microbench-coverage guard is auditable.
  - Positive Tests:
    - At least 5 of the 8 microbench cells (per DEC-1 RESOLVED) re-join the headline `comparison_bundle.json` with `predictor_result_kind != typed_unavailable` and `typed_reason != cache_model_underconfigured`. The Round C summary lists the identities of all admitted microbench cells under a `microbench_admitted_cells` block (slug + shape_point per cell).
    - For each of the 8 microbench slugs (`working_set_capacity_sweep`, `l2_thrash_pointer_chase`, `dram_stream_stride_mix`, `carveout_sensitivity_probe`, `l2_persisting_window_sweep`, `l2_prefetch_hint_sweep`, `warm_cold_pair_relaunch`, `register_spill_local_mem`), a new larger shape variant (e.g., `trace_l2_accuracy_xl`) is added under `workloads/simbench_bw/l1/<slug>/`; the new shape is sized so that every active reuse bin clears the 50-sample threshold after a single sweep pass.
    - All 8 microbench cells get NEW `binary_sha256` and `kernel_signature_sha256` (recorded in the Round C regeneration manifest); the prior microbench bundles in `data/local_labels/sweeps/sweep_20260511T030924Z_7189cda8e43f/` are NOT re-joined under the new join keys (the new Round C sweep dir produces fresh bundles).
    - Label-side `label_repeatability_failed` for the affected microbench cells clears below the existing stddev thresholds (L2 stddev <= 0.03, L1tex stddev <= 0.01, per `repeatability_aggregator.py:664-718`).
  - Negative Tests:
    - A code change that lowers `MINIMUM_BIN_SAMPLE_COUNT=50` (line 46 of `profiler.py`) OR lowers the `CI_HALF_WIDTH_TARGET` predicate (line 463) MUST be rejected by a static-guard test analogous to the existing no-NCU static guard.
    - A microbench cell whose newly-regenerated bin still fails the 50-sample threshold MUST continue to emit `cache_model_underconfigured` (typed-fail-closed discipline preserved).
    - A Round C summary missing the `microbench_admitted_cells` identity block (only reporting count N) MUST be flagged by the round-summary structural scanner.
    - A Round C implementation that uses repeat-and-scale (the alternative path) without adding new larger shape variants MUST be rejected — DEC-3 RESOLVED to "larger shapes for all 8".

- AC-7: Trace-sidecar polluter root-caused (Round Polluter); the Round 20 `regenerate_in_place` workaround removed. The 5 named suspect tests are a SEED list — the actual offender may be in a sibling test file; the fail-closed assertion approach in `handle_trace` itself catches the offender regardless of which test invokes it.
  - Positive Tests:
    - `workloads/simbench_bw/run_suite.py::handle_trace` carries a fail-closed assertion (or equivalent test fixture wiring) that `args_out_dir_root` resolves under a `tmp_path` when invoked from pytest; the assertion fires on at least one test in the suite. SEED suspect list (NOT exhaustive — investigation may identify a sibling test):
      1. `workloads/simbench_bw/tests/test_run_suite_handle_trace_user_defined_folders.py`
      2. `workloads/simbench_bw/tests/test_microbench_correctness_typed_unavailable.py`
      3. `workloads/simbench_bw/tests/test_run_suite_trace_mode.py`
      4. Any sibling test that constructs `row.slug` from a dict and omits `args_out_dir_root` monkeypatch
      5. Any sibling test that calls `handle_trace` with a real `DeviceProps` and default `args_out_dir_root`
    - After the offending test is fixed (proper `tmp_path` / monkeypatch), an aggregate pytest run of `tools/ workloads/` produces NO diff under `workloads/simbench_bw/results/dram_stream_stride_mix/` — the file `trace_trace_00.json` and its 3 sibling files (manifest, nvbit summary, `trace_trace_00/` subdir) are NOT re-emitted.
    - The Round 20 `regenerate_in_place` workaround is removed in a separate cleanup commit, and producer-side discipline is restored.
  - Negative Tests:
    - An attempt to declare AC-7 closed while ANY post-aggregate-suite-run diff appears under `workloads/simbench_bw/results/dram_stream_stride_mix/` MUST fail the round contract's idempotence assertion.
    - Removing the `regenerate_in_place` workaround BEFORE identifying and fixing the offending test MUST be blocked by the round contract's slice ordering.

- AC-8: portable_lcd traces RESTORED via Path A for ALL 14 missing slugs (per DEC-2.c RESOLVED); pre-existing failing test from Round 15 retired. Authoritative roster per `workloads/portable_lcd/roster.toml` is 15 slugs total; `gemm_hmma_bf16_sm80` is the only one with `traces/`, so 14 slugs are missing traces. Path B (live-gate) is NO LONGER an acceptable alternative for closure per the user's DEC-2.c choice.
  - Positive Tests:
    - For each of the 14 missing slugs (`conv2d_implicit_gemm_bf16_sm80`, `cross_entropy_fwd_fp32_sm80`, `embedding_gather_int32_sm80`, `flash_attn_bwd_bf16_sm80`, `flash_attn_fwd_bf16_sm80`, `gemm_hmma_fp16_sm80`, `gemm_imma_int8_sm80`, `gemm_sparse_2to4_bf16_sm80`, `layernorm_fwd_bf16_sm80`, `rmsnorm_bwd_bf16_sm80`, `rmsnorm_fwd_bf16_sm80`, `silu_fwd_bf16_sm80`, `softmax_safe_bf16_sm80`, `swiglu_fwd_bf16_sm80`), `workloads/portable_lcd/<slug>/traces/` is populated by a re-run of the NVBit tracer (`NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool` rebuild + per-slug sweep) on the 3090 Ti.
    - The NVBit tracer rebuild record (host gcc version, NVBit build commit, CUDA toolkit version) is recorded in the Round portable_lcd summary per the `Verified-live-sm_86` evidence requirement.
    - The aggregate pytest run no longer carries a pre-existing portable_lcd failure (Round 15's carried-forward failure is retired); `workloads/portable_lcd/verify_kernel_contracts.py` passes for all 15 slugs (1 pre-existing + 14 newly-restored).
  - Negative Tests:
    - Partial restoration (some slugs restored, others left missing) MUST be rejected — the round contract requires all 14 missing slugs to be restored or the round does not close.
    - A live-gate fallback (Path B) attempt MUST be rejected — DEC-2.c RESOLVED to Path A only.
    - A trace dir whose contents do NOT pass the `verify_kernel_contracts.py` per-slug fatbin slice + NVBit trace + suite SASS coverage + manifest schema validity checks MUST cause that slug's restoration to fail.

- AC-9: Headline-validity guard (cross-cutting; lands in the LAST round of the loop). Fail-closed if the headline bundle violates ANY of three conditions; the guard records ALL conditions that fire (not just the first), so multi-failure cases (e.g., the current on-disk bundle which fails BOTH `insufficient_microbench_coverage` AND `ci_width_exceeded`) are fully captured. Schema semantics pinned: the bundle is STILL emitted on guard fire (so its trajectory remains auditable) but its top-level `headline_validity_status` field equals `typed_unavailable` and the typed `headline_validity_failed_reasons` field (an ARRAY of enum values, NOT singular — per Codex Round 2 multi-failure correction) lists EVERY sub-condition that fired. Bundle emission itself does NOT fail — the validity verdict is a top-level diagnostic alongside the existing `relative_error_stats` block. The new fields MUST be added to `workloads/simbench_bw/schemas/comparison_bundle.schema.json` with `headline_validity_status` enum `["valid", "typed_unavailable"]` and `headline_validity_failed_reasons` array-of-enum (item enum: `["insufficient_prediction_variance", "insufficient_microbench_coverage", "ci_width_exceeded"]`); the array MUST be empty when status is `valid` and MUST be non-empty when status is `typed_unavailable`.
  - Positive Tests:
    - The guard lives in (or is invoked by) `workloads/simbench_bw/comparison_runner.py::run_comparison` (line 1414) and runs unconditionally on every aggregate suite invocation.
    - On a bundle where the per-cell prediction variance (stddev of `predicted_hit_rate` across all admitted cells, computed using the same per-bucket framing as `relative_error_stats_per_predictor_design_coverage`) is BELOW 0.10 (per DEC-2.a RESOLVED), the bundle IS emitted with `headline_validity_status=typed_unavailable` and `headline_validity_failed_reasons` includes `insufficient_prediction_variance`.
    - On a bundle where the count of microbench cells contributing to the headline (i.e., non-`typed_unavailable` cells whose join-key slug appears in the FULL 8-cell microbench subset of `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` — both friendly: `working_set_capacity_sweep`, `l2_thrash_pointer_chase`, `dram_stream_stride_mix`; AND hostile: `carveout_sensitivity_probe`, `l2_persisting_window_sweep`, `l2_prefetch_hint_sweep`, `warm_cold_pair_relaunch`, `register_spill_local_mem`) is BELOW 5 (per DEC-1 RESOLVED), the bundle IS emitted with `headline_validity_status=typed_unavailable` and `headline_validity_failed_reasons` includes `insufficient_microbench_coverage`.
    - On a bundle whose CI width exceeds 0.65 (per DEC-2.b RESOLVED), the bundle IS emitted with `headline_validity_status=typed_unavailable` and `headline_validity_failed_reasons` includes `ci_width_exceeded`.
    - The current on-disk bundle at `data/local_labels/sweeps/sweep_20260511T030924Z_7189cda8e43f/comparison/comparison_bundle.json` is correctly re-classified as `typed_unavailable` by a unit test that loads it through the new guard, AND `headline_validity_failed_reasons` contains BOTH `insufficient_microbench_coverage` AND `ci_width_exceeded` (multi-failure capture verified).
    - The schema at `workloads/simbench_bw/schemas/comparison_bundle.schema.json` is extended (additive widening, modeled on the proto3 `optional` Round 16 pattern) to require BOTH `headline_validity_status` (string enum) AND `headline_validity_failed_reasons` (array of enum) at the top level; an offline schema-validator test asserts both fields are present on every newly-emitted bundle.
  - Negative Tests:
    - A guard that has a suppression flag (e.g., `--allow-stale-headline`, `--skip-headline-validity`) MUST be rejected — the guard MUST NOT be bypassable.
    - A guard that fires on a bundle satisfying ALL three thresholds MUST fail a positive-witness test (i.e., status MUST be `valid` and `headline_validity_failed_reasons` MUST be the empty array; no false positives).
    - A bundle-emission code path that emits a bundle without setting `headline_validity_status` AND `headline_validity_failed_reasons` MUST be flagged by a static-scan test.
    - A schema change that removes either field from the required top-level fields MUST be rejected by the offline schema-validator test.
    - A guard implementation that records only the FIRST sub-condition that fires (instead of ALL fires) MUST fail the multi-failure unit test that loads the current on-disk bundle and asserts both reasons are present.

- AC-10: Stale pre-R12 host-absolute-paths queue entry removed (1-line cleanup; lands in the same commit that opens Round A0's contract — NOT a separate round).
  - Positive Tests:
    - The goal-tracker `Queued Side Issues` section in `.humanize/rlcr/<new-session-id>/goal-tracker.md` does NOT contain a "pre-R12 host-absolute paths" entry; the cleanup commit's diff includes ONLY the deletion of that line plus the new round-A0 contract entry.
    - A search across the new `.humanize/rlcr/<new-session-id>/` directory for "pre-R12" or "host-absolute" returns no matches except in archived round summaries from the prior 35-round loop.
  - Negative Tests:
    - The cleanup MUST NOT land as its own round — adding a Round 0 or Round X solely for this cleanup MUST be rejected by the round-contract structural scanner (loop opens with Round A0, not Round X).
    - A commit that deletes the queue entry without ALSO opening Round A0's contract MUST be rejected by the cleanup-with-round-open invariant.

- AC-11: Per-round shadow headline diagnostics emitted on every round closure, even before the validity guard lands. Scanner pattern: a new test under `tools/local_labels/test_round_summary_shadow_diagnostics_scanner.py` (modeled on the prior loop's existing structural scanners under `tools/local_labels/test_sweep_runbook.py` and `tools/local_labels/test_check_admission_floor.py`) asserts the `shadow_headline_diagnostics` block is present in every committed round summary under the new `.humanize/rlcr/2026-05-12_<HHMMSS>/` directory tree.
  - Positive Tests:
    - Each round summary (A0, A1, B, C, Polluter, portable_lcd) contains a `shadow_headline_diagnostics` block reporting: per-cell prediction variance, microbench-admitted cell IDENTITIES (not just count, per AC-6 alignment), excluded reason counts (by typed reason), and CI width.
    - The shadow diagnostics use the same calculation as the eventual AC-9 validity guard, so the guard's "lands last" placement does not block visibility of the round-by-round trajectory.
    - The scanner test file path follows the prior loop's discipline: `tools/local_labels/test_round_summary_shadow_diagnostics_scanner.py`.
  - Negative Tests:
    - A round summary missing the `shadow_headline_diagnostics` block MUST be flagged by the round-summary structural scanner.
    - Shadow diagnostics whose values disagree with a fresh re-computation from the same sweep MUST cause a unit test to fail.
    - A scanner-test file in a path other than `tools/local_labels/` MUST be flagged (the scanner's location is part of the convention).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation lands all six rounds in dependency order with full coverage: Round A0 emits `regeneration_delta.json` quantifying contributions from all 4 staleness sources (R16/R17/R24/R28) for the 5 activation cells (microbench cells are deferred to Round C since DEC-3 RESOLVED to replace their binaries); Round A1 re-collects all 5 activation labels under the R28-fixed launch-skip and types pre-R28 labels as `superseded_by_launch_skip_fix`; Round B lands all 4 model refinements (slice geometry + sector-edge + write-policy + CTA/SM interleave) under AC-4, plus the calibration head under AC-5 with train-on-all + shadow LOOCV residual table per DEC-4 RESOLVED; Round C rehabilitates all 8 microbench cells via new larger shape variants per DEC-3 RESOLVED (each cell gets a new `binary_sha256` + `kernel_signature_sha256`); Round Polluter root-causes the offender, fixes the test, AND removes the Round 20 `regenerate_in_place` workaround; Round portable_lcd restores all 14 missing `traces/` directories (out of 15 total slugs in `workloads/portable_lcd/roster.toml`) via Path A on the 3090 Ti per DEC-2.c RESOLVED (rebuild NVBit tracer + per-slug sweep); the headline-validity guard lands in the final round with all 3 fail-closed conditions enforced (variance ≥ 0.10, microbench-coverage ≥ 5 of 8, CI width ≤ 0.65); per-round shadow-diagnostic blocks are recorded for every round; the 1-line pre-R12 queue cleanup lands in the same commit that opens Round A0.

### Lower Bound (Minimum Acceptable Scope)
The implementation lands the dependency-critical path: Round A0 (offline regeneration + delta for the 5 activation cells; microbench cells deferred to Round C since DEC-3 RESOLVED to "larger shapes for all 8" replaces them with new binaries), Round A1 (5 activation re-collections under R28-fixed launch-skip with supersession), Round B sub-slice 1 (diagnostics) + at least ONE Round B sub-slice 2 refinement (slice geometry, the highest-bias source per Codex), Round C admits at least 5 of 8 microbench cells under new larger shapes (per DEC-3 RESOLVED; `MINIMUM_BIN_SAMPLE_COUNT` and `CI_HALF_WIDTH_TARGET` unchanged), Round Polluter root-causes the offending test AND removes the Round 20 `regenerate_in_place` workaround (no relaxation — workaround removal is required to close AC-7), Round portable_lcd restores traces for all 14 missing slugs via Path A (per DEC-2.c RESOLVED — Path B is no longer an acceptable alternative), the validity guard lands in the final round enforcing ALL 3 fail-closed conditions (variance ≥ 0.10 per DEC-2.a + microbench-coverage ≥ 5 of 8 per DEC-1 + CI width ≤ 0.65 per DEC-2.b — no relaxation; AC-9 is non-bypassable), per-round shadow diagnostics are recorded, and the 1-line cleanup lands in the Round A0 commit. Round B sub-slice 3 (calibration head — train-on-all + shadow LOOCV per DEC-4 RESOLVED) and Round B sub-slices 2.2/2.3/2.4 (sector-edge, write-policy, CTA/SM interleave) are queued as follow-up rounds rather than blocking lower-bound closure — those are the actual Lower-Bound relaxations vs the Upper Bound.

### Allowed Choices
- Can use:
  - The local NCU exception under `tools/local_labels/` for live label collection on the 3090 Ti (CLAUDE.md Rule 1, local-label exception).
  - Existing producer infrastructure: `tools/hrd_profiler/profiler.py`, `tools/hrd_profiler/observation_builder.py`, `workloads/simbench_bw/cache_predictor.py`, `workloads/simbench_bw/comparison_runner.py`, `tools/local_labels/comparison_runner` orchestrator, `tools/local_labels/run_headline_cohort_sweep.py`, `tools/local_labels/run_headline_cohort_predict_compare.py`.
  - Existing typed-fail-closed mechanisms: `cache_model_underconfigured` (`profiler.py:481`), `label_repeatability_failed` (`repeatability_aggregator.py:664-718`), `predictor_friendly`/`predictor_hostile` bucketing (`comparison_runner.py:175`), MPI 5-tuple pairing (`trace_convergence_ledger.py:177-188`), 7-tuple join key (`comparison_runner.py:99-107`).
  - The existing `relative_error_stats_per_predictor_design_coverage` per-bucket schema (`workloads/simbench_bw/schemas/comparison_bundle.schema.json:450-459`) — already breaks out friendly vs. hostile.
  - The RLCR loop pattern from `.humanize/rlcr/2026-05-10_23-00-08/` (round contracts + goal-tracker + finalize phase + structural scanners + Codex review per round).
  - Per-microbench-cell choice between (a) larger shapes via new `trace_l2_accuracy_xl`-class shape variants and (b) repeat-and-scale via increased microbench loop iteration count + proportional NCU repeat count, AS PINNED BY DEC-3.
  - Per-portable_lcd-slug choice between Path A (restore traces) and Path B (live-gate test), AS PINNED BY DEC-2.
- Cannot use:
  - Lowering `tools/hrd_profiler/profiler.py::MINIMUM_BIN_SAMPLE_COUNT=50` (line 46) OR `CI_HALF_WIDTH_TARGET` (line 463) — both bin-admission predicates stay; AC-6 enforces this.
  - Landing the Round B calibration head before Round B diagnostics + model refinements demonstrably improve per-cell variance — AC-5 negative test enforces ordering.
  - Landing the headline-validity guard in any round before the LAST one — AC-9 placement is non-negotiable.
  - Landing a guard suppression flag (e.g., `--allow-stale-headline`) — AC-9 negative test enforces no-bypass.
  - Any cross-arch sm_120 work in this loop — out of scope; queued for a separate session and a separate RLCR loop.
  - NCU on cloud or non-local server GPUs (CLAUDE.md Rule 1, absolute), nvidia-smi clock locking / persistence / MIG (Rule 2), CUPTI Events / Metrics / Range Profiling / PM Sampling (Rule 3), host-driver work (Rule 4), simulator revival (Rule 5).
  - Cohort-pivot from the 13-cell simbench_bw cohort to a portable_lcd cohort — the cohort definition stays at the 13 cells in `_HEADLINE_COHORT_13` (`run_headline_cohort_sweep.py:46`, `run_headline_cohort_predict_compare.py:42`).
  - Touching the 7-tuple join key fields in `workloads/simbench_bw/comparison_runner.py:99-107` — the join contract is invariant; AC-2 supersession types old labels rather than altering the join.

> **Note on Deterministic Designs**: The dependency ordering A0 → A1 → B → C → Polluter → portable_lcd is fixed by the user's deliverable. Within Round B, the sub-slice ordering (diagnostics → model refinements → calibration head) is fixed. The `MINIMUM_BIN_SAMPLE_COUNT=50` constraint is fixed. The cross-arch sm_120 deferral is fixed. These are non-negotiable; path boundaries narrow accordingly. Allowed choices are reserved for per-microbench rehab path, per-portable_lcd-slug policy, and the user-pinned thresholds (DEC-1 through DEC-4).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Open a new RLCR session at `.humanize/rlcr/2026-05-12_<HHMMSS>/` modeled on the prior `.humanize/rlcr/2026-05-10_23-00-08/` infrastructure. The new `goal-tracker.md` names the Ultimate Goal as: "Trustworthy `Verified-live-sm_86` L2-sector-hit-rate predictor-vs-NCU geomean baseline" (replacing the prior loop's "first artifact of the right shape"). Six round contracts queue in dependency order. Each round closes via failing-test-first discipline + Codex review per round + structural-scanner enforcement (plan-token markers, parity literals, sidecar variable names per the Round 35 finalize-summary).

For Round A0, regenerate predictor bundles in place against the existing 13-cell traces under `workloads/simbench_bw/l1/<slug>/traces/`; emit a new sweep dir under `data/local_labels/sweeps/<new_id>/` with a side-by-side `regeneration_delta.json` enumerating per-cell deltas attributable to R16/R17/R24/R28. The 1-line pre-R12 queue-entry cleanup lands in the same commit that opens A0's contract.

For Round A1, re-run the 3090 Ti NCU sweep on the 5 activation cells with the corrected `--launch-skip`; type old labels in `data/local_labels/label_excluded.json` with new typed reason `superseded_by_launch_skip_fix`; re-join via `workloads/simbench_bw/comparison_runner.py::run_comparison` (line 1414).

For Round B, sub-slice 1 extends `_walk_launch` (in `tools/hrd_profiler/observation_builder.py`, lines 132-260) to AGGREGATE the existing per-access cold/finite state into per-launch sums; sub-slice 2 lands one model refinement at a time (slice geometry first since it's the largest-bias source per Codex), each with a synthetic-launch witness; sub-slice 3 adds the calibration head ONLY after sub-slices 1+2 demonstrably improve variance (AC-5 negative test enforces this).

For Round C, choose per-microbench-cell between larger shapes (new shape variants under `workloads/simbench_bw/l1/<slug>/`) and repeat-and-scale (increased loop iterations + NCU repeats); the choice is pinned per-cell by DEC-3.

For Round Polluter, instrument `workloads/simbench_bw/run_suite.py::handle_trace` with a fail-closed assertion that `args_out_dir_root` resolves under `tmp_path` from pytest; iterate the 5 suspect tests until the assertion fires; remove the Round 20 `regenerate_in_place` workaround as a final cleanup commit.

For Round portable_lcd, choose between Path A (rebuild tracer + per-slug sweep) and Path B (live-gate test under `SIMBENCH_REQUIRE_LIVE_GPU=1`), per DEC-2.

The headline-validity guard lands in the last round merged (typically portable_lcd or a finalize phase). It MUST not be bypassable; the AC-9 negative test enforces this.

### Relevant References

- `workloads/simbench_bw/comparison_runner.py` — comparison logic + 7-tuple join key (lines 99-107) + `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` mapping (line 175) + `run_comparison` entrypoint (line 1414). NOTE: `tools/local_labels/run_headline_cohort_predict_compare.py` only orchestrates this; the comparison logic itself lives at the simbench_bw path.
- `workloads/simbench_bw/cache_predictor.py` — `predict_cache_hit_rate` (lines 478-751); already returns `typed_unavailable` (no scalar) when upstream HRD reports `bundle_underconfigured` (lines 685-717); already does bootstrap CI over per-launch hit rates with 2000 resamples (lines 728-751).
- `tools/hrd_profiler/observation_builder.py::_walk_launch` (lines 132-260) — fully-associative LRU walker with per-access cold/finite state machine; Round B sub-slice 1 aggregates these existing per-access tags into per-launch sums; Round B sub-slice 2 modifies the LRU model itself.
- `tools/hrd_profiler/profiler.py::MINIMUM_BIN_SAMPLE_COUNT=50` (line 46) and the second predicate `CI_HALF_WIDTH_TARGET` (line 463) — both untouched per AC-6; the typed `cache_model_underconfigured` reason emits at line 481.
- `data/local_labels/label_admitted.json` (8.7KB, 13-cell admitted set) and `data/local_labels/label_excluded.json` (1.1KB, typed-exclusion set with reasons `label_repeatability_failed`, `label_repeatability_missing`, `label_schema_admission_failed`); Round A1 adds typed reason `superseded_by_launch_skip_fix`.
- `tools/local_labels/repeatability_aggregator.py` (lines 664-718) — existing pattern for stddev-threshold-based exclusion (L2 stddev > 0.03, L1tex stddev > 0.01 → demoted to `label_excluded.json` with `label_repeatability_failed`); Round A1's supersession follows the same pattern.
- `tools/local_labels/select_best_config_labels.py` (lines 113-264) — existing rogue-stale filter that rejects per-config aggregator output if `binary_sha256` does not match the admitted-verdict-table; Round A1 negative-test reuses this filter.
- `tools/adaptive_sampler/trace_convergence_ledger.py::manifest_pairing_index_dict` (lines 177-188) — 5-field MPI tuple (slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256). The 7-tuple join key in `comparison_runner.py:99-107` extends MPI with `arch` + `ncu_metric_version`. AC-2 supersession types old labels but does NOT alter the join contract.
- `tools/local_labels/evidence_manifest.py::_check_comparison_semantic_binding` (line 652) — existing pairing-index check that AC-1's regenerated bundle MUST pass.
- `workloads/simbench_bw/run_suite.py::handle_trace` — Round Polluter's instrumentation site (sidecar variable name "sidecar" is a structural-scanner-watched parity-test literal per Round 35 finalize-summary; do not rename).
- `workloads/simbench_bw/schemas/comparison_bundle.schema.json` (lines 80-191 per-cell record shape; lines 450-459 `relative_error_stats_per_predictor_design_coverage` per-bucket stats; both `predictor_friendly` and `predictor_hostile` keys required) — already supports the per-bucket break-out the AC-9 validity guard reads.
- `tools/local_labels/check_admission_floor.py` (Round 33 work) — existing admission-floor check that the loop's structural scanners already enforce; new typed reasons added in this plan reuse the same diagnostic pattern.
- `.humanize/rlcr/2026-05-10_23-00-08/` (prior 35-round loop) — battle-tested template for round contracts, goal-tracker, finalize-summary, structural-scanner literal-watching.

## Dependencies and Sequence

### Milestones

1. M1 — Round A0 (offline regeneration + 1-line cleanup): depends on no prior milestones; can begin immediately.
   - Phase A0.a: Open new `.humanize/rlcr/2026-05-12_<HHMMSS>/` session; write goal-tracker.md (Ultimate Goal text + queued side issues, with the pre-R12 entry already removed); commit Round A0 contract.
   - Phase A0.b: Re-run `tools/hrd_profiler/profiler.py` + `workloads/simbench_bw/cache_predictor.py` against existing 13-cell traces; emit new predictor bundles under a new sweep dir.
   - Phase A0.c: Re-run `workloads/simbench_bw/comparison_runner.py::run_comparison` against the existing label set; emit new `comparison_bundle.json` + side-by-side `regeneration_delta.json`.
   - Phase A0.d: Round A0 summary records per-cell deltas attributable to R16/R17/R24/R28; queues Round A1 contract.

2. M2 — Round A1 (live re-collection of 5 activation labels): depends on M1.
   - Phase A1.a: Add typed reason `superseded_by_launch_skip_fix` to the typed-reason enum + schema in `data/local_labels/label_excluded.json` writer; failing-test-first witness asserts the new reason is admitted by the schema.
   - Phase A1.b: Move the 5 pre-R28 activation labels into `label_excluded.json` with the new typed reason.
   - Phase A1.c: Run the sweep entrypoint on the 3090 Ti with corrected `--launch-skip`; collect 5 new label rows.
   - Phase A1.d: Re-join via `comparison_runner.py::run_comparison`; round summary names device + driver + CUDA toolkit + NVBit build + observed clocks (`Verified-live-sm_86`).

3. M3 — Round B (predictor-saturation fixes; 3 sub-slices in strict order): depends on M1; benefits from but does not strictly require M2.
   - Phase B.1 (diagnostics, AC-3): aggregate existing per-access cold/finite/state-machine data in `_walk_launch` into per-launch sums; failing-test-first witness with synthetic launches.
   - Phase B.2 (model refinements, AC-4): land sub-AC-4.1 (slice geometry) first; then sub-AC-4.2 (sector-edge), sub-AC-4.3 (write-policy), sub-AC-4.4 (CTA/SM interleave) — each as its own slice with its own witness.
   - Phase B.3 (calibration head, AC-5): MUST NOT land if Phase B.1 + B.2 have not demonstrably improved per-cell variance; fits monotone/logistic head on per-cell `(predicted, label)` pairs from the post-B.2 regenerated bundle; training/eval split per DEC-4.

4. M4 — Round C (microbench rehabilitation): depends on M1; benefits from but does not strictly require M3.
   - Phase C.a: Per-microbench-cell decision (larger shapes vs repeat-and-scale, per DEC-3); for "larger shapes", add new shape variants under `workloads/simbench_bw/l1/<slug>/`; for "repeat-and-scale", increase microbench loop iteration count + NCU repeat count.
   - Phase C.b: Re-run sweep + predictor stack for the affected microbench cells; new bundles must produce `predictor_result_kind != typed_unavailable` and `typed_reason != cache_model_underconfigured`.
   - Phase C.c: Recollect labels with proportionally more NCU repeats; ensure label-side `label_repeatability_failed` clears below threshold.
   - Phase C.d: Re-run `comparison_runner.py::run_comparison`; verify ≥N microbench cells (N from DEC-1) re-join the headline.

5. M5 — Round Polluter (trace-sidecar polluter root-cause): independent of M2/M3/M4; can run in parallel with them as a separate worktree if operator capacity allows.
   - Phase P.a: Instrument `workloads/simbench_bw/run_suite.py::handle_trace` with fail-closed assertion on `args_out_dir_root`; failing-test-first witness exercises one of the 5 suspect tests.
   - Phase P.b: Iterate suspect tests until assertion fires; identify offender.
   - Phase P.c: Fix offender (proper `tmp_path` / monkeypatch); re-run aggregate suite; verify no diff under `dram_stream_stride_mix/`.
   - Phase P.d: Remove Round 20 `regenerate_in_place` workaround as final cleanup commit; verify producer-side discipline restored.

6. M6 — Round portable_lcd (restore traces or live-gate): independent of M2/M3/M4/M5; can run in parallel.
   - Phase L.a: Per-slug or blanket decision on Path A (restore) vs Path B (live-gate), per DEC-2.
   - Phase L.b (Path A): rebuild NVBit tracer (`NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool`); per-slug sweep on 3090 Ti; commit `traces/` directories.
   - Phase L.b (Path B): gate `verify_kernel_contracts` test under `SIMBENCH_REQUIRE_LIVE_GPU=1`; emit typed-skip reason `nvbit_trace_missing_skipped_offline` when traces absent.
   - Phase L.c: Verify aggregate suite no longer carries Round 15's pre-existing failure.

7. M7 — Headline-validity guard + finalize (AC-9 + AC-11 enforcement): depends on M1, M2, M3, M4, M5, M6.
   - Phase F.a: Add BOTH required top-level fields to bundle schema + emitter in `workloads/simbench_bw/comparison_runner.py::run_comparison`: (1) `headline_validity_status` (string enum `["valid", "typed_unavailable"]`); (2) `headline_validity_failed_reasons` (ARRAY of enum, item-enum `["insufficient_prediction_variance", "insufficient_microbench_coverage", "ci_width_exceeded"]`; empty when status is `valid`, non-empty when status is `typed_unavailable`).
   - Phase F.b: Implement 3 fail-closed conditions (variance, microbench-coverage, CI width); thresholds per DEC-2.a, DEC-2.b, DEC-1. Guard records EVERY firing condition in `headline_validity_failed_reasons` (not just the first).
   - Phase F.c: Add static-scan test asserting NO bundle-emission code path emits a bundle without setting BOTH fields.
   - Phase F.d: Add unit test loading the current on-disk bundle (geomean 0.3289, all-1.0) and asserting the guard correctly classifies it as `typed_unavailable` with `headline_validity_failed_reasons` containing BOTH `insufficient_microbench_coverage` AND `ci_width_exceeded` (multi-failure capture).
   - Phase F.e: Finalize phase (per the prior loop's `.humanize/rlcr/2026-05-10_23-00-08/finalize-summary.md` model): code-simplifier review of any opportunistic dead-code removal; round summary records `Cross-arch-verified=N/A` (sm_120 deferred).

The dependency graph: M1 → {M2, M3, M4} → M7; {M5, M6} run in parallel with M2/M3/M4; M7 depends on all of M1-M6 closing; M3 internally enforces B.1 → B.2 → B.3 ordering; per-round shadow-diagnostic AC-11 emission happens in every round summary from M1 onward (its emitter scaffolding lands in M1 even though the guard's fail-closed teeth land in M7).

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|-----------|
| task1 | Open new RLCR session at `.humanize/rlcr/2026-05-12_<HHMMSS>/`; write goal-tracker.md (Ultimate Goal "Trustworthy `Verified-live-sm_86` L2-sector-hit-rate predictor-vs-NCU geomean baseline"); commit Round A0 contract; same commit removes pre-R12 host-absolute-paths queued-issue entry. | AC-10, AC-1 | coding | - |
| task2 | Add per-round `shadow_headline_diagnostics` emission scaffolding to round-summary writer (no fail-closed teeth yet — those land in task20). | AC-11 | coding | task1 |
| task3 | Re-run `tools/hrd_profiler/profiler.py` + `workloads/simbench_bw/cache_predictor.py` against existing 13-cell traces; emit new predictor bundles under new sweep dir. | AC-1 | coding | task1 |
| task4 | Re-run `workloads/simbench_bw/comparison_runner.py::run_comparison` against existing label set; emit new `comparison_bundle.json` + side-by-side `regeneration_delta.json` enumerating per-cell deltas from R16/R17/R24/R28. | AC-1 | coding | task3 |
| task5 | Round A0 summary; commit Round A1 contract. | AC-1 | coding | task4 |
| task6 | Add typed reason `superseded_by_launch_skip_fix` to `data/local_labels/label_excluded.json` writer + schema; failing-test-first witness asserts new reason admitted. | AC-2 | coding | task5 |
| task7 | Move 5 pre-R28 activation labels into `label_excluded.json` with `superseded_by_launch_skip_fix`. | AC-2 | coding | task6 |
| task8 | Live sweep on 3090 Ti with corrected `--launch-skip` for 5 activation cells; commit 5 new label rows. Round summary names device + driver + CUDA toolkit + NVBit build + observed clocks (`Verified-live-sm_86`). | AC-2 | coding | task7 |
| task9 | Re-join via `comparison_runner.py::run_comparison`; commit re-joined bundle; commit Round B contract. | AC-2 | coding | task8 |
| task10 | Round B sub-slice 1 (diagnostics): aggregate existing per-access cold/finite state in `_walk_launch` into per-launch `cold_count`, `finite_miss_count`, `unique_sector_count`, reuse-distance percentiles, `working_set_sector_count`, per-op load/store class. Failing-test-first witness with synthetic all-cold and all-warm launches. | AC-3 | coding | task9 |
| task11 | Round B sub-slice 2.1 (slice geometry): replace `capacity_bytes // 32` with slice-geometry-aware effective capacity probed from device-properties manifest. Witness: per-launch hit rate decreases on synthetic single-slice working-set kernel. | AC-4.1 | coding | task10 |
| task12 | Round B sub-slice 2.2 (sector-edge accounting at 32B granularity). Witness: 16B-aligned 32B loads spanning sector boundaries report 2× sector-access count. | AC-4.2 | coding | task11 |
| task13 | Round B sub-slice 2.3 (write-policy classification). Witness: store-only synthetic kernel reports `store_count > 0`, `load_count == 0`. | AC-4.3 | coding | task12 |
| task14 | Round B sub-slice 2.4 (CTA/SM interleaving). Witness: N concurrent CTAs touching same sectors report cross-CTA reuse. | AC-4.4 | coding | task13 |
| task15 | Round B sub-slice 3 (calibration head): per DEC-4 RESOLVED, monotone/logistic head fit on ALL 5 activation cells' `(predicted, label)` pairs from post-task14 regenerated bundle (train-on-all, no held-out split). ALSO emit shadow LOOCV residual table (5 leave-one-out folds) in Round B summary as overfit signal (NOT used as a gate). AC-5 negative test enforces this slice MUST NOT land before tasks 10-14. | AC-5 | coding | task14 |
| task16 | Round C: per DEC-3 RESOLVED, add NEW larger shape variants (e.g., `trace_l2_accuracy_xl`) under `workloads/simbench_bw/l1/<slug>/` for ALL 8 microbench cells (`working_set_capacity_sweep`, `l2_thrash_pointer_chase`, `dram_stream_stride_mix`, `carveout_sensitivity_probe`, `l2_persisting_window_sweep`, `l2_prefetch_hint_sweep`, `warm_cold_pair_relaunch`, `register_spill_local_mem`); re-run sweep + predictor stack; recollect labels with proportional NCU repeats; re-run `comparison_runner.py::run_comparison`. Verify ≥5 of 8 microbench cells admitted (per DEC-1 RESOLVED). All 8 microbench cells get new `binary_sha256` + `kernel_signature_sha256`; prior microbench bundles in `data/local_labels/sweeps/sweep_20260511T030924Z_7189cda8e43f/` no longer match the new join keys. | AC-6 | coding | task5 |
| task17 | Round C static-guard test: a code change that lowers `MINIMUM_BIN_SAMPLE_COUNT=50` (line 46) OR `CI_HALF_WIDTH_TARGET` (line 463) MUST be rejected by the new static-guard test. | AC-6 | coding | task16 |
| task18 | Round Polluter: instrument `workloads/simbench_bw/run_suite.py::handle_trace` with fail-closed assertion on `args_out_dir_root` under pytest; iterate 5 suspect tests until assertion fires; fix offender (proper `tmp_path` / monkeypatch); verify no diff under `dram_stream_stride_mix/` after aggregate run; remove Round 20 `regenerate_in_place` workaround as final cleanup. | AC-7 | coding | task1 |
| task19 | Round portable_lcd: per DEC-2.c RESOLVED, Path A (rebuild NVBit tracer via `NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool` + per-slug sweep on 3090 Ti) for all 14 missing slugs (out of 15 per `workloads/portable_lcd/roster.toml`). Restore `workloads/portable_lcd/<slug>/traces/` for each missing slug. Round summary records NVBit build commit + CUDA toolkit version + observed clocks (`Verified-live-sm_86`). Verify `verify_kernel_contracts.py` passes for all 15 slugs and Round 15 carried-forward failure is retired. | AC-8 | coding | task1 |
| task20 | Headline-validity guard: extend `workloads/simbench_bw/schemas/comparison_bundle.schema.json` with two required top-level fields — `headline_validity_status` (enum `["valid", "typed_unavailable"]`) and `headline_validity_failed_reasons` (array of enum, item-enum `["insufficient_prediction_variance", "insufficient_microbench_coverage", "ci_width_exceeded"]`, empty when status is `valid`, non-empty when status is `typed_unavailable`). Implement guard in `workloads/simbench_bw/comparison_runner.py::run_comparison` enforcing all 3 fail-closed conditions per RESOLVED thresholds: variance stddev ≥ 0.10 (DEC-2.a), microbench-coverage ≥ 5 of 8 microbench cells (DEC-1), CI width ≤ 0.65 (DEC-2.b); the guard records EVERY sub-condition that fires, not just the first. Bundle is STILL emitted on guard fire (top-level status + array diagnostic, no bundle-emission failure). Add static-scan test asserting NO bundle-emission code path emits a bundle without setting both fields. Add unit test loading current on-disk `comparison_bundle.json` (geomean 0.3289) and asserting it is correctly classified `typed_unavailable` with `headline_validity_failed_reasons` containing BOTH `insufficient_microbench_coverage` AND `ci_width_exceeded` (multi-failure capture verified; the current bundle has 0 microbench cells contributing — fewer than 5 — and CI width 0.659 — over 0.65 — so both fire). | AC-9 | coding | task9, task15, task16, task18, task19 |
| task21 | Codex review per round (one `/humanize:ask-codex` invocation per Round closure: A0, A1, B, C, Polluter, portable_lcd, finalize). | AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8, AC-9 | analyze | task5, task9, task15, task16, task18, task19, task20 |
| task22 | Finalize phase: code-simplifier review of opportunistic dead-code removal (modeled on `.humanize/rlcr/2026-05-10_23-00-08/finalize-summary.md`); finalize-summary.md records `Cross-arch-verified=N/A` (sm_120 deferred). | AC-9 | coding | task20, task21 |

## Claude-Codex Deliberation

### Agreements
- Cross-arch sm_120 is explicitly deferred; this loop targets sm_86 only on the 3090 Ti.
- Round B's sub-slice ordering (diagnostics → model refinements → calibration head) is non-negotiable; AC-5 negative test enforces the calibration head MUST NOT land before AC-3 + AC-4.
- `MINIMUM_BIN_SAMPLE_COUNT=50` MUST stay; AC-6 static-guard test enforces this.
- The headline-validity guard MUST land in the LAST round and MUST be non-bypassable; AC-9 negative tests enforce both.
- The 1-line pre-R12 host-absolute-paths queue cleanup lands in the same commit that opens Round A0; it is NOT a separate round (AC-10).
- The 5 specific tests Codex named are the right starting investigation surface for the polluter root-cause.

### Resolved Disagreements
- **Comparison module path**: Draft cited `tools/local_labels/comparison_runner.py`; Codex correctly flagged that the comparison logic lives at `workloads/simbench_bw/comparison_runner.py` (the local-labels module is just an orchestrator at `tools/local_labels/run_headline_cohort_predict_compare.py:32`). All AC-1, AC-2, AC-9 paths corrected to `workloads/simbench_bw/comparison_runner.py`.
- **MPI vs join key arity**: Codex initially mentioned "all 7 join fields" against `manifest_pairing_index`; verification showed MPI is a 5-tuple (`tools/adaptive_sampler/trace_convergence_ledger.py:177-188`) and the comparison_runner.py join key is a 7-tuple extending MPI with `arch` + `ncu_metric_version`. Plan now distinguishes the two; AC-1 references the 5-tuple MPI check via `evidence_manifest.py::_check_comparison_semantic_binding` and the 7-tuple join is invariant.
- **Per-bucket stats already exist**: Codex's CANDIDATE_CRITERIA included "AC-B: report per-bucket stats for friendly/hostile". Verification showed `relative_error_stats_per_predictor_design_coverage` is ALREADY required by the schema (`comparison_bundle.schema.json:450-459`) with both `predictor_friendly` and `predictor_hostile` keys. Plan reframes this as: AC-9 validity guard READS the existing per-bucket stats and fails closed on insufficient coverage in the microbench-friendly bucket.
- **CI half-width gate**: Codex flagged `MINIMUM_BIN_SAMPLE_COUNT=50` as the only bin-admission threshold; verification showed a SECOND predicate at `profiler.py:463` (`CI_HALF_WIDTH_TARGET`). Plan extends AC-6's negative test to cover BOTH predicates, not just sample count.
- **Existing supersession gap**: Codex's MISSING_REQUIREMENTS noted "Define label supersession rules"; verification confirmed no existing `superseded_*` rule exists in `tools/local_labels/`. Plan adds typed reason `superseded_by_launch_skip_fix` (AC-2), modeled on the existing `label_repeatability_failed` exclusion pattern in `repeatability_aggregator.py:664-718`.
- **Per-round shadow diagnostics**: Codex's MISSING_REQUIREMENTS asked for "per-round shadow headline diagnostics even if final fail-closed guard lands last"; plan added AC-11 explicitly for this, with the emission scaffolding landing in task2 (early in M1).
- **`_HEADLINE_COHORT_13` "duplication"**: Codex flagged this as duplicated risk; verification showed the two definitions have DIFFERENT TUPLE SHAPES (3-tuple in `run_headline_cohort_sweep.py:46` for binaries, 2-tuple in `run_headline_cohort_predict_compare.py:42` for slug+shape_point). The duplication is intentional, not a bug; plan does NOT add a deduplication task. Recorded for traceability.

### Convergence Status
- Final Status: **CONVERGED** (Codex Round 4 verdict; 4 review rounds completed).
- Round 1: 4 REQUIRED_CHANGES + 5 OPTIONAL_IMPROVEMENTS surfaced; all 4 REQUIRED + 4 of 5 OPTIONAL applied.
- Round 2: 2 NEW_REQUIRED_CHANGES surfaced (residual portable_lcd cardinality "13 missing" in milestone/DEC sections; AC-9 multi-failure semantics — singular reason vs array); both applied.
- Round 3: 1 NEW_REQUIRED_CHANGE surfaced (residual singular `headline_validity_failed_reason` in Phase F.a milestone description); applied.
- Round 4: zero NEW_REQUIRED_CHANGES; CONVERGED.
- Pending User Decisions: ALL 6 (DEC-1, DEC-2.a, DEC-2.b, DEC-2.c, DEC-3, DEC-4) RESOLVED via AskUserQuestion in Phase 6 of gen-plan execution; status fields updated from PENDING to RESOLVED with the user's chosen values.

## Pending User Decisions

- DEC-1: Microbench-coverage minimum for AC-9 + AC-6 closure.
  - Claude Position (initial default): At LEAST 3 microbench cells from the friendly subset (`working_set_capacity_sweep`, `l2_thrash_pointer_chase`, `dram_stream_stride_mix`).
  - Codex Position: N/A — open question.
  - Tradeoff Summary: Tighter minimum (e.g., 5 of 8) maximizes coverage credibility but raises the chance Round C cannot close at all. Looser minimum (e.g., 1) makes closure trivial but defeats the guard's intent.
  - Decision Status: RESOLVED — User chose "Tighten variance + coverage" — microbench-coverage minimum is **≥5 of the 8 total microbench cells** (NOT just the 3-friendly subset; "covers half the 8 microbench cells" is the user's explicit framing). AC-6 lower bound (≥5 of 8 admitted) and AC-9 microbench-coverage condition are aligned at this threshold. Implication: Round C MUST admit at least 5 microbench cells across the friendly+hostile mix, including possibly some predictor_hostile cells; the predictor_hostile cells contribute to the count even if their relative error is high (the count is over `predictor_result_kind != typed_unavailable`, not over per-cell error).

- DEC-2.a: Per-cell prediction-variance floor for AC-9.
  - Claude Position (initial default): stddev ≥ 0.05.
  - Codex Position: N/A — open question.
  - Tradeoff Summary: 0.05 is conservative; 0.10 forces wider discrimination but may fire on legitimately-clustered regimes; 0.02 would allow near-saturation regimes to pass.
  - Decision Status: RESOLVED — User chose "Tighten variance + coverage" — variance threshold is **stddev ≥ 0.10** for `predicted_hit_rate` across all admitted cells. Implication: AC-9 fires `insufficient_prediction_variance` when admitted cells cluster within a 10-percentage-point band. This is a stricter discrimination requirement than the initial default and may surface predictor-saturation residual sooner.

- DEC-2.b: Maximum CI width for AC-9.
  - Claude Position (initial default): CI width ≤ 0.65.
  - Codex Position: N/A — open question.
  - Tradeoff Summary: Per-bucket CI-width gating is more nuanced but adds complexity; tightening to 0.40 risks blocking closure; keeping 0.65 just rules out the current bundle.
  - Decision Status: RESOLVED — User did NOT pick the "Tighten CI width to ≤0.40" option, so the default ≤ 0.65 stands. Implication: the current on-disk bundle (CI [0.0945, 0.7533], width 0.659) is correctly classified `typed_unavailable` with `ci_width_exceeded`, but a regenerated bundle with CI width slightly under 0.65 would pass this sub-condition.

- DEC-2.c: portable_lcd policy.
  - Claude Position (initial default): Path B (live-gate) for all 14 missing slugs.
  - Codex Position: N/A — open question.
  - Tradeoff Summary: Path B is fast and host-only; Path A produces tangible artifacts but requires NVBit tracer rebuild + per-slug sweep effort; mixed forbidden by Codex.
  - Decision Status: RESOLVED — User chose **Path A: restore traces for all 14 missing slugs** (out of 15 total per `workloads/portable_lcd/roster.toml`). Implication: Round portable_lcd scope expands from "gate one test" to "rebuild NVBit tracer (`NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool`) + per-slug sweep on 3090 Ti for all 14 missing slugs". This is significantly more operator wall-clock time but produces tangible `traces/` artifacts that future work (e.g., portable_lcd cohort pivot) can reuse. AC-8 positive test now requires Path A specifically; Path B is no longer an acceptable alternative for closure.

- DEC-3: Per-microbench rehabilitation path (Round C).
  - Claude Position (initial default): Repeat-and-scale by default; per-cell fallback to larger-shapes from task10 diagnostics.
  - Codex Position: N/A — open question.
  - Tradeoff Summary: Repeat-and-scale preserves binary identity; larger shapes give guaranteed threshold-clearance but change `binary_sha256` + `kernel_signature_sha256`; drop/replace is a cohort change.
  - Decision Status: RESOLVED — User chose **"Larger shapes default for all 8 microbench cells"**. Implication: Round C adds new shape variants (e.g., `trace_l2_accuracy_xl`-class) under `workloads/simbench_bw/l1/<microbench_slug>/` for ALL 8 microbench slugs (`working_set_capacity_sweep`, `l2_thrash_pointer_chase`, `dram_stream_stride_mix`, `carveout_sensitivity_probe`, `l2_persisting_window_sweep`, `l2_prefetch_hint_sweep`, `warm_cold_pair_relaunch`, `register_spill_local_mem`). All 8 microbench cells will have NEW `binary_sha256` and `kernel_signature_sha256`, invalidating their prior bundles for join-key purposes. The 7-tuple join contract is preserved (no schema change) but the "old" microbench bundles in `data/local_labels/sweeps/sweep_20260511T030924Z_7189cda8e43f/` no longer match the new microbench join keys; the new sweep dir created in Round C produces fresh bundles. AC-1's R16/R17/R24/R28 ablation only applies to the 5 ACTIVATION cells (which retain original binaries); microbench cells get fresh bundles in Round C, not regeneration deltas in Round A0. Round A0's `regeneration_delta.json` MUST explicitly note the microbench cells are deferred to Round C and not represented in the per-source ablation.

- DEC-4: Calibration-head training/eval split (AC-5 task15).
  - Claude Position (initial default): Train-on-all + shadow LOOCV residual table.
  - Codex Position: N/A — open question.
  - Tradeoff Summary: Train-on-all maximizes fit but offers no overfit detection; 4-vs-1 LOOCV with gating is principled but unstable on small corpus; deferring entirely is most conservative.
  - Decision Status: RESOLVED — User chose **"Train-on-all + shadow LOOCV residual table"**. Implication: AC-5 task15 fits a monotone (or logistic) calibration head on the per-cell `(predicted, label)` pairs from the post-task14 regenerated bundle WITHOUT a held-out eval split; ALSO produces a shadow LOOCV residual table (leave-one-out per cell, 5 folds) printed in the Round B summary as overfit signal but NOT used as a gating threshold. The Round B summary documents the train-on-all decision explicitly. A follow-up loop is queued to add held-out eval if the cohort grows beyond 13 cells (e.g., when sm_120 cross-arch labels join).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", "Round A0/A1/B/C/Polluter/portable_lcd", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code: e.g., `headline_validity_failed` (typed reason name) rather than "AC-9 reason"; `superseded_by_launch_skip_fix` (typed reason name) rather than "Round A1 reason"; `cold_count` / `finite_miss_count` (diagnostic field names) rather than "AC-3 fields".
- The structural-scanner-watched parity literal `sidecar` in `workloads/simbench_bw/run_suite.py` MUST stay as-is per the Round 35 finalize-summary; do not rename to e.g. `slowdown_sidecar` even when extending nearby logic.
- All new typed reasons added to `data/local_labels/label_excluded.json` MUST extend the existing enum/schema (additive only; no breaking schema change), modeled on the proto3 `optional` widening pattern from Round 16.
- All new diagnostic fields added to per-launch records MUST flow through the existing `_BuildResult.launches` shape; no parallel data path.

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# Trustworthy L2 Sector Hit Rate Baseline on sm_86

## Original Idea

Brainstorm a repo-grounded idea for closing the L2-accuracy gap on the local 3090 Ti by addressing all queued side issues + two follow-on rounds (regeneration, predictor-saturation fix), with cross-arch sm_120 EXPLICITLY DEFERRED.

CURRENT STATE
The 35-round RLCR loop closed its narrow Ultimate Goal: a `Verified-live-sm_86` headline geomean predictor-vs-NCU relative-error on L2 sector hit rate. On-disk: `data/local_labels/sweeps/sweep_20260511T030924Z_7189cda8e43f/comparison/comparison_bundle.json` carries `geomean=0.3289`, CI `[0.0945, 0.7533]`, 5 admitted activation cells. The number is "first artifact of the right shape", not credible accuracy.

CONFIRMED ROOT CAUSES (deep research + Codex review)

1. Headline bundle is STALE under 4 producer/runner code shifts (R16 cold-as-miss denominator, R17 all-cold-launch admission, R24 comparable_cells_per_arch widening, R28 NCU --launch-skip warmup offset bug — labels collected with wrong launch-skip).

2. Predictor SATURATES at 1.0 on the on-disk bundle. CODEX CORRECTION: re-running CURRENT cache_predictor.py on existing traces would give ~0.938-0.979, NOT 1.0. Remaining high bias is in `tools/hrd_profiler/observation_builder.py::_walk_launch` (fully-associative sector LRU; deterministic trace-order walk; ignores set/slice geometry, sector-edge effects, write policy class, CTA/SM interleaving).

3. ALL 8 microbench predictor bundles return `cache_model_underconfigured`. Root cause: `tools/hrd_profiler/profiler.py::MINIMUM_BIN_SAMPLE_COUNT=50`. The trace_00 traces are too small (per-cell admitted_bins: 1-3, underconfigured: 1-12). 5 of 8 ALSO fail label-side `label_repeatability_failed`.

4. Trace-sidecar polluter (`workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/trace_trace_00.json` re-emitted on every aggregate run; mitigated by `regenerate_in_place` workaround but root cause unidentified). Hypothesis: a test invokes `handle_trace` with row.slug from a dict + DEFAULT `args_out_dir_root` (no monkeypatch). Likely candidates: `test_run_suite_handle_trace_user_defined_folders.py`, `test_microbench_correctness_typed_unavailable.py`, `test_run_suite_trace_mode.py`.

5. portable_lcd nvbit_trace_missing for 13 of 14 slugs (only gemm_hmma_bf16_sm80 has `traces/`). Either restore traces via tracer rebuild + sweep, or live-gate the test under SIMBENCH_REQUIRE_LIVE_GPU=1.

6. Pre-R12 host-absolute paths queue entry is STALE — Round 27 already cleaned all 26 committed bundles; producer-side R12 fix is in place. Just drop the goal-tracker entry.

CODEX-RECOMMENDED ROUND ORDERING
A0 offline: regenerate predictor bundles from CURRENT code on existing traces; quantify stale-code delta.
A1 live: re-collect 5 activation NCU labels with FIXED launch-skip offset (R28's bug fix).
B: predictor-saturation fixes — (1) emit diagnostics first (cold_count, finite_miss_count, unique sectors, reuse-distance percentiles, working-set sectors, per-op load/store class); (2) test set/slice-aware effective capacity, sector-edge accounting in `observation_builder.py`, write-policy classification, CTA/SM interleaving model; (3) constrained calibration head (monotone/logistic) ONLY after (1)+(2).
C: microbench rehabilitation — increase trace size (NOT lower MINIMUM_BIN_SAMPLE_COUNT=50); add larger `trace_l2_accuracy` shapes OR repeat/scale microbench loops until every active reuse bin has >=50 samples; recollect labels with more repeats.

ADDITIONAL CODEX-IDENTIFIED GAP
Add a "headline validity guard" — fail closed if all included predictions have near-zero variance, or if NO microbench cells contribute, or if CI width stays as wide as the current [0.0945, 0.7533].

DELIVERABLE I WANT FROM gen-idea
A repo-grounded draft idea document that:
- Names the ultimate goal: "Trustworthy `Verified-live-sm_86` L2-sector-hit-rate predictor-vs-NCU geomean baseline" (replacing "first" with "trustworthy").
- Sequences the rounds A0 → A1 → B → C → polluter → portable_lcd in dependency order.
- For each round: motivation, scope, deliverable artifact, success criterion (what makes it "done").
- Calls out the headline-validity guard as a cross-cutting acceptance criterion landed at the end.
- Explicitly defers cross-arch sm_120 (NOT in scope).
- Identifies the 5 specific tests under suspicion for the polluter (so the polluter round has a starting point).
- For predictor saturation fixes, explicitly orders: diagnostics first, model refinements second, calibration head last.
- For microbench rehab, names the two paths (larger shapes vs repeat-and-scale) and the constraint (don't lower the 50-sample threshold).
- Calls the "drop stale pre-R12 queue entry" a 1-line cleanup, not a round.

TECHNICAL CONSTRAINTS (CLAUDE.md)
- Local NCU exception under `tools/local_labels/` (Rule 1) is the ONLY ground-truth surface allowed; this work falls under that exception.
- No driver root, no privileged CUPTI, no host-driver work (Rules 2/3/4).
- No simulator revival (Rule 5).
- Active measurement stack: CUPTI Activity, NVBit (built tracer at `tools/nvbit_tracer/tracer_tool/tracer_tool.so`), %clock64, static SASS, NVML covariates.
- Two-Device Local Development Model: this work targets sm_86 only (RTX 3090 Ti); sm_120 cross-arch is queued separately.

## Primary Direction: Sequential RLCR Loop Restart with Dependency-Ordered Rounds

### Rationale

The user's explicit deliverable enumerates dependency order (A0 → A1 → B → C → polluter → portable_lcd) and asks for "round" granularity with motivation/scope/artifact/success criteria — that maps directly onto the 35-round RLCR loop pattern just closed at HEAD `255c661`. The prior loop's `.humanize/rlcr/2026-05-10_23-00-08/` infrastructure (round-contracts, goal-tracker, finalize phase, structural scanners enforcing plan-token markers and parity literals) is battle-tested precedent for exactly this kind of multi-round dependency-ordered work. Lifting D2's single-push velocity, D3's parallelism gains, or D5's calibration-bypass shortcut would each shed an axis the user named explicitly: "round" granularity, dependency ordering, and the requirement that the calibration head land **last** within the predictor-saturation round. Sequential RLCR is the only direction that preserves all three.

### Approach Summary

Open a new RLCR session at `.humanize/rlcr/2026-05-12_<HHMMSS>/` and run six dependency-ordered rounds against a single `goal-tracker.md` whose Ultimate Goal text replaces "first artifact of the right shape" with **"Trustworthy `Verified-live-sm_86` L2-sector-hit-rate predictor-vs-NCU geomean baseline"**. Each round closes its own contract via failing-test-first discipline, lands one Codex-reviewed commit slice (or a small ordered series), and queues the next round only after the prior round's success criterion is met on disk:

- **Round A0 — Offline regeneration & stale-code delta quantification.**
  - *Motivation*: the on-disk `comparison_bundle.json` (geomean 0.3289, CI [0.0945, 0.7533], 5 cells) was produced by pre-R16 producer code. Codex flagged that re-running current `cache_predictor.py` on existing traces would yield ~0.938–0.979 hit rates, not the saturated 1.0 the bundle claims. We need the delta in writing before any model change.
  - *Scope*: re-run `tools/hrd_profiler/profiler.py` + `workloads/simbench_bw/cache_predictor.py` on the 13 already-collected sweep traces; re-run `tools/local_labels/comparison_runner.py`; emit a side-by-side `regeneration_delta.json` artifact at `data/local_labels/sweeps/<new_sweep_id>/comparison/` that records old vs. new per-cell hit rates and the new geomean + CI. No NCU re-collection in this round.
  - *Deliverable artifact*: `regeneration_delta.json` + a regenerated `comparison_bundle.json` committed under a new sweep directory with `source_commit` pinned to current HEAD.
  - *Success criterion*: the regenerated bundle's per-cell predictions vary across the 5 admitted cells (no longer all 1.0000), AND the `regeneration_delta.json` quantifies which of the 4 staleness sources (R16/R17/R24/R28) contribute most to the geomean shift. Round closes `Offline-verified` since no live GPU runs.

- **Round A1 — Live re-collection of activation NCU labels with fixed launch-skip.**
  - *Motivation*: R28 caught that the 5 admitted activation labels (`layernorm/rmsnorm/silu_correctness/silu_timing/softmax_safe`) were collected with `--launch-skip=0`, which profiles the first warmup launch instead of the first measured launch. The numerator (NCU hit rate) is therefore measuring the wrong kernel instance.
  - *Scope*: invoke `tools/local_labels/run_suite.py` (or the appropriate sweep entrypoint) on the 3090 Ti with the corrected launch-skip offset for those 5 cells only; recompute `comparison_bundle.json` joining new labels with A0's regenerated predictions; record device, driver, CUDA toolkit, NVBit build, and observed clocks per the `Verified-live-sm_86` evidence requirements.
  - *Deliverable artifact*: a new sweep directory with the 5 re-collected NCU labels + a re-joined `comparison_bundle.json`.
  - *Success criterion*: all 5 cells re-join the bundle (no `comparison_runner_join_key_unmatched`), AND the launch-skip metadata stamped on each label matches the corrected value, AND the round summary names the sm_86 device + observed clocks. Closes `Verified-live-sm_86`.

- **Round B — Predictor-saturation fixes (3 sub-slices, ordered).**
  - *Motivation*: even after A0+A1, Codex predicts the LRU walker in `tools/hrd_profiler/observation_builder.py::_walk_launch` will still over-predict because it (a) is fully-associative, (b) walks deterministically in trace order, (c) ignores set/slice geometry, sector-edge effects, write policy class, and CTA/SM interleaving.
  - *Scope* (strict ordering — calibration head MUST be last):
    1. **Diagnostics first**: extend `_walk_launch` (or its caller) to emit `cold_count`, `finite_miss_count`, `unique_sector_count`, reuse-distance percentiles (p50/p90/p99), `working_set_sector_count`, and per-op load/store class as new fields on the per-launch HRD output. No model behavior change.
    2. **Model refinements second**: add set/slice-aware effective capacity (probe the sm_86 L2 slice geometry from the device-properties manifest), sector-edge accounting at the 32B granularity, write-policy classification, and a coarse CTA/SM interleaving approximation. Each refinement lands as its own slice with a failing-test-first witness asserting predicted hit rate moves in the expected direction on a synthetic launch.
    3. **Calibration head last**: only after diagnostics + model refinements both committed and re-running A0 produces meaningfully varying predictions, add a constrained (monotone or logistic) post-hoc calibration head fit on the 5 activation cells. Forbid this slice from landing if (1) and (2) have not both demonstrably improved per-cell variance.
  - *Deliverable artifact*: a regenerated bundle with per-cell predictions whose stddev across the 5 cells exceeds a documented floor (e.g., >= 0.05) AND whose geomean rel-err improves over A0's regenerated baseline.
  - *Success criterion*: per-cell prediction variance clears the documented floor, AND no slice was merged out of order (verified by the round summary's commit-list ordering). Closes `Verified-live-sm_86` if A1's labels were re-used; otherwise `Offline-verified` and queues a follow-up.

- **Round C — Microbench rehabilitation.**
  - *Motivation*: ALL 8 microbench predictor bundles return `cache_model_underconfigured` because trace_00 traces are too small for `MINIMUM_BIN_SAMPLE_COUNT=50` in `tools/hrd_profiler/profiler.py`. Net effect: zero microbench cells contribute to the headline geomean, so the original "memory-stress on L2" cohort intent is invisible.
  - *Scope* (two paths, choose by per-cell feasibility — DO NOT lower the 50-sample threshold):
    1. **Larger shapes**: add larger shape variants under `workloads/simbench_bw/l1/<microbench_slug>/` (e.g., `trace_l2_accuracy_xl`) sized so that every active reuse bin clears 50 samples after a single pass.
    2. **Repeat-and-scale**: increase the per-microbench loop iteration count so the existing shape produces enough sector-stream samples per bin, and recollect labels with proportionally more NCU repeats to hold label-side `label_repeatability_failed` below threshold.
  - *Deliverable artifact*: regenerated microbench predictor bundles with `result_kind != typed_unavailable` and `typed_reason != cache_model_underconfigured` for at least 5 of 8 microbench slugs, AND the headline `comparison_bundle.json` shows microbench cells contributing.
  - *Success criterion*: at least 5 microbench cells re-join the headline geomean, AND the 50-sample threshold is unchanged in `profiler.py` (verified by a static scan in the round summary). Closes `Verified-live-sm_86`.

- **Round Polluter — Trace-sidecar polluter root-cause.**
  - *Motivation*: `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/trace_trace_00.json` (and 3 sibling files) re-emit on every aggregate suite run; Round 20's mitigation handles symptom only. The 5 specific tests under suspicion are the starting investigation surface:
    1. `workloads/simbench_bw/tests/test_run_suite_handle_trace_user_defined_folders.py`
    2. `workloads/simbench_bw/tests/test_microbench_correctness_typed_unavailable.py`
    3. `workloads/simbench_bw/tests/test_run_suite_trace_mode.py`
    4. (any sibling test that calls `handle_trace` with `row.slug` constructed from a dict)
    5. (any sibling test that omits a monkeypatch on `args_out_dir_root`)
  - *Scope*: instrument `handle_trace` with a fail-closed assertion that `args_out_dir_root` resolves under a `tmp_path` when invoked from pytest; run the suspect tests one-by-one until the assertion fires; fix the offending test to pass `tmp_path` (or equivalent monkeypatch) and remove Round 20's `regenerate_in_place` workaround so the producer-side discipline is restored.
  - *Deliverable artifact*: a failing-test-first commit demonstrating the polluter, then a fix commit removing it, then the cleanup commit dropping the `regenerate_in_place` workaround.
  - *Success criterion*: aggregate suite run produces no diff under `workloads/simbench_bw/results/dram_stream_stride_mix/`. Closes `Offline-verified` (host-only test work).

- **Round portable_lcd — Live-gate or restore traces.**
  - *Motivation*: 13 of 14 portable_lcd slugs lack `traces/` dirs (only `gemm_hmma_bf16_sm80` has one); the verify_kernel_contracts test fails `nvbit_trace_missing` on every aggregate run, polluting the test signal.
  - *Scope* (choose one):
    1. **Restore traces**: rebuild the NVBit tracer for 3090 Ti (`NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool`) and run the per-slug sweep that emits `workloads/portable_lcd/<slug>/traces/`.
    2. **Live-gate**: gate the verify_kernel_contracts test under `SIMBENCH_REQUIRE_LIVE_GPU=1` so it only runs when traces are expected to be present, with a typed-skip diagnostic when they are not.
  - *Deliverable artifact*: either 13 new `traces/` dirs committed under `workloads/portable_lcd/`, or a gating commit on the test plus a documented runbook entry on how to restore traces.
  - *Success criterion*: `portable_lcd` no longer carries a pre-existing failing test in the aggregate suite (Round 15's carried-forward failure is finally retired). Closes `Verified-live-sm_86` if traces restored, else `Offline-verified`.

**Cross-cutting acceptance criterion — Headline-validity guard (lands in the LAST round merged, typically portable_lcd or in a finalize phase):** add a fail-closed assertion in `tools/local_labels/comparison_runner.py` (or its downstream `comparison_bundle.json` emitter) that rejects the bundle if any of the following hold:
- All included predictions have near-zero variance (stddev across cells < documented floor).
- No microbench cells contribute (Round C must already have shipped microbench cells, otherwise this guard fires).
- CI width stays as wide as the current `[0.0945, 0.7533]` (i.e., > 0.65).

The guard is a typed `headline_validity_failed` reason that surfaces in the bundle's typed_unavailable path; it is NOT a CI hardware test, it runs on every aggregate suite invocation, and it landing **last** ensures it's the final gate that re-classifies the current "first artifact" bundle as `typed_unavailable` until A0+A1+B+C have all closed.

**1-line cleanup (not a round):** drop the stale pre-R12 host-absolute paths entry from the goal-tracker `Queued Side Issues` section in the same commit that opens Round A0's contract — Round 27 already cleaned all 26 committed bundles, the producer-side R12 fix is in place, the entry is dead.

**Explicitly out of scope (cross-arch sm_120 deferred):** none of the six rounds opens any `Verified-live-sm_120` work. The cross-arch sm_120 sweep, the sm_120-side classifier extensions, and the cross-arch `Cross-arch-verified` status closure remain queued for a separate session and a separate RLCR loop.

### Objective Evidence

- **Prior 35-round RLCR loop infrastructure exists**: `.humanize/rlcr/2026-05-10_23-00-08/` with goal-tracker, round-contracts, finalize-summary, and Codex-review-passed evidence (HEAD `255c661`, finalize HEAD `e58708f`). The pattern is the project's proven multi-round dependency-ordered work mechanism.
- **Headline bundle on disk**: `data/local_labels/sweeps/sweep_20260511T030924Z_7189cda8e43f/comparison/comparison_bundle.json` carries `geomean=0.3289`, CI `[0.0945, 0.7533]`, 5 admitted cells (all predict 1.0000) — the "first artifact" the new loop replaces.
- **Saturation cause located**: Codex review fingerprinted `tools/hrd_profiler/observation_builder.py::_walk_launch` (fully-associative LRU, deterministic trace order, no slice geometry/sector edges/write policy/CTA-SM interleave) as the remaining bias source after A0's regeneration.
- **Microbench underconfig root cause**: `tools/hrd_profiler/profiler.py::MINIMUM_BIN_SAMPLE_COUNT=50` (line 46); per-cell admitted_bins 1–3 vs. underconfigured_bins 1–12 across all 8 microbench slugs.
- **Polluter timing**: file `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/trace_trace_00.json` last re-emitted 2026-05-12 05:49:33 — exactly during the most recent aggregate suite run.
- **portable_lcd state**: 13 of 14 slugs missing `traces/` dirs on disk; only `gemm_hmma_bf16_sm80` has one.
- **Pre-R12 host-absolute paths is provably stale**: Round 27's path-rewrite already cleaned all 26 committed predictor bundles; both committed sweeps now show repo-relative `workloads/simbench_bw/results/...` paths.
- **CLAUDE.md gates**: Rule 1's `tools/local_labels/` exception is the only ground-truth surface available; Rules 2-4 forbid driver root, privileged CUPTI, and host-driver work; Rule 5 forbids simulator revival. All six rounds stay within these gates.
- **Two-Device Local Development Model**: the loop is explicitly sm_86-only; cross-arch sm_120 is queued separately, matching the user's deferral.

### Known Risks

- **Round B over-runs its sub-slice budget**: predictor-saturation refinements (set/slice-aware capacity, sector-edge accounting, write-policy classification, CTA/SM interleaving) are 4 distinct model changes, each with non-trivial test surface. Mitigation: each sub-slice lands as a separate commit with its own failing-test-first witness; if any sub-slice exceeds 3 attempts to clear Codex review, it queues as a follow-up sub-round rather than blocking B's closure.
- **A1 cannot run without operator availability on the 3090 Ti**: the launch-skip re-collection requires a live GPU sweep with NCU. If the operator is on the sm_120 host, A1 stalls until the next 3090 Ti session. Mitigation: A0 (offline) is fully decoupled and can land in either session; A1 is the only round that hard-requires the 3090 Ti.
- **Round C's "larger shapes" path may invalidate prior microbench traces**: adding `trace_l2_accuracy_xl` shapes changes the cohort definition; old microbench bundles become incomparable. Mitigation: prefer the "repeat-and-scale" path when feasible (preserves shape but increases sample count); reserve "larger shapes" only for slugs where repeat-and-scale cannot reach 50 samples.
- **Polluter root cause may not be in the 5 named tests**: the hypothesis ("`handle_trace` with row.slug from a dict + default args_out_dir_root") is a deep-research best guess. Mitigation: the fail-closed assertion approach in `handle_trace` itself catches the offender regardless of which test invokes it; the 5 named tests are the starting search order, not the closed candidate set.
- **Headline-validity guard could fire spuriously on early A0+A1 bundles**: if Round B is queued behind Round A1 and the validity guard lands in the LAST round, it might be tempting to land it earlier and then suppress it. Mitigation: the guard MUST land last and MUST NOT have any suppression flag; if the guard fires after A0+A1 alone, that is the correct signal that B+C are still required.
- **Codex review may demand additional rounds mid-stream**: if Codex flags new gaps during B or C reviews, the loop expands. Mitigation: the loop is allowed to grow — six rounds is the floor, not the ceiling. Any new round queues as B'/C'/etc. without changing the dependency ordering of existing rounds.
- **Auto-memory loss on session pivot**: if the 3090 Ti session pivots to sm_120 mid-loop, `.humanize/rlcr/` and `.humanize/bitlesson.md` survive raw scp but not git fetch (per CLAUDE.md). Mitigation: do all session pivots via raw scp during this loop; defer the sm_120 cross-arch work until after this loop's closure.

## Alternative Directions Considered

### Alt-1: Operator-Driven Single Push

**Rationale**: collapse the six dependency-ordered rounds into a single workstream of ~6 commits with continuous aggregate-suite validation, no formal RLCR contracts. **Approach**: open a feature branch off `5060_3090ti_cross_dev`, land each round's slice as a single commit with the same failing-test-first discipline but no per-round contract/summary file. **Evidence**: the project's commit cadence (e58708f, 255c661, aa4f71f, 1a19eb1, 13bbdc2 — all single-purpose commits within the prior 24h window) shows a single-push workflow is achievable. **Risks**: loses the per-round Codex-review checkpoint that caught the saturation-cause correction during gen-idea exploration; loses the goal-tracker as a queryable structured artifact; harder to recover from a mid-stream pivot. **Why not primary**: user explicitly asked for "round" granularity and dependency ordering; single-push optimizes for velocity but discards the audit-trail discipline that protected the prior 35-round loop from regressions.

### Alt-2: Dependency-Graph Parallel Tracks

**Rationale**: model the six rounds as a DAG where independent task cohorts run in parallel — A0 and Polluter and portable_lcd have no inter-dependencies, so they can land concurrently while A1 → B → C run as the critical path. **Approach**: open three concurrent worktrees (A0, polluter+portable_lcd, A1+B+C critical path); each worktree lands its commits to a topic branch; integration commit at the end. **Evidence**: the EnterWorktree tool and the project's fish-shell + Linux substrate support concurrent worktrees; D3's exploration found 3 of 6 rounds have no inter-dependencies. **Risks**: the headline-validity guard depends on every other round being closed, so the integration commit becomes a bottleneck regardless; risk of merge conflicts when concurrent worktrees touch the same observation_builder.py / comparison_runner.py / profiler.py files. **Why not primary**: the user explicitly named "dependency order" as the structuring principle; parallelism gains are real but contradict the explicit deliverable. Could fold back if A0 + Polluter + portable_lcd land as a parallel side-cohort during Round A0's contract.

### Alt-3: Validity-Guard-First Inverted Design

**Rationale**: land the headline-validity guard FIRST as a forcing function — it immediately re-classifies the current `comparison_bundle.json` as `typed_unavailable`, removing any incentive to treat 0.3289 as a credible number while the remediation rounds are in flight. **Approach**: open the new loop with the validity guard as Round 0; A0 → A1 → B → C → polluter → portable_lcd then close one validity-guard exit condition each. **Evidence**: the validity guard is the cleanest typed-fail-closed pattern in the codebase (matches BL-20260509-strict-admission-contract-first); landing it first prevents the common project anti-pattern of treating a stale headline as evidence. **Risks**: the validity guard with no remediation behind it produces a `typed_unavailable` bundle for an indefinite window; if the loop stalls, the project has no headline number AT ALL. **Why not primary**: user's explicit ordering puts the validity guard "as a cross-cutting acceptance criterion landed at the end". Invertion would directly contradict the deliverable. The forcing-function value is real and should fold back as the **primary's** rationale for why the validity guard's slice is non-negotiable in the final round.

### Alt-4: Calibration Head Bypass (Post-Comparison Correction Layer)

**Rationale**: skip the model-refinement work in Round B entirely; replace it with an isotonic-regression calibration head fit on the 5 activation cells' (predicted, label) pairs, applied post-comparison-runner. **Approach**: add a `tools/local_labels/calibration_head.py` that consumes `comparison_bundle.json`, fits a monotone calibration on the per-cell pairs, and emits a `calibrated_bundle.json` alongside. **Evidence**: isotonic regression is a well-understood post-hoc correction; sklearn or a hand-rolled PAV implementation is ~50 LOC. **Risks**: (a) violates the user's explicit "calibration head LAST" ordering — it would land calibration FIRST; (b) hides the underlying LRU-walker bias rather than fixing it, so any new microbench cell admitted in Round C re-exposes the bias; (c) overfits to 5 cells, producing a calibration that doesn't generalize. **Why not primary**: directly contradicts the user's "diagnostics first, model refinements second, calibration head last" ordering. Could fold back as Round B's third sub-slice (which is exactly where the primary places it).

### Alt-5: Cohort Pivot to portable_lcd Kernels

**Rationale**: replace the 13-cell simbench_bw cohort with portable_lcd kernels (conv2d_implicit_gemm, flash_attn_fwd/bwd, gemm_hmma_fp16, etc.) which have larger working sets that exercise L2 more aggressively, naturally clearing the MINIMUM_BIN_SAMPLE_COUNT=50 threshold and giving the LRU walker more discriminative material. **Approach**: shift the headline cohort definition; rebuild traces for the 13 missing portable_lcd slugs; treat simbench_bw as the secondary cohort. **Evidence**: portable_lcd kernels are documented as larger working set in the workload tree; D6's exploration found the cohort size and kernel diversity would naturally improve discrimination. **Risks**: (a) cohort change invalidates 35 rounds of simbench_bw infrastructure; (b) portable_lcd has its own `nvbit_trace_missing` problem (13 of 14 slugs); (c) the simbench_bw "memory-stress on L2" cohort intent is the prior loop's stated design — pivoting drops that intent rather than honoring it. **Why not primary**: the user named "queued side issues" + "two follow-on rounds (regeneration, predictor-saturation fix)" — implying repair, not replacement. Cohort pivot folds back as a Round-portable_lcd opportunity if "restore traces" is the chosen sub-path.

## Synthesis Notes

**Why D1 over D2 (Operator-Driven Single Push)**: D1 keeps the per-round Codex review checkpoint; D2 collapses it. Codex review is the mechanism that surfaced the "predictor would actually predict 0.938-0.979, not 1.0" correction during this very gen-idea session — losing that checkpoint per round would compound similar misreads.

**Why D1 over D3 (Dependency-Graph Parallel Tracks)**: D1 honors the user's explicit "dependency order" framing. D3's parallelism is real (A0 + Polluter + portable_lcd genuinely have no inter-dependencies), and could fold back as concurrent worktrees within Round A0's contract window if velocity becomes a constraint. The primary direction does not preclude this.

**Why D1 over D4 (Validity-Guard-First)**: the user's ordering puts the validity guard at the END, not the beginning. D4's forcing-function logic is correct on the merits but inverts the explicit deliverable. The primary preserves D4's value by making the validity guard's "lands last" placement non-negotiable and by typing its rejection reason explicitly (`headline_validity_failed`).

**Why D1 over D5 (Calibration Head Bypass)**: the user's ordering for Round B is "diagnostics first, model refinements second, calibration head last". D5 lands calibration FIRST. The primary preserves D5's actual value (calibration as the final closure step) by making it Round B's third sub-slice, with a hard guard that it cannot land before sub-slices 1 and 2.

**Why D1 over D6 (Cohort Pivot to portable_lcd)**: the user named "queued side issues" + "follow-on rounds" — repair language, not replacement language. Pivoting cohorts would also drop 35 rounds of simbench_bw infrastructure. D6's value is real for the portable_lcd round itself (the "restore traces" sub-path is the cohort-pivot path under a different name), but the primary headline cohort stays simbench_bw.

**Folded-back elements** (each alternative's best contribution to the primary):
- D2 → Within each round, prefer single-commit slices when the contract is small (avoid commit-spam ceremony for trivial changes).
- D3 → A0 + Polluter + portable_lcd may launch as concurrent worktrees if operator-time is a bottleneck; the dependency-order narrative still governs the merge order.
- D4 → The headline-validity guard's "lands last" placement is treated as non-negotiable, with an explicit typed-reason name (`headline_validity_failed`).
- D5 → The calibration head sub-slice in Round B is gated on per-cell variance demonstrably improving from sub-slices 1 and 2 first (no shortcut admitted).
- D6 → If portable_lcd round chooses "restore traces", the resulting traces remain available for any future cohort-pivot decision.

**Cross-cutting deferrals (explicitly NOT in this loop)**:
- Cross-arch sm_120 work (queued separately per Two-Device Local Development Model and the user's explicit deferral).
- L1 hit-rate and IPC/occupancy/wall-clock predictors (the prior loop and this loop both target L2 sector hit rate only; broader telemetry-axis expansion is a follow-on goal).
- Simulator revival (forbidden by Rule 5; not on the table).
- Any privileged CUPTI / driver-root / NCU on cloud (forbidden by Rules 1-4; not on the table).

**1-line cleanup (not a round, lands in the same commit that opens Round A0's contract)**: drop the stale pre-R12 host-absolute paths entry from the goal-tracker `Queued Side Issues` section. Round 27 already cleaned all 26 committed bundles; the entry is dead.

--- Original Design Draft End ---
