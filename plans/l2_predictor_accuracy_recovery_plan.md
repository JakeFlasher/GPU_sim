# L2 Predictor Accuracy Recovery (sm_86 — AC-4 Sub-Slice Empirical Closure + AC-5 Calibration Application)

## Goal Description

Reduce the L2 sector hit-rate predictor's systematic over-prediction on sm_86 — current admitted geomean relative error is 0.319 (95% CI [0.125, 0.640]) with per-cell range 0.051–0.739 on the 5-cell activation cohort — by empirically closing the four deferred AC-4 sub-slice refinements (slice geometry, sector edges, write policy, CTA/SM interleaving) in dependency order, then applying the AC-5 calibration head as a post-prediction transform if the variance precondition is satisfied. Each sub-slice closure MUST be gated by a hard before/after artifact over the admitted-cell corpus — contract closure (witness tests pass) is NOT sufficient; the headline must measurably move OR the absence of movement must be classified and recorded (e.g., "this corpus is single-TB so AC-4.4 produces a no-op delta on it"). Scope: sm_86 only; cross-architecture transfer (sm_120, sm_100) and live AC-T5.1 SASS regeneration continue to defer to the `Tier-2-cross-arch` follow-up loop per the user's 2026-05-16 directive.

Key context: the active baseline plan `plans/cache_predictor_l2_trustworthy_baseline_plan.md` defined AC-4.1–AC-4.4 + AC-5 as Round B sub-slices; the draft analysis discovered that all four sub-slices are LARGELY already implemented in `tools/hrd_profiler/observation_builder.py` + `workloads/simbench_bw/cache_predictor.py` + `workloads/simbench_bw/calibration_head.py`, so the work is empirical-impact verification + witness formalization + variance-gate adjudication, not new physics. The Codex first-pass analysis flagged that the existing `data/local_labels/audit_evidence/ac5_variance_precondition_diagnostic/variance_trajectory.json` records `current_stddev=0.0217`, well below the 0.10 floor — so AC-5 calibration application may be structurally blocked on the current 5-cell corpus regardless of AC-4 improvements, and the plan MUST handle this fail-closed path honestly.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Implementation-state inventory of the four AC-4 sub-slices. A committed artifact maps each sub-slice (AC-4.1 / 4.2 / 4.3 / 4.4) to (a) the producer-side code path, (b) the existing witness test (if any), (c) the bundle field that carries its parameter, (d) any known no-op condition on the current admitted corpus (e.g., single-TB cells produce no CTA-interleave delta).
  - Positive Tests:
    - Inventory artifact `data/sm_86_predictor_accuracy_recovery/ac4_implementation_state_inventory.json` exists and lists exactly 4 entries (AC-4.1 through AC-4.4), each carrying `code_paths` (≥1), `existing_witness_tests` (≥0), `bundle_fields_consumed` (≥1), `known_no_op_conditions` (string).
    - Each entry's `code_paths` list contains at least one entry under `tools/hrd_profiler/`, `workloads/simbench_bw/`, or `tools/local_labels/`.
  - Negative Tests:
    - An inventory missing any of the four sub-slices is rejected by a scanner.
    - An entry with an empty `code_paths` list is rejected.
    - An entry that references a `code_paths` value that does not exist on disk is rejected by a path-existence scanner.

- AC-2: Bundle provenance scanner enforces that every emitted sm_86 predictor bundle in `data/local_labels/sweeps/<sweep_id>/predictor_bundles/` carries the post-refinement geometry/interleave/second-layer fields with non-default values. Default values (`cache_model_num_l2_slices=1` indicating no slice geometry; `cache_model_cta_interleave_enabled=false` indicating no interleave; `cache_model_hrd_second_layer_enabled=false` indicating Maeda layer disabled) MUST be rejected as stale per-bundle metadata.
  - Positive Tests:
    - A scanner reads every bundle in the latest sm_86 sweep and asserts `cache_model_num_l2_slices > 1` (typically 6 per `tools/hrd_profiler/device_properties.json`), `cache_model_cta_interleave_enabled == True` for multi-TB launches, `cache_model_hrd_second_layer_enabled == True` everywhere.
    - The scanner emits per-bundle diagnostic entries naming which fields are non-default and which are still default.
  - Negative Tests:
    - A bundle with `cache_model_num_l2_slices=1` triggers the scanner's typed defect `stale_predictor_bundle_default_num_l2_slices`.
    - A bundle missing `cache_model_geometry_provenance` triggers `bundle_geometry_provenance_missing`.
    - A bundle whose `cache_model_geometry_provenance.manifest_sha256` does not match the on-disk `tools/hrd_profiler/device_properties.json` SHA-256 triggers `bundle_geometry_provenance_stale`.

- AC-3: Per-sub-slice ablation artifact reports BEFORE/AFTER admitted-cell predictions for each AC-4 sub-slice. The artifact is the load-bearing "headline moved" evidence — contract closure (witness tests pass) without measurable per-cell prediction change on the corpus is INSUFFICIENT.
  - Positive Tests:
    - Artifact `data/sm_86_predictor_accuracy_recovery/ac4_per_slice_ablation.json` carries one entry per AC-4 sub-slice with fields: `sub_slice_id` (`AC-4.1` | `AC-4.2` | `AC-4.3` | `AC-4.4`), `before_predicted_hit_rate` (per cell), `after_predicted_hit_rate` (per cell), `before_signed_residual` (per cell), `after_signed_residual` (per cell), `before_relative_error` (per cell), `after_relative_error` (per cell), `before_geomean_rel_err` (scalar), `after_geomean_rel_err` (scalar), `before_pred_stddev` (scalar), `after_pred_stddev` (scalar).
    - The artifact covers all 5 admitted activation cells from the latest sm_86 sweep (`layernorm_fwd_bf16_sm80/timing_shape`, `rmsnorm_fwd_bf16_sm80/timing_shape`, `silu_fwd_bf16_sm80/correctness_shape`, `silu_fwd_bf16_sm80/timing_shape`, `softmax_safe_bf16_sm80/timing_shape`).
    - For each sub-slice with `after_geomean_rel_err < before_geomean_rel_err`, the artifact records `headline_moved: true`; for any sub-slice with `after_geomean_rel_err >= before_geomean_rel_err`, the artifact records `headline_moved: false` AND a `no_op_classification` enum value from the documented set (`single_tb_corpus`, `working_set_fits_in_one_slice`, `no_access_kind_in_trace`, `single_launch_corpus`, `other`).
  - Negative Tests:
    - An artifact with any sub-slice missing the 12 required fields is rejected.
    - A `headline_moved: false` entry without a `no_op_classification` is rejected.
    - A `headline_moved: true` entry whose `after_geomean_rel_err >= before_geomean_rel_err` (logical contradiction) is rejected.

- AC-4: Per-sub-slice empirical closure. Each of the four sub-slices closes only when (a) its witness test passes AND (b) its AC-3 ablation entry shows either `headline_moved: true` OR `headline_moved: false` with a documented `no_op_classification` cleared as expected by the corpus shape.
  - AC-4.1: Slice geometry empirical closure. The slice-aware capacity calculation at `observation_builder.py::_walk_launch` lines ~244-260 + 391-393 + 602-605 + 621 is exercised on every admitted bundle.
    - Positive: `test_slice_overflow_decreases_hit_rate` passes (synthetic witness with documented >5% margin). The latest sm_86 sweep's bundles ALL carry `cache_model_num_l2_slices=6` per the device_properties manifest. The AC-3 ablation entry for AC-4.1 either shows `after_geomean_rel_err < before_geomean_rel_err` OR documents `working_set_fits_in_one_slice` as the no-op classification per cell.
    - Negative: a synthetic single-slice working-set kernel that should NOT move hit rate under slice-aware capacity (negative control) MUST NOT show a delta — `test_small_working_set_unchanged_hit_rate` regression test enforces this.
  - AC-4.2: Sector-edge accounting formalization. The sector expansion loop at `observation_builder.py` lines ~545-547 + the `SectorEdgeAccountingTest` class lines ~295-487 cover all 4 contract cells (aligned-fits, aligned-crosses, sub-32B-fits, sub-32B-crosses).
    - Positive: `SectorEdgeAccountingTest` passes with explicit docstring tying each test method to one of the four contract cells. The AC-3 ablation entry documents the per-cell delta (expected to be small on the current corpus because the activation kernels use 16-bit-aligned 32-byte loads).
    - Negative: a synthetic 16B-aligned 32B load spanning a sector boundary that emits only 1 sector access (vs the contract-required 2) is rejected by the witness.
  - AC-4.3: Write-policy classification with fresh-trace requirement. The producer must emit traces that populate `memory_op_access_kind` (load/store/atomic/async_copy); legacy traces lacking this field route to `unspecified_kind_count` and produce a no-op for AC-4.3.
    - Positive: at least one admitted-cell trace in the latest sm_86 sweep has populated `memory_op_access_kind` per-access fields (verifiable via a scanner over `data/local_labels/sweeps/<sweep_id>/predictor_bundles/<bundle>/observations.json` or equivalent). A new synthetic store-only kernel test asserts `store_count > 0, load_count == 0` and that the predictor's hit rate correctly excludes no-write-allocate paths. A new synthetic load-only kernel test asserts the inverse.
    - Negative: a bundle whose access-kind counts are all `unspecified_kind_count` triggers a typed `legacy_trace_missing_access_kind` defect AND the AC-4.3 ablation entry documents `no_access_kind_in_trace` as the no-op classification (rather than asserting empirical impact).
  - AC-4.4: CTA/SM interleave with empirical gate. The round-robin merge at `observation_builder.py::_round_robin_merge_tb_accesses` lines ~301-327 must be exercised by at least one multi-TB admitted cell where shared sectors across CTAs are demonstrably present.
    - Positive: `CtaInterleaveAggregationTest` passes; the test class is extended with a new multi-CTA shared-sector witness showing interleave=True increases hit rate vs interleave=False by ≥10% margin on the synthetic fixture. The AC-3 ablation entry for AC-4.4 documents the per-cell delta on the multi-TB admitted cells (e.g., silu_fwd timing_shape with 1,024 CTAs is the canonical headline candidate).
    - Negative: a synthetic disjoint-working-set multi-CTA fixture MUST show ≤1% delta between interleave=True and interleave=False (no false positive). A single-TB cell that should produce a no-op MUST NOT be misclassified as `headline_moved: true`.

- AC-5: Geometry sanity audit. A committed artifact compares `tools/hrd_profiler/device_properties.json` sm_86 entry against (a) `cudaDeviceProp::l2CacheSize` on the actual 3090 Ti (RTX 3090 Ti GA102 die), (b) any cuobjdump or nvidia-smi output that reveals slice count, (c) the published per-architecture-whitepaper L2 partition information, and records the rationale for the chosen `num_l2_slices=6` and `slice_hash_model=sector_id_mod_num_l2_slices`. The audit explicitly documents the modulo-slice-hash as a coarse approximation (real hardware may use address-bit XOR / page coloring) and states the residual modeling-uncertainty bound.
  - Positive Tests:
    - Artifact `data/sm_86_predictor_accuracy_recovery/geometry_sanity_audit.json` exists with fields: `arch`, `sku`, `manifest_num_l2_slices`, `cuda_device_prop_l2_cache_size_bytes`, `published_partition_count`, `chosen_slice_hash_model`, `modeling_uncertainty_classification` (one of `published_exact_match`, `published_approximation`, `unverified`), `audit_provenance` (git commit + tooling).
    - The audit's `manifest_num_l2_slices` matches `tools/hrd_profiler/device_properties.json` exactly (lockstep check).
  - Negative Tests:
    - An audit asserting `modeling_uncertainty_classification: published_exact_match` without `published_partition_count` populated is rejected.
    - A drift between `manifest_num_l2_slices` and the audit's recorded value triggers a typed `geometry_manifest_audit_drift` defect.

- AC-6: AC-5 variance-precondition + apply-policy adjudication. The follow-up loop reads `data/local_labels/audit_evidence/ac5_variance_precondition_diagnostic/variance_trajectory.json` AFTER the AC-4 sub-slices land and re-emits the diagnostic. If `current_stddev >= 0.10` (the parent plan's documented floor), the AC-5 calibration head is APPLIED as a post-prediction transform inside `cache_predictor.predict_cache_hit_rate(...)` (or as a compare-time transform inside `comparison_runner.py` per DEC-4 resolution); the canonical `predicted_hit_rate` field reflects the calibrated value AND a new diagnostic `raw_predicted_hit_rate` field carries the pre-calibration scalar. If `current_stddev < 0.10`, the AC-5 calibration head is NOT applied; the predictor bundle's `headline_validity_failed_reasons` includes `calibration_variance_precondition_failed` and the bundle emits with `result_kind="typed_unavailable"` per the typed-fail-closed contract.
  - Positive Tests:
    - Post-AC-4 re-emission of `variance_trajectory.json` records the updated `current_stddev` value.
    - If `current_stddev >= 0.10`, at least one admitted predictor bundle's `predicted_hit_rate` differs from its `raw_predicted_hit_rate` field (verifying the transform fired).
    - If `current_stddev < 0.10`, the bundle's `headline_validity_failed_reasons` includes `calibration_variance_precondition_failed` AND the bundle emits with `result_kind="typed_unavailable"`.
  - Negative Tests:
    - A bundle with `predicted_hit_rate == raw_predicted_hit_rate` while the variance gate is satisfied (calibration should have fired but didn't) triggers a typed `calibration_apply_not_fired_after_gate_pass` defect.
    - A bundle with `result_kind="admitted"` while the variance gate failed (calibration should have fail-closed but didn't) triggers `calibration_variance_gate_bypass`.

- AC-7: Residual root-cause classification report. After all AC-4 sub-slices land and AC-5 either applies or fail-closes, a committed artifact `data/sm_86_predictor_accuracy_recovery/residual_root_cause_report.md` classifies the remaining sm_86 admitted-cell relative error into named buckets: `slice_geometry_residual`, `cta_interleave_residual`, `write_policy_residual`, `cross_launch_state_residual`, `sector_edge_residual`, `label_variance_floor`, `unsupported_corpus_shape`. Per-bucket quantification cites the AC-3 ablation deltas where possible.
  - Positive Tests:
    - Report exists with all 7 named buckets enumerated.
    - For each bucket, the report has either a quantified residual (cite AC-3 ablation delta in pp) OR a documented "unattributable / mixed-attribution" entry.
    - The sum of attributed residuals + unattributable residuals approximately equals the post-AC-4 geomean rel_err on the admitted corpus.
  - Negative Tests:
    - A report missing any of the 7 buckets is rejected by a scanner.
    - A report whose per-bucket attribution sum is wildly inconsistent with the post-AC-4 geomean (e.g., off by >50% relative) is flagged as a `residual_attribution_inconsistency` defect.

- AC-8: Final sm_86 sweep summary. A committed sweep under `data/local_labels/sweeps/<sweep_id>/comparison_bundle.json` emits with: admitted-cell identities (5 activation cells + ≥1 microbench cell if available), per-arch `coverage_gate_verdict.meets_floor=true` for sm_86, per-arch `relative_error_stats` (mean/stdev/min/max + quartiles when n≥4), per-arch `relative_error_geomean_ci` block, `headline_validity_failed_reasons` array reflecting the AC-5 gate outcome, evidence status `Verified-live-sm_86`.
  - Positive Tests:
    - Bundle validates against `workloads/simbench_bw/schemas/comparison_bundle.schema.json`.
    - Bundle records the sweep's source_commit AND the `tools/local_labels/fit_calibration_head.py` was at that source_commit (via existing `cat-file` source-commit gate from the prior loop's Round 12 closure).
    - The Round summary in the new RLCR loop session records `Verified-live-sm_86` evidence status per CLAUDE.md Two-Device Local Development Model.
  - Negative Tests:
    - A bundle that records `Verified-live-sm_120` (without RTX 5060 laptop hardware in this loop) is rejected by an evidence-status scanner.
    - A bundle that records `Cross-arch-verified` is rejected (this loop is sm_86-only per scope).
    - A bundle with `result_kind="admitted"` but `headline_validity_failed_reasons` non-empty (cardinality contradiction) is rejected.

- AC-9: Loop discipline. Each round in the follow-up RLCR loop closes by Codex review per the existing framework's stop-hook; each round contract names slices in dependency order, includes failing-test-first witness criteria, names which AC-4 sub-slice is the round's target, and respects the post-commit recapture invariant. The variance-precondition + AC-5 apply decision is gated in the round it lands; no AC-5 apply round may close before the AC-5 variance precondition is empirically verified post-AC-4.
  - Positive Tests:
    - Round contract for every round under `.humanize/rlcr/<loop_session>/round-N-contract.md` names exactly one AC-4 sub-slice OR the AC-5 apply decision OR a documented post-AC-7 closure round; each round summary cites its AC-3 ablation entry contribution.
    - Codex review of each round's summary uses the standard verdict template.
    - The bitlesson scanner enforces BL selection per BL-20260506 + BL-20260507 + BL-20260509 for every round.
  - Negative Tests:
    - A round contract that bundles two AC-4 sub-slices into one round is flagged (`round_scope_overcommitted`).
    - A round summary that claims an AC-4 sub-slice closed without citing an AC-3 ablation entry is flagged (`ac4_closure_without_ablation_artifact`).
    - A round that lands AC-5 apply without first emitting the post-AC-4 variance precondition artifact is flagged (`ac5_apply_without_precondition_verification`).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The follow-up loop lands all four AC-4 sub-slices with empirical before/after ablation artifacts over the 5 admitted activation cells; for each sub-slice that produces a no-op delta on the current corpus, a `no_op_classification` is documented from the closed enum set; if AC-5's variance precondition is satisfied post-AC-4, the calibration head is applied as a canonical-value transform inside `cache_predictor.predict_cache_hit_rate(...)` with the raw pre-calibration value emitted as a diagnostic field; the residual root-cause report quantifies remaining error per category; the final sm_86 sweep emits a `Verified-live-sm_86` comparison bundle with admitted geomean rel_err measurably lower than the pre-loop baseline (0.319) OR a documented explanation of why the corpus shape prevents further improvement without expansion; an optional new synthetic microbench canary is added per AC-4 sub-slice that no-ops on the current corpus to enable empirical verification on a future cohort.

### Lower Bound (Minimum Acceptable Scope)

The follow-up loop lands the AC-1 implementation-state inventory + AC-2 bundle provenance scanner + AC-3 ablation artifact for at least the AC-4.1 slice-geometry sub-slice (the single-largest-bias source per the parent plan and Codex's analysis); for the remaining three AC-4 sub-slices, the ablation artifact either records measurable per-cell movement OR documents `no_op_classification` per cell honestly; AC-5 fails closed (`calibration_variance_precondition_failed`) if the variance precondition does not improve; the residual root-cause report at least identifies which residual category dominates on the post-AC-4 corpus; the final sm_86 sweep emits a `Verified-live-sm_86` comparison bundle, even if its admitted geomean rel_err is unchanged from the 0.319 baseline (the loop's value in that case is the empirical evidence that the AC-4 sub-slices cannot move the headline without corpus expansion — that is itself a publishable finding under the project's research framing).

### Allowed Choices

- Can use:
  - Existing `tools/hrd_profiler/observation_builder.py` walker logic (slice-geometry, sector-edge, write-policy, CTA-interleave) without re-implementation — the work is witness completion + empirical verification.
  - Existing `workloads/simbench_bw/calibration_head.py` `PlattCalibrator` / `IsotonicCalibrator` / `BetaCalibrator` classes (DEC-T9 simplicity_one_se_rule head selection).
  - Existing `tools/hrd_profiler/device_properties.json` manifest as the geometry source of truth (audit but do not silently override).
  - Existing `comparison_runner.py` headline-validity guard (extended with the new `calibration_variance_precondition_failed` typed reason if AC-5 fail-closes).
  - Existing `tools/local_labels/` NCU label capture pipeline for any new synthetic canary additions (sm_86 host).
  - Bounded-arm result_kind emission for any cell where the scalar prediction cannot be trusted (per the alternative direction in the draft; folded in as a fall-through when AC-5 fail-closes).
  - Re-using LOOCV residual computation patterns from `tools/local_labels/fit_calibration_head.py`.
- Cannot use:
  - NCU profiling on cloud or server GPUs (CLAUDE.md Rule 1 absolute).
  - Driver-root operations / nvidia-smi clock locking (Rule 2).
  - Privileged CUPTI surfaces (Rule 3).
  - Host-driver work / CUDA toolchain upgrades on this host (Rule 4) — the existing CUDA 13.2 toolchain is the binding constraint.
  - sm_120 / sm_100 / cross-arch evidence (per user's 2026-05-16 directive; defers to `Tier-2-cross-arch` follow-up loop).
  - ML-regression replacement of the Smith-Eklund LRU walker (explicitly rejected by the parent plan as a "hides bias rather than fixes it" alternative).
  - Calibration head application without the AC-5 variance precondition being satisfied or fail-closed honestly (per parent plan AC-5 ordering invariant).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The loop's central insight is that AC-4 sub-slice CODE is largely already in place; what's missing is EMPIRICAL VERIFICATION on the admitted-cell corpus + an HONEST classification of which sub-slices produce no-op deltas because the corpus shape doesn't exercise them. The four sub-slices land in dependency order with three closure shapes per round:

```
Round k (per sub-slice):
  1. Read AC-1 inventory; identify the code path + existing witness for this sub-slice
  2. Generate BEFORE bundles by re-running the current sweep with the sub-slice's flag at its old default value
  3. Generate AFTER bundles by re-running with the new flag set per device_properties.json manifest
  4. Diff the bundles into an AC-3 ablation artifact entry
  5. If headline moved → record `headline_moved: true`
     If headline did not move → diagnose corpus reason + record `no_op_classification`
  6. Update AC-2 bundle provenance scanner if any new field-presence assertion is needed
  7. Commit + round summary cites the ablation entry
```

After all four sub-slices land, the AC-5 apply round:

```
  1. Re-emit `variance_trajectory.json` on the post-AC-4 admitted-cell predictions
  2. If current_stddev >= 0.10 → fit + apply the calibration head; emit `raw_predicted_hit_rate` + transform `predicted_hit_rate`
  3. If current_stddev < 0.10 → fail closed with `calibration_variance_precondition_failed`
  4. Either way, emit a comparison bundle that records the decision in `headline_validity_failed_reasons`
```

The residual root-cause report (AC-7) is the final consolidation round; it cites each AC-3 ablation entry and quantifies what fraction of the residual 0.319 baseline error remains attributable to each named bucket.

### Relevant References

- `tools/hrd_profiler/observation_builder.py` — `_walk_launch` per-slice walker + `_round_robin_merge_tb_accesses` + sector-edge expansion loop
- `tools/hrd_profiler/device_properties.json` — sm_86 num_l2_slices=6, slice_hash_model
- `tools/hrd_profiler/test_observation_builder.py` — existing AC-4 witnesses (`SectorEdgeAccountingTest`, `CtaInterleaveAggregationTest`, `test_slice_overflow_decreases_hit_rate`)
- `workloads/simbench_bw/cache_predictor.py` — `predict_cache_hit_rate` per-launch aggregation + bundle emission
- `workloads/simbench_bw/calibration_head.py` — `PlattCalibrator` / `IsotonicCalibrator` / `BetaCalibrator` classes
- `tools/local_labels/fit_calibration_head.py` — LOOCV residual computation; train-on-all + shadow LOOCV per DEC-4 of the parent plan
- `data/local_labels/calibration_head_v1.json` + `calibration_evaluation_v1.json` — existing fit artifacts (Platt LOOCV p90 = 0.3015)
- `data/local_labels/audit_evidence/ac5_variance_precondition_diagnostic/variance_trajectory.json` — current_stddev = 0.0217 (below 0.10 floor) — the load-bearing diagnostic for AC-6
- `tools/local_labels/run_headline_cohort_predict_compare.py` — orchestrator that threads all 4 AC-4 parameters from device_properties to cache_predictor
- `workloads/simbench_bw/comparison_runner.py` — consumer-side headline-validity guard + bundled-arm intake
- `workloads/simbench_bw/schemas/comparison_bundle.schema.json` — bundle schema with `headline_validity_failed_reasons` enum + bounded-arm support
- `data/local_labels/sweeps/sweep_20260514T165947Z_a1closure_ebe60dd/comparison_bundle.json` — current baseline (admitted geomean rel_err 0.319)
- `plans/cache_predictor_l2_trustworthy_baseline_plan.md` — parent plan with AC-4 + AC-5 spec; DEC-4 train-on-all decision; DEC-T5 LOOCV p90 trust threshold 0.30; DEC-T9 simplicity_one_se_rule head selection

## Dependencies and Sequence

### Milestones

1. Milestone A: Inventory + scanner foundation (parallelizable)
   - Phase A1: AC-1 implementation-state inventory artifact (audits code paths + existing witnesses)
   - Phase A2: AC-2 bundle provenance scanner (enforces non-default geometry/interleave/second-layer fields)
   - Phase A3: AC-5 geometry sanity audit (manifest vs device + published partition data)
2. Milestone B: Per-sub-slice empirical closure (dependency order: B1 → B2 → B3 → B4)
   - Phase B1: AC-4.1 slice-geometry empirical closure + AC-3 ablation entry
   - Phase B2: AC-4.2 sector-edge formalization + AC-3 ablation entry
   - Phase B3: AC-4.3 write-policy empirical closure (with fresh-trace gate) + AC-3 ablation entry
   - Phase B4: AC-4.4 CTA-interleave empirical closure + AC-3 ablation entry
3. Milestone C: AC-5 apply-or-fail-closed adjudication
   - Phase C1: Re-emit post-AC-4 variance precondition diagnostic
   - Phase C2: Apply calibration head IF variance gate satisfied (transform `predicted_hit_rate` + emit `raw_predicted_hit_rate` diagnostic); else fail closed with `calibration_variance_precondition_failed`
4. Milestone D: Consolidation + final sweep
   - Phase D1: AC-7 residual root-cause classification report
   - Phase D2: AC-8 final sm_86 sweep with `Verified-live-sm_86` evidence + comparison bundle emission
   - Phase D3: Loop finalize round per AC-9 (simplify review + finalize summary)

Cross-milestone dependencies:
- Milestone B depends on Milestone A (need inventory + scanner before per-sub-slice closure)
- Milestone C depends on Milestone B (variance precondition is meaningful only post-AC-4)
- Milestone D depends on Milestone C (residual report classifies what remains after AC-5 decision)

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Generate the AC-4 implementation-state inventory artifact at `data/sm_86_predictor_accuracy_recovery/ac4_implementation_state_inventory.json` — audit code paths, existing witnesses, bundle fields, known no-op conditions per sub-slice | AC-1 | coding | - |
| task2 | Build the bundle provenance scanner at `tools/sm_86_predictor_accuracy_recovery/test_bundle_provenance_scanner.py` enforcing non-default geometry/interleave/second-layer fields on every sm_86 predictor bundle | AC-2 | coding | task1 |
| task3 | Generate the geometry sanity audit artifact at `data/sm_86_predictor_accuracy_recovery/geometry_sanity_audit.json` comparing manifest vs cudaDeviceProp vs published partition data | AC-5 | analyze | task1 |
| task4 | Execute AC-4.1 slice-geometry empirical closure: regenerate sm_86 bundles with `num_l2_slices=6` AND `=1`; diff into AC-3 ablation entry; document `no_op_classification` per cell if needed | AC-4.1 + AC-3 | coding | task2, task3 |
| task5 | Execute AC-4.2 sector-edge formalization: rename `SectorEdgeAccountingTest` class with explicit AC-4.2 docstring + per-method contract-cell tie; AC-3 ablation entry documents per-cell delta (expected small on activation corpus) | AC-4.2 + AC-3 | coding | task4 |
| task6 | Execute AC-4.3 write-policy empirical closure: add synthetic load-only + store-only kernel tests; verify ≥1 admitted cell trace has populated `memory_op_access_kind`; emit `legacy_trace_missing_access_kind` defect if not; AC-3 ablation entry documents per-cell delta or `no_access_kind_in_trace` no-op | AC-4.3 + AC-3 | coding | task5 |
| task7 | Execute AC-4.4 CTA-interleave empirical closure: extend `CtaInterleaveAggregationTest` with multi-CTA shared-sector witness (≥10% margin); regenerate sm_86 bundles with `cta_interleave_enabled=True` AND `=False`; diff into AC-3 ablation entry on multi-TB cells | AC-4.4 + AC-3 | coding | task6 |
| task8 | Re-emit `variance_trajectory.json` on the post-AC-4 admitted-cell predictions; record updated `current_stddev` value | AC-6 | coding | task7 |
| task9 | Conditional: IF `current_stddev >= 0.10`, fit calibration head per DEC-T9 simplicity_one_se_rule + APPLY transform inside cache_predictor (canonical `predicted_hit_rate` reflects calibrated value; `raw_predicted_hit_rate` diagnostic field added); regenerate bundles | AC-6 + AC-5 (parent plan) | coding | task8 |
| task10 | Conditional: IF `current_stddev < 0.10`, emit bundles with `headline_validity_failed_reasons` including `calibration_variance_precondition_failed` AND `result_kind="typed_unavailable"`; document the fail-closed path in the round summary | AC-6 | coding | task8 |
| task11 | Generate the residual root-cause classification report at `data/sm_86_predictor_accuracy_recovery/residual_root_cause_report.md` citing per-bucket AC-3 ablation deltas | AC-7 | analyze | task9 or task10 |
| task12 | Run the final sm_86 sweep via `tools/local_labels/run_headline_cohort_predict_compare.py` on the local RTX 3090 Ti; emit a comparison bundle with `Verified-live-sm_86` evidence status; record per-arch coverage_gate_verdict + relative_error_stats + geomean CI | AC-8 | coding | task11 |
| task13 | Finalize the loop with the simplify review + finalize summary per AC-9 (mirroring the parent loop's Round 19 closure pattern); emit final round summary citing AC-1 through AC-8 closures + the AC-9 discipline | AC-9 | coding | task12 |

Note on task4–task7 sequencing: Codex's first-pass analysis recommended dependency-order execution (AC-4.1 → 4.2 → 4.3 → 4.4) because slice geometry is the single-largest-bias source. If during execution AC-4.1's ablation entry shows the headline already moves significantly, the round may opt to short-circuit task5–task7 into a single "remaining sub-slices land together for formalization" round per DEC-1 (see Pending User Decisions) — the user must adjudicate whether to optimize for empirical headline movement or formal sub-slice closure.

## Claude-Codex Deliberation

### Agreements

- AC-4 sub-slice code is LARGELY ALREADY IMPLEMENTED in `tools/hrd_profiler/observation_builder.py` + `workloads/simbench_bw/cache_predictor.py` + the device_properties manifest infrastructure; the loop's work is empirical verification + witness formalization + variance-gate adjudication, not new physics.
- The plan's dependency-order sequence (AC-4.1 → 4.2 → 4.3 → 4.4 → AC-5) follows the parent plan's spec and Codex's identification of slice geometry as the single-largest-bias source.
- "Contract closure" (witness tests pass) is INSUFFICIENT for AC-4 sub-slice closure; an AC-3 ablation artifact with measurable per-cell prediction change (or a documented no-op classification) is required.
- Cross-architecture (sm_120, sm_100) and live AC-T5.1 SASS regeneration continue to defer per the user's 2026-05-16 directive; the follow-up loop scope is sm_86-only.
- AC-5 calibration head application MUST NOT bypass the variance precondition gate; if the post-AC-4 stddev remains below 0.10, the bundle fail-closes with `calibration_variance_precondition_failed` rather than emitting a low-trust calibrated value.
- The parent plan's AC-9 typed-fail-closed discipline (no suppression flags) extends to this loop's new typed reasons.

### Resolved Disagreements

- **Topic: Where does the calibration transform apply?**
  - Claude initial position: apply as a compare-time transform in `comparison_runner.py` to minimize producer-side schema churn.
  - Codex first-pass critique: a compare-time transform would conflate predictor identity (the bundle's `predicted_hit_rate` field) with consumer-side interpretation, harder to audit per-bundle.
  - Resolution: apply as a producer-side transform inside `cache_predictor.predict_cache_hit_rate(...)`; emit `raw_predicted_hit_rate` as a new diagnostic field so the pre-calibration value is preserved. This is recorded in AC-6 positive tests + the conditional task9 description. (The compare-time alternative is left as a folded-in option in Alt-2 of the draft if producer-side proves infeasible.)
- **Topic: How is "variance improvement" measured for AC-5 gating?**
  - Claude initial position: per-cell prediction stddev across the admitted corpus.
  - Codex first-pass critique: the parent plan's AC-9 wording is ambiguous between (a) per-cell prediction spread, (b) per-cell residual variance, (c) signed-bias magnitude reduction. The plan should pin one.
  - Resolution: pin to per-cell prediction stddev (matches `current_stddev` field in the existing `variance_trajectory.json` artifact). The 0.10 threshold is the parent plan's documented floor. Recorded in AC-6 + the threshold-cite in the Goal Description.

### Convergence Status

- Final Status: `converged` (Phase 5 ran a single convergence round in Phase 3's first-pass Codex critique; the candidate plan v1 incorporated Codex's CORE_RISKS + MISSING_REQUIREMENTS + CANDIDATE_CRITERIA verbatim; no high-impact DISAGREE items remain at the AC / Path Boundary / Task Breakdown level. Pending user decisions are scope-level, not plan-correctness-level, and are surfaced as PENDING DECs for explicit user adjudication.)

## Pending User Decisions

- DEC-1: Should this follow-up loop optimize for closing formal deferred AC-4 criteria, or for reducing the current sm_86 geomean error even if that means prioritizing corpus expansion (new microbench canaries) over more AC-4 witnesses?
  - Claude Position: optimize for formal AC-4 closure per the draft's stated scope; corpus expansion is an Alt-5 (synthetic canary) fold-in that the user can opt into mid-loop if AC-3 ablation entries show the current corpus structurally cannot exercise certain sub-slices.
  - Codex Position: optimize for sm_86 geomean reduction; corpus expansion attacks the observed "AC-4 no-op on current corpus" problem directly.
  - Tradeoff Summary: formal closure honors the existing plan's deferred work but risks producing a "contract met / headline unchanged" outcome. Geomean reduction honors the user's actual goal (predictor accuracy) but requires more live-label cost (NCU runs on sm_86 + new kernel build) and shifts the loop's scope. The right answer depends on whether the user views the loop's value as (a) closing the parent plan's deferred items as documented or (b) measurably reducing the predictor's bias.
  - Decision Status: `PENDING`

- DEC-2: Is AC-5 calibration head application allowed to land if the AC-4 sub-slices are proven wired but the post-AC-4 variance precondition (`current_stddev >= 0.10`) remains UNSATISFIED?
  - Claude Position: AC-5 MUST fail closed when the variance precondition fails. The parent plan's invariant ("AC-5 MUST NOT land if AC-4 sub-slices have not demonstrably improved per-cell prediction variance") is structural; bypassing it would produce low-trust calibrated predictions that satisfy formal AC-5 but mislead downstream consumers.
  - Codex Position: agrees with fail-closed; suggests the loop explicitly emit `calibration_variance_precondition_failed` as a new typed reason in `comparison_bundle.schema.json::headline_validity_failed_reasons.items.enum` and adopt the Alt-2 bounded-arm fall-through (emit `result_kind="bounded"` instead of `typed_unavailable` to preserve some signal).
  - Tradeoff Summary: fail-closed protects against false-precision but produces a `typed_unavailable` bundle that excludes from the headline geomean entirely. Bounded-arm fall-through preserves some signal (interval contains truth at documented coverage) at the cost of fitting a conformal interval on small N. The user must decide whether AC-5 fail-closure produces (a) typed_unavailable (more conservative) or (b) bounded-arm (more informative but more complex).
  - Decision Status: `PENDING`

- DEC-3: Should cross-launch L2 state (warm cache across kernel launches within a session) become an explicit AC in this loop, given rmsnorm (3 launches) and softmax (2 launches) have multi-launch structure, or remain deferred?
  - Claude Position: defer cross-launch state as a queued follow-up. The four AC-4 sub-slices are well-scoped and the draft's primary direction was incremental closure of those four; adding a fifth sub-slice mid-loop violates BL-20260509 bounded-recovery scope.
  - Codex Position: at minimum, the residual root-cause report (AC-7) should classify the cross-launch state residual as a named bucket so its contribution is quantifiable; whether it becomes its own sub-slice with implementation is a scope question.
  - Tradeoff Summary: explicit AC adds 1-2 rounds of work but addresses a documented contributor to the rmsnorm/softmax over-prediction. Deferral keeps loop scope bounded but leaves a known headline-residual contributor unattributed. The plan's AC-7 already includes `cross_launch_state_residual` as a named bucket per Codex's recommendation; the user's decision is whether to escalate beyond that.
  - Decision Status: `PENDING`

- DEC-4: Should the canonical headline `predicted_hit_rate` field reflect the calibrated value (post-AC-5 apply), or should the calibrated value be emitted only as a diagnostic until the LOOCV p90 clears the trust threshold (0.30)?
  - Claude Position: when AC-5 applies (variance gate satisfied), the canonical `predicted_hit_rate` SHOULD reflect the calibrated value; the pre-calibration scalar is preserved as `raw_predicted_hit_rate` diagnostic. The parent plan's DEC-4 (train-on-all) is consistent with this — calibration is the predictor's output, not a downstream diagnostic.
  - Codex Position: agrees in spirit but suggests caution given Platt LOOCV p90 = 0.3015 is marginally above the trust threshold; recommends the round summary explicitly cite which value is the canonical headline + which is the diagnostic.
  - Tradeoff Summary: canonical-calibrated produces sharper headlines but commits to the calibration at producer-time; diagnostic-only is more conservative but adds consumer-side complexity (downstream must decide whether to use raw or calibrated). The user's decision determines which field carries the canonical comparison value in `comparison_runner.py`.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers (the existing project's `tools/test_plan_token_scanner.py` enforces this; respect its exemption table).
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead. Examples: name the AC-4.1 witness `test_slice_overflow_decreases_hit_rate` (not `test_ac4_1`); name the AC-3 ablation field `headline_moved_under_slice_geometry_refinement` (not `headline_moved_ac4_1`).

### Active-scope binding constraints (carry-over from parent plan + CLAUDE.md)
- The 2026-05-16 user directive defers all cross-architecture work (sm_120, sm_100) to the `Tier-2-cross-arch` follow-up loop. The follow-up loop's scope MUST stay sm_86-only.
- CLAUDE.md Rules 1-5 are binding; the local NCU exception under `tools/local_labels/` + `data/local_labels/` applies for sm_86 on the local RTX 3090 Ti.
- The host CUDA 13.2 toolchain limit blocks AC-T5.1 live SASS regeneration; that work is OUT OF SCOPE for this loop.
- The `.humanize/rlcr/` loop artifacts are gitignored and mechanically blocked from `git add`; round contracts MUST NOT require committing them.
- Round contracts MUST follow the existing framework's discipline (failing-test-first witnesses, named slices in dependency order, post-commit recapture invariant) per AC-9.

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

# Producer-Side L2 Predictor Refinement Through AC-4 Sub-Slice Closure

## Original Idea

L2 sector hit-rate predictor systematically over-predicts on sm_86 by 22-40 percentage points (admitted geomean rel_err = 0.319, range 0.051–0.739). Root-cause analysis from deep research across cache_predictor.py + observation_builder.py + the plan + KB cards + the 5 admitted cells:

**Measured error pattern (sweep_20260514T165947Z_a1closure_ebe60dd):**
- silu_fwd correctness_shape (1 KB, 4 CTAs, single-SM fit): 5.1% rel_err ← model accurate when assumptions hold
- silu_fwd timing_shape (1 MB, 1024 CTAs): 73.9% rel_err ← worst; cross-CTA interleave + slice-geometry failure
- layernorm timing (1 MB, 64 CTAs, 1 launch): 29.7%
- rmsnorm timing (1 MB, 64 CTAs, 3 launches: reduce+sum+norm): 42.7%
- softmax timing (1 MB, 64 CTAs, 2 launches: reduce+norm): 67.8%

Predictor systematically over-predicts (94-98% predicted vs 54-89% NCU truth).

**Root causes (in descending bias order, per Codex's earlier analysis + KB cards):**

1. **Slice-geometry not modeled (AC-4.1 deferred)** — GA102 has 12 × 512 KB L2 partitions. `_walk_launch` defaults to `num_l2_slices=1` (fully-associative). Real per-slice capacity ceiling is 1/12 of total; the model gives full L2 to any working set. "Single-largest bias source per Codex" (plan line 142).

2. **Cross-CTA/SM interleaving not modeled (AC-4.4 deferred)** — sequential per-TB walk assumes perfect temporal CTA isolation. silu_fwd timing's 1,024 concurrent CTAs compete for the same L2 sectors; the sequential model inflates reuse hits.

3. **Cross-launch L2 state not carried forward** — each launch analyzed in isolation with cold L2. rmsnorm (3 launches) and softmax (2 launches) have warm L2 between kernels that the model can't see.

4. **Write-policy not classified (AC-4.3 deferred)** — all stores treated identically; write-allocate misses not distinguished.

5. **Sector-edge accounting (AC-4.2 deferred)** — sub-32B aligned-crosses double-count sector touches.

6. **No calibration head APPLIED to predictions** — Platt is fit per-cell but only used downstream as an admissibility check; the raw Smith-Eklund LRU output goes directly to comparison (no calibration transform). Platt LOOCV p90 = 0.302 (above DEC-T5 trust threshold 0.30).

7. **Maeda 2017 HRD second layer EXISTS but isn't enabled by default** — `hrd_second_layer_enabled` is a kwarg; producer may pass False, dropping the flat-L2 capacity ceiling.

**Active scope vs deferred per the active plan (cache_predictor_l2_trustworthy_baseline_plan.md):**
- AC-4.1 (slice geometry) + AC-4.2 (sector edges) + AC-4.3 (write policy) + AC-4.4 (CTA/SM interleave): all deferred Round B sub-slices
- AC-5 (calibration head application): gated behind AC-4 demonstrably improving per-cell variance
- AC-9 variance floor: stddev >= 0.10 across admitted cells (current sweep stddev unknown but the over-prediction is so uniform it may not pass)

Draft an idea for a follow-up RLCR loop (`L2-predictor-accuracy-recovery` or similar) that incrementally lands the four AC-4 sub-slices in dependency order (slice geometry → sector edges → write policy → CTA interleave) with the AC-5 calibration head applied at the end. Each sub-slice must regenerate the predictor bundles and show measurable per-cell variance improvement before the next slice lands. The follow-up loop's scope is sm_86-only (matches the current loop's active scope); cross-arch transfer defers to the existing Tier-2-cross-arch follow-up.

## Primary Direction: Producer-Side Model Refinement (user-proposed)

### Rationale

Land the 4 deferred AC-4 sub-slices in the original plan's dependency order (slice geometry → sector edges → write policy → CTA interleave) on the producer side, then apply the AC-5 calibration head. Each sub-slice regenerates predictor bundles and proves per-cell variance improvement before the next slice lands. This is the "fix the physics" angle straight from the existing plan.

### Approach Summary

**Core Design: "Formalize Existing Physics into Explicit Sub-Slice Witnesses"**

The four AC-4 sub-slices are NOT new physics implementations — they are already present in the codebase and actively used by the local-label orchestrator. The refinement work is FORMALIZATION + WITNESS COMPLETION:

1. **AC-4.1 (Slice Geometry)**: The slice-aware capacity calculation and per-slice LRU walker already live at `observation_builder.py:391-393, 602-605, 621`. The discriminating test `test_slice_overflow_decreases_hit_rate` (2164-2226) already proves the model. Work: (a) rename test to explicitly reference AC-4.1, (b) add docstring verifying it matches AC-4.1 positive-test criteria (working set exceeds slice capacity → hit rate drops >5%), (c) add formal assertion for the 50% miss-gap margin. Estimated: 10-20 LOC of marker + doc.

2. **AC-4.2 (Sector-Edge Accounting)**: The sector expansion loop at lines 545-547 already correctly expands sub-32B accesses across boundaries. The `SectorEdgeAccountingTest` class (295-487) already locks all 4 contract cells (aligned-fits, aligned-crosses, sub-32B-crosses, sub-32B-fits). Work: (a) rename class to explicitly reference AC-4.2, (b) add docstring tying each test method to its contract cell and the AC-4.2 spec, (c) verify all 4 are present. Estimated: 10-15 LOC of marker + doc.

3. **AC-4.3 (Write-Policy Classification)**: The `access_kind` enum and `_is_no_write_allocate` filter already exist (`cache_predictor.py:170-187`). Per-launch aggregation of `load_count`/`store_count`/`atomic_count` already present (`observation_builder.py:262-277`). Work: add synthetic-launch tests (a) load-only kernel asserts `load_count > 0, store_count == 0`, (b) store-only kernel asserts inverse, (c) predictor hit rate excludes no-write-allocate paths. Estimated: 70-100 LOC of new test code.

4. **AC-4.4 (CTA/SM Interleaving)**: The round-robin merge function `_round_robin_merge_tb_accesses` (301-327) and the conditional call (579-580) already exist. The `CtaInterleaveAggregationTest` class (897+) already locks sequential vs interleaved behavior. Work: extend the test class with a NEW multi-CTA shared-sector witness showing: (a) interleave_enabled=True reduces reuse distance (more hits) vs False, (b) hit-rate improvement is strictly documented (>10% margin on a synthetic overlapping-sector fixture), (c) disjoint working sets show 0% delta (no false positives). Estimated: 50-80 LOC of new test code extension.

**Bundle-Side Changes (Already Integrated)**: All 4 parameters (`num_l2_slices`, `cta_interleave_enabled`, `hrd_second_layer_enabled`, `cache_model_geometry_provenance`) already flow from device_properties manifest → orchestrator → cache_predictor → bundle at lines 233-239 of `run_headline_cohort_predict_compare.py`. No new bundle fields required.

**Plan Alignment**: AC-4 sub-slices land AFTER AC-3 (diagnostics aggregation, which is already complete). Each sub-slice must have a failing-test-first witness that proves the model improvement. AC-5 (calibration head) MUST NOT land before all 4 sub-slices demonstrably improve variance on the 13-cell sweep. The slice-geometry (AC-4.1) is the "largest-bias source per Codex" and should land first among the 4.

### Objective Evidence

- `tools/hrd_profiler/observation_builder.py` (789 LOC): Pass-1 per-TB access collection + Pass-2 per-slice sector LRU walker already implement `_walk_launch(num_l2_slices, cta_interleave_enabled, hrd_second_layer_enabled)` (lines 330-339), slice-aware sector routing via `sector_id % num_l2_slices` (line 621), per-slice ReuseDistanceTracker instances (lines 602-605), second-layer flat tracker for Maeda 2017 HRD (lines 606-608, 626-646), round-robin CTA interleave via `_round_robin_merge_tb_accesses` (lines 301-327), per-access write-policy tags (`access_kind`, `memory_op_class`) propagated to observation dict (lines 558-560, 671-672), sector-edge accounting loop for sub-32B spans (lines 545-547), per-launch diagnostics aggregation via `summarize_launch` (lines 161-278).
- `workloads/simbench_bw/cache_predictor.py` (1,076 LOC): `_LaunchAggregate` dataclass (lines 191-221) accepting `hit_count`, `miss_count`, `cold_count`, `no_write_allocate_excluded_count`; `_is_no_write_allocate` filter (lines 170-187) checking `access_kind == STORE and op_class in {SHARED, CLUSTER_DSMEM}`; bundle-level parameters `cache_model_num_l2_slices`, `cache_model_num_l2_slices_source`, `cache_model_cta_interleave_enabled`, `cache_model_hrd_second_layer_enabled`, `cache_model_geometry_provenance` (lines 549-575, 938-956).
- `tools/hrd_profiler/device_properties.py` (235 LOC): 5-tuple lookup (arch, sku, driver_version, cuda_toolkit_version, nvbit_build) → `DevicePropertiesEntry`; manifest at `tools/hrd_profiler/device_properties.json` pins `num_l2_slices=6` for sm_86 RTX 3090 Ti (note: original idea's "12 × 512 KB" was the GA102 die spec; the manifest reflects the published 6-partition logical view).
- `tools/hrd_profiler/test_observation_builder.py`: `SectorEdgeAccountingTest` (lines 295-487, 192 LOC), `CtaInterleaveAggregationTest` (lines 897-1051, 150+ LOC), `test_slice_overflow_decreases_hit_rate` (lines 2164-2226, AC-4.1 positive witness with documented >5% margin), `test_small_working_set_unchanged_hit_rate` (lines 2228-2275, AC-4.1 negative witness), `test_num_l2_slices_default_is_backward_compatible` (lines 2277-2295, backward-compat guard).
- `tools/local_labels/run_headline_cohort_predict_compare.py` (lines 225-240): orchestrator already resolves device properties via `_resolve_local_cache_model_geometry()` and threads all 4 model parameters into `cache_predictor.predict_cache_hit_rate()` with `cache_model_hrd_second_layer_enabled=True`.
- `workloads/simbench_bw/schemas/cache_predictor.schema.json` (lines 82-132): already defines bundle-level fields for all 4 parameters with full provenance documentation; `cache_model_geometry_provenance_scanner` test exists to enforce cross-checks.

### Known Risks

- **Test-Coverage Gap for AC-4.3 & AC-4.4**: Write-policy classification and CTA interleaving are fully implemented but lack synthetic-launch witnesses demonstrating their correctness. If the witnesses reveal bugs in the aggregation logic (e.g., kind_counts not being accumulated correctly, or round-robin merge producing incorrect per-access ordering), the implementation path requires fixes in `observation_builder.py`'s hot path before witnesses can pass. Mitigation: start with AC-4.3 witness to probe for aggregation bugs early.
- **Variance Improvement Contingency**: The plan assumes AC-4 sub-slices will "demonstrably improve per-cell variance" per AC-5's negative test. If on-device empirical testing shows variance improvement is <5% or non-monotone across sub-slices, the calibration head (AC-5) must be deferred to a follow-up loop. Material risk on sm_86 where the manifest's slice count (6) is moderate and cross-CTA contention may be limited.
- **Device-Specific Geometry Coupling**: The sm_86 entry in `device_properties.json` is pinned to `num_l2_slices=6`. If the manifest entry is wrong (e.g., actual hardware partitions differently, given the original idea's "12 × 512 KB" claim), all AC-4 witnesses will pass but on-device prediction accuracy will remain poor. Mitigation: the "empirical variance improvement" gate in AC-5 will catch this if geometry is fundamentally mis-specified; first round should audit manifest against `cudaDeviceProp` / `nvidia-smi --query-gpu=l2_cache_size` output.
- **Backward-Compat on Legacy Traces**: The `hrd_second_layer_enabled=True` flag in `run_headline_cohort_predict_compare.py` applies to ALL 13 cells. If any legacy traces were emitted before the second-layer infrastructure existed, they may have stale sector-stream sidecars that don't account for the dual-gate logic. Mitigation: the schema enforces bundle-level semantics; mismatches are caught at join time.

## Alternative Directions Considered

### Alt-1: Calibration-Head-First Transform
- Gist: Skip the expensive AC-4 producer-side model refinements; instead, immediately apply the already-fit Platt calibration head (`A=1.3559, B=-0.3725` from `calibration_head_v1.json`) as a monotone post-hoc transform on raw Smith-Eklund LRU output in `comparison_runner.py`'s scalar-path join (line 2011–2012). The Platt LOOCV p90 = 0.3015 is marginally above the 0.30 trust threshold but the +23.5pp absolute improvement on p90 residuals (uncalibrated 0.394 → calibrated 0.302) is the cheapest available accuracy lift. Implementation surface: ~80-120 LOC for transform integration + new diagnostic field `calibrated_predicted_hit_rate`.
- Objective Evidence:
  - `workloads/simbench_bw/calibration_head.py` (420 LOC): three calibrator classes with unified `fit(...) → transform(...) → (calibrated, out_of_envelope_mask)` surface.
  - `data/local_labels/calibration_head_v1.json` + `calibration_evaluation_v1.json`: trained Platt coefficients + LOOCV table (Platt p90=0.3015, Isotonic p90=0.282, Beta p90=0.448).
  - `workloads/simbench_bw/comparison_runner.py` lines 2011-2024 (scalar-path join — insertion point identified).
  - Plan AC-5 ordering constraint (lines 59-65) is commit-time, not evaluation-time; transform application before AC-4 commits is plausibly within plan spirit if AC-5 commit itself defers.
- Why not primary: The plan explicitly forbids landing the AC-5 calibration head before AC-4 sub-slices demonstrate variance improvement ("MUST NOT land if AC-4 sub-slices have not demonstrably improved per-cell prediction variance"); applying calibration before AC-4 commits risks hiding the underlying physics gap and may over-correct after AC-4 lands, plus the envelope (0.9375–0.9771, only 0.039 range) is tight enough that any post-AC-4 prediction shift would route cells to `typed_unavailable` via `calibration_reuse_distance_out_of_envelope`.

### Alt-2: Bounded-Arm Uncertainty Emission
- Gist: Instead of trying to make the scalar prediction more accurate, accept the physics gap as known model uncertainty and emit `result_kind="bounded"` bundles (`hit_rate_lower`, `hit_rate_upper`) whose interval CONTAINS the NCU truth at a documented coverage rate (e.g., 90-95%). The schema, comparison_runner intake, per-arch bounded-arm statistics, and midpoint-canonical relative-error logic all already exist; the only new work is a `ConformalCalibrator` wrapper (~50-80 LOC) that fits isotonic quantile regression on the existing (predicted, label) pairs + the cache_predictor emission extension (~80-120 LOC). NO schema changes required.
- Objective Evidence:
  - `workloads/simbench_bw/schemas/cache_predictor.schema.json` lines 160-205: `result_kind` enum admits `"bounded"`; `hit_rate_lower`/`hit_rate_upper` required when bounded; constraint `hit_rate_lower <= hit_rate_upper`.
  - `workloads/simbench_bw/schemas/comparison_bundle.schema.json` lines 92-119, 453-510: per-cell bounded conditional + per-arch `bounded_arm_relative_error_stats` block with lower/upper arm stats already required.
  - `workloads/simbench_bw/comparison_runner.py` lines 1184-1241 (bounded intake), 1359-1375 (midpoint canonical), 1592-1633 (`_bounded_arm_relative_error_stats`) already fully implemented.
  - `workloads/simbench_bw/cache_predictor.py` lines 295-329: `_bootstrap_ci_over_launches()` machinery already in use for scalar CI; same machinery can produce conformal quantiles.
- Why not primary: Bounded arm trades sharper headline predictions for calibrated coverage — it satisfies AC-9 typed-fail-closed discipline by being honest about uncertainty but does not actually improve the underlying model; the primary direction is the canonical "fix the physics" path the plan explicitly prescribes, and the bounded arm is better folded in AFTER AC-4 has landed (as a wider envelope for the residual model uncertainty AC-4 cannot eliminate).

### Alt-3: Workload-Class Stratification
- Gist: The existing schema already carries a `predictor_design_coverage ∈ {predictor_friendly, predictor_hostile}` field with 8 friendly + 5 hostile cells (`_PREDICTOR_DESIGN_COVERAGE_BY_CELL` constant in `comparison_runner.py:547-592`). Don't fix the model uniformly; branch on the bucket: leave Smith-Eklund LRU for friendly cells, deploy fail-closed `typed_unavailable` (typed_reason=`predictor_model_insufficient_for_bucket_coverage`) or wide bounded-interval for hostile cells. Per-bucket stat aggregation already exists; the comparison runner naturally excludes typed_unavailable from headline via existing `typed_unavailable_handling="exclude"` contract pin. Implementation surface: cell-level bucket lookup in `cache_predictor.predict_cache_hit_rate()` + conditional result_kind branch (~50 LOC).
- Objective Evidence:
  - `workloads/simbench_bw/comparison_runner.py:547-592`: `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` constant + `tests/test_predictor_design_coverage_partition.py` enforce lockstep with the headline cohort.
  - `workloads/simbench_bw/comparison_runner.py:1560-1589`: `_per_bucket_relative_error_stats()` aggregator + schema-required `relative_error_stats_per_predictor_design_coverage` block.
  - Plan AC-9 lines 104-117: per-bucket framing already used for `insufficient_microbench_coverage` typed-fail-closed gate.
- Why not primary: Stratification REPORTS the model's known scope without FIXING it — it changes the headline-presentation framing (friendly geomean as canonical, hostile as diagnostic) but doesn't reduce per-cell error on the hostile cells the user actually wants more accurate predictions for. Better folded in AFTER AC-4 to characterize residual hostility.

### Alt-4: ML Regression Head Replacing LRU Walker
- Gist: Abandon the per-launch Smith-Eklund LRU walk entirely; fit a `RandomForestRegressor` or `GradientBoostingRegressor` directly on per-cell features (working_set_bytes, num_ctas, num_launches, threads_per_cta, arch_token, mean_reuse_distance) → NCU_hit_rate using the 13 label cells. Cache model becomes a feature input, not the predictor. New module `workloads/simbench_bw/ml_regression_head.py` (~80-120 LOC) parallel to existing `calibration_head.py`; schema `algorithm_version` enum extension to `"ml_regression_v1"`.
- Objective Evidence:
  - `tools/local_labels/run_headline_cohort_predict_compare.py:126-140`: 13-cell cohort definition.
  - `requirements.txt`: `scikit-learn==1.8.0` pinned per DEC-T8.
  - `tools/local_labels/fit_calibration_head.py:201-228`: `_loocv_residuals()` + `_bootstrap_p90_se()` reusable for ML head.
  - `workloads/simbench_bw/calibration_head.py`: direct precedent for `fit(...) → transform(...) → (output, mask)` interface.
- Why not primary: The plan explicitly rejects ML-replaces-physics in its alternatives discussion (lines 517-529) on the grounds that it "hides the underlying LRU-walker bias rather than fixing it"; 13-cell LOOCV is a degenerate regime where p90 ceiling is noise-dominated; cross-arch generalization (sm_120 in the future Tier-2-cross-arch loop) would not transfer; severe overfitting risk on 13 training cells.

### Alt-5: Synthetic Canary Instrumentation
- Gist: Before any model refinement, add a set of 4 synthetic micro-kernels under `workloads/simbench_bw/l1/<canary_slug>/` designed to ISOLATE each bias source (one slice-geometry canary, one CTA-interleave canary, one cross-launch canary, one write-allocate canary). Run them through the current predictor + NCU and measure the per-cause bias. Use those measurements to PRIORITIZE which AC-4 sub-slice yields the largest accuracy improvement BEFORE committing to the AC-4.1→4.4 sequence. Cost: ~2 hours wall time on local hardware (~150-300 LOC per canary + label capture + bundle regeneration).
- Objective Evidence:
  - `tools/hrd_profiler/test_walk_launch_second_hrd_layer.py` (338 LOC): proven failing-test-first synthetic-witness pattern for slice-geometry.
  - `workloads/simbench_bw/l1/` (8 existing microbench slugs): standardized per-slug template (`main.cu` + `<slug>.cu` + `Makefile` + binary `<slug>_run`) with `-gencode arch=compute_86,code=sm_86` build target.
  - `tools/local_labels/label_driver.sh` (804 LOC) + `parse_ncu_metrics.py` (460 LOC): NCU label-capture pipeline tested in `test_label_driver_integration.py` (1446 LOC).
  - `tools/hrd_profiler/test_observation_builder.py` per-bias-source test classes (SectorEdgeAccountingTest, WritePolicyClassificationTest, CtaInterleaveAggregationTest, SliceAwareCapacityTest) all provide reference fixtures.
- Why not primary: Canary instrumentation does not FIX the predictor — it produces diagnostic priors for prioritizing the AC-4 sub-slice order. The primary direction already commits to landing all 4 sub-slices in the plan-prescribed dependency order (slice geometry first per "single-largest bias source per Codex"), so the canary's prioritization value is largely pre-decided; canaries are better folded in as failing-test fixtures inside each AC-4.x sub-slice round rather than as a standalone preflight loop.

## Synthesis Notes

The primary direction's discovery that all 4 AC-4 sub-slices are LARGELY ALREADY IMPLEMENTED dramatically narrows the follow-up loop's true scope — the work is witness formalization (AC-4.1 + AC-4.2: ~25-35 LOC of doc/markers) + new synthetic-launch tests (AC-4.3: ~70-100 LOC; AC-4.4: ~50-80 LOC), not new physics. This makes the loop's expected duration much shorter than the plan's "4 sub-slices × 1 round each" naïve estimate; closer to 2 rounds (one for AC-4.1+4.2 formalization + AC-4.3 witnesses, one for AC-4.4 + AC-5 calibration apply), plus 1 round for cohort regeneration + variance-improvement evidence. If during execution the variance improvement gate for AC-5 is missed, **Alt-2 (bounded-arm uncertainty)** is the natural fold-in: emit a wider conformal interval as the AC-5 deliverable instead of a scalar calibrated prediction; the schema, comparison runner, and per-arch stats already exist. **Alt-3 (workload-class stratification)** is the natural AC-9 presentation fold-in: report per-bucket geomeans separately so the headline reflects the friendly-cell accuracy without hiding the hostile-cell residual. **Alt-5 (synthetic canaries)** should be folded into each AC-4.x sub-slice's failing-test-first witness rather than run as a standalone preflight. **Alt-1 (calibration-first)** and **Alt-4 (ML regression)** are both explicitly contradicted by the plan and should remain rejected unless the variance-improvement gate proves AC-4 cannot close the gap (in which case Alt-1 becomes the next escalation, with Alt-4 deferred to a much larger cross-arch cohort).

--- Original Design Draft End ---
