# sm_86 Submission-Grade Evidence Expansion Plan

## Goal Description

Move the current sm_86 non-root GPU predictor evidence from a low-coverage prototype state to a submission-grade evidence package for wall-clock timing, L1/TEX hit-rate, and L2 hit-rate prediction against NCU ground truth.

The starting point is intentionally treated as insufficient for a top-tier conference claim: timing has a strong 13-cell result but weak generalization evidence; L1 has only 6 scalar headline cells with 14 typed-unavailable rows; L2 has a good headline score but fails robustness gates (`loocv_holdout_relative_gap=0.441 > 0.2`, `loso_max_relative_change=0.814 > 0.25`). This plan therefore prioritizes benchmark/cell coverage, split discipline, label repeatability, typed-unavailable budgets, baselines, ablations, and claim discipline before any stronger claim is made.

A dedicated evaluatable cell means one concrete sm_86 `(slug, shape_point, cache_layer)` with:

- an existing workload directory and built binary;
- predictor trace sidecars under the RTX 3090 Ti / sm_86 result tree;
- a staged and schema-valid measurement manifest;
- provenance audit pass;
- NCU labels captured only through `tools/local_labels/` into `data/local_labels/`;
- repeatability admission under the existing thresholds;
- predictor bundle readiness with no publication-slice bundle failure;
- inclusion in the final evaluator ledger and split manifest.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. The `AC-*` items are current RLCR completion gates for this implementation loop.

- AC-1: Current-evidence gap audit and claim discipline are explicit.
  - Positive Tests (expected to PASS):
    - A committed gap-audit artifact records the current wall-clock, L1, and L2 metrics from the existing artifacts and marks them `not_submission_grade`.
    - The audit records timing as 13 cells, L1 as 6 scalar headline cells plus 14 typed-unavailable rows, and L2 robustness as failing the existing LOOCV/holdout and LOSO thresholds.
    - The submission claim map states that no wall-clock, L1, or L2 claim is publication-grade until its own coverage, split, accuracy, and robustness gates pass.
  - Negative Tests (expected to FAIL):
    - Any report that calls the current 13-cell timing result, 6-cell L1 result, or robustness-failing L2 result publication-grade is rejected.
    - Any report that hides typed-unavailable counts or bundle failures behind scalar-only accuracy is rejected.

- AC-2: Expansion roster covers enough benchmark breadth and replacement capacity before live NCU runs.
  - Positive Tests:
    - A schema-valid sm_86 expansion roster contains at least 42 primary candidate cells, at least 35 unique slugs, and at least 10 replacement cells.
    - The roster uses existing workload roots first: `workloads/portable_lcd/`, `workloads/simbench_bw/l1/`, and sm_86-compatible rows under `workloads/simbench_bw/shapes/`.
    - The roster satisfies family minima before live collection: DL primitives >= 4 unique slugs, GEMM >= 4, attention/conv >= 2, memory-stress >= 3, irregular-hostile >= 4, vendor-baseline >= 1.
    - At least 6 slugs have at least 3 planned shape points each.
  - Negative Tests:
    - A roster below 42 primary cells, below 35 unique slugs, missing any required family bucket, or lacking replacement cells is rejected.
    - A roster that depends on unsupported sm_86 kernels such as WGMMA/TMA/cluster-only/NVFP4-only paths is rejected unless it carries a typed exclusion and replacement.

- AC-3: Every candidate is preflighted into a trace/manifest/provenance-complete live-run unit.
  - Positive Tests:
    - Each live candidate has binary path, binary SHA-256, source commit, kernel signature, arch, shape point, NCU metric version, trace refs, and measurement manifest before NCU runs.
    - `stage_measurement_manifests_from_predictor_trace.py`, `validate_measurement_manifest.py`, and `provenance_audit.py` pass for every live candidate.
    - NCU metric availability is probed and recorded before the live sweep starts.
  - Negative Tests:
    - Missing trace sidecars, binary hash drift, kernel-signature drift, unvalidated manifests, or provenance-audit failures block the live candidate.
    - NCU invocation outside `tools/local_labels/` or writes outside `data/local_labels/` are rejected.

- AC-4: Live NCU admission produces a publication-sized evaluator cohort without loosening repeatability.
  - Positive Tests:
    - Live sweeps use the existing local-label path with cache controls `all` and `none`, warmup counts including `0`, `1`, and `5`, repeats >= 5, and the L1/L2 sector-hit repeatability threshold fixed at stddev <= 0.03.
    - The final evaluator ledger has at least 31 admitted cells and at least 30 unique slugs; the practical target is 38-42 evaluator-ready cells and 32-35 unique slugs.
    - Every publication-slice cell carries NCU labels for timing cycles, NCU ns, L1/TEX sector hit rate, and L2 sector hit rate, or a typed exclusion outside the publication slice.
    - The final publication slice has zero `bundle_failed`, zero `cache_model_underconfigured`, zero `canonical_pre_calibration_features_absent`, and zero timing binary SHA mismatch.
  - Negative Tests:
    - A publication slice below 31 admitted cells or 30 unique slugs is rejected.
    - Any repeatability threshold widening, silent drop, partial untyped label, or publication-slice bundle failure is rejected.

- AC-5: Split-by-slug calibration, validation, and sealed holdout are enforced.
  - Positive Tests:
    - A schema-valid split artifact assigns cells by slug, not by individual cell leakage, with calibration >= 15 cells, validation >= 8 cells, and holdout >= 8 cells.
    - No slug appears in more than one split, and no slug contributes more than one third of any split.
    - The holdout report is generated once after validation gates pass; its artifact records the model/source commit used for the one-shot evaluation.
  - Negative Tests:
    - Same-slug leakage across calibration/validation/holdout is rejected.
    - Re-running holdout after a model or threshold change without invalidating the previous holdout report is rejected.

- AC-6: Accuracy and robustness gates pass per metric, with explicit L1 promotion rules.
  - Positive Tests:
    - Timing cycles: log-space geomean relative error <= 0.15, median absolute log-relative error <= 0.10, bootstrap upper <= 0.25, leave-one-slug-out geomean <= 0.15, and no family median absolute log error > 0.15.
    - Deploy ns crosswalk: CUPTI-vs-NCU and CUDA-event-vs-NCU deltas are reported for every timing cell; the publication-tier gate requires median relative delta <= 5%, max relative delta <= 10%, repeat count >= 5, and dual-source pass rate >= 0.9.
    - L2: full-cohort geomean relative error <= 0.08, holdout geomean <= 0.12, per-bucket p90 absolute error <= 0.15, LOOCV/holdout relative gap <= 0.20, and LOSO max relative change <= 0.25.
    - L1: a load-bearing L1 claim is allowed only if scalar coverage reaches at least 70% of publication-slice cells, holdout scalar count >= 5, geomean relative error <= 0.10, median absolute error <= 0.06, and `predictor_l1_sass_provenance_missing` <= 10%. Otherwise L1 is emitted as a diagnostic/appendix result with a typed non-promotion reason.
  - Negative Tests:
    - Any metric report that passes scalar accuracy while failing its coverage, split, or robustness gate is rejected.
    - A load-bearing L1 claim below the scalar coverage or holdout floor is rejected.
    - A timing report that relies only on cell-level LOOCV while failing group-heldout validation is rejected.

- AC-7: Baselines, ablations, and statistical reporting are publication-grade.
  - Positive Tests:
    - Baselines run on the exact same holdout cells: Nugteren/RDH where applicable, always-hit, cohort mean, uncalibrated walker, previous v4/Platt baseline, v5 head, and timing prior-only/residual variants.
    - Reports include geomean relative error, MAPE, MdAPE, median absolute error, signed residual bias, Pearson, Spearman, per-family breakdowns, scalar/typed-unavailable counts, paired deltas, and 95% confidence intervals.
    - At minimum, ablations cover no calibration head, no gridDim/CTA-wave feature, no high-footprint correction, no reduction correction, no warmup/cache-control tuning, timing prior-only, timing residual-only, and leave-one-family-out.
  - Negative Tests:
    - Baselines on a different cell set than the holdout are rejected.
    - Reports without typed-unavailable counts, confidence intervals, or paired baseline deltas are rejected.

- AC-8: Typed-unavailable budgets are first-class gates rather than footnotes.
  - Positive Tests:
    - Publication reports include per-metric typed-unavailable counts and typed reasons.
    - L2 publication slice has zero predictor-bundle absence and zero untyped comparison failures.
    - Timing publication slice has zero binary SHA mismatch and zero untyped repeatability failure.
    - L1 emits either a promoted result that satisfies AC-6 or a typed non-promotion artifact naming the missing provenance/coverage reason.
  - Negative Tests:
    - Any scalar-only report that omits coverage denominator or typed failures is rejected.
    - Any untyped exclusion in the publication slice is rejected.

- AC-9: Final evidence bundle is reproducible, scoped, and compliant.
  - Positive Tests:
    - The final evidence bundle includes the roster, candidate preflight, raw label sidecar inventory, admitted/excluded ledgers, split artifact, predictor bundles, accuracy reports, baseline reports, ablation reports, SHA-256 manifest, and claim-to-evidence map.
    - Frozen sm_120 and frozen L2 invariants either pass unchanged or any intentional change is explicitly outside this loop.
    - Static guards pass: no NCU outside quarantine, no privileged CUPTI/event leakage, no clock-lock/power/MIG/driver mutation, no plan-token leakage in implementation code.
  - Negative Tests:
    - Missing reproducibility artifacts, hidden local state, cross-arch claims without matching evidence, or mutation of frozen invariants fails closure.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The loop delivers a publication-grade sm_86 evidence package: at least 38 evaluator-ready cells over at least 32 unique slugs; wall-clock timing, L2, and promoted L1 all pass their coverage, split, holdout, robustness, baseline, ablation, and typed-unavailable gates; the evidence bundle is reproducible and compliant; cross-arch remains explicitly out of claim unless separately proven.

### Lower Bound (Minimum Acceptable Scope)

The loop must at least deliver the roster, preflight, live NCU admission, split-by-slug, L2 publication-grade holdout report, timing group-heldout report, and L1 diagnostic/non-promotion artifact. If L1 cannot reach promotion coverage, the loop must not pretend it did; it must emit a typed non-promotion result and keep L1 out of the load-bearing claim.

### Allowed Choices

- Can use: existing portable LCD kernels, existing simbench L1/microbench kernels, sm_86-compatible shape files, additive shape points, local NCU capture under `tools/local_labels/`, existing comparison runners and publication gates, new schema-valid audit/report artifacts, read-only NVML/CUPTI Activity, and regularized deploy-legal model features.
- Cannot use: NCU outside the quarantine path, privileged CUPTI metrics/events as deployable inputs, clock/power/persistence/MIG mutation for the submitted measurements, simulator revival, achieved NCU counters as predictor inputs, silent threshold widening, hidden holdout reuse, or scalar-only reporting without coverage denominators.

## Feasibility Hints and Suggestions

### Conceptual Approach

Start by making the evidence denominator explicit. Build a roster that intentionally overbooks the paper target, preflight every candidate, and run live NCU only after the traces/manifests/provenance are complete. Recover existing admitted-but-not-evaluator-ready cells first, then add breadth from active portable LCD kernels and sm_86-compatible simbench rows. Keep noisy memory-stress and irregular-hostile cells in the roster because they are essential for credibility, but carry replacements because they are likely to fail repeatability.

### Relevant References

- `workloads/simbench_bw/comparison_contract_sm86_submission.toml` - submission target and sm_86 scope.
- `tools/sm_86_predictor_accuracy_recovery_v2/publication_gate.py` - family, depth, cell-count, and split-balance gates.
- `data/sm_86_predictor_accuracy_recovery_v2/expansion_candidates.json` - current trace-present candidate inventory.
- `workloads/portable_lcd/roster.toml` - active LCD-portable kernel roster.
- `tools/local_labels/stage_measurement_manifests_from_predictor_trace.py` - manifest staging from predictor traces.
- `tools/local_labels/provenance_audit.py` - pre-live provenance audit.
- `tools/local_labels/sweep_repeatability_matrix.py` - live NCU repeatability matrix driver.
- `workloads/simbench_bw/timing_cycle_predictor.py` - current timing predictor surface and deployability guards.
- `tools/local_labels/fit_monotonic_calibration_head_v5.py` - L2 calibration head fitting.
- `tools/local_labels/fit_l1_calibration_head.py` and `tools/local_labels/emit_l1_closure_artifacts.py` - L1 fitting and closure path.
- `data/local_labels/calibration_head_v5_robustness_sm_86.json` - current robustness failure surface.

## Dependencies and Sequence

### Milestones

1. Evidence baseline and roster design:
   - Emit current-gap audit and claim map.
   - Generate the expanded candidate roster and replacement pool.
   - Validate family, depth, and compatibility before live collection.
2. Preflight and live admission:
   - Stage manifests and run provenance audit.
   - Run live NCU repeatability sweeps.
   - Regenerate predictor bundles and evaluator ledger.
3. Split and locked modeling:
   - Create slug-disjoint calibration/validation/holdout.
   - Fit or adjust timing/L1/L2 predictors using calibration and validation only.
   - Lock thresholds before holdout.
4. Publication-grade reporting:
   - Run holdout once.
   - Emit accuracy, robustness, baseline, ablation, typed-unavailable, and claim-to-evidence reports.
   - Run compliance and frozen-invariant guards.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

Every `AC-*` must be covered by at least one task. Every task must target at least one `AC-*`. Do not target `FUT-*`, `DEC-*`, or `-` in the Target AC column.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Emit current evidence gap audit and claim-to-evidence map skeleton | AC-1 | coding | - |
| task2 | Generate and validate expanded sm_86 candidate roster with family/depth/replacement gates | AC-2 | coding | task1 |
| task3 | Ask Codex to review roster for unsupported sm_86 paths, family gaps, and likely NCU-repeatability risks | AC-2 | analyze | task2 |
| task4 | Implement or repair candidate preflight: manifest staging, provenance audit aggregation, metric-availability report | AC-3 | coding | task2 |
| task5 | Recover admitted-but-not-evaluator-ready cells and regenerate bundle readiness reports | AC-4, AC-8 | coding | task4 |
| task6 | Run live NCU repeatability sweeps through local-label tooling and emit admitted/excluded evaluator ledgers | AC-4, AC-8 | coding | task4 |
| task7 | Generate slug-disjoint calibration/validation/holdout split and sealed-holdout guard | AC-5 | coding | task6 |
| task8 | Add timing group-heldout/LOSO validation and GEMM residual diagnostics | AC-6 | coding | task7 |
| task9 | Refit or repair L2 reporting under holdout, LOOCV/holdout, LOSO, and per-bucket gates | AC-6 | coding | task7 |
| task10 | Repair L1 scalar coverage or emit typed L1 non-promotion artifact | AC-6, AC-8 | coding | task7 |
| task11 | Add baseline and ablation runners/reports over the exact same holdout cell set | AC-7 | coding | task8, task9, task10 |
| task12 | Ask Codex for independent review of statistical claims, baseline fairness, and holdout leakage | AC-5, AC-7 | analyze | task11 |
| task13 | Emit final evidence bundle, SHA-256 manifest, and compliance/frozen-invariant verification | AC-9 | coding | task11, task12 |

## Future Work / Out of Scope

- FUT-1: Cross-architecture sm_120 promotion.
  - Source DEC: DEC-1
  - Current-loop handoff: AC-9
  - Promotion trigger: sm_86 evidence bundle passes and the same protocol can be reproduced on sm_120.
- FUT-2: A new benchmark suite from scratch.
  - Source DEC: DEC-2
  - Current-loop handoff: AC-2
  - Promotion trigger: existing portable LCD and simbench candidates cannot reach publication-grade coverage.
- FUT-3: Top-tier paper writing and figure generation.
  - Source DEC: DEC-3
  - Current-loop handoff: AC-9
  - Promotion trigger: final evidence bundle passes with publication-grade status.

## Claude-Codex Deliberation

### Agreements

- Current results are not top-tier-submission ready.
- Coverage expansion and typed-unavailable budgets are higher priority than model tuning alone.
- The plan should target above the nominal 31-cell / 30-slug gate to survive live NCU attrition.
- Holdout must be slug-disjoint and touched once.
- L1 must not be load-bearing until scalar coverage and holdout gates pass.

### Resolved Disagreements

- Scope of the main claim: one reviewer preferred L2-only. The chosen resolution keeps wall-clock, L1, and L2 in the plan, but applies metric-specific promotion gates. L1 may be demoted to diagnostic if coverage remains insufficient.
- Cross-arch timing: cross-arch is deferred because sm_86 is not yet publication-grade.

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- DEC-1: Cross-arch inclusion in the immediate loop.
  - Claude Position: Defer until sm_86 passes this protocol.
  - Codex Position: Defer; current sm_120 evidence should remain exploratory.
  - Tradeoff Summary: Cross-arch is valuable but multiplies collection and interpretation risk.
  - Decision Status: `defer_cross_arch`
- DEC-2: Whether to build a new benchmark suite from scratch.
  - Claude Position: Use existing workload roots first.
  - Codex Position: Use existing portable LCD and simbench candidates first.
  - Tradeoff Summary: Existing assets are already integrated with labels, traces, and publication gates.
  - Decision Status: `use_existing_workloads_first`
- DEC-3: Whether a non-promoted L1 result blocks the entire loop.
  - Claude Position: No; it blocks only a load-bearing L1 claim if typed non-promotion is explicit.
  - Codex Position: Same.
  - Tradeoff Summary: The loop should not fake L1 coverage, but it can still deliver timing and L2 evidence.
  - Decision Status: `l1_claim_requires_promotion_gate`

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.
