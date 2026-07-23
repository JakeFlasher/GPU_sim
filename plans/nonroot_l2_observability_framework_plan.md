# Non-Root GPU Cache Observability Framework with L2 Prediction (sm_86 Submission Track)

## Goal Description

Deliver a sm_86-only submission track for top-tier computer-architecture
venues (ASPLOS / HPCA / ISCA / MICRO target umbrella) for a **non-root GPU
L2 cache observability framework with calibrated L2 hit-rate prediction
and L1 as appendix-only diagnostics**. The central paper claim is
re-scoped from "L1+L2 cache hit-rate predictor" (the prior R48 framing)
to a defensible L2-only-with-L1-diagnostics framing that aligns with
`workloads/simbench_bw/comparison_contract.toml::scope.cache_layers = ["L2"]`
and demotes the L1 hostile-cohort catastrophe (93%-100% rel_err on
carveout / working_set / l2_thrash / register_spill) from a headline
claim to an appendix diagnostic surface.

The plan exercises the existing infrastructure (~30 microbenches under
`workloads/simbench_bw/l1/`, ~20 production-flavored kernels under
`workloads/portable_lcd/`) to lift the admitted cohort from the
smoke-test floor (5 cells per arch; current state at HEAD `85bef166` has
9 admitted cells in a 13-cell headline runner cohort) to the
publication-grade floor (10 cells per arch per
`comparison_contract.toml::coverage_gate.publication_grade_floor_per_arch`)
to the submission-target gate (>= 30 unique slugs across >= 31 cells;
matches the top-tier peer-review breadth bar). A locked
calibration / validation / holdout protocol with split-by-slug
membership, balancing invariants, and operational blinding rules
prevents calibration leakage. Head-to-head accuracy comparisons against
Nugteren 2014 reuse-distance L2 model, an uncorrected-walker baseline,
and two trivial baselines (mean / last-level) are emitted alongside
three headline ablations (disable L2 high-footprint correction; disable
L2 reduction correction; collapse L1 subfamily dispatch as appendix-only).

Explicitly out of scope per user direction at round-30 entry:
cross-architecture validation on sm_120 (RTX 5060 laptop unavailable);
NCU on cloud / server GPUs (CLAUDE.md Rule 1); GPU-driver-root operations
(Rule 2); privileged CUPTI surfaces (Rule 3); host-driver work (Rule 4);
simulator revival (Rule 5); ALGORITHM_VERSION bump (a future plan can
add an L1 model upgrade as a separate effort); real-application
wall-clock validation. The submission claim is explicitly scoped to
sm_86 Ampere GA102 on a single RTX 3090 Ti host; the paper acknowledges
this scope in the abstract and frames sm_120 cross-arch verification as
a documented follow-up.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative
tests for deterministic verification. All count units are stated
explicitly (cells vs unique slugs).

- AC-1: sm_86 submission-track contract overlay exists and is honored by the headline runner.
  - Positive Tests (expected to PASS):
    - `workloads/simbench_bw/comparison_contract_sm86_submission.toml` exists with `scope.arch_set=["sm_86"]`, `cache_layers=["L2"]`, `device_set=["RTX 3090 Ti"]`, `coverage_gate.publication_grade_floor_per_arch=10`, `coverage_gate.cohort_paper_target_unique_slugs=30`, `coverage_gate.cohort_paper_target_cells=31`, `coverage_gate.aspirational_target_unique_slugs=40`, `prior_art_handling="head_to_head_for_RDH_and_trivial_only_contextual_otherwise"`.
    - The base `workloads/simbench_bw/comparison_contract.toml` is unchanged (the project's two-arch program description is preserved as the broader contract; the overlay is the submission's view).
    - `tools/local_labels/run_headline_cohort_predict_compare.py` accepts a `--contract-overlay` flag that, when provided, overrides scope checks against the base contract.
  - Negative Tests (expected to FAIL):
    - Invoking the runner with the overlay flag pointed at the overlay file AND a sm_120 cell in the cohort manifest emits typed_diagnostic `cell_arch_outside_overlay_scope` and exits non-zero.
    - The overlay file with a missing required field (e.g., absent `cohort_paper_target_unique_slugs`) fails a schema-validation test.

- AC-2: re-scope production surfaces to L2-only with L1 diagnostics in a SEPARATE artifact.
  - Positive Tests:
    - `workloads/simbench_bw/cache_predictor.py` carries a module-level scope docblock at file top that cites the overlay's `cache_layers=["L2"]` verbatim and declares L1 as appendix-only diagnostics with no L2-prediction-path influence.
    - `tools/local_labels/run_headline_cohort_predict_compare.py` emits L2 bundles to `data/sm_86_predictor_accuracy_recovery_v2/predictor_bundles/` AND emits a SEPARATE L1 diagnostic artifact `data/sm_86_predictor_accuracy_recovery_v2/l1_diagnostics/<slug>__<shape>.json` whose path is NEVER globbed by `run_comparison`.
    - `README.md` framing aligns with the overlay scope (L2-only prediction with L1 as diagnostics).
  - Negative Tests:
    - A test that injects a fake L1TEX bundle into `predictor_bundles/` triggers `join_key_unmatched` exclusion (existing behavior) AND the new L1 diagnostic at `l1_diagnostics/<slug>__<shape>.json` is consumed by a diagnostic-only verifier without entering the headline geomean.
    - A scoped source-text scan (limited to tracked prose / docs under `workloads/simbench_bw/cache_predictor.py`, `README.md`, and the overlay docs; with an allowlist for the scope docblock itself) fails if any current-tense "L1 prediction" claim appears outside the appendix-diagnostic block.

- AC-3: cohort manifest as single source of truth.
  - Positive Tests:
    - `data/sm_86_predictor_accuracy_recovery_v2/cohort_manifest.json` exists with one entry per cell carrying: `slug`, `shape_point`, `family_bucket`, `split_assignment` (one of `calibration` / `validation` / `holdout`), `expected_trace_path`, `expected_label_path`, `admission_status` (one of `admitted` / `attempted_excluded`), `exclusion_typed_reason` (closed enum: `trace_budget_exceeded`, `ncu_repeatability_fail`, `build_fail`, `typed_unavailable_emit`, `trace_missing`, `label_missing`, `none_admitted`), `identity_tuple` (slug + shape_point + binary_sha256 + kernel_signature_sha256 + source_commit).
    - The manifest declares `schema_version` and validates against a strict JSON-schema validator emitted at `tools/sm_86_predictor_accuracy_recovery_v2/schemas/cohort_manifest.schema.json`.
    - `tools/local_labels/run_headline_cohort_predict_compare.py` loops over the manifest's `admission_status="admitted"` entries; the prior `_HEADLINE_COHORT_13` hardcoded tuple is replaced by a manifest loader.
    - Every Direction-2/3/5 test references the manifest as its cohort source.
  - Negative Tests:
    - A duplicate `(slug, shape_point)` in the manifest fails schema validation.
    - A manifest entry whose `expected_trace_path` does not resolve on disk causes admission to fall back to `attempted_excluded` with `exclusion_typed_reason="trace_missing"`.
    - The schema validator rejects a manifest whose `admission_status` carries a value outside the closed enum.

- AC-4: cohort expansion to >= 31 admitted cells across >= 30 unique slugs covering 6 family buckets.
  - Positive Tests:
    - `cohort_manifest.json` reports >= 31 entries with `admission_status="admitted"`.
    - The admitted set contains >= 30 distinct `slug` values.
    - Workload-family coverage test asserts per-bucket minima against the admitted set: DL-primitives >= 4 slugs, GEMM-family >= 4 slugs, attention-conv >= 2 slugs, memory-stress >= 3 slugs, irregular-hostile >= 4 slugs, vendor-baselines >= 1 slug. Total >= 18 covered-family-bucket slugs; the remaining >= 12 slugs may distribute freely across the 6 buckets to reach >= 30 unique slugs.
    - Depth-first dimension: >= 6 unique slugs carry >= 3 distinct `shape_point` values each in the admitted set.
    - Every admitted family bucket appears in >= 2 of the 3 splits (prevents family-skewed splits).
  - Negative Tests:
    - The publication-target gate fails when ANY of: admitted cells < 31 OR unique slugs < 30 OR any family-bucket minimum unmet OR balancing invariant (AC-7) violated OR family-coverage-across-splits invariant unmet. The negative gate uses OR semantics: a single violation is sufficient to fail.
    - An explicit fixture test exercises each OR branch independently (six fixtures, each violating exactly one criterion; all six fail the gate).

- AC-5: two-tier gate separation between contract floor and submission gate.
  - Positive Tests:
    - The headline runner enforces the contract floor (`publication_grade_floor_per_arch=10`) as a sanity gate; failing it stops the pipeline with `cohort_below_contract_floor` typed diagnostic.
    - The submission gate (`cohort_paper_target_unique_slugs=30` AND `cohort_paper_target_cells=31`) is enforced by a separate runner phase; failing it emits a typed_diagnostic `cohort_below_paper_target` and prevents the holdout-evaluation report from being marked `publication-grade`.
    - The runner exposes both gate verdicts in its emitted report.
  - Negative Tests:
    - A cohort with 10 <= cells < 31 passes the contract floor but fails the submission gate; the holdout report is emitted but tagged `submission_gate_not_met`.

- AC-6: cohort replacement policy locked at freeze time with attempted / admitted accounting.
  - Positive Tests:
    - `cohort_manifest.json` carries a `frozen_at_commit` field; a post-freeze swap of any cell requires (a) a fresh `sweep_id` AND (b) a documented `exclusion_typed_reason` from the closed enum AND (c) an append to the manifest's `replacement_history` block.
    - The manifest reports `attempted_cell_count` (all attempted cells regardless of admission outcome) AND `admitted_cell_count`; `attempted >= cohort_paper_target_cells` AND `admitted >= publication_grade_floor`.
    - Every excluded cell carries an `exclusion_typed_reason` from the closed enum.
  - Negative Tests:
    - A post-freeze replacement attempt without `sweep_id` rotation triggers a fail-closed assertion.
    - An exclusion with `exclusion_typed_reason` outside the closed enum fails the schema validator.

- AC-7: locked calibration / validation / holdout split with split-by-slug + balancing + operational blinding.
  - Positive Tests:
    - `data/sm_86_predictor_accuracy_recovery_v2/cohort_split.json` exists with disjoint membership: calibration >= 14 unique slugs (>= 15 cells), validation >= 8 unique slugs (>= 8 cells), holdout >= 8 unique slugs (>= 8 cells). Total split membership matches the admitted cohort.
    - Split-by-slug invariant: every shape_point of any given slug lands in the SAME split (so the holdout is true never-seen kernels, not "we've seen this kernel at another shape").
    - Balancing invariant: NO single slug contributes more than 1/3 of any split's cell count (prevents a multi-shape slug from dominating a split when shape_points are unequal across slugs).
    - Operational blinding rule: the calibration runner and the validation runner REFUSE to accept `data/sm_86_predictor_accuracy_recovery_v2/sealed_holdout_labels.json` as an input path; the calibration-mode label loader has no code path that opens that file. (Note: the ContextVar guard is a testable code-level defense, not a security boundary; this operational rule is the load-bearing protection.)
    - The negative-leakage test exercises four leak paths during calibration mode and asserts each raises: (a) direct `label_loader.load(path)` call on the sealed file; (b) cached label-map preload that includes a sealed cell; (c) any baseline runner reading the sealed file during calibration-set fitting; (d) any report-generator code reading the sealed file during a calibration-set report.
  - Negative Tests:
    - A split file with overlap between any two sets fails membership-disjoint validation.
    - A split-by-slug violation (different shape_points of the same slug in different splits) fails the invariant test.
    - A balancing violation (one slug contributes > 1/3 of any split's cells) fails the balancing test.
    - A family-coverage-across-splits violation (a family bucket present in only 1 split) fails the family-balance test.
    - A holdout evaluation that reads calibration-set truth values to compute holdout rel_err is a soft-leak; the test asserts only the sealed file is opened during holdout evaluation.

- AC-8: locked thresholds and factors re-derived from CALIBRATION-SET-ONLY evidence.
  - Positive Tests:
    - `data/sm_86_predictor_accuracy_recovery_v2/locked_protocol.md` documents: HARD-1 / HARD-2 / HARD-3 bands, L2 reduction factor, L2 high-footprint per-slug factors, AC-9 structural-signature thresholds, AND the L1 subfamily factors (silu_elementwise / shared_reduction) - all derived from the calibration set ONLY.
    - `data/sm_86_predictor_accuracy_recovery_v2/locked_thresholds.json` carries the values; `cache_predictor.py` reads them at runtime via the locked-thresholds loader, NOT from hardcoded module constants.
    - L1 subfamily factors live in locked_thresholds.json only because the L1 diagnostic artifact reads them; they do NOT enter any L2 prediction code path. A grep / static-source-text test confirms the L2 prediction code path does not reference L1 subfamily factors.
    - Validation-set evaluation gates calibration termination: if validation rel_err diverges from calibration rel_err by more than a documented threshold (calibration-time choice), the calibration factors are rejected and the loop iterates.
  - Negative Tests:
    - A factor-derivation script that reads validation-set or holdout-set truth values fails the calibration-only test.
    - A `cache_predictor.py` change that hardcodes a threshold without updating `locked_thresholds.json` fails the threshold-lockstep test.

- AC-9: holdout headline accuracy evaluation report.
  - Positive Tests:
    - `data/sm_86_predictor_accuracy_recovery_v2/holdout_evaluation_report.json` is emitted ONLY after the unblind step (after validation-set pass AND after the unblind_token ContextVar is set by the holdout-evaluation runner).
    - The report carries per-cell `predicted` / `truth` / `rel_err` / `bucket`, plus aggregate metrics (geomean rel_err, MAPE, MdAPE, Pearson rho with 95% CI, Spearman rho with 95% CI), per-family breakdowns, and `attempted` / `admitted` accounting.
    - Holdout-set HARD-1 / HARD-2 / HARD-3 bands enforced under the new overlay; pass / fail recorded explicitly per band.
    - The report records `cohort_manifest.lock` digest (sha256 of the manifest content at report time) so a reader can prove which manifest produced which report.
    - The report records the calibration sweep_id, validation sweep_id, and holdout sweep_id separately.
  - Negative Tests:
    - Emitting the holdout report without a prior passing validation-set evaluation fails the gate.
    - A holdout report whose `cohort_manifest.lock` digest does not match the manifest at evaluation time fails the manifest-pinning test.

- AC-10: baseline comparisons and headline ablations.
  - Positive Tests:
    - Nugteren 2014 RDH clean-room baseline implemented at `tools/baseline_runners/nugteren2014_rdh_baseline.py`, consuming the existing HRD profile + sharing locality + divergence proxies feature inputs. Versioned with `BASELINE_VERSION = "nugteren2014_v1"`.
    - Uncorrected-walker baseline: this work's L2 walker output WITHOUT the high-footprint OR reduction corrections.
    - Trivial baselines: `predict_mean` (constant cohort mean), `predict_last_level` (constant 1.0).
    - Each baseline emits `(model_name, model_version, input_features, calibration_set_id, prediction, comparable_metric)` per cell to `data/sm_86_predictor_accuracy_recovery_v2/baseline_comparison/<baseline_name>__<cell_id>.json`.
    - Three headline ablations: (a) disable L2 high-footprint correction; (b) disable L2 reduction correction; (c) report L1 subfamily collapse as APPENDIX-ONLY (the L1 dispatch operates on the diagnostic artifact, not the headline prediction; the collapsed-factor variant is reported in an appendix table).
    - Baselines + ablations report includes failure / coverage accounting: per-baseline `attempted` / `comparable` / `failed_typed_reason` counts.
    - MDM 2020 and Accel-Sim are emitted as CONTEXTUAL prior-art table entries (paper-text comparison with published numbers, not head-to-head), justified by `comparison_contract.toml::scope.prior_art_handling = "contextual_only"`.
  - Negative Tests:
    - A baseline runner that opens the sealed holdout labels file during fitting raises.
    - A baseline whose `BASELINE_VERSION` field is missing fails the baseline-completeness test.
    - The L1-subfamily-collapse ablation appearing in the HEADLINE ablations report (rather than the appendix) fails the appendix-scope test.

- AC-11: full metric reporting with confidence intervals and per-family breakdowns.
  - Positive Tests:
    - Per-cell and aggregate report carries: geomean rel_err, MAPE, MdAPE (median absolute percentage error), Pearson rho with 95% CI (computed via Fisher transform), Spearman rho with 95% CI (computed via permutation or bootstrap), per-family breakdown, attempted / admitted accounting.
    - No headline-only aggregate is emitted; every aggregate number carries its per-family breakdown.
    - CI methods are documented in `locked_protocol.md`: bootstrap as default for all aggregates; Fisher transform for Pearson; permutation (or bootstrap) for Spearman.
  - Negative Tests:
    - A report claim missing any of (geomean rel_err, MAPE, MdAPE, Pearson, Spearman, CIs, per-family breakdown) fails the schema validator.

- AC-12: reviewer-facing artifact bundles with sha256 digests.
  - Positive Tests:
    - `data/sm_86_predictor_accuracy_recovery_v2/frozen_protocol_bundle/` directory contains `cohort_manifest.json` + `cohort_split.json` + `locked_protocol.md` + `locked_thresholds.json` AND a `manifest.sha256` file listing the sha256 of each.
    - `data/sm_86_predictor_accuracy_recovery_v2/final_evaluation_bundle/` directory contains `holdout_evaluation_report.json` + `baseline_comparison_report.json` + `ablation_report.json` AND a `manifest.sha256` file listing the sha256 of each.
    - `data/sm_86_predictor_accuracy_recovery_v2/claim_mapping.md` lists each paper claim with the exact report field + test that backs it. Each row carries: claim text, report file path + JSON pointer, test file path + test name.
  - Negative Tests:
    - A bundle missing any required artifact fails the bundle-completeness test.
    - A `manifest.sha256` entry that does not match the actual file's sha256 fails the digest-pinning test.

- AC-13: pre-existing R48 evidence-provenance discipline preserved.
  - Positive Tests:
    - Post-commit cascade-recap per BL-20260508 still passes at the new HEAD; source_commit drift verifier at `tools/local_labels/test_post_commit_source_commit_drift.py` still passes; identity-tuple lockstep tests still pass; byproduct manifest_pairing_index sidecars still emitted by the producer code path.
    - `ALGORITHM_VERSION = "smith_eklund_lru_v1"` at `workloads/simbench_bw/cache_predictor.py` is unchanged.
  - Negative Tests:
    - Any predictor-touching change without a forward-reference `cache_predictor.py` block fails the drift verifier.
    - Any modification under `tools/nvbit_tracer/` or to the CUPTI Activity collector fails the path-boundary test.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality
and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation completes all 13 acceptance criteria with full test
coverage AND extends to >= 40 unique slugs (the `aspirational_target`)
covering all 6 family buckets at the higher end of each bucket minimum.
The L1 diagnostic artifact path is exercised end-to-end on the full
admitted cohort. The baseline-comparison set includes Nugteren 2014 +
uncorrected-walker + two trivial baselines, plus a depth-first
exploration of CI methods (bootstrap + Fisher + permutation + jackknife
variants documented as a methodology table). The reviewer-facing bundle
includes a synthetic stress-test appendix exercising holdout blinding
under deliberate adversarial conditions (synthetic cells, replaced
cells, manifest mutation attempts).

### Lower Bound (Minimum Acceptable Scope)

The implementation meets the submission gate (>= 30 unique slugs / >= 31
cells admitted) with the minimum per-family bucket coverage, the locked
calibration / validation / holdout protocol with the balancing rules +
operational blinding rule + 4-leak-path negative test, the
manifest-driven runner, the L1-separated diagnostic artifact, the three
head-to-head baselines (Nugteren 2014 + uncorrected-walker + one
trivial), the three headline ablations, and the full metric report with
CIs. The L1-subfamily-collapse ablation is appendix-only. MDM 2020 +
Accel-Sim are contextual table entries only.

### Allowed Choices

- Can use: existing R48 infrastructure (NVBit producer, CUPTI Activity
  collector, local-label driver, cascade-recap chain); existing R25
  NVBit slowdown budget (`NVBIT_SLOWDOWN_BUDGET_MAX_RATIO = 400.0`);
  existing tracer outputs (HRD profile, sharing locality, divergence
  proxies, order projections); existing per-slug factor dispatch in
  `cache_predictor.py`.
- Cannot use: NCU on any cloud / server GPU; sm_120 evidence (deferred
  to a future plan); ALGORITHM_VERSION bump; nvasim / Accel-Sim /
  GPGPU-Sim source code (consuming published numbers as contextual prior
  art is allowed); NVBit producer-side changes; CUPTI Activity collector
  changes; modifications under `tools/nvbit_tracer/`.

> **Note on Deterministic Designs**: This plan carries several fixed
> design choices (cohort target = >= 30 unique slugs / >= 31 cells;
> split by slug with 1/3 balancing rule; L2-only paper claim with L1 as
> appendix-only diagnostics; Nugteren 2014 + uncorrected-walker +
> trivial baselines as the head-to-head set; MDM 2020 + Accel-Sim as
> contextual-only). The Allowed Choices section above narrows the
> implementation surface accordingly.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These
> are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The plan is sequenced as five milestones (A through E). Milestones A
and B are prerequisites for any holdout-set sweep. Milestones C and D
together produce the calibration / validation / holdout sweeps.
Milestone E emits the reviewer-facing artifacts.

```
Milestone A: re-scope production surfaces + contract overlay
  - Phase A.1: contract overlay file + runner --contract-overlay flag
  - Phase A.2: cache_predictor.py scope docblock + README re-prose
  - Phase A.3: L1 diagnostic separate-artifact emission path
  - Phase A.4: scoped source-text scan + L1-diagnostic-only verifier

Milestone B: cohort manifest + expansion
  - Phase B.1: cohort_manifest.json schema + strict validator
  - Phase B.2: migrate run_headline_cohort_predict_compare.py from
              hardcoded tuple to manifest loader
  - Phase B.3: live trace + NCU label generation for 20+ new cells on
              sm_86 (using existing R25 NVBit slowdown budget +
              local-label driver), populating manifest with admission
              outcomes
  - Phase B.4: family-coverage tests + replacement-policy lock

Milestone C: locked split + blinding
  - Phase C.1: split design + cohort_split.json with split-by-slug +
              balancing invariants
  - Phase C.2: sealed_holdout_labels.json file + operational
              calibration-runner refusal-to-accept rule
  - Phase C.3: ContextVar guard + 4-leak-path negative test
  - Phase C.4: split-membership-disjoint + balancing + family-balance tests

Milestone D: calibration + validation + holdout sweeps
  - Phase D.1: re-derive HARD bands, factors, structural-signature
              thresholds from calibration set only
  - Phase D.2: validation-set evaluation gating calibration termination
  - Phase D.3: locked_protocol.md + locked_thresholds.json frozen
  - Phase D.4: holdout-set unblind + holdout_evaluation_report.json

Milestone E: baselines + ablations + reviewer bundles
  - Phase E.1: Nugteren 2014 RDH baseline clean-room
  - Phase E.2: uncorrected-walker + trivial baselines
  - Phase E.3: three headline ablations + appendix L1-collapse ablation
  - Phase E.4: full metric report (MAPE/MdAPE/Pearson/Spearman with CIs)
  - Phase E.5: claim_mapping.md + frozen_protocol_bundle +
              final_evaluation_bundle with sha256 digests
```

### Relevant References

- `plans/l1_l2_accuracy_gate_closure_plan.md` - the R48 plan whose closure (HEAD `85bef166`) is the starting point.
- `workloads/simbench_bw/comparison_contract.toml` - the project's base contract; the overlay is built on top without modifying it.
- `workloads/simbench_bw/cache_predictor.py` - production predictor; the L2 prediction path stays load-bearing; L1 path is re-scoped to diagnostic-only.
- `tools/local_labels/run_headline_cohort_predict_compare.py` - current headline runner with hardcoded `_HEADLINE_COHORT_13`; migrated to manifest-driven in this plan.
- `data/sm_86_predictor_accuracy_recovery_v2/l2_high_footprint_correction_shadow_report.json` - current shadow report (9 admitted of 13 cohort cells); seed for the calibration set.
- `references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf` - Nugteren 2014 RDH baseline source.
- `references/papers/knowledge_base/active/2020__mdm_the_gpu_memory_divergence_model.pdf` - MDM 2020 (contextual prior-art only).
- `references/papers/knowledge_base/active/2020__accel_sim_an_extensible_simulation_framework_for_validated_gpu_modeling.pdf` - Accel-Sim 2020 (contextual prior-art only).
- `references/papers/knowledge_base/active/2015__cross_architecture_performance_prediction_xapp.pdf` - XAPP 2015 (out-of-scope for this plan; future cross-arch plan reference).
- `tools/local_labels/test_post_commit_source_commit_drift.py` - drift verifier; must continue to pass.
- `.humanize/bitlesson.md` - BL-20260508 (cascade-recap), BL-20260520 (predictor-touching commit chain), BL-20260517 (user-escalation pattern).

## Dependencies and Sequence

### Milestones

1. Milestone A (Re-scope production surfaces + contract overlay): prerequisite for everything else. No new sweep required.
   - Phase A.1 through A.4 sequential.
   - Output: overlay file; runner `--contract-overlay` flag; L1 diagnostic artifact path; scope docblock; scoped source-text scan.
2. Milestone B (Cohort manifest + expansion): depends on Milestone A. Prerequisite for Milestones C/D/E.
   - Phase B.1 then B.2 sequential; Phase B.3 can run in parallel with B.2 once the manifest schema is locked.
   - Output: `cohort_manifest.json` with >= 31 admitted cells across >= 30 unique slugs; manifest-driven runner.
3. Milestone C (Locked split + blinding): depends on Milestone B's manifest. Phase C.4 negative-leakage test is the load-bearing reviewer-defensible artifact.
   - Phase C.1 through C.4 sequential.
   - Output: `cohort_split.json`; `sealed_holdout_labels.json`; operational + ContextVar blinding; 4-leak-path negative test.
4. Milestone D (Calibration + validation + holdout sweeps): depends on Milestones B + C. Phase D.4 unblind step is the final pre-bundle gate.
   - Phase D.1 through D.4 sequential.
   - Output: `locked_protocol.md`; `locked_thresholds.json`; `holdout_evaluation_report.json`.
5. Milestone E (Baselines + ablations + reviewer bundles): depends on Milestone D for the locked predictor. Phase E.1 + E.2 can run in parallel; E.5 is the final reviewer-facing artifact.
   - Phase E.1 and E.2 parallel; E.3 depends on E.1+E.2; E.4 depends on E.3; E.5 depends on E.4.
   - Output: `baseline_comparison_report.json`; `ablation_report.json`; `claim_mapping.md`; `frozen_protocol_bundle/`; `final_evaluation_bundle/`.

Relative dependencies between components: contract overlay enables every downstream test; manifest schema enables the runner migration and every gate test; locked split + sealed labels enable the calibration/validation loop; locked thresholds enable the holdout evaluation; the holdout report enables the baselines and ablations to compare against the framework's headline number on the same cells.

## Task Breakdown

Each task carries exactly one routing tag: `coding` (Claude implements
directly) or `analyze` (executed via `/humanize:ask-codex`, results
integrated by Claude).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1  | Write `comparison_contract_sm86_submission.toml` overlay file with all required fields | AC-1 | coding | - |
| task2  | Add `--contract-overlay` flag to `run_headline_cohort_predict_compare.py`; wire scope-check override | AC-1 | coding | task1 |
| task3  | Schema-validation test for overlay file (required-field coverage) | AC-1 | coding | task1 |
| task4  | Add module-level scope docblock to `cache_predictor.py` citing overlay verbatim | AC-2 | coding | task1 |
| task5  | Add L1-diagnostic-artifact emission path to runner; separate from `predictor_bundles/` | AC-2 | coding | task2, task4 |
| task6  | Re-prose `README.md` framing section to match overlay scope | AC-2 | coding | task4 |
| task7  | Scoped source-text scan + L1-diagnostic-only verifier test | AC-2 | coding | task4, task5 |
| task8  | Design `cohort_manifest.schema.json` (strict JSON schema with schema_version, identity_tuple, exclusion enum) | AC-3 | analyze | - |
| task9  | Write `cohort_manifest.json` schema validator at `tools/sm_86_predictor_accuracy_recovery_v2/validate_cohort_manifest.py` | AC-3 | coding | task8 |
| task10 | Migrate `run_headline_cohort_predict_compare.py` from `_HEADLINE_COHORT_13` to manifest-driven loader | AC-3 | coding | task9 |
| task11 | Initial cohort_manifest.json seeded with the 13 current cohort cells (9 admitted + 4 typed_unavailable) | AC-3, AC-6 | coding | task9 |
| task12 | Capture live traces for 20+ new candidate cells on sm_86 (3090 Ti) using existing R25 NVBit budget | AC-4 | coding | task11 |
| task13 | Generate NCU labels for the new cells via `tools/local_labels/label_driver.sh` into `data/local_labels/sweeps/sweep_<calibration_id>/` | AC-4 | coding | task12 |
| task14 | Run predictor against each new cell; populate manifest with admission outcomes + typed exclusion reasons | AC-4, AC-6 | coding | task13 |
| task15 | Workload-family coverage test (per-bucket minima + family-balance-across-splits) | AC-4 | coding | task11 |
| task16 | Depth-first count test (>= 6 slugs with >= 3 distinct shape_points each) | AC-4 | coding | task11 |
| task17 | Six-fixture OR-gate test for AC-4 negative gate (each fixture violates exactly one criterion) | AC-4 | coding | task15, task16 |
| task18 | Two-tier gate test: contract floor (10 cells) sanity gate AND submission gate (30 unique slugs / 31 cells) separation | AC-5 | coding | task14 |
| task19 | Cohort replacement-policy lock test (frozen_at_commit + sweep_id rotation + replacement_history append) | AC-6 | coding | task14 |
| task20 | Design `cohort_split.json` schema with split-by-slug invariant + balancing invariant + family-balance | AC-7 | analyze | task14 |
| task21 | Implement split-by-slug assignment algorithm with 1/3 balancing constraint + family-balance | AC-7 | coding | task20 |
| task22 | Write `sealed_holdout_labels.json` file containing holdout-cell NCU truth values | AC-7 | coding | task14, task21 |
| task23 | Implement operational blinding rule: calibration/validation runners refuse to accept sealed-labels path | AC-7 | coding | task22 |
| task24 | Implement ContextVar `unblind_token` guard on label loader | AC-7 | coding | task22 |
| task25 | Write 4-leak-path negative test (direct loader / cached map / baseline runner / report generator) | AC-7 | coding | task23, task24 |
| task26 | Membership-disjoint + split-by-slug + balancing + family-balance test (positive + negative fixtures) | AC-7 | coding | task21 |
| task27 | Re-derive HARD-1/2/3 bands, L2 reduction factor, L2 high-footprint per-slug factors, structural-signature thresholds from CALIBRATION SET ONLY | AC-8 | analyze | task22, task23 |
| task28 | Re-derive L1 subfamily factors from calibration set only (appendix-only; not part of L2 prediction path) | AC-8 | analyze | task22, task23 |
| task29 | Write `locked_thresholds.json` + add reader to `cache_predictor.py` (no hardcoded module constants) | AC-8 | coding | task27, task28 |
| task30 | Write `locked_protocol.md` with full design rationale + CI method choices | AC-8, AC-11 | coding | task27, task28 |
| task31 | Implement validation-set evaluation as calibration-termination gate | AC-8 | coding | task29 |
| task32 | Grep / static-source-text test: L2 prediction code path does NOT reference L1 subfamily factors | AC-8 | coding | task29 |
| task33 | Implement holdout-evaluation runner (sets unblind_token; emits `holdout_evaluation_report.json`) | AC-9 | coding | task24, task29 |
| task34 | Compute aggregate metrics with CIs (geomean rel_err, MAPE, MdAPE, Pearson w/ Fisher CI, Spearman w/ permutation CI), per-family breakdown | AC-9, AC-11 | coding | task33 |
| task35 | Implement Nugteren 2014 RDH baseline at `tools/baseline_runners/nugteren2014_rdh_baseline.py` | AC-10 | coding | task29 |
| task36 | Implement uncorrected-walker baseline | AC-10 | coding | task29 |
| task37 | Implement predict_mean + predict_last_level trivial baselines | AC-10 | coding | task29 |
| task38 | Implement three headline ablations (disable high-footprint; disable reduction correction; L1-collapse moved to APPENDIX) | AC-10 | coding | task29, task33 |
| task39 | Failure / coverage accounting in baseline report (attempted / comparable / failed_typed_reason per baseline) | AC-10 | coding | task35, task36, task37 |
| task40 | Emit `baseline_comparison_report.json` + `ablation_report.json` with full metrics + per-family | AC-10, AC-11 | coding | task38, task39 |
| task41 | Write contextual prior-art table for MDM 2020 + Accel-Sim (paper-text comparison; no head-to-head) | AC-10 | analyze | task40 |
| task42 | Compute `cohort_manifest.lock` digest; record in every report | AC-9, AC-12 | coding | task11, task33, task40 |
| task43 | Write `claim_mapping.md` (each paper claim with its report field + test) | AC-12 | coding | task33, task40 |
| task44 | Assemble `frozen_protocol_bundle/` directory + `manifest.sha256` | AC-12 | coding | task22, task29, task30 |
| task45 | Assemble `final_evaluation_bundle/` directory + `manifest.sha256` | AC-12 | coding | task33, task40, task43 |
| task46 | Bundle-completeness + digest-pinning tests | AC-12 | coding | task44, task45 |
| task47 | Verify R48 cascade-recap discipline still passes at the new HEAD (post-commit drift; identity-tuple lockstep; byproduct MPI) | AC-13 | coding | task14, task33 |
| task48 | Forward-reference block in `cache_predictor.py` for the locked-thresholds reader (predictor-touching per BL-20260520) | AC-13 | coding | task29 |
| task49 | Cascade-recap commit per BL-20260508 after each predictor-touching change | AC-13 | coding | task29, task48 |

## Claude-Codex Deliberation

### Agreements

- The sm_86 submission-track contract overlay (AC-1) reconciles the
  contract mismatch (`comparison_contract.toml::arch_set = ["sm_86", "sm_120"]`
  vs sm_86-only submission) without modifying the base contract.
- Manifest-driven cohort membership (replacing `_HEADLINE_COHORT_13`
  with `cohort_manifest.json`) closes hidden cohort drift and
  hardcoded-runner risk.
- Separating L1 diagnostics from `predictor_bundles/` closes headline
  contamination.
- The holdout blinding design (sealed labels + ContextVar guard +
  operational refusal-to-accept rule + split-by-slug + balancing) is
  materially stronger than the draft's ContextVar-only approach.
- Treating Accel-Sim / MDM 2020 as contextual prior-art only is
  acceptable under
  `comparison_contract.toml::scope.prior_art_handling = "contextual_only"`.
- Full metric reporting (MAPE + MdAPE + Pearson + Spearman with CIs +
  per-family breakdowns) closes the headline-only-aggregate weakness.
- Preserving R48 provenance discipline as a hard compatibility
  requirement (no ALGORITHM_VERSION bump; cascade-recap stays;
  source_commit drift verifier stays; identity-tuple lockstep stays).

### Resolved Disagreements

- Cohort math (cells vs unique slugs vs split units): Claude v1
  proposed `>= 31 cells / >= 15 unique kernels`. Codex pass 2 disagreed:
  top-tier breadth requires `>= 30 unique kernels`. Claude v2 reconciled
  to `>= 30 unique slugs / >= 31 cells` (primary unit = unique slugs;
  cells derived from split sizes). Split sizes per split = unique slug
  count; balancing rule (no slug > 1/3 of any split) prevents multi-shape
  slugs from dominating.
- AC-4 negative gate logic: Claude v1 used AND; Codex pass 2 caught
  this. Claude v2 fixed to OR semantics (any single violation fails the
  gate). A six-fixture OR-branch test (task17) pins the semantics.
- Two-tier gate separation: Claude v1 conflated
  `publication_grade_floor=10` and `cohort_paper_target=31`. Codex
  pass 2 demanded separation. Claude v2 introduces two-tier gates
  (contract floor as sanity gate; submission gate as publication-grade
  marker).
- L1 status ambiguity: Claude v1 left L1 subfamily factors in the L2
  prediction path; Codex pass 2 demanded clarification. Claude v2
  explicitly states L1 factors live in `locked_thresholds.json` ONLY
  because the L1 diagnostic artifact reads them; the L2 prediction code
  path does NOT reference them (enforced by a grep test, task32). L1
  ablation moved to appendix-only.
- Holdout blinding language: Claude v1 used "sealed" terminology
  suggesting cryptographic protection. Codex pass 2 disagreed. Claude
  v2 re-frames as a testable code-level guard (ContextVar) PLUS an
  operational rule (calibration runners refuse to accept the
  sealed-labels path). No security claim is made.
- Split-by-slug + depth-first balancing: Codex pass 2 flagged that
  multi-shape slugs could dominate a split. Claude v2 adds the 1/3
  balancing invariant + the family-balance-across-splits invariant.

### Convergence Status

- Final Status: `converged`
- Convergence rounds executed: 2 (Codex pass 1; Claude candidate v1;
  Codex pass 2; Claude candidate v2; Codex pass 3 AGREE with no
  remaining required changes and no unresolved items).

## Pending User Decisions

(none) - all five candidate-v1 DEC items were resolved through the
user's prior in-conversation direction at gen-plan invocation:

- DEC-1 (L1 in public artifacts?): resolved to appendix-only
  diagnostics per the user's "demotes the L1 hostile catastrophe"
  wording.
- DEC-2 (cohort target size?): resolved to >= 30 unique slugs / >= 31
  cells per the user's "30+ kernel evaluation cohort on sm_86" wording.
- DEC-3 (MDM 2020 baseline scope?): resolved to contextual-only per
  `comparison_contract.toml::scope.prior_art_handling`.
- DEC-4 (depth vs breadth?): resolved to hybrid (>= 30 unique slugs
  WITH >= 6 multi-shape slugs for depth coverage).
- DEC-5 (submission venue tier?): resolved to top-tier umbrella
  (ASPLOS / HPCA / ISCA / MICRO) per the user's "targeting the
  submission to top-tier conferences" wording. The plan's deliverables
  make the work competitive at ASPLOS / HPCA after Milestones A through
  E land. ISCA / MICRO are reachable but would additionally require a
  future cross-arch plan (sm_120) to land first; the current plan does
  NOT block that future work.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments MUST NOT contain plan-specific
  terminology such as `AC-N`, `Milestone`, `Step`, `Phase`, or similar
  workflow markers (the existing `tools/test_plan_token_scanner.py`
  enforces this repo-wide).
- These terms are for plan documentation only, not for the resulting
  codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g.,
  "cohort manifest validator" not "AC-3 validator"; "holdout evaluation
  runner" not "Phase D.4 runner"; "publication-grade cohort gate" not
  "AC-5 two-tier gate").

### Predictor-Touching Commit Chain (BL-20260520)

Every change that touches `workloads/simbench_bw/cache_predictor.py`
(the scope docblock in task4; the locked-thresholds reader in task29;
the forward-reference block in task48) is a predictor-touching commit
and triggers the BL-20260508 cascade-recap discipline: a single
fix-commit followed by a chore-recap commit that stamps the 11
drift-sensitive artifacts to the fix-commit. The R48 cascade-recap
helpers (`refresh_admitted_label_source_commit.py`,
`refresh_predictor_trace_source_commit.py`,
`recapture_l2_high_footprint_artifacts.py`,
`recapture_l2_reduction_artifacts.py`,
`compute_predicted_l1_per_admitted_cell.py`,
`backfill_trace_byproduct_mpi_sidecars.py`) continue to apply.

### Trace Capture Discipline

New live-trace captures for cohort expansion (task12) use the existing
R25 NVBit slowdown budget (`NVBIT_SLOWDOWN_BUDGET_MAX_RATIO = 400.0`).
Cells exceeding the budget enter the typed-diagnostic path with
`exclusion_typed_reason="trace_budget_exceeded"` - this is expected
behavior, not a regression. A handful of irregular cells
(`l2_thrash_pointer_chase` variants, `register_spill_local_mem`
variants, `global_atomic_histogram`) may legitimately exceed the cap
and become `attempted_excluded`; the manifest records this without
prejudice to the publication-target gate (`attempted >=
cohort_paper_target_cells` is sufficient even if `admitted` is lower,
provided `admitted >= publication_grade_floor`).

### CI Method Defaults

- Geomean rel_err: 95% CI via bootstrap (10000 resamples; bias-corrected
  and accelerated variant).
- MAPE / MdAPE: 95% CI via bootstrap.
- Pearson rho: 95% CI via Fisher transform (z-transform on rho;
  inverse-transform on the z-interval bounds).
- Spearman rho: 95% CI via permutation test (default 10000 permutations)
  with a bootstrap fallback when the permutation distribution is
  degenerate.

All CI methods are documented in `locked_protocol.md` and pinned in
`locked_thresholds.json::ci_methods`.

### Cohort Target Count-Unit Clarification

The `cohort_paper_target_cells = 31` value is the split-count floor
(derived from calibration >= 15 + validation >= 8 + holdout >= 8 cells
under the unique-slug split). When the depth-first dimension is
exercised (>= 6 slugs with >= 3 shape_points each), the practical
admitted cell count rises significantly above 31 (roughly 40-60 cells
for a 30-unique-slug + 6-depth-first cohort). The plan reports both
`unique_slugs` and `cells` in every gate and accounting summary so a
reader cannot confuse the two.

--- Original Design Draft Start ---

# Non-Root GPU Cache Observability Framework with L2 Prediction (sm_86 Submission Track)

## Original Idea

The preceding R48 plan (`plans/l1_l2_accuracy_gate_closure_plan.md`) reached
its acceptance criteria at HEAD `85bef166` (Codex round-29 verdict: 16/16 ACs
MET, 0 mainline gaps, 0 blocking, all HARD targets PASS). However, an
independent review by gpt-5.5 (codex xhigh) and the project's own
`workloads/simbench_bw/comparison_contract.toml` agree that the **current
experimental scope does not meet the project's own publication-grade bar** at
top-tier computer-architecture venues (ISCA / MICRO / HPCA / ASPLOS):

- `comparison_contract.toml::coverage_gate` declares
  `publication_grade_floor_per_arch = 10`; the current admitted cohort is **9
  cells** (5 friendly + 4 hostile), at the contract's `smoke_test_floor_per_arch = 5`.
- `comparison_contract.toml::scope.cache_layers = ["L2"]` (narrowed back from
  L1+L2 because the comparison driver is L2-only); a paper claiming "L1+L2
  cache predictor" misrepresents the driver's scope.
- The 4 "predictor-hostile" cells (carveout_sensitivity_probe,
  working_set_capacity_sweep, l2_thrash_pointer_chase, register_spill_local_mem)
  carry L1 rel_err **93%–100%** — a reviewer-fatal headline if the paper
  claims an L1 predictor under any framing.
- All evidence is sm_86 only. Cross-arch sm_120 verification is "explicitly
  deferred" with a replication assumption — the kind of language reviewers
  cite in fatal-flaw rejection paragraphs.
- HARD-3 was widened mid-flight from 0.15 to a [0.10, 0.30] band, and AC-5
  criterion 4 was reformulated mid-flight from `bin[0].mass >= 0.90` to
  `hrd_long_rd_tail_mass >= 0.10 at rd >= 4096`. Both are user-sanctioned in
  the goal-tracker Plan Evolution Log, but **"user-sanctioned" carries no
  weight in peer review** — reviewers will read these as post-hoc evaluation
  criterion moves to fit the result.

This plan opens the next phase of work, re-scoping the central paper
contribution from "L1+L2 cache hit-rate predictor" to a defensible
**non-root GPU L2 cache observability framework with an L2 prediction model**,
aligning the paper claim with the comparison driver's actual scope, and
exercising the project's existing infrastructure (~30 SimBench microbenches
+ ~20 portable_lcd kernels) to lift the evaluation cohort from smoke-test
floor (5 cells) to publication-grade floor (≥10 cells) and beyond.

**Explicit scoping constraint per user direction (round-30 entry)**:
**cross-architecture validation on sm_120 is OUT OF SCOPE for this plan**
because the RTX 5060 laptop is unavailable. All directions in this plan are
sm_86-only. The work assumes a future plan will pick up sm_120 cross-arch
verification once the laptop becomes available; that future plan's outputs
are NOT a precondition for this plan's submission readiness.

## Re-Scoping Decision (Load-Bearing)

The central contribution claim in any paper produced by this work pivots
from "L1+L2 cache hit-rate predictor" to one of:

- **Option A (recommended)**: **"A non-root GPU L2 cache observability
  framework with calibrated L2 hit-rate prediction"**. The non-root
  measurement methodology (CUPTI Activity + NVBit + read-only NVML; no NCU,
  no driver root, no privileged CUPTI) is the central novelty; the L2
  prediction model is the framework's first validated capability; L1 is a
  *diagnostic* surface, not a prediction claim. This aligns with
  `comparison_contract.toml::scope.cache_layers = ["L2"]` verbatim.
- **Option B (alternative)**: maintain the L1+L2 framing AND seriously
  improve L1 on the hostile cohort by introducing either a reuse-distance
  histogram (RDH) model (mirroring Nugteren 2014's
  `references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf`)
  or a learned ML head (mirroring SynPerf 2026 / NeuroScalar 2025). This is
  higher-ceiling but adds 4–8+ weeks to the schedule and increases scope
  risk.

This plan defaults to **Option A** for the central contribution but reserves
Direction 4 (below) as an optional Option-B path for the L1 hostile cohort.

## Existing Infrastructure Available (Pre-Verified)

The repo already has the infrastructure to evaluate a much broader cohort
than the current 9 cells. From `workloads/simbench_bw/l1/`:

- **30 microbench kernels** including: `bf16_hmma_gemm_traceable`,
  `cp_async_shared_tile_copy`, `dsmem_pairwise_latency`, `dsmem_ring_bandwidth`,
  `fp16_hmma_gemm_traceable`, `fp32_fma_dep_chain`, `fp4_e2m1_mma_dep_chain`,
  `fp6_e2m3_saturation`, `fp6_e3m2_mma_dep_chain`, `fp8_e4m3_qmma_dep_chain`,
  `fp8_e5m2_saturation`, `global_atomic_histogram`, `int8_imma_gemm_traceable`,
  `l2_persisting_window_sweep`, `l2_prefetch_hint_sweep`, `nvfp4_blockscale_gemm`,
  `shared_bank_conflict_stride`, `sparse_2to4_bf16_gemm`, `sparse_2to4_fp8_gemm`,
  `sparse_2to4_int8_gemm`, `tf32_hmma_gemm_traceable`, `tma_tensor_copy_2d`,
  `vendor_fp8_matmul_cublaslt`, `vendor_gemm_cublas`, `warp_divergence_reconvergence`,
  plus the 4 current hostile cells.

From `workloads/portable_lcd/`:

- **~20 production-flavored kernels** including: CUTLASS GEMM variants
  (`gemm_hmma_bf16_sm80`, `gemm_hmma_fp16_sm80`, `gemm_imma_int8_sm80`,
  `gemm_sparse_2to4_bf16_sm80`), Flash Attention (`flash_attn_fwd_bf16_sm80`,
  `flash_attn_bwd_bf16_sm80`), `conv2d_implicit_gemm_bf16_sm80`,
  `embedding_gather_int32_sm80`, `cross_entropy_fwd_fp32_sm80`,
  `rmsnorm_bwd_bf16_sm80`, plus the 4 current friendly cells (layernorm,
  rmsnorm forward, softmax_safe, silu).

Total available: ~50 distinct kernels on sm_86. The publication-grade
floor under `comparison_contract.toml` (≥10 per arch) is comfortably
reachable; the aspirational target (≥20 across) is also reachable. The
top-tier reviewer expectation (25–30 unique kernels minimum, 40–60 strong)
is reachable with one focused expansion round.

## Directions

### Direction 1 (load-bearing): Re-scope paper contribution to L2-only with L1 diagnostics

Codify the L2-only scope in the paper narrative AND in three production
surfaces that currently leak L1 prediction claims:

1. **`tools/local_labels/run_headline_cohort_predict_compare.py`** —
   the comparison driver. Currently emits L2 bundles only via
   `cp.predict_cache_hit_rate`. Add a typed-diagnostic emission path for
   L1 (via `cp.predict_l1_from_trace_dir` / `cp.predict_l1_hit_rate_from_manifest`)
   that produces side-channel L1 hit-rate predictions WITHOUT entering the
   headline geomean. This is the production-side implementation of the
   "L1 as diagnostic, not claim" framing.
2. **`workloads/simbench_bw/cache_predictor.py`** — add a module-level
   documentation block at the top of the file declaring the L2 prediction
   scope and the L1 diagnostic scope, citing
   `comparison_contract.toml:135` (the `cache_layers = ["L2"]` declaration)
   verbatim. The block becomes the paper's Section 3 (Scope and Coverage)
   evidence anchor.
3. **`README.md`** — re-prose the project framing section to align
   with the L2-with-L1-diagnostic scope. Currently the README says "L1/L2
   cache hit ratio" without the L2-only-prediction caveat.

Three new tests:

- `tools/test_l2_only_predictor_scope.py`: a static-source-text scanner that
  fails if `workloads/simbench_bw/cache_predictor.py` carries any
  current-tense "L1 prediction" claim outside the L1-as-diagnostic block.
- `tools/test_l1_diagnostic_path_does_not_enter_headline.py`: a unit test
  that exercises `run_headline_cohort_predict_compare.py` against a fixture
  cell with both L1 and L2 bundles emitted, and asserts the headline
  geomean only sums over the L2 bundle.
- `tools/test_comparison_contract_scope_lockstep.py`: asserts the
  `cache_layers` array in `comparison_contract.toml` is `["L2"]` and that
  the paper-claim production surfaces above all match.

### Direction 2 (load-bearing): Cohort expansion to ≥30 unique kernels on sm_86

Use the existing SimBench + portable_lcd infrastructure to lift the
admitted cohort from 9 cells to **at least 30 unique kernels** (the lower
end of top-tier reviewer expectation; the project's own
`aspirational_target_total = 20` is a floor, not the ceiling). Target
distribution across workload families (mirroring what 2026 ISCA/MICRO/HPCA
reviewers will count):

- **DL primitives (≥6)**: layernorm, rmsnorm, softmax_safe, silu,
  cross_entropy_fwd, embedding_gather.
- **GEMM family (≥6)**: fp16_hmma, bf16_hmma, tf32_hmma, fp8_e4m3_qmma,
  fp4_e2m1_mma, int8_imma; plus optional sparse variants
  (sparse_2to4_bf16_gemm, sparse_2to4_int8_gemm).
- **Attention / Conv (≥3)**: flash_attn_fwd, flash_attn_bwd,
  conv2d_implicit_gemm.
- **Memory-system stress (≥6)**: l2_prefetch_hint_sweep,
  l2_persisting_window_sweep, dsmem_pairwise_latency, dsmem_ring_bandwidth,
  cp_async_shared_tile_copy, tma_tensor_copy_2d.
- **Irregular / hostile (≥6)**: l2_thrash_pointer_chase, register_spill_local_mem,
  global_atomic_histogram, warp_divergence_reconvergence, shared_bank_conflict_stride,
  carveout_sensitivity_probe, working_set_capacity_sweep, fp6_e2m3_saturation,
  fp8_e5m2_saturation.
- **Vendor baselines (≥2)**: vendor_gemm_cublas, vendor_fp8_matmul_cublaslt
  (these document the gap between production kernels and our predictor).

For each cohort cell, this direction requires:

1. Live trace capture on sm_86 (3090 Ti) under the existing R25 NVBit slowdown
   budget (`NVBIT_SLOWDOWN_BUDGET_MAX_RATIO = 400.0` post-R25; cells exceeding
   this enter the trace-budget-exceeded typed-diagnostic path).
2. Label generation via the existing local-label NCU driver at
   `tools/local_labels/label_driver.sh`, into `data/local_labels/sweeps/sweep_<new_id>/`.
3. Predictor bundle emission via `workloads/simbench_bw/cache_predictor.py`
   end-to-end against each cell.
4. Admission via `run_headline_cohort_predict_compare.py` into the new
   headline cohort.

Two new tests:

- `tools/sm_86_predictor_accuracy_recovery_v2/test_publication_grade_cohort_size.py`:
  parametrized test asserting the admitted cohort meets the publication-grade
  floor (≥10 per arch from `comparison_contract.toml`) AND a stronger
  paper-target floor (≥30 unique kernels for the submission claim).
- `tools/sm_86_predictor_accuracy_recovery_v2/test_workload_family_coverage.py`:
  asserts the admitted cohort covers each of the 6 workload-family buckets
  named above with at least the named minimum count.

### Direction 3 (load-bearing): Locked held-out protocol

Replace the current calibration-equals-evaluation mode with a frozen
**calibration / validation / holdout** split. The split is set BEFORE
the next live sweep; correction factors, AC-5 structural predicate
thresholds, HARD-target bounds, and admission rules are frozen at
calibration time and EVALUATED on the holdout set without further tuning.

Concrete structure:

- **Calibration set (~50% of cohort, ≥15 cells)**: the cells used to
  derive the L2 high-footprint correction factors, the L1 subfamily
  factors, the AC-5 4-criterion threshold values, the HARD-1/2 band, and
  the HARD-3 band. The current 9-cell cohort lands here as the seed.
- **Validation set (~25% of cohort, ≥8 cells)**: used DURING calibration
  to detect overfit (factors that hit the calibration set but blow the
  validation set are rejected; the calibration loop iterates with the
  validation set as the early-stopping criterion).
- **Holdout set (~25% of cohort, ≥8 cells)**: NEVER touches the
  calibration or validation loops. The paper's reported accuracy
  numbers are the holdout-set rel_err geomeans. The reviewer-facing
  story is: "we calibrated on N cells; we validated on M cells; we
  evaluated on K cells the calibration never saw."

Three new artifacts:

- `data/sm_86_predictor_accuracy_recovery_v2/cohort_split.json`: pinned
  split-set membership with a manifest_pairing_index–style identity
  tuple for each cell (slug, shape_point, binary_sha256,
  kernel_signature_sha256, source_commit) so a future calibration run
  cannot accidentally leak across sets.
- `data/sm_86_predictor_accuracy_recovery_v2/locked_protocol.md`:
  human-readable description of the protocol, frozen thresholds, and
  the rationale for each (including AC-5 criterion 4's
  `hrd_long_rd_tail_mass >= 0.10 at rd >= 4096` re-derived from
  *calibration-set-only evidence* and re-validated against
  validation-set positive/negative controls).
- `data/sm_86_predictor_accuracy_recovery_v2/holdout_evaluation_report.json`:
  the paper-headline-grade evaluation report emitted only after the
  holdout set is unblinded; carries one row per holdout cell with
  predicted, truth, rel_err, and bucket.

Two new tests:

- `tools/sm_86_predictor_accuracy_recovery_v2/test_cohort_split_membership.py`:
  asserts disjoint membership across the three sets AND that each set
  meets its minimum-size floor.
- `tools/sm_86_predictor_accuracy_recovery_v2/test_holdout_blinding_discipline.py`:
  asserts the calibration code path never reads the holdout cells'
  truth values (enforced by a runtime ContextVar guard around the
  NCU-label reader: holdout cells' truth values raise on access during
  calibration mode).

The HARD-3 widening and AC-5 criterion 4 reformulation from the previous
plan are RE-DERIVED here from calibration-set-only evidence. The
re-derivation is documented in `locked_protocol.md` as a
**calibration-time design decision**, not as a mid-flight evolution.
This is the load-bearing surface that converts the prior plan's "user
sanctioned" criterion moves into "documented design decisions before
evaluation" — the only re-framing that survives peer review.

### Direction 4 (optional, alternative Option-B path): L1 hostile-cohort model upgrade

Reserve as a fall-back direction if the user later wants the L1
prediction claim back in scope. Two sub-paths:

- **D4a (RDH-based)**: implement a reuse-distance-histogram L1 model
  following Nugteren 2014's structure. The HRD profile + sharing
  locality + divergence proxies already collected per-trace in the
  existing tracer infrastructure are direct feature inputs. Effort
  estimate: 4–6 weeks. Expected L1 hostile-cohort rel_err reduction:
  90%+ → 30–50% range based on Nugteren-class results on similar
  workloads.
- **D4b (ML head)**: implement a small learned head over the existing
  HRD/sharing-locality/divergence/order-projection features (mirroring
  SynPerf 2026 / NeuroScalar 2025). Effort estimate: 6–8 weeks
  including baseline ablation against D4a. Expected L1 hostile-cohort
  rel_err reduction: similar or better than D4a if features are
  sufficient.

This direction is OUT OF SCOPE for the initial plan unless the user
sanctions it after Phase A completes. The default L2-only-with-L1-diagnostic
framing (Direction 1) is the higher-confidence path to a top-tier
submission.

### Direction 5 (load-bearing): Baseline comparisons and ablations

The plan's most reviewer-impactful missing piece. Implement head-to-head
accuracy and runtime comparisons against:

1. **Nugteren 2014 reuse-distance L2 model** —
   `references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf`.
   Implement a clean-room version of the model using the same HRD profile
   inputs the framework already collects. Compare on the holdout cohort.
2. **Accel-Sim cache hit-rate output** —
   `references/papers/knowledge_base/active/2020__accel_sim_an_extensible_simulation_framework_for_validated_gpu_modeling.pdf`.
   This is the gold-standard cycle-accurate cache prediction in the prior
   art. Per CLAUDE.md Rule 5 the simulator itself stays out of scope,
   but consuming Accel-Sim's published validation numbers as a
   normalized baseline is in-scope. Document the comparison methodology
   carefully to satisfy the "head-to-head requires normalized
   workloads/metrics" caveat at `comparison_contract.toml:138`.
3. **MDM 2020** —
   `references/papers/knowledge_base/active/2020__mdm_the_gpu_memory_divergence_model.pdf`.
   Memory-divergence-aware model; comparison on the irregular and DL
   primitive workload families.
4. **Trivial baselines**: always-predict-mean, always-predict-last-level,
   uncorrected walker output (the framework's L2 prediction BEFORE the
   high-footprint correction). These set the "you must beat this"
   floor and quantify the corrections' contribution.

Three ablation tracks:

- **Ablate L2 high-footprint correction**: rerun the holdout evaluation
  with the high-footprint allowlist disabled. Documents the correction's
  contribution to the headline number.
- **Ablate L2 reduction-family correction**: rerun with the R47
  allowlist disabled.
- **Ablate per-slug factor dispatch**: rerun with the L1 subfamily
  per-slug dispatch collapsed back to the single-factor R46 model.

Three new metrics (in addition to rel_err):

- **MAPE** (Mean Absolute Percentage Error) across the holdout cohort.
- **Pearson correlation** between predicted and truth across cells.
- **Spearman rank correlation** between predicted and truth across cells
  (rank-preserving accuracy, less sensitive to scale).

Two new tests:

- `tools/test_baseline_comparison_runner.py`: exercises the baseline-runner
  driver on a synthetic-cohort fixture and asserts each baseline emits a
  comparable per-cell prediction.
- `tools/test_ablation_runner.py`: exercises each ablation switch
  independently and asserts the headline geomean changes consistently
  with the ablation's expected effect (e.g. disabling high-footprint
  correction must increase hostile_l2 geomean).

### Direction 6 (deferred, NOT IN THIS PLAN): cross-architecture validation

**Explicitly out of scope for this plan** per user direction at round-30
entry: the sm_120 RTX 5060 laptop is currently unavailable. This
direction is listed here so reviewers (and future plans) understand the
gap and the path forward, but no work in this plan depends on sm_120
evidence.

A future plan will pick up:

- Re-running the full Direction 2 cohort on sm_120.
- Calibrate-on-one / test-on-the-other and leave-one-architecture-out
  splits.
- A cross-arch transfer story closer to XAPP 2015's framing
  (`references/papers/knowledge_base/active/2015__cross_architecture_performance_prediction_xapp.pdf`).
- Per-arch covariate analysis (clock / thermal / power / async-copy
  semantics) using the existing read-only NVML covariate path.

The current plan's paper claim is **explicitly scoped to sm_86 (Ampere
GA102)**. The submission narrative acknowledges this scoping in the
abstract and frames it as a limit the work plans to remove in follow-up,
not as an assumption that the result generalizes.

## Hard Constraints

- All headline accuracy numbers are reported on the holdout set, NOT the
  calibration set.
- The L2 prediction scope is enforced by `comparison_contract.toml::scope.cache_layers = ["L2"]`.
- The cohort meets the publication-grade floor (≥10 per arch) and the
  paper-target floor (≥30 unique kernels) as defined in Direction 2.
- All correction factors, AC-5 thresholds, HARD-target bands are FROZEN
  before the holdout sweep; mid-sweep tuning is forbidden.
- Pre-existing R48 evidence-provenance discipline is preserved: post-commit
  cascade-recapture per BL-20260508; source_commit drift verifiers per
  `tools/local_labels/test_post_commit_source_commit_drift.py`;
  identity-tuple lockstep tests; byproduct manifest_pairing_index sidecars.
- ALGORITHM_VERSION at `workloads/simbench_bw/cache_predictor.py::ALGORITHM_VERSION = "smith_eklund_lru_v1"`
  remains unchanged for Directions 1, 2, 3, 5 (all corrections are
  post-hoc multiplicative adjustments + diagnostic surfaces; no new
  prediction algorithm). Direction 4 (optional, L1 model upgrade) would
  require a new ALGORITHM_VERSION; it is out of scope unless explicitly
  promoted.
- CLAUDE.md Rules 1–5 unchanged: no NCU on cloud/server GPUs (local-label
  exception at `data/local_labels/` for sm_86 + sm_120); no GPU-driver-root
  operations; no privileged CUPTI surfaces; no host-driver work; no
  simulator revival.

## Path Boundaries

- Direction 1 (re-scope): edits `workloads/simbench_bw/cache_predictor.py`
  (header doc block), `tools/local_labels/run_headline_cohort_predict_compare.py`
  (L1 diagnostic path), `README.md`, plus 3 new tests under `tools/`.
- Direction 2 (cohort expansion): writes new traces + labels under
  `data/local_labels/sweeps/sweep_<new_id>/` and
  `workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/`. No production
  predictor logic change. Adds 2 new tests.
- Direction 3 (locked protocol): writes `cohort_split.json`,
  `locked_protocol.md`, `holdout_evaluation_report.json` under
  `data/sm_86_predictor_accuracy_recovery_v2/`. Edits the headline runner
  to honor the holdout-blinding ContextVar guard. Adds 2 new tests.
- Direction 5 (baselines + ablations): adds a new
  `tools/baseline_runners/` subdirectory with one driver per baseline,
  consuming the existing HRD/sharing-locality/etc feature outputs.
  Reads only; emits comparison artifacts under
  `data/sm_86_predictor_accuracy_recovery_v2/baseline_comparison/`. Adds
  2 new tests.

Out of bounds: NVBit producer-side changes (already at the
post-R25 widened cap); CUPTI Activity collector changes; any
modification under `tools/nvbit_tracer/`; any production
ALGORITHM_VERSION bump (Direction 4 is optional and out of scope).

## Feasibility Notes

- Direction 1 effort: ~1–2 weeks. Mostly prose + 3 narrow tests.
- Direction 2 effort: ~3–4 weeks (live trace capture for 20+ new cells on
  sm_86 + NCU label generation in the local-label quarantine + sweep
  admission). Trace budget headroom is comfortable for most cells; a
  handful (l2_thrash_pointer_chase, register_spill_local_mem variants,
  global_atomic_histogram) may exceed the 400x NVBit slowdown cap and
  enter the typed-diagnostic path — that's expected and within the
  R25 cap-widening discipline.
- Direction 3 effort: ~1–2 weeks (split design + ContextVar blinding
  guard + protocol document + 2 tests).
- Direction 5 effort: ~3–4 weeks (3 baseline clean-room implementations
  + 3 ablation tracks + new metric computations + 2 tests).
- **Total Phase-A+B+C-without-cross-arch: ~8–12 weeks of focused work.**

The non-root infrastructure (CUPTI Activity collector, NVBit tracer,
read-only NVML reader, local-label driver, post-commit-recapture chain)
is already in place and exercised; Direction 2's cohort expansion is the
dominant time cost, and it scales linearly with cell count.

## Pending User Decisions

1. **Confirm Direction 4 (L1 hostile-cohort model upgrade) deferral**:
   keep L1 as diagnostic-only for this submission, or promote D4a / D4b
   to in-scope? Recommended: defer.
2. **Confirm cohort target size**: this plan aims for ≥30 unique kernels.
   The aggressive top-tier target is 40–60. Stop at 30 (faster, still
   meets reviewer minimum), or push for 40+? Recommended: target 30 with
   an option to push to 40 in a Direction-2 follow-up round.
3. **Confirm baseline set**: this plan implements Nugteren 2014, MDM 2020,
   plus trivial baselines and consumes Accel-Sim published numbers. Add
   GCoM 2022, Photon 2023, or a learned baseline (SynPerf / NeuroScalar)
   to the set? Recommended: defer additional baselines to a follow-up
   round; the proposed set is the minimum reviewer-credible coverage.
4. **Confirm submission venue tier**: this plan's deliverables make the
   work competitive at ASPLOS / HPCA after Phases A+B+C. ISCA / MICRO
   are reachable but would require Direction 6 (cross-arch) to land
   first. Confirm target venue.

## Out of Scope (Explicitly)

- sm_120 cross-architecture validation (Direction 6, deferred per user
  direction).
- NCU profiling on cloud / server GPUs (CLAUDE.md Rule 1).
- GPU-driver-root operations (Rule 2).
- Privileged CUPTI surfaces (Rule 3).
- Host-driver work (Rule 4).
- Simulator revival of nvasim / Accel-Sim / GPGPU-Sim source (Rule 5);
  consuming published validation numbers is in scope, building or
  validating a simulator is not.
- ALGORITHM_VERSION bump (would require a separate plan with a separate
  re-validation cycle; Directions 1, 2, 3, 5 are all post-hoc
  multiplicative + diagnostic and preserve `smith_eklund_lru_v1`).
- Real-application end-to-end wall-clock validation (the original
  `README.md` framing); this plan stays in cache-hit-rate prediction
  scope. A future plan can add the wall-clock layer once the L2
  predictor's coverage and cross-arch evidence are established.

--- Original Design Draft End ---
