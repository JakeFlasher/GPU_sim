# RTX 5060 sm_120 L2 Observability and Prediction 30-Cell Cohort Bounded Loop

## Goal Description

Land one bounded RLCR loop on the RTX 5060 laptop (sm_120 Blackwell consumer, GB206 die) that expands the sm_120 L2 cache evaluation cohort from the current 5 admitted cells to a TOTAL of 30 admitted sm_120 L2 cells, frozen before measurement and stratified across memory-behavior family buckets, with each cell validated against NCU ground-truth labels produced under the CLAUDE.md Rule 1 local-label exception. The loop closes at `Verified-live-sm_120` evidence status only; cross-arch promotion to `Cross-arch-verified` is queued for a follow-up loop once the RTX 3090 Ti (sm_86) host frees up.

The headline contribution is reframed from "L1+L2 cache predictor" to **"counter-free, non-root GPU L2 cache observability and hit-rate prediction on production Blackwell hardware, validated against NCU ground truth via the local-label exception."** The predictor itself stays counter-free at prediction time (no hardware counters, no driver root, no privileged CUPTI); NCU is used ONLY as the ground-truth oracle that the predictor's L2 hit-rate predictions are compared against, and ONLY through `tools/local_labels/` writing into `data/local_labels/`. L1 prediction is "implemented but not connected to the headline protocol" (per the existing `tools/cache_predictor/l1_walker.py` plus the headline driver `tools/local_labels/run_headline_cohort_predict_compare.py` invoking only `cp.predict_cache_hit_rate` for L2); L1 driver connection remains DEFERRED per the existing `workloads/simbench_bw/comparison_contract.toml` `cache_layers = ["L2"]` pin.

The constraint that the RTX 3090 Ti (sm_86) host and all cloud GPU providers are busy is absorbed: no AC requires sm_86 live evidence, sm_90 or sm_100 zero-shot, cloud collection, Accel-Sim execution, Nugteren 2014 head-to-head, MDM 2020 head-to-head, L1 driver connection, driver-root mutation, or privileged CUPTI use. The loop sustains the v8 / `sm120_only_decoupled_framework_plan.md` precedent that "sm_120 IS the mainline on the laptop" so the stop-hook receives unambiguous in-scope vs deferred signals and the Mainline Drift Circuit Breaker pattern observed in the v7 / M1 loops does not re-trigger.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. All criteria are binary-closeable in a single RLCR loop on the receiving laptop session.

- AC-1: Laptop-only sm_120 claim guard (offline-verifiable). The plan body plus a new claim-guard test assert no active comparison bundle, plan summary, test, or round summary asserts sm_86 live results, cross-arch promotion, Accel-Sim execution, Nugteren 2014 head-to-head, MDM 2020 head-to-head, L1 driver-connected predictions, sm_90 or sm_100 zero-shot, driver-root mutation, or privileged CUPTI use during this loop's evidence reporting. The guard is a CLAIM guard (scans active comparison bundles, plan summaries, tests, and round summaries) rather than a token-absence scan over code, so legitimate negative-test fixtures and historical reads remain permitted.
  - Positive Tests (expected to PASS):
    - The claim-guard test runs against all round-summary files emitted during the loop and asserts no forbidden numeric claim appears.
    - The claim-guard test runs against `data/comparison_bundles/<latest>/comparison_bundle.json` (or equivalent active path) and asserts `top_level_admission_scope == "target_arch=sm_120"` and `arches` lists only `sm_120`.
    - The active framework path (driver + comparison_runner + headline producer) exercises sm_120 only when `--target-arch sm_120` is specified.
  - Negative Tests (expected to FAIL):
    - A synthetic round-summary fixture containing the literal `"sm_86 live"` accuracy claim makes the claim-guard test fail.
    - A synthetic comparison-bundle fixture with `top_level_admission_scope == "cross_arch_require_both_arches"` makes the claim-guard test fail.
    - A synthetic round-summary fixture asserting a numeric Nugteren / Accel-Sim / MDM result makes the claim-guard test fail.

- AC-2: Materialized and frozen 30+10 sm_120 L2 candidate roster (offline-verifiable). Before any NCU measurement runs, a Round 0 task emits a materialization table covering all candidates. The frozen roster lives at `data/local_labels/sm120_cohort_expansion_roster.toml` with exactly 30 primary plus 10 replacement candidates. Each candidate row carries `slug`, `shape_point`, `family`, `family_bucket`, `level`, `arch_gate`, `split`, `binary_path`, `manifest_path`, `trace_json_path`, `roster_source`, and `candidate_order`. The candidate unit is `(slug, shape_point)`; the same `slug` may appear with different `shape_point` values to fill multiple cells.
  - AC-2.1: Schema validation. Every candidate row's `arch_gate` includes `"sm_120"`; `level == "L2"` for simbench_bw rows OR the row's `roster_source == "portable_lcd"` with an explicit `level = "L2"` cohort-roster override; `family` is drawn from the union of `family` field values present in `workloads/simbench_bw/roster.toml` and `workloads/portable_lcd/roster.toml`; `family_bucket` is one of {`memory_stress`, `tensor_compute`, `activation_norm`, `other`}; `split` is one of {`calibration`, `validation`, `holdout`}.
    - Positive: A schema-valid `sm120_cohort_expansion_roster.toml` admits via the new validator.
    - Negative: A roster row missing `family_bucket` is rejected.
    - Negative: A roster row with `arch_gate = ["sm_86", "sm_90"]` (no `sm_120`) is rejected.
  - AC-2.2: Diversity constraints. No `family_bucket` exceeds 35% of the 30 primary cells. Family-bucket minimums met: at least 6 cells in `memory_stress`, at least 6 cells in `tensor_compute`, at least 6 cells in `activation_norm`. Cells in `other` count toward the 30 but not toward any minimum.
    - Positive: A roster with 8 memory_stress + 8 tensor_compute + 8 activation_norm + 6 other admits.
    - Positive: A roster with 6 / 7 / 7 / 10 admits (each minimum met; max bucket = 10/30 ≈ 33% under cap).
    - Negative: A roster with 5 memory_stress + 8 tensor_compute + 8 activation_norm + 9 other is rejected (memory_stress minimum unmet).
    - Negative: A roster with 11 memory_stress + 6 tensor_compute + 6 activation_norm + 7 other is rejected (memory_stress over 35% cap).
  - AC-2.3: Split-count pinning. The 30 primary cells split exactly as 15 calibration + 8 validation + 7 holdout. Each split contains at least 1 cell from each of the 3 required family buckets so no calibration/validation/holdout slice is bucket-degenerate.
    - Positive: A roster with the exact 15/8/7 split and per-bucket-per-split minimums admits.
    - Negative: A 16/7/7 split is rejected.
    - Negative: A 15/8/7 split where the holdout slice has 0 memory_stress cells is rejected.
  - AC-2.4: Family-bucket mapping commit. A sibling file at `data/local_labels/sm120_cohort_family_buckets.toml` commits the mapping from raw `family` values (e.g., `gemm_hmma`, `dram_stream`, `layernorm_fwd`, `softmax`) to derived buckets (`tensor_compute`, `memory_stress`, `activation_norm`, `other`). Validator checks both raw `family` and derived `family_bucket` per row.
    - Positive: A row with `family = "gemm_hmma"` and `family_bucket = "tensor_compute"` admits.
    - Negative: A row with `family = "gemm_hmma"` and `family_bucket = "activation_norm"` is rejected (mapping mismatch).
  - AC-2.5: Roster-preservation pinning. The Round 0 contract pins BOTH the commit SHA at which the frozen roster lands AND a SHA256 content hash of the frozen roster file. Downstream rounds verify both before any measurement and reject when either drifts.
    - Positive: A Round-N contract that quotes the Round 0 commit SHA and SHA256 admits.
    - Negative: A Round-N contract where the file SHA256 has changed since Round 0 is rejected.

- AC-2.5: Pre-measurement buildability preflight (offline-verifiable). Before any NCU sweep starts, an offline preflight task runs against the AC-2 materialization table and proves for all 40 candidates (30 primary + 10 replacement): the `binary_path` exists on the receiving host (or builds cleanly via the candidate's `roster_source` Makefile), the `manifest_path` resolves, the expected `trace_json_path` is producible by the existing tracer toolchain, and `arch_gate` includes `"sm_120"`. The preflight emits a buildability-witness sidecar at `data/local_labels/sm120_cohort_buildability_witness.json` carrying per-candidate `(slug, shape_point, resolved_binary_path, binary_sha256_if_built, manifest_path, expected_trace_json_path, arch_gate, source_commit, preflight_tool_version)`. The sidecar is pinned in the Round 0 contract.
  - Positive Tests (expected to PASS):
    - For each of 40 candidates the buildability-witness sidecar has a non-empty `resolved_binary_path` or a Makefile invocation that built it during preflight.
    - The sidecar's `preflight_tool_version` matches the active preflight tool's `__version__` constant.
  - Negative Tests (expected to FAIL):
    - A candidate whose `binary_path` cannot be resolved or built makes the preflight fail and prevents AC-3 from starting.
    - A candidate whose `arch_gate` does NOT include `"sm_120"` makes the preflight fail.

- AC-3: 30 TOTAL sm_120 L2 admitted NCU label cells (live). After the AC-2.5 preflight passes, the operator runs the existing `tools/local_labels/sweep_repeatability_matrix.py` plus `tools/local_labels/label_driver.sh` against the frozen roster's primary candidates (with replacement-pool substitution allowed only via typed exclusion). At loop close, `data/local_labels/label_admitted.json` contains AT LEAST 30 cells where `arch == "sm_120"` AND the cell maps to a row whose `cache_layer == "L2"` (or is admitted-by-default through the portable_lcd registry's L2-trace path). Each NEW admission carries the current 17-field schema (`arch`, `binary_sha256`, `cache_hit_rate_l1tex_sector_pct`, `cache_hit_rate_l2_sector_pct`, `evidence_kind`, `kernel_signature_sha256`, `l1tex_label_status`, `n_repeats`, `ncu_metric_version`, `profiler_version`, `schema_version`, `shape_point`, `slug`, `source_commit`, `source_commit_override_from`, `stddev_l1tex_sector`, `stddev_l2_sector`). Pre-existing cells (the 43 cells already carrying 17 fields and the 6 cells carrying 16 fields) are preserved with their historical field shape; AC-3 does NOT migrate historical cells. `profiler_version` is pinned to the NCU 2026.1.x metric semantics declared at the comparison contract's `ncu_metric_version` field. Repeatability verdict via `tools/local_labels/repeatability_aggregator.py` with N>=5.
  - AC-3.1: Cell-count target. The total count of cells where `arch == "sm_120"` in `data/local_labels/label_admitted.json` is at least 30 at loop close.
    - Positive: The aggregator confirms 30 (or more) sm_120 cells admitted.
    - Negative: A loop close with 29 sm_120 admitted cells is rejected.
  - AC-3.2: NEW-cell schema. Every new admission carries all 17 fields and `arch == "sm_120"` and `cache_layer == "L2"` (resolvable from the candidate roster).
    - Positive: A schema validator emits zero schema-defect tokens for new admissions.
    - Negative: A new admission missing `n_repeats` is rejected before merge.
  - AC-3.3: Historical-cell preservation. The 44 sm_86 historical cells plus the 5 existing sm_120 cells are byte-equivalent to their pre-loop content at the corresponding ranges of `label_admitted.json`.
    - Positive: A byte-diff scanner over the relevant ranges emits zero deltas.
    - Negative: A test that mutates the existing `swiglu_fwd_bf16_sm80/trace_l2_accuracy_xl` cell value fails the preservation check.
  - AC-3.4: Profiler-version pinning. Every new admission's `profiler_version` matches the contract's pinned `ncu_metric_version` family (currently `ncu_2026.1.x`).
    - Positive: All new admissions emit `profiler_version` recognized by the version validator.
    - Negative: A new admission carrying `profiler_version = "ncu_2025.x"` is rejected.

- AC-4: 30 joinable sm_120 L2 predictor bundles with scalar-or-bounded headline threshold (live). For each admitted cell in AC-3, the existing headline driver `tools/local_labels/run_headline_cohort_predict_compare.py` produces a corresponding sm_120 NVBit-derived predictor bundle whose `manifest_pairing_index.arch == "sm_120"` and `cache_layer == "L2"`. The comparison runner `workloads/simbench_bw/comparison_runner.py` joins each predictor bundle to its NCU label cell on the contract join key (8 fields per `workloads/simbench_bw/comparison_contract.toml`: `slug`, `shape_point`, `binary_sha256`, `source_commit`, `kernel_signature_sha256`, `arch`, `ncu_metric_version`, `cache_layer`). Headline coverage is the count of cells whose comparison result kind is `scalar` OR `bounded` (typed-unavailable cells are tallied separately and excluded from headline coverage per the contract's `typed_unavailable_handling = "exclude"` pin).
  - AC-4.1: Bundle production. Every AC-3 admitted sm_120 cell has a matching predictor bundle pinned in the active comparison run.
    - Positive: The comparison run's bundle-input list count equals the admitted-cell count (>= 30).
    - Negative: A run with one fewer predictor bundle than admitted cells is rejected.
  - AC-4.2: Join-key drift. Zero cells emit `comparison_runner_join_key_unmatched` for join-key drift in any of the 8 contract fields.
    - Positive: The runner's exclusion tally for `comparison_runner_join_key_unmatched` is zero.
    - Negative: A run where two cells have mismatched `binary_sha256` between predictor bundle and label is rejected.
  - AC-4.3: Scalar-or-bounded headline threshold. `scalar_or_bounded_headline_cells_per_arch["sm_120"] >= 24` (allowing up to approximately 20% typed-unavailable cells; typed-unavailable cells are tallied but NOT counted toward headline coverage).
    - Positive: A run with 25 scalar/bounded sm_120 cells and 5 typed-unavailable sm_120 cells admits.
    - Negative: A run with 22 scalar/bounded sm_120 cells and 8 typed-unavailable sm_120 cells is rejected (below the 24 threshold).

- AC-5: sm_120-only L2 headline bundle (live). The comparison runner emits a single sm_120-only L2 headline bundle over the >=30 admitted cells. The bundle carries (at minimum): scalar geomean of per-cell relative errors per `comparison_contract.toml` `error_formula`, bootstrap CI at the contract-pinned `bootstrap_resamples = 2000` and `ci_confidence = 0.95`, per-arch admitted-cell count, per-arch scalar/bounded headline cell count, typed exclusion tallies per the contract's `per_arch_diagnostics_config.diagnostic_vocabulary`, AND `top_level_admission_scope == "target_arch=sm_120"` (matching the existing schema pattern at `workloads/simbench_bw/schemas/comparison_bundle.schema.json`; no schema migration required for this loop). The bundle reports BOTH a full-cohort geomean (primary "calibration sanity" number) AND a holdout-only geomean (the 7-cell generalization estimate per AC-2.3). Per the DEC-4 resolution, AC-5 closure requires only that the numbers be emitted with full provenance; there is no accuracy threshold on either geomean.
  - AC-5.1: Schema validation. The emitted bundle passes strict validation against `workloads/simbench_bw/schemas/comparison_bundle.schema.json`.
    - Positive: A jsonschema run with the bundle emits zero violations.
    - Negative: A bundle missing `bootstrap_ci_low` / `bootstrap_ci_high` for the headline geomean is rejected.
  - AC-5.2: Scope claim. `top_level_admission_scope == "target_arch=sm_120"` and the bundle's arch list contains exactly `sm_120`.
    - Positive: A bundle with the expected scope and arch values admits.
    - Negative: A bundle with `top_level_admission_scope == "cross_arch_any_arch"` is rejected.
  - AC-5.3: Dual-geomean reporting. The bundle carries both `full_cohort_geomean` and `holdout_geomean` keys (or equivalent), each with its own bootstrap CI half-width.
    - Positive: A bundle with both keys populated admits.
    - Negative: A bundle with only `full_cohort_geomean` and no `holdout_geomean` is rejected.
  - AC-5.4: Round-summary tag. The Round-N close summary for the round that emits the headline bundle explicitly tags the bundle as "sm_120-only L2; no cross-arch transfer claim; no head-to-head baseline claim".
    - Positive: The summary file contains this tag string (case-insensitive substring match against the canonical phrasing).
    - Negative: A summary without this tag is rejected by the claim-guard sub-test.

- AC-6: Calibration head v4 sanity with split-leakage scanner (live). The existing `fit_calibration_head.py` is invoked on the AC-2 calibration split (exactly 15 cells per AC-2.3) and emits `data/local_labels/calibration_head_v4.json` carrying `target_arch == "sm_120"`, training cohort size == 15, training timestamp, `source_commit`, and a SHA256 pin of the active `comparison_contract.toml`. The active framework path selects v4 when `--target-arch sm_120` is specified; v3 is explicitly retired from the active path; v1 and v2 remain on disk as historical artifacts but are NOT consumed by any active sm_120 code path. A separate evaluation sidecar `calibration_evaluation_v4.json` reports train/validation/holdout residual triples separately (15/8/7 split per AC-2.3).
  - AC-6.1: v4 emission with correct cohort size. `calibration_head_v4.json` carries `training_cohort_size == 15` and `target_arch == "sm_120"`.
    - Positive: The v4 metadata matches the expected fields.
    - Negative: A v4 file with `training_cohort_size == 30` is rejected.
  - AC-6.2: Split-leakage provenance scanner. A new provenance scanner reads the AC-2 frozen roster's split tags and asserts that NONE of the 8 validation cells nor the 7 holdout cells appears in v4's training-cell list.
    - Positive: The scanner's leakage tally is exactly zero.
    - Negative: A synthetic v4 file where a holdout cell appears in the training-cell list fails the scanner.
  - AC-6.3: Train/validation/holdout residual triple. `calibration_evaluation_v4.json` carries separate residual-statistics blocks for each of the 3 splits, each block including geomean residual, max residual, and per-cell residuals indexed by `(slug, shape_point)`.
    - Positive: The evaluation file's three residual blocks each contain N cells matching the split sizes (15/8/7).
    - Negative: An evaluation file missing the validation residual block is rejected.
  - AC-6.4: v3 retirement and v1/v2 historical-only confirmation. The active framework path (the headline driver, the comparison runner, the provenance scanner) selects v4; a provenance scanner asserts no active code path consumes v1, v2, or v3 calibration heads when `--target-arch sm_120`.
    - Positive: A grep over the active code path produces no v3/v2/v1 active consumption.
    - Negative: A code change that re-introduces v3 selection in the active path fails the scanner.

- AC-7: Deferred-baseline and L1 claim guard with closure-addendum re-statement (offline-verifiable). The same claim-guard family used in AC-1 is widened to also reject: any active comparison bundle / plan summary / test assertion / round summary asserting a Nugteren 2014 RDH numeric result, an Accel-Sim execution result, an MDM 2020 numeric result, an L1 driver-connected prediction (any cell where the predictor bundle's `cache_layer == "L1"` AND `evidence_kind == "live"`), a cross-arch sm_86 live result during this loop, or a sm_90 / sm_100 zero-shot prediction. The guard is a CLAIM guard, not a file-absence guard, so harmless documentation stubs (e.g., a paragraph describing future Nugteren work) are permitted. The loop cannot emit COMPLETE without a closure addendum re-stating that `workloads/simbench_bw/comparison_contract.toml` `prior_art_handling = "contextual_only"` and `cache_layers = ["L2"]` both remain active pins.
  - Positive Tests (expected to PASS):
    - The claim-guard test emits zero violations across all active round summaries and the headline bundle.
    - The closure addendum re-statement appears in the COMPLETE-emitting round summary with both pin quotes present.
  - Negative Tests (expected to FAIL):
    - A synthetic round-summary fixture asserting "our predictor beats Nugteren 2014 by 12% on holdout" fails the guard.
    - A synthetic round-summary fixture asserting an L1 driver-connected live prediction fails the guard.
    - A COMPLETE-emitting round summary that omits the closure-addendum pin re-statement is rejected.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The maximum acceptable implementation lands all seven AC closures with the following scope: a frozen 30-primary + 10-replacement candidate roster with all 40 candidates passing the AC-2.5 buildability preflight; 30 admitted sm_120 L2 cells with the full 17-field NEW-cell schema and pinned NCU 2026.1.x metric semantics; 24 or more scalar-or-bounded headline-comparable cells (typed-unavailable held under ~20%); a calibration head v4 trained on the exact 15-cell calibration split with full train/validation/holdout residual triples reported; a sm_120-only L2 headline bundle reporting BOTH the full-cohort geomean (primary calibration-sanity number) AND the 7-cell holdout-only geomean (generalization estimate) each with bootstrap CI; AC-1 and AC-7 claim guards covering all deferred-baseline + cross-arch + L1 + driver-root + privileged-CUPTI claim surfaces; and the closure-addendum re-statement of the `prior_art_handling = "contextual_only"` and `cache_layers = ["L2"]` pins.

### Lower Bound (Minimum Acceptable Scope)

The minimum acceptable implementation lands ALL seven ACs but admits the following relaxations: 30 admitted cells (the AC-3 cell-count target is binary; no relaxation here), 24 or more scalar-or-bounded headline cells (the AC-4 threshold is binary), the frozen roster admits with the 15/8/7 split per AC-2.3 (binary). The acceptable lower bound applies to LOOP TERMINATION SAFETY: if the AC-2 family-bucket minimums cannot be reached from the buildable pool because the available roster lacks enough memory_stress kernels, the loop EXPLICITLY ESCALATES to the operator with a typed defect token rather than silently relaxing the constraint. The minimum bundle reports the full-cohort geomean with bootstrap CI; the holdout-only geomean is reported as a value but its absence (e.g., if all 7 holdout cells admit typed-unavailable) is permitted with a typed diagnostic. No accuracy threshold gates AC-5 closure per DEC-4.

### Allowed Choices

Implementation decisions are constrained by the binding `comparison_contract.toml` pins, the CLAUDE.md rule set, and the buildable-only DEC-3 resolution. The choices below pin the load-bearing surfaces.

- Can use:
  - `tools/local_labels/sweep_repeatability_matrix.py` plus `tools/local_labels/label_driver.sh` for NCU label production on sm_120 under the local-label exception (CLAUDE.md Rule 1).
  - `tools/local_labels/run_headline_cohort_predict_compare.py` for predictor bundle production.
  - `workloads/simbench_bw/comparison_runner.py` for the headline bundle emission.
  - `fit_calibration_head.py` for the calibration head v4 training.
  - `tools/local_labels/repeatability_aggregator.py` for the N>=5 repeatability verdict.
  - `tools/local_labels/merge_admitted_cells.py` for cross-host-safe label merge.
  - `tools/nvbit_tracer/tracer_tool/tracer_tool.so` (built on sm_120 at HEAD; the live tracer used to produce predictor bundles).
  - Already-buildable simbench_bw L2 slugs (sm_120-gated) plus already-buildable portable_lcd slugs as the candidate pool.
  - Existing schemas (`workloads/simbench_bw/schemas/comparison_bundle.schema.json`, `data/local_labels/label_admitted.schema.json`).

- Cannot use:
  - NCU OUTSIDE `tools/local_labels/` (forbidden by CLAUDE.md Rule 1; static guard at `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` enforces this).
  - `nvidia-smi` for clock locking, persistence, power, MIG, or compute-mode changes (CLAUDE.md Rule 2).
  - CUPTI Events, Metrics, Range Profiling, PM Sampling, or PerfWorks (CLAUDE.md Rule 3).
  - Driver installation, DKMS, MOK, `modprobe nvidia`, or reboot workflows (CLAUDE.md Rule 4).
  - Simulator-based prediction (CLAUDE.md Rule 5).
  - Cross-arch sm_86 live evidence in this loop (the RTX 3090 Ti host is busy).
  - sm_90 / sm_100 zero-shot prediction in this loop.
  - Cloud or server GPU collection scripts in this loop.
  - Building new scaffolded simbench_bw L2 rows in this loop (per DEC-3 Buildable-only). Building new kernels is a separate out-of-scope loop.
  - Nugteren 2014 RDH baseline runner, Accel-Sim head-to-head, MDM 2020 head-to-head, or any `tools/baseline_runners/` implementation in this loop (deferred; AC-7 claim guard enforces).
  - L1 driver connection (connecting `tools/cache_predictor/l1_walker.py` to the headline driver), or re-widening `comparison_contract.toml` `cache_layers` to `["L1", "L2"]` (deferred; AC-7 claim guard enforces).

> **Note on Deterministic Design**: This plan pins the candidate count to 30 primary + 10 replacement, the split to exactly 15/8/7, the family-bucket minimums to >=6 each, the family-bucket cap to 35%, the scalar-or-bounded headline threshold to >=24, and the NCU metric to `lts__t_sector_hit_rate.pct` per the existing `comparison_contract.toml`. These quantitative targets are HARD REQUIREMENTS for AC closure (per the DEC-4 resolution that AC-5 only emits numbers, the geomean values themselves are NOT hard targets — only the structural artifacts are). The Allowed Choices reflect the existing infrastructure; reaching the binding ACs without using the existing pipeline would be a scope explosion and is out of scope.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The loop's high-level flow follows the verified sm120_only_decoupled_framework_plan precedent. Round 0 reads the frozen plan and the buildability witness; produces the materialization table from the union of `workloads/simbench_bw/roster.toml` (filtered to `arch_gate` includes `sm_120` AND `level == "L2"`) and `workloads/portable_lcd/roster.toml`; commits the frozen roster + family-buckets mapping + buildability witness; closes AC-2 and AC-2.5. Round 1 implements the AC-1 and AC-7 claim guards as host-only tests with synthetic fixtures; the claim guards stay green throughout the loop. Round 2 runs the AC-2.5 preflight; if it fails for any candidate, replace from the 10-cell replacement pool and re-run preflight. Rounds 3-N (operator-driven; expect 4-8 rounds) run the AC-3 NCU sweep against the frozen primary roster, 5-10 cells per round, emitting the per-launch label sidecars and the N>=5 repeatability verdict via `repeatability_aggregator.py`, then merging via `merge_admitted_cells.py` once each cell admits. Round N+1 produces the predictor bundles via `run_headline_cohort_predict_compare.py` and the comparison runner emits the full sm_120-only L2 headline bundle (full-cohort geomean + 7-cell holdout-only geomean), closing AC-4 and AC-5. Round N+2 trains the calibration head v4 on the 15-cell calibration split and produces the evaluation sidecar with train/validation/holdout residual triples, closing AC-6. Round N+3 emits the closure-addendum re-statement and runs the claim-guard family one final time, closing AC-7 and emitting COMPLETE.

The operator wall-clock for the 30-cell NCU sweep is approximately 3-10 hours total (the `tools/local_labels/SWEEP_RUNBOOK.md` budget for a narrowed 5-cell sweep is ~0.5-1.6 hours; extrapolating linearly for 30 cells yields ~3-10 hours with non-linear NCU-stall variance). The receiving session should sweep in 5-10-cell batches with the replacement pool available, rather than attempting a single 30-cell pass.

### Relevant References

- `CLAUDE.md` - Rules 1-5; local-label exception bounding (Rule 1); two-device development model.
- `MANIFEST.md` - Active repo scope.
- `workloads/simbench_bw/comparison_contract.toml` - Binding scope, join key, geomean rule, coverage gates, per-arch diagnostics vocabulary.
- `workloads/simbench_bw/comparison_runner.py` - Headline bundle producer; 7-field join logic; bootstrap CI emission; per-arch diagnostics tallies.
- `workloads/simbench_bw/schemas/comparison_bundle.schema.json` - Comparison bundle strict schema.
- `workloads/simbench_bw/schemas/cache_predictor.schema.json` - Predictor bundle schema.
- `data/local_labels/label_admitted.json` - Existing 49-cell admitted set; current sm_120 amended cohort (5 cells); will grow by >=25.
- `data/local_labels/calibration_head_v3.json` - Existing sm_120-trained PlattCalibrator on 5 cells; will be retired from active path in favor of v4.
- `tools/local_labels/sweep_repeatability_matrix.py` - Arch-agnostic sweep harness.
- `tools/local_labels/label_driver.sh` - The single NCU-invocation entry point; quarantine boundary.
- `tools/local_labels/repeatability_aggregator.py` - N>=5 repeatability verdict.
- `tools/local_labels/merge_admitted_cells.py` - Append-only cross-host merge with strict validation.
- `tools/local_labels/SWEEP_RUNBOOK.md` - Operator wall-clock and dry-run guidance.
- `tools/local_labels/run_headline_cohort_predict_compare.py` - Headline driver; emits L2 predictor bundles via `cp.predict_cache_hit_rate`; currently selects calibration head v3 for sm_120.
- `tools/cache_predictor/l1_walker.py` - L1 predictor implementation; remains disconnected from the headline driver per the L2-only contract pin.
- `tools/nvbit_tracer/tracer_tool/tracer_tool.so` - Live NVBit tracer built on sm_120 (sha256 `3ad7d0d2...` at HEAD `7cf475ed`).
- `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_5060_laptop/sm_120/trace_trace_00/` - Reference admitted live trace evidence proving end-to-end sm_120 NVBit ingestion works on the receiving host.
- `workloads/simbench_bw/roster.toml` - 80 rows; 79 scaffold; the L2 + sm_120 subset narrows further to roughly 28 buildable rows.
- `workloads/portable_lcd/roster.toml` - 15 rows; all currently buildable on sm_120; lacks `level` and `split` fields (cohort roster supplies via override).
- `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` - Existing NCU quarantine enforcement.
- `plans/sm120_only_decoupled_framework_plan.md` - The just-converged precedent plan whose `sm_120 IS the mainline on the laptop` framing this plan continues.
- `references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf`, `references/papers/knowledge_base/active/2020__accel_sim_an_extensible_simulation_framework_for_validated_gpu_modeling.pdf`, `references/papers/knowledge_base/active/2020__mdm_the_gpu_memory_divergence_model.pdf` - Deferred-baseline PDFs on disk; head-to-head implementation is out of scope per AC-7.

## Dependencies and Sequence

### Milestones

1. Milestone 1 — Roster, family buckets, buildability preflight, claim guards (offline-only):
   - Phase A: Materialize candidates from the two rosters; produce family-bucket mapping; draft cohort roster.
   - Phase B: Implement AC-2 schema validator (arch_gate, level, family, family_bucket, split) + AC-2.4 mapping validator.
   - Phase C: Commit `data/local_labels/sm120_cohort_expansion_roster.toml`, `data/local_labels/sm120_cohort_family_buckets.toml`. Pin commit SHA + content SHA256 in the Round 0 contract per AC-2.5.
   - Phase D: Implement AC-2.5 buildability preflight tool; run against all 40 candidates; emit `data/local_labels/sm120_cohort_buildability_witness.json`.
   - Phase E: Implement AC-1 and AC-7 claim guards as host-only pytest with synthetic fixtures. Both pass against the current `data/comparison_bundles/` state.

2. Milestone 2 — NCU label production (live, operator-driven):
   - Phase A: Operator runs `sweep_repeatability_matrix.py --candidates <primary-roster-slice>` for the first 5-10 candidates; emits per-launch sidecars under `data/local_labels/<slug>/sm_120/`.
   - Phase B: `repeatability_aggregator.py` emits N>=5 verdict per cell.
   - Phase C: `merge_admitted_cells.py` appends admitted cells to `label_admitted.json`. Per-cell `label_admitted.json` count advances.
   - Phase D: Repeat A-C until AC-3.1 cell count >= 30. Replacement-pool substitution permitted only via typed exclusion of a replaced primary cell (logged in the Round summary).
   - Phase E: Run AC-3 schema validator + AC-3.3 byte-equivalence scanner over `label_admitted.json` final state.

3. Milestone 3 — Predictor bundle production and headline emission (live):
   - Phase A: Run `run_headline_cohort_predict_compare.py` against the AC-3 admitted set; emit sm_120 predictor bundles under the active comparison-run directory.
   - Phase B: Comparison runner joins on the 8-field contract key; emits the headline bundle. Verify AC-4 join-key drift = 0 and `scalar_or_bounded_headline_cells_per_arch["sm_120"] >= 24`.
   - Phase C: Verify the bundle's `top_level_admission_scope == "target_arch=sm_120"`, dual-geomean (full-cohort + holdout-only) populated, schema-valid.
   - Phase D: Round summary tags the bundle per AC-5.4.

4. Milestone 4 — Calibration head v4 fit and evaluation (live):
   - Phase A: Invoke `fit_calibration_head.py` on the 15-cell calibration split; emit `calibration_head_v4.json`.
   - Phase B: Implement and run the AC-6.2 split-leakage provenance scanner; emit `calibration_evaluation_v4.json` with train/validation/holdout triples.
   - Phase C: Update the active framework path (driver + comparison_runner + headline producer) to select v4 when `--target-arch sm_120`; retire v3 from active path.

5. Milestone 5 — Closure addendum and COMPLETE:
   - Phase A: Re-run AC-1 + AC-7 claim guards on the final state.
   - Phase B: Emit closure-addendum re-statement in the COMPLETE-emitting round summary quoting `prior_art_handling = "contextual_only"` and `cache_layers = ["L2"]`.
   - Phase C: Emit COMPLETE.

Relative dependencies: Milestone 1 has no live-GPU dependency and gates Milestone 2 (no NCU sweep starts without AC-2.5 preflight admitting). Milestone 2 produces the labels that gate Milestone 3 (no predictor bundles without admitted labels). Milestone 3 produces the headline bundle that gates Milestone 4 (calibration fits against the AC-3 admitted set, not against arbitrary cells). Milestone 5 depends on Milestones 1-4 completing.

## Task Breakdown

Each task includes exactly one routing tag: `coding` (implemented by Claude) or `analyze` (executed via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Inventory buildable candidate (slug, shape_point) pairs from the two rosters; produce a materialization-table draft listing all candidates whose `arch_gate` includes `sm_120` AND (simbench_bw `level == "L2"` OR portable_lcd default). | AC-2 | coding | - |
| task2 | Codex review of the candidate inventory for diversity feasibility: do we have at least 6 buildable kernels in each of `memory_stress`, `tensor_compute`, `activation_norm`? Identify any family-bucket shortfalls that would block AC-2.2. | AC-2 | analyze | task1 |
| task3 | Commit `data/local_labels/sm120_cohort_family_buckets.toml` mapping raw `family` values to derived buckets. Implement AC-2.4 mapping validator. | AC-2.4 | coding | task2 |
| task4 | Commit the frozen `data/local_labels/sm120_cohort_expansion_roster.toml` with 30 primary + 10 replacement candidates, split 15/8/7, per-bucket-per-split minimums met. | AC-2 | coding | task3 |
| task5 | Implement AC-2 schema validator (arch_gate, level, family, family_bucket, split, family-cap 35%, family-bucket minimums >=6, exact split counts) + AC-2.5 roster-preservation pinning (commit SHA + content SHA256). | AC-2.1, AC-2.2, AC-2.3, AC-2.5 | coding | task4 |
| task6 | Implement AC-2.5 buildability preflight tool. Emit `data/local_labels/sm120_cohort_buildability_witness.json` per the AC-2.5 field schema. | AC-2.5 | coding | task4 |
| task7 | Run the buildability preflight against all 40 candidates; substitute from the replacement pool only via typed exclusion until all 40 admit. | AC-2.5 | coding | task6 |
| task8 | Implement AC-1 claim guard as host-only pytest with synthetic fixtures (positive + negative). Run against current `data/comparison_bundles/` state. | AC-1 | coding | task5 |
| task9 | Implement AC-7 claim guard family (no Nugteren / Accel-Sim / MDM numeric claim; no L1 driver-connected live prediction; no cross-arch sm_86 live; no sm_90/sm_100 zero-shot; closure-addendum pin re-statement). | AC-7 | coding | task8 |
| task10 | Codex review of claim-guard test coverage: are there round-summary or comparison-bundle fields that could legitimately carry these claims but are not covered by the guard's scan logic? | AC-1, AC-7 | analyze | task9 |
| task11 | Operator-driven NCU label sweep via `tools/local_labels/sweep_repeatability_matrix.py` against the frozen primary roster (10-cell batches). Each cell admits via `repeatability_aggregator.py` N>=5 verdict and merges via `merge_admitted_cells.py`. | AC-3 | coding | task7, task9 |
| task12 | Implement AC-3 schema validator (17-field NEW-cell schema; profiler_version pinning) + AC-3.3 historical-cell byte-equivalence scanner. | AC-3.2, AC-3.3, AC-3.4 | coding | task11 |
| task13 | Run `tools/local_labels/run_headline_cohort_predict_compare.py` against the admitted >=30 sm_120 cells; emit predictor bundles under the active comparison-run directory. | AC-4 | coding | task11 |
| task14 | Run `workloads/simbench_bw/comparison_runner.py` to emit the sm_120-only L2 headline bundle (full-cohort + holdout-only geomean). Verify AC-4.2 join-key drift = 0 and AC-4.3 scalar-or-bounded threshold >= 24. | AC-4, AC-5 | coding | task13 |
| task15 | Implement AC-5.3 dual-geomean schema check and AC-5.4 round-summary tagging hook. | AC-5.3, AC-5.4 | coding | task14 |
| task16 | Codex review of the emitted headline bundle for honest framing: does the bundle accurately represent the sm_120-only L2 scope without overclaiming cross-arch transfer or head-to-head baseline? | AC-5 | analyze | task15 |
| task17 | Invoke `fit_calibration_head.py` against the AC-2 calibration split (exactly 15 cells); emit `data/local_labels/calibration_head_v4.json`. | AC-6.1 | coding | task14 |
| task18 | Implement AC-6.2 split-leakage provenance scanner (validation + holdout cells must NOT appear in training-cell list). | AC-6.2 | coding | task17 |
| task19 | Emit `calibration_evaluation_v4.json` with train/validation/holdout residual triples per AC-6.3. | AC-6.3 | coding | task18 |
| task20 | Update the active framework path (driver + comparison_runner + headline producer) to select v4 when `--target-arch sm_120`; retire v3 from active path; implement AC-6.4 provenance scanner. | AC-6.4 | coding | task19 |
| task21 | Codex review of v4 calibration sanity: do the train/validation/holdout residual numbers tell a coherent story (no train < holdout < validation inversion)? Is v4 worse than v3 at the same headline metric, and if so, is the regression acceptable per the calibration-sanity-not-final-generalization framing? | AC-6 | analyze | task20 |
| task22 | Emit the closure-addendum re-statement quoting `prior_art_handling = "contextual_only"` and `cache_layers = ["L2"]` in the COMPLETE-emitting round summary. Re-run AC-1 + AC-7 claim guards on the final state. | AC-7 | coding | task21 |
| task23 | Codex final-state review: does the closure-addendum re-statement match the contract pins byte-for-byte? Does the COMPLETE-emitting round summary contain all required claim-guard-friendly tags? | AC-7 | analyze | task22 |

## Claude-Codex Deliberation

### Agreements

- Single bounded RLCR loop sized for the receiving sm_120 laptop session is the correct shape; multi-phase A/B/C planning re-triggers the v7/M1 stop-hook structural-stall pattern and is out of scope for this loop.
- The headline contribution wording "counter-free, non-root GPU L2 cache observability and hit-rate prediction on production Blackwell hardware, validated against NCU ground truth via the local-label exception" is the right framing (per the user's gen-plan precursor decision); it preserves the modeling contribution while making the systems constraint central.
- AC-3's cell-count target should be 30 TOTAL sm_120 cells (existing 5 count; ~25 new admissions required), NOT 30 NEW (which would mean 35 total).
- Typed-unavailable cells are tallied but NOT counted toward headline coverage per `comparison_contract.toml` `typed_unavailable_handling = "exclude"`.
- The 7-field join described in the draft is incorrect; the contract pins an 8-field join (including `cache_layer`).
- The candidate identity unit is `(slug, shape_point)`, not just `slug`; multiple shape points from one slug may fill different cells.
- Family-bucket mapping must be committed as a sibling file and validated both by raw `family` and derived `family_bucket`.
- portable_lcd cells need explicit `level = "L2"` and `split` overrides in the cohort roster since the portable_lcd source roster lacks these fields.
- AC-2.5 buildability preflight is necessary before any NCU sweep starts.
- Roster preservation requires BOTH commit SHA AND SHA256 content hash, not just one.
- Pinned 15/8/7 split is reasonable; alternatives (14/8/8 or 16/7/7) are statistically slightly different but not materially better.
- Claim guards (AC-1, AC-7) are better than token-absence scans because they avoid blocking legitimate negative-test fixtures and historical reads.

### Resolved Disagreements

- Cell-count semantics. Claude position: 30 new cells beyond the existing 5 (35 total). Codex position: 30 total. Resolution: 30 TOTAL per Codex; the existing 5 cells count toward AC-3; the implementation produces approximately 25 new admissions. Rationale: cleaner downstream split math; matches the contract's structural floors (5 smoke, 10 publication, 20 aspirational, 30 research-cohort).
- Typed-unavailable accounting. Claude position v1: typed-unavailable counts toward 30. Codex position: tallied but excluded. Resolution: AC-4.3 introduces a separate scalar-or-bounded headline threshold (>=24) that excludes typed-unavailable; AC-3 counts all admitted cells regardless of result kind. Rationale: matches the contract's `typed_unavailable_handling = "exclude"` and the schema's `headline_cells_per_arch` semantics.
- Scope-lock approach. Claude position v1: scan code paths for forbidden tokens (`arch == "sm_86"`, `target_arch = "sm_86"`). Codex position: claim guard (no active comparison bundle / plan summary / test / round summary asserts forbidden results). Resolution: claim guard wins. Rationale: avoids blocking legitimate historical-read or negative-test code; preserves the spirit of the scope lock without false positives.
- Baseline file-absence guard. Claude position v1: AC-7 fails if `tools/baseline_runners/` exists. Codex position: claim guard, not file-absence guard. Resolution: claim guard wins (harmless documentation stubs allowed). Rationale: same as above.
- Join-field count. Claude position v1: 7-field join. Codex position: 8-field per `comparison_contract.toml`. Resolution: 8 fields including `cache_layer`. Rationale: contract is binding; the draft's "7-field" wording was stale.
- top_level_admission_scope value. Claude position v1: "sm_120-only enum value". Codex position: `target_arch=sm_120` (the existing schema pattern). Resolution: `target_arch=sm_120` per Codex. Rationale: matches the existing schema; no schema migration required for this loop.

### Convergence Status

- Final Status: `converged`
- Convergence Rounds Executed: 2 (initial first-pass + 1 verification round)
- Codex Round 1: produced 7 REQUIRED_CHANGES; all incorporated into candidate plan v2.
- Codex Round 2: produced 0 REQUIRED_CHANGES; explicitly declared `PLAN_CONVERGENCE_STATUS=converged subject to the two carried user decisions`.

## Pending User Decisions

All four DECs surfaced during Codex's two-round convergence loop have been resolved by the operator before plan finalization. Recorded here for traceability.

- DEC-1: AC-3 cell-count semantics — 30 TOTAL vs 30 NEW (in addition to existing 5)?
  - Claude Position: 30 NEW (35 total).
  - Codex Position: 30 TOTAL (cleaner downstream split math; matches contract floors).
  - Tradeoff Summary: 30 TOTAL is cleaner and contract-aligned; 30 NEW makes the split math (15/8/7) inconsistent with the existing 5-cell amended cohort.
  - Decision Status: 30 TOTAL (operator confirmed during gen-plan precursor; Codex Round 1 review confirmed).

- DEC-2: Typed-unavailable cells count toward the 30 admitted, or scalar/bounded only?
  - Claude Position: Count toward 30.
  - Codex Position: Tallied but excluded from headline coverage (contract pin `typed_unavailable_handling = "exclude"`).
  - Tradeoff Summary: Counting typed-unavailable inflates the cohort count but breaks the geomean's defensibility. The two-threshold AC-3 + AC-4 resolution (30 admitted total; 24 scalar-or-bounded headline-comparable) honors both intuitions.
  - Decision Status: Typed-unavailable tallied but NOT counted toward headline; AC-4.3 separate threshold (>=24 scalar-or-bounded).

- DEC-3: AC-2 candidate roster — buildable-only vs carve out a small scaffold-build slice?
  - Claude Position: Carve out a small scaffold-build slice (3-5 of 30 slots) to guarantee the L2 memory-stress minimum is real.
  - Codex Position: Position A (buildable-only) is pragmatic for 3-5 week single-loop closure; Position B (carve-out) widens the engineering surface and re-triggers the stop-hook stall.
  - Tradeoff Summary: Buildable-only is fastest to close but L2 memory-stress minimum may be hard to satisfy; replacement pool absorbs gaps. Scaffold-build slice guarantees the family-bucket minimum but ~1 extra week.
  - Decision Status: BUILDABLE-ONLY (operator confirmed via AskUserQuestion during Phase 6). Buildable kernels: ~28 simbench_bw L2+sm_120 rows + ~15 portable_lcd rows ≈ 43 candidates; 30 primary + 10 replacement (40 total) fits within the buildable pool.

- DEC-4: AC-5 close target — accuracy threshold on holdout geomean vs emit numbers only?
  - Claude Position: Emit numbers only; AC-5 closes structurally without an accuracy gate.
  - Codex Position: Position A (emit numbers only) is loop-safe; Position B (require holdout geomean with CI + no typed-unavailable majority on holdout) gives a stronger binary signal but is sensitive to 1-2 unlucky cells in the 7-cell holdout.
  - Tradeoff Summary: Numbers-only is the calibration-sanity-not-final-generalization framing; an accuracy gate would force the loop to either tune the predictor (scope creep) or fail closure even when the structural work is correct.
  - Decision Status: EMIT NUMBERS ONLY (operator confirmed via AskUserQuestion during Phase 6). AC-5.3 requires both full-cohort and holdout-only geomeans + their CIs be reported, but no numeric threshold gates AC-5 closure.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, `task1`, `task23`, or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code: e.g., `sm120_cohort_buildability_witness.json`, `verify_split_leakage_v4()`, `claim_guard_no_cross_arch_assertion`, `family_bucket_validator`.

### Provenance and Discipline

- Every NEW commit landed by this loop carries `Source Rounds: <round-id>` in its commit message for trace-back to the round contract.
- The Round 0 contract pins both `git rev-parse HEAD` and `sha256sum data/local_labels/sm120_cohort_expansion_roster.toml`; downstream rounds verify both before any measurement.
- Per the `BL-20260507-arch-token-multi-surface-lockstep` discipline (in `.humanize/bitlesson.md` when present on the receiving host), any new `family_bucket` enum value lands in lockstep across the cohort roster + family-buckets mapping + validator + tests.
- Per the `BL-20260509-strict-admission-contract-first` discipline, AC-2 and AC-2.5 schema validators land test-first (synthetic positive + negative fixtures committed before the validator implementation).
- Per the `BL-20260508-post-commit-recapture-discipline`, every live-sweep round (Milestone 2 + 3 + 4 Phase A) records device + driver + CUDA toolkit + NVBit build sha256 + observed clocks/throttle state in the round summary.

### Cross-Host Survival Notes

- The `data/local_labels/sm120_cohort_expansion_roster.toml`, `data/local_labels/sm120_cohort_family_buckets.toml`, `data/local_labels/sm120_cohort_buildability_witness.json`, and all new admitted cells in `data/local_labels/label_admitted.json` are committed and survive via git.
- The built `tools/nvbit_tracer/tracer_tool/tracer_tool.so` and per-slug compiled binaries do NOT survive cross-host transport cleanly; if the loop pivots hosts mid-flight, rebuild per the `HANDOFF.md` toolchain discipline (`NVCC_CCBIN=g++-14` + `nvcc -ccbin=/usr/bin/g++-14`).
- The `.humanize/rlcr/<session-id>/` loop state is host-local; the receiving host MUST `rm -rf .humanize/rlcr/` before starting a fresh loop.

### Out-of-Scope Follow-Up Loops

The following loops are EXPLICITLY DEFERRED and are NOT part of this plan. They are recorded here so the receiving session does not accidentally widen scope:

1. Cross-arch sm_86 30-cell mirror loop (lifts the sm_120 cohort to `Cross-arch-verified` once the RTX 3090 Ti host is free).
2. Nugteren 2014 RDH baseline runner build loop (implements `tools/baseline_runners/nugteren2014_rdh_baseline.py`).
3. Accel-Sim contextual table loop.
4. MDM 2020 contextual table loop.
5. L1 hostile-cohort model upgrade loop (reuse-distance or ML head for hostile L1 cells; the L1 walker exists but is disconnected).
6. L1 driver connection loop + contract re-widening (`cache_layers` back to `["L1", "L2"]`).
7. sm_90 / sm_100 zero-shot cross-arch extrapolation loop.
8. Cloud / server collection bootstrap loop (Path 3 of the historical three-fold completion gate; revisit only after item 1 closes).

None of items 1-8 is in this loop's scope. AC-7's claim guard enforces this boundary.

### Honesty Discipline

- Every round summary explicitly tags the evidence status of each AC closure: `Offline-verified`, `Verified-live-sm_120`. NO summary tags any AC `Cross-arch-verified` or `Verified-live-sm_86` (cross-arch promotion is queued for the follow-up loop).
- The headline bundle is explicitly tagged "sm_120-only L2; no cross-arch transfer claim; no head-to-head baseline claim" per AC-5.4.
- The closure-addendum re-statement (AC-7) quotes the contract's `prior_art_handling = "contextual_only"` and `cache_layers = ["L2"]` pins verbatim.
- Per `BL-20260509-rlcr-recovery-round-bounded-scope`, the loop's round contracts treat every deferred item (cross-arch + baselines + L1) as an explicit DEFERRAL with a typed marker, so the stop-hook receives unambiguous in-scope vs deferred signals.

--- Original Design Draft Start ---

# sm_120 L2 Observability + Prediction 30-Cell Bounded Loop

## Original Idea

The active plan `plans/sm120_only_decoupled_framework_plan.md` closed
all 9 ACs (AC-D1..D9) at Round 4; Rounds 5-147 are steady-state
"closure addendum" phase with one closure class crossing the 120-flag
circuit-breaker threshold. The loop has converged: it is time to
pivot to a new RLCR loop whose mainline is **expanding the sm_120 L2
evaluation cohort from 5 admitted cells to 30 admitted cells, frozen
before measurement, stratified across memory-behavior families**.

**Central contribution re-scope (binding)**: the project's headline
contribution is reframed from "L1+L2 cache predictor" to
**"counter-free, non-root GPU L2 cache observability and hit-rate
prediction on production Blackwell hardware, validated against NCU
ground truth via the CLAUDE.md Rule 1 local-label exception."** The
predictor itself stays counter-free at prediction time (no hardware
counters, no driver root, no privileged CUPTI); NCU is used ONLY as
the ground-truth oracle that the predictor's L2 hit-rate predictions
are compared against, and ONLY under the local-label exception
(`tools/local_labels/` + `data/local_labels/`), on local devices
(RTX 5060 laptop sm_120; RTX 3090 Ti sm_86 when available). L1 is
explicitly "implemented but not connected to the headline protocol"
(per `tools/cache_predictor/l1_walker.py` existing + the headline
driver `tools/local_labels/run_headline_cohort_predict_compare.py`
calling only `cp.predict_cache_hit_rate` for L2); L1 connection
remains deferred per the comparison contract's existing L2-only pin
at `workloads/simbench_bw/comparison_contract.toml:135`.

**Constraint absorbed**: the next loop executes entirely on the RTX
5060 laptop (sm_120). The RTX 3090 Ti (sm_86) host is busy and
cloud GPU providers are busy. Cross-arch live evidence is unavailable
for this loop. Evidence status is bounded to
`Verified-live-sm_120` (per CLAUDE.md "Two-Device Local Development
Model" item 3); no AC may claim `Cross-arch-verified` or require
sm_86 / sm_90 / sm_100 / cloud / Accel-Sim execution.

The loop is a **single bounded RLCR loop sized for ~3-5 weeks**, NOT
a multi-phase Phase A/B/C plan. Multi-phase planning is explicitly
out of scope to avoid the v7/M1 structural-stall pattern where the
stop-hook treats deferred items as incomplete and fires the Mainline
Drift Circuit Breaker. Cross-arch (sm_86) re-validation, Nugteren
2014 RDH baseline runner build, Accel-Sim head-to-head, MDM 2020
head-to-head, full ablation harness, and L1 hostile-cohort ML head
upgrade are all explicitly DEFERRED to follow-up loops, recorded as
"deferred-baseline guard" ACs so the loop's review surfaces receive
unambiguous in-scope vs deferred signals.

## Primary Direction: Single Bounded 30-Cell sm_120 L2 Cohort Loop

### Rationale

The current 5-cell sm_120 admitted set (`layernorm_fwd_bf16_sm80`,
`rmsnorm_fwd_bf16_sm80`, `silu_fwd_bf16_sm80`,
`softmax_safe_bf16_sm80` at `timing_shape` +
`swiglu_fwd_bf16_sm80` at `trace_l2_accuracy_xl`) sits at exactly
`comparison_contract.toml:119`'s
`smoke_test_floor_per_arch = 5` — the minimum that lets the
comparison runner emit a scalar geomean rather than the
smoke-test-floor diagnostic. It is NOT at the publication-grade
floor of `coverage_gate.publication_grade_floor_per_arch = 10`
(`comparison_contract.toml:120`), nor at the aspirational
`aspirational_target_total = 20` (line 121), nor at the
larger 30-cell research-cohort target that this loop pursues.

Naive expansion to "30 admitted cells" gives the wrong incentive:
adding any 5 more activation-family cells trivially meets the
publication floor without exercising the predictor against
memory-behavior diversity. The acceptance criterion shape must
require **stratified diversity** (per Codex red-team 2026-05-26):
30 primary + 10 replacement cells, frozen before measurement, with
family caps and minimum cell counts per family. This both prevents
floor-gaming and bounds the loop below the 90-slug naive upper
bound (95 distinct slugs across `workloads/simbench_bw/roster.toml`
and `workloads/portable_lcd/roster.toml`).

The infrastructure to execute this loop is already on disk and
verified through 146 rounds of the prior plan:

  * **NCU label production**: `tools/local_labels/` driver +
    `sweep_repeatability_matrix.py` (already arch-neutral with
    `sm_120 → sm_120-laptop` mapping at line 873) +
    `label_driver.sh` (arch-capability gate validates
    `sm_120-laptop` without code change at lines 213-237) +
    `repeatability_aggregator.py` (schema-agnostic to arch) +
    `merge_admitted_cells.py` (append-only cross-host merge with
    strict schema validation).
  * **Predictor bundle production**: the headline driver
    `tools/local_labels/run_headline_cohort_predict_compare.py`
    already emits L2 predictor bundles via
    `cp.predict_cache_hit_rate` and currently selects the
    sm_120-trained `calibration_head_v3.json` PlattCalibrator
    when `--target-arch sm_120` is specified (lines 83-100).
  * **Comparison runner**: `workloads/simbench_bw/comparison_runner.py`
    consumes `comparison_contract.toml`, joins on the 7-field key
    (lines 62-70 of the contract), emits scalar geomean over
    admitted cells with bootstrap CI when smoke-test floor is met.
  * **NVBit live trace evidence**: `tools/nvbit_tracer/tracer_tool/tracer_tool.so`
    builds on sm_120 (sha256 `3ad7d0d2...` at HEAD `7cf475ed`);
    `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_5060_laptop/sm_120/trace_trace_00/`
    carries an admitted `dynamic_trace.pb` proving end-to-end
    sm_120 NVBit ingestion works on this host.
  * **Calibration head training**: `fit_calibration_head.py`
    already trains the sm_120-only PlattCalibrator
    `calibration_head_v3.json` from the 5-cell training pool;
    the same path supports retraining as the cohort grows.
  * **Stop-hook structural-stall mitigation**: the prior
    `sm120_only_decoupled_framework_plan.md` proved the
    "sm_120 IS the mainline on the laptop" framing closes
    cleanly (R4 ADVANCED on all 9 ACs); no host-constrained
    deferrals in the loop's AC set.

### Approach Summary

A single RLCR loop on the RTX 5060 laptop closes at
`Verified-live-sm_120` by landing seven binary-closeable ACs:

1. **AC-1 Laptop-only scope lock** (offline-verifiable). The plan
   body + a new static-guard test asserts every loop AC requires
   ONLY `Verified-live-sm_120` evidence; no AC requires sm_86,
   sm_90, sm_100, cloud, Accel-Sim execution, driver-root, or
   privileged CUPTI. The static guard scans this plan file +
   any new code paths for forbidden tokens
   (`Cross-arch-verified`, `arch == "sm_86"` in NEW code,
   `target_arch = "sm_86"` in NEW code, cloud-arch admission,
   etc.). Sanctioned by CLAUDE.md "Two-Device Local Development
   Model" item 3.

2. **AC-2 Frozen 30-cell sm_120 candidate roster** (offline-verifiable).
   A new committed roster at
   `data/local_labels/sm120_cohort_expansion_roster.toml` defines
   30 primary + 10 replacement sm_120 L2 cells BEFORE measurement,
   with family tags (`memory_stress`, `tensor_compute`,
   `activation`, `reduction`, `normalization`, `gemm`,
   `attention_subkernel`, `bandwidth_microbench`) and split tags
   (`calibration`, `validation`, `holdout`). Schema validates;
   no family exceeds 35% cap; family-minimum constraints met
   (>=6 memory-stress, >=6 tensor/compute-heavy,
   >=6 activation/reduction/normalization). Replacements may
   only be substituted with a typed exclusion reason. The frozen
   roster's commit SHA is pinned in the round-1 contract so
   downstream rounds cannot silently mutate it.

3. **AC-3 30 admitted sm_120 L2 NCU labels** (live, operator-driven).
   Using ONLY the local-label exception under
   `tools/local_labels/` + `data/local_labels/`,
   `data/local_labels/label_admitted.json` reaches >=30 sm_120 L2
   cells (in addition to the existing 5 sm_120 + 44 sm_86
   historical cells). Each new cell carries:
   * 7-field join key per `comparison_contract.toml:62-70`
     (slug, shape_point, binary_sha256, source_commit,
     kernel_signature_sha256, arch, ncu_metric_version, cache_layer)
   * `arch = "sm_120"` + `cache_layer = "L2"`
   * `ncu_metric_version` matching the contract's
     `ncu_metric_version = "ncu_2026.1.x"` (line 45)
   * `profiler_version` provenance (NCU 2026.1.1.0 build 37634170
     per Round 146 device record)
   * Repeatability verdict via
     `tools/local_labels/repeatability_aggregator.py` N>=5
     (per prior loop's `BL-20260509-strict-admission-contract-first`)
   * NO `.ncu-rep*` artifact outside the
     `data/local_labels/` quarantine.

4. **AC-4 30 joinable sm_120 L2 predictor bundles** (live).
   Each admitted cell has a corresponding sm_120 NVBit-derived
   predictor bundle with
   `manifest_pairing_index.arch = "sm_120"`,
   `cache_layer = "L2"`, and a successful join against its
   NCU label cell on the 7-field key. Built via the existing
   `tools/local_labels/run_headline_cohort_predict_compare.py`
   driver; bundle production exercises the live sm_120 NVBit
   tracer at `tools/nvbit_tracer/tracer_tool/tracer_tool.so`.
   Zero cells admit as `comparison_runner_join_key_unmatched`
   for a join-key drift reason; cells that admit via
   `predictor_typed_unavailable` count toward the 30 if their
   typed reason is one of the contract's
   `per_arch_diagnostics_config.diagnostic_vocabulary` values
   (`comparison_contract.toml:152-157`).

5. **AC-5 sm_120-only L2 headline bundle** (live).
   The comparison runner
   (`workloads/simbench_bw/comparison_runner.py`) emits a
   sm_120-only L2 headline bundle over the >=30 admitted cells.
   The bundle reports:
   * scalar geomean of per-cell relative errors
     (`comparison_contract.toml:84` formula)
   * bootstrap CI at
     `bootstrap_resamples = 2000` /
     `ci_confidence = 0.95` (lines 99-100)
   * per-arch admitted cell count
     (>=30 for sm_120; 0 for any other arch in
     the active path)
   * typed exclusion tallies per
     `per_arch_diagnostics_config.diagnostic_vocabulary`
   * `top_level_admission_scope` MUST NOT include
     `cross_arch_*` enum values.
   The bundle's wire format passes
   `workloads/simbench_bw/schemas/comparison_bundle.schema.json`
   strict validation. The summary explicitly tags the bundle as
   "sm_120-only L2; no cross-arch transfer claim".

6. **AC-6 Calibration sanity, NOT final generalization** (live).
   Refresh the sm_120 PlattCalibrator head against the new
   `calibration` split (predeclared via AC-2 tags). Emit a new
   `data/local_labels/calibration_head_v4.json` with metadata:
   * training cohort size (must equal AC-2 `calibration` split
     count; no holdout/validation leakage)
   * `target_arch = "sm_120"`
   * training timestamp + source_commit
   * comparison_contract.toml SHA256 pin
   * train/validation/holdout residual triples per the AC-2 split.
   A provenance scanner enforces that the active framework path
   (driver + comparison_runner + headline producer) selects
   v4 when `--target-arch sm_120` is specified; v1/v2 sm_86 heads
   remain on disk as historical artifacts but are NOT consumed
   by the active sm_120 path. Binary close requires:
   * no train/test leakage (provenance scanner asserts no
     `holdout`-tagged cell appears in the training set)
   * train/validation/holdout residual triple is reported
   * the v3 head is explicitly retired from the active path
     (provenance scanner asserts no consumer references
     v3 when v4 is present).

7. **AC-7 Deferred-baseline + L1 guard** (offline-verifiable).
   A new plan-body + static-guard test records the following
   as DEFERRED for this loop, with explicit "do not implement
   in this loop" markers:
   * Nugteren 2014 RDH baseline runner
     (`tools/baseline_runners/nugteren2014_rdh_baseline.py`
     remains absent)
   * Accel-Sim head-to-head
   * MDM 2020 head-to-head
   * L1 driver connection
     (`tools/cache_predictor/l1_walker.py` stays disconnected)
   * `tools/ablation/` directory (stays absent)
   * Cross-arch sm_86 re-validation
   * sm_90 / sm_100 zero-shot
   The guard fails the loop if any of the above is introduced.
   Closure addendum guard: the loop cannot exit COMPLETE
   without an explicit re-statement that
   `comparison_contract.toml:138`
   `prior_art_handling = "contextual_only"` remains the active
   contract pin and no contradicting head-to-head claim is in
   the round summaries.

### Objective Evidence

- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/CLAUDE.md` rules 1-5 (no NCU on cloud/server; local-label exception under `tools/local_labels/` + `data/local_labels/`; no driver-root; no privileged CUPTI; no host-driver work; no simulator revival)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/plans/sm120_only_decoupled_framework_plan.md` (current plan; all 9 ACs closed at R4; loop converged R5-R147)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/workloads/simbench_bw/comparison_contract.toml:119-122` smoke_test_floor_per_arch=5, publication_grade_floor_per_arch=10, aspirational_target_total=20, require_both_arches_for_cross_arch=true
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/workloads/simbench_bw/comparison_contract.toml:135` cache_layers=["L2"] pin (narrowed back from ["L1","L2"] at R10 commit 75f7f8ab6c)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/workloads/simbench_bw/comparison_contract.toml:138` prior_art_handling="contextual_only"
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/workloads/simbench_bw/comparison_contract.toml:62-70` 7-field join key
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/data/local_labels/label_admitted.json` schema_version 4, currently 5 sm_120 + 44 sm_86 cells
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/data/local_labels/calibration_head_v3.json` sm_120-trained PlattCalibrator on 5 cells (training timestamp 2026-05-25T04:05:10Z; supersedes v1/v2)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/tools/local_labels/run_headline_cohort_predict_compare.py:83-100` v3 head selection on sm_120
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/tools/local_labels/sweep_repeatability_matrix.py:873` sm_120 → sm_120-laptop arch mapping (arch-neutral sweep harness)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/tools/local_labels/label_driver.sh:213-237` arch-capability gate validates sm_120-laptop without code change
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/tools/local_labels/merge_admitted_cells.py` append-only cross-host merge with strict schema validation
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/tools/local_labels/SWEEP_RUNBOOK.md:333-373` wall-clock budget ~0.5-1.6 hr per 5-cell narrowed sweep on the laptop with dry-run capability (extrapolates to ~3-10 hr operator wall-clock for a 30-cell sweep)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/tools/nvbit_tracer/tracer_tool/tracer_tool.so` sha256 `3ad7d0d284160783ee927d0a879d7050fb1ec737fdb955774d98c6cef45bca06` at HEAD `7cf475ed` (live on sm_120; R146 VMM partial-unmap fix landed)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/workloads/simbench_bw/results/dram_stream_stride_mix/rtx_5060_laptop/sm_120/trace_trace_00/dynamic_trace.pb` admitted (proves end-to-end sm_120 NVBit ingestion)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/workloads/simbench_bw/roster.toml` + `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/workloads/portable_lcd/roster.toml` together provide ~95 distinct slugs available for cohort expansion (5 currently admitted; ~90 not yet admitted)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf` Nugteren 2014 PDF on disk (no runner code; deferred per AC-7)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/references/papers/knowledge_base/active/2020__accel_sim_an_extensible_simulation_framework_for_validated_gpu_modeling.pdf` Accel-Sim 2020 PDF on disk (deferred per AC-7)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/references/papers/knowledge_base/active/2020__mdm_the_gpu_memory_divergence_model.pdf` MDM 2020 PDF on disk (deferred per AC-7)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/tools/cache_predictor/l1_walker.py` L1 walker implemented but disconnected from headline driver (deferred per AC-7)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/plans/tier2_rd_calibration_cross_arch_plan.md` sm_120 + sm_100 cross-arch scope explicitly DEFERRED to follow-up loop (`Tier-2-cross-arch`)
- `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler/.humanize/skill/2026-05-26_15-10-52-59776-94b2a9bd/output.md` Codex red-team response (2026-05-26 15:10 UTC) confirming bounded single-loop scope, deferring Nugteren/Accel-Sim/MDM/L1, requiring stratified cohort diversity not just count

### Known Risks

- **Operator wall-clock for a 30-cell NCU sweep**: extrapolating
  the 5-cell narrowed sweep budget of ~0.5-1.6 hr
  (`SWEEP_RUNBOOK.md:333-373`) linearly gives ~3-10 hr operator
  time for 30 cells, but kernel-specific NCU stalls (per the
  prior failed cell `fp32_fma_dep_chain/trace_00` on sm_86,
  and the more recent sm_120 amendment cohort) can extend
  any single cell to multiple hours. **Mitigation**: AC-2's
  10-cell replacement pool absorbs failed cells via typed
  exclusion reasons; AC-3 closure requires >=30 admitted, not
  the entire roster, so operator can substitute from the
  replacement pool without re-opening the frozen roster.
- **Family-cap constraint may not be satisfiable from the
  available roster**: the 30/35%/family-minimum constraint
  (AC-2) requires the laptop session to identify >=6 cells in
  each of three family categories. If the available 95 slugs
  do not have enough memory-stress or tensor-compute kernels
  to clear the minimums, AC-2 cannot close. **Mitigation**:
  the round-0 contract enumerates candidate slugs per family
  using the existing roster contents; if any family is
  short, escalate to the user for either family redefinition
  (broaden the tag) or kernel addition (a separate
  out-of-scope loop adds new portable_lcd kernels). The
  loop does NOT silently relax the constraint.
- **Stop-hook structural-stall trap**: the loop harness's
  stop-hook treated explicitly-deferred items as incomplete
  in the v7 (3090 Ti, sm_86-only scope) and M1 (5060 laptop,
  broader scope) loops, firing the Mainline Drift Circuit
  Breaker after 3 consecutive STALLED verdicts. **Mitigation**:
  this loop is sm_120-only on the laptop (matching the
  sm120_only_decoupled_framework_plan precedent that closed
  cleanly at R4 ADVANCED); the AC-7 deferred-baseline guard
  records cross-arch + Nugteren + Accel-Sim + MDM + L1 as
  EXPLICIT DEFERRALS so the loop's review surfaces receive
  unambiguous in-scope vs deferred signals.
- **Calibration retraining (AC-6) may invalidate the existing
  sm_120-only headline framing**: refitting the PlattCalibrator
  on a larger 30-cell training pool will produce a head whose
  hyperparameters differ from v3's 5-cell-trained values. If
  v4 has materially worse validation/holdout residuals than
  v3, the headline geomean will degrade. **Mitigation**:
  AC-6 requires reporting train/validation/holdout residuals
  separately; the round summary is allowed to report
  "headline geomean degraded from N% to N+M% after v4
  retraining" as a HONEST finding (this is calibration
  sanity, not final generalization claim). The contract's
  bootstrap CI absorbs the per-cell variance.
- **Cross-host evidence-merge invariance**: the 44 sm_86
  historical cells in `label_admitted.json` carry the
  pre-existing schema_version=4 format; the cohort expansion
  must preserve them byte-for-byte. **Mitigation**:
  `merge_admitted_cells.py` already validates strictly on
  both inputs (precedent from v3 R23 audit); AC-3 closure
  test asserts cell count = 44 (sm_86 historical, unchanged)
  + 5 (existing sm_120) + N>=30 (new sm_120) where N is the
  loop's new admissions count.
- **Contract drift risk**: tightening AC-2's diversity
  constraints may surface a gap in the comparison runner's
  typed-exclusion vocabulary (`per_arch_diagnostics_config.diagnostic_vocabulary`,
  `comparison_contract.toml:152-157`). Adding a new exclusion
  value requires a coordinated update across the contract,
  the runner, the schema, and the regression tests, per
  `BL-20260509-strict-admission-contract-first`. **Mitigation**:
  AC-3 closure test admits ALL new cells via the EXISTING
  vocabulary; if a new vocabulary value is needed, that
  becomes a separate slice with its own slice-by-AC commit
  ordering.

## Alternative Directions Considered

### Alt-1: 10-cell publication-grade floor only (Codex-fallback narrow scope)

- Gist: Land only the `publication_grade_floor_per_arch = 10`
  threshold (`comparison_contract.toml:120`); admit 5 new
  sm_120 cells beyond the existing 5 for a total of 10. No
  stratified diversity constraint; cohort grows from the
  existing activation family + 5 more arbitrary cells.
  Calibration head v4 trained on 10 cells; comparison runner
  re-emits headline.
- Why not primary: meets the publication floor trivially but
  does not exercise the predictor against memory-behavior
  diversity. The headline geomean would be biased by family
  over-representation (activations dominate). The next loop
  after this one would re-pay the diversity work plus
  re-train v4 → v5; better to land 30-cell stratified now.

### Alt-2: 30-cell unstratified expansion (operator-easy mode)

- Gist: Same 30-cell target as primary but without family-cap
  or family-minimum constraints. AC-2 reduces to "30 cells
  frozen before measurement"; AC-3 admits >=30 regardless of
  family distribution.
- Why not primary: floor-gaming risk per Codex red-team —
  the publication floor of 10 is "trivially met by adding
  any 5 more kernels." 30 is also trivially met without
  diversity. The headline geomean's defensibility against
  reviewers asking "what does this cover?" depends on
  diversity, not just count.

### Alt-3: 30-cell stratified expansion + Nugteren 2014 RDH baseline runner build (Phase A+C combined)

- Gist: Primary's AC-1..6 + an additional AC-8 that builds
  `tools/baseline_runners/nugteren2014_rdh_baseline.py` from
  the references/papers/knowledge_base/active/2014__*.pdf
  paper, producing a side-by-side Nugteren-vs-our-predictor
  table for the 30-cell cohort.
- Why not primary: ~800-1500 LOC new baseline implementation
  in a single loop is the M1 / v7 scope-creep pattern. The
  comparison_contract.toml:138 `prior_art_handling = "contextual_only"`
  pin explicitly defers head-to-head until normalized
  workloads/metrics are established. Building the baseline
  runner inside the cohort-expansion loop re-triggers the
  structural-stall pattern. Deferred to a dedicated follow-up
  loop with its own RLCR contract scoped purely to baseline
  implementation.

### Alt-4: 30-cell stratified expansion + L1 driver connection (Phase A + L1 combined)

- Gist: Primary's AC-1..6 + an additional AC-8 that connects
  the existing `tools/cache_predictor/l1_walker.py` to the
  headline driver `tools/local_labels/run_headline_cohort_predict_compare.py`,
  widening `comparison_contract.toml:135` `cache_layers` back
  to `["L1", "L2"]` and re-emitting the headline over both
  layers.
- Why not primary: the L1 hostile-cohort catastrophe history
  (93-100% rel_err on hostile cells per
  `plans/l1_l2_accuracy_gate_closure_plan.md`) means
  re-widening the contract scope without first fixing the
  hostile cells reverts the R10 retrenchment for no benefit.
  The right sequencing is: land the L2 30-cell cohort first
  (this loop), then a separate loop adds L1 connection AFTER
  the L1 hostile-cohort model upgrade (Phase C optional
  Direction 4 per the user's original multi-phase framing).

### Alt-5: Multi-phase Phase A/B/C plan (user's original framing)

- Gist: Generate a single large plan covering Phase A
  (cohort expansion + scope alignment, ~5-8 weeks) + Phase B
  (cross-arch + locked protocol, ~5-8 weeks) + Phase C
  (baselines + ablations + L1 model upgrade, ~6-14 weeks).
  Downstream RLCR loops slice this plan into ~3-5 week
  chunks.
- Why not primary: Codex red-team (2026-05-26) explicitly
  pushed against this framing on grounds that multi-phase
  plans re-trigger the v7/M1 stop-hook structural-stall
  pattern when downstream phases (e.g., Phase B requiring
  sm_86) are deferred. The user confirmed the single
  bounded-loop scope choice via the gen-plan precursor
  prompt. Phase B (cross-arch sm_86) is unavailable in this
  session (sm_86 host busy); Phase C (baselines + L1) is
  scope creep for a single loop. Deferred to follow-up loops
  with their own RLCR contracts.

## Synthesis Notes

The primary direction is **the bounded single-loop subset of the
user's original Phase A** with three explicit refinements per
Codex red-team (2026-05-26): (a) the 30-cell expansion criterion
requires stratified diversity (family caps + family minimums), not
just count, to defeat the publication-floor-gaming risk and
preserve headline-geomean defensibility under reviewer scrutiny;
(b) the central contribution wording is sharpened from
"non-root GPU cache observability framework with an L2 prediction
model" (user's original) to **"counter-free, non-root GPU L2
cache observability and hit-rate prediction on production Blackwell
hardware, validated against NCU ground truth via the local-label
exception"** (user-approved framing), which preserves the modeling
contribution while making the systems constraint central and
explicitly invoking the local-label NCU exception that validates
the L2 hit-rate predictions; (c) Nugteren 2014 RDH baseline runner
build, Accel-Sim head-to-head, MDM 2020 head-to-head, L1 driver
connection, full ablation harness, and cross-arch sm_86
re-validation are all EXPLICITLY DEFERRED via AC-7's static guard
so the loop's review surfaces receive unambiguous in-scope vs
deferred signals. The seven ACs are sized for a single ~3-5 week
RLCR loop on the RTX 5060 laptop; combined operator wall-clock for
the 30-cell NCU sweep is ~3-10 hours (extrapolated from the
SWEEP_RUNBOOK 5-cell narrowed budget). Closure status is
`Verified-live-sm_120` only; cross-arch promotion to
`Cross-arch-verified` is queued for a follow-up loop once the
RTX 3090 Ti host frees up. The follow-up loop sequence, in
order, is: (1) cross-arch sm_86 30-cell mirror loop (lifts to
`Cross-arch-verified`); (2) Nugteren 2014 RDH baseline runner
build loop; (3) Accel-Sim contextual table loop; (4) MDM 2020
contextual table loop; (5) L1 hostile-cohort model upgrade loop
(reuse-distance or ML head); (6) L1 driver connection +
contract re-widening loop. None of (1)-(6) is in this loop's
scope.

--- Original Design Draft End ---
