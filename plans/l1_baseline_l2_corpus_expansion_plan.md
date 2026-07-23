# L1 Predictor Baseline And L2 Corpus Variance Unlock (sm_86 Follow-Up Loop)

> **Normative sections**: the Goal Description, Acceptance Criteria, Path Boundaries, Dependencies and Sequence, and Task Breakdown sections of this plan are AUTHORITATIVE. The "--- Original Design Draft Start ---" appendix at the bottom is a SUPERSEDED archive preserved for traceability (per the gen-plan command's draft-completeness rule); it may contain language that pre-dates the convergence-round revisions (e.g., hard-coded "192 KiB", `stddev_prediction_trajectory`, v1→v2 schema language, exactly-13-cells gating, memory_op-driven L1 classifier wording, 7-test-shape AC-8). Where the appendix conflicts with the normative sections above, the normative sections govern.

## Goal Description

Land Plan 1 of the two-plan recovery sequence for the cache hit-rate predictor: build a SASS-grounded L1 predictor baseline that emits scalar / bounded / typed_unavailable bundles for the cache_hit_rate_l1tex_sector metric; migrate the local-label admitted-cell schema to retain the L1 mean (currently only stddev_l1tex_sector survives aggregation, so today's pipeline cannot compute L1 relative error); extend the comparison runner to a layer-aware join path producing per-arch L1 and L2 diagnostics independently; admit 6 additional sm_86 L2 cells from the existing HEADLINE_COHORT_13 unlabeled set via the operator runbook; recompute the cross-cell prediction stddev and emit a calibration-decision artifact without applying calibration; design a no-leakage feature-discovery matrix for the silu correctness/timing missing-feature instance; and commit an interim diagnostic report identifying which cells are Plan-2 accuracy candidates vs physics-challenge cells. The whole loop runs on sm_86 only (RTX 3090 Ti); cross-arch carries forward to the Tier-2-cross-arch follow-up loop per the user's 2026-05-16 directive.

This plan does NOT apply calibration (Platt / Isotonic / Beta), does NOT add new L2 physics features (DVFS covariate / eviction-chaining / hit-under-miss queue — all deferred to Plan 2), does NOT regenerate the prior loop's final report (also Plan 2), and does NOT promote L1 set-associative refinement unless DEC-2 fires.

**HARD numerical requirements (per user 2026-05-17 decision; all 3 must hit for Plan 1 closure):**
- L2 geomean rel_err < 0.15 on the predictor-friendly bucket (8 cells; was 0.319 across the 7-cell uniform corpus). The predictor-hostile bucket (5 cells: carveout / persisting_window / prefetch_hint / warm_cold / register_spill) is reported separately but is NOT in scope for the <0.15 target — those cells are diagnostic-coverage by design.
- L1 geomean rel_err < 0.20 on the cells where SASS / cache-op provenance is available AND the L1 mean survived schema migration. The friendly-bucket scoping mirrors L2.
- Cross-cell L2 prediction stddev >= 0.10 over the expanded 13-cell cohort (Codex's trajectory rule: 4-6 new cells with delta_pred ~0.25-0.30 from the current ~0.956 cluster).

The honest ceiling: the user's ultimate goal is <10% rel_err for ALL L1 and L2 metrics. Codex's critical review establishes that target depends on Plan 2's physics work plus calibration application. Plan 1 sets the conditions under which Plan 2 can deliver: a layer-aware comparison path, a working L1 baseline, a variance-unlocked corpus, and a feature catalog — AND Plan 1 demonstrates that the corpus + L1 baseline can hit the 3 intermediate hard targets above. A round whose work lands all 9 ACs green but misses any of the 3 numerical targets does NOT close Plan 1; it instead emits typed_unavailable on the failing target and routes to a Plan-1.5 corpus-or-baseline-strengthening round (which may add more NCU cells, invoke DEC-2 set-associative promotion at the >50% trigger, or revisit AC-6 covariate decisions). This is materially different from the prior loop's "fail-closed with typed_unavailable is acceptable closure" stance — Plan 1's hard metrics mean typed_unavailable closure is NOT a Plan 1 success state.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. The 9 ACs split into two concurrent sub-streams plus shared infrastructure: sub-stream C (L1: AC-1, AC-2, AC-3, AC-8) and sub-stream A (L2 corpus + discovery: AC-4, AC-5, AC-6). AC-7 (report) depends on both; AC-9 (loop discipline) is cross-cutting.

- AC-1: SASS / cache-op provenance scaffold. A new per-PC instruction-cache-op classifier consumes cuobjdump output for each admitted-cell binary and emits an `instruction_cache_op` enum per program counter. Wire vocabulary uses underscore-separated tokens to keep schema validators simple: `LDG_E_CA` / `LDG_E_CG` / `LDG_E_CS` / `LDG_E_EF` / `LDG_E_STRONG_SM` / `LDG_E_STRONG_GPU` / `LDL` / `STL` / `LDGSTS` / `ATOM` / `TEX` / `OTHER`. Dotted SASS mnemonics (`LDG.E.CA` etc.) appear only as parser input examples; the wire / schema / test fixtures all use the underscore form. The trace sidecar gains a per-PC mapping; the predictor's L1 head fail-closes with typed_reason `predictor_l1_sass_provenance_missing` when the mapping is absent or the binary is not introspectable. The classifier is per-arch tolerant per the prior loop's sass_confirm.py precedent: sm_86 (Ampere GA102) recognizes `LDG_E_CA` / `LDG_E_CG` / `LDG_E_CS` / `LDG_E_EF` / `LDL` / `STL` / `LDGSTS` / `ATOM` / `TEX` / `OTHER` from live cuobjdump output. Blackwell-only tokens `LDG_E_STRONG_SM` and `LDG_E_STRONG_GPU` are covered by CANNED cuobjdump TEXT FIXTURES only in Plan 1 (live sm_120 coverage defers to the Tier-2-cross-arch follow-up loop).
  - Positive Tests (expected to PASS):
    - A synthetic kernel binary containing exactly one `LDG.E.CA` load returns a per-PC mapping with that PC tagged `LDG_E_CA` and `l1_eligible=true`.
    - A canned cuobjdump text fixture containing `LDG.E.STRONG.SM` returns the per-PC mapping with `LDG_E_STRONG_SM` and `l1_eligible=false` (bypass). Plan 1 does NOT require a real sm_86 binary to exercise this token.
    - An `LDGSTS` async-copy instruction is tagged `LDGSTS` and routes to the typed_unavailable arm at the L1 predictor head (no implicit L1 staging assumption).
    - A real sm_86 admitted-cell binary (silu_fwd_bf16_sm80, layernorm_fwd_bf16_sm80) parses without error and produces a non-empty per-PC mapping with every load-class PC tagged using the underscore-vocabulary tokens that the classifier supports on Ampere.
    - The `instruction_cache_op` enum in the new schema enforces the closed set of 12 underscore-separated tokens.
    - The `_TYPED_REASONS` frozenset in `workloads/simbench_bw/cache_predictor.py`, the `typed_reason.enum` in `workloads/simbench_bw/schemas/cache_predictor.schema.json`, and the `comparison_runner.py` accepted-bundle-shape rejection list all carry `predictor_l1_sass_provenance_missing` in lockstep (paired-control: removing the literal from any one of the three surfaces fires the lockstep test).
  - Negative Tests (expected to FAIL / be rejected):
    - A binary whose cuobjdump output cannot be parsed (corrupt SASS) is rejected with typed_reason `predictor_l1_sass_provenance_missing` AND a diagnostic in the bundle's `evidence.sass_provenance_status` field.
    - An invalid `instruction_cache_op` value (e.g., `FOO_BAR` or the dotted form `LDG.E.CA`) in the schema fixture is rejected by jsonschema validation. The wire format is the underscore form ONLY.
    - A bundle that emits a non-empty L1 hit-rate scalar arm WITHOUT a per-PC mapping is rejected as a contract violation by a paired-control test.
  - AC-1.1: Memory-space classification is preserved orthogonally. The new `instruction_cache_op` field coexists with the existing `memory_op_class` field (GLOBAL / SHARED / LOCAL / CLUSTER_DSMEM / ATOMIC); the classifier does NOT replace `memory_op_class` because they answer different questions (memory space vs cache operator).
    - Positive: a global load with LDG.E.CA carries `memory_op_class=GLOBAL` AND `instruction_cache_op=LDG_E_CA`.
    - Negative: any access whose `memory_op_class=SHARED` AND `instruction_cache_op=LDG_E_*` fires a defect (impossible combination — shared loads use LDS / LDSM mnemonics).

- AC-2: L1 capacity provenance + per-SM baseline model. The bundle records `l1tex_capacity_bytes`, `l1tex_capacity_source` (enum: `device_properties_manifest` / `cuda_func_attribute_query` / `cuda_device_attribute_query` / `static_card_constant`), `l1_shared_carveout_source` (enum: `device_properties_manifest` / `cuda_func_attribute` / `default`), `static_shared_bytes`, `dynamic_shared_bytes`, and `l1_model_class` (enum: `fully_associative_baseline` / `set_associative_refinement_pending`). **No L1 capacity constant is hard-coded into Plan 1 implementation or tests; every test fixture must source the value from a manifest, a CUDA-attribute query stub, or a `static_card_constant` table entry, and the provenance source MUST be recorded in `l1tex_capacity_source`. A bundle whose `l1tex_capacity_source` is missing or whose source is `static_card_constant` without a corresponding entry in the published static-card table is rejected.** The per-SM walker requires a CTA-to-SM scheduling assumption recorded as `cta_to_sm_schedule_policy` (enum: `round_robin_homogeneous` / `occupancy_derived` / `unknown`); when `unknown`, the L1 path emits typed_unavailable with reason `predictor_l1_cta_to_sm_schedule_unknown` (CTAs on different SMs do not evict each other's L1 lines, so guessing biases the prediction). The hit decision is sector-with-line-allocation: a hit fires when the 32B sector is in the per-SM LRU stack AND its parent 128B line is resident. Plan 1's L1 model is `fully_associative_baseline`; set-associative refinement is deferred to Plan 2 unless DEC-2 fires. Because the fully-associative baseline is known to potentially over-predict in conflict-heavy patterns, AC-8 includes an explicit set-conflict diagnostic fixture (see AC-8) that documents this without promoting set-associative modeling into Plan 1.
  - Positive Tests (expected to PASS):
    - A synthetic kernel with grid=8 blocks and `round_robin_homogeneous` scheduling distributes blocks evenly across the SM count derived from the test fixture's device-properties manifest, and the walker's per-SM LRU stacks correctly reflect the partition.
    - The bundle's `l1tex_capacity_bytes` field is populated from a manifest-sourced value (whatever the fixture provides — the test does NOT compare against a hard-coded constant); `l1tex_capacity_source` records the provenance enum verbatim.
    - The `l1_shared_carveout_source` field defaults to `device_properties_manifest` when the manifest is present.
    - A kernel with shared-memory carveout reduces the effective L1 capacity by the carveout amount; the carveout policy provenance is preserved in the bundle.
    - The `l1_model_class` field defaults to `fully_associative_baseline` and is enforced as an enum in the schema.
  - Negative Tests (expected to FAIL / be rejected):
    - A trace lacking grid_dim / block_dim metadata is rejected with typed_reason `predictor_l1_cta_to_sm_schedule_unknown`.
    - A bundle whose `l1tex_capacity_source` is missing OR whose source value is unrecognized is rejected at parse time.
    - A bundle whose `l1tex_capacity_bytes` exceeds the source manifest's recorded value (or the static-card-table maximum if `static_card_constant`) is rejected as a provenance-inconsistency defect.
    - A bundle whose `l1_model_class=set_associative_refinement_pending` AND emits a scalar L1 hit rate is rejected (the refinement-pending class must emit typed_unavailable or bounded).
    - The L2 path is byte-identical to the prior loop's predictor for all 7 currently-admitted cells (regression guard).

- AC-3: Layer-aware comparison path + admitted-label schema migration. The current `local_label_admitted.schema.json` is pinned at `schema_version=3`; AC-3 bumps it to `schema_version=4`. The v4 schema adds `cache_hit_rate_l1tex_sector_pct` (nullable, allowing absence per the L1-status semantic below) as a sibling of `cache_hit_rate_l2_sector_pct`, and adds `l1tex_label_status` (enum: `mean_present` / `mean_absent_label_missing` / `mean_absent_provenance_missing`). A cell whose v3 input lacked the L1 mean is migrated to v4 with `cache_hit_rate_l1tex_sector_pct=null` AND `l1tex_label_status` set to the matching absent-cause; the L1 comparison-runner path emits typed_unavailable for that cell. The L2 path continues to work unchanged on the migrated cells. `repeatability_aggregator.py` emits the L1 mean alongside the existing stddev (when per-repeat L1 values are populated; otherwise emits null + the appropriate `l1tex_label_status`). The bundle wire shape is per-layer: each predictor bundle carries a `cache_layer` enum (`L1TEX` or `L2`) plus an independent `result_kind` (scalar / bounded / typed_unavailable). The producer emits ONE bundle per (cell, cache_layer) pair, so an L1 typed_unavailable bundle and an L2 scalar bundle coexist for the same cell without ambiguity. The `comparison_runner.py` 7-tuple join expands to a `(slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256, arch, ncu_metric_version, cache_layer)` 8-tuple; per-arch L1 + L2 diagnostics emit in separate blocks mirroring the existing L2 structure. `comparison_contract.toml::cache_layers` widens from `["L2"]` to `["L1", "L2"]`. V3 admitted records (without the L1 mean OR `l1tex_label_status`) are rejected at parse time with typed reason `local_label_admitted_l1_mean_migration_required`, forcing the operator to re-aggregate.
  - Positive Tests (expected to PASS):
    - A migrated v4 admitted-cell record with `cache_hit_rate_l1tex_sector_pct` (non-null) and `l1tex_label_status=mean_present` validates against the v4 schema.
    - A migrated v4 admitted-cell record with `cache_hit_rate_l1tex_sector_pct=null` and `l1tex_label_status=mean_absent_label_missing` validates against the v4 schema; the L1 comparison-runner path emits typed_unavailable for that cell while the L2 path continues normally.
    - `repeatability_aggregator.py` on a fresh sweep emits both means with values in [0.0, 100.0] (percent units) when per-repeat L1 values are populated; emits `cache_hit_rate_l1tex_sector_pct=null` + `l1tex_label_status=mean_absent_label_missing` when L1 sidecar values are not present in the per-repeat data.
    - `comparison_runner.py` consuming the migrated admitted set + per-layer predictor bundles emits per-arch diagnostics with separate `l1tex` and `l2` rel_err stat blocks; each block reports `n_cells`, `n_scalar`, `n_bounded`, `n_typed_unavailable` alongside the geomean.
    - The `comparison_contract.toml::cache_layers` value is parsed as `["L1", "L2"]` and the runner refuses to emit an L1 diagnostic block when the contract pins `["L2"]` only (per-contract gating).
    - Per-layer predictor bundles co-exist for the same cell: an L1-layer typed_unavailable bundle (because SASS provenance missing) AND an L2-layer scalar bundle (because L2 has its existing predictor) both validate, and the comparison runner joins them on the 8-tuple including `cache_layer`.
  - Negative Tests (expected to FAIL / be rejected):
    - A v3 admitted-cell record (missing both the L1 mean field AND `l1tex_label_status`) is rejected at parse time with typed reason `local_label_admitted_l1_mean_migration_required`.
    - A v4 record where `cache_hit_rate_l1tex_sector_pct=null` AND `l1tex_label_status=mean_present` is rejected (contradiction: cannot be `mean_present` if value is null).
    - A bundle that emits `cache_layer=L1TEX` against a comparison contract pinned to `["L2"]` only is rejected.
    - A comparison runner attempting L1 join when the admitted set has only `stddev_l1tex_sector` (no mean, no `l1tex_label_status` v4 field) fires `comparison_runner_l1tex_label_mean_missing` BEFORE any join.
    - The L2 join continues to work on migrated cells (v4 with `cache_hit_rate_l1tex_sector_pct=null` is still L2-comparable), confirming the L1 absence does NOT regress L2 coverage.

- AC-4: L2 corpus expansion with cohort accounting. The operator runs `tools/local_labels/SWEEP_RUNBOOK.md` on local RTX 3090 Ti for up to 6 unlabeled HEADLINE_COHORT_13 cells (working_set_capacity_sweep, dram_stream_stride_mix, carveout_sensitivity_probe, warm_cold_pair_relaunch, l2_persisting_window_sweep, l2_prefetch_hint_sweep), all at shape_point=`trace_l2_accuracy_xl`. The acceptance threshold is **at least 4 of 6 cells admitted at n_repeats >= 5** (the same as the Lower Bound floor); any cells that fail label repeatability or fail the join contract are honestly excluded with a typed reason recorded in the cohort-accounting report. The resulting `label_admitted.json` carries AT MINIMUM 11 sm_86 cells (7 prior + 4 of 6 new); the Upper Bound is 13 cells (7 + 6). The interim report and the comparison runner stratify per-bucket geomeans (predictor_friendly vs predictor_hostile) using the R21 `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` partition. The <0.15 L2 target applies ONLY to the predictor_friendly bucket; the predictor_hostile bucket is reported separately as diagnostic coverage. AC-4's deliverable is a cohort-accounting artifact that reports per-cell admission_status (admitted / excluded_label_repeatability / excluded_join_key_unmatched / excluded_operator_skipped) with a typed reason for every exclusion.
  - Positive Tests (expected to PASS):
    - After AC-4 lands, `data/local_labels/label_admitted.json` contains at least 11 sm_86 cells whose `(slug, shape_point)` tuples are a subset of `_HEADLINE_COHORT_13`.
    - The cohort-accounting artifact records the admission_status for each of the 13 HEADLINE_COHORT_13 cells (no cell silently absent).
    - The comparison bundle's `per_arch_diagnostics.per_arch.sm_86.relative_error_stats_per_predictor_design_coverage.predictor_friendly` block has its `n_cells` field reflecting the actual admitted-friendly count.
    - The comparison bundle's `predictor_hostile` block has its `n_cells` field reflecting the actual admitted-hostile count.
    - Each admitted cell's L2-layer predictor bundle joins on the AC-3 8-tuple (slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256, arch, ncu_metric_version, cache_layer=L2); no `comparison_runner_join_key_unmatched` for any admitted cell. (Note: the legacy 7-tuple key is the AC-3 8-tuple's prefix for backward-compat against L2-only sweeps that predate the cache_layer field; AC-3 lockstep tests cover the transition.)
    - Each admitted cell's NCU sweep has n_repeats >= 5 (the admission floor).
  - Negative Tests (expected to FAIL / be rejected):
    - An admitted cell with stddev_l2_sector > 0.03 is excluded with `label_repeatability_failed` BEFORE reaching the comparison runner; the cohort-accounting artifact records the exclusion.
    - The L2 predictor-friendly geomean target (<0.15) is computed ONLY over the friendly bucket; hostile cells contribute zero to that specific number, and a paired-control test asserts the runner does NOT mix them.
    - A cell admitted at a different source_commit than the predictor sidecar fires `comparison_runner_join_key_unmatched` (no silent mapping).
    - A HEADLINE_COHORT_13 cell that is NOT in the cohort-accounting artifact (silently dropped) is rejected by an AC-4 scanner that walks the 13-cell set and asserts every cell has an admission_status entry.

- AC-5: Variance trajectory + calibration decision inputs (no application). After AC-4, recompute cross-cell L2 prediction stddev over the expanded 13-cell cohort and emit `data/sm_86_predictor_accuracy_recovery_v2/ac5_variance_trajectory_post_corpus_expansion.json` with fields: `cross_cell_stddev` (measured), `documented_floor_stddev=0.10`, `precondition_status` (`satisfied` / `not_satisfied`), `heuristic_stddev_projection` (Codex's `delta_pred / sqrt(n+1)` rule applied as a heuristic projection, not a physical formula), `loocv_inputs` (per-cell predictions + truth for Plan 2 calibration LOOCV fit), and `calibration_application_verdict` (`deferred_to_plan_2_per_dec_1` regardless of precondition status). The plan does NOT apply Platt / Isotonic / Beta calibration to per-cell predictions in this loop; DEC-1 forbids it in Plan 1. The artifact's existence is the load-bearing deliverable; whether the precondition is satisfied is the Plan-2-readiness signal.
  - Positive Tests (expected to PASS):
    - The artifact exists, validates against a new schema, and records `cross_cell_stddev` as the unbiased sample stddev (ddof=1) of the per-cell scalar predictions.
    - When the measured stddev >= 0.10, `precondition_status="satisfied"`.
    - When the measured stddev < 0.10, `precondition_status="not_satisfied"`.
    - `calibration_application_verdict` always equals `deferred_to_plan_2_per_dec_1` regardless of precondition status (Plan 1 NEVER applies calibration).
    - The committed predictor bundles do NOT carry a `calibration_applied=true` flag for any cell in Plan 1.
    - The artifact's `heuristic_stddev_projection` field records the `delta_pred / sqrt(n+1)` heuristic value as documentation, NOT as a physical formula or load-bearing prediction.
  - Negative Tests (expected to FAIL / be rejected):
    - A Plan 1 commit that mutates a predictor bundle's `predicted_hit_rate` via a Platt transform fires the AC-5 paired-control test (the committed predictor must equal the raw Smith-Eklund output).
    - An artifact with `precondition_status=satisfied` AND `cross_cell_stddev < 0.10` is rejected by the scanner (contradiction).
    - An artifact with `calibration_application_verdict != deferred_to_plan_2_per_dec_1` is rejected (Plan 1 is explicit).

- AC-6: No-leakage feature discovery matrix. Per the synthesis-note insight that all candidate covariates are ALREADY collected (in cupti_activity_sidecar / nvml_covariate / phase_cycle_ledger / kernel.proto), AC-6's deliverable is (a) a JOIN utility `tools/local_labels/extend_launch_parameters.py` that produces one `launch_parameters_sidecar.json` per (slug, shape_point) by joining the existing schemas on `start_ns` (CUPTI Activity) / `repeat_index` / `kernel_id`; (b) a correlation analysis on the silu_fwd_bf16_sm80 correctness vs timing shape-point pair (canonical missing-feature instance: both predict 0.9375; truth differs by 0.3526 absolute) PLUS one contrast cell (layernorm_fwd_bf16_sm80 timing; a non-silu cell where the predictor also over-predicts but by less); (c) a candidate-feature catalog at `data/sm_86_predictor_accuracy_recovery_v2/ac6_missing_feature_candidates.json` structured as `{features: [{feature_name, sidecar_source_file, schema_owner, expected_physical_sign}, ...], cells: [{slug, shape_point, signed_residual, per_feature_observations: [{feature_name, observed_value, availability_at_prediction_time, correlation_method, sample_size, computed_effect}, ...]}, ...], recommendations: [{feature_name, recommendation_for_plan_2 (predictor_input / covariate / discard / inconclusive), justification}, ...]}`. Feature-level metadata lives at the top; per-cell observations are explicit rows; recommendations are derived. DVFS state is documented as `label-provenance-only` (DEC-5 forbids it as a predictor input in Plan 1; Plan 2 may revisit per DEC-5 adjudication). No model changes in Plan 1.
  - Positive Tests (expected to PASS):
    - The candidate-feature catalog enumerates the 5 named candidate covariates (per-warp occupancy, blockDim, gridDim, DVFS state, MSHR pressure proxy) in the top-level `features` array with all 4 feature-level fields populated for each.
    - The JOIN utility runs against the currently-admitted cells + the admitted subset of the 6 new cells (per AC-4's actual admission count) AND produces one launch_parameters_sidecar per admitted cell.
    - The catalog's `cells` array records per-cell signed_residual and per-feature observations for EVERY admitted cell.
    - Every entry in the `recommendations` array uses one of the 4 enum values (`predictor_input` / `covariate` / `discard` / `inconclusive`); `inconclusive` is acceptable and expected when sample size is too small for a definitive correlation verdict.
    - DVFS-state fields are marked `availability_at_prediction_time=label-provenance-only` in every per-feature observation AND carry `plan_2_prohibition_status=forbidden_requires_explicit_reopen` per the DEC-5 resolution (Plan 1 forbids DVFS-as-predictor-input in both Plan 1 AND Plan 2; lifting in Plan 2 requires an explicit Plan-2 DEC reopening with portability evidence).
  - Negative Tests (expected to FAIL / be rejected):
    - A catalog entry marking DVFS as `prediction_time_available` without an explicit DEC-5 user override is rejected.
    - A catalog entry whose `availability_at_prediction_time=leakage_risk` AND `recommendation_for_plan_2=predictor_input` is rejected (leakage cannot become predictor input).
    - A `recommendations` entry whose value is outside the 4-value enum is rejected at parse time.
    - The JOIN utility producing a sidecar with monotone-impossible values (e.g., negative occupancy, blockDim=0) is rejected.

- AC-7: Interim diagnostic report. A new artifact at `data/sm_86_predictor_accuracy_recovery_v2/interim_diagnostic_report.md` mirrors the structure of the prior loop's `residual_root_cause_report.md` but covers both L1 and L2. Per-cell rows: slug, shape_point, ncu_truth_l1, predicted_l1, rel_err_l1, ncu_truth_l2, predicted_l2, rel_err_l2, predictor_design_coverage_bucket, plan_2_candidate. The `plan_2_candidate` field uses these explicit enum values: `already_meets_target_no_action_needed` (current rel_err < 0.10 on the relevant layer), `predictor_friendly_plan_2_accuracy_target` (in-domain for Smith-Eklund; Plan 2 physics + calibration is expected to close the gap below 0.10), `predictor_hostile_physics_challenge_plan_2` (out-of-domain for the current Smith-Eklund baseline; Plan 2 physics work required and may or may not reach 0.10), and `typed_unavailable_persistent` (the L1 or L2 path emits typed_unavailable persistently because of missing SASS provenance, missing scheduling info, or shared-only kernel structure). Per-bucket headline: friendly-bucket geomean rel_err for L1 and L2 separately; hostile-bucket geomeans listed as diagnostic coverage (NOT in the headline target). Quote the AC-5 variance-precondition verdict + the AC-6 candidate-feature catalog top entries. The audit replaces the prior loop's geomean-sum-check (mathematically invalid because geomeans do not add) with a PER-CELL RESIDUAL DECOMPOSITION: for each cell, compute `signed_residual = predicted - truth`, decompose into `attributed_to_named_bucket` (per the 7 plan-pinned buckets if applicable) + `unattributable_per_cell`, and verify the per-cell decomposition arithmetic BEFORE aggregating per-bucket geomeans of `|signed_residual|/truth` separately.
  - Positive Tests (expected to PASS):
    - The report exists, contains a Per-Cell Residual Decomposition section showing each admitted cell's `signed_residual = predicted - truth` with the decomposition arithmetic verified.
    - The Per-Bucket Geomeans section reports `friendly_l1_geomean`, `friendly_l2_geomean`, `hostile_l1_geomean`, `hostile_l2_geomean` SEPARATELY with their `n_cells`, `n_scalar`, `n_bounded`, `n_typed_unavailable` counts. No additive sum-check across geomeans is required (because geomeans of relative errors do not add).
    - Every cell in `_HEADLINE_COHORT_13` appears as a row with an admission_status (from AC-4's cohort-accounting artifact) AND a `plan_2_candidate` enum value.
    - The `plan_2_candidate` column classifies each cell into one of the 4 explicit enum values.
    - The report cites the AC-5 artifact's precondition_status verbatim.
  - Negative Tests (expected to FAIL / be rejected):
    - A report missing the per-bucket headline geomeans (with n_cells / n_scalar / n_bounded / n_typed_unavailable counts) is rejected by a scanner.
    - A report whose per-cell residual decomposition arithmetic does not equal the cited `signed_residual` value (within float precision) is flagged.
    - A report classifying a `predictor_hostile` bucket cell as `predictor_friendly_plan_2_accuracy_target` is flagged (the bucket-vs-candidate mapping is enforced).
    - A report classifying an admitted cell with rel_err < 0.10 as anything OTHER than `already_meets_target_no_action_needed` is flagged.
    - **A Plan-1-closure report that claims closure WITHOUT meeting all 3 hard numerical targets is rejected by a Plan-1-closure scanner** (per user 2026-05-17 hard-targets decision). The report MUST emit a `plan_1_closure_status` field with one of three values: `closed_all_targets_hit` (proceeds to AC-9 close), `pending_targets_not_yet_hit_plan_1_5_required` (routes to a Plan-1.5 strengthening round), or `aborted_unrecoverable_per_user_decision` (only the user can flip to this state via an explicit override).

- AC-8: L1 paired-control test suite. Eight specific test shapes (Codex's Phase-3 critique named the first seven; the Round-2 convergence loop adopted shape 8 as the set-conflict diagnostic) prove that the L1 baseline behaves correctly on canonical kernel patterns: (1) `LDG_E_CA` vs `LDG_E_CG` global-load variant on identical stride (modeled differently); (2) working set BELOW vs ABOVE L1 capacity with fixed L2-fit size (capacity ceiling fires); (3) shared-memory-only kernel produces typed_unavailable on L1 path with reason `predictor_l1_no_modeled_l1_eligible_accesses` (a new typed reason distinct from `predictor_l1_sass_provenance_missing` because shared-only kernels HAVE SASS but have nothing for L1 to model); (4) local-memory spill kernel is NOT treated as normal global (`LDL` / `STL` route differently per the cache-op classifier); (5) async-copy (`LDGSTS`) kernel fail-closes with `predictor_l1_sass_provenance_missing` unless explicit staging is proven; (6) same binary with different blockDim tests per-SM occupancy sensitivity; (7) unaligned sector-crossing loads pin the denominator behavior. Additionally (per Codex's optional improvement adopted as required): (8) a SET-CONFLICT DIAGNOSTIC fixture — a kernel whose access pattern would suffer set-conflict misses on real set-associative L1 hardware; the fully-associative baseline is EXPECTED to over-predict on this fixture; the test ASSERTS the over-prediction explicitly and records the gap as documented expected behavior of the fully-associative baseline (this fixture is NOT a fail-state; it documents the baseline's known limitation that Plan 2 may address via DEC-2). These are unit tests, not live-GPU witnesses; synthetic SASS / trace fixtures suffice. The intent is to lock the L1 baseline's behavior on edge cases before Plan 2 layers physics on top. ALL 8 test shapes are REQUIRED to pass (Plan 1 does not accept deferred shapes; if a shape is genuinely not implementable in Plan 1, it must be removed from AC-8 with documented rationale, NOT left as a failing test).
  - Positive Tests (expected to PASS):
    - Each of the 8 test shapes has at least one positive control proving the expected behavior.
    - The full suite runs under `uv run --isolated --with protobuf --with pytest pytest -q` in under 30 seconds (host-only; no live GPU).
    - The L2 path is byte-identical for the same 8 fixtures (regression guard: L1 additions did not perturb L2).
    - The set-conflict diagnostic fixture (shape 8) records the expected fully-associative over-prediction magnitude in the test's documented assertion (the gap is a feature, not a bug, of the baseline; the assertion confirms the baseline behaves as the model class promises).
  - Negative Tests (expected to FAIL / be rejected):
    - Each of the 8 test shapes has at least one negative control proving the failure mode (e.g., for shape 1, a fixture where the classifier mis-tags `LDG_E_CA` as `LDG_E_CG` is rejected by the assertion).
    - Removing the L1 schema migration field reproduces the AC-3 negative test in the L1 paired-control suite.

- AC-9: Loop discipline. Mirror the prior loop's AC-9 verbatim: per-round Codex review (this plan); failing-test-first witnesses for each AC; post-commit recapture for live-derived bundle fields per BL-20260508; BitLesson selection per round; plan file frozen via `--track-plan-file` (BL-20260510); a new Tier-2-cross-arch follow-up entry in the goal-tracker explicitly carrying forward the L1 + L2 cohort to sm_120 / sm_100. The plan-token scanner discipline (BL-20260507 / BL-20260510) is preserved: no `AC-N`, `Round N`, `Milestone`, `Phase`, `Step` literals in implementation files outside the narrow `_EXEMPT_FILES` set.
  - Positive Tests (expected to PASS):
    - Every round commit includes a failing-test-first witness landed in the same commit as the production code.
    - The plan-token scanner (`tools/test_plan_token_scanner.py`) passes on every commit.
    - The goal-tracker's Plan Evolution Log has one row per round with Reason + Impact on AC.
    - The new Tier-2-cross-arch follow-up entry in the goal-tracker is committed by Plan 1's final round.
    - **Plan 1 closure (Codex emits COMPLETE) requires ALL of: (a) the 9 ACs land with green tests, (b) all 3 hard numerical targets hit (per user 2026-05-17 decision), (c) the AC-7 report's `plan_1_closure_status=closed_all_targets_hit`. A round that satisfies (a) but not (b) routes to Plan-1.5 strengthening, NOT to AC-9 close.**
  - Negative Tests (expected to FAIL / be rejected):
    - A round that lands production code without a paired failing-test-first witness is rejected by Codex review.
    - A commit that introduces a plan-token literal in an implementation file outside the exempt list is rejected by the scanner.
    - The plan file is not modified during the active RLCR session (any edit is rejected).
    - A Plan-1 closure attempt with `plan_1_closure_status=pending_targets_not_yet_hit_plan_1_5_required` is rejected by AC-9; the loop continues to a Plan-1.5 strengthening round.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

All 9 ACs land with green paired-control tests; the L1 SASS / cache-op classifier covers all 12 underscore-vocabulary enum tokens with per-arch tolerance (live sm_86 coverage for the 10 Ampere-recognizable tokens; canned-fixture coverage for the 2 Blackwell-only tokens); the L1 capacity model records full carveout provenance with `l1tex_capacity_source` populated for every bundle AND `cta_to_sm_schedule_policy` set to `round_robin_homogeneous` for all admitted cells; the admitted-cell schema migration v3→v4 carries both L1 and L2 means (no nulls) with non-regressing L2 acceptance; all 6 new L2 cells are admitted at 5+ NCU repeats (13-cell cohort total); the variance trajectory artifact exists with a satisfied precondition (stddev >= 0.10); the candidate-feature catalog has at least one `recommendations` entry with value `predictor_input` (vs `inconclusive`) based on a documented effect-size + sample-size justification; the interim report contains the per-cell residual decomposition with verified arithmetic AND per-bucket headline geomeans hit `friendly_l2 < 0.15` AND `friendly_l1 < 0.20`; the L1 paired-control suite passes all 8 test shapes (including the set-conflict diagnostic); AC-9 loop discipline holds across all rounds with no plan-file edits, no skipped Codex reviews, and a new Tier-2-cross-arch follow-up entry committed.

### Lower Bound (Minimum Acceptable Scope)

All 9 ACs land with green paired-control tests; the L1 SASS / cache-op classifier covers at least the 6 sm_86-relevant enum tokens (`LDG_E_CA` / `LDG_E_CG` / `LDG_E_CS` / `LDG_E_EF` / `LDL` / `OTHER`) from live sm_86 cuobjdump output, with canned-fixture coverage for the 2 Blackwell-only tokens; the L1 capacity model records all required provenance fields with capacity sourced from a fixture-provided manifest (no hard-coded constant); the admitted-cell schema migration v3→v4 completes WITH non-regressing L2 acceptance; the cohort-accounting artifact records admission_status for ALL 13 HEADLINE_COHORT_13 cells; the variance trajectory artifact exists; the candidate-feature catalog has all 5 named covariates enumerated; the interim report exists with per-cell rows AND per-bucket headlines (with counts); the L1 paired-control suite passes ALL 8 test shapes; AC-9 loop discipline holds; AND **all 3 hard numerical targets hit**: friendly_l2 < 0.15, friendly_l1 < 0.20, stddev >= 0.10.

**Per user 2026-05-17 decision, the 3 numerical targets are HARD requirements for Plan 1 closure.** A round that lands all 9 ACs green but misses any of the 3 targets does NOT close Plan 1; it routes to Plan-1.5 strengthening (more NCU cells, DEC-2 set-associative promotion at >50% trigger, or revisited AC-6 covariate decisions). Honest typed_unavailable diagnostics on the failing target are required (per the no-false-precision discipline) but are NOT Plan-1-closure-acceptable on their own. The plan does NOT regress to false precision; missing targets surface as fail-closed `typed_unavailable` arms with diagnostic catalogs AND a new Plan-1.5 round contract documenting the corpus-or-baseline-strengthening work needed before re-attempting closure.

### Allowed Choices

- Can use:
  - Smith-Eklund stack-distance + Maeda HRD second-layer pattern (the existing L2 predictor architecture; mirror for L1).
  - sklearn (numpy 2.4.5, scipy 1.17.1, scikit-learn 1.8.0 already pinned in `requirements.txt`) for any LOOCV / correlation analysis in AC-6.
  - JSON-schema + Protobuf for any schema additions; additive-widening pattern (proto3 `optional`, new enum values appended) per the existing wire-extension policy in CLAUDE.md.
  - Cuobjdump / nvdisasm / cu++filt for SASS extraction in AC-1 (no driver-root, all read-only on committed binaries).
  - jsonschema-backed validation for new schemas (mirror of the existing `cache_predictor.schema.json` discipline).
  - `--track-plan-file` for RLCR plan-file freezing (BL-20260510).
  - Fully-associative LRU baseline for the L1 walker (Plan 2 may promote to set-associative per DEC-2).
  - The operator runbook `tools/local_labels/SWEEP_RUNBOOK.md` for AC-4 NCU collection on local RTX 3090 Ti.
- Cannot use:
  - NCU / nv-nsight-cu-cli on cloud or server GPUs (CLAUDE.md Rule 1, absolute except the local-label exception under `tools/local_labels/`).
  - nvidia-smi clock locking, persistence, power, MIG, or compute-mode changes (Rule 2).
  - Privileged CUPTI surfaces: Events, Metrics, Range Profiling, PM Sampling, PerfWorks (Rule 3; only CUPTI Activity is allowed).
  - Host-driver work: driver install, DKMS, MOK, `modprobe nvidia`, reboot workflows (Rule 4).
  - Simulator code (nvasim, Accel-Sim, GPGPU-Sim) — Rule 5; the L1 / L2 predictor uses Smith-Eklund stack distance, not phase simulation.
  - Calibration application in Plan 1 (Platt / Isotonic / Beta transforms on `predicted_hit_rate`); defer to Plan 2 per DEC-1.
  - L2 physics features (DVFS covariate as predictor input, eviction-chaining model, hit-under-miss queue, prefetch model); defer to Plan 2.
  - Cross-arch work (sm_120, sm_100) — Tier-2-cross-arch follow-up loop per the user's 2026-05-16 directive.
  - L1 set-associative refinement unless DEC-2 fires (Plan 1 baseline is fully-associative).
  - Plan-token literals (`AC-N`, `Round N`, `Milestone`, `Phase`, `Step`) in implementation files outside the narrow `_EXEMPT_FILES` set.
  - Editing the plan file after the RLCR session starts (BL-20260510).

> **Note on Deterministic Design**: Per user 2026-05-17 decision, the 3 numerical targets (friendly_l2 < 0.15, friendly_l1 < 0.20, stddev >= 0.10) are HARD requirements alongside the 9 ACs. Plan 1 closure requires BOTH the 9 ACs land with green tests AND all 3 numerical targets hit. A miss on any target routes to a Plan-1.5 strengthening round (more NCU cells / DEC-2 set-associative promotion / AC-6 covariate revisit) before re-attempting closure. The metric-vs-AC distinction in the prior loop's DEC-4 (where DEC-4 confirmed metric-aspirational) is OVERRIDDEN for Plan 1 by the user's explicit hard-requirement decision.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Order the rounds by Codex's strict dependency order: sub-stream C (L1 head: AC-1 → AC-2 → AC-3 → AC-8) and sub-stream A (L2 corpus + discovery: AC-4 → AC-5 → AC-6) run concurrently; AC-7 (report) depends on both; AC-9 (loop discipline) is cross-cutting on every round.

A possible per-AC implementation path:

- AC-1: Start by reading the existing `tools/perturbation_harness/sass_confirm.py` as the template for cuobjdump invocation + LDG regex matching. Build a new `tools/cache_predictor/l1_cache_op_classifier.py` that calls cuobjdump on each admitted-cell binary, walks per-PC, and emits a JSON manifest `<binary>_cache_op_map.json` using the underscore-vocabulary tokens (`LDG_E_CA` etc.). Add `instruction_cache_op` to the dynamic_trace.pb sidecar schema (additive proto3 optional field). The predictor's L1 head reads the manifest at predict time; missing manifest → typed_unavailable.

- AC-2: Extend `tools/hrd_profiler/observation_builder.py` with a sibling of `_walk_launch` that maintains a per-SM LRU stack instead of a per-slice LRU stack. Use the existing reuse_distance_tracker.py infrastructure. The CTA-to-SM schedule is a function from `block_index → sm_id`; for `round_robin_homogeneous`, this is `block_index % sm_count` where `sm_count` is sourced from the test fixture's device-properties manifest (NOT a plan-text constant — every fixture provides the value and the bundle records `l1tex_capacity_source` to make the provenance auditable). Record the scheduling policy AND the source manifest in the bundle's evidence block.

- AC-3: Schema migration is the highest-risk surface. Bump `local_label_admitted.schema.json` from `schema_version=3` to `schema_version=4`; add nullable `cache_hit_rate_l1tex_sector_pct` plus `l1tex_label_status` enum (`mean_present` / `mean_absent_label_missing` / `mean_absent_provenance_missing`). The aggregator (`tools/local_labels/repeatability_aggregator.py`) gains LOC to emit the L1 mean alongside the existing stddev computation OR emit null + the matching `l1tex_label_status` when the per-repeat L1 values are not populated. The predictor's per-layer bundle wire shape carries `cache_layer` enum (L1TEX / L2) plus an independent `result_kind`. The comparison runner's join expands to an 8-tuple including `cache_layer`, computing L1 and L2 rel_err in parallel via independent per-arch diagnostic blocks (each reporting `n_cells` / `n_scalar` / `n_bounded` / `n_typed_unavailable` alongside geomean).

- AC-4: NO new code — operator runs the SWEEP_RUNBOOK on local RTX 3090 Ti. Estimated wall-clock: 6 cells × 20 min/cell = 2 hours of operator NCU time + ~30 min of admission + audit. Verify each new cell's join key matches the predictor sidecar before declaring the cell admitted.

- AC-5: A new diagnostic script `tools/sm_86_predictor_accuracy_recovery_v2/compute_variance_trajectory.py` reads the 13-cell predictor bundles, computes the cross-cell scalar stddev, and emits the JSON artifact. The script is ~100 LOC; the schema is new (~50 LOC).

- AC-6: The JOIN utility is ~150 LOC; it walks the per-repeat sidecars under `data/local_labels/runs/`, joins on (slug, shape_point, repeat_index, kernel_id), and emits one launch_parameters_sidecar per (slug, shape_point). The correlation analysis uses scipy.stats.pearsonr or scipy.stats.spearmanr on the per-cell signed_residual; the catalog is ~200 LOC.

- AC-7: The report is generated via `/humanize:ask-codex` per the prior loop's task11 pattern (analyze tag). Codex receives the AC-5 + AC-6 artifacts + the per-cell rel_err table and produces the report markdown. The plan-token scanner is satisfied because the report goes under `data/sm_86_predictor_accuracy_recovery_v2/` (eligible for narrow exempt-file widening if it carries load-bearing literals).

- AC-8: Synthetic SASS / trace fixtures in `tools/cache_predictor/test_l1_paired_controls.py`. Each of the 8 shapes (the 7 Codex Phase-3 shapes + the Round-2-adopted shape 8 set-conflict diagnostic) gets a small generator that produces a synthetic dynamic_trace.pb + a synthetic cache-op manifest. Shape 8 documents the fully-associative baseline's expected over-prediction on conflict-heavy patterns. ~400 LOC total.

- AC-9: Mirror the prior loop's AC-9 verbatim. New goal-tracker section for Tier-2-cross-arch follow-up; BitLesson selection per round; per-round summary + contract files; post-commit recapture for any live-derived bundle fields.

### Relevant References

- `workloads/simbench_bw/cache_predictor.py` — entry point for adding `predict_l1_hit_rate`; mirrors the existing `predict_cache_hit_rate` shape.
- `tools/hrd_profiler/observation_builder.py` — `_walk_launch` is the precedent for the per-SM L1 walker; lines 462-477 (per-slice L2 geometry).
- `tools/hrd_profiler/reuse_distance_tracker.py` — Fenwick-tree LRU stack reusable for per-SM tracking.
- `tools/perturbation_harness/sass_confirm.py` — cuobjdump + per-arch SASS regex template for AC-1.
- `workloads/simbench_bw/comparison_runner.py` — `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` (R21 friendly/hostile partition), `_per_bucket_relative_error_stats()`, join logic for AC-3 extension.
- `workloads/simbench_bw/schemas/cache_predictor.schema.json` — `_TYPED_REASONS` enum location for AC-1 lockstep.
- `workloads/simbench_bw/schemas/local_label_admitted.schema.json` — current `schema_version=3`; AC-3 migrates to `schema_version=4` (adds nullable `cache_hit_rate_l1tex_sector_pct` + `l1tex_label_status` enum).
- `tools/local_labels/repeatability_aggregator.py` — emits stddev; needs +4 LOC for L1 mean.
- `tools/local_labels/SWEEP_RUNBOOK.md` — operator runbook for AC-4.
- `workloads/simbench_bw/schemas/cupti_activity_sidecar.schema.json` + `nvml_covariate.schema.json` + `phase_cycle_ledger.schema.json` + `tools/nvbit_tracer/traces_enhanced/dynamic_trace/kernel.proto` — the 4 existing schemas that AC-6's JOIN utility consumes.
- `data/sm_86_predictor_accuracy_recovery/residual_root_cause_report.md` — prior loop's AC-7 report; structural template for AC-7's interim report.
- `data/sm_86_predictor_accuracy_recovery/loop_finalize_summary.md` — prior loop's AC-9 finalize summary; template for Plan 1's loop-finalize artifact.

## Dependencies and Sequence

### Milestones

1. Milestone-L1 (Sub-stream C): L1 baseline predictor with SASS-grounded provenance, layer-aware comparison, and paired-control coverage.
   - Section A: AC-1 SASS / cache-op provenance scaffold lands first (it is the input to AC-2's L1 walker).
   - Section B: AC-2 L1 capacity provenance + per-SM baseline model consumes AC-1's manifest.
   - Section C: AC-3 layer-aware comparison path + admitted-label schema migration consumes AC-1 + AC-2 outputs.
   - Section D: AC-8 L1 paired-control test suite consumes AC-1 + AC-2 + AC-3.

2. Milestone-A (Sub-stream A): L2 corpus expansion + variance verdict + missing-feature discovery.
   - Section A: AC-4 L2 corpus expansion runs concurrently with Milestone-L1 (operator runs the SWEEP_RUNBOOK on local hardware; no code dependency on Milestone-L1).
   - Section B: AC-5 variance trajectory + calibration-decision inputs consumes AC-4 + the existing predictor.
   - Section C: AC-6 no-leakage feature discovery matrix consumes the existing 4 schemas + AC-4 labels for the contrast cell.

3. Milestone-Final: Cross-stream synthesis + loop closure.
   - Section A: AC-7 interim diagnostic report consumes outputs of all preceding ACs.
   - Section B: AC-9 loop discipline closes the loop with the Tier-2-cross-arch follow-up entry.

Dependencies:
- Milestone-L1 (Sections A → B → C → D) is strictly serial.
- Milestone-A Section A is concurrent with all of Milestone-L1.
- Milestone-A Sections B and C are concurrent with each other but both depend on Section A.
- Milestone-Final depends on both Milestone-L1 (all sections) and Milestone-A (all sections).

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Build cuobjdump-backed L1 cache-op classifier (`tools/cache_predictor/l1_cache_op_classifier.py`) + per-PC manifest emitter + dynamic_trace.pb sidecar schema additive widening + lockstep tests across `_TYPED_REASONS` / `cache_predictor.schema.json::typed_reason.enum` / `comparison_runner.py` accepted-bundle-shape | AC-1 | coding | - |
| task2 | Extend `tools/hrd_profiler/observation_builder.py` with per-SM L1 walker + capacity provenance fields + CTA-to-SM schedule policy + sector-with-line-allocation hit decision + paired controls | AC-2 | coding | task1 |
| task3 | Migrate `local_label_admitted.schema.json` v3 → v4 (add nullable `cache_hit_rate_l1tex_sector_pct` + `l1tex_label_status` enum); update `repeatability_aggregator.py` to emit L1 mean alongside the existing L1 stddev (or null + the matching `l1tex_label_status` when L1 sidecar values are absent); define per-layer predictor bundle wire shape (each bundle carries `cache_layer` enum L1TEX / L2 + independent `result_kind`); extend `comparison_runner.py` 7-tuple join to the 8-tuple including `cache_layer` and emit per-arch L1 + L2 diagnostic blocks (each block reports `n_cells` / `n_scalar` / `n_bounded` / `n_typed_unavailable` alongside geomean); widen `comparison_contract.toml::cache_layers` to ["L1", "L2"]; lockstep schema tests for v3-record rejection with typed reason `local_label_admitted_l1_mean_migration_required` | AC-3 | coding | task1, task2 |
| task4 | Operator runs `tools/local_labels/SWEEP_RUNBOOK.md` on local RTX 3090 Ti for 6 unlabeled HEADLINE_COHORT_13 cells (working_set_capacity_sweep, dram_stream_stride_mix, carveout_sensitivity_probe, warm_cold_pair_relaunch, l2_persisting_window_sweep, l2_prefetch_hint_sweep) at trace_l2_accuracy_xl; admit to label_admitted.json with cohort-accounting diagnostics | AC-4 | coding | - |
| task5 | Build `tools/sm_86_predictor_accuracy_recovery_v2/compute_variance_trajectory.py` + new schema for the variance-trajectory artifact + paired-control tests asserting Plan 1 never applies calibration | AC-5 | coding | task4 |
| task6 | Build `tools/local_labels/extend_launch_parameters.py` JOIN utility; emit launch_parameters_sidecar.json per cell; correlation analysis on silu correctness/timing + layernorm timing contrast; candidate-feature catalog (`data/sm_86_predictor_accuracy_recovery_v2/ac6_missing_feature_candidates.json`) with feature-level metadata + per-cell observations + recommendations. Catalog generation may start before AC-4 finishes (against the 7 currently-admitted cells), but the catalog cannot CLOSE until AC-4 admission outcomes are known so that the catalog's `cells` array covers the actual admitted set. | AC-6 | coding | task4 |
| task7 | Generate interim diagnostic report (`data/sm_86_predictor_accuracy_recovery_v2/interim_diagnostic_report.md`) per the prior loop's residual_root_cause_report.md template + per-cell residual decomposition (verified arithmetic) + independent per-bucket geomeans (friendly_l1, friendly_l2, hostile_l1, hostile_l2 each reported with n_cells / n_scalar / n_bounded / n_typed_unavailable counts; NO additive sum-check across geomeans) + plan_2_candidate classification using the 4-value enum (`already_meets_target_no_action_needed` / `predictor_friendly_plan_2_accuracy_target` / `predictor_hostile_physics_challenge_plan_2` / `typed_unavailable_persistent`) | AC-7 | analyze | task3, task5, task6 |
| task8 | Author synthetic SASS / trace fixtures and 8 L1 paired-control test shapes in `tools/cache_predictor/test_l1_paired_controls.py` — including the new set-conflict diagnostic fixture (shape 8) that documents the fully-associative baseline's expected over-prediction on conflict-heavy patterns. ALL 8 shapes are required to pass; deferring a shape requires removing it from AC-8 with documented rationale | AC-8 | coding | task1, task2, task3 |
| task9 | Loop discipline: per-round Codex review; failing-test-first witnesses; post-commit recapture; BL selection per round; plan-file frozen via --track-plan-file; commit Tier-2-cross-arch follow-up entry in goal-tracker | AC-9 | coding | task1, task2, task3, task4, task5, task6, task7, task8 |

## Claude-Codex Deliberation

### Agreements

- Plan 1 is the first of a two-plan recovery sequence; Plan 2 (deferred) handles L2 physics features, calibration application, and final reporting per Codex's critical-review dependency order.
- The L1 head is necessary for the L1+L2 target; bypass-only is insufficient (per-SM capacity + carveout + Smith-Eklund instantiation required).
- The corpus expansion targets the 6 unlabeled HEADLINE_COHORT_13 cells (variance-expansion gold mine; cells already have .cu + NVBit trace infrastructure).
- AC-6's candidate covariates are ALREADY collected in 4 existing sidecar schemas; the deliverable is a JOIN utility + correlation analysis + catalog, NOT a new sidecar schema.
- The silu correctness/timing split is the canonical missing-feature instance (both predict 0.9375; truth differs by 0.35 absolute); AC-6 must analyze it AND a non-silu contrast cell.
- Plan 1 does NOT apply calibration regardless of the new variance-precondition state (DEC-1).
- Plan 1 is sm_86-only; cross-arch defers to Tier-2-cross-arch follow-up per the user's 2026-05-16 directive.

### Resolved Disagreements

- "L1 predictor head as L2 mirror" (Claude's draft proposed widening cache_layers from ["L2"] to ["L1", "L2"] as a small change): Codex argued that's too small — it changes metric semantics, ground-truth field selection, schemas, comparison output shape, and bundle shape. Resolution: AC-3 is restructured as a full layer-aware comparison path WITH admitted-label schema migration (AC-3 becomes the largest surface in the plan, not a small addition).
- "memory_op_class drives the L1 bypass classifier" (Claude's draft): Codex pointed out that `memory_op_class` distinguishes memory space (GLOBAL/SHARED/LOCAL), NOT the SASS cache operator. Resolution: AC-1 is restructured as a SASS / cache-op provenance scaffold (cuobjdump-backed per-PC classifier); `memory_op_class` and `instruction_cache_op` are orthogonal fields (AC-1.1 enforces the orthogonality with a paired-control test).
- "L1 ground truth is already joinable" (Claude's draft): Codex showed that `label_admitted.json` only retains `stddev_l1tex_sector`, not the L1 mean — the comparison runner cannot compute L1 rel_err today. Resolution: AC-3 is migration-required (NOT migration-optional); the v4 schema rejects v3 records with typed reason `local_label_admitted_l1_mean_migration_required`.
- "L2 friendly-bucket geomean < 0.15 target" (Codex argued this is under-supported without ≥1 L2 model change OR scope limited to predictor-friendly cells): Resolution: the target is explicitly scoped to the predictor-friendly bucket only (per DEC-4 confirming targets are aspirational optimization trends). Hostile cells reported separately as diagnostic coverage.
- "AC-2 fully-associative LRU on per-SM L1" (Codex argued this may reproduce L2's compressed-high-prediction failure): Resolution: AC-2 is explicit that `l1_model_class=fully_associative_baseline` is Plan 1's choice; set-associative refinement deferred to Plan 2 unless DEC-2 fires. AC-8 shape-8 set-conflict diagnostic fixture explicitly documents the baseline's over-prediction behavior on conflict-heavy patterns.
- "AC-6 is sidecar join utility" (Codex argued this is under-scoped): Resolution: AC-6 is expanded to require correlation analysis + a no-leakage feature matrix structured as `{features: [...], cells: [...], recommendations: [...]}` with feature-level metadata at top, per-cell observations, and explicit recommendation enum including `inconclusive`.
- "Per-SM L1 walker needs CTA-to-SM scheduling" (Codex argued the existing per-slice L2 round-robin merge is too coarse for L1): Resolution: AC-2 explicitly records `cta_to_sm_schedule_policy` (enum) and fail-closes with typed reason `predictor_l1_cta_to_sm_schedule_unknown` when the policy is `unknown`. `round_robin_homogeneous` is the baseline; `occupancy_derived` is allowed when the workload provides per-SM occupancy.

### Round-2 Resolved Disagreements (Codex convergence round 1)

- "AC-2 hard-codes 192 KiB L1 capacity for sm_86": Codex argued the constant is not safe without live device-props evidence. Resolution: AC-2 removes all hard-coded capacity constants from implementation and tests; every bundle records `l1tex_capacity_source` (enum: device_properties_manifest / cuda_func_attribute_query / cuda_device_attribute_query / static_card_constant); test fixtures source the value from a manifest, not from plan text.
- "AC-3 schema migration v1→v2 is version-stale" (current schema is already v3 per the in-repo definition): Resolution: AC-3 migrates v3 → v4; the typed reason rename to `local_label_admitted_l1_mean_migration_required` is non-version-stale (does not embed the schema version number).
- "AC-3 underspecified bundle wire shape": Resolution: AC-3 defines per-layer bundles — each predictor bundle carries `cache_layer` enum (L1TEX / L2) plus an independent `result_kind`; ONE bundle per (cell, cache_layer) pair; the comparison runner expands to an 8-tuple join including `cache_layer`.
- "AC-3 silent on L1 label nullability": Resolution: AC-3 specifies `cache_hit_rate_l1tex_sector_pct` is NULLABLE in v4; cells with no L1 sidecar value migrate with `cache_hit_rate_l1tex_sector_pct=null` AND `l1tex_label_status=mean_absent_*`; the L2 path continues to work on those cells.
- "AC-4 exactly-13-cells contradicts Lower Bound 4-of-6": Resolution: AC-4 acceptance threshold is "at least 4 of 6 cells admitted at n_repeats >= 5" (matching the Lower Bound); the cohort-accounting artifact records admission_status for ALL 13 HEADLINE_COHORT_13 cells; Upper Bound (13 total) is aspirational, not a Lower-Bound floor.
- "AC-6 |correlation| >= 0.5 is false precision on small N": Resolution: AC-6 removes the magnitude-correlation hard requirement; the `recommendations` enum gains `inconclusive` as an explicit acceptable value when sample size is too small for a definitive verdict.
- "AC-7 geomean sum-check is mathematically invalid": Resolution: AC-7 replaces the sum-check with a per-cell residual decomposition (`signed_residual = predicted - truth` with arithmetic verified BEFORE aggregating per-bucket geomeans separately); per-bucket geomeans are reported as independent values WITHOUT an additive sum-check.
- "AC-7 `plan_2_candidate` semantics ambiguous": Resolution: the enum is expanded to 4 explicit values (`already_meets_target_no_action_needed` / `predictor_friendly_plan_2_accuracy_target` / `predictor_hostile_physics_challenge_plan_2` / `typed_unavailable_persistent`); paired-control test enforces the bucket-vs-candidate mapping.
- "AC-8 5-of-7 acceptance undermines loop discipline": Resolution: AC-8 requires ALL 8 test shapes to pass (shape 8 = set-conflict diagnostic added per Codex's optional improvement); deferring shapes from AC-8 requires removing them with documented rationale, not leaving them failing.
- "AC-1 enum-spelling inconsistency between text and tests": Resolution: AC-1 normalizes to the underscore-vocabulary tokens (`LDG_E_CA` etc.); dotted SASS mnemonics appear only as parser input examples; schema enum + test fixtures use the underscore form exclusively.
- "AC-1 tests require sm_86 binary for Blackwell-only tokens": Resolution: Plan 1's live sm_86 coverage is the 10 Ampere-recognizable tokens; Blackwell-only tokens (`LDG_E_STRONG_SM` / `LDG_E_STRONG_GPU`) are covered via CANNED CUOBJDUMP TEXT FIXTURES only; live sm_120 coverage defers to the Tier-2-cross-arch follow-up loop.
- "task6 missing AC-4 dependency": Resolution: task6's Depends On column updated to `task4`; the catalog generation may start before AC-4 finishes (against the 7 currently-admitted cells), but cannot CLOSE until AC-4 admission outcomes are known.
- Optional-improvement adoptions: (a) `stddev_prediction_trajectory` renamed to `heuristic_stddev_projection` to clarify the heuristic nature; (b) the new typed reason `predictor_l1_no_modeled_l1_eligible_accesses` is added to AC-8 shape 3 (distinct from `predictor_l1_sass_provenance_missing`); (c) the per-arch L1/L2 diagnostic blocks include `n_cells` / `n_scalar` / `n_bounded` / `n_typed_unavailable` counts; (d) AC-8 shape 8 (set-conflict diagnostic) documents the fully-associative baseline's known over-prediction on conflict-heavy patterns.

### Convergence Status

- Final Status: `partially_converged` after 3 convergence rounds with second-pass Codex. Round 1 surfaced 12 REQUIRED_CHANGES (all integrated) plus 5 OPTIONAL_IMPROVEMENTS (all adopted) plus 3 genuinely-unresolved UNRESOLVED items. Round 2 verified the AC text fixes landed but identified residual stale text in task breakdown / feasibility hints / appendix (all integrated). Round 3 verification found 2 final stale-text propagations in Feasibility Hints (AC-8 wording said "7 shapes"; Relevant References said "v1 schema migration to v2") — both integrated in this pass. The plan's normative sections (Goal, ACs, Path Boundaries, Dependencies, Task Breakdown) are now internally consistent across all sections; the appendix is explicitly marked SUPERSEDED. 2 DECs (DEC-2, DEC-5) remain genuinely PENDING for user adjudication; both have documented Claude recommendations that the RLCR loop can adopt by default if the user does not override before loop start.

## Pending User Decisions

The Round-1 convergence loop resolved 5 of the originally-raised 7 decisions by adopting the language directly into AC text. Only the 2 below remain genuinely unresolved (Claude and Codex have different positions that the AC text could not unilaterally settle). See `## Claude-Codex Deliberation > Round-2 Resolved Disagreements` for the resolved 5 (DEC-1, DEC-3, DEC-4, DEC-6, DEC-7 plus DEC-5's Plan-1 scope) and the AC sections that codify each resolution.

- DEC-2: L1 set-associative refinement promotion trigger for Plan 1. **RESOLVED by user 2026-05-17**: defer set-associative refinement to Plan 2 UNLESS AC-3's fully-associative baseline produces rel_err > 50% on any single cell (compromise stricter than Claude's 30% threshold and looser than Codex's unconditional deferral). When the trigger fires, promote into Plan 1 as a new AC-2.2 sub-criterion: per-cell associativity probe + new walker mode + new ablation. The AC-8 shape-8 set-conflict diagnostic fixture documents the baseline's expected over-prediction on conflict-heavy patterns AND serves as the canary for the 50% trigger (if a real cell's L1 rel_err exceeds 50%, AC-2.2 is invoked).
  - Decision Status: RESOLVED 2026-05-17.

- DEC-5: DVFS-as-predictor-input policy (both Plan 1 AND Plan 2). **RESOLVED by user 2026-05-17**: Plan 1 EXPLICITLY FORBIDS DVFS as a predictor input in BOTH Plan 1 AND Plan 2; lifting the prohibition in Plan 2 requires an explicit Plan-2 DEC reopening backed by evidence that DVFS state is prediction-time-available AND portable across runtime contexts (cloud vs local; sm_86 vs sm_120). AC-6's catalog records DVFS as `availability_at_prediction_time=label-provenance-only` with an additional `plan_2_prohibition_status=forbidden_requires_explicit_reopen` field per cell entry. This is the default-forbidden + explicit-reopen pattern.
  - Decision Status: RESOLVED 2026-05-17.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead (e.g., for AC-1 use `l1_cache_op_classifier`, for AC-3 use `local_label_admitted_l1_mean_migration`, etc. — version-neutral names, not `_v2_` or `_v4_` literals which would rot when the schema bumps again)
- The plan-token scanner at `tools/test_plan_token_scanner.py` enforces this rule; any new file must pass without exemption-table widening unless the file's load-bearing semantics genuinely require a plan-pinned identifier (in which case the exemption requires a per-file rationale comment per the prior loop's BL-20260509 discipline)

### RLCR Loop Conventions
- The plan file `plans/l1_baseline_l2_corpus_expansion_plan.md` MUST be frozen during the active RLCR session via `--track-plan-file` (BL-20260510). Any in-loop adjustment to scope, decision points, or AC text lands in the goal-tracker's Plan Evolution Log, NOT as a plan-file edit.
- Per-round Codex review is mandatory; the loop only closes via the framework's COMPLETE → Code Review → no [P0-P9] findings sequence.
- BitLesson selection per round per the project's `.humanize/bitlesson.md`; new lessons land when patterns recur per BL-20260509's recurrence-threshold rule.
- Post-commit recapture for live-derived bundle fields per BL-20260508 (Plan 1's main live-derived surface is AC-4's NCU labels).

--- Original Design Draft Start ---

# Concurrent L1 Predictor Baseline And L2 Variance Unlock

## Original Idea

# L1 baseline + L2 corpus expansion + missing-physics-feature discovery (sm_86 follow-up loop)

## Starting state (end of L2 predictor accuracy recovery loop)

The just-closed RLCR loop (session `2026-05-17_11-26-33`) closed all 9 ACs for the L2 cache hit-rate predictor at:
- post-AC-4 geomean rel_err = 0.319 (32% systematic over-prediction)
- per-cell rel_err range 0.051 (silu correctness, 5.15%) to 0.739 (silu timing, 73.9%)
- cross-cell prediction stddev = 0.018 (below 0.10 floor; calibration head NOT applied)
- AC-8 final sweep at `data/local_labels/sweeps/sweep_20260517T111017Z_57b09f4/` emits `result_kind=typed_unavailable` with `headline_validity_failed_reasons=[calibration_loocv_p90_residual_exceeded, calibration_variance_precondition_failed]`

The honest verdict from the root-cause report: 99.74% of the residual is in unattributable/corpus-shape buckets. ALL four "structural" sub-slices (slice geometry, sector edge, CTA interleave, cross-launch state) no-opped because the 7-cell corpus is uniform (all working sets fit in L2; all aligned; all disjoint-CTA; all single-launch). Only the write-policy sub-slice moved the headline by -0.00084.

## The user's ultimate goal

<10% relative error for ALL L1 and L2 cache modelling metrics. Codex's critical review (the 2026-05-17 deep analysis at `.humanize/skill/2026-05-17_22-34-24-3582787-73a6835b/output.md`) calls this aspirational but not credible as a guaranteed sm_86-only close criterion. The pragmatic two-plan structure Codex recommends:
- Plan 1 (this idea): L1 baseline + L2 corpus expansion + missing-physics-feature discovery; target <20% L1 rel_err + <15% L2 geomean rel_err + variance-precondition unlock
- Plan 2 (deferred sequential follow-up): L2 physics features (DVFS covariate, eviction-chaining model, hit-under-miss queue) + calibration application decision + final reporting; target <10% L2 + maintain L1 baseline

This idea covers Plan 1 only; Plan 2 is queued as a sequential follow-up plan (NOT a side-issue; it depends on Plan 1's corpus + L1 baseline).

## Why Plan 1 first (Codex's dependency order)

The strict linear order Codex recommends: **C (L1 head) → A (L2 corpus expansion) → B (L2 physics) → calibration decision → D (regenerate reports)**. C and A can run concurrently. Plan 1 covers C + A + the variance-trajectory update; Plan 2 covers B + calibration + D.

Two critical observations from Codex:

1. **The silu evidence proves the gap is missing physics feature, not just corpus shape**. silu_fwd_bf16_sm80/correctness_shape and silu_fwd_bf16_sm80/timing_shape both predict 0.9375 under the current predictor; truth differs by 0.3526 absolute (0.8916 vs 0.5390). Same slug, same predicted hit rate, vastly different truth → predictor lacks a feature that distinguishes the two shape points. Plan 1 must DESIGN the trace/sidecar schema for the missing covariates (timing-vs-correctness shape, occupancy, DVFS state, MSHR proxy) BEFORE Plan 2 collects labels — otherwise Plan 2 collects labels without explanatory fields and learns nothing.

2. **Bypass-only L1 is not enough**. Codex recommends bypass classifier + per-SM L1 capacity model + L1-specific Smith-Eklund instantiation + carveout policy. Initial fully-associative LRU OK; set-associative refinement deferred. Typed-unavailable arm for missing SASS/cache-op provenance is mandatory.

## What this loop produces (Plan 1 ACs proposed)

5-6 ACs:

**AC-1: L1 bypass classifier** (`tools/cache_predictor/l1_bypass_classifier.py`). SASS-aware: distinguishes L1-eligible (.CA, .CG, .CS, .EF default LDG) vs bypassing (.STRONG.SM, .STRONG.GPU, atomics, local-memory, async-copy without L1 staging). Consumes the existing NVBit sector_stream + per-PC SASS classifier output. Emits per-record L1-eligibility tag. Typed-unavailable when SASS provenance missing.

**AC-2: L1 capacity model** (extension to `tools/hrd_profiler/observation_builder.py`). Per-SM L1 capacity = 192 KiB (sm_86). Records carveout policy as provenance (sm_86 has 4 KiB/16 KiB/32 KiB/64 KiB/100 KiB shared-memory carveout configurations). L1 walker tracks per-SM LRU at the warp-line granularity (not whole-GPU). Fully associative initial approximation; set-associative refinement deferred. Bundle field `cache_model_l1_capacity_bytes` + `cache_model_l1_carveout_policy_source` (mirrors existing L2 fields). 

**AC-3: L1 predictor head** (`workloads/simbench_bw/cache_predictor.py::predict_l1_hit_rate`). Mirrors L2 architecture (Smith-Eklund stack-distance + Maeda HRD second-layer optional). Emits scalar/bounded/typed_unavailable bundle shapes. Updates `comparison_contract.toml::cache_layers` from `[L2]` to `[L1, L2]`. Schema lockstep with `cache_predictor.schema.json`. New typed reason `predictor_l1_sass_provenance_missing` (typed_unavailable on missing classifier input). New paired-control tests.

**AC-4: L2 corpus expansion** (run on local RTX 3090 Ti per CLAUDE.md Rule 1 exception). Cost: ~20 min/cell + 8 operator commands. Target 6 variance-expansion cells from the existing HEADLINE_COHORT_13 unlabeled set: working_set_capacity_sweep, dram_stream_stride_mix, carveout_sensitivity_probe, warm_cold_pair_relaunch, l2_persisting_window_sweep, l2_prefetch_hint_sweep. Each cell already has a `.cu` + an NVBit trace; need NCU label sweep + admission. Operator runbook at `tools/local_labels/SWEEP_RUNBOOK.md` is the load-bearing reference; no new scripts required. Plan 1 includes the operator runbook execution as a manual-step gate (not a coding task — the operator runs the commands on local hardware).

**AC-5: Variance trajectory update + diagnostic**. After AC-4 lands, recompute cross-cell prediction stddev. Codex's estimate using the trajectory rule (`delta_pred / sqrt(n+1)`): with n=7 current cells and a target stddev=0.10, need 4-6 new cells with delta_pred ~0.25-0.30 from the current ~0.956 cluster (i.e., predictions <=0.70). The 6 variance-expansion cells span L2-thrash regimes; their predictions span ~0.0 (l2_thrash_pointer_chase already labeled) to ~0.95 (carveout_sensitivity_probe at default carveout). New artifact `data/sm_86_predictor_accuracy_recovery_v2/ac5_variance_trajectory_post_corpus_expansion.json` with cross-cell stddev recomputation + verdict (precondition_status=satisfied or not_satisfied) + Codex-formulated stddev-prediction trajectory ("with N cells, expected stddev = ..."). NO calibration application in this plan — that's Plan 2.

**AC-6: Missing-physics-feature discovery + sidecar schema design**. The silu correctness/timing split is the canonical instance. Investigate what feature(s) distinguish the two shape points. Candidates: per-warp occupancy delta, kernel launch parameters (block count + thread count), DVFS state (record-only, NOT lock-clocks per Rule 2), MSHR pressure proxy. Design the trace sidecar schema additions for those covariates. Emit a candidate-feature catalog at `data/sm_86_predictor_accuracy_recovery_v2/ac6_missing_feature_candidates.json`. NO model changes in this plan — Plan 2 will integrate these candidates into the predictor.

**AC-7: Interim diagnostic report**. Mirrors the structure of the prior loop's residual_root_cause_report.md but covers both L1 and L2. Report the per-cell L1 + L2 rel_err for the expanded corpus (7 + 6 = 13 cells); name the per-bucket attribution (slice geometry / sector edge / write policy / CTA interleave / cross-launch / label-variance / missing-feature). Quote the variance-precondition verdict + the candidate-feature catalog summary. Identify which cells are CANDIDATES for <10% rel_err in Plan 2 (the predictor-friendly ones) vs which need new physics (the silu split + thrash cells).

## Out of scope for this Plan 1

- Plan 2 work: L2 physics features (DVFS covariate, eviction-chaining model, hit-under-miss queue), calibration application decision, AC-7-style final report. ALL queued for Plan 2.
- Cross-arch (sm_120, sm_100) work. Continues to defer to the Tier-2-cross-arch follow-up loop per the user's 2026-05-16 directive.
- The 25 unlabeled microbench cells. Filter: only 2 are sm_86-compatible (bf16_hmma_gemm_traceable, cp_async_shared_tile_copy); the rest require Hopper/Blackwell. Both sm_86-compatible non-cohort cells stay queued for either Plan 2 or a separate cohort-expansion round.
- Flash-attention / mamba / diffusion cells with >60 MB working sets. These need .cu implementations to exist first; that's a separate prerequisite plan (NOT in scope here).

## Decision points the plan must surface

1. **Calibration application policy under the new precondition state**. If AC-5 reveals stddev >= 0.10 AND LOOCV p90 < DEC-T5 threshold, should Plan 1 apply Platt or stay typed-unavailable? Codex recommends "apply calibration only after A+B" — so the answer is NO (defer to Plan 2). Plan 1's AC-5 verdict is precondition-only.

2. **The L1 set-associative refinement**. Codex says "initial fully-associative approximation; set-associative refinement can remain deferred." Resolved per DEC-2: defer to Plan 2 UNLESS the fully-associative baseline produces rel_err > 50% on any single cell (user decision 2026-05-17; compromise stricter than Claude's 30% threshold and looser than Codex's unconditional deferral).

3. **The "track plan file" RLCR setting**. The prior loop ran with `--track-plan-file` (frozen plan during session). Plan 1 SHOULD also be frozen (per BL-20260510) — recommend the same setting.

4. **DEC-1-style adjudication on the AC-6 missing-feature scope**. The silu split is one instance; there are likely others (layernorm/rmsnorm both predict ~0.97 but truth differs by 0.07; less dramatic). Should AC-6 also investigate the layernorm/rmsnorm gap, or focus only on silu? Recommend: focus on silu (the canonical example) + name layernorm/rmsnorm as queued instances.

## Success criteria

Plan 1 closes when:
- L1 bypass classifier + capacity model + predictor head + comparison-runner L1 path all land with paired-control tests (AC-1 to AC-3)
- 6 additional L2 labels admitted on sm_86 via the operator runbook (AC-4)
- Cross-cell prediction stddev recomputed; verdict artifact committed (AC-5)
- Missing-feature candidate catalog committed (AC-6)
- Interim diagnostic report committed (AC-7)
- All 9 prior-loop ACs remain green (no regressions on AC-8 typed-unavailable schema/cardinality, no regressions on AC-9 loop discipline)
- AC-9 loop discipline observed: per-round Codex review, failing-test-first witnesses, post-commit recapture, BL selection per round
- New Tier-2-cross-arch follow-up entry in the goal-tracker explicitly carries forward the L1 + L2 corpus to sm_120/sm_100

Codex's pragmatic target ranges for the interim report:
- L2 geomean rel_err < 0.15 (was 0.319; expanded corpus + averaging in well-predicted cells should reach this)
- L1 geomean rel_err < 0.20 on the 13-cell expanded cohort
- Cross-cell L2 prediction stddev >= 0.10 (variance precondition satisfied so Plan 2 can apply calibration)
- L1 typed_unavailable arm fires correctly on cells with missing SASS provenance


## Primary Direction: Dual-Head Parallel Build

### Rationale

Build the L1 predictor head + expand the L2 labeled corpus + design the missing-feature catalog in one composite loop, with concurrent sub-streams (C and A can run in parallel per Codex). This is the most aligned with the idea body's proposed AC-1 through AC-7 structure.

### Approach Summary

**Dual-Head Parallel Build for L1 Baseline + L2 Corpus Expansion**

Implement three concurrent sub-streams (C and A run in parallel per Codex; B deferred to Plan 2):

1. **AC-1: L1 Bypass Classifier** (`tools/cache_predictor/l1_bypass_classifier.py`, ~150 LOC)
   - Consumes per-access `memory_op_class` field from observation_builder (already present in observations dict).
   - SASS-aware binary classification: L1-eligible (ld.global.{ca,cg,cs}, default LDG, ld.local) vs bypassing (ld.global.strong.sm, ld.global.strong.gpu, ld.volatile, atomics, async-copy).
   - Outputs per-record `l1_bypass_tag` (enum: "eligible", "bypassing", "unknown") + `sass_provenance_status` ("matched", "unavailable").
   - Typed-unavailable path: `predictor_l1_sass_provenance_missing` when cubin/SASS not available.

2. **AC-2: L1 Capacity Model Extension** (modify `tools/hrd_profiler/observation_builder.py`, ~100 LOC delta)
   - Per-SM L1 capacity = 192 KiB (sm_86). Carveout policy as provenance (4/16/32/64/100 KiB shared-memory carveout).
   - Add fields to launch record: `cache_model_l1_capacity_bytes`, `cache_model_l1_carveout_policy_source`.
   - LRU walker tracks per-SM warp-line (128 B) granularity. Fully-associative approximation (refinement deferred to Plan 2).
   - Bundle-side: extend cache_predictor.schema.json with L1-specific carveout + capacity provenance fields.

3. **AC-3: L1 Predictor Head** (`workloads/simbench_bw/cache_predictor.py::predict_l1_hit_rate`, ~300 LOC new entry point)
   - Mirrors L2 architecture: Smith-Eklund stack-distance (line-level, not sector), optional Maeda HRD second layer.
   - Input: per-launch aggregates from L1-bypassing-filtered observations.
   - Output shape: `{"result_kind": "scalar"/"bounded"/"typed_unavailable", "predicted_l1_hit_rate": float, ...}`.
   - New typed reason: `predictor_l1_sass_provenance_missing`.
   - Update comparison_contract.toml: change `cache_layers: ["L2"]` to `cache_layers: ["L1", "L2"]`.
   - New paired-control tests: L1 classification edge cases, L1 capacity gate, L1 vs L2 orthogonality.

4. **AC-4: L2 Corpus Expansion via Operator Runbook** (manual gate; no new code)
   - Operator runs `tools/local_labels/SWEEP_RUNBOOK.md` on local RTX 3090 Ti for 6 variance-expansion cells.
   - Cost: ~20 min/cell + 8 operator commands per cell.
   - Admits 6 new sm_86 label cells into `data/local_labels/label_admitted.json`.
   - Artifact: updated label_admitted.json with 13-cell cohort (7 prior + 6 new).

5. **AC-5: Variance Trajectory Recomputation + Verdict** (new artifact, ~50 LOC diagnostic script)
   - Recompute cross-cell prediction stddev over expanded 13-cell L2 cohort.
   - Emit `data/sm_86_predictor_accuracy_recovery_v2/ac5_variance_trajectory_post_corpus_expansion.json` with:
     - `cross_cell_stddev`: measured stddev after AC-4 labels admitted.
     - `variance_precondition_status`: "satisfied" (stddev >= 0.10) or "not_satisfied" (stddev < 0.10).
     - `stddev_prediction_trajectory`: formula per Codex's delta_pred rule (e.g., "expected stddev with N=13 cells: ...").
   - Decision artifact: does expanded corpus unlock the variance precondition for Plan 2 calibration?

6. **AC-6: Missing-Physics-Feature Discovery + Sidecar Schema Design** (~200 LOC + design doc)
   - Investigate silu correctness/timing split (canonical instance: both predict ~0.9375, truth differs by 0.3526 absolute).
   - Candidate covariates: per-warp occupancy delta, kernel launch parameters (blockDim, gridDim), DVFS state (record-only, no clock-lock), MSHR pressure proxy.
   - Emit `data/sm_86_predictor_accuracy_recovery_v2/ac6_missing_feature_candidates.json` with per-cell feature proposals.
   - Design trace sidecar schema additions (no model changes in Plan 1).
   - Name layernorm/rmsnorm gap (0.07 difference) as queued instance; focus only on silu.

7. **AC-7: Interim Diagnostic Report** (restructured from prior loop's report, ~500 LOC doc)
   - Per-cell L1 + L2 rel_err for expanded 13-cell cohort.
   - Per-bucket attribution (slice geometry / sector edge / write policy / CTA interleave / cross-launch / label-variance / missing-feature).
   - Quote variance-precondition verdict from AC-5.
   - Identify Plan-2-candidate cells (predictor-friendly, <10% likelihood) vs physics-challenge cells (thrash, silu split).

**Concurrency Model:**
- Sub-stream C (L1 head: AC-1, AC-2, AC-3) and sub-stream A (L2 corpus: AC-4, AC-5, AC-6) run in parallel.
- AC-7 depends on both and runs last.
- All ACs must land with green tests before AC-9 loop discipline recapture.

**Success Gate:** All 9 prior-loop ACs remain green + L1 rel_err < 0.20 + L2 geomean rel_err < 0.15 + variance precondition verdict artifact committed + missing-feature catalog committed.

### Objective Evidence

- `workloads/simbench_bw/cache_predictor.py` (1,245 LOC): Core L2 predictor with Smith-Eklund stack-distance + bootstrap CI; non-circular input validation discipline; per-launch aggregation from observation_builder; variance-precondition gate for calibration (lines 685-731). Precedent for an L1 sibling head following identical bundle/typed-unavailable patterns.
- `workloads/simbench_bw/comparison_runner.py` (200+ lines): Joins predictor bundles vs NCU labels on 7-field key, relative-error computation, geomean aggregation. Already pins `cache_layers: ["L2"]` in comparison_contract.toml (line 134); trivially extensible to `["L1", "L2"]`.
- `tools/hrd_profiler/observation_builder.py` (500+ LOC): Already emits `memory_op_class` field on every per-access observation (lines 486-495). MemoryOpClass enum: GLOBAL, SHARED, LOCAL, CLUSTER_DSMEM, ATOMIC. The cache_predictor already consumes this field for write-policy filtering (`_is_no_write_allocate` at line 174) — the L1 bypass classifier reuses the same per-access tagging.
- `tools/perturbation_harness/sass_confirm.py` (170 LOC): Cuobjdump-backed SASS extraction + LDG regex matching for cache modifiers. Per-arch SASS mnemonic tolerance (Ampere LDG.E.CA vs Blackwell LDG.E.STRONG.SM, lines 45-53). Kernel-scoped section extraction to avoid helper functions. This is the precedent for AC-1's SASS-aware classifier.
- `tools/hrd_profiler/observation_builder.py` lines 462-477: Per-slice L2 geometry (`num_l2_slices` parameter, slice-aware sector capacity). Mirror structure available for AC-2's per-SM L1 tracking.
- `workloads/simbench_bw/l1/` contains all 6 AC-4 target cells (working_set_capacity_sweep, dram_stream_stride_mix, carveout_sensitivity_probe, warm_cold_pair_relaunch, l2_persisting_window_sweep, l2_prefetch_hint_sweep) with .cu + NVBit trace infrastructure already in place; AC-4 only requires running the operator SWEEP_RUNBOOK (no new cell creation).
- `workloads/simbench_bw/tests/test_comparison_runner.py` + `test_cache_predictor.py`: Bidirectional structural lockstep tests + synthetic fixture builders (`_make_mpi`, `_make_input_digests`) + schema validation + non-circular discipline tests + typed-unavailable arm tests. Direct templates for new paired-control tests on L1 surfaces.
- `workloads/simbench_bw/schemas/cache_predictor.schema.json`: `result_kind` enum already covers `["scalar", "bounded", "typed_unavailable"]`; extend `_TYPED_REASONS` frozenset for L1-specific reasons (e.g., `predictor_l1_sass_provenance_missing`).
- `tools/local_labels/SWEEP_RUNBOOK.md`: Stable operator interface for running NCU + repeatability aggregation; already supports adding new candidate cells via `--candidates <ARCH>/<SLUG>/<SHAPE_POINT>` flags (line 74). Cost per cell: ~20 min × 40 NCU invocations + 8 operator commands.
- `data/local_labels/label_admitted.json`: Current admitted set is 7 sm_86 cells; schema shows cells carry `cache_hit_rate_l2_sector_pct` + `stddev_l2_sector` + `cache_hit_rate_l1tex_sector` + `confidence_interval_l1tex_sector` (L1 truth is ALREADY collected for 23+30 cells — only the L1 predictor is missing).
- `workloads/simbench_bw/cache_predictor.py` lines 685-731: Variance-precondition gate already implemented; `_VARIANCE_PRECONDITION_FLOOR = 0.10` constant; returns typed_unavailable with `headline_validity_failed_reasons: ["calibration_variance_precondition_failed"]` when stddev < 0.10. AC-5 only requires recomputing cross-cell stddev over expanded 13-cell cohort and emitting diagnostic artifact.
- `data/sm_86_predictor_accuracy_recovery/ac4_per_slice_ablation.json`: Existing AC artifact structure showing per-cell diagnostics with per-slice residual attribution. AC-6 mirrors this pattern: emit `ac6_missing_feature_candidates.json` with per-cell covariate proposals.

### Known Risks

- **L1 carveout policy ambiguity on deployed binaries:** The .cu source may not encode the carveout intent. Mitigation: AC-2 sources carveout from the comparison_contract.toml pinned value or a device_properties manifest entry, with typed-unavailable arm for missing provenance.
- **SASS provenance unavailability for cloud traces:** sm_120 traces may not have accessible cubin/SASS. Mitigation: AC-1 types this as `predictor_l1_sass_provenance_missing` (fail-closed); L1 predictor emits typed_unavailable on cloud traces. Plan 1 scope is sm_86-only per CLAUDE.md Rule 1.
- **Variance floor (0.10) may not be achievable with 13 cells:** Codex's trajectory rule suggests 4-6 cells sufficient, but empirical variance distribution may differ. Mitigation: AC-5 reports actual stddev; if still < 0.10, Plan 2 decides whether to apply calibration or defer further corpus expansion.
- **Missing-feature discovery scope creep:** Silu split is clear, but layernorm/rmsnorm/others may each warrant investigation. Mitigation: AC-6 focuses on silu only; names layernorm/rmsnorm as queued; Plan 2 prioritizes by rel_err severity.
- **L1 fully-associative approximation may be too coarse:** Set-associativity could matter for high-contention workloads. Mitigation: AC-3's rel_err > 30% on any cell triggers Plan 2 promotion of set-associative refinement. Plan 1 accepts fully-associative as baseline.
- **Test-first discipline on AC-1 classifier:** Requires SASS fixtures + per-arch mnemonic coverage before implementation. Mitigation: Use existing sass_confirm.py test fixtures as templates; paired-control tests authored alongside the classifier stub.

## Alternative Directions Considered

### Alt-1: Cohort-Stratified Predictor Ensemble
- Gist: Instead of one comprehensive predictor build, leverage the existing R21 `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` partition (8 friendly + 5 hostile cells) as a routing mechanism. Build a shallow classifier that dispatches per-cell to specialist heads: friendly cells use the existing Smith-Eklund scalar path; hostile cells route to `result_kind="typed_unavailable"` (with new typed reason `predictor_model_insufficient_for_bucket_coverage`) OR `result_kind="bounded"` (via isotonic quantile regression on 5-cell pairs). The pinned `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` becomes the model selector rather than just a diagnostic label. Implementation surface: ~50-100 LOC in `cache_predictor.predict_cache_hit_rate()` for cell-keyed dispatch + ~20-40 LOC in comparison_runner + 5-10 new paired-control tests + 1 new enum entry in `cache_predictor.schema.json`. Honors Codex's "silu_correctness already at 5% rel_err proves the predictor CAN be accurate" by routing predictor-friendly cells to the working scalar path and fail-closing on cells where Smith-Eklund cannot model the dominant physics.
- Objective Evidence:
  - `workloads/simbench_bw/comparison_runner.py:547-637` — pinned `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` (13-cell dict), `_PREDICTOR_DESIGN_COVERAGE_BUCKETS` tuple, lockstep test `test_predictor_design_coverage_partition.py::test_contract_pin_set_equals_headline_cohort`.
  - `workloads/simbench_bw/comparison_runner.py:2087-2127` — every included-cell record already carries `"predictor_design_coverage": bucket_label` (schema enforces enum).
  - `workloads/simbench_bw/comparison_runner.py:1601-1633` — `_per_bucket_relative_error_stats()` already aggregates per-cell errors by bucket; emits `relative_error_stats_per_predictor_design_coverage` under `per_arch_diagnostics.per_arch.<arch>`.
  - `workloads/simbench_bw/cache_predictor.py:270-329` — `_typed_unavailable_bundle()` constructor used for existing fail-closed conditions; lockstep with `cache_predictor.schema.json` lines 29-63.
  - `workloads/simbench_bw/schemas/cache_predictor.schema.json` lines 160-205 + `comparison_bundle.schema.json` lines 453-510 — `result_kind="bounded"` + `hit_rate_lower`/`hit_rate_upper` schemas + per-arch `bounded_arm_relative_error_stats` aggregator are already implemented and tested.
  - Prior-art reference: `plans/l2_predictor_accuracy_recovery_plan.md` lines 411-417 explicitly proposes this design as "Alt-3 Workload-Class Stratification" with ~50 LOC surface estimate.
- Why not primary: Strategically depends on AC-4 (Plan 2's L2 physics features) NOT closing the friendly-cell gap; if AC-4 closes the gap uniformly, the ensemble becomes unnecessary. Tactically smaller than the primary (~50 LOC vs ~1700 LOC), but solves a narrower problem (model dispatch only, no L1 baseline, no corpus expansion).

### Alt-2: Empirical Regression Correction
- Gist: Skip new physics entirely; fit a learned correction layer (sklearn `LinearRegression` or `GradientBoostingRegressor` with max_depth=3-4) that maps `[predicted_hit_rate, grid_dim_total, block_dim_total, num_registers, slug_onehot, shape_point_onehot]` to NCU truth via a residual delta Δ. Inference: `corrected = clip(predicted + Δ(features), 0, 1)`. Mirrors the existing `calibration_head.py` 1D Platt/Isotonic/Beta fit pattern but with a multi-feature regressor. Training set is the 5 admitted sm_86 cells (N=5-7 unique pairs); LOOCV p90 from `fit_calibration_head.py` template. Fallback: revert to PlattCalibrator if LOOCV p90 > 0.20.
- Objective Evidence:
  - `workloads/simbench_bw/calibration_head.py` — existing 1D calibration framework (Platt, Isotonic, Beta) with `fit(predicted, label, weights=None)` + `transform(predicted) -> (calibrated, out_of_envelope_mask)` interface; extensible to multi-feature regressor.
  - `tools/local_labels/fit_calibration_head.py` lines 200-228 — `_loocv_residuals()` function provides ready LOOCV pipeline; only feature matrix changes from 1D to N-D.
  - `requirements.txt` — numpy 2.4.5, scipy 1.17.1, scikit-learn 1.8.0 already pinned (no new dependencies).
  - `tools/nvbit_tracer/traces_enhanced/dynamic_trace/kernel.proto` lines 5, 8, 9 — kernel.proto already captures `grid_dim`, `block_dim`, `number_of_registers` per launch.
  - `data/local_labels/label_admitted.json` — per-cell `n_repeats`, `stddev_l1tex_sector`, `stddev_l2_sector` already available as features.
  - CLAUDE.md Rule 5 ("no simulator revival") does NOT apply: a learned regressor is empirical data-fitting (identical in spirit to existing Platt/Isotonic/Beta calibrators), not a discretized hardware simulator.
- Why not primary: N=5-7 training cells is below the rule-of-thumb 10×(feature_count) minimum; high overfitting risk; slug-specific generalization unproven; sidesteps the physics-modeling problem rather than solving it. The user's stated goal includes "understanding cache modeling" which a black-box regression undermines.

### Alt-3: Schema-First Instrumentation
- Gist: Defer all model changes; the entire Plan 1 becomes "expand the trace + sidecar schema to capture missing covariates on existing 7 cells + correlate which covariates explain the silu split." NO L1 predictor build, NO L2 corpus expansion, NO new predictor code. Just join the data that's ALREADY being collected (in 4 separate sidecar streams) into per-launch records, then run a correlation analysis on the silu correctness/timing pair. Deliverable: `workloads/simbench_bw/schemas/launch_parameters_sidecar.schema.json` (v1) + `data/sm_86_predictor_accuracy_recovery_v2/ac6_silu_split_diagnostic_candidates.json` (ranked covariate table) + `tools/local_labels/extend_launch_parameters.py` utility. All 5 candidate covariates (launch geometry, occupancy, phase duration vector, DVFS state, MSHR proxy) are ALREADY collected in `cupti_activity_sidecar.schema.json`, `nvml_covariate.schema.json`, `phase_cycle_ledger.schema.json`, and `kernel.proto` — only the per-launch JOIN + correlation analysis is missing.
- Objective Evidence:
  - `workloads/simbench_bw/schemas/cupti_activity_sidecar.schema.json` lines 114-131 — grid/block/regs_per_thread/shared_mem fields already captured per launch.
  - `workloads/simbench_bw/schemas/nvml_covariate.schema.json` lines 68-96 — per-repeat NVML samples (timestamp_ns, power_mw, temp_c, throttle_reasons, sm_clock_mhz, mem_clock_mhz) at >=100 Hz already collected.
  - `workloads/simbench_bw/schemas/phase_cycle_ledger.schema.json` — per-phase cycle counts at semantic boundaries (%clock64 injection).
  - `tools/nvbit_tracer/traces_enhanced/dynamic_trace/kernel.proto` lines 8-26 — kernel message carries grid_dim, block_dim, cluster_dim.
  - `workloads/simbench_bw/schemas/kernel_feature_vector.schema.json` lines 153-169 — `memory_access_stride_distribution` already analyzed (stride_buckets_bytes + stride_counts).
  - `workloads/simbench_bw/schemas/measurement_manifest.schema.json` lines 100-134 — `enable_cupti_activity`, `nvml_polling_interval_ns`, `nvml_sidecar_paths` already wired in for timing runs.
  - `tools/local_labels/sidecar_emitter.py` lines 27-115 — `build_sidecar()` constructs sidecars from environment variables; extending shape is non-breaking (new fields simply do not appear in env-var set if absent).
- Why not primary: Solves only the missing-feature-discovery slice; does not deliver L1 baseline, L2 corpus expansion, or variance-precondition unlock. The user's idea body explicitly scopes all four deliverables. Schema work alone does not move the headline rel_err and would require a second loop to act on the findings.

### Alt-4: Per-Shape-Point Specialization
- Gist: Surgical fix targeting only the canonical silu correctness/timing split (both predict 0.9375; truth differs by 0.35 absolute). A new committed pin map at `data/local_labels/shape_point_parameter_pins.json` keyed by `(slug, shape_point)` carries a scalar `hit_rate_adjustment_factor` (e.g., 0.85 for `silu_fwd_bf16_sm80/timing_shape`). Comparison runner applies the multiplier post-join: `predicted_adjusted = clip(predicted * factor, 0, 1)`. ~80-120 LOC in comparison_runner + ~50 LOC for the pin map loader.
- Objective Evidence:
  - `workloads/simbench_bw/cache_predictor.py:1086-1090` — `manifest_pairing_index` is guaranteed present in every emitted bundle (scalar and typed_unavailable arms); `slug` + `shape_point` are accessible.
  - `workloads/simbench_bw/comparison_runner.py:2052-2077` — scalar-arm join site is the natural insertion point for a multiplier (no new bundle fields needed if the adjustment stays consumer-side).
  - `tools/hrd_profiler/device_properties.json` — committed JSON pin-map precedent (schema_version, entries[]).
  - Grep confirms `shape_point` is currently ONLY used as a join_key field; no prior art for shape_point as a model input or per-cell parameter override.
- Why not primary: No prior art for per-cell parameter overrides; speculative; violates the plan's AC-5 ordering gate (the multiplier is a form of local calibration applied before AC-4 sub-slices have demonstrated variance improvement); overfits on 13-cell LOOCV; non-generalizable (each new cell requires a manual pin map entry).

### Alt-5: Trace Amplification + Repeats
- Gist: Instead of new physics or new cells, COLLECT more repeats per cell (30 instead of 5; sqrt(6)=2.45x CI tightening reduces per-cell stddev from ~0.025 to ~0.010) AND patch the NVBit trace producer to persist multi-launch traces (currently the producer emits one `dynamic_trace.pb` per shape directory; rmsnorm/softmax are conceptually multi-launch but committed as single-launch). The multi-launch capture would FINALLY give the existing AC-4.5 `cross_launch_state_carryover` infrastructure something to operate on (currently `cross_launch_state_residual = 0.0` because every committed trace has n_launches=1). Plan 1 = bump `--repeats` flag in sweep matrix + Protobuf schema bump (add `launch_index` field to `Kernel` message) + activation_trace_orchestrator emits per-launch traces.
- Objective Evidence:
  - `tools/local_labels/sweep_repeatability_matrix.py` lines 875-877 — `--repeats` flag default is 5; parameterized, no hardcoded constant.
  - `data/sm_86_predictor_accuracy_recovery/residual_root_cause_report.md` lines 62, 85-86 — `label_variance_floor = 0.050` geomean contribution; per-cell `stddev_l2_sector` 0.01-0.03 in `label_admitted.json`.
  - `tools/hrd_profiler/observation_builder.py` lines 330-339, 626-646 — `_walk_launch(cross_launch_state_carryover=...)` parameter + per-launch LRU tracker initialization already implemented; just lacks live multi-launch data.
  - `tools/local_labels/activation_trace_orchestrator.py` lines 79-83 — parses workload-supplied `n_launches` at sidecar level; tracer Protobuf output is the bottleneck.
  - `tools/nvbit_tracer/traces_enhanced/pb_trace/include/kernel.pb.h` — Protobuf `Kernel` message extensible with a new `uint32 launch_index` field (backward-compatible proto3 addition).
- Why not primary: 6x repeat scaling implies ~21-28 hours wall-clock per full sweep on RTX 3090 Ti (sequential only); multi-launch capture has uncertain payoff (AC-4.5 plan spec estimates 0.5-2% relative-error improvement, could be measurement noise); variance-floor reduction alone closes only ~0.025 of the 0.319 headline residual; the silu split (a missing-feature problem, not a measurement-variance problem) remains untouched.

## Synthesis Notes

The primary direction (Dual-Head Parallel Build) is structurally sound but borrows critical refinements from the alternatives. Two foldable enhancements:

1. **From Alt-3 (Schema-First)**: AC-6's missing-feature discovery should explicitly REUSE the existing CUPTI Activity / NVML / phase-cycle-ledger / kernel.proto schemas rather than design a new sidecar from scratch. Alt-3's audit confirms ALL 5 candidate covariates (launch geometry, occupancy, phase duration, DVFS state, MSHR proxy) are already collected in production schemas — AC-6's deliverable becomes a JOIN utility (`extend_launch_parameters.py`) + a correlation analysis on the existing 7 cells, not a new schema. This reduces AC-6's implementation surface by ~150 LOC and accelerates the variance-trajectory verdict.

2. **From Alt-1 (Cohort-Stratified Ensemble)**: AC-3's L1 predictor head should adopt the `predictor_design_coverage` bucket label for L1 cells from day one, mirroring the R21 partition. The L1 typed_unavailable arm (`predictor_l1_sass_provenance_missing`) naturally maps to the same fail-closed pattern Alt-1 proposes. If Plan 2 later promotes the ensemble dispatch, L1 cells inherit the partition without retrofitting. This adds ~20 LOC to AC-3 but future-proofs against Plan 2's dispatch decision.

Three rejected enhancements:

- **Alt-2 (Empirical Regression)**: rejected because the user's stated goal includes "understanding cache modeling," and a black-box regression undermines diagnostic interpretability. Mentioned in AC-6 as a "could be Plan 2 fallback if physics modeling stalls" but not part of Plan 1.
- **Alt-4 (Per-Shape-Point Specialization)**: rejected because the multiplier approach overfits on 13-cell LOOCV and conflicts with the AC-5 ordering gate. The silu split is better addressed by AC-6's covariate discovery (a physics solution) than by a manual pin map (a patch).
- **Alt-5 (Trace Amplification + Repeats)**: deferred to a future infrastructure round. Multi-launch capture has clear value for AC-4.5 but requires Protobuf schema work + tracer rebuild that exceeds Plan 1's scope. Repeat amplification alone (sqrt(N) variance reduction) closes only ~0.025 of the 0.319 headline residual and trades operator wall-clock 6x for marginal benefit; better to spend that budget on AC-4's 6 new cells (variance expansion across the cohort, not variance reduction per cell).

Adopting enhancements 1 and 2 above leaves the primary direction's 7-AC structure intact while reducing the implementation surface from ~1700 LOC to ~1550 LOC and tightening AC-3/AC-6 against future architectural promotions.

--- Original Design Draft End ---
