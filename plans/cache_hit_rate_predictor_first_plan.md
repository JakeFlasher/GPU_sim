# Cache-Hit-Rate Predictor v1: Non-Circular Analytical Baseline + Comparison Contract + Coverage Scaffold

## Goal Description

Stand up the first analytical cache-hit-rate predictor for the project's non-root GPU profiling stack, anchored on a NON-CIRCULAR design (the predictor consumes RAW reuse-distance evidence — per-launch reuse-distance histograms recomputed from the producer-side `sector_stream/*.jsonl` records — NOT the HRD profiler's already-modeled `hit_count` / `hit_rate` fields), pinned to an explicit predictor↔NCU comparison contract (NCU metric definition, 7-field join key, error formula, epsilon policy, admitted-cell coverage gate), and sequenced through milestones that resolve the four blocking issues (label stabilization, predictor build, HRD coverage scaling, cross-arch ablation) toward a reportable predictor-vs-NCU geomean relative error number on the local sm_120 + sm_86 evidence corpus. The downstream goal is <10% (and ideally <5%) geomean relative error suitable for top-tier academic conference submission (ISCA / MICRO / HPCA / SIGMETRICS / ASPLOS / SC).

The plan is repo-grounded. The Smith-Eklund LRU stack-distance algorithm is already implemented at `tools/hrd_profiler/reuse_distance_tracker.py` (Fenwick-tree O(log N) tracker) and the modeled-hit rule lives at `tools/hrd_profiler/observation_builder.py:34-46` (`sector_reuse_distance < cache_model_spec["capacity_bytes"] // 32`). The bootstrap-CI machinery is pinned at `tools/hrd_profiler/profiler.py:40-66` (`BOOTSTRAP_RESAMPLES=2000`, `CI_HALF_WIDTH_TARGET=0.05`, `MINIMUM_BIN_SAMPLE_COUNT=50`, kernel-launch resampling unit). The 45/16/16 calibration/validation/holdout split exists at `workloads/simbench_bw/splits.toml`. The strict-schema admission pattern (typed-exception / typed-unavailable + `to_json()` + `TOKEN`) lives at `workloads/simbench_bw/phase_cycle_ledger_consumer.py:44-109`. The per-shape sidecar verifier loop with full 5-field `manifest_pairing_index` provenance check (slug + shape_point + binary_sha256 + source_commit + kernel_signature_sha256) was widened in commit `d2ec483` and is the reference for the new predictor verifier hook in `workloads/simbench_bw/verify_artifacts.py`. The new predictor module is a thin verifier-side consumer wrapping these existing surfaces, NOT a from-scratch implementation.

The "non-circular" framing is critical (Codex first-pass core-risk #2): if the predictor merely re-aggregates the HRD profiler's already-modeled `hit_count` / `hit_rate` fields, it is an aggregator, not a predictor. The plan binds the predictor's input to RAW reuse-distance evidence (per-record sector access stream + Fenwick-tree-recomputed histograms), so the prediction is independent of the HRD profiler's modeled output even though the underlying algorithm is shared.

The "reportable number" framing is bounded by the comparison contract (Codex first-pass core-risk #1, #3, #4): no headline predictor-vs-NCU geomean relative error claim is admissible until the contract pins the NCU metric (e.g., `lts__t_sectors_lookup_hit / lts__t_sectors`, sector-basis hit rate fraction), the 7-field join key, the error formula (`|predicted − ground_truth| / max(ground_truth, epsilon)`), and the epsilon policy. The plan defers the headline number to Milestone 4; earlier milestones surface predictor-self-consistency (CI half-width per kernel) and admitted-cell coverage diagnostics, which are reportable progress without requiring stable labels.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Predictor schema is strict and provenance-bound.
  - Positive Tests:
    - Schema file at `workloads/simbench_bw/schemas/cache_predictor.schema.json` validates with `additionalProperties: false`; `required` keys include `schema_version`, `algorithm_version`, `manifest_pairing_index` (5 fields per the R4 widening), `input_sidecar_digests` (object listing per-input sidecar SHA-256), `result_kind` (one of `scalar` / `bounded` / `typed_unavailable`), `bootstrap_seed`, `bootstrap_resamples`, `epsilon`, `cache_model_capacity_bytes`, plus result-arm-specific fields (`predicted_hit_rate` + `ci_low` + `ci_high` + `ci_half_width` for `scalar`; `hit_rate_lower` + `hit_rate_upper` for `bounded`; `typed_reason` for `typed_unavailable`).
    - A complete fixture bundle admits via `verify_artifacts.py` with the new `_verify_cache_predictor_run_summaries` verifier function mirroring the R4 divergence_proxies pattern.
  - Negative Tests:
    - An extra unknown field rejects with strict-schema typed diagnostic.
    - Mismatched `kernel_signature_sha256` between bundle's `manifest_pairing_index` and the run summary's `kernel_signature_sha256` rejects with `BudgetViolationUntracked: cache_predictor manifest_pairing_index.kernel_signature_sha256 ...` mirroring the R4 divergence_proxies error format.
    - Mismatched `binary_sha256` rejects similarly.

- AC-2: Predictor is non-circular (consumes raw reuse-distance evidence, NOT HRD profiler's modeled hit_rate).
  - Positive Tests:
    - The predictor's input ingestion code at `workloads/simbench_bw/cache_predictor.py` reads per-launch reuse-distance histograms recomputed from `sector_stream/*.jsonl` records via the Fenwick-tree tracker at `tools/hrd_profiler/reuse_distance_tracker.py`; the predictor's `input_sidecar_digests` lists `sector_stream` (and optionally raw `dynamic_trace.pb`) but NOT `hrd_profile`.
    - Unit test against a synthetic reuse-distance fixture (with hand-computed expected hit rate via Smith-Eklund integration) passes.
  - Negative Tests:
    - A test that wires `hrd_profile.json`'s `hit_count` / `hit_rate` field as predictor input rejects with typed `circular_predictor_input_rejected`.
    - A `manifest_pairing_index.input_sidecar_digests` field that does NOT include `sector_stream` rejects.

- AC-3: Predictor emits typed-unavailable cleanly on currently-underconfigured trace_00 captures.
  - Positive Tests:
    - The current `dram_stream_stride_mix/trace_00` capture on sm_120 (24,577 accesses, `bundle_underconfigured=true`) returns `result_kind=typed_unavailable` with `typed_reason=hrd_layer_unavailable` or `cache_model_underconfigured`.
    - The captured trace bundle's existing `bundle_underconfigured=true` propagates to the predictor's typed-unavailable arm.
  - Negative Tests:
    - The predictor does NOT emit `result_kind=scalar` when `bundle_underconfigured=true`.
    - An attempt to compute geomean over a `result_kind=typed_unavailable` cell rejects via the typed coverage-table check.

- AC-4: Comparison contract pinned (NCU metric, join key, error formula, epsilon policy).
  - Positive Tests:
    - A new `tools/comparison_contract/contract.toml` (or `workloads/simbench_bw/comparison_contract.toml`) pins:
      - **NCU metric**: explicit name and definition (e.g., `lts__t_sectors_lookup_hit_rate.pct` for L2 sector-basis hit rate, with documented numerator / denominator).
      - **7-field join key**: `(slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256, arch, ncu_metric_version)`.
      - **Error formula**: relative error = `|predicted − ground_truth| / max(ground_truth, epsilon)` (with ground_truth and predicted both expressed as fractions ∈ [0, 1]).
      - **Epsilon policy**: `epsilon = 0.01` (rationale: hit rates near 0 destabilize relative error; absolute-error fallback when ground_truth < epsilon).
      - **Geomean aggregation**: log-space mean of relative errors across admitted cells, bootstrap CI per the same `BOOTSTRAP_RESAMPLES=2000` discipline.
      - **Admitted-cell coverage gate**: minimum N admitted cells per arch before a headline geomean is claimed (N pinned via DEC-3 user decision).
    - The comparison runner consumes only admitted (label × predictor) cells; rejects unmatched cells with typed diagnostic.
  - Negative Tests:
    - A predictor cell joining to an NCU label whose `ncu_metric_version` differs is rejected.
    - A geomean computation when admitted-cell count is below the AC-4 gate rejects with typed `headline_geomean_admitted_cell_count_below_threshold`.
    - A geomean computation that uses an error formula or epsilon different from the contract rejects with typed `comparison_contract_drift`.

- AC-5: NCU labels stabilized via experimentally-isolated knob matrix.
  - Positive Tests:
    - The label-stabilization experiment matrix isolates four knobs (warmup_count ∈ {0, 1, 5, 10}, replay_mode ∈ {kernel, application}, repeats_required ∈ {5, 10, 20}, optional inter-launch L2 flush ∈ {off, on}) and admits cells where the resulting L2 stddev clears the existing 0.03 threshold.
    - At least one of the two formerly-excluded cells (sm_120 fp32_fma_dep_chain trace_00 OR sm_86 gemm_hmma_bf16_sm80 correctness_shape) admits via the matrix.
    - `data/local_labels/label_admitted.json` gains at least one cell on each arch (or, where one arch fails to admit, the typed-failure record explicitly documents which knobs were attempted and which residual variance source remains).
  - Negative Tests:
    - The `stddev_threshold` is NOT loosened from `0.03` in this loop (loosening requires a separate dedicated round with an explicit pre-registered justification).
    - Switching `replay_mode` does NOT bypass the per-cell `binary_sha256_consistent` check; the recorded sidecar's `replay_mode` field reflects the actual mode used.
    - A label cell whose 5+ repeats mix `warmup_count` or `replay_mode` across repeats is rejected as protocol-incoherent.

- AC-6: HRD coverage scaling preserves workload identity (trace_50 / trace_75 admit only when invariance gates pass).
  - Positive Tests:
    - At least 3 memory-access slugs (e.g., `dram_stream_stride_mix`, plus 2 of `l2_thrash_pointer_chase`, `warm_cold_pair_relaunch`, `carveout_sensitivity_probe`) gain `trace_50` and / or `trace_75` shape definitions in `workloads/simbench_bw/shapes/<slug>.toml`.
    - Each new shape's HRD bundle admits with `bundle_underconfigured=false`.
    - Each new shape passes a workload-identity invariance gate: `kernel_signature_sha256` matches `trace_00`'s, `predicted_bytes` is within `trace_budget_mib`, and structural ratios (e.g., `divergence_proxies_sector_stream_records_consumed / trace_access_count`) stay within a documented tolerance.
    - The new shapes appear in admitted captures on the laptop (sm_120) and on the 3090 Ti (sm_86) when the cross-host transport delivers them.
  - Negative Tests:
    - A shape that clears bin counts by changing the memory footprint class (e.g., grid expansion that crosses an L2-fit boundary) is rejected as a different workload cell with typed `workload_identity_invariance_violated`.
    - A shape that exceeds `trace_budget_mib` is rejected with typed `trace_volume_exceeded`.
    - The 34-bucket reuse-distance partition at `tools/hrd_profiler/profiler.py:43-66` is NOT modified in this loop (alternative bin partitions are deferred to a separate dedicated round).

- AC-7: Bounded prediction interval for underconfigured cells.
  - Positive Tests:
    - When `bundle_underconfigured=true` but at least one bin admits, the predictor emits `result_kind=bounded` with `hit_rate_lower` (treat typed-unavailable bins as misses) and `hit_rate_upper` (treat typed-unavailable bins as hits).
    - The bounded interval admits via the strict schema and is consumable by the geomean calculator (which records both bounds and reports geomean over the lower / upper / midpoint variants for transparency).
  - Negative Tests:
    - The bounded path does NOT silently drop typed-unavailable bins and report a scalar.
    - `hit_rate_lower > hit_rate_upper` is rejected as malformed.
    - The bounded path is NOT used when `bundle_underconfigured=false` (in that case `result_kind=scalar` is required).

- AC-8: Geomean is reproducible.
  - Positive Tests:
    - The geomean computation is gated on a pinned `admitted_cells[]` list, a pinned `bootstrap_seed`, the AC-4 epsilon policy, and the AC-4 error formula.
    - The geomean output bundle records `admitted_cells`, `bootstrap_seed`, `epsilon`, `error_formula`, `geomean_relative_error`, `geomean_ci_low`, `geomean_ci_high`, `bootstrap_resamples`.
    - Re-running with the same admitted-cell list + seed produces a byte-identical geomean output.
  - Negative Tests:
    - Re-running with a perturbed admitted-cell list OR seed produces a measurable delta the runner reports as `geomean_drift_detected`.
    - The geomean computation does NOT silently include `result_kind=typed_unavailable` cells.

- AC-9: Cross-arch transfer is gated on admitted labels on both arches.
  - Positive Tests:
    - When both `data/local_labels/sm_86/` and `data/local_labels/sm_120/` carry at least N admitted cells (N pinned per DEC-3), the transfer-learning evaluation reports per-arch MAPE plus train-on-sm_86-test-on-sm_120 MAPE plus train-on-sm_120-test-on-sm_86 MAPE.
    - The covariate / invariant feature split mirrors the AC-8 (R0/R3) decomposition documented in `.humanize/rlcr/2026-05-08_22-54-53/round-0-summary.md` (must-match invariants vs expected-to-differ covariates).
  - Negative Tests:
    - Cross-arch evaluation rejects when one arch has zero admitted cells.
    - A schema-version mismatch between sm_86 v7 baseline labels and sm_120 fresh labels rejects with typed `cross_arch_label_schema_drift`.
    - Training on labels with mismatched `ncu_metric_version` rejects.

- AC-10: Prior-art comparison is citation-safe and scope-bounded.
  - Positive Tests:
    - A comparison section in the round summary cites NVArchSim (HPCA 2021), Accel-Sim (MICRO 2020 + extensions 2023-2024), GPGPU-Sim (MICRO 2009), and the Smith-Eklund cache paper (2014) by paper title + venue + year.
    - For workloads / metrics that do NOT match published prior-art harnesses, the comparison is explicitly framed as `contextual_only`, NOT head-to-head.
  - Negative Tests:
    - A claim of "head-to-head superiority" for a workload / metric / arch combination not normalized against the prior-art paper rejects with typed `prior_art_comparison_unmatched_workload`.
    - AccelWattch is NOT cited (not in the local references corpus); the round summary explicitly notes its exclusion from the comparison set.
    - No simulator binary is set up or invoked (Rule 5 anti-goal).

- AC-11: Per-record L1 bypass detection scaffold (lands; default off).
  - Positive Tests:
    - A new `tools/cache_predictor/l1_bypass_classifier.py` consumes per-record sector_stream entries plus the per-PC SASS classifier output (Ampere `LDG.E.CA/CG/CS` vs Blackwell `LDG.E.STRONG.SM/STRONG.GPU/EF` per BL-20260505) and routes each access to L1 / L2 individually.
    - The per-record routing is opt-in (gated by a config flag, default off); the default predictor still uses the binary-level `l1_bypass_cases` annotation from `cache_model_capacity_source`.
    - Unit tests pin the classifier output for synthetic SASS-disassembly fixtures on both arches.
  - Negative Tests:
    - The default predictor's accuracy is NOT regressed by the L1-bypass scaffold's mere presence.
    - The L1-bypass routing does NOT widen the SASS regex per BL-20260505 anti-goal lockstep (it consumes the existing widened regex from `tools/perturbation_harness/sass_confirm.py:45-53`).
    - The instruction.proto wire format is NOT modified in this loop (per-record cache-modifier metadata extension is deferred).

- AC-12: Set-associative refinement is deferred behind an ablation gate.
  - Positive Tests:
    - The plan documents an explicit ablation ladder: fully-associative-LRU baseline → per-record L1 bypass → bounded-associativity (RRIP/DRRIP) → ATD sampling — each step gated on the prior step's residual error not closing to <5%.
    - The schema's `associativity_policy` enum stays restricted to `["fully_associative_lru_approximation"]` until the ablation gate triggers.
  - Negative Tests:
    - RRIP / DRRIP / ATD code does NOT land in this loop unless the ablation gate triggers.
    - The schema is NOT widened pre-emptively.

- AC-13: Test surface lands.
  - Positive Tests:
    - Unit tests at `workloads/simbench_bw/tests/test_cache_predictor_*.py` cover: synthetic reuse-distance fixture round-trip (Smith-Eklund integration vs hand-computed expected output); strict schema validation; typed-unavailable arm; bounded-interval arm; manifest_pairing_index 5-field cross-check (mirroring R4 divergence_proxies); comparison-contract join + error formula + epsilon policy + geomean reproducibility.
    - Integration test with a real (admitted) trace bundle + (eventually) admitted NCU label produces a non-typed-unavailable scalar prediction OR bounded interval and joins cleanly to the comparison contract.
    - `python3 -m pytest tools/test_plan_token_scanner.py -q` continues to pass after the new test files land (no `Round N` / `AC-N` / `Milestone` / `task1..N` markers in test docstrings or assertion messages).
  - Negative Tests:
    - A test that fails by accidentally consuming the HRD profiler's `hit_rate` field rejects with the AC-2 circularity check.
    - A test docstring that contains forbidden plan-token markers fails the scanner.

- AC-14: Anti-goals and CLAUDE.md hard rules honored throughout.
  - Positive Tests:
    - All NCU invocations live under `tools/local_labels/` and write only to `data/local_labels/` (Rule 1).
    - No `nvidia-smi` clock-lock / persistence / MIG / compute-mode-mutation invocations land (Rule 2).
    - Only CUPTI Activity is used; no Events / Metrics / Range Profiling / PM Sampling / PerfWorks (Rule 3).
    - No host-driver work (Rule 4). No simulator revival (Rule 5).
  - Negative Tests:
    - `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` continues to pass.
    - Tracker arch token sets, dram workload's `ARCH_GATE`, HANDOFF.md, top-level READMEs, per-tree READMEs, BLOCKED_ON_3090TI.md are NOT modified in this loop (per the carry-forward anti-goal).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

All 14 ACs land in the loop: predictor module + strict schema + admission helper + verifier hook (AC-1, AC-2, AC-3); comparison contract pinned with admitted-cell coverage gate (AC-4); label stabilization knob matrix admits at least one cell per arch (AC-5); HRD coverage scaling delivers at least 3 memory-access slugs with trace_50 / trace_75 (AC-6); bounded-interval arm for underconfigured cells (AC-7); reproducible geomean with bootstrap CI (AC-8); cross-arch transfer evaluation reports per-arch + transfer MAPE (AC-9); prior-art comparison cites NVArchSim / Accel-Sim / GPGPU-Sim / Smith-Eklund as contextual prior art (AC-10); per-record L1 bypass scaffold lands as opt-in (AC-11); set-associative ablation ladder documented but RRIP / DRRIP code not yet landed (AC-12); test surface complete (AC-13); anti-goals honored (AC-14). The first reportable predictor-vs-NCU geomean relative error number lands at the loop close.

### Lower Bound (Minimum Acceptable Scope)

AC-1, AC-2, AC-3, AC-4, AC-5 (or its typed-failure variant), AC-7, AC-8, AC-13, AC-14. The predictor module + comparison contract + bounded-interval arm + reproducible geomean + tests + anti-goals land; label stabilization (AC-5) lands either as an admitted-cell increment OR as an explicit typed-failure record documenting residual variance source; HRD coverage scaling (AC-6), cross-arch ablation (AC-9), prior-art comparison (AC-10), L1-bypass scaffold (AC-11) defer to follow-up loops. The first reportable number is "predictor self-consistency" (CI half-width per kernel + admitted-cell coverage diagnostic), not the headline predictor-vs-NCU MAPE — the headline number is a follow-up loop's deliverable.

### Allowed Choices

- Can use: existing `tools/hrd_profiler/reuse_distance_tracker.py` Fenwick-tree implementation; existing `phase_cycle_ledger_consumer.py` admission-helper pattern; existing `verify_artifacts.py` verifier-loop pattern (incl. R4 5-field `manifest_pairing_index` widening); existing strict-schema discipline (`additionalProperties: false`, `$defs/`, `enum`); the existing 45/16/16 split at `splits.toml`; the bootstrap CI machinery at `tools/hrd_profiler/profiler.py`; the SASS classifier at `tools/perturbation_harness/sass_confirm.py:45-53`; the local-label NCU exception at `tools/local_labels/` + `data/local_labels/`; pure-Python + NumPy (no sklearn / torch / TensorFlow imports — match repo style); native gcc-14 / nvcc-13.2 toolchain.
- Cannot use: NCU on cloud / server (Rule 1); GPU-driver-root operations (Rule 2); privileged CUPTI surfaces (Rule 3); host-driver work (Rule 4); simulator revival (Rule 5); modification of `instruction.proto` wire format in this loop (per-record cache-modifier metadata extension deferred); modification of HANDOFF.md / top-level README.md / per-tree READMEs / BLOCKED_ON_3090TI.md (carry-forward anti-goal); modification of the 34-bucket reuse-distance partition (deferred); widening of tracer arch token sets without lockstep; loosening of `stddev_threshold` from 0.03 (deferred); RRIP / DRRIP / ATD implementation (deferred behind AC-12 ablation gate); set-associative `associativity_policy` enum widening (deferred).

> **Note on Deterministic Designs**: The predictor's algorithm (Smith-Eklund LRU stack distance) is fixed per the draft; the schema discipline (`additionalProperties: false`, 5-field `manifest_pairing_index`, R4 verifier pattern) is fixed per existing repo discipline. The path boundaries' flexibility lives between the upper and lower bounds (which milestones land in this loop vs follow-up loops).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Module surfaces that the loop is expected to add or extend:

- **`workloads/simbench_bw/cache_predictor.py`** (new): the main predictor module. Public API: `predict_cache_hit_rate(sector_stream_path: Path, cache_model_spec: dict, manifest_pairing_index: dict, *, bootstrap_seed: int) → CachePredictorBundle`. Internally calls into a Fenwick-tree LRU stack-distance tracker (reusing `tools/hrd_profiler/reuse_distance_tracker.py`) to recompute per-launch reuse-distance histograms from the sector_stream sidecar; applies the modeled-hit rule (`rd < capacity_bytes // 32`); aggregates per-bin hit rates weighted by observation count; emits bootstrap CI per the existing `BOOTSTRAP_RESAMPLES=2000` discipline; surfaces typed-failure / typed-unavailable reasons via the `phase_cycle_ledger_consumer.py` admission pattern.
- **`workloads/simbench_bw/schemas/cache_predictor.schema.json`** (new): strict JSON Schema with `additionalProperties: false`, 5-field `manifest_pairing_index` enum, three result-arm `oneOf` (scalar / bounded / typed_unavailable), strict typed_reason enum (e.g., `hrd_layer_unavailable`, `cache_model_underconfigured`, `circular_predictor_input_rejected`, `comparison_contract_drift`, `headline_geomean_admitted_cell_count_below_threshold`, `workload_identity_invariance_violated`, `cross_arch_label_schema_drift`, `prior_art_comparison_unmatched_workload`, `bundle_underconfigured`).
- **`workloads/simbench_bw/comparison_contract.toml`** (new): the predictor↔NCU comparison contract per AC-4. Pins NCU metric, 7-field join key, error formula, epsilon, geomean aggregation, admitted-cell coverage gate.
- **`workloads/simbench_bw/comparison_runner.py`** (new): consumes admitted (predictor × NCU label) cells, applies the comparison contract, emits the geomean bundle.
- **`workloads/simbench_bw/verify_artifacts.py`** (extended): adds `_verify_cache_predictor_run_summaries` mirroring `_verify_divergence_proxies_run_summaries`'s R4 5-field `manifest_pairing_index` cross-check.
- **`tools/local_labels/label_driver.sh`** (extended): per-cell `replay_mode` flag (default `kernel`; per-slug override to `application` for memory-bound slugs), per-cell `warmup_count` flag (forwarded to the workload binary), per-cell optional inter-launch L2 flush kernel hook.
- **`tools/local_labels/repeatability_aggregator.py`** (extended): per-cell `repeats_required` override (from a TOML registry, defaults preserved); does NOT loosen `stddev_threshold`.
- **`workloads/simbench_bw/shapes/<slug>.toml`** (extended for ≥3 memory-access slugs): add `[[trace_shapes]]` entries for `trace_50`, `trace_75` mirroring `carveout_sensitivity_probe.toml`'s working three-shape exemplar.
- **`tools/cache_predictor/l1_bypass_classifier.py`** (new, AC-11; opt-in): per-record L1-bypass routing.
- **`workloads/simbench_bw/tests/test_cache_predictor_*.py`** (new): unit + integration tests per AC-13.

### Relevant References

- `tools/hrd_profiler/reuse_distance_tracker.py` — Fenwick-tree O(log N) LRU stack-distance tracker; lift verbatim or via a thin wrapper.
- `tools/hrd_profiler/observation_builder.py` lines 34-46 — modeled-hit rule (`rd < capacity_bytes // 32`); the canonical Smith-Eklund operationalization.
- `tools/hrd_profiler/profiler.py` lines 40-66 + 134-150 — bootstrap CI constants + `_percentile_bootstrap_ci_launch_unit`; the resampling unit the predictor mirrors byte-for-byte.
- `workloads/simbench_bw/phase_cycle_ledger_consumer.py` lines 44-109 — typed-exception / typed-unavailable admission pattern (`TOKEN`, `to_json`).
- `workloads/simbench_bw/verify_artifacts.py` lines 1567-1635 — the R4-widened divergence_proxies `manifest_pairing_index` 5-field cross-check (post-commit `d2ec483`); the new predictor verifier mirrors this pattern exactly.
- `workloads/simbench_bw/verify_artifacts.py` lines 2755-2810 — the trace_convergence_confidence verifier's `manifest_run_summary_drift` block; the canonical full-5-field cross-check pattern.
- `workloads/simbench_bw/schemas/divergence_proxies.schema.json` — the 5-field `manifest_pairing_index` schema constraint; the predictor schema mirrors this.
- `tools/perturbation_harness/sass_confirm.py` lines 45-53 — cross-arch SASS classifier (Ampere `LDG.E.CA/CG/CS` + Blackwell `LDG.E.STRONG.SM/STRONG.GPU/EF` per BL-20260505).
- `tools/local_labels/{label_driver.sh, parse_ncu_metrics.py, repeatability_aggregator.py, README.md, TASK7_RUNBOOK.md}` — the local-label collection driver + admission helper.
- `data/local_labels/{label_admitted, label_excluded, label_repeatability}.json` — the current label state (empty admitted; 2 excluded).
- `workloads/simbench_bw/{run_suite.py, roster.toml, splits.toml}` — orchestration + 45/16/16 split.
- `workloads/simbench_bw/shapes/carveout_sensitivity_probe.toml` — the working three-shape exemplar (`trace_00`, `trace_50`, `trace_75`).
- `workloads/simbench_bw/results/dram_stream_stride_mix/blackwell_consumer_unknown/sm_120/trace_trace_00/` — the live admitted sm_120 trace bundle (HEAD `d2ec483`, 30922/30922 records, `bundle_underconfigured=true`).
- `references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf` — primary prior art for cache-hit / miss prediction from reuse behavior.
- `references/papers/knowledge_base/active/{2009__gpgpu_sim, 2020__accel_sim, 2021__nvarchsim, 2022__gcom}.pdf` — additional prior-art papers for AC-10 contextual comparison.
- `.claude/knowledge/measurement/` — distilled cards on non-root measurement framing.
- `.claude/skills/validating-nonroot-gpu-measurement/SKILL.md` — skill card for the non-root measurement specialty.

## Dependencies and Sequence

### Milestones

1. **Milestone 1 — Comparison contract + predictor scaffold** (closes AC-1, AC-2, AC-3, AC-4, AC-13 partial, AC-14 partial). Lands the schema, the predictor module, the comparison contract, and the verifier hook. Reportable: predictor self-consistency (CI half-width per kernel) + admitted-cell coverage diagnostic. The current `dram_stream_stride_mix/trace_00` capture on sm_120 emits typed-unavailable cleanly; no headline geomean yet.
   - Phase A: schema + predictor module skeleton with typed-unavailable arm.
   - Phase B: Fenwick-tree-based predictor (lifting `tools/hrd_profiler/reuse_distance_tracker.py`).
   - Phase C: comparison contract pinned at `comparison_contract.toml`.
   - Phase D: verifier hook in `verify_artifacts.py` mirroring R4 divergence_proxies.

2. **Milestone 2 — Label stabilization** (closes AC-5). Per-cell `replay_mode` + `warmup_count` + `repeats_required` overrides land in `tools/local_labels/`; the formerly-excluded cells re-run; `data/local_labels/label_admitted.json` gains its first cells (or the typed-failure record explicitly documents residual variance). Stddev threshold not loosened.
   - Phase A: per-cell knob registry (TOML).
   - Phase B: experiment matrix execution against the two failed cells.
   - Phase C: admit results or typed-failure documentation.

3. **Milestone 3 — HRD coverage scaling** (closes AC-6). Add `trace_50` / `trace_75` shape definitions for ≥3 memory-access slugs; admit the new HRD bundles with `bundle_underconfigured=false`; pass the workload-identity invariance gate.
   - Phase A: shape definitions (mirroring `carveout_sensitivity_probe.toml`'s exemplar).
   - Phase B: live sm_120 captures; verify_artifacts admission.
   - Phase C: workload-identity invariance gate.

4. **Milestone 4 — First reportable geomean** (closes AC-7, AC-8). Bounded-interval arm + reproducible geomean computation over the M2-admitted labels × M3-admitted HRD profiles. The first headline predictor-vs-NCU geomean relative error number lands.
   - Phase A: bounded-interval arm in the predictor.
   - Phase B: comparison runner + geomean computation per AC-8.
   - Phase C: round summary records the headline number with bootstrap CI.

5. **Milestone 5 — Cross-arch ablation** (closes AC-9). Per-arch + train-on-sm_86-test-on-sm_120 + train-on-sm_120-test-on-sm_86 MAPE; covariate / invariant feature split mirrors AC-8 (R0/R3) decomposition.
   - Phase A: feature extraction pipeline.
   - Phase B: calibration head training (linear regression in NumPy).
   - Phase C: per-arch + cross-arch evaluation table.

6. **Milestone 6 — Prior-art comparison + L1-bypass scaffold + ablation gate** (closes AC-10, AC-11, AC-12). Citation-safe contextual table; opt-in per-record L1-bypass scaffold; documented ablation ladder (no RRIP/DRRIP/ATD code).
   - Phase A: prior-art comparison table.
   - Phase B: `tools/cache_predictor/l1_bypass_classifier.py` scaffold.
   - Phase C: ablation-ladder documentation (set-associative refinement deferred).

### Dependency Graph

- M1 has no in-loop dependencies (predictor scaffold + comparison contract are independently self-consistent against synthetic fixtures).
- M2 depends on the existing `tools/local_labels/` driver and its admission registry (no in-loop dependency).
- M3 depends on the existing `run_suite.py --mode trace --shape-point <NAME>` orchestration (no in-loop dependency).
- M4 depends on M1 (predictor exists) + M2 (labels admitted) + M3 (HRD profiles admitted).
- M5 depends on M2 (labels on both arches) + M4 (predictor + comparison runner).
- M6's prior-art table depends on M4 (the contextual comparison cites the headline number); M6's L1-bypass scaffold is independent.

## Task Breakdown

Each task carries exactly one routing tag.

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Author `workloads/simbench_bw/schemas/cache_predictor.schema.json` with strict `additionalProperties: false`, 5-field `manifest_pairing_index`, three result-arm `oneOf` (scalar / bounded / typed_unavailable), and the typed_reason enum | AC-1 | coding | - |
| task2 | Author `workloads/simbench_bw/cache_predictor.py` skeleton with the predictor public API, typed-unavailable arm, and admission-helper pattern (mirrors `phase_cycle_ledger_consumer.py:44-109`) | AC-1, AC-3 | coding | task1 |
| task3 | Wire the predictor's input ingestion to read `sector_stream/*.jsonl` records, recompute per-launch reuse-distance histograms via `tools/hrd_profiler/reuse_distance_tracker.py`, and apply the Smith-Eklund modeled-hit rule from `observation_builder.py:34-46`. Reject `hrd_profile.json` as input (typed `circular_predictor_input_rejected`) | AC-2 | coding | task2 |
| task4 | Author `workloads/simbench_bw/comparison_contract.toml` pinning NCU metric, 7-field join key, error formula, epsilon policy, geomean aggregation, admitted-cell coverage gate. Document the rationale for each pinned value | AC-4 | coding | - |
| task5 | Author `workloads/simbench_bw/comparison_runner.py` consuming admitted (predictor × NCU label) cells per the contract; emit the geomean bundle with bootstrap CI | AC-4, AC-8 | coding | task4, task1 |
| task6 | Extend `workloads/simbench_bw/verify_artifacts.py` with `_verify_cache_predictor_run_summaries` mirroring R4 divergence_proxies pattern (full 5-field manifest_pairing_index cross-check) | AC-1, AC-13 | coding | task1 |
| task7 | Extend `tools/local_labels/label_driver.sh` with per-cell `replay_mode` and `warmup_count` overrides; extend `tools/local_labels/repeatability_aggregator.py` with per-cell `repeats_required`; do NOT loosen `stddev_threshold` | AC-5 | coding | - |
| task8 | Run the label-stabilization experiment matrix against the two formerly-excluded cells (sm_120 fp32_fma_dep_chain trace_00 + sm_86 gemm_hmma_bf16_sm80 correctness_shape) | AC-5 | coding | task7 |
| task9 | Add `trace_50` and / or `trace_75` shape definitions for ≥3 memory-access slugs in `workloads/simbench_bw/shapes/<slug>.toml` (mirror `carveout_sensitivity_probe.toml`'s exemplar) | AC-6 | coding | - |
| task10 | Run live sm_120 captures of the new shapes; admit via `verify_artifacts.py`; pass the workload-identity invariance gate | AC-6 | coding | task9 |
| task11 | Wire the bounded-interval arm in the predictor for underconfigured cells (lower / upper / midpoint variants) | AC-7 | coding | task3 |
| task12 | Author the geomean-reproducibility test fixture: pin admitted-cell list, bootstrap seed, epsilon, error formula; verify byte-identical re-run | AC-8, AC-13 | coding | task5 |
| task13 | Run cross-arch transfer evaluation: per-arch MAPE + train-on-sm_86-test-on-sm_120 MAPE + train-on-sm_120-test-on-sm_86 MAPE | AC-9 | coding | task8, task10, task5 |
| task14 | Cross-arch invariants vs covariates analysis (analyze tag): map which features are arch-invariant vs arch-specific; produce the feature-split documentation that the calibration head consumes | AC-9 | analyze | task13 |
| task15 | Author `tools/cache_predictor/l1_bypass_classifier.py` scaffold (opt-in, default off); consume the existing per-arch SASS regex from `tools/perturbation_harness/sass_confirm.py:45-53` | AC-11 | coding | task3 |
| task16 | Author the ablation-ladder documentation (fully-assoc-LRU baseline → per-record L1 bypass → bounded-associativity → ATD); pin the gate criterion (residual error not closing to <5%) | AC-12 | coding | task13 |
| task17 | Author the prior-art contextual comparison table (NVArchSim, Accel-Sim, GPGPU-Sim, Smith-Eklund 2014); explicitly mark unmatched workloads as `contextual_only` | AC-10 | coding | task13 |
| task18 | Prior-art comparison verification (analyze tag): /humanize:ask-codex with the comparison table + cited paper PDFs from `references/papers/knowledge_base/active/`; verify scope-bounding | AC-10 | analyze | task17 |
| task19 | Author the unit + integration test surface at `workloads/simbench_bw/tests/test_cache_predictor_*.py` covering AC-1 through AC-12 | AC-13 | coding | task3, task5, task6, task11, task15 |
| task20 | Round summary recording device + driver + CUDA + NVBit hash + gcc + observed clocks; cuobjdump --list-elf outputs; freshness SHA chain; admitted-cell tables; geomean output (when M4 lands); cross-arch table (when M5 lands); prior-art table (when M6 lands). Two-Device Local Development Model provenance per CLAUDE.md item 5. The summary explicitly distinguishes self-consistency reporting from the headline predictor-vs-NCU MAPE | AC-14 | coding | task5, task8, task10, task13, task17 |

## Claude-Codex Deliberation

### Agreements

- The Smith-Eklund LRU stack-distance algorithm is the right baseline; the Fenwick-tree implementation already exists in repo.
- The strict-schema + 5-field `manifest_pairing_index` discipline (R4 commit `d2ec483`) is the right scaffold for the new predictor's verifier hook.
- The `phase_cycle_ledger_consumer.py:44-109` typed-exception / typed-unavailable admission pattern is the right scaffold for the predictor's failure modes.
- The 45/16/16 calibration / validation / holdout split at `splits.toml` is the right scaffold for the LOO + bootstrap CI evaluation harness.
- Prior-art comparison must be citation-safe (contextual_only for unmatched workloads), not head-to-head (Codex first-pass core-risk #8).
- Anti-goals: no NCU on cloud / server, no driver root, no privileged CUPTI, no host-driver work, no simulator revival.

### Resolved Disagreements

- **Circularity of the predictor's input** (Codex first-pass core-risk #2): Codex flagged that wrapping `hrd_profile.json`'s already-modeled `hit_count` / `hit_rate` would make the predictor an aggregator, not an independent predictor. Resolution: AC-2 explicitly binds the predictor's input to RAW reuse-distance evidence (per-record `sector_stream/*.jsonl` + Fenwick-tree-recomputed histograms); rejects `hrd_profile.json` as input with typed `circular_predictor_input_rejected`.
- **First reportable number framing** (Codex first-pass core-risk #1): Codex flagged that with empty admitted labels and unavailable HRD coverage, the first deliverable is a predictor artifact + typed-unavailable records, not a predictor-vs-NCU geomean. Resolution: M1's reportable output is "predictor self-consistency" (CI half-width per kernel + admitted-cell coverage diagnostic); M4 is the first headline predictor-vs-NCU geomean number, gated on M2 + M3 closing.
- **NCU metric semantics** (Codex first-pass core-risk #3): Codex flagged that NCU L2 sector hit rate has multiple definitions (with / without atomics, sector vs request, with / without TLB). Resolution: AC-4's comparison contract pins the exact NCU metric (e.g., `lts__t_sectors_lookup_hit_rate.pct` for L2 sector-basis hit rate fraction); the contract documents numerator / denominator / scope.
- **Geomean stability** (Codex first-pass core-risk #4): Codex flagged that hit rates near 0 / 1 destabilize relative error. Resolution: AC-4 pins an `epsilon=0.01` policy; the error formula uses `max(ground_truth, epsilon)` as the denominator. Geomean is computed in log-space.
- **Larger trace shapes change behavior** (Codex first-pass core-risk #5): Codex flagged that scaling axes can shift the cache regime entirely. Resolution: AC-6 adds a workload-identity invariance gate (`kernel_signature_sha256` matches, structural ratios within tolerance, `predicted_bytes` within `trace_budget_mib`).
- **Cross-arch transfer is premature** (Codex first-pass core-risk #7): Codex flagged that with empty `label_admitted.json`, transfer learning has no training data. Resolution: AC-9 is gated on M2 + M5 closing (admitted labels on both arches); the lower-bound path defers M5 to a follow-up loop.
- **Set-associative refinement risks speculative modeling** (Codex first-pass core-risk #8): Codex flagged that NVIDIA L2 set indexing / slice hashing / replacement policy are not publicly documented and RRIP/DRRIP could become an unvalidated simulator-like detour. Resolution: AC-12 defers RRIP / DRRIP / ATD code behind an explicit ablation gate (residual error not closing to <5%); the schema's `associativity_policy` enum stays restricted.
- **Per-record L1 bypass requires producer-side wire change** (Codex first-pass technical-gap #6): Codex flagged that `instruction.proto` does not currently carry per-record cache-modifier metadata. Resolution: AC-11's scaffold lands as opt-in (default off), reading from the post-hoc SASS classifier rather than from the proto wire format; the proto wire-format extension is deferred.
- **Quantized reuse-distance bins miss capacity-threshold-inside-a-bin cases** (Codex first-pass technical-gap #3): Codex flagged that when a cache-capacity threshold falls inside a bin, the predictor needs raw distances or finer histograms. Resolution: AC-2 binds the predictor's input to RAW reuse-distance evidence (recomputed from `sector_stream`), so the per-launch histogram is the predictor's own (finer-grained) — NOT the HRD profiler's 34-bucket pre-bucketed output.
- **Underconfigured-bin handling** (Codex first-pass alternative-direction #3): Codex suggested bounded prediction intervals rather than dropping or imputing typed-unavailable bins. Resolution: AC-7 implements the bounded-interval arm (`hit_rate_lower` = treat unavailable bins as misses; `hit_rate_upper` = treat as hits).

### Convergence Status

- Final Status: `partially_converged after one Codex first-pass review` — pending second-pass review and resolution of any new disagreements; pending user decisions DEC-1 through DEC-8.
- All 8 Codex first-pass CORE_RISKS are addressed in the AC structure or resolved-disagreement entries above.
- All 10 Codex first-pass MISSING_REQUIREMENTS are folded into AC-1 / AC-4 / AC-5 / AC-6 / AC-8 / AC-9 wording.
- All 10 Codex first-pass TECHNICAL_GAPS are addressed via AC-2 (raw evidence), AC-7 (bounded intervals), AC-11 (per-record bypass scaffold opt-in), AC-12 (ablation gate), AC-9 (cross-arch label gate), AC-10 (citation-safe contextual comparison), AC-4 (NCU metric pinning).
- 6 of Codex's ALTERNATIVE_DIRECTIONS are folded into the milestone sequencing or the AC framing; the remaining alternatives (e.g., "raw-MRC predictor first") are equivalent to the chosen primary because AC-2 already binds the predictor to raw evidence.
- All 8 Codex first-pass QUESTIONS_FOR_USER are carried as Pending User Decisions DEC-1 through DEC-8 below.
- All 10 Codex first-pass CANDIDATE_CRITERIA are subsumed by the 14 ACs in this plan.

## Pending User Decisions

- DEC-1: Exact NCU metric for the headline number.
  - Claude Position: `lts__t_sectors_lookup_hit_rate.pct` (L2 sector-basis hit rate fraction; sector basis matches the predictor's modeled-hit rule which operates in 32B-sector units; consistent with `cache_model_capacity_bytes // 32` units in `observation_builder.py:34-46`).
  - Codex Position: N/A — open question (Codex flagged the metric needs explicit pinning; did not propose a specific name).
  - Tradeoff Summary: L1tex hit rate (e.g., `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct`) is the alternative; covers a different cache layer. The plan should pick one; reporting both is permissible but the headline geomean targets one.
  - Decision Status: `PENDING`

- DEC-2: Headline target quantity (hit rate vs miss rate vs sector counts).
  - Claude Position: relative error on hit rate (∈ [0, 1]) with `epsilon=0.01` to handle hit rates near 0; this matches the typical cache-modeling literature (Smith-Eklund 2014) reporting convention.
  - Codex Position: N/A — open question.
  - Tradeoff Summary: Miss rate has the same relative-error stability issues near 1; sector counts (predicted vs measured raw counts) avoids the rate-stability issue but couples to access-count covariance which becomes per-arch covariate. The plan recommends hit rate + epsilon; the choice should be documented in the comparison contract.
  - Decision Status: `PENDING`

- DEC-3: Minimum admitted-cell count required before headline geomean is "reportable".
  - Claude Position: 5 admitted cells per arch minimum (smoke-test floor); 10+ per arch for a defensible publication-grade result; mention 20+ across the 45-row calibration split as the aspirational target.
  - Codex Position: N/A — open question (Codex flagged that "a single admitted cell should be a smoke test, not a conference-grade result").
  - Tradeoff Summary: Setting the floor too low risks publishing on small-sample noise; too high blocks the loop's lower-bound deliverable. The 5/10/20 thresholds match the existing 45/16/16 split's leave-one-out semantics.
  - Decision Status: `PENDING`

- DEC-4: Predictor input source — raw reuse-distance evidence (recomputed from `sector_stream`) vs HRD profiler's modeled `hit_count` / `hit_rate`.
  - Claude Position: RAW reuse-distance evidence (recomputed via `tools/hrd_profiler/reuse_distance_tracker.py` Fenwick-tree from `sector_stream/*.jsonl`); reject `hrd_profile.json` as input with typed `circular_predictor_input_rejected`. AC-2 binds this.
  - Codex Position: same — Codex's first-pass critique #2 explicitly flagged the circularity risk.
  - Tradeoff Summary: Raw evidence is more code (predictor recomputes histograms) but produces an independent prediction; modeled output is faster but circular. Claude + Codex aligned.
  - Decision Status: `RESOLVED — RAW reuse-distance evidence (AC-2 binds)`

- DEC-5: Underconfigured HRD cells — strict refusal, bounded intervals, or imputation.
  - Claude Position: bounded intervals via AC-7 (`hit_rate_lower` = treat unavailable bins as misses; `hit_rate_upper` = treat as hits); the strict-refusal arm exists at AC-3 for cells where every bin is underconfigured (`bundle_underconfigured=true` AND zero admitted bins).
  - Codex Position: bounded intervals (Codex first-pass alternative-direction #3 explicitly suggested this).
  - Tradeoff Summary: Strict refusal is statistically cleaner but produces no number for many current cells; imputation introduces bias; bounded intervals are honest about uncertainty without dropping data.
  - Decision Status: `RESOLVED — bounded intervals via AC-7 + strict refusal via AC-3 for fully-underconfigured cells`

- DEC-6: NCU label target state — warmed steady-state vs application-replay vs single-launch trace state.
  - Claude Position: prefer application-replay state (matches `replay_mode=application` in NCU); the trace-evidence side captures single-launch trace state; the comparison contract (AC-4) explicitly documents the alignment / mismatch and uses `epsilon` to tolerate residual mismatch.
  - Codex Position: N/A — open question (Codex flagged the alignment must be made explicit, did not pick).
  - Tradeoff Summary: warmed steady-state matches typical NCU defaults but requires extra warmup loops; application-replay is independent per repeat (closer to the predictor's per-launch view); single-launch trace state is what the trace evidence already represents but most NCU defaults differ. The plan's bias is application-replay because it minimizes the trace-evidence-vs-label-evidence mismatch; the contract documents the choice.
  - Decision Status: `PENDING`

- DEC-7: Trace shape scaling — preserve memory footprint vs allow grid expansion.
  - Claude Position: preserve memory footprint (AC-6's invariance gate requires `kernel_signature_sha256` match + `predicted_bytes` within `trace_budget_mib` + structural ratios within tolerance); shape scaling primarily means more iterations / repeated observations, not larger working sets.
  - Codex Position: same — Codex first-pass core-risk #5 + alternative-direction #5 both pushed for invariance gates.
  - Tradeoff Summary: preserving footprint produces a "same workload, more samples" cell; allowing grid expansion produces a "different workload" cell that should be its own row. The plan's bias is preservation; expanded-footprint shapes can be added as additional rows in roster.toml in a follow-up loop.
  - Decision Status: `RESOLVED — preserve memory footprint per AC-6 invariance gate`

- DEC-8: Prior-art comparison scope — qualitative / contextual vs reproduce-and-compare.
  - Claude Position: qualitative / contextual comparison only in this loop (AC-10 marks unmatched workloads as `contextual_only`); reproduce-and-compare requires running NVArchSim / Accel-Sim / GPGPU-Sim against this repo's workload roster, which violates Rule 5 (simulator revival).
  - Codex Position: same — Codex first-pass alternative-direction #6 + core-risk #8 both flagged this as a scope risk.
  - Tradeoff Summary: contextual-only is honest about the comparison's limits; reproduce-and-compare gives a direct head-to-head but is out of scope per Rule 5. Claude + Codex aligned.
  - Decision Status: `RESOLVED — qualitative / contextual only per AC-10 + Rule 5`

## Implementation Notes

### Code Style Requirements

- Implementation code (`.cu`, `.py`, `.sh`) and comments must NOT contain plan-specific terminology such as "AC-N", "Milestone N", "Step N", "Phase N", "Round N", "task1..N", or similar workflow markers. These belong in plan documentation only.
- Test docstrings and assertion messages follow the same rule (per the existing `tools/test_plan_token_scanner.py` discipline).
- Use descriptive, domain-appropriate naming in code instead.

### Two-Device Local Development Model Reminders

- This loop produces evidence on both local arches. The cross-arch promotion (AC-9) requires admitted labels on both sm_86 and sm_120; Milestone 5 explicitly gates on this.
- The cross-host transport semantics (raw scp vs git fetch) per CLAUDE.md "Session Handoff" still apply: `data/local_labels/` survives both transports (committed); `.humanize/rlcr/<session>/` is host-local; built `.so` files do not survive transport cleanly and must be rebuilt on the receiving host.
- Per CLAUDE.md item 5, the round summary records on which device the live evidence was produced (device, SM, driver, CUDA toolkit, NVBit build, observed clocks / probes / throttle state).

### Critical Rules Reminders (carry-forward from CLAUDE.md)

- Rule 1: NCU only via `tools/local_labels/` + `data/local_labels/`. No NCU on cloud or server. No `.ncu-rep` artifacts outside the quarantine.
- Rule 2: no GPU-driver-root operations.
- Rule 3: no privileged CUPTI surfaces (CUPTI Activity is allowed).
- Rule 4: no host-driver work.
- Rule 5: no simulator revival (NVArchSim / Accel-Sim / GPGPU-Sim are CITATION-ONLY prior art per AC-10; no simulator binaries are set up or invoked).

### Loop Exit Discipline

The canonical exit for this loop is to let the stop-hook emit `COMPLETE` once Codex returns `ADVANCED` with the headline geomean number landed (upper bound) OR with the lower-bound deliverable (predictor module + comparison contract + bounded-interval arm + tests + anti-goals + at least one of M2 / M3 closing) admitted. `/humanize:cancel-rlcr-loop` is fallback only.

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

# Analytical LRU Predictor Anchored Cross-Arch NCU Validation

## Original Idea

Brainstorm a repo-grounded research direction that resolves the four blocking issues currently preventing the non-root GPU profiler from producing a reportable predictor-vs-NCU geomean relative error number on the local 5060 laptop (sm_120, Blackwell consumer, GB206). The downstream goal is <10% and ideally <5% geomean relative error suitable for top-tier academic conference submission (e.g. ISCA / MICRO / HPCA / SIGMETRICS / ASPLOS / SC).

Project context (binding):
- The project is "non-root GPU measurement and prediction for wall-clock behavior and cache hit ratio" per `CLAUDE.md`. There is currently NO predictor module in the repo (grep `predict*.py` / `model*.py` / `predictor*.py` returns nothing). The just-closed RLCR loop closed the measurement-side wire contract (NVBit producer + verify_artifacts.py admission on sm_120 for `dram_stream_stride_mix/trace_00`); the prediction-side comparison is the open research problem.
- `data/local_labels/label_admitted.json` is empty (`cells: []`); both attempted NCU label cells are in `label_excluded.json` with `typed_reason: label_repeatability_failed` — sm_120 fp32_fma_dep_chain trace_00 (5 repeats, L2 stddev=0.0605 vs threshold 0.03), sm_86 gemm_hmma_bf16_sm80 correctness_shape (5 repeats, L2 stddev=0.1414).
- The captured trace bundle reports its own underconfiguration: `trace_convergence_admitted=false`, `trace_convergence_typed_reason=hrd_layer_unavailable`, `cache_model_underconfigured_typed_reason=cache_model_underconfigured`, `bundle_underconfigured=true`, `underconfigured_bin_count=5`, `typed_unavailable_count=6`. `trace_access_count=24577` for the current trace_00 shape (grid=1×1×1, block=32×1×1; 32 threads).
- The cache geometry currently uses `associativity_policy=fully_associative_lru_approximation`, `sector_size_bytes=32`, `line_size_bytes=128`, `sectors_per_line=4`, `geometry_source=constant`, `l1_bypass_cases=[".cg","atomic"]`. L2 size is read from `cuda_device_prop` (32 MiB on sm_120, 6 MiB on sm_86). HRD profile params are `bootstrap_resamples=2000`, `ci_half_width_target=0.05`, `minimum_bin_sample_count=50`.

Hard constraints (binding per `CLAUDE.md`):
- Critical Rule 1: NCU is allowed ONLY through `tools/local_labels/` and `data/local_labels/` on the two local devices (RTX 3090 Ti sm_86 + RTX 5060 laptop sm_120). No NCU on cloud or server. No `.ncu-rep` artifacts outside the quarantine.
- Critical Rule 2: no GPU-driver-root operations (no nvidia-smi clock locks, no persistence, no MIG, no compute-mode toggles).
- Critical Rule 3: no privileged CUPTI surfaces (no Events / Metrics / Range Profiling / PM Sampling / PerfWorks). CUPTI Activity allowed.
- Critical Rule 4: no host-driver work (no driver install / DKMS / MOK / modprobe / reboot).
- Critical Rule 5: no simulator revival (no nvasim, Accel-Sim, GPGPU-Sim revival as active direction).
- Anti-goals from prior plans: no simulator revival; no widening of tracer arch token sets without lockstep; no edits to HANDOFF.md / top-level READMEs / per-tree READMEs / BLOCKED_ON_3090TI.md without dedicated tracked-doc-hygiene loops; no completion-gate overclaim ("the only", "sole", "completes" without scope-bounding).

The four blocking issues to brainstorm:

1. **Stabilize NCU labels.** Both repeatability failures point at L2 variance (L1 stddev=0 for both, L2 stddev 2-5x threshold). Hypothesized causes: cache-pollution from prior NCU replays (NCU's kernel-replay buffering re-runs the kernel many times, contaminating L2 across repeats); `warmup_count: 0` in the labels (admitting cold-cache variance); the difference between `replay_mode=kernel` (re-warms L1 between launches but not L2 across replays) and `replay_mode=application` (each repeat is an independent process). Concrete moves to consider: bump `warmup_count`, switch replay mode for memory-bound slugs, add an inter-launch L2 flush kernel, raise `repeats_required` from 5 to 10-20, deferred-tighten the `stddev_threshold` only after diagnosing variance source, isolate the GPU between repeats (no `nvidia-smi -lgc` since Rule 2 forbids it; what other knobs?).

2. **Build the baseline predictor.** Hooks already exist: per-shape `divergence_proxies/`, `hrd_profile/`, `sector_stream/`, `sharing_locality/`, `order_projections/` sidecars + `cache_model_capacity_bytes` + `l1_bypass_cases`. The Smith-Eklund LRU stack-distance algorithm against the HRD profile (with sm_120 32 MiB L2 capacity, sm_86 6 MiB L2 capacity) is the canonical first comparison and gives a reportable number. The brainstorm should identify: which sidecar is the natural input, what schema the predictor outputs (per-kernel hit-rate + bootstrap CI, mirroring the existing `bootstrap_resamples=2000` discipline), which existing helper (e.g., `phase_cycle_ledger_consumer.py`'s admission style) is the right pattern for the predictor's typed-failure / typed-unavailable reasons, and how the predictor surfaces to `verify_artifacts.py`.

3. **Fix the HRD layer.** `hrd_layer_unavailable` is gated by `bundle_underconfigured=true` + `underconfigured_bin_count=5`. Each HRD bin needs `minimum_bin_sample_count=50` samples. Current trace_00 has only 24577 dynamic accesses across 1 block × 32 threads — far too few for 5 quantile bins. The brainstorm should identify: which larger trace shapes (trace_50 / trace_75 / etc.) exist or could be added per workload to clear the bin-sample threshold, the storage-vs-trace-budget tradeoff for the higher access counts (the slug currently has `trace_budget_mib=64`), and which HRD bin partition (quantile vs power-of-two reuse-distance windows) gives the lowest variance per access count.

4. **Cross-arch ablation.** AC-8 split (must-match invariants vs expected-to-differ covariates) is the scaffold; the brainstorm should map: which invariant features (schema-level, allocation, exclusion taxonomy, per-trace structural counts) versus which covariate features (SASS mnemonic class, instruction count, compression distribution, NVML clock/power/temp, L2 capacity, sm_count, total_global_mem) belong on which side of a cross-arch transfer-learning split. Specifically: train on sm_86 v7 baseline labels at `data/local_labels/sm_86/`, test on sm_120 fresh labels at `data/local_labels/sm_120/`; OR train on a shared-arch invariant feature set with arch-specific calibration head; OR jointly train and report per-arch MAPE plus cross-arch transfer MAPE.

Stretch targets the brainstorm should explore briefly:

- Per-record L1 bypass detection: the SASS classifier already distinguishes Ampere `LDG.E.CA/CG/CS` and Blackwell `LDG.E.STRONG.SM/STRONG.GPU/EF` per BL-20260505. Routing each access to L1 vs L2 individually (not just at the binary level via `l1_bypass_cases`) is one of the highest-impact accuracy levers.
- Set-associativity refinement: replace `fully_associative_lru_approximation` with a 3-way RRIP / DRRIP approximation + ATD sampling on top of the existing sector_stream.
- MSHR / serialization model: `mshr_pressure_proxy` is currently a proxy field; queue-depth modeling against per-SM MSHR counts is the next-tier accuracy lever for memory-bound kernels.
- TLB / page-walk: NCU's L1tex hit-rate includes TLB-induced misses; subtracting the TLB component from NCU labels OR adding a small TLB model removes a systematic positive bias on large working sets.
- Calibration discipline: roster.toml has ~20 slugs; train/test split + leave-one-out + bootstrap CI on the geomean (with the existing `ci_half_width_target=0.05`) is what top-tier reviewers will demand.
- Comparison to prior art: NVArchSim, GPGPU-Sim, Accel-Sim, AccelWattch all publish per-workload accuracy tables; the submission needs a head-to-head.

Repo paths the brainstorm will want to reference:
- `data/local_labels/{label_admitted,label_excluded,label_repeatability}.json`
- `tools/local_labels/{label_driver.sh, parse_ncu_metrics.py, repeatability_aggregator.py, README.md, TASK7_RUNBOOK.md}`
- `workloads/simbench_bw/{run_suite.py, verify_artifacts.py, roster.toml}`
- `workloads/simbench_bw/shapes/<slug>.toml`
- `workloads/simbench_bw/results/<slug>/<sku>/sm_<N>/trace_<shape>/{divergence_proxies, hrd_profile, sector_stream, sharing_locality, order_projections, perturbation_run_ledger, convergence_confidence}/*.json`
- `workloads/simbench_bw/phase_cycle_ledger_consumer.py` (admission-helper pattern reference)
- `tools/perturbation_harness/sass_confirm.py` (per-arch SASS regex per BL-20260505)
- `references/papers/knowledge_base/` (prior-art measurement / prediction / simulator papers)
- `.claude/knowledge/measurement/` and `.claude/knowledge/blackwell/` distilled cards
- `.claude/skills/validating-nonroot-gpu-measurement/SKILL.md`

Output: a directed-swarm idea draft suitable for handoff to `/humanize:gen-plan`. The draft should articulate (a) the primary research direction with rationale + objective evidence + known risks, (b) 3-5 alternative directions with their tradeoffs and why-not-primary, (c) acceptance-criteria seed candidates, (d) path-boundary suggestions (upper / lower bound), and (e) the natural sequencing of the four blocking issues into milestones.

## Primary Direction: Analytical LRU Predictor First

(See structured plan above for the synthesized primary direction with rationale, approach summary, objective evidence, and known risks.)

## Alternative Directions Considered

(See structured plan above for Alt-1 through Alt-5 with tradeoff analysis.)

## Synthesis Notes

(See structured plan's milestone sequencing for how the alternatives fold in as Milestones 2-6.)

--- Original Design Draft End ---
