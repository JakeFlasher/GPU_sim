# M1 Lower-Bound Completion: Predictor↔NCU Comparison Runner + Predictor Verifier (Test-First Defect-Token Lockstep)

## Goal Description

Complete the Milestone 1 lower bound for the non-root GPU cache-hit-rate predictor by authoring two unimplemented code surfaces and their narrow test surface, under a strict test-first contract-lockstep discipline that mirrors `BL-20260509-strict-admission-contract-first`'s bidirectional `asserted == DEFECT_TOKENS` pattern. Specifically:

1. `workloads/simbench_bw/comparison_runner.py` — a library-first module (with a thin CLI wrapper) that loads `workloads/simbench_bw/comparison_contract.toml`, joins admitted predictor bundles (emitted by `cache_predictor.write_bundle`) against admitted NCU label cells (loaded from `data/local_labels/label_admitted.json`) on the contract-pinned 7-field key, normalizes NCU percent to fraction, computes the relative-error formula with `epsilon=0.01`, applies the result-arm dispatch policy (scalar / bounded / typed_unavailable), enforces the coverage gate, and emits a deterministic schema-valid output bundle.
2. `_verify_cache_predictor_run_summaries` in `workloads/simbench_bw/verify_artifacts.py` — a verifier hook structurally mirroring the R4 `_verify_divergence_proxies_run_summaries` exemplar (commit `d2ec483`), enforcing strict schema validation, 5-field `manifest_pairing_index` cross-check, on-disk sidecar digest verification, non-circular forbidden-keys rejection, bounded-arm `hit_rate_lower <= hit_rate_upper`, and a `bundle_underconfigured` cross-check against the sibling `hrd_profile.json`.

Both surfaces are gated by a defect-token frozenset (`_RUNNER_DEFECT_TOKENS`, `_VERIFIER_DEFECT_TOKENS`) and a bidirectional structural lockstep test (`asserted == DEFECT_TOKENS`, set equality, NOT subset-of). The negative-control test for every defect token is committed BEFORE the implementation that satisfies it lands; this ordering is verifiable from `git log` and is the procedural enforcer that prevents the just-stalled loop's "ships latent bugs" failure mode.

This plan is intentionally narrow: it advances AC-4 narrow (comparison contract IS consumed by a runner that emits deterministic JSON), AC-13 narrow (negative + positive control tests for every defect token), and AC-1 extension (verifier hook locks the contract permanently). It explicitly defers AC-5 (label stabilization, M2), AC-6 (HRD coverage scaling, M3), AC-7 (bounded-arm runtime production, M4), AC-8 (geomean reproducibility fixture against real labels, M4), AC-9 (cross-arch transfer, M5), AC-10 / AC-11 / AC-12 (prior-art comparison + L1 bypass + ablation gates, M6) to subsequent plans in the decomposed sequence.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. The test surface is committed BEFORE the implementation that satisfies it lands.

- AC-1: `_verify_cache_predictor_run_summaries` enforces the predictor bundle integrity contract.
  - Positive Tests (expected to PASS):
    - A schema-valid `cache_predictor_bundle.json` emitted by `predict_cache_hit_rate` whose 5-field `manifest_pairing_index` matches the enclosing run summary's fields and whose `input_sidecar_digests.sector_stream.sha256` matches the on-disk file's actual sha256 admits silently (returns no defects).
    - A schema-valid `typed_unavailable` bundle with `typed_reason=cache_model_underconfigured` and an empty `input_sidecar_digests` admits silently (the per-arm sector_stream relaxation in `cache_predictor.schema.json` allows this; verifier respects it).
    - A run summary whose `cache_predictor_bundle_path` field is absent (no predictor bundle was emitted for this trace) admits silently with no defects (this is the legitimate skip path — different from "path is set but missing on disk").
  - Negative Tests (expected to FAIL with a typed defect):
    - A bundle whose `manifest_pairing_index.slug` differs from the run summary's `slug` produces typed `cache_predictor_manifest_pairing_index_mismatch` with `slug` in the detail.
    - A bundle whose `input_sidecar_digests.sector_stream.sha256` does NOT match the on-disk file's actual sha256 produces typed `cache_predictor_sidecar_digest_drift` with `sector_stream` in the detail.
    - A bundle whose `input_sidecar_digests` carries any of `{hrd_profile, hit_count, hit_rate, model_hit, modeled_hit_rate, hrd_profile_path}` is hard-rejected with typed `cache_predictor_circular_predictor_input_rejected` (the schema's `additionalProperties: false` already rejects, but the verifier MUST surface the typed defect even if the schema validation step would have).
    - A bounded-arm bundle with `hit_rate_lower > hit_rate_upper` produces typed `cache_predictor_bounded_arm_inversion`.
    - A `result_kind=scalar` or `result_kind=bounded` bundle whose sibling `<trace_dir>/hrd_profile/hrd_profile.json` carries `bundle_underconfigured=true` produces typed `cache_predictor_bundle_underconfigured_with_admitted_arm` (locks the runtime admission gate at the verifier layer).
    - A bundle that fails strict schema validation (missing required field, additional unknown property, malformed typed_reason) produces typed `cache_predictor_schema_invalid`.
    - A run summary whose `cache_predictor_bundle_path` is set but the file does not exist on disk produces typed `cache_predictor_run_summary_missing_path` (distinguishes "intentional skip" from "broken provenance").

- AC-2: `comparison_runner.py` consumes the comparison contract and emits a deterministic schema-valid output bundle.
  - Positive Tests:
    - An explicit contract loader (`load_comparison_contract(path: Path | None = None) -> ComparisonContract`) returns a structurally-validated runtime contract object whose fields match the on-disk file's pinned constants exactly (`epsilon=0.01`, `bootstrap_resamples=2000`, `ci_confidence=0.95`, `bounded_arm_handling`, the 7-field join key list, the coverage-gate thresholds 5/10/20). The default path resolves to `workloads/simbench_bw/comparison_contract.toml`; a tests-supplied path or contract object overrides. Module import does NOT perform IO.
    - Two synthetic admitted predictor bundles (scalar arm, hit rates 0.40 and 0.60) joined against two synthetic admitted NCU label cells (ground-truth fractions 0.50 and 0.50) on matching 7-field keys produce a deterministic output bundle with predicted hit rates (0.40, 0.60), ground-truth fractions (0.50, 0.50), per-cell relative errors (0.20, 0.20), and a log-space geomean of relative errors that rounds to 0.20 within float tolerance.
    - The same input fixture, under the same `bootstrap_seed`, produces a byte-identical output bundle on re-run; the cell-level bootstrap CI fields (`geomean_ci_low`, `geomean_ci_high`, `geomean_ci_half_width`) are deterministic across runs (resampling unit = cell; seed = `bootstrap_seed` from the contract or override; the resampling discipline mirrors `cache_predictor._bootstrap_ci_over_launches` with the unit rebinding).
  - Negative Tests (7-field join near-miss tests, one for each field):
    - A predictor bundle whose `slug` differs by one character from any label cell's `slug` is recorded as typed `comparison_runner_join_key_unmatched` (no crash; geomean is computed over remaining matches; the bundle is recorded in `excluded_cells` with `reason=join_key_unmatched` and `unmatched_field=slug`).
    - Same near-miss for `shape_point` produces `comparison_runner_join_key_unmatched` with `unmatched_field=shape_point`.
    - Same near-miss for `binary_sha256` produces `comparison_runner_join_key_unmatched` with `unmatched_field=binary_sha256`.
    - Same near-miss for `source_commit` produces `comparison_runner_join_key_unmatched` with `unmatched_field=source_commit`.
    - Same near-miss for `kernel_signature_sha256` produces `comparison_runner_join_key_unmatched` with `unmatched_field=kernel_signature_sha256`.
    - Same near-miss for `arch` produces `comparison_runner_join_key_unmatched` with `unmatched_field=arch`.
    - Same near-miss for `ncu_metric_version` produces `comparison_contract_drift` with `unmatched_field=ncu_metric_version` (per DEC-1: the contract's pinned value is the binding floor; near-miss on `ncu_metric_version` is escalated to a contract-drift defect, NOT a join-unmatched exclusion, because the contract version is a protocol pin not a per-cell key).
  - Negative Tests (contract-drift floor; one per pinned constant):
    - The runner refuses to load a contract whose `epsilon` differs from the pinned `0.01` (typed `comparison_contract_drift` with `epsilon` in the detail) — pins the contract-version-discipline floor.
    - The runner refuses to load a contract whose `bootstrap_resamples` differs from the pinned `2000` (typed `comparison_contract_drift` with `bootstrap_resamples` in the detail).
    - The runner refuses to load a contract whose `ci_confidence` differs from the pinned `0.95` (typed `comparison_contract_drift` with `ci_confidence` in the detail).
    - The runner refuses to load a contract whose 7-field join-key list does not exactly match the pinned ordered list (typed `comparison_contract_drift` with `join_key` in the detail).
    - The runner refuses to load a contract whose coverage-gate thresholds (smoke-test floor / publication-grade floor / aspirational target) do not exactly match the pinned values (typed `comparison_contract_drift` with `coverage_gate` in the detail).

- AC-3: Result-arm dispatch policy is pinned and verifiable. (Two distinct denominators: a label cell that matches a predictor bundle on all 7 join-key fields is "comparable"; the comparable set is the DIAGNOSTIC coverage denominator — used for reporting how many cells were attempted, how many of those matched, and the per-gate failure breakdown. Within the comparable set, predictor bundles with `result_kind in {scalar, bounded}` are "headline" cells; bundles with `result_kind=typed_unavailable` are "comparable-but-not-headline" cells. The geomean is computed ONLY over headline cells. The admission floor — the coverage-gate threshold of 5 from the contract — is evaluated against the headline-cell count, NOT the comparable-cell count. The output bundle reports BOTH counts so a reader can audit gate decisions: `comparable_cells_per_arch` is the diagnostic count; `headline_cells_per_arch` is the admission-floor count.)
  - Positive Tests:
    - A `result_kind=typed_unavailable` predictor bundle that matches a label cell on all 7 fields is recorded in `excluded_cells` with `reason=predictor_typed_unavailable` and the upstream `typed_reason` preserved; the cell IS counted toward the comparable-set denominator (so coverage diagnostics distinguish "predictor refused" from "no label exists"); the cell is NOT counted toward the headline-cell numerator and is excluded from the geomean.
    - A `result_kind=bounded` predictor bundle (synthetic; predictor does not yet emit bounded today) produces a per-cell record with `relative_error` set to the midpoint-derived error (per resolved DEC-3: midpoint canonical), `relative_error_lower_diagnostic` set to the error from `hit_rate_lower`, and `relative_error_upper_diagnostic` set to the error from `hit_rate_upper`. The bundle's headline `geomean` is computed using `relative_error` only.
    - A `result_kind=scalar` predictor bundle produces the single relative error per the contract formula and contributes to the headline-cell numerator.
  - Negative Tests:
    - An unknown `result_kind` value (any string outside `{scalar, bounded, typed_unavailable}`) hard-rejects with typed `comparison_runner_unknown_result_kind`.
    - A scalar bundle whose `predicted_hit_rate` is outside `[0, 1]` hard-rejects with typed `comparison_runner_predicted_hit_rate_out_of_range` (defensive: predictor schema already enforces, but runner is the second admission boundary for non-bundle-shaped inputs).
    - A bundle whose `result_kind=typed_unavailable` but the upstream `typed_reason` is NOT in the predictor schema's enum (a syntactically-valid-but-operationally-unknown value) hard-rejects with typed `comparison_runner_typed_reason_out_of_enum`.

- AC-4: Coverage-gate semantics are pinned and verifiable.
  - Positive Tests:
    - With `headline_cells_per_arch ≥ 5` (smoke-test floor pinned in `comparison_contract.toml`), the output bundle's `result_kind` is the admitted arm and the geomean is reported.
    - With `headline_cells_per_arch < 5`, the output bundle's `result_kind=typed_unavailable` with `typed_reason=headline_geomean_admitted_cell_count_below_threshold`; the bundle records both `comparable_cells_per_arch` and `headline_cells_per_arch` (and the threshold) so a future reader can audit the gate decision.
  - Negative Tests:
    - A run with `headline_cells_per_arch == 4` (one below floor) does NOT emit a scalar geomean (the floor is enforced bidirectionally — the test surface's positive control admits at 5, the negative control rejects at 4).
    - A run with zero headline cells (every predictor bundle excluded by typed_unavailable or join-key-unmatched) emits `typed_unavailable` with the threshold reason.
    - A run where `comparable_cells_per_arch >= 5` BUT `headline_cells_per_arch < 5` (every comparable cell is predictor-typed-unavailable) emits `typed_unavailable` with the threshold reason — the comparable-set is large enough but the headline subset is not. The output bundle records both counts so the diagnostic distinction is auditable.

- AC-5: Bidirectional defect-token structural lockstep is exact (the prior-loop-failure-mode lock).
  - Positive Tests:
    - For the runner: `set(test_module.asserted_runner_defect_tokens) == comparison_runner._RUNNER_DEFECT_TOKENS` (set equality, not subset-of). Every entry in `_RUNNER_DEFECT_TOKENS` has at least one negative-control test that asserts it.
    - For the verifier: `set(test_module.asserted_verifier_defect_tokens) == verify_artifacts._VERIFIER_DEFECT_TOKENS` (set equality, not subset-of). Every entry in `_VERIFIER_DEFECT_TOKENS` has at least one negative-control test that asserts it.
    - `_RUNNER_DEFECT_TOKENS` and `_VERIFIER_DEFECT_TOKENS` are SEPARATE frozensets from `cache_predictor._TYPED_REASONS` (the predictor's typed-reason enum); a parity test asserts the three sets are disjoint OR explicitly intersect by a documented overlap list (no accidental overlap).
  - Negative Tests:
    - Adding a token to `_RUNNER_DEFECT_TOKENS` without adding a corresponding negative-control test produces a test failure naming the orphan token.
    - Asserting a token in a test that is not in `_RUNNER_DEFECT_TOKENS` produces a test failure naming the orphan assertion.
    - The plan-token scanner (`tools/test_plan_token_scanner.py`) returns `23 passed` after this round's commits.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. The narrow scope is binding: every AC above must close cleanly, and no widening into deferred ACs is permitted.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes:
- A frozen output JSON shape for the comparison runner (whether as a separate `comparison_bundle.schema.json` file under `workloads/simbench_bw/schemas/` or as a strict structural contract pinned in test fixtures — the choice is recorded in DEC-2 below).
- `comparison_runner.py` as a library-first Python module with a typed-exception API (`ComparisonContractDrift`, `ComparisonRunnerJoinKeyUnmatched`, `ComparisonRunnerResultKindUnknown`, etc.) and a thin `__main__` CLI wrapper that accepts `--contract`, `--predictor-bundles-glob`, `--label-admitted-path`, `--output`. The library is the test surface; the CLI is operator-facing.
- `_verify_cache_predictor_run_summaries` in `verify_artifacts.py`, structurally mirroring `_verify_divergence_proxies_run_summaries` (5-field manifest_pairing_index cross-check, schema validation, sidecar digest verification, bounded-arm invariant, bundle_underconfigured cross-check), with the cache-predictor-specific defect tokens.
- A new test file `workloads/simbench_bw/tests/test_comparison_runner.py` covering AC-2, AC-3, AC-4, and AC-5 (runner side).
- A new test file `workloads/simbench_bw/tests/test_verify_cache_predictor_run_summaries.py` covering AC-1 and AC-5 (verifier side).
- A shared test fixture module (under `workloads/simbench_bw/tests/conftest.py` extension or new `workloads/simbench_bw/tests/conftest_cache_predictor_comparison.py`) housing `_make_synthetic_admitted_predictor_bundle()`, `_make_synthetic_ncu_label_cell()`, `_make_synthetic_run_summary_with_cache_predictor_path()`, and `_RUNNER_DEFECT_TOKENS` / `_VERIFIER_DEFECT_TOKENS` re-export helpers for test discovery.
- A new `## Pending User Decisions` resolution section in the round summary (DEC-1, DEC-2, DEC-3 status), the bidirectional structural lockstep gate output (`pytest -q` with `pass_count == defect_token_count + positive_controls`), and the plan-token scanner gate output (must return `23 passed`).

### Lower Bound (Minimum Acceptable Scope)

The implementation includes:
- The defect-token frozensets (`_RUNNER_DEFECT_TOKENS`, `_VERIFIER_DEFECT_TOKENS`) and synthetic-fixture builder.
- The full negative-control test surface for every defect token (AC-5 lockstep mandatory; AC-1, AC-2, AC-3, AC-4 negative tests authored).
- A minimum-viable `comparison_runner.py` library module that satisfies the runner-side test surface — it provides an explicit contract loader (`load_comparison_contract(path: Path | None = None)`; default path = `workloads/simbench_bw/comparison_contract.toml`; module import does NOT perform IO), joins on the 7-field key, applies the relative-error formula with `epsilon=0.01`, dispatches the three result arms (per AC-3 coverage-denominator semantics; bounded arm uses midpoint canonical per resolved DEC-3), applies the coverage gate, computes the cell-level bootstrap CI on the geomean (resampling unit = cell; seed semantics deterministic per AC-2's byte-identical-re-run positive control), and emits a bundle that schema-validates against `workloads/simbench_bw/schemas/comparison_bundle.schema.json` (per resolved DEC-2: separate schema file). The CLI wrapper may be deferred; only the library is required.
- A minimum-viable `_verify_cache_predictor_run_summaries` that satisfies the verifier-side test surface (AC-1 positive + negative tests pass).
- The plan-token scanner (`tools/test_plan_token_scanner.py`) passes with the count expected for this round's commits (currently `23 passed`; if test-file additions widen the scanner's recognized count, the new count is recorded verbatim in the round summary's gate output).
- Round summary records every gate's executable command output verbatim (no paraphrased validation claims, per BL-20260509-rlcr-recovery-round-bounded-scope).

The lower bound is NOT a partial implementation that defers ACs to follow-up rounds — every AC above must close. The lower bound differs from the upper bound only in one optional element: the `__main__` CLI wrapper for `comparison_runner.py` (DEC-2 was resolved to "separate schema file" so the output schema authoring is in scope for both bounds).

Bootstrap CI on the geomean IS lower-bound mandatory (not deferred): the existing `cache_predictor._bootstrap_ci_over_launches` provides the proven pattern, the rebinding to cell-level resampling is a small surface change, and AC-2's byte-identical-re-run positive control depends on the deterministic seed semantics. The output bundle's `geomean_ci_low`, `geomean_ci_high`, `geomean_ci_half_width` fields are required by `comparison_bundle.schema.json`'s admitted arm (per resolved DEC-2).

### Allowed Choices

- Can use:
  - Standard library only for `comparison_runner.py` (json, math, statistics, argparse, pathlib, dataclasses, tomllib).
  - The existing `cache_predictor._bootstrap_ci_over_launches` pattern for cell-level bootstrap CI on the geomean (rebind the resampling unit from "launch" to "cell"; the bootstrap discipline is unchanged).
  - The existing `verify_artifacts._verify_divergence_proxies_run_summaries` structural pattern as the verifier hook template.
  - The existing `phase_cycle_ledger_consumer.py` typed-exception pattern (`TOKEN` class attribute + `to_json()` method) for runner exception surfaces.
  - `jsonschema` for strict schema validation in the verifier (already a dependency for the predictor test surface).
  - Synthetic on-disk files in `pytest`'s `tmp_path` for digest-verification tests (the fixture writes real bytes; sha256 verifies against them).
- Cannot use:
  - Any modeled-hit field from sidecars (`hit_count`, `hit_rate`, `model_hit`, `modeled_hit_rate`, `predicted_hit_rate` from `hrd_profile.json` are forbidden in both the runner's input and the verifier's cross-check; the predictor enforces this at its API boundary, the schema enforces it via `additionalProperties: false`, the verifier enforces it as a typed defect).
  - NCU invocation outside `tools/local_labels/` (Critical Rule 1).
  - Any GPU-driver-root operation (Critical Rule 2).
  - Any privileged CUPTI surface (Critical Rule 3).
  - Any host-driver work (Critical Rule 4).
  - Any simulator revival (Critical Rule 5).
  - Plan-token markers (`AC-N`, `Milestone`, `Phase`, `Step`, `Round N`) in implementation code or comments (the plan-token scanner will reject these).
  - Edits to top-level docs (`HANDOFF.md`, top-level `README.md`, per-tree `README.md`s, `BLOCKED_ON_3090TI.md`) — these are out of scope per the parent plan's anti-goals.

> **Note on Deterministic Designs**: This plan is largely deterministic — the comparison contract is already pinned, the schema patterns are already fixed, and the test-first discipline is the binding constraint. The narrow scope is the load-bearing decision; the upper / lower bounds differ only in the optional elements above. The three open design questions (DEC-1, DEC-2, DEC-3) are the only meaningful flexibility points.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The natural per-round task ladder, derived from the test-first discipline:

1. **Pre-test scaffolding** (single commit before any test or implementation): Author the defect-token frozensets in skeleton form (just the names, no implementation calls them yet). Author the synthetic-fixture builder. Author `workloads/simbench_bw/schemas/comparison_bundle.schema.json` (per resolved DEC-2) with the strict structural pattern (additionalProperties: false; manifest_pairing_index analog; oneOf arms admitted / typed_unavailable; admitted-arm required fields including the geomean + bootstrap CI fields and the per-cell records with relative_error + diagnostic-bound fields per resolved DEC-3). Commit.

2. **Test surface authoring** (single commit after task 1): Author every negative-control test (one per defect token) and every positive-control test (the happy paths for AC-1 / AC-2 / AC-3 / AC-4). The bidirectional structural lockstep test (AC-5) lands in this commit. The tests fail at this point because the implementation does not yet exist (or imports raise `AttributeError`). Commit. The fact that the tests are committed BEFORE the implementation is verifiable from `git log`.

3. **Runner implementation** (single commit after task 2): Implement `comparison_runner.py` to satisfy the runner-side test surface. The contract loader + 7-field join + relative-error computation + three-arm dispatch + coverage-gate enforcement + output-bundle emission. The runner-side tests pass after this commit. Commit.

4. **Verifier implementation** (single commit after task 3): Implement `_verify_cache_predictor_run_summaries` in `verify_artifacts.py` to satisfy the verifier-side test surface. The 5-field manifest_pairing_index cross-check + schema validation + sidecar digest verification + bounded-arm invariant + bundle_underconfigured cross-check. The verifier-side tests pass after this commit. Commit.

After task 4, the bidirectional structural lockstep test passes (the asserted-defect-token sets match the implementation's frozensets exactly). The plan-token scanner remains green.

### Relevant References

- `workloads/simbench_bw/cache_predictor.py` — the predictor module the runner consumes; carries `_TYPED_REASONS` frozenset (11 entries) the runner / verifier must NOT accidentally overlap; carries `_validate_input_keys` non-circular discipline and `_read_structural_underconfiguration` admission gate; carries `_bootstrap_ci_over_launches` for cell-level bootstrap reuse.
- `workloads/simbench_bw/schemas/cache_predictor.schema.json` — predictor schema; the verifier hook validates against this; the comparison-bundle schema (if DEC-2 chooses one) mirrors its discipline (additionalProperties: false; oneOf arm exclusivity; per-arm required clauses).
- `workloads/simbench_bw/comparison_contract.toml` — the contract; the runner loads this at module init; structural lockstep test asserts every pinned constant matches the runtime contract object.
- `workloads/simbench_bw/verify_artifacts.py:1468-1672` — `_verify_divergence_proxies_run_summaries` exemplar (R4 commit `d2ec483`); the new verifier hook is a structural carbon copy with cache-predictor-specific defect tokens.
- `workloads/simbench_bw/phase_cycle_ledger_consumer.py` — the typed-exception pattern (TOKEN attr + to_json()) the runner mirrors.
- `workloads/simbench_bw/tests/test_cache_predictor.py` — the existing 24-test surface; the new tests extend the same conftest discipline (synthetic bundles via `_make_mpi()` and `_make_input_digests()`); fixture builders expand to comparison-runner-side cells.
- `data/local_labels/{label_admitted,label_excluded}.json` — current state: empty admitted, two excluded with typed_reason `label_repeatability_failed`. Synthetic NCU label cells will be hand-authored in tests; real cells land in M2 (AC-5) follow-up plan.
- `workloads/simbench_bw/splits.toml` — 45 calibration / 16 validation / 16 holdout split assignments; if DEC-2 resolves to "three split bundles", the runner emits one per split (filtering predictor bundles by slug ∈ split.assignments).
- `tools/test_plan_token_scanner.py` — the scanner that must return `23 passed`. The new code must NOT contain plan-token markers.
- `.humanize/bitlesson.md` — `BL-20260508-post-commit-recapture-discipline` (the verifier's HEAD-derived field re-anchor discipline applies if any output bundle records HEAD); `BL-20260509-strict-admission-contract-first` (the bidirectional `asserted == DEFECT_TOKENS` lockstep is the procedural enforcer); `BL-20260509-rlcr-recovery-round-bounded-scope` (the per-round bounded scope discipline is the meta-constraint).

## Dependencies and Sequence

### Milestones

This plan is intentionally a single-milestone plan; it advances ONE milestone (M1 lower-bound completion) within a single RLCR round. No multi-milestone sequencing.

1. **M1-Lower-Bound Completion**: The narrow scope above closes within a single RLCR round.
   - Phase A: Pre-test scaffolding — defect-token frozensets, synthetic-fixture builder, output JSON shape (DEC-2 resolution). One commit.
   - Phase B: Test surface authoring — every negative-control + positive-control test, bidirectional structural lockstep. One commit. Tests fail at this point because no implementation exists.
   - Phase C: Runner implementation — `comparison_runner.py` library + (optional, upper-bound) thin CLI wrapper. One commit. Runner-side tests pass.
   - Phase D: Verifier implementation — `_verify_cache_predictor_run_summaries` in `verify_artifacts.py`. One commit. Verifier-side tests pass.

The four commits are ordered: scaffolding → tests → runner → verifier. The test-first ordering is verifiable from `git log` and is the procedural enforcer of the test-first discipline (a future reviewer cannot mistake "test-first" for "test-along").

Subsequent plans (out of scope for this plan) sequence as: Plan B (M2 — label stabilization, closes AC-5); Plan C (M3 — HRD coverage scaling, closes AC-6); Plan D (M4 — first reportable geomean against real labels, closes AC-7 + AC-8); Plan E (M5 — cross-arch transfer, closes AC-9); Plan F (M6 — prior-art comparison + L1 bypass scaffold + ablation gates, closes AC-10 + AC-11 + AC-12).

### Dependency Graph

- task1 (scaffolding) has no in-loop dependencies.
- task2 (test surface) depends on task1's frozensets and fixture builder.
- task3 (runner) depends on task2's tests existing (so it can be validated against them).
- task4 (verifier) depends on task2's tests existing AND task3's runner output shape (the verifier's bundle_underconfigured cross-check needs the same on-disk path conventions the runner emits).

There are no out-of-loop dependencies (the comparison contract already exists; the predictor module already exists; the schema already exists; the divergence_proxies verifier exemplar already exists).

## Task Breakdown

Each task carries exactly one routing tag.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Author the defect-token frozensets (`_RUNNER_DEFECT_TOKENS` in `comparison_runner.py`, `_VERIFIER_DEFECT_TOKENS` in `verify_artifacts.py`), the synthetic-fixture builder (`_make_synthetic_admitted_predictor_bundle`, `_make_synthetic_ncu_label_cell` (with `profiler_version` matching the contract's pinned `ncu_metric_version` per DEC-1 contract-side pinning), `_make_synthetic_run_summary_with_cache_predictor_path`), and the output schema `workloads/simbench_bw/schemas/comparison_bundle.schema.json` (per DEC-2: separate schema file; mirrors `cache_predictor.schema.json`'s discipline — additionalProperties: false; manifest_pairing_index analog; oneOf arms for `admitted` / `typed_unavailable`; admitted arm requires `geomean`, `geomean_ci_low`, `geomean_ci_high`, `geomean_ci_half_width`, `comparable_cells_per_arch`, `headline_cells_per_arch`, per-cell records with `predicted_hit_rate`, `ground_truth_fraction`, `relative_error` (canonical; midpoint for bounded arm per DEC-3), `relative_error_lower_diagnostic` and `relative_error_upper_diagnostic` (bounded arm only); typed_unavailable arm requires runner-level typed_reason from its own enum). Skeleton modules only — no business logic; just the names and frozensets and fixture functions and the schema file. Commit before any test or implementation lands. | AC-5 (token-set definitions); AC-1, AC-2, AC-3, AC-4 (fixture + schema support) | coding | - |
| task2 | Author the full test surface at `workloads/simbench_bw/tests/test_comparison_runner.py` and `workloads/simbench_bw/tests/test_verify_cache_predictor_run_summaries.py`. Every defect token in the frozensets has at least one negative-control test that constructs a malformed input and asserts the typed defect fires; every happy-path AC has a positive-control test. The bidirectional structural lockstep test (`set(asserted_defect_tokens) == _DEFECT_TOKENS`) lands here. Tests fail at commit time because the implementation does not yet exist. The test-first commit ordering is part of the deliverable. | AC-1, AC-2, AC-3, AC-4, AC-5 (all tests) | coding | task1 |
| task3 | Implement `workloads/simbench_bw/comparison_runner.py` library — explicit contract loader `load_comparison_contract(path: Path | None = None) -> ComparisonContract` (default path = on-disk `comparison_contract.toml`; module import does NOT perform IO; loader rejects contracts whose `epsilon`, `bootstrap_resamples`, `ci_confidence`, 7-field join-key list, or coverage-gate thresholds differ from pinned values with typed `comparison_contract_drift`), 7-field key join across admitted predictor bundles + admitted NCU label cells with per-field near-miss diagnostics, NCU percent-to-fraction normalization (inline at point-of-use; the contract's `lts__t_sectors_lookup_hit_rate.pct` metric is divided by 100 before applying the error formula), result-arm dispatch per AC-3 coverage-denominator semantics (scalar / bounded contribute to headline numerator; typed_unavailable contributes to comparable denominator only; bounded arm uses midpoint canonical per resolved DEC-3, with `relative_error_lower_diagnostic` and `relative_error_upper_diagnostic` populated for audit), error formula |predicted - ground_truth| / max(ground_truth, epsilon), log-space geomean over headline cells, cell-level bootstrap CI (resampling unit = cell; deterministic seed semantics; output fields `geomean_ci_low`, `geomean_ci_high`, `geomean_ci_half_width`), coverage-gate enforcement (`headline_cells_per_arch ≥ 5` floor), label-side `profiler_version` cross-check against contract's pinned `ncu_metric_version` (per resolved DEC-1: contract-side pinning), deterministic output-bundle emission validating against `workloads/simbench_bw/schemas/comparison_bundle.schema.json`. Library-first; thin CLI wrapper optional (upper bound). All runner-side tests from task2 pass. | AC-2, AC-3, AC-4, AC-5 (runner side) | coding | task2 |
| task4 | Implement `_verify_cache_predictor_run_summaries` in `workloads/simbench_bw/verify_artifacts.py` — strict schema validation against `cache_predictor.schema.json`, 5-field `manifest_pairing_index` cross-check binding to the enclosing run summary, on-disk sidecar SHA-256 verification (`sector_stream` mandatory on scalar / bounded; `dynamic_trace_pb` cross-checked when present), forbidden-keys rejection (any of `hrd_profile`, `hit_count`, `hit_rate`, `model_hit`, `modeled_hit_rate`, `hrd_profile_path` in `input_sidecar_digests` is hard-rejected), bounded-arm `hit_rate_lower <= hit_rate_upper`, bundle_underconfigured cross-check (scalar / bounded result_kind requires sibling `hrd_profile.json` to NOT report `bundle_underconfigured=true`). All verifier-side tests from task2 pass. The defect-token frozenset matches `_VERIFIER_DEFECT_TOKENS` exactly (bidirectional structural lockstep). | AC-1, AC-5 (verifier side) | coding | task2, task3 |

## Claude-Codex Deliberation

### Agreements

- The test-first contract-lockstep discipline (mirroring BL-20260509-strict-admission-contract-first) is the right framing for this round; both Claude and Codex agree it directly addresses the prior loop's "ships latent bugs / false validation claims" failure mode.
- The narrow scope (AC-4 narrow + AC-13 narrow + AC-1 extension only; AC-5 / AC-6 / AC-7 / AC-8 / AC-9 / AC-10 / AC-11 / AC-12 deferred) is necessary; the just-stalled loop's structural failure was scope-overpromise, and the recovery is per-round bounded scope.
- The verifier hook MUST mirror the R4 `_verify_divergence_proxies_run_summaries` structural pattern (5-field manifest_pairing_index cross-check, schema validation, sidecar digest verification, cached-vs-recomputed consistency).
- The runner is library-first (with an optional CLI wrapper at upper-bound) — Codex's TECHNICAL_GAPS #2 surfaced the question; Claude's draft already framed it as library-first; both agree.
- `_RUNNER_DEFECT_TOKENS` and `_VERIFIER_DEFECT_TOKENS` are SEPARATE frozensets from `cache_predictor._TYPED_REASONS` — token-set separation prevents brittle coupling per Codex's TECHNICAL_GAPS #4. A parity test asserts the three sets are disjoint or explicitly intersect by a documented overlap list.
- Synthetic fixtures must be NAMED as synthetic-only and must NOT imply real-trace progress (per Codex's CORE_RISK #5). Tests that consume synthetic fixtures must include `synthetic` in the test name or fixture name; round summary explicitly distinguishes synthetic-fixture coverage from real-trace coverage.
- Token production must be tied to externally-visible failure records (typed exceptions or output-bundle defect entries), not private exception names — per Codex's TECHNICAL_GAPS #1.
- Negative tests for the 7-field join include near-miss tests for ALL seven fields (per Codex's MISSING_REQUIREMENTS #4), not just the missing-field case.

### Resolved Disagreements

- **Token-complete-but-behavior-thin risk** (Codex CORE_RISK #1): Claude's initial draft framed test-first as the discipline; Codex correctly noted that token-completeness alone is not enough. Resolution: the test surface includes BOTH defect-token negative-controls AND positive-control tests for every happy-path AC. The bidirectional `asserted == DEFECT_TOKENS` lockstep is the floor (catches missing tokens); the positive-control tests are the ceiling (catch token-complete-but-behavior-thin). Both layers are mandated in AC-5 explicitly.
- **Output JSON shape: separate schema vs fixture-pinned** (Codex MISSING_REQUIREMENT #1): Claude's draft was agnostic; Codex argued the canonical shape needs to be frozen BEFORE tests. Resolution: this is recorded as DEC-2 (Pending User Decision) — the user picks separate schema file (more rigorous, more upfront work) or fixture-pinned (simpler, less upfront). Either choice locks the shape before the test surface lands. (Codex round-1 lean: separate schema file.)
- **`ncu_metric_version` location** (draft's open Q1 + Codex QUESTIONS_FOR_USER): three options surveyed; Claude's draft favored Alt-4 (contract-side pinning); Codex listed all three as user-decisions. Resolution: DEC-1 (Pending User Decision) records the three options; the test surface aligns with the chosen option (the AC-2 7-field near-miss test for `ncu_metric_version` exercises the chosen surface). (Codex round-1 lean: contract-side pinning with label-side `profiler_version`.)
- **Bounded-arm canonical** (draft's open Q3 + Codex QUESTIONS_FOR_USER): three options (midpoint canonical / three variants / explicit user choice). Resolution: DEC-3 (Pending User Decision) records the choice; Claude's tentative position is "midpoint canonical + three diagnostic variants" per Alt-5 forward-compat reasoning, but the user resolves. (Codex round-1 lean: midpoint canonical + lower/upper diagnostic fields only.)
- **Runner CLI vs library-first** (Codex TECHNICAL_GAPS #2): both agree library-first; the CLI wrapper is upper-bound only. Lower bound omits the CLI; the library is fully testable without it.
- **Coverage-gate scope creep** (Codex CORE_RISK #2): the contract pins 5/10/20 thresholds; the runner enforces ONLY the smoke-test floor (5) at this round's scope. Tiered admission (5-9 vs 10-19 vs 20+) is deferred to M4 (Plan D) per the contract's downstream use. AC-4's positive control admits at 5; the negative control rejects at 4.
- **Verifier discovery contract** (Codex MISSING_REQUIREMENT #3): the run summary records `cache_predictor_bundle_path` as a known field; the verifier loads bundles whose path is referenced in the summary. Missing-path-with-set-field is typed `cache_predictor_run_summary_missing_path` (defect, AC-1 negative test). Field-absent is the legitimate skip path (AC-1 positive test). The two cases are distinguished explicitly.
- **Coverage denominator semantics** (Codex round-1 DISAGREE on AC-3): Claude's v1 said typed_unavailable bundles were excluded from BOTH numerator and coverage-gate denominator. Codex correctly pointed out the brainstorm draft's intent was "excluded from geomean but counted toward coverage gate denominator." Resolution: AC-3 now distinguishes `comparable cells` (matched on 7-field; the coverage-gate denominator) from `headline cells` (comparable AND `result_kind in {scalar, bounded}`; the geomean numerator). The coverage-gate floor of 5 applies to `headline_cells_per_arch`. Typed_unavailable cells contribute to the comparable denominator (so coverage diagnostics distinguish "predictor refused" from "no label exists") but NOT to the headline numerator.
- **Module-init contract loading** (Codex round-1 DISAGREE on AC-2): Claude's v1 said the runner "loads the contract at module init." Codex correctly flagged this as brittle (import-time IO; hard to test alternative paths). Resolution: AC-2 now requires an explicit `load_comparison_contract(path: Path | None = None)` library function with a default-path resolution; module import does NOT perform IO; tests pass an explicit path or contract object.
- **Bootstrap CI scope** (Codex round-1 DISAGREE on Lower Bound): Claude's v1 implied bootstrap CI was in scope but didn't pin its semantics. Codex asked for either "required with deterministic seed semantics" or "explicitly defer." Resolution: bootstrap CI on the geomean IS lower-bound mandatory; the existing `cache_predictor._bootstrap_ci_over_launches` is the proven pattern; the resampling unit rebinding to cell-level is a small surface change; AC-2's byte-identical-re-run positive control depends on deterministic seed semantics. The output-bundle fields `geomean_ci_low`, `geomean_ci_high`, `geomean_ci_half_width` are required.
- **Plan-token scanner count** (Codex round-1 OPTIONAL_IMPROVEMENT): Claude's v1 hard-coded "23 passed." Resolution: Lower Bound now says "passes with the count expected for this round's commits" and notes that if test-file additions widen the scanner's recognized count, the new count is recorded verbatim in the round summary's gate output.
- **`admitted_cells_per_arch` rename** (Codex round-1 OPTIONAL_IMPROVEMENT): renamed to `headline_cells_per_arch` throughout AC-4 to make the coverage-denominator distinction explicit.

### Convergence Status

- Final Status: `converged`. Three convergence rounds were executed: round 1 surfaced 4 REQUIRED_CHANGES (typed_unavailable / coverage-denominator reconciliation; 7 explicit near-miss tests; module-init contract loading replaced with explicit loader; bootstrap CI semantics pinned); round 2 surfaced 1 REQUIRED_CHANGE (AC-3 wording conflict between "comparable IS denominator" and "headline gates admission"); round 3 confirmed the AC-3 fix and the optional improvements (additional contract-drift tests for `ci_confidence` / join-key / coverage thresholds; rename to `headline_cells_per_arch`) and declared `CONVERGENCE_READY: yes`. DEC-1, DEC-2, DEC-3 were resolved by the user during the gen-plan loop (recommended options chosen for all three: contract-side pinning; separate schema file; midpoint canonical with diagnostic lower/upper). The plan is ready for `/humanize:start-rlcr-loop`.

## Pending User Decisions

All three Pending User Decisions were resolved during the convergence loop. They are recorded here for traceability; no items remain `PENDING`.

- DEC-1: Where does `ncu_metric_version` live for the predictor side of the join?
  - Claude Position: Contract-side pinning. The `comparison_contract.toml` carries the pinned `ncu_metric_version`; predictor bundles do NOT carry it (predictor doesn't know the NCU version it'll be compared against). The runner cross-checks label-side `profiler_version` (extracted from per-launch sidecars or from the admitted-cell record) against the contract's pinned value; mismatches produce typed `comparison_contract_drift` with `ncu_metric_version` in the detail.
  - Codex Position: Codex round-1 lean: contract-side pinning with label-side `profiler_version`.
  - Tradeoff Summary: Option A (predictor-bundle field) requires a predictor schema bump and breaks the predictor's "doesn't know its NCU comparator" property. Option B (contract-side pinning) is cleanest at the cost of eventually widening `tools/local_labels/repeatability_aggregator.py` to propagate `profiler_version` into `label_admitted.json` cells. Option C (per-comparison sidecar) is intermediate but adds a new artifact tree.
  - Decision Status: `RESOLVED: contract-side pinning`. Implementation note: this round's synthetic NCU label fixtures carry an explicit `profiler_version` field matching the contract's pinned `ncu_metric_version`. The live `repeatability_aggregator.py` widening (to propagate `profiler_version` into real `label_admitted.json` cells) is M2 scope (the AC-5 label-stabilization plan); deferred from this round per anti-goals. The runner-side cross-check logic exists in this round's runner code (using the synthetic-fixture `profiler_version`); when M2 lands real cells with the propagated field, no runner change is required.

- DEC-2: How is the comparison runner's output JSON shape locked?
  - Claude Position: Author a separate `workloads/simbench_bw/schemas/comparison_bundle.schema.json` mirroring the predictor schema's discipline (additionalProperties: false; oneOf arms for admitted / typed_unavailable; comparison_run_pairing_index analog; strict per-arm required fields). The runner emits to the schema; the verifier reads the schema; the schema is the canonical contract.
  - Codex Position: Codex round-1 lean: separate schema file.
  - Tradeoff Summary: Separate schema file = more rigorous, mirrors the existing schema family (43 files under `workloads/simbench_bw/schemas/`), enables future schema validators to reuse it; cost = upfront authoring + per-arm exclusivity oneOf clauses + integration with verifier's strict-mode validation. Fixture-pinned shape = simpler, less upfront work; cost = the canonical shape lives in test fixtures rather than as a tracked artifact, harder for future readers to audit.
  - Decision Status: `RESOLVED: separate schema file`. Implementation note: task1 includes authoring `workloads/simbench_bw/schemas/comparison_bundle.schema.json` (additionalProperties: false; manifest_pairing_index analog binding the comparison-run identity; oneOf arms for `admitted` / `typed_unavailable`; admitted arm carries `geomean`, `geomean_ci_low`, `geomean_ci_high`, `geomean_ci_half_width`, `comparable_cells_per_arch`, `headline_cells_per_arch`, per-cell records with predicted_hit_rate / ground_truth_fraction / relative_error / per-arm-dispatch fields; typed_unavailable arm carries typed_reason from the runner-level enum). The schema lands in task1's pre-test scaffolding commit, BEFORE the test surface, so tests in task2 schema-validate every emitted bundle.

- DEC-3: How does the runner handle bounded-arm predictor bundles for the headline number?
  - Claude Position: Pin midpoint-error-of-bounded-cell as the canonical headline error per forward-compat reasoning. Lower and upper errors are diagnostic-only (saved in the per-cell record). When AC-7's bounded-arm runtime production lands in M4, no contract change is required — M4 produces bounded bundles, the runner consumes them midpoint-canonically.
  - Codex Position: Codex round-1 lean: midpoint canonical with lower/upper diagnostic fields only.
  - Tradeoff Summary: Midpoint canonical = single headline; lower/upper diagnostic-only; downstream consumer is unambiguous. Three variant rows = three rows in the output bundle's per-cell records, each with its own error; downstream consumer chooses; risk = consumer-side fork. Three variant metrics = the bundle reports three geomean variants (lower-geomean / upper-geomean / midpoint-geomean); downstream consumer reports all three; risk = headline ambiguity in publication.
  - Decision Status: `RESOLVED: midpoint canonical + lower/upper diagnostic fields`. Implementation note: per-cell record fields are `relative_error` (the canonical headline error; `predicted_hit_rate` for scalar arm or midpoint of `[hit_rate_lower, hit_rate_upper]` for bounded arm), `relative_error_lower_diagnostic` (bounded arm only; the error computed from `hit_rate_lower`), and `relative_error_upper_diagnostic` (bounded arm only; the error computed from `hit_rate_upper`). The output bundle's headline `geomean` is computed over `relative_error` only. Synthetic bounded-arm tests assert all three fields are populated correctly when `result_kind=bounded`.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Phase`, `Step`, `Round N`, or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead. Examples:
  - Reference the defect tokens by their literal names (`comparison_contract_drift`, `comparison_runner_join_key_unmatched`), not by AC-N labels.
  - Reference the contract terms by their TOML field names (`epsilon`, `bootstrap_resamples`, `ci_confidence`, `bounded_arm_handling`), not by AC-N labels.
  - The `tools/test_plan_token_scanner.py` gate enforces this; it must return `23 passed` after every commit in this round.

### Test-First Commit Ordering Discipline

- Each task lands in a SEPARATE commit, in the order task1 → task2 → task3 → task4. The ordering is verifiable from `git log` and is the procedural enforcer of the test-first discipline.
- The task2 commit (test surface) lands BEFORE the task3 / task4 commits (implementations); a future reviewer can `git checkout` the task2 SHA and observe that the tests fail (the implementation does not yet exist).
- Round summary records each commit's SHA and the gate output at that SHA verbatim (no paraphrased validation claims, per BL-20260509-rlcr-recovery-round-bounded-scope).

### Synthetic-Fixture Discipline

- Every synthetic-fixture-driven test name MUST include the literal substring `synthetic` (e.g., `test_runner_admits_synthetic_scalar_pair`, `test_verifier_rejects_synthetic_inverted_bounded`).
- The round summary explicitly distinguishes synthetic-fixture coverage (this round's scope) from real-trace coverage (deferred to M2 follow-up plan once `data/local_labels/label_admitted.json` is non-empty).
- No synthetic-fixture test is permitted to imply that AC-5 (label stabilization, M2 scope) has progressed.

### BitLesson Anchors

- `BL-20260508-post-commit-recapture-discipline` — applies if any output bundle records a HEAD-derived field; the round's last commit is the implementation; no live capture is required for this round (synthetic fixtures only), so the post-commit recapture discipline is satisfied trivially.
- `BL-20260509-strict-admission-contract-first` — the bidirectional `asserted == DEFECT_TOKENS` lockstep is the procedural enforcer of the test-first contract.
- `BL-20260509-rlcr-recovery-round-bounded-scope` — the per-round bounded scope discipline is the meta-constraint; the deferral table at the end of the round summary names AC-5 / AC-6 / AC-7 / AC-8 / AC-9 / AC-10 / AC-11 / AC-12 with their target follow-up plan, NOT as silent drops.

### Anti-Goals (binding for this round)

- No re-implementation of AC-1 / AC-2 / AC-3 from the parent plan (already credible-correctness after commit `5ddb990` on the prior loop).
- No widening into AC-5 (label stabilization), AC-6 (HRD coverage scaling), AC-7 (bounded arm runtime production), AC-8 (geomean reproducibility against real labels), AC-9 (cross-arch transfer), AC-10 (prior-art comparison), AC-11 (L1 bypass scaffold), AC-12 (ablation gate). Those belong to subsequent plans (Plan B through Plan F per the decomposition table in the parent plan / brainstorm draft).
- No widening of `tools/test_plan_token_scanner.py` exemption table; if any new file trips the scanner, re-prose the offending text rather than exempt the file.
- No edits to `HANDOFF.md` / top-level `README.md` / per-tree `README.md`s / `BLOCKED_ON_3090TI.md` — these are tracked-doc-hygiene scope and require a dedicated plan per `BL-20260507-tracked-doc-hygiene-after-substantive-work`.
- No widening of tracer arch token sets without lockstep (`BL-20260507-arch-token-multi-surface-lockstep`).
- No `instruction.proto` wire-format changes.
- No 34-bucket reuse-distance partition modifications.
- No `stddev_threshold` loosening from 0.03.
- No completion-gate overclaim ("the only", "sole", "completes" without scope-bounding) — the round summary uses scope-bounded language ("M1 lower bound is closed", NOT "the cache predictor is complete").

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

# Test-First Defect-Token Lockstep For Predictor Comparison Surfaces

## Original Idea

Brainstorm a repo-grounded, novel, and high-quality research direction for completing Milestone 1's lower bound on the cache-hit-rate predictor — specifically the two remaining unimplemented code surfaces: (a) `workloads/simbench_bw/comparison_runner.py` and (b) `_verify_cache_predictor_run_summaries` in `workloads/simbench_bw/verify_artifacts.py` — plus their narrow test surface. The brainstorm output will become the input to `/humanize:gen-plan` and ultimately to a fresh, narrow RLCR loop replacing the just-stalled oversized loop.

Project context (binding):
- The just-stalled RLCR loop at `.humanize/rlcr/2026-05-09_10-55-58/` ran R0 (REGRESSED) -> R1 (STALLED) -> R2 (STALLED). The Mainline Drift Circuit Breaker fired after 3 consecutive non-advancing rounds. The structural problem was that the original plan's lower bound (9 ACs, ~20 tasks: schema + predictor module + non-circular ingestion + verifier hook + comparison runner + label stabilization + experiment matrix + bounded arm + geomean reproducibility + broader tests) was too large for any single round under the loop's all-or-nothing review semantics. The recovery path is to decompose into smaller plans; THIS brainstorm is for Plan A (M1 lower-bound completion).
- AC-1 / AC-2 / AC-3 are now credible-correctness after commit `5ddb990`:
  - `workloads/simbench_bw/schemas/cache_predictor.schema.json` (strict, 5-field manifest_pairing_index, 3-arm oneOf with not.anyOf exclusivity, per-arm `input_sidecar_digests.required: ["sector_stream"]` enforcement on scalar/bounded; typed_unavailable may omit sector_stream)
  - `workloads/simbench_bw/cache_predictor.py` (public API; non-circular discipline via `_validate_input_keys` rejecting hrd_profile/hit_count/hit_rate/model_hit; Smith-Eklund LRU stack-distance via `tools/hrd_profiler/observation_builder.build_launches_from_trace_dir`'s Fenwick-tree tracker; bundle_underconfigured admission gate via `_read_structural_underconfiguration` reading STRUCTURAL fields (bundle_underconfigured, underconfigured_bin_count, admitted_bin_count, typed_reason, scope) from sibling hrd_profile.json + convergence_confidence.json — NEVER hit_count / hit_rate)
  - Verified live: predict_cache_hit_rate on `workloads/simbench_bw/results/dram_stream_stride_mix/blackwell_consumer_unknown/sm_120/trace_trace_00` correctly emits `result_kind=typed_unavailable, typed_reason=cache_model_underconfigured, admitted_bin_count=2, underconfigured_bin_count=5`.
- AC-4 partially landed: `workloads/simbench_bw/comparison_contract.toml` (NCU metric `lts__t_sectors_lookup_hit_rate.pct`; 7-field join key `(slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256, arch, ncu_metric_version)`; error formula `|predicted - ground_truth| / max(ground_truth, epsilon)` with `epsilon = 0.01`; geomean rule = log-space mean over admitted cells; coverage gates 5 / 10 / 20). The runner that consumes this contract does NOT yet exist.
- AC-13 partial: `workloads/simbench_bw/tests/test_cache_predictor.py` has 24 tests (schema arm-exclusivity + non-circular + sector_stream per-arm + bundle_underconfigured admission gate + structural-field non-circular reader). The verifier-hook test surface and runner test surface are absent.
- `data/local_labels/label_admitted.json` is still empty `{cells: []}`. The two attempted NCU label cells are in `label_excluded.json` with `typed_reason: label_repeatability_failed`. AC-5 (label stabilization) is M2 scope and explicitly OUT of this brainstorm.

The two specific unimplemented surfaces this brainstorm covers:

(1) `workloads/simbench_bw/comparison_runner.py`. Consumes admitted predictor bundles (per-kernel-per-launch from `predict_cache_hit_rate` writes via `cache_predictor.write_bundle`) and admitted NCU label cells (from `data/local_labels/label_admitted.json`) under the contract pinned in `comparison_contract.toml`. The brainstorm should articulate:
  - The join semantics: 7-field key `(slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256, arch, ncu_metric_version)`. NCU side carries `ncu_metric_version`; predictor side does not yet — what is the right shape for predictor-side `ncu_metric_version`? An explicit field on the bundle? An implicit "predictor was developed against contract X" fact? A separate per-comparison sidecar?
  - NCU percent-to-fraction normalization: NCU metric `lts__t_sectors_lookup_hit_rate.pct` is a percentage in [0, 100]; the predictor emits a fraction in [0, 1]. The runner must normalize BEFORE applying the error formula. Where is the normalization fact recorded — runner code, contract field, or sidecar metadata?
  - Bounded arm comparison: when the predictor emits `result_kind=bounded`, the runner must report three error variants (lower-bound-error / upper-bound-error / midpoint-error) per the contract or pick one canonical variant. AC-7's bounded arm is M4 scope, but the runner must define the protocol now so M4 is a producer-side widening, not a contract change.
  - Typed_unavailable exclusion: cells where the predictor emits typed_unavailable MUST be excluded from the geomean but counted toward the coverage gate denominator. The exclusion record must be itself an admitted artifact (not silent) — what shape?
  - Coverage gate semantics: contract pins 5/10/20 thresholds. Below-5 produces `typed_unavailable: headline_geomean_admitted_cell_count_below_threshold`. What about 5-9 (admitted but flagged as low-confidence) vs 10-19 vs 20+ (full-confidence)?
  - Bootstrap CI on the geomean: cell-level resampling (the unit of variation is the admitted cell, not the per-launch sample). What seed discipline preserves AC-8's byte-identical re-run requirement?
  - Output bundle schema: the runner emits ONE bundle per (calibration_set, validation_set, holdout_set) split per `splits.toml` — three bundles total per run? Or one bundle with three result fields? The schema needs a strict shape mirroring the predictor schema's discipline (additionalProperties: false; oneOf arms for admitted / typed_unavailable; manifest_pairing_index analogue for the comparison run itself).

(2) `_verify_cache_predictor_run_summaries` in `workloads/simbench_bw/verify_artifacts.py`. The brainstorm should articulate:
  - The pattern to mirror: R4 commit `d2ec483`'s `_verify_divergence_proxies_run_summaries` is the canonical exemplar (5-field manifest_pairing_index cross-check; sidecar digest verification; bundle-to-summary binding).
  - What the verifier MUST enforce on every cache-predictor bundle:
    - schema-validates against `cache_predictor.schema.json` (strict mode)
    - all 5 manifest_pairing_index fields match the enclosing run summary's fields
    - `input_sidecar_digests.sector_stream.sha256` matches the on-disk file's actual sha256 (witness integrity)
    - `input_sidecar_digests.dynamic_trace_pb.sha256` if present matches the on-disk file's actual sha256
    - `input_sidecar_digests` MUST NOT contain forbidden keys (`hrd_profile`, `hit_count`, `hit_rate`, `model_hit`, `modeled_hit_rate`, `hrd_profile_path`) — this is the schema-side circular_predictor_input_rejected enforcement
    - bounded arm: `hit_rate_lower <= hit_rate_upper`
    - typed_unavailable arm: typed_reason is in schema enum (already enforced by schema, but the verifier should reject typed_reason values that are syntactically valid but operationally unknown to this verifier version)
    - bundle_underconfigured cross-check: if the predictor bundle is `result_kind=scalar` or `result_kind=bounded`, the sibling `hrd_profile/hrd_profile.json`'s `bundle_underconfigured` MUST be False or absent (this is the same bug Codex's prior review caught at the runtime layer; the verifier locks the contract permanently)
  - Cache-predictor bundles where the typed_reason names a CRITICAL-RULE violation (e.g., circular_predictor_input_rejected, comparison_contract_drift, input_sidecar_digest_drift) MUST be hard-rejected (not just typed-failure-recorded); the verifier treats them as evidence-integrity failures.
  - The verifier's typed-defect frozenset should mirror the contract-first lockstep pattern from BL-20260509-strict-admission-contract-first (defect tokens + bidirectional structural lockstep with the test suite via `asserted == DEFECT_TOKENS`, not subset-of).
  - Run-summary cross-binding: the run-summary JSON should record per-cache-predictor-bundle paths, hashes, and source_commit so verifier can match them; the post-commit re-anchor discipline of BL-20260508 applies.

The brainstorm output should articulate (a) the primary direction with rationale + objective evidence (citing concrete repo paths and existing exemplars) + known risks; (b) 3-5 alternative directions and why-not-primary; (c) acceptance-criteria seed candidates that map to AC-4 and AC-13 narrow scope (NOT into AC-5/AC-6/AC-7/AC-8/AC-9/AC-10/AC-11/AC-12 which belong to subsequent plans); (d) path-boundary suggestions (upper / lower bound) sized to fit a single tractable RLCR round (~3-4 tasks total); (e) the natural sequencing of the two surfaces (comparison_runner.py and the verifier hook) into a per-round task ladder.

Hard constraints (binding per `CLAUDE.md`):
- Critical Rule 1: NCU only via `tools/local_labels/` and `data/local_labels/` on the two local devices. No NCU on cloud or server. No `.ncu-rep` artifacts outside the quarantine.
- Critical Rule 2: no GPU-driver-root operations.
- Critical Rule 3: no privileged CUPTI surfaces (CUPTI Activity allowed).
- Critical Rule 4: no host-driver work.
- Critical Rule 5: no simulator revival.
- The non-circular discipline: the predictor MUST NEVER consume modeled `hit_count` / `hit_rate` / `model_hit` fields from any sidecar; only the structural metadata is admissible.

Anti-goals for THIS plan:
- No re-implementation of AC-1/AC-2/AC-3 (already credible-correctness after commit `5ddb990`).
- No widening into AC-5 (label stabilization), AC-6 (HRD coverage scaling), AC-7 (bounded arm runtime), AC-9 (cross-arch transfer), AC-10 (prior-art comparison), AC-11 (L1 bypass scaffold), AC-12 (ablation gate). Those belong to subsequent plans.
- No widening of `tools/test_plan_token_scanner.py` exemption table.
- No edits to HANDOFF.md / top-level README.md / per-tree READMEs / BLOCKED_ON_3090TI.md.
- No widening of tracer arch token sets without lockstep.
- No `instruction.proto` wire-format changes.
- No 34-bucket reuse-distance partition modifications.
- No `stddev_threshold` loosening.
- No completion-gate overclaim.

Repo paths the brainstorm will want to reference:
- `plans/cache_hit_rate_predictor_first_plan.md` (parent plan; M1 phase D = verifier hook; AC-4 + AC-8 partial = comparison_runner.py)
- `workloads/simbench_bw/cache_predictor.py` (the predictor module)
- `workloads/simbench_bw/schemas/cache_predictor.schema.json` (the schema)
- `workloads/simbench_bw/comparison_contract.toml` (the pinned contract)
- `workloads/simbench_bw/verify_artifacts.py` (hook target file)
- `workloads/simbench_bw/phase_cycle_ledger_consumer.py` (admission-helper exemplar)
- `data/local_labels/{label_admitted,label_excluded,label_repeatability}.json` (label-side artifact tree)
- `workloads/simbench_bw/splits.toml` (45/16/16 calibration/validation/holdout split)
- `tools/hrd_profiler/observation_builder.py` (Fenwick-tree LRU + `_BuildResult` shape)
- `tools/hrd_profiler/reuse_distance_tracker.py` (Smith-Eklund stack-distance algorithm)
- `.claude/skills/validating-nonroot-gpu-measurement/SKILL.md`
- `references/papers/knowledge_base/active/` (prior-art citations, future use)
- `.humanize/bitlesson.md` (BL-20260508-post-commit-recapture-discipline; BL-20260509-strict-admission-contract-first; BL-20260509-rlcr-recovery-round-bounded-scope)

Output: a directed-swarm idea draft suitable for handoff to `/humanize:gen-plan`. Quality bar: novel framings considered (not just rote translation of plan tasks 5+6 into prose), clear path-boundary discipline (lower-bound = ~3 tasks fits a single round), explicit acknowledgement of the just-stalled oversized loop's failure mode and how this plan's narrowed scope avoids it.

## Primary Direction: Joint Test-First Contract Lockstep

### Rationale

Authors the test surface FIRST — negative-control tests for every defect token before any implementation — making the contract itself the primary artifact and the implementation a derived satisfaction of the test surface, mirroring `BL-20260509-strict-admission-contract-first`'s bidirectional `asserted == DEFECT_TOKENS` discipline. This framing was selected as primary because it directly addresses the just-stalled loop's two named failure modes (R1 STALLED on "ships latent bugs that should have been caught by tests + false validation claims"; R2 STALLED on "the schema relaxation was too broad" — both are surfaces a test-first contract would have blocked at the test layer before the implementation landed). The empty `data/local_labels/label_admitted.json` makes synthetic-fixture-first unavoidable for M1; this direction turns that constraint into deliberate discipline.

### Approach Summary

Implement both `workloads/simbench_bw/comparison_runner.py` and `_verify_cache_predictor_run_summaries` (in `workloads/simbench_bw/verify_artifacts.py`) under a **Joint Test-First Contract Lockstep** discipline: author the full test surface BEFORE any implementation, using negative-control tests for every defect token and synthetic admitted-predictor-bundle + synthetic NCU-label fixtures. The test suite enforces bidirectional structural lockstep (`asserted == DEFECT_TOKENS`, mirroring BL-20260509) — every typed defect token in the runner + verifier's frozenset MUST have a corresponding negative-control test, and every test MUST assert a token from the frozenset.

**Tractable per-round task ladder (~3 tasks, sized to fit a single RLCR round):**

1. **Author the joint defect-token frozenset and synthetic-fixture builder** — define `_RUNNER_DEFECT_TOKENS` and `_VERIFIER_DEFECT_TOKENS` frozensets in their respective modules; expose them via `__all__` so tests can import and assert against them. Author `_make_synthetic_admitted_predictor_bundle()` (parameterized by result_kind, manifest_pairing_index, sector_stream digest), `_make_synthetic_ncu_label_cell()` (parameterized by 7-field join key, profiler_version, ground_truth_pct), `_make_synthetic_run_summary_with_cache_predictor_path()` fixture builders in a new `workloads/simbench_bw/tests/conftest_cache_predictor_comparison.py` (or extend existing conftest).

2. **Author the test surface FIRST** — at `workloads/simbench_bw/tests/test_comparison_runner.py` and `workloads/simbench_bw/tests/test_verify_cache_predictor_run_summaries.py`. Each defect token gets one negative-control test (constructs a malformed input, asserts the typed defect fires) AND one positive-control companion (the well-formed input admits). The bidirectional structural lockstep test (`test_runner_defect_tokens_match_test_surface_exactly`) imports `_RUNNER_DEFECT_TOKENS` and asserts `asserted_in_tests == _RUNNER_DEFECT_TOKENS` (set equality, not subset). Same pattern for the verifier.

3. **Implement minimum-viable `comparison_runner.py` and `_verify_cache_predictor_run_summaries`** that satisfy the test surface — a runner that loads the contract, joins on the 7-field key, normalizes NCU percent to fraction, computes the error formula, applies the coverage gate, and emits a schema-valid output bundle; a verifier hook that mirrors `_verify_divergence_proxies_run_summaries` (5-field manifest_pairing_index cross-check; schema validation; sidecar digest verification; non-circular forbidden-keys rejection; bounded-arm `hit_rate_lower <= hit_rate_upper`; bundle_underconfigured cross-check). NO production code is written until the test suite is in place; once the suite is locked, the implementation is "fill in the contract" work.

**Open design questions the brainstorm explicitly leaves open for `/humanize:gen-plan` Codex first-pass:**

- Predictor-side `ncu_metric_version`: explicit bundle field, comparison-side pinning (Alt-4), or per-comparison sidecar? The Joint Test-First approach is agnostic; the test surface assertions name the contract drift typed token regardless of where the version lives.
- Output bundle shape: omnibus single bundle vs three split-bundles per `splits.toml` (Alt-3's domain). The test fixtures should construct both shapes to keep the option open.
- Bounded-arm dispatch: midpoint-canonical vs three-variant-array (Alt-5's domain). The test surface should construct synthetic bounded bundles and exercise both options.

**Path boundaries:**
- **Lower bound** (must close in this round): defect-token frozenset + synthetic fixtures + test surface + minimum-viable runner + minimum-viable verifier hook. AC-4 narrow (runner exists and consumes contract) + AC-13 narrow (test surface asserts the contract). ~3 tasks.
- **Upper bound** (round can close on lower bound; upper bound only if time permits): real-trace integration tests once AC-5 (label stabilization, M2 scope) lands; performance profiling of the geomean computation; verifier hook on a real captured trace bundle; AC-13 broader coverage (the 6-test test surface for AC-1..AC-12 from the parent plan's task19).

**Acceptance-criteria seed candidates** (mapped to AC-4 and AC-13 narrow scope only):

- **AC-4-narrow** (comparison contract IS consumed): `comparison_runner.py` loads `comparison_contract.toml` at module init; rejects bundles whose 7-field join key fails to match a label cell with typed `comparison_contract_drift`; normalizes NCU percent to fraction inline at point-of-use; computes the error formula with `epsilon=0.01` exactly; emits a schema-valid output bundle.
- **AC-13-narrow** (test surface enforces contract bidirectionally): every entry in `_RUNNER_DEFECT_TOKENS` has a corresponding negative-control test that constructs a malformed input and asserts the typed defect; the bidirectional structural lockstep test asserts `asserted_in_tests == _RUNNER_DEFECT_TOKENS` (set equality); same for `_VERIFIER_DEFECT_TOKENS`.
- **AC-1-extension** (verifier hook enforces strict schema + non-circular): `_verify_cache_predictor_run_summaries` mirrors `_verify_divergence_proxies_run_summaries`'s structural pattern (5-field manifest_pairing_index cross-check; cached-vs-recomputed consistency; schema validation); rejects bundles where `input_sidecar_digests` carries any of `{hrd_profile, hit_count, hit_rate, model_hit, modeled_hit_rate, hrd_profile_path}`; rejects bundles where bounded-arm `hit_rate_lower > hit_rate_upper`; rejects bundles where `result_kind in {scalar, bounded}` but sibling `hrd_profile.json` reports `bundle_underconfigured=true`.

### Objective Evidence

- BL-20260509-strict-admission-contract-first at `.humanize/bitlesson.md:96-104` — source of the bidirectional structural lockstep pattern and the three-round R20→R21→R22 evolution showing contract-first design catches defects earlier than failure-mode-first design. R20 helper (none) admitted 98/100 as 100/100; R21 helper (failure-mode-first; 13 defect tokens; CUPTI-only identity check; subset-of test lockstep) admitted wrong-NVML-identity probe as `ok=True, defects=[]`; R22 helper (contract-first; 20 defect tokens; identity check on all 4 surfaces; `asserted == DEFECT_TOKENS` bidirectional) correctly rejected both probes.
- `workloads/simbench_bw/phase_cycle_ledger_consumer.py:32-81` — fixture builder pattern (`_admissible_ledger()` parameterized for positive+negative), typed exception classes with `TOKEN` + `to_json()`, registry loader with per-kernel overrides.
- `workloads/simbench_bw/tests/test_phase_cycle_ledger_consumer.py:1-150` — 9 tests covering registry defaults, per-kernel overrides, overhead exceeded, boundary unresolved (both positive + negative), demonstrating fixture-driven test-first discipline.
- `workloads/simbench_bw/tests/test_cache_predictor.py:1-250` — current 24-test surface (schema arm-exclusivity + non-circular + sector_stream per-arm + bundle_underconfigured admission gate + structural-field non-circular reader); `_make_mpi()` and `_make_input_digests()` are minimal fixture builders that the new conftest_cache_predictor_comparison.py would extend.
- `workloads/simbench_bw/comparison_contract.toml:1-130` — pinned 7-field join key, error formula, epsilon=0.01, coverage gates (smoke_test_floor_per_arch=5, publication_grade_floor_per_arch=10, aspirational_target_total=20).
- `workloads/simbench_bw/cache_predictor.py:62-74` — typed-reason enum with 11 entries; lines 77-100 show typed exception `__init__` + `to_json()` pattern (the model the new `_RUNNER_DEFECT_TOKENS` frozenset and the new `_VERIFIER_DEFECT_TOKENS` frozenset should mirror).
- `workloads/simbench_bw/verify_artifacts.py:1468-1672` (R4 commit `d2ec483`) — `_verify_divergence_proxies_run_summaries` demonstrates the lockstep verifier: schema validation, manifest-pairing identity cross-check (5-field tuple bind), cached-vs-recomputed consistency check on per-bundle count fields, rejects on mismatch. **The new `_verify_cache_predictor_run_summaries` is a structural carbon copy of this function with cache-predictor-specific defect tokens.**
- `workloads/simbench_bw/tests/test_dual_source_calibrated_bound_contract.py:1-80` — fixture + validator pattern with default fallback, negative-control tests for misconfigured overrides; another precedent for the synthetic-fixture-driven negative-control discipline.
- `workloads/simbench_bw/splits.toml:1-18` — 45 calibration / 16 validation / 16 holdout (assignments at lines 20-327); the runner's coverage-gate code paths read this and bind it to per-split admission.

### Known Risks

- **Label stabilization (AC-5 out of scope)**: Synthetic NCU label cells in the fixture must be hand-authored to avoid circular self-verification. If AC-5 lands in M2, the synthetic fixtures remain regression-locked; real labels will replace them post-promotion. Mitigation: clearly mark the fixture builder's output as synthetic in test names and conftest comments.
- **Fixture path dependencies**: Synthetic bundles reference imaginary paths in `input_sidecar_digests` (e.g., `trace_00/sector_stream/sector_stream.jsonl`). The runner/verifier must not attempt filesystem access on fixture bundles; tests mock or skip I/O-dependent code paths. Mitigation: the fixture factory writes synthetic on-disk files in `tmp_path` so digests verify against real bytes.
- **Geomean numerical stability**: Bootstrap CI at 2000 resamples + log-space mean computation are well-trodden (prior art in HRD profiler), but the relative-error-space geomean can be numerically unstable if any cell's relative error is very small or zero. The `epsilon=0.01` policy gates this, but tests should cover boundary cases (ground_truth near epsilon, predicted==ground_truth, predicted>>ground_truth).
- **Contract evolution**: The `comparison_contract.toml` is versioned (`schema_version=1`); if future rounds bump the contract (e.g., new error formula, epsilon widening), the runner's and verifier's `DEFECT_TOKENS` must widen in lockstep. The test suite's `asserted == DEFECT_TOKENS` discipline enforces this visibility.
- **Test-first discipline must be enforced procedurally**: this approach only works if the test suite is committed BEFORE the implementation lands. If both land in one commit, a future reviewer cannot distinguish "test-first" from "test-along". Mitigation: per-task git-commit ordering (test commit precedes implementation commit) is part of the plan's task ladder.

## Alternative Directions Considered

### Alt-1: Verifier-First Integrity Boundary

- Gist: Treat the verifier as the dominant artifact (the integrity firewall) and the runner as a downstream consumer that trusts pre-verified bundles. The architecture sequences as: implement `_verify_cache_predictor_run_summaries` first with bidirectional test coverage, then implement `comparison_runner.py` consuming verified bundles. The runner depends entirely on the verifier's pass/fail gate; no runner code consumes or modifies predictions. Frames "admission" as the primary research output and the geomean number as a secondary derived value.
- Objective Evidence:
  - `workloads/simbench_bw/verify_artifacts.py:1468-1672` — R4 commit `d2ec483` divergence_proxies verifier (template structural pattern; defect-token frozenset; 5-field manifest binding; cached-count recomputation).
  - `workloads/simbench_bw/verify_artifacts.py:1071-1270` — HRD verifier Section 10 three-way bundle_underconfigured invariant cross-checking (precedent for the cache-predictor verifier's bundle_underconfigured cross-check).
  - `workloads/simbench_bw/schemas/cache_predictor.schema.json:159-175` — typed_reason enum with 11 values; the verifier's defect-token frozenset must mirror this enum exactly per `BL-20260507-arch-token-multi-surface-lockstep`.
  - `workloads/simbench_bw/phase_cycle_ledger_consumer.py:159-246` — admission-helper precedent (typed exceptions with defect_kind tokens, silent-pass-on-success).
- Why not primary: Excellent fit and confidence (`high`), but architecturally biased toward the verifier as the primary research artifact. The just-stalled loop's failure mode was "ships latent bugs that should have been caught by tests" — the test-first discipline (D3) addresses this more directly than verifier-first because the test-first surface catches both runner AND verifier defects pre-implementation, whereas verifier-first only catches predictor-bundle defects pre-runner. The two directions fold cleanly: D3's test surface includes the verifier's negative-control tests; D1's structural pattern is what the verifier implementation satisfies.

### Alt-2: Runner-First Geomean Producer

- Gist: Treat the runner as the dominant artifact (the publication-grade geomean machine) and the verifier as a thin schema gate. The architecture sequences as: implement `comparison_runner.py` first with full bootstrap CI, error-formula computation, log-space geomean, coverage-gate semantics, and an omnibus output bundle; the verifier becomes a thin hygiene gate that checks per-bundle schema validity and manifest-pairing identity. Frames the predictor-vs-NCU number as the primary research output and integrity gating as a derived hygiene concern.
- Objective Evidence:
  - `workloads/simbench_bw/cache_predictor.py:177-211` — `_bootstrap_ci_over_launches` resampling pattern (BOOTSTRAP_RESAMPLES=2000, CI_CONFIDENCE=0.95, seed-deterministic RNG).
  - `tools/hrd_profiler/profiler.py` — `_percentile_bootstrap_ci_launch_unit` precedent (whole-launch resampling; per-bin aggregation) that mirrors the runner's cell-level resampling unit.
  - `workloads/simbench_bw/comparison_contract.toml:1-130` — fully pinned contract (the runner only needs to read it).
  - `workloads/simbench_bw/splits.toml:1-18` — 45/16/16 split structure for per-split runner output.
- Why not primary: The risk of "thin verifier" recreates the prior loop's failure mode: integrity bugs in the predictor go undetected longer. The just-stalled R2 review explicitly named "scalar bundles can omit sector_stream" as a schema-laxness bug — that's a verifier-side gap that runner-first deprioritizes. D3 (test-first) folds runner-first's bootstrap CI and geomean computation work in unchanged but mandates the verifier negative-control tests be written FIRST.

### Alt-3: Comparison-Bundle Schema Co-Design

- Gist: Treat the runner's OUTPUT bundle schema as the central design surface — author a new `comparison_bundle.schema.json` mirroring the predictor schema's strict discipline (additionalProperties: false; oneOf arms for admitted / typed_unavailable; comparison_run_pairing_index analog; strict per-arm required fields). The runner emits to the schema; the verifier reads the schema. Frames the typed comparison output as a first-class artifact analogous to the predictor bundle.
- Objective Evidence:
  - `workloads/simbench_bw/schemas/cache_predictor.schema.json:1-241` — direct precedent (additionalProperties: false; manifest_pairing_index 5-field; oneOf 3-arm exclusivity with not.anyOf clauses; per-arm input_sidecar_digests required clauses).
  - `workloads/simbench_bw/schemas/` (43 schema files): divergence_proxies.schema.json, perturbation_run_ledger.schema.json, bug_detection_ledger.schema.json, metamorphic_pair_ledger.schema.json — the schema family already follows the additionalProperties:false + per-cell-array + aggregate-envelope pattern.
  - `workloads/simbench_bw/schemas/local_label_sidecar.schema.json:24-28` — label sidecar's existing fields (slug, shape_point, repeat_index, launch_index, binary_sha256, profiler_family, profiler_version, cache_hit_rate_l2_sector, confidence_interval_l2_sector) inform the new comparison_bundle's join-key fields.
- Why not primary: Strong fit but adds a new schema file as a prerequisite, increasing surface area. D3 (test-first) treats the output bundle's structural shape as a SET of test fixtures rather than a separate schema artifact — the runner's output structure is fixed by the negative-control tests' positive companions. If the user prefers the explicit schema artifact, this folds into D3 as task 0.5: author `comparison_bundle.schema.json` BEFORE the test surface, then write tests that schema-validate every emitted bundle.

### Alt-4: Contract-Pinned ncu_metric_version Protocol

- Gist: Frame the predictor-side `ncu_metric_version` question as a contract-protocol-version problem rather than a per-bundle field problem. The `comparison_contract.toml` carries the protocol version (already pinned at `"ncu_2026.1.x"`); predictor bundles compatible with the protocol revision are admissible; no per-bundle field is added to the predictor schema. The runner enforces the protocol by binding label-side `profiler_version` (extracted from per-launch sidecars) to the contract's pinned `ncu_metric_version` and rejecting drift with typed `comparison_contract_drift`.
- Objective Evidence:
  - `workloads/simbench_bw/comparison_contract.toml:31-37` — `ncu_metric_version = "ncu_2026.1.x"` already pinned as a contract surface.
  - `data/local_labels/sm_120/fp32_fma_dep_chain/trace_00/label_0_0.json:26` — `profiler_version: "Version 2026.1.1.0 (build 37634170) (public-release)"` already in the per-launch sidecar.
  - `tools/local_labels/repeatability_aggregator.py:279-324` — current admitted-cell schema does NOT carry profiler_version; widening it to include profiler_version would be the migration path.
  - `.humanize/bitlesson.md:75-84` — `BL-20260507-arch-token-multi-surface-lockstep` names the structural lockstep pattern that prevents protocol drift.
  - `workloads/simbench_bw/cache_predictor.py:62-74` — typed-reason enum already includes `comparison_contract_drift`, ready for ncu_metric_version-specific use.
- Why not primary: This direction answers ONE of the open design questions (where ncu_metric_version lives) but does not advance the M1 surface implementations directly. Its scope is narrow: the contract revision, the runner's contract-version check, and a structural lockstep regression test (~50-100 LOC). It folds cleanly into D3 as a specific design decision the test surface encodes (the `_RUNNER_DEFECT_TOKENS` frozenset includes `comparison_contract_drift`; the negative-control test constructs a label cell with mismatched `profiler_version` and asserts the typed defect fires).

### Alt-5: Bounded-Arm-Forward M4 Producer Widening

- Gist: Design the runner's output protocol as forward-compatible from day one — the runner accepts bounded predictor bundles and emits lower / upper / midpoint error variants — so M4's bounded-arm runtime production lands as a producer-side widening rather than a contract change. Pin midpoint-error-of-bounded-cell as the canonical headline error; lower and upper errors are diagnostic-only (saved in evidence).
- Objective Evidence:
  - `workloads/simbench_bw/comparison_contract.toml:95` — `bounded_arm_handling = "report_three"` is a forward-declaration in the contract; the runner must implement this from day one even though no bounded-arm production path exists.
  - `workloads/simbench_bw/schemas/cache_predictor.schema.json:137-148, 202-222` — bounded-arm schema definition (hit_rate_lower, hit_rate_upper, underconfigured_bin_count, admitted_bin_count) ready for synthetic bundles.
  - `workloads/simbench_bw/cache_predictor.py:364-666` — typed-arm dispatch pattern (`_typed_unavailable_bundle` factory + scalar return path) shows how a future `_bounded_bundle` factory would slot in.
  - `CLAUDE.md:236-259` — wire-extension policy permits "additive, version- or capability-gated fields" producer-side; the bounded-arm production is exactly this shape.
- Why not primary: This direction is a CONSTRAINT on the runner's design, not a primary direction itself. It says "design the runner this way", but the test-first discipline (D3) is the framing that ENFORCES that design. The bounded-arm-forward protocol becomes a synthetic-fixture in D3's test surface: `_make_synthetic_bounded_bundle()` constructs a `result_kind=bounded` bundle; the runner's negative-control test asserts the runner correctly dispatches the three error variants; the positive-control test asserts the headline error is the midpoint. The forward-compatibility property is a derived consequence of the test-first discipline, not its motivation.

## Synthesis Notes

The five alternatives fold cleanly into the primary's test-first task ladder rather than competing with it:

- **Alt-1 (Verifier-First)** contributes the structural pattern for `_verify_cache_predictor_run_summaries`: 5-field manifest_pairing_index cross-check, schema validation, cached-vs-recomputed consistency check, defect-token frozenset. The primary direction's third task (implement minimum-viable verifier) IS Alt-1's verifier; the difference is sequencing — D3 writes the verifier's tests first, D1 writes the verifier first.
- **Alt-2 (Runner-First)** contributes the runner's bootstrap CI, log-space geomean, and coverage-gate computation logic. The primary direction's third task (implement minimum-viable runner) IS Alt-2's runner; the difference is sequencing — D3 writes the runner's tests first, D2 writes the runner first.
- **Alt-3 (Schema Co-Design)** contributes the option of an explicit `comparison_bundle.schema.json` artifact. If the user prefers explicit schema artifact, this slots in as a pre-task 0.5 before the test surface; the test surface then schema-validates every emitted bundle. If the user prefers test-fixture-as-implicit-schema, the option is deferred and the runner's output shape is pinned by the test fixtures' positive companions.
- **Alt-4 (Contract-Pinned ncu_metric_version Protocol)** contributes a specific design decision about where ncu_metric_version lives. The primary direction's defect-token frozenset includes `comparison_contract_drift`; the negative-control test for that token is the surface where Alt-4's design lands (as the specific drift the test exercises).
- **Alt-5 (Bounded-Arm-Forward Widening)** contributes a specific design constraint on the runner's output shape. The primary direction's synthetic-fixture builder includes `_make_synthetic_bounded_bundle()`; the negative-control test for `bounded_arm_dispatch_misuse` asserts the runner correctly handles the three-variant protocol.

If `/humanize:gen-plan`'s Codex first-pass argues a different alternative should be primary, the synthesis path is unchanged: the primary direction defines the WHAT (the contract surface lockstep with the test surface) and the alternatives define the HOW for individual sub-decisions. The plan that emerges should preserve the test-first task ordering as a non-negotiable: the negative-control tests for every defect token are committed BEFORE the implementation that satisfies them lands, in separate per-task git commits, so the test-first discipline is verifiable from `git log` and a future reviewer cannot mistake "test-first" for "test-along".

This narrowed scope (one round, ~3 tasks, lower-bound = test scaffolding + minimum-viable runner + minimum-viable verifier) avoids the just-stalled loop's structural failure mode (9 ACs / 20 tasks in one round produced under-validated implementations and false validation claims) by making validation surface a first-class deliverable BEFORE implementation. The bidirectional `asserted == DEFECT_TOKENS` lockstep — proven across three RLCR rounds at R20 → R21 → R22 per BL-20260509 — is the procedural enforcer that makes the test-first discipline durable.

--- Original Design Draft End ---
