# Per-Arch Diagnostic Layer For Cache Predictor Validation With Live-Only Floor Discipline

## Goal Description

Extend `workloads/simbench_bw/comparison_runner.py` with a per-arch diagnostic layer (`per_arch_diagnostics`) that aggregates per-cell `relative_error` into per-arch summary statistics, exclusion-token tallies, and a coverage-gate verdict. Lock the harness contract this round so when AC-5.2 sm_120 evidence eventually lands on the RTX 5060 laptop, the operator runs the harness without code changes and the per-arch verdict block reflects the live cross-arch admission state.

In addition, this plan tightens the project's evidence discipline by adding a required `evidence_kind: live | synthetic` field to the local-label admitted-cell schema (bumping `local_label_admitted.schema.json` `schema_version` from 2 to 3), migrating the 5 already-admitted sm_86 cells in `data/local_labels/label_admitted.json` to `evidence_kind: live`, and updating `comparison_runner.py`'s top-level admitted gate so the floor counts only `live` cells. Synthetic test fixtures must declare `evidence_kind: synthetic`; they no longer satisfy the smoke-test floor. This makes the existing `Cross-arch-verified` discipline mechanically enforceable: a fixture-only sm_120 leg cannot accidentally appear admitted.

The harness is exercised on the local 3090 Ti against:

1. The 5 real sm_86 admitted cells in `data/local_labels/label_admitted.json` (migrated to schema_version=3 with `evidence_kind: live`), joined to synthesized matching predictor bundles using the same 7-field join keys read from the on-disk admitted file.
2. A synthetic 5-cell sm_120 fixture (declared `evidence_kind: synthetic`) that exercises the wire format end-to-end without sm_120 hardware. Its per-arch `coverage_gate_verdict.meets_floor` is `false` because the live count is zero.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: `per_arch_diagnostics` field shape is locked in `workloads/simbench_bw/schemas/comparison_bundle.schema.json` as a top-level OPTIONAL field with the wrapper shape:
  - `per_arch_diagnostics.per_arch`: object whose `propertyNames` matches the regex `^sm_[0-9]+$`; each per-arch entry has required keys `{arch, n_comparable_cells, n_headline_cells, statistics_status, relative_error_stats, excluded_reason_counts, coverage_gate_verdict}`; strict `additionalProperties: false` at every nested level.
  - `per_arch_diagnostics.unattributed_excluded_reason_counts`: object with three required keys (the three `_EXCLUDED_DIAGNOSTIC_VALUES` tokens) with non-negative integer values.
  - `relative_error_stats`: oneOf with three branches each binding `statistics_status` via `const` (so the schema enforces that quartile fields can ONLY appear when `statistics_status == "quartile_emitted"`).
  - The `additionalProperties: false` discipline is preserved at every level; existing bundles WITHOUT the field continue to validate (additive change; no `comparison_bundle.schema.json` schema_version bump).
  - Positive Tests:
    - A bundle whose `per_arch_diagnostics.per_arch` contains records for every arch in `contract.arch_set` validates against the updated schema.
    - A bundle WITHOUT `per_arch_diagnostics` validates (backward-compatible additive change).
    - Each oneOf branch correctly binds `statistics_status` const; a `quartile_emitted` record with all 7 stat fields validates.
  - Negative Tests:
    - A record with an unknown additional key fails validation.
    - A record missing a required key fails validation.
    - A `statistics_status` value outside the three-token enum fails validation.
    - A quartile field (`q25`, `q50`, `q75`) appearing on a record where `statistics_status != "quartile_emitted"` fails validation (oneOf const-binding catches this).
    - An `excluded_reason_counts` key outside the three-token enum fails validation.

- AC-2: `_per_arch_diagnostics(included_cells, excluded_cells, arch_set, contract) -> dict` produces the wrapper shape. Internally splits responsibilities:
  - `_per_arch_stats(included_cells_for_arch, contract) -> dict | None` consumes the per-arch slice of `included_cells` (filtered by `join_key.arch`) and produces the `relative_error_stats` block per AC-3.
  - `_per_arch_excluded_counts(excluded_cells_for_arch) -> dict` consumes the per-arch slice of `excluded_cells` (filtered by `join_key.arch == arch`) and produces the `excluded_reason_counts` block per AC-4.
  - Top-level `_unattributed_excluded_counts(excluded_cells_lacking_arch)` consumes excluded records lacking `join_key.arch` and produces the `unattributed_excluded_reason_counts` block.
  - Per-arch `n_comparable_cells_for_arch` formula: `(count of included_cells with join_key.arch == arch) + (count of excluded_cells with join_key.arch == arch and diagnostic in _EXCLUDED_DIAGNOSTIC_VALUES)`. Excluded records lacking `join_key.arch` do NOT affect any per-arch `n_comparable_cells`; they contribute only to `unattributed_excluded_reason_counts`. Excluded records whose diagnostic falls outside `_EXCLUDED_DIAGNOSTIC_VALUES` are rejected by the runner before tally per AC-4 (regression guard).
  - Per-arch `n_headline_cells_for_arch` formula: count of `included_cells` with `join_key.arch == arch` (the stats sample).
  - Positive Tests:
    - 5 included sm_86 cells + 1 excluded sm_86 cell with diagnostic `comparison_contract_drift` produces `n_comparable_cells == 6`, `n_headline_cells == 5` for sm_86.
    - A mixed sm_86/sm_120 fixture where sm_86 cells have `relative_error` in `[0.05, 0.10]` and sm_120 cells have `relative_error` in `[0.20, 0.30]` yields disjoint per-arch min/max ranges.
  - Negative Tests:
    - Swapping sm_120 `relative_error` values must NOT change sm_86 stats (per-arch independence regression guard).

- AC-3: `statistics_status` enum and small-n statistics behavior are pinned:
  - `quartile_emitted` (n_headline_cells >= 4): `relative_error_stats` has 7 fields (mean, stdev, min, max, q25, q50, q75).
  - `insufficient_cells_for_quartile_statistics` (1 <= n_headline_cells <= 3): `relative_error_stats` has 4 fields (mean, stdev, min, max); quartile fields forbidden.
  - `no_cells` (n_headline_cells == 0): `relative_error_stats == JSON null`.
  - Stdev semantics: for `n_headline_cells >= 2`, `stdev = statistics.stdev(values)` (Python sample stddev, n-1 denominator); for `n_headline_cells == 1`, `stdev = 0.0` (special case; `statistics.stdev()` raises for n=1).
  - Quartile method: `statistics.quantiles(data, n=4, method="exclusive")` (Python default; Tukey hinges).
  - Positive Tests:
    - 5-cell fixture `[0.10, 0.20, 0.30, 0.40, 0.50]` produces `quartile_emitted` with the exact mean / stdev / min / max / q25 / q50 / q75 expected from `statistics.quantiles(..., method="exclusive")`.
    - 3-cell fixture produces `insufficient_cells_for_quartile_statistics` with mean / stdev / min / max only.
    - 1-cell fixture produces `insufficient_cells_for_quartile_statistics` with `stdev == 0.0`.
    - 0-cell fixture produces `no_cells` with `relative_error_stats == null`.
  - Negative Tests:
    - A quartile field on a record where `statistics_status != "quartile_emitted"` fails schema validation (oneOf const-binding).

- AC-4: `excluded_reason_counts` vocabulary uses the three runner-side `_EXCLUDED_DIAGNOSTIC_VALUES` tokens (`comparison_contract_drift`, `comparison_runner_join_key_unmatched`, `predictor_typed_unavailable`):
  - `per_arch_diagnostics.per_arch[arch].excluded_reason_counts`: object REQUIRES all three keys with non-negative integer values; zero defaults emitted; counts only excluded records WITH `join_key.arch == arch` and diagnostic in the three-token enum.
  - `per_arch_diagnostics.unattributed_excluded_reason_counts`: object REQUIRES the same three keys with non-negative integer values; zero defaults emitted; counts only excluded records lacking `join_key.arch`.
  - Tally rule: dispatch each excluded record by `(record["join_key"]["arch"] if present else UNATTRIBUTED)` and by `record["diagnostic"]` (which must be one of the three tokens; runner rejects records with out-of-vocab diagnostic before tally).
  - Positive Tests:
    - A fixture with 1 unmatched + 2 contract_drift + 1 typed_unavailable per-arch tallies as `{comparison_runner_join_key_unmatched: 1, comparison_contract_drift: 2, predictor_typed_unavailable: 1}` for that arch.
    - An arch with zero exclusions emits all-zero tally `{comparison_runner_join_key_unmatched: 0, comparison_contract_drift: 0, predictor_typed_unavailable: 0}`.
    - A fixture with one excluded record lacking `join_key` entirely tallies into `unattributed_excluded_reason_counts.comparison_contract_drift`.
  - Negative Tests:
    - An excluded record with a diagnostic outside the three-token enum is rejected by the runner before tally (regression guard against vocabulary drift).
    - A tally key outside the schema enum fails validation.

- AC-5: `coverage_gate_verdict` reports per-arch live-evidence admission floor (DEC-1 RESOLVED to live-only floor counting):
  - Per-arch object shape: `{arch, headline_count_live, headline_count_synthetic, floor, meets_floor}` where `floor` is `contract.coverage_gate.smoke_test_floor_per_arch` and `meets_floor` is the boolean `headline_count_live >= floor` (synthetic cells do NOT satisfy the floor).
  - Top-level `admitted` `result_kind` gate semantics CHANGE this round: the gate becomes `any(headline_count_live[a] >= floor for a in arch_set)`. Existing tests asserting admitted on synthetic-only fixtures are updated per AC-11.
  - The runner derives `headline_count_live` and `headline_count_synthetic` from each included cell's `evidence_kind` field (added in AC-10).
  - Positive Tests:
    - 5 live sm_86 cells produce `headline_count_live == 5`, `headline_count_synthetic == 0`, `meets_floor == true` for sm_86.
    - 5 synthetic sm_120 cells produce `headline_count_live == 0`, `headline_count_synthetic == 5`, `meets_floor == false` for sm_120.
    - Top-level `result_kind` is `admitted` when at least one arch has `headline_count_live >= floor`; `typed_unavailable` (with `typed_reason: headline_geomean_admitted_cell_count_below_threshold`) when no arch meets the live floor.
  - Negative Tests:
    - A per-arch `coverage_gate_verdict` that omits any of the four required fields fails schema validation.
    - A behavior-preserving regression test that fakes 5 synthetic sm_120 cells without any live cells must NOT produce `result_kind == "admitted"` (live-only floor catches this).

- AC-6: The contract loader gains a `[per_arch_diagnostics_config]` section in `workloads/simbench_bw/comparison_contract.toml` that pins the diagnostic vocabulary + quartile threshold for drift detection. `ComparisonContract` dataclass gains a `per_arch_diagnostics_config` sub-config field; the loader parses it and validates set-equality between the contract's `diagnostic_vocabulary` (list of 3 strings) and the runner's `_EXCLUDED_DIAGNOSTIC_VALUES` constant; `quartile_min_n` is pinned to 4. Drift produces a typed contract loader exception (NOT a silent fallback).
  - Positive Tests:
    - The pinned section parses to `ComparisonContract.per_arch_diagnostics_config`; `set(contract.per_arch_diagnostics_config.diagnostic_vocabulary) == _EXCLUDED_DIAGNOSTIC_VALUES`.
    - `contract.per_arch_diagnostics_config.quartile_min_n == 4`.
  - Negative Tests:
    - A contract with a drifted vocabulary (e.g., missing `predictor_typed_unavailable`) produces a typed loader exception; the runner surfaces this as `comparison_contract_drift` on subsequent invocations.
    - A contract with `quartile_min_n != 4` produces a typed loader exception.
    - A contract missing the section produces a typed loader exception (after this round; the section is required in lower bound).
    - A section with an extra unknown key produces a typed loader exception.

- AC-7: `per_arch_diagnostics` serialization is byte-deterministic. Same inputs and same `bootstrap_seed` produces byte-identical output files when written via `write_bundle()` (which already uses `json.dump(..., sort_keys=True)`). `per_arch_diagnostics.per_arch` keys are alphabetically sorted (e.g., `sm_120` before `sm_86`); per-arch nested keys are alphabetically sorted; per-arch and unattributed `excluded_reason_counts` entries are alphabetically sorted by token.
  - Positive Tests:
    - Two consecutive runs with the same fixture + seed produce byte-identical files via `diff` comparison.
    - An ordering-flip test (e.g., dict-iteration relying on Python hash randomization) is detected by a regression test that flips dict insertion order and confirms identical output.
  - Negative Tests:
    - A non-deterministic dict iteration introduced into `_per_arch_diagnostics(...)` (e.g., a `set()` walk without `sorted()`) is detected by the byte-identical test.

- AC-8: Cross-arch fixture test surface lives at `workloads/simbench_bw/tests/test_cross_arch_comparison_harness.py` and exercises both arches end-to-end:
  - Positive Test 1 (real sm_86 round-trip): 5 real sm_86 cells from on-disk `data/local_labels/label_admitted.json` (post-AC-10 migration to schema_version=3 + `evidence_kind: live`) joined to 5 synthesized matching predictor bundles built via a new fixture-builder helper that reads on-disk join keys. `per_arch_diagnostics.per_arch.sm_86`: `n_comparable_cells == 5`, `n_headline_cells == 5`, `statistics_status == "quartile_emitted"`, `coverage_gate_verdict.headline_count_live == 5`, `coverage_gate_verdict.meets_floor == true`.
  - Positive Test 2 (synthetic sm_120 wire-format guard): 5 synthetic sm_120 cells declared `evidence_kind: synthetic` + 5 matching synthetic bundles. `per_arch_diagnostics.per_arch.sm_120`: `n_comparable_cells == 5`, `n_headline_cells == 5`, `statistics_status == "quartile_emitted"`, `coverage_gate_verdict.headline_count_live == 0`, `coverage_gate_verdict.headline_count_synthetic == 5`, `coverage_gate_verdict.meets_floor == false`.
  - Negative Test 1 (small-n via mismatch): a fixture with 4 sm_86 cells (NOT 5) + 4 predictor bundles where ONE bundle has `binary_sha256` differing by 1 hex character produces 1 entry in `per_arch_diagnostics.per_arch.sm_86.excluded_reason_counts.comparison_runner_join_key_unmatched`; `n_comparable_cells == 4`, `n_headline_cells == 3`; `statistics_status == "insufficient_cells_for_quartile_statistics"`; `coverage_gate_verdict.meets_floor == false`.
  - Negative Test 2 (no cells): a fixture with 0 sm_120 cells produces `n_comparable_cells == 0`, `n_headline_cells == 0`, `statistics_status == "no_cells"`, `relative_error_stats == null`.
  - Negative Test 3 (unattributed bucket): a fixture with 1 predictor bundle that lacks `manifest_pairing_index` entirely (so the runner emits `comparison_contract_drift` before resolving join_key) produces 1 entry in `per_arch_diagnostics.unattributed_excluded_reason_counts.comparison_contract_drift`; per-arch records are unaffected.
  - Negative Test 4 (schema-level): a label record missing `source_commit` fails strict label-admitted schema validation BEFORE reaching the runner (mirrors the existing `test_synthetic_v2_strict_admitted_profiler_version_drift_routes_to_contract_drift` pattern).

- AC-9: Test-first commit ordering and hygiene-gate compliance per slice (`BL-20260509-strict-admission-contract-first` discipline):
  - For each AC slice (AC-1 through AC-8, AC-10, AC-11), the failing-test commit precedes the implementation commit (verifiable via `git log` SHA ordering).
  - `tools/test_plan_token_scanner.py` returns `23 passed` after every commit; this scanner's regex covers the verbatim plan-token vocabulary (e.g., `AC-\d+`, `Milestone`, `Phase`, `Step`, `Round \d+`, `M2 plan`).
  - Widened workflow-marker grep `grep -RInE 'DEC-|[Pp]lan line|M1 plan|M2 plan|upper-bound|lower-bound scope' tools/local_labels/ workloads/simbench_bw/` returns no matches; this grep covers additional vocabulary not in the regex scanner. Together both gates cover the full plan-token vocabulary; neither is claimed to cover the full set on its own. The genuinely-domain `[hit_rate_lower, hit_rate_upper]` exception applies as documented in the prior label-stabilization plan.
  - Positive Tests: `git log` shows test commit SHA before implementation commit SHA per slice; both gates green after every commit.
  - Negative Tests: an implementation commit landing without a preceding test commit is detected by a `git log` audit; a plan-token marker introduced into any code/test file is detected by either the regex scanner or the widened grep.

- AC-10: `evidence_kind` field on the local-label admitted-cell schema (DEC-1 RESOLVED to live-only floor):
  - `workloads/simbench_bw/schemas/local_label_admitted.schema.json` `schema_version` is bumped from 2 to 3; new required field `evidence_kind` is added with enum `["live", "synthetic"]`; `additionalProperties: false` discipline preserved.
  - `data/local_labels/label_admitted.json` is migrated to `schema_version: 3`; each of the 5 already-admitted sm_86 cells is updated with `evidence_kind: "live"` (manual migration committed in this round).
  - `tools/local_labels/repeatability_aggregator.py` is updated to emit `evidence_kind: "live"` on every admitted cell record (the aggregator only runs on actual NCU sweep output, so all aggregator-emitted cells are `live` by construction).
  - Synthetic test fixtures (e.g., `_make_synthetic_ncu_label_cell` in `workloads/simbench_bw/tests/conftest_cache_predictor_comparison.py`) are updated to emit `evidence_kind: "synthetic"` by default; tests that exercise the live-evidence path explicitly override to `evidence_kind: "live"`.
  - Positive Tests:
    - `schema_version=3` admitted-cell record with `evidence_kind: "live"` validates.
    - `schema_version=3` admitted-cell record with `evidence_kind: "synthetic"` validates.
    - The 5 migrated cells in `data/local_labels/label_admitted.json` validate against the v3 schema.
  - Negative Tests:
    - `schema_version=2` (pre-migration) record fails strict v3 schema validation.
    - `evidence_kind` outside the enum (e.g., `"unknown"`) is rejected.
    - Missing `evidence_kind` is rejected (required field).

- AC-11: Aggregate test suite regression preservation. The aggregate suite `pytest tools/local_labels/ workloads/simbench_bw/tests/ tools/test_plan_token_scanner.py -q` returns the same passing count as the pre-merge baseline (currently `1329 passed, 2 skipped, 170 subtests`). Synthetic fixtures across the test corpus are updated to declare `evidence_kind` per AC-10; the existing `test_synthetic_v2_strict_admitted_label_file_runs_to_admitted_result_sm120` test is split into two tests:
  - `test_synthetic_v2_strict_admitted_label_file_with_live_evidence_admits_sm120`: cells declared `evidence_kind: "live"`; result is `admitted` (preserves the wire-format guard semantics).
  - `test_synthetic_v2_strict_admitted_label_file_with_synthetic_evidence_routes_to_typed_unavailable_sm120`: cells declared `evidence_kind: "synthetic"`; result is `typed_unavailable` (live-only floor enforcement; no synthetic-fixture admission).
  - Positive Tests:
    - Aggregate suite returns the same passing count as the pre-merge baseline; no new failures.
    - The two split tests both pass and assert their respective `result_kind`.
    - All other synthetic-fixture-using tests pass after their fixtures gain `evidence_kind`.
  - Negative Tests:
    - A test that omits `evidence_kind` from a synthetic fixture fails the strict v3 schema validation (regression guard against fixture drift).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation includes AC-1 through AC-11 closed; per-arch diagnostics object covers every arch in `contract.arch_set` deterministically (including zero-count arches with `no_cells` status); contract amendment `[per_arch_diagnostics_config]` section is added with pinned vocabulary + threshold; the `evidence_kind` schema field is added with `live` and `synthetic` enum values; the 5 sm_86 admitted cells are migrated to schema_version=3; the synthetic-sm_120 wire-format-guard test is split into live and synthetic variants; cross-arch fixture test surface includes both real-sm_86 round-trip + synthetic-sm_120 wire-format guard with `meets_floor: false`; the existing prior-plan plan-token scanner + widened workflow-marker grep stay clean; the aggregate test suite returns the same passing count as the pre-merge baseline; signed-residual statistics (in addition to absolute relative_error) are queued for a follow-on round; bounded-arm `relative_error_lower/upper` aggregation is queued for a follow-on round; per-arch bootstrap CI is queued for a follow-on round.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes AC-1 through AC-11 closed (all 11 acceptance criteria are in lower bound; DEC-1's live-only-floor resolution makes AC-10 + AC-11 mandatory because the schema migration must land lockstep with the runner's gate change). The per-arch diagnostics object covers `contract.arch_set`'s two arches (`sm_86`, `sm_120`); cross-arch fixture test exercises 5 real sm_86 + 5 synthetic sm_120 cells; the existing 1329-test suite returns the same passing count as the pre-merge baseline.

### Allowed Choices

- Can use:
  - Python stdlib `statistics` module for mean / stdev / quantiles (`statistics.stdev` for sample stddev with n>=2; pin `0.0` for n=1; `statistics.quantiles(data, n=4, method="exclusive")` for quartiles with n>=4).
  - The existing `comparison_runner.py` `_bootstrap_ci_over_cells(...)` function for an OPTIONAL upper-bound per-arch CI extension (NOT required for lower-bound closure).
  - The existing `conftest_cache_predictor_comparison.py` synthetic fixture builders (`_make_synthetic_admitted_predictor_bundle`, `_make_synthetic_ncu_label_cell`, `_make_synthetic_manifest_pairing_index`) extended with explicit `arch` / `device` / `evidence_kind` overrides.
  - `jsonschema` for strict schema validation in tests (already a project dependency).
  - `tomllib` (Python 3.11+) for contract loader extension (`[per_arch_diagnostics_config]` section parsing).
  - The existing `tools/local_labels/repeatability_aggregator.py` admitted-cell emission path as the model for adding `evidence_kind: "live"`.
- Cannot use:
  - Any NCU invocation outside `tools/local_labels/` (CLAUDE.md Critical Rule 1).
  - Any GPU-driver-root operation (Critical Rule 2).
  - Any privileged CUPTI surface (Critical Rule 3).
  - Any host-driver work (Critical Rule 4).
  - Any simulator revival (Critical Rule 5).
  - Plan-token markers in implementation code or comments — covered by AC-9.
  - Loosening `stddev_threshold = 0.03` in `tools/local_labels/repeatability_aggregator.py` (the existing static-source-scan test enforces this).
  - Widening or reordering the 7-field join key in `comparison_contract.toml` (out of scope; the M1 contract is binding).
  - Bumping `comparison_bundle.schema.json` `schema_version` (the `per_arch_diagnostics` field is OPTIONAL additive; existing bundles without it must continue to validate). The `local_label_admitted.schema.json` schema_version bump (2 → 3) is the ONLY schema bump this round; it is mandated by AC-10.
  - Adding a silent TOML config (the `[per_arch_diagnostics_config]` section MUST land with a real `ComparisonContract` sub-config + loader; or the section MUST be omitted entirely; the existing anti-pattern of `require_both_arches_for_cross_arch` being silent is NOT to be perpetuated for new additions).
  - Wiring `require_both_arches_for_cross_arch` into the runner's admission gate (queued for a follow-on plan; DEC-7 RESOLVED to defer).
  - Edits to plan files (BL-20260510-plan-file-frozen-during-active-rlcr-session lesson; this plan is the new tracked plan and is not the active session's plan when the loop starts).

> **Note on Deterministic Designs**: The DEC-1 resolution to live-only floor makes AC-5 + AC-10 + AC-11 deterministic — the schema MUST gain `evidence_kind`; the gate MUST count only `live` cells; the existing 5 admitted cells MUST be migrated; the synthetic-sm_120 test MUST be split. There is no choice axis on these. Choice axes remain on the per-arch statistics layer (statistics_status enum encoding, quartile method, stdev for n=1).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The natural per-slice ladder, derived from the test-first discipline:

1. **AC-10 evidence_kind schema slice**: failing tests committed first asserting the v3 schema validates `live` and `synthetic` enum values + rejects pre-migration v2 records + rejects records missing `evidence_kind`. Implementation lands the `local_label_admitted.schema.json` bump 2 → 3; migrates the 5 cells in `data/local_labels/label_admitted.json` to v3 + `evidence_kind: "live"`; updates `repeatability_aggregator.py` to emit `evidence_kind: "live"` on every admitted record.

2. **AC-11 fixture migration slice**: failing tests committed first asserting `_make_synthetic_ncu_label_cell` requires `evidence_kind`; the existing synthetic-sm_120 admitted test is split into the live-and-admitted variant + the synthetic-and-typed_unavailable variant. Implementation lands the conftest fixture builder updates + the test split + scope-walks the rest of the test corpus to add `evidence_kind: "synthetic"` to every synthetic-fixture call site that does not already specify it. Aggregate suite returns the same passing count.

3. **AC-1 + AC-4 schema + vocabulary slice**: failing tests committed first for the `per_arch_diagnostics` schema shape + `excluded_reason_counts` vocabulary lockstep. Implementation lands the `comparison_bundle.schema.json` amendment + the runner-side vocabulary check.

4. **AC-2 + AC-3 stats slice**: failing tests committed first for per-arch statistics independence + small-n behavior + stdev semantics + quartile method. Implementation lands `_per_arch_diagnostics(...)` with its three internal helpers.

5. **AC-5 verdict slice**: failing tests committed first for `coverage_gate_verdict` per-arch with `headline_count_live` / `headline_count_synthetic` / `meets_floor` semantics + top-level admitted gate change to live-only floor + the regression guard that synthetic-only sm_120 produces `typed_unavailable`. Implementation lands the gate change + the per-arch verdict computation.

6. **AC-6 contract amendment slice**: failing tests committed first for `[per_arch_diagnostics_config]` section parse + drift detection. Implementation lands `ComparisonContract` sub-config + loader + drift exceptions.

7. **AC-7 determinism slice**: failing test committed first for byte-identical-re-run with fixed seed + alphabetical sorting of nested keys. Implementation confirms `sort_keys=True` path; adds deterministic-ordering helpers if needed.

8. **AC-8 cross-arch fixture slice**: failing tests committed first for the real-sm_86 round-trip (5 cells from `data/local_labels/label_admitted.json` + matching predictor bundles built via the new fixture-builder helper) + synthetic-sm_120 (5 cells with `evidence_kind: "synthetic"`) + small-n + unattributed-bucket + schema-level negative tests. Implementation lands `workloads/simbench_bw/tests/test_cross_arch_comparison_harness.py` + the fixture-builder helper.

9. **AC-9 hygiene slice**: continuous (every commit). Plan-token scanner + widened workflow-marker grep are post-commit gates per `BL-20260508-post-commit-recapture-discipline`.

The 9 slices land as ~22 commits (11 test-first + 11 impl).

### Relevant References

- `workloads/simbench_bw/comparison_runner.py` — runner; key surfaces: `_RUNNER_DEFECT_TOKENS` (lines 63-65), `_PREDICTOR_TYPED_UNAVAILABLE_DIAGNOSTIC` + `_EXCLUDED_DIAGNOSTIC_VALUES` (lines 134-138), `run_comparison()` join + filter + per-arch tally (lines 668-912), `comparable_per_arch` / `headline_per_arch` counter init + update (lines 750-751, 778-781, 817, 837), bundle assembly (lines 876-883), top-level admitted gate ("any arch meets floor"; lines 885-897), `write_bundle` with `sort_keys=True` (lines 915-922), bounded-arm path emitting canonical `relative_error` + `relative_error_lower/upper_diagnostic` (lines 818-836).
- `workloads/simbench_bw/comparison_contract.toml` — pinned contract; key sections: `[scope]` `arch_set = ["sm_86", "sm_120"]` (lines 132-137), `[coverage_gate]` `smoke_test_floor_per_arch` (line 118), `require_both_arches_for_cross_arch` silent field (line 121, queued for DEC-7 follow-on).
- `workloads/simbench_bw/schemas/comparison_bundle.schema.json` — `additionalProperties: false` discipline; `included_cells` / `excluded_cells` per-cell record schemas; `relative_error_lower_diagnostic` / `relative_error_upper_diagnostic` precedent (lines 118-136); singular `geomean` / `geomean_ci_*` (lines 163-181).
- `workloads/simbench_bw/schemas/local_label_admitted.schema.json` — admitted-cell schema; current schema_version=2; AC-10 bumps to 3 + adds `evidence_kind`.
- `workloads/simbench_bw/tests/test_comparison_runner.py` — 44 existing tests; synthetic-sm_120 wire-format guard test (lines 1280-1366); `_RUNNER_DEFECT_TOKENS` / `_EXCLUDED_DIAGNOSTIC_VALUES` bidirectional set-equality tests (lines 56-75, 134-139). The synthetic-sm_120 test is split per AC-11.
- `workloads/simbench_bw/tests/conftest_cache_predictor_comparison.py` — synthetic fixture builders; AC-10 + AC-11 update them to require `evidence_kind` per cell.
- `data/local_labels/label_admitted.json` — 5 sm_86 admitted cells; AC-10 migrates to schema_version=3 + `evidence_kind: "live"`.
- `tools/local_labels/repeatability_aggregator.py` — admitted-cell emitter; AC-10 updates to emit `evidence_kind: "live"` on every record.
- `tools/test_plan_token_scanner.py` — the 23-passing scanner gate enforced by AC-9.
- `.humanize/bitlesson.md`:
  - `BL-20260507-arch-token-multi-surface-lockstep`: the schema/runner enum lockstep pattern AC-6 mirrors.
  - `BL-20260509-strict-admission-contract-first`: test-first per slice in AC-9.
  - `BL-20260509-hygiene-validation-grep-must-be-case-insensitive`: the widened workflow-marker grep in AC-9.
  - `BL-20260508-post-commit-recapture-discipline`: post-commit gate validation in AC-9.
  - `BL-20260510-plan-file-frozen-during-active-rlcr-session`: the procedural lesson from the prior loop session; this plan is a new tracked plan and the active session's plan-modification-stop-hook applies if this plan is wired into a future RLCR loop.

## Dependencies and Sequence

### Milestones

1. **M-A: Schema Migration + Fixture Lift** (AC-10, AC-11)
   - Phase A1: `local_label_admitted.schema.json` bump 2 → 3 + `evidence_kind` field; migration of `data/local_labels/label_admitted.json` 5 cells; `repeatability_aggregator.py` emission update.
   - Phase A2: synthetic-fixture conftest update + corpus-wide migration of synthetic fixtures + split of the existing synthetic-sm_120 admitted test into live and synthetic variants; aggregate suite returns same passing count.

2. **M-B: Per-Arch Diagnostics Schema + Vocabulary** (AC-1, AC-4)
   - Phase B1: `comparison_bundle.schema.json` amendment + schema-validation tests.
   - Phase B2: `excluded_reason_counts` vocabulary lockstep tests + runner-side computation.

3. **M-C: Per-Arch Statistics + Verdict** (AC-2, AC-3, AC-5)
   - Phase C1: `_per_arch_diagnostics()` + `_per_arch_stats()` + `_per_arch_excluded_counts()` + `_unattributed_excluded_counts()` impl + small-n behavior + n=1 stdev pin + quartile method pin.
   - Phase C2: `coverage_gate_verdict` per-arch + top-level admitted gate change to live-only floor + regression guard for synthetic-only fixture.

4. **M-D: Contract Amendment + Determinism** (AC-6, AC-7)
   - Phase D1: `[per_arch_diagnostics_config]` section + `ComparisonContract` sub-config + loader.
   - Phase D2: byte-identical-re-run test.

5. **M-E: Cross-Arch Harness + Hygiene** (AC-8, AC-9)
   - Phase E1: `test_cross_arch_comparison_harness.py` with all positive + negative tests + new fixture-builder helper.
   - Phase E2: plan-token scanner + widened workflow-marker grep gates per commit (continuous).

### Dependency Graph

- AC-10 has no in-loop dependencies; lands first (M-A is the foundation).
- AC-11 depends on AC-10 (fixture migration follows the schema bump).
- AC-1 depends on AC-10 + AC-11 (the `per_arch_diagnostics` schema reads the `evidence_kind`-aware bundle output).
- AC-2 / AC-3 / AC-4 depend on AC-1 (the schema is the contract pin every per-arch field validates against).
- AC-5 depends on AC-10 (top-level gate uses `evidence_kind`-classified counts) and AC-2/AC-3/AC-4.
- AC-6 depends on AC-1 + AC-4 (the contract lockstep pins the runner's emitted shape).
- AC-7 depends on AC-2 / AC-3 / AC-4 / AC-5 (the determinism test exercises the populated bundle).
- AC-8 depends on AC-1 through AC-5 + AC-10 + AC-11 (the test fixture exercises every populated field on both arches).
- AC-9 is continuous (every commit).

There are no out-of-loop tooling dependencies. The cross-host transfer is a project-discipline dependency (HANDOFF.md + CLAUDE.md "Session Handoff"), NOT a tooling dependency for this plan.

## Task Breakdown

Each task carries exactly one routing tag.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1a | Failing tests for AC-10: v3 schema validates `live` and `synthetic`; rejects pre-migration v2 records; rejects records missing `evidence_kind`. | AC-10, AC-9 | coding | - |
| task1b | Implementation: bump `local_label_admitted.schema.json` schema_version 2 → 3 + add `evidence_kind` enum field; migrate the 5 cells in `data/local_labels/label_admitted.json` to schema_version=3 + `evidence_kind: "live"`; update `tools/local_labels/repeatability_aggregator.py` to emit `evidence_kind: "live"` on every admitted record. | AC-10 | coding | task1a |
| task2a | Failing tests for AC-11: synthetic fixtures across the test corpus require `evidence_kind`; the existing synthetic-sm_120 admitted test is split into live-and-admitted + synthetic-and-typed_unavailable variants. | AC-11, AC-9 | coding | task1b |
| task2b | Implementation: update `conftest_cache_predictor_comparison.py` synthetic fixture builders to require `evidence_kind`; corpus-walk every test using the synthetic fixtures and add `evidence_kind: "synthetic"` (or `evidence_kind: "live"` where appropriate); split the named test; confirm aggregate suite returns the same passing count. | AC-11 | coding | task2a |
| task3a | Failing tests for AC-1: schema with `per_arch_diagnostics` object validates positive cases; rejects unknown / missing / out-of-enum fields; oneOf branches bind `statistics_status` const. | AC-1, AC-9 | coding | task2b |
| task3b | Implementation: amend `workloads/simbench_bw/schemas/comparison_bundle.schema.json` to add OPTIONAL `per_arch_diagnostics` wrapper object with strict `additionalProperties: false`; oneOf branches per `statistics_status`. | AC-1 | coding | task3a |
| task4a | Failing tests for AC-4: `excluded_reason_counts` vocabulary lockstep against `_EXCLUDED_DIAGNOSTIC_VALUES`; per-arch + unattributed buckets; zero defaults emitted. | AC-4, AC-9 | coding | task3b |
| task4b | Implementation: `_per_arch_excluded_counts()` + `_unattributed_excluded_counts()` helpers tally per-arch exclusion records using the three-token vocabulary. | AC-4 | coding | task4a |
| task5a | Failing tests for AC-2 + AC-3: per-arch stats independence; n_comparable_cells / n_headline_cells formulas; small-n behavior; n=1 stdev pinned to 0.0; quartile method pinned to `statistics.quantiles(..., method="exclusive")`. | AC-2, AC-3, AC-9 | coding | task4b |
| task5b | Implementation: `_per_arch_diagnostics()` + `_per_arch_stats()` helpers compute mean / stdev / min / max / quartiles with the `statistics_status` enum gate; n=1 special case; oneOf-aligned shape. | AC-2, AC-3 | coding | task5a |
| task6a | Failing tests for AC-5: `coverage_gate_verdict` per-arch with `headline_count_live` / `headline_count_synthetic` / `meets_floor` semantics; top-level admitted gate uses live-only floor; synthetic-only sm_120 produces `typed_unavailable`. | AC-5, AC-9 | coding | task5b |
| task6b | Implementation: per-arch `coverage_gate_verdict` block in `run_comparison()` output; top-level admitted gate changed to `any(headline_count_live[a] >= floor for a in arch_set)`. | AC-5 | coding | task6a |
| task7a | Failing tests for AC-6: `[per_arch_diagnostics_config]` section parse + drift detection; set-equality between contract and runner enums; `quartile_min_n` pinned to 4. | AC-6, AC-9 | coding | task6b |
| task7b | Implementation: `ComparisonContract` sub-config field + loader extension for `[per_arch_diagnostics_config]`; drift exceptions surface as `comparison_contract_drift` on subsequent invocations. | AC-6 | coding | task7a |
| task8a | Failing tests for AC-7: byte-identical-re-run with fixed seed; `per_arch_diagnostics.per_arch` keys alphabetically sorted; nested keys alphabetically sorted; `excluded_reason_counts` entries alphabetically sorted; `unattributed_excluded_reason_counts` alphabetically sorted. | AC-7, AC-9 | coding | task7b |
| task8b | Implementation: confirm `sort_keys=True` path; add deterministic-ordering helpers for nested set walks if needed. | AC-7 | coding | task8a |
| task9a | Failing tests for AC-8: real-sm_86 round-trip (5 cells from on-disk `label_admitted.json` + matching predictor bundles) + synthetic-sm_120 (5 cells `evidence_kind: "synthetic"`); per-arch diagnostics covers both arches; small-n + unattributed-bucket + schema-level negative tests. | AC-8, AC-9 | coding | task8b |
| task9b | Implementation: `workloads/simbench_bw/tests/test_cross_arch_comparison_harness.py` + new fixture-builder helper that synthesizes predictor bundles directly from on-disk `label_admitted.json` join keys. | AC-8 | coding | task9a |

## Claude-Codex Deliberation

### Agreements

- `comparison_runner.py` already partitions per-arch via `comparable_per_arch` / `headline_per_arch` dicts; the `per_arch_diagnostics` extension is a mechanical bookkeeping addition.
- `test_comparison_runner.py` and `conftest_cache_predictor_comparison.py` provide the synthetic fixture infrastructure; the cross-arch harness test (AC-8) reuses them.
- `write_bundle()`'s `sort_keys=True` establishes byte-determinism at the runner level; AC-7 is mostly a regression guard.
- `_EXCLUDED_DIAGNOSTIC_VALUES` (3 tokens, including `predictor_typed_unavailable`) is the correct vocabulary for `excluded_cells[].diagnostic`, not `_RUNNER_DEFECT_TOKENS` (2 tokens, runner-defect-only). Documented at `comparison_runner.py` lines 134-138.
- The `[per_arch_diagnostics_config]` section MUST land with a real `ComparisonContract` sub-config + loader; or it MUST be omitted. The existing anti-pattern of `require_both_arches_for_cross_arch` (silent in TOML, never loaded) is NOT to be perpetuated.
- DEC-1 resolution to live-only floor counting is enforced by adding `evidence_kind` to the admitted-cell schema; this is a substantive but bounded scope expansion.

### Resolved Disagreements

- Excluded-reason vocabulary: candidate plan v1 named `_RUNNER_DEFECT_TOKENS`; corrected to `_EXCLUDED_DIAGNOSTIC_VALUES`. Resolution: AC-4 names the three-token enum.
- Synthetic-vs-live evidence semantics: the user resolved DEC-1 to live-only floor (Option B). Resolution: AC-10 adds `evidence_kind` to the schema; AC-5 uses live-only floor; AC-11 splits the synthetic-sm_120 test into live and synthetic variants.
- Schema_version bump strategy: `comparison_bundle.schema.json` does NOT bump (per_arch_diagnostics is OPTIONAL additive); `local_label_admitted.schema.json` DOES bump (2 → 3) per AC-10's evidence_kind addition. Two distinct schemas; one is additive, one is breaking.
- Quartile small-n: canonical resolution is `statistics_status` enum with three values + omission of quartile fields below threshold. Schema oneOf branches bind `statistics_status` via `const`.
- n=1 stdev: pinned to `0.0` (special case); `statistics.stdev()` raises for n=1, so the helper handles this with an explicit branch.
- Quartile method: pinned to `statistics.quantiles(data, n=4, method="exclusive")` (Python default, Tukey hinges).
- Bounded-arm aggregation: aggregate canonical `relative_error` only this round; bounded-arm `relative_error_lower/upper_diagnostic` aggregation is queued for a follow-on plan.
- Unattributed exclusion bucket location: pinned to `per_arch_diagnostics.unattributed_excluded_reason_counts` (inside the wrapper object, NOT at the bundle's top level).
- `require_both_arches_for_cross_arch` silent field: kept silent this round; cross-arch admission gate semantics are queued for a follow-on plan.
- AC-8 negative-case arithmetic: corrected to start with 4 sm_86 cells + 1 mismatch = 3 remaining → `insufficient_cells_for_quartile_statistics` (consistent with AC-3's small-n branch).
- Hygiene scanner overclaim: AC-9 narrowed to cite the regex scanner + widened grep separately; together both gates cover the full plan-token vocabulary.

### Convergence Status

- Final Status: `converged`. Three convergence rounds executed (Codex first-pass + Round 1 + Round 2 + Round 3). Round 3 returned no `REQUIRED_CHANGES` and no `DISAGREE`; one OPTIONAL_IMPROVEMENT (AC-2 wording alignment) was folded into the final plan silently. DEC-1 was resolved via `AskUserQuestion` to Option B (live-only floor, strict); DEC-2 / DEC-3 / DEC-4 / DEC-5 / DEC-6 / DEC-7 were resolved during convergence. No `PENDING` items remain.

## Pending User Decisions

- DEC-1: Should synthetic sm_120 cells count toward `coverage_gate_verdict.meets_floor` in `per_arch_diagnostics`, or should only LIVE local-label evidence count?
  - Claude Position: count-based only (Option A) — bounded scope; preserves existing tests; no schema bump.
  - Codex Position: live-only (Option B) — stricter; required schema bump and broader test corpus migration.
  - Tradeoff Summary: Option A preserves backward compatibility but lets fixture-only fixtures appear admitted; Option B is mechanically rigorous but breaks the existing `schema_version=2` admitted-cell records and forces every synthetic-fixture call site to declare `evidence_kind`.
  - Decision Status: RESOLVED (user chose Option B / Live-only floor). AC-5, AC-10, AC-11 reflect this resolution.

- DEC-2: Should `per_arch_diagnostics` be required or optional in `comparison_bundle.schema.json`?
  - Claude Position: optional / additive (no schema_version bump).
  - Codex Position: agrees optional is safer.
  - Tradeoff Summary: optional preserves backward compatibility; required forces all consumers to update.
  - Decision Status: RESOLVED (optional / additive).

- DEC-3: Should every arch in `contract.arch_set` appear in `per_arch_diagnostics` with zero-count records, or only arches with at least one comparable cell?
  - Claude Position: every arch always present (deterministic cross-arch readiness).
  - Codex Position: agrees every-arch-always.
  - Tradeoff Summary: every-arch-always is deterministic; only-non-empty has smaller bundles but non-deterministic shape.
  - Decision Status: RESOLVED (every arch in `contract.arch_set` always present).

- DEC-4: Should `per_arch_diagnostics` include signed residual statistics (`predicted - ground_truth`) in addition to absolute `relative_error`?
  - Claude Position: absolute `relative_error` only this round; signed residuals queued.
  - Codex Position: agrees deferring signed residuals is a clean scope split.
  - Tradeoff Summary: relative_error only is simpler; both is denser but doubles the schema surface.
  - Decision Status: RESOLVED (absolute `relative_error` only; signed residuals queued for a follow-on plan).

- DEC-5: Should quartile fields be omitted, null, or present-with-status when `n_headline_cells < 4`?
  - Claude Position: omitted with `statistics_status` carrying the explanation; schema uses oneOf branches to enforce omission.
  - Codex Position: agrees omission with status flag is clean.
  - Tradeoff Summary: omit is strict; null is uniform; present-with-status is denser.
  - Decision Status: RESOLVED (omit-with-status; oneOf branches with const-binding `statistics_status`).

- DEC-6: Should the contract loader gain a `[per_arch_diagnostics_config]` section pinning the diagnostic vocabulary + quartile threshold for drift detection?
  - Claude Position: yes (lower bound). Adds explicit lockstep between contract and runner; mirrors `BL-20260507-arch-token-multi-surface-lockstep`.
  - Codex Position: yes IF the section is loaded; do not add a silent section (anti-pattern of `require_both_arches_for_cross_arch`).
  - Tradeoff Summary: yes is stricter lockstep; no is simpler scope.
  - Decision Status: RESOLVED (yes; add `[per_arch_diagnostics_config]` + `ComparisonContract` sub-config + loader; in lower bound).

- DEC-7: Should `require_both_arches_for_cross_arch` (currently silent in TOML) be wired into the runner's admitted-gate logic this round, or kept as a separate follow-on?
  - Claude Position: separate follow-on. This round preserves the existing "any arch meets floor" top-level gate (modified to count only live cells per AC-5); cross-arch-strictness semantics are queued.
  - Codex Position: agrees the silent field is anti-pattern but acknowledges this is bounded-scope work for a separate plan.
  - Tradeoff Summary: defer is bounded scope; wire-in is stricter cross-arch admission.
  - Decision Status: RESOLVED (defer to a separate follow-on plan; the silent field stays in TOML but unused by the runner this round).

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", "Round N", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.
- The widened workflow-marker grep `grep -RInE 'DEC-|[Pp]lan line|M1 plan|M2 plan|upper-bound|lower-bound scope' tools/local_labels/ workloads/simbench_bw/` MUST return no matches in implementation surfaces (subject to the genuinely-domain `[hit_rate_lower, hit_rate_upper]` exception).
- The `tools/test_plan_token_scanner.py` 23-passing scanner gate enforces the regex-based vocabulary (`AC-\d+`, `Milestone`, `Phase`, `Step`, `Round \d+`, `M2 plan`).
- Together both gates cover the full plan-token vocabulary; both must be green after every commit.

--- Original Design Draft Start ---

The full draft is preserved at `.humanize/ideas/draft-a-new-idea-to-advance-the-cache-pr-20260510-134608.md`. The plan above incorporates the draft's full content (primary direction: cross-arch predictor-vs-label harness; per-arch statistical aggregation extending `comparison_runner.py`'s already-arch-aware bookkeeping; cross-arch fixture exercise on the 3090 Ti against the 5 sm_86 admitted cells + a synthetic 5-cell sm_120 fixture; ready to render `Cross-arch-verified` verdicts when AC-5.2 sm_120 evidence eventually lands) plus the Codex first-pass critique adjustments (vocabulary correction from `_RUNNER_DEFECT_TOKENS` to `_EXCLUDED_DIAGNOSTIC_VALUES`; explicit small-n and stdev semantics; oneOf schema branches binding `statistics_status` via const; unattributed-bucket location; deterministic ordering) plus the user's DEC-1 resolution to live-only floor (adding `evidence_kind: live | synthetic` to the admitted-cell schema; bumping `local_label_admitted.schema.json` schema_version 2 → 3; migrating the 5 admitted sm_86 cells; updating the runner's gate to count only live cells; splitting the synthetic-sm_120 admitted test).

--- Original Design Draft End ---
