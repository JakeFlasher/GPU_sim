# L2 Reduction-Family Hit-Rate Correction For Cache Predictor

## Goal Description

Add a post-gridDim residual class correction to the L2 cache hit-rate prediction in `workloads/simbench_bw/cache_predictor.py` targeting the L2 reduction-kernel family (`rmsnorm_fwd_bf16_sm80`, `softmax_safe_bf16_sm80`). The correction is a multiplicative scaling applied after the existing gridDim alpha penalty, mirroring the R46 L1 reduction-kernel correction architecturally but operating as an independent L2-head classifier (no piggybacking on the L1 reduction allowlist; the existing L2-isolation regression test forbids that coupling).

Two co-equal goals:

1. **Reduce hostile-cell L2 over-prediction**: `rmsnorm_fwd_bf16_sm80` (L2 rel_err 0.425) and `softmax_safe_bf16_sm80` (L2 rel_err 0.674) currently over-predict L2 hit-rate by 30-50%. The correction targets these two cells directly.
2. **Widen HARD-1 margin**: friendly_l2 currently 0.14558 (margin 0.004 below the 0.15 cap). The correction should preserve HARD-1 PASS and ideally widen the margin by improving these two cells (both belong to the friendly cohort).

Pointer-chase (`l2_thrash_pointer_chase`) and local-memory-spill (`register_spill_local_mem`) corrections are explicitly deferred to a follow-up round pending predictor-side structural evidence beyond slug matching — each currently has only one motivating cell, which would constitute target-fitting.

Constraints:
- HARD-1 (friendly_l2 < 0.15) preserved or improved.
- HARD-2 (friendly_l1) byte-identical (after deterministic JSON serialization, excluding provenance SHA fields).
- HARD-3 (cross_cell_stddev) allowed to move within `0.10 <= cross_cell_stddev <= 0.15`.
- Per-cell L2 regression budget: no admitted cell's rel_err may increase by more than 0.02 absolute.
- `ALGORITHM_VERSION` unchanged (the correction is a post-hoc multiplicative adjustment, not a model architectural change).
- L1 head byte-identical (bidirectional L1/L2 head isolation).
- No new bundle fields (internal-only correction).
- Plan-token discipline diff-scoped to new/modified lines.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: L2 reduction-family multiplicative correction applies to the closed allowlist `{rmsnorm_fwd_bf16_sm80, softmax_safe_bf16_sm80}` and only those slugs.
  - Positive Tests (expected to PASS):
    - For each allowlist slug, the bundle's L2 `predicted_hit_rate` equals `clamp(point_pre * factor, 0.0, 1.0)` where `point_pre` is the L2 point estimate after the existing gridDim alpha penalty and `factor` is the calibrated reduction-family correction factor.
    - `ci_low` and `ci_high` transformed by the same `clamp(x * factor, 0, 1)` function; `ci_half_width` recomputed from corrected `ci_low/ci_high`.
    - Allowlist-exact-match assertion: the L2 reduction allowlist frozenset equals `{rmsnorm_fwd_bf16_sm80, softmax_safe_bf16_sm80}` (prevents accidental expansion).
  - Negative Tests (expected to FAIL when criterion is violated):
    - `silu_fwd_bf16_sm80` (member of L1 reduction allowlist but L2-accurate) L2 bundle byte-identical with and without the new L2 correction code path active.
    - `layernorm_fwd_bf16_sm80` (also in L1 reduction allowlist) L2 bundle byte-identical.
    - A non-reduction slug (`working_set_capacity_sweep`) L2 bundle byte-identical.

- AC-2: This round must not land pointer-chase L2 correction surface.
  - Positive Tests:
    - Scanner verifies absence of any `_L2_POINTER_CHASE_*` identifier in this round's diff.
    - Scanner verifies absence of `l2_thrash_pointer_chase` as a key in any new L2 correction allowlist constant.
  - Negative Tests:
    - Any `_L2_POINTER_CHASE_*` identifier or allowlist entry naming `l2_thrash_pointer_chase` triggers scanner failure.

- AC-3: This round must not land local-memory-spill L2 correction surface.
  - Positive Tests:
    - Scanner verifies absence of any `_L2_LOCAL_MEM_SPILL_*` identifier in this round's diff.
    - Scanner verifies absence of `register_spill_local_mem` as a key in any new L2 correction allowlist constant.
  - Negative Tests:
    - Any `_L2_LOCAL_MEM_SPILL_*` identifier or allowlist entry naming `register_spill_local_mem` triggers scanner failure.

- AC-4: Hard-target preservation after the correction lands.
  - Positive Tests:
    - friendly_l2 geomean_rel_err < 0.15 (HARD-1).
    - friendly_l1 geomean_rel_err byte-identical to pre-correction value `0.153676764668927` (HARD-2), under deterministic serialization excluding provenance SHA fields.
    - `0.10 <= cross_cell_stddev <= 0.15` (HARD-3 lower and upper bound).
  - Negative Tests:
    - Any hard-target verdict failure aborts admission via the existing closure-audit gate.

- AC-5: Per-cell L2 regression budget across every admitted cell.
  - Positive Tests:
    - For every admitted L2 cell C, `rel_err_after(C) - rel_err_before(C) <= 0.02` where `rel_err` is computed via the existing comparison-runner contract formula and epsilon (no new metric definition).
    - Shadow report enumerates each cell with columns `(slug, class_matched, factor_applied, point_pre, point_post, truth, rel_err_before, rel_err_after, delta, budget_pass)` sorted descending by `delta`.
  - Negative Tests:
    - Any cell with `delta > 0.02` triggers deterministic test failure naming the cell.

- AC-6: Bidirectional L1/L2 head isolation.
  - Positive Tests:
    - The existing `test_bundle_l2_byte_identical_regardless_of_l1_correction` passes unchanged (L1 reduction correction does not mutate L2 bundles).
    - New `test_bundle_l1_byte_identical_regardless_of_l2_correction` asserts: for an L1-reduction-allowlist slug, the L1 bundle is byte-identical whether the new L2 correction code path is active or inert (L2 correction does not mutate L1 bundles).
  - Negative Tests:
    - Any L1 bundle mutation triggered by the L2 correction code path triggers failure.

- AC-7: Correction order, CI handling, and `ALGORITHM_VERSION` consistency.
  - Positive Tests:
    - gridDim alpha penalty applies FIRST (existing flow unchanged); L2 class correction applies AFTER as multiplicative scaling.
    - `ci_low` and `ci_high` transformed by identical `clamp(x * factor, 0, 1)`; `ci_half_width` recomputed from corrected bounds.
    - `ALGORITHM_VERSION` value unchanged from pre-correction state (asserted by a dedicated test comparing the constant's pre/post value).
  - Negative Tests:
    - Order swap (class correction before gridDim) triggers an explicit ordering assertion failure.
    - Missing CI transformation triggers failure.
    - Accidental `ALGORITHM_VERSION` bump triggers failure.

- AC-8: Schema integrity (no new bundle fields) and byte-identity scope.
  - Positive Tests:
    - Bundle JSON schema validation passes for all emitted bundles post-correction.
    - For unaffected cells (any cell NOT in the L2 reduction allowlist), bundles are byte-identical after deterministic JSON serialization, excluding `manifest_pairing_index.source_commit` and `predictor_build_commit` (which move per BL-20260508 recapture flow).
  - Negative Tests:
    - Any unexpected new field in emitted bundles triggers schema validation failure.
    - Any byte-difference in an unaffected cell's bundle outside the excluded provenance SHA fields triggers test failure.

- AC-9: Plan-token hygiene, diff-scoped to new/modified lines.
  - Positive Tests:
    - Diff-scoped scanner verifies no NEW or MODIFIED line in this round introduces literals matching `AC-\d`, `Milestone`, `Phase\s+\d`, `Step\s+\d` outside the static guard's existing exempt list.
    - Pre-existing debt in unchanged files is out of scope.
  - Negative Tests:
    - Any new/modified line containing such a literal triggers scanner failure naming file and line.

- AC-10: Post-commit recapture of tracked artifacts (BL-20260508).
  - Positive Tests:
    - After each fix-commit X that mutates predictor logic, the recapture script regenerates tracked artifacts under `data/` so that each artifact's `source_commit` field references commit X (the code-commit that GENERATED the artifact, NOT the artifact-containing chore-recapture commit, which cannot reference its own SHA).
    - Regenerated artifacts are committed in a separate chore-recapture commit per BL-20260508.
    - Validation compares each tracked artifact's `source_commit` to the most recent predictor-touching fix-commit and asserts they match.
  - Negative Tests:
    - Any tracked artifact whose `source_commit` is stale (references an earlier commit than the most recent predictor-touching fix-commit) triggers test failure.

- AC-11: Coverage of both L2 prediction entry points.
  - Positive Tests:
    - `predict_cache_hit_rate` (trace-walking path; has gridDim surface) applies gridDim alpha penalty FIRST, then L2 class correction.
    - `predict_from_aggregates` (aggregate path; no gridDim surface) applies the L2 class correction directly to the aggregate-derived point estimate (no gridDim composition needed because the aggregate path does not consume launch-level gridDim).
    - For an allowlist slug, both entry points apply the L2 reduction-family factor under matching inputs.
  - Negative Tests:
    - Either entry point bypassing the L2 class correction for an allowlist slug triggers test failure.

- AC-12: Composition and clamp edge tests via factor-injection helper.
  - Positive Tests:
    - Composition test: gridDim alpha penalty composed with L2 class correction in the documented order produces `clamp(point_pre_class * factor, 0, 1)` for an allowlist slug and a non-allowlist slug.
    - Clamp-at-zero test: with `factor=0` injected via test helper, the corrected point estimate equals 0.0; `ci_low/ci_high` both 0.0; `ci_half_width` 0.0.
    - Clamp-at-one test: with `factor=2.0` and `point_pre=0.6` injected via helper, the corrected point estimate saturates at 1.0; `ci_high` saturates at 1.0.
    - CI-boundary-straddling test: with `ci_low=0.4`, `ci_high=0.9`, `factor=1.5` injected, `ci_high` saturates at 1.0, `ci_low = 0.6`, `ci_half_width = (1.0 - 0.6) / 2 = 0.2`.
  - Negative Tests:
    - Any composition, clamp, or `ci_half_width` recomputation regression triggers failure.

### Monitoring Goals (Advisory; not Acceptance)

- MG-1: hostile_l2 geomean_rel_err measurably decreases from baseline 0.5156.
- MG-2: `rmsnorm_fwd_bf16_sm80` L2 rel_err measurably reduced from 0.425.
- MG-3: `softmax_safe_bf16_sm80` L2 rel_err measurably reduced from 0.674.
- MG-4: HARD-1 friendly_l2 geomean_rel_err improves (drops below 0.14558), widening the HARD-1 margin from baseline 0.004.
- MG-5: Calibrated factor recorded in a deterministic fixture (e.g. `data/sm_86_predictor_accuracy_recovery_v2/l2_reduction_correction_calibration.json`) capturing input bundles, residual ratios, and computed factor for reproducibility.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation includes the L2 reduction-family multiplicative correction applied to exactly two slugs (`rmsnorm_fwd_bf16_sm80`, `softmax_safe_bf16_sm80`), CI bound and `ci_half_width` recomputation, full positive paired controls per allowlist slug, full negative paired controls (`silu_fwd_bf16_sm80`, `layernorm_fwd_bf16_sm80`, a non-reduction slug), allowlist-exact-match assertion, bidirectional L1/L2 head isolation tests (existing + new reverse direction), composition tests with gridDim + clamp-at-zero/at-one/CI-straddling via factor-injection helper, `ALGORITHM_VERSION` unchanged-assertion test, scanner assertions for absence of pointer-chase and local-mem-spill surface, bundle schema validation across all admitted cells, a shadow report covering every admitted L2 cell with per-cell regression budget verification, deterministic calibration provenance fixture, post-commit recapture for tracked artifacts, and diff-scoped plan-token hygiene.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes the L2 reduction-family multiplicative correction for the 2 allowlisted slugs, positive paired controls for at least one allowlist slug, the three required negative paired controls (`silu_fwd_bf16_sm80`, `layernorm_fwd_bf16_sm80`, a non-reduction slug), bidirectional L1/L2 head isolation tests, the shadow report with per-cell regression budget verification, hard-target verification, and post-commit recapture. Clamp-at-one and CI-straddling tests may be deferred only if the calibrated factor is empirically in `(0, 1]` and verified by the shadow report; clamp-at-zero remains mandatory.

### Allowed Choices

- Can use: internal classifier inside `workloads/simbench_bw/cache_predictor.py` (frozenset constant + factor constant), multiplicative scaling via `clamp(x * factor, 0, 1)`, paired controls in `workloads/simbench_bw/tests/`, deterministic JSON fixture for calibration provenance, BL-20260508 recapture script for tracked artifacts.
- Cannot use: piggybacking the L1 reduction classifier (`_L1_REDUCTION_KERNEL_SLUG_ALLOWLIST`) for the L2 correction (the existing L2-isolation regression test forbids that coupling), modifying the L1 head or its constants, adding new fields to the bundle JSON schema, single-cell allowlist (each L2 correction surface requires at least two members or structural evidence), pointer-chase or local-mem-spill correction in this round, `ALGORITHM_VERSION` bump, cohort-membership changes for `l2_thrash_pointer_chase` in this round.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. Read active sweep predictor bundles at `data/local_labels/sweeps/sweep_20260514T165947Z_a1closure_ebe60dd/predictor_bundles/` for `rmsnorm_fwd_bf16_sm80` and `softmax_safe_bf16_sm80`. For each, compute the L2 point estimate AFTER the gridDim alpha penalty (call it `point_pre_class`). Compute residual ratio `r_slug = truth_l2 / point_pre_class`. The calibrated factor is the geometric mean of `r_slug` across the two allowlist slugs (or a minimum-MSE fit — choose one in the calibration fixture and document the choice).
2. Add constants near the existing L1 reduction allowlist in `workloads/simbench_bw/cache_predictor.py`:

   ```python
   _L2_REDUCTION_KERNEL_SLUG_ALLOWLIST: frozenset[str] = frozenset({
       "rmsnorm_fwd_bf16_sm80",
       "softmax_safe_bf16_sm80",
   })
   _L2_REDUCTION_KERNEL_CORRECTION_FACTOR: float = <calibrated value>
   ```

3. In the L2 prediction finalization path (after the existing `_GRIDDIM_PENALTY_ALPHA * log_density` term), call a new helper `_apply_l2_reduction_correction(point, ci_low, ci_high, slug)` that returns the clamp-and-scale-transformed `(point, ci_low, ci_high, ci_half_width)` when slug is in the allowlist, else returns the inputs unchanged.
4. Apply the helper in BOTH `predict_cache_hit_rate` (trace-walking path; composes after gridDim) and `predict_from_aggregates` (aggregate path; applies directly, no gridDim composition).
5. Add paired controls, bidirectional isolation tests, composition tests with factor-injection helper, the shadow report, and the diff-scoped plan-token scan.

### Relevant References

- `workloads/simbench_bw/cache_predictor.py` — main predictor module containing the L1 reduction allowlist (`_L1_REDUCTION_KERNEL_SLUG_ALLOWLIST` + `_L1_REDUCTION_KERNEL_CORRECTION_FACTOR = 0.32`), `_GRIDDIM_PENALTY_ALPHA = 0.110`, and both L2 entry points (`predict_cache_hit_rate`, `predict_from_aggregates`). The new L2 reduction-family constants and helper land in this file.
- `workloads/simbench_bw/tests/test_l1_reduction_kernel_correction.py` — the existing L2-isolation regression test `test_bundle_l2_byte_identical_regardless_of_l1_correction` (model for the reverse-direction test the new plan adds).
- `workloads/simbench_bw/tests/test_l1_paired_controls.py` — paired-control test pattern to mirror for L2.
- `workloads/simbench_bw/tests/test_cache_predictor_gridDim.py` — composition test pattern for the gridDim alpha penalty.
- `data/sm_86_predictor_accuracy_recovery_v2/interim_diagnostic_report_sidecar.json` — baseline numbers for `friendly_l2`, `friendly_l1`, `cross_cell_stddev`, and per-cell L2 rel_err values used in the original investigation.
- `data/local_labels/sweeps/sweep_20260514T165947Z_a1closure_ebe60dd/predictor_bundles/` — active sweep bundles supplying `truth_l2_hit_rate` and `predicted_l2_hit_rate` for calibration.
- BL-20260508 — post-commit recapture discipline for tracked artifacts (sed-replace prior fix-commit SHA in tracked artifact `source_commit` fields).
- BL-20260507 — 3-surface lockstep discipline (NOT triggered this round; classifier is internal-only; documented for context if a future round elevates it to wire-visible).

## Dependencies and Sequence

### Milestones

1. **Calibration**: derive `_L2_REDUCTION_KERNEL_CORRECTION_FACTOR` from active sweep post-gridDim residuals for the 2 allowlist slugs.
   - Compute per-slug residual ratio.
   - Combine into a single factor (geometric mean or minimum-MSE).
   - Record source bundles, ratios, and final factor in a deterministic JSON fixture for reproducibility.

2. **Implementation**: add allowlist + factor constants; implement the correction helper; apply at both L2 entry points.
   - Constants near the existing L1 reduction allowlist.
   - Helper `_apply_l2_reduction_correction`.
   - Application in `predict_cache_hit_rate` after gridDim alpha penalty.
   - Application in `predict_from_aggregates` (direct, no gridDim).
   - CI bound and `ci_half_width` recomputation.

3. **Testing**: positive and negative paired controls, bidirectional L1/L2 isolation, composition and clamp edge tests via factor-injection helper, `ALGORITHM_VERSION` unchanged-assertion, scanner assertions, bundle schema validation.

4. **Shadow report and budget verification**: per-cell before/after across every admitted L2 cell with regression-budget compliance check.

5. **Hard-target verification**: HARD-1 (< 0.15), HARD-2 byte-identical, HARD-3 in `[0.10, 0.15]`.

6. **Post-commit recapture**: BL-20260508 SHA sed-replace across tracked artifacts after each fix-commit.

7. **Plan-token hygiene scan**: diff-scoped scanner against new/modified lines.

Dependencies:
- Milestone 2 depends on Milestone 1 (factor value).
- Milestone 3 depends on Milestone 2 (code under test).
- Milestone 4 depends on Milestones 2-3 (correction logic + tests pass before the shadow report runs).
- Milestone 5 depends on Milestone 4 (shadow report confirms hard-target preservation).
- Milestone 6 depends on Milestone 5 (recapture after the final fix-commit).
- Milestone 7 runs alongside Milestone 3 (continuous scan).

## Task Breakdown

Each task includes exactly one routing tag:
- `coding`: implemented by Claude.
- `analyze`: executed via Codex (`/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Compute calibrated `_L2_REDUCTION_KERNEL_CORRECTION_FACTOR` from active sweep post-gridDim residuals for rmsnorm + softmax_safe; write deterministic calibration fixture | MG-5 | analyze | - |
| task2 | Add `_L2_REDUCTION_KERNEL_SLUG_ALLOWLIST` (rmsnorm_fwd_bf16_sm80, softmax_safe_bf16_sm80) and `_L2_REDUCTION_KERNEL_CORRECTION_FACTOR` constants near existing L1 reduction allowlist in cache_predictor.py | AC-1 | coding | task1 |
| task3 | Implement `_apply_l2_reduction_correction` helper transforming (point, ci_low, ci_high) via clamp(x * factor, 0, 1) and recomputing ci_half_width | AC-1, AC-7, AC-12 | coding | task2 |
| task4 | Apply helper in `predict_cache_hit_rate` AFTER existing gridDim alpha penalty | AC-7, AC-11 | coding | task3 |
| task5 | Apply helper in `predict_from_aggregates` (no gridDim composition; direct application) | AC-11 | coding | task3 |
| task6 | Add positive paired controls for each allowlist slug; allowlist-exact-match assertion | AC-1 | coding | task4, task5 |
| task7 | Add negative paired controls for silu_fwd_bf16_sm80, layernorm_fwd_bf16_sm80, and a non-reduction slug (working_set_capacity_sweep) | AC-1 | coding | task4, task5 |
| task8 | Add bidirectional L1/L2 isolation test: new `test_bundle_l1_byte_identical_regardless_of_l2_correction` | AC-6 | coding | task4 |
| task9 | Add composition test (gridDim + L2 class correction in documented order); clamp-at-zero, clamp-at-one, CI-straddling tests via factor-injection helper | AC-7, AC-12 | coding | task4 |
| task10 | Add `ALGORITHM_VERSION` unchanged-assertion test | AC-7 | coding | task4 |
| task11 | Add scanner assertions: no `_L2_POINTER_CHASE_*` or `_L2_LOCAL_MEM_SPILL_*` identifiers in the diff; no `l2_thrash_pointer_chase` or `register_spill_local_mem` in any new L2 allowlist | AC-2, AC-3 | coding | task6 |
| task12 | Add bundle schema validation test across all admitted cells; verify byte-identity for unaffected cells excluding provenance SHA fields | AC-8 | coding | task6 |
| task13 | Generate shadow report (table: slug, class_matched, factor, point_pre/post, truth, rel_err_before/after, delta, budget_pass) sorted by descending delta | AC-5 | analyze | task4, task5 |
| task14 | Verify per-cell regression budget (`delta <= 0.02 abs`) across every admitted L2 cell | AC-5 | analyze | task13 |
| task15 | Verify hard-target preservation: friendly_l2 < 0.15, friendly_l1 byte-identical, 0.10 <= cross_cell_stddev <= 0.15 | AC-4 | analyze | task13 |
| task16 | Verify monitoring goals (hostile_l2 decreases, rmsnorm/softmax_safe L2 rel_err decreases, HARD-1 margin widens) | MG-1..MG-4 | analyze | task15 |
| task17 | Diff-scoped plan-token hygiene scan against new/modified lines | AC-9 | coding | task6 |
| task18 | Post-commit recapture: regenerate tracked artifacts, sed-replace prior source_commit with new fix-commit SHA, commit recapture chore | AC-10 | coding | task15 |

## Claude-Codex Deliberation

### Agreements

- L2 correction must live on a separate classifier surface; piggybacking the L1 reduction allowlist would break the existing `test_bundle_l2_byte_identical_regardless_of_l1_correction` regression guard.
- `silu_fwd_bf16_sm80` and `layernorm_fwd_bf16_sm80` (both in L1 reduction allowlist) must be explicit L2 negative controls — their L2 rel_err is already near-zero post-gridDim, so a blanket reduction-family L2 factor would regress them.
- Internal-only classifier (no new bundle fields, no schema lockstep) is the right default this round. BL-20260507 lockstep is documented for context but not triggered.
- Post-gridDim ordering is correct: gridDim alpha penalty applies FIRST as an additive residual on the model output; the L2 class factor applies AFTER as a multiplicative residual on the gridDim-corrected estimate.
- CI bounds must transform with the same `clamp(x * factor, 0, 1)` function as the point estimate; `ci_half_width` recomputed from corrected `ci_low/ci_high`.
- Bidirectional L1/L2 isolation tests are required (existing forward direction + new reverse direction).
- Per-cell regression budgeting must be enforceable (not just monitoring), with deterministic test failure naming any offending cell.
- Single-cell allowlists are target-fitting; pointer-chase and local-mem-spill require structural evidence beyond slug matching and are deferred.
- Staged approach (reduction-family this round; pointer/spill follow-up) is preferred over a 3-correction monolithic round.

### Resolved Disagreements

- **L1 vs L2 classifier reuse**: Original draft suggested mirroring the L1 reduction allowlist directly. Codex flagged the existing L2-isolation regression test plus the fact that `silu_fwd_bf16_sm80` is in the L1 allowlist but L2-accurate. **Resolution**: L2 correction lives on a separate allowlist (`_L2_REDUCTION_KERNEL_SLUG_ALLOWLIST`) with only `rmsnorm_fwd_bf16_sm80` and `softmax_safe_bf16_sm80`; `silu_fwd_bf16_sm80` and `layernorm_fwd_bf16_sm80` are explicit negative controls.
- **Scope of corrections this round**: Original draft proposed 3 corrections (reduction, pointer-chase, local-mem-spill) in one round. Codex flagged single-cell-fitting risk for pointer-chase and local-mem-spill. **Resolution**: stage — reduction-family only this round; pointer/spill deferred until structural predictor evidence (beyond slug matching) is available.
- **HARD-3 byte-identity**: Original draft assumed HARD-3 byte-identical. Codex noted L2 corrections WILL move `cross_cell_stddev` (computed across cells from L2 predictions) and existing tooling treats HARD-3 as `>= 0.10`, not only `<= 0.15`. **Resolution**: HARD-3 acknowledged to move; bound set to `0.10 <= cross_cell_stddev <= 0.15` (user confirmed).
- **AC-9 plan-token scope**: Original AC-9 applied to all implementation files. Codex flagged that pre-existing files may already contain plan-token literals as legacy debt. **Resolution**: AC-9 diff-scoped to new/modified lines only; pre-existing debt is out of scope.
- **Byte-identity precision**: Codex flagged that "byte-identical" can fail on map ordering or float formatting variance even when semantically equivalent. **Resolution**: byte-identity specified as comparison after deterministic JSON serialization; provenance SHA fields (`manifest_pairing_index.source_commit`, `predictor_build_commit`) explicitly excluded per the BL-20260508 recapture flow.
- **AC-10 self-referential SHA**: Codex flagged that a commit cannot contain its own SHA. **Resolution**: AC-10 rewritten — the artifact's `source_commit` references the code-commit that GENERATED the artifact (the prior fix-commit), not the artifact-containing chore-recapture commit.
- **Two L2 entry points**: Codex flagged that both `predict_cache_hit_rate` and `predict_from_aggregates` exist as L2 prediction surfaces. **Resolution**: AC-11 explicitly requires both paths apply the L2 class correction. Trace-walking path composes after gridDim; aggregate path applies directly (no gridDim surface).
- **`ALGORITHM_VERSION` choice**: Codex required explicit choice (not "unchanged OR bumped"). **Resolution**: unchanged (user confirmed). The correction is a post-hoc multiplicative adjustment, not a model architectural change.

### Convergence Status

- Final Status: `partially_converged`
- Reason: 3 convergence rounds executed (max). Round 3 produced a small set of REQUIRED_CHANGES (AC-10 wording, `ALGORITHM_VERSION` choice, DEC-3 / DEC-7 explicit values, calibration source naming, byte-identity precision) — all resolved during Phase 6 via user-decision resolution plus direct plan refinement. The remaining `UNRESOLVED` item (DEC-4 hostile_l2 advisory target) is intentionally advisory-only and does not gate admission.

## Pending User Decisions

- DEC-1: Internal-only L2 classifier vs wire-visible.
  - Claude Position: Internal-only (no new bundle fields; preserve byte-identity for unaffected cells).
  - Codex Position: Agrees — avoid BL-20260507 lockstep unless wire-visibility is required by a consumer.
  - Tradeoff Summary: Internal-only minimizes scope; wire-visible would require schema + frozenset + comparison_runner 3-surface lockstep.
  - Decision Status: Resolved — internal-only.

- DEC-2: `l2_thrash_pointer_chase` cohort reclassification (predictor_friendly → predictor_hostile).
  - Claude Position: Defer to a separate decision (do not bundle with correction).
  - Codex Position: Agrees — "do no harm" cohort reclassification audit should be a separate decision.
  - Tradeoff Summary: Reclassification would widen HARD-1 margin artificially without addressing the predictor's structural defect.
  - Decision Status: Deferred to a separate decision/round.

- DEC-3: Per-cell regression budget threshold.
  - Claude Position: `<= 0.01` absolute rel_err increase (tight).
  - Codex Position: 0.01 reasonable; must apply to every admitted cell deterministically.
  - User Decision: `<= 0.02` absolute (looser).
  - Tradeoff Summary: 0.02 accepts small regressions on already-accurate cells in exchange for calibration flexibility; tighter values risk multiple calibration iterations.
  - Decision Status: Resolved — `<= 0.02` absolute.

- DEC-4: hostile_l2 monitoring target.
  - Claude Position: `< 0.30` (advisory, monitoring only).
  - Codex Position: Advisory only; should not gate admission.
  - User Decision: No specific numerical target as AC; only the monitoring goal that hostile_l2 measurably decreases (per MG-1).
  - Tradeoff Summary: A specific hostile_l2 numerical target would over-constrain; "measurably decreases" leaves room for calibration to find the achievable bound.
  - Decision Status: Resolved — advisory monitoring only (no AC).

- DEC-5: New bundle evidence fields for L2 correction provenance.
  - Claude Position: No new fields (internal-only correction; preserve byte-identity).
  - Codex Position: Agrees — emitting evidence fields breaks byte-identity for unaffected cells.
  - Tradeoff Summary: Internal-only minimizes scope; provenance traced through the calibration fixture and round summary instead.
  - Decision Status: Resolved — no new bundle fields.

- DEC-6: Scope this round.
  - Claude Position: Reduction-family only; pointer-chase + local-mem-spill deferred.
  - Codex Position: Agrees — single-cell allowlists are target-fitting; structural evidence required for pointer/spill.
  - Tradeoff Summary: Staged approach trades a multi-round implementation for lower regression risk.
  - Decision Status: Resolved — reduction-family only.

- DEC-7: HARD-3 `cross_cell_stddev` bound.
  - Claude Position: `<= 0.15` (upper bound only).
  - Codex Position: Existing tooling treats HARD-3 as `>= 0.10`; both bounds required.
  - User Decision: `0.10 <= cross_cell_stddev <= 0.15`.
  - Tradeoff Summary: Adding the lower bound respects existing tooling contract; upper-bound-only would risk inconsistency with the closure-audit gate.
  - Decision Status: Resolved — `0.10 <= cross_cell_stddev <= 0.15`.

- DEC-8: `ALGORITHM_VERSION` behavior.
  - Claude Position: Unchanged (post-hoc multiplicative adjustment is not a model architectural change).
  - Codex Position: Must be explicit; either unchanged OR bumped with rationale.
  - User Decision: Unchanged.
  - Tradeoff Summary: Unchanged preserves consumer-parsing compatibility; bump would force version negotiation across all consumers.
  - Decision Status: Resolved — unchanged.

- DEC-9: Primary goal of this next-phase plan.
  - Claude Position: Either hostile-cell headroom OR HARD-1 margin widening (single primary).
  - Codex Position: Depends on user preference.
  - User Decision: Both equally.
  - Tradeoff Summary: Both goals are addressed by the same correction — reducing rmsnorm/softmax_safe L2 over-prediction reduces hostile rel_err and improves friendly_l2 margin (both slugs are in the friendly cohort).
  - Decision Status: Resolved — co-equal goals.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Phase \d`, `Step \d`, or similar workflow markers, scoped to NEW or MODIFIED lines in this round's diff (pre-existing debt is out of scope per AC-9).
- Use descriptive, domain-appropriate naming in code:
  - `_L2_REDUCTION_KERNEL_SLUG_ALLOWLIST` (not `_AC_1_ALLOWLIST`).
  - `_L2_REDUCTION_KERNEL_CORRECTION_FACTOR` (not `_M2_FACTOR`).
  - `_apply_l2_reduction_correction` (not `_apply_step_4_correction`).
- Comments documenting WHY the correction exists may reference the factual situation ("L2 head over-predicts reduction-kernel hit-rate by ~30-50%; correction calibrated against active sweep residuals") but must not encode workflow markers.
- The calibration fixture path is the canonical reference for factor provenance; comments may name the fixture path.

### Project-Specific Constraints (per CLAUDE.md)

- No NCU / Nsight Compute on cloud or server GPUs. Local-label exception (NCU permitted under `tools/local_labels/` and `data/local_labels/` quarantine) applies for the calibration sweep already captured — no new NCU invocation outside the quarantine.
- No GPU-driver-root operations (no `nvidia-smi` clock locking / persistence / power / MIG / compute-mode changes).
- No privileged CUPTI surfaces (Events / Metrics / Range Profiling / PM Sampling / PerfWorks). CUPTI Activity is allowed.
- No host-driver work.
- No simulator revival.
- Two-device cross-architecture discipline: this round is expected to land on one architecture (sm_86, RTX 3090 Ti); cross-architecture verification on sm_120 (RTX 5060 laptop) is a follow-up obligation. Round summary must specify the verification status per CLAUDE.md (`Offline-verified`, `Verified-live-sm_86`, `Verified-live-sm_120`, or `Cross-arch-verified`).

--- Original Design Draft Start ---

# Access-Pattern Allowlist Correction For L2 Hostile Cells

## Original Idea

L2 prediction shows large rel_err on a 4-cell class: register_spill_local_mem (L2 rel_err 2.92, pred 0.998 vs truth 0.255), l2_thrash_pointer_chase (rel_err 0.88, pred 0.998 vs truth 0.531), softmax_safe_bf16_sm80 (rel_err 0.67, pred 0.958 vs truth 0.573), rmsnorm_fwd_bf16_sm80 (rel_err 0.43, pred 0.969 vs truth 0.679). Friendly_l2 geomean is 0.146 (passes HARD-1 < 0.15 by 0.004 margin); hostile_l2 geomean is 0.516 with only 2 cells. All cells share identical bundle inputs (grid=64, slices=6, hrd_second_layer=True, cta_interleave=True, capacity=6MB). The predictor's L2 model has no class-specific dimension to distinguish reduction kernels, pointer-chase, or local-memory spill access patterns. Predictor systematically over-predicts L2 hit rate. The L1 head got a reduction-kernel correction (factor 0.32) in R46; L2 has no such correction. Brainstorm next-phase strategies: (1) introduce no-leakage L2 predictor inputs that capture access-pattern dimensions (e.g. inter-cta sharing ratio, intra-launch reuse, line-residency-density from NVBit trace), (2) class-specific L2 corrections analogous to L1 reduction-kernel allowlist, (3) cohort substitution to replace highest-rel_err hostile cells, (4) refined L2 walker model that consumes per-block access counts directly. Hard targets are all PASS but the margin on HARD-1 is thin (0.004); hostile cells offer the biggest accuracy headroom.

## Primary Direction: Access-Pattern Classifier Allowlist

### Rationale

Mirrors the proven R46 L1 reduction-kernel correction pattern in the cleanest one-to-one way: small surgical addition to the L2 head with an explicit class allowlist, scaling factor, and paired-control tests — no new wire-format fields, no new bundle inputs, no new walker model. Smallest implementation surface of all six directions while directly attacking the three identified hostile classes (reduction, pointer-chase, local-mem spill).

### Approach Summary

Introduce a class-specific L2 correction analogous to the R46 L1 reduction-kernel allowlist. After the existing gridDim density alpha=0.110 correction in the L2 prediction path at `tools/cache_predictor/cache_predictor.py` (around line 1509, where the L2 hit-rate is finalized), apply a multiplicative scaling factor when the cell's `arch_capabilities` / class-tag matches one of three families: (a) reduction-style memory access (rmsnorm_fwd, softmax_safe, layernorm_fwd, silu_timing — already characterized via R46's existing reduction-kernel allowlist), (b) pointer-chase / dependent-load access (l2_thrash_pointer_chase and any cell with `dependent_load_chain=True` if such a tag exists, else a hard-coded slug allowlist), (c) local-memory spill access (register_spill_local_mem and cells with high `local_mem_byte_ratio`).

The scaling factor is calibrated from the same source data the L1 head used: the truth/predicted ratios in the active sweep predictor bundles. Estimated factors from the hostile-cell investigation: reduction-on-L2 ≈ 0.65, pointer-chase ≈ 0.53, local-mem-spill ≈ 0.26 (these scale the predicted hit-rate down toward truth). Each factor lives behind a typed constant in `cache_predictor.py` with the same provenance comment as the L1 reduction factor 0.32. The allowlist itself lives in a frozenset alongside the existing `_REDUCTION_KERNEL_SLUGS` constant — 3-surface lockstep (frozenset + schema enum + comparison_runner constant) per BL-20260507 if the allowlist is wire-visible, else a single-surface internal constant if the classification stays inside `cache_predictor.py`.

Paired controls land at `tools/cache_predictor/test_cache_predictor.py` mirroring the existing reduction-kernel test: one positive control per family (asserts the correction lowers the predicted hit-rate for an allowlisted cell), one negative control (asserts a non-allowlisted cell — e.g. a vanilla GEMM — is unaffected), and one boundary test confirming the correction composes cleanly with the gridDim alpha correction (i.e. order-independent within the floating-point tolerance the harness already enforces).

Closure-audit gate impact: +3-9 new paired controls land in the existing cache_predictor sweep; the static plan-token guard at `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` is unaffected because constants are not plan-tokens.

### Objective Evidence

- `tools/cache_predictor/cache_predictor.py:895-963` — existing L1 reduction-kernel correction with factor 0.32 (the precedent this direction mirrors verbatim for the L2 head).
- `tools/cache_predictor/cache_predictor.py:1509` — current L2 prediction path where the gridDim density alpha=0.110 correction lands; the new class-specific scaling factor would compose immediately after.
- `tools/cache_predictor/test_cache_predictor.py` — paired controls for the L1 reduction factor (the existing test pattern the new L2 controls would mirror).
- `data/sm_86_predictor_accuracy_recovery_v2/interim_diagnostic_report_sidecar.json` — hostile-cell rel_err numbers, calibration source for the three scaling factors.
- `data/local_labels/sweeps/sweep_20260514T165947Z_a1closure_ebe60dd/predictor_bundles/` — active sweep bundles supplying both predicted hit-rate and truth hit-rate for the 4 hostile cells; calibration source.
- R46 round summary (Plan-1.5 history) — original justification for the L1 reduction-kernel allowlist + factor; identical justification shape applies to L2.

### Known Risks

- Hostile-cell sample size is small (4 cells total); calibrated factors risk overfitting to these specific cells rather than generalizing to the broader class. Mitigation: cross-validation against any other cells that share access-pattern signatures (e.g. additional softmax / rmsnorm variants in the sweep).
- The three families are heuristic; a cell with a mixed access pattern (e.g. softmax with local-mem spill) gets only the first matching factor. Mitigation: ordering rule documented in code, paired control verifies the chosen ordering.
- Allowlist maintenance burden grows with each new cell added to the corpus. Mitigation: keep the allowlist next to the existing R46 one so the maintenance surface is centralized.
- If `cta_interleave=True` + `hrd_second_layer=True` is the actual source of over-prediction (not access pattern), the correction will mask but not diagnose the underlying model defect. Mitigation: pair with Alt-6 (working-set gate) as a structural fail-closed check.

## Alternative Directions Considered

### Alt-1: Inter-CTA Sharing Ratio Input
- Gist: Introduce a new no-leakage predictor input — inter-CTA sharing ratio computed from the NVBit trace — that captures whether different CTAs touch overlapping cache lines. Plumb the ratio through the L2 head as an additional regression feature. Existing `tools/sharing_locality/classifier.py:94-186` already produces line-level sharing signatures from L1 traces; extending to L2 line-granularity would reuse the same observation_builder path at `tools/hrd_profiler/observation_builder.py:522-649`.
- Objective Evidence:
  - `tools/hrd_profiler/observation_builder.py:522-649` — existing per-block observation builder ingesting NVBit traces; the natural site for an inter-CTA sharing ratio aggregate.
  - `tools/sharing_locality/classifier.py:94-186` — existing line-level sharing classifier producing per-trace sharing signatures.
  - `tools/cache_predictor/cache_predictor.py:895-964` — L1 reduction correction; the model for a new feature plumbed through the L2 head.
  - Active sweep predictor bundles in `data/local_labels/sweeps/sweep_20260514T165947Z_a1closure_ebe60dd/predictor_bundles/` — bundle schema with room for a new scalar field via proto3 `optional`.
- Why not primary: significantly more invasive — adds a new wire-format field (producer + schema + validator lockstep per CLAUDE.md), regenerates predictor bundles for all cells, and requires a re-calibration of the L2 head's full feature weights rather than a surgical multiplicative correction. Higher generalizability but higher cost; better as a Phase-2 follow-up after the allowlist proves the class-specific signal is real.

### Alt-2: Hostile Cohort Substitution
- Gist: Use the existing task12 substitution preflight to replace the two highest-rel_err hostile cells (register_spill_local_mem rel_err 2.92, l2_thrash_pointer_chase rel_err 0.88) with alternative microbench cells from the candidate pool, then re-verdict the hard targets. The substitution path is already verdict-emitting (`task12_substitution_feasible_min_subs_1`) and 30+ prior sweeps prove live NCU collection works for new candidates.
- Objective Evidence:
  - `tools/sm_86_predictor_accuracy_recovery_v2/task12_substitution_preflight.py` — existing substitution preflight with verdict emission.
  - `data/local_labels/sweeps/` — 30+ prior sweeps demonstrating live NCU label collection paths.
  - `tools/sm_86_predictor_accuracy_recovery_v2/test_*` — existing substitution scanner tests.
- Why not primary: substitution improves the headline number without addressing the predictor's structural defect; the L2 head still over-predicts on reduction / pointer-chase / spill patterns, just on cells excluded from the cohort. Cohort changes are reversible later, but the structural correction is what unlocks broader generalization.

### Alt-3: Per-Block L2 Walker From NVBit
- Gist: Build a per-block L2 walker that consumes NVBit address traces directly and simulates L2 line-residency at the block granularity, mirroring the existing L1 walker architecture (`tools/cache_predictor/l1_walker.py` + `l1_trace_adapter.py` + `l1_production_helper.py`). The walker emits a predicted hit-rate per cell that feeds the L2 head as a high-fidelity prior.
- Objective Evidence:
  - `tools/cache_predictor/l1_walker.py` — existing L1 walker architecture as the precedent for the L2 walker.
  - `tools/cache_predictor/l1_trace_adapter.py` + `tools/cache_predictor/l1_production_helper.py` — the adapter + production helper modules the L2 variants would mirror.
  - NVBit trace artifacts under `workloads/simbench_bw/l1/<slug>/traces/` — input source.
- Why not primary: ~1200-1800 new LOC across 3 modules, plus new wire-format integration and a calibration loop. Highest-fidelity option but largest implementation surface and longest review cycle; the allowlist correction can land in 1-2 rounds while the walker would take 5-8.

### Alt-4: Bucket Re-classification Audit
- Gist: Audit whether l2_thrash_pointer_chase is correctly classified as `predictor_friendly` — it likely belongs in `predictor_hostile`. Moving it would drop friendly_l2 geomean to ~0.09 and widen the HARD-1 PASS margin from 0.004 to ~0.06. Pure classification fix, no model change.
- Objective Evidence:
  - `data/sm_86_predictor_accuracy_recovery_v2/interim_diagnostic_report_sidecar.json` — current cohort classification.
  - `tools/sm_86_predictor_accuracy_recovery_v2/` — cohort-classification module locations.
- Why not primary: medium confidence — the re-classification might be correct, but without a principled criterion the move risks looking like target-fitting. Better as a paired-control with the primary correction (verify the corrected predictor on the cell justifies re-classification on intrinsic difficulty, not on convenience).

### Alt-5: Working-Set vs Capacity Gate
- Gist: Add a structural fail-closed gate that compares the cell's unique sector count (already at `tools/hrd_profiler/observation_builder.py:262`) against the L2 capacity (6MB / 128B = 49152 sectors). When the working set exceeds capacity by a configurable threshold, the predictor refuses to claim high hit-rate and emits a typed fail-closed diagnostic instead.
- Objective Evidence:
  - `tools/hrd_profiler/observation_builder.py:262` — `unique_sector_count` already computed; no new wire field.
  - `tools/cache_predictor/cache_predictor.py` — existing fail-closed exception patterns (e.g. `L1ProductionHelperInstructionMalformed`).
- Why not primary: a fail-closed gate diagnoses but doesn't correct — the predictor returns an exception rather than a refined number, which the AC-7 sidecar would then propagate as a verdict-failure rather than a passing prediction. Better as a paired companion to the primary (gate flags the pathology, allowlist corrects the prediction).

## Synthesis Notes

The primary direction (Access-Pattern Classifier Allowlist) is deliberately the smallest-surface intervention. If during plan execution the calibrated factors turn out to overfit or fail to generalize, the natural escalation path folds in Alt-1 (Inter-CTA Sharing Ratio Input) as a principled feature replacing the heuristic allowlist, while keeping the allowlist as the fallback. Alt-5 (Working-Set vs Capacity Gate) is the recommended paired companion — it lands as a fail-closed structural check that diagnoses cases the allowlist might miss, without competing for the same correction surface. Alt-4 (Bucket Re-classification Audit) is the recommended paired-control on the primary's success — once the predictor's hit-rate prediction for l2_thrash_pointer_chase becomes accurate, re-evaluating its `predictor_friendly` classification on intrinsic-difficulty grounds is defensible. Alt-2 (Hostile Cohort Substitution) and Alt-3 (Per-Block L2 Walker) are deferred: Alt-2 because it masks rather than fixes, Alt-3 because the implementation cost dwarfs the per-round budget. Both remain available as Phase-3+ follow-ups if the allowlist + sharing-ratio combination still leaves headroom.

--- Original Design Draft End ---
