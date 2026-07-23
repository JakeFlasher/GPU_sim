# sm_86 Cohort Regeneration + Standalone sm_86 v5 Head on 3090 Ti (sm_86-develop-only)

## Goal Description

Complete the sm_86 half of the v5 footprint-aware per-bucket monotonic L2 hit-rate
calibration work on this RTX 3090 Ti (sm_86, Ampere GA102) — and ONLY the sm_86 half.
The sm_86 cohort's derived sidecars and predictor bundles have already been
regenerated locally (18 of 20 sm_86 label cells reach bundle admission; 2 fail closed
with typed reasons; the artifacts are committed). The remaining work is to fit a
STANDALONE sm_86 v5 calibration head over the admitted sm_86 cells and emit
sm_86-scoped closure artifacts (sm_86-native accuracy, baseline comparison, robustness,
and the residual-structure falsification verdict over the sm_86 slice), reaching
`Verified-live-sm_86` evidence.

This loop is **sm_86-develop-only by design**. Per CLAUDE.md's Two-Device Local
Development Model, cross-architecture work is sequential and multi-session: develop /
test / validate / commit on sm_86 first (this loop), then do the same on sm_120 (RTX
5060 laptop) in a SEPARATE later session. Cross-arch *verification* (the joint same-74
head, the cross-arch transfer 2× rule, and the `Cross-arch-verified` evidence status)
cannot be achieved inside a single session and is **explicitly deferred** to the
follow-on sm_120 session. The cross-arch discipline files (CLAUDE.md's two-device
model, HANDOFF.md, the `Cross-arch-verified` status, the sm_120 freeze test, the
`test_sm120_loop_claim_guard`) remain standing as forcing functions that keep sessions
decoupled — this loop respects them by stopping at `Verified-live-sm_86` and never
claiming cross-arch closure.

The 30 sm_120 admitted artifacts at HEAD `b1646ef7` are treated as **FROZEN and
untouched** inputs. They are NOT combined into this loop's head; the standalone sm_86
head is fit independently, and a regression guard proves the sm_120 artifacts do not
drift. The joint same-74 head is assembled later, in the sm_120 session, when both
arches' live evidence is present.

The v5 plan's IMMUTABLE designs (6-feature canonical vector, monotonic constraint
signs `[+1, -1, -1]`, pre-registered model-class hierarchy with composite scoring, fold
construction protocol, attribution procedure, falsification hypothesis, Nugteren 2014
RDH baseline) are inherited verbatim. Only the **cohort scope narrows from same-74 to
the sm_86 slice** for this loop. CLAUDE.md Rules 1-5 (no NCU outside the
`data/local_labels/` quarantine, no GPU-driver-root, no privileged CUPTI, no
host-driver work, no simulator revival) remain binding and non-negotiable — they are
project constants, unrelated to the cohort-scope narrowing.

### Cohort ground truth (corrects the prior same-74 framing)

- 30 sm_120 cells — FROZEN at `b1646ef7`, untouched this loop (regression-guarded).
- 20 sm_86 cells in `data/local_labels/label_admitted.json` (`arch == "sm_86"`).
  - 18 reach predictor-bundle admission (`status="admitted"` in the regen ledger).
  - 2 fail closed (`status="bundle_failed"`, typed reason
    `cache_model_underconfigured` — the walker's HRD bin coverage exceeds the
    `5e-3` share-override threshold on these small-N cells).
- 24 additional sm_86 cells are excluded at the label layer
  (`data/local_labels/label_excluded.json`: 18 `label_repeatability_failed`,
  6 `label_provenance_duplicate`). They are TERMINAL_TYPED_FAILURE in the staging
  manifest, never candidates for this loop.

The prior plan's "44 sm_86 / same-74" language reflected the pre-relabel sweep; the
relabel sweep narrowed the admitted-label set to 20. This loop scopes to those 20
label cells (18 admitted bundles).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. AC-1 through AC-4, AC-10, and AC-11 describe work that is
ALREADY COMPLETE and committed (the sm_86 artifact-production lane); the fresh loop
re-validates them and must keep them green. AC-5 (freeze guard), AC-6, AC-7, and AC-9
describe the remaining sm_86-scoped head + closure work.

- AC-1: A per-cell regeneration ledger covers every sm_86 cell in
  `data/local_labels/label_admitted.json` (`arch == "sm_86"`). The ledger lives at
  `data/local_labels/sm86_cohort_regen_ledger.json`, keyed by the 6-tuple
  `(arch, slug, shape_point, binary_sha256, kernel_signature_sha256,
  label_source_commit)`. Per-entry fields: `status` (closed enum, AC-1.1),
  `typed_reason` (non-admitted only), `binary_path`, `trace_dir`,
  `predictor_bundle_path`, `sidecar_paths` (the six task4 sidecars +
  `sector_stream_mpi`), `attempted_at` (UTC ISO-8601), `producer_commit`,
  `artifact_sha256` (eight digests — predictor bundle + dynamic trace + six sidecars),
  `validator_status`, `live_evidence_status`.
  - AC-1.1: `status` ∈ `{admitted, rebuild_binary_failed, trace_failed, sidecar_failed,
    bundle_failed, schema_invalid, identity_mismatch, trace_volume_exceeded,
    slowdown_budget_exceeded, drop_rate_exceeded}` (closed). The ledger validates
    against `workloads/simbench_bw/schemas/sm86_cohort_regen_ledger.schema.json`.
  - AC-1.2: `live_evidence_status` for sm_86 ledger entries is one of
    `{Offline-verified, Verified-live-sm_86}`. `Cross-arch-verified` is NOT emitted by
    this loop (it requires the sm_120 session's matching live witness). The schema's
    enum may retain `Cross-arch-verified` as a forward-reference, but the sm_86 ledger
    never uses it.
    - Positive: every sm_86 cell has exactly one ledger entry; admitted entries have
      all eight `artifact_sha256` digests non-null;
      `tools/local_labels/test_sm86_cohort_regen_ledger_coverage.py` passes.
    - Negative: a ledger missing any sm_86 cell fails the coverage scanner; an
      admitted entry with any null artifact digest fails the schema validator; a
      `live_evidence_status="Cross-arch-verified"` entry in the sm_86 ledger fails.

- AC-2: Each admitted sm_86 cell ends with one canonical predictor bundle via the
  two-stage cascade (DEC-3, inherited): Stage 1 admits a legacy bundle whose
  `manifest_pairing_index` 5-tuple matches the label AND `predictor_build_commit >=
  b1646ef7`; Stage 2 emits a fresh bundle via the walker. The bundle carries the
  canonical 6-feature `canonical_pre_calibration_features` block and the 5-tuple MPI
  matching the label cell verbatim. `bundle_admission_path ∈ {legacy_match,
  fresh_regen}`.
  - AC-2.1: Identity (`manifest_pairing_index.{slug, shape_point, binary_sha256,
    source_commit, kernel_signature_sha256}`) is SEPARATE from producer provenance
    (`predictor_build_commit`). The MPI is a closed 5-field object (no `arch` key);
    `tools/local_labels/test_build_predictor_bundle_for_cell.py` pins this.
  - AC-2.2: Forbidden modeled-hit keys (`hrd_profile`, `hrd_profile_path`,
    `hit_count`, `hit_rate`, `model_hit`, `modeled_hit_rate`, `predicted_hit_rate`)
    absent from `input_sidecar_digests`.
  - AC-2.3: Stale-bundle quarantine: legacy bundles with same `(slug, shape_point)` but
    mismatched 5-tuple are counted in
    `v5_cohort_staging_manifest.json::summary.legacy_bundle_quarantined` (an aggregate
    BUNDLE count, summed per row) and surfaced per-cell as
    `legacy_bundle_quarantined_count`. NOT a `suggested_operator_action` value.

- AC-3: Each admitted sm_86 cell has the six NVBit-derived sidecars at the canonical
  results-root convention `workloads/<suite>/results/<slug>/rtx_3090_ti/sm_86/
  trace_<shape_point>/`: `dynamic_trace.pb`, `sector_stream/sector_stream.jsonl` +
  sibling `sector_stream/manifest_pairing_index.json`, `sharing_locality/
  sharing_locality.json`, `hrd_profile/hrd_profile.json`,
  `divergence_proxies/divergence_proxies.json`, `order_projections/
  order_projections.json`. Each passes its offline schema validator and binds to the
  label cell's identity (closed 5-field MPI, no `arch` leak).
  - AC-3.1: Trace budget within `slowdown_ratio <= NVBIT_SLOWDOWN_BUDGET_MAX_RATIO`
    (1200) AND `drop_rate <= 0.05` AND `trace_volume_within_budget`, OR the cell fails
    closed via a typed terminal reason. Trace-volume terminal failures may delete the
    overbudget trace; the ledger records the deletion.
  - AC-3.2: Shape-name normalization: only current shape names bind; legacy
    `trace_00`-style names do NOT.

- AC-4: The staging manifest `data/local_labels/v5_cohort_staging_manifest.json`
  carries the closed `suggested_operator_action` enum `{READY, REBUILD_BINARY_LOCAL,
  REBUILD_TRACE_LOCAL, REGENERATE_SIDECARS_LOCAL, REBUILD_BUNDLE_LOCAL,
  BUNDLE_TYPED_UNAVAILABLE, BUNDLE_NON_SCALAR, CROSS_HOST_SCP, CROSS_HOST_SCP_SIDECARS,
  TERMINAL_TYPED_FAILURE}` plus a separate `action_typed_reason` string (present when
  action ∈ `{BUNDLE_TYPED_UNAVAILABLE, BUNDLE_NON_SCALAR, TERMINAL_TYPED_FAILURE}`).
  Schema-pinned in `workloads/simbench_bw/schemas/v5_cohort_staging_manifest.schema.json`.
  The manifest is a full-cohort inventory (it enumerates all 74 cohort cells: 30
  sm_120 + 20 sm_86 admitted-label + 24 sm_86 excluded) and is arch-agnostic.
  - Invariant: `summary.ready_count + summary.terminal_typed_failure_count ==
    cohort_size` (exact equality). The legacy colon-form action value fails the schema
    validator.
  - Note: the manifest inventorying sm_120 rows is NOT a cross-arch closure operation
    — it reports each cell's local staging state. It stays as-is.

- AC-5 (FROZEN sm_120 GUARD — kept): The 30 sm_120 admitted artifacts at HEAD
  `b1646ef7` are FROZEN. `tools/local_labels/test_sm120_cohort_frozen_at_b1646ef7.py`
  asserts (a) each sm_120 predictor bundle path is unchanged from `b1646ef7`, (b) its
  sha256 is unchanged, (c) no sm_120 cell's 5-tuple changed. The test runs BOTH before
  the sm_86 head-fit step (task9) AND immediately after, so any drift the sm_86 work
  induces on the frozen sm_120 artifacts is caught. This is a non-mutation regression
  guard, NOT a cross-arch closure dependency — it requires no live sm_120.
  - Positive: the freeze test admits both before and after the sm_86 head fit.
  - Negative: a synthetic mutation of any sm_120 bundle or 5-tuple lifts
    `sm_120_cohort_frozen_violation` at either invocation point.

- AC-6 (STANDALONE sm_86 head): The sm_86 v5 calibration head is fit over ONLY the
  admitted sm_86 cells, producing `data/local_labels/sm_86_v5_head.json` with
  `status="ok"`, `cohort_scope.admitted_count == admitted_sm86_count` (the sm_86
  admitted count, NOT 30 + N), and non-empty `per_bucket_provenance`. Per-bucket fit
  applies where the bucket has `n_train >= 8`; under-sized buckets fall back to
  `kind="pooled_fallback"` per the inherited v5 rule. The frozen `sm_120_v5_head.json`
  is NOT touched, and the joint `calibration_head_v5.json` same-74 head is NOT produced
  this loop (it is deferred to the sm_120 session).
  - AC-6.1: `admitted_sm86_count >= 8` is REQUIRED for the sm_86 head fit (DEC-1 floor,
    inherited; ensures per-bucket fits have data). Currently met (18 admitted). If a
    future sm_86-only session admits fewer than 8, the loop escalates to operator
    review rather than closing.
    - Positive: `admitted_sm86_count >= 8` AND `sm_86_v5_head.json::status="ok"`.
    - Negative: `admitted_sm86_count < 8` after the head-fit step fails; an
      `sm_86_v5_head.json` with `status="indeterminate_insufficient_data"` after a
      non-zero admitted count fails.
  - AC-6.2: The sm_86 head fit consumes ONLY sm_86 admitted bundles (scoped bundle
    glob); it does not read sm_120 bundles and does not emit cross-arch transfer
    diagnostics. Any per-arch / transfer block in an sm_86-scoped active-path artifact
    records `sm_120` legs as `status="deferred_cross_arch_requires_sm120_session"` with
    a backing ledger citation — NOT as a closure gate.
    - Positive: the sm_86 head's source provenance shows it was fit over the sm_86
      bundle set only; transfer legs (if the artifact carries them) are marked
      deferred with citation.
    - Negative: an sm_86-scoped head whose fit silently mixed in sm_120 bundles, or
      whose transfer leg claims a finite cross-arch number, fails.

- AC-7 (sm_86-scoped closure artifacts): The sm_86 closure artifacts report the sm_86
  slice plus a frozen-sm_120 regression guard. Cross-arch / same-74 slices are
  DEFERRED.
  - AC-7.1: `calibration_head_v5_accuracy_gates.json` (or an sm_86-scoped equivalent)
    carries a `per_slice` block with rows for `sm_86_only` (the admitted sm_86 cells,
    evaluated) and `same_30_frozen_b1646ef7` (REGRESSION GUARD — equals the b1646ef7
    frozen replay within 1e-9, proving the sm_120 fit was not disturbed). The
    `same_74`, `same_30_under_same74_head`, `sm_120_only`, and `holdout_same_74` rows
    are NOT required this loop (deferred to the sm_120 session).
    - Positive: the `sm_86_only` slice is schema-valid; `same_30_frozen_b1646ef7`
      equals the frozen values within 1e-9.
    - Negative: drift in `same_30_frozen_b1646ef7` lifts `frozen_b1646ef7_drift`.
  - AC-7.2: `calibration_head_v5_baseline_comparison.json` (sm_86-scoped) compares the
    head against the inherited baselines (`nugteren`, `cohort_mean`, `always_hit`,
    `uncalibrated_walker`, `v4_platt`) over the sm_86 slice with `n ==
    admitted_sm86_count`. Per-column typed-missing values record a typed reason.
  - AC-7.3: Robustness diagnostics (`calibration_head_v5_robustness.json`, sm_86-scoped)
    re-evaluate LOOCV vs holdout and LOSO over the sm_86 slice.

- AC-8 (cross-arch transfer): **DEFERRED to the sm_120 session.** The inherited v5
  cross-arch transfer gate (epsilon-floored 2× rule, both `(source_arch, target_arch)`
  directions, transfer median ≤ 0.12) cannot be evaluated with a single arch's live
  evidence and is OUT OF SCOPE for this loop. Any sm_86-scoped artifact that carries a
  transfer block records the cross-arch legs as
  `status="deferred_cross_arch_requires_sm120_session"` with a backing ledger citation.
  This loop makes NO cross-arch transfer claim.
  - Positive: the closure artifacts record the cross-arch transfer as explicitly
    deferred with a citation; no finite cross-arch transfer number is asserted.
  - Negative: an artifact claiming a satisfied 2× cross-arch transfer gate (or any
    finite `sm_86_to_sm_120` / `sm_120_to_sm_86` transfer error) on this 3090 Ti loop
    fails — it would be a cross-arch claim the session cannot witness.

- AC-9 (sm_86-slice falsification): The falsification arm runs the inherited
  residual-structure verdict — the joint contribution of
  `(footprint_ratio + long_rd_tail_mass)` to per-bucket residual reduction — over the
  sm_86 admitted slice and records the verdict (`confirmed` ≥30%, `refuted` <10%, or
  `indeterminate` 15-29%) in `calibration_head_v5_falsification.json`. This sm_86-slice
  verdict is INFORMATIONAL; the authoritative same-74 falsification verdict is deferred
  to the sm_120 session and recorded as
  `closure_path="deferred_cross_arch_requires_sm120_session"`.
  - Loop closure does NOT gate on the falsification verdict. The sm_86-only loop closes
    on: (a) the sm_86 head fits cleanly (AC-6), (b) the sm_86-native accuracy slice +
    the frozen-30 regression guard hold (AC-7), (c) AC-5 freeze holds before and after,
    (d) AC-10 debt clear, (e) AC-11 compliance clean.
    - Positive: the sm_86-slice falsification verdict is recorded (any of
      confirmed / refuted / indeterminate), the same-74 verdict is recorded as deferred
      with citation, and the loop's closure gates (a)-(e) hold.
    - Negative: an artifact recording `closure_path="research_success"` or
      `"verdict_falsification"` over a *same-74* slice on this loop fails (that closure
      is the sm_120 session's to make).

- AC-10: Pre-existing pytest debt cleaned (ALREADY COMPLETE; keep green).
  - AC-10.1: `tools/arch_capabilities.json` declares the v5 entry points;
    `tools/test_arch_capabilities_coverage.py` passes.
  - AC-10.2: The Tier-2 `source_commit` pins in
    `tools/sm_86_predictor_accuracy_recovery_v2/` artifacts are refreshed (recapture
    scripts run); `test_artifact_source_commit_matches_predictor_fix_commit` and the
    cell-identity binding test pass.

- AC-11: Compliance gate. CLAUDE.md Rules 1-5 hold throughout.
  - Positive: `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` passes;
    `tools/test_plan_token_scanner.py` passes.
  - Negative: any NCU reference outside the `data/local_labels/` quarantine, any
    GPU-driver-root op, any privileged CUPTI surface, any simulator-revival reference,
    or any plan-token in an implementation file fails the relevant guard.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
All 20 admitted-label sm_86 cells reach `admitted` or a typed terminal status (current:
18 admitted + 2 typed `bundle_failed`); the standalone sm_86 v5 head fits cleanly with
per-bucket fits where `n_train >= 8`; the sm_86-native accuracy slice meets the
inherited v5 single-arch accuracy targets where a 18-cell slice permits; the
`same_30_frozen_b1646ef7` regression guard holds; the sm_86 baseline comparison and
robustness slices land; the sm_86-slice falsification verdict is recorded;
AC-10 debt clear; AC-11 compliance clean; the host-only `pytest tools/
workloads/simbench_bw/tests/` sweep returns 0 failed. Cross-arch transfer, the joint
same-74 head, and the `Cross-arch-verified` status are explicitly out of scope
(deferred to the sm_120 session).

### Lower Bound (Minimum Acceptable Scope)
The regeneration ledger covers all 20 sm_86 label cells with terminal status (current:
18 `admitted` + 2 typed `bundle_failed`). **At least 8 sm_86 cells reach
`status="admitted"`** (DEC-1 floor; currently 18). The standalone sm_86 v5 head fits
(`status="ok"`) over the admitted cells; the `same_30_frozen_b1646ef7` regression guard
holds; AC-5 freeze passes before AND after the head-fit step; AC-10 debt clear; AC-11
compliance clean. The sm_86-native accuracy gates MAY fall short of the inherited
research-success targets (a single 18-cell arch slice is underpowered for some
buckets); the loop still closes because cross-arch research-success / verdict-
falsification is the sm_120 session's deliverable, recorded here as deferred. **If
`admitted_sm86_count < 8` or the AC-5 freeze guard trips, the loop escalates to
operator review instead of closing.** Cross-arch transfer + same-74 closure remain
deferred.

### Allowed Choices
- Can use: `workloads/simbench_bw/run_suite.py --mode trace` as the canonical NVBit
  trace entry point (raw `CUDA_INJECTION64_PATH=` injection only as an implementation
  detail inside the runner); the CUPTI Activity collector via `--enable-cupti-activity`
  for wall-clock timing; the sidecar producers under `tools/sharing_locality/`,
  `tools/hrd_profiler/`, `tools/sector_stream/`, `tools/divergence_proxies/`,
  `tools/order_projections/`; the per-cell bundle builder
  `tools/local_labels/build_predictor_bundle_for_cell.py`; the fitter
  `tools/local_labels/fit_monotonic_calibration_head_v5.py` invoked with a bundle glob
  SCOPED to the sm_86 admitted bundles (to produce the standalone sm_86 head); the
  ledger builder `tools/local_labels/build_sm86_cohort_regen_ledger.py`; NCU on this
  local 3090 Ti for ground-truth labels into `data/local_labels/` ONLY (CLAUDE.md Rule
  1 local-label exception); rebuilding per-slug binaries with `NVCC_CCBIN=g++-14`.
- Cannot use: changes to the monotonic constraint signs, the 6-feature canonical
  schema, the model-class hierarchy, the fold construction protocol, the attribution
  procedure, or the falsification hypothesis (all PRE-REGISTERED v5 deterministic
  designs); fitting a joint same-74 head or mutating `sm_120_v5_head.json` /
  `calibration_head_v5.json` (the sm_120 head + joint head are the sm_120 session's
  deliverable); claiming any cross-arch transfer number or `Cross-arch-verified`
  status; NCU outside `data/local_labels/`; GPU-driver-root operations; privileged
  CUPTI surfaces; CUDA host-driver installation / DKMS / MOK; simulator revival; new
  external benchmark-suite admission.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only.

### Conceptual Approach

1. The sm_86 artifact-production lane (preflight, binary builds, NVBit traces, the six
   sidecars, the two-stage bundle cascade, the regen ledger, the staging manifest) is
   ALREADY COMPLETE and committed (18 admitted + 2 typed `bundle_failed`; ledger +
   manifest closed). The fresh loop's Round 0 re-validates this committed baseline
   (the combined gate is green at 680 passed) and treats it as the `Verified-live-sm_86`
   starting point — it does NOT re-run the live sweeps unless an artifact is found
   missing or drifted.
2. Run the AC-5 freeze test BEFORE the head-fit step to establish the frozen-sm_120
   baseline.
3. Fit the standalone sm_86 head: invoke
   `tools/local_labels/fit_monotonic_calibration_head_v5.py` with a `--bundle-glob`
   scoped to the sm_86 admitted bundles (e.g.
   `data/local_labels/sweeps/sm86_cohort_regen_*/comparison/predictor_bundles/sm_86__*.json`),
   writing an sm_86-scoped head artifact (`sm_86_v5_head.json`). Do NOT invoke the
   joint `regenerate_v5_artifacts.py` wrapper in its default same-74 mode — its default
   glob is sm_120-only and its active-path step computes cross-arch transfer. If the
   wrapper is used, it must be invoked in an sm_86-scoped mode that skips the
   cross-arch transfer + joint-head steps and leaves `sm_120_v5_head.json` /
   `calibration_head_v5.json` untouched.
4. Emit the sm_86-scoped closure artifacts: accuracy gates with the `sm_86_only` slice
   + the `same_30_frozen_b1646ef7` regression guard; baseline comparison over the sm_86
   slice; robustness (LOOCV / LOSO) over the sm_86 slice; the residual-structure
   falsification verdict over the sm_86 slice (informational). Record cross-arch
   transfer + the same-74 falsification verdict as
   `deferred_cross_arch_requires_sm120_session` with backing ledger citations.
5. Run the AC-5 freeze test AFTER the head-fit step to prove no sm_120 drift.
6. Run the host-only pytest sweep. The cross-arch readiness tests
   (`test_run_v5_active_path_comparison.py` dormant-path tests,
   `test_require_both_arches_for_cross_arch.py`, `test_cross_arch_comparison_harness.py`,
   the same-74 rows in `test_v5_cohort_coverage_scanner.py`) are mock/synthetic-fixture
   tests of the FUTURE sm_120 cross-arch paths; they pass today and remain — they are
   NOT closure gates for this loop and need not change.

### Relevant References
- `tools/local_labels/fit_monotonic_calibration_head_v5.py` — fitter (invoke with
  sm_86-scoped `--bundle-glob`).
- `tools/local_labels/build_predictor_bundle_for_cell.py` — two-stage bundle cascade.
- `tools/local_labels/build_sm86_cohort_regen_ledger.py` — ledger producer (8-key
  digest emission; provenance scanner).
- `tools/local_labels/emit_v5_cohort_staging_manifest.py` — staging manifest (closed
  enum; exact invariant).
- `tools/local_labels/test_sm120_cohort_frozen_at_b1646ef7.py` — AC-5 freeze guard.
- `tools/local_labels/regenerate_v5_artifacts.py` — joint wrapper; default glob is
  sm_120-only + active-path step is cross-arch. Use only in an sm_86-scoped mode or not
  at all this loop.
- `tools/local_labels/run_v5_active_path_comparison.py` — cross-arch transfer driver;
  its transfer legs are the sm_120 session's scope.
- `tools/local_labels/emit_v5_closure_artifacts.py` — accuracy gates / baseline /
  robustness / falsification emitter.
- `tools/baseline_runners/nugteren2014_rdh_baseline.py` — Nugteren baseline.
- `workloads/simbench_bw/cache_predictor.py` — walker.
- `data/local_labels/sm_120_v5_head.json` — FROZEN reference (do not touch).

## Dependencies and Sequence

### Lane A: sm_86 artifact regeneration — COMPLETE (re-validated, not re-run)
Preflight, binary builds, NVBit traces, six sidecars, two-stage bundle cascade, regen
ledger, staging manifest. Committed; the fresh loop confirms they are green.

### Lane B: sm_86-scoped head + closure (the loop's mainline work)
Phase B0: AC-5 freeze test (pre-head baseline).
Phase B1: Fit standalone sm_86 v5 head (sm_86-scoped bundle glob).
Phase B2: Codex review of the sm_86 head fit (analyze).
Phase B3: Emit sm_86-scoped closure artifacts (accuracy + baseline + robustness +
sm_86-slice falsification; cross-arch legs recorded deferred).
Phase B4: AC-5 freeze test (post-head verification).
Phase B5: Codex review of the sm_86-slice falsification verdict + deferral
documentation (analyze).

### Lane C: pre-existing debt — COMPLETE (re-validated)
`tools/arch_capabilities.json` declarations; Tier-2 `source_commit` recapture.

### Lane D: compliance closure
Host-only pytest sweep + static guard + plan-token scanner + final Codex review.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | (COMPLETE) Per-cell sm_86 preflight matrix at `data/local_labels/sm86_cohort_preflight.json`. Re-validate replayability. | AC-1 | coding | - |
| task2 | (COMPLETE) Missing sm_86 binaries built with `NVCC_CCBIN=g++-14`; ARCH_GATE-excluded slugs carry typed terminal reasons. | AC-1 | coding | task1 |
| task3 | (COMPLETE) NVBit traces via `run_suite.py --mode trace` under the canonical sm_86 results root. | AC-3 | coding | task2 |
| task4 | (COMPLETE) Six sidecars per cell (sharing_locality, hrd_profile, sector_stream + MPI sibling, divergence_proxies, order_projections); schema + label-MPI validated. | AC-3 | coding | task3 |
| task5 | (COMPLETE) Two-stage bundle cascade (legacy_match + fresh_regen); canonical 6-feature block; identity vs provenance split; 18 admitted, 2 typed `bundle_failed`. | AC-2 | coding | task4 |
| task6 | (COMPLETE) Staging manifest closed enum + `action_typed_reason` split + `legacy_bundle_quarantined` bundle-sum + exact `ready+terminal==cohort_size` invariant. | AC-4 | coding | task1 |
| task7 | (COMPLETE) Regen ledger with 6-tuple key + 8-key `artifact_sha256` + provenance scanner. | AC-1 | coding | task5 |
| task8 | (COMPLETE) sm_120 freeze test `test_sm120_cohort_frozen_at_b1646ef7.py`; invoked before AND after the head-fit step (task9). | AC-5 | coding | - |
| task9 | Fit the STANDALONE sm_86 v5 head over the admitted sm_86 bundles only (`fit_monotonic_calibration_head_v5.py` with an sm_86-scoped `--bundle-glob`), emitting `sm_86_v5_head.json` with `status="ok"` + per-bucket provenance (per-bucket where `n_train>=8`, else pooled_fallback). Do NOT touch `sm_120_v5_head.json` or produce the joint `calibration_head_v5.json`. Run AC-5 freeze before and after. Two-commit landing per BL-20260527 (Commit A: code; Commit B: artifact re-emit with source_commit pinned to Commit A). | AC-6 | coding | task5, task7, task8 |
| task10 | Codex review of the standalone sm_86 head fit: per-bucket provenance, monotonic-constraint adherence, model-class selection, `cohort_scope.admitted_count == admitted_sm86_count`, confirmation that no sm_120 bundle was mixed in and the frozen sm_120 head is untouched. | AC-6 | analyze | task9 |
| task11 | Emit sm_86-scoped closure artifacts: accuracy gates with `sm_86_only` slice + `same_30_frozen_b1646ef7` regression guard (1e-9 equality); baseline comparison over the sm_86 slice; robustness (LOOCV/LOSO) over the sm_86 slice; record cross-arch transfer + same-74 falsification as `deferred_cross_arch_requires_sm120_session` with backing ledger citations. | AC-7, AC-8, AC-9 | coding | task9 |
| task12 | Codex review of the sm_86-slice falsification verdict + the deferral documentation: confirm no cross-arch / same-74 closure is claimed, the deferred legs carry citations, and the loop's closure gates (AC-6 + AC-7 + AC-5) are sound. | AC-9 | analyze | task11 |
| task13 | (COMPLETE) `tools/arch_capabilities.json` v5 entry-point declarations. | AC-10.1 | coding | - |
| task14 | (COMPLETE) Tier-2 `source_commit` recapture (`recapture_l2_reduction_artifacts.py`, `recapture_l2_high_footprint_artifacts.py`). | AC-10.2 | coding | - |
| task15 | Run host-only pytest sweep `uv run --with-requirements requirements.txt --with protobuf python -m pytest tools/ workloads/simbench_bw/tests/ -q`; expected 0 failed; validate `test_no_root_or_ncu_static_guard.py` + `test_plan_token_scanner.py` clean. The cross-arch readiness tests remain (mock-based; testing future sm_120 paths) and must stay green. | AC-10, AC-11 | coding | task13, task14, task11 |
| task16 | Final Codex review of the sm_86-only closure state: per-AC verdict; confirm `Verified-live-sm_86` (NOT `Cross-arch-verified`); ratify the cross-arch deferral. | All | analyze | task15 |

## Cross-Arch Deferral Rationale

Per CLAUDE.md's Two-Device Local Development Model, a round verified live on one arch is
NOT verified on the other; `Cross-arch-verified` requires matching live witnesses on
BOTH the RTX 3090 Ti (sm_86) and the RTX 5060 laptop (sm_120), produced in their own
sessions. Cross-arch verification is therefore inherently multi-session and cannot be
achieved inside this single sm_86 loop. This loop produces `Verified-live-sm_86`
artifacts only. The following are EXPLICITLY DEFERRED to a follow-on sm_120 session and
must NOT be claimed here:

- The joint same-74 calibration head (`calibration_head_v5.json` over 30 sm_120 + the
  sm_86 cells).
- Cross-arch transfer evaluation (the `(sm_86 → sm_120)` and `(sm_120 → sm_86)` 2× rule;
  `per_arch_transfer_diagnostics` cross legs).
- The same-74 closure slices (`same_74`, `same_30_under_same74_head`, `sm_120_only`,
  `holdout_same_74`) in the accuracy gates.
- The same-74 falsification verdict (`closure_path ∈ {research_success,
  verdict_falsification}` over same-74).
- The `Cross-arch-verified` evidence status.

The cross-arch discipline files remain standing as the forcing function that keeps the
two sessions decoupled: CLAUDE.md's two-device model and four evidence statuses,
HANDOFF.md's machine-to-machine transfer protocol, the `test_sm120_loop_claim_guard`
(which forbids cross-arch-verified claims in round summaries), the
`sm86_cohort_regen_ledger.schema.json` `Cross-arch-verified` forward-reference, and the
AC-5 freeze guard. None of these are edited by this loop; they protect the boundary.

## Pending User Decisions

All scoping decisions are resolved.

- DEC-1 (admit floor): `admitted_sm86_count >= 8` REQUIRED for the standalone sm_86
  head fit (no engineering-typed-failure escape); escalate below 8. Currently met (18).
- DEC-2 (thresholds): inherited verbatim from the v5 plan; applied to the sm_86 slice.
- DEC-3 (legacy bundles): two-stage cascade (legacy_match first, fresh_regen fallback);
  unchanged.
- DEC-4 (cohort scope): RESOLVED by operator — this loop is **sm_86-develop-only** with
  a **standalone sm_86 head** (the sm_120 head + joint same-74 head + cross-arch
  transfer + cross-arch falsification + `Cross-arch-verified` status are deferred to the
  sm_120 session). The 30 sm_120 artifacts at `b1646ef7` stay frozen and untouched.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments MUST NOT contain plan-specific terminology
  ("AC-", "Milestone", "Step", "Phase", "Round N", or similar workflow markers). These
  belong in the plan document only. Use descriptive, domain-appropriate naming. The
  plan-token scanner (`tools/test_plan_token_scanner.py`) is binding.
- Round summaries MUST record `Verified-live-sm_86` (or `Offline-verified`) — never
  `Cross-arch-verified` — and must document that cross-arch transfer / same-74 closure
  is deferred, per the `test_sm120_loop_claim_guard` contract.

### Fresh-Loop Mechanics
- The prior loop (session `.humanize/rlcr/2026-05-28_13-24-40/`, rounds 0-34) was
  cancelled; its state is preserved as `cancel-state.md`. A fresh `start-rlcr-loop`
  invocation initializes a NEW session directory + a NEW goal-tracker from THIS revised
  plan. The prior loop's IMMUTABLE goal-tracker section (same-74 cross-arch) is
  historical and is not edited; the fresh loop's new goal-tracker reflects the sm_86-
  only scope above.
- CLAUDE.md and HANDOFF.md are NOT edited — they are durable contracts encoding the
  multi-session cross-arch reality this pivot affirms.

## Output File Convention

This operative plan supersedes the prior "Cross-Arch Same-74 Closure" framing for the
sm_86-develop-only loop. The original v5 design draft is preserved below the
`--- Original Design Draft Start ---` marker for reference (it is the source of the
inherited IMMUTABLE constants); where the draft describes same-74 / cross-arch closure,
the operative plan above narrows the cohort scope to the sm_86 slice and defers
cross-arch work to the sm_120 session.

--- Original Design Draft Start ---

# Footprint-Aware Per-Bucket Monotonic Calibration For Counter-Free L2 Observability

## Goal Description

Replace the brittle hand-tuned multiplicative correction stack in the non-root GPU L2 cache hit-rate predictor with a footprint-aware, per-bucket, monotonically-constrained learnable calibration head ("v5") that consumes a canonical pre-calibration feature vector, fits on the existing 30 sm_120 + 44 sm_86 NCU-labeled cohort, and demonstrates cross-architecture transferability between Ampere (RTX 3090 Ti, sm_86) and Blackwell (RTX 5060 laptop, sm_120). The new head is evaluated head-to-head against a clean-room reimplementation of the Nugteren 2014 reuse-distance-histogram baseline applied to the SAME 74 cells, plus against trivial baselines (uncalibrated walker, current v4 PlattCalibrator, cohort mean, always-hit), so the headline accuracy claim is comparable to published numbers without admitting external benchmark suites in this loop. Suite admission (Rodinia / Parboil / CUTLASS / FlashAttention) is deferred to a sequential follow-on loop. The contribution is framed as **counter-free, non-root L2 observability with calibrated, monotonic, cross-arch-transferable hit-rate prediction**, targeting HPCA 2027 / ISCA 2027 / MICRO 2027 / ASPLOS 2027 cycles. The work lands within one bounded RLCR loop on the existing cohort; engineering closure is decoupled from research-success accuracy gates via a falsification-report path.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Canonical pre-calibration feature vector exposed per-launch by the walker. The 6-feature vector consumed by the new head is `{raw_prediction, footprint_ratio, long_rd_tail_mass, intra_lane_reuse, global_share, gridDim_density}`. `footprint_ratio` is computed as `(unique_sector_count * 32) / cache_model_capacity_bytes`. The existing 3-stage multiplicative correction chain (gridDim penalty + per-slug L1/L2 factors + Platt 2-param head) is retired from the active path; the predictor emits ONE canonical pre-calibration score.
  - Positive Tests (expected to PASS):
    - `observation_builder` emits each feature in the canonical schema for every admitted cell across the 30+44 cohort with no `null` or `missing` placeholders.
    - A schema-validation test against `workloads/simbench_bw/schemas/predictor_bundle.schema.json` (extended to include the 6-feature block) admits all committed bundles.
    - The active comparison_runner consumes ONLY the canonical pre-calibration score and the 6-feature vector — no per-slug factor table import survives in the active path.
  - Negative Tests (expected to FAIL):
    - A synthetic predictor bundle that omits any of the 6 features fails the schema validator.
    - A synthetic active-path call that injects a per-slug factor multiplier raises an `ActivePathDomainMismatchError`.
    - A predictor bundle whose `footprint_ratio` is negative or exceeds 100 fails the per-feature range validator.

- AC-2: v5 per-bucket monotonic calibration head emitted with constrained-regression provenance. `data/local_labels/calibration_head_v5.json` carries `feature_schema`, `model_class_per_bucket`, `monotonic_constraints_by_feature` (only on `used_features`), `fold_assignments` (deterministic stratified, seed=20260527), `selected_hyperparameters`, `dependency_versions`, `training_cell_ids`, `source_commit`, `dependency_availability`, and `composite_score_formula`. v4 is retired from active path; v1/v2/v3 remain historical-only.
  - AC-2.1: Monotonic constraint signs are correct: `raw_prediction = +1`, `footprint_ratio = -1`, `long_rd_tail_mass = -1`, `intra_lane_reuse = +1`, `global_share = 0` (unconstrained), `gridDim_density = -1`. Recorded verbatim in provenance per bucket; only used features have constraint entries.
    - Positive: `calibration_head_v5.json::monotonic_constraints_by_feature` matches the canonical sign table; a sign-direction unit test passes at fit time on synthetic monotonic data.
    - Negative: a perturbation that flips `footprint_ratio` to `+1` makes the synthetic-monotonicity unit test fail.
  - AC-2.2: Pre-registered model-class hierarchy with composite scoring: candidate classes are (a) sklearn `IsotonicRegression` on a frozen linear projection of `{raw_prediction, footprint_ratio, long_rd_tail_mass}` with weights `[+1, -1, -1]` (no learnable projection), (b) `pygam.LinearGAM` with sign-constrained smooths, (c) `lightgbm.LGBMRegressor` with `monotone_constraints` on the 6-feature vector. Selection uses the composite score (primary: mean CV geomean relative error; secondary tiebreaker: median absolute hit-rate error; tertiary: log-loss with `eps=1e-6` clipping and fractional Bernoulli labels). Simpler-model tiebreaker applies on the PRIMARY metric only: within ≤2% primary-score difference, fewer parameters wins. XGBoost is admitted only as a last-resort fallback if both sklearn IR and pygam are unavailable.
    - Positive: the selected model class is recorded in provenance; the composite-score formula matches `(primary, secondary, tertiary)` exactly; the simpler-model tiebreaker rule is recorded verbatim; a fit-time unit test against fixture data converges to the registered hierarchy when the primary score is tied within 2%.
    - Negative: a synthetic provenance file omitting `composite_score_formula` fails the v5 schema validator; a fit run that ignores the simpler-model tiebreaker (selects LightGBM when IR is within 2% on primary) fails the registered-protocol regression test.
  - AC-2.3: Per-bucket minimum-N rule: per-bucket head fits only if N≥8 admitted cells in that `(arch, family_bucket)` pair. Otherwise the bucket falls back to a POOLED monotone head with `family_bucket` as a one-hot covariate; the fallback decision is recorded per-bucket as `kind: 'per_bucket' OR 'pooled_fallback'`.
    - Positive: each per-bucket entry in `calibration_head_v5.json` carries `n_train ≥ 8` for `kind=per_bucket` and `kind=pooled_fallback` for `n_train < 8`.
    - Negative: a synthetic v5 head with `n_train=5` and `kind=per_bucket` fails the v5 schema validator.

- AC-3: Accuracy gate on existing 30+44 cell cohort. The v5 head meets each sub-gate; metrics are unweighted (the AC gate); weighted variants (weight = `log10(n_observations_per_cell)`) are reported as advisory.
  - AC-3.1: Full-cohort median absolute hit-rate error ≤ 0.06.
    - Positive: `comparison_bundle.json::full_cohort_median_absolute_error <= 0.06`.
    - Negative: a synthetic predictor whose median absolute error is 0.10 fails the gate scanner.
  - AC-3.2: Full-cohort geomean relative error ≤ 0.15.
    - Positive: `comparison_bundle.json::full_cohort_geomean <= 0.15`.
    - Negative: synthetic geomean 0.20 fails the scanner.
  - AC-3.3: Holdout geomean relative error ≤ 0.25 (over the 7 sm_120 + 10 sm_86 holdout cells).
    - Positive: `comparison_bundle.json::holdout_geomean <= 0.25`.
    - Negative: synthetic holdout geomean 0.40 fails the scanner.
  - AC-3.4: Per family-bucket with N ≥ 8: p90 absolute hit-rate error ≤ 0.15. Buckets with N < 8 report the metric but the gate is advisory (relaxed; surfaced in the falsification report if breached).
    - Positive: every family-bucket with N ≥ 8 emits `p90_abs_err <= 0.15`; the scanner enumerates and confirms.
    - Negative: a synthetic family-bucket with N=10 and p90 = 0.20 fails the scanner.

- AC-4: Cross-arch transfer gate. The v5 head is fit per-arch and per (source_arch → target_arch) combination; transfer error is reported.
  - AC-4.1: For each `(source_arch, target_arch)` pair `(sm_86 → sm_120)` and `(sm_120 → sm_86)`: `transfer_geomean_relative_error ≤ 2 × max(native_geomean_relative_error, 0.03)` (epsilon floor 0.03 prevents instability when native is near-zero).
    - Positive: comparison_bundle carries `per_arch_transfer_diagnostics` block with both transfer directions satisfying the epsilon-floored 2× rule.
    - Negative: synthetic transfer at 3× native error fails the scanner.
  - AC-4.2: Transfer absolute error still ≤ 0.12 in median.
    - Positive: `transfer_median_absolute_error <= 0.12`.
    - Negative: synthetic 0.18 fails.

- AC-5: Clean-room Nugteren 2014 RDH baseline reimplemented and compared head-to-head on the SAME 74 cells. The baseline lives at `tools/baseline_runners/nugteren2014_rdh_baseline.py`, consumes the same NVBit traces, and emits a 6-column comparison table.
  - AC-5.1: v5 beats Nugteren baseline on median absolute hit-rate error on ≥60% of admitted cells.
    - Positive: `comparison_bundle.json::v5_vs_nugteren_beat_fraction >= 0.60`.
    - Negative: synthetic 0.40 fails.
  - AC-5.2: v5 aggregate median absolute error < Nugteren aggregate median absolute error.
    - Positive: `v5_aggregate_median_abs_err < nugteren_aggregate_median_abs_err`.
    - Negative: synthetic case where v5 ≥ Nugteren fails.
  - AC-5.3: Trivial baselines (`cohort_mean=0.6`, `always_hit=1.0`, `uncalibrated_walker`, `v4_PlattCalibrator`) emitted in the same comparison table for reviewer context.
    - Positive: comparison table has all 6 columns populated for every admitted cell.
    - Negative: a column missing for any cell fails the schema validator.

- AC-6: Robustness gates.
  - AC-6.1: LOOCV residuals and held-out residuals within 20% of each other on the primary score.
    - Positive: `|LOOCV_primary - holdout_primary| / max(LOOCV_primary, eps) <= 0.20`.
    - Negative: a 30% gap fails.
  - AC-6.2: No single-slug removal changes the headline primary error by more than 25%.
    - Positive: leave-one-slug-out scanner emits `max_relative_change <= 0.25`.
    - Negative: a synthetic slug whose removal changes headline error by 40% fails.

- AC-7: Falsification arm, engineering closure, and compliance. The loop closes successfully if EITHER (i) AC-3 + AC-4 + AC-5 all pass (research success) OR (ii) a falsification report is emitted at `data/local_labels/calibration_head_v5_falsification.json` documenting which targets were missed, the per-bucket residual structure, and whether the falsification hypothesis is confirmed or refuted. The compliance test always runs.
  - AC-7.1: Falsification hypothesis testable: `footprint_ratio + long_rd_tail_mass` joint contribution to v4→v5 error reduction ≥ 30% (confirmed) OR < 10% (refuted) per the pre-registered Shapley/attribution procedure (background = per-arch training-set mean; perturbation = permutation-based marginal contribution; metric = absolute hit-rate error reduction vs v4; aggregation = per `(arch, family_bucket)` then averaged; for `IsotonicRegression`-on-projection the attribution is decomposed proportional to `|w_feature| / sum(|w|)` — recorded as an attribution convention, not a full SHAP estimate through the nonlinear isotonic transform).
    - Positive: `calibration_head_v5_falsification.json::joint_contribution_pct` is present and either ≥30 (confirmed) or <10 (refuted).
    - Negative: a contribution of 15-29% is recorded as "indeterminate" and does NOT count as falsification closure.
  - AC-7.2: Compliance — static guard at `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` passes; no NCU outside `data/local_labels/`; no privileged CUPTI / driver-root operations introduced; no simulator path revived.
    - Positive: static guard scans clean.
    - Negative: a synthetic file referencing `nvidia-smi --lock-gpu-clocks` outside the quarantine fails the guard.
  - AC-7.3: Reliability diagnostics emitted as advisory artifacts: per-bucket calibration slope, intercept, and binned expected calibration error (10 equal-width bins). Not a closure gate.
    - Positive: diagnostics file present, schema-valid.
    - Negative: missing diagnostics file fails the schema validator (but does NOT block AC-7 closure if research-success path was taken).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The implementation extends `observation_builder` with the 6-feature canonical pre-calibration vector, retires the current 3-stage multiplicative correction chain from the active path, fits a v5 per-bucket monotonic head selected from a pre-registered hierarchy (IsotonicRegression on frozen projection / pygam monotone GAM / LightGBM monotone GBM), evaluates cross-arch transfer in both directions on the existing 30+44 cell cohort, reimplements a clean-room Nugteren 2014 RDH baseline plus four trivial baselines, emits the 6-column comparison table, runs the falsification-arm permutation/Shapley attribution, emits the reliability diagnostics + bootstrap-CI adjuncts, and records full provenance (feature schema, model class, constraints, fold assignments, dependency versions, training cell IDs, composite-score formula, source_commit, dependency availability). All test infrastructure (positive + negative fixtures) lands alongside the implementation; the existing 533+9 focused suite preserves its passing state.

### Lower Bound (Minimum Acceptable Scope)
The implementation lands AC-1 (canonical feature vector + active-path retirement), AC-2 with sklearn `IsotonicRegression` on the frozen 3-feature projection only (no pygam, no LightGBM evaluation), AC-3 OR a complete falsification report (per AC-7's engineering-closure path), and AC-7 (compliance + falsification arm). The Nugteren baseline reimplementation (AC-5) and trivial baselines may be deferred to the falsification report if a fast convergence path is required, provided the deferral is explicitly documented in the closure summary. Cross-arch transfer evaluation (AC-4) is mandatory because it carries the central novelty claim; under the lower bound the transfer numbers may be reported without meeting the 2×-native-error gate, but they must be emitted.

### Allowed Choices
- Can use: sklearn `IsotonicRegression` (preferred, already pinned); pygam `LinearGAM` with sign-constrained smooths (if dependency available); LightGBM `LGBMRegressor` with monotone constraints (if pygam unavailable); XGBoost (last-resort fallback only). NVBit-derived dynamic sector-stream traces. CUPTI Activity wall-clock timing. CUDA events. Read-only NVML covariates. Static SASS inspection.
- Cannot use: GPU-driver root operations (`nvidia-smi` clock-lock / persistence / power / MIG / compute-mode changes). Privileged CUPTI surfaces (Events / Metrics / Range Profiling / PM Sampling / PerfWorks). NCU outside `data/local_labels/` quarantine. Simulator revival (no `nvasim`, Accel-Sim, or simulator-validation plans as active project direction per CLAUDE.md Rule 5). Cloud GPU access. New external benchmark-suite admission (Rodinia / Parboil / CUTLASS / FlashAttention) in this loop — those are explicitly deferred. Learnable projections in the IsotonicRegression model class (the linear projection weights are FROZEN at `[+1, -1, -1]` to avoid calibration leakage).

> **Note on Deterministic Designs**: The monotonic constraint signs, composite scoring hierarchy, fold construction protocol, attribution procedure, and falsification hypothesis are all pre-registered with fixed values. The model-class selection has hierarchical fallback rather than free choice. The feature vector is fixed at 6 entries. The cohort is fixed at the existing 30 sm_120 + 44 sm_86 cells.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The current predictor pipeline applies three sequential multiplicative corrections to a saturating Smith-Eklund walker, with hand-tuned per-slug factors derived from 2-cell calibration. This is replaced by a one-step learnable head consuming a richer feature vector:

```
walker.cache_predictor.predict(trace):
    # one canonical pre-calibration score
    raw_prediction = stack_distance_walker(trace, modeled_hit_rule)
    # 6-feature canonical vector
    features = {
        raw_prediction,
        footprint_ratio = unique_sector_count * 32 / cache_model_capacity_bytes,
        long_rd_tail_mass,
        intra_lane_reuse,
        global_share,
        gridDim_density,
    }
    bucket = classify_family_bucket(slug)
    head = v5_head_for(bucket, arch)  # per-bucket if N>=8 else pooled_fallback
    return head.predict(features)
```

Fold construction uses deterministic stratified round-robin: for each `(arch, family_bucket)` stratum, sort cells by `hash(seed=20260527, cell_identity_tuple)`, then assign folds by round-robin over the sorted order. Fallback to non-stratified 3-fold if any stratum has fewer cells than the minimum-per-fold criterion; the fallback decision is logged in provenance.

The Nugteren 2014 baseline at `tools/baseline_runners/nugteren2014_rdh_baseline.py` is a clean-room reimplementation of the published reuse-distance histogram model. It is NOT a simulator — it is an analytical predictor that consumes the same NVBit trace, computes RDH per the published equations, and emits a hit-rate prediction. No SASS execution, no thread-warp-block-level simulation. This stays within CLAUDE.md Rule 5.

Falsification arm uses permutation-based feature importance on the v5 model output, with the metric being `(v4_residual_abs - v5_residual_abs)` per cell. For the preferred IsotonicRegression-on-projection model, the linear projection weights provide an analytical decomposition; for GAM/GBM the standard permutation importance is computed via the same protocol.

### Relevant References
- `workloads/simbench_bw/cache_predictor.py` — Smith-Eklund stack-distance walker; the active multiplicative correction stack to retire.
- `workloads/simbench_bw/calibration_head.py` — IsotonicCalibrator + PlattCalibrator + BetaCalibrator scaffolding (lines 151-199, 329-414).
- `tools/local_labels/fit_calibration_head.py` — LOOCV + simplicity-selection scaffolding (lines 277-384).
- `tools/local_labels/fit_calibration_head_v4_sm120.py` — v4 fitter to evolve into v5.
- `tools/local_labels/inventory_cohort_candidates.py::_classify_family_bucket` — family-bucket routing.
- `tools/hrd_profiler/observation_builder.py:286-310` — per-launch working-set sector exports.
- `tools/hrd_profiler/cache_model_spec.py:80-165` — per-arch L2 capacity resolution (sm_86=96MB, sm_120=192MB).
- `data/local_labels/sm120_cohort_family_buckets.toml` — frozen bucket vocabulary.
- `data/sm_86_predictor_accuracy_recovery_v2/cohort_manifest.json` — sm_86 44-cell cohort with split assignments.
- `data/local_labels/label_admitted.json` — 30 sm_120 admitted cells with 15-cell calibration split.
- `plans/nonroot_l2_observability_framework_plan.md` — AC-10 contract for the Nugteren clean-room baseline.
- `workloads/simbench_bw/comparison_runner.py::_bootstrap_ci_over_cells` — 2000-resample bootstrap CI infrastructure.
- `CLAUDE.md` — local-label exception, Verified-live-* evidence statuses, no-simulator rule.

## Dependencies and Sequence

### Milestones

1. **M1 — Canonical feature vector**: extend `observation_builder` and the predictor bundle schema with the 6-feature canonical vector; retire the 3-stage multiplicative correction chain from the active path.
   - Phase A: extend `observation_builder` outputs to include `footprint_ratio`; verify the other 5 features are already exposed.
   - Phase B: extend `predictor_bundle.schema.json` with the new feature block; regenerate the 30+44 committed bundles via the existing pipeline (no new NCU runs required).
   - Phase C: retire the per-slug factor table and the gridDim penalty from the active path in `cache_predictor.py`; emit only the canonical pre-calibration score.

2. **M2 — v5 per-bucket head**: fit and emit `calibration_head_v5.json` using the pre-registered model-class hierarchy and composite scoring.
   - Phase A: implement `PerBucketCalibratorEnsemble` in `calibration_head.py` extending the `_BaseCalibrator` surface; wire fold-stratified round-robin construction.
   - Phase B: implement the three candidate model classes (sklearn IR-on-projection, pygam GAM if available, LightGBM monotone GBM if available); record dependency availability.
   - Phase C: implement composite scoring + simpler-model tiebreaker; select the v5 model class.
   - Phase D: emit `calibration_head_v5.json` with full provenance.

3. **M3 — Cross-arch transfer**: fit per-arch heads on sm_86 and sm_120 calibration splits independently; compute transfer error in both directions.
   - Phase A: implement the per-arch fitting routine; emit per-arch head sidecars.
   - Phase B: implement the transfer-evaluation harness in `comparison_runner.py`; emit `per_arch_transfer_diagnostics`.

4. **M4 — Clean-room Nugteren baseline + trivial baselines**: reimplement Nugteren 2014 RDH at `tools/baseline_runners/nugteren2014_rdh_baseline.py`; emit the 6-column comparison table.
   - Phase A: implement the Nugteren RDH analytical model (no simulator); verify on synthetic fixtures.
   - Phase B: implement the trivial baselines (cohort_mean, always_hit, uncalibrated_walker, v4); emit the comparison table.

5. **M5 — Falsification arm + reliability diagnostics**: permutation-Shapley attribution on `(v4 vs v5)` residual reduction; per-bucket reliability diagnostics; bootstrap-CI adjunct.
   - Phase A: implement the attribution procedure; emit `calibration_head_v5_falsification.json`.
   - Phase B: implement reliability diagnostics + per-cell bootstrap CIs; emit advisory artifacts.

Dependencies: M2 depends on M1 (feature vector must exist). M3 depends on M2 (v5 head must exist per arch). M4 is independent of M2/M3 (can run in parallel). M5 depends on M2 + M3 + M4 (attribution requires the full 6-column comparison + v5 cross-arch heads).

## Task Breakdown

Each task is tagged `coding` (Claude) or `analyze` (Codex).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Extend observation_builder + predictor_bundle.schema.json with the 6-feature canonical vector; backfill footprint_ratio for the 30+44 committed bundles | AC-1 | coding | - |
| task2 | Retire the 3-stage multiplicative correction chain from cache_predictor.py active path; emit one canonical pre-calibration score | AC-1 | coding | task1 |
| task3 | Add positive + negative tests for the feature schema and active-path retirement | AC-1 | coding | task2 |
| task4 | Implement PerBucketCalibratorEnsemble + deterministic stratified round-robin fold construction in calibration_head.py | AC-2, AC-2.3 | coding | task3 |
| task5 | Implement candidate model classes: sklearn IsotonicRegression-on-frozen-projection, pygam LinearGAM (if available), LightGBM LGBMRegressor (if available); record dependency_availability | AC-2.2 | coding | task4 |
| task6 | Implement composite scoring (primary: CV geomean rel err; secondary: median abs err; tertiary: log-loss with eps=1e-6 + fractional Bernoulli labels) + simpler-model tiebreaker on primary; record composite_score_formula | AC-2.2 | coding | task5 |
| task7 | Codex review of the model-class selection protocol + monotonic-constraint signs + fold-construction stratification logic for soundness and small-N safety | AC-2, AC-2.1, AC-2.2, AC-2.3 | analyze | task6 |
| task8 | Emit data/local_labels/calibration_head_v5.json with full provenance (feature_schema, model_class_per_bucket, monotonic_constraints_by_feature, fold_assignments, selected_hyperparameters, dependency_versions, training_cell_ids, source_commit, dependency_availability, composite_score_formula) | AC-2 | coding | task7 |
| task9 | Implement per-arch fitting routine; emit per-arch v5 head sidecars (sm_86_v5_head.json, sm_120_v5_head.json) | AC-4 | coding | task8 |
| task10 | Implement transfer-evaluation harness in comparison_runner.py; emit per_arch_transfer_diagnostics block (sm_86_to_sm_120 + sm_120_to_sm_86) with epsilon-floored 2× rule + absolute error gate | AC-4, AC-4.1, AC-4.2 | coding | task9 |
| task11 | Implement clean-room Nugteren 2014 RDH analytical baseline at tools/baseline_runners/nugteren2014_rdh_baseline.py; verify on synthetic fixtures and confirm no simulator-revival path | AC-5 | coding | task1 |
| task12 | Codex review of the Nugteren reimplementation: equations match published 2014 HPCA paper; no SASS execution / thread-warp-block-level simulation; clean-room provenance documented | AC-5, AC-7.2 | analyze | task11 |
| task13 | Implement trivial baselines (cohort_mean=0.6, always_hit=1.0, uncalibrated_walker, v4_PlattCalibrator); emit the 6-column comparison table in comparison_bundle.json | AC-5.3 | coding | task11 |
| task14 | Run the v5 vs Nugteren vs trivial-baselines head-to-head on the 30+44 cells; emit v5_vs_nugteren_beat_fraction + aggregate medians | AC-5.1, AC-5.2 | coding | task13 |
| task15 | Implement the accuracy-gate scanner enforcing AC-3.1-3.4 (median absolute, full-cohort geomean, holdout geomean, per-bucket p90); emit per-bucket residuals with N<8 advisory flag | AC-3, AC-3.1, AC-3.2, AC-3.3, AC-3.4 | coding | task10 |
| task16 | Implement permutation-based Shapley attribution for footprint_ratio + long_rd_tail_mass joint contribution to v4→v5 error reduction; emit calibration_head_v5_falsification.json with confirmed/refuted/indeterminate verdict | AC-7.1 | coding | task15 |
| task17 | Codex review of the falsification arm's attribution procedure: background distribution, perturbation semantics, metric semantics, aggregation level, isotonic-projection decomposition convention | AC-7.1 | analyze | task16 |
| task18 | Implement robustness scanner: LOOCV-vs-holdout primary-metric gap ≤20% test; leave-one-slug-out scanner with ≤25% headline-error change threshold | AC-6.1, AC-6.2 | coding | task15 |
| task19 | Implement reliability diagnostics: per-bucket calibration slope + intercept + binned expected calibration error (10 equal-width bins on predicted hit rate); per-cell bootstrap CI adjunct (2000 resamples reusing _bootstrap_ci_over_cells) | AC-7.3 | coding | task15 |
| task20 | Run the no-root / no-NCU-outside-quarantine / no-simulator-revival compliance scanners on the v5 active path | AC-7.2 | coding | task19 |
| task21 | Codex final-state review of the v5 head + transfer diagnostics + Nugteren comparison + falsification report + reliability diagnostics + compliance scanners | AC-2 through AC-7 | analyze | task20 |

## Claude-Codex Deliberation

### Agreements
- One bounded RLCR loop on the existing 30 sm_120 + 44 sm_86 cells is the correct scope; deferring Rodinia/Parboil/CUTLASS/FlashAttention keeps the loop honest.
- A single canonical pre-calibration score (replacing the 3-stage multiplicative chain) is the right direction; the training/deployment domain mismatch must be removed completely.
- Exposing `footprint_ratio`, locality features, long-RD tail, and grid-density as a canonical feature vector is grounded in the existing walker/observation surface.
- Cross-arch transfer as the central novelty is stronger than another within-arch calibration result.
- Clean-room Nugteren RDH applied to the same 74 cells is a reasonable like-for-like baseline matching the repository's AC-10 contract direction.
- The falsification arm makes the claim testable rather than just "v5 improved".
- The 2027 venue family (HPCA 2027 / ISCA 2027 / MICRO 2027 / ASPLOS 2027) is the right target.

### Resolved Disagreements
- **Monotonic constraint signs**: Codex caught that the initial proposal had `footprint_ratio = +1` (Claude error). Resolution: signs are now `raw_prediction=+1, footprint_ratio=-1, long_rd_tail_mass=-1, intra_lane_reuse=+1, global_share=0, gridDim_density=-1`, encoded in AC-2.1 with a fit-time sign-direction unit test.
- **Univariate vs multivariate model class**: Codex noted IsotonicRegression is univariate and can't directly consume the 6-feature vector. Resolution: AC-2.2 pre-registers a hierarchy — IsotonicRegression on a FROZEN 3-feature linear projection `[+1, -1, -1]` (preferred), pygam GAM with sign-constrained smooths (secondary), LightGBM monotone GBM (fallback). Projection weights are frozen at registration time, not learned.
- **Small-N risk on per-bucket fits**: Codex flagged per-bucket fitting on 74 total cells as overfit-prone. Resolution: AC-2.3 requires N≥8 per-bucket, otherwise pooled fallback with bucket as one-hot covariate; per-bucket `kind` recorded in provenance.
- **Selection metric alignment with AC gates**: Codex caught CV log-loss alone could win while losing AC-3 absolute-error gates. Resolution: composite score with primary = CV geomean relative error, secondary = median absolute error, tertiary = log-loss; simpler-model tiebreaker on primary only.
- **Engineering closure vs research success**: Codex required the loop to be able to close even if AC-3/AC-4 fail. Resolution: AC-7 widened — closure path (i) all gates pass OR (ii) falsification report emitted; both paths are acceptable engineering closure.
- **Fold-construction stratification**: Codex caught that modulo-hash on `(arch, family_bucket)` doesn't guarantee balanced strata. Resolution: deterministic stratified round-robin — sort cells within each stratum by `hash(seed=20260527, cell_identity_tuple)`, assign folds by round-robin over the sorted order; fallback to non-stratified 3-fold only when a stratum cannot satisfy the minimum-per-fold criterion; both decisions recorded.
- **Falsification attribution procedure**: Codex required precise definition of background, perturbation, metric, and aggregation. Resolution: AC-7.1 records the full procedure verbatim, including the isotonic-projection proportional-weight decomposition as an attribution convention (not a full SHAP through the nonlinear transform).
- **Trivial baselines**: Codex requested cohort_mean, always_hit, uncalibrated_walker, v4_PlattCalibrator alongside Nugteren. Resolution: AC-5.3 mandates the 6-column comparison table.
- **Suite admission this loop vs next**: Codex pushed to lift Rodinia/Parboil to primary. Resolution: deferred to follow-on loop with explicit publication-comparability tradeoff documented in the Deferred Work section below; the Nugteren clean-room baseline is the in-loop substitute.

### Convergence Status
- Final Status: `converged`
- Convergence rounds executed: 3 (max).
- All round-3 REQUIRED_CHANGES were incorporated into v4 (fold-construction stratification, composite scoring alignment, feature-usage clarity, attribution procedure, dependency-absence logging). Round-3 optional improvements (weighted vs unweighted metrics, reliability diagnostics, simpler-model tiebreaker scope clarity) were also folded in. No high-impact DISAGREE remains.

## Pending User Decisions

All decisions resolved by Claude per the project's standing autonomy directive (user is asleep; Claude discusses with Codex and decides). No PENDING items remain.

- DEC-1: Conference target.
  - Claude Position: target HPCA 2027 / ISCA 2027 / MICRO 2027 / ASPLOS 2027 multi-cycle; pick the best fit at submission time.
  - Codex Position: same; 2026 cycles are past (HPCA 2026 was Jan-Feb, ASPLOS 2026 cycles were 2025, MICRO 2026 full-paper was April 7 2026, ISCA 2026 is June-July 2026 happening now). HPCA 2027 abstract July 24, 2026 and paper July 31, 2026 AoE — viable if the loop converges fast.
  - Tradeoff Summary: 2027 cycle gives breathing room; HPCA 2027 is the earliest 2027 venue and most consistent with the architecture framing. ISCA/MICRO/ASPLOS 2027 are later in the cycle and admit broader empirical scope.
  - Decision Status: RESOLVED — target HPCA 2027 first; fall back to ISCA/MICRO/ASPLOS 2027 if the convergence schedule slips past July 2026.

- DEC-2: Model class hierarchy.
  - Claude Position: pre-register a hierarchy with composite scoring; let the holdout-isolated CV decide the winner.
  - Codex Position: conditional accept — composite scoring must be aligned with AC gates and the simpler-model tiebreaker must apply on the primary metric only.
  - Tradeoff Summary: sklearn IsotonicRegression on a frozen projection is the smallest-surface, lowest-risk option; pygam GAM adds expressiveness at the cost of a new dependency; LightGBM monotone GBM is most expressive but riskiest at N=74.
  - Decision Status: RESOLVED — sklearn IR-on-frozen-projection preferred; pygam GAM secondary; LightGBM monotone GBM fallback; XGBoost last-resort. Selection by composite score (primary: CV geomean rel err; secondary tiebreaker: median abs err; tertiary: log-loss); simpler-model tiebreaker on primary only within ≤2% difference.

- DEC-3: Rodinia/Parboil suite admission this loop or deferred.
  - Claude Position: deferred to a sequential follow-on loop to keep this loop bounded.
  - Codex Position: conditional accept — deferring is reasonable but limits external publication comparability; the next-loop suite admission must be explicitly committed.
  - Tradeoff Summary: admitting suites in this loop adds 24-60h of NCU labeling time on a single device and risks loop drift; deferral preserves bounded scope at the cost of weaker cross-paper comparability until the next loop closes.
  - Decision Status: RESOLVED — defer to the next loop; publication-comparability tradeoff documented in the Deferred Work section below; the clean-room Nugteren baseline applied to the same 74 cells is the in-loop comparability substitute.

## Deferred Work (Out of Scope This Loop)

- **Rodinia / Parboil / CUTLASS / FlashAttention suite admission**: requires 24-60h of single-device NCU labeling time and per-kernel build/compatibility work. Scheduled as the next loop. The publication's accuracy claim, when this loop completes, is *"v5 beats clean-room Nugteren 2014 on the L2 hit-rate task across the LCD-portable + simbench_bw cohort with cross-arch validation"* — NOT *"v5 beats Nugteren on Parboil"*. The next loop elevates the claim by admitting Rodinia/Parboil/CUTLASS/FlashAttention and reporting Parboil-comparable MAPE.
- **Calibrated confidence-interval coverage as headline metric**: per-cell bootstrap CIs are emitted as adjunct artifacts (AC-7.3 + comparison_bundle adjunct) but coverage rate is not a closure gate this loop. Promoting to headline framing is a follow-on framing decision tied to the suite-admission loop's findings.
- **Multi-feature SHAP through the nonlinear isotonic transform**: the AC-7.1 attribution uses a proportional-weight decomposition convention for IsotonicRegression-on-projection rather than full SHAP through the nonlinear transform. A full SHAP-via-KernelSHAP-on-isotonic implementation is a follow-on theoretical-rigor extension.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers (per the plan-token scanner at `tools/test_plan_token_scanner.py`).
- The 35-file exemption list in the scanner exists for prior cross-track work; any new files added by this plan must NOT rely on the exemption list and must use domain-appropriate naming.
- Use descriptive, domain-appropriate naming in code: e.g., `monotonic_calibration_head` not `v5_calibration_head_per_milestone_M2`; `footprint_ratio_feature_extractor` not `ac1_feature_extractor`.
- Comments may reference Nugteren 2014, Smith-Eklund, PAV, Kull-Filho-Flach 2017 — these are literature references, not plan tokens.
- Provenance JSON files (`calibration_head_v5.json`, `calibration_head_v5_falsification.json`) carry `source_commit`, `schema_version`, `feature_schema` field names — these are domain artifacts and do not violate the no-plan-token rule.

### Architecture-Coupling Discipline (per CLAUDE.md)
- Every live evidence claim must record device + driver + CUDA toolkit + NVBit build + observed clocks/throttle state.
- A round verified live on one arch is NOT verified on the other; cross-arch claims require matching live witnesses.
- Wire-extension policy: the new `footprint_ratio` feature is added as a producer-side additive field in the predictor_bundle schema; the consumer-side schema validator widening lands in lockstep with the producer-side change.

### Dependency Surface
- sklearn 1.8.0 is already pinned in `requirements.txt`; IsotonicRegression is available.
- pygam (optional, ≥0.9.0) and LightGBM (optional, ≥4.0) are admitted IF they pass the existing repo's dependency-policy check; their absence is recorded in v5 provenance as `dependency_availability.{pygam, lightgbm}: 'unavailable'` and the fallback to the next tier of the hierarchy is logged.
- XGBoost (last resort) is admitted only if sklearn IR + pygam + LightGBM all fail.
- No new GPU-side dependency. No new CUDA toolkit requirement. No new NVBit build.

--- Original Design Draft Start ---

# Per-Bucket Monotonic Calibration For Counter-Free L2 Observability

## Original Idea

Goal: brainstorm a publishable-quality research direction that elevates the current sm_120 L2 observability / hit-rate prediction system to a top-tier conference bar (ISCA / MICRO / HPCA / ASPLOS 2026). The system is at `/home/jakeshea/3090ti_5060laptop_cross_study/non_root_gpu_profiler`; CLAUDE.md Rule 5 prohibits simulator revival, so the angle is observability + prediction, not simulation.

================ CURRENT-STATE RESEARCH FINDINGS ================

(1) PREDICTOR ARCHITECTURE
- Core algorithm: Smith-Eklund stack-distance reuse modeling in `workloads/simbench_bw/cache_predictor.py:1-9`. Walks the NVBit dynamic sector-stream trace via `observation_builder`, extracts per-launch sector reuse distances, applies modeled-hit rule (`rd < capacity_bytes // 32`), aggregates per-launch hit rates via arithmetic mean, applies three sequential multiplicative corrections, clamps to [0,1].
- The three corrections: (a) gridDim-density penalty (log2-scaled subtractive); (b) L2 reduction-kernel correction (per-slug factors 0.60–0.88 for layernorm/softmax/rmsnorm); (c) L2 high-footprint correction (per-slug factors 0.26/0.54 for register_spill/pointer_chase).
- Walker saturates near 0.998 for many cells; high-footprint factors calibrated against only 2 positive controls + 2 negatives — classic 2-cell overfit.
- v4 PlattCalibrator (sigmoid(A*x+B), 2 parameter) fits on 15-cell calibration split; chosen by LOOCV simplicity selection rule.

(2) SOTA-GAP ANALYSIS
- Nugteren 2014 (HPCA): 6-8% mean absolute miss-rate error on Parboil + PolyBench, Ocelot-emulator trace input.
- MDM 2020 (MICRO): 13.9% mean performance prediction error vs 162% for GPUMech.
- Accel-Sim 2020 (ISCA): L2 read NRMSE 0.03 / correlation 1.00 on TITAN V; L1D explicitly hard to correlate.
- Recent ML predictors (MLCRP, SynPerf, NeuSight): none report a clean L2-hit-rate rel-err on a standard suite; mostly end-to-end perf/latency.
- "Counter-free non-root GPU profiling" framing has NO claimant in published lit. NVBit-based profilers (cuThermo, MemSysExplorer) instrument access but don't position as the central observability problem.
- Current system: 50% full-cohort / 189% holdout rel-err — 5-30x worse than published bar.

(3) CALIBRATION ARCHITECTURE FLAWS
- Flaw 1: multiplicative factors break stability when prediction is already close to truth. silu_fwd correctness: predicted=0.22, truth≈0.23, rel_err_before=0.045; after multiplying by factor=0.22, prediction=0.0484, rel_err_after=0.343. Factors are derived as `ceil_to_2dp(geomean(NCU truths))` assuming walker baseline near 1.0, NOT for intermediate-stage predictions.
- Flaw 2: domain-mismatch in PlattCalibrator. Trained on post-gridDim/pre-class-correction values, deployed on intermediate corrected values.
- Flaw 3: fixed-constant per-slug factors are blind to cell-level error magnitude. Round 27 shadow_report shows `per_cell_budget_pass=false` with breaches on silu × 2 + layernorm/timing.
- Candidate replacements (ranked): (i) per-bucket LightGBM regression on {sector-stream stats, footprint ratio, intra-lane-reuse, global_share, gridDim, kernel family}; (ii) per-bucket monotonic Isotonic Regression (PAV algorithm, no overfit risk, but ≥3 pairs per bucket); (iii) per-bucket Platt with residual-magnitude sample-weights to down-weight cells already close to truth.

(4) COHORT + WORKLOAD COVERAGE GAPS
- Current cohort: 30 cells. Memory_stress (10): 6 simbench_bw L1 microbenches + 4 portable_lcd. Tensor_compute (10): 6 HMMA/IMMA GEMMs at 256×256×256 (toy), 2 FlashAttention BH=4 Sq=64 Sk=128 (sub-production), 1 sparse GEMM, 1 conv. Activation_norm (10): RMSNorm/LayerNorm/SwiGLU/Softmax fwd/bwd.
- Available-but-not-admitted buildable pool: 29 simbench_bw L1 microbenches (including sm_120-exclusive WGMMA/QMMA/NVFP4/TMA/DSMEM probes), plus vendor cuBLAS variants.
- Missing for publication-grade scope: production-scale GEMM (4096²), large-seq FlashAttention (Sq=2048-8192), depthwise + grouped conv, ROPE, batch-norm backprop, distributed primitives, dynamic sparsity. Plus Rodinia + Parboil + DeepBench + CUTLASS-shaped GEMM tiles for cross-paper comparability with Nugteren/Accel-Sim baselines.
- Cross-arch coverage gap: cohort claims sm_86+sm_90+sm_120 but only sm_120 has live evidence. sm_86 (RTX 3090 Ti) is documented as "queued for follow-up loop".

================ CONSTRAINTS (NON-NEGOTIABLE) ================
- CLAUDE.md Rule 1: NCU only into `data/local_labels/`; driver code only under `tools/local_labels/`. The local-label exception is the ONLY path to ground truth.
- CLAUDE.md Rule 2-4: no `nvidia-smi` lock-clocks / persistence / power / MIG / privileged CUPTI / driver-root operations.
- CLAUDE.md Rule 5: NO simulator revival. The contribution MUST be observability/prediction framing, not simulation.
- Hardware: 2 local devices only — RTX 3090 Ti (sm_86 Ampere) + RTX 5060 laptop (sm_120 Blackwell). No cloud GPUs accessible.
- Time horizon: aiming for a 2026 submission cycle (ISCA paper deadline typically Nov/Dec; HPCA Aug; MICRO Apr; ASPLOS Sep).

================ ASK ================
Brainstorm a SINGLE coherent research direction that:
1. Closes the accuracy gap to within ~10% rel-err on the publication-standard suite (Rodinia + Parboil + at minimum FlashAttention/CUTLASS).
2. Replaces the brittle hand-tuned multiplicative correction stack with a principled, learnable, monotonic calibration architecture.
3. Demonstrates the counter-free / non-root angle as the central contribution (not a sideshow), with a concrete user-facing claim (e.g., "our predictor recovers NCU-grade L2 hit-rate on locked-counter cloud-GPU-equivalent profiles within 8% absolute error using only CUPTI Activity + NVBit traces").
4. Has cross-arch validation (sm_86 + sm_120 minimum); does NOT require sm_90 / sm_100 / cloud GPUs.
5. Includes a measurable falsifiable hypothesis (so reviewers can't accuse it of just being engineering work).
6. Lands within ONE bounded RLCR loop of work (no multi-loop epics).

Please produce a draft idea with: title, headline claim (1-2 sentences), 3-4 measurable acceptance criteria with numeric targets, the falsification path, the in-scope vs out-of-scope split, and the most-likely-to-trip risks. Keep it tight — under 800 words.

## Primary Direction: Learned Monotonic Calibration Cascade

### Rationale

Replace the hand-tuned multiplicative correction stack (gridDim penalty + per-slug L1/L2 factors + Platt head) with a single learnable per-bucket monotonic calibration model (isotonic regression PAV per family-bucket, or per-bucket LightGBM with monotonicity constraints), fitting on the existing calibration split with the existing label corpus.

### Approach Summary

Replace the hand-tuned multiplicative correction stack (gridDim penalty α + per-slug L1/L2 reduction factors + Platt 2-parameter sigmoid) with a learnable-per-family-bucket architecture:

1. **Core mechanism**: Partition the 30-cell sm_120 cohort by family-bucket (memory_stress / tensor_compute / activation_norm / other) per the existing bucketing contract (`data/local_labels/sm120_cohort_family_buckets.toml`, lines 1–101).
2. **Learnable head per bucket**: Fit a separate monotonic calibrator (per-bucket `IsotonicCalibrator` via `sklearn.isotonic.IsotonicRegression` with PAV algorithm) on each family-bucket's (predicted, label) pairs from the calibration split, replacing the global `PlattCalibrator`.
3. **Inference routing**: At prediction time, classify the input kernel by family-bucket (existing routine in `tools/local_labels/inventory_cohort_candidates.py:_classify_family_bucket`), then apply the trained bucket-specific isotonic head. Buckets with <3 training pairs fall back to a global fallback (per-bucket Beta if N≥3, otherwise the existing global Platt).
4. **Envelope + fail-closed**: Each bucket-head retains the existing envelope-check surface (`training_predicted_min / training_predicted_max`) to route out-of-envelope predictions to typed reasons rather than extrapolating.
5. **Affected components**: `workloads/simbench_bw/calibration_head.py` (add `PerBucketCalibratorEnsemble` class wrapping `dict[bucket_name, IsotonicCalibrator]`; reuse existing `_BaseCalibrator` surface fit + transform). `tools/local_labels/fit_calibration_head_v4_sm120.py` → v5 variant that partitions rows by family-bucket at fit time; spawns per-bucket LOOCV; emits multi-head sidecar with per-bucket provenance. `workloads/simbench_bw/comparison_runner.py` adds family-bucket resolution at prediction time; routes to bucket-specific head or fallback.

### Objective Evidence

- `IsotonicCalibrator` + `BetaCalibrator` already exist in `workloads/simbench_bw/calibration_head.py` (lines 151–199, 329–414): non-parametric PAV + 3-param beta, sharing the same `_BaseCalibrator` interface (fit, transform, envelope tracking).
- Simplicity-selection rule (DEC-T9) pinned in `tools/local_labels/fit_calibration_head.py:_choose_head_simplicity_one_se()` (lines 354–384): existing LOOCV + standard-error gating infrastructure extends to per-bucket selection.
- LOOCV + residual aggregation scaffold in `fit_calibration_head.py:_loocv_residuals()` (lines 277–304) and `_per_head_residual_block()` (lines 329–338): already decoupled from global fit; can be called per-bucket.
- Family-bucket vocabulary established in `data/local_labels/sm120_cohort_family_buckets.toml`: closed enum `{memory_stress, tensor_compute, activation_norm, other}`; mapping already validated by the AC-2.4 family-bucket validator.
- Cohort roster includes `family_bucket` field implicitly via `data/local_labels/sm120_cohort_expansion_roster.toml`; routed through `fit_calibration_head_v4_sm120.py:resolve_effective_splits()` (lines 97–157).
- Dependency surface is minimal: `scikit-learn==1.8.0` already pinned in `requirements.txt` line 34 (allowed per DEC-T8 for IsotonicCalibrator). No LightGBM required; isotonic is sufficient for monotonicity in the small-N regime.
- Existing infrastructure LOC: 421 lines `calibration_head.py` + 666 lines `fit_calibration_head.py` + 581 lines `fit_calibration_head_v4_sm120.py` = 1,668 LOC total. Per-bucket ensemble adds an estimated 150–200 LOC (partition logic + per-bucket sidecar emission + fallback routing).
- Prior art in repo: `calibration_head_v1`, v2, v3, v4 sidecars show iterative calibration refinement. v3 skip (legacy host lacked sklearn) + v4 PlattCalibrator simplicity choice show that head selection is already pluggable.
- Comparison contract pins 7-tuple join key (`comparison_contract.toml` lines 61–71): no family-bucket field required; classification happens inside `comparison_runner` at inference time, keeping the join key stable.

### Known Risks

- **Small-bucket underfitting**: activation_norm or `other` may have <5 training pairs; isotonic regression risks overfitting at N<3. Mitigation: pre-compute minimum-bucket threshold; route small buckets to fallback (global Platt or per-bucket Beta if N≥3).
- **Cross-arch validation gap**: current roster is sm_120-only. Extrapolating per-bucket architecture to sm_86 (RTX 3090 Ti) requires the existing 44-cell sm_86 cohort to be partitioned by the same family-bucket vocabulary; plan must explicitly admit sm_86 as a parallel training arm rather than a deferred follow-up.
- **Domain-mismatch amplification**: per-bucket head sees only intra-bucket (predicted, label) pairs. If gridDim correction or L2-reduction factors are bucket-correlated (e.g., activation_norm over-predicts post-gridDim), the bucket-level isotonic fit may amplify the misalignment rather than smooth it. Requires per-bucket exploratory analysis before committing.
- **Envelope-violation cascade**: if a bucket's training envelope is narrow (e.g., memory_stress saturates near 0.99), holdout cells outside the envelope explode into typed-unavailable counts, reducing per-split residual statistics. Monitor per-bucket envelope coverage in the v5 evaluation block.

## Alternative Directions Considered

### Alt-1: Footprint-Capacity-Aware Reuse Model
- Gist: Extend the Smith-Eklund walker in `cache_predictor.py` with an explicit working-set-to-L2-capacity ratio term so capacity-overflow thrashing is modeled at the source rather than patched downstream. The 4-criterion high-footprint gate (`cache_predictor.py:3205-3268`) already detects long-RD tail mass ≥0.10 as an empirical overflow marker; this proposal lifts that gate into a continuous dampening surface (sigmoid or piecewise-linear function of `(unique_sector_count × 32) / cache_model_capacity_bytes`), trained against the existing 30-cell calibration split.
- Objective Evidence:
  - `tools/hrd_profiler/observation_builder.py:286-310` already exports per-launch `working_set_sector_count`, `unique_sector_count`, `cold_count`, `finite_miss_count`.
  - `workloads/simbench_bw/cache_predictor.py:3205-3268` 4-criterion gate; `cache_predictor.py:3129-3176` per-slug factors 0.26 / 0.54 documented as 2-cell calibration.
  - `tools/hrd_profiler/cache_model_spec.py:80-165` resolves per-arch L2 capacity via CUDA `l2CacheSize` (96 MB sm_86 / 192 MB sm_120).
- Why not primary: Addresses the saturation symptom at the raw-walker layer but inherits the same small-N calibration problem as the primary while introducing additional reuse-distance modeling risk; the primary's per-bucket head subsumes this signal as one input feature without committing to a particular functional form.

### Alt-2: Calibrated Confidence Intervals as the Headline Contribution
- Gist: Reframe the paper's central claim from point-accuracy to coverage-calibrated confidence intervals — "our 95% CI covers the NCU ground-truth value for ≥90% of admitted cells with median width Y" — making predictor usability in admission-gate / optimization-search workflows the headline. Bootstrap CI machinery (2000 resamples, 0.95 confidence, AC-5.3 dual-geomean contract) is already deployed; the missing piece is a per-cell coverage-rate scanner.
- Objective Evidence:
  - `workloads/simbench_bw/comparison_runner.py:_bootstrap_ci_over_cells()` computes 2000-resample log-space CI with pinned seed.
  - `workloads/simbench_bw/schemas/comparison_bundle.schema.json:145-165` already requires `relative_error_lower_diagnostic` / `relative_error_upper_diagnostic` when `predictor_result_kind == "bounded"`.
  - `workloads/simbench_bw/calibration_head.py` documents the Kull-Filho-Flach 2017 BetaCalibrator (extensible to interval output).
  - No `ci_coverage` / `coverage_rate` scanner currently exists; greenfield surface.
- Why not primary: Pure framing pivot; while it complements an accuracy fix, it does not by itself close the 5-30x SOTA gap. Better deployed as a secondary evaluation metric ON TOP of the primary's accuracy improvement.

### Alt-3: Cross-Architecture Transferability Study (Ampere ↔ Blackwell)
- Gist: Train the calibration head on 44 admitted sm_86 cells (RTX 3090 Ti, NCU-labeled, already live in `data/sm_86_predictor_accuracy_recovery_v2/cohort_manifest.json`), then transfer to sm_120 (RTX 5060 laptop, 15-cell calibration split) and report transfer error vs native-arch error as the central cross-arch novelty claim. HANDOFF.md's three-fold completion gate Path 2 explicitly positions sm_120 transfer evaluation as the laptop session's reason to exist.
- Objective Evidence:
  - `data/sm_86_predictor_accuracy_recovery_v2/cohort_manifest.json` carries 44 admitted sm_86 cells with split assignment.
  - `data/sm_86_predictor_accuracy_recovery_v2/cohort_split.json` defines the 24-cell sm_86 calibration split.
  - `tools/local_labels/fit_calibration_head_v4_sm120.py` pins `target_arch=sm_120`; the file naming already anticipates a parallel sm_86 fitter.
  - `CLAUDE.md` defines `Verified-live-sm_86` / `Cross-arch-verified` evidence statuses.
  - `plans/tier2_rd_calibration_cross_arch_plan.md` explicitly scopes the cross-arch transfer follow-up.
- Why not primary: This is the natural EVALUATION modality for the primary, not a competing architectural fix. Folding the sm_86 cohort into the per-bucket calibration training (rather than transferring after-the-fact) is the stronger experimental design.

### Alt-4: Standard-Benchmark-Suite Adoption + Head-to-Head Baseline
- Gist: Admit 3-5 Rodinia (bfs, hotspot, srad) + 2 Parboil (tpacf, mri-gridding) kernels into the cohort, AND reimplement Nugteren 2014's reuse-distance baseline at `tools/baseline_runners/nugteren2014_rdh_baseline.py` (already contracted at AC-10 of `plans/nonroot_l2_observability_framework_plan.md`). This enables direct numeric comparison against the published 6-8% MAPE bar on Parboil.
- Objective Evidence:
  - Zero Rodinia/Parboil kernels currently in the repo (grep yields no matches outside reference documentation).
  - Vendor-kernel adoption precedent in `workloads/simbench_bw/l1/vendor_*` (4 cuBLAS/cublasLt wrapper slugs).
  - `plans/nonroot_l2_observability_framework_plan.md` AC-10 explicitly lists the Nugteren clean-room baseline as a load-bearing comparison.
  - `tools/local_labels/sm120_cohort_buildability_preflight.py` (~220 LOC) supports per-slug preflight including vendor-kernel-style adoption.
- Why not primary: Loop-scope strain. 12-15 new NCU label slots at 2-4 hours each (single-device, CLAUDE.md Rule 1 forbids parallel NCU) = 24-60 hours of measurement. Pre-Volta source patches likely. Better as a sequential follow-on AFTER the calibration architecture is fixed and the predictor demonstrably hits <10% on the existing cohort.

### Alt-5: Information-Theoretic Feature Ablation + Accuracy Ceiling Analysis
- Gist: Systematically ablate the 6-7 NVBit-derived features (sector-RD histogram, footprint, intra-lane reuse, global_share, long-RD tail, CTA interleave, cross-launch carryover) against the NCU label set; derive an empirical Bayes / CV upper bound on what any non-counter predictor could achieve. Transforms the contribution from "we improved accuracy" to "we proved the ceiling and met it" — reviewer-resistant theoretical rigor.
- Objective Evidence:
  - `tools/sm_86_predictor_accuracy_recovery/ac3_ablation_recomputation_helper.py` proves per-slice ablation tooling exists.
  - `data/sm_86_predictor_accuracy_recovery/ac4_per_slice_ablation.json` artifact structure (before/after per cell, geomean_rel_err, headline_moved, per_cell_residual) already established for 5 sub-slices.
  - `data/sm_86_predictor_accuracy_recovery_v2/ac6_missing_feature_candidates.json` enumerates gridDim, blockDim, occupancy, DVFS via paired-delta correlation.
  - sklearn already pinned; scipy/numpy available for permutation/Shapley importance.
- Why not primary: Statistical power is genuinely limited at N=30 (or N=21 sm_86); cross-feature interactions cannot be cleanly isolated. Becomes a stronger fit AS the falsification arm of the primary rather than a standalone direction.

## Synthesis Notes

The five alternatives orbit a shared accuracy-gap target and each folds cleanly into the primary as an extension rather than a competitor. **Alt-1 (footprint-overflow term)** is the most natural fold-in: its capacity-ratio feature plugs directly into the primary's per-bucket input vector and is already implicitly captured by the 4-criterion gate's long-RD tail signal. **Alt-2 (CI coverage)** extends the primary's evaluation rather than competing with it — each per-bucket isotonic head emits LOOCV residuals that bootstrap into per-cell 95% CIs at zero marginal cost, and reporting coverage rate becomes a no-cost adjunct to the headline accuracy table. **Alt-3 (cross-arch transfer)** is the natural validation modality for the primary: train per-bucket heads on the 44-cell sm_86 cohort, deploy on the 15-cell sm_120 calibration split, report transfer-error as the central cross-arch claim, and explicitly use the architecture-coupling discipline in `CLAUDE.md` to guard the claim. **Alt-4 (Rodinia/Parboil + Nugteren baseline)** is the suite-coverage prerequisite for cross-paper comparability; the AC-10 contract at `plans/nonroot_l2_observability_framework_plan.md` already commits to the Nugteren clean-room baseline, and admitting 3-5 Rodinia kernels into the primary's training cohort is a sequential follow-on that should happen in a subsequent loop once the calibration architecture is proven on the existing 30+44 cell base. **Alt-5 (information-theoretic ablation)** becomes the primary's falsification arm: ablating each input feature against the per-bucket isotonic ceiling tells us whether the calibration architecture is information-limited (Alt-5 says ceiling = current error) or merely model-class-limited (Alt-5 says ceiling < current error, primary closes the gap). If a future reviewer prefers a more theoretical paper, Alt-5 becomes the primary and the per-bucket isotonic head reduces to one of the empirical bound estimators it analyzes.

--- Original Design Draft End ---

--- Original Design Draft End ---
