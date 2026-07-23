# Tier-2 RD Calibration + Cross-Arch Transfer Draft (post-research, 2026-05-16)

> Input draft for `/humanize:gen-plan`. This document captures the
> output of a deep research pass (1 Explore opcode-inventory agent
> + 1 general-purpose paper-and-literature agent + 1 codex
> consultation, all run in parallel on 2026-05-16) on the
> reuse-distance threshold calibration approach for the deferred
> Tier-2 metric modeling work and the Blackwell opcode coverage
> gap. The draft does NOT propose to reopen the active RLCR loop
> (`.humanize/rlcr/2026-05-14_22-59-20/`, currently at Round 6
> closing AC-2 launch-provenance); it scopes a SEPARATE follow-up
> loop that lands AFTER the active loop closes its remaining
> mainline AC-3 through AC-9 work. CLAUDE.md Rules 1-5 remain
> binding; `MINIMUM_BIN_SAMPLE_COUNT=50` and
> `CI_HALF_WIDTH_TARGET=0.05` remain pinned per AC-6 negative
> tests; the Two-Device Local Development Model (sm_86 RTX 3090 Ti
> + sm_120 RTX 5060 laptop as equal-citizen local GPUs with NCU
> available; sm_100 cloud B200 as NCU-forbidden target) carries
> through.

## 1. Origin and Authoritative References

### 1.1 Active RLCR loop (precondition for this plan)

The active loop at `.humanize/rlcr/2026-05-14_22-59-20/` is closing
the trustworthy `Verified-live-sm_86` L2-sector-hit-rate
predictor-vs-NCU geomean baseline per
`plans/cache_predictor_l2_trustworthy_baseline_plan.md`. As of
Round 6 (HEAD `2e68c24`):

- AC-1 closed (regeneration-delta scanner + provenance recapture).
- AC-2 closed (5 activation cells re-collected with R28 launch-skip
  fix; pre-R28 labels typed `superseded_by_launch_skip_fix`;
  launch-provenance scanner durable against duplicate entries +
  placeholder `nvbit_build_reason` values).
- AC-3 through AC-9 remain pending (Round B sub-slices, Round C
  microbench rehab, Round Polluter, Round portable_lcd, headline
  validity guard).
- AC-5 calibration head is sub-slice 3 of Round B and is gated
  behind AC-3 (`_walk_launch` diagnostics) + AC-4 (four model
  refinements) demonstrably improving per-cell variance.

This Tier-2 plan is the natural CONTINUATION after the active loop
closes. It MUST NOT preempt or replace the active loop's mainline
work; it queues for a future RLCR session.

### 1.2 Authoritative source documents

- **Active plan**:
  `plans/cache_predictor_l2_trustworthy_baseline_plan.md` (AC-1 through
  AC-11; deferred Tier-2 statement in Goal Description; AC-4 sub-slice
  ordering; AC-5 negative-test ordering; AC-9 last-round placement;
  AC-9 no-bypass discipline; AC-6 pinned constants).
- **Active goal-tracker**:
  `.humanize/rlcr/2026-05-14_22-59-20/goal-tracker.md` (Round 6 status;
  cross-arch sm_120 work explicitly deferred; pinned thresholds).
- **Project rules**: `CLAUDE.md` Rules 1-5; static guard at
  `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`.
  Local-label exception under `tools/local_labels/` and
  `data/local_labels/` for sm_86 + sm_120 only.
- **Two-Device Local Development Model**: `CLAUDE.md` Section
  "Two-Device Local Development Model" — sm_86 (RTX 3090 Ti, Ampere)
  and sm_120 (RTX 5060 laptop, Blackwell consumer/workstation, GB206)
  as equal-citizen local GPUs.
- **Predictor producer**: `tools/hrd_profiler/observation_builder.py`
  (`_walk_launch` lines 132-260; per-slice walker since Round 12 at
  line 391; CTA-interleave Pass-2 path at lines 415-436).
- **Predictor consumer**: `workloads/simbench_bw/cache_predictor.py`
  (`predict_cache_hit_rate` lines 478-751; `_TYPED_REASONS` frozenset
  lines 89-101; bootstrap CI 2000 resamples lines 728-751).
- **Profiler constants**: `tools/hrd_profiler/profiler.py`
  (`MINIMUM_BIN_SAMPLE_COUNT=50` line 46; `CI_HALF_WIDTH_TARGET=0.05`
  line 45; typed reason `cache_model_underconfigured` line 100).

### 1.3 Research-pass deliverables (the input to this draft)

The 2026-05-16 research pass produced three independent reports:

1. **Opcode-inventory report** (Explore agent): inventories
   `tools/nvbit_tracer/isa_def/blackwell_opcode.h` against the
   Blackwell SASS reference; confirms only `OMMA` + `QMMA` are
   recognized from the Blackwell-new set; identifies critical TMEM
   gap (`LDT`/`LDTM`/`STT`/`STTM`/`UTCCP`/`UTCBAR`/`UTC*MMA`/
   `UTCSHIFT`) plus paired-lane FP variants
   (`FADD2`/`FFMA2`/`FMUL2`/`FHADD`/`FHFMA`/`FMNMX3`); flags
   "tcgen03" as not a real NVIDIA term.
2. **Paper survey + literature report** (general-purpose agent):
   identifies `references/papers/knowledge_base/inbox/maeda2017.pdf`
   as the canonical HRD paper that `tools/hrd_profiler/` is named
   after (HPCA 2017, DOI 10.1109/HPCA.2017.11; Algorithm 1 at p. 148);
   identifies `inbox/jiang2022.pdf` as the Maeda-group sharing-aware
   RD follow-on (TPDS 2022) directly relevant to AC-4.4
   (CTA/SM interleaving); confirms inbox CCS 2023 + USENIX Sec 2024
   are off-topic security papers (the latter useful only for L2
   write-policy + non-inclusive-L2 empirical confirmation per
   AC-4.3); 40+ active-corpus papers cataloged.
3. **Codex consultation** (`humanize:ask-codex`, 195s response):
   focused 5-part defensible plan covering threshold
   parameterization, calibration-head choice, fail-closed envelope,
   literature grounding, and round-by-round scope bounds. Saved at
   `.humanize/skill/2026-05-16_14-13-56-278667-748300b1/output.md`.

## 2. Headline Findings That Drive This Plan

### 2.1 H1: `inbox/maeda2017.pdf` IS the HRD paper our project descends from

`tools/hrd_profiler/` is named after Maeda et al.'s Hierarchical
Reuse Distance method. Their paper reports flat RD has 34.7%-55%
mean error on sectored caches; HRD reduces that to 17.5%-18.2%
(Maeda 2017 §V.D Table VI, p. 152). Our `_walk_launch` is currently
a single-layer flat RD with per-slice partitioning bolted on at
line 391; Algorithm 1 at p. 148 is a near drop-in second-layer
extension at the same `O(N log M)` cost.

**Plan implication**: AC-4.1 (slice-geometry-aware effective
capacity) should be reframed as "add a second HRD layer at L2-slice
granularity" using Maeda 2017 Algorithm 1, NOT as yet-another
flavor of set-associativity tuning.

### 2.2 H2: Plan AC-4.1 phrasing in the active plan is partially stale

`observation_builder.py:391` already does
`max(1, capacity_bytes // num_l2_slices // 32)` — per-slice geometry
is in (Round 12 / Round 19 work). What's left for AC-4.1 is the
SECOND HRD layer (per-slice working-set count, mechanistically
grounded by Maeda 2017), not per-slice itself.

**Plan implication**: a 1-line amendment paragraph in the active
plan's AC-4 section corrects the stale phrasing and cites Maeda 2017
§V.D Table VI as the mechanistic justification. This is a non-scope
amendment that lands inside the active loop, NOT in this Tier-2
follow-up.

### 2.3 H3: Blackwell tracer opcode coverage is incomplete

`tools/nvbit_tracer/isa_def/blackwell_opcode.h` recognizes only
`OMMA` (FP4 MMA) + `QMMA` (FP8 MMA) from the Blackwell-new set.
Missing:

- **Memory-touching (critical for RD accounting)**:
  `LDT`/`LDTM` (TMEM load), `STT`/`STTM` (TMEM store), `UTCCP`
  (SMEM→TMEM async copy). On any sm_100 kernel using `tcgen05.*`
  PTX, these silently drop from `_walk_launch`, biasing predicted
  L2 hit rate upward.
- **Compute-only (lower stakes for RD accounting, but unknown-opcode
  risk)**: `UTCSHIFT`, `UTCBAR`, `UTCHMMA`, `UTCIMMA`, `UTCOMMA`,
  `UTCQMMA`, `UGETNEXTWORKID`, `UVIRTCOUNT`, paired-lane `FADD2`,
  `FFMA2`, `FMUL2`, `FHADD`, `FHFMA`, `FMNMX3`.

For the cloud target sm_120 (consumer Blackwell, no TMEM path), the
gap is mostly compute-only. For the sm_100 datacenter target (which
this Tier-2 plan must produce predictions for), the memory-touching
gap is real.

**Plan implication**: a tracer-side opcode-coverage extension is
required before the Tier-2 cross-arch sm_100 dry-run round
(Tier2-E). The extension follows the producer-side wire-extension
policy in CLAUDE.md (additive enum widening; consumer-side schema
must widen in lockstep).

### 2.4 H4: "tcgen03" is not a real NVIDIA term

PTX/SASS publishes `tcgen05` (Blackwell) and `wgmma` (Hopper, the
4th-gen tensor path); 3rd-gen tensor work is Ampere DMMA, already
covered by `OP_DMMA → DP_OP` in `tools/nvbit_tracer/isa_def/ampere_opcode.h`.
This plan treats "tcgen03" as user shorthand for either Ampere DMMA
(already covered) or a typo for tcgen05; no `tcgen03` opcode is
added.

### 2.5 H5: Calibration-head functional form is a real open design question

Codex argues for ridge-regularized monotone Platt-style 2-parameter
head (`y_cal = sigmoid(a · logit(clamp(y_raw)) + b)`, `a ≥ 0`);
literature agent argues for isotonic (PAV) with arch-shift covariate,
falling back to beta calibration (Kull/Filho/Flach AISTATS 2017,
3-param) if isotonic over-fits. The literature agent's argument that
the current `1.0000`-saturation is asymmetric over-confidence (which
Platt's symmetric sigmoid cannot fix) is technically correct; codex's
counter-argument that isotonic at N=13 is one PAV step from
memorization is also correct.

**Plan implication**: the Tier2-B round implements ALL THREE heads
(isotonic, Platt, beta) at modest cost (~50 LOC each); Tier2-D
reports LOOCV residuals for all three and lets the data pick. The
round summary records the winner and why; the AC-5 scanner-test
admits any of the three with a typed enum field.

### 2.6 H6: Inbox papers should be promoted / archived

- Promote `inbox/maeda2017.pdf` → `active/2017__hierarchical_reuse_distance_…`.
- Promote `inbox/jiang2022.pdf` → `active/2022__shalom_condens_sharing_aware_rd_…`.
- Move `inbox/2023-ccs.pdf` → `archive/` (TLB reverse-engineering;
  off-topic for RD calibration).
- Move `inbox/usenixsecurity24-zhang-zhenkai.pdf` → `archive/`
  (timer-free GPU cache attack; off-topic for RD calibration but
  cite under AC-4.3 for write-policy + non-inclusive-L2 empirical
  confirmation).

This is housekeeping that lands in Round Tier2-A or earlier.

## 3. Goal Description for the Tier-2 Follow-Up Plan

Land a defensible Tier-2 RD threshold calibration head and
cross-architecture transfer story for the L2 sector hit-rate
predictor:

1. Add Maeda 2017 HRD second-layer threshold (per-slice working-set
   count) on top of the existing per-slice partitioning, closing
   AC-4.1 with corrected mechanistic semantics.
2. Implement three calibration heads (isotonic with arch-shift
   covariate, ridge-regularized monotone Platt, beta calibration);
   persist trained heads as JSON sidecars with full provenance.
3. Live sm_120 NCU label sweep on the RTX 5060 laptop (the
   `Verified-live-sm_120` cohort that does not yet exist).
4. Fit shared head on pooled sm_86 + sm_120 (predicted, label, arch)
   tuples; emit `calibration_loocv_residuals` block for all three
   heads; record winner; close AC-5 with `Cross-arch-verified`
   evidence.
5. Capture sm_100 NVBit trace in the cloud (NCU-free per CLAUDE.md
   Rule 1); run trained head; emit `cross_arch_zero_shot_prediction`
   block with `result_kind = scalar_zero_shot_inferred`,
   LOOCV-derived confidence band, and
   `transfer_status = extrapolated_no_target_labels`. This is the
   headline cross-arch transfer story.
6. Tracer-side: extend `tools/nvbit_tracer/isa_def/blackwell_opcode.h`
   with the missing TMEM family (`LDT`/`LDTM` → `LOAD_OP`, `STT`/`STTM`
   → `STORE_OP`, `UTCCP` → `LOAD_OP`, `UTC*MMA`/`UTCSHIFT`/`UTCBAR`
   → `TENSOR_CORE_OP`); widen the proto schema enum + offline
   validator in lockstep per CLAUDE.md wire-extension policy.
7. AC-9 widening: add five new typed reasons
   (`calibration_reuse_distance_out_of_envelope`,
   `calibration_covariate_out_of_envelope`,
   `trace_opcode_coverage_incomplete`,
   `unsupported_blackwell_memory_path`,
   `target_arch_unlabeled_extrapolation`); each propagates through
   `headline_validity_failed_reasons` per AC-9 multi-failure capture.
8. Round summaries record device + driver + CUDA toolkit + NVBit
   build + observed clocks per the per-architecture evidence
   discipline; sm_120 round summary records `Verified-live-sm_120`;
   sm_86+sm_120 calibration round records `Cross-arch-verified`;
   sm_100 dry-run round records `Verified-live-sm_100` for the trace
   (NOT for any NCU evidence — Rule 1 absolute on cloud).

## 4. Acceptance Criteria (Tier-2 AC-T1 through AC-T8)

The Tier-2 plan opens fresh AC numbering with the `AC-T<n>` prefix
to avoid collision with the active plan's `AC-<n>` numbering.

### AC-T1 — HRD second-layer threshold (Round Tier2-A)

Add Maeda 2017 Algorithm 1 second-layer to `_walk_launch`; emit
per-launch `reuse_distance_slice_p50/p90/p99` field; commit
`tools/hrd_profiler/device_properties.json` keyed by (arch, sku);
sourced from `cudaDeviceProp` at trace-capture time, committed
alongside the trace, NOT re-read at predict time.

- **Positive**: synthetic kernel that fits in
  `effective_sector_capacity` but exceeds `sectors_per_slice` MUST
  drop predicted hit rate by >5% vs flat-RD predictor.
- **Negative**: a `cache_model_capacity_bytes` value not sourced
  from the manifest MUST be rejected by the new
  `cache_model_geometry_provenance` scanner.
- **Evidence**: `Verified-live-sm_86` + `Verified-live-sm_120`.

### AC-T2 — Three-head calibration infrastructure (Round Tier2-B)

New module `workloads/simbench_bw/calibration_head.py` with
`IsotonicCalibrator`, `PlattCalibrator`, `BetaCalibrator` classes;
each exposes `fit(predicted, label, arch)` and `transform(predicted,
arch) → calibrated`. Persist trained heads as JSON sidecars at
`data/local_labels/calibration_head_v1.json` with provenance:
training cells (slug + shape_point + arch), training-pair count,
training timestamp, library version, trained-head SHA-256.

- **Positive**: monotonicity witness (predicted in → calibrated out
  preserves order for all three heads); envelope-violation witness
  (raw value outside training range fires the typed reason).
- **Negative**: a `calibration_head` commit landing BEFORE AC-T1
  closes MUST be rejected by the round-contract slice ordering
  scanner.
- **Evidence**: `Offline-verified` only.

### AC-T3 — Live sm_120 NCU label sweep (Round Tier2-C)

Run `tools/local_labels/run_headline_cohort_sweep.py` on the RTX 5060
laptop (the second equal-citizen local GPU per the Two-Device Local
Development Model). Emit `data/local_labels/sm_120/` mirror of the
sm_86 label structure with the same 5-tuple `manifest_pairing_index`
schema. Type any label that fails repeatability stddev as
`label_repeatability_failed`. Round summary records device + driver
+ CUDA toolkit + NVBit build + observed clocks per the
`Verified-live-sm_120` evidence requirement.

- **Positive**: at least 5 of the 5 activation cells admit on sm_120;
  microbench cell admission count matches the active plan's AC-6
  outcome on sm_86 (whatever that turned out to be).
- **Negative**: a sweep that omits the device + driver + CUDA toolkit
  + NVBit build + observed-clocks block MUST be flagged by the
  round-summary structural scanner.
- **Evidence**: `Verified-live-sm_120`.

### AC-T4 — Cross-arch calibration head fit + LOOCV (Round Tier2-D)

Pool sm_86 + sm_120 (predicted, label, arch) tuples; fit all three
heads with arch covariate; report LOOCV residuals for all three;
record winner in round summary. AC-9 extension: add
`calibration_loocv_p90_residual_exceeded` typed reason (threshold
pinned by user DEC).

- **Positive**: round summary names the chosen head and the LOOCV
  p50/p90/p99 residuals for all three. The chosen head's calibrated
  predictions for sm_86 cells (via leave-arch-out) lie within X% of
  the sm_86 NCU labels; the chosen head's calibrated predictions for
  sm_120 cells (via leave-arch-out) lie within X% of the sm_120 NCU
  labels. (The actual X is computed by this round and recorded in
  the AC-T4 closure summary.)
- **Negative**: a calibration head whose LOOCV p90 residual exceeds
  the user-pinned threshold MUST emit
  `calibration_loocv_p90_residual_exceeded`; the bundle is still
  emitted, but the calibrated scalar is `typed_unavailable`.
- **Evidence**: `Cross-arch-verified` (this round earns the status).

### AC-T5 — Tracer-side Blackwell opcode coverage extension (Round Tier2-E-prep)

Add to `tools/nvbit_tracer/isa_def/blackwell_opcode.h` (and the
trace-opcode + operation-type enums in lockstep):

- `LDT`, `LDTM` → `LOAD_OP` (or `TENSOR_CORE_LOAD_OP` if the reserved
  enum is activated).
- `STT`, `STTM` → `STORE_OP` (or `TENSOR_CORE_STORE_OP`).
- `UTCCP` → `LOAD_OP` (SMEM-source async copy).
- `UTC*MMA`, `UTCSHIFT`, `UTCBAR`, `UGETNEXTWORKID`, `UVIRTCOUNT`,
  `FADD2`, `FFMA2`, `FMUL2`, `FHADD`, `FHFMA`, `FMNMX3` →
  appropriate compute / control category.

Widen `workloads/simbench_bw/schemas/comparison_bundle.schema.json`
+ `tools/local_labels/import_nvbit_exclusion.py::VALID_OP_CLASSES`
+ `tools/local_labels/import_trace_contract.py` typed-reason set in
lockstep per CLAUDE.md wire-extension policy.

- **Positive**: a synthetic `tcgen05.ld.sync.aligned.*` PTX kernel
  compiled `-arch=sm_100a` is traceable end-to-end without
  unknown-opcode errors; the resulting trace's
  `memory_op_class_counts` includes a non-zero `LOAD_OP` bucket
  attributable to `LDT`.
- **Negative**: a producer-side commit that adds a new opcode
  without widening the consumer-side schema enum MUST be rejected
  by the existing schema-validator test (per the wire-extension
  policy).
- **Evidence**: `Offline-verified` (synthetic kernels and unit
  tests; no live sm_100 hardware required for this round).

### AC-T6 — Zero-shot sm_100 dry-run (Round Tier2-E)

Capture an sm_100 (cloud B200) NVBit trace under the existing
non-root contract (NVBit-via-LD_PRELOAD is the explicitly allowed
surface per `validating-nonroot-gpu-measurement` SKILL.md). Run the
trained AC-T4 head on the trace; emit a
`cross_arch_zero_shot_prediction` block with `result_kind =
scalar_zero_shot_inferred`, LOOCV-derived confidence band, and
`transfer_status = extrapolated_no_target_labels`.

- **Positive**: sm_100 trace is captured without unknown-opcode
  errors (i.e., AC-T5 extension is sufficient for the workloads
  under test); the predicted hit rate is emitted with a
  LOOCV-derived confidence band; the bundle's
  `headline_validity_status = typed_unavailable` and
  `headline_validity_failed_reasons` includes
  `target_arch_unlabeled_extrapolation` (a non-suppressing
  typed reason that flags the absence of target-arch labels but
  does not invalidate the scalar by itself).
- **Negative**: an attempt to mark the sm_100 prediction as
  `Verified-live-sm_100` for any NCU-derived evidence MUST be
  rejected by the static no-NCU guard (Rule 1 absolute on cloud
  GPUs).
- **Evidence**: `Verified-live-sm_100` for the NVBit trace ONLY;
  NO NCU labels are collected on sm_100. The round summary names
  the cloud GPU model (B200), driver, CUDA toolkit, NVBit build,
  and observed clocks per the trace-capture evidence requirement.

### AC-T7 — Headline-validity guard widening (Round Tier2-F, last round)

Widen `workloads/simbench_bw/comparison_runner.py::run_comparison`'s
AC-9 validity-guard with the five new typed reasons (per H3 + codex
consultation):

- `calibration_reuse_distance_out_of_envelope` — fires when the
  input RD distribution falls outside the per-cell calibration
  envelope (e.g., normalized features p99/C_eff,
  working_set_sectors/C_eff exceed the maximum training value).
- `calibration_covariate_out_of_envelope` — fires when NVML clock /
  thermal / throttle covariates fall outside the union of sm_86 +
  sm_120 covariate ranges.
- `trace_opcode_coverage_incomplete` — fires when the tracer
  encounters opcode families the classifier does not recognize.
- `unsupported_blackwell_memory_path` — fires when the missing
  opcode family is memory-touching (per the H3 distinction).
- `target_arch_unlabeled_extrapolation` — fires when the target
  arch has no NCU labels and the prediction is zero-shot.

Add a static-scan test that no calibration-head code path may emit a
bundle without setting both the calibrated `predicted_hit_rate` and
the appropriate envelope arm (no silent extrapolation).

- **Positive**: each of the five new reasons fires under a
  positive-witness test; on a bundle satisfying all envelopes, the
  status is `valid` and `headline_validity_failed_reasons` is the
  empty array.
- **Negative**: any new typed reason that has a suppression flag
  MUST be rejected (per AC-9 no-bypass discipline); a guard
  implementation that records only the FIRST sub-condition that
  fires (instead of ALL fires) MUST fail the multi-failure unit
  test.
- **Evidence**: `Offline-verified` (extension to the existing AC-9
  guard; no new live evidence required).

### AC-T8 — Inbox housekeeping + active plan amendment (Round Tier2-A or earlier)

Per H6:

- Promote `inbox/maeda2017.pdf` → `active/2017__hierarchical_reuse_distance_…`
  with proper rename pattern.
- Promote `inbox/jiang2022.pdf` → `active/2022__shalom_condens_sharing_aware_rd_…`.
- Move `inbox/2023-ccs.pdf` → `archive/` with one-line off-topic note.
- Move `inbox/usenixsecurity24-zhang-zhenkai.pdf` → `archive/` with
  one-line off-topic note that nonetheless cites it for AC-4.3
  write-policy + non-inclusive-L2 empirical confirmation.
- Update `references/papers/knowledge_base/manifest.json`.
- Append a 1-line correction paragraph to the active plan's AC-4
  section (per H2) noting that per-slice geometry already lands at
  `observation_builder.py:391` and that the remaining AC-4.1 work
  is the Maeda 2017 second HRD layer.

- **Positive**: `git diff --name-only` for the housekeeping commit
  shows only the four PDF moves + the manifest update + the active
  plan 1-line amendment; no producer code is touched.
- **Negative**: an attempt to promote an `inbox/` paper without
  recording it in `manifest.json` MUST be rejected by a manifest
  scanner (if one exists; if not, this is a minor side-issue, not a
  blocker).
- **Evidence**: `Offline-verified`.

## 5. Path Boundaries

### 5.1 Upper Bound (Maximum Acceptable Scope)

All eight ACs (AC-T1 through AC-T8) close in dependency order:
Tier2-A (AC-T1 + AC-T8 housekeeping) → Tier2-B (AC-T2) → Tier2-C
(AC-T3) → Tier2-D (AC-T4) → Tier2-E-prep (AC-T5) → Tier2-E (AC-T6)
→ Tier2-F (AC-T7). The headline cross-arch transfer story lands on
sm_100 with full `target_arch_unlabeled_extrapolation` typed-honesty.

### 5.2 Lower Bound (Minimum Acceptable Scope)

The dependency-critical path: Tier2-A (AC-T1 second HRD layer +
AC-T8 housekeeping) → Tier2-B (AC-T2 head infrastructure with
isotonic + Platt only; defer beta) → Tier2-D-restricted (AC-T4 fit
on sm_86-only with `target_arch_unlabeled_extrapolation` flag
perpetually set; defer Tier2-C sm_120 sweep + Tier2-E sm_100
dry-run) → Tier2-F-restricted (AC-T7 widens AC-9 with three of the
five new typed reasons:
`calibration_reuse_distance_out_of_envelope`,
`target_arch_unlabeled_extrapolation`, and
`trace_opcode_coverage_incomplete`; defer
`calibration_covariate_out_of_envelope` and
`unsupported_blackwell_memory_path`). The lower bound delivers a
defensible sm_86-only Tier-2 baseline; the upper bound delivers the
cross-arch story.

### 5.3 Allowed Choices (per-AC)

- AC-T2: head implementations may use scikit-learn isotonic +
  hand-rolled 2-param Platt + hand-rolled 3-param beta; OR three
  hand-rolled implementations to avoid the scikit-learn dep. The
  choice is recorded in the round summary.
- AC-T3: sm_120 sweep may run all 13 cells (5 activation + 8
  microbench) OR the 5 activation cells only with microbenchmarks
  deferred to a Tier2-C-bis round; pinned per user DEC.
- AC-T4: chosen head may be isotonic, Platt, or beta — data picks.
- AC-T5: tracer extension may classify TMEM ops as
  `LOAD_OP`/`STORE_OP` (the simple path) OR activate the reserved
  `TENSOR_CORE_LOAD_OP`/`TENSOR_CORE_STORE_OP` enum (the more
  precise path); the simple path is the lower-bound default per
  YAGNI; the precise path is the upper-bound option.
- AC-T6: sm_100 cloud capture may use any cloud B200 instance the
  user has access to; the round summary records the specific
  instance type + driver + CUDA toolkit + NVBit build.
- AC-T7: typed-reason names are negotiable (codex's names are
  recommendations, not specifications); the round summary records
  the chosen names + the rationale.

### 5.4 Cannot Use

- Lowering `MINIMUM_BIN_SAMPLE_COUNT=50` or `CI_HALF_WIDTH_TARGET=0.05`
  (AC-6 negative test from active plan; carries forward).
- Landing AC-T2 (calibration head infra) before AC-T1 (HRD second
  layer demonstrably improves variance) — AC-T2 negative test
  enforces ordering.
- Landing AC-T7 (validity guard widening) before AC-T6 (zero-shot
  sm_100 dry-run) — AC-T7 placement is non-negotiable (last round).
- Adding any suppression flag to the new typed reasons (AC-9
  no-bypass discipline carries forward).
- NCU on sm_100 cloud (Rule 1 absolute); driver-root operations
  (Rule 2); privileged CUPTI surfaces (Rule 3); host-driver work
  (Rule 4); simulator revival (Rule 5).
- Per-arch heads with ≤13 labels per arch (codex + literature agent
  both reject).
- Neural calibration heads ("overfit theater" per codex; "explicitly
  assumes target labels" per literature agent).
- Using sm_120 alone as authority for sm_100 prediction (consumer
  Blackwell ≠ B200; pool both arches).

## 6. Open DEC Points (User Decides Before Tier-2 Opens)

- **DEC-T1** — Calibration-head functional form: data-driven (all
  three heads implemented; Tier2-D picks winner) vs pin one a-priori
  (Platt per codex OR isotonic per literature agent). Recommendation:
  data-driven (cheap; ~50 LOC × 3).
- **DEC-T2** — Number of new AC-9 typed reasons: codex's full set
  of 5 vs literature agent's restricted set of 2. Recommendation:
  full 5-reason set (more defensible; separates RD envelope,
  covariate envelope, opcode coverage, memory-path coverage,
  target-arch labeling).
- **DEC-T3** — sm_120 sweep cohort size: 5 activation cells only vs
  full 13 cells (5 activation + 8 microbench). Recommendation: full
  13 if microbench cells are stable on sm_120 hardware; 5 activation
  only if microbench rebuild on sm_120 turns out to be a separate
  round.
- **DEC-T4** — sm_100 cloud target: B200 vs the GB202 RTX PRO 6000
  Blackwell mentioned in `validating-nonroot-gpu-measurement`
  SKILL.md (which is sm_120, not sm_100). The active plan and codex
  both refer to "sm_100 cloud B200"; this DEC reconciles which is
  the actual deployment target.
- **DEC-T5** — `calibration_loocv_p90_residual_exceeded` threshold
  pinning: needs a numeric value before AC-T4 closes (codex
  defers; user must pin).
- **DEC-T6** — Tracer-side TMEM op classification: simple path
  (`LOAD_OP`/`STORE_OP`) vs precise path (activate reserved
  `TENSOR_CORE_LOAD_OP`/`TENSOR_CORE_STORE_OP` enum).

## 7. Feasibility Hints

- The active loop's RLCR pattern (failing-test-first, Codex review
  per round, structural scanners, typed-fail-closed, post-commit
  recapture, slice-by-slice ordering) carries forward verbatim.
- Per-slice math at `observation_builder.py:391` is byte-for-byte
  pre-Round-12 collapse-equivalent when `num_l2_slices == 1`,
  preserving offline-validator admissibility on legacy traces.
- The `cta_interleave_enabled` flag at
  `observation_builder.py:415-436` already gates the AC-4.4
  CTA/SM interleaving model; jiang2022's Shalom decomposition
  informs the eventual Pass-2 refinement but is NOT a blocker for
  Tier-2.
- The `Cross-arch-verified` evidence status from CLAUDE.md is
  EARNED by AC-T4 (the round that fits the head on pooled
  sm_86 + sm_120 and reports leave-arch-out residuals).
- The `Verified-live-sm_100` evidence status applies to the NVBit
  trace ONLY in AC-T6 (NOT to any NCU-derived evidence; Rule 1
  absolute on cloud).
- Reading order for the round-by-round implementer: this draft →
  Maeda 2017 §V.D Table VI + Algorithm 1 (p. 148, 152) → codex
  consultation result at
  `.humanize/skill/2026-05-16_14-13-56-278667-748300b1/output.md` →
  active plan's AC-4 / AC-5 / AC-9 sections → the three independent
  research reports.

## 8. Notes on Prior-Art Citations to Carry Forward

- **Mattson, Gecsei, Slutz, Traiger 1970** — IBM Systems Journal —
  stack-distance theoretical foundation.
- **Maeda, Cai, Xu, Wang, Tian 2017** — HPCA — HRD; the project's
  namesake. PROMOTE TO ACTIVE.
- **Jiang, Maeda, Feng, Chen, Chen, Li, Xu 2022** — IEEE TPDS —
  Shalom + Condens; sharing-aware multi-thread RD; informs AC-4.4.
  PROMOTE TO ACTIVE.
- **Nugteren, van den Braak, Corporaal, Bal 2014** — HPCA — GPU
  cache model from RD theory. ALREADY ACTIVE.
- **Niu, Dinda, Memik, Memik 2012** — IPDPS — PARDA parallel RD.
- **Waldspurger, Park, Garthwaite, Ahmad 2015** — FAST — SHARDS
  approximate RD.
- **Ardalani, Lestourgeon, Sankaralingam, Zhu 2015** — ISCA — XAPP
  cross-arch performance prediction. ALREADY ACTIVE.
- **Platt 1999** — sigmoid calibration.
- **Zadrozny, Elkan 2002** — KDD — isotonic calibration.
- **Kull, Filho, Flach 2017** — AISTATS — beta calibration.
- **Villa, Lustig, Yan, Bolotin, Fu, Chatterjee, Jiang, Nellans 2021**
  — HPCA — NVArchSim. ALREADY ACTIVE.
- **Zhang, Cai, Guo, Yao, Gao 2024** — USENIX Security —
  INVALIDATE+COMPARE; cite for L2 write-allocate + non-inclusive
  empirical confirmation only. ARCHIVED.
