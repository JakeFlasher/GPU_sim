# Tier-2 Reuse-Distance Calibration (sm_86 Active Scope; Cross-Architecture Deferred)

> Follow-up to the Trustworthy L2 Sector Hit-Rate Baseline plan
> at `plans/cache_predictor_l2_trustworthy_baseline_plan.md`. This
> plan opens a SEPARATE RLCR loop that lands AFTER the active loop
> at `.humanize/rlcr/2026-05-14_22-59-20/` closes its remaining
> AC-3 through AC-9 work. CLAUDE.md Rules 1-5 remain binding;
> `MINIMUM_BIN_SAMPLE_COUNT=50` and `CI_HALF_WIDTH_TARGET=0.05` at
> `tools/hrd_profiler/profiler.py` lines 45-46 remain pinned per
> the active plan's AC-6 negative tests.

## Active Scope (Per User Directive 2026-05-16)

This plan's ACTIVE SCOPE is **sm_86-only Tier-2 RD calibration
calibrated against local NCU ground truth on the RTX 3090 Ti**.
All sm_120 features and all sm_100 cross-arch features are
**DEFERRED to a follow-up RLCR loop** (`Tier-2-cross-arch`) that
opens only after this plan closes with an accurate, NCU-calibrated
sm_86 cache-behavior model. The user's stated workflow is:

1. **This plan (sm_86 active scope)**: develop, test, and calibrate
   the Maeda 2017 HRD second-layer + calibration-head infrastructure
   on sm_86 alone, with local NCU labels as ground truth. The
   accuracy story is "NCU-calibrated on the 3090 Ti within the
   AC-T4 LOOCV-residual envelope" (per DEC-T5 pinned at 0.30).
2. **Follow-up `Tier-2-cross-arch` loop (deferred)**: extend the
   calibrated sm_86 model to sm_120 (consumer Blackwell, RTX 5060
   laptop) and sm_100 (datacenter B200, cloud, NCU-forbidden).
   This loop opens only after this plan closes; building on the
   sm_86-validated model is a precondition for the more complex
   Blackwell modeling work.

### ACs in Active Scope (sm_86 only)

These ACs land in this plan's RLCR loop:

- **AC-T0** — Loop bootstrap (with active-loop precondition).
- **AC-T1** — Maeda 2017 HRD second-layer threshold (sm_86 only).
- **AC-T2** — Three-head calibration-head infrastructure
  (Isotonic + Platt + Beta) per DEC-T1; scikit-learn permitted
  per DEC-T8; pooled across sm_86 cells per DEC-T12.
- **AC-T4** — sm_86-only calibration fit with LOOCV residuals
  for all three heads; `simplicity_one_se_rule` picks the
  winner per DEC-T9; `target_arch = sm_86_only` flag set; no
  leave-arch-out (single-arch fit); the
  `calibration_loocv_p90_residual_exceeded` typed reason fires
  at LOOCV p90 > 0.30 per DEC-T5.
- **AC-T7** — Headline-validity guard widening with the FOUR
  sm_86-relevant typed reasons:
  `calibration_reuse_distance_out_of_envelope`,
  `calibration_covariate_out_of_envelope`,
  `trace_opcode_coverage_incomplete`,
  `calibration_loocv_p90_residual_exceeded`. The two cross-arch
  reasons (`unsupported_blackwell_memory_path`,
  `target_arch_unlabeled_extrapolation`) defer.
- **AC-T8** — Inbox housekeeping + 1-line active-plan annotation.
- **AC-T9** — Tier-2 RLCR loop discipline (Codex review per round,
  failing-test-first, structural scanners, post-commit recapture).
- **AC-T10** — Per-round shadow-headline diagnostics.

### ACs Deferred to Follow-Up Loop

These ACs do NOT land in this plan; they queue for the
`Tier-2-cross-arch` follow-up RLCR loop:

- **AC-T3** — Live sm_120 NCU label sweep on the RTX 5060 laptop.
- **AC-T5** — NVBit tracer-side Blackwell opcode coverage
  extension (LDT/LDTM/STT/STTM/UTCCP/UTC*MMA/UTCBAR/UTCSHIFT
  plus the new `MemoryOpClass::kTMEM` enum value per DEC-T6).
- **AC-T6** — Zero-shot sm_100 NVBit trace capture and prediction
  emission under the non-root contract; the
  `cross_arch_zero_shot_prediction.schema.json` and
  `Trace-live-sm_100` evidence-status string also defer.
- **AC-T7's two cross-arch typed reasons**:
  `unsupported_blackwell_memory_path` and
  `target_arch_unlabeled_extrapolation`.
- **AC-T4's leave-arch-out evaluation** (requires sm_120 labels
  from AC-T3).

### DEC Resolutions Affected

- DEC-T1, DEC-T2 (four-reason set in this loop), DEC-T5,
  DEC-T8, DEC-T9, DEC-T12, DEC-T13: ACTIVE per the resolutions
  in the Pending User Decisions section.
- DEC-T3, DEC-T4, DEC-T6, DEC-T7, DEC-T10, DEC-T11, DEC-T14:
  RESOLVED-BUT-DEFERRED (the resolution choices stand for when
  the follow-up `Tier-2-cross-arch` loop opens; no work
  happens in this plan).

### Why this matters scientifically

The user's framing is correct: an accurate, NCU-calibrated sm_86
cache-behavior model is a **scientific precondition** for the
more-complex sm_120 + sm_100 work. Without a defensible sm_86
baseline (LOOCV-validated calibration head + Maeda 2017 HRD
second-layer producer), any sm_120/sm_100 extrapolation has no
ground-truth anchor. This plan delivers that anchor; the
follow-up loop builds the cross-arch transfer claim on top of it.

## Goal Description

Land a defensible Tier-2 reuse-distance threshold calibration head
and cross-architecture transfer story for the L2 sector hit-rate
predictor at `workloads/simbench_bw/cache_predictor.py`. The plan
extends `tools/hrd_profiler/observation_builder.py`'s reuse-distance
walker with a Maeda 2017 Hierarchical Reuse Distance (HRD)
second-layer threshold; implements a calibration head module fit
on pooled sm_86 + sm_120 NCU labels with predeclared model-selection
rule; collects sm_120 NCU labels under the local-label exception on
the RTX 5060 laptop; extends the NVBit Blackwell opcode coverage
under the producer-side wire-extension policy in CLAUDE.md to admit
the missing TMEM family (`LDT`/`LDTM`/`STT`/`STTM`/`UTCCP` plus
the compute-only `UTCBAR`/`UTCSHIFT`/`UTC*MMA` variants); captures
an sm_100 NVBit trace under the existing non-root contract (no NCU);
emits a zero-shot sm_100 prediction with explicit
`extrapolated_no_target_labels` typed-honesty; widens the
headline-validity guard at
`workloads/simbench_bw/comparison_runner.py::run_comparison` with
six new typed reasons (five from the prior research synthesis plus
`calibration_loocv_p90_residual_exceeded` from AC-T4). The plan
respects the active loop's RLCR
discipline (failing-test-first, Codex review per round, structural
scanners, typed-fail-closed, post-commit recapture, slice-by-slice
ordering) and extends it with Tier-2-specific shadow diagnostics.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and
negative tests for deterministic verification. Acceptance criteria
use the `AC-T<n>` prefix to avoid collision with the active plan's
`AC-1`..`AC-11` numbering.

- AC-T0: Tier-2 RLCR loop opens with the active L2-baseline
  loop's state DOCUMENTED for traceability. The original
  AC-T0 precondition ("Tier-2 opens only after the active
  loop closes AC-3 through AC-9 of the trustworthy-baseline
  plan") was RELAXED by user directive 2026-05-16 (Round 4
  scope-narrowing) when the user explicitly invoked
  `/humanize:start-rlcr-loop --privacy --skip-quiz
  --track-plan-file plans/tier2_rd_calibration_cross_arch_plan.md`
  with the active loop at Round 6. The relaxation is binding
  per the user's authority over plan precondition policy.
  - Positive Tests:
    - The Tier-2 `goal-tracker.md` IMMUTABLE SECTION names this
      plan file by relative path and references the active plan
      by relative path.
    - The Tier-2 `goal-tracker.md` Plan Evolution Log Round 0
      entry records the active loop's state at Tier-2 open time
      (session id, current_round, HEAD SHA, pending ACs).
    - Tier-2 round contracts queue in dependency order (Tier2-A
      through Tier2-F) per the Dependencies and Sequence section.
  - Negative Tests:
    - A Tier-2 goal-tracker IMMUTABLE SECTION that omits the
      active-loop-state Plan Evolution Log entry MUST be
      flagged at Round 0 close.
    - A Tier-2 round contract that lands AC-T2 before AC-T1's
      synthetic witness MUST be rejected by the round-contract
      structural scanner.

- AC-T1: Maeda 2017 HRD second-layer threshold lands in
  `tools/hrd_profiler/observation_builder.py` for the sector cache
  decision specifically. The line / allocation / region layers
  remain unchanged; only the sector layer gains the second HRD
  level per Maeda 2017 Algorithm 1 (HPCA 2017 p. 148). This is a
  NEW Tier-2 surface and does NOT retroactively close the active
  plan's AC-4.1 — the active loop owns its own AC-4.1 closure
  using whatever per-slice / per-set semantics it pins at close
  time.
  - AC-T1.1: Second HRD layer implementation. Definitions:
    - `effective_sector_capacity` is the SINGLE-LAYER (flat-RD)
      sector-count threshold at which the existing walker
      transitions sector accesses from "hit" to "miss"; i.e.,
      `total_l2_capacity_bytes // 32` in the legacy
      `num_l2_slices == 1` collapse path.
    - `sectors_per_slice` is the PER-SLICE sector-count
      threshold introduced by the existing per-slice walker at
      `tools/hrd_profiler/observation_builder.py` line 391:
      `max(1, capacity_bytes // num_l2_slices // 32)`.
    - Maeda 2017 second HRD layer adds a SECOND threshold check
      against `sectors_per_slice` AFTER the first layer's
      `effective_sector_capacity` check; an access that is a
      hit at the first layer can become a miss at the second
      layer if its per-slice working-set count exceeds
      `sectors_per_slice` (the multi-granularity locality
      effect Maeda's Algorithm 1 captures, p. 148).
    - Positive: a synthetic kernel that fits in
      `effective_sector_capacity` (single-layer hit) but
      exceeds `sectors_per_slice` (second-layer miss) SHOULD
      drop the predicted hit rate by at least 5 percentage
      points relative to the single-layer flat-RD baseline at
      the active loop's close commit. The 5-percentage-point
      figure is an OPTIMIZATION TARGET (per user DEC), not a
      hard close-gate: the round summary records the actual
      observed delta; deltas smaller than 5pp do NOT block
      round closure but trigger a `shadow_hrd_layer_low_impact`
      diagnostic in the round's `shadow_headline_diagnostics`
      block per AC-T10. The actual delta value and its
      synthetic-fixture identity are committed alongside the
      AC-T1.1 round artifacts so AC-T4 has the data to
      evaluate whether the second HRD layer materially
      contributed to predictor variance.
    - Positive: a synthetic kernel whose access stream is exactly
      the active loop's last regression-locked
      `CtaInterleaveAggregationTest` fixture continues to produce
      the same per-launch hit-rate values (additive HRD layer must
      not regress the single-layer behavior on existing fixtures).
    - Negative: a refactor that changes per-launch hit rate on the
      single-layer fixture MUST cause the regression-lock witness
      to fail.
  - AC-T1.2: Device-properties manifest provenance
    - Positive: `tools/hrd_profiler/device_properties.json` is
      committed alongside any new sweep, keyed by `(arch, sku,
      driver_version, cuda_toolkit_version, nvbit_build)`; values
      are sourced from `cudaDeviceProp` at trace-capture time and
      committed alongside the trace.
    - Positive: every new predictor bundle records
      `cache_model_geometry_provenance` as a structured object
      with `l2_capacity_bytes`, `sector_size_bytes`,
      `num_l2_slices`, `num_l2_slices_source`, `slice_hash_model`,
      `write_policy_model`, `cta_interleave_model`,
      `geometry_confidence`, and the manifest sha256.
    - Negative: a `cache_model_capacity_bytes` value not sourced
      from the manifest MUST be rejected by a new
      `cache_model_geometry_provenance_scanner` test (modeled on
      `tools/local_labels/test_admitted_label_launch_provenance_scanner.py`).
    - Negative: a manifest entry missing any of `(arch, sku,
      driver_version, cuda_toolkit_version, nvbit_build)` MUST be
      rejected by the manifest-shape scanner.

- AC-T2: Calibration-head module infrastructure at
  `workloads/simbench_bw/calibration_head.py` with predeclared
  model-selection rule and predeclared cross-arch parameterization
  (per DEC-T12; see Pending User Decisions). The default
  parameterization is "pooled global head with `arch` recorded
  ONLY as evaluation metadata" — `arch` is NOT a fitted feature;
  the calibrator sees only `(predicted, label)` pairs. Per-arch
  residuals are reported in the AC-T4 evaluation block but the
  fit itself is single-arch-blind. This avoids implicit per-arch
  heads and matches the small-N regime (≤26 total cells across
  both arches).
  - AC-T2.1: Calibrator implementations
    - Positive: the module exposes at minimum `IsotonicCalibrator`
      and `PlattCalibrator` classes; each exposes
      `fit(predicted, label, weights=None)` and
      `transform(predicted) -> calibrated`. The `arch` argument
      is NOT a fitted feature per the AC-T2 default
      parameterization (DEC-T12); it is recorded ONLY in the
      training-row provenance under AC-T2.2 and surfaces in the
      AC-T4 leave-arch-out evaluation block.
      `BetaCalibrator` is implemented per DEC-T8 outcome
      (omitted under the restricted-set decision; included under
      the full-set decision).
    - Positive: monotonicity witness — for any input array
      sorted ascending, the calibrated output array is also
      sorted ascending (each calibrator).
    - Positive: envelope-violation witness — an input value
      outside `[min(training_predicted), max(training_predicted)]`
      causes `transform` to emit an out-of-envelope flag the
      consumer can route to the typed-reason path under AC-T7.
    - Negative: a calibration-head commit landing BEFORE AC-T1
      synthetic witnesses pass MUST be rejected by the
      round-contract slice-ordering scanner.
  - AC-T2.2: Trained-head sidecar provenance
    - Positive: `data/local_labels/calibration_head_v1.json`
      records training cells (slug + shape_point + arch),
      training-pair count, training timestamp, calibrator
      library version (or `stdlib_only=true`), trained-head
      SHA-256, comparison_contract.toml pin (sha256), the
      7-tuple `manifest_pairing_index` join keys for every
      training row, and the source commit at training time. The
      sidecar shape is enforced by a NEW JSON Schema at
      `workloads/simbench_bw/schemas/calibration_head_v1.schema.json`
      with `additionalProperties: false`, mirroring the
      discipline of the existing `comparison_bundle.schema.json`.
    - Positive: a new
      `tools/local_labels/test_calibration_head_provenance_scanner.py`
      admits a structurally valid sidecar and rejects:
      missing-field defects, training rows that fail the 7-tuple
      join against `data/local_labels/label_admitted.json` and
      the corresponding predictor bundles, and a
      `comparison_contract.toml` sha256 that does not match the
      committed contract.
    - Negative: a sidecar whose training row cannot be joined
      back to admitted labels and predictor bundles by the
      7-tuple key (per `workloads/simbench_bw/comparison_runner.py`
      lines 99-107) MUST be rejected with typed defect
      `calibration_training_row_unjoinable:<row_index>`.

- AC-T3 **[DEFERRED TO Tier-2-cross-arch follow-up loop]**: Live
  sm_120 NCU label sweep on the RTX 5060 laptop under
  the local-label exception (CLAUDE.md Rule 1). Per Round 4 user
  directive, this AC does NOT open in this plan; the criteria
  below define the AC's shape for when the follow-up loop opens.
  - Positive Tests:
    - `tools/local_labels/run_headline_cohort_sweep.py` runs on
      sm_120 and emits `data/local_labels/sm_120/` artifacts
      mirroring the sm_86 label structure; every emitted row
      carries the same 5-tuple `manifest_pairing_index` schema
      that sm_86 rows use (per
      `tools/adaptive_sampler/trace_convergence_ledger.py:177-188`).
      The slug names retain the existing `*_sm80` pattern (e.g.,
      `layernorm_fwd_bf16_sm80`) since the slug names denote the
      compile-time PTX/cubin arch ABI under which the workload
      was originally tuned, not the runtime device — Blackwell
      sm_120 forward-compatibly executes sm80 ABI cubins per
      `.claude/knowledge/blackwell/blackwell-compatibility.md`.
      The actual run arch is recorded separately in the row's
      `arch` provenance field (`sm_120`).
    - At minimum the 5 activation cells (layernorm,
      rmsnorm, silu_correctness, silu_timing, softmax_safe)
      admit on sm_120; the microbench cell admission count is
      pinned by DEC-T3 (see Pending User Decisions).
    - The Tier2-C round summary records GPU model (RTX 5060
      laptop), driver version, CUDA toolkit version, NVBit build,
      observed clocks, and throttle state per the
      `Verified-live-sm_120` evidence requirement in CLAUDE.md.
    - A new `data/local_labels/sm_120/label_admitted_launch_provenance.json`
      sidecar records `selected_warmup_count`,
      `selected_cache_control`, `selected_launch_skip`,
      `selected_n_repeats`, `sweep_dir_id`, `verdict_path`, and
      `nvbit_build_status` / `nvbit_build_reason` for every
      admitted activation row (mirrors the sm_86 sidecar shape
      from active-loop Round 5).
    - `tools/local_labels/test_admitted_label_launch_provenance_scanner.py`
      is extended to ADMIT a parallel sm_120 manifest with the
      same five activation slug+shape identities, with a typed
      `arch=sm_120` discriminator on the entry-count and
      duplicate-detection clauses (preserves the active loop's
      Round 6 hardening — duplicate-entry rejection,
      placeholder-reason rejection, count-of-five enforcement).
  - Negative Tests:
    - A re-join attempt that silently mixes sm_86 and sm_120
      rows MUST fail the
      `tools/local_labels/select_best_config_labels.py` rogue-
      stale filter (existing arch-mismatch rejection).
    - A round summary that omits device + driver + CUDA toolkit
      + NVBit build + observed-clocks block MUST be flagged by
      the round-summary structural scanner.
    - The extended launch-provenance scanner MUST reject a
      sm_120 manifest with fewer than five activation entries
      AND MUST reject a sm_120 manifest with placeholder
      `nvbit_build_reason` values (same denylist as sm_86 from
      active-loop Round 6: `n/a`, `na`, `none`, `not applicable`,
      `unavailable`, case-folded after strip).

- AC-T4: Cross-arch calibration head fit on pooled sm_86 + sm_120
  (predicted, label, arch) tuples with predeclared selection rule.
  - Positive Tests:
    - The Tier2-D round emits a `calibration_evaluation_v1.json`
      block recording: per-head LOOCV residual table (p50, p90,
      p99 for each of `IsotonicCalibrator`, `PlattCalibrator`,
      and (per DEC-T8) `BetaCalibrator`), per-head
      leave-arch-out residual table (train sm_86 score sm_120;
      train sm_120 score sm_86; explicit two-fold-only honesty
      labelled `n_arches=2`), the uncalibrated-baseline residual
      table for the same cells, and the chosen head's name plus
      the rationale token from the predeclared selection rule.
    - The selection rule (per DEC-T9) is recorded by name in
      the round contract before the round opens (one of:
      `loocv_p90_minimum`, `leave_arch_out_p90_minimum`,
      `simplicity_one_se_rule`, `predeclared_isotonic`,
      `predeclared_platt`); the round summary cites the rule
      name and the rule's tie-breaker.
    - The chosen head's calibrated predictions for sm_86 cells
      via leave-arch-out lie within X% of the sm_86 NCU labels;
      the chosen head's calibrated predictions for sm_120 cells
      via leave-arch-out lie within X% of the sm_120 NCU labels.
      X is computed by this round and recorded in the AC-T4
      closure summary; the round contract pins an upper-bound
      threshold per DEC-T5 below which the round may close.
    - AC-9 widening (lockstep with AC-T7): a
      `calibration_loocv_p90_residual_exceeded` typed reason
      fires when the chosen head's LOOCV p90 residual exceeds
      the DEC-T5-pinned threshold; the bundle is still emitted,
      with a calibrated-scalar arm of `typed_unavailable`.
  - Negative Tests:
    - A round that reports only the chosen head's residuals
      (omitting the other heads or the uncalibrated baseline)
      MUST be flagged by the round-summary structural scanner.
    - A round that names a selection rule different from the
      one pinned in the round contract MUST be rejected by the
      contract scanner.
    - A calibration head that produces non-monotone behavior
      on any LOOCV fold MUST fail the per-head monotonicity
      witness (carried forward from AC-T2.1).

- AC-T5 **[DEFERRED TO Tier-2-cross-arch follow-up loop]**: NVBit tracer-side Blackwell opcode coverage extension
  for the missing TMEM family. Lands as a producer-side additive
  enum widening per the wire-extension policy in CLAUDE.md
  (consumer-side schema widens in lockstep).
  - AC-T5.1: Producer-side opcode classification
    - Positive: `tools/nvbit_tracer/isa_def/blackwell_opcode.h`
      gains `LDT`, `LDTM`, `STT`, `STTM`, `UTCCP`, `UTCSHIFT`,
      `UTCBAR`, `UTCHMMA`, `UTCIMMA`, `UTCOMMA`, `UTCQMMA`,
      `UGETNEXTWORKID`, `UVIRTCOUNT`, `FADD2`, `FFMA2`, `FMUL2`,
      `FHADD`, `FHFMA`, `FMNMX3` mappings.
    - Positive: `tools/nvbit_tracer/isa_def/trace_opcode.h`
      enum is widened with corresponding `OP_*` values (additive
      to the end of the existing enum list).
    - Positive: `tools/nvbit_tracer/tracer_tool/memory_op_class.h`
      and
      `tools/nvbit_tracer/tracer_tool/memory_op_access_kind.h`
      classifiers are extended so the new memory-touching opcodes
      route to the correct `MemoryOpClass` and
      `MemoryOpAccessKind` (TMEM ops route per DEC-T6 below; the
      compute-only opcodes route to the existing compute path).
    - Positive: the proto schema at
      `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto`
      is updated additively (new enum values appended to existing
      `MemoryOpClass` and `MemoryOpAccessKind` per the wire-
      extension policy; old readers stay forward-compatible).
      Generated bindings (`.pb.cc` / `.pb.h` / `_pb2.py`) are
      regenerated as part of the build, NOT committed unless
      the project already tracks generated bindings (which it
      does NOT today; the regeneration step lives in the build
      recipe). The C++ unit tests at
      `tools/nvbit_tracer/tracer_tool/tests/memory_op_class_test.cc`
      and `memory_op_access_kind_test.cc` admit the new opcodes
      with positive controls.
    - Positive: a synthetic `tcgen05.ld.sync.aligned.*` PTX
      kernel compiled `-arch=sm_100a` (offline, no live sm_100
      hardware required) emits SASS containing `LDT` per
      `cuobjdump --dump-sass`; `nvdisasm` confirms the opcode
      mnemonic; the AC-T5.1 `memory_op_class_test.cc` and
      `memory_op_access_kind_test.cc` unit tests admit `LDT`
      with the correct `MemoryOpClass` and `MemoryOpAccessKind`
      classification per DEC-T6 and DEC-T14 outcomes. End-to-end
      live NVBit trace execution on sm_100 hardware is deferred
      to AC-T6 (offline witness here is SASS-presence and
      classifier-correctness, not tracer-execution).
    - Negative: a producer-side commit that adds a new opcode
      without widening the `MemoryOpClass` enum in
      `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto`
      AND without updating
      `tools/nvbit_tracer/import_nvbit_exclusion.py` MUST be
      rejected by the existing exclusion-taxonomy parity test
      at `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`
      (or an extension of it).
  - AC-T5.2: Consumer-side schema widening + headline propagation
    - Positive: the JSON Schema enums at
      `workloads/simbench_bw/schemas/comparison_bundle.schema.json`,
      `workloads/simbench_bw/schemas/nvbit_unsupported_memory_path.schema.json`,
      and
      `tools/nvbit_tracer/import_nvbit_exclusion.py::VALID_OP_CLASSES`
      are widened in lockstep; an offline schema-validator test
      asserts every newly-emitted bundle and exclusion sidecar
      passes schema validation against the widened enums.
    - Positive: the `unsupported_memory_path.json` exclusion
      sidecar carries any opcode the classifier now recognizes
      under a typed `path_resolved` arm (so downstream consumers
      can distinguish "newly classified" from "still unknown").
    - Positive: the headline-validity guard (lockstep with
      AC-T7) PROMOTES the unsupported-memory-path sidecar into
      `headline_validity_failed_reasons` when the sidecar's
      typed reason is `unsupported_memory_path` AND the missing
      opcode is memory-touching per the AC-T5.1 classifier
      tables.
    - Negative: an offline schema-validator test failure on the
      widened enum MUST cause the AC-T5.2 round to not close.
    - Negative: a producer-side classifier change without a
      lockstep schema enum widening MUST be rejected by the
      schema-validator test (per the CLAUDE.md wire-extension
      policy).

- AC-T6 **[DEFERRED TO Tier-2-cross-arch follow-up loop]**: Zero-shot sm_100 NVBit trace capture and prediction
  emission under the non-root contract (no NCU; CLAUDE.md Rule 1
  absolute on cloud GPUs).
  - Positive Tests:
    - A cloud B200 NVBit trace is captured via
      `LD_PRELOAD=tools/nvbit_tracer/tracer_tool/tracer_tool.so`
      + `CUDA_INJECTION64_PATH=...` per the existing non-root
      contract documented in
      `.claude/knowledge/measurement/nvbit-injection-playbook.md`;
      the AC-T5 opcode coverage extension is sufficient for the
      workloads under test (no unknown-opcode errors emitted).
    - The trained AC-T4 head consumes the sm_100 trace and emits
      a NEW artifact at
      `data/cross_arch_predictions/sm_100/<sweep_id>/cross_arch_zero_shot_prediction.json`
      with a NEW schema at
      `workloads/simbench_bw/schemas/cross_arch_zero_shot_prediction.schema.json`
      (separate from `comparison_bundle.schema.json` to avoid
      polluting the labeled-arch enum surface). The artifact
      records `result_kind = scalar_zero_shot_inferred` (NEW
      enum value local to this NEW schema; no widening of the
      existing `result_kind: ["admitted", "typed_unavailable"]`
      in `comparison_bundle.schema.json`), the calibrated
      predicted hit rate, the LOOCV-derived confidence band
      from AC-T4, the `transfer_status =
      extrapolated_no_target_labels` field, and the AC-T7
      `headline_validity_failed_reasons` array including
      `target_arch_unlabeled_extrapolation`.
    - The Tier2-E round summary records the cloud GPU model
      (B200), driver, CUDA toolkit, NVBit build, and observed
      clocks under a NEW evidence-status string `Trace-live-sm_100`
      (NOT `Verified-live-sm_100` — the latter would conflate
      NVBit-only trace evidence with NCU-derived evidence the
      cloud rule forbids; per DEC-T7 below). The round summary
      explicitly cites CLAUDE.md Rule 1 as binding for sm_100.
  - Negative Tests:
    - Any sm_100 cloud artifact under
      `data/cross_arch_predictions/sm_100/` containing a
      `.ncu-rep` or `.ncu-repz` file, or any reference to `ncu`
      or `nv-nsight-cu-cli`, MUST cause the existing static
      guard at
      `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`
      to fail (the existing guard already covers all paths
      outside the `tools/local_labels/` and `data/local_labels/`
      quarantine; the AC-T6 negative test asserts the guard
      catches sm_100-cloud artifacts).
    - A round summary that names the sm_100 evidence as
      `Verified-live-sm_100` MUST be flagged by the
      round-summary structural scanner (DEC-T7 RESOLVED to
      admit `Trace-live-sm_100` only; `Verified-live-sm_100`
      remains forbidden).
    - An attempt to emit the zero-shot artifact through
      `comparison_bundle.json` (instead of the new dedicated
      schema) MUST be rejected by the schema-validator
      (existing `comparison_bundle.schema.json` enum admits only
      `admitted` and `typed_unavailable`; the validator MUST
      NOT silently accept a `scalar_zero_shot_inferred` value).

- AC-T7: Headline-validity guard widening at
  `workloads/simbench_bw/comparison_runner.py::run_comparison`
  with six new typed reasons. (NOTE: codex second-pass flagged
  inconsistency between AC-T4's
  `calibration_loocv_p90_residual_exceeded` and AC-T7's enum;
  resolution is to admit the AC-T4 reason as the sixth typed
  reason in the AC-T7 enum so the headline-validity propagation
  stays load-bearing for calibration fit quality.)
  - Positive Tests:
    - The guard widens the existing AC-9 `headline_validity_failed_reasons`
      enum in lockstep with `comparison_bundle.schema.json` (per
      the CLAUDE.md wire-extension policy) to include:
      `calibration_reuse_distance_out_of_envelope`,
      `calibration_covariate_out_of_envelope`,
      `trace_opcode_coverage_incomplete`,
      `unsupported_blackwell_memory_path`,
      `target_arch_unlabeled_extrapolation`,
      `calibration_loocv_p90_residual_exceeded`.
    - Each new reason fires under a positive-witness unit test
      that constructs a synthetic bundle violating the
      condition and asserts the reason appears in the array.
    - On a bundle satisfying all envelopes, the
      `headline_validity_status = valid` and
      `headline_validity_failed_reasons = []` (preserves the
      active AC-9 invariant: empty when valid, non-empty when
      `typed_unavailable`).
    - A multi-failure unit test constructs a synthetic bundle
      violating two of the new reasons simultaneously and
      asserts both appear in the array (preserves the active
      AC-9 multi-failure-capture invariant).
    - A static-scan test asserts no calibration-head code path
      may emit a `calibration_head_v1.json` consumer's bundle
      output without setting both the calibrated
      `predicted_hit_rate` (or its `typed_unavailable` arm)
      AND the appropriate envelope arm.
  - Negative Tests:
    - Any new typed reason that has a suppression flag
      (e.g., `--allow-out-of-envelope`, `--skip-opcode-coverage`)
      MUST be rejected by the AC-T7 guard scanner (preserves
      the active AC-9 no-bypass discipline).
    - A guard implementation that records only the FIRST
      sub-condition that fires (instead of ALL fires) MUST fail
      the multi-failure unit test.
    - A bundle with `headline_validity_status = valid` AND a
      non-empty `headline_validity_failed_reasons` array MUST
      fail the schema validator.
    - A schema change that REMOVES any of the five new typed
      reasons from the enum MUST be rejected by an offline
      schema-stability test.

- AC-T8: Inbox housekeeping plus 1-line active-plan amendment.
  Lands in the same commit that opens the Tier-2 round contract
  for AC-T1 (analogous to the active plan's AC-10 cleanup
  pattern).
  - Positive Tests:
    - `references/papers/knowledge_base/inbox/maeda2017.pdf` is
      promoted to `references/papers/knowledge_base/active/`
      with the canonical name pattern
      `2017__hierarchical_reuse_distance_…`.
    - `references/papers/knowledge_base/inbox/jiang2022.pdf` is
      promoted to `references/papers/knowledge_base/active/`
      with the canonical name pattern
      `2022__shalom_condens_sharing_aware_rd_…`.
    - `references/papers/knowledge_base/inbox/2023-ccs.pdf` is
      moved to `references/papers/knowledge_base/archive/` with
      a one-line off-topic note in the manifest.
    - `references/papers/knowledge_base/inbox/usenixsecurity24-zhang-zhenkai.pdf`
      is moved to `references/papers/knowledge_base/archive/`
      with a one-line off-topic note that nonetheless cites it
      under AC-4.3 of the active plan for L2 write-allocate
      and non-inclusive empirical confirmation.
    - `references/papers/knowledge_base/manifest.json` is
      updated with the four moves.
    - `plans/cache_predictor_l2_trustworthy_baseline_plan.md`
      receives a 1-line ANNOTATION paragraph in the AC-4
      section noting that per-slice geometry already lands at
      `tools/hrd_profiler/observation_builder.py` line 391 and
      that THIS Tier-2 plan (this file) ADDS A SEPARATE Maeda
      2017 second-layer experiment AFTER active AC-4.1 closes.
      The annotation MUST NOT redefine, narrow, or widen the
      active AC-4.1 closure semantics; the active loop owns
      its own AC-4.1 closure shape.
  - Negative Tests:
    - The cleanup MUST NOT land as its own round — adding a
      Tier2-X round solely for this cleanup MUST be rejected
      by the round-contract structural scanner.
    - A commit that promotes inbox PDFs without updating
      `manifest.json` MUST be rejected by the manifest scanner
      (analogous to the active loop's evidence-manifest
      pairing check at `tools/local_labels/evidence_manifest.py`).

- AC-T9: Tier-2 RLCR loop discipline analog of active AC-10.
  - Positive Tests:
    - Each Tier-2 round is closed by a `/humanize:ask-codex`
      review invocation (one per round).
    - Each Tier-2 round's contract carries the same shape as
      the active loop's contracts under
      `.humanize/rlcr/2026-05-14_22-59-20/round-<n>-contract.md`:
      named slices in dependency order, failing-test-first
      witness, structural-scanner pass requirements,
      post-commit recapture invariant
      (`predictor_build_commit ∈ {HEAD, HEAD^}` for any
      bundle the round emits).
    - The Tier-2 finalize phase runs `simplify` review per the
      active loop's Round-finalize pattern.
  - Negative Tests:
    - A Tier-2 round closed without a Codex review invocation
      MUST be rejected by the round-summary structural scanner.
    - A Tier-2 round contract that does not declare its slice
      ordering MUST be rejected by the contract scanner.

- AC-T10: Per-round shadow-headline diagnostics for Tier-2 rounds
  (analog of active AC-11).
  - Positive Tests:
    - Each Tier-2 round summary contains a
      `shadow_headline_diagnostics` block reporting per-cell
      prediction variance, microbench-admitted cell IDENTITIES
      (slug + shape_point per cell, not just count), excluded
      reason counts (by typed reason), CI width, the
      calibration head's per-arch LOOCV residual values,
      and (Tier2-E only) the zero-shot sm_100 confidence band.
    - The Tier-2 scanner extends
      `tools/local_labels/test_round_summary_shadow_diagnostics_scanner.py`
      to admit the new Tier-2 fields with a typed
      `arch_set_includes=["sm_86", "sm_120"]` discriminator
      (and `sm_100` once Tier2-E lands).
  - Negative Tests:
    - A Tier-2 round summary missing the
      `shadow_headline_diagnostics` block MUST be flagged by
      the extended scanner.
    - Shadow diagnostics whose values disagree with a fresh
      re-computation from the same sweep MUST cause a unit
      test to fail (preserves the active AC-11 invariant).

## Path Boundaries

Path boundaries define the acceptable range of implementation
quality and choices.

### Upper Bound (Maximum Acceptable Scope)

All ten acceptance criteria close in dependency order: AC-T0 (loop
opens) → AC-T8 (housekeeping in opening commit) → AC-T1 (HRD
second layer + device manifest) → AC-T2 (calibration heads + sidecar
provenance, all three heads implemented per DEC-T8 outcome) →
AC-T3 (sm_120 live label sweep — full 13-cell cohort per DEC-T3
outcome) → AC-T4 (cross-arch fit with leave-arch-out and
uncalibrated-baseline residuals for all heads, predeclared
selection rule per DEC-T9) → AC-T5 (full producer + consumer
opcode coverage extension with regenerated proto bindings) →
AC-T6 (cloud B200 NVBit capture and zero-shot prediction emission
under `Trace-live-sm_100` evidence per DEC-T7) → AC-T7 (full
six-reason guard widening) → AC-T9 + AC-T10 (loop discipline
plus shadow diagnostics carried throughout). Cross-arch transfer
story closes with explicit
`transfer_status = extrapolated_no_target_labels` typed-honesty
on every sm_100 prediction; no marketing claim of "validated on
sm_100".

### Lower Bound (Minimum Acceptable Scope)

The dependency-critical path: AC-T0 + AC-T8 (loop opens with
housekeeping) → AC-T1.1 + AC-T1.2 (second HRD layer + device
manifest, mandatory for any meaningful Tier-2 work) → AC-T2 with
restricted head set (`IsotonicCalibrator` + `PlattCalibrator`
only; `BetaCalibrator` deferred per DEC-T8 restricted-set
decision) → AC-T4 with sm_86-only fit and `target_arch =
sm_86_only` flag perpetually set on any inference path (defers
AC-T3 sm_120 sweep and AC-T6 sm_100 zero-shot dry-run to a
follow-up loop) → AC-T7 widens the guard with the FOUR
sm_86-active typed reasons
(`calibration_reuse_distance_out_of_envelope`,
`calibration_covariate_out_of_envelope`,
`trace_opcode_coverage_incomplete`,
`calibration_loocv_p90_residual_exceeded`; defers
`unsupported_blackwell_memory_path` and
`target_arch_unlabeled_extrapolation` to the same follow-up
loop, since both fire only on cross-arch evidence — Blackwell
memory paths and zero-shot target-arch inference respectively).
AC-T5 producer-side opcode classification DEFERS to the same
follow-up loop (Round 4 user directive narrowed scope; Blackwell
tracer extension is not part of the sm_86-only active scope).
AC-T9 + AC-T10 carry the loop discipline and shadow diagnostics
throughout the
lower-bound rounds.

**Lower-bound closure semantics** (per DEC-T13): when AC-T3 and
AC-T6 are deferred, the loop CANNOT claim
`Cross-arch-verified` evidence status for AC-T4 and CANNOT
claim any cross-architecture transfer. The lower-bound loop is
explicitly an `sm_86-only Tier-2 calibration follow-up`, NOT a
cross-architecture transfer loop. The AC-T4 round summary MUST
record `target_arch = sm_86_only` and MUST NOT emit any
`cross_arch_zero_shot_prediction.json` artifact. A separate
follow-up RLCR loop (call it Tier-2-cross-arch) MUST be opened
later to land AC-T3 + AC-T6 + the deferred AC-T7 reasons; the
lower-bound loop's finalize-summary MUST queue this follow-up
under its `Explicitly Deferred` section.

### Allowed Choices

- AC-T1.1 implementation: Maeda 2017 Algorithm 1 may be
  implemented via per-slice secondary `ReuseDistanceTracker`
  instances (the simple path, mirroring the existing per-slice
  pattern at
  `tools/hrd_profiler/observation_builder.py` lines 386-391) OR
  via a unified two-layer stack distance tracker (the precise
  path; closer to Maeda 2017 §IV exposition). The choice is
  recorded in the round summary.
- AC-T2 calibrator library policy: scikit-learn `IsotonicRegression`
  + hand-rolled 2-parameter Platt, OR three hand-rolled
  implementations to avoid the scikit-learn dependency. The
  choice is pinned by DEC-T8.
- AC-T3 sm_120 sweep cohort size: 5 activation cells only OR
  full 13 cells (5 activation + 8 microbench). Pinned by DEC-T3.
- AC-T4 winner selection rule: pinned by DEC-T9 to one of
  `loocv_p90_minimum`, `leave_arch_out_p90_minimum`,
  `simplicity_one_se_rule`, `predeclared_isotonic`,
  `predeclared_platt`. The chosen rule is recorded in the
  round contract before the round opens.
- AC-T5.1 TMEM op classification: simple path
  (`LDT`/`LDTM` → `MemoryOpClass::kGlobal`,
  `STT`/`STTM` → `MemoryOpClass::kGlobal`, `UTCCP` →
  `MemoryOpClass::kShared`) OR a NEW `MemoryOpClass::kTMEM`
  enum value (additive widening). Pinned by DEC-T6.
- AC-T6 cloud target: B200 (sm_100 datacenter) OR an alternative
  Blackwell cloud (e.g., RTX PRO 6000 sm_120-class). Pinned by
  DEC-T4 re-scope.
- AC-T6 zero-shot artifact format: dedicated new schema (the
  default per the codex-flagged enum-collision finding) OR
  inline within `comparison_bundle.json` with an additional
  `result_kind` enum value. Pinned by DEC-T10; default is the
  dedicated schema.
- AC-T6 sm_100 evidence-status string: `Trace-live-sm_100`
  (pinned per DEC-T7 RESOLVED). The alternative
  `Verified-live-sm_100` is NOT available; CLAUDE.md is
  widened in the AC-T6 opening commit to admit
  `Trace-live-sm_100` only, preserving the discipline that
  `Verified-live-<arch>` requires NCU-level evidence.

### Cannot Use

- **Round contracts that require committing `.humanize/` loop
  artifacts (`goal-tracker.md`, `round-N-contract.md`,
  `round-N-summary.md`, `round-N-review-result.md`)**. These
  files are gitignored per `.gitignore:11` AND mechanically
  blocked by the framework's `loop-bash-validator.sh` hook AND
  forbidden by CLAUDE.md's Session Handoff section. A round
  contract slice that says "commit Round N artifacts" is
  un-satisfiable; round contracts MUST instead say "Round N
  source-tree changes (if any) commit normally; loop artifacts
  remain host-local per project convention." Codex's review
  rubric will treat host-local loop artifacts as satisfied
  when the round summary cites them by path (the framework's
  `@<path>` resolves them for review without requiring git
  tracking). Added post-circuit-breaker 2026-05-16 when the
  prior `.humanize/rlcr/2026-05-16_15-26-09/` loop hit
  3-consecutive-STALL drift on this exact incompatibility.
- Lowering `MINIMUM_BIN_SAMPLE_COUNT=50` or
  `CI_HALF_WIDTH_TARGET=0.05` at
  `tools/hrd_profiler/profiler.py` lines 45-46 (active AC-6
  negative test carries forward).
- Landing AC-T2 (calibration head infra) before AC-T1's
  synthetic witnesses pass (AC-T2.1 negative test).
- Landing AC-T7 (validity guard widening) before AC-T6 (zero-shot
  sm_100 dry-run) when the upper-bound scope is selected; AC-T7
  placement preserves the active AC-9 last-round discipline
  (the guard must NOT widen until the new typed reasons have
  observable witnesses).
- Adding any suppression flag to the new typed reasons
  (active AC-9 no-bypass discipline carries forward).
- NCU on sm_100 cloud (CLAUDE.md Rule 1 absolute);
  driver-root operations (Rule 2); privileged CUPTI surfaces
  (Rule 3); host-driver work (Rule 4); simulator revival
  (Rule 5).
- Per-arch heads with at most ~13 labels per arch (codex first-
  pass and prior research pass converged on this rejection).
- Neural calibration heads (codex first-pass: "overfit theater";
  literature agent: "explicitly assumes target labels"; both
  reject).
- Using sm_120 alone as authority for sm_100 prediction
  (consumer Blackwell GB206 differs from datacenter B200 in L2
  size, dual-die topology, scheduling pressure; pool both arches
  per AC-T4).
- Mixing the zero-shot sm_100 prediction artifact into
  `comparison_bundle.json`'s existing
  `result_kind: ["admitted", "typed_unavailable"]` enum without
  the dedicated schema arm (codex first-pass schema-collision
  finding).
- Naming the sm_100 evidence as `Verified-live-sm_100`
  (DEC-T7 RESOLVED to admit `Trace-live-sm_100` instead;
  `Verified-live-sm_100` is forbidden because CLAUDE.md
  Rule 1 forbids NCU on cloud GPUs, and
  `Verified-live-<arch>` requires NCU-level evidence per
  the active loop's two-device discipline).
- Touching the 7-tuple `manifest_pairing_index` join key at
  `workloads/simbench_bw/comparison_runner.py` lines 99-107
  (active plan's invariant carries forward).

> **Note on Deterministic Designs**: The Tier-2 dependency order
> AC-T0 → AC-T8 → AC-T1 → AC-T2 → AC-T3 → AC-T4 → AC-T5 → AC-T6
> → AC-T7 (with AC-T9 + AC-T10 throughout) is fixed by the
> draft's mechanism-then-data-then-transfer sequencing. The
> AC-T1.1 synthetic-witness ordering before AC-T2 is fixed.
> The cloud no-NCU rule (CLAUDE.md Rule 1) is fixed.
> The active loop's pinned constants are fixed. The
> `MEMORY_OP_CLASS_*` lockstep with `instruction.proto` enum
> widening is fixed (per CLAUDE.md wire-extension policy).
> Allowed choices apply only to per-AC implementation tactics
> and the six pinned DECs.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only.
> These are conceptual suggestions, not prescriptive
> requirements.

### Conceptual Approach

Open the Tier-2 RLCR session at `.humanize/rlcr/<new-session-id>/`
modeled on the active session at `.humanize/rlcr/2026-05-14_22-59-20/`.
The new `goal-tracker.md` names the Ultimate Goal as: "Defensible
Tier-2 RD calibration head and cross-arch transfer story for the
L2 sector hit-rate predictor." Ten Tier-2 round contracts queue
in dependency order. Each round closes via failing-test-first
discipline + Codex review per round + structural-scanner
enforcement + post-commit recapture (the active loop's pattern
verbatim).

For Tier2-A (AC-T1 + AC-T8 cleanup), implement Maeda 2017
Algorithm 1 as a second-layer per-slice working-set tracker on
top of the existing fully-associative LRU walker; the additive
shape preserves byte-for-byte legacy output when the new layer
is disabled (mirrors the existing `num_l2_slices == 1` collapse
path at `observation_builder.py` line 382-385).

For Tier2-B (AC-T2), implement calibrators in pure
NumPy/SciPy (or scikit-learn per DEC-T8); persist trained heads
as JSON sidecars with full provenance (training cells +
comparison_contract.toml pin + 7-tuple join keys + source commit
recapture); add the sidecar provenance scanner modeled on the
existing `tools/local_labels/test_admitted_label_launch_provenance_scanner.py`
including duplicate-detection and placeholder-rejection clauses.

For Tier2-C (AC-T3), run the local-label sweep on the RTX 5060
laptop using the existing `tools/local_labels/run_headline_cohort_sweep.py`
entrypoint; emit the sm_120 mirror of the sm_86 label tree under
`data/local_labels/sm_120/`; create the sm_120 launch-provenance
sidecar; extend the launch-provenance scanner with the
`arch=sm_120` discriminator. The repeatability stddev thresholds
(L2 ≤ 0.03, L1tex ≤ 0.01 per
`tools/local_labels/repeatability_aggregator.py` lines 664-718)
remain pinned for sm_120.

For Tier2-D (AC-T4), pool sm_86 + sm_120 (predicted, label, arch)
tuples; fit each calibrator; emit the per-head residual evaluation
block with LOOCV + leave-arch-out + uncalibrated-baseline columns;
record the chosen head per the predeclared selection rule from
the round contract.

For Tier2-E-prep (AC-T5), implement producer-side opcode
classification first as a risk-burn-down round (no live sm_100
required); validate via synthetic `tcgen05.ld.sync.aligned.*`
PTX kernels compiled `-arch=sm_100a` offline; widen consumer-side
schema enums in lockstep per the CLAUDE.md wire-extension policy
(modeled on the active loop's Round 24 producer-side `kUnspecified`
→ `unsupported_memory_path` lockstep with consumer-side
`VALID_OP_CLASSES` widening at
`tools/nvbit_tracer/import_nvbit_exclusion.py`).

For Tier2-E (AC-T6), capture cloud B200 NVBit trace via
`LD_PRELOAD` + `CUDA_INJECTION64_PATH`; consume via the
trained Tier2-D head; emit the dedicated zero-shot artifact at
`data/cross_arch_predictions/sm_100/<sweep_id>/cross_arch_zero_shot_prediction.json`
with the new schema `cross_arch_zero_shot_prediction.schema.json`
under `workloads/simbench_bw/schemas/`. The round summary
records `Trace-live-sm_100` evidence (per DEC-T7 default).

For Tier2-F (AC-T7), widen the AC-9 typed-reason enum at
`workloads/simbench_bw/comparison_runner.py::run_comparison`
with the FOUR sm_86-active typed reasons
(`calibration_reuse_distance_out_of_envelope`,
`calibration_covariate_out_of_envelope`,
`trace_opcode_coverage_incomplete`,
`calibration_loocv_p90_residual_exceeded`); widen the JSON
Schema enum in lockstep; add the per-reason positive witnesses
+ the multi-failure witness + the suppression-flag negative
test; finalize phase runs `simplify` review. The two
cross-arch typed reasons (`unsupported_blackwell_memory_path`
and `target_arch_unlabeled_extrapolation`) DEFER to the
follow-up Tier-2-cross-arch loop.

### Relevant References

- `plans/cache_predictor_l2_trustworthy_baseline_plan.md` — the
  active plan; AC-1..AC-11 numbering reference; AC-9 guard
  template that AC-T7 widens; AC-10 cleanup-with-round-open
  pattern that AC-T8 mirrors; AC-11 shadow-diagnostics scanner
  that AC-T10 extends.
- `.humanize/rlcr/2026-05-14_22-59-20/` — the active loop;
  round contracts, goal-tracker structure, post-commit
  recapture discipline, structural-scanner literal-watching.
- `tools/hrd_profiler/observation_builder.py` — the predictor
  producer; `_walk_launch` lines 132-744 (already contains
  per-slice walker since Round 12 at line 391; CTA-interleave
  Pass-2 path at lines 415-436; per-launch diagnostic emission
  at lines 161-278; aggregator at lines 330-452; entrypoint
  `build_launches_from_trace_dir` at line 648).
- `tools/hrd_profiler/profiler.py` — typed-fail-closed constants
  `MINIMUM_BIN_SAMPLE_COUNT=50` (line 46) and
  `CI_HALF_WIDTH_TARGET=0.05` (line 45);
  `cache_model_underconfigured` typed reason (line 100).
- `workloads/simbench_bw/cache_predictor.py` — the predictor
  consumer; `predict_cache_hit_rate` (lines 478-751);
  `_TYPED_REASONS` frozenset (lines 89-101); bootstrap CI 2000
  resamples (lines 728-751).
- `workloads/simbench_bw/comparison_runner.py` — `run_comparison`
  entrypoint (line 1414); 7-tuple `manifest_pairing_index` join
  key (lines 99-107); `_PREDICTOR_DESIGN_COVERAGE_BY_CELL`
  (line 175); AC-9 validity-guard placement.
- `workloads/simbench_bw/schemas/comparison_bundle.schema.json` —
  enum surfaces; `result_kind: ["admitted", "typed_unavailable"]`;
  per-cell `predictor_result_kind: ["scalar", "bounded"]`;
  `headline_validity_status` + `headline_validity_failed_reasons`
  surfaces from the active AC-9.
- `workloads/simbench_bw/schemas/nvbit_unsupported_memory_path.schema.json` —
  exclusion sidecar schema that AC-T5.2 promotes.
- `tools/nvbit_tracer/isa_def/blackwell_opcode.h` (293 lines) —
  current Blackwell opcode map (only OMMA + QMMA from
  Blackwell-new set); AC-T5.1 extends.
- `tools/nvbit_tracer/isa_def/trace_opcode.h` (286 lines) —
  TraceInstrOpcode enum; AC-T5.1 extends.
- `tools/nvbit_tracer/tracer_tool/memory_op_class.h` and
  `memory_op_access_kind.h` — classifier headers; AC-T5.1
  extends.
- `tools/nvbit_tracer/tracer_tool/exclusion_aggregator.h` —
  emits `unsupported_memory_path.json` exclusion sidecar that
  AC-T5.2 promotes into `headline_validity_failed_reasons`.
- `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto` —
  proto schema for `memory_op_class` and `memory_op_access_kind`
  fields; AC-T5.1 widens additively.
- `tools/nvbit_tracer/import_nvbit_exclusion.py` and
  `tools/nvbit_tracer/test_import_nvbit_exclusion.py` and
  `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py` —
  consumer-side exclusion taxonomy; AC-T5.2 widens in
  lockstep.
- `tools/local_labels/test_admitted_label_launch_provenance_scanner.py` —
  scanner template that AC-T2.2 + AC-T3 extend; the active
  loop's Round 6 hardening (duplicate-detection,
  placeholder-rejection, count-of-five enforcement) carries
  forward.
- `tools/local_labels/test_round_summary_shadow_diagnostics_scanner.py` —
  scanner template that AC-T10 extends.
- `tools/local_labels/select_best_config_labels.py` — rogue-stale
  filter (already rejects mismatched-identity payloads); AC-T3
  negative test reuses.
- `tools/adaptive_sampler/trace_convergence_ledger.py` lines
  177-188 — 5-field `manifest_pairing_index` schema; AC-T3
  preserves invariant.
- `references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf` —
  Nugteren 2014 HPCA; canonical RD-on-GPU prior art.
- `references/papers/knowledge_base/inbox/maeda2017.pdf` (to be
  promoted per AC-T8) — Maeda 2017 HPCA; Algorithm 1 (p. 148)
  is the AC-T1 mechanistic basis.
- `references/papers/knowledge_base/inbox/jiang2022.pdf` (to be
  promoted per AC-T8) — Jiang/Maeda 2022 IEEE TPDS; Shalom +
  Condens; informs AC-T1 CTA-interleaving consideration.
- `.claude/knowledge/measurement/nvbit-injection-playbook.md` —
  non-root NVBit injection contract; grounds AC-T6 cloud
  capture feasibility.
- `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md` —
  DVFS covariate discipline; grounds the
  `calibration_covariate_out_of_envelope` typed reason.
- `.claude/knowledge/blackwell/blackwell-tuning.md` — L2
  capacity numbers per arch; grounds AC-T1.2 device-properties
  manifest content.
- The 2026-05-16 deep research outputs at
  `.humanize/skill/2026-05-16_14-13-56-278667-748300b1/output.md`
  (codex consultation) and the parallel agent reports
  (referenced inline in the draft preserved at the bottom of
  this plan file).

## Dependencies and Sequence

### Milestones

1. Milestone Tier2-A — Loop opens with mechanistic foundation
   - Phase A1: AC-T0 (Tier-2 RLCR session opens at
     `.humanize/rlcr/<new-session-id>/`; round contracts
     queued; goal-tracker initialized). Precondition: active
     loop's AC-3..AC-9 closed.
   - Phase A2: AC-T8 (inbox housekeeping + active-plan
     1-line correction lands in the same commit that opens
     the AC-T1 round contract).
   - Phase A3: AC-T1.1 (Maeda 2017 second HRD layer in
     `_walk_launch` with synthetic witnesses).
   - Phase A4: AC-T1.2 (device-properties manifest + scanner).
   - Phase A5: AC-T9 round-discipline + AC-T10
     shadow-diagnostics scanner extension carry through
     this and every subsequent milestone.
2. Milestone Tier2-B — Calibration head infrastructure
   - Phase B1: AC-T2.1 (calibrator implementations + monotonicity
     witness + envelope-violation witness).
   - Phase B2: AC-T2.2 (trained-head sidecar provenance + scanner).
   - Depends on: Tier2-A (AC-T1 synthetic witnesses pass).
3. Milestone Tier2-C **[DEFERRED TO Tier-2-cross-arch follow-up loop]** — Live sm_120 label sweep
   - Phase C1: Pre-flight validation (RTX 5060 laptop NVBit
     build + CUDA toolkit + driver state).
   - Phase C2: AC-T3 sweep + label admission + launch-provenance
     sidecar emission + scanner extension.
   - Depends on: Tier2-A (predictor producer stable; predictor
     bundles regenerable on sm_120).
   - Independent of: Tier2-B (calibration head infra; can run
     in parallel with Tier2-B if scheduling permits).
4. Milestone Tier2-D — Cross-arch calibration fit
   - Phase D1: AC-T4 pool + fit + LOOCV + leave-arch-out +
     uncalibrated-baseline residual evaluation block.
   - Phase D2: AC-T4 selection rule resolution + chosen-head
     persistence + AC-9 typed-reason scaffolding for
     `calibration_loocv_p90_residual_exceeded`.
   - Depends on: Tier2-B (calibration heads exist) AND Tier2-C
     (sm_120 labels exist).
5. Milestone Tier2-E **[DEFERRED TO Tier-2-cross-arch follow-up loop]** — sm_100 zero-shot transfer
   - Phase E1: AC-T5.1 producer-side opcode classification
     extension (risk-burn-down; runs in parallel with Tier2-B
     or Tier2-C if scheduling permits).
   - Phase E2: AC-T5.2 consumer-side schema widening + lockstep
     test.
   - Phase E3: AC-T6 cloud B200 NVBit capture + zero-shot
     prediction emission via dedicated artifact.
   - Depends on: Tier2-D (trained head exists) AND AC-T5.1
     (producer can trace sm_100 binaries).
6. Milestone Tier2-F — Headline-validity guard widening + finalize
   - Phase F1: AC-T7 widens AC-9 typed-reason enum with the
     five new reasons; positive + negative + multi-failure
     witnesses pass.
   - Phase F2: Finalize phase runs `simplify` review;
     AC-T9 + AC-T10 final sweep; loop close.
   - Depends on: ALL prior milestones.

Active Scope (per Round 4 user directive): Tier2-C and Tier2-E
(both phases) defer to the follow-up Tier-2-cross-arch loop;
AC-T4 fits sm_86-only with a perpetual `target_arch = sm_86_only`
flag; AC-T7 widens with the FOUR sm_86-active typed reasons
(`calibration_reuse_distance_out_of_envelope`,
`calibration_covariate_out_of_envelope`,
`trace_opcode_coverage_incomplete`,
`calibration_loocv_p90_residual_exceeded`; the remaining
two — `unsupported_blackwell_memory_path`,
`target_arch_unlabeled_extrapolation` — defer to the
follow-up Tier-2-cross-arch loop per DEC-T13).

## Task Breakdown

Each task includes exactly one routing tag: `coding` (Claude
implements) or `analyze` (Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Confirm active loop's AC-3..AC-9 closed; open Tier-2 RLCR session at `.humanize/rlcr/<new-session-id>/`; initialize goal-tracker.md with this plan's IMMUTABLE SECTION; queue ten round contracts in dependency order | AC-T0 | coding | - |
| task2 | Inbox housekeeping: promote `inbox/maeda2017.pdf` and `inbox/jiang2022.pdf` to `active/`; archive `inbox/2023-ccs.pdf` and `inbox/usenixsecurity24-zhang-zhenkai.pdf`; update `references/papers/knowledge_base/manifest.json`; append 1-line AC-4 correction paragraph to `plans/cache_predictor_l2_trustworthy_baseline_plan.md` | AC-T8 | coding | task1 |
| task3 | Implement Maeda 2017 second HRD layer in `tools/hrd_profiler/observation_builder.py::_walk_launch`; add synthetic-kernel witnesses (positive + negative); preserve byte-for-byte legacy output when the new layer is disabled | AC-T1.1 | coding | task2 |
| task4 | Add `tools/hrd_profiler/device_properties.json` manifest keyed by (arch, sku, driver_version, cuda_toolkit_version, nvbit_build); add `cache_model_geometry_provenance` field to predictor bundles; implement `cache_model_geometry_provenance_scanner` test | AC-T1.2 | coding | task3 |
| task5 | Codex review per Tier2-A round closure | AC-T0, AC-T1, AC-T8 | analyze | task4 |
| task6 | Implement `workloads/simbench_bw/calibration_head.py` with `IsotonicCalibrator` + `PlattCalibrator` (and `BetaCalibrator` per DEC-T8); add monotonicity + envelope-violation witnesses for each calibrator | AC-T2.1 | coding | task5 |
| task7 | Define `data/local_labels/calibration_head_v1.json` schema + provenance fields; implement `tools/local_labels/test_calibration_head_provenance_scanner.py` with positive + negative controls (missing fields; 7-tuple join failures; comparison_contract.toml sha256 mismatch) | AC-T2.2 | coding | task6 |
| task8 | Codex review per Tier2-B round closure | AC-T2 | analyze | task7 |
| task9 | **[DEFERRED TO Tier-2-cross-arch]** Run `tools/local_labels/run_headline_cohort_sweep.py` on RTX 5060 laptop for sm_120 cohort (size per DEC-T3); emit `data/local_labels/sm_120/` mirror; type repeatability failures via existing `repeatability_aggregator.py`; commit Tier2-C round summary with `Verified-live-sm_120` evidence statement | AC-T3 | coding | DEFERRED |
| task10 | **[DEFERRED TO Tier-2-cross-arch]** Create `data/local_labels/sm_120/label_admitted_launch_provenance.json` sidecar; extend `tools/local_labels/test_admitted_label_launch_provenance_scanner.py` with `arch=sm_120` discriminator preserving Round 6 hardening (duplicate-detection, placeholder-rejection, count-of-five) | AC-T3 | coding | DEFERRED |
| task11 | **[DEFERRED TO Tier-2-cross-arch]** Codex review per Tier2-C round closure | AC-T3 | analyze | DEFERRED |
| task12 | Fit each calibrator from task6 on sm_86-only (predicted, label) tuples; emit `calibration_evaluation_v1.json` with LOOCV residuals + uncalibrated-baseline residuals for all three heads (Isotonic + Platt + Beta); resolve chosen head per DEC-T9 `simplicity_one_se_rule` pinned in the round contract; record `target_arch = sm_86_only` in the round summary. Leave-arch-out evaluation is DEFERRED with AC-T3 to the follow-up loop | AC-T4 | coding | task7 |
| task13 | Add `calibration_loocv_p90_residual_exceeded` typed reason scaffolding to `workloads/simbench_bw/comparison_runner.py::run_comparison`; positive-witness unit test fires the reason when chosen head's LOOCV p90 exceeds DEC-T5 pinned threshold 0.30 | AC-T4 | coding | task12 |
| task14 | Codex review per Tier2-D round closure | AC-T4 | analyze | task13 |
| task15 | **[DEFERRED TO Tier-2-cross-arch]** Extend `tools/nvbit_tracer/isa_def/blackwell_opcode.h` with `LDT`/`LDTM`/`STT`/`STTM`/`UTCCP`/`UTCSHIFT`/`UTCBAR`/`UTCHMMA`/`UTCIMMA`/`UTCOMMA`/`UTCQMMA`/`UGETNEXTWORKID`/`UVIRTCOUNT`/`FADD2`/`FFMA2`/`FMUL2`/`FHADD`/`FHFMA`/`FMNMX3`; widen `tools/nvbit_tracer/isa_def/trace_opcode.h` enum additively | AC-T5.1 | coding | DEFERRED |
| task16 | **[DEFERRED TO Tier-2-cross-arch]** Extend `tools/nvbit_tracer/tracer_tool/memory_op_class.h` and `memory_op_access_kind.h` classifiers per DEC-T6 outcome (kTMEM enum value); update proto schema at `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto`; extend C++ unit tests at `memory_op_class_test.cc` and `memory_op_access_kind_test.cc` with positive controls for the new opcodes | AC-T5.1 | coding | DEFERRED |
| task17 | **[DEFERRED TO Tier-2-cross-arch]** Synthetic `tcgen05.ld.sync.aligned.*` PTX kernel compiled `-arch=sm_100a` offline; assert `cuobjdump --dump-sass` emits SASS containing `LDT`; `nvdisasm` confirms the opcode mnemonic; `memory_op_class_test.cc` and `memory_op_access_kind_test.cc` unit tests admit `LDT` with the correct classification per DEC-T6 + DEC-T14 outcomes | AC-T5.1 | coding | DEFERRED |
| task18 | **[DEFERRED TO Tier-2-cross-arch]** Widen JSON Schema enums at `workloads/simbench_bw/schemas/comparison_bundle.schema.json` and `workloads/simbench_bw/schemas/nvbit_unsupported_memory_path.schema.json` and `tools/nvbit_tracer/import_nvbit_exclusion.py::VALID_OP_CLASSES`; extend `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py` to enforce lockstep | AC-T5.2 | coding | DEFERRED |
| task19 | **[DEFERRED TO Tier-2-cross-arch]** Promote `unsupported_memory_path.json` sidecar into `headline_validity_failed_reasons` when typed reason is `unsupported_memory_path` AND missing opcode is memory-touching per the AC-T5.1 classifier tables | AC-T5.2 | coding | DEFERRED |
| task20 | **[DEFERRED TO Tier-2-cross-arch]** Codex review per Tier2-E-prep (AC-T5) round closure | AC-T5 | analyze | DEFERRED |
| task21 | **[DEFERRED TO Tier-2-cross-arch]** Capture cloud B200 NVBit trace via `LD_PRELOAD` + `CUDA_INJECTION64_PATH`; record cloud GPU model + driver + CUDA toolkit + NVBit build + observed clocks in Tier2-E round summary | AC-T6 | coding | DEFERRED |
| task22 | **[DEFERRED TO Tier-2-cross-arch]** Add new schema `workloads/simbench_bw/schemas/cross_arch_zero_shot_prediction.schema.json` (separate from `comparison_bundle.schema.json`); emit `data/cross_arch_predictions/sm_100/<sweep_id>/cross_arch_zero_shot_prediction.json` with `result_kind = scalar_zero_shot_inferred` (NEW enum value local to NEW schema), calibrated predicted hit rate, LOOCV-derived confidence band, `transfer_status = extrapolated_no_target_labels`, `headline_validity_failed_reasons` array including `target_arch_unlabeled_extrapolation` | AC-T6 | coding | DEFERRED |
| task23 | **[DEFERRED TO Tier-2-cross-arch]** Use `Trace-live-sm_100` evidence-status string in Tier2-E round summary (per DEC-T7 RESOLVED); widen CLAUDE.md's Two-Device Local Development Model section to a Three-Device discipline admitting `Trace-live-sm_100`; cite CLAUDE.md Rule 1 as binding for sm_100; add round-summary structural-scanner clause that ALWAYS flags `Verified-live-sm_100` | AC-T6 | coding | DEFERRED |
| task24 | **[DEFERRED TO Tier-2-cross-arch]** Codex review per Tier2-E (AC-T6) round closure | AC-T6 | analyze | DEFERRED |
| task25 | Widen `headline_validity_failed_reasons` enum in `workloads/simbench_bw/comparison_runner.py::run_comparison` and `comparison_bundle.schema.json` with the FOUR sm_86-active typed reasons (`calibration_reuse_distance_out_of_envelope`, `calibration_covariate_out_of_envelope`, `trace_opcode_coverage_incomplete`, `calibration_loocv_p90_residual_exceeded`); per-reason positive-witness unit tests; multi-failure witness; suppression-flag negative test; static-scan test for calibration-head bundle emission. The two cross-arch reasons (`unsupported_blackwell_memory_path`, `target_arch_unlabeled_extrapolation`) are DEFERRED with AC-T6 to the follow-up Tier-2-cross-arch loop | AC-T7 | coding | task14 |
| task26 | Per-round shadow-headline diagnostics emission throughout Tier2-A, Tier2-B, Tier2-D, Tier2-F (only the sm_86-active milestones); extend `tools/local_labels/test_round_summary_shadow_diagnostics_scanner.py` with `arch_set_includes=["sm_86"]` discriminator. Tier2-C and Tier2-E discriminator extensions DEFER with their parent ACs | AC-T10 | coding | task5, task8, task14, task25 |
| task27 | Codex review per Tier2-F (AC-T7) round closure | AC-T7 | analyze | task25, task26 |
| task28 | Finalize phase: `/simplify` review of opportunistic dead-code removal; finalize-summary.md records `sm_86_only` closure for this loop (NO `Cross-arch-verified`; NO `Trace-live-sm_100` — those defer with AC-T3/T5/T6); finalize-summary `Explicitly Deferred` section queues the follow-up `Tier-2-cross-arch` RLCR loop with AC-T3, AC-T5, AC-T6, AC-T4's leave-arch-out, AC-T7's two cross-arch typed reasons, and DEC-T3/T4/T6/T7/T10/T11/T14 carried forward | AC-T9 | coding | task27 |

## Claude-Codex Deliberation

### Agreements

- Mechanistic per-slice / per-arch geometry from a static
  device-properties manifest (NOT learned thresholds; NOT runtime
  NVML reads). Codex first-pass and the prior research pass both
  converge.
- Pooled global calibration head with `arch` recorded ONLY as
  evaluation metadata (NOT as a fitted feature; NOT per-arch
  heads at ≤13 labels per arch). The calibrator signature is
  `fit(predicted, label)`; per-arch residuals surface in the
  AC-T4 leave-arch-out evaluation block. Both Claude and codex
  converged on this in Round 2.
- Pool sm_86 + sm_120 once both labeled; never use sm_120 alone
  as authority for sm_100 prediction. Both converge.
- Honest sm_100 claim shape: `transfer_status =
  extrapolated_no_target_labels`; never market as "validated on
  sm_100". Both converge.
- Reject neural calibration heads at this cohort size. Both
  converge.
- Five new typed-fail-closed reasons span the right surfaces
  (RD envelope, covariate envelope, opcode coverage, memory-path
  coverage, target-arch labeling). Codex first-pass and the
  prior consultation converge on the set.
- Active loop's pinned constants (`MINIMUM_BIN_SAMPLE_COUNT=50`,
  `CI_HALF_WIDTH_TARGET=0.05`) carry forward unchanged.
- Wire-extension policy (additive enum widening; consumer-side
  schema widens in lockstep) is the right discipline for
  AC-T5.1 + AC-T5.2.
- AC-T8 inbox housekeeping (promote maeda2017 + jiang2022;
  archive 2023-ccs + usenixsecurity24) is uncontroversial.

### Resolved Disagreements

**Round 1 (first-pass codex critique → candidate plan v1)**

- AC-T1 ownership: Claude's draft language ("AC-T1 closes
  AC-4.1 with corrected mechanistic semantics") implied
  retroactive closure of an active-plan AC. Codex first-pass
  flagged the collision. Resolution: AC-T1 is reframed as a
  NEW Tier-2 surface that BUILDS ON whatever per-slice /
  per-set semantics the active loop's AC-4.1 close commit
  pins; the active loop owns its own AC-4.1 closure.
- AC-T6 schema collision: Claude's draft proposed
  `result_kind = scalar_zero_shot_inferred` as a new value in
  the existing `comparison_bundle.schema.json` enum. Codex
  first-pass flagged the schema collision (existing enum is
  `["admitted", "typed_unavailable"]` with per-cell
  `predictor_result_kind: ["scalar", "bounded"]`). Resolution:
  AC-T6 emits a SEPARATE artifact at
  `data/cross_arch_predictions/sm_100/<sweep_id>/cross_arch_zero_shot_prediction.json`
  under a NEW dedicated schema
  `cross_arch_zero_shot_prediction.schema.json`; the existing
  `comparison_bundle.schema.json` enum is unchanged.
- AC-T6 evidence-status terminology: Claude's draft proposed
  `Verified-live-sm_100` for the NVBit-only trace. Codex
  first-pass flagged that CLAUDE.md only defines
  `Verified-live-sm_86`, `Verified-live-sm_120`, and
  `Cross-arch-verified`; the new vocabulary risks blurring the
  no-NCU cloud rule. Resolution: AC-T6 default uses
  `Trace-live-sm_100` (a NEW evidence-status string scoped to
  NVBit-only trace evidence on sm_100); DEC-T7 escalates the
  vocabulary choice to the user.
- Calibration-head data-dredging risk: Claude's draft proposed
  "implement all three heads and let LOOCV pick". Codex
  first-pass flagged that LOOCV winner-selection at N=10..26
  cells is data dredging without a predeclared rule.
  Resolution: AC-T4 requires the selection rule to be PINNED IN
  THE ROUND CONTRACT BEFORE THE ROUND OPENS (one of five named
  rules); DEC-T9 escalates the rule choice to the user.
- AC-T1 multi-layer scope: Claude's draft did not specify
  whether the second HRD layer applies to all four
  `_walk_launch` layers (sector, line, allocation, region) or
  only the sector layer. Codex first-pass flagged the
  ambiguity. Resolution: AC-T1.1 explicitly scopes the second
  HRD layer to the sector cache decision only; line /
  allocation / region layers remain unchanged.
- AC-T5 widening surface area: Claude's draft mentioned only
  `comparison_bundle.schema.json` + `import_nvbit_exclusion.py`
  + `import_trace_contract.py`. Codex first-pass flagged the
  missing tracer-side surfaces (memory_op_class.h,
  memory_op_access_kind.h, C++ unit tests, regenerated proto
  bindings, exclusion sidecar schema, parity tests).
  Resolution: AC-T5.1 + AC-T5.2 explicitly enumerate every
  surface that must widen in lockstep.

**Round 2 (second-pass codex review → candidate plan v2)**

- AC-T8 active-plan amendment scope: Codex second-pass flagged
  that v1's amendment to `cache_predictor_l2_trustworthy_baseline_plan.md`
  redefined the active loop's AC-4.1 closure semantics, which
  contradicts AC-T1's "do not retroactively close active AC"
  resolution from round 1. Resolution: AC-T8 amendment narrowed
  to a strict ANNOTATION ("Tier-2 ADDS A SEPARATE Maeda
  second-layer experiment AFTER active AC-4.1 closes"); MUST
  NOT redefine, narrow, or widen the active AC-4.1 closure
  semantics.
- AC-T5.1 offline witness feasibility: Codex second-pass
  flagged that v1's "traceable end-to-end without unknown-
  opcode errors" implied live NVBit trace execution which is
  infeasible offline (NVBit requires running on compatible
  hardware). Resolution: AC-T5.1 offline witness reframed as
  cuobjdump+nvdisasm SASS-presence check + C++ classifier
  unit tests; live tracer execution moves entirely to AC-T6.
  DEC-T14 records the validation split.
- AC-T4/AC-T7 enum inconsistency: Codex second-pass noted that
  v1 added `calibration_loocv_p90_residual_exceeded` in AC-T4
  but AC-T7 still listed exactly five new typed reasons.
  Resolution: AC-T7 admits the LOOCV residual reason as the
  sixth typed reason; DEC-T2 updated to reflect six (not five).
- AC-T2/AC-T4 calibrator parameterization: Codex second-pass
  flagged that `fit(predicted, label, arch)` could become an
  implicit per-arch head if `arch` enters as a fitted feature.
  Resolution: AC-T2 default parameterization pinned to
  "pooled global head with `arch` recorded ONLY as evaluation
  metadata"; the calibrator signature is `fit(predicted, label)`
  (no arch input to fit); per-arch residuals reported in the
  AC-T4 evaluation block. DEC-T12 records this choice.
- Lower-bound closure semantics: Codex second-pass flagged
  that v1's lower-bound (deferring AC-T3 + AC-T6) was
  presented as "Tier-2 closure" while it is actually an
  sm_86-only calibration follow-up. Resolution: lower bound
  explicitly rebrands as `sm_86-only Tier-2 calibration
  follow-up`; AC-T4 round summary MUST record
  `target_arch = sm_86_only` and MUST NOT emit any
  `cross_arch_zero_shot_prediction.json` artifact; the
  lower-bound finalize-summary MUST queue a follow-up
  `Tier-2-cross-arch` RLCR loop. DEC-T13 records this choice.
- Plan-file hygiene: Codex second-pass flagged the appended
  draft tail (preserved per the gen-plan template's
  Rule 11 requirement) as containing contradictory older
  AC-T language (e.g., the draft says "AC-T1 closes AC-4.1"
  which contradicts the v1 + v2 resolutions). Resolution: a
  preamble note added at the top of the appended draft
  section clarifies that the structured plan above supersedes
  any conflicting AC-T language in the draft; the draft is
  preserved for traceability of the original human input.

**Round 4 (user directive 2026-05-16 → sm_86-only Active Scope narrowing)**

- Scope collapse: User explicitly directed that this plan focus
  on sm_86 development, testing, and calibration against local
  NCU ground truth on the 3090 Ti. All sm_120 features defer to
  a follow-up `Tier-2-cross-arch` loop that opens only after an
  accurate NCU-calibrated sm_86 cache-behavior model is in place.
  Resolution: a high-visibility "Active Scope (Per User Directive
  2026-05-16)" section added near the top of the plan
  enumerating which ACs are in active scope (AC-T0, AC-T1, AC-T2,
  AC-T4 with `target_arch=sm_86_only`, AC-T7 with four of the six
  typed reasons, AC-T8, AC-T9, AC-T10) and which are deferred
  (AC-T3, AC-T5, AC-T6, AC-T4's leave-arch-out evaluation,
  AC-T7's two cross-arch typed reasons). The DEC resolutions
  for cross-arch items (DEC-T3, DEC-T4, DEC-T6, DEC-T7, DEC-T10,
  DEC-T11, DEC-T14) keep their values but are marked
  RESOLVED-BUT-DEFERRED. DEC-T13 is elevated from "lower-bound
  recommendation" to "primary active scope". The plan title
  updates from "Tier-2 Reuse-Distance Calibration and
  Cross-Architecture Transfer" to "Tier-2 Reuse-Distance
  Calibration (sm_86 Active Scope; Cross-Architecture
  Deferred)".

### Convergence Status

- Round 1 status: round-1 REQUIRED_CHANGES were six
  resolvable items; all six addressed in candidate plan v1.
- Round 2 status: round-2 REQUIRED_CHANGES were six
  resolvable items; all six addressed in candidate plan v2.
- Round 3 status: codex confirmation pass found three stale
  text references outside the main ACs that contradicted v2's
  main-AC fixes. The three REQUIRED_CHANGES were purely
  text-hygiene cleanup. Claude applied the cleanup in a
  Round-3-hygiene revision: "five new typed reasons" →
  "six new typed reasons" everywhere; task17 reworded to
  remove "run NVBit tracer offline" and replace with
  cuobjdump+nvdisasm+classifier-unit-test wording;
  Deliberation Agreements item updated to "pooled global
  calibration head with arch as evaluation metadata only".
- Round 4 status: user-directive scope narrowing. All 14 DECs
  resolved (4 from Phase 6 + 4 from continued Phase 6 +
  3 auto-applied consensus defaults + 3 RESOLVED-BUT-DEFERRED
  marker updates). Active-Scope section added.
- Final Status: `converged` after Round 4 scope narrowing.
  All REQUIRED_CHANGES from Rounds 1, 2, 3 addressed; all
  14 Pending User Decisions resolved; the plan's Active Scope
  is sm_86-only Tier-2 calibration with cross-arch work
  explicitly deferred to a named follow-up loop.
- Convergence rounds executed: 4 (Round 1 first-pass critique
  → candidate v1; Round 2 second-pass review → candidate v2;
  Round 3 confirmation → v2-hygiene revision; Round 4 user
  directive → Active Scope narrowing).

## Pending User Decisions

- DEC-T1: Calibration-head functional form selection.
  - Claude Position: Implement all three calibrators
    (`IsotonicCalibrator` + `PlattCalibrator` +
    `BetaCalibrator`); use a predeclared selection rule to
    pick the winner per DEC-T9.
  - Codex Position: Restricted set (Platt + isotonic only);
    hold beta behind explicit DEC. Beta's three parameters on
    ≤13 labels is risky.
  - Tradeoff Summary: All-three is more breadth and avoids
    pinning a head a-priori, at the cost of implementation
    work and a heavier model-selection concern. Restricted set
    is safer with smaller surface but loses beta's asymmetric
    over-confidence handling.
  - Decision Status: **RESOLVED — All three calibrators
    (Isotonic + Platt + Beta)**. AC-T2.1 implements all three;
    AC-T4 LOOCV evaluation evaluates all three; DEC-T9's
    `simplicity_one_se_rule` picks the winner from the full
    set (simplicity order: Platt < Isotonic < Beta).

- DEC-T2: Number of new AC-9 typed reasons.
  - Claude Position: Full SIX-reason set
    (`calibration_reuse_distance_out_of_envelope`,
    `calibration_covariate_out_of_envelope`,
    `trace_opcode_coverage_incomplete`,
    `unsupported_blackwell_memory_path`,
    `target_arch_unlabeled_extrapolation`,
    `calibration_loocv_p90_residual_exceeded`). The sixth
    reason was added in Claude/Codex round 2 to resolve the
    AC-T4/AC-T7 enum inconsistency codex second-pass flagged.
  - Codex Position: Six-reason set is more defensible than
    five (the LOOCV residual reason is clearly a headline-
    validity concern; keeping it only in
    `calibration_evaluation_v1.json` would split the
    propagation surface). Codex second-pass converges with
    Claude on this.
  - Tradeoff Summary: More enum churn vs. more honest
    typed-fail-closed surface. The active loop's pattern
    favors enumerated reasons over catch-all "typed_unavailable"
    to preserve auditable failure-mode breakdown.
  - Decision Status: **RESOLVED — Four-of-six typed reasons
    in Active Scope; two cross-arch reasons DEFERRED**.
    Post-Round-4 user directive narrows the scope to sm_86.
    In Active Scope, AC-T7 widens with FOUR typed reasons:
    `calibration_reuse_distance_out_of_envelope`,
    `calibration_covariate_out_of_envelope`,
    `trace_opcode_coverage_incomplete`,
    `calibration_loocv_p90_residual_exceeded`. The two
    cross-arch reasons —
    `unsupported_blackwell_memory_path` (requires AC-T5
    Blackwell tracer extension) and
    `target_arch_unlabeled_extrapolation` (fires only on
    cross-arch zero-shot inference) — defer to the follow-up
    `Tier-2-cross-arch` loop. The six-reason set is the
    eventual long-term enum; the four-reason subset is what
    lands in this plan.

- DEC-T3: sm_120 sweep cohort size.
  - Claude Position: Full 13 cells (5 activation + 8
    microbench) if microbench cells are stable on sm_120
    hardware.
  - Codex Position: Codex first-pass deferred this DEC; it
    surfaces only as a scheduling concern, not a methodology
    concern.
  - Tradeoff Summary: Full cohort gives stronger leave-arch-out
    statistics; activation-only is faster to land but limits
    the AC-T4 cross-arch evaluation breadth.
  - Decision Status: **RESOLVED-BUT-DEFERRED — Full 13 cells
    (5 activation + 8 microbench), DEFERRED to follow-up
    `Tier-2-cross-arch` loop per Round 4 user directive
    2026-05-16**. When AC-T3 opens (in the follow-up loop),
    it sweeps the full cohort; any microbench cell that fails
    the existing repeatability stddev thresholds
    (L2 ≤ 0.03, L1tex ≤ 0.01 per
    `tools/local_labels/repeatability_aggregator.py:664-718`)
    types as `label_repeatability_failed`; the round summary
    records the admission count and excluded cells with their
    typed reasons. AC-T4 in the follow-up loop uses whatever
    cohort admits. In THIS plan, AC-T3 does NOT open.

- DEC-T4: sm_100 cloud target reconciliation.
  - Claude Position: Datacenter B200 (sm_100, CC 10.0).
  - Codex Position: B200 is sm_100 (CC 10.0); the RTX PRO 6000
    Blackwell GB202 mentioned in
    `validating-nonroot-gpu-measurement` SKILL.md is sm_120-class
    (CC 12.0). DEC-T4 needs to clarify whether the Tier-2
    target is datacenter Blackwell sm_100 only OR "any
    Blackwell cloud trace".
  - Tradeoff Summary: sm_100 (B200) is the headline scientific
    target with the largest L2 (126 MB) and dual-die topology;
    the cloud RTX PRO 6000 (sm_120) target is the same arch as
    the local RTX 5060 laptop and so is a smaller cross-arch
    transfer claim. Both are valid Tier-2 targets; the user
    must pin which.
  - Decision Status: **RESOLVED-BUT-DEFERRED — Datacenter B200
    (sm_100, CC 10.0), DEFERRED to follow-up
    `Tier-2-cross-arch` loop per Round 4 user directive
    2026-05-16**. When AC-T6 opens (in the follow-up loop), it
    targets cloud B200. In THIS plan, AC-T6 does NOT open.

- DEC-T5: `calibration_loocv_p90_residual_exceeded` threshold
  pinning.
  - Claude Position: Pin numerically before AC-T4 round opens.
    No default proposed; the threshold depends on the AC-T4
    fit's actual LOOCV distribution.
  - Codex Position: Codex first-pass deferred this DEC.
  - Tradeoff Summary: A loose threshold lets weak fits pass;
    a tight threshold can fail the round on small-sample
    noise. Recommend pinning AFTER a Tier2-D dry-run produces
    the LOOCV distribution; pin to a value that admits the
    "best of three" head's actual p90 plus a small margin.
  - Decision Status: **RESOLVED — Pin at 0.30 a-priori**.
    AC-T4's `calibration_loocv_p90_residual_exceeded` fires
    when the chosen head's LOOCV p90 relative-error residual
    exceeds 0.30 (30 percentage points). The AC-T4 round
    contract pins this numeric threshold before the round
    opens; the round summary records the actual LOOCV p90
    against the threshold. The 0.30 figure is conservative
    relative to the active loop's Round 3 geomean CI width
    (~0.44 from `comparison_bundle.json` per the AC-1 closure
    evidence) and forces AC-T4 to produce a meaningfully
    better fit than the unconstrained baseline. If the Tier2-D
    dry-run shows the best head's LOOCV p90 is reliably above
    0.30, the AC-T4 round legitimately fails and the round
    contract must be amended to either accept a higher
    threshold (with explicit user DEC) or expand the cohort.

- DEC-T6: Tracer-side TMEM op classification path.
  - Claude Position: Simple path
    (`LDT`/`LDTM` → `MemoryOpClass::kGlobal`,
    `STT`/`STTM` → `MemoryOpClass::kGlobal`,
    `UTCCP` → `MemoryOpClass::kShared`).
  - Codex Position: Either path is valid; the precise path
    (NEW `MemoryOpClass::kTMEM` enum value) is more accurate
    semantically but adds enum churn. Codex first-pass did
    not pin a recommendation.
  - Tradeoff Summary: Simple path is YAGNI default and
    minimizes enum churn. Precise path is more accurate and
    enables future TMEM-specific RD accounting (a hypothetical
    AC-T1.3 second HRD layer specifically for TMEM cache
    behavior, which is out of this plan's scope).
  - Decision Status: **RESOLVED-BUT-DEFERRED — Precise path
    (NEW `MemoryOpClass::kTMEM` enum value), DEFERRED to
    follow-up `Tier-2-cross-arch` loop per Round 4 user
    directive 2026-05-16**. When AC-T5.1 opens (in the
    follow-up loop), it adds a `kTMEM` value to the
    `MemoryOpClass` enum in
    `tools/nvbit_tracer/tracer_tool/memory_op_class.h`
    AND to the proto schema at
    `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto`
    (additive widening per CLAUDE.md wire-extension policy).
    `LDT`/`LDTM` → `kTMEM` (load); `STT`/`STTM` → `kTMEM`
    (store); `UTCCP` → stays `kShared`. The lockstep schema
    widening lands in AC-T5.2;
    `tools/nvbit_tracer/import_nvbit_exclusion.py::VALID_OP_CLASSES`
    and `workloads/simbench_bw/schemas/comparison_bundle.schema.json`
    `memory_op_class` enum BOTH widen to admit `tmem`. In
    THIS plan, AC-T5 does NOT open.

- DEC-T7: sm_100 evidence-status terminology.
  - Claude Position: `Trace-live-sm_100` (NEW string scoped to
    NVBit-only trace evidence on sm_100); preserves
    CLAUDE.md's vocabulary discipline that
    `Verified-live-<arch>` requires the AC-relevant runtime
    surface evidence (which on sm_100 cannot be NCU per Rule
    1). Update CLAUDE.md to admit the new string into the
    Two-Device Local Development Model section.
  - Codex Position: First-pass codex flagged this exact
    issue; recommends the new vocabulary OR an existing
    typed annotation rather than `Verified-live-sm_100`.
  - Tradeoff Summary: New vocabulary is honest and prevents
    drift; existing vocabulary minimizes CLAUDE.md churn but
    blurs the no-NCU cloud rule. Recommend the new vocabulary.
  - Decision Status: **RESOLVED-BUT-DEFERRED —
    `Trace-live-sm_100` (new vocabulary), DEFERRED to
    follow-up `Tier-2-cross-arch` loop per Round 4 user
    directive 2026-05-16**. When AC-T6 opens (in the
    follow-up loop), the CLAUDE.md update widens the
    Two-Device Local Development Model section to a
    Three-Device discipline admitting `Trace-live-sm_100`.
    In THIS plan, AC-T6 does NOT open and CLAUDE.md remains
    on the Two-Device discipline.

- DEC-T8: Calibrator library dependency policy.
  - Claude Position: Permit scikit-learn (`IsotonicRegression`)
    plus hand-rolled 2-parameter Platt and 3-parameter beta;
    document the dependency in the Tier-2 round summaries.
  - Codex Position: First-pass codex flagged scikit-learn as
    a non-trivial dependency add; alternative is three
    hand-rolled implementations (numpy + scipy.special only).
  - Tradeoff Summary: scikit-learn is well-tested and saves
    implementation work. Hand-rolled keeps the dependency
    surface small and matches the project's existing
    minimal-dep posture.
  - Decision Status: **RESOLVED — scikit-learn permitted**.
    AC-T2.1 uses `sklearn.isotonic.IsotonicRegression` for
    `IsotonicCalibrator`; `PlattCalibrator` and
    `BetaCalibrator` are hand-rolled (numpy + scipy.special)
    since scikit-learn does not ship 2-parameter Platt or
    3-parameter beta calibration as first-class APIs. The
    Tier-2 opening round (Tier2-A) adds scikit-learn to the
    project's `requirements.txt` (or equivalent dependency
    pin); the AC-T2.2 sidecar provenance records
    `calibrator_libraries: {"isotonic": "sklearn",
    "platt": "stdlib_handrolled",
    "beta": "stdlib_handrolled"}` with the sklearn version
    number for reproducibility. The round summary cites
    sklearn's `IsotonicRegression` documentation URL for
    reviewers.

- DEC-T9: AC-T4 winner selection rule.
  - Claude Position: `simplicity_one_se_rule` — pick the
    simplest head whose LOOCV p90 is within one standard
    error of the best p90.
  - Codex Position: First-pass codex flagged the data-dredging
    risk and proposed predeclared rules. The five candidates
    are: `loocv_p90_minimum`, `leave_arch_out_p90_minimum`,
    `simplicity_one_se_rule`, `predeclared_isotonic`,
    `predeclared_platt`.
  - Tradeoff Summary: `loocv_p90_minimum` maximizes
    fit-quality at risk of overfitting noise.
    `leave_arch_out_p90_minimum` favors cross-arch
    generalization but is noisier with only two arches.
    `simplicity_one_se_rule` matches the literature norm for
    small-N regression. `predeclared_*` removes selection
    drama entirely at the cost of giving up the data-driven
    win.
  - Decision Status: **RESOLVED — `simplicity_one_se_rule`**.
    Round simplicity ordering: `PlattCalibrator` (2 params) <
    `IsotonicCalibrator` (PAV step function) <
    `BetaCalibrator` (3 params). The AC-T4 round contract
    pins this rule by name before the round opens; the round
    summary records both the chosen head and the LOOCV p90
    distance from the best.

- DEC-T10: Zero-shot sm_100 artifact location.
  - Claude Position: Dedicated new schema at
    `workloads/simbench_bw/schemas/cross_arch_zero_shot_prediction.schema.json`
    with `result_kind = scalar_zero_shot_inferred` as a new
    enum value local to that schema.
  - Codex Position: Same — first-pass flagged the
    `comparison_bundle.json` enum collision and recommended
    the dedicated schema.
  - Tradeoff Summary: Dedicated schema avoids polluting the
    labeled-arch enum and signals the unlabeled-target
    semantic clearly. The cost is a small consumer-side
    surface area to add (one new schema file, one new
    artifact directory).
  - Decision Status: **RESOLVED-BUT-DEFERRED — Dedicated
    schema, DEFERRED to follow-up `Tier-2-cross-arch` loop
    per Round 4 user directive 2026-05-16**. When AC-T6
    opens (in the follow-up loop), the new schema
    `workloads/simbench_bw/schemas/cross_arch_zero_shot_prediction.schema.json`
    lands. In THIS plan, AC-T6 does NOT open.

- DEC-T11: AC-T5 producer-side opcode classification ordering.
  - Claude Position: AC-T5.1 lands as Tier2-E-prep round
    (after Tier2-D, before Tier2-E AC-T6). This is the
    default sequence.
  - Codex Position: First-pass codex flagged risk-burn-down
    alternative — land AC-T5.1 in parallel with Tier2-B or
    Tier2-C as risk-burn-down for AC-T6.
  - Tradeoff Summary: Default sequence keeps the dependency
    graph linear. Risk-burn-down parallel sequence detects
    sm_100 trace feasibility issues earlier (e.g., if
    `tcgen05.ld.sync.aligned.*` cannot actually be compiled
    `-arch=sm_100a` in this toolchain, the discovery happens
    BEFORE AC-T4 invests in cross-arch fit work). The
    parallel path is recommended if scheduling permits.
  - Decision Status: **RESOLVED-BUT-DEFERRED — Risk-burn-down
    (AC-T5.1 in parallel with Tier2-B / Tier2-C), DEFERRED to
    follow-up `Tier-2-cross-arch` loop per Round 4 user
    directive 2026-05-16**. When AC-T5 opens (in the
    follow-up loop), AC-T5.1 (producer-side classification +
    offline witnesses) opens AS SOON AS the follow-up loop's
    Tier2-A-equivalent closes, in parallel with Tier2-B
    calibration-head extensions and Tier2-C sm_120 sweep;
    AC-T5.2 (schema widening) depends on AC-T5.1 closing.
    In THIS plan, AC-T5 does NOT open.

- DEC-T12: Cross-arch calibrator parameterization.
  - Claude Position: Pooled global head with `arch` recorded
    ONLY as evaluation metadata (NOT as a fitted feature).
    The calibrator sees only `(predicted, label)` pairs;
    per-arch residuals are reported in the AC-T4 evaluation
    block but the fit itself is single-arch-blind. This
    avoids implicit per-arch heads (the failure mode codex
    second-pass flagged when the AC-T2 fit signature was
    `fit(predicted, label, arch)`).
  - Codex Position: Codex second-pass requires this to be
    pinned in the plan; pooled-only or arch-offset are both
    valid; Claude's pooled-only is simpler and matches the
    small-N regime.
  - Tradeoff Summary: Pooled-only is the simplest and matches
    the small-N regime; arch-offset would add a per-arch
    intercept (`y_cal = sigmoid(a · logit(p) + b + c·arch_id)`)
    but at 13-26 cells the offset c is poorly identified.
    Pooled-only is the recommended default; the arch-offset
    variant is available if DEC-T12 escalates.
  - Decision Status: **RESOLVED — Pooled-only with arch as
    evaluation metadata**. Under the post-Round-4 Active
    Scope (sm_86-only; see top-of-plan Active Scope section),
    the "pooled across arches" framing simplifies further:
    the fit pools across CELLS within the sm_86 arch alone;
    the arch dimension is constant in this loop. When the
    follow-up Tier-2-cross-arch loop adds sm_120 + sm_100
    data, the pooled-only discipline carries forward
    unchanged.

- DEC-T13: Lower-bound closure semantics.
  - Claude Position: Lower-bound (deferring AC-T3 + AC-T6)
    closes as `sm_86-only Tier-2 calibration follow-up`,
    NOT as Tier-2 cross-arch closure. The AC-T4 round
    summary records `target_arch = sm_86_only`, MUST NOT
    emit any `cross_arch_zero_shot_prediction.json` artifact,
    and MUST queue a follow-up `Tier-2-cross-arch` RLCR loop
    in the finalize-summary's `Explicitly Deferred` section.
  - Codex Position: Codex second-pass flagged that the
    lower-bound's framing as "Tier-2 closure" is misleading
    when AC-T3 + AC-T6 defer; recommends the explicit rename
    Claude adopted.
  - Tradeoff Summary: Honest naming prevents downstream
    consumers from assuming the lower-bound closure includes
    cross-arch transfer. The cost is a slightly longer plan
    name and an extra finalize-summary section.
  - Decision Status: **RESOLVED — ELEVATED to primary scope
    per Round 4 user directive 2026-05-16**. The
    sm_86-only Tier-2 calibration follow-up is no longer the
    "lower bound"; it is the **primary active scope** for this
    plan. The prior upper-bound (sm_120 sweep + sm_100
    zero-shot + Blackwell tracer extension) is moved to the
    new Active Scope section's "Deferred to follow-up loops"
    block. See the top-of-plan Active Scope section.

- DEC-T14: AC-T5 validation split (offline vs. live).
  - Claude Position: AC-T5.1 validation is OFFLINE only
    (cuobjdump SASS-presence + classifier unit tests on
    `memory_op_class_test.cc` and `memory_op_access_kind_test.cc`).
    AC-T6 carries the live NVBit trace execution on cloud
    B200 hardware (AC-T6 is the only round that requires
    live sm_100 hardware access).
  - Codex Position: Codex second-pass flagged that the
    original AC-T5.1 phrase "traceable end-to-end without
    unknown-opcode errors" implied live tracer execution
    which is not feasible offline; the split Claude adopted
    resolves this.
  - Tradeoff Summary: The offline-only AC-T5.1 path is the
    only feasible discipline since live sm_100 access is a
    Tier2-E precondition, not a Tier2-E-prep precondition.
    The cost is that AC-T5.1 closure does NOT prove the
    classifier handles real Blackwell binaries — that proof
    lands in AC-T6.
  - Decision Status: **RESOLVED — DEFERRED with AC-T5 per
    Round 4 user directive 2026-05-16**. AC-T5 entirely
    defers to the follow-up Tier-2-cross-arch loop; DEC-T14
    is moot in this plan's Active Scope.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain
  plan-specific terminology such as "AC-T", "Tier-2",
  "Milestone", "Step", "Phase", or similar workflow markers.
  These terms belong in the plan document, not in the
  resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.
  For example: `IsotonicCalibrator` (not `AC_T2_1_IsotonicHead`),
  `cache_model_geometry_provenance` (not
  `tier2_a_geometry_block`),
  `cross_arch_zero_shot_prediction` (not
  `ac_t6_zero_shot_artifact`),
  `calibration_loocv_p90_residual_exceeded` (not
  `tier2_d_residual_exceeded_reason`).
- Round-summary documents under
  `.humanize/rlcr/<new-session-id>/round-<n>-summary.md` MAY
  reference AC-T<n> identifiers since those are plan-document
  artifacts, not codebase artifacts.

--- Original Design Draft Start ---

> **Preamble note (added by gen-plan Round 2 hygiene fix)**:
> The structured plan ABOVE this line is the AUTHORITATIVE
> Tier-2 plan after two rounds of Claude/Codex convergence
> review. The draft preserved BELOW this line is the original
> human input (per gen-plan template Rule 11) and contains
> earlier AC-T language (e.g., "AC-T1 closes AC-4.1" in the
> draft's §3 Goal Description and AC-T1 narrative) that
> CONTRADICTS the post-Round-2 structured plan above
> (resolved in §"Resolved Disagreements" → Round 1 → AC-T1
> ownership; reinforced in §"Resolved Disagreements" → Round 2
> → AC-T8 active-plan amendment scope). When the structured
> plan and the draft conflict, the structured plan governs.

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

--- Original Design Draft End ---
