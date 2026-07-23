# sm_86 Non-Root Wall-Clock / Kernel-Duration Prediction (Focused Solve)

## Goal Description

Build and VALIDATE, live on the RTX 3090 Ti (sm_86, GA102/Ampere; HEAD `083c178a`, branch
`cleanup/pre-merge-simplification`), a non-root, NCU-free predictor of per-kernel execution time.
The DVFS-stable **cycle count** (`gpc__cycles_elapsed.max`) is the sole load-bearing prediction
headline; wall-clock **nanoseconds** are a derived, clock-uncertainty-banded diagnostic that is only
promoted to a deploy-realistic headline behind an explicit CUPTI-Activity-vs-NCU agreement
(crosswalk) artifact AND a cohort long enough to clear the measured dual-source noise floor. NCU is
used ONLY on this local card to capture ground-truth labels, quarantined to `tools/local_labels/` →
`data/local_labels/`. At deploy there is no NCU, no driver-root, no clock-lock, and no privileged
CUPTI; the predictor consumes only NVBit address/SASS traces, static SASS, launch geometry, and
read-only NVML / CUPTI-Activity covariates.

This is a dedicated effort split out of `plans/sm86_unified_multimetric_predictor_plan.md`: per the
operator decision it SUPERSEDES that plan's timing legs (its AC-6..AC-10 / task10–task13), executes
additively, leaves the cache/traffic legs and all frozen invariants untouched, and merges the
revised timing module back after closure.

The wall-clock leg "fails dramatically" today because of a **regime collision**, not a built
predictor failing: (1) no duration predictor, scorer, roofline prior, or cycle→ns runner exists
(greenfield — `workloads/simbench_bw/comparison_runner.py` scores cache hit-rate only); (2) all 12
labeled sm_86 timing cells are 2310.4–7110.4 ns while a deploy-validatable headline needs kernels
above the **provenanced 110,352 ns p99 dual-source noise floor**
(`workloads/simbench_bw/golden/cupti_dual_source_tolerance.json`) — zero cells qualify for the
planned >200 µs gate; (3) cycles is DVFS-stable (implied clock 1.87 GHz, CV 0.35 %) but ns is merely
cycles × a near-constant 6.4 ns-quantized clock, so an ns-MAPE double-counts the cycle-MAPE; (4) the
single 1350 MHz NVML conversion anchor (sustained-matmul throttle) mis-matches the labels' ~1867 MHz
boost, injecting a ~38 % systematic ns bias; (5) no timing accuracy gate exists anywhere in code.
The solve therefore: makes cycles the headline, authors a long-duration deploy-validatable cohort,
fixes the conversion anchor, builds the predictor, defines and passes a real cycle-accuracy gate, and
either passes or honestly refutes the deploy-ns leg.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic
verification. The `AC-*` items are the current RLCR completion gates for this loop. CLAUDE.md Rules
1–5 (NCU quarantine; no clock-lock/persistence/power/MIG mutation; no privileged CUPTI but CUPTI
Activity allowed; no host-driver; no simulator) are binding throughout. All eight ACs are HARD gates.

- AC-1: **Timing-prediction contract reset (cycles headline; ns gated).** A committed, schema-valid
  timing-contract artifact (produced under the timing toolchain) declares: cycles
  (`gpc__cycles_elapsed.max`) is the sole publication headline target; ns is a derived
  clock-uncertainty interval / diagnostic that may NOT close `research_success` on its own; the
  publication gate (N ≥ 10, log-space geomean relative error ≤ 0.15, median absolute log-relative
  error ≤ 0.10, 95 % bootstrap upper bound ≤ 0.25) and the smoke tier (N ≥ 5, reports only); the
  deploy-ns prerequisite (a CUPTI-vs-NCU crosswalk) and floor-relative cohort threshold (≥ ~552 µs
  for a 20 % bound); the terminal vocabulary (`research_success` / `verdict_falsification` /
  `deferred_cross_arch_requires_sm120_session`); and the sm_86-only evidence statuses.
  - Positive Tests: the contract artifact validates against its schema and a parser resolves every
    threshold + the cycles-only-headline rule + the ns-crosswalk prerequisite.
  - Negative Tests: a contract that permits a sub-floor ns-MAPE (or any ns leg) to close
    `research_success` is rejected; a contract asserting a co-equal ns headline on sub-10 µs cells
    is rejected.
  - AC-1.1: **Exact cycle-error definition (pinned formulas).** For each scored cell `i` with NCU
    true cycles `t_i > 0` and predicted cycles `p_i > 0`, the per-cell signed log-ratio is
    `r_i = ln(p_i / t_i)`. The three headline metrics are: (a) log-space geomean relative error
    `G = exp(mean_i(|r_i|)) − 1` (gate `G ≤ 0.15`); (b) median absolute log-relative error
    `M = median_i(|r_i|)` (gate `M ≤ 0.10`); (c) the 95 % bootstrap upper bound on `G ≤ 0.25`
    (2000 resamples, CI 0.95, resampling cells with replacement). Predictions and labels are strictly
    positive (cycles ≥ thousands), so NO epsilon floor is applied — unlike the bounded-fraction cache
    runner. The contract (AC-1) pins these formulas verbatim and the timing runner (AC-6) implements
    them.
    - Positive: a fixture of `(p_i, t_i)` pairs scores to the `G` / `M` / bootstrap values the pinned
      formula prescribes.
    - Negative: a runner that computes `G` or `M` via the bounded-fraction cache formula
      (`|pred − true| / max(true, eps)`) instead of the log-ratio formula is rejected.

- AC-2: **Static-SASS structural features + structural-work artifact + GA102 peak constants.** A new
  static-SASS extractor (`static_sass_features.py` does not exist today) emits, per kernel binary, a
  schema-valid `static_sass_features` record (instruction-class histogram + register usage; the
  existing closed schema is left unchanged). The structural FLOP/byte estimate the roofline prior
  consumes is emitted in a SEPARATE NEW schema-valid static-work artifact whose schema lists
  `claim_class` (`static_sanity_check_only`) in `required` so a record cannot omit the claim-class
  discipline. GA102 peak constants (FP32 / tensor FLOP·s⁻¹, DRAM GB·s⁻¹, base/boost SM clock, memory
  clock) are added ADDITIVELY (to `tools/hrd_profiler/device_properties.json` — which declares no
  top-level `additionalProperties:false` — or a sourced sidecar) with explicit provenance, and the SM
  count is re-derived (not guessed) from a live device query / vendor spec.
  - Positive Tests: the extractor emits a schema-valid `static_sass_features` record AND a schema-valid
    static-work artifact carrying the FLOP/byte estimate with a present `claim_class` field; the GA102
    peaks artifact validates and records source/provenance; the SM count matches a live device query.
  - Negative Tests: a static-work record missing `claim_class`, or asserting the FLOP/byte estimate as
    a stand-alone identifiability claim beyond a sanity-check covariate, is rejected; attempting to add
    a FLOP/byte field into the closed `static_sass_features.schema.json` (`additionalProperties:false`)
    fails validation; an unsourced / fabricated peak or a guessed SM count fails the provenance check.

- AC-3: **Label-matched NVML short-burst reference anchor (read-only).** A committed, schema-valid
  read-only NVML short-burst anchor artifact — captured under a workload matched to the inference
  launch pattern so it reflects boost (~1.87 GHz), not the sustained-throttle 1350 MHz operating
  point — records workload, poll rate, accepted/rejected samples, and clock dispersion, AND carries a
  first-class regime discriminator (a new `artifact_kind`, or an additive `anchor_regime ∈
  {sustained, short_burst}` / `workload_class` field, since the current
  `nvml_reference_operating_point` schema is `additionalProperties:false` with a fixed `artifact_kind`)
  so the anchor loader selects by regime; the `cycles_to_ns` conversion + anchor loader select the
  `short_burst` anchor for the short regime and CANNOT silently consume the sustained 1350 MHz
  artifact, while still failing closed on missing / implausible clock.
  - Positive Tests: the short-burst anchor artifact exists, validates, carries the regime
    discriminator, and records its provenance; the loader selects it for short-regime conversion and
    the converted ns of a labeled cell lands within the clock-quantization band of its NCU ns.
  - Negative Tests: any clock-lock / persistence / power / MIG mutation fails (Rule 2); a missing or
    implausible (outside [300, 3000] MHz) clock fails closed; a short-regime conversion that resolves
    the sustained (`anchor_regime=sustained`, 1350 MHz) artifact is rejected; an anchor artifact
    lacking the regime discriminator fails.

- AC-4: **Long-duration deploy-validatable timing cohort (scoped, additive).** Either (a) ≥ 10
  sm_86 long timing cells — each with NCU duration ≥ the chosen ~552 µs floor — captured against ONE
  current binary identity (consistent `binary_sha256`, `kernel_signature_sha256`, `source_commit`),
  each carrying cycles + ns + L1 + L2 labels and passing per-unit repeatability (cycles/ns CV ≤ 0.05,
  fractions stddev ≤ 0.03, n ≥ 5) AND the dual-source pass-rate gate, recorded in a SCOPED additive
  timing-cohort artifact (kept out of root `label_admitted.json` until closure, then merged) so the
  root "every sm_86 cell carries timing OR a typed exclusion" invariant and the frozen-L2 / sm_120
  freeze are never perturbed; OR (b) a schema-valid `verdict_falsification` artifact records why the
  long cohort could not be populated. Each long cell must ALSO carry the deploy-legal feature
  artifacts AC-5 needs to score it (NVBit `summarize_launch` sector summaries + the AC-2 static-SASS /
  static-work records), not labels alone, so it is label-complete AND scorable. Branch (b) is a
  TERMINAL BLOCKING outcome, not a `research_success` substitute: if the long cohort is not populated,
  AC-6's publication gate cannot pass on a long cohort and the loop terminates in the Lower-Bound
  BLOCKED/escalation state.
  - Positive Tests: a coverage scanner confirms ≥ 10 long cells each ≥ floor against one binary with
    all four labels + repeatability metadata + the deploy-legal feature artifacts in the scoped
    artifact; the merge into root is purely additive (root cohort key set unchanged for pre-existing
    cells); OR the `verdict_falsification` artifact validates and is schema-correct.
  - Negative Tests: zero cells ≥ floor admitted to the headline; mixed `binary_sha256` for the same
    shape; a long cell missing L1, L2, timing, or its NVBit/static-SASS feature artifacts; an untyped
    exclusion or a silent cross-binary bind; a scoped→root merge that mutates a frozen-L2 or sm_120
    cell — each fails. A `verdict_falsification` branch (b) counted as `research_success` fails.

- AC-5: **Cycles execution-time predictor (pedestal + analytical prior + regularized residual).** A
  predictor emits, per cell, `pedestal_cycles` (nonnegative fixed launch/drain overhead) + an
  analytical work term (`compute_cycles` from the static-SASS / FLOP mix; `memory_cycles` from NVBit
  sector traffic × Ampere latency constants, amortized by the existing predicted L1/L2 sector hit
  rates and active-blocks theoretical occupancy) + a strongly-regularized ML residual → a final cycle
  prediction, with full feature provenance, model version, and typed-unavailable reasons; all inputs
  are deploy-legal.
  - Positive Tests: a predictor bundle for an admitted cell records `pedestal_cycles`, the analytical
    prior term, the residual term, the final prediction, the feature vector, provenance, and model
    version; both the prior and residual contributions are separately auditable.
  - Negative Tests: using a timing label, achieved occupancy, IPC, any privileged-counter / NCU
    metric, OR a `%clock64`-derived value as a SILENT required headline input fails the
    deployability-leakage guard; a prediction lacking the pedestal/prior/residual decomposition fails.

- AC-6: **Cycles accuracy runner + gate (HARD pass).** A dedicated timing accuracy runner uses a
  timing-specific join key (the cache runner's 8-tuple MINUS `cache_layer`, PLUS `metric_target` =
  cycles — it must NOT reuse the `cache_layer` join), reuses the pinned bootstrap (2000 resamples,
  CI 0.95, log-space geomean) and coverage floors (smoke 5 / publication 10 per arch), emits regime
  tags (short/long) per cell, and reports cycle log-geomean relative error + median absolute
  log-relative error + 95 % bootstrap upper bound — PASSING the AC-1 publication gate (N ≥ 10,
  geomean ≤ 0.15, median ≤ 0.10, bootstrap upper ≤ 0.25) on the long cohort.
  - Positive Tests: the runner emits the three metrics + bootstrap CI + per-arch coverage + regime
    tags over the long cohort and the publication gate passes; an N ≥ 5 smoke run reports without
    closing `research_success`.
  - Negative Tests: a runner that reuses the cache-layer 8-tuple join (carrying `cache_layer`) fails;
    below-floor coverage or a missing bootstrap yields a typed-unavailable, not a pass; a publication
    closure declared on N < 10 fails.

- AC-7: **Deploy-ns crosswalk + honest closure.** A same-binary / same-shape CUPTI-Activity+cudaEvent
  vs NCU `gpu__time_duration.sum` agreement (crosswalk) artifact exists and validates BEFORE any
  deploy-ns claim; closure emits `research_success` for the passed cycles headline and, for the
  deploy-ns leg, EITHER a passing floor-relative ns result (long cohort ≥ ~552 µs, 20 % bound) OR a
  schema-valid `verdict_falsification` citing the measured 110 µs floor; ns is reported as a
  clock-uncertainty interval derived from the AC-3 anchor, never as a silent co-equal headline.
  - Positive Tests: the crosswalk artifact exists + validates; closure emits `research_success`
    (cycles) plus either a deploy-ns pass or a `verdict_falsification`; the reported ns is an interval
    derived from the label-matched anchor.
  - Negative Tests: an NCU-ns conversion diagnostic labeled as a deploy wall-clock result fails; an
    `indeterminate` (attribution in [10 %, 30 %) or n < 8) falsification counted as closure fails; a
    deploy-ns point result claimed without the crosswalk artifact fails.

- AC-8: **Compliance + frozen-invariant gate (sm_86-only).** The no-root / no-NCU static guard, the
  plan-token scanner, the frozen L2 v5 byte-identity guard, the sm_120 cohort-freeze guard, the
  190-file path-boundary guard, the schema-v5 lockstep checks, and the deployability-leakage guard
  all pass; every new timing artifact is additive; round summaries record `Verified-live-sm_86` or
  `Offline-verified` only.
  - Positive Tests: all listed guards pass; the frozen L2 v5 head sha256 and headline geomean
    reproduce within 1e-9, the sm_120 cohort is byte-identical, and the path-boundary baseline
    (190 files, each pinned by SHA-256) matches byte-for-byte after all timing work.
  - Negative Tests: NCU invoked outside the quarantine, a privileged-CUPTI surface, a
    clock-lock/persistence/power/MIG mutation, an sm_120 mutation, a frozen-L2 byte drift, a byte
    edit to the CUPTI collector (`cupti_activity_collector.cu/.h`) or NVBit tracer
    (`tools/nvbit_tracer/`) — both out of scope for this plan, edited not at all — a plan-token in implementation
    code, or any cross-arch / sm_120 claim on this host — each fails the relevant guard.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
All eight ACs met: a cycles-headline timing-prediction contract; a static-SASS/FLOP extractor +
sourced GA102 peaks; a label-matched read-only NVML short-burst anchor; a ≥ 10-cell long-duration
(≥ ~552 µs) deploy-validatable cohort carrying cycles + ns + L1 + L2 against one binary (ideally
N ≥ 15 with ≥ 4 memory-bound and ≥ 4 compute-bound cells per DEC-4); a pedestal + roofline-prior +
regularized-residual cycle predictor; a dedicated timing runner that HARD-PASSES the cycle-accuracy
gate; a CUPTI-vs-NCU ns crosswalk with a passing floor-relative deploy-ns headline; full closure +
compliance; the whole loop `Verified-live-sm_86` with the frozen invariants green.

### Lower Bound (Minimum Acceptable Scope)
The cycles headline is the load-bearing deliverable and MUST hard-pass: AC-1 contract + AC-2
extractor/peaks + AC-3 anchor + AC-4 long cohort (≥ 10 cells, all four labels, one binary) + AC-5
predictor + AC-6 cycle-accuracy gate PASS + AC-8 compliance. For the deploy-ns leg (AC-7), the
minimum acceptable outcome is an emitted, schema-valid `verdict_falsification` (cycles passes;
deploy-ns honestly refuted below the floor) — never a silent or "reported-only" ns. If the cycle
gate itself is genuinely unreachable after exhausting the latitude below, the loop terminates in an
escalation / BLOCKED state with a decomposed residual report — NOT a declared `research_success`.

### Allowed Choices
- Can use: NVBit address/SASS traces + `%clock64` ledger (ablation/feature only), CUPTI Activity,
  CUDA events, read-only NVML covariates, static SASS via `cuobjdump`/`nvdisasm`, a roofline/interval
  analytical prior + pedestal + regularized ML residual, the existing predicted L1/L2 hit-rate heads
  as covariates, the pinned bootstrap (2000/0.95) + coverage floors (5/10) with a NEW timing-specific
  join key, NCU on this local 3090 Ti for labels ONLY into `data/local_labels/` via
  `tools/local_labels/`, additive shape_points on existing portable_lcd slugs and new long kernels.
  The executing agent has WIDE LATITUDE to research, refactor, or rewrite the timing prediction
  surface — the predictor, the timing runner, the label/capture tooling, the new timing schemas, and
  the analysis code — to reach the cycle-accuracy gate. This latitude EXCLUDES the immutable 190-file
  path-boundary producer stack (the CUPTI collector `cupti_activity_collector.cu/.h` and the NVBit
  tracer `tools/nvbit_tracer/`), which is out of scope and not edited at all this loop.
- Cannot use: privileged CUPTI (Events/Metrics/Range/PM/PerfWorks); NCU outside the quarantine;
  `nvidia-smi` clock/power/persistence/MIG mutation or clock-locking; host-driver install/DKMS/MOK/
  reboot; simulator revival; achieved occupancy or IPC as deployable inference inputs; a silent
  `%clock64` headline input; any edit to the frozen L2 v5 head scoring, the sm_120 frozen cohort, or
  the 190-file path-boundary producer stack (CUPTI collector + NVBit tracer) — these are byte-frozen
  and OUT OF SCOPE for this plan, not edited at all; any sm_120 / cross-arch claim or artifact on this
  machine.

> **Note on Deterministic Designs**: The architecture (cycles-primary headline, pedestal + roofline
> prior + regularized residual, label-matched NVML anchor, scoped-additive long cohort, ns gated
> behind a crosswalk + honest falsification) is fixed per the draft and the Codex consults; the path
> boundaries are correspondingly narrow. Remaining latitude is confined to HOW the predictor and the
> long cohort are built, and to the RESOLVED decisions DEC-1..DEC-9.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only — conceptual suggestions, not
> prescriptive requirements.

### Conceptual Approach
1. **Contract first (M0).** Author the timing-prediction contract + its schema/validator, encoding
   the cycles-only headline, the DEC-2 thresholds, the ns-crosswalk prerequisite, the floor-relative
   ns threshold, the terminal vocabulary, and the evidence statuses.
2. **Substrate (M1-prep).** Build the static-SASS / FLOP extractor against
   `static_sass_features.schema.json` (sanity-check claim-class) and add sourced GA102 peaks
   additively; capture the label-matched read-only NVML short-burst anchor and rewire `cycles_to_ns`
   to prefer it for the short regime.
3. **Cohort (M1).** Cohort-FIRST: scale 3–5 candidate portable_lcd slugs (e.g.
   `gemm_hmma_bf16_sm80`, `conv2d_implicit_gemm_bf16_sm80`, `flash_attn_fwd_bf16_sm80`, plus ≥ 1
   memory-bound / divergent kernel) via new long shape_points; probe NCU cycles/ns CV + the
   dual-source pass-rate at the long size BEFORE full feature/tracing work; then scale to ≥ 10
   (aim ≥ 15, balanced) deterministic single-binary cells carrying cycles + ns + L1 + L2 in the
   scoped additive cohort artifact.
4. **Predictor (M2).** Fit `pedestal_cycles + analytical_prior + regularized_residual`; let the prior
   + pedestal dominate at small N; record both terms; leakage-guard the inputs.
5. **Score (M3).** Dedicated timing runner with the timing-specific join key (no `cache_layer`),
   reusing the bootstrap + floors; report log-geomean / median / bootstrap-upper + regime tags;
   hard-pass the cycle gate.
6. **ns + closure (M4).** Build the same-binary/shape CUPTI-vs-NCU crosswalk; emit `research_success`
   for cycles + a deploy-ns pass OR `verdict_falsification`; ns as a clock-uncertainty interval.
7. **Compliance (M5).** Verify every guard green and every new artifact additive.

### Relevant References
- `plans/research/sm86_wallclock_prediction_research_dossier.md` — the verified research dossier (§1–§8).
- `tools/local_labels/capture_timing_labels.py` — the NCU cycles/ns capture + CV gate + additive
  splice + typed-exclusion pattern to extend for the combined cycles+ns+L1+L2 long-cell capture.
- `tools/local_labels/metric_registry.json` — `exec_cycles_gpc_max` / `exec_duration_ns` (units, CV
  rule, `deployability=label_only`).
- `data/local_labels/label_admitted.json` — the 12 short timing cells (the regime-collision evidence).
- `data/local_labels/nvml_reference_operating_point_sm_86.json` + its schema — the 1350 MHz anchor
  to supersede for the short regime; `workloads/simbench_bw/nvml_covariate_join.py:225-257`
  (`cycles_to_ns`, fail-closed).
- `workloads/simbench_bw/golden/cupti_dual_source_tolerance.json` — the provenanced 110,352 ns p99 floor.
- `workloads/simbench_bw/cupti_activity_collector.cu/.h` + `schemas/cupti_activity_sidecar.schema.json`
  — the deploy observable (IMMUTABLE producer; consume its output for the crosswalk).
- `workloads/simbench_bw/comparison_runner.py` — bootstrap 2000/0.95 + floors 5/10/20 + the 8-tuple
  (incl. `cache_layer`) join to reuse-constants-but-not-join-shape.
- `tools/hrd_profiler/observation_builder.py` — NVBit `summarize_launch` features (the shared input).
- `tools/hrd_profiler/device_properties.json` — cache geometry today; GA102 peaks added additively.
- `workloads/simbench_bw/schemas/static_sass_features.schema.json` — the (sanity-check-only) schema
  the new extractor must satisfy.
- `workloads/portable_lcd/roster.toml` + slug `main.cu` files (`--shape_point`) — the long-cohort
  population path (`timing_00` today; add long shape_points additively).
- `tools/local_labels/emit_v5_closure_artifacts.py` + `test_sm86_v5_closure_artifacts.py` — the
  closure/falsification verdict bands (confirmed ≥ 30 % / refuted < 10 % / indeterminate [10 %,30 %) /
  insufficient n < 8) and the terminal-state vocabulary.
- `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` — the no-NCU + 190-file
  path-boundary guard; `tools/local_labels/test_frozen_l2_v5_regression_guard.py`,
  `test_sm120_cohort_frozen_at_b1646ef7.py`, `test_no_deployability_leakage_guard.py` — frozen guards.

## Dependencies and Sequence

### Milestones
1. **M0 — Contract reset** (AC-1). The timing-prediction contract + schema/validator; the
   cycles-only-headline, thresholds, ns-crosswalk prerequisite, terminal vocabulary.
2. **M1 — Substrate + cohort + anchor** (AC-2, AC-3, AC-4).
   - Phase A: static-SASS/FLOP extractor + sourced GA102 peaks (AC-2).
   - Phase B: label-matched read-only NVML short-burst anchor + `cycles_to_ns` rewire (AC-3).
   - Phase C: cohort-first long-kernel authoring + combined cycles+ns+L1+L2 capture into the scoped
     additive cohort artifact (AC-4).
3. **M2 — Cycles predictor** (AC-5). Pedestal + analytical prior + regularized residual; depends on
   M1 (features from AC-2, anchor from AC-3, labels from AC-4).
4. **M3 — Accuracy runner + gate** (AC-6). Timing-specific join key; bootstrap + floors; hard-pass.
   Depends on M1 (labels) + M2 (predictions).
5. **M4 — ns crosswalk + closure** (AC-7). CUPTI-vs-NCU crosswalk; `research_success` for cycles +
   deploy-ns pass OR `verdict_falsification`. Depends on M1/M2/M3.
6. **M5 — Compliance** (AC-8). All guards green; artifacts additive. Depends on all milestones.

Dependencies are by artifact, not time: the contract (M0) governs every downstream gate; the
substrate (M1 A/B) and the long cohort (M1 C) feed the predictor (M2); the predictor feeds the runner
(M3); the runner + crosswalk feed closure (M4); compliance (M5) audits everything.

## Task Breakdown

Each task carries exactly one routing tag (`coding` = Claude; `analyze` = Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Author the timing-prediction contract artifact + its schema + validator test (cycles-only headline; DEC-2 publication/smoke thresholds; ns-crosswalk prerequisite; floor-relative ns threshold; terminal vocabulary; sm_86-only evidence statuses) | AC-1 | coding | - |
| task2 | Build the static-SASS extractor (structural histogram + register usage → the unchanged closed `static_sass_features` schema) PLUS a SEPARATE new static-work artifact + schema (the FLOP/byte estimate, with `claim_class` in `required`); add sourced+provenanced GA102 peak constants additively (no closed-schema edit) and re-derive the SM count from a live device query | AC-2 | coding | - |
| task3 | Codex review: ratify the static-SASS FLOP-extraction approach, the GA102 peak values + provenance, and the sanity-check claim-class discipline (no over-claimed identifiability) | AC-2 | analyze | task2 |
| task4 | Capture the label-matched read-only NVML short-burst anchor artifact; rewire `cycles_to_ns` to prefer it for the short regime, refuse the sustained 1350 MHz anchor for short-regime point ns, and keep failing closed on missing/implausible clock | AC-3 | coding | task1 |
| task5 | Cohort-first: add long (≥ ~552 µs) shape_points to candidate slugs (gemm/conv/attention + ≥ 1 memory-bound/divergent), deterministic single-binary build, capture cycles+ns+L1+L2 into a scoped additive cohort artifact with per-unit repeatability + dual-source pass-rate + a coverage scanner; else emit `verdict_falsification` | AC-4 | coding | task1, task4 |
| task6 | Codex review: long-cohort population strategy (3–5 probe shapes; CV + dual-source pass-rate thresholds at the long size; memory/compute-bound balance per DEC-4; the scoped→root merge policy per DEC-5) | AC-4 | analyze | task5 |
| task7 | Build the cycles predictor: `pedestal_cycles` + analytical roofline prior (compute/memory cycles from static-SASS/FLOP + NVBit sectors × Ampere latency, amortized by predicted L1/L2 hit rates + theoretical occupancy) + strongly-regularized ML residual; emit predictor bundles with provenance/model-version; leakage-guarded inputs | AC-5 | coding | task2, task4, task5 |
| task8 | Codex review: the roofline+pedestal+residual model design (falsifiable structure, feature set, regularization vs N, error propagation, no-leakage, no silent `%clock64` headline input) | AC-5 | analyze | task7 |
| task9 | Build the dedicated timing accuracy runner: timing-specific join key (8-tuple minus `cache_layer`, plus `metric_target`=cycles); reuse pinned bootstrap (2000/0.95) + coverage floors (5/10); emit regime tags; report log-geomean/median/bootstrap-upper; hard-pass the AC-1 publication gate | AC-6 | coding | task5, task7 |
| task10 | Build the same-binary/shape CUPTI-Activity-vs-NCU `gpu__time_duration.sum` crosswalk artifact; emit closure (`research_success` for cycles + deploy-ns pass OR `verdict_falsification` citing the 110 µs floor); report ns as a clock-uncertainty interval from the AC-3 anchor | AC-7 | coding | task4, task5, task9 |
| task11 | Codex review: the ns crosswalk methodology + closure/falsification verdict logic (bands, indeterminate-not-closure, interval honesty, NCU-vs-deploy observable distinction) | AC-7 | analyze | task10 |
| task12 | Verify the compliance + frozen-invariant gate: no-root/no-NCU static guard, plan-token scanner, frozen-L2 v5 byte-identity, sm_120 freeze, 190-file path boundary, schema-v5 lockstep, deployability-leakage; confirm every new timing artifact is additive | AC-8 | coding | task1, task2, task4, task5, task7, task9, task10 |

## Future Work / Out of Scope

- FUT-1: **Formal two-regime (short/long) wall-clock architecture (D2).** The full short-overhead /
  long-compute regime model, with a tuned boundary tied to the measured floor.
  - Source DEC: DEC-6
  - Current-loop handoff: AC-6 (the runner emits per-cell regime tags now, enabling the later split).
  - Promotion trigger: a follow-up loop after the long cohort is populated and stable.
- FUT-2: **`%clock64` phase-cycle ledger taken live as an ablation feature.** Wire the existing
  (offline-only) ledger producer to emit a live JSON and report a with/without ablation.
  - Source DEC: DEC-7
  - Current-loop handoff: AC-5 (negative test forbids a silent `%clock64` headline input).
  - Promotion trigger: after the cycles headline passes and an ablation is warranted.
- FUT-3: **sm_120 cross-arch wall-clock validation.** Port + validate the timing predictor on the
  RTX 5060 laptop (sm_120) host.
  - Source DEC: DEC-9
  - Current-loop handoff: AC-8 (records `Verified-live-sm_86` only; asserts no cross-arch claim).
  - Promotion trigger: a development session on the separate sm_120 host.
- FUT-4: **10 % floor-relative deploy-ns headline (cohort ≥ ~1.10 ms).** Tighten the deploy-ns bound
  from the 20 % default to 10 % by growing the cohort above ~1.10 ms.
  - Source DEC: DEC-2
  - Current-loop handoff: AC-7 (this loop delivers the crosswalk + EITHER the 20 % floor-relative
    deploy-ns result OR an honest `verdict_falsification`).
  - Promotion trigger: after the 20 % headline passes and longer kernels are authored.
- FUT-5: **Recover the 7 `timing_binary_sha256_mismatch` short cells.** Deterministically rebuild +
  recapture to raise short-regime cycle coverage from 12 toward 19.
  - Source DEC: DEC-8
  - Current-loop handoff: AC-4 (the long cohort, not the short recovery, is the headline path).
  - Promotion trigger: opportunistic, after the long cohort + cycle gate are secured.
- FUT-6: **Merge the revised timing module back into the unified multi-metric loop.** Reintegrate the
  validated timing legs into `plans/sm86_unified_multimetric_predictor_plan.md`.
  - Source DEC: DEC-3
  - Current-loop handoff: AC-8 (this loop executes additively, leaving cache/traffic legs untouched).
  - Promotion trigger: after this loop closes (`research_success` or emitted `verdict_falsification`).

## Claude-Codex Deliberation

### Agreements
- The wall-clock leg fails due to a regime collision (greenfield predictor + 2–7 µs cohort vs > 200 µs
  gate vs 110 µs floor), not a built predictor failing.
- Cycles is the sole load-bearing headline; ns is a derived interval gated behind a CUPTI-vs-NCU
  crosswalk; ns-MAPE on the current cohort double-counts cycle-MAPE and is not honest.
- The accuracy gate (absent today) is cycle log-geomean ≤ 0.15 / median ≤ 0.10 / bootstrap-upper
  ≤ 0.25 at N ≥ 10; the L2 bounded-fraction median 0.06 does NOT transfer to a log-target error.
- A long-duration (≥ ~552 µs) deploy-validatable cohort is the real critical path; cohort-first
  probing precedes the model fit.
- The static-SASS/FLOP extractor must be built from scratch; GA102 peaks must be sourced additively;
  the SM count must be re-derived, not guessed.
- The dedicated timing runner reuses the bootstrap + coverage-floor constants but defines its own
  join key (no `cache_layer`).
- The CUPTI collector + NVBit tracer are immutable path-boundary producers; the predictor consumes
  their existing output.
- Honest refutation (`verdict_falsification`) is a first-class terminal outcome for any leg that
  cannot pass; a silent or "reported-only" ns is unacceptable.

### Resolved Disagreements
- **Cohort-first vs cycles-first.** Codex preferred cohort-first (probe 3–5 long shapes, measure CV +
  dual-source pass-rate, then scale) over building the model on the empty deploy-valid cohort.
  Resolution: cohort-first sequencing inside M1 (task5 probes before full capture), with the cycle
  predictor (M2) consuming the populated cohort. Adopted.
- **Scoped artifact vs immediate root admission.** Codex flagged that the root "every sm_86 cell has
  timing or a typed exclusion" invariant + the frozen-L2 / sm_120 freeze make immediate root splices
  risky. Resolution: new long cells live in a SCOPED additive cohort artifact until closure, then
  merge (DEC-5). Adopted.
- **ns-MAPE as a co-equal headline (the unified plan's AC-8).** Rejected: ns is gated behind the
  crosswalk + floor-relative cohort, otherwise it is a clock-uncertainty interval only.
- **Second-Codex convergence round (5 REQUIRED_CHANGES, all applied).** (1) AC-1.1 pins the exact
  cycle-error formulas (`r_i = ln(p_i/t_i)`; `G = exp(mean|r_i|)−1 ≤ 0.15`; `M = median|r_i| ≤ 0.10`;
  bootstrap-upper ≤ 0.25), since the L2 bounded-fraction formula must not be copied. (2) AC-2 moves the
  FLOP/byte estimate into a SEPARATE new schema-valid static-work artifact (the closed
  `static_sass_features.schema.json` forbids the field). (3) AC-2 requires `claim_class` in the
  static-work schema's `required` (the schema otherwise lets a record omit the discipline). (4) AC-3
  adds a first-class anchor regime discriminator so short-regime conversion cannot silently consume the
  sustained 1350 MHz artifact. (5) The path boundary + AC-8 + Implementation Notes tighten the 190-file
  producer stack to byte-frozen / OUT OF SCOPE (no "amendment + re-pin" loophole). Three optional
  improvements were also applied: AC-4 marks the `verdict_falsification` branch a TERMINAL BLOCKING
  outcome (not a `research_success` substitute) and requires per-cell deploy-legal feature artifacts;
  FUT-4 wording reconciled with AC-7's falsification branch.

### Convergence Status
- Final Status: `converged` after one second-Codex convergence round (all 5 REQUIRED_CHANGES + 3
  OPTIONAL_IMPROVEMENTS applied). Decisions DEC-1..DEC-9 are RESOLVED; no `PENDING` items remain. One
  non-blocking operator-tightening option is noted on DEC-4 (raise the long cohort from N ≥ 10 to a
  balanced N ≥ 15 hard gate); the plan is executable at the locked N ≥ 10 floor and the operator may
  tighten it on review.

## Pending User Decisions

- DEC-1: Scope / ambition of the wall-clock solve.
  - Claude Position: full solve (cohort expansion + deploy-ns) with hard-pass + wide refactor latitude.
  - Codex Position: full solve is buildable but the long cohort is the real critical path.
  - Tradeoff Summary: full solve yields a deploy-realistic wall-clock headline at the cost of real
    GPU/kernel-authoring work; cycles-only would be smaller but yields no deploy ns number.
  - Decision Status: **RESOLVED — FULL SOLVE (operator).** Cohort expansion in scope; hard-pass with
    wide latitude to research/refactor/rewrite the timing toolchain.
- DEC-2: The wall-clock accuracy gate.
  - Claude Position: adopt Codex's defined bar.
  - Codex Position: cycles N ≥ 10, log-geomean ≤ 0.15, median ≤ 0.10, bootstrap-upper ≤ 0.25; smoke
    N ≥ 5; deploy-ns only with a crosswalk + cohort ≥ ~552 µs (20 % bound).
  - Tradeoff Summary: the L2 bounded-fraction gate does not transfer to a log-target error; tighter
    bars risk infeasibility on N≈10–20 of narrow dynamic range.
  - Decision Status: **RESOLVED — adopt Codex's gate (operator)**; the 10 % bound is FUT-4.
- DEC-3: Relationship to the active unified RLCR loop.
  - Claude Position: supersede its timing legs.
  - Codex Position: supersede AC-6..AC-10 (not buildable as written); merge back after closure.
  - Tradeoff Summary: superseding avoids two efforts mutating the same timing artifacts; the
    cache/traffic legs and frozen invariants stay untouched.
  - Decision Status: **RESOLVED — SUPERSEDE the unified timing legs (operator)**; pause its
    task10–task13; merge back via FUT-6.
- DEC-4: Long-cohort size beyond the N ≥ 10 floor.
  - Claude Position: N ≥ 10 hard floor; aim N ≥ 15 with ≥ 4 memory-bound and ≥ 4 compute-bound cells.
  - Codex Position: consider N ≥ 15 balanced to reduce overfit on the residual.
  - Tradeoff Summary: more balanced cells reduce overfit but cost more authoring/capture work.
  - Decision Status: **RESOLVED — N ≥ 10 hard (publication floor); N ≥ 15 balanced is the upper-bound
    goal.** Non-blocking operator-tightening option (raised by the second-Codex round): the residual
    over-fit risk is materially lower if N ≥ 15 with ≥ 4 memory-bound + ≥ 4 compute-bound cells is
    made a HARD gate rather than an aim. The plan is executable at the locked N ≥ 10 floor; the
    operator may tighten this to N ≥ 15-balanced-hard on review.
- DEC-5: New long timing cells — scoped artifact vs immediate root admission.
  - Claude Position: scoped additive cohort artifact until closure, then merge into root.
  - Codex Position: keep fresh long cells out of root `label_admitted.json` until schema/guards pass.
  - Tradeoff Summary: a scoped artifact protects the root timing-invariant + frozen-L2/sm_120 freeze
    during iteration at the cost of a merge step.
  - Decision Status: **RESOLVED — scoped additive cohort artifact until closure, then additive merge.**
- DEC-6: Two-regime architecture now vs later.
  - Claude Position: emit regime tags now; defer the full two-regime model.
  - Codex Position: defer D2 but require regime tags in the runner now.
  - Tradeoff Summary: tagging now preserves auditability without premature boundary tuning.
  - Decision Status: **RESOLVED — regime tags now; full two-regime model is FUT-1.**
- DEC-7: `%clock64` phase-cycle ledger.
  - Claude Position: ablation-only; never a silent headline input.
  - Codex Position: ablation-only; it has no live ledger and cannot supply f(t).
  - Tradeoff Summary: keeping it ablation-only avoids a hidden headline dependency.
  - Decision Status: **RESOLVED — ablation-only; live ledger is FUT-2.**
- DEC-8: Recover the 7 `timing_binary_sha256_mismatch` short cells.
  - Claude Position: not on the critical path; opportunistic only.
  - Codex Position: do not spend critical-path time; build fresh long cells against one binary.
  - Tradeoff Summary: recovery adds short-regime cycle coverage but the headline path is the long cohort.
  - Decision Status: **RESOLVED — opportunistic only; FUT-5.**
- DEC-9: Execution branch.
  - Claude Position: execute on current HEAD `083c178a` / `cleanup/pre-merge-simplification`, additive.
  - Codex Position: execute on current HEAD; avoid arch-config-unification churn.
  - Tradeoff Summary: staying additive on the current branch avoids interaction with in-flight
    arch-config work; sm_120 cross-arch is FUT-3.
  - Decision Status: **RESOLVED — current HEAD, additive; cross-arch is FUT-3.**

## Implementation Notes

### Code Style Requirements
- Implementation code and comments MUST NOT contain plan-specific terminology ("AC-", "Milestone",
  "M0/M1/...", "Step", "Phase", "Round N", or similar workflow markers). These belong in the plan
  document only. Use descriptive, domain-appropriate naming. The plan-token scanner is binding.
- Round summaries MUST record `Verified-live-sm_86` or `Offline-verified` — never `Cross-arch-verified`
  — and must not introduce any sm_120 / cross-arch artifact on this machine.
- NCU invocation remains confined to `tools/local_labels/` → `data/local_labels/`; no other path may
  invoke NCU or write `.ncu-rep*`.
- The frozen L2 v5 head, the sm_120 frozen cohort, and the 190-file path-boundary producer stack
  (CUPTI collector + NVBit tracer) are byte-frozen and OUT OF SCOPE for this plan; they are not edited
  at all, and any byte drift is a hard stop. The predictor consumes the producer stack's existing
  output only.

--- Original Design Draft Start ---

# Non-Root Wall-Clock / Kernel-Duration Prediction on sm_86 (RTX 3090 Ti) — Focused Solve

## Original Idea

Build and VALIDATE a non-root, NCU-free predictor of per-kernel **wall-clock duration** on the
RTX 3090 Ti (sm_86, GA102, Ampere; driver 595.71.05), as a dedicated effort separated out of the
broader unified multi-metric program. NCU is used ONLY on this local 3090 Ti to capture
ground-truth timing labels, quarantined to `tools/local_labels/` → `data/local_labels/`. At deploy
there is no NCU, no driver-root, no clock-lock, and no privileged CUPTI; the predictor must work
from NVBit address/SASS traces, static SASS, launch geometry, and read-only NVML / CUPTI-Activity
covariates only.

This draft is grounded in a byte-verified research dossier
(`plans/research/sm86_wallclock_prediction_research_dossier.md`, 9-area research workflow + 131
adversarial verification checks) and a follow-up grounded Codex design discussion
(`.humanize/skill/2026-05-31_00-09-50-*/output.md`). Load-bearing facts below carry file
citations; the dossier §8 Verification Ledger records what was refuted and must not be relied on.

### Why the wall-clock leg "fails dramatically" today — a REGIME COLLISION, not a built-predictor failure

1. **The predictor is greenfield — nothing exists.** There is no roofline/interval cycle prior, no
   ML residual head, no cycles→ns timing runner, and no cycle-MAPE/ns-MAPE scoring arm.
   `workloads/simbench_bw/comparison_runner.py` scores cache hit-rate only (verified: zero
   timing-scoring tokens). The unified plan's AC-8 (roofline + residual → cycles), AC-9 (>200µs
   headline gate), AC-10 (%clock64 ablation) are planned-and-unbuilt
   (`plans/sm86_unified_multimetric_predictor_plan.md:245` "greenfield; none exists today").

2. **Cohort ⟂ gate ⟂ floor are disjoint.** All 12 labeled sm_86 timing cells are 2310.4–7110.4 ns
   (mean 3551.5, median 3.10 µs, dynamic range only 3.08×) — `data/local_labels/label_admitted.json`.
   The planned headline gate admits only kernels **>200 µs** (`...plan.md:56`) → **0 of 12 qualify**.
   The deploy-time dual-source CUPTI-Activity + cudaEvent observable carries a **provenanced sm_86
   p99 noise floor of 110,352 ns** (`workloads/simbench_bw/golden/cupti_dual_source_tolerance.json`,
   n=1000 reps of `rmsnorm_fwd_bf16`, git sha a0cabd4e) — every label is ~15–48× below it. There is
   no populated window where a deploy-realistic wall-clock prediction can be validated.

3. **cycles is clean; ns is the failing leg.** `gpc__cycles_elapsed.max` is DVFS-stable: implied
   clock across the 12 cells is 1.8608–1.8798 GHz, CV 0.35%; per-cell cycle CV 0.00223–0.021596.
   ns is merely cycles × a near-constant clock, quantized to a 6.4 ns grid; ns-MAPE double-counts
   cycle-MAPE and adds only the (mis-calibrated) clock conversion.

4. **ns anchor is mis-regime'd → ~38% systematic bias.** Cycles→ns conversion uses a single
   1350 MHz NVML reference (`data/local_labels/nvml_reference_operating_point_sm_86.json:17`,
   median SM clock under a sustained fp32 4096² matmul at 90 °C / 294.5 W with thermal slowdown
   active), but the labeled kernels ran at ~1867 MHz boost. Converting at 1350 MHz over-predicts ns
   by 37.8–39.2% on every cell before any model error.

5. **No accuracy gate is defined anywhere.** `git grep mape -- data/` is empty; the only scored
   sm_86 gates are L1/L2 hit-rate (bounded-fraction) gates. The wall-clock bar itself does not exist.

6. **Deploy-time feature poverty.** At deploy there is no NCU (cycles only via %clock64 or
   prediction), no clock-lock (Rule 2), and the SM clock is a free DVFS variable. The only non-root
   per-kernel clock probe — the NVML polling thread — samples at 200 Hz / 5 ms
   (`cupti_activity_collector.cu:985`); a 2–7 µs kernel window contains zero in-window NVML samples,
   so per-kernel effective clock is unmeasurable non-root and the covariate regression undersamples
   (requires ≥4 launches + ≥10% residual-variance reduction, `nvml_covariate_acceptance.toml:30`).
   Both timing metrics are `deployability=label_only` (`metric_registry.json:105,123`) — they may
   NEVER be predictor inputs.

## Operator Decisions (LOCKED — do not relitigate)

- **DEC-A — Scope = FULL SOLVE (cohort expansion in scope).** Author/scale long-running kernels
  (>200 µs, target ≥ ~552 µs = 5× the 110 µs floor so a 20% floor-relative ns claim is even
  possible), capture fresh labels against a single current binary build, build the cycles predictor
  AND a deploy-realistic ns headline, and HARD-PASS the gates. Real GPU + kernel-authoring work is
  authorized. The executing agent has wide latitude to research, refactor, or rewrite the timing
  toolchain/framework to reach a gate pass, bounded by the constraints and frozen invariants below.
- **DEC-B — Accuracy gate = Codex's defined bar** (adjustable plan decision, recorded as a DEC):
  - Cycles headline (publication-grade, HARD): N ≥ 10 cells; log-space geomean relative error
    ≤ 0.15; median absolute log-relative error ≤ 0.10 (NOT 0.06 — the L2 0.06 median is a
    bounded-fraction gate and does not transfer to a log-target multiplicative error); 95% bootstrap
    upper bound ≤ 0.25 (2000 resamples, CI 0.95, log-space geomean — reuse the existing bootstrap).
  - Cycles smoke (N ≥ 5): reports the same metrics but CANNOT close `research_success` unless the
    publication-grade gate passes.
  - Deploy-ns gate: no point-MAPE gate UNLESS a same-binary/same-shape CUPTI-Activity-vs-NCU
    `gpu__time_duration.sum` agreement (crosswalk) artifact exists AND the cohort duration is large
    enough for a floor-relative bound (≥ ~552 µs for 20%; ≥ ~1.10 ms for 10%).
  - Coverage floors reuse the existing smoke-5 / publication-10 per-arch floors
    (`comparison_runner.py:280-281`).
- **DEC-C — Supersede the unified loop's timing legs.** Pause the active unified RLCR loop's timing
  tasks (`.humanize/rlcr/2026-05-30_02-22-44`, task10–task13 / AC-6..AC-10). Execute this dedicated
  wall-clock plan additively; keep the unified loop's cache/traffic legs and all frozen invariants
  untouched; merge the revised timing module back after closure. AC-8/AC-9 as written are not
  buildable (AC-8 reports ungated ns-MAPE; AC-9's headline cohort is empty).

## Primary Direction: Cycles-First Roofline+Pedestal+Residual, ns via Crosswalk+Anchor-Fix, Long-Cohort Headline, Honest Refutation Fallback

Ranked directions (from the Codex discussion): **D1 primary**, **D6 second (cohort expansion —
necessary for any real deploy headline)**, **D5 third (anchor fix — mandatory for ns diagnostics)**,
**D4 fallback (emit `verdict_falsification`, never silent)**; D2 is the eventual two-regime
architecture once long kernels exist; D3 (direct covariate-de-confounded dual-source) waits because
it is inert on sub-poll kernels.

### Approach Summary

1. **Cycles is the SOLE positive headline target.** Predict `gpc__cycles_elapsed.max` and demote ns
   to a derived interval / NCU-conversion diagnostic (Q1/Q2). Model the cycle target as
   **`pedestal_cycles + analytical_work_term + regularized_residual`** (Q8): a nonnegative
   fixed launch/drain overhead pedestal + a SynPerf/GPUMech/GCoM-shaped analytical prior
   (`compute_cycles` from static-SASS instruction/FLOP mix; `memory_cycles` from NVBit sector
   traffic × Ampere latency constants, amortized by the existing predicted L1/L2 hit rates and
   active-blocks occupancy) + a SMALL, strongly-regularized ML residual (the prior must dominate at
   N≈12–50).
2. **Populate the Goldilocks window (D6).** Author/scale long-running kernels (>200 µs, target
   ≥ ~552 µs) using the existing `--shape-point` parameterization (e.g. `timing_01`/`timing_02`
   geometries per slug), capture wall-clock + L1 + L2 labels against ONE current binary build
   (so the AC-11-style all-three-labels invariant holds and binary_sha256 matches), and diversify
   away from the GEMM-heavy composition (add memory-bound / divergent kernels) so both rooflines are
   exercised. De-risk per the highest-risk note below.
3. **Fix the ns anchor (D5).** Recalibrate the NVML reference under a label-matched short-burst
   workload (so the anchor reflects ~1867 MHz boost, not 1350 MHz sustained throttle), and record
   concurrent NVML telemetry as session context, NOT per-kernel truth. Do NOT fit a per-kernel clock
   model from N=12; per-kernel effective clock is unmeasurable non-root for sub-poll kernels.
   `cycles_to_ns` must keep failing closed on missing/implausible clock (`nvml_covariate_join.py:225-257`).
4. **Build the ns crosswalk before any deploy-ns claim (Q2).** Emit a same-binary/same-shape
   CUPTI-Activity+cudaEvent vs NCU `gpu__time_duration.sum` agreement artifact. The scored ns target
   (NCU) and the deploy observable (dual-source CUPTI/event) are DIFFERENT observables with no
   crosswalk today; an ns-MAPE against NCU measures the conversion, not deploy accuracy.
5. **Score with a dedicated timing runner.** Reuse the bootstrap (2000/0.95, log-space geomean) and
   coverage-floor machinery, but define a timing-specific metric/target join key — do NOT cargo-cult
   `comparison_runner.py`'s 8-tuple cache-layer join (it pins `cache_layer`,
   `comparison_runner.py:270-279`). Report cycle log-geomean / median / bootstrap-CI and gate per DEC-B.
6. **Honest closure (D4).** If the long cohort cannot be populated or the deploy-ns leg cannot clear
   its floor-relative bound, close cycles on `research_success` if it passes and emit a schema-valid
   `verdict_falsification` artifact for the deploy-ns-below-floor leg, citing the measured 110 µs p99
   floor. `confirmed ≥30% / refuted <10% / indeterminate 15–29% (NOT closure)`
   (`emit_v5_closure_artifacts.py:78-80`). A silent "reported ns-MAPE" is unacceptable.

### Feature set (cycles predictor; all deploy-legal)

- NVBit `summarize_launch` counts (cold/finite-miss/unique-sector counts, RD percentiles,
  per-op-class/per-access-kind counts, atomics) — `tools/hrd_profiler/observation_builder.py:193-273`
  (already emitted).
- Static SASS instruction mix + FLOP count via `cuobjdump`/`nvdisasm` — **new extractor; does not
  exist** (thin spot).
- Launch geometry → theoretical occupancy (`occupancy_limit_warps_theoretical` is the only
  feature-eligible occupancy metric, `metric_registry.json:405`).
- Predicted L1/L2 sector hit rates (from the existing frozen cache heads) to amortize memory cycles.
- GA102 device peaks (FP32/Tensor FLOP/s, DRAM GB/s, peak clock) — **absent from
  `device_properties.json` (cache geometry only); must be sourced from vendor spec or fit** (thin spot).
- FORBIDDEN as inputs: timing labels themselves, achieved occupancy, IPC, any privileged-counter
  metric (all `label_only` / admin-gated).

## Milestones and Acceptance-Criteria Skeleton (to be formalized by gen-plan)

- **M0 — Timing contract reset (HARD).** Plan/contract artifact states cycles-only headline, the ns
  diagnostic/crosswalk rules, the publication/smoke thresholds (DEC-B), and the non-closure rules.
  Positive: the contract artifact encodes all of the above and validates. Negative: any criterion
  allowing a co-equal ns-MAPE headline on sub-10 µs cells fails.
- **M1 — Cohort + anchor (HARD).**
  - AC-cohort: ≥10 long timing cells above the chosen deploy-ns floor are captured against one
    current binary with wall-clock+L1+L2, OR a typed `verdict_falsification` records why the long
    cohort is unavailable. Negative: a zero-cell >200 µs headline cannot pass.
  - AC-anchor: a label-matched short-burst NVML anchor artifact exists; the 1350 MHz sustained-throttle
    anchor is not used for short-regime point ns. Negative: missing/implausible clock fails closed.
- **M2 — Cycles predictor (HARD).** Predictor emits `pedestal_cycles`, analytical prior, residual,
  feature provenance, and a cycle prediction. Negative: using timing labels / achieved occupancy /
  IPC / privileged counters as inputs fails (leakage guard).
- **M3 — Accuracy runner (HARD).** Timing runner reports log-cycle geomean + median + bootstrap CI
  (2000/0.95) under coverage floors 5/10 and PASSES the DEC-B cycles gate. Negative: below-floor
  coverage or missing bootstrap yields a typed-unavailable, not a silent pass.
- **M4 — ns crosswalk + closure (HARD).** Same-binary/shape CUPTI-vs-NCU crosswalk artifact exists
  before any deploy-ns claim; emits `research_success` for passed cycles and, where applicable,
  `verdict_falsification` for deploy-ns. Negative: an NCU-ns conversion diagnostic labeled as deploy
  wall-clock is rejected; an indeterminate (15–29%) falsification is not closure.
- **M5 — Compliance (HARD).** no-root/no-NCU static guard + plan-token scanner + frozen-L2 guard +
  sm_120-freeze guard + 190-file path-boundary guard all pass. Negative: NCU outside the quarantine,
  privileged CUPTI, clock-lock, sm_120 mutation, frozen-L2 byte drift, or producer-stack byte drift
  fails.

## Hard Constraints and Frozen Invariants (MUST NOT BREAK)

- CLAUDE.md Rules 1–5 (`CLAUDE.md:8-45`): NCU only inside `tools/local_labels/`→`data/local_labels/`;
  no GPU-driver-root / clock-lock / persistence / power / MIG mutation; no privileged CUPTI
  (Events/Metrics/Range/PM/PerfWorks) — CUPTI Activity IS allowed; no host-driver work; no simulator
  revival.
- Frozen L2 v5 head byte-identity: `data/local_labels/sm_86_v5_head.json` sha256
  `9597457b…`, headline geomean reproducing within 1e-9
  (`tools/local_labels/test_frozen_l2_v5_regression_guard.py`).
- sm_120 cohort freeze at commit `b1646ef7` — timing writes share `label_admitted.json` with these
  cells and must not re-key/reorder them.
- 190-file path-boundary producer stack (NVBit %clock64 producer + CUPTI collector) is IMMUTABLE —
  the predictor must work from existing producer output
  (`workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py:966-997`).
- Additive / lockstep `schema_version=5` contract (`additionalProperties:false` at every layer);
  new label families ride the optional `unit_tagged_metrics` array; a true bump lands in lockstep
  across producer + sidecar schema + admitted schema + aggregator + parser + import_label.
- Plan-token discipline: implementation code carries no `AC-`/`Milestone`/`M0`/`Round N` markers.
- Evidence status on this host: `Verified-live-sm_86` or `Offline-verified` only; never
  `Cross-arch-verified`. Terminal vocabulary: `research_success` / `verdict_falsification` /
  `deferred_cross_arch_requires_sm120_session`.

## Known Risks

- **Highest risk:** creating a diverse long-duration cohort that is deploy-validatable WITHOUT
  disturbing frozen/shared artifacts. De-risk: first generate 3–5 scaled shapes, run timing-only NCU
  labels inside the quarantine, check duration + CV BEFORE full feature/tracing work, and store the
  timing cohort in a SCOPED ADDITIVE artifact until duplicate-key and frozen-L2 effects are proven safe.
- N≈12–50 over-fits the residual → regularize hard; let the analytical prior + pedestal dominate.
- `gpc__cycles_elapsed.max` is a `.max`-across-GPC elapsed window dominated by launch/drain overhead
  at small sizes → the pedestal term is essential; the roofline prior has little signal below ~200 µs.
- Roofline homogeneity assumption (uniform peak FLOPs/BW) — add explicit memory-vs-compute bottleneck
  selection.
- sha256-mismatch build drift recurs as the cohort grows → deterministic builds + single-binary
  capture per cell.
- The deploy-ns leg may be structurally unachievable below the floor → must terminate in an emitted
  `verdict_falsification`, not a silent failure.

## Scoping Decisions for gen-plan to formalize (beyond the locked DEC-A/B/C)

- D-recover: do NOT spend the critical path recovering the 7 `timing_binary_sha256_mismatch` cells;
  build fresh long-shape cells against one current binary (Codex Q6). (Recover only opportunistically
  for extra cycle coverage.)
- D-clock64: keep the %clock64 phase-cycle ledger ablation-only (never a silent headline input); it
  has never produced a live JSON and cannot supply f(t) (Codex Q7). May be taken live as the ablation.
- D-branch: execute on current HEAD `083c178a` / branch `cleanup/pre-merge-simplification`; keep all
  changes additive and avoid arch-config-unification churn (Codex Q9).

## Relevant References

- `plans/research/sm86_wallclock_prediction_research_dossier.md` — the verified research dossier (§1–§8).
- `.humanize/skill/2026-05-31_00-09-50-*/output.md` — the grounded Codex design discussion.
- `plans/sm86_unified_multimetric_predictor_plan.md` — the broader plan whose timing legs (AC-6..AC-10)
  this supersedes.
- `tools/local_labels/capture_timing_labels.py`, `metric_registry.json`,
  `data/local_labels/label_admitted.json`, `timing_label_exclusions_sm_86.json`,
  `nvml_reference_operating_point_sm_86.json`.
- `workloads/simbench_bw/golden/cupti_dual_source_tolerance.json`, `cupti_activity_collector.cu/.h`,
  `nvml_covariate_join.py`, `nvml_covariate_acceptance.toml`, `comparison_runner.py`.
- `tools/hrd_profiler/observation_builder.py`, `device_properties.json`.
- `tools/local_labels/emit_v5_closure_artifacts.py` / `test_sm86_v5_closure_artifacts.py` — the
  closure/falsification vocabulary.

--- Original Design Draft End ---
