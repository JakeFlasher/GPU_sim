# SOTA-Grounded Non-Root Performance Model: Full Multi-Milestone NCU-Replacement Loop (sm_86)

## Goal Description

Convert the SOTA-research dossier
(`plans/research/sota_nonroot_perf_model_next_phase_dossier.md`) into the
project's accurate-and-fast non-root NCU replacement on sm_86: deterministic
T1 replay counts with NCU lockstep validation, a scheduler-aware /
MSHR-aware / sectored L1 model rebuilt per the published root-cause fixes,
T2 traffic streams, a GCoM-lite analytical cycles model with SASS
control-bit stall readout and a DVFS-decomposed ns interval, and a
per-(cell, metric_key) evidence harness that issues the honest per-metric
replacement verdict — all under CLAUDE.md Rules 1-5, with the producer
hazards hardened and the wall-clock same-slug leakage demoted first.

**Scope decision (DEC-1, user):** full multi-milestone scope. **Scope-risk
note (recorded honestly):** the previous loop completed one of seven
milestones; this plan contains six milestones. Milestones are ordered
information-per-round and each AC is independently closable, so a
max-iterations end leaves a coherent prefix (hygiene + honesty + T1) rather
than a broken middle. Round contracts must not start a later milestone
while an earlier milestone's ACs are open, unless explicitly queued as
parallel non-conflicting lanes (M2 is the designated parallel-safe lane
after M0).

Standing user-confirmed constraints: timing demotion in-scope (DEC-2);
consumer-side only — no edits to the byte-pinned NVBit tracer / CUPTI
collector tree (DEC-3); sm_120 live work typed-deferred (DEC-4); trace
sampling deferred (DEC-5); no new Python dependencies (DEC-6); numeric
thresholds directional + structural gates hard (DEC-7).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.
The `AC-*` items are current RLCR completion gates for this implementation loop.

> ALL ACs are mandatory current-loop completion gates. Numeric accuracy
> thresholds are DIRECTIONAL (DEC-7): reported achieved-vs-target with CIs;
> a threshold MISS closes as an honest typed verdict artifact, never silence
> and never a fake pass. Structural gates (typed accounting, closed-identity
> joins, schema validity, quarantine, non-vacuity floors, sealed-holdout
> governance) are HARD.

- AC-1: Producer/artifact hygiene gate closes the three queued hazards
  before any bulk regeneration.
  - AC-1.1: `run_v5_active_path_comparison` cannot clobber the committed
    `data/local_labels/sm_86_v5_head.json`: its default output resolves to a
    non-committed path (or an explicit opt-in flag is required for the
    committed path), pinned by a guard test.
    - Positive Tests (expected to PASS):
      - Running the comparison with defaults leaves the committed head
        byte-identical (frozen-head guard passes).
      - The guard test asserting the default output path is non-committed
        passes.
    - Negative Tests (expected to FAIL):
      - A regression pointing the default output back at the committed head
        path fails the guard test.
  - AC-1.2: The `regenerate_v5_artifacts` wrapper is split-aware: its fitter
    step runs with the sealed cohort-split arguments (cal+val fit, holdout
    excluded) or fails closed with a typed error.
    - Positive Tests:
      - A unit test asserts holdout cells are excluded from the wrapper's
        fit cohort.
    - Negative Tests:
      - A wrapper invocation that would fit on holdout cells raises the
        typed error.
  - AC-1.3: No test mutates committed artifacts: the timing-crosswalk
    restamp writes to a temp path; a deterministic path-scoped artifact
    guard (pre/post sha256 over the committed `data/local_labels/` tree
    plus untracked-file detection under `data/local_labels/`) brackets the
    test sweep — not a bare `git status` check.
    - Positive Tests:
      - Pre/post hashes identical across the full sweep; zero untracked
        artifacts introduced.
    - Negative Tests:
      - A test writing into `data/local_labels/` is flagged by the guard;
        the pre-fix crosswalk restamp reproduces the failure.

- AC-2: The wall-clock cycle headline is made slug-disjoint-honest (DEC-2).
  - The same-slug scaled-neighbor leave-one-out arm in
    `workloads/simbench_bw/timing_cycle_predictor.py` is demoted to a
    diagnostic channel (mirroring the correction-factor demotion pattern):
    the deployable validation score comes from the ridge-on-roofline
    predictor with NO same-slug label substitution; the same-slug values
    persist only in a clearly-labeled diagnostic field; the model version
    string names the demotion.
  - The full artifact/test surface is regenerated or updated in lockstep:
    `timing_cycle_predictions_sm_86.json`,
    `timing_accuracy_report_sm_86.json`, `timing_closure_sm_86.json`,
    `deploy_ns_crosswalk_sm_86.json` (if closure regeneration touches it),
    `timing_prediction_contract_sm_86.json`, the matching schemas,
    `workloads/simbench_bw/timing_accuracy_runner.py`,
    `workloads/simbench_bw/timing_closure.py`, and their tests
    (`workloads/simbench_bw/tests/test_timing_closure.py` et al.). The
    4-slug cohort limitation is recorded as a typed limitation field.
  - Positive Tests:
    - Regenerated artifacts are schema-valid; the diagnostic channel
      preserves the prior same-slug scores; an honest slug-disjoint MISS
      (or a typed falsification/no-claim verdict) validates as an
      acceptable closing state.
  - Negative Tests:
    - A deployable headline computed with same-slug substitution is
      rejected by a new guard test.
    - A closure artifact deriving `publication_pass` / `research_success`
      from the demoted diagnostic arm fails validation.

- AC-3: T1 count labels exist (or are typed-unavailable with probe
  evidence) across the admitted sm_86 corpus, under quarantine.
  - AC-3.1: The 8 probe-confirmed counters (`smsp__inst_executed.sum`,
    global op ld/st/atom/red request counters, global ld/st sector
    counters, `l1tex__t_requests.sum`) are captured via the additive
    `unit_tagged_metrics` splice by a driver under `tools/local_labels/`
    only, n_repeats >= 5, relative_tolerance rule.
    - Positive Tests:
      - Admitted cells carry count blocks; schema-valid; registry binding
        resolves.
    - Negative Tests:
      - A count captured outside `data/local_labels/` or by a driver
        outside `tools/local_labels/` is rejected (Rule 1 guard).
  - AC-3.2: Counts are integer-stable across repeats; a counter with
    disagreeing repeats is typed `label_count_unstable`, never silently
    averaged.
    - Positive Tests:
      - A stability statistic is recorded per counter; an unstable fixture
        cell produces the typed exclusion.
    - Negative Tests:
      - An averaged unstable counter passing as a label is rejected.
  - AC-3.3: The 7 probe_pending identifiers (local ld/st, texture, 4x
    surface) are resolved by `probe_ncu_metric_availability.py` on the
    3090 Ti: each flips to available (and is captured) or becomes a typed
    fail-closed exclusion, in registry + inventory lockstep.
    - Positive Tests:
      - Committed probe artifact present; the lockstep test passes.
    - Negative Tests:
      - A registry flip without a committed probe artifact is rejected.
  - AC-3.4: Sealed-holdout governance: capturing count labels on holdout
    cells is permitted (labels are ground truth, not model fit), but
    producer/model tuning and any threshold/configuration selection use
    ONLY calibration/validation cells or synthetic fixtures; holdout cells
    are scored once, at final evaluation, with the evidence trail recorded
    in the validator artifact.
    - Positive Tests:
      - Validator artifacts record the holdout-touch count (== 1) and the
        cal/val-only tuning trail.
    - Negative Tests:
      - A producer configuration change after a holdout score is rejected
        (retune-after-holdout guard).

- AC-4: A deterministic NVBit replay-count producer reconstructs T1 counts
  per admitted cell from unsampled traces, with semantics pinned by live
  microfixtures BEFORE corpus scoring.
  - AC-4.1: Live observation-contract microfixtures on the 3090 Ti pin, per
    metric: `smsp__inst_executed.sum` predication semantics
    (issued-warp-instruction vs active-lane counting), inactive-lane
    handling, ATOM-vs-RED attribution, vector-width (32/64/128b) request
    formation, alignment/32B sectoring, local-memory routing, and the
    rollup scope of `l1tex__t_requests.sum` (which pipes/op classes it
    includes). Each discovered semantic lands in a committed
    contract-matching artifact consumed by the validator.
    - Positive Tests:
      - Each microfixture's NCU label matches its designed count under the
        discovered semantic; the contract artifact is schema-valid.
    - Negative Tests:
      - Scoring a metric with no contract-matching entry is rejected; a
        fixture mismatch leaves the metric typed `contract_mismatch`
        (fail-closed), not scored.
  - AC-4.2: Producer: builds on existing reductions
    (`tools/sector_stream/`, the observation-builder import contract);
    active-mask x predication x per-op-class counting; coalescing to
    requests; 32B sectoring; closed identity (binary_sha256 +
    kernel_signature + shape) on every emission. Exact-support accounting:
    sampled/truncated/budget-limited traces are typed
    `trace_support_insufficient`; unsupported op classes route through the
    existing kUnspecified -> unsupported_memory_path fallback and are
    counted in a typed coverage denominator. Broad rollups
    (`l1tex__t_requests.sum`) are scored ONLY when every included op class
    (per the AC-4.1 contract) is either modeled or absent from the cell's
    trace; otherwise typed `rollup_unsupported` (fail-closed diagnostic).
    - Positive Tests:
      - Synthetic-fixture unit tests pin counting semantics (masks,
        predication, vector width, ATOM-vs-RED, alignment); the producer
        emits schema-valid per-cell counts with full typed accounting.
    - Negative Tests:
      - A sampled trace scored as exact is rejected; an ATOM counted as RED
        in fixtures is caught; a broad rollup scored with an unmodeled
        included op class is rejected; a join on anything weaker than the
        closed identity tuple is rejected.

- AC-5: An NCU lockstep validator delivers the per-metric T1
  count-replacement verdict on sm_86 with live evidence.
  - Join: producer counts to AC-3 labels by the closed identity tuple only.
    Per-(cell, metric) verdict artifact (schema added early): one of
    {replacement_pass, replacement_miss, low_support, contract_mismatch,
    rollup_unsupported, trace_support_insufficient, label_count_unstable,
    label_unavailable}; p50/p90/max relative error per metric and
    per-op-class with denominator floors. Directional targets: p50 <= 1%,
    p90 <= 3%, max <= 5% — reported achieved-vs-target; a MISS is an
    honest typed outcome, not a failure to close this AC.
  - HARD non-vacuity floor: >= 10 scored (cell, metric) pairs spanning
    >= 5 distinct slugs, including at least one cell from each of the
    ld/st, atom-vs-red, and vector-width factorial coverage families, and
    >= 4 of the 8 confirmed metrics scored on >= 5 cells each. Scored means
    a real label joined to a real producer count — typed exclusions do not
    count toward the floor.
  - Coverage report: measured per-op-class coverage across active-mask,
    predication, vector-width, ATOM-vs-RED, and local-memory cases.
    Evidence: Verified-live-sm_86 for the runtime claim (real NVBit traces,
    real quarantined NCU captures; device/driver/toolkit/NVBit metadata in
    the round summary). Typed fail-closed exceptions are not live evidence.
  - Positive Tests:
    - Validator artifact schema-valid; floor met; coverage report present;
      live sweep ran on the 3090 Ti with the metadata recorded.
  - Negative Tests:
    - A validator run below the floor cannot claim any replacement_pass
      (vacuity guard); a join on a weaker identity is rejected; a threshold
      MISS silently upgraded to pass is rejected.

- AC-6: Per-arch measurement substrate (sm_86) for cache/timing modeling.
  - AC-6.1: A P-chase microbenchmark suite (new tool subtree outside the
    pinned boundary; %clock64/clock64() timing; no clock locking) measures
    on the 3090 Ti: L1/L2/global/shared latencies, L1 capacity/sets/ways
    evidence, set-mapping evidence, replacement-policy evidence, MSHR-count
    evidence, and a carveout-capacity sweep; it emits a committed per-arch
    static card with repeat counts, dispersion, NVML covariates
    (clock/thermal/throttle), and typed `geometry_unresolved` fallbacks for
    any property the probe cannot pin under DVFS noise.
    - Positive Tests:
      - Card schema-valid; every property carries either a value +
        dispersion + repeats or a typed `geometry_unresolved`; NVML
        covariates recorded per run.
    - Negative Tests:
      - A card value without dispersion/repeats is rejected; an NCU-derived
        number in the card is rejected (the card must be counter-free by
        construction).
  - AC-6.2: A static SASS control-bit extractor (new consumer-side tool)
    parses cuobjdump/nvdisasm output for stall counts, dependence-counter
    masks/increments, yield bits, and reuse bits per instruction, with a
    parser contract (supported toolkit versions pinned, per-arch fixtures,
    fail-closed `unknown_encoding` typing, PC/function join to dynamic
    traces).
    - Positive Tests:
      - Fixture SASS parses to the expected control fields; unknown
        encodings type fail-closed; the join to NVBit vpc is verified on a
        live trace.
    - Negative Tests:
      - A parse that silently drops unknown control fields is rejected; an
        extractor consuming NCU output is rejected.

- AC-7: L1 walker rebuild per the dossier's F1-F3 root causes; honest
  slug-disjoint re-score.
  - AC-7.1: Scheduler-aware interleaving: the per-SM walk replaces strict
    block-by-block concatenation with a warp-pool scheduler (one memory
    request per warp per turn, latency-proportional re-insertion), with
    co-residency derived from launch geometry + occupancy.
    - Positive Tests:
      - Synthetic fixtures with known interleaving-sensitive hit rates pass
        (the ordered-round-robin baseline reproduces the documented error;
        the warp-pool walk corrects it).
    - Negative Tests:
      - A walk that concatenates blocks end-to-end (the old behavior)
        fails the interleaving fixtures.
  - AC-7.2: Finite-MSHR model: same-line merge + stall-on-exhaustion, with
    the MSHR count sourced from the AC-6.1 card (or typed
    `geometry_unresolved` -> a documented default with sensitivity
    reported).
    - Positive Tests:
      - MSHR fixtures (merge, exhaustion-stall) pass; the sensitivity note
        exists when the card types unresolved.
    - Negative Tests:
      - An unlimited-MSHR walk fails the exhaustion fixture.
  - AC-7.3: HRD(32,128) two-layer sector/line accounting replaces the
    single-stack approximation (building on the existing
    cache_model_hrd_second_layer machinery); carveout-aware capacity from
    the AC-6.1 sweep; set-mapping/replacement per card evidence (LRU only
    as a documented approximation where the card types unresolved).
    - Positive Tests:
      - Sector-vs-line fixtures (partial-line eviction) pass; carveout
        fixture cells track the swept capacity.
    - Negative Tests:
      - A flat single-granularity walk fails the sectored fixtures; an
        unsourced geometry constant (no card entry, no typed default) is
        rejected.
  - AC-7.4: Honest re-score: L1 predictions regenerate over the admitted
    corpus under AC-3.4 sealed-holdout governance (tuning on cal/val or
    fixtures only; holdout scored once); slug-disjoint accuracy +
    recomputed robustness gates (LOOCV/LOSO) for the NEW head are reported
    achieved-vs-target; correction factors REMAIN demoted; per-slug
    heuristics (grid-density penalty) are replaced by structural features
    (sharing classification, divergence/DPKI) or explicitly retained with a
    typed approximation status.
    - Positive Tests:
      - Regenerated L1 artifacts schema-valid; the achieved-vs-target table
        is present (a MISS with honest reporting is an acceptable close);
        robustness is recomputed for the new head.
    - Negative Tests:
      - A re-introduced truth-fitted per-slug factor in the deployable path
        is rejected; copying forward old-head robustness numbers is
        rejected.

- AC-8: T2 traffic streams labeled and modeled (or typed).
  - AC-8.1: The 5 available cache_stream counters —
    `dram__sectors_read.sum`, `dram__sectors_write.sum`,
    `lts__t_requests_srcunit_tex.sum`,
    `l1tex__m_xbar2l1tex_read_sectors.sum`,
    `l1tex__m_l1tex2xbar_write_sectors.sum` — are captured as labels over
    the admitted corpus (AC-3 mechanics: quarantine, repeats, additive
    schema, integer-stability); the 2 excluded DRAM op-split aliases stay
    typed fail-closed rows.
    - Positive Tests:
      - Labels captured + schema-valid with stability statistics.
    - Negative Tests:
      - An alias counter resurrected without probe evidence is rejected.
  - AC-8.2: A miss-stream traffic predictor composed from the rebuilt
    walker (AC-7) + sector streams + write-policy filters; directional
    targets p50 <= 10%, p90 <= 25% relative count error with denominator
    floors; a traffic primitive missing its gate is reported `diagnostic`
    and bars promotion of any hit-rate derived from it; hit-rate reporting
    gains the absolute-pp bucket framing (p50 <= 5pp, p90 <= 12pp,
    low/mid/high buckets).
    - Positive Tests:
      - Predictor artifact reports achieved-vs-target with floors; the
        pp-bucket report is present.
    - Negative Tests:
      - A traffic count whose label was never captured cannot be scored
        (validator fails closed, no vacuous pass).

- AC-9: GCoM-lite cycles + DVFS-decomposed ns interval; timing cohort made
  slug-disjoint-scoreable.
  - AC-9.1: Timing cohort slug expansion: timing labels (cycles + ns,
    existing capture mechanics) captured over >= 10 distinct slugs
    (reusing admitted corpus slugs whose binaries pin; rebuilding stale
    timing-exclusion binaries as needed), so slug-disjoint scoring is
    structurally possible. Rebuilt binaries require fresh
    capture/admission sidecars; committed binary_sha256 pins are NEVER
    hand-edited to make stale labels bind. Cells that cannot bind (sha256
    mismatch) stay typed.
    - Positive Tests:
      - >= 10 distinct slugs carry admitted timing labels at matching
        binary hashes; rebuilds carry fresh sidecars.
    - Negative Tests:
      - A hand-edited binary_sha256 pin is rejected by the recurrence
        guard; a stale-binary cell scored as bound is rejected.
  - AC-9.2: Analytical cycles model: an interval/CPI-stack evaluator
    consuming (i) the AC-6.2 control-bit stalls, (ii) the AC-6.1 latency
    card, (iii) T1 per-op counts (AC-4/AC-5) and L1/L2 hit predictions
    (AC-7 + the frozen v5 head) for memory terms with
    MSHR-batching/queueing corrections, (iv) launch geometry/occupancy. A
    ridge residual is permitted on top (cal/val only). NO same-slug
    substitution in any deployable channel (the AC-2 guard extends to this
    model). The evaluator stays analytical/trace-replay — no per-cycle
    simulation machinery (Rule 5).
    - Positive Tests:
      - Model artifacts schema-valid; component terms (compute, memory,
        stall, imbalance) are individually inspectable; deploy-legal
        feature assertions pass.
    - Negative Tests:
      - A deployable cycles number sourced from a same-slug channel is
        rejected; a label-only field in the inference feature vector is
        rejected.
  - AC-9.3: Slug-disjoint scoring: directional target geomean log-relative
    error <= 0.20 (stretch 0.15) on cycles, with LOSO/LOOCV recomputed for
    this head; ns published as an interval via the
    max(alpha/f_mem, beta/f_core) + t0 decomposition with NVML-observed
    clocks; a point-ns claim only for cells above the dual-source noise
    floor with in-window NVML samples, else interval-only (typed).
    - Positive Tests:
      - Achieved-vs-target + robustness tables present (honest MISS
        acceptable); ns intervals carry the decomposition parameters and
        clock evidence; the deploy crosswalk is honest.
    - Negative Tests:
      - A point-ns claim below the noise floor or without in-window NVML
        samples is rejected; an interval computed from a clock outside the
        plausibility band is rejected.

- AC-10: Per-(cell, metric_key) evidence harness; independent review; final
  compliance.
  - AC-10.1: The harness generalizes scoring to per-tier, per-metric
    (cell, metric_key) verdicts across the registry rows exercised by this
    plan (T1 counts, L1/L2 hit rates, T2 traffic, cycles/ns): every
    exercised row lands in exactly one bucket {main, appendix, diagnostic,
    typed_unavailable} with achieved-vs-target numbers and CIs; the
    baseline panel runs ONCE on the sealed holdout (always-hit/miss,
    family-mean, uncalibrated walker, uncalibrated reuse-distance, prior
    v4/v5, static roofline timing, event-only timing); ablations cover
    no-interleaving, no-MSHR, no-HRD-second-layer, no-control-bit-stalls,
    no-covariates.
    - Positive Tests:
      - Harness artifacts schema-valid; every exercised row bucketed
        exactly once; one-touch holdout trail recorded.
    - Negative Tests:
      - A metric promoted to main with a failed robustness or typed-budget
        gate is rejected; a harness scoring a metric without slug-disjoint
        validation evidence is rejected.
  - AC-10.2: `claim_to_evidence_map.md` + the readiness report regenerate
    from harness outputs; typed-unavailable budget accounting is enforced
    (T1 = 0 silent gaps, T2 <= 10%, T3 explicit diagnostic budget).
    - Positive Tests:
      - Regenerated map/report match harness outputs; budget accounting
        artifact validates.
    - Negative Tests:
      - A silent (untyped) gap in any exercised T1 row is rejected.
  - AC-10.3: Independent Codex review of statistical claims, baseline
    fairness, gate calibration, and holdout leakage passes; AC-wide
    compliance re-verification (static guards, plan-token scanner,
    quarantine, path-boundary byte-identity, label-commit-last +
    POST-commit drift/recapture battery per commit, and re-verified at
    every closing HEAD); a per-AC evidence-status table is emitted; zero
    net-new failures vs the documented queued residual; optional
    consumer-side experiments provably do not affect scored outputs or
    reserved wire tags.
    - Positive Tests:
      - Review verdicts recorded; full battery green; evidence table lists
        every AC with its status; residual reconciliation matches the
        queued ledger.
    - Negative Tests:
      - An implementation file containing plan tokens fails the scanner; a
        new NCU reference outside quarantine fails the guard; a scored
        output influenced by an optional experiment is rejected.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
AC-1..AC-10 fully closed with Verified-live-sm_86 evidence on every
runtime-claiming criterion (AC-4.1, AC-5, AC-6.1, AC-7.4 live re-trace where
needed, AC-8.1, AC-9.1), plus optional diagnostic-only consumer-side
static-SASS cache-op classification experiments (touching no scored output
and no reserved wire tag) and opportunistic probe resolution of the
mem_clock/dram_bytes registry identifiers if NCU exposes them trivially
during the AC-3.3 probe run.

### Lower Bound (Minimum Acceptable Scope)
ALL of AC-1..AC-10 closed as written — completion is not weakened. The
scoped minimums live inside the ACs themselves: AC-5's hard non-vacuity
floor (>= 10 scored pairs / >= 5 slugs / coverage families) is the minimum
scored surface; AC-3.3 and AC-8.1 may legitimately close with typed
fail-closed exclusions (that IS closure, with probe evidence); AC-2,
AC-7.4, and AC-9.3 may legitimately close on honest slug-disjoint MISSes
with diagnostic shadows and typed limitations recorded; AC-6.1 may close
with typed `geometry_unresolved` entries for properties DVFS noise cannot
pin. Milestones close strictly in order M0 -> M5 (M2 parallel-safe after
M0); if the loop ends at max iterations, the closed prefix stands and open
milestones carry to a successor plan via the goal tracker.

### Allowed Choices
- Capture: extending `capture_timing_labels.py` vs sibling capture scripts
  (all under `tools/local_labels/`); one capture sweep vs incremental
  per-metric sweeps.
- Producer substrate: `tools/sector_stream/` vs the observation-builder
  import contract; the walker rebuild may live in `tools/cache_predictor/`
  or a sibling module consumed by `cache_predictor.py`.
- New producers (P-chase suite, control-bit extractor): new tool subtrees
  (e.g., `tools/microbench_cards/`, `tools/sass_control_bits/`) outside the
  pinned boundary.
- Cannot use: edits to the byte-pinned `tools/nvbit_tracer/` + CUPTI
  collector tree (DEC-3); new ML model families or Python dependencies
  (DEC-6); sampling policies as scored inputs (DEC-5; FUT-4); sm_120 live
  work (DEC-4; FUT-5); simulator integration of any kind (Rule 5); the
  sealed holdout for tuning; NCU-derived numbers in any deployable input
  path (labels/validation only).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**Trace once -> summarize once -> predict many metrics.** One NVBit trace
per representative cell feeds (i) the deterministic T1 count reconstructor
(active-mask x predication x per-op-class counting -> PRT-style coalescing
to requests -> 32B sectoring; published fidelity for this signal path:
requests within ~0.001%, sectors within ~0.28% of hardware), (ii) the
rebuilt L1 walker (warp-pool FIFO interleaving with latency-proportional
re-insertion, finite MSHR merge/stall, HRD(32,128) two-layer
sector/line accounting, per-arch geometry from the P-chase card), and
(iii) the GCoM-lite cycles evaluator (interval/CPI stack whose
fixed-latency dependence stalls are read nearly directly from SASS control
bits via cuobjdump — stall counter, six dependence counters, yield — and
whose memory terms compose T1 counts with L1/L2 hit predictions and
MSHR-batching/queueing corrections). The ns leg derives from cycles via
the t0 + max(alpha/f_mem, beta/f_core) DVFS decomposition with read-only
NVML clock covariates, published as intervals below the dual-source noise
floor. Literature anchors: Nugteren HPCA'14 (6.4-8.3% L1 MAE), Maeda
HPCA'17 HRD, MDM HPCA'20, GCoM ISCA'22, Huerta MICRO'25 (13.45% Ampere
cycles MAPE), Buelow CGF'22, DSO 2024 — full survey in the dossier
(sections 3-5 of the Original Design Draft below).

### Relevant References
- `plans/research/sota_nonroot_perf_model_next_phase_dossier.md` — the
  research dossier (failure-mode->fix map, SOTA survey, architecture).
- `workloads/simbench_bw/cache_predictor.py` — L2/L1 prediction entry
  points, demotion registry, v5 active path.
- `tools/cache_predictor/l1_walker.py` — the L1 walk to be rebuilt
  (block-sequential, fully-associative LRU baseline).
- `tools/hrd_profiler/observation_builder.py` — 4-layer reuse-distance
  walker + import contract; HRD second-layer machinery.
- `tools/sector_stream/{generator,consume_trace}.py` — per-warp
  request/sector reconstruction substrate for the T1 producer.
- `tools/local_labels/capture_timing_labels.py` — the additive
  `unit_tagged_metrics` splice mechanism to extend for count metrics.
- `tools/local_labels/probe_ncu_metric_availability.py` +
  `tools/local_labels/metric_registry.json` — probe + registry lockstep
  (8 count metrics `available`, 7 `probe_pending`).
- `workloads/simbench_bw/timing_cycle_predictor.py`,
  `timing_accuracy_runner.py`, `timing_closure.py` — the wall-clock surface
  for the AC-2 demotion and the AC-9 model.
- `tools/local_labels/run_v5_active_path_comparison.py`,
  `tools/local_labels/regenerate_v5_artifacts.py`,
  `workloads/simbench_bw/tests/test_timing_closure.py` — the three AC-1
  hazard sites.
- `workloads/simbench_bw/static_sass_features.py` — existing static SASS
  extractor (mnemonic histogram; sanity-check FLOP weights).
- `tools/cloud/trace_budgets.py` + `workloads/simbench_bw/trace_budgets.toml`
  — slowdown/byte budgets the producer must respect.
- `.claude/knowledge/measurement/` cards — nvbit-injection-playbook,
  cupti-activity-api, clock64-globaltimer-dvfs, nonroot-cloud-contract.
- `references/papers/knowledge_base/active/` — the local PDF corpus cited
  throughout the dossier.

## Dependencies and Sequence

### Milestones
1. M0 (AC-1, AC-2): hygiene + wall-clock honesty. Everything afterward
   regenerates artifacts, so the hazard guards and the timing demotion land
   first.
   - Phase A: the three hazard guards (AC-1.1/1.2/1.3).
   - Phase B: the same-slug demotion + lockstep artifact regeneration
     (AC-2).
2. M1 (AC-3, AC-4, AC-5): the T1 count-replacement headline.
   - Phase A: probe + capture widening (AC-3).
   - Phase B: observation-contract microfixtures, then the producer
     (AC-4.1 -> AC-4.2).
   - Phase C: lockstep validator + live sweep + Codex evidence review
     (AC-5).
3. M2 (AC-6): per-arch substrate (P-chase card + control-bit extractor) —
   the designated parallel-safe lane after M0.
4. M3 (AC-7): L1 walker rebuild — consumes the AC-6.1 card; benefits from
   T1 per-op counts as coverage denominators.
5. M4 (AC-8): T2 traffic — consumes AC-7's walker and AC-3's capture
   mechanics.
6. M5 (AC-9, AC-10): cycles + ns interval (consumes AC-4/5 counts, AC-6
   substrate, AC-7 hit predictions), then the per-metric harness,
   independent review, and final compliance.

Dependencies are structural, not temporal: a later milestone may not start
while an earlier one's ACs are open, except M2 which may run in parallel
after M0 closes.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

Every `AC-*` must be covered by at least one task. Every task must target at least one `AC-*`. Do not target `FUT-*`, `DEC-*`, or `-` in the Target AC column.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Comparison-output guard + frozen-head protection | AC-1.1 | coding | - |
| task2 | Split-aware regeneration wrapper + typed error + unit test | AC-1.2 | coding | - |
| task3 | Path-scoped artifact guard (pre/post hashes + untracked detection) + crosswalk temp-path fix + artifact restore | AC-1.3 | coding | - |
| task4 | Timing same-slug demotion: predictor change, full artifact/test lockstep regeneration, guard tests, typed limitation field | AC-2 | coding | task1, task2, task3 |
| task5 | Probe the 7 pending identifiers; registry/inventory lockstep flips or typed exclusions | AC-3.3 | coding | - |
| task6 | Capture widening to the 8 confirmed counters over the admitted corpus; integer-stability typing; holdout governance trail | AC-3.1, AC-3.2, AC-3.4 | coding | task5 |
| task7 | Live observation-contract microfixtures + committed contract-matching artifact + schema | AC-4.1 | coding | task3 |
| task8 | Deterministic replay-count producer + exact-support and rollup fail-closed accounting + synthetic fixtures | AC-4.2 | coding | task7 |
| task9 | Lockstep validator + verdict schema + non-vacuity floor + coverage report; live sm_86 sweep with evidence metadata | AC-5 | coding | task6, task8 |
| task10 | Codex review of T1 evidence: counting units, replay semantics, ATOM-vs-RED, predication, rollup matching, holdout governance | AC-5 | analyze | task9 |
| task11 | P-chase microbenchmark suite + committed per-arch static card (latencies, MSHR, set mapping, replacement, carveout) | AC-6.1 | coding | task3 |
| task12 | Static SASS control-bit extractor + parser contract + live-join witness | AC-6.2 | coding | - |
| task13 | Walker rebuild: warp-pool interleaving + finite-MSHR model | AC-7.1, AC-7.2 | coding | task11 |
| task14 | HRD(32,128) two-layer + carveout-aware capacity + mapping/replacement wiring | AC-7.3 | coding | task13 |
| task15 | L1 honest re-score + structural-feature replacement of per-slug heuristics + recomputed robustness | AC-7.4 | coding | task14 |
| task16 | T2 label capture (5 counters) + typed alias rows | AC-8.1 | coding | task6 |
| task17 | Miss-stream traffic predictor + pp-bucket hit-rate reporting | AC-8.2 | coding | task15, task16 |
| task18 | Timing cohort slug expansion + binary rebuild with fresh sidecars + typed exclusions | AC-9.1 | coding | task4 |
| task19 | GCoM-lite cycles evaluator (control bits + latency card + T1 counts + cache terms) + ridge residual | AC-9.2 | coding | task9, task11, task12, task15 |
| task20 | Slug-disjoint cycle scoring + ns interval decomposition + honest deploy crosswalk | AC-9.3 | coding | task18, task19 |
| task21 | Per-(cell, metric_key) harness + baseline panel + ablations (one-touch holdout) | AC-10.1 | coding | task9, task15, task17, task20 |
| task22 | claim-to-evidence map + readiness report regeneration + typed-budget accounting | AC-10.2 | coding | task21 |
| task23 | Codex statistical/baseline-fairness/gate-calibration/holdout-leakage review | AC-10.3 | analyze | task21, task22 |
| task24 | Final compliance verification + per-AC evidence-status table + queued-residual reconciliation | AC-10.3 | coding | task23 |

## Future Work / Out of Scope

Future, deferred, post-work, successor-loop, and out-of-scope items belong here, not under `## Acceptance Criteria`.

- FUT-4: Sieve instruction-count stratification + Photon warp-type
  early-stop wired into the trace-budget machinery as pre-declared
  sampling policies (dossier section 4.5).
  - Source DEC: DEC-5
  - Current-loop handoff: AC-5 (exact-lockstep verdicts make
    sampled-vs-exact deltas measurable)
  - Promotion trigger: AC-5 closed with at least one replacement_pass
    metric, so a sampled-trace variant can be scored against the exact
    baseline.
- FUT-5: sm_120 legs — P-chase static card, count/timing/traffic label
  capture, cross-arch few-shot transfer recipes (hardware embeddings /
  domain-invariant features), and cross-arch verification of every
  behavioral claim this plan lands as Verified-live-sm_86 (dossier
  section 4.4).
  - Source DEC: DEC-4
  - Current-loop handoff: AC-6.1 (the per-arch card schema and typed
    geometry_unresolved vocabulary are arch-parameterized by construction)
  - Promotion trigger: physical alternating-session move to the RTX 5060
    laptop.
- FUT-6: Producer-side SASS cache-op classifier filling the reserved NVBit
  wire tags 12-13 (lifts predictor_l1_sass_provenance_missing at the
  source; requires byte-pinned-tree re-pin governance) (dossier
  section 5.2).
  - Source DEC: DEC-3
  - Current-loop handoff: AC-6.2 (the consumer-side control-bit extractor
    establishes the SASS-parsing contract a producer-side classifier would
    reuse)
  - Promotion trigger: a dedicated re-pin round is approved for the pinned
    tree.
- FUT-7: Deployable HARD-2 L1 closure via slug-disjoint correction-factor
  re-derivation IF the AC-7 rebuild leaves family-specific residuals
  (requires >= 3 slug-disjoint members per affected family; corpus
  growth).
  - Source DEC: DEC-7; Source AC: AC-7.4 conditional outcome
  - Current-loop handoff: AC-7.4 (the honest re-score names the residual
    families, if any)
  - Promotion trigger: AC-7.4 verdict shows family-coupled residual AND
    the named families gain >= 3 slug-disjoint members.

## Claude-Codex Deliberation

### Codex-First Findings (Analysis v1)
- Highest risks: draft-as-roadmap exceeds one-loop throughput; T1
  "easy near-exact" needs explicit lockstep proof for rollup/predication/
  ATOM-vs-RED semantics; L1 SOTA transfer is the weakest accuracy
  assumption (depends on geometry facts not yet captured); sampling vs
  exact counts are in tension; the timing same-slug arm must be demoted
  before any new headline; producer hazards must be first-class ACs.
- Candidate criteria proposed by Codex v1 (hygiene gate, label gate, T1
  producer gate, lockstep gate, evidence gate, scope gate) were adopted
  into AC-1/AC-3/AC-4/AC-5 and the evidence-status requirements.

### Agreements
- Hygiene first: the three producer hazards are pre-bulk-regeneration
  blockers and first-class ACs (AC-1).
- Honesty first: the same-slug timing arm is structurally the same
  truth-fitted leakage pattern the project just demoted on the cache side;
  demotion precedes any new timing claim (AC-2).
- T1 replay counts are the strongest near-term NCU-replacement claim, but
  only with observation-contract microfixtures, exact-support accounting,
  rollup fail-closed handling, a hard non-vacuity floor, and sealed-holdout
  governance (AC-3/4/5).
- Analytical-first at this label budget: labels are validation, not fit
  data; learned components stay low-capacity (ridge/isotonic) on cal/val
  only.
- Governance: consumer-side only this loop (pinned tree untouched), sm_120
  typed-deferred, sampling deferred, no new dependencies.
- The cycles evaluator stays analytical/trace-replay; Codex round 3
  explicitly confirmed no simulator-revival drift (Rule 5).

### Resolved Disagreements
- **Loop scope** (round 1-2 vs user): Claude + both Codex passes
  recommended a narrowed one-loop T1-first plan ("roadmap split"); the
  USER decided on the full multi-milestone plan. Resolution: full scope
  adopted (DEC-1) with the scope-risk note, strict milestone ordering, and
  independently-closable ACs so a max-iterations end leaves a coherent
  prefix.
- **Lower bound vs AC semantics** (round 1): v1's lower bound weakened
  AC closure. Resolution: all ACs mandatory; scoped minimums moved inside
  the ACs (non-vacuity floor, typed-exclusion closure, honest-MISS
  closure).
- **"Live sweep green" ambiguity** (round 1): resolved by separating HARD
  structural gates from DIRECTIONAL numeric targets; a numeric MISS closes
  as an honest typed verdict.
- **NCU rollup risk** (round 1): `l1tex__t_requests.sum` spans more pipes
  than the producer models. Resolution: AC-4.1 contract-matching
  microfixtures + AC-4.2 rollup_unsupported fail-closed typing.
- **`smsp__inst_executed.sum` semantics** (round 1): resolved by live
  microfixture discovery before scoring (AC-4.1).
- **"git status clean" determinism** (round 1): resolved by path-scoped
  pre/post hashing + untracked detection (AC-1.3).
- **Sub-AC task coverage** (round 1): resolved by task rows targeting
  sub-ACs explicitly.
- **FUT-7 linkage** (round 3): FUT-7 lacked a Source DEC. Resolution:
  `Source DEC: DEC-7; Source AC: AC-7.4 conditional outcome`.
- **Battery wording** (round 3): AC-10.3 strengthened to "per commit, and
  re-verified at every closing HEAD".

### Convergence Status
- Final Status: `converged`
- Rounds executed: 3 (round 1: 9 required changes -> v2; round 2: clean
  pass on v2; round 3 on the user-expanded v3: 2 required changes, both
  applied here). All Codex round-3 optional improvements (full
  cache-stream identifiers in AC-8.1, fresh-sidecar rule in AC-9.1,
  holdout-governance cross-reference in AC-7.4) are incorporated.

## Pending User Decisions

All decisions below were resolved by the user in this gen-plan session
(2026-06-04). None remain `PENDING`.

- DEC-1: Loop scope — narrowed one-loop T1-first vs full multi-milestone.
  - Claude Position: one-loop T1-first with FUT roadmap (matches observed
    single-loop throughput).
  - Codex Position: same ("roadmap split").
  - Tradeoff Summary: full scope risks a max-iterations end; narrowed scope
    defers the L1/cycles work the project ultimately needs.
  - Decision Status: `Full multi-milestone plan` (user). Mitigations: strict
    milestone ordering, independently-closable ACs, scope-risk note.
- DEC-2: Demote the timing same-slug arm now, accepting a worse honest
  headline.
  - Claude Position: demote now. Codex Position: demote now.
  - Tradeoff Summary: honest MISS + diagnostic shadow vs a leaky pass.
  - Decision Status: `Demote now` (user).
- DEC-3: Pinned-tree policy this loop.
  - Claude Position: consumer-side only; wire emission deferred (FUT-6).
  - Codex Position: same.
  - Tradeoff Summary: avoids a re-pin governance round; defers
    producer-side cache-op provenance.
  - Decision Status: `Consumer-side only` (user). Linked: FUT-6.
- DEC-4: sm_120 scope this loop.
  - Claude Position: typed deferral (FUT-5). Codex Position: same.
  - Tradeoff Summary: no Blackwell session dependency; cross-arch
    verification waits for the alternating-session move.
  - Decision Status: `Typed deferral` (user). Linked: FUT-5.
- DEC-5: Trace sampling (Sieve/Photon) scope.
  - Claude Position: defer until exact T1 lockstep passes (FUT-4).
  - Codex Position: same.
  - Tradeoff Summary: avoids sampled-vs-exact confounds in the first
    lockstep claim; defers the "fast" amortization win.
  - Decision Status: `Deferred` (user). Linked: FUT-4.
- DEC-6: New Python dependencies (pygam/lightgbm monotone heads).
  - Claude Position: none this loop (isotonic/ridge only).
  - Codex Position: same.
  - Tradeoff Summary: environment stability vs richer pre-registered model
    candidates.
  - Decision Status: `No new dependencies` (user).
- DEC-7: Numeric-threshold posture for ALL of this plan's numbers (T1
  p50 <= 1% / p90 <= 3% / max <= 5%; L1/L2 geomeans; T2 p50 <= 10% /
  p90 <= 25%; cycles geomean <= 0.20; robustness gates per-head).
  - Claude Position: directional achieved-vs-target with hard
    structural/integrity gates (inherits the parent plans' precedent).
  - Codex Position: same.
  - Tradeoff Summary: honest closure on misses vs hard numeric completion
    gates that could block an otherwise-sound loop.
  - Decision Status: `Directional + hard structural` (user).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead
- The repository's plan-token scanner enforces this; it must stay green at
  every closing HEAD

### Process Requirements
- CLAUDE.md Rules 1-5 bind every task: NCU only via `tools/local_labels/`
  into `data/local_labels/` on the local 3090 Ti; no driver-root
  operations or clock locking; no privileged CUPTI (Activity allowed); no
  host-driver work; no simulator revival.
- The standing round-close protocol applies: every round that lands ANY
  commit closes label-commit-last (the documented refresh cycle) and runs
  the drift/recapture battery POST-commit at the closing HEAD.
- Round summaries record the per-AC evidence status (Offline-verified /
  Verified-live-sm_86 / Verified-live-sm_120 / Cross-arch-verified) with
  device, driver, CUDA toolkit, NVBit build, and observed clocks/throttle
  state for live evidence; behavioral vs timing vs SASS-layout evidence is
  distinguished per the architecture-coupling discipline.
- Frozen surfaces must not move: the L2 v5 head byte-identity guard, the
  30-cell sm_120 cohort freeze, and the 190-file path-boundary pin remain
  binding; new producers live outside the pinned tree.
- The sealed holdout is touched once per fitted/scored head, at final
  evaluation, with the trail recorded (AC-3.4 governance applies to every
  AC that scores against labels).

--- Original Design Draft Start ---

# RESEARCH DOSSIER — Accurate + Fast Non-Root GPU Performance Modeling, Next Phase (NCU Replacement: Wall-Clock / L1 / L2, sm_86 + sm_120)

Repo: `/home/jakeshea/3090ti_5060laptop_cross_study/3090ti_5060laptop_cross_study/non_root_gpu_profiler_codex`
Date: 2026-06-04 · Branch: `cleanup/pre-merge-simplification` (HEAD `b37bc7d5`)
Scope: seeds a `/humanize:gen-plan` implementation plan for the phase AFTER
`plans/sm86_multimetric_ncu_replacement_plan.md` (42-round loop, complete) and
`plans/sm86_ncu_replacement_continuation_plan.md` (AC-1 executed; loop complete).

> **Research provenance.** This dossier synthesizes (a) a 13-agent research
> workflow (4 repo-state analysts, 5 local-PDF deep-readers over the curated
> 64-paper corpus, 4 web researchers covering 2023–2026 literature), (b) an
> independent `codex exec` deep-research report (read-only, web-enabled), and
> (c) direct session verification of the load-bearing repo artifacts. Facts
> marked **[verified-this-session]** were re-checked directly against the
> artifact bytes in this session; other repo numbers were extracted by the
> repo-state analysts with `file:line` citations quoted from the artifacts.

---

## 1. Executive Summary

The two completed plans built the **corpus/ontology spine** and ran the
**pivotal controlled refit**. The honest post-loop position is bimodal:

- **L2 sector hit rate is a real, competitive, deployable result**: the frozen
  v5 isotonic-on-walker head scores full-cohort geomean relative error
  **0.0642** (gate ≤0.15 PASS), median abs **0.0592** (≤0.06 PASS), sealed
  holdout geomean **0.0552** (≤0.25 PASS), and beats a clean-room
  Nugteren-HPCA-2014 reimplementation on 75% of 40 cells. Literature context:
  the best published trace-driven L2 results are ~13–16% error (PPT-GPU,
  Buelow 2022), so 6.4% on a 40-cell sm_86 cohort is at or beyond published
  SOTA — *on accuracy*. **On robustness it still fails**: LOOCV-vs-holdout gap
  0.3385 (gate ≤0.20) and LOSO max 0.7076 (gate ≤0.25), driven by
  register_spill_local_mem (0.708), l2_prefetch_hint_sweep (0.573),
  working_set_capacity_sweep (0.501).
- **L1 sector hit rate is honestly broken**: deployable friendly-L1 geomean
  **1.947** (>190% relative error, HARD-2 MISS) after the truth-fitted
  correction factors were demoted to `diagnostic_truth_fitted_demoted`; L1
  closure geomean 0.89 over 30 cells. The SOTA root-cause match is exact and
  actionable (§3): the L1 walker has **no intra-SM warp interleaving, no MSHR
  model, LRU + fully-associative assumptions that real GPU L1s violate, and a
  fixed (non-carveout-aware) capacity** — precisely the error sources
  Nugteren 2014 quantified at up to ~50pp and fixed with a warp-pool
  scheduler + finite-MSHR model at 6.4% MAE.
- **Wall-clock cycles exists but its headline is leakage-suspect**
  [verified-this-session]: `timing_cycle_predictor.py` (roofline prior + ridge
  residual) reports a passing publication gate (cycle geomean 0.0559, n=13),
  but `MODEL_VERSION = "timing_cycle_ridge_v3_same_slug_local_residual"` and
  the 13-cell cohort spans only **4 slugs** (gemm×5, conv2d×4, flash_attn×2,
  dot×2), so **every** leave-one-out cell had a same-slug neighbor and the
  LOO predictions substitute same-slug NCU-label scaling — the same circular
  pattern whose cache-side analogue was just demoted. The deployable
  slug-disjoint cycles claim is **open**. The deploy-ns leg is already honest
  (interval-only, 1890 MHz short-burst anchor, `point_result_claimed=false`).
- **T1 replay counts and T2 traffic streams — the plan's own headline
  "NCU count-replacement" claim — are entirely unstarted** (zero of 15 count
  metrics captured; no producer; no lockstep validator; continuation
  task4–task15 all `pending`).

The SOTA study converges, from three independent directions (local PDF
corpus, 2023–2026 web literature, codex deep research), on one architecture:

> **Trace once → summarize once → predict many metrics.** A one-pass NVBit
> trace summarizer feeds (i) a deterministic T1 replay-count reconstructor
> (published fidelity: requests within ~0.001%, sectors within ~0.28% of
> hardware — the easiest and highest-confidence NCU-replacement claim), (ii)
> a scheduler-aware, MSHR-aware, sectored (HRD 32B/128B) cache walker for
> L1/L2 hit rates (published: 3.7–6.4% L1 MAE), (iii) a GCoM-lite analytical
> cycles model whose dependence stalls are **read nearly directly from SASS
> control bits** (stall/yield/dependence counters via cuobjdump — the
> single biggest non-root accuracy lever, per the MICRO'25 modern-core line:
> 13.45% MAPE Ampere, 17.41% Blackwell), and (iv) a DVFS decomposition
> `T(fc,fm) = t0 + max(α/fm, β/fc)` that converts cycles to NVML-conditioned
> ns intervals. Low-capacity ML (isotonic/ridge, optionally an
> efficiency-factor head bounded by the analytical ceiling) sits ON TOP of
> analytical priors, never replaces them, at this label budget. No published
> system fuses all of this under non-root constraints across Ampere +
> Blackwell — that fusion is this project's novelty claim.

---

## 2. Current State (Grounded)

### 2.1 What the completed loops landed

- **Cohort**: 44 sm_86 cells / 37 slugs, sealed split cal-28/val-8/holdout-8
  (frozen at commit `501723b0`); 40 cells admitted for v5 fit/score (4 holdout
  cells `predictor_bundle_absent`); L1 head cohort 30 cells. Full label store:
  74 cells (44 sm_86 + 30 sm_120 frozen at `b1646ef7`)
  (`data/sm_86_predictor_accuracy_recovery_v2/cohort_split.json`,
  `data/local_labels/sm_86_v5_head.json`, `data/local_labels/label_admitted.json`).
- **L2 accuracy gates** (`calibration_head_v5_accuracy_gates_sm_86.json`):
  full-cohort geomean 0.0642 / median abs 0.0592 / holdout 0.0552 — all PASS.
  Per-bucket p90 absolute-error gates FAIL: memory_stress p90 0.2346 (n=9),
  other p90 0.3587 (n=15) vs threshold 0.15.
- **L2 robustness gates** (`calibration_head_v5_robustness_sm_86.json`): FAIL.
  LOOCV-vs-holdout gap 0.3385 (≤0.20), LOSO max 0.7076 (≤0.25). Pre-refit
  values were 0.441 / 0.814 — the expanded corpus improved both but the
  pivotal hypothesis ("the corpus expansion is the structural fix") is **not
  confirmed**. Worst LOSO slugs: register_spill_local_mem 0.7076,
  l2_prefetch_hint_sweep 0.5734, working_set_capacity_sweep 0.5011.
- **Baseline panel** (`calibration_head_v5_baseline_comparison_sm_86.json`):
  v5 geomean 0.0628 vs uncalibrated walker 0.111, Nugteren-2014 reimpl 0.117
  (v5 wins 75% of cells), always-hit 0.117, cohort-mean 0.325, v4 Platt 0.203.
- **L1**: deployable friendly-L1 geomean **1.9469743371673827** (HARD-2 MISS,
  target <0.20); hostile-L1 0.3172; the diagnostic-shadow channel (the
  would-have-been factor-corrected values) reports 0.0148 but is **not
  deployable** (truth-fitted circularity). L1 closure geomean 0.89
  (`interim_diagnostic_report_sidecar.json`, shadow reports,
  `l1_closure_comparison_sm_86.json`).
- **Demotion registry**: all four correction-factor families
  (l1_silu_elementwise, l1_shared_reduction, l2_reduction_kernel,
  l2_high_footprint_low_reuse) carry
  `correction_factor_status=diagnostic_truth_fitted_demoted`; production
  helpers no-op the deployable value (`cache_predictor.py:3028-3065`).
  Blocking fact for re-derivation: silu has 1 slug, shared_reduction has 2 —
  no slug-disjoint data exists for those families.
- **Falsification** (`calibration_head_v5_falsification_sm_86.json`): v4→v5
  joint contribution 14.33% — **indeterminate** band [10%,30%); v4 MAE 0.1686
  vs v5 0.1323 (21.5% residual reduction). The prior "confirmed" verdict was
  an artifact of the truth-fitted factor leak.
- **plan_1_closure_status**: `aborted_unrecoverable_per_user_decision`.

### 2.2 Wall-clock truth [verified-this-session]

- `workloads/simbench_bw/timing_cycle_predictor.py` exists: static roofline
  prior (GA102 tensor peak × 0.066 efficiency prior, from
  `tools/hrd_profiler/ga102_peak_constants.json`) + ridge residual (λ=10) on a
  14-feature deploy-legal vector (label-only fields actively blocked by
  `assert_feature_vector_deploy_legal`).
- `MODEL_VERSION = "timing_cycle_ridge_v3_same_slug_local_residual"`: at LOO
  validation time, `_same_slug_scaled_neighbor_prediction` (line ~444)
  replaces the ridge prediction with `true_cycles(nearest same-slug cell) ×
  work_ratio` whenever a same-slug training cell exists.
- The 13-cell cohort (`timing_cycle_predictions_sm_86.json`) spans **4
  slugs**: conv2d_implicit_gemm_bf16_sm80 ×4, dram_streaming_dot_product ×2,
  flash_attn_fwd_bf16_sm80 ×2, gemm_hmma_bf16_sm80 ×5 ⇒ every LOO cell had a
  same-slug neighbor ⇒ the committed `publication_pass` (cycle geomean
  0.055856, median abs log 0.010734, bootstrap upper 0.106601, in
  `timing_closure_sm_86.json`) is a **same-slug result**, not a slug-disjoint
  deployable claim. Evidence status `Offline-verified`.
- The deploy-ns leg is interval-honest: `anchor_regime=short_burst`,
  `sm_clock_ref_mhz=1890.0`, `clock_uncertainty_fraction=0.06`,
  `point_result_claimed=false`. Dual-source noise floor 110352 ns (committed,
  provenanced). NCU cycles label (`gpc__cycles_elapsed.max`) is DVFS-stable;
  ns label inherits clock estimation error.

### 2.3 Unexecuted continuation work (all `pending` in the goal tracker)

task4 (widen NCU label capture to 8 probe-confirmed T1 counters), task5
(probe 7 pending identifiers + mem_clock/dram_bytes/pct-peak), task6 (NVBit
replay-count producer), task7 (T1 lockstep validator), task8 (Codex T1
review), task9 (texture/surface disposition), task10 (rebuild/re-pin 22
timing-exclusion binaries + stale readiness report), task11 (T2 traffic
labels + predictor), task12 (T3 re-score over rebuilt cohort), task13
(per-(cell,metric_key) harness + baselines/ablations), task14 (Codex stats
review), task15 (final compliance). Zero T1 count metrics exist in
`label_admitted.json` today.

### 2.4 Measurement substrate (features available vs absent)

Available now (producers committed and live on sm_86):
- **NVBit dynamic trace** (`tools/nvbit_tracer/`, byte-pinned path-boundary):
  per-instruction pc, active/predicate masks (proto3 optional), per-lane
  addresses (3 compression formats + allocation registry), 10-value
  `memory_op_class`, data width, warp/SM ids, vpc;
  `instruction_cache_op` wire slot **reserved but producer classifier not
  landed** (today resolves UNSPECIFIED ⇒ `predictor_l1_sass_provenance_missing`).
- **%clock64 phase ledger** (env-gated injection; function/basic-block/bar
  phase records) — built, available for phase timing evidence.
- **CUPTI Activity collector** (dual-source vs CUDA events; noise floor
  max(5%, 5µs); calibrated p99 110352 ns) + **NVML covariates** (200 Hz; typed
  `nvml_alignment_gap` for sub-poll kernels).
- **Static SASS extractor** (`static_sass_features.py`): 9-family mnemonic
  histogram, register/spill usage, separate static FLOP/byte estimate —
  `claim_class=static_sanity_check_only`; order-of-magnitude FLOP weights.
- **Device peaks** for sm_86 (`device_constants.json`: 1008 GB/s DRAM,
  1890 MHz boost; GA102 peak table in `tools/hrd_profiler/`); **all sm_120
  device constants are null** (`deferred_no_live_evidence`).
- **Trace budgets** (`tools/cloud/trace_budgets.py`): slowdown cap 1200×,
  live sm_86 ratio 434.72×, drop-rate cap 0.05, per-cell byte caps 2 GiB
  default down to 64 MiB with `threadblock_stride_N` / `kernel_limit_N`
  sampling grammar; sub-1ms baselines suppressed from the slowdown gate
  (µs-kernels inflate to ~14000–19000×).

Absent / planned (gaps the next phase must fill):
- Producer-side SASS cache-op classifier (wire tags 12–13 reserved).
- Per-instruction **dependence/control-bit extraction** (stall counter,
  dependence counters, yield — readable from static SASS via cuobjdump; not
  yet extracted).
- **P-chase / latency microbenchmark suite** for per-arch latency + cache
  geometry constants (L1/L2 latency, MSHR count, set mapping, replacement) —
  the published numbers (A100: global ~290 / L2 ~200 / L1 33 / shared 23/19
  cycles; GB203 Blackwell: L2 ~358, global ~876, L1 30–40) are
  close-cousin-arch facts that must be re-measured non-root on the actual
  3090 Ti and 5060 laptop.
- One-pass **trace summarizer / feature store** (sector summaries and
  mnemonics are currently reconstructed offline per-consumer rather than
  emitted as one canonical feature bundle).
- 13/15 portable_lcd roster slugs lack host-local traces; 4 holdout cells
  `predictor_bundle_absent`; 22 timing-exclusion binaries stale (task10 debt).

### 2.5 Producer hazards queued from the last loop (must fix before bulk regen)

- `run_v5_active_path_comparison` clobbers `sm_86_v5_head.json` (producer
  collision — observed once: admitted_count=0 stub).
- `regenerate_v5_artifacts` wrapper runs a split-unaware fitter step.
- A sweep test restamps `deploy_ns_crosswalk_sm_86.json` in place (test
  pollution).
- Tools-sweep failure baseline residual: 6 entries (nested-provenance ×3,
  sidecar-schema ×2, ledger digests) + simbench ac5 pair.

---

## 3. Failure-Mode → SOTA-Fix Map

| # | Observed failure (repo evidence) | Root cause | SOTA fix (source, published numbers) |
|---|---|---|---|
| F1 | L1 deployable geomean 1.947; closure 0.89 | `l1_walker.py:_walk_one_launch_fa` walks per-block accesses **sequentially block-by-block** — no intra-SM warp interleaving; **no MSHR model at all** | Nugteren HPCA'14: FIFO warp-pool (one memory request per warp, re-insert delayed by latency) + finite-MSHR merge/stall + latency-miss class. Ablated impact up to ~50pp; full model 6.4–8.3% L1 MAE, 268× faster than GPGPU-Sim |
| F2 | "Set-conflict over-prediction" documented; carveout cells predictor-hostile | L1 modeled as fully-associative LRU with fixed capacity; real GPU L1 is **non-LRU** with XOR-hashed set mapping and runtime-carveout capacity | Mei&Chu 2015: P-chase methodology reverse-engineers size/sets/ways/mapping/replacement **non-root** (%clock64 only). Run on both cards to populate per-arch static cards; make capacity carveout-aware |
| F3 | Sector vs line handled by a single LRU stack approximation | Flat reuse distance cannot track partial-line (sector) eviction | Maeda HPCA'17 HRD(32,128): one-pass two-granularity RD (propagate-on-infinity), >3× sectored miss-rate error reduction; repo already has `cache_model_hrd_second_layer_enabled` to build on |
| F4 | LOSO offenders: register_spill 0.708, l2_prefetch 0.573, working_set 0.501; inter-CTA reuse approximated by `alpha*log2(grid_volume/sm_count)` | Per-slug/per-family heuristics instead of structural features | SHALOM TPDS'22: classify accesses local / sequential-sharing / concurrent-sharing / private; MDM HPCA'20: DPKI + MSHR-batching + NoC/DRAM queueing features (arch-independent, NVBit-derivable) |
| F5 | Correction factors circular (truth-fitted) → demoted; HARD-2 reopened | Labels used as fit inputs on the same cells | Analytical-first discipline (Wang HPCA'19 hybrid; GCoM): labels become **validation**, not fit data — the structural escape from the 35–50-label budget |
| F6 | Timing publication_pass is same-slug leakage on a 4-slug cohort | LOO substitutes same-slug NCU-label scaling | GCoM-lite cycles (interval/CPI-stack) + **SASS control-bit stall readout** (Huerta MICRO'25: 13.45% Ampere / 17.41% Blackwell MAPE; control bits readable via cuobjdump = non-root near-direct dependence-stall readout); slug-disjoint gates mandatory |
| F7 | LOOCV/LOSO robustness gates still fail after corpus expansion | 0-parameter isotonic on a 3-feature frozen projection + 4 family buckets; richer monotone candidates never fit (deps absent); tiny per-bucket N | Analytical priors carry the structure; low-capacity heads only (ridge/isotonic); HELP/Multi-Predict few-shot device embeddings for cross-arch; CDMPP domain-invariant features; report robustness achieved-vs-target honestly |
| F8 | ns target structurally noisy (110 µs floor, DVFS, sub-poll kernels) | Single-anchor cycles→ns; NVML 200 Hz cannot see µs kernels | DSO 2024: `T(fc,fm)=t0+max(α/fm,β/fc)` compute/memory-clock decomposition; Wang/Wu DVFS structure (3.5–4.8% MAPE) with NVBit-reconstructed counts replacing forbidden counters; keep ns as interval until DVFS residual proven |
| F9 | Traces cost 434.7× and dominate; 13/15 portable_lcd slugs untraced | Full per-instruction tracing of everything | Sieve IISWC'23: stratify invocations by NVBit dynamic instruction count alone (1.2% error vs PKS 16.5%; 8–98× less profiling; explicitly names NVBit); Photon MICRO'23 warp-type/BB-stability early stop; cuThermo single-block sampling; PKA one-occupancy-wave guard |
| F10 | T1/T2 headline unstarted | — | Accel-Sim NVBit tracer line: reconstructed requests within ~0.001%, transactions/sectors within ~0.28% of hardware — deterministic, near-exact, the highest-confidence NCU-replacement claim available; PRT coalescing abstraction (Huerta MICRO'25) |

---

## 4. SOTA Survey Synthesis (per target)

### 4.1 Wall-clock cycles (T3 headline)

- **Lineage**: GPUMech (MICRO'14, interval analysis, 13–14% vs sim, 97×) →
  MDM (HPCA'20, MSHR-batching + NoC/DRAM queueing, 13.9% vs sim / 40% vs real
  GTX 1080; fixes 162–298% GPUMech error on memory-divergent apps; collects
  inputs **via NVBit by name**) → GCoM (ISCA'22, sub-core + sectored-L1 +
  MDM embedded, 10.0% MAE vs Accel-Sim, **24.4% vs real RTX 2060**, cross-GPU
  25.5% on RTX 3070; 7.6–345× speedups). All consume per-warp NVBit
  SASS+address traces; none needs counters at predict time.
- **Modern-core line** (Huerta et al., arXiv 2503.20481 + MICRO'25): the
  decisive non-root lever — NVIDIA dependence handling is compiler-managed
  and **encoded in SASS control bits** (stall counter 0–15, six dependence
  counters 0–63, yield, DEPBAR.LE), readable statically via cuobjdump.
  CGGTY issue policy; no operand collector; PRT coalescing; RF bank-conflict
  rules. Validated MAPE: 13.45% Ampere A6000, 17.41% Blackwell RTX 5070 Ti
  (corr 0.99) vs Accel-Sim 34.03%. **13–17% is the literature floor for
  cycles on this signal set**; analytical (non-simulator) variants should
  expect the 20–40% real-hardware band (GCoM 24.4%, MDM 40%).
- **Latency constants** (Abdelkhalik HPEC'22 A100; Jarmusch 2025 GB203;
  Arafa 2019 methodology): global ~290 / L2 ~200 / L1 33 / shared 23,19
  cycles (A100); Blackwell L2 ~358, global ~876.7, unified INT32/FP32 pipe,
  2 FP64 units/SM. All obtained with %clock64 + SASS only — **re-measure on
  the actual two cards with the same non-root method**.
- **Tensor kernels are the hard tail**: AMALI (ISCA'25) 23.59% MAPE on LLM
  kernels (GCoM degrades to 127.56%); the 2026 microbenchmark-driven stage
  model (TMA/TMEM/math stages) hits 1.31% on B200 tensor GEMMs but is weak on
  irregular apps; naive roofline shows >95% error on modern kernels.
- **Hybrid heads**: NeuSight (ASPLOS'25) roofline-bounded tile-MLP, 8.1%
  unseen-GPU avg; SynPerf (2026) analytical-pipeline features + shallow MLP,
  6.1% seen / **11.4% unseen incl. an sm_120-class RTX PRO 6000** — the
  "efficiency factor against an analytical ceiling" framing is the right
  shape for a small learned head at tiny label budgets.

### 4.2 L1/L2 sector hit rate (T2)

- Counterless trace-driven cache modeling is proven: Nugteren 2014 (6.4–8.3%
  L1 miss-rate MAE; warp-pool + MSHR + associativity + latency-miss
  extensions, each ablated), Buelow CGF'22 (NVBit→LRU sector sim validated
  vs NCU: **L1 3.73%, L2 15.81%**), PPT-GPU (~13% L1/L2 hit-rate error,
  450× faster than GPGPU-Sim), MLCRP (IEEE Access'25, ML over reuse
  profiles, <5% MAE, ~4 orders faster than cycle-accurate).
- The repo's L2 (6.4% geomean) **already beats the published L2 numbers**;
  the repo's L1 is far below SOTA — and §3 F1–F3 explain exactly why, with
  the fix list (warp interleaving, MSHRs, HRD(32,128), per-arch set
  hash/replacement, carveout-aware capacity).
- What a per-warp trace walker typically gets wrong (web sweep synthesis):
  single deterministic interleaving; global LRU vs non-oracle eviction;
  line-not-sector fills; no L1 bank/port contention; no finite
  MSHR/miss-merging; mishandled writes (L1 write-through/no-allocate vs L2
  write-back semantics).
- IPOLY pseudo-random L2 set indexing (Huerta, citing Jia) — a naive modulo
  mapping mispredicts conflict misses; the hash must be widened for
  Blackwell's larger L2.

### 4.3 T1 replay counts + T2 traffic streams

- Accel-Sim/Macsim NVBit tracer evidence: **requests within ~0.001%,
  transactions (32B sectors) within ~0.28%** of hardware counters. The
  per-instruction address + active-mask + opcode stream the repo already
  collects is sufficient to rebuild `inst_executed`, per-op requests, and
  sector counts almost exactly. This is the easiest, most defensible
  NCU-replacement claim — **build it first**.
- PRT (pending-request-table) coalescing abstraction (Huerta MICRO'25) maps
  NVBit addresses+masks → requests → sectors deterministically.
- MDM's equations directly produce T2 quantities (DRAM sectors via
  LLCMissRate×BlockSize; NoC/xbar traffic via #SMs×M×L^NoCService), so the
  T2 stream model and the cycles model share components.
- Risk register: ~0.28% sector residual includes sync-barrier internals NVBit
  misses; ATOM-vs-RED semantics; predication; Blackwell async-copy/TMA paths
  must route through the existing `kUnspecified` fail-closed fallback until
  classified.

### 4.4 ML/learned predictors at a 35–50-label budget

- Label-budget verdict (PDF deep-read): XAPP needed 122 datapoints, Braun 189
  kernels/GPU, SynPerf ~1M samples, NeuroScalar ~1.5B simulator-labeled
  instructions. **No pure-ML method was demonstrated at this repo's budget.**
  Analytical-first with labels-as-validation (Wang HPCA'19 hybrid, 17.04%
  MAPE, zero training labels) is the only structurally sound escape.
- Consumer-DVFS warning that matches the repo's failed gates: Braun's random
  forest hit 8.86–13.86% on professional GPUs but **52% median MAPE on a
  consumer GTX1650** (uncontrollable DVFS) — the exact risk profile of the
  5060 **laptop**. NVML clock/thermal/power/throttle covariates are
  mandatory inputs, not optional.
- Few-shot device transfer toolkit for sm_86→sm_120: HELP (NeurIPS'21,
  ~10 samples/new device) and Multi-Predict (AutoML'23, 4–10 samples via
  learnable hardware embeddings); CDMPP (EuroSys'24) domain-invariant
  representation + balanced sampling (10.85% cross-device). Borrow the
  hardware-embedding + domain-adaptation recipes when the sm_120 leg opens.
- Tree ensembles cannot extrapolate beyond their training range — a
  structural conflict with honest slug-disjoint LOSO where a held-out slug
  may be the extreme. Keep heads monotone/low-capacity.

### 4.5 Fast-model engineering (the "fast" requirement)

- **Sieve (IISWC'23)** — adopt wholesale as the inter-kernel layer: stratify
  kernel invocations by NVBit dynamic instruction count (CoV threshold
  θ=0.4), trace ONE representative per stratum, weight by instruction count,
  reconstruct via weighted-harmonic-mean IPC. 1.2% avg error vs silicon
  (PKS 16.5%), 8–98× less profiling; cross-arch relative-perf error 1.5%
  (vs PKA 9.8%) — directly serves the sm_86↔sm_120 decoupling goal. Sieve
  itself names NVBit as the intended lightweight tool.
- **Photon (MICRO'23)** — intra-kernel: warp-type identity (identical
  instruction/PC sequence) + basic-block issue→retire slope stability as the
  trace-termination signal (IPC is NOT reliably stable on GPUs — do not use
  IPC-stability stops). Re-tune thresholds per-arch.
- **PKA (MICRO'21)** — the "one full occupancy wave before trusting
  stability" guard; weighted-sum reconstruction generalizes to DRAM
  utilization.
- **cuThermo (2025)** — single-thread-block sampling to bound NVBit overhead.
- **Concorde (ISCA'25, CPU)** — represent the trace as compact per-component
  performance-bound distributions; O(1) inference independent of trace
  length; attractive shape for the one-pass summarizer's output.
- Net effect: the 434.7×-per-cell trace cost stops being the binding
  constraint because most invocations are never fully traced.

### 4.6 Non-root precedent and the novelty claim

Counterless NCU-style metric reconstruction exists in fragments: Accel-Sim
tracer (counts), Buelow/PPT-GPU/MLCRP (cache hit rates), DSO/Wang-Wu (DVFS
wall-clock, but counter-fed), NeuSight/SynPerf (ns, but no cache/sector
metrics), Penrose (counter-dependent, opposite design), eBPF/CUPTI-activity
lines (timing only). **No published system fuses NVBit sector/cycle
reconstruction + DVFS cycles→ns decomposition + read-only NVML covariates,
calibrated against quarantined NCU labels, across Ampere AND Blackwell.**
That fusion — with typed fail-closed exclusions and slug-disjoint honest
gates — is the project's publishable contribution.

---

## 5. Recommended Next-Phase Architecture

### 5.1 Per-target model classes

| Target | Model class | First deployable form | Literature anchor |
|---|---|---|---|
| T1 replay counts (inst_executed, per-op requests, ld/st sectors, l1tex requests) | Deterministic trace replay: active-mask × PRT coalescing × 32B sectoring per op-class | Exact-or-typed counts + NCU lockstep validator (p50 ≤1%, p90 ≤3%, low-support floors) | Accel-Sim tracer 0.001%/0.28% |
| L1 sector hit rate | Scheduler-aware sectored walker: warp-pool FIFO interleaving + finite MSHR + HRD(32,128) + per-arch set-hash/replacement/carveout from P-chase cards | Analytical hit rate + existing isotonic calibration; correction factors stay demoted | Nugteren 6.4%, Buelow 3.73%, Maeda >3× |
| L2 sector hit rate | Frozen v5 baseline + traffic-first upgrade (predict requested/hit/miss sectors, then derive rate); MDM divergence + SHALOM sharing features for the LOSO offenders | v5 stays the deployable head until a challenger beats it slug-disjoint | repo 0.064 vs published 13–16% |
| T2 traffic streams (dram/xbar sectors) | Miss-stream composition from the cache walker + MDM queueing identities | p50 ≤10%, p90 ≤25% with denominator floors (continuation AC-5 gates) | MDM; Buelow L2 ~16% context |
| Wall-clock cycles | GCoM-lite: SASS control-bit stall readout + interval/CPI stack + MDM memory terms + per-arch latency tables (P-chase) ; ridge residual only on top | Slug-disjoint cycle geomean with uncertainty; same-slug arm demoted to diagnostic | Huerta 13.45%/17.41%; GCoM 24.4% real-hw |
| Wall-clock ns | `T(fc,fm)=t0+max(α/fm,β/fc)` over predicted cycles + NVML covariates; dual-source crosswalk | Interval claim (already honest); point claim only for kernels above the noise floor with in-window NVML samples | DSO 4.6%; Wang/Wu 3.5–4.8% |

### 5.2 Feature pipeline (new producers, all non-root)

1. **One-pass NVBit trace summarizer / feature store**: per-kernel canonical
   bundle — dynamic opcode-class counts, per-PC memory streams, sector
   histograms, RD/HRD bins (32B + 128B), active-mask/divergence (DPKI)
   features, coalescing distributions, per-PC cache-op classes (once the
   classifier lands), %clock64 phase deltas, launch/occupancy, NVML/CUPTI
   covariates. Raw full traces become a debugging/label-side mode.
2. **Static SASS control-bit extractor**: stall counter, dependence-counter
   masks, yield, DEPBAR thresholds, reuse bits per instruction (cuobjdump
   parse; new tool OUTSIDE the byte-pinned tracer tree).
3. **P-chase microbenchmark suite** (per-arch static cards): L1/L2/shared/
   global latencies, MSHR count, L1 set mapping + replacement evidence,
   carveout capacity sweep — %clock64-bracketed, run on both local cards.
4. **Producer-side SASS cache-op classifier** filling the reserved wire slots
   (lifts `predictor_l1_sass_provenance_missing`).
5. **Sieve stratification + Photon early-stop** wired into the trace-budget
   machinery as pre-declared sampling policies.

### 5.3 Build order (information-per-round)

0. **Hygiene gate**: fix the two producer-collision hazards + crosswalk test
   pollution (§2.5); demote the timing same-slug arm to diagnostic (mirror
   of the correction-factor demotion) and re-score the ridge slug-disjoint.
1. **T1 producer + lockstep** (continuation task4–task7): widen labels to the
   8 probe-confirmed counters; build the deterministic count reconstructor;
   validate lockstep. Highest confidence, fastest claim, unlocks F-feature
   counts for everything else (DVFS model inputs, coverage denominators).
2. **L1 walker rebuild** (F1–F3 fixes + P-chase static cards): warp-pool
   interleaving, finite MSHRs, HRD(32,128), per-arch geometry; re-score L1
   honestly. This is the largest accuracy delta available anywhere in the
   project (1.947 → published 0.04–0.10 band if SOTA transfers).
3. **T2 traffic streams** (task11): labels + miss-stream model.
4. **Cycles (GCoM-lite) + ns interval** (task12 upgraded): control-bit
   extractor + interval model + DVFS decomposition; timing cohort slug
   expansion (4 → ≥10 slugs) so slug-disjoint scoring is possible at all.
5. **Per-metric evidence harness** (task13–15): per-(cell, metric_key)
   verdicts, baselines, ablations, claim-to-evidence regeneration, Codex
   stats review, final compliance.

### 5.4 Corpus strategy

- Grow sm_86 timing-labeled slug diversity first (the 4-slug timing cohort is
  the binding constraint, not total cell count).
- Family-targeted growth (per codex + the LOSO offenders): register/spill,
  prefetch-hint, working-set-capacity families need ≥3 slug-disjoint members
  each before any family-coupled feature is deployable.
- sm_120: defer live work to the alternating-session model; when it opens,
  apply the HELP/Multi-Predict few-shot recipe + per-arch P-chase cards
  rather than refitting from scratch.

---

## 6. Accuracy + Speed Targets Calibrated by Literature

Proposed as DIRECTIONAL targets (DEC-7 precedent: report achieved-vs-target
with CIs; structural/integrity gates stay HARD):

| Metric | Target | Literature justification |
|---|---|---|
| T1 counts | p50 ≤1%, p90 ≤3%, max ≤5% rel. error | Accel-Sim 0.001%/0.28% headroom |
| L1 sector hit rate | geomean rel ≤0.15; per-cell ≤10pp p90 | Nugteren 6.4–8.3%, Buelow 3.73% |
| L2 sector hit rate | hold ≥ v5 (0.064); fix per-bucket p90 ≤0.15 | repo already ≥ SOTA |
| Cycles (slug-disjoint) | geomean log-rel ≤0.20 (stretch 0.15) | Huerta 13–17% floor; GCoM 24.4% real-hw |
| ns | interval coverage ≥90% at stated clock band | DSO/Wang-Wu structure; 110 µs floor honored |
| Robustness | LOOCV gap ≤0.20, LOSO ≤0.25 re-tested per head | unchanged project gates |
| Trace budget | ≤1200× per traced representative; ≥5× amortized profiling reduction via stratification | Sieve 8–98× |
| Model runtime | per-kernel prediction ≪ kernel runtime (analytical + O(N log N) walker; no per-cycle simulation) | Nugteren 268×, PPT-GPU 450× class |

---

## 7. Hard Constraints & Frozen Invariants (carry forward verbatim)

- CLAUDE.md Rules 1–5: NCU only via `tools/local_labels/` →
  `data/local_labels/` on the two local cards; no nvidia-smi state mutation
  (no clock locking — DVFS handled by covariates, never controls); no
  privileged CUPTI (Activity allowed); no host-driver work; **no simulator
  revival** (analytical, trace-replay, and ML models are in scope; HyFiSS/
  Accel-Sim/GCStack are cited for abstractions and validation methodology
  only — no integration).
- Frozen surfaces: L2 v5 head byte-identity + regression guard; 30-cell
  sm_120 cohort at `b1646ef7`; 190-file path-boundary pin over
  `tools/nvbit_tracer/` + the CUPTI collector (new producers live OUTSIDE the
  pinned tree or re-pin in the same commit per the documented protocol);
  additive-only label schema (v5) with lockstep consumer updates.
- Validation discipline: slug-disjoint splits, sealed holdout touched once,
  typed fail-closed exclusions (never silent gaps), four evidence statuses
  (Offline-verified / Verified-live-sm_86 / Verified-live-sm_120 /
  Cross-arch-verified), falsification artifacts as first-class outcomes.
- Plan-token scanner: no `AC-*`/Milestone/Round markers in implementation
  code; the standing label-commit-last + POST-commit drift/recapture battery
  applies to any round that lands a commit.
- Architecture-coupling discipline: every latency/geometry/scheduler constant
  is a per-arch covariate (measured by the P-chase suite), never a baked
  cross-arch invariant; unknown sm_120 opcodes route through
  `kUnspecified` → `unsupported_memory_path`.

---

## 8. Open Questions / Decisions for Operator + Codex

- **Q1 — Scope and order.** Adopt the §5.3 build order (hygiene → T1 → L1
  rebuild → T2 → cycles → harness)? Alternatives: (a) finish continuation
  task4–15 as written before any SOTA upgrades; (b) L1-first (biggest
  accuracy delta) before T1. The dossier recommends §5.3: T1 is the fastest
  defensible claim and feeds features to everything else.
- **Q2 — Timing same-slug demotion.** Demote
  `timing_cycle_ridge_v3_same_slug_local_residual`'s same-slug arm to
  diagnostic (mirroring the correction-factor demotion) and accept that the
  slug-disjoint cycles number will initially be much worse than 0.056 — or
  keep the same-slug arm as the headline with an explicit caveat? The
  dossier strongly recommends demotion: the project's credibility rests on
  the honesty standard it already set in this loop.
- **Q3 — Robustness-gate posture.** The LOOCV/LOSO gates have now failed
  through two loops. Options: (a) keep gates as-is and target the L1
  rebuild + structural features at the named offenders; (b) re-scope the
  gates per-head (a fresh analytical head may pass where the calibration
  head cannot); (c) report-as-miss permanently under DEC-7. Recommendation:
  (a)+(b) — new heads get their own recomputed gates.
- **Q4 — New-producer governance.** The P-chase suite, control-bit
  extractor, SASS cache-op classifier, and trace summarizer are NEW
  producers. Confirm they live outside the byte-pinned tree (new
  `tools/` subtrees) vs amending the path-boundary pin. The NVBit tracer
  wire format should need only additive changes (cache-op classifier fills
  reserved tags 12–13).
- **Q5 — Sampling in scope?** Wire Sieve stratification + Photon early-stop
  into trace budgets this phase (the "fast" half of the user goal), or defer
  to a follow-up loop once the accuracy spine is rebuilt? Recommendation:
  land Sieve stratification (cheap, pure offline arithmetic over existing
  budgets) this phase; Photon early-stop as a stretch/follow-up.
- **Q6 — sm_120 this phase?** The session model alternates devices. Does
  this plan include a Verified-live-sm_120 leg (P-chase cards + label
  capture on the laptop) or stay sm_86-only with typed deferrals? The
  Braun-GTX1650 evidence (52% MAPE under consumer DVFS) says the sm_120
  timing leg NEEDS the covariate machinery to be ready before it runs.
- **Q7 — Model-class governance.** Install pygam/lightgbm to actually fit
  the pre-registered monotone GAM/GBM candidates (recording the environment
  change), or keep isotonic-only until the analytical rebuild lands?
- **Q8 — Corpus growth quantum.** Codex recommends growing toward ~150 sm_86
  cells before promoting new residual heads; the continuation plan's bar was
  ≥3 slug-disjoint members per factor family. Pick the binding number per
  family rather than a global count?

---

## 9. Verification Notes

- **[verified-this-session]**: `timing_cycle_predictor.py` MODEL_VERSION +
  same-slug substitution mechanics (lines ~53, ~444–532); the 4-slug/13-cell
  timing cohort composition (`timing_cycle_predictions_sm_86.json`); the
  `timing_closure_sm_86.json` publication-gate scores and interval-only
  deploy-ns posture; codex-cited repo paths all exist (`tools/cache_predictor/
  l1_walker.py`, `tools/hrd_profiler/reuse_distance_tracker.py`,
  `tools/sector_stream/`, `data/local_labels/timing_predictions/`).
- Repo gate/robustness/falsification numbers were extracted by the
  repo-state analysts directly from the named artifacts with quoted values;
  the dossier reproduces them exactly as quoted. Paper numbers in §4 come
  from PDF deep-reads of the local corpus (with section/table citations
  recorded in the workflow results) and from the cited URLs.
- Known tension to re-verify during planning: the L1 closure geomean is
  reported as 0.89 by `l1_closure_comparison_sm_86.json` while the
  deployable friendly-L1 headline is 1.947 in the shadow/diagnostic sidecar —
  different cohorts/buckets (30-cell closure vs friendly-bucket headline);
  both are MISSES but they are different denominators. Do not conflate.

---

## 10. Knowledge Consulted

- `.claude/knowledge/INDEX.md` — KB routing (measurement/blackwell cards
  + paper corpus skills).
- `.claude/knowledge/measurement/` — nvbit-injection-playbook.md (bounded
  instrumentation + A/B overhead discipline), cupti-activity-api.md (allowed
  CUPTI surface; separate-runs join), clock64-globaltimer-dvfs.md (%clock64
  semantics; covariates-not-controls), nonroot-cloud-contract.md,
  nvarchsim-vs-this-project.md (validation-discipline prior art).
- `.claude/knowledge/blackwell/` — rtx_pro_6000_blackwell_capabilities.md,
  blackwell-tuning.md, blackwell-compatibility.md, blackwell-tensor-perf.md
  (sm_120 occupancy ceilings, lineage split, no-NCU-derived-numbers rule).
- `references/papers/knowledge_base/` — selected_index.md,
  problem_playbook.md (paper routing + the "When The Playbook Is Wrong"
  escape clause), manifest.json focus curation, and the active/ PDF corpus
  (GPUMech, MDM, GCoM, Hong-Kim, Nugteren 2014, Maeda HRD 2017, SHALOM 2022,
  Mei&Chu 2015, Huerta 2025 ×2, Abdelkhalik 2022, Jarmusch 2025, NeuroScalar
  2025, XAPP 2015, Braun 2021, Wang 2019, PKA 2021, Photon 2023, Sieve 2023,
  Arafa 2019, SynPerf 2026, GCL-Sampler 2026).
- `plans/research/sm86_wallclock_prediction_research_dossier.md` — prior
  dossier (sibling-repo era; its "no duration head exists" claim is now
  superseded in this repo — see §2.2).
- `.humanize/rlcr/2026-06-04_01-48-48/goal-tracker.md` + round summaries —
  loop outcome and queued residual.

## 11. References (web)

- NeuSight: https://arxiv.org/abs/2407.13853 (ASPLOS 2025)
- SynPerf/PipeWeave: https://arxiv.org/abs/2601.14910 (2026)
- Microbenchmark-driven analytical modeling (Blackwell stages): https://arxiv.org/abs/2605.04178 (2026)
- Huerta et al., Analyzing Modern NVIDIA GPU Cores: https://arxiv.org/abs/2503.20481 (MICRO'25 extension: dl.acm.org/doi/full/10.1145/3725843.3756041)
- HyFiSS: https://dl.acm.org/doi/10.1109/MICRO61859.2024.00022 (MICRO 2024)
- GCStack/GCScaler: https://dl.acm.org/doi/10.1145/3695053.3731068 (ISCA 2025)
- AMALI: https://dl.acm.org/doi/10.1145/3695053.3731064 (ISCA 2025)
- PPT-GPU-Mem: https://dl.acm.org/doi/10.1145/3392717.3392761 (ICS 2020)
- MLCRP: IEEE Access 2025 (Yonsei DTL)
- Buelow et al., Fine-Grained Memory Profiling: https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14671 (CGF 2022)
- Accel-Sim NVBit tracer: https://github.com/accel-sim/accel-sim-framework (ISCA 2020)
- DSO: https://arxiv.org/html/2407.13096v1 (2024)
- Wang/Wu DVFS models: https://arxiv.org/abs/1701.05308; https://www.mdpi.com/2079-3197/8/2/37
- HELP: https://arxiv.org/abs/2106.08630 (NeurIPS 2021); Multi-Predict: https://arxiv.org/pdf/2306.02459 (AutoML 2023)
- CDMPP: https://arxiv.org/abs/2311.09690 (EuroSys 2024)
- cuThermo: https://arxiv.org/abs/2507.18729 (2025)
- Omniwise (contrast case): https://arxiv.org/abs/2506.20886 (2025)
- Penrose (contrast case, counter-dependent): https://arxiv.org/html/2509.21762v1 (2025)
- Habitat: https://arxiv.org/abs/2102.00527 (ATC 2021)
- Locality quantitative study: https://link.springer.com/article/10.1007/s10766-022-00729-2 (2022)

--- Original Design Draft End ---
