# sm_120 Cache-Only Trace-Replay Predictor For Counter-Free L2 Hit-Ratio Observability

## Scope and Framing (binding contract — read first)

This plan opens a **fresh, sm_120-develop-only RLCR loop** on the RTX 5060
Laptop GPU (sm_120, Blackwell consumer, GB206 die, CUDA 13.2). It **supersedes**
`plans/per_bucket_monotonic_calibration_counter_free_l2_observability_plan.md`
(the "v5" learned per-bucket monotonic calibration plan), which (a) plateaued on
accuracy in the current environment and (b) carried a deferred-but-present
cross-architecture evaluation surface that fired the mainline-drift circuit
breaker twice (v7 R2/3/4; v5 R46/47/48). The v5 plan and its 47-round narrative
are retained as **historical / seed reference only**; this plan does not advance,
amend, or depend on it.

### Rule-5 binding contract (what we build, and what we refuse to build)

The pivot is to build an **ultrafast, cache-only trace-replay hit-rate
predictor**: it consumes already-collected NVBit sector/address traces and emits
calibrated L2 (and diagnostic L1) cache-hit-ratio estimates with uncertainty. It
is a **cache hit-rate model**, the mechanistic sibling of the Nugteren-2014 RDH
baseline already in the repo (`tools/baseline_runners/nugteren2014_rdh_baseline.py`).

**This is NOT a GPU simulator and does NOT revive CLAUDE.md Rule 5's forbidden
direction.** The boundary is binding and tracked here:

- **Allowed (this plan):** NVBit dynamic address/sector trace → cache-only
  set-associative / reuse-distance replay → predicted L1/L2 hit ratio; offline
  replay of observed memory references; analytical RDH / stack-distance /
  set-associative cache modeling; calibration against quarantined local NCU
  labels in `data/local_labels/`; reporting "hit-rate prediction",
  "counter-free observability", "cache trace-replay predictor".
- **Forbidden (Rule 5, absolute):** reviving `nvasim`, Accel-Sim, GPGPU-Sim, or
  any simulator-validation plan; simulating instruction issue, warp scheduling,
  DRAM timing, SM pipelines, occupancy, interconnect, or full-kernel runtime;
  treating any simulator as ground truth; framing the contribution as a
  "validated GPU simulator". The README sentence "The novelty is not a GPU
  simulator" remains true under this plan — the novelty is **counter-free,
  non-root cache-hit-ratio observability**, and the replay engine is the *means*,
  not the headline.

The colloquial phrase "cache simulator" (operator framing, 2026-05-29) is
honored as a **cache-occupancy trace-replay model strictly scoped to hit-ratio**;
all tracked text, code identifiers, and round summaries use
"cache-only trace-replay predictor" / "cache hit-rate model" and never
"GPU simulator". (Codex independent review, 2026-05-29, concurred with this
scoping; see Claude-Codex Deliberation.)

### sm_120-only decoupling contract (cross-arch is OUT of this loop)

Modeled on the closed-clean precedent `plans/sm120_only_decoupled_framework_plan.md`:

- The active path runs on the RTX 5060 Laptop (sm_120) **only**. No new code
  path may filter `arch == "sm_86"` or pin `target_arch = "sm_86"`; no new
  evidence may join to or be SHA-coupled to an sm_86 artifact; no new test or
  scanner may require an sm_86 cell to exist for the framework to be valid.
- `Cross-arch-verified` is **removed from this plan's active evidence
  vocabulary**. The only in-scope statuses are `Offline-verified` and
  `Verified-live-sm_120`. **`Verified-live-sm_120` is SUFFICIENT for this
  plan's closure.**
- The 44 sm_86 cells in `data/local_labels/label_admitted.json` and the
  `data/sm_86_predictor_accuracy_recovery_v2/` corpus stay on disk as historical
  reference; the active sm_120 path neither consumes them nor regresses when
  they are absent (they are silently skipped as `predictor_bundle_absent`).
- CLAUDE.md "Two-Device Local Development Model" and its four evidence statuses
  remain **binding at the repository level** — this is per-session evidence
  DISCIPLINE (record device / driver / CUDA / NVBit build / observed clocks for
  every live claim; a round verified on one arch is not verified on the other),
  **not a loop-closure gate**. Cross-architecture transfer is a separate,
  sequential, multi-session follow-on (see Deferred Work).

### Anti-stall binding rule (encode in tracked text — it has fired twice)

The mainline-drift circuit breaker has now fired twice on cross-arch closure
gating (v7 R2/3/4; v5 R46/47/48). The corrective lesson previously lived only in
machine-local auto-memory that does **not** survive scp/git. It is therefore
pinned here as a binding contract clause:

> **A sm_120-only loop NEVER carries any sm_86 / same-74 / cross-arch-transfer
> criterion into its closure gates, and its engineering-closure escape hatch
> (the falsification report) MUST be reachable on the single-host sm_120 surface.
> `Verified-live-sm_120` is sufficient for closure. Two consecutive STALLED
> verdicts trigger an honest stand-down, NOT compensating broadening.**

---

## Goal Description

Build a non-root, **cache-only trace-replay predictor** that converts NVBit
sector streams into calibrated L1/L2 cache-hit-ratio estimates on Blackwell
(sm_120), replacing privileged NCU cache counters without reviving full GPU
simulation. The engine performs **sampled set-associative sector-cache replay**
(32 B sector granularity, 128 B line grouping) over the already-collected
sector/address traces of the existing 30 sm_120 NCU-labeled cells, selects its
unknown Blackwell-L2 parameters (slice count / hash, associativity, replacement
policy, effective capacity) from a small **pre-registered discrete grid** via
stratified cross-validation, and applies at most a tiny one-dimensional
monotone **residual correction** on top of the mechanistic prediction. The
replay predictor **subsumes** the uncalibrated stack-distance walker and the
clean-room Nugteren-2014 RDH baseline (both kept as ablations) and
**complements** the plateaued v5 learned head (kept as a baseline column). The
work lands within one bounded sm_120-develop-only RLCR loop; engineering closure
is decoupled from the research-success accuracy gates via a falsification-report
path that separates trace-fidelity, model-parameter, and residual-calibration
failure modes. Target venues: HPCA / ISCA / MICRO / ASPLOS 2027.

**Why the pivot.** The v5 learned per-bucket calibration head plateaued and is
unimprovable in the current `uv` environment: full-cohort median absolute
hit-rate error 0.0755 (gate ≤0.06) and holdout geomean relative error 0.301
(gate ≤0.25), with only `sklearn` isotonic-on-frozen-projection fittable
(`pygam` / `lightgbm` / `xgboost` unavailable). Separately, the clean-room
Nugteren RDH baseline currently degenerates toward always-hit on the 30-cell
aggregate (median absolute error ≈0.342 ≈ `always_hit`), which shows a
fully-associative reuse-distance model misses the conflict / slice / hash /
replacement realism that a set-associative replay engine captures. A mechanistic
replay model is the credible path to both **accuracy** (conflict/capacity
realism) and **speed** (O(N) replay, sampling), and it is a cleaner
"replace NCU" story than another within-arch learned calibrator.

## Acceptance Criteria

Following TDD philosophy, each criterion carries positive and negative tests for
deterministic verification. All accuracy/throughput numbers are measured on the
existing **30 sm_120 NCU-labeled cells** (4 family buckets: `tensor_compute`
N=9, `activation_norm` N=7, `memory_stress` N=7, `other` N=7).

- **AC-1: Scope and framing compliance.** The plan, code identifiers, artifact
  field names, and round summaries use "cache-only trace-replay predictor" /
  "cache hit-rate model"; no active "GPU simulator" framing; no `nvasim`,
  Accel-Sim, GPGPU-Sim, simulator-validation, or timing/architecture simulation
  path is introduced.
  - Positive: a scope-guard test asserts the new engine module's public docstring
    declares "cache-only … not a GPU simulator" and enumerates the
    forbidden-surface refusal; the static plan-token scanner
    (`tools/test_plan_token_scanner.py`) stays green on all new implementation
    files.
  - Negative: a synthetic engine docstring claiming "GPU timing simulator" fails
    the scope-guard test; a synthetic file modeling warp scheduling / DRAM
    latency fails review under this AC.

- **AC-2: L2 sector-cache replay engine.** A new engine (proposed
  `tools/cache_replay/sector_cache_replay.py`) consumes the existing
  `dynamic_trace.pb` / `sector_stream.jsonl` for an admitted cell and emits a
  per-cell L2 result carrying: predicted L2 hit ratio, a cold / conflict /
  capacity miss breakdown, the full cache-parameter provenance
  (capacity_bytes, sector_bytes=32, line_bytes=128, num_slices, slice_hash_id,
  associativity, replacement_policy), and an uncertainty interval. It reuses the
  device-property surface (`tools/hrd_profiler/{device_properties,cache_model_spec}.py`)
  for the knowable L2 capacity (RTX 5060 Laptop: `l2_cache_size_bytes = 33,554,432`
  = 32 MiB).
  - Positive: replaying every admitted sm_120 cell produces a schema-valid result
    with all provenance fields populated and `0 ≤ predicted_hit_ratio ≤ 1`; a
    fixed synthetic trace with a known reuse pattern yields the analytically
    expected hit ratio within 1e-6.
  - Negative: a result missing the miss-breakdown or any cache-parameter
    provenance field fails the result schema validator; a replay that emits a
    hit ratio outside [0,1] fails the range validator.

- **AC-3: Pre-registered Blackwell-L2 parameter grid (no overfitting).** Unknown
  parameters are selected from a fixed, pre-registered discrete grid totalling
  **≤ 75 candidate configurations** via a staged search (capacity-scale ×
  associativity × replacement first; slice-hash tuned second), recorded verbatim
  in provenance: capacity_scale ∈ {0.75, 1.0, 1.25}; associativity ∈ {8, 16, 32};
  replacement ∈ {LRU, SRRIP, BIP}; slice-hash ∈ a fixed set of 4–8 bit-mixing
  candidates; prefetch ∈ {off, next-sector/next-line} (next-* admitted only if a
  specific microbench supports it). Selection is by stratified CV on the primary
  metric; the simplest config within 2% of best wins; then at most a
  one-dimensional monotone residual correction is fit. No free-form ML model is
  fit over the 30 cells.
  - Positive: the grid is enumerated in code as a frozen constant of ≤75 entries;
    the selected config + the simpler-within-2% tie-break + the residual
    correction degree are recorded verbatim in provenance; a CV unit test on
    fixture data converges to the registered selection rule.
  - Negative: a grid expansion beyond 75 configs fails the grid-size guard; a
    selection that ignores the simpler-within-2% tie-break fails the
    registered-protocol regression test; a residual head with >1 free dimension
    fails the provenance validator.

- **AC-4: Accuracy gate vs the 30 sm_120 NCU labels.** The replay predictor
  (with residual correction) meets, on the 30-cell cohort:
  full-cohort median absolute hit-rate error ≤ **0.055**; full-cohort geomean
  relative error ≤ **0.12**; holdout geomean relative error ≤ **0.25** (7
  holdout cells); per family-bucket p90 absolute error ≤ **0.15** for every
  bucket with N ≥ 7. Metrics are unweighted (the gate); weighted variants are
  advisory.
  - Positive: the comparison bundle reports each metric at or below its threshold;
    a gate scanner enumerates and confirms all four.
  - Negative: a synthetic predictor at median absolute error 0.10 fails the gate
    scanner; a synthetic bucket (N=8, p90=0.20) fails the per-bucket scanner.
  - Baseline reference (to beat): v5 learned head sits at median abs 0.0755 /
    holdout geomean 0.301; the gate is the genuine, reachable mainline accuracy
    work this loop exists to do.

- **AC-5: Subsumes baselines (head-to-head on the same 30 cells).** The replay
  predictor beats the v5 learned head on ≥ **20/30** cells and the Nugteren-2014
  RDH baseline on ≥ **24/30** cells (median absolute hit-rate error per cell);
  and its aggregate median absolute error is strictly lower than every baseline:
  v5, Nugteren RDH, uncalibrated stack-distance walker, v4 PlattCalibrator,
  cohort_mean (0.6), always_hit (1.0).
  - Positive: the comparison table has all baseline columns populated for every
    admitted cell; `replay_vs_v5_beat_fraction ≥ 0.667`,
    `replay_vs_nugteren_beat_fraction ≥ 0.80`; replay aggregate < each baseline
    aggregate.
  - Negative: a baseline column missing for any cell fails the schema validator;
    a synthetic case where replay aggregate ≥ any baseline fails the scanner.

- **AC-6: Ultrafast (measurable throughput).** The replay path throughput is
  measured and falsifiable: an optimized replay processes ≥ **100 M** sector
  accesses/sec, OR (if the C++/extension replay is deferred to a follow-on)
  the Python replay is ≥ **10×** the current `tools/hrd_profiler/reuse_distance_tracker.py`
  Fenwick stack-distance throughput on the same traces. A benchmark JSON records
  sectors/sec, wall time, input bytes/sec, CPU model, and source_commit. The NVBit
  trace-collection time is reported **separately** (the "ultrafast" claim is
  scoped to replay, not to NVBit tracing, which may not beat NCU wall time;
  see AC-7 for the sampled-trace mitigation).
  - Positive: the benchmark JSON is emitted, schema-valid, and meets one of the
    two throughput bars; a regression test pins the bar.
  - Negative: a benchmark JSON below both bars fails the throughput gate; a
    benchmark conflating NVBit trace time into the replay rate fails the
    separation check.

- **AC-7: Sampled replay fidelity.** A sampled replay mode (set sampling 1/32 to
  1/128, extrapolated with Wilson/bootstrap CI) processing ≤ **10%** of sector
  records differs from full replay by ≤ **0.02** absolute hit ratio on ≥ **27/30**
  cells AND preserves every AC-4 pass/fail verdict.
  - Positive: a sampled-vs-full comparison report shows ≤0.02 delta on ≥27/30
    cells and identical AC-4 verdicts.
  - Negative: a sampling mode that flips any AC-4 verdict, or exceeds 0.02 delta
    on >3 cells, fails the sampling-fidelity gate.

- **AC-8: Validation + robustness (advisory).** A deterministic stratified split,
  LOOCV, leave-one-slug-out (LOSO), and a 2000-resample bootstrap CI are emitted.
  The selected config must be stable under removal of any single cell, or be
  flagged. **Per the sm_120-only scope, the robustness thresholds (LOOCV-vs-holdout
  gap ≤ 0.20; per-slug LOSO headline change ≤ 0.25) are ADVISORY at N=30**
  (small-N variance, not a model defect) — they surface in the falsification
  report but do not gate closure.
  - Positive: the validation artifact carries split assignments, LOOCV / LOSO /
    bootstrap-CI values, and a per-cell stability flag; schema-valid.
  - Negative: a missing bootstrap-CI block or an unflagged unstable config fails
    the validation-artifact schema validator.

- **AC-9: Falsification arm + engineering-closure decoupling.** The loop closes
  successfully if EITHER (i) AC-4 + AC-5 + AC-6 all pass (research success) OR
  (ii) a falsification report at
  `data/local_labels/cache_replay_falsification.json` is emitted that documents
  which gates were missed and **separates the failure into trace-fidelity vs
  model-parameter vs residual-calibration modes**, with a confirmed/refuted
  verdict on the pre-registered hypothesis (set-associative conflict/slice
  realism reduces median absolute error vs the fully-associative Nugteren RDH by
  ≥ 30% confirmed / < 10% refuted). Both disjuncts are **terminal on the
  sm_120 host** — no cross-arch / same-74 precondition anywhere.
  - Positive: the falsification report is present, schema-valid, and either AC-4+5+6
    pass or the report carries `joint_contribution_pct` ≥30 (confirmed) or <10
    (refuted) plus the three-way failure-mode decomposition.
  - Negative: a falsification report that gates on any "final cross-arch / same-74"
    artifact, or omits the failure-mode decomposition, fails the report validator.

- **AC-10: Compliance + sm_120-only claim guard.** The repo-wide static guard
  (`workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`) passes; no
  NCU outside `tools/local_labels/` + `data/local_labels/`; no driver-root, no
  privileged CUPTI, no simulator-revival path. Additionally, a claim-guard AC
  (modeled on `sm120_l2_observability_30_cell_cohort_bounded_loop_plan.md`)
  asserts the active comparison bundle's `top_level_admission_scope ==
  "target_arch=sm_120"` and `arches == ["sm_120"]`, and that round summaries
  contain no `sm_86` / `Cross-arch-verified` claims — enforced by the live
  `tools/local_labels/sm120_loop_claim_guard.py`.
  - Positive: static guard scans clean; the claim guard admits the sm_120-only
    bundle and rejects a synthetic `cross_arch_require_both_arches` scope.
  - Negative: a synthetic file referencing `nvidia-smi --lock-gpu-clocks` outside
    quarantine fails the static guard; a synthetic round summary claiming
    `Verified-live-sm_86` fails the claim guard.

- **AC-11: L1 diagnostic-only.** L1 hit-ratio is emitted as a diagnostic surface
  only (it requires per-SM/CTA scheduling assumptions and `.ca`/`.cg`/bypass
  semantics). L1 is promoted to a load-bearing claim ONLY if it independently
  reaches median absolute error ≤ 0.10 with explicit L1 eligibility / bypass
  provenance — otherwise it stays diagnostic and does not gate closure.
  - Positive: the L1 diagnostic artifact is schema-valid and clearly marked
    `load_bearing=false` unless the ≤0.10 + provenance bar is met.
  - Negative: an L1 result silently entering the headline geomean without meeting
    the promotion bar fails the load-bearing-scope guard.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Implement the `tools/cache_replay/` set-associative sector-cache replay engine
(O(N) headline path + sampled mode + Wilson/bootstrap CI), the pre-registered
≤75-config Blackwell-L2 parameter grid with staged stratified-CV selection, the
≤1-D monotone residual calibrator, the full 6+-column comparison table
(replay, v5, Nugteren RDH, uncalibrated walker, v4 Platt, cohort_mean,
always_hit), the throughput benchmark harness, the validation + advisory
robustness artifacts, the falsification report, and the L1 diagnostic surface.
All positive + negative test fixtures land alongside. An optional C++/Python-
extension replay core is in the upper bound only if Python cannot meet AC-6's
≥10× bar; otherwise it is deferred.

### Lower Bound (Minimum Acceptable Scope)
Land AC-1 (scope/framing), AC-2 (L2 replay engine consuming existing traces),
AC-3 (pre-registered grid + selection), AC-10 (compliance + claim guard), and
EITHER AC-4+AC-5+AC-6 (research success) OR AC-9's falsification report
(engineering closure). The sampled mode (AC-7), L1 diagnostic (AC-11), and the
C++ optimization may be deferred to the falsification report or a follow-on if a
fast convergence path is required, provided each deferral is documented in the
closure summary.

### Allowed Choices
- **Can use:** the existing `dynamic_trace.pb` / `sector_stream.jsonl` traces;
  `tools/sector_stream/{generator,consume_trace}.py`;
  `tools/hrd_profiler/{device_properties,cache_model_spec,reuse_distance_tracker}.py`;
  `tools/baseline_runners/nugteren2014_rdh_baseline.py`; `numpy`; `sklearn`
  isotonic for the ≤1-D residual; CUDA device attributes / read-only NVML for the
  knowable L2 capacity; static SASS cache-operator hints; a C++ / Python-extension
  replay core if needed for AC-6.
- **Cannot use:** GPU-driver-root operations; privileged CUPTI surfaces; NCU
  outside `data/local_labels/` quarantine; simulator revival (no `nvasim` /
  Accel-Sim / GPGPU-Sim / timing-architecture simulation per Rule 5); cloud GPU
  access; new external benchmark-suite admission (Rodinia / Parboil / CUTLASS /
  FlashAttention) this loop; **any cross-arch transfer evaluation, sm_86 cohort
  fit, or same-74 joint fit**; a free-form ML model fit over the 30 cells (the
  residual is ≤1-D monotone only).

> **Note on Deterministic Designs.** The parameter grid, staged selection rule,
> simpler-within-2% tie-break, residual-correction degree, sector/line
> granularity, and falsification hypothesis are pre-registered with fixed
> values. The cohort is fixed at the existing 30 sm_120 cells.

## Feasibility Hints and Suggestions

### Conceptual Approach (Codex-reviewed, 2026-05-29)
1. **Primary engine:** sampled set-associative **sector-cache replay** over the
   NVBit sector stream — 32 B sectors grouped into 128 B lines; per-set LRU/SRRIP/BIP
   stacks; candidate slice-hash → set-index mapping. O(N) per access on the
   headline path (NOT O(N log N) Fenwick stack distance, which is reserved for
   the walker ablation).
2. **Calibration:** select the unknown Blackwell-L2 parameters from the
   pre-registered ≤75-config grid by stratified CV; then fit at most a ≤1-D
   monotone residual `label − replay_prediction` over 3–5 interpretable features.
   Do NOT fit a large parameter space over 30 labels.
3. **Speed:** compact integer sector IDs, a streaming parser (no per-access
   Python dict), set sampling (1/32–1/128) with Wilson/bootstrap CI extrapolation,
   and parallel partitioning by launch and sampled set. A C++/extension core if
   pure Python cannot hit ≥10× the current Fenwick path.
4. **L1:** keep diagnostic this loop; it needs per-SM/CTA scheduling and bypass
   semantics that are higher-risk than the L2 surface.

### Known Risks (and mitigations)
- **"Ultrafast" overclaim:** full NVBit address tracing can dominate runtime, so
  a replay-only speed win is insufficient on its own → report trace time
  separately (AC-6) and add the sampled-trace/replay mode (AC-7); scope the speed
  claim to replay.
- **Overfitting 30 labels with Blackwell unknowns:** → tiny pre-registered grid +
  simpler-within-2% tie-break + ≤1-D residual (AC-3).
- **Trace fidelity:** coalescing, sector expansion, unsupported-Blackwell-memory
  paths, prefetch, compression, ECC, L1/TLB effects → enumerated as the
  falsification report's trace-fidelity failure mode (AC-9); `num_l2_slices`
  currently falls back to 1 (`source=constant`) and MUST be calibrated, not
  assumed, for a real Blackwell claim.
- **Replacement realism:** do not assume pure LRU; the grid tests LRU/SRRIP/BIP.

### Relevant References
- `tools/baseline_runners/nugteren2014_rdh_baseline.py` — the RDH cache model
  (category precedent: a cache model that explicitly is not a simulator).
- `tools/hrd_profiler/reuse_distance_tracker.py` — existing stack-distance engine
  (the AC-6 throughput baseline + the walker ablation).
- `workloads/simbench_bw/schemas/sector_stream.schema.json` — sector-stream wire
  format the engine consumes.
- `references/papers/` — RDH / stack-distance / sampled-cache-simulation prior art
  (prior art only, per MANIFEST: not the project direction).

## Dependencies and Sequence

### Milestones
- **M0 — Fresh-loop carryover triage (Round 0).** Per the supersession of the v5
  plan: `@unittest.skip` (NOT delete — operator chose skip+keep-dormant) the
  cross-arch test surfaces (`test_require_both_arches_for_cross_arch.py` already
  skipped; the `Same74Readiness*` suite; the 74-pin / cross_arch_transfer tests
  in `test_v5_cohort_coverage_scanner.py`) with a reason pointing at this plan;
  keep the sm_120-only enforcement tests green; confirm the v5 sm_86 emitters
  stay dormant (no new churn). Update `HANDOFF.md` + `MANIFEST.md` to name this
  plan as the active sm_120-develop-only direction and demote the stale
  M1/M2/3090-Ti framing to historical. Data-staging audit: confirm the 30 sm_120
  cells' `sector_stream.jsonl` / `dynamic_trace.pb` survived the session.
- **M1 — L2 replay engine (AC-1, AC-2).** Build `tools/cache_replay/` engine +
  result schema + scope guard; replay all 30 cells.
- **M2 — Parameter-grid calibration (AC-3).** Pre-registered ≤75-config grid +
  staged stratified-CV selection + ≤1-D residual.
- **M3 — Accuracy + baselines (AC-4, AC-5).** Comparison table; meet or falsify
  the accuracy gates; head-to-head subsumption.
- **M4 — Throughput + sampling (AC-6, AC-7).** Benchmark harness; sampled-mode
  fidelity.
- **M5 — Validation + falsification + closure (AC-8, AC-9, AC-10, AC-11).**
  Validation/robustness artifacts; falsification report; compliance + claim
  guard; L1 diagnostic; closure summary.

### Sequence notes
M0 precedes all. M1 → M2 → M3 are the accuracy spine. M4 depends on M1 (engine).
M5 depends on M3 (accuracy result) + M4 (speed result). Engineering closure
(AC-9) is reachable after M3 even if AC-4/5/6 miss.

## Task Breakdown

- task0 [M0]: Round-0 carryover triage + HANDOFF.md/MANIFEST.md update + staging audit.
- task1 [M1]: `tools/cache_replay/sector_cache_replay.py` engine + `cache_replay_result.schema.json`.
- task2 [M1]: scope-guard test (AC-1) + result schema validator (AC-2) + synthetic-trace exactness test.
- task3 [M2]: frozen ≤75-config grid + staged stratified-CV selector + simpler-within-2% tie-break.
- task4 [M2]: ≤1-D monotone residual calibrator + provenance fields.
- task5 [M3]: comparison-table emitter (replay + 6 baselines) + per-cell beat-fraction + aggregate scanner.
- task6 [M3]: AC-4 accuracy gate scanner (positive + negative fixtures).
- task7 [M4]: throughput benchmark harness + benchmark JSON schema + ≥10× / ≥100M gate.
- task8 [M4]: sampled replay mode + Wilson/bootstrap CI + sampled-vs-full fidelity report.
- task9 [M5]: validation artifact (split/LOOCV/LOSO/bootstrap) + advisory robustness.
- task10 [M5]: falsification report emitter + three-way failure-mode decomposition.
- task11 [M5]: compliance (static guard) + claim-guard AC wiring to `sm120_loop_claim_guard.py`.
- task12 [M5]: L1 diagnostic surface (load_bearing=false unless promotion bar met).
- task13 [M5]: closure summary recording evidence status (`Verified-live-sm_120` or `Offline-verified`) + device/driver/CUDA/NVBit/clock metadata.

## Claude-Codex Deliberation

### Agreements
- The pivot object is a **cache-only trace-replay hit-rate predictor**, NOT a GPU
  simulator; this stays inside the README "replace NCU cache-hit-ratio
  measurement" goal and outside Rule 5. Tracked text uses
  "cache-only trace-replay predictor"; the Nugteren RDH baseline is the category
  precedent. (Codex concurred verbatim with the operator's Rule-5 read.)
- Architecture is a **hybrid**: mechanistic sampled set-associative sector replay
  + a tiny ≤1-D monotone residual. Pure StatStack/StatCache is NOT the mainline
  (the current failure mode needs conflict/slice/hash/replacement realism, shown
  by Nugteren's degeneration to ≈always-hit).
- The "ultrafast" claim is scoped to **replay** throughput (≥100M sectors/sec
  optimized, or ≥10× the current Fenwick path) with NVBit trace time reported
  separately and a sampled mode added; otherwise the speed claim is not
  defensible.
- Overfitting is prevented by a **pre-registered ≤75-config grid + simpler-within-2%
  tie-break + ≤1-D residual**, selected by stratified CV.

### Resolved Disagreements
- "Cache simulator" (operator phrasing) vs Rule 5 → resolved by scoping it as a
  cache-occupancy replay model strictly for hit-ratio and forbidding all
  architectural/timing simulation; the project's "novelty is not a simulator"
  framing is preserved.
- L1 as a load-bearing claim → resolved to **diagnostic-only** this loop
  (per-SM/CTA + bypass complexity); promotion bar gated at ≤0.10 median abs.
- Whether to delete the dormant cross-arch machinery → resolved (operator) to
  **skip + keep dormant**, not delete, to avoid churn during the pivot.

### Convergence Status
Converged on the cache-only replay direction, the hybrid architecture, the
falsifiable speed/accuracy ACs, and the Rule-5-respecting framing. Open items are
the user decisions below.

## Pending User Decisions

- **PUD-1 (Rule-5 stance):** This plan treats the cache-replay predictor as
  **compatible with** CLAUDE.md Rule 5 (a cache hit-rate model, not a GPU
  simulator). If you instead want to overturn the project's "novelty is not a
  simulator" stance, that requires an explicit CLAUDE.md Rule 5 + README
  amendment — flag it and it will be done separately. **Default: compatible (no
  CLAUDE.md/README change).**
- **PUD-2 (C++ now vs Python-first):** AC-6 admits either ≥100M sectors/sec
  (optimized C++/extension) or ≥10× the current Python Fenwick path. **Default:
  Python-first; C++ deferred unless the ≥10× bar is missed.**
- **PUD-3 (cross-arch follow-on plan):** the v5 plan's AC-4 cross-arch contract
  remains preserved in that (historical) plan. **Default: no separate follow-on
  plan authored now**; cross-arch transfer of the replay model is a future
  sequential, two-host loop (see Deferred Work).

## Deferred Work (Out of Scope This Loop)

- **Cross-arch transfer evaluation (sm_86 ↔ sm_120):** out of scope; a sequential,
  multi-session follow-on (develop+commit on sm_86 first, then sm_120, with
  cross-host scp per CLAUDE.md). The verbatim transfer-gate contract is preserved
  in the historical
  `plans/per_bucket_monotonic_calibration_counter_free_l2_observability_plan.md`
  (AC-4) until a dedicated follow-on plan is authored.
- **L1 as a load-bearing claim** (AC-11 promotion path).
- **C++/extension replay core** if Python meets AC-6's ≥10× bar.
- **External benchmark-suite admission** (Rodinia / Parboil / CUTLASS /
  FlashAttention).

## Implementation Notes

### Code Style Requirements
Match surrounding repo idiom: deterministic seeds (reuse `seed=20260527` family),
atomic JSON writes, strict draft-2020-12 schemas with positive + negative
fixtures, `source_commit` provenance on every emitted artifact, lazy
package-qualified imports with repo-root-on-`sys.path` for dual library/CLI
modules (`BL-20260529-repo-root-on-syspath-before-package-qualified-tools-import`),
and the two-commit landing cadence (Commit A = code/test, Commit B = re-run the
regen wrapper + lockstep restamp) where artifacts carry `source_commit`. Run all
Python via `uv run`.

### Architecture-Coupling Discipline (per CLAUDE.md — DISCIPLINE, not a closure gate)
Every live-evidence claim records device + driver + CUDA toolkit + NVBit build +
observed clocks / throttle state. A round verified live on one arch is NOT
verified on the other. The replay engine's wire format (result schema, cache-
parameter provenance) is architecture-independent; arch-specific extensions are
additive and capability-gated. **`Verified-live-sm_120` is sufficient for this
plan's closure; `Cross-arch-verified` is not a gate here.** The Blackwell L2
parameters (slices, hash, associativity, replacement, effective capacity) are
treated as **calibrated covariates, not invariants** — they are sm_120 + CUDA-13.2
facts, not contracts.

### Dependency Surface
`uv`-managed Python; `numpy`; `sklearn` (isotonic only, for the ≤1-D residual);
optional C++ toolchain (`g++-14` via `NVCC_CCBIN=g++-14`) for the replay core.
`pygam` / `lightgbm` / `xgboost` are unavailable in the environment and are NOT
required by this plan (the residual is ≤1-D monotone, not a free ML model).

### Binding compliance pointers
- Static guard: `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`.
- Plan-token scanner: `tools/test_plan_token_scanner.py` (implementation files;
  `plans/*.md` are exempt).
- sm_120-only claim guard: `tools/local_labels/sm120_loop_claim_guard.py`.
