# NCU-Replacement Continuation: Refit, T1 Count Lockstep, and Per-Metric Evidence Closure (sm_86)

## Goal Description

Continuation of `plans/sm86_multimetric_ncu_replacement_plan.md` after its 42-round
RLCR loop (2026-06-02_16-29-10) ended at max iterations with the corpus/ontology
spine complete but the predictive validation unstarted. That loop landed: the
41-metric ontology + composition DAG (AC-1 met), the T0 static occupancy
calculator with exact-agreement evidence (AC-2 met), a 45-admitted/63-attempted
cohort with a sealed slug-disjoint cal-29/val-8/holdout-8 split (AC-5 met), and
a factorial corpus where EVERY op-class metric with any sm_86 coverage clears
the ≥3 slug-disjoint bar (14 of 19 load-bearing metrics eligible; only
texture + 4 surface op-classes remain uncovered). What it did NOT do: refit any
predictor over the expanded corpus (the v5 heads still describe the
pre-expansion 18-cell sm_86 / 30-cell sm_120 cohorts and still FAIL both
robustness gates — LOOCV gap 0.441 vs ≤0.20, LOSO 0.814 vs ≤0.25 on sm_86;
1.805 on sm_120), build any T1 replay-count producer or NCU lockstep validator
(task7 absent; ZERO of the 15 T1 count metrics are even CAPTURED as NCU labels),
model any T2 cross-level traffic stream, or emit the per-(cell, metric_key)
evidence harness. The claim-to-evidence map accordingly holds every metric at
`not_submission_grade`.

This continuation turns the corpus into the claim. Priority order is chosen by
information-per-round:

1. **The pivotal unexecuted experiment first** — fold the
   `PENDING_REFIT_FACTORIAL_SLUGS` + register-cap variants + recapture cells
   into a controlled head refit and re-run the robustness gates. The entire
   28-round corpus campaign was justified as "the structural fix for LOOCV
   0.441 / LOSO 0.814"; one refit round tests that hypothesis.
2. **The load-bearing headline second** — T1 replay counts are the plan's own
   "non-root NCU count-replacement" main claim, are in-principle EXACTLY
   reconstructable from NVBit traces (per the measurement KB: per-instruction
   addresses, active masks, deterministic 32-byte sectoring), and are the one
   tier where "exact or near-exact" is achievable rather than modeled. Requires
   widening the NCU label capture to the count metrics first (today the labels
   carry ONLY the two sector hit rates + two timing counters).
3. **Close the structural skeleton** — texture/surface disposition, T2 traffic
   labels/models, T3 compositions — then the per-metric evidence harness and
   the final submission-grade verdict per metric.

The replacement thesis is per-tier and honest: T1 counts exact-or-typed,
T2 hit-rates/traffic modeled with slug-disjoint robustness, T3 timing
DVFS-bounded with covariates, T0 static exact. A metric that cannot be labeled
stably (NCU-replay variance), probed (no canonical identifier on this card), or
generated (no sm_86 workload) is a TYPED, fail-closed exclusion — never a
silent gap.

## Acceptance Criteria

> Numeric thresholds below inherit the parent plan's operator decision DEC-7:
> they are DIRECTIONAL TARGETS reported achieved-vs-target with CIs; the
> structural/integrity gates (typed-unavailable accounting, slug-disjoint
> splits, sealed holdout touched once, no label leakage into features, schema
> validity, compliance Rules 1-5) remain HARD.

- AC-1: A controlled predictor-head refit folds the expanded corpus into the
  fitted T2 heads and re-evaluates accuracy + robustness on the sealed split.
  - Positive Tests:
    - The refit cohort is derived from `is_base_predictor_fit_cell` AFTER
      removing the transitional exclusions: the 11 base factorial cells leave
      `PENDING_REFIT_FACTORIAL_SLUGS`, the 4 register-cap variants and the 10
      recapture cells are folded (the 4 recapture cells lacking sm_86 traces
      are live re-traced first), so the v5/L1 fit cohort grows from 18 to the
      full admitted base set (~35+ cells across ≥26 slugs).
    - Fitting uses calibration+validation only; the sealed holdout
      (`sealed_holdout_labels.json`) is evaluated ONCE after gates pass; the
      artifact records the model/source commit.
    - The refit emits the same gate artifacts (accuracy gates, baseline
      comparison, robustness, falsification, reliability) over the NEW cohort,
      reporting achieved-vs-target for: median abs ≤ 0.06, geomean rel ≤ 0.15,
      holdout geomean ≤ 0.25, LOOCV-vs-holdout gap ≤ 0.20, LOSO max ≤ 0.25.
    - The L1 head's provenance blockage is repaired before the refit is scored:
      the confirmed `_PC_RE` classifier regex bug and the secondary walker
      intra-block-reuse gap (both root-caused in
      `carveout_l1_head_diagnostic.json`) land together, lifting L1 scalar
      coverage from 6/20 toward the ≥70% floor and fixing the
      carveout_sensitivity_probe L1 catastrophic miss (pred 0.0013 vs truth
      0.9751).
    - Per-slug correction factors that are set EQUAL to NCU truth (the
      l1/l2_reduction subfamily factors) are either re-derived from
      slug-disjoint calibration data or demoted to diagnostic with a typed
      flag; no NCU-truth-fitted constant ships as a deployable prediction.
  - Negative Tests:
    - A refit that re-tunes after a holdout miss is rejected (demote, don't
      retune).
    - A refit fit cohort that still silently excludes an admitted base cell
      without a typed reason is rejected.
    - Robustness numbers copied forward from the 18-cell head are rejected;
      the gates must be recomputed on the refit cohort.

- AC-2: The 15 T1 replay-count metrics are labeled (or typed-unavailable with
  probe evidence) across the admitted corpus.
  - Positive Tests:
    - The NCU label capture path (`tools/local_labels/` only, quarantine
      preserved) is widened to collect the 8 probe-confirmed T1 counters
      (`smsp__inst_executed.sum`, global op ld/st/atom/red requests, global
      ld/st sectors, `l1tex__t_requests.sum`) per admitted cell, with the same
      n_repeats ≥ 5 + per-metric repeatability statistic and additive label
      schema (no breaking bump).
    - The 7 probe_pending T1 identifiers (local ld/st, texture, 4×surface) are
      resolved by `probe_ncu_metric_availability.py` on the 3090 Ti: each
      becomes `available` (and is then captured) or a fail-closed typed
      `sm_86_probe_*` exclusion recorded in the registry + inventory in
      lockstep.
    - Count labels are integer-stable across repeats (exact-replay metrics);
      a counter whose repeats disagree is typed `label_count_unstable`, not
      averaged silently.
  - Negative Tests:
    - A count label captured outside `data/local_labels/` or by a driver
      outside `tools/local_labels/` is rejected (Rule 1).
    - A registry row flipped to `available` without a committed probe artifact
      is rejected.

- AC-3: An NVBit replay-count producer + NCU lockstep validator delivers the T1
  count-replacement claim (parent AC-3/task7).
  - Positive Tests:
    - A producer derives per-cell predicted counts from the NVBit trace
      artifacts (instruction records, active masks, per-op classification,
      32-byte sector formation) for: `smsp__inst_executed.sum`, global op
      {ld,st,atom,red} requests, global {ld,st} sectors, local {ld,st} requests
      (now generable via the 3-slug local-memory family), and
      `l1tex__t_requests.sum`.
    - A lockstep validator joins predictions to the AC-2 labels by the closed
      cell identity and reports p50 ≤ 1%, p90 ≤ 3%, max ≤ 5% relative count
      error (directional), with a minimum-denominator floor typing low-support
      cells `low_support` rather than scoring near-zero denominators.
    - Measured per-op-class coverage is recorded across active-mask,
      predication, vector-width (32b/64b/128b), alignment/cache-op,
      ATOM-vs-RED, and local-memory cases — using the factorial families built
      for exactly this purpose — and
      `emit_ac4_load_bearing_coverage.py` gains the measured-count sub-measure
      alongside the contract-signature basis.
    - Unsupported op classes route through the existing `kUnspecified` →
      `unsupported_memory_path` fail-closed fallback and are counted in a typed
      coverage denominator.
    - NVBit slowdown stays within the existing trace budgets; cells whose
      bounded/sampled trace cannot support an exact count are typed
      `trace_support_insufficient`, never silently extrapolated.
  - Negative Tests:
    - An ATOM counted as RED (or vice versa) surviving validation is rejected.
    - A replay count exceeding its gate without typed demotion is rejected.
    - A lockstep join on anything weaker than the closed identity tuple
      (binary_sha256 + kernel_signature + shape) is rejected.

- AC-4: task6 closes: texture/surface get a final disposition and the corpus
  registration surface is fully consistent.
  - Positive Tests:
    - EITHER texture (TEX/TLD) and surface (SULD/SUST/SURED/SUATOM) factorial
      families land end-to-end (offline → live admission → materialization,
      the established 3-stage pattern), OR a committed fail-closed
      typed-unavailable disposition artifact + tests records why sm_86
      generation is not reliable (probe evidence for the 5 identifiers +
      SASS-generation evidence), and the 5 coverage rows stay
      `typed_unavailable_no_coverage` with that disposition referenced.
    - The queued registration-surface debts are cleared: stale
      `submission_bundle_readiness_report.json` regenerated over the current
      corpus; stale `register_spill_local_mem.toml` launch geometry corrected
      under the existing recurrence-guard pattern; the timing-exclusion
      binaries rebuilt/re-pinned (15 absent + 7 sha256-mismatch) so the timing
      label surface can grow.
  - Negative Tests:
    - A texture/surface family counted toward coverage without
      repeatability-admitted NCU labels is rejected.
    - A disposition artifact that asserts unavailability without probe/SASS
      evidence is rejected.

- AC-5: The T2 cross-level traffic streams are labeled and modeled (or typed).
  - Positive Tests:
    - The 5 available cache_stream counters (`dram__sectors_{read,write}.sum`,
      `lts__t_requests_srcunit_tex.sum`, `l1tex__m_xbar2l1tex_read_sectors…`,
      `l1tex__m_l1tex2xbar_write_sectors…`) are captured as labels on the
      admitted corpus (same AC-2 capture mechanics).
    - A model-derived predictor (miss-stream composition from the existing
      cache model + sector streams) reports p50 ≤ 10%, p90 ≤ 25% relative
      count error (directional) with denominator floors; a traffic primitive
      missing its gate is reported `diagnostic` and bars promotion of any
      hit-rate derived from it.
    - The 2 excluded DRAM op-split aliases stay fail-closed typed rows.
    - Hit-rate reporting moves to the parent plan's absolute-pp framing
      (p50 ≤ 5pp, p90 ≤ 12pp with low/mid/high buckets) in the refit artifacts.
  - Negative Tests:
    - A traffic count whose label was never captured cannot be scored
      (validator fails closed, no vacuous pass).

- AC-6: T3 closes to its honest DVFS-bounded form.
  - Positive Tests:
    - The timing-label surface grows over the rebuilt binaries (AC-4) and the
      cycles/duration heads are re-scored over the wider cohort with the
      parent gates (holdout geomean ≤ 0.15 log-relative, bootstrap 95% upper
      ≤ 0.25, repeatability CV ≤ 5%), keeping the existing publication-pass
      13-cell result as the floor; NVML covariates recorded per cell.
    - `mem_clock_hz_covariate` and `dram_bytes_per_second` identifiers are
      probe-resolved (or typed); the 3 `derived_pct_peak` rows either gain
      per-subsystem peak constants (provenance-tracked, into
      `device_constants.json`) for the exact speed-of-light composition, or
      their DRAM-proxy composition is explicitly retained with a typed
      `proxy_composition` status — no silent approximation presented as exact.
    - The composition DAG evaluates every derived row end-to-end from
      delivered primitives on at least the holdout cells (IPC, dram bytes/s,
      pct-of-peak), with availability cascading for missing primitives.
  - Negative Tests:
    - A pct-of-peak number published from the DRAM proxy without the typed
      proxy status is rejected.
    - A timing claim from a cell with a stale binary sha256 is rejected.

- AC-7: A per-(cell, metric_key) evidence harness produces the per-metric
  submission verdict (parent AC-8) and the system passes final review.
  - Positive Tests:
    - `publication_gate` / `comparison_runner` generalize to per-tier,
      per-metric `(cell, metric_key)` scoring across all 41 registry rows;
      every row lands in exactly one bucket {main, appendix, diagnostic,
      typed_unavailable} with achieved-vs-target numbers and CIs.
    - The baseline panel runs on the sealed holdout once: always-hit/miss,
      family-mean, uncalibrated replay, uncalibrated reuse-distance, prior
      v4/v5, static roofline timing, CUPTI/CUDA-event-only timing,
      NVML-prior-only.
    - Ablations cover: no active-mask, no cache-op, no warmup-state, no
      atomics-contention, no high-footprint correction, no timing-residual.
    - `claim_to_evidence_map.md` and the readiness report are regenerated from
      the harness outputs; the typed-unavailable budget accounting (T0=0,
      T1=0, T2 ≤ 10%, T3 explicit diagnostic budget) is enforced per DEC-4.
    - task12-style independent Codex review of statistical claims, baseline
      fairness, gate calibration, and holdout leakage passes; the AC-9
      compliance surface (static guards, plan-token, quarantine, legacy-path
      compatibility, sm_120 deferred discipline) is re-verified over the
      completed system.
  - Negative Tests:
    - A metric promoted to `main` whose robustness or typed-budget gate failed
      is rejected.
    - A harness that scores a metric without slug-disjoint validation evidence
      is rejected.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Everything above on sm_86 only, plus the housekeeping queue. No sm_120 live
work (DEC-6 stands until the physical move); no new model families beyond the
existing walker/isotonic/ridge classes unless an AC-1 robustness failure
demands one (then propose, don't improvise); no simulator revival (Rule 5).

### Lower Bound (Minimum Acceptable Scope)
AC-1 (the refit + recomputed robustness verdict, whatever the numbers say) and
AC-2+AC-3 for the 8 probe-confirmed T1 counters over ≥ the contract-locked
factorial slugs. Everything else may demote to typed/diagnostic with honest
reporting.

### Allowed Choices
Capture widening may be one sweep or incremental per-metric sweeps; the T1
producer may start from the existing NVBit sector summaries or full traces;
texture/surface may resolve as families or disposition; the refit may keep the
isotonic head class or add the optional GAM/GBM candidates by installing the
missing dependencies (record the environment change).

## Dependencies and Sequence

### Milestones
- M1 (AC-1): Feature-fold + L1 provenance repair + controlled refit + gates.
  The pivotal experiment: does the expanded corpus fix LOOCV/LOSO?
- M2 (AC-2 → AC-3): T1 label widening + probe closure, then the replay-count
  producer + lockstep validator + measured per-op coverage. The headline.
- M3 (AC-4, AC-5): task6 disposition + binary rebuild/re-pin + T2 traffic
  labels/models.
- M4 (AC-6): T3 probe/composition closure + wider timing scoring.
- M5 (AC-7): Per-metric harness, baselines/ablations, claim-to-evidence regen,
  Codex stats review, final compliance.

## Task Breakdown

| Task | Description | AC | Tag | Depends |
|------|-------------|----|----|---------|
| task1 | Live re-trace the 4 sm_86-trace-absent recapture cells; run the feature pipeline over the 10 recapture cells; fold variants + factorial slugs (empty `PENDING_REFIT_FACTORIAL_SLUGS`) into `is_base_predictor_fit_cell` | AC-1 | coding | - |
| task2 | Land the `_PC_RE` classifier fix + walker intra-block-reuse fix together (per the committed diagnostic); regenerate L1 sass-provenance sidecars; lift L1 scalar coverage | AC-1 | coding | - |
| task3 | Controlled v5/L1 refit on cal+val; recompute accuracy/baseline/robustness/falsification/reliability artifacts; evaluate sealed holdout once; demote-or-rederive NCU-truth-fitted correction factors | AC-1 | coding | task1, task2 |
| task4 | Widen `tools/local_labels/` capture to the 8 confirmed T1 counters (additive schema + repeatability stat per counter); capture over the admitted corpus | AC-2 | coding | - |
| task5 | Probe the 7 pending T1 identifiers (+ mem_clock, dram_bytes, 3 pct-of-peak, 2 DRAM op-splits) on the 3090 Ti; flip registry+inventory in lockstep or type fail-closed | AC-2, AC-6 | coding | - |
| task6 | Build the NVBit replay-count producer (inst/requests/sectors per op-class from traces) | AC-3 | coding | task4 |
| task7 | Build the T1 lockstep validator (closed-identity join, p50/p90/max gates, low_support floors, fail-closed unsupported ops); add measured-count coverage to the AC-4 report | AC-3 | coding | task6 |
| task8 | Ask Codex to review the T1 count-error evidence + per-op coverage for observation-contract fidelity (counting units, replay semantics, ATOM-vs-RED, predication) | AC-3 | analyze | task7 |
| task9 | Texture/surface: probe + attempt SASS generation; land families end-to-end OR the fail-closed disposition artifact + tests | AC-4 | coding | task5 |
| task10 | Rebuild/re-pin the 22 infrastructural timing-exclusion binaries; regenerate the stale readiness report + fix stale register_spill geometry | AC-4 | coding | - |
| task11 | Capture the 5 T2 traffic-stream labels; build the miss-stream traffic predictor; score against p50≤10%/p90≤25% with floors; wire hit-rate pp-bucket reporting | AC-5 | coding | task4 |
| task12 | T3: re-score cycles/duration over the rebuilt timing cohort; resolve per-subsystem peaks or typed proxy status for pct-of-peak; end-to-end DAG evaluation on holdout cells | AC-6 | coding | task5, task10 |
| task13 | Generalize the harness to (cell, metric_key); run baselines + ablations on sealed holdout; regenerate claim-to-evidence map + readiness verdict + typed budgets | AC-7 | coding | task3, task7, task11, task12 |
| task14 | Ask Codex for the independent statistical/baseline/holdout-leakage review | AC-7 | analyze | task13 |
| task15 | Final compliance verification over the completed system (static guards, quarantine, plan-token, legacy paths, sm_120 deferred discipline) | AC-7 | coding | task13, task14 |

## Feasibility Hints and Suggestions

- The capture-widening (task4) is the critical path for everything T1/T2; do it
  with one additive label-schema change and one full-matrix sweep over the
  existing 35 admitted cells, reusing the repeatability aggregator's per-metric
  statistic machinery. Counts should be replay-deterministic; treat any
  non-integer-stable counter as a finding, not noise to average.
- The NVBit side already records `inst_count`, `observed_access_count`, and
  32-byte sector counts in the committed sector summaries; the producer's first
  increment can score `smsp__inst_executed.sum` + global ld/st requests/sectors
  on the 13 contract-locked factorial slugs before generalizing.
- Stability ground truth from the 42-round loop (carry into validator design):
  reuse-heavy cells admit at cache=none warm, streaming cells at cache=all
  cold, data-dependent-index cells only at warmup1/cache_all; the structurally
  unstable set (warm_cold_pair_relaunch, cp_async_shared_tile_copy,
  shared_bank_conflict_stride, and the uncapped register_spill) should be typed
  out of count-label scoring up front rather than retried.
- The refit's most likely failure mode is the family buckets: memory_stress was
  the weak bucket pre-expansion (2.8% residual reduction, worst ECE) and the
  expansion added mostly memory/atomics/local cells — re-derive the bucket map
  before fitting.
- Knowledge base cards to consult during implementation:
  `.claude/knowledge/measurement/nvbit-injection-playbook.md` (trace fidelity +
  slowdown budgets), `cupti-activity-api.md` (timing joins),
  `clock64-globaltimer-dvfs.md` (T3 covariates),
  `nonroot-cloud-contract.md` (allowed-surface contract).

## Compliance

CLAUDE.md Rules 1-5 bind every task: NCU only via `tools/local_labels/` into
`data/local_labels/` on the local 3090 Ti; no privileged CUPTI; no
driver-root operations; read-only NVML; no simulator revival. The plan-token
scanner, no-NCU/no-root static guard, and zero-net-new-failure discipline
continue per round.
