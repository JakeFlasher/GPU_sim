# Tiered Metric-Ontology NCU-Replacement Predictor (sm_86, Submission-Grade Multi-Metric)

## Goal Description

Elevate the non-root GPU predictor from a single-layer cache/timing prototype to a **submission-grade, multi-metric NCU-replacement predictor** on sm_86 (RTX 3090 Ti) that covers the full target NCU metric set (~40 metrics, "if available") and is validated cell-by-cell against NCU ground-truth labels collected only via `tools/local_labels/` into `data/local_labels/` (CLAUDE.md Rule 1).

The organizing thesis is a **metric ontology + composition DAG**: the ~40 metrics are not 40 independent prediction heads but a small set of *primitive observation problems* plus *suffix-derived normalizations*. Every metric is assigned a `metric_class` and a risk tier, and derived metrics are computed deterministically from primitives + provenance-tracked device constants:

- **T0 static** — launch/resource arithmetic (`launch__occupancy_limit_*`), computed by an occupancy calculator from `cuobjdump -res-usage` registers + static shared memory + block dim + device constants. Exact, no GPU run.
- **T1 replay_count** — instruction/request/sector counts (`smsp__inst_executed.sum`, `l1tex__t_requests_pipe_lsu_mem_global_op_{ld,st,atom,red}.sum`, `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum`, local op ld/st requests) from NVBit trace replay, **validated in lockstep against NCU observation-point semantics**. Deterministic, DVFS-free, but only "exact" after semantic validation.
- **T2 cache_stream** — cross-level sector traffic and hit-rates (`l1tex__t_sector_hit_rate.pct`, `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`, `lts__t_sector_hit_rate.pct`, `l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum`, `l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum`, `dram__sectors_op_{read,write}.sum`, `lts__t_requests_srcunit_tex.sum`, texture/surface requests) from a reuse-distance/miss-stream model. The hard modeling tier.
- **T3 time_clock / derived_pct_peak** — DVFS/scheduler-bound timing and throughput (`gpu__time_duration.sum`, `gpc__cycles_elapsed.{max,avg.per_second}`, `sm__cycles_active.avg`, `dram__cycles_elapsed.avg.per_second`, `sm__warps_active.avg.*`, `sm__inst_executed.avg.per_cycle_elapsed`, `dram__bytes.sum.per_second`, the three `*throughput.avg.pct_of_peak_sustained_elapsed`) from the timing model + read-only NVML covariates + composition.

Derived ratios/percentages (`hit_rate = 1 - miss/req`, `throughput_pct = traffic/time/device_peak`, `dram_bytes_per_s = sectors*32/time`, `ipc = inst/cycles`, achieved-warps from theoretical occupancy + scheduler correction) are computed by a composition engine and validated *separately* from their primitives so composition cannot hide denominator error. The arch-specific surface collapses to a single provenance-tracked device-constant table, so the predictor logic is identical across sm_86 (active) and sm_120 (deferred, design retained).

This plan **supersedes** the half-done predecessor `plans/sm86_submission_grade_evidence_expansion_plan.md` (which reached task6: 49 admitted cells / 35 unique slugs, but never refit heads or built a valid slug-disjoint split) by **importing that 49-cell cohort** and **regenerating the stale `cohort_split.json`**, while keeping the existing single-metric L1/L2 `comparison_runner` path working unchanged.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Numeric gates are stated per tier because count metrics (exact), rate metrics (modeled), and timing metrics (DVFS-bound) must not share one gate.

> **Operator decisions (resolved):** (1) The headline is a **tiered claim** — T0/T1 exact/near-exact counts are the load-bearing "non-root NCU count-replacement" result, T2 cache/traffic is load-bearing only if its gates are met, and **T3 timing/throughput is appendix/diagnostic** unless it clears its DVFS gates. (2) The per-tier **numeric error/robustness thresholds below are DIRECTIONAL TARGETS**, not hard binary cutoffs: the predictor must demonstrably improve toward them and every report must state achieved-vs-target with CIs, but a metric narrowly missing a numeric target is reported (and tier-placed) on its evidence rather than auto-failed. (3) **Structural/integrity gates remain HARD** regardless: T0 exact-agreement-or-typed-unavailable, coverage denominators, typed-unavailable budgets, slug-disjoint split with no leakage, no NCU-label leakage into features, schema validity, and compliance. Where a negative test below says a numeric miss is "rejected," read it as "barred from main-claim promotion and demoted to diagnostic with its achieved number reported," not "the loop fails."

- AC-1: A metric-ontology + composition-DAG registry covers the full target set with typed classes, NCU observation-point contracts, composition formulas, and per-arch availability.
  - Positive Tests (expected to PASS):
    - The registry (extending `tools/local_labels/metric_registry.json` + `metric_registry.py`) contains an entry for every target metric in the draft's list, each carrying a `metric_class` in {`static`,`replay_count`,`cache_stream`,`time_clock`,`derived_ratio`,`derived_pct_peak`,`typed_unavailable`}, a `risk_tier` in {T0,T1,T2,T3}, a `ncu_observation_contract` (counting unit, rollup, numerator/denominator, op classes, unsupported-op behavior), and `per_arch_availability` for sm_86 and sm_120.
    - Each `derived_ratio`/`derived_pct_peak` entry declares a `composition` block (`formula`, `dependencies`, required `device_constants`); a `CompositionDAGResolver` topologically sorts the DAG, evaluates formulas with a restricted (non-`eval`) evaluator, and reproduces a known derived value from its primitives within floating-point tolerance on a synthetic fixture.
    - Availability cascades: if a primitive is `typed_unavailable` on an arch, every dependent derived metric resolves to `typed_unavailable` with a dependency reason.
    - task1 emits an auditable target-metric inventory table (one row per target metric: `metric_class`, `risk_tier`, primitive-or-derived, `per_arch_availability`, and ground-truth source) so AC-1 coverage is checkable at a glance.
  - Negative Tests (expected to FAIL):
    - A registry with a metric lacking `metric_class`, `risk_tier`, or (for derived metrics) a `composition` block is rejected.
    - A cyclic composition DAG, an unknown formula variable, or a formula referencing a missing device constant is rejected with a typed reason (`composition_dag_cyclic`, `composition_unknown_symbol`, `device_constant_missing`).
    - A `label_only`/non-deployable primitive used as a deployable predictor input is rejected (no NCU-derived ground-truth leakage into features).

- AC-2: A provenance-tracked device-constant table and a static occupancy calculator deliver the T0 metrics exactly or as typed-unavailable.
  - Positive Tests:
    - A single arch-keyed device-constant table (consolidating `tools/hrd_profiler/device_properties.json`, `ga102_peak_constants.json`, `tools/cache_predictor/l1_static_card_table.json`) exposes sector size, regs/SM, smem/SM, max warps/SM, max blocks/SM, SM count, L1/L2 capacity, peak DRAM BW — each with mandatory `source`/`provenance`; missing constants fail closed.
    - An occupancy calculator computes `launch__occupancy_limit_{blocks,registers,shared_mem,warps}` from `cuobjdump -res-usage` registers + static/dynamic shared memory + block dim + device constants, and matches the NCU launch-section values exactly on every cell where NCU exposes them.
  - Negative Tests:
    - A device constant with empty provenance is rejected (cannot fabricate a peak).
    - An occupancy-limit prediction emitted without the required resource inputs, instead of a typed-unavailable, is rejected.
    - A hardcoded sm_86 constant used directly in predictor logic (bypassing the table) is rejected by a guard.

- AC-3: T1 replay counts are validated in lockstep against NCU observation-point semantics on dedicated microbenches.
  - Positive Tests:
    - For `smsp__inst_executed.sum`, `l1tex__t_requests_pipe_lsu_mem_global_op_{ld,st,atom,red}.sum`, `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum`, `l1tex__t_requests_pipe_lsu_mem_local_op_{ld,st}.sum`, and `l1tex__t_requests.sum`: trace-replay-derived counts vs NCU report p50 relative count error <= 1%, p90 <= 3%, max <= 5% across lockstep microbenches.
    - Each T1 metric is exercised across active-mask, predication, vector-width, alignment, cache-operator, ATOM-vs-RED, and (where applicable) local-memory cases, with explicit per-op-class coverage recorded.
    - Unsupported op classes route through the existing `kUnspecified`/`unsupported_memory_path` typed fallback (fail-closed), never a silent admission.
    - Lockstep microbench fixtures must exceed a minimum count/denominator floor; cells below the floor are typed `low_support` and excluded from the count-error gate rather than scored at a near-zero denominator.
  - Negative Tests:
    - A T1 metric admitted with a silent unsupported-op count is rejected.
    - A replay count that exceeds the per-metric error gate without a typed-unavailable demotion is rejected.
    - An ATOM-vs-RED misclassification (RED counted as ATOM or vice versa) that survives validation is rejected.

- AC-4: A factorial microbenchmark corpus supplies dedicated evaluatable cells per metric, with each phenomenon recurring across multiple slugs.
  - Positive Tests:
    - New/extended kernel families cover the under-covered surfaces: register-spill/local-memory (`-maxrregcount` + volatile-local + dynamic-index sweeps), paired ATOM-vs-RED (return-value-used vs ignored; bin-count/skew/Zipf sweeps), texture 1D/2D point-filter footprint sweeps (below-L1 / near-L2 / above-L2), reuse-distance hit-rate sweeps (working sets 0.25x-4x L1 and L2; warmup {0,1,5}; cache-control {all,none}), and request/sector-formation (alignment offsets, vector widths, partial masks, uncoalesced). Surface ops are included where reliably generable on sm_86, else typed-unavailable.
    - Each new family registers via the existing surface (`roster.toml`, `shapes/<slug>.toml`, `common.mk`, correctness refs) and is regression-locked by `-Xptxas=-v` register counts / SASS signatures via the kernel-contract verifier.
    - Every load-bearing metric has >= 3 slug-disjoint families exercising it (the structural fix for LOSO collapse).
    - Each new factorial cell has NCU ground-truth labels collected via `tools/local_labels/` (quarantine path) and repeatability-admitted through the existing aggregator (the same stddev <= 0.03 sector / log-relative timing gate), so it is an admitted evaluatable cell, not just a built binary.
  - Negative Tests:
    - A factorial family whose `-maxrregcount` sweep does not actually change spill behavior (confounded signal) is rejected by the contract verifier.
    - A new factorial cell counted toward coverage without repeatability-admitted NCU labels is rejected.
    - A new slug added without a roster entry, shape file, or correctness reference is rejected by `verify_coverage.py`.
    - A metric whose only evaluatable cells come from a single slug (no slug-disjoint coverage) is barred from a load-bearing claim.

- AC-5: A slug-disjoint calibration/validation/holdout split is regenerated from the imported 49-cell cohort and sealed.
  - Positive Tests:
    - A new split artifact is generated from the **union of** the imported task6 cohort (49 cells / 35 slugs) **and the newly admitted factorial cells (AC-4)**, assigning cells by slug with calibration >= 15 cells, validation >= 8 cells, holdout >= 8 cells and >= 8 unique slugs; no slug appears in more than one split; no slug exceeds one third of any split.
    - The split records its source cohort digest and supersedes the stale `cohort_split.json` (cal11/val5/holdout4 at floors 10/4/3), which is explicitly retired.
    - The holdout is sealed and evaluated once after validation gates pass; all tuning uses calibration/validation only; a holdout failure DEMOTES the affected claim (to appendix/diagnostic) rather than triggering a retune; the sealed-holdout artifact records the model/source commit used.
  - Negative Tests:
    - Same-slug leakage across splits is rejected.
    - Treating the pre-existing `cohort_split.json`/`sealed_holdout_labels.json` as a completed split is rejected (digest mismatch with the 49-cell cohort).
    - Re-running the holdout after a model/threshold change without invalidating the previous holdout report is rejected.

- AC-6: T2 cache/traffic modeling passes per-bucket and robustness gates, fixing the current failure surface.
  - Positive Tests:
    - Cross-level traffic primitives (`l1tex__m_xbar2l1tex_read_sectors…`, `l1tex__m_l1tex2xbar_write_sectors…`, `dram__sectors_op_{read,write}.sum`, `lts__t_requests_srcunit_tex.sum`) have their OWN model-derived count gate (looser than exact T1 because they are miss-stream-derived): p50 relative count error <= 10%, p90 <= 25%, with a denominator floor typing low-traffic cells `low_support`. A traffic primitive that misses this gate is reported as `diagnostic`, not promoted as a load-bearing count, and bars promotion of any hit-rate derived from it.
    - Hit-rates (`l1tex__t_sector_hit_rate.pct`, `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`, `lts__t_sector_hit_rate.pct`) report **absolute percentage-point error** with low/mid/high hit-rate buckets represented: p50 <= 5 pp, p90 <= 12 pp.
    - Robustness improves over the current failure surface: LOOCV-vs-holdout relative gap <= 0.20 and LOSO max relative change <= 0.25 (replacing the current `0.441` / `0.814`).
    - Low-request cells are typed `low_support` via a denominator floor rather than scored normally.
  - Negative Tests:
    - A hit-rate report passing relative error while failing absolute pp error or per-bucket coverage is rejected.
    - A robustness regression (gap > 0.20 or LOSO > 0.25) admitted as a load-bearing T2 claim is rejected.
    - A derived hit-rate that passes while its primitive traffic counts fail their own gate is rejected.

- AC-7: T3 timing/throughput metrics use DVFS-appropriate gates and cascade derived percentages from their dependencies.
  - Positive Tests:
    - `gpu__time_duration.sum` and `gpc__cycles_elapsed.max` use log-relative error: holdout geomean <= 0.15, bootstrap 95% upper <= 0.25, repeatability CV <= 5% for admitted labels; group-heldout (slug-disjoint) and leave-one-family-out are reported.
    - `*throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes.sum.per_second`, IPC, and achieved-warps are claimable only if their primitive traffic + timing dependencies pass; otherwise they cascade to `typed_unavailable` with dependency reasons.
    - Read-only NVML clock/throttle/power covariates are recorded for every T3 cell.
  - Negative Tests:
    - A T3 metric scored with the T1 exact-count gate is rejected (wrong gate class).
    - A derived pct-of-peak metric reported as passing while a dependency is unavailable is rejected.
    - A timing claim relying only on cell-level LOOCV while failing group-heldout is rejected.

- AC-8: A tiered, per-metric evidence harness with baselines, ablations, and a claim-to-evidence map gates the submission.
  - Positive Tests:
    - The harness (extending `tools/sm_86_predictor_accuracy_recovery_v2/publication_gate.py` and `workloads/simbench_bw/comparison_runner.py` additively over a generic `(cell, metric_key)` join) applies the per-tier gates above and reports per-metric AND per-tier results with typed-unavailable counts and coverage denominators front-and-center.
    - Baselines run on the exact same holdout cells per metric: always-hit/always-miss (rates), cohort/family mean, uncalibrated replay, uncalibrated reuse-distance, prior v4/v5 head, static roofline timing, CUPTI/CUDA-event-only timing, NVML-prior-only timing.
    - Ablations cover (at minimum): no active-mask modeling, no cache-op modeling, no cross-SM interleaving, no warmup state, no atomics-contention feature, no high-footprint correction, no timing-residual head.
    - Reports include geomean/MdAPE/median-absolute-error, signed bias, Pearson/Spearman, per-family breakdowns, 95% CIs, paired baseline deltas, and both cell-weighted and traffic-weighted statistics.
    - A claim-to-evidence map labels every metric `main` / `appendix` / `diagnostic` / `typed_unavailable`, with primitive and composed error reported separately.
  - Negative Tests:
    - A baseline run on a different cell set than the holdout is rejected.
    - A scalar-only report omitting coverage denominators, typed-unavailable counts, or paired baseline deltas is rejected.
    - A metric promoted to `main` whose tier gate did not pass is rejected.

- AC-9: The expansion is compliant, backward-compatible, and preserves cross-arch design.
  - Positive Tests:
    - Static guards pass: no NCU outside `tools/local_labels/`+`data/local_labels/`, no privileged CUPTI/events, no clock/power/MIG/driver mutation, no plan-token leakage in implementation code.
    - The legacy single-metric L1/L2 path (`comparison_runner.py` + `comparison_contract_sm86_submission.toml`) continues to pass unchanged for existing cache-layer bundles (multi-metric is additive/feature-gated).
    - All sm_120 cross-arch design (the `sm120_*` tooling, sm_120 data trees, `arch_capabilities`, the device-constant table's sm_120 rows) is retained; every sm_120 artifact carries an explicit `sm_120_deferred_no_live_evidence` status; no sm_120 live claim is made.
    - Schema-version discipline: additive registry/sidecar changes bump `registry_version` (and `label_schema_version` only on a breaking change), in lockstep across producer + schema + validators.
  - Negative Tests:
    - Any NCU invocation or `.ncu-rep*` write outside the quarantine path fails closed.
    - A change that breaks the legacy L1/L2 comparison path is rejected.
    - A cross-arch (sm_120) accuracy claim without matching sm_120 live evidence is rejected.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The loop delivers a tiered submission-grade evidence package on sm_86: the full ontology+DAG registry over the entire target metric set with typed availability; T0 occupancy and T1 replay counts validated to the exact/near-exact gates and promoted as the load-bearing "non-root NCU count-replacement" claim; T2 cache/traffic passing per-bucket + robustness gates as a load-bearing modeled claim; T3 timing/throughput reported with DVFS-appropriate gates (main if it passes, diagnostic/appendix otherwise); a factorial corpus giving slug-disjoint coverage per load-bearing metric; baselines, ablations, slug-disjoint sealed holdout, and a claim-to-evidence map; full compliance and retained cross-arch design. Cross-arch sm_120 remains explicitly out of claim.

### Lower Bound (Minimum Acceptable Scope)

At minimum the loop delivers: the ontology+DAG registry + device-constant table; the regenerated slug-disjoint split from the imported 49-cell cohort; T0 occupancy + T1 replay-count validation passing their gates as the load-bearing claim; the tiered evidence harness with baselines and typed-unavailable budgets; and either T2/T3 promoted results OR explicit typed non-promotion artifacts naming the missing coverage/robustness. The loop must not fake T2/T3 coverage; an honest typed non-promotion keeps T2/T3 out of the load-bearing claim while T0/T1 carry it.

### Allowed Choices

- Can use: the existing proto-registry, NVBit tracer + sector_stream, HRD reuse-distance + L1 walker, `parse_ncu_metrics.py`, the publication gate + comparison runner, existing portable-LCD/simbench kernels and sm_86-compatible shapes, additive factorial kernels and shape points, local NCU capture under `tools/local_labels/`, read-only NVML/CUPTI Activity, a restricted formula evaluator, and regularized deploy-legal model features.
- Cannot use: NCU outside the quarantine path, privileged CUPTI metrics/events/range/PM as deployable inputs, clock/power/persistence/MIG mutation, simulator revival, achieved NCU counters as predictor inputs (no label leakage), `eval()` for composition formulas, silent threshold widening, hidden holdout reuse, or scalar-only reporting without coverage denominators.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Make the metric ontology the spine first: extend `metric_registry.json` with `metric_class`, `risk_tier`, `ncu_observation_contract`, `composition`, and `per_arch_availability`; build a `CompositionDAGResolver` that validates the DAG and evaluates derived metrics. Land the device-constant table + occupancy calculator (T0) and the NVBit replay-count lockstep (T1) before any modeling, because counts are deterministic and either match NCU or expose a tracer-semantics gap to fix. Only then import the 49-cell cohort, regenerate the slug-disjoint split, refit the T2 cache/traffic model (reframed as cross-level sector-stream prediction with hit-rates as derived ratios) and the T3 timing model on calibration/validation only, and touch the sealed holdout once. Score everything through a generic `(cell, metric_key)` harness that applies the per-tier gate, runs baselines/ablations on the identical holdout, and emits the claim-to-evidence map. Sequence delivery tier-by-tier so hard T2/T3 failures cannot hide behind a large metric count.

### Relevant References

- `tools/local_labels/metric_registry.json` + `metric_registry.py` — proto-registry to extend (20 entries; families cache_hit_rate/execution_time/ipc/occupancy/traffic).
- `data/local_labels/ncu_metric_availability_probe_sm_86.json` — per-metric availability (29 targets/23 available); occupancy_limit_* flagged not-in-counter-catalog (T0 static).
- `tools/local_labels/parse_ncu_metrics.py` — NCU CSV parsing + raw-counter fallback to reuse in the lockstep validator.
- `tools/nvbit_tracer/`, `tools/sector_stream/{generator,consume_trace}.py`, `tools/cache_predictor/{l1_cache_op_classifier,l1_walker,l1_trace_adapter}.py` — replay/request/sector path for T1.
- `tools/hrd_profiler/{reuse_distance_tracker,cache_model_spec,observation_builder,device_properties,ga102_peak_constants}.py/.json` + `tools/cache_predictor/l1_static_card_table.json` — T2 model + device constants.
- `tools/sm_86_predictor_accuracy_recovery_v2/publication_gate.py`, `workloads/simbench_bw/comparison_runner.py`, `comparison_contract_sm86_submission.toml`, `tools/baseline_runners/` — evidence harness + baselines to generalize.
- `data/sm_86_predictor_accuracy_recovery_v2/task6_*` (49-cell cohort), `cohort_split.json` (stale; supersede), `calibration_head_v5_robustness_sm_86.json` (failure surface).
- `plans/arch_config_unification_merge_prep_draft.md` — the ~160 arch-coupled-value map for the device-constant decoupling.
- `plans/sm86_submission_grade_evidence_expansion_plan.md` — superseded predecessor; import its cohort + reuse AC-5/6/7/8 gate ideas.

## Dependencies and Sequence

### Milestones

1. Ontology spine + T0 static (architecture-independent core):
   - Phase A: Extend the metric registry (classes, tiers, observation contracts, composition, per-arch availability) over the full target set; build the composition-DAG resolver.
   - Phase B: Consolidate the provenance-tracked device-constant table; implement the occupancy calculator; deliver T0 occupancy_limit_* exact-or-typed-unavailable.
2. T1 replay-count lockstep + corpus expansion:
   - Phase A: Add factorial microbench families (local-mem, ATOM-vs-RED, texture/surface, request/sector formation, hit-rate regimes, occupancy/warp tails).
   - Phase B: Build the NVBit replay request/sector/instruction counters and the NCU lockstep validator; pass the T1 count-error gates with per-op coverage.
3. Cohort import + split + T2 modeling:
   - Phase A: Import the 49-cell task6 cohort; regenerate the slug-disjoint cal/val/holdout split; retire the stale split.
   - Phase B: Refit the cross-level traffic / hit-rate model on cal/val only; pass per-bucket + robustness gates.
4. T3 timing/throughput + evidence harness + reports:
   - Phase A: Refit the timing model; compose throughput/IPC/achieved-warps with dependency cascade; apply DVFS gates.
   - Phase B: Generalize the gate harness to per-tier/per-metric; run baselines + ablations on the sealed holdout (once); emit tiered + per-metric + claim-to-evidence reports; run compliance + cross-arch-retention guards.

The dependency spine is: ontology spine (Milestone 1) gates everything; T0/T1 (Milestones 1B-2) are validated before any modeling; the regenerated split (Milestone 3A) gates both T2 (3B) and T3 (4A); the evidence harness (4B) consumes all tier outputs. Cross-arch sm_120 design is carried but not live-validated this loop.

## Task Breakdown

Each task includes exactly one routing tag (`coding` = Claude, `analyze` = Codex via `/humanize:ask-codex`). Every AC is covered; every task targets an AC.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Extend metric registry with metric_class/risk_tier/ncu_observation_contract/composition/per_arch_availability over the full target set; schema + validator | AC-1 | coding | - |
| task2 | Implement CompositionDAGResolver (restricted evaluator, topo-sort, cycle/symbol/constant checks, availability cascade, label-leakage guard) | AC-1 | coding | task1 |
| task3 | Consolidate provenance-tracked device-constant table (arch-keyed, sm_86 + sm_120 rows) + fail-closed loader | AC-2, AC-9 | coding | task1 |
| task4 | Implement static occupancy calculator (block_dim + static/dynamic shared memory + cuobjdump -res-usage registers + device constants) for launch__occupancy_limit_*; exact-or-typed-unavailable | AC-2 | coding | task3 |
| task5 | Ask Codex to review the NCU observation-point contracts (counting units, ATOM-vs-RED, texture/surface, predication/masks) for fidelity risks | AC-1, AC-3 | analyze | task1 |
| task6 | Add factorial microbench families + roster/shape/correctness registration + kernel-contract regression locks; collect + repeatability-admit NCU labels for the new cells via tools/local_labels/ | AC-4 | coding | task1 |
| task7 | Build NVBit replay request/sector/instruction counters + NCU lockstep validator; pass T1 count-error gates + per-op coverage; fail-closed unsupported ops | AC-3 | coding | task5, task6 |
| task8 | Import 49-cell task6 cohort + newly admitted factorial cells; regenerate slug-disjoint cal/val/holdout split over the union; retire stale cohort_split.json; sealed-holdout guard | AC-5 | coding | task6 |
| task9 | Refit cross-level traffic / hit-rate (T2) model on cal/val; per-bucket pp gates + robustness (LOOCV<=0.20, LOSO<=0.25) + denominator floors | AC-6 | coding | task2, task7, task8 |
| task10 | Refit timing (T3) primitive model on cal/val, then compose throughput/IPC/achieved-warps with dependency cascade; DVFS gates + NVML covariates | AC-7 | coding | task8, task9 |
| task11 | Generalize publication_gate + comparison_runner to per-tier/per-metric `(cell, metric_key)` harness; baselines + ablations on sealed holdout; tiered/per-metric/claim-to-evidence reports | AC-8 | coding | task9, task10 |
| task12 | Ask Codex for independent review of statistical claims, baseline fairness, per-tier gate calibration, and holdout-leakage | AC-5, AC-8 | analyze | task11 |
| task13 | Compliance + backward-compat + cross-arch-retention verification (static guards, legacy L1/L2 path unchanged, sm_120 deferred-status, schema versioning) | AC-9 | coding | task11, task12 |

## Claude-Codex Deliberation

### Agreements

- Current results are not submission-grade (LOOCV gap 0.441 > 0.20; LOSO 0.814 > 0.25; L1 ~30% scalar coverage); a tiered, multi-metric NCU-replacement claim is the right next phase.
- The ~40 metrics decompose into primitives + derived normalizations; primitives and derived metrics must be validated separately.
- "Exact replay counts" is conditional on matching NCU observation-point semantics; T1 must be lockstep-validated on semantic microbenches before any modeling.
- The plan must absorb the half-done predecessor: import the 49-cell cohort, regenerate the stale slug-disjoint split, and keep the legacy single-metric comparison path working (additive).
- Delivery should be tier-sequenced (T0/T1 → T2 → T3) so hard failures cannot hide behind a large metric count.
- The arch-specific surface collapses to a provenance-tracked device-constant table; sm_120 stays design-only (deferred).

### Resolved Disagreements

- All-40-at-once vs tier-by-tier: resolved to **register all metrics (coverage = full set, "if available" → typed_unavailable) but sequence and gate delivery tier-by-tier**, with T0/T1 load-bearing first. This honors "cover all metrics" without an unrealistic single-shot modeling claim.
- Supersede vs fold the predecessor: resolved to **supersede** `sm86_submission_grade_evidence_expansion_plan.md` while **importing the 49-cell cohort** and reusing its split/refit/report gate ideas; the predecessor's remaining tasks 7-13 are absorbed into this plan's Milestones 3-4, not run separately.
- T2 primitive traffic-count gating (second-Codex review, UNRESOLVED → resolved): traffic primitives get their own model-derived count gate (p50 <= 10%, p90 <= 25%, denominator floor); a traffic primitive failing it is `diagnostic` and bars promotion of any hit-rate derived from it (AC-6).
- Sealed-split composition (second-Codex review, UNRESOLVED → resolved): the split is built over the **union** of the imported 49-cell cohort and the newly admitted factorial cells (AC-4/AC-5), because the multi-metric claim needs the new cells; continuity is preserved by recording the imported cohort's digest.

### Convergence Status

- Two Codex passes ran: a first-pass planning analysis (CORE_RISKS / MISSING_REQUIREMENTS / TECHNICAL_GAPS / ALTERNATIVE_DIRECTIONS / QUESTIONS_FOR_USER / CANDIDATE_CRITERIA) and a convergence review (AGREE / DISAGREE / REQUIRED_CHANGES / OPTIONAL_IMPROVEMENTS / UNRESOLVED). All REQUIRED_CHANGES were folded in (T2 primitive gates; task DAG deps task5→task7, task2→task9, task9→task10; union split; occupancy-calculator inputs; T1 denominator floor; auditable metric inventory). No must-fix items remain.
- Final Status: `converged`

## Pending User Decisions

- DEC-1: Minimum load-bearing metric count for the paper claim.
  - Claude Position: Make T0 (4 occupancy) + T1 (the available instruction/request/sector counts) the load-bearing "non-root NCU count-replacement" claim (~12-16 metrics), with T2 hit-rates/traffic also load-bearing if robustness passes; register all ~40 with typed availability.
  - Codex Position: 12-15 defensible T0/T1/T2 metrics as the main claim is far more defensible than asserting all ~40.
  - Tradeoff Summary: Full-40 coverage maximizes scope but risks a catalog of typed-unavailable/semantically-mismatched metrics; a tiered claim is smaller but stronger.
  - Decision Status: `RESOLVED — tiered claim` (register all ~40 with typed availability; T0/T1 load-bearing, T2 load-bearing-if-gated, T3 diagnostic).
- DEC-2: Is T3 throughput/pct-of-peak in the main claim or appendix/diagnostic?
  - Claude Position: Diagnostic/appendix until T3 robustness passes (DVFS-bound); promote to main only if AC-7 gates pass.
  - Codex Position: Defer T3 to appendix/diagnostic until timing robustness passes.
  - Tradeoff Summary: Including T3 in the main claim broadens the result but inherits DVFS/scheduler error.
  - Decision Status: `RESOLVED — appendix/diagnostic` (T3 promoted to main only if it clears its DVFS gates).
- DEC-3: Is the first submission claim explicitly "tiered NCU replacement" (T0/T1 replacement-grade; T2/T3 modeled/diagnostic)?
  - Claude Position: Yes — frame the headline as tiered, with T0/T1 as replacement-grade and T2/T3 as modeled.
  - Codex Position: Yes; a reduced, tiered claim is much more defensible at a top-tier venue.
  - Tradeoff Summary: A tiered framing is honest and review-resilient but less sweeping than "predicts all NCU metrics".
  - Decision Status: `RESOLVED — yes, tiered NCU replacement` (T0/T1 replacement-grade; T2/T3 modeled/diagnostic).
- DEC-7: Are the per-tier numeric error/robustness thresholds hard cutoffs or directional targets?
  - Claude Position: Hard requirements (demote below-gate metrics to diagnostic).
  - Codex Position: Per-tier gates with fail-closed promotion.
  - Tradeoff Summary: Hard gates prevent claim drift but can shrink the headline set; directional targets keep promotion evidence-based but require disciplined achieved-vs-target reporting to stay honest.
  - Decision Status: `RESOLVED — directional targets` (numeric accuracy/robustness thresholds are targets reported achieved-vs-target with CIs; structural/integrity gates remain hard — see the Acceptance Criteria preamble).
- DEC-4: Typed-unavailable budget per tier.
  - Claude Position: Zero for T0/T1 eligible metrics; <= 10% for T2; a larger explicit diagnostic budget for T3.
  - Codex Position: Same (0 / 0 / <=10% / larger-for-T3).
  - Tradeoff Summary: Strict budgets force honesty but may shrink the headline metric set.
  - Decision Status: `RESOLVED — 0 / 0 / <=10% / larger-diagnostic-for-T3` (Codex-confirmed: T0 and T1 eligible metrics 0; T2 <=10%; T3 explicit larger diagnostic-only budget, reported per metric/tier).
- DEC-5: sm_120 artifact status.
  - Claude Position: Schema/config-only, no live gates; every sm_120 artifact carries explicit `sm_120_deferred_no_live_evidence` status.
  - Codex Position: Explicit `sm_120_deferred_no_live_evidence` status on every artifact.
  - Tradeoff Summary: Explicit status prevents accidental cross-arch claims; pure schema-only is lighter but less auditable.
  - Decision Status: `RESOLVED — schema/config-only, explicit deferred status` (Codex-confirmed: no live gates; every sm_120 artifact carries `sm_120_deferred_no_live_evidence`).
- DEC-6: Cross-arch sm_120 live validation timing (matches the user's stated deferral).
  - Claude Position: Defer all sm_120 live work to the future physical-machine move; retain design only this loop.
  - Codex Position: Defer; sm_120 stays exploratory/config-only.
  - Tradeoff Summary: Cross-arch is valuable but multiplies collection/interpretation risk and is blocked on hardware availability.
  - Decision Status: `defer_cross_arch` (per the user's explicit instruction; recorded for traceability)

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", "task1", or similar workflow markers.
- These terms are for plan documentation only. In code, use descriptive, domain-appropriate names (e.g., `metric_class`, `risk_class`, `occupancy_limit`, `replay_count`, `cache_stream`, `time_clock`, `derived_ratio`) that describe the domain concept, not the plan structure.
- All new artifacts must be schema-valid and additive; preserve the frozen L2 v5 head and the legacy single-metric comparison path byte-for-byte unless a change is explicitly in scope.

--- Original Design Draft Start ---

# A Metric-Ontology Composition DAG For Non-Root NCU-Metric Prediction

## Original Idea

Next-phase research direction: elevate the non-root GPU predictor from the current sm_86 cache/timing prototype to a submission-grade, multi-metric NCU-replacement predictor that covers ~40 NCU metrics and is validated against NCU ground-truth labels (collected only via tools/local_labels/ into data/local_labels/, per CLAUDE.md). Build on the existing repo (workloads/simbench_bw l1/l2 microbenches + workloads/portable_lcd; tools/nvbit_tracer; tools/sector_stream; tools/cache_predictor L1 walker; tools/hrd_profiler reuse-distance; tools/local_labels parse_ncu_metrics + ncu_metric_availability_probe_sm_86.json with 29 targets/23 available; tools/sm_86_predictor_accuracy_recovery_v2 publication gate). Keep all sm_120 cross-arch design; sm_120 live study is DEFERRED to a future machine move but the design must decouple hardware/arch-specific config from the generic model.

CORE IDEA: treat the ~40 metrics not as 40 independent heads but via a METRIC ONTOLOGY + COMPOSITION DAG with classes {static (launch/resource arithmetic), replay_count (NVBit instruction/request/sector formation), cache_stream (reuse-distance/miss-stream modeling), time_clock (timing+NVML covariates), derived_ratio, derived_pct_peak, typed_unavailable}. Primitives feed derived metrics (hit_rate = 1 - miss/req; throughput.pct = traffic/time/device_peak; dram bytes/s = sectors*32/time; IPC = inst/cycles; achieved_warps from theoretical occupancy + scheduler correction). Validate primitives AND derived metrics separately. The arch-specific surface collapses to a device-constant table (sector size, regs/SM, smem/SM, max warps/SM, SM count, peak DRAM BW, L2 size) with provenance — the generic predictor stays identical across sm_86/sm_120.

CRITICAL RISK (per Codex review): "exact replay counts" is conditional on matching NCU OBSERVATION-POINT SEMANTICS (counting unit L1TEX/XBAR2L1/LTS/DRAM; active mask; predication; vector width; alignment; coalescing; cache operator; ATOM vs RED; texture/surface sampler state; L2 global interleaving invisible to per-CTA trace). The replay model must be lockstep-validated against NCU on semantic microbenches BEFORE cache/timing heads are trusted.

RISK TIERS drive per-tier evidence gates: T0 exact-static (occupancy_limit_*) -> exact-agreement-or-typed-unavailable; T1 deterministic-NCU-validated (smsp__inst_executed.sum, global op ld/st/atom/red requests, global ld/st sectors, local op ld/st requests) -> count-error gates + per-op coverage + fail-closed on unsupported op classes; T2 high-risk modeling (l1tex/l2 sector hit-rates, xbar<->l1 sectors, dram sectors, lts requests from tex, texture/surface requests) -> absolute error in percentage points + per-bucket low/mid/high regimes + slug-disjoint holdout; T3 DVFS/scheduler (time_duration, cycles, *.per_second, warps_active, IPC, throughput.pct) -> log-relative error + repeatability + bootstrap CI + NVML throttle covariates + group-heldout + leave-family-out.

BENCHMARK EXPANSION = dedicated metric families (not just more kernels): register-spill/local-memory sweeps (volatile local arrays, dynamic indexing, -maxrregcount); paired ATOM-vs-RED kernels with bin-count/skew/Zipf sweeps; texture 1D/2D point-filter footprint sweeps (below-L1/near-L2/above-L2); pitched-2D surface read/write (surface atom/red typed-unavailable if not reliably generated on sm_86); reuse-distance hit-rate sweeps at 0.25-4x L1/L2 working sets with warmup {0,1,5} and cache-control {all,none}; request/sector-formation kernels (alignment offsets, vector widths, partial active masks, uncoalesced); occupancy/warp sweeps with grid tails SM_count*N+r. Goal: dedicated evaluatable cells per metric and controlled factorial families so each phenomenon recurs across multiple slugs/shapes (this is the fix for the current LOOCV gap 0.441 / LOSO 0.814 robustness failures).

EVIDENCE DESIGN: grouped/tier gates first, per-metric gates second; baselines (always-hit/miss, mean-by-family, uncalibrated replay, uncalibrated reuse-distance, prior v4/v5 head, static roofline timing, CUPTI/CUDA-event-only timing, NVML-prior-only); ablations (no active-mask, no cache-op, no cross-SM interleaving, no warmup state, no atomics-contention, no high-footprint correction, no timing-residual head); report primitive-family AND composed-metric error with typed-unavailable counts front-and-center; cell-weighted and traffic-weighted statistics; denominator floors for low-support ratios.

HIGHEST-LEVERAGE SEQUENCE for the next loop: (1) extend metric registry/probe/parser to the full ~40 with typed per-arch availability; (2) build the metric ontology + composition DAG + explicit NCU observation-point contracts per count metric; (3) add the local/texture/surface/atom-red/request-sector-formation microbenches + hit-rate-regime sweeps; (4) validate replay counts (T0/T1) against NCU BEFORE touching cache/timing; (5) regenerate slug-disjoint cal/val/holdout split from the 49-cell task6 cohort, then refit cache (T2) and timing (T3) heads on cal/val only; (6) emit tiered + per-metric accuracy/robustness/baseline/ablation reports and a claim-to-evidence map.

## Primary Direction: Metric Ontology & Composition DAG Registry

### Rationale

Centers the loop on a single declarative metric registry + composition DAG (each metric declares its class, NCU observation-point contract, composition formula, and per-arch availability), turning ~40 targets into a small set of primitive estimators plus a deterministic composition engine — distinct from the data-corpus, modeling, replay, and validation-harness angles because it is the framework/architecture spine they all bind to, and because the repo already ships a proto-registry (`tools/local_labels/metric_registry.json`, 20 entries across 6 families) to extend rather than replace.

### Approach Summary

Build a metric-ontology + composition-DAG layer as the spine of the multi-metric predictor, extending the **existing committed proto-registry** `tools/local_labels/metric_registry.json` + `tools/local_labels/metric_registry.py` (which already binds `key → ncu_metric → unit → family → carrier → parser_transform → aggregation → repeatability_rule → nullable → deployability → availability → cache_layer`, families currently `{cache_hit_rate, execution_time, ipc, occupancy, occupancy_theoretical, traffic}`). Three additive concepts:

1. **Metric-class taxonomy.** Add a `metric_class` enum `{static, replay_count, cache_stream, time_clock, derived_ratio, derived_pct_peak, typed_unavailable}` to each entry, declaring its observation-point semantics and risk tier (T0–T3).
2. **Composition DAG.** Add an optional `composition` block to derived entries: `formula` (e.g. `hit_rate = 1 - miss/req`, `throughput_pct = (sectors*32/time)/device_peak_dram_bw`, `ipc = inst/cycles`), `dependencies` (primitive `ncu_metric` ids the formula needs), and `per_arch_availability` (arch→bool). A `CompositionDAGResolver` validates dependency availability, topologically sorts the DAG (rejecting cycles), routes primitives through their observation-point contract, and emits per-cell composition bundles. Primitives and derived metrics are scored separately so composition cannot hide denominator error.
3. **Unified device-constant table.** Formalize the existing per-arch constant patterns (`tools/hrd_profiler/device_properties.json`, `ga102_peak_constants.json`, `tools/cache_predictor/l1_static_card_table.json`) into one provenance-tracked table (sector size, regs/SM, smem/SM, max warps/SM, SM count, peak DRAM BW, L1/L2 capacity, L2 slices) keyed by arch with mandatory `source`/`provenance`, so the generic predictor stays identical across sm_86/sm_120 and the only arch-specific surface is the constants.

The registry is the binding point existing tools (`cache_predictor.py`, `hrd_profiler/cache_model_spec.py`, `parse_ncu_metrics.py`, `comparison_runner.py`) attach to; composition is opt-in for the new ~40-metric heads and applied at emit time. No forced refactor of existing single-metric paths.

### Objective Evidence

- `tools/local_labels/metric_registry.json` (16 KB; `registry_version`, `label_schema_version`, `units[]`, `deployability_kinds[]`, `parser_transforms[]`, `repeatability_rules[]`, `carrier_kinds[]`, `metrics[]` with 20 entries) and `tools/local_labels/metric_registry.py` (`MetricEntry` dataclass) — the proto-registry to extend; no composition field yet, so the extension is additive.
- `data/local_labels/ncu_metric_availability_probe_sm_86.json` — already a per-metric availability surface (29 targets / 23 available on sm_86; each entry carries `ncu_metric`, `ncu_type` Counter/Ratio, `rollup`, `registry_key`); the 4 `launch__occupancy_limit_*` are flagged `launch_config_metric_not_in_counter_catalog` (NCU computes them without counters → T0 static).
- Device-constant precedent with provenance discipline: `tools/cache_predictor/l1_static_card_table.json` (+ `.schema.json` requiring `source_reference`), `tools/hrd_profiler/device_properties.py` (`DevicePropertiesEntry`: `l2_capacity_bytes`, `sector_size_bytes`, `num_l2_slices`, `manifest_sha256`), `tools/hrd_profiler/ga102_peak_constants.json`, `workloads/simbench_bw/schemas/device_peak_constants.schema.json` (rejects empty provenance so a fabricated peak cannot pass).
- Composition intent already foreshadowed: `data/local_labels/timing_prediction_contract_sm_86.json` declares the nanosecond leg `role: derived_clock_uncertainty_interval` (derived = cycles × clock-conversion), i.e. an existing derived-metric pattern.
- Runtime composition precedent: `tools/cache_predictor/l1_walker.py` `PerSmL1Result.hit_rate = hit_count/total_eligible_accesses`; `tools/hrd_profiler/cache_model_spec.py` `resolve_cache_model_spec()`/`validate_declared_cache_model_spec()` already resolve live-CUDA + static-card + fallback into one spec (a composition-resolver pattern).
- Schema-versioning discipline: `local_label_sidecar.schema.json` / `local_label_admitted.schema.json` pin `schema_version: 5` in lockstep with the registry `label_schema_version: 5`; the v4→v5 bump added a `unit_tagged_metrics[]` block — precedent for additive composition surface without breaking consumers.

### Known Risks

- **Formula evaluation safety**: free-form `formula` strings need a restricted/safe evaluator (not `eval()`), with variable names validated against registry keys and hard fail-close on parse error.
- **Availability cascade**: if a primitive is `excluded` on an arch, every derived metric depending on it must cascade to excluded; needs a recursive reachability pass + tests that perturb primitive availability and audit derived status.
- **Per-arch formula divergence**: some derived formulas differ by arch (e.g. throughput_pct peak); support `per_arch_formula_overrides` with a default, rather than one hard-coded formula.
- **Device-constant coverage gaps**: not every constant exists for every arch (e.g. `l1_ways` optional); the engine must declare required constants per formula and emit a typed `device_constant_missing` reason when absent rather than silently fall through.
- **Cyclic / mis-declared DAG**: reject non-DAG structures at load time with a typed `composition_dag_cyclic` reason via topological-sort validation.

## Alternative Directions Considered

### Alt-1: Replay-Fidelity Lockstep (Exact-Count Tier First)
- Gist: Make NVBit trace-replay counts match NCU's exact observation-point semantics (active mask, predication, vector width, coalescing, cache operator, ATOM-vs-RED, counting unit) via deterministic semantic microbenches, and ship the T0/T1 exact-count metrics as the first provable submission claim before any cache/timing modeling. Extends the existing `tools/sector_stream/` pipeline with a per-op `request_stream` and a lockstep validator that compares trace-derived counts against NCU CSV per launch.
- Objective Evidence:
  - `tools/nvbit_tracer/import_trace_contract.py` — confirms each memory instruction already carries `active_mask`, `predicate_mask`, `data_width`, `memory_op_class`, reconstructed lane addresses.
  - `tools/sector_stream/consume_trace.py` + `generator.py` — existing pure-function sector model (`active_lane_count`, `touched_32B_sectors`, `sectors_per_request`); request-stream is a straightforward addition.
  - `tools/local_labels/parse_ncu_metrics.py` — already parses NCU CSV per launch with raw-counter fallback; a validator can reuse it. The T1 exact-count metrics are all `available: true` in the sm_86 probe.
  - `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py` — established producer/consumer typed-reason lockstep discipline to mirror.
- Why not primary: It is one (high-value) tier of the ontology rather than the spine; without the registry/composition layer it cannot express or score the derived and modeled tiers, so it is best sequenced as the first set of primitives the primary's DAG validates.

### Alt-2: Factorial Microbenchmark Corpus
- Gist: Treat corpus design as the bottleneck — build controlled factorial kernel/shape families (register-spill/local-mem, ATOM-vs-RED with skew/Zipf, texture 1D/2D footprint sweeps, pitched-2D surface, reuse-distance hit-rate regimes, request/sector formation, occupancy/warp tails) so each metric phenomenon recurs across many slugs/shapes — the direct structural fix for the LOOCV 0.441 / LOSO 0.814 robustness failures.
- Objective Evidence:
  - Existing single-axis prototypes to generalize: `workloads/simbench_bw/l1/dram_stream_stride_mix/` and `l2_prefetch_hint_sweep/` (perturbation_build variant sources), `working_set_capacity_sweep` (shape-axis sweep pattern).
  - Partial families to extend: `register_spill_local_mem` (params exist but shape file doesn't sweep `live_regs`), `global_atomic_histogram` (atomics family has one member, no paired RED).
  - Gaps confirmed by grep: no texture, surface, or request/sector-formation kernels exist in `l1/`/`l2/` today; `common.mk` ARCH_GATE + per-kernel Makefile + `roster.toml` + `shapes/<slug>.toml` + `verify_coverage.py` are the registration surface.
- Why not primary: It produces the data the model is trained/scored on, but without the ontology/DAG and tiered gates the new cells have no typed targets or scoring contract; it is the highest-value *input* to the primary, not the spine.

### Alt-3: Cross-Level Traffic / Miss-Stream Modeling
- Gist: Reframe the hard T2 cache-hit problem as predicting cross-level sector traffic (L1→L2 via xbar, L2→DRAM) from the reuse-distance model, with hit-rate %s derived as ratios (`hit = req - miss`, miss stream of one level = request stream of the next) and denominator-floor handling, replacing the current per-slug multiplicative correction band-aids.
- Objective Evidence:
  - `tools/hrd_profiler/reuse_distance_tracker.py` (per-layer Fenwick-tree trackers `sec_rd`, `line_rd`, `alloc_rd`, `region_rd`), `observation_builder.py` (per-launch `unique_sector_count`, `working_set_sector_count`, RD percentiles), `cache_model_spec.py` (L2 capacity/sector/slices).
  - `data/local_labels/calibration_head_v5_robustness_sm_86.json` (the LOOCV 0.441 / LOSO 0.814 failure surface) and `data/sm_86_predictor_accuracy_recovery_v2/l2_high_footprint_correction_*.json` (current per-slug correction factors the composition model would replace).
- Why not primary: It is the highest-risk modeling tier (T2) and depends on both the registry (to declare the cross-level metrics) and the corpus (hit-rate regimes); attacking it first risks the same robustness failures without the structural fixes the other directions provide.

### Alt-4: Tiered Evidence & Publication-Gate Harness
- Gist: Center on the validation infrastructure — generalize the single-metric L2 gate into per-tier/per-metric gates (T0 exact-agreement, T1 count-error + per-op coverage + fail-closed, T2 absolute pct-point error + low/mid/high regimes + slug-disjoint holdout, T3 log-rel + repeatability + bootstrap CI + NVML covariates + group/leave-family-out), with baselines, ablations, primitive-vs-composed error propagation, cell- and traffic-weighted stats, and a claim-to-evidence map.
- Objective Evidence:
  - `tools/sm_86_predictor_accuracy_recovery_v2/publication_gate.py` (5 cohort sub-gates), `workloads/simbench_bw/comparison_runner.py` + `comparison_contract_sm86_submission.toml` (bootstrap-CI geomean, typed headline-validity reasons), `tools/baseline_runners/nugteren2014_rdh_baseline.py` (always-hit/cohort-mean/uncalibrated-walker/v4-Platt baselines).
  - `data/local_labels/calibration_head_v5_accuracy_gates_sm_86.json` (per-family `per_bucket_p90_absolute_error` + gate_status), `data/sm_86_predictor_accuracy_recovery/ac4_per_slice_ablation.json` (per-cell before/after error + `headline_moved`), `cohort_split.json` + `sealed_holdout_labels.json`.
- Why not primary: It is how claims are gated/reported, not what is predicted; it consumes the registry's metric classes and the model's outputs, so it is best built alongside but downstream of the spine.

### Alt-5: Arch-Decoupled Device-Constant Model
- Gist: Center on the generic-vs-arch-specific separation — one provenance-tracked device-constant table + an arch-free predictor + an occupancy calculator (cuobjdump `-res-usage` regs + static smem + blockdim + device props) so the identical model runs on sm_86 now and sm_120 later, directly serving the deferred cross-arch retention requirement.
- Objective Evidence:
  - `tools/arch_capabilities.py` + `.json` (per-arch capability + script registry), `tools/hrd_profiler/device_properties.json` / `ga102_peak_constants.json` / `tools/cache_predictor/l1_static_card_table.json` (existing per-arch constant tables, sm_86-populated), `plans/arch_config_unification_merge_prep_draft.md` (maps ~160 arch-coupled hardcoded values across 6 subsystems and proposes the per-arch card + loader + fallback pattern).
  - Hardcoded couplings to factor: `cache_predictor.py` `_SM_COUNT_BY_ARCH`/`_GRIDDIM_PENALTY_ALPHA`, `run_suite.py` `_static_card_num_l2_slices_for_props`; no cuobjdump/occupancy-calculator exists yet (occupancy currently NCU-label-sourced).
- Why not primary: The device-constant table is a sub-component the primary already absorbs (concept 3), and the cross-arch live validation is explicitly deferred; promoting it to primary would over-index on the deferred arch axis instead of the sm_86 submission spine.

## Synthesis Notes

The five alternatives are not competitors to the primary but its phased build-out: the ontology/composition-DAG registry is the spine, and each alternative supplies one organ that plugs into it. The cleanest fold is to keep the registry primary and sequence the others as the loop's milestones — Alt-5's device-constant table is already concept (3) of the primary and lands first (it is the arch-decoupling foundation and unblocks the T0 occupancy-limit metrics); Alt-1's replay-fidelity lockstep delivers the T0/T1 primitives the DAG validates before any modeling; Alt-2's factorial corpus supplies the dedicated evaluatable cells (and is the structural fix for the LOOCV/LOSO failures) that the T1–T3 metrics are scored on; Alt-3's cross-level traffic model is the T2 organ that turns primitives into the hard hit-rate/traffic derived metrics; and Alt-4's tiered gate harness is the scoring/reporting contract that reads the registry's `metric_class` to apply per-tier gates and emit the claim-to-evidence map. If the user preferred a different primary, the most natural alternate spine is Alt-4 (gate harness) — it could drive the work top-down from the evidence contract — but it would still need the registry's metric ontology to know which tier each of the ~40 metrics belongs to, which is why the registry leads.

--- Original Design Draft End ---
