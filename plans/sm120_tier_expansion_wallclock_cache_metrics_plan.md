# sm_120 Wall-Clock Execution-Time Prediction, >=50-Cell Tri-Label Cohort, And Broader Counter-Free Cache-Metric Prediction

## Goal Description

Run ONE sm_120-only RLCR loop, internally staged into dependency-ordered milestones, that extends the non-root, counter-free, NOT-a-GPU-simulator predictor along three axes the user requested, every one closing by research-success OR honest engineering-closure falsification and terminal on the live RTX 5060 laptop (`Verified-live-sm_120`):

1. **Wall-clock execution-time predictor.** Predict per-kernel `duration_ns` from NVBit trace features + read-only NVML covariates, scored by per-kernel SMAPE (primary) with an MAE-with-floor diagnostic, on a slug-held-out split against declared baselines. Ground truth is the already-collected, non-root **CUPTI dual-source `duration_ns`** (CUPTI Activity reconciled against `cudaEventElapsedTime`), never NCU `gpu__time_duration.sum`.
2. **Cohort expansion to >=50 cells.** Reach >=50 admitted `(arch, slug, shape_point)` sm_120 cells, each carrying wall-clock + L1 + L2 ground truth, via depth-first collection over already-built `portable_lcd` slugs first, authoring kernels only if the repeatability/distribution gates cannot net 50.
3. **Broader counter-free cache-metric predictors.** Add a typed (`fraction|count|ratio`) NCU metric registry and a SEPARATE typed label path (without mutating the percentage-only `parse_ncu_metrics.py` contract), predictors for the cache/memory metric allowlist, and the L1->miss-stream->L2 filter prerequisite for any L1/L2 re-widening. The L2 sector hit-rate family is the named required-success target; all other cache/memory targets (other hit rates, ratios, counts) close by success-or-honest-falsification.

This is non-root counter-free observability, not a GPU simulator: all predictors are empirical feature->scalar regressions or trace-normalization cache accounting; no instruction-issue, warp-scheduling, DRAM-timing, pipeline, occupancy, or runtime simulation, and no analytic roofline as a primary time model (Rule 5). NCU runs only on the local device into `data/local_labels/`, driven only from `tools/local_labels/` (Rule 1).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests. The `AC-*` items are the current RLCR completion gates.

- AC-1: Compliance and Rule-5 framing hold across the expansion.
  - Positive Tests:
    - `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` scans clean with all new code present; new NCU collection lives only in `tools/local_labels/` writing only `data/local_labels/`.
    - Schema/doc prose that previously said outputs are what "any simulator predicting wall-clock must reproduce" is reframed to "any predictor/model estimating wall-clock duration", and a prose-token guard asserts no new surface uses GPU-simulator framing.
  - Negative Tests:
    - A synthetic file framing the duration model as a "wall-clock GPU simulator" fails the new prose-token guard.
    - A synthetic NCU invocation outside `tools/local_labels/`, a privileged-CUPTI surface, or a driver-root/clock-lock flag fails the static guard.

- AC-2: A committed wall-clock duration label artifact admits the cohort with repeatability gating.
  - Positive Tests:
    - `data/local_labels/duration_admitted.json` validates against a new closed schema; a CUPTI-provenance repeatability aggregator admits a cell from a fixture of N>=5 dual-source-reconciled repeats whose duration robust-CV is within the ceiling, recording median `duration_ns`, robust-CV, reconciled-repeat count, an NVML covariate summary, and full provenance (CUPTI sidecar hash, `binary_sha256`, `source_commit`, arch, slug, shape_point).
    - Acceptance thresholds are read from a named config (`duration_repeatability_acceptance.toml`): `min_reconciled_repeats=5`, `cv_ceiling=0.05`, `short_kernel_ns_threshold=5000`, `short_kernel_abs_jitter_floor_ns=2000` (short kernels gate on absolute stddev instead of CV).
  - Negative Tests:
    - A cell whose repeats fail dual-source reconciliation, or whose robust-CV exceeds `cv_ceiling` (or short-kernel absolute jitter exceeds the floor), emits a typed exclusion rather than an admit.
    - A duration label carrying NCU `gpu__time_duration` provenance is rejected (the label source must be CUPTI dual-source).

- AC-3: The expanded cohort is integrity-gated at >=50 `(arch, slug, shape_point)` cells with tri-label coverage.
  - Positive Tests:
    - A frozen, commit-pinned roster + validator admits >=50 sm_120 `(arch, slug, shape_point)` cells with family-bucket minima (memory_stress / tensor_compute / activation_norm each >=6), <=35% of cells per bucket, a per-slug cap of <=4 shape_points per slug, a slug-held-out split (every shape_point of a holdout slug is excluded from calibration), and required source-artifact presence (trace + L2 + L1 + duration evidence) per cell.
    - Each admitted cell carries L2 sector hit rate, L1TEX sector hit rate, and a `duration_admitted.json` duration label.
  - Negative Tests:
    - A roster with a slug exceeding the per-slug cap, a family bucket below its minimum, a bucket above 35%, or a slug appearing in both calibration and holdout fails the validator before any live collection.
    - A cell missing any of its three label sources fails the tri-label coverage check.

- AC-4: The wall-clock predictor emits leakage-checked, baseline-relative predictions and closes terminally.
  - Positive Tests:
    - The predictor emits per-cell `duration_ns` predictions and a schema-valid report containing: the slug-held-out split identity, the four declared baselines (per-slug mean, global median, shape-size-only, NVML-only) with their held-out scores, the predictor's held-out SMAPE (primary) and MAE-with-epsilon-floor (diagnostic), a short-kernel handling record, a leakage-check result, a failure-mode decomposition (feature-fidelity / model-capacity / DVFS-non-stationarity), and a terminal verdict in `{research_success, engineering_closure_refuted, indeterminate}`.
    - Closure is reached when the predictor's held-out SMAPE is strictly lower than the strongest baseline's held-out SMAPE (research success) OR the report records `engineering_closure_refuted` with the full decomposition.
  - Negative Tests:
    - Any shape_point of a holdout slug appearing in the calibration set fails the leakage check.
    - A report scored by plain MAPE on near-zero-duration kernels (no epsilon/short-kernel rule) fails the report validator.

- AC-5: An L1->miss-stream->L2 filter normalizes the replay input before any L1/L2 cache-metric comparison.
  - Positive Tests:
    - A trace-normalization stage partitions accesses per-SM/CTA (`sm_id`/`warpid_sm`, with a provenance-tagged round-robin fallback) and applies `.ca`/`.cg` cache-op eligibility, emitting only the L1-miss stream into the L2 replay; a synthetic mixed-`.ca`/`.cg`, multi-SM trace yields the analytically expected L1 pass/miss partition and the expected L1-miss-fed L2 result.
    - The stage is documented and guarded as trace-normalization / cache-accounting, not scheduler or timing simulation.
  - Negative Tests:
    - A full-stream (un-filtered) L2 replay is rejected for an L1/L2 headline comparison.
    - A cell lacking the metadata needed to partition (no `sm_id` and no derivable CTA->SM mapping) under headline-live mode without an explicitly admitted round-robin-fallback provenance tag fails closed rather than silently using the fallback.

- AC-6: A typed NCU metric registry + a separate typed label path admit the cache/memory allowlist without breaking the percentage-only parser.
  - Positive Tests:
    - A typed metric registry declares each target as `{ncu_token, kind in (fraction|count|ratio), unit, bounds, denominator-if-ratio, admissible_layers, repeatability_stat, prediction_source, comparison_formula}`; a separate typed extraction path admits one `fraction`, one `ratio`, and one `count` fixture with registry metadata, while `parse_ncu_metrics.py`'s `[0,1]`-fraction contract is unchanged (its own tests stay green).
    - The allowlist is exactly: hit rates (`l1tex__t_sector_hit_rate.pct`, `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`, `lts__t_sector_hit_rate.pct`); request counts (`l1tex__t_requests_pipe_lsu_mem_global_op_ld/st/atom/red.sum`, unified L1 total requests, `lts__t_requests_srcunit_tex.sum`); sector counts (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld/st.sum`, `l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum`, `l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum`, `dram__sectors_read/write.sum`); sectors-per-request ratios; and the derived totals.
  - Negative Tests:
    - An occupancy / launch-limit, compute (cycles/IPC/MIPS/warp-insts), or `gpu__time_duration` metric is rejected by the allowlist.
    - A registry entry whose `kind` does not match its admission path (e.g. a `count` routed through the fraction parser) fails validation.

- AC-7: The metric comparison runner handles every target kind's join, bounds, and error geometry.
  - Positive Tests:
    - For each target kind the runner joins a valid typed label to its predictor on the full identity key, applying per-target bounds, ratio denominators, an epsilon/floor for fractions, and a count-appropriate error geometry (absolute / count-floor, not unbounded relative).
    - A typed-unavailable target is recorded with a typed reason rather than crashing the run.
  - Negative Tests:
    - A zero-denominator ratio or a negative/non-integer count fails with a typed reason.
    - A metric whose registry version does not match the committed contract fails with a metric-version-drift typed reason.

- AC-8: Each cache/memory target closes terminally; the named required-success family delivers and the rest are success-or-falsification.
  - Positive Tests:
    - The L2 sector hit-rate target, evaluated on the L1-filtered replay (AC-5), reaches a load-bearing predictor (its accuracy gate passes) OR emits a schema-valid falsification report; the loop's cache claim additionally requires at least this one named family to reach a load-bearing predictor.
    - Every other cache/memory target closes by either a load-bearing predictor (its declared gate) OR a schema-valid count/metric falsification report carrying: the attempted NVBit-trace->NCU mapping, the aggregation-domain mismatch, a coalescing/replay/write-allocation explanation where applicable, the comparison geometry used, and a terminal verdict.
  - Negative Tests:
    - A raw-count target whose NVBit-trace->NCU mapping is non-comparable is recorded as `engineering_closure_refuted` with evidence, NOT silently reported as a failed-accuracy predictor.
    - The cache claim is rejected if NO named family reaches a load-bearing predictor (the L2 sector hit-rate family failing to deliver without a falsification report fails closed).

- AC-9: A multi-claim closure registry records a terminal, sm_120-host-terminal verdict per target.
  - Positive Tests:
    - A unified registry artifact validates against a closed schema; each target record carries `{target_id, kind, ac_coverage, verdict in (research_success|engineering_closure_refuted|indeterminate), evidence_artifact_paths, verified_live_sm120_proof (device/driver/CUDA/NVBit/observed-clocks), success_or_falsification_details, cross_arch_precondition: false}`.
    - Every target (duration + each cache/memory metric) has exactly one terminal verdict and a `Verified-live-sm_120` proof block.
  - Negative Tests:
    - A target record gating on a cross-arch / second-host precondition, or with `verdict = indeterminate` presented as a closure, fails the registry validator.
    - A target lacking an `evidence_artifact_paths` entry or a `verified_live_sm120_proof` block fails validation.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
All nine ACs satisfied with positive and negative tests: a committed CUPTI-provenance `duration_admitted.json`, a >=50-cell tri-label sm_120 cohort, a baseline-beating wall-clock predictor, a working L1->miss-stream->L2 filter, a typed metric registry + separate typed label path admitting the full cache/memory allowlist, a comparison runner handling fraction/ratio/count geometry, load-bearing predictors for the L2 sector hit-rate family plus at least one additional hit-rate-or-ratio family (with counts either load-bearing or falsified-with-evidence), and a multi-claim closure registry with a terminal `Verified-live-sm_120` verdict per target — without reviving simulator code or framing.

### Lower Bound (Minimum Acceptable Scope)
Each predictive target reaches a TERMINAL verdict (research-success OR honest engineering-closure falsification), and specifically: AC-1 compliance green and prose reframed; AC-2 `duration_admitted.json` admits the cohort under the named repeatability config; AC-3 >=50 `(arch, slug, shape_point)` cells pass the frozen-roster gates with tri-label coverage; AC-4 the duration predictor emits the leakage-checked baseline-relative report and closes (beat-strongest-baseline or refuted-with-decomposition); AC-5 the L1->L2 miss-filter is built and full-stream L2 is rejected for L1/L2 mode; AC-6 the typed registry + separate label path admit the allowlist with the fraction parser unchanged; AC-7 the comparison runner handles all three kinds; AC-8 the L2 sector hit-rate family reaches a load-bearing predictor (or is falsified-with-evidence) and every other cache/memory target carries a terminal success-or-falsification verdict (counts MAY all close by falsification); AC-9 the registry carries a terminal sm_120-host verdict per target. The loop closes when every target has a terminal verdict and the named required-success family plus the duration target each have a verdict.

### Allowed Choices
- Can use: NVBit dynamic traces, CUPTI Activity, CUDA events, read-only NVML covariates, local-quarantine NCU (only under `tools/local_labels/` writing `data/local_labels/`), the existing dependency-tiered calibration head (Isotonic/GAM/GBM/XGBoost), the `tools/cache_replay/` gate + falsification + advisory-validation harness, and the existing cohort/repeatability machinery; sklearn / pygam / lightgbm / xgboost; sector-basis prediction first.
- Cannot use: cloud/server NCU, privileged CUPTI (Events/Metrics/Range/PM/PerfWorks), driver-root or clock-locking, host-driver work, any instruction-issue / warp-scheduling / DRAM-timing / pipeline / occupancy / interconnect / runtime simulation, an analytic roofline/bandwidth model as the primary time predictor (roofline only as a diagnostic covariate), or any "GPU simulator" framing of the deliverables.

## Feasibility Hints and Suggestions

### Conceptual Approach
Stage the loop so each milestone has a hard stop point. Land the compliance/prose + contract skeletons first (the `duration_admitted.json` schema, the typed metric registry). Run the cohort campaign path-A (more shape_points of already-built `portable_lcd` slugs) producing trace + NCU L1/L2 labels + CUPTI duration per cell, gated by the frozen roster validator. Aggregate the duration labels (new CUPTI-provenance aggregator mirroring `repeatability_aggregator.py`). Train the duration predictor by retargeting `monotonic_calibration_head.py` to `duration_ns` (generalize the `[0,1]` clip to a per-target bound), with NVML covariate conditioning via `nvml_covariate_join.py`, scored by `compute_validation_metric.py::per_kernel_mape` extended to SMAPE + MAE-floor, and a slug-held-out split. Build the L1->miss-stream->L2 filter as a per-SM/CTA trace-normalization stage reusing the `tools/cache_predictor/l1_walker.py` per-SM model, then re-widen `comparison_contract.toml` `cache_layers` to `[L1, L2]`. Add the typed cache/memory targets through the new typed label path + comparison runner, and route every target through the multi-claim closure registry so each closes by research-success-or-honest-falsification.

### Relevant References
- `workloads/simbench_bw/cupti_activity_collector.cu`, `schemas/cupti_activity_sidecar.schema.json`, `run_suite.py --mode timing` — CUPTI dual-source duration ground truth.
- `tools/local_labels/repeatability_aggregator.py`, `check_admission_floor.py`, `merge_admitted_cells.py`, `expand_cohort_manifest_from_catalog.py`, `sm120_cohort_roster_validator.py`, `data/local_labels/sm120_cohort_inventory.json` — cohort + repeatability machinery.
- `workloads/simbench_bw/monotonic_calibration_head.py`, `canonical_features.py`, `nvml_covariate_join.py`, `compute_validation_metric.py` — calibration head, features, NVML covariates, scorer.
- `tools/cache_replay/` (`sector_cache_replay.py`, `accuracy_gate.py`, `falsification.py`, `validation.py`, `artifact_io.py`) — replay + gate + falsification harness.
- `tools/cache_predictor/l1_walker.py`, `l1_trace_adapter.py`, `l1_cache_op_classifier.py`, `tools/nvbit_tracer/` instruction/threadblock proto + `import_trace_contract.py` — L1 per-SM model + per-access metadata for the L1->L2 filter.
- `tools/local_labels/parse_ncu_metrics.py`, `label_driver.sh`, `workloads/simbench_bw/schemas/local_label_*.schema.json`, `comparison_contract.toml`, `test_metric_name_lockstep.py` — the wire surfaces the typed registry coordinates.
- `read_ncu_report.py` — the canonical metric enumeration the cache/memory allowlist is drawn from.

## Dependencies and Sequence

### Milestones
1. Milestone M1 — Compliance + contracts.
   - Phase A: AC-1 prose reframe + Rule-5 prose-token guard + static-guard re-green.
   - Phase B: AC-2 `duration_admitted.json` schema + `duration_repeatability_acceptance.toml`; AC-6 typed metric registry skeleton.
2. Milestone M2 — >=50-cell tri-label cohort.
   - Phase A: frozen roster + AC-3 validator (caps/buckets/split/tri-label preflight).
   - Phase B: path-A trace + NCU L1/L2 + CUPTI duration collection on already-built slugs.
3. Milestone M3 — Duration label + predictor.
   - Phase A: AC-2 CUPTI-provenance aggregator admits >=50 cells.
   - Phase B: AC-4 duration predictor + baselines + slug-held-out eval + falsification report.
4. Milestone M4 — L1->miss-stream->L2 filter.
   - Phase A: AC-5 per-SM/CTA filter + eligibility + fail-closed fallback + tests.
5. Milestone M5 — Typed cache/memory metrics.
   - Phase A: AC-6 separate typed label path + allowlist; AC-7 comparison runner geometry.
   - Phase B: AC-8 cache-metric predictors (named L2 family load-bearing; ratios; count feasibility-or-falsification).
6. Milestone M6 — Multi-claim closure.
   - Phase A: AC-9 closure registry + per-target terminal verdicts.

Dependencies: M2 depends on M1's label contracts; M3 depends on M2; M4 is independent of M3 and gates L1 re-widening in M5; M5 depends on M1's registry and M4's filter; M6 depends on all. Milestone boundaries are natural operator split points if the loop is later divided.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Reframe simulator->predictor prose on the affected schemas/docs; add a Rule-5 prose-token guard for the new surfaces; keep the no-root/no-NCU/no-privileged static guard green | AC-1 | coding | - |
| task2 | New `duration_admitted.json` closed schema + `duration_repeatability_acceptance.toml` + CUPTI-provenance repeatability aggregator (N>=5, dual-source pass, CV/short-kernel gates, NVML summary, typed exclusions) | AC-2 | coding | task1 |
| task3 | Frozen commit-pinned >=50-cell roster + roster validator (per-slug cap, bucket minima, <=35%/bucket, slug-held-out split, tri-label source-artifact preflight) | AC-3 | coding | task2 |
| task4 | Path-A live collection on already-built sm_120 slugs: NVBit trace + NCU L1/L2 labels + CUPTI duration per cell; admit via the aggregators | AC-3 | coding | task3 |
| task5 | Wall-clock predictor: retarget the calibration head to `duration_ns` (per-target bound) + NVML covariate conditioning + baselines + slug-held-out SMAPE/MAE-floor + AC-4 falsification report contract | AC-4 | coding | task4 |
| task6 | Codex adversarial review of the duration predictor: leakage (slug-held-out), baseline-beating validity, and a DVFS-leakage check (does it still beat baselines without thermal/clock covariates?) | AC-4 | analyze | task5 |
| task7 | L1->miss-stream->L2 filter: per-SM/CTA partition (`sm_id`/`warpid_sm` + fail-closed provenance-tagged round-robin fallback) + `.ca`/`.cg` eligibility + synthetic-trace tests + reject full-stream L2 for L1/L2 mode | AC-5 | coding | task1 |
| task8 | Typed NCU metric registry + separate typed (`fraction|count|ratio`) label path enforcing the allowlist, leaving `parse_ncu_metrics.py` percentage-only contract unchanged; lockstep test parameterized over the registry | AC-6 | coding | task2 |
| task9 | Metric comparison runner: per-target bounds, ratio denominators, fraction epsilon/floor, count error-geometry, typed-unavailable, and metric-version-drift typed reasons | AC-7 | coding | task8 |
| task10 | Codex count-feasibility analysis: whether NVBit-trace counts are comparable to NCU's post-coalescing/replay/aggregation-domain counters, producing the AC-8 falsification evidence for non-comparable count targets | AC-8 | analyze | task9 |
| task11 | Cache-metric predictors over the L1-filtered replay: L2 sector hit-rate family load-bearing; additional hit-rate/ratio families; raw counts via the success-or-falsification AC-8 path + count falsification report contract | AC-8 | coding | task7, task9, task10 |
| task12 | Multi-claim closure registry + closed schema: per-target terminal verdict, AC coverage, evidence paths, `Verified-live-sm_120` proof, `cross_arch_precondition: false` | AC-9 | coding | task5, task11 |

## Future Work / Out of Scope

- FUT-1: Expand to >=50 DISTINCT slugs (build the unbuilt L2 simbench corpus) for a distinct-slug publication grain.
  - Source DEC: DEC-1
  - Current-loop handoff: AC-3 (records the distinct-slug count alongside the `(arch, slug, shape_point)` cell count)
  - Promotion trigger: a reviewer / publication requirement for >=50 distinct slugs.
- FUT-2: sm_86 cross-arch replication of the duration + cache-metric predictors on a separate RTX 3090 Ti session.
  - Source DEC: DEC-2
  - Current-loop handoff: AC-9 (records `Verified-live-sm_120` only, with `cross_arch_precondition: false`)
  - Promotion trigger: a cross-arch (`Cross-arch-verified`) claim is required.
- FUT-3: Unified multi-target prediction framework (one registry-driven head emitting all targets) replacing the per-target heads.
  - Source DEC: DEC-2
  - Current-loop handoff: AC-6 (the typed metric registry the unified head would consume)
  - Promotion trigger: the duration target validates the per-target abstraction and >=2 targets are load-bearing.
- FUT-4: Promote raw-count predictors to load-bearing.
  - Source DEC: DEC-3
  - Current-loop handoff: AC-8 (the per-count feasibility verdict)
  - Promotion trigger: the count feasibility analysis shows an NVBit-trace->NCU-count mapping that passes a count-appropriate accuracy gate.

## Claude-Codex Deliberation

### Agreements
- Wall-clock label source is the non-root CUPTI dual-source `duration_ns`, never NCU `gpu__time_duration`.
- Predict nanoseconds with NVML clock/thermal/power/throttle as conditioning covariates (no cycle conversion, no clock locking).
- Requirement #3 scope is a cache/memory allowlist drawn from `read_ncu_report.py`, EXCLUDING occupancy, compute (cycles/IPC/MIPS), and NCU time.
- Do NOT mutate `parse_ncu_metrics.py`'s percentage-only contract; add a separate typed (`fraction|count|ratio`) label path + registry.
- The L1->miss-stream->L2 filter is a current-loop prerequisite for any L1/L2 re-widening (NCU's L2 observes only the L1-miss stream).
- sm_120-only, `Verified-live-sm_120`; sm_86 deferred to a separate host (FUT-2).
- Each predictive target closes by research-success OR honest engineering-closure falsification; the duration predictor needs declared baselines, a slug-held-out split, and a short-kernel-stable secondary metric.
- A single staged loop is defensible (Codex AGREE) because per-target falsification closure decouples targets so no single dependency holds the loop hostage; milestone boundaries are operator-splittable.

### Resolved Disagreements
- Single loop vs split into sequential loops: Codex initially recommended splitting into three loops; Claude proposed one staged loop with per-target honest-falsification closure. Resolution: ONE staged loop with dependency-ordered milestones as operator-splittable stop points; Codex agreed this is RLCR-valid once a Task Breakdown, per-AC falsification report contracts, pinned thresholds, Path Boundaries, and a load-bearing closure registry are added (all incorporated).
- AC-8 success/falsification semantics ("families MUST deliver" vs ">=1 family passes"): Resolved to Codex option 1 — at least one NAMED family (the L2 sector hit-rate family on the L1-filtered replay) must reach a load-bearing predictor; all other cache/memory targets close by success-or-honest-falsification (counts may all falsify).
- Count predictability: Codex flagged that NVBit-trace counts may not be comparable to NCU's post-coalescing/replay counters. Resolved: raw counts are gated by AC-8's feasibility-or-falsification path (task10 Codex analysis + a count falsification report), not assumed load-bearing.

### Convergence Status
- Final Status: `converged`
- Convergence rounds executed: 1 (Codex first-pass analysis + 1 Codex convergence review; all REQUIRED_CHANGES incorporated; structure endorsed).

## Pending User Decisions

The following product-positioning decisions were resolved by Claude (informed by two Codex passes) per the operator's standing "discuss-with-Codex, decide-myself, do-not-block" preference; they are recorded here as revisitable should the user prefer a different positioning.

- DEC-1: Does ">=50 cells" mean >=50 DISTINCT slugs, or >=50 `(arch, slug, shape_point)` cells?
  - Claude Position: `(arch, slug, shape_point)` grain (repo-native join grain; path A reaches it without new kernels).
  - Codex Position: acceptable ONLY if the publication claim explicitly states `(arch, slug, shape_point)`; otherwise distinct slugs are needed.
  - Tradeoff Summary: the `(slug, shape_point)` grain is fast (existing binaries) but lower implementation diversity; distinct slugs are more defensible for publication but require building the unbuilt L2 simbench corpus.
  - Decision Status: Resolved — this loop targets >=50 `(arch, slug, shape_point)` cells with the claim stated explicitly; distinct-slug expansion is FUT-1.
- DEC-2: Which target is the headline if milestones conflict, and may scope be dropped?
  - Claude Position: the wall-clock duration predictor is the headline for positioning; all ACs are mandatory (each via success-or-falsification), so priority is operational only, not a scope-drop lever.
  - Codex Position: must be declared so the lower bound is unambiguous.
  - Tradeoff Summary: declaring all ACs mandatory keeps the closure gate firm; designating a droppable target would weaken it.
  - Decision Status: Resolved — all ACs mandatory; duration is the positioning headline; FUT-2/FUT-3 hold the deferred architecture/cross-arch work.
- DEC-3: Must the loop deliver at least one raw-count predictor, or may all counts close by falsification?
  - Claude Position: counts may all close by honest falsification; the L2 sector hit-rate family is the named required-success target and the hit-rate/ratio families carry the load-bearing cache claim.
  - Codex Position: must be declared up front (resolved by the AC-8 semantics fix).
  - Tradeoff Summary: requiring a count predictor risks blocking the loop on an empirically non-comparable target; allowing falsification keeps closure honest.
  - Decision Status: Resolved — counts close by success-or-falsification (FUT-4 promotes them); the named L2 family must deliver.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific workflow terminology such as `AC-`, `Milestone`, `Phase`, `Step`, `Round <N>`, or similar markers (a tracked plan-token scanner enforces this). Use descriptive, domain-appropriate names.
- New code and comments must use predictor/model framing, never "GPU simulator" / "simulator predicting wall-clock"; the Rule-5 prose-token guard (AC-1) enforces this on the new surfaces.
- Any new NCU metric / label field / op-class lands as a coordinated single-commit lockstep change across the driver argv, the parser-to-sidecar mapping, the sidecar emitter, the closed schemas, the import denylist + aggregator, and the comparison contract + runner + lockstep test (the consumer-coupled wire policy).

--- Original Design Draft Start ---

# Counter-Free Wall-Clock Execution-Time Prediction For Non-Root GPU Kernels

## Original Idea

# Extend the non-root counter-free GPU predictor: wall-clock execution-time prediction, a >=50-cell sm_120 cohort, and broader NCU cache-metric predictors

## Motivation

The project today predicts exactly ONE NCU-ground-truth quantity — the **L2 sector hit
rate** (`lts__t_sector_hit_rate.pct`) — on a 30-cell sm_120 cohort, with the L1TEX sector
hit rate collected-but-not-headline and everything else absent. The two stated research
targets in README/CLAUDE.md are **wall-clock behavior** and **cache hit ratio**, but only
the cache side has a scored predictor, and even that is narrowed to L2 sector hit rate.
This idea extends the predictor along three axes the user requested, while staying strictly
inside the non-root, counter-free, NOT-a-GPU-simulator contract (CLAUDE.md Rules 1-5).

## The three requirements (user-stated)

1. **Wall-clock accuracy via an actual execution-time predictor.** Build a predictor that
   estimates per-kernel execution time and is scored against measured ground truth across
   the benchmark suite.
2. **Expand the test cohort to >=50 cells**, each carrying wall-clock, L1, and L2 ground
   truth. Reuse existing kernels/microbenches where possible; author or rewrite kernels if
   the existing catalog cannot reach 50.
3. **Broader cache-metric predictors.** Beyond the L2/L1 *sector hit rate*, implement
   predictors for the other essential cache/memory measurements enumerated in the canonical
   template `read_ncu_report.py`.

## Resolved scoping decisions (deep research + Codex consultation; do-not-guess items closed)

These decisions were reached by reading the repo end-to-end and consulting Codex
(gpt-5.5:high). They are the binding scope for this work.

### D1 — Requirement #3 metric allowlist (cache/memory only; compute/occupancy excluded)
`read_ncu_report.py` enumerates 33 NCU metrics but is inherited DEAD code (absent
`ncu_report_helper`, hardcoded foreign path, CUDA 11.x). Its enumeration is the metric
*menu*, not the project's intent. The in-scope predictor targets are the **cache/memory**
subset:
- **Hit rates:** `l1tex__t_sector_hit_rate.pct` (L1 unified), `lts__t_sector_hit_rate.pct`
  (L2), and the not-yet-collected `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`
  (L1 global-load read hit rate).
- **Request counts:** GMEM load/store requests
  (`l1tex__t_requests_pipe_lsu_mem_global_op_ld/st.sum`), global atomic/reduction requests
  (`...op_atom/red.sum`), unified L1 total requests, unified L2 total requests
  (`lts__t_requests_srcunit_tex.sum`), and the derived GMEM-total / atomic+reduction totals.
- **Transaction / sector counts:** GMEM load/store sectors
  (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld/st.sum`), L2 read/write sectors
  (`l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum`,
  `l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum`), DRAM read/write sectors
  (`dram__sectors_read/write.sum`), and their derived totals.
- **Sectors-per-request ratios:** read and write transactions-per-request (derived).
- **EXCLUDED** from this effort: occupancy / launch-occupancy limits, compute throughput
  (cycles, IPC, MIPS, warp-instruction counts), and `gpu__time_duration.sum` — wall-clock
  belongs to requirement #1 and must stay non-NCU-derived. This exclusion is an explicit
  scope decision (the user pointed at `read_ncu_report.py`, so it is stated, not silent).

### D2 — Parser discipline: a new typed path, not a contract break
`tools/local_labels/parse_ncu_metrics.py` is structurally percentage-only (admits [0,100]
-> fraction, clamps (100,105] -> 1.0, raw-counter hit/(hit+miss) fallback for .pct
overcount, hard-rejects non-fractions). Do NOT mutate that narrow, valuable contract. Add a
SEPARATE typed extraction path that carries an explicit `unit` per metric
(`fraction` | `count` | `ratio`) so counts/ratios have a first-class admission path. Each
new `.pct` ratio metric gets its own raw-counter pair entry if it exhibits the Blackwell
>105 overcount.

### D3 — Wall-clock label source: CUPTI dual-source duration_ns (NOT NCU time)
Ground truth for the execution-time predictor is the already-collected, non-root
**dual-source CUPTI Activity `duration_ns`** reconciled against host
`cudaEventElapsedTime` (`cupti_activity_collector.cu`, `run_suite.py --mode timing`,
CLOCK_MONOTONIC, with a reconciliation admission gate). Do NOT add NCU
`gpu__time_duration.sum` to the schema — keeping the target non-root-derived strengthens
the project thesis and avoids a non-fraction NCU parser/schema branch.

### D4 — Duration label artifact: new `duration_admitted.json`
Persist wall-clock labels in a NEW committed artifact (`data/local_labels/duration_admitted.json`)
with its own closed schema and a parallel CUPTI-provenance repeatability aggregator:
N>=5 repeats, dual-source reconciliation must pass, a CV/robust-CV ceiling (analogous to
the 0.03 stddev L2 gate), and full provenance (CUPTI sidecar hash, binary_sha256,
source_commit, arch, slug, shape_point, NVML covariate linkage). Do NOT bolt a nullable
duration field onto the NCU-cache `local_label_admitted` v4 artifact (different provenance,
different units; would force a real v4->v5 parse-time-breaking migration).

### D5 — Prediction units: nanoseconds, DVFS as covariates
Predict **nanoseconds** (the CUPTI observable). Clocks cannot be locked (Rule 2), so feed
NVML clock / thermal / power / throttle state as conditioning covariates (not a cycle
conversion layer). Report scope as the current sm_120 host.

### D6 — Rule-5 boundary (the key safety line)
An **empirical supervised regression** from NVBit trace features + NVML covariates to a
scalar `duration_ns`, scored by per-kernel MAPE, calibrated with the existing
dependency-tiered monotone/bucketed head, is INSIDE Rule 5. FORBIDDEN: simulating
instruction issue, warp scheduling, SM pipelines, DRAM queue/timing, interconnect,
occupancy, or full-kernel runtime; and presenting the result as a "validated GPU
simulator". Roofline / bandwidth formulas may appear only as diagnostic covariates, never
as the primary time model. Schema/doc prose that currently says outputs are what "any
simulator predicting wall-clock must reproduce" should be reframed to
"any predictor/model estimating wall-clock duration". Rule 5 has no static guard today; a
prose-token guard for the new surfaces would help.

### D7 — Cohort cell grain: (arch, slug, shape_point); path A first
A "cell" = a unique `(arch, slug, shape_point)` tuple (the existing join grain). Reach >=50
admitted sm_120 cells **path-A first**: depth-first over additional shape_points of the ~14
already-built `portable_lcd` slugs (trace + NCU label + CUPTI duration + merge only — no new
kernels), which lifts 30 -> ~61 candidates. The inventory has 59 distinct slugs / 169
candidate cells; 139 unadmitted; 31 candidates sit on already-built binaries. Preserve the
N>=5 / 0.03-stddev repeatability gate and family-bucket distribution invariants
(memory_stress / tensor_compute / activation_norm each >=6, <=35% per bucket); add per-slug
caps and a slug-held-out validation split so depth does not collapse diversity. If the
gates cannot net 50, then author/rewrite kernels (as the user explicitly permitted).

### D8 — Architecture scope: sm_120-only (Verified-live-sm_120)
All three requirements are scoped to the live RTX 5060 laptop (sm_120). sm_86 traces cannot
be produced on this host; the matching sm_86 work is deferred to a separate RTX 3090 Ti
session. Closure is `Verified-live-sm_120`; no cross-arch criterion gates it.

### D9 — L1 prediction requires the L1->L2 miss-filter first
If broadening re-widens `comparison_contract.toml` `cache_layers` back to `[L1, L2]`, the
documented **L1 -> miss-stream -> L2 filter chain** must be built FIRST (per-SM/CTA
scheduling partition keyed on `sm_id`/`warpid_sm`, plus `.ca`/`.cg` cache-op eligibility),
because NCU's L2 sees only the L1-miss stream and the current full-stream replay
systematically over-predicts L2. This filter is a trace-normalization / cache-accounting
layer, NOT a scheduler or timing simulator. Predict on the **sector basis** first; request
counts and transactions-per-request are secondary targets layered on afterward.

## Reusable substrate (build on, do not reinvent)

- Wall-clock ground truth: `cupti_activity_collector.cu`, `cupti_activity_sidecar.schema.json`,
  `run_suite.py --mode timing` (dual-source, reconciled).
- Generic per-kernel MAPE scorer: `compute_validation_metric.py::per_kernel_mape` (its
  docstring already names wall-clock duration as an example target).
- Dependency-tiered calibration engine: `monotonic_calibration_head.py`
  (Isotonic -> monotone GAM -> monotone GBM -> XGBoost; `PerBucketCalibratorEnsemble` keyed
  on `(arch, family_bucket)`; SHA-256 deterministic folds).
- Metric-agnostic accuracy-gate / falsification / advisory-validation harness under
  `tools/cache_replay/`.
- Feature substrate: `canonical_features.py` 6-feature contract plus an unconsumed pool
  (NVBit inst_count, divergence/MSHR proxies, NVML clock/thermal/power/throttle covariates
  with `nvml_covariate_join.py`, the `%clock64` `phase_cycle_ledger`).
- Cohort machinery: `tools/local_labels/` (`check_admission_floor.py`,
  `inventory_cohort_candidates.py`, `expand_cohort_manifest_from_catalog.py`,
  `merge_admitted_cells.py`, `repeatability_aggregator.py`), `sm120_cohort_inventory.json`.
- Local NCU label driver: `tools/local_labels/label_driver.sh` (the only sanctioned NCU
  invocation; quarantined to `data/local_labels/`).

## Recommended phasing (one coherent plan)

1. Policy/prose cleanup: reframe "simulator predicting wall-clock" schema prose to
   "predictor/model estimating wall-clock"; consider a static Rule-5 wording guard for the
   new surfaces.
2. Define the target contracts: a new `duration_admitted.json` schema (CUPTI wall-clock) and
   a separate cache/memory metric allowlist + a typed (fraction/count/ratio) NCU label path.
3. Expand the sm_120 cohort to >=50 `(slug, shape_point)` cells via already-built
   portable_lcd candidates first; preserve N>=5 + repeatability + family-bucket gates.
4. Implement admitted CUPTI wall-clock duration labels + train the first `duration_ns`
   predictor (NVBit + NVML covariate features, dependency-tiered calibration), scored by
   per-kernel MAPE with an accuracy gate + falsification arm.
5. Restore L1/L2 sector-basis cache prediction only after the L1-miss-stream->L2 filter is
   in place (fixes the systematic L2 over-prediction).
6. Add the selected cache/memory count + ratio targets via the new non-fraction parser path.
7. Defer the matching sm_86 campaign to a separate RTX 3090 Ti session.

## Pending decision to surface (do not block)

Only one product-scope question remains genuinely open: whether ">=50 cells" must mean
">=50 DISTINCT slugs" for publication positioning (vs the repo-native `(arch, slug,
shape_point)` grain, which path A satisfies). The repo schema + inventory support the cell
= `(arch, slug, shape_point)` definition; if reviewers later require distinct slugs, build
the unbuilt L2 simbench corpus as a follow-on.

## Non-negotiable constraints

- NCU only on local sm_86/sm_120 into `data/local_labels/`, driven only from
  `tools/local_labels/` (Rule 1). No driver-root / clock-locking (Rule 2). No privileged
  CUPTI; CUPTI Activity allowed (Rule 3). No host-driver work (Rule 4). No simulator
  revival or simulator framing (Rule 5).
- Any new metric/label/field is a coordinated single-commit lockstep change across the
  driver argv, parser-to-sidecar mapping, sidecar emitter, schema, import denylist +
  aggregator, and comparison contract + runner + lockstep test (consumer-coupled wire
  policy).
- Evidence status for closure is `Verified-live-sm_120`.

## Primary Direction: Wall-Clock Duration Predictor

### Rationale

Treats requirement #1 (the execution-time predictor) as the load-bearing deliverable and spine that the cohort and cache-metric work feed, distinct from the cache-metric and cohort angles.

### Approach Summary

Build a nanosecond-scale kernel execution-time predictor fitted on the already-collected CUPTI Activity dual-source duration measurement (`duration_ns` reconciled against `cudaEventElapsedTime`), as a distinct second prediction claim alongside the existing L2-hit-rate model. The design has four layers, each reusing proven machinery:

1. **Ground-truth label layer (`duration_admitted.json`).** Add a CUPTI-provenance repeatability aggregator that mirrors `repeatability_aggregator.py`'s contract (N>=5 repeats per `(arch, slug, shape_point)` cell, a CV ceiling analogous to the 0.03 stddev L2 gate, dual-source reconciliation must pass) and emits an admitted-duration artifact with full provenance (CUPTI sidecar hash, `binary_sha256`, `source_commit`, NVML covariate linkage). Wall-clock labels stay strictly non-root (CUPTI Activity + CUDA events), never NCU `gpu__time_duration.sum`.

2. **Feature -> nanoseconds regression head.** Feed the target-agnostic `canonical_features.py` vector plus NVML clock/thermal/power covariates (`nvml_covariate_join.py`) into the existing `monotonic_calibration_head.py` hierarchy (Isotonic -> monotone GAM -> monotone GBM -> XGBoost; `PerBucketCalibratorEnsemble` keyed on `(arch, family_bucket)`), retargeted to predict `duration_ns` instead of an L2 fraction. The only mechanical change is generalizing the model classes' hard-coded `[0,1]` clip to a per-target bound.

3. **Scoring + gate + falsification.** Reuse `compute_validation_metric.py::per_kernel_mape` (its docstring already names wall-clock duration as an example target) and route the predictions through the metric-agnostic `tools/cache_replay/` accuracy-gate + falsification + advisory-validation harness, so the timing claim closes by EITHER a MAPE accuracy gate OR an honest engineering-closure falsification report.

4. **DVFS conditioning.** Apply the proven `nvml_covariate_join.py` residual-regression pattern (thermal/power/clock correction coefficients fitted on per-launch NVML samples aligned within the kernel lifetime window) to the duration residual, gated by `nvml_covariate_acceptance.toml` thresholds. Predict nanoseconds with DVFS state as conditioning covariates rather than converting to cycles.

The work stays inside Rule 5: an empirical feature->scalar regression on measured durations, with no instruction-issue / warp-scheduling / DRAM-timing / pipeline / runtime simulation, and no roofline-as-primary-model. Affected components: a new `duration_admitted.json` schema + aggregator under `tools/local_labels/`, a duration sibling of the calibration head under `workloads/simbench_bw/`, and the `tools/cache_replay/` gate/falsification harness keyed on the new target.

### Objective Evidence

- `workloads/simbench_bw/cupti_activity_collector.cu` — dual-source `duration_ns` collection with the canonical reconciliation formula and sidecar emission already operational.
- `workloads/simbench_bw/schemas/cupti_activity_sidecar.schema.json` — `duration_ns`, `cuda_event_wall_time_ms`, `dual_source_reconciliation_gap_ns`, `dual_source_reconciliation_passed` defined since schema_version 1.
- `workloads/simbench_bw/run_suite.py::handle_timing` — timing orchestration with per-repeat CUPTI sidecar emission, dual-source classification, and a typed exit code (`EXIT_CUPTI_DUAL_SOURCE_NO_PASSING_REPEATS`) for no-passing-repeats.
- `workloads/simbench_bw/compute_validation_metric.py::per_kernel_mape` — a generic (measured, predicted) -> MAPE scorer; reusable for `duration_ns` by swapping the measured-dict key.
- `workloads/simbench_bw/monotonic_calibration_head.py` — target-agnostic Isotonic/GAM/GBM/XGBoost per-bucket ensemble; only the `[0,1]` clip is hard-coded and easily parameterized.
- `workloads/simbench_bw/canonical_features.py` — the 6-feature canonical vector is fully decoupled from the L2 target; monotonic constraint signs are a parameterizable dict.
- `workloads/simbench_bw/nvml_covariate_join.py` (+ `nvml_covariate_acceptance.toml`) — the per-launch NVML covariate residual-regression pattern, directly applicable to duration residuals.
- `tools/local_labels/repeatability_aggregator.py` — the N>=5 + stddev-gate cell-aggregation contract to mirror for a CUPTI-provenance duration aggregator.
- `tools/cache_replay/` (`accuracy_gate.py`, `falsification.py`, `validation.py`, `artifact_io.py`) — metric-agnostic gate/falsification/advisory harness keyed by any per-kernel scalar target.

### Known Risks

- CUPTI Activity `duration_ns` requires `--enable-cupti-activity`; falls back to `cuda_event_wall_time_ms` via the dual-source hard-gate, with typed exit codes already defined for unavailability.
- NVML alignment gaps: short-duration kernels may have zero NVML samples in the lifetime window, leaving the residual correction unavailable for those cells.
- A new `duration_admitted.json` schema + a parallel CUPTI-provenance aggregator are net-new surfaces; provenance must stay cleanly separated from the NCU-cache label artifact.
- Monotonic constraint signs and gate thresholds for nanoseconds are empirically unknown and must be tuned/validated on the >=50-cell cohort before the duration gate is load-bearing (overfitting risk on a small per-bucket cohort).
- The wall-clock predictor sits directly on the Rule-5 interpretive line (no static guard exists); the empirical-regression framing and "predictor not simulator" prose must be explicit.

## Alternative Directions Considered

### Alt-1: Tri-Label Cohort Scale-Out
- Gist: Make the >=50-cell cohort the organizing deliverable — a tri-label (wall-clock + L1 + L2) evidence base. Extend the CUPTI dual-source sidecar + label schema to carry duration alongside the existing L2/L1TEX hit rates, extend `repeatability_aggregator.py` to gate and persist all three targets per cell, and run a path-A trace+label+CUPTI-duration collection campaign over the already-built `portable_lcd` slugs (more shape_points) plus simbench L2 rows, allocating by family bucket to satisfy the distribution invariants and lift 30 -> 50+ admitted cells.
- Objective Evidence:
  - `tools/local_labels/sm120_cohort_inventory.py` + `data/local_labels/sm120_cohort_inventory.json` — 169 candidate `(slug, shape_point)` cells classified into family buckets; only 30 admitted.
  - `tools/local_labels/check_admission_floor.py`, `merge_admitted_cells.py`, `expand_cohort_manifest_from_catalog.py`, `sm120_cohort_roster_validator.py`, `sm120_cohort_buildability_preflight.py` — the operational admission + roster pipeline.
  - `tools/local_labels/repeatability_aggregator.py` — N>=5 + stddev<=0.03 per-target gate; already emits `label_repeatability.json` / `label_admitted.json` / `label_excluded.json`.
  - `workloads/portable_lcd/` — all ~15 LCD-portable kernels have built `<slug>_run` binaries (sm_86/sm_90/sm_120 arch-gated), so path-A needs trace+label+merge only.
  - `workloads/simbench_bw/schemas/local_label_sidecar.schema.json` — currently L2 (mandatory) + L1TEX (optional); no duration field yet (expansion point).
- Why not primary: It is a data-production campaign whose value is realized only once the predictors it feeds exist, so it is better sequenced as the evidence base under the wall-clock predictor rather than the headline contribution.

### Alt-2: Unified Multi-Target Prediction Framework
- Gist: Generalize the predictor stack into one framework where a single shared 6-feature vector feeds a registry of per-target calibration heads (target -> {unit, bounds, monotonic constraint signs, gate thresholds, label source}), emitting `{duration_ns, L1/L2 hit rates, cache counts/ratios}` together. Parameterize the model classes' `[0,1]` clip into per-target bounds, extend cell records from a single `label` to a `labels` dict, instantiate one `PerBucketCalibratorEnsemble` per target, and move gate thresholds + the comparison-contract metric pin into the target registry.
- Objective Evidence:
  - `workloads/simbench_bw/monotonic_calibration_head.py` (`PerBucketCalibratorEnsemble`) — almost entirely target-agnostic; the only target coupling is the `[0,1]` clip in the IR/GAM predict methods.
  - `workloads/simbench_bw/compute_validation_metric.py` (`MetricRegistry`, lines ~143-172) — a proven in-repo registry pattern to mirror as a `TargetRegistry`.
  - `workloads/simbench_bw/canonical_features.py` — the 6-feature vector and `MONOTONIC_CONSTRAINT_SIGNS` dict are target-decoupled and parameterizable per target.
  - `workloads/simbench_bw/comparison_contract.toml` — the single `[ncu_metric]` pin + `[0,1]`-bounded error formula is the refactor surface for a multi-target overlay.
  - `tools/cache_replay/accuracy_gate.py` — hard-coded hit-rate thresholds to move into a per-target registry.
- Why not primary: It is the largest-surface, most-speculative angle (~800-1000 new LOC + cross-target fold-stratification and per-target threshold/monotonicity unknowns), so it is better adopted incrementally once the first new target (duration) proves the abstraction.

### Alt-3: Typed NCU Metric Registry And Wire Extension
- Gist: Center the work on the label/parser/schema extension machinery — a typed metric registry (`tools/local_labels/ncu_metric_registry.py`) declaring each metric as a dataclass (`metric_name`, `metric_type` fraction|count|ratio, `cache_layer`, sidecar/admitted field names, optional status enum, raw-counter fallback pair, CI units, validation range) that drives all ~7 coupled wire surfaces (parser, driver `--metrics`, sidecar emitter, both schemas, comparison contract, comparison runner/aggregator, lockstep test) from one declaration, so adding any cache/memory metric is config-driven and lockstep-safe.
- Objective Evidence:
  - `tools/local_labels/parse_ncu_metrics.py` (lines ~102-111) — the hardcoded 2-entry `_PCT_TO_RAW_COUNTER_PAIR` + percentage-only range validation to replace with a registry-driven typed path.
  - `tools/local_labels/label_driver.sh` (~line 765) — hardcoded `--metrics` string; the registry would generate it / accept a `--metric-set` filter.
  - `workloads/simbench_bw/schemas/local_label_sidecar.schema.json` + `local_label_admitted.schema.json` — closed (`additionalProperties:false`) schemas with the L1TEX `dependentRequired` + `l1tex_label_status` enum coupling pattern.
  - `tools/local_labels/test_metric_name_lockstep.py` — the single-metric lockstep enforcement to parameterize.
  - Git precedent: commit 877119a0 (L1TEX v3->v4 additive migration) and c9579606 (`source_commit_override_from` additive-no-bump) show how additive wire extensions land.
- Why not primary: It is enabling infrastructure for requirement #3 rather than a headline predictive result, so it is better motivated and validated by the first concrete new metric it carries than pursued as the lead.

### Alt-4: L1 To L2 Miss-Filter Trace Normalization
- Gist: Fix the documented dominant failure mode — the full-stream replay over-predicts L2 because NCU's L2 observes only the L1-miss stream — by building a per-SM/CTA-partitioned L1 filter that sits between trace decode and L2 replay and emits only L1-miss accesses into the L2 model. Reuse the existing per-SM fully-associative L1 walker + CTA-to-SM scheduling, preserve trace-walk order and inter-launch carryover, and respect `.ca`/`.cg` cache-op eligibility. Framed explicitly as trace-normalization / cache-accounting, not scheduler or timing simulation.
- Objective Evidence:
  - `data/local_labels/cache_replay_falsification.json` — documents trace-fidelity as the dominant failure mode (16/30 cells over-predicted by >=0.1; mean signed over-prediction ~0.34) and prescribes the L1-miss-stream->L2 filter.
  - `tools/cache_predictor/l1_walker.py` (`_walk_one_launch_fa`, `block_to_sm_id`) — a completed per-SM fully-associative L1 model with CTA-to-SM scheduling to reuse as the filter core.
  - `tools/cache_replay/sector_cache_replay.py` (`decode_cell_access_stream`, `replay_access_stream`) — the monolithic global-L2 replay the filter would feed a filtered stream.
  - `tools/nvbit_tracer/.../instruction.proto` — optional `sm_id` / `warpid_sm` / `warpid_tb` + `instruction_cache_op` per-access metadata enabling per-SM partitioning + cache-op eligibility.
  - `tools/hrd_profiler/observation_builder.py` (`LaunchSurvivingState`, stable TB walk-order) — the per-launch carryover + ordering precedent.
- Why not primary: It is the prerequisite that makes honest L1/L2 cache prediction possible (D9) but is orthogonal to the wall-clock predictor and gates only the cache-side of the program, so it is better sequenced as the cache-prediction enabler than the overall lead.

### Alt-5: Multi-Claim Falsification Harness
- Gist: Organize the expansion around scaling the single-claim closure infrastructure to N independent per-target closures: each of wall-clock, L1, and each cache metric gets its own accuracy gate, advisory validation artifact, and falsification arm (with the three-way trace-fidelity/model-parameter/residual-calibration decomposition and a confirmed/refuted verdict), aggregated into a unified falsification registry where every target's engineering closure stays terminal on the single sm_120 host. Generalize `accuracy_gate.py` thresholds into a per-target registry and extract the validation/diagnostic machinery into target-agnostic bases.
- Objective Evidence:
  - `tools/cache_replay/falsification.py` — `build_falsification_report` already splits research-success vs engineering-closure with a three-way failure-mode decomposition to generalize per target.
  - `tools/cache_replay/accuracy_gate.py` — pure metric-scalar gate evaluator to parameterize by target.
  - `tools/cache_replay/validation.py` — target-agnostic LOOCV/LOSO/bootstrap helpers to extract into a shared base.
  - `tools/cache_replay/l1_diagnostic.py` — the metric-specific load-bearing-guard precedent (`load_bearing` flag + promotion bar) for new diagnostic targets.
  - `tools/sm_86_predictor_accuracy_recovery_v2/publication_gate.py` — a multi-criterion gate aggregating independent verdicts under OR-semantics, the precedent for a unified registry verdict.
  - `data/local_labels/cache_replay_falsification.json` — the committed honest-negative (refuted) closure precedent.
- Why not primary: It is the evidence-discipline scaffolding around the predictors rather than a predictor itself, so it is best applied once there are multiple concrete targets (duration + cache metrics) to close, with the wall-clock claim as the first instance.

## Synthesis Notes

The primary (Wall-Clock Duration Predictor) is the user-prioritized headline and the cleanest high-confidence build, but it is most powerful as the spine of a sequenced program that folds in the alternatives in dependency order. Alt-1 (Tri-Label Cohort Scale-Out) supplies the >=50-cell evidence base the duration predictor (and every cache metric) is scored on, so it is the immediate enabler folded under the primary. Alt-3 (Typed NCU Metric Registry) is the wire machinery that makes requirement #3's broader cache metrics — and the new duration label — land lockstep-safe; it should be adopted the moment the duration label proves the additive-extension pattern. Alt-4 (L1->L2 Miss-Filter) is the hard prerequisite (D9) for any honest L1/L2 cache-metric claim and should precede re-widening `cache_layers`, but it is independent of the timing spine. Alt-5 (Multi-Claim Falsification Harness) generalizes the existing closure discipline so each new target (duration first, then each cache metric) closes by research-success-or-honest-falsification, preventing the small-cohort overfitting and threshold-tuning risks the primary flags. Alt-2 (Unified Multi-Target Framework) is the eventual convergence of all of these into one registry-driven predictor; it is best deferred until the duration target has validated the per-target abstraction empirically, so that the framework generalizes from two proven targets rather than one. A plan built from the primary would therefore sequence: cohort scale-out + duration label/predictor first, then the typed-registry + L1-filter enabling the broader cache metrics, with the falsification harness and unified framework as the closing discipline and architecture.

--- Original Design Draft End ---
