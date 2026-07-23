# sm_86 Unified Multi-Metric Non-Root GPU Predictor (Execution Time + L1/L2 Hit Rates + Memory Traffic)

## Goal Description

Extend the repository's single live deliverable — a non-root L2-sector-hit-rate predictor (NVBit reuse-distance walker → frozen v5 monotonic calibration head, scored by `comparison_runner.py`) — into a **UNIFIED multi-metric non-root predictor** covering, on the RTX 3090 Ti (sm_86) only:

1. **kernel EXECUTION TIME** — predict `gpc__cycles_elapsed.max` (cycles, DVFS-stable) as the primary target and derive `gpu__time_duration.sum` (ns) via a read-only NVML SM-clock covariate at inference (cycle-MAPE + ns-MAPE reported);
2. **L1 + L2 cache HIT RATES** — promote L1 (`l1tex__t_sector_hit_rate.pct`, plus L1 read-hit) from appendix-only diagnostic to a load-bearing predicted+validated metric alongside the existing L2 (`lts__t_sector_hit_rate.pct`);
3. **memory-TRAFFIC COUNTS** — predict GMEM/L2/DRAM requests, sectors, and transactions (plus the probe-confirmed subset of atomics/reductions and L1/L2 total requests).

All targets are validated against NCU ground-truth labels produced **only** inside the `tools/local_labels/` → `data/local_labels/` quarantine on this local card. The cohort is expanded to **≥50 admitted sm_86 cells**, where every admitted cell carries wall-clock + L1 + L2 labels and passes the repeatability gate. The work is organized as one RLCR loop with internally sequenced milestones (M0 label-substrate+L2-freeze → M1 L1-promotion → M2 execution-time → M3 cohort→50 → M4 traffic → closure), per the operator's decision to run the full program in a single loop. All work targets `Verified-live-sm_86`; cross-arch / sm_120 work is out of scope on this machine (standing operator directive) and recorded under Future Work.

The architecture is the Codex-ratified **unified core + modular heads** design: one shared ingestion/feature/error-budget layer (NVBit address+SASS traces, static SASS, launch geometry, read-only NVML covariates, CUPTI Activity timing, NCU labels) feeding modular per-metric heads. The existing L2 v5 calibration head is **frozen and untouched**; new metrics use either a bounded `[0,1]` regression head (fractions: L1 hit rate) or a log-target nonnegative head (counts/ns/cycles). The NCU label pipeline gains per-metric unit tags (`{unit: pct|count|ns|cycles}`) and bumps the admitted schema `4→5` in lockstep across producer + sidecar schema + admitted schema + aggregator + parser + `import_label`.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. The `AC-*` items are the current RLCR completion gates for this loop. (CLAUDE.md Rules 1-5 — NCU quarantine, no clock-lock, no privileged CUPTI, no host-driver, no simulator — are binding throughout.)

- AC-1: **Schema-v5 label substrate + metric registry, landed in lockstep.** A committed metric registry enumerates every target metric's canonical name, `unit ∈ {pct, count, ns, cycles}`, valid range, parser transform, aggregation rule, nullable policy, and deployability status (label-only vs feature-eligible). The sidecar schema and admitted schema bump to `schema_version 5` and the producer (`label_driver.sh`/`sidecar_emitter.py`), parser (`parse_ncu_metrics.py`), aggregator (`repeatability_aggregator.py`), and `import_label.py` accept the unit-tagged records.
  - Positive Tests (expected to PASS):
    - A v5 fixture carrying L1/L2 (pct), `gpc__cycles_elapsed.max` (cycles), `gpu__time_duration.sum` (ns), and a traffic count (count) validates against both v5 schemas and round-trips through producer → parser → aggregator → `import_label`.
    - The metric registry resolves each target metric to its unit + transform; `import_label` admits the new fields under label-style names.
  - Negative Tests (expected to FAIL):
    - A mixed v4/v5 artifact (a v4 record fed to a v5 consumer, or vice versa) is rejected with a typed error.
    - A `count`/`ns`/`cycles` value pushed through the percentage `/100` path is rejected; a field named `metric_*`/`counter_*` trips the `import_label` denylist.
  - AC-1.1: The percentage path is unchanged for L1/L2 hit rates (still `/100`, `>105%` rejected); only `count`/`ns`/`cycles` metrics use the unit-tagged non-percentage path.

- AC-2: **The frozen L2 v5 result stays green.** The committed L2 v5 comparison output, coverage gate, bootstrap CI, accuracy-gate headline, and `sm_86_v5_head.json` reproduce the pre-change baseline after the schema-v5 wrapping and the L1 contract widening.
  - Positive Tests: re-running the L2 v5 fit + comparison after all M0/M1 changes reproduces the committed L2 head (byte-identical model fields) and the L2 headline geomean within `1e-9`.
  - Negative Tests: any change that alters an L2 join key, an L2 score, or the frozen L2 head fails a regression guard.

- AC-3: **NCU metric-availability probe artifact for this sm_86 host.** A committed probe artifact (produced under `tools/local_labels/`) records, for every target metric in `read_ncu_report.py`, the exact NCU metric ID, unit, aggregation/rollup, and availability on this card (3090 Ti, `ncu_2026.1.x`). `get_L1_Total_Requests` is resolved to a concrete metric ID and the `gpu__time_duration.sum` spelling is confirmed.
  - Positive Tests: the probe artifact lists each target metric with a resolved NCU ID + unit + availability; `get_L1_Total_Requests` and `gpu__time_duration.sum` are pinned to confirmed IDs.
  - Negative Tests: a target metric unavailable/renamed on sm_86 is recorded as `excluded` with a typed reason, NOT stubbed with a fabricated value; a predictor head that targets an unprobed/unavailable metric fails.

- AC-4: **No-deployability-leakage guard.** Theoretical occupancy / launch limits (static-derivable) may be predictor input features; achieved occupancy (`sm__warps_active...pct_of_peak_sustained_active`) and IPC (`sm__inst_executed.avg.per_cycle_elapsed`) appear ONLY as labels/diagnostics, never as deployable inference inputs.
  - Positive Tests: a leakage test confirms theoretical-occupancy features are allowed and that no inference feature file contains achieved occupancy or IPC.
  - Negative Tests: an inference feature file containing achieved occupancy or IPC fails the leakage test.

- AC-5: **L1 hit rate promoted to load-bearing.** The submission contract widens `cache_layers=[L2]` → `[L1TEX, L2]`, the APPENDIX-ONLY `SCOPE_DOCBLOCK` guard is retired (replaced by a guard that still rejects an *unvalidated* L1 claim), and a SEPARATE bounded-`[0,1]` L1 calibration head INSTANCE (the `PerBucketCalibratorEnsemble` pattern + monotonic constraints, independent fit, leaving the L2 v5 head byte-frozen; per DEC-8) produces predictions scored by the comparison runner with bootstrap CI and L1 closure artifacts under the smoke-5 / publication-10 coverage floor (DEC-10).
  - Positive Tests: `[L1TEX, L2]` labels import, are scored, and emit L1 closure artifacts; an admitted cell's L1 prediction carries a bootstrap CI; the static-source scan no longer fails on a current-tense, *validated* L1 prediction claim.
  - Negative Tests: an admitted cell missing its L1 label fails admission; an L1 prediction *asserted* without being scored by the runner fails the replacement guard.

- AC-6: **NVML covariate reference operating point calibrated.** A committed calibration sweep artifact records the reference SM clock (and temp/power reference), the sampling method, accepted kernels, and rejected samples; the covariate regression is no longer inert.
  - Positive Tests: the calibration artifact exists with a populated reference operating point; the ns conversion consumes the calibrated read-only NVML clock.
  - Negative Tests: a missing/implausible NVML clock sample prevents the ns-headline conversion (fails closed) rather than silently substituting a default.

- AC-7: **Execution-time labels captured and unit-separated.** Admitted cells carry `gpc__cycles_elapsed.max` (cycles) and `gpu__time_duration.sum` (ns) as unit-tagged v5 labels, each passing the per-unit repeatability gate.
  - Positive Tests: a timing-roster admitted cell carries both timing labels with per-unit repeatability metadata.
  - Negative Tests: a cycles/ns label routed through the percentage parser path is rejected; a timing-roster kernel missing its timing label fails timing admission.

- AC-8: **Execution-time predictor (roofline analytical prior + ML residual).** A new timing runner (reusing `comparison_runner.py`'s 7-field join, coverage gate, and bootstrap) scores a predictor that emits a cycles prediction from a roofline analytical prior (work/bytes/sectors/cache/active-blocks/bottleneck from NVBit + static SASS + launch geometry) plus an ML residual, converted to ns via the calibrated NVML clock. Reports cycle-MAPE + ns-MAPE with bootstrap CI over the headline roster, under the smoke-5 / publication-10 coverage floor (DEC-10).
  - Positive Tests: the timing runner emits cycle-MAPE + ns-MAPE + bootstrap CI + per-arch coverage over the headline cohort; the head's roofline term and residual term are both recorded in provenance.
  - Negative Tests: the predictor consuming a label-only feature (achieved occupancy / IPC) fails (ties to AC-4); a timing prediction reported without passing the coverage gate fails.

- AC-9: **Timing headline noise gate (>200µs hard exclusion; appendix-allowed).** Kernels with NCU duration `>200µs` and sufficient repeats form the headline timing cohort; kernels `≤200µs` are excluded from the headline cycle/ns MAPE but emitted as appendix/diagnostic with a noise-floor caveat (the ~110µs dual-source p99 floor dominates relative error below 200µs).
  - Positive Tests: a `>200µs` kernel is in the headline timing MAPE; a `≤200µs` kernel appears only in the appendix block.
  - Negative Tests: a `≤200µs` kernel counted in the headline timing MAPE fails the gate.

- AC-10: **`%clock64` phase-cycle ledger is ablation-only.** The timing closure report compares accuracy with and without the `%clock64`-derived features; the ledger is an optional feature/ablation, never a silent required headline input.
  - Positive Tests: the closure report contains a with/without-`%clock64` ablation; the headline timing metric is produced with the ledger absent.
  - Negative Tests: a configuration in which a `%clock64`-derived value is a silent required headline input fails.

- AC-11: **Cohort expanded to ≥50 admitted sm_86 cells (HARD GATE).** ≥50 sm_86 cells reach `admitted` status, each with consistent `binary_sha256`, `n_repeats≥5`, a repeatability pass under the per-unit criteria, AND carrying wall-clock + L1 + L2 labels. The 6 `label_provenance_duplicate` exclusions are repaired or excluded with typed reasons; the 18 `label_repeatability_failed` exclusions are triaged (clustered tunable kernel-level vs systemic device-level) and either re-admitted or excluded with typed reasons; the true count of sm_86-capable roster kernels lacking cells is re-derived from `roster.toml` ARCH_GATE tokens + `label_admitted.json`; new/rebuilt kernels are authored as needed.
  - Positive Tests: the admitted set has ≥50 sm_86 cells, each carrying wall-clock + L1 + L2 and passing the per-unit repeatability gate; a coverage scanner confirms the count and the all-three-labels invariant.
  - Negative Tests: `<50` admitted sm_86 cells, OR any admitted cell missing wall-clock/L1/L2, OR an admitted cell failing the per-unit repeatability gate, fails closure.
  - AC-11.1: Per-unit repeatability criteria (DEC-7) — fractions keep `stddev ≤ 0.03`; cycles/ns use coefficient-of-variation `CV ≤ 0.05`; counts use relative-tolerance `≤ 0.05`. These provisional values are finalized from the M3 calibration sweep and may be loosened (documented) ONLY if the ≥50 hard gate is at risk.

- AC-12: **Memory-traffic count predictors.** Log-target nonnegative heads predict the AC-3-probe-confirmed mandatory traffic-metric subset (GMEM load/store requests + sectors, DRAM sectors, L2 total requests), using a `log1p` transform with explicit zero handling and inverse-transform error reporting, scored with bootstrap CI under the smoke-5 / publication-10 coverage floor (DEC-10). Atomics/reductions + sectors-per-request ratios are deferred to FUT-3 (DEC-9).
  - Positive Tests: zero and nonzero count labels score correctly through the log-target head; the traffic runner emits per-metric error + bootstrap CI over the probe-confirmed subset.
  - Negative Tests: a negative or fractional count label is rejected; a traffic metric not confirmed by the AC-3 probe is not silently stubbed.

- AC-13: **This loop emits complete closure artifacts for every new head.** For each of the L1, execution-time, and traffic heads, this loop emits an admitted manifest, an excluded manifest with typed reasons, comparison outputs, a coverage report, a bootstrap CI artifact, the config, the git SHA, and the schema version.
  - Positive Tests: each new head's closure artifact set is present and schema-valid.
  - Negative Tests: a missing closure artifact for any new head fails RLCR closure.

- AC-14: **Compliance gate (sm_86-only; CLAUDE.md Rules 1-5).** The no-root/no-NCU static guard and the plan-token scanner pass; NCU invocation stays within `tools/local_labels/` → `data/local_labels/`; no privileged CUPTI / clock-lock / host-driver / simulator surface is introduced; round summaries record `Verified-live-sm_86` (or `Offline-verified`), never `Cross-arch-verified`.
  - Positive Tests: `test_no_root_or_ncu_static_guard.py` + `test_plan_token_scanner.py` pass; the sm_120/cross-arch claim guard passes.
  - Negative Tests: any NCU outside the quarantine, any privileged CUPTI/clock-lock op, any sm_120/cross-arch claim, or any plan-token in implementation code fails the relevant guard.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
All 14 acceptance criteria met: the unified core + modular-heads substrate is in place; L1 hit rate, execution time (cycles + ns), and the mandatory traffic-count subset are predicted and validated against NCU on ≥50 admitted sm_86 cells each carrying wall-clock + L1 + L2 labels; the schema-v5 unit-tagged label pipeline is landed in lockstep; the frozen L2 v5 result remains green; every new head emits full closure artifacts; and the whole loop is `Verified-live-sm_86` with the compliance gate clean.

### Lower Bound (Minimum Acceptable Scope)
Because the operator chose to run the full program in one loop with ≥50 cells as a hard gate, the lower bound largely coincides with the upper bound: schema-v5 lockstep + frozen-L2 green (AC-1, AC-2) + NCU probe + leakage guard (AC-3, AC-4) + L1 promotion (AC-5) + NVML calibration + execution-time predictor over the >200µs headline cohort (AC-6–AC-10) + ≥50 admitted cells with all-three labels (AC-11) + at least the mandatory traffic-count subset predicted (AC-12) + closure artifacts (AC-13) + compliance (AC-14). The only documented give: the traffic head may launch with the mandatory subset only, deferring the extended subset (atomics/reductions, sectors-per-request) to Future Work per DEC-9.

### Allowed Choices
- Can use: reuse of `comparison_runner.py`'s join/coverage/bootstrap (hybrid) plus NEW timing and traffic runners; NEW bounded-`[0,1]` and log-target heads (the v5 `PerBucketCalibratorEnsemble` pattern, frozen for L2); the schema-v5 unit-tag mechanism; NVBit address+SASS traces, CUPTI Activity, read-only NVML, static SASS via `cuobjdump`/`nvdisasm`, the `%clock64` ledger (ablation/feature); a roofline analytical prior + ML residual for timing; NCU on this local 3090 Ti for labels ONLY into `data/local_labels/` via `tools/local_labels/`.
- Cannot use: privileged CUPTI (Events/Metrics/Range/PM/PerfWorks); NCU outside the quarantine; `nvidia-smi` clock/power/persistence/MIG mutation or clock-locking; host-driver install/DKMS/MOK/reboot; simulator revival; any change to the frozen L2 v5 head's scoring semantics; achieved occupancy or IPC as deployable inference inputs; any sm_120 / cross-arch claim or artifact on this machine.

> **Note on Deterministic Designs**: The architecture (unified core + modular heads, freeze L2, cycles-primary timing, roofline+residual, schema-v5 unit tags, bounded vs log-target heads) is fixed per the draft and the Codex consult; the path boundaries are correspondingly narrow. Remaining latitude is confined to the PENDING decisions (DEC-7–DEC-10).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only — conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. **Label substrate first (M0).** Add a metric registry + per-metric unit tags; bump both schemas to v5 and update producer/parser/aggregator/`import_label` in one lockstep change; keep the percentage path intact for fractions and add the non-percentage path for counts/ns/cycles. Land a frozen-L2 regression guard and a leakage guard, and run the NCU metric-availability probe to pin exact metric IDs (resolving `get_L1_Total_Requests`).
2. **L1 promotion (M1).** Widen `comparison_contract*.toml` `cache_layers` to `[L1TEX, L2]`, retire the appendix-only `SCOPE_DOCBLOCK` guard (replace with a "must-be-scored" guard), and fit a bounded-`[0,1]` L1 head reusing the calibration pattern.
3. **Execution-time predictor (M2).** Calibrate the NVML reference operating point via a local sweep; capture `gpc__cycles_elapsed.max` + `gpu__time_duration.sum` labels; build the roofline analytical prior + ML residual; score via a new timing runner with cycle-MAPE + ns-MAPE; gate the headline to `>200µs` kernels; keep `%clock64` as an ablation.
4. **Cohort to ≥50 (M3).** Re-derive sm_86-capable kernels lacking cells, repair the 6 provenance duplicates, triage the 18 repeatability failures, author/rebuild kernels, and re-sweep so ≥50 cells carry wall-clock + L1 + L2 and pass the per-unit repeatability gate.
5. **Traffic predictors (M4).** Log-target heads over the probe-confirmed mandatory traffic subset (GMEM/DRAM/L2 requests+sectors), with zero handling and inverse-transform error reporting.

### Relevant References
- `workloads/simbench_bw/comparison_runner.py` — 7-field join, coverage gate, bootstrap CI (reuse for hybrid timing/traffic runners).
- `workloads/simbench_bw/cache_predictor.py` + `tools/hrd_profiler/observation_builder.py` — the walker + per-launch structural intermediates (shared feature source).
- `workloads/simbench_bw/monotonic_calibration_head.py` + `tools/local_labels/fit_monotonic_calibration_head_v5.py` — the head pattern to freeze (L2) and clone (bounded / log-target).
- `tools/local_labels/parse_ncu_metrics.py`, `repeatability_aggregator.py`, `import_label.py`, `label_driver.sh`/`sidecar_emitter.py` — the label-pipeline lockstep surface.
- `workloads/simbench_bw/schemas/local_label_{sidecar,admitted}.schema.json` — the schemas to bump to v5.
- `workloads/simbench_bw/comparison_contract.toml` + `comparison_contract_sm86_submission.toml` — the contract whose `cache_layers` widens to `[L1TEX, L2]`.
- `workloads/simbench_bw/cupti_dual_source_live_admission.py` + `nvml_covariate_join.py` — timing-label substrate + covariate regression.
- `device_properties.json` — pinned sm_86 cache geometry (traffic-model inputs).
- `tools/baseline_runners/nugteren2014_rdh_baseline.py` — precedent analytical baseline.
- `read_ncu_report.py` — the target-metric spec to reproduce.
- `plans/sm86_cohort_regen_3090ti_plan.md` — exemplar single-metric RLCR loop structure.

## Dependencies and Sequence

### Milestones
1. **M0 — Label substrate + L2 freeze + probe + leakage** (AC-1, AC-2, AC-3, AC-4)
   - Phase A: metric registry + schema-v5 lockstep (producer/parser/aggregator/import_label).
   - Phase B: frozen-L2 regression guard; no-deployability-leakage guard.
   - Phase C: NCU metric-availability probe (resolve `get_L1_Total_Requests`, `gpu__time_duration.sum`).
2. **M1 — L1 promotion** (AC-5) — depends on M0 (schema + probe).
3. **M2 — Execution-time predictor** (AC-6, AC-7, AC-8, AC-9, AC-10) — depends on M0 (timing labels via v5 + NVML calibration); can score on currently-admitted >200µs kernels before M3 completes.
4. **M3 — Cohort → ≥50** (AC-11) — depends on M0/M1/M2 (every cell must carry wall-clock + L1 + L2).
5. **M4 — Traffic predictors** (AC-12) — depends on M0 (AC-3 probe confirms the metric subset).
6. **Closure** (AC-13, AC-14) — depends on all milestones; emits per-head closure artifacts and verifies the compliance gate.

Dependencies are by artifact, not time: the schema-v5 substrate (M0) underlies every new label family; the NVML calibration (M2) gates ns-headline conversion; the cohort gate (M3) consumes all label families; the traffic subset (M4) consumes the M0 probe.

## Task Breakdown

Each task carries exactly one routing tag (`coding` = Claude; `analyze` = Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Add the metric registry + per-metric unit tags; bump sidecar+admitted schemas to v5; update producer/parser/aggregator/import_label in lockstep; keep the pct path for fractions, add the non-pct path for counts/ns/cycles | AC-1 | coding | - |
| task2 | Build the frozen-L2 v5 regression guard (replay L2 head + comparison; assert byte-identical model fields + headline within 1e-9 after M0/M1 changes) | AC-2 | coding | task1 |
| task3 | Implement the NCU metric-availability probe under `tools/local_labels/` emitting a committed probe artifact (exact IDs/units/availability per target metric) | AC-3 | coding | task1 |
| task4 | Codex review: confirm `get_L1_Total_Requests` and `gpu__time_duration.sum` against the NCU catalog / source `ncu_report_helper.py`; ratify the probe's metric-ID resolutions and exclusion semantics | AC-3 | analyze | task3 |
| task5 | Implement the no-deployability-leakage guard test (theoretical occupancy feature-eligible; achieved occupancy + IPC label-only) | AC-4 | coding | task1 |
| task6 | Promote L1 to load-bearing: widen `cache_layers`→`[L1TEX,L2]`, retire the appendix-only `SCOPE_DOCBLOCK` guard (replace with a must-be-scored guard), fit the bounded-[0,1] L1 head, emit L1 closure | AC-5 | coding | task1, task2 |
| task7 | Codex review: L1-promotion contract change vs the frozen L2 (confirm no L2 join/score drift; ratify the replacement guard) | AC-5 | analyze | task6 |
| task8 | Calibrate the NVML covariate reference operating point via a local sweep; emit the calibration artifact; wire ns-conversion to fail closed on missing/implausible clock | AC-6 | coding | task1 |
| task9 | Capture `gpc__cycles_elapsed.max` + `gpu__time_duration.sum` labels (unit-tagged v5), per-unit-separated from the pct path | AC-7 | coding | task1, task3 |
| task10 | Build the execution-time predictor (roofline analytical prior + ML residual) + a new timing runner reusing comparison_runner join/coverage/bootstrap; report cycle-MAPE + ns-MAPE | AC-8 | coding | task8, task9 |
| task11 | Codex review: roofline-prior + ML-residual model design (falsifiable structure, feature set, error propagation, dependency ordering) | AC-8 | analyze | task10 |
| task12 | Implement the >200µs headline timing noise gate (≤200µs excluded from headline, appendix-allowed) | AC-9 | coding | task10 |
| task13 | Add the `%clock64` ablation-only path (with/without comparison; headline produces without the ledger) | AC-10 | coding | task10 |
| task14 | Expand the cohort to ≥50 admitted sm_86 cells: re-derive sm_86-capable kernels lacking cells, build/rebuild kernels + shape points, repair the 6 provenance duplicates, re-sweep so each cell carries wall-clock+L1+L2 and passes the per-unit repeatability gate | AC-11 | coding | task1, task6, task9 |
| task15 | Codex review: triage the 18 `label_repeatability_failed` exclusions by clustering (tunable kernel-level vs systemic device-level) and set the per-unit repeatability criteria | AC-11 | analyze | task14 |
| task16 | Build the memory-traffic count predictors (log-target/log1p heads, zero handling, inverse-transform error) over the AC-3-probe-confirmed mandatory subset; score with bootstrap CI | AC-12 | coding | task3, task4 |
| task17 | Emit complete closure artifacts (admitted+excluded manifests, comparison outputs, coverage report, bootstrap CI, config, git SHA, schema version) for the L1 / execution-time / traffic heads | AC-13 | coding | task6, task10, task16 |
| task18 | Verify the compliance gate: no-root/no-NCU static guard + plan-token scanner + sm_86-only/cross-arch claim guard | AC-14 | coding | task1, task6, task10, task16 |

## Future Work / Out of Scope

- FUT-1: **sm_120 cross-arch validation** (`Cross-arch-verified`, joint same-74 head, sm_120 L2-geometry pinning, cross-arch transfer). 
  - Source DEC: DEC-6
  - Current-loop handoff: AC-14 (this loop records `Verified-live-sm_86` only and asserts no cross-arch claim).
  - Promotion trigger: a development session on the separate RTX 5060 laptop (sm_120) host.
- FUT-2: **Achieved occupancy + IPC as PREDICTED targets** (this loop treats them as label-only / diagnostics).
  - Source DEC: DEC-4
  - Current-loop handoff: AC-4 (leakage guard keeps them out of inference inputs; they are captured as labels).
  - Promotion trigger: a follow-up loop that models counter-derived targets with a defensible non-root surrogate.
- FUT-3: **Extended traffic-metric subset** (atomics/reductions, sectors-per-request ratios) beyond the RESOLVED mandatory subset (GMEM load/store requests+sectors, DRAM sectors, L2 total requests).
  - Source DEC: DEC-9 (RESOLVED)
  - Current-loop handoff: AC-3 (the probe records availability of the extended metrics) + AC-12 (mandatory subset validated).
  - Promotion trigger: after the mandatory traffic subset validates and the probe confirms the extended metrics are stable on sm_86.

## Claude-Codex Deliberation

### Agreements
- Sequenced milestones are the correct shape (M0→M1→M2→M3→M4→closure), even when executed within a single loop, to preserve failure attribution.
- Unified core + modular heads; freeze the L2 v5 head; bounded-`[0,1]` head for fractions and log-target head for counts/ns/cycles.
- Cycles-primary timing target with ns derived via read-only NVML clock; roofline analytical prior + ML residual as the headline; `%clock64` as ablation only.
- Schema-v5 with per-metric unit tags landed in lockstep; the percentage parser path is preserved for fractions and a separate path handles counts/ns/cycles.
- Achieved occupancy + IPC are label-only (never deployable inputs); theoretical occupancy is a static feature.
- An NCU metric-availability probe must precede traffic-predictor commitments; unavailable metrics are excluded, not stubbed.
- The `>200µs` headline timing gate is needed because the ~110µs dual-source floor dominates relative error below it.

### Resolved Disagreements
- **Loop scope (Codex: sequenced successor loops; Operator: one loop).** Codex recommended scoping this plan to Foundation+Execution-Time (Option C) with cohort-to-50 and traffic as successor loops, citing failure-attribution cost. The operator chose Option A — the full program in a single loop. Resolution: one loop, but with the milestone sequencing retained internally (M0→M4) and a frozen-L2 regression guard (AC-2) + per-milestone closure artifacts (AC-13) to preserve attributability. Operator decision is authoritative.
- **Deferred ACs.** Per the operator's Option A, no milestone is deferred; only inherently out-of-scope items (sm_120 cross-arch, occupancy/IPC-as-targets, extended traffic subset) are `FUT-*`.

### Convergence Status
- Final Status: `converged` — all decisions (DEC-1 through DEC-10) are RESOLVED; no PENDING items remain. The plan is ready for `/humanize:start-rlcr-loop`.

## Pending User Decisions

- DEC-1: Loop scope (full program vs sequenced).
  - Decision Status: **RESOLVED — Option A (full program in one loop)**, milestone-sequenced internally.
- DEC-2: Is "≥50 admitted sm_86 cells" a hard gate?
  - Decision Status: **RESOLVED — hard gate** (AC-11 does not close below 50, and every admitted cell must carry wall-clock+L1+L2).
- DEC-3: Is the >200µs headline-timing roster a hard exclusion?
  - Decision Status: **RESOLVED — hard exclusion from the headline; appendix-allowed** (AC-9).
- DEC-4: Which metric groups are prediction targets vs features/labels?
  - Decision Status: **RESOLVED — targets = wall-clock + L1/L2 hit rates + traffic counts; occupancy + cycles/IPC are features/labels** (achieved occupancy + IPC label-only → FUT-2).
- DEC-5: Promote L1 to load-bearing?
  - Decision Status: **RESOLVED — yes, widen `cache_layers` to `[L1TEX, L2]`** (AC-5).
- DEC-6: Cross-arch closure policy?
  - Decision Status: **RESOLVED — sm_86-first; sm_120 deferred to its own host** (standing operator directive: this machine is sm_86-only) → FUT-1.
- DEC-7: Per-unit repeatability thresholds for cycles/ns/counts (AC-11.1).
  - Claude Position: fractions keep `stddev ≤ 0.03`; cycles/ns use a coefficient-of-variation bound (tentative CV ≤ 0.05); counts use a relative-tolerance bound (tentative ≤ 0.05); exact values set from the M3 calibration sweep.
  - Codex Position: cycles/ns/counts need a separate CV/absolute-tolerance rule distinct from the 0.03 fraction rule; exact thresholds require local noise calibration.
  - Tradeoff Summary: tighter thresholds shrink the admitted cohort (risking the ≥50 gate); looser thresholds admit noisier labels. The exact numbers depend on observed per-metric noise and should be pinned by the M3 sweep.
  - Decision Status: **RESOLVED — per-unit provisional thresholds, finalized from the M3 calibration sweep.** Fractions keep `stddev ≤ 0.03`; cycles/ns use coefficient-of-variation `CV ≤ 0.05`; counts use relative-tolerance `≤ 0.05`. The M3 sweep finalizes the exact numbers and may loosen (documented) only if the ≥50 hard gate (AC-11) is at risk. Baked into AC-11.1.
- DEC-8: L1 head — reuse the frozen-L2-style bounded calibration head or allow L1 its own head/tuning.
  - Claude Position: a SEPARATE bounded-`[0,1]` head INSTANCE (same `PerBucketCalibratorEnsemble` pattern, independent fit) so the L2 head stays frozen.
  - Codex Position: L1 is allowed its own monotonic head/tuning.
  - Tradeoff Summary: reusing the exact pattern minimizes new code and matches L2; allowing independent tuning may improve L1 accuracy at the cost of a second tuning surface.
  - Decision Status: **RESOLVED — a SEPARATE bounded-`[0,1]` head INSTANCE** reusing the `PerBucketCalibratorEnsemble` pattern + monotonic constraints with an independent fit; the L2 v5 head stays byte-frozen. Baked into AC-5.
- DEC-9: Mandatory vs aspirational traffic-metric subset for AC-12.
  - Claude Position: mandatory = GMEM load/store requests + sectors, DRAM sectors, L2 total requests; aspirational (→ FUT-3) = atomics/reductions, sectors-per-request ratios.
  - Codex Position: which exact traffic metrics are mandatory vs aspirational needs an explicit decision, gated on the AC-3 probe results.
  - Tradeoff Summary: a narrow mandatory subset de-risks AC-12; the extended subset broadens the contribution but inherits NCU-availability + modeling uncertainty.
  - Decision Status: **RESOLVED — mandatory = GMEM load/store requests + sectors, DRAM sectors, L2 total requests** (each dropped if the AC-3 probe finds it unavailable on sm_86). Aspirational = atomics/reductions + sectors-per-request ratios → FUT-3. Baked into AC-12.
- DEC-10: Minimum per-head coverage-gate floors (L1, cycles, ns, traffic).
  - Claude Position: reuse the existing smoke floor (5 cells) / publication floor (10 cells) per head.
  - Codex Position: define the minimum coverage gate per head explicitly.
  - Tradeoff Summary: reusing 5/10 keeps consistency with the L2 contract; per-head custom floors could reflect differing label noise.
  - Decision Status: **RESOLVED — reuse the existing smoke floor (5 cells) / publication floor (10 cells) per head** (L1, cycles, ns, traffic), with bootstrap CI required per head. Baked into AC-5/AC-8/AC-12 coverage gates.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments MUST NOT contain plan-specific terminology ("AC-", "Milestone", "M0/M1/...", "Step", "Phase", "Round N", or similar workflow markers). These belong in the plan document only. Use descriptive, domain-appropriate naming. The plan-token scanner (`tools/test_plan_token_scanner.py`) is binding.
- Round summaries MUST record `Verified-live-sm_86` (or `Offline-verified`) — never `Cross-arch-verified` — and must not introduce any sm_120 / cross-arch artifact on this machine (per the standing sm_86-only directive and the `test_sm120_loop_claim_guard` contract).
- NCU invocation remains confined to `tools/local_labels/` → `data/local_labels/`; no other path may invoke NCU or write `.ncu-rep*`.

--- Original Design Draft Start ---

# Unified Roofline-Residual Substrate For Non-Root Multi-Metric GPU Prediction

## Original Idea

Brainstorm a draft research/engineering idea for the non-root GPU measurement+prediction repo (sm_86-ONLY on this RTX 3090 Ti machine, now and all future sessions; sm_120 is the separate 5060-laptop host's scope and is deferred/out-of-scope here). All work targets Verified-live-sm_86.

GOAL: Extend the stack from "L2 sector-hit-rate only" to a UNIFIED multi-metric non-root predictor covering (a) kernel EXECUTION TIME, (b) L1+L2 cache hit rates, and (c) memory-TRAFFIC counts, all validated against NCU ground-truth labels (NCU quarantined to tools/local_labels/ -> data/local_labels/ on this local card), over an EXPANDED >=50-admitted-sm_86-cell cohort where every cell carries wall-clock + L1 + L2 labels.

Three threads:
1) EXECUTION-TIME PREDICTOR (greenfield; none exists today — wall-clock is only measured via CUPTI Activity duration_ns + cudaEvent, dual-source-reconciled, p99 noise floor ~110us). Codex-decided design: predict gpc__cycles_elapsed.max (cycles, DVFS-stable) as PRIMARY target; derive gpu__time_duration.sum (ns) as SECONDARY via read-only NVML SM clock at inference; report cycle-MAPE + ns-MAPE. Headline model = ROOFLINE analytical prior (predicted work / bytes / sectors / cache behavior / active blocks / memory-vs-compute bottleneck from NVBit address+SASS traces + static SASS + launch geometry) + ML RESIDUAL fit vs NCU labels (SynPerf-style). %clock64 phase-cycle ledger as ablation/feature, not the main claim. Restrict headline timing roster to kernels >200us (floor-dominated below). NVML covariate reference operating point must be calibrated (currently inert).

2) COHORT EXPANSION to >=50 ADMITTED sm_86 cells (today 18 admitted of 20 sm_86 labels; 24 excluded = 18 label_repeatability_failed + 6 label_provenance_duplicate; 2 bundle_failed). Catalog: 36 l1 + 29 l2 + 14 l3 workload dirs, ~81 roster rows. Path: build sm_86-capable roster kernels currently lacking cells (RE-DERIVE the true count from roster ARCH_GATE tokens + label_admitted), add shape_points to admitted kernels, repair the 6 provenance duplicates (metadata-fixable), triage the 18 repeatability failures by clustering (tunable kernel-level vs systemic device-level), and author NEW kernels/microbenches as needed. Every new/modified cell must pass the repeatability gate (stddev<=0.03 fraction, n_repeats>=5, consistent binary_sha256) AND must carry the new timing + L1 + L2 label families.

3) BROADER NCU METRIC PREDICTORS (target list = read_ncu_report.py, a vendored SPEC, not runnable; ncu_report_helper.py absent). User-scoped TARGETS = wall-clock + L1/L2 hit rates + memory-traffic COUNTS: unified-L1 hit rate (l1tex__t_sector_hit_rate.pct), L1 read-hit rate (l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct), L2 hit rate (lts__t_sector_hit_rate.pct, already done), GMEM/L2/DRAM requests+sectors+transactions (l1tex__t_requests/sectors_pipe_lsu_mem_global_op_{ld,st,atom,red}.sum, l1tex__m_xbar2l1tex/l1tex2xbar sectors, dram__sectors_{read,write}.sum, lts__t_requests_srcunit_tex.sum, get_L1_Total_Requests~=l1tex__t_requests.sum), atomics/reductions, sectors-per-request ratios. PROMOTE L1 hit rate to LOAD-BEARING (widen submission contract from cache_layers=[L2] to [L1TEX,L2]; retire the APPENDIX-ONLY SCOPE_DOCBLOCK guard). Occupancy (theoretical = static-derivable feature; ACHIEVED occupancy + IPC = LABEL-ONLY, counter-derived, NEVER deployable inputs) and cycles/IPC are FEATURES/labels, not standalone deployable targets.

ARCHITECTURE (Codex-decided): UNIFIED core+memory framework = ONE shared ingestion/feature/error-budget layer (NVBit address+SASS traces, static SASS, launch geometry, read-only NVML covariates, CUPTI Activity timing, NCU labels) feeding MODULAR heads (cache / traffic / occupancy-features / cycles-time). HYBRID harness: reuse comparison_runner.py's 7-field join + typed-unavailable + coverage-gate + bootstrap; add a NEW timing runner + new heads; FREEZE the existing L2 v5 calibration head untouched (it is load-bearing). Head types: bounded [0,1] regression head for FRACTIONS (L1 hit rate, occupancy fraction); LOG-TARGET nonnegative regression head for COUNTS/ns/cycles/traffic (NOT the isotonic projection). LABEL PIPELINE extension: the NCU parser is structurally percentage-only (divides by 100, rejects >105%); add PER-METRIC UNIT TAGS ({unit: pct|count|ns|cycles}) + label-style field names (avoid metric_*/counter_* denylist), bump admitted schema_version 4->5, lockstep producer + sidecar schema + admitted schema + repeatability_aggregator + parser + import_label.

CONSTRAINTS: CLAUDE.md Rules 1-5 binding (NCU only in the quarantine; no clock-lock/persistence/power; CUPTI Activity only, NO privileged CUPTI Events/Metrics/Range/PM/PerfWorks; no host-driver install; no simulator). sm_86-only (per standing directive). Open plan-TODOs: confirm get_L1_Total_Requests against the source ncu_report_helper.py; pin sm_120 L2 geometry is NOT needed here (sm_86-only); calibrate the NVML reference operating point via a local sweep.

## Primary Direction: Unified Roofline-Residual Substrate

### Rationale

Treats all three threads as ONE unified framework — a single shared ingestion/feature/error-budget layer feeding modular per-metric heads, with the execution-time headline = roofline analytical prior + ML residual — maximally reusing the load-bearing comparison/calibration infrastructure while keeping a falsifiable analytical story; distinct from the incremental or dataset-first angles because the architecture itself (shared core + modular heads, not three bolt-on models) is the contribution and is the architecture Codex independently recommended.

### Approach Summary

Build a single shared ingestion/feature/error-budget layer feeding modular per-metric heads (the "Unified Roofline-Residual Substrate", URAS): a shared core (label+bundle loading, v4→v5 schema validation, canonical features, CUPTI dual-source timing reconciliation + NVML covariate join, epsilon-floored error + bootstrap CI); an execution-time head (gpc__cycles_elapsed.max primary via roofline prior + monotonic ML residual, ns secondary via NVML clock, >200us headline roster, %clock64 ablation); multi-metric cache+traffic heads (L1 promoted to load-bearing, bounded-[0,1] hit-rate heads, log-target traffic-count heads, theoretical occupancy as covariate, achieved occupancy+IPC label-only); modular per-(metric,arch) heads reusing fold seed 20260527 with the L2 v5 head frozen; an additive schema v5 with per-metric unit tags landed in lockstep; cohort grown 18→≥50 carrying timing+L1+L2; and a hybrid comparison_runner extension (metric_target + per-metric error formulas + gate).

### Objective Evidence

- workloads/simbench_bw/comparison_runner.py — 7-field join, three-arm dispatch, per-arch coverage gate (5/10 floors), bootstrap CI, epsilon-floored rel-err, v5 head loading: the hybrid-runner reuse surface.
- workloads/simbench_bw/cache_predictor.py + tools/hrd_profiler/observation_builder.py — summarize_launch emits cold/finite_miss/unique_sector counts, RD percentiles, per_op_class/per_access_kind counts, atomics: the shared feature intermediates.
- workloads/simbench_bw/monotonic_calibration_head.py + tools/local_labels/fit_monotonic_calibration_head_v5.py — PerBucketCalibratorEnsemble, model-class hierarchy, composite scoring, deterministic fold seed 20260527: head pattern to freeze (L2) and clone (bounded/log-target).
- workloads/simbench_bw/canonical_features.py — 6-feature vector + monotonic sign table + ActivePathDomainMismatchError guard: the shared feature contract.
- cupti_dual_source_live_admission.py + nvml_covariate_join.py — duration_ns / cuda_event_wall_time_ms / reconciliation_gap_ns + covariate regression (reference operating point inert): timing-label substrate.
- read_ncu_report.py:140,146 names gpc__cycles_elapsed_max + gpu__time_duration_sum; roofline_coordinate.schema.json + %clock64 phase-cycle ledger exist as roofline/cycle scaffolding.
- data/local_labels/calibration_head_v5_*.json + emit_v5_closure_artifacts.py/emit_v5_robustness_artifact.py — closure-artifact pattern reusable per metric.

### Known Risks

- Roofline analytical assumption drift (homogeneous peak FLOPs/BW); untrained capability risks extrapolation error.
- L1 promotion complexity: backfill canonical features + family-bucket taxonomy + retire the SCOPE_DOCBLOCK guard in lockstep with the contract.
- Sub-200us timing variance dominated by ~110us dual-source p99 floor; headline-roster restriction mitigates but bounds applicability.
- Multi-metric error-budget coupling: frozen L2 + learned correlated metrics can decouple budgets; gate must reject tandem-threshold violations.
- %clock64 optional in legacy traces; cycle/roofline path must tolerate CUPTI+NVML-only fallback.
- Lockstep schema surface (additionalProperties:false): v4->5 must update producer+both schemas+aggregator+parser+import_label atomically.

## Alternative Directions Considered

### Alt-1: Measurement-Contract-First Metrology
Lead with the metric-agnostic unit-tagged label/parser/schema-v5/coverage substrate; predictors slot in downstream. Folds into the primary as its enabling label-pipeline phase. Not primary: ships contracts without a headline predictive result.

### Alt-2: Multi-Metric Microbenchmark Suite (dataset-first)
Headline = the expanded ≥50-cell repeatability-gated sm_86 corpus with paired wall-clock+L1+L2+traffic ground truth. Folds into the primary as thread 2 (the cohort it validates on). Not primary: a dataset-only framing underplays the execution-time modeling contribution.

### Alt-3: Joint Multi-Task Learned Model
One shared-representation model jointly predicting time/cache/traffic, exploiting cross-metric correlation. Not primary: CONFIDENCE low — N≈18-50 cells is marginal for a multi-task learner; per-task monotonic-sign ambiguity; deferred higher-variance bet. (Note: sm_86 cells DO carry L1 labels — the subagent's contrary claim is incorrect.)

### Alt-4: White-Box Analytical / Label-Light
Derive traffic+cache+cycles structurally from the walker + static SASS + analytical model; NCU only calibrates a few free params. Folds into the primary as HOW the roofline prior + structural heads compute their analytical term. Not primary: as a standalone framing it under-uses the labeled cohort + proven calibration machinery the residual stage needs.

### Alt-5: Incremental Per-Metric Extension (de-risked)
Ship metrics one at a time cloning the proven v5 head + comparison_runner pattern, each via its own RLCR loop. Folds into the primary as the rollout SEQUENCING (timing→L1→traffic). Not primary: triples boilerplate and forgoes the shared core + cross-metric error budget.

## Synthesis Notes

The five alternatives are largely complementary phases/facets of the primary rather than rivals: Alt-1 is the enabling label-pipeline phase; Alt-2 is the ≥50-cell cohort the primary validates on; Alt-4 is how the roofline prior + structural heads compute their analytical term; Alt-5 is the safe rollout sequencing inside the unified substrate. If the user prefers minimal architecture, fold toward Alt-5 (clone the v5 pattern for execution-time only); if cross-metric statistical power proves high, Alt-3 (joint multi-task) could replace the modular heads (a deferred higher-variance bet). Two factual guards: (1) sm_86 cells DO already carry L1 labels (cache_hit_rate_l1tex_sector_pct, l1tex_label_status: mean_present); (2) the RTX 3090 Ti is GA102, and the exact roster row count (~72 vs 81) + the count of sm_86-capable kernels lacking cells must be re-derived from roster.toml ARCH_GATE tokens before sizing the cohort expansion.

--- Original Design Draft End ---
