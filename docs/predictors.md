# Predictors

> **Historical scope.** This page records the predecessor analytical predictor
> lanes, their gates, and their negative evidence. The active project is the
> event-driven Rust simulator described by root `CLAUDE.md` and
> [`docs/ARCHITECTURE.md`](ARCHITECTURE.md); this page is not a competing
> deployment contract or publication headline. NCU values remain validation
> targets, and the current binding headline is `gpc__cycles_elapsed.max`.

The predecessor suite studied GPU-kernel performance from deployable
user-space signals and kept each predictor lane's target, gate, and honesty
posture separate. Its historical results remain useful for diagnostic and
regression context only; they must not be treated as simulator features, same-cell
calibration inputs, or current publication claims.

The implementation and its measured states are recorded below. The active
simulator architecture is documented in [ARCHITECTURE.md](ARCHITECTURE.md).

```
                         THE THREE LANES (target · gate · status)
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ Lane A  L2 sector-hit-rate   lts__t_sector_hit_rate.pct   sm_86 VALIDATED   │
   │ Lane B  L1/TEX sector-hit    cache_hit_rate_l1tex_..pct   sm_86 SCORED-only │
   │ Lane C  elapsed cycles       gpc__cycles_elapsed.max      both REFUTED/lbl  │
   └───────────────────────────────────────────────────────────────────────────┘
        every per-arch constant (SM count, geometry, anchor clock, cache-op
        default) is an INPUT card — see arch_profiles/cards/*.json
```

---

## Lane A — L2 sector-hit-rate (parametric prior + v5 monotone head)

**Target metric:** `lts__t_sector_hit_rate.pct`
**Model:** analytical Smith–Eklund stack-distance predictor → learned monotone
calibration head.
**Gate:** `0.06 / 0.15 / 0.25` (median-abs / geomean-rel / holdout-geomean-rel).

### Pipeline

```
 dynamic_trace.pb                                    NCU label (fit-time only)
 (RAW reuse distance,                                lts__t_sector_hit_rate.pct
  recomputed via Fenwick-tree LRU)                          │
        │                                                   │
        ▼                                                   ▼
 ┌──────────────────────┐   raw_pred + 6      ┌──────────────────────────────┐
 │ cache_predictor.py   │   monotone feats    │ v5 PerBucketCalibratorEnsemble│
 │ analytical predictor │────────────────────▶│ IsotonicRegressionOnFrozen    │
 │ modeled-hit rule:    │                     │ Projection  (pygam/LGBM/XGB   │
 │ rd < capacity//32    │                     │  when importable)             │
 │ + gridDim density α  │                     │ monotonic_calibration_head.py │
 │ + L2 high-footprint  │                     └──────────────┬────────────────┘
 │   per-slug factors   │                                    │ calibrated hit rate
 └──────────┬───────────┘                                    ▼
   circular_predictor_input_rejected           ┌──────────────────────────────┐
   (any hrd hit_rate/hit_count input killed)   │  ACCURACY GATE 0.06/0.15/0.25 │
                                               └──────────────┬────────────────┘
              sm_86 ──────► PASS  (Verified-live-sm_86)       │       sm_120
              med-abs 0.0592 <0.06 · geomean 0.0642 <0.15 ────┘       ──► REFUTED
              holdout 0.0552 <0.25 · n=40 admitted (n_fit=34)         (see below)
```

- The analytical stage (`workloads/simbench_bw/cache_predictor.py`) is
  **non-circular by construction**: it recomputes per-launch reuse-distance
  evidence from the producer-side `dynamic_trace.pb`, *not* from the HRD
  profiler's already-modeled `hit_rate`/`hit_count`. Any input carrying those
  fields is rejected with `circular_predictor_input_rejected`
  (`cache_predictor.py:1-21`).
- The head (`monotonic_calibration_head.py`) fits a **per-bucket** ensemble; when
  pygam/LightGBM are unavailable (the pinned env has only sklearn) the only
  fittable class is `IsotonicRegressionOnFrozenProjection` — 1-D PAV on a frozen
  `(raw_prediction, footprint_ratio, long_rd_tail_mass)` projection with weights
  `(+1, −1, −1)` (`monotonic_calibration_head.py:1-39`). Fitter:
  `tools/local_labels/fit_monotonic_calibration_head_v5.py`.

### sm_86 — VALIDATED

Gate thresholds live in
`data/local_labels/calibration_head_v5_accuracy_gates_sm_86.json:143-146`
(`geomean_max 0.15`, `median_abs_max 0.06`, `holdout_max 0.25`, per-bucket p90
0.15 advisory). Frozen head result at HEAD:

| gate | value | bar | verdict |
|---|---|---|---|
| full-cohort median-abs | **0.05923** | ≤ 0.06 | PASS (barely) |
| full-cohort geomean-rel | **0.06420** | ≤ 0.15 | PASS |
| holdout geomean-rel | **0.05524** | ≤ 0.25 | PASS |
| admitted cells | **40** (n_fit 34, 6 holdout) | — | `Verified-live-sm_86` |

The per-bucket p90 is **advisory**: `memory_stress` 0.2346 (n=9) and
`other` 0.3587 (n=15) exceed 0.15 but are not part of the recorded pass
criterion. The retained L2 v5 head is `data/local_labels/sm_86_v5_head.json`.

### sm_120 — REFUTED (and *why* the frozenset is frozen)

`data/local_labels/sm120_cache_metric_closure.json`:
`verdict = engineering_closure_refuted`, `cache_claim_load_bearing = false`,
`named_target_gate.median_abs_error = 0.27629`, **22** cache/memory targets all
refuted. Root cause is **not** the analytical model — it is the L1→L2 miss
filter going inert on Blackwell:

```
 tools/cache_predictor/l1_cache_op_classifier.py:309
   _BARE_LDG_E_DEFAULTS_TO_CA = frozenset({"sm_86"})          ← sm_120 NOT in set

   sm_86:   bare  LDG.E                → LDG_E_CA   → L1-eligible → filter WORKS
   sm_120:  LDG.E.U16.CONSTANT,             ┐
            desc[UR..] (descriptor loads)   ┴─────► OTHER → bypass → filter INERT
   ⇒ every miss forwarded to L2 unfiltered ⇒ L2 hit-rate prediction refuted
```

Adding `sm_120` to that frozenset should follow a live 5060
cuobjdump/microbenchmark that documents the true `.ca`/bypass semantics. The
ArchProfile represents the current unknown explicitly — the
sm_120 card carries `"bare_ldg_e_default": "unknown_typed_unavailable"`
(`arch_profiles/cards/sm_120-laptop.json`), so bare `LDG.E` keeps mapping to
`OTHER`. The `cache_replay` subsystem (`data/local_labels/cache_replay_*.json`)
is a **non-load-bearing sm_120 ablation**, not a predictor: even its best learned
arm scores 0.0755 median-abs — still above the 0.055 target — so it does not
rescue the lane. The next experiment is a live Blackwell load-semantics and
cache-geometry characterization.

---

## Lane B — L1/TEX sector-hit-rate (walker + classifier + bounded head)

**Target metric:** `cache_hit_rate_l1tex_sector_pct`
**Model:** per-SM LRU walker → cache-op classifier → bounded-[0,1] calibration head.
**Gate:** coverage-gated, **NOT accuracy-gated** (this is the honesty crux).

### Pipeline

```
 cuobjdump SASS text ─► l1_cache_op_classifier.py ─► per-PC cache-op map
                        (12-token wire vocab:          (which accesses are
                         LDG_E_CA … OTHER,              L1-eligible vs bypass)
                         host-only, no NVBit)                    │
                                                                 ▼
 per-block access seq ─► l1_walker.py ──────────────► per-launch L1 hit counters
 (line_id, l1_eligible)   per-SM fully-assoc LRU,      (bypass excluded from BOTH
                          sector-with-line-alloc hit    numerator & denominator)
                          cap = capacity_bytes // 128           │
                                                                ▼
                                              bounded-[0,1] L1 head → COVERAGE gate
                                              (sm_86_l1_head.json)         │
                          sm_86 ──► SCORED (meets_floor=true) ─────────────┘
                          rel-geomean 0.89 · n=44 (30 scalar / 14 typed_unavail)
                          NO hard accuracy gate · median abs residual ≈ 4%
                          sm_120 ──► UNCALIBRATED (no L1 head; cache_layers=["L2"])
```

- The walker (`tools/cache_predictor/l1_walker.py`) models each SM's L1 as an
  independent fully-associative LRU line cache (CTAs on different SMs do not evict
  each other). Hit rule = **sector-with-line-allocation**; bypass accesses
  (`l1_eligible=False`) are excluded from both numerator and denominator.
- The classifier (`l1_cache_op_classifier.py`) records the **actual SASS opcode**,
  not the semantic class: the Blackwell `.ca` encoding `LDG.E.STRONG.SM` is kept
  as `LDG_E_STRONG_SM` (L1-bypass on Blackwell), never collapsed into the Ampere
  `LDG_E_CA`. It is cuobjdump-backed and host-only.

### sm_86 — SCORED, not accuracy-gated

`data/local_labels/l1_closure_comparison_sm_86.json`: `result_kind = admitted`,
`relative_error_stats_l1tex.geomean = 0.8905` over **44** cells (30 scalar / 14
`typed_unavailable`), `coverage_gate_verdict.meets_floor = true`. The apparent
"0.89 error" is a **relative-geomean artifact**: L1 hit rates on streaming kernels
sit near zero, so `|pred−true|/true` inflates even though the **median absolute
residual is ≈ 4%** (q50 ≈ 0.041). There is deliberately **no hard L1 accuracy
gate** in the runner — "L1 load-bearing" means *scored + coverage-gated*, held to
the L2 standard only for coverage, never for accuracy. `cache_layers` is a
per-arch card: `sm_86 → ["L1","L2"]`.

### sm_120 — UNCALIBRATED

No sm_120 L1 head, no sm_120 L1 predictions exist; the sm_120 card forces
`cache_layers = ["L2"]` so no L1TEX cell is broadcast (a uniform `["L1","L2"]`
would flood every sm_120 L1 cell as `join_key_unmatched`). The two Blackwell-only
classifier tokens are exercised via canned cuobjdump text fixtures only; live
sm_120 coverage remains future experimental work.

---

## Lane C — wall-clock (cycles-first target, ns as a derived interval)

**Target metric:** `gpc__cycles_elapsed.max` — the **operating-state-bound**
elapsed cycle count, declared the *sole* publication headline. Nanoseconds are a
**derived interval**, never a co-equal point.
**Model:** roofline prior + ridge residual (`pedestal_roofline_ridge_residual`).
**Merge policy:** **labels + refutation only** — no positive wall-clock claim ships.

### Why cycles, not nanoseconds

Predicting `duration_ns` directly inherits the DVFS confound. Tree B's
OLS-on-`log10(duration_ns)` is **refuted** in
`data/local_labels/duration_predictor_report.json`: held-out SMAPE **1.766** vs
the strongest baseline (`shape_size_only`) **0.891** — and the NVML clock
covariates are the *entire harmful delta* (`predictor_smape_without_nvml` equals
the baseline exactly). The clock spanned **180–2762 MHz (15.34×)** across the
cohort. Cycles avoid treating nanoseconds as an independent target, but they are
not invariant under DVFS or boost; comparisons remain bound to the recorded
operating state and capture protocol.

### Pipeline

```
 launch geometry + static SASS work + GA102 peak constants
        │
        ▼
 ┌─────────────────────────────┐   FORBIDDEN_FEATURE_FIELDS blocks labels as
 │ timing_cycle_predictor.py   │   inputs: gpc__cycles_elapsed.max, gpu__time_
 │ roofline prior:             │   duration.sum, clock64, IPC, occupancy …
 │   dyn-FLOP ÷ tensor peak     │   (deploy-legal feature vector only)
 │   × 0.066 efficiency prior   │
 │ + ridge residual (λ=10)      │──────────────┐  predicted cycles
 │ v3 same-slug arm → DIAGNOSTIC│              │
 └─────────────────────────────┘              ▼
                              ┌──────────────────────────────────────────┐
                              │ PUBLICATION GATE (contract-pinned):       │
   sm_86 CYCLE HEADLINE ─────▶│  geomean-rel ≤0.15 · med-abs-log ≤0.1     │
   med-abs-log 0.0414  PASS   │  bootstrap-upper ≤0.25 · ≥10 cells        │
   geomean-rel 0.1583  FAIL ──┤                                           │
   bootstrap    0.3099 FAIL ──┘  ⇒ verdict_falsification (n=13, GEMM-heavy)│
                              └──────────────────────────────────────────┘
        │ cycles × 1000 / mhz × [1 ± 0.06]      (mhz = short-burst 1890 anchor)
        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ DEPLOY-ns CROSSWALK (separate leg): CUPTI-Activity & cudaEvent vs NCU  │
 │  agreement_pass on 13/13 cells · noise floor 110352 ns · bound 0.20    │
 │  ⇒ research_success  (ns is a DERIVED interval, not a point headline)  │
 └──────────────────────────────────────────────────────────────────────┘
```

- The predictor (`timing_cycle_predictor.py`) is deploy-legal: labels can be fit
  targets but never inference inputs (`FORBIDDEN_FEATURE_FIELDS`, line 100). The
  v3 same-slug scaled-neighbor arm was **demoted to a diagnostic-only channel**;
  the deployable score uses no same-slug substitution.
- The short-burst **1890 MHz** anchor (not the 1350 MHz sustained-throttle anchor)
  removes the ~38% cycles→ns bias; the ns leg emits an interval via
  `cycles_to_ns(..., regime="short_burst") × [1 ± 0.06]` (`timing_closure.py:79`).

### Status

`data/local_labels/timing_predictions/timing_closure_sm_86.json`:
`cycle_terminal_state = verdict_falsification` (an **honest near-miss** — geomean
is 0.008 over the bar; the n=13 GEMM-heavy cohort's bootstrap upper bound is
clearly over), `deploy_ns_terminal_state = research_success`,
`overall_terminal_state = verdict_falsification`. The cycle headline therefore
**does not pass**; only the deploy-ns crosswalk closes. **sm_120** has no cycle
labels and no short-burst anchor → `deferred_requires_sm120_cycle_capture`; the
refuted OLS is not carried forward.

---

## Cross-lane invariants

| property | Lane A (L2) | Lane B (L1) | Lane C (cycles) |
|---|---|---|---|
| sm_86 status | **VALIDATED** (gate pass) | **SCORED** (coverage only) | cycle near-miss / **ns pass** |
| sm_120 status | **REFUTED** | uncalibrated | **deferred** |
| hard accuracy gate | 0.06 / 0.15 / 0.25 | none (by design) | 0.15 / 0.10 / 0.25 |
| deploy-legal features | yes | yes | yes (`FORBIDDEN_FEATURE_FIELDS`) |
| per-arch knobs | ArchProfile card | ArchProfile card | ArchProfile card |

Every per-arch number a lane touches — SM count, L1/L2 geometry, `griddim_penalty_alpha`,
`bare_ldg_e_default`, the NVML anchor clocks, `cache_layers` — is an **input card**
under `arch_profiles/cards/`, resolved by `arch_profiles/loader.py` with
**fail-closed** behavior on unknown devices (no silent sm_86 fallback). See
[archprofile.md](archprofile.md) for the current card and resolver design.

The SOL-ExecBench baseline corpus that feeds kernels into these lanes is
documented at [sol_execbench/README.md](sol_execbench/README.md).
