# Trace-sampling golden ablation — fixed-design non-inferiority adjudication

Status: **ACCEPTED and binding for task17 / Round 26**

Date: 2026-07-10

Scope: the real-golden selection-unit ablation only. This adjudication does not
change the sampler, estimator, manifest, random streams, seeds, fractions,
accuracy thresholds, or floating-point comparison epsilon.

## Decision

Replace universal wave-versus-independent non-regression on **every metric of
every cell** with a predeclared composite acceptance rule:

1. both arms retain the existing absolute count and rate accuracy limits in
   every admissible cell;
2. the wave-segment arm must be non-inferior to the independent-block arm for
   both count and rate on the pooled, fixed validation design;
3. the manifest must contain a nonempty, predeclared
   `co_residency_sensitive` witness set, every such witness must retain
   wave-rate non-inferiority, and at least one such witness must show a strict,
   non-vacuous wave-rate advantage;
4. exact coverage, payload fairness, CI/calibration/error-budget checks,
   admission negative controls, and count heterogeneity checks remain required;
   and
5. deterministic adverse controls must prove that pooled success cannot hide a
   lost sensitive witness, a pooled count/rate regression, or an inaccurate
   arm.

Per-cell wave-versus-independent ordering remains reported for every cell. A
neutral cell's ordering reversal is a diagnostic, not by itself a task17
failure, when all five conditions above hold. This is stronger than a pooled-only
rule, while avoiding a universal claim that the design and finite experiment do
not support.

## Binding source design

The Fable-5 design and its simulator-grounded adjudication establish the claim
that task17 is meant to test:

- `plans/sm86_trace_sampling_strategy_fable5_design.md` makes complete
  concurrency segments the primary sampling units because independent block
  thinning discards cache-order/cohort information. Its section 7 accuracy
  criterion is explicitly p95 over **seeds x cells**, and its ablation asks for
  wave-segment versus independent-block evidence documenting the co-residency
  effect. It does not require stochastic dominance in every cell.
- `plans/sm86_trace_sampling_strategy_fable5_design_adjudication.md` accepts
  whole-block wave segments because replay consumes literal `sm_id`,
  `warpid_sm`, per-SM L1 state, and shared L2 state. It also requires
  ghost-differenced segment replay and explicitly recognizes cross-segment L2
  warmth as an approximation. It does not claim that clustering minimizes the
  error of every derived metric in every cell.
- The real harness correctly narrows the empirical interpretation to
  `whole_segment_vs_independent_block_selection_unit` and sets
  `causal_co_residency_claim: false`: flat L2 replay is a selection-unit and
  cache-order proxy, not a causal concurrency experiment. This adjudication
  preserves that claim boundary.

Consequently, the scientific obligation is to show that the production PSU
preserves absolute fidelity, is non-inferior over the declared validation
design, and wins where a predeclared cache-order/cohort-sensitive witness says
the PSU distinction should matter. It is not to prove universal pointwise loss
dominance over an estimator using a different PSU.

## Independent predeclaration authority

The runtime manifest cannot be its own authority for cell roles or witness
labels. The implementation must independently pin both the complete manifest
bytes and the positive-cell expectation map under a versioned policy. For this
adjudication the manifest SHA-256 is
`ecc40493d77f7b400f512a89b1b6c31d87798281ea27cab5af9b6299e429962b`, and the
binding positive map is `gs_stream_copy=neutral`,
`gs_stream_reuse=co_residency_sensitive`, and `gs_var_stride=neutral`. Any
relabel, role change, deletion, addition, label/path edit, or byte-level manifest
change must fail the certification until a new design adjudication and policy
version deliberately replaces this authority.

## Exact empirical estimand

Let:

- `C` be the exact set of subset-admissible positive cells in the hash-pinned
  manifest;
- `C_sensitive` be the nonempty subset whose manifest expectation was declared
  `co_residency_sensitive` before this adjudication (the current declaration is
  already committed in `8d3ade0e`);
- `F = {1/16, 1/8, 1/4}` and `S = {0, 1, 2, 3, 4}`;
- `W` be the production phase x launch-tercile, capped-Neyman, whole-segment,
  calibrated HT/ratio arm with per-segment ghost differencing; and
- `B` be the independently drawn whole-block baseline, replayed cold once, with
  the same realized block count, certainty membership, full-trace reference,
  decoder/cache path, and its declared inverse-probability/calibration rule.

For arm `a`, cell `c`, fraction `f`, and seed `s`, define

```
E_count[a,c,f,s] = abs(T_hat[a,c,f,s] - T_full[c]) / T_full[c]
E_rate [a,c,f,s] = abs(R_hat[a,c,f,s] - R_full[c])
```

where `T_full` and `R_full` are the common production full-trace sector-count
and L2-hit-rate references. Define `Q95_cell[m,a,c]` as NumPy's existing p95 of
the 15 values over `F x S`, and `Q95_pool[m,a]` as the same p95 over the complete
`C x F x S` ledger. These are finite, fixed-design benchmark statistics. They
are not an expectation over an unbounded RNG population, a confidence bound on
stochastic dominance, or a causal co-residency effect size.

The independent arm is an intentionally structurally wrong baseline for the
simulator's cohort semantics, not an oracle that must have larger realized loss
in every finite cell. Equal realized block counts make the resource comparison
fair; they do not equalize effective PSU count, replay order, cache state, or
estimator variance.

## Why sector-Neyman does not imply universal per-cell dominance

The production allocation uses per-segment sector traffic as the pilot variable.
In the ideal uncapped, continuous, equal-cost stratified-total setting,
`n_h proportional to N_h * S_h` minimizes the approximate variance of that
specific additive target among allocations over the same strata and PSU. The
implemented design also has probability-one certainty segments, integer
Hamilton rounding, per-stratum floors and caps, and a calibrated ratio
estimator. None of those facts provides a theorem that
`Q95_cell[m,W,c] <= Q95_cell[m,B,c]` for every `m` and `c`.

In particular:

- L2 hit rate is a nonlinear ratio of cache-history-dependent hit and sector
  totals, not the additive sector pilot used by Neyman allocation.
- The baseline samples many individual blocks while the production arm samples
  fewer correlated clusters at the same block budget. Cluster preservation is
  necessary to avoid semantic thinning bias, but can increase finite-sample
  variance on a neutral cell whose intra-cluster values are similar.
- Ghost differencing and cold-once independent replay deliberately expose
  different cache-order information. A realized error ordering therefore mixes
  selection variance with the very state-preservation distinction under test.
- Capping, certainty inclusion, phase crossing, and integer allocation prevent
  the textbook continuous Neyman optimum from transferring even to every
  additive target, much less to every ratio metric.
- A p95 computed from 15 fixed trials is a high order statistic. Its ordering
  can reverse under finite draws without either estimator violating absolute
  accuracy or the pooled design changing conclusion.

Universal per-cell dominance would therefore be an unsupported strengthening
of the Fable-5 claim. Enforcing it after observing a neutral-cell reversal would
reward seed/RNG/selector tuning to this ledger rather than improve the validity
of the selection-unit claim.

## Round 25 ledger used for adjudication

The authoritative artifact is
`data/local_labels/campaign/golden_set_report.json`, with tracer
`nvbit-enhanced-tracer-v5`, stats pass `counting-pass-sector-aux-v2`, and
estimator `phase-tercile-neyman-srs-ht-ghost-v2`. It contains three admissible
positive cells x three fractions x five seeds = 45 trials, plus six correctly
forbidden negative controls. Its only failure is
`gs_var_stride: ablation::wave_segment_rate_regression`.

All values below are p95 errors; columns are wave / independent:

| Cell | Predeclared expectation | Count relative error | Rate absolute error | Existing absolute limits |
|---|---|---:|---:|---:|
| `gs_stream_copy` | neutral | `0 / 0` | `0 / 1.1102230246251565e-16` | both pass |
| `gs_stream_reuse` | co-residency-sensitive | `0 / 1.9737298215558337e-16` | `0.007968072771094126 / 0.014580887083023264` | both pass |
| `gs_var_stride` | neutral | `0.00039198196348200453 / 0.0006332822519184368` | `0.001250896241994581 / 0.0007399460388754273` | both pass |
| **Pooled** | all 45 trials | `0.0002478024767413449 / 0.0004542098641333341` | `0.007407643699112909 / 0.014487372504340333` | both pass |

The immutable comparison epsilon is `1e-12`; count and rate absolute limits
remain `0.05` relative and `0.02` absolute. The sensitive witness's rate
advantage is `0.006612814311929139`, far above floating-point epsilon without
introducing a fitted margin. The neutral `gs_var_stride` rate reversal is
`0.0005109502031191538`; its wave error is still only about 6.3% of the existing
absolute rate-error limit, and its count result favors wave segments.

The non-ablation ledger also passes: count/rate CI coverage are both `1.0`, raw
ghost-rate coverage is `0.9111111111111111`, the sampling-induced error p95 is
`0.007407643699112909` against allowance `0.12034872516556291`, count evidence is
non-vacuously heterogeneous on `gs_var_stride`, and all six negative cells are
typed `wave_forbid` (including the atomic histogram's atomic gate). Thus the
observed result is specifically a finite neutral-cell arm ordering, not an
absolute-fidelity, coverage, error-budget, admission, or sensitive-witness
failure.

## Alternatives considered

### A. Retain universal per-cell count-and-rate dominance — rejected

This is stricter syntactically but tests an unsupported universal property. It
is not stated by Fable-5, does not follow from sector-Neyman allocation, and
turns a 15-draw neutral-cell order statistic into the task17 scientific claim.
The likely remedies would be post-label RNG or selector tuning, both forbidden.

### B. Modify seeds, streams, fractions, epsilon, or the selector — rejected

Changing a draw after observing which arm won is label leakage. Expanding the
seed set now would also redefine the observed estimand after inspection. A
selector change optimized to `gs_var_stride` would conflate benchmark fitting
with production correctness. All are outside Round 26.

### C. Add a per-cell non-inferiority margin — rejected

A new margin chosen after seeing `0.0005109502031191538` would be an especially
transparent fitted threshold. Reusing the `0.02` absolute accuracy limit as a
pairwise margin would blur two separate questions and permit a materially worse
arm so long as both remain accurate. The composite rule uses no new numeric
margin.

### D. Use pooled evidence alone — rejected

Pooling matches the Fable-5 accuracy estimand but can hide a selection-unit loss
on the only cell designed to exercise cache-order/cohort sensitivity. It can
also pass vacuously with an empty sensitive roster. Pooled comparison is
necessary, not sufficient.

### E. Collapse count and rate into one score — rejected

Any weighting would introduce a new post-observation tuning choice and could
let a strong count result conceal a rate failure (or vice versa). Count and rate
remain separate gates.

### F. Composite fixed-design rule — accepted

This rule matches the scope of the evidence: cellwise absolute safety,
design-level comparative performance, a targeted non-vacuous selection-unit
witness, and deterministic tests against masking. It retains every observed
cell and every diagnostic while making only claims the experiment can support.

## Binding composite acceptance rule

The real-golden verdict passes only when **all** clauses below pass. Existing
global task17 gates not mentioned here are unchanged and remain conjunctive.

### G0 — complete and fair evidence

- Ablation slugs exactly equal the subset-admissible positive slugs; neither set
  may be empty.
- Every cell has exactly `len(F) * len(S) = 15` trials on the unchanged fraction
  and seed grid.
- Every paired trial has equal realized block counts, uses the same production
  full-trace reference, and the independent draw demonstrably partially retains
  at least one segment. The ledger must carry a concrete population-segment and
  selected-intersection witness from which strict partial retention can be
  recomputed; a supplied Boolean alone is not evidence.
- Every cell has a valid predeclared expectation in `{neutral,
  co_residency_sensitive}`, and `C_sensitive` is nonempty.
- The manifest seal and every runtime slug, role, and expectation match the
  independent versioned policy above.

### G1 — per-cell absolute accuracy

For every `c in C` and both `a in {W,B}`:

```
Q95_cell[count,a,c] <= 0.05
Q95_cell[rate, a,c] <= 0.02
```

No pooled improvement can conceal an inaccurate arm or cell.

### G2 — pooled selection-unit non-inferiority

Both metrics must independently satisfy the unchanged strict comparison:

```
Q95_pool[count,W] <= Q95_pool[count,B] + 1e-12
Q95_pool[rate, W] <= Q95_pool[rate, B] + 1e-12
```

This is the comparative performance claim over the declared fixed design.

### G3 — co-residency-sensitive witness

For every `c in C_sensitive`:

```
Q95_cell[rate,W,c] <= Q95_cell[rate,B,c] + 1e-12
```

and at least one `c in C_sensitive` must satisfy the non-vacuity condition

```
Q95_cell[rate,W,c] + 1e-12 < Q95_cell[rate,B,c].
```

The strict witness uses only the existing floating-point epsilon, not a fitted
effect-size threshold. This is an observed selection-unit/cache-order witness,
not a causal hardware co-residency claim. Neutral cells are ineligible to
satisfy G3.

### G4 — surrounding task17 validity gates

All existing production-arm accuracy, CI coverage/width, calibration,
minimum-segment, convergence, ghost-depth, and sim-vs-NCU error-budget checks
remain required per cell and in aggregate. Count heterogeneity must remain
non-vacuous. The exact manifest size and every declared negative control remain
required; a gate-inadmissible cell must still be forbidden rather than scored.

### G5 — deterministic adverse controls

The composite evaluator must be a deterministic function of a supplied
ablation ledger and must be tested with handcrafted ledgers. It must reject, at
minimum:

1. missing/duplicate cell coverage or an empty sensitive set;
2. a post-hoc expectation relabel, role change, or manifest deletion;
3. either arm exceeding either G1 limit in any cell;
4. pooled wave count regression while rate and witness pass;
5. pooled wave rate regression while count and witness pass;
6. a sensitive-cell wave rate regression even when neutral cells make both
   pooled comparisons pass;
7. no strict sensitive rate advantage (an equality-only or vacuous witness);
8. mismatched block budgets or an independent arm that accidentally preserves
   every whole segment.

A companion positive control must contain a neutral-cell rate ordering reversal
while G0-G4 pass, proving that the implementation replaced only the unsupported
universal-neutral clause. These controls use literal values and no RNG, trace,
or NCU label.

## Reporting and claim discipline

The report schema is version 2 and must continue to emit each cell's four p95
values and the two legacy
diagnostic booleans. It must not relabel the observed
`gs_var_stride` rate ordering as a win. Instead it should distinguish:

- `per_cell_absolute_accuracy_ok` — binding;
- `pooled_selection_unit_noninferiority_ok` — binding;
- `co_residency_sensitive_witness_ok` — binding;
- `universal_per_cell_noninferiority` — diagnostic, expected to remain false on
  the Round 25 ledger; and
- `composite_acceptance_ok` — the new binding ablation verdict.

The top-level report may pass only from `composite_acceptance_ok` together with
all unchanged task17 gates. The interpretation must remain
`whole_segment_vs_independent_block_selection_unit`, and
`causal_co_residency_claim` must remain false. No cell, trial, or typed
diagnostic may be dropped from the artifact.

## Expected adjudication of the frozen Round 25 ledger

Under this rule the frozen ledger satisfies G0-G4:

- every arm/cell is absolutely accurate;
- pooled wave count and rate errors are both lower;
- the predeclared `gs_stream_reuse` witness has a strict wave-rate advantage;
- `gs_var_stride` remains present and honestly reports its neutral-cell rate
  reversal; and
- all surrounding fidelity and negative-control gates pass.

That expected result is a consequence check, not a fitted numeric rule. Any
future version/manifest run is evaluated from scratch against the same
composite contract and deterministic adverse controls.
