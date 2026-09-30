# Fast NCU-metric projection surrogate

This is a new alternative to requiring exact SM86 physical mechanisms. It
targets the observable NCU metrics directly, while preserving the native
replay as the mechanism/reference path.  It does **not** claim to reconstruct
physical pipes, cache owners, read-from sources, or NCU's internal timing
triggers.

## Core idea

Use a two-stage projection:

```text
trace/control -> sparse frontier features -> NCU metric projection
```

The projection consumes only:

* simulator-derived `sim_event_cycles`, `sim_last_issue_cycles`, and
  `sim_gpc_envelope_cycles`;
* simulated instruction population;
* log-scaled work size;
* an independently calibrated regression trained on a disjoint workload
  family.

No `ncu_*` field or aggregate NCU pipe counter is a model feature. The NCU
values are labels only. A warm-cache transfer mode learns only the median
warm/cold ratio from the disjoint calibration family and applies it to the
held-out family.

Implementation: `tools/sm86_ncu_projection_surrogate.py`.

The projection now emits training-residual intervals and a nearest-training
feature-distance diagnostic. Every scored row receives either `candidate` or
`candidate_fallback`; the fallback widens its interval as feature distance
grows, so missing shape coverage no longer drops a workload or presents it as
an exact estimate.

## Hierarchical archetype refactor for the large flash-backward row

The large `flash_attn_bwd_bf16_sm80` row is now handled by an explicit
archetype path rather than by weakening the generic distance threshold.  Independent
live NCU anchors were collected for the same runner at `BH/Sq/D = 4/16/16`:
`Sk=64` and `Sk=256`.  The scored target is `Sk=128`, so it is an interior
point of an independent bracket.  Additional `BH=2,Sk=64` and `BH=8,Sk=64`
anchors show why total instruction count is not a safe substitute: the
`BH=8,Sk=64` anchor has almost the target's instruction population but a
different endpoint behavior.

`tools/sm86_ncu_projection_surrogate.py` now accepts
`--feature-mode frontier-archetype`, `--archetype-matrix`, and `--shape-specs`.
The generic frontier prediction and its original feature distance remain in
the report.  A row is overridden only when its shape metadata exactly matches
`BH`, `Sq`, and `D` of at least two independent anchors and its `Sk` lies
strictly inside the anchor bracket.  Outside that bracket the generic
frontier/trace fallback supplies a candidate interval; no threshold is lowered
and no extrapolation is hidden.  The
archetype prediction is a linear interpolation in `Sk`, with the observed
anchor range retained as a conservative interval.  It is an endpoint model,
not a claim about physical pipe, cache, or issue mechanisms.

The cold large holdout now reports **2.59% GPC MAPE / 2.70% duration MAPE**
(`results/alpha/sm86_ncu_projection_frontier_archetype_goal_cold_20260929.json`).
For the refactored flash row alone, the independent bracket predicts 200,436
GPC cycles versus 199,819 (0.31%) and 148,523 ns versus 148,032 (0.33%).
Projection runtime is 0.00564 s for the seven-row score and the projection-only
speedup versus retained replay wall is 1,216x.  The generic feature-distance
diagnostic remains visible, while the effective path is marked
`projection_path=independent_same_shape_linear_bracket`.

A separate warm-cache shape matrix is retained under
`results/alpha/sm86_flash_bwd_shape_calibration_goal_20260929_warm/`.  With the
disjoint-family warm/cold transfer ratio, the warm large holdout is
**7.24% GPC MAPE / 7.33% duration MAPE** and the flash row is 8.23% / 8.49%.
This is materially better than the unrefactored warm transfer result, but it
also shows that a `--cache-control none` shape bracket is not a universal cache
state: the warm endpoint depends on process/campaign history.  The result is
therefore retained as a cache-domain observation, not promoted to an exact
warm-cache law.

The paired-transition variant is retained at
`results/alpha/sm86_flash_bwd_shape_calibration_goal_20260929_paired_warm/`.
It profiles each independent shape cold and immediately warm, then selects
`--archetype-cache-transfer skip` for the already-warm archetype while keeping
the disjoint-family transfer for generic rows.  This improves the warm large
holdout to **6.10% GPC MAPE / 6.15% duration MAPE** and predicts the fresh flash
target within **0.354% / 0.352% APE**.  The pairing narrows cache-domain error
without asserting a proprietary cache ownership or persistence protocol.

## Live holdout results

Training used the earlier campaign; scoring used the fresh goal campaigns.
The split is **cohort-disjoint** (`base` versus `large`), but the cohorts are
not guaranteed semantically independent: they contain related GEMM,
attention, and normalization families. These numbers are therefore useful
holdouts, not independent generalization proof.

| score lane | NCU GPC MAPE | NCU duration MAPE | projection time | replay-wall speedup |
|---|---:|---:|---:|---:|
| cold, score base | 3.59% | 3.61% | 0.00072 s | 5,383× |
| cold, score large | 3.49% | 3.61% | 0.00067 s | 10,181× |
| warm, score base, transfer | 10.54% | 10.52% | 0.00696 s | 556× |
| warm, score large, transfer | 8.14% | 8.13% | 0.00905 s | 763× |

Reports and predictions are retained under
`results/alpha/sm86_ncu_projection_surrogate_goal_20260929/`.

An independent six-probe calibration attempt (issue, scheduler-contention,
and barrier probes) is retained under
`results/alpha/sm86_independent_ncu_calibration_goal_20260929/`. Directly
extrapolating its absolute frontier-to-NCU regression to the production
workloads gives 43–71% MAPE. This is a useful negative result: independent
resource probes must calibrate normalized rates/constraints, not be treated as
absolute kernel-duration exemplars. The production-cohort surrogate numbers
above are therefore not promoted to independent-generalization accuracy.

The direct static trace-only mode currently produces 47–74% MAPE. A dynamic
protobuf-walking variant reduced that only to roughly 42–63% MAPE and took
about four seconds for these 15 traces, so neither is admitted as an accuracy
or speed path yet. The frontier mode is the promising hybrid: it preserves a
small causal summary and replaces detailed event timing with a cheap
measurement projection. Its reported speedup is projection-only because the
frontier currently comes from the existing sparse replay.

## Why this can replace the blockers for the requested objective

* **Physical pipes:** the projection learns the effect of route ambiguity on
  aggregate NCU outcomes; no route label is required.
* **CUDA ABI/ISA:** native traces supply the semantic workload; the surrogate
  does not need to generate a new cubin for metric prediction.
* **Cache ownership:** cold/warm transfer models cache-state effects as an
  endpoint correction, without asserting an owner or read-from source.
* **Timing endpoints:** the output is explicitly an NCU-duration/GPC projection,
  not simulator completion mislabeled as NCU time.

## Required next slice

The current projection features still come from sparse replay frontiers. To
reach the intended many-magnitude speedup end to end, replace those inputs
with a vectorized trace-stat/frontier extractor that runs directly over
protobuf, control sidecar, and launch metadata. Before using the metric for a
strong accuracy claim, add genuinely independent calibration workloads (the
current base/large split is only a cohort holdout), fit normalized
resource/rate constraints rather than absolute kernel-duration coefficients,
quantile
uncertainty with distance-scaled fallback intervals for unmatched kernels, and fresh NCU campaigns
only after the model is frozen for that holdout.

The first normalized calibration anchors are in
`results/alpha/sm86_independent_ncu_calibration_goal_20260929/resource_rate_card.json`.
They cover issue, scheduler-contention, and barrier/rearm shapes; they should
enter the next model as bounded rate intervals and resource-class priors, not
as exact per-opcode timing constants.

This is an NCU-metric surrogate, not a declaration that the four hidden
hardware mechanisms have been discovered.

The follow-on endpoint-lattice alternative is documented separately in
`docs/sm86_latent_endpoint_lattice_20260929.md`. Its design is informed by
NeuSight's tile-and-wave decomposition ([arXiv:2407.13853](https://arxiv.org/abs/2407.13853)),
PPT-GPU's public hybrid PTX/SASS and reuse-profile model
([repository](https://github.com/NMSU-PEARL/PPT-GPU)), Photon sampled simulation,
and Accel-Sim's separation of control bits, scoreboards, sub-core pipelines,
and correlation. These are design analogies, not imported validation labels.
