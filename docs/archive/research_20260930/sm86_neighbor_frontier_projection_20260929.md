# SM86 independent neighbor-frontier sensitivity (2026-09-29)

## Idea

The promoted frontier/archetype projector already captures the global
issue/event/envelope relation, but related kernels with nearly identical
frontiers can still differ by a small endpoint-efficiency residual.  This
optional sensitivity path transfers that residual from the nearest
**independent** static semantic signature:

```text
signature = static opcode fractions + barrier/wait/stall densities
neighbor ratio = NCU / analytical frontier base   (training cohort only)
candidate = (1 - alpha) * promoted generic + alpha * neighbor ratio transfer
```

The signature is read from captured instruction metadata and does not inspect
protobuf timing, NCU counters, or physical-pipe fields.  The neighbor labels
come only from the independent calibration cohort.  The exact flash shape
archetype still overrides the generic path.  The feature is disabled by
default (`--neighbor-correction-gpc 0 --neighbor-correction-duration 0`) so
the promoted model cannot regress accidentally.

This is a narrow analogue of representative-warp calibration in
GCStack/GCScaler, but without claiming that a static neighbor represents the
same physical warp or scheduler state.  See the [GCStack/GCScaler source](https://github.com/yonsei-hpcp/gcstack_gcscaler)
and its [CPI-stack description](https://github.com/yonsei-hpcp/gcstack_gcscaler/blob/main/gcstack/README.md).

## Exploratory score-only result

The retained sensitivity run uses `alpha_gpc=0.5`, `alpha_duration=0`, and
`k=1`.  It is **not promoted**: the alpha was not pre-registered before all
score sensitivity exploration, so it is reported as a bounded model
sensitivity rather than an independent generalization claim.

| scope | GPC MAPE | duration MAPE | model time |
|---|---:|---:|---:|
| cold large (7 rows) | 2.305% | 2.626% | 0.062 s |
| cold base (8 rows) | 3.406% | 3.400% | 0.048 s |
| cold all (15 rows) | 2.892% | 3.039% | 0.110 s |

The baseline promoted cold full-campaign values are 3.101% GPC / 3.039%
duration.  Thus the sensitivity reduces the aggregate GPC error without
changing duration, but no production/config default is changed.  Reports are
under `results/alpha/sm86_neighbor_frontier_projection_goal_20260929/`.
All 15 rows remain `candidate` with one exact-shape and fourteen generic
frontier-covered rows; no OOD refusal or row drop is introduced.

## Boundary impact

* **Pipe/issue:** nearest semantic class transfers residual efficiency but does
  not identify a physical pipe or issue slot.
* **ABI:** compiler/template effects remain in the residual; no ABI generator is
  inferred.
* **Cache:** no cache owner/read-from state is used; phase-specific residuals
  remain separate.
* **Timing:** duration and GPC remain independent outputs; the correction is
  not a common-clock relation.

The path is a useful low-cost extrapolation probe (about 104x aggregate
projection speedup in the retained cold report), not blocker closure.
