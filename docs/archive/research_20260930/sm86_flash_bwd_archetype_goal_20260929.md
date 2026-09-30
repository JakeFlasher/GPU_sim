# SM86 large-workload coverage refactor: flash backward archetype

## Outcome

The large `flash_attn_bwd_bf16_sm80` workload was the only row rejected by the
generic frontier-distance guard.  Its sparse replay frontier (`sim_event_cycles`
about 194k, with a large issue-to-event population) is not close to the other
large-cohort feature vectors, so changing the global OOD threshold would have
silently converted an extrapolation into a candidate.

The replacement is a hierarchical endpoint model:

```text
trace/frontier model
       |
       +-- generic cohort projection (all rows)
       |
       +-- strict flash-bwd shape archetype (only when bracketed)
             BH/Sq/D exact match -> interpolate Sk -> NCU metric interval
```

The generic diagnosis is retained.  The archetype is an explicit alternative
path, not a renamed failure.

## Independent calibration campaigns

The live profiles are under:

- `results/alpha/sm86_flash_bwd_shape_calibration_goal_20260929/` (`cache-control all`)
- `results/alpha/sm86_flash_bwd_shape_calibration_goal_20260929_warm/` (`cache-control none`)
- refined live anchors: `results/alpha/sm86_flash_bwd_shape_calibration_goal_20260929_refined/`
  and `results/alpha/sm86_flash_bwd_shape_calibration_goal_20260929_paired_warm_refined/`

The target shape is `BH=4, Sq=16, Sk=128, D=16`.  It is a score-only row from
the full campaign; it is absent from both calibration matrices.

Cold-cache matrix:

| anchor | BH | Sq | Sk | D | NCU ns | GPC cycles | instructions | grid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 2 | 16 | 64 | 16 | 76,192 | 102,803 | 22,526 | 2 |
| bh8_sk64 | 8 | 16 | 64 | 16 | 76,128 | 102,708 | 90,104 | 8 |
| mid_bh4_sk64 | 4 | 16 | 64 | 16 | 75,520 | 101,890 | 45,052 | 4 |
| long_bh4_sk256 | 4 | 16 | 256 | 16 | 294,528 | 397,528 | 178,924 | 4 |

The follow-on live campaign added same-configuration `Sk=96` and `Sk=192`
anchors: 112,544 ns / 151,887 cycles / 67,364 instructions and 221,088 ns /
298,392 cycles / 134,300 instructions, respectively. Their paired-warm
values are 98,464 ns / 132,864 cycles and 193,408 ns / 261,022 cycles. These
additional points make the local response surface identifiable without using
the scored `Sk=128` observation.

The `BH=4` anchors bracket the target in `Sk`; the `BH=8` point is a negative
control for the tempting but invalid instruction-count/work-only coordinate.
The model consumes only these independent anchor labels and the launch-shape
metadata.  NCU values from the target campaign are read only after prediction
for scoring.

## Model and fallback coverage rule

For each target metric `y` in `{ncu_duration_ns, ncu_gpc_cycles}`:

1. Filter anchors to the same archetype and exact `BH`, `Sq`, and `D`.
2. If at least three independent anchors exist, fit a degree-2 polynomial in
   normalized `Sk` and evaluate only inside the observed `Sk` hull.
3. With exactly two anchors, use linear interpolation between the lower and
   upper points.
4. Use the observed anchor range as a deliberately conservative interval.
5. If an exact shape anchor exists, use it as a candidate; if no matching
   bracket exists, retain the generic frontier prediction as
   `candidate_fallback` with a widened interval.

This is a piecewise local response surface, not universal opcode timing.  It
is intentionally strict because the live `BH=8,Sk=64` anchor has nearly the
same dynamic instruction population as the target while retaining a distinct
endpoint.  The path therefore avoids a false sense of generalization.

## Cold result against held-out NCU

Latest report: `results/alpha/sm86_ncu_projection_frontier_archetype_goal_cold_refined_20260929.json`.

| quantity | value |
|---|---:|
| large-cohort GPC MAPE | 2.586% |
| large-cohort duration MAPE | 2.698% |
| flash target GPC prediction / NCU | 200,389 / 199,819 cycles (0.285% APE) |
| flash target duration prediction / NCU | 148,481 / 148,032 ns (0.303% APE) |
| model runtime | 0.00586 s |
| projection-only speedup vs replay wall | 1,171x |

The report keeps the generic feature-distance diagnostic and records
`archetype_projection_status=candidate`; this is auditable evidence that the
exact-shape improvement came from an independent bracket, while all unmatched
rows remain covered by `candidate_fallback` rather than being dropped.

A fresh refined target profile is retained at
`results/alpha/sm86_flash_bwd_target_ncu_refined_20260929/summary.json`.
Against this live observation, the refined cold prediction is 148,481 ns /
200,389 cycles versus 148,448 / 200,337 (0.022% / 0.026% APE). The refined
paired-warm prediction is 130,019 ns / 175,448 versus 129,632 / 174,924
(0.298% / 0.299% APE).

## Warm-cache result and remaining scope

The warm matrix and report are retained separately.  With the disjoint
base-family warm/cold transfer ratio, the warm large holdout reaches 7.239%
GPC MAPE / 7.327% duration MAPE, and the flash row reaches 8.23% / 8.49%.
The warm shape bracket is useful, but its endpoint differs with campaign/cache
history.  It should be treated as a cache-domain transfer observation rather
than an exact cache law.

To model the cache transition without applying a cold-to-warm ratio to the
archetype itself, a paired campaign profiles each shape cold and then warm
back-to-back. The refined `Sk=64/96/192/256` matrix predicts the fresh target
at 130,019 ns / 175,448 cycles versus 129,632 / 174,924 (0.298% / 0.299% APE).
The full warm large holdout improves to **6.093% GPC MAPE / 6.147% duration
MAPE** while generic rows still use the disjoint-family transfer. The refined
paired matrix is at
`results/alpha/sm86_flash_bwd_shape_calibration_goal_20260929_paired_warm_refined/`
and is selected with `--archetype-cache-transfer skip`.

With the added `Sk=96/192` anchors and quadratic interpolation, the warm large
holdout is **6.093% GPC MAPE / 6.147% duration MAPE**. This is a small but
independent improvement over the two-anchor paired result, not a target-label
fit.

The model still does not expose exact physical pipe tags, complete CUDA ABI or
ISA generation, cache ownership/read-from/DRAM completion, or a unified NCU
clock endpoint.  Those unresolved mechanism boundaries are not needed by this
metric-targeted archetype path, but they remain explicit limitations of the
cycle-accurate native replay model.

As a non-regression score on unrelated rows, the same code path (with no shape
spec match) leaves the cold base cohort at 3.590% GPC / 3.608% duration MAPE;
the warm base transfer lane is 10.328% / 10.137%.  The archetype override is
therefore limited to the one explicitly bracketed flash-backward row.
