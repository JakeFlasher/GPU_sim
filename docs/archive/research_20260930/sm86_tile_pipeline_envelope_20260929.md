# SM86 tile/resource pipeline envelope (2026-09-29)

## Purpose

This note records a new **committee** predictor motivated by recent tile- and
pipeline-oriented GPU performance work.  It is deliberately not a replacement
for native replay and does not close any of the four external mechanisms.  The
model turns each supported shape into three semantic demand classes and uses a
max-plus endpoint envelope:

```text
math         = independent ALU + FMA + tensor population
memory       = independent LSU population
instruction  = independent total instruction population
endpoint     = max(math / rate_math,
                   memory / rate_memory,
                   instruction / rate_instruction)
```

The classes are **not** physical SM86 pipe or sub-unit tags.  The rates are
median demand/endpoint rates from independent calibration rows.  A q10/q90
rate envelope is retained as an uncertainty interval, not a confidence
interval.  Resource populations are first interpolated in `Sk` only inside an
exact `BH/Sq/D` anchor hull.  Rows outside that hull are covered by an
independent frontier/trace fallback with a distance-scaled interval; the
coverage output never refuses a workload.

The implementation is `tools/sm86_tile_pipeline_envelope.py`.  It reads NCU
labels only from an explicitly independent anchor matrix and loads score-row
NCU values after prediction for APE reporting.  The target row cannot enter the
fit and target pipe counters are never features.

## Literature and source-code clues

* **NeuSight** decomposes a kernel into smaller tiles and aggregates
  tile-granularity predictions.  Its public predictor source uses separate
  operation predictors, derives memory/operation-per-output features, and
  aggregates phase latency rather than fitting one opaque whole-kernel scalar.
  See [arXiv:2407.13853](https://arxiv.org/abs/2407.13853), the
  [paper implementation](https://github.com/scai-tech/NeuSight), and the
  operation aggregation in
  [`neusight/Prediction/predictor.py`](https://github.com/scai-tech/NeuSight/blob/main/neusight/Prediction/predictor.py).
* **TileSight** makes a tile resource vector, legal dependency/order envelope,
  prologue/steady-state/epilogue phases, and tile reuse-distance cache model
  first-class.  We adopt the resource-vector and envelope discipline, but keep
  cache ownership/read-from semantics explicitly unknown on NVIDIA.
  See [arXiv:2607.22432](https://arxiv.org/abs/2607.22432).
* **PipeWeave/SynPerf** combine analytical heterogeneous-pipeline demand with a
  lightweight learned interaction model.  Here the interaction is replaced by
  a transparent max envelope so no same-workload hidden label is needed.
  See [arXiv:2601.14910](https://arxiv.org/abs/2601.14910).
* **PPT-GPU** uses scalable analytical memory/reuse profiles and accepts PTX or
  SASS inputs.  Its reuse-profile idea motivates keeping memory as a separate
  demand class rather than assigning a guessed physical owner.
  See the [PPT-GPU repository](https://github.com/NMSU-PEARL/PPT-GPU) and its
  [reuse-profile paper](https://doi.org/10.1145/3291051).
* **GPGPU-Sim/Accel-Sim** source separates cycle stages, scoreboard checks,
  logical pipeline availability, memory requests, and response/retirement.
  In the public `shader.cc`, `cycle()` calls writeback, execute, operand read,
  issue, decode, and fetch separately; the scheduler checks the scoreboard
  before choosing a logical output register set.  This is the same reason the
  new model reports competing frontiers instead of inventing one physical
  route.  See the [GPGPU-Sim source](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/dev/src/gpgpu-sim/shader.cc)
  and [Accel-Sim framework](https://github.com/accel-sim/accel-sim-framework).

## Live score-only result

The fresh full campaigns remain the independent score domain:

| score campaign | supported rows | cold/warm GPC APE | cold/warm duration APE | model time | replay wall | projection speedup |
|---|---:|---:|---:|---:|---:|---:|
| `full_3090ti_20260929_refined_cold_live` | 15/15 covered; 1 exact + 14 fallback | 3.112% | 3.050% | ~0.01 s/family | ~0.666 s total | coverage path |
| `full_3090ti_20260929_refined_warm_live` | 15/15 covered; 1 exact + 14 fallback | 8.421% | 8.483% | ~0.01 s/family | ~0.666 s total | coverage path |

Machine-readable reports are:

* `results/alpha/sm86_tile_pipeline_envelope_goal_20260929/cold_large.json`
* `results/alpha/sm86_tile_pipeline_envelope_goal_20260929/warm_large.json`
* `results/alpha/sm86_tile_pipeline_envelope_goal_20260929/coverage_summary.json`

The matching `cold_base.json` and `warm_base.json` reports cover the eight base
rows; the summary combines all fifteen fresh workloads.  Exact-shape flash
APE remains 0.493%/0.483% cold and 0.273%/0.265% warm.  Fallback rows are
explicitly marked `candidate_fallback`, not silently relabeled as exact.

For the cold score row the envelope predicts `200,708.8` GPC cycles and
`148,715.5` ns versus `199,725` and `148,000` measured by fresh NCU.  The
paired-warm prediction is `175,571.8` cycles and `130,103.3` ns versus
`175,094` and `129,760` measured.  The promoted quadratic frontier-archetype
path remains lower-error on this row (`0.333%/0.325%` cold and
`0.202%/0.199%` warm on the same fresh campaign), so this envelope is retained
as a more interpretable committee member rather than promoted by novelty.

The envelope's broad anchor-rate intervals are intentionally conservative.  A
single target row resolves exactly because it has four independent
`BH=4,Sq=16,D=16` anchors at `Sk=64,96,192,256`; the other fourteen rows use
the independent frontier/trace fallback.  Thus every fresh workload has a
prediction and interval, while exact-shape provenance remains visible.

## Blocker mapping

| unresolved boundary | what this abstraction contributes | what it still cannot claim |
|---|---|---|
| physical pipe/issue tags | semantic resource demand classes and a partial-order bottleneck envelope | per-PC physical sub-unit, issue slot, or dynamic route |
| unrestricted CUDA ABI/ISA | shape/resource endpoint behavior can be scored from existing trace/replay inputs | general SASS/control-word/ELF/QMD generation |
| cache ownership/read-from/DRAM completion | cold and paired-warm anchor matrices become separate endpoint rate domains | line owner, read-from source, dirty payload, or complete DRAM persistence |
| unified timing endpoint | duration and GPC cycle rates are predicted independently | equivalence of NCU WaitForIdle, CUPTI, CUDA event, device timer, and simulator completion |

The model therefore makes progress on the requested accuracy/runtime target
without renaming an unobserved mechanism as “resolved.”  Additional anchors
can replace fallback rows, but no workload is dropped merely because its
shape is absent.
