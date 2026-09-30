# SM86 latent-endpoint lattice: blocker replacements

Updated 2026-09-29. This note records a new metric-oriented alternative to
recovering proprietary microarchitectural fields. It is deliberately not a
claim that the physical mechanisms have been discovered.

## Research synthesis

Several public simulators and predictors point to the same useful abstraction:
predict a small causal execution unit and aggregate it, instead of simulating
every hidden hardware transition.

* **NeuSight** decomposes a kernel into tiles, predicts tile throughput, and
  bounds/aggregates the result with execution laws. Its paper explicitly uses
  kernel names, input dimensions, thread-block counts, and tile sizes rather
  than requiring per-instruction hardware traces. See the [arXiv paper](https://arxiv.org/abs/2407.13853), especially the tile-size and
  tile-and-wave sections. Its public [source repository](https://github.com/scai-tech/NeuSight)
  separates dataset processing, model, operator-graph, prediction, and tracing
  code, which supports treating tile extraction and endpoint prediction as
  independent seams.
* **PPT-GPU** exposes a hybrid PTX/SASS trace path and separates a kernel,
  warp-scheduler, and reuse-profile memory model. Its public source tree
  contains `kernels.py`, `warp_scheduler.py`, `warps.py`, and
  `memory_model.py`; the project README describes the hybrid scalable
  trace-driven model. See [PPT-GPU](https://github.com/NMSU-PEARL/PPT-GPU).
* **Accel-Sim/GPGPU-Sim** retains control bits, scoreboard/dependence state,
  sub-core pipeline state, and a correlator rather than collapsing all causes
  into one scalar. The modern-GPU public fork documents separate sub-core and
  memory pipelines, control-bit interpretation, enhanced scoreboards, and
  static instruction metadata. See [modern-gpu-simulator-micro-2025]
  (https://github.com/upc-arco/modern-gpu-simulator-micro-2025) and the
  [Accel-Sim framework](https://accel-sim.github.io/).
* **Photon** uses representative functional/timing samples and reconstructs
  larger executions, providing a speed-oriented alternative to full detailed
  simulation. See the [Photon paper]
  (https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf).
* Nsight Compute documents why the target endpoint cannot simply be renamed to
  simulator completion: application replay, cache control, clock control,
  semaphore placement, and serialization change the measured duration. See
  [Nsight Compute's workload-duration guide]
  (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

The common design lesson is not to pretend that hidden fields are available;
it is to preserve the observable causal frontiers and learn a bounded endpoint
map over them.

## Model

The new executable model is
`tools/sm86_tile_lattice_projection.py`. Its feature vector contains only:

```text
I = sim_last_issue_cycles
E = sim_event_cycles
L = sim_gpc_envelope_cycles
T = max(E - I, 0)                  # typed completion/tail proxy
Q = max(L - E, 0)                  # launch-envelope proxy
W = sim_warp_instructions
G = trace block/thread population and trace-inst density
```

The default endpoint-lattice response is a ridge-fitted affine surface over
`I, E, L, T, Q, log(W), G`, plus two bounded products (`E*log(W)` and
`L*log(blocks)`). This retains separate issue/event/launch frontiers. It does
not feed NCU counters or aggregate pipe counters back into the features.

For a named shape archetype, a second local response surface is selected only
when:

1. the archetype, `BH`, `Sq`, and `D` match independent anchors exactly;
2. the target `Sk` lies inside two independent anchor values; and
3. the target NCU row is absent from the calibration matrix.

Within that bracket, each target metric is linearly interpolated and the
observed anchor range is retained as a conservative interval. Outside the
bracket, the local path retains the generic lattice prediction as
`candidate_fallback` with a distance-scaled interval; it never silently calls
that row shape-exact, but it also does not drop a score.

The existing `frontier-archetype` path remains the lower-error production
candidate. The lattice path is retained as a simpler independent model and a
committee/uncertainty member, not silently substituted where it regresses.

## Blocker substitutions

### Physical pipe and issue tags

Replace unavailable per-PC physical route tags with a **resource mixture**:
class-level instruction populations, issue frontier, scoreboard tail, and
aggregate route counters constrain a set of feasible route distributions. The
model predicts endpoint cost from the mixture; it does not assign one hidden
pipe to each dynamic instruction. Isolated instruction probes provide
class-local anchors, while mixed kernels retain a partial order/interval.

### CUDA ABI/ISA generation

Replace unrestricted native code generation with a **semantic launch IR**:
operation family, operand widths, tile shape, memory regions, synchronization,
and launch envelope. NVIDIA PTX compilation, Mesa NAK, and the bounded
NAK-to-CUDA-ELF bridge are producers of that IR/native artifact, not evidence
that a general independent CUDA ABI has been implemented. The endpoint model
can score an existing trace without generating a new cubin.

### Cache ownership and DRAM completion

Replace hidden owner/read-from state with a **cache-phase latent variable**:
`cold`, `paired-warm`, `steady-warm`, or `unknown`. Each phase has a transfer
interval measured from independent cold/warm pairs. The phase is a model input
or an explicit refusal, never an inferred proprietary ownership story. The
paired flash-backward campaign demonstrates why this matters: same-shape
`Sk=64/256` anchors predict the warm `Sk=128` target within 0.354% / 0.352%.

### Unified timing endpoint

Replace endpoint identity with an **endpoint transform**:

```text
simulator frontiers -> latent endpoint lattice -> NCU duration/GPC metric
```

The transform is calibrated from disjoint workloads and emits residual
intervals. It does not equate CUPTI, CUDA events, simulator completion, and
NCU semaphores. Nsight's documented start/end semaphore placement and cache
replay rules remain explicit endpoint metadata.

## Current measured results

`results/alpha/sm86_tile_lattice_goal_cold_large_20260929.json` gives the
simple lattice model 2.746% GPC and 2.862% duration MAPE on the seven-row large
holdout, at 1,048x projection-only speedup. Its flash-backward row is resolved
by the independent shape bracket rather than the generic distance model.

The paired-warm run at
`results/alpha/sm86_tile_lattice_goal_warm_paired_large_20260929.json` reaches
6.314% GPC / 6.367% duration MAPE at 1,165x speedup. The lower-error
frontier-archetype paired-warm run remains 6.101% / 6.151%; this confirms that
the lattice abstraction is competitive but should remain a committee member,
not a replacement selected after seeing the target labels.

With the refined `Sk=96/192` anchors, the corresponding lattice scores are
2.742% / 2.858% cold and 6.305% / 6.362% paired-warm. The refined
frontier-archetype path is 2.586% / 2.698% cold and 6.093% / 6.147% warm.

The fresh refined target lane measures 148,448 ns / 200,337 cycles cold and
129,632 ns / 174,924 cycles paired-warm. The quadratic archetype prediction is
within 0.022% / 0.026% cold and 0.298% / 0.299% warm.

The lower-error hierarchical candidate remains
`results/alpha/sm86_ncu_projection_frontier_archetype_goal_cold_20260929.json`:
2.589% / 2.702% MAPE and 1,216x speedup. This comparison is intentional: the
lattice is a novel bounded model, but no model is admitted merely for novelty
or because it hides a regression.

On the unrelated base holdout, the lattice is 4.377% / 4.370% MAPE, versus the
existing frontier-archetype path at 3.590% / 3.608%. That is a known regression
for the current small calibration set, so the lattice is not the default.

## What is and is not resolved

For the requested **NCU-oriented performance objective**, the combination of
frontier projection, shape-lattice interpolation, cache-phase transfer, and
interval refusal provides a practical replacement for all four missing fields.
It is much faster than detailed replay and is evaluated against held-out NCU
measurements.

For the stronger **cycle-accurate mechanism objective**, the blockers remain:
exact physical pipe/issue tags, unrestricted independent CUDA ABI/ISA and
control-word generation, complete cache ownership/read-from/DRAM semantics,
and a single public timing endpoint. The substitutions resolve their effects
as bounded latent variables; they do not prove those hidden mechanisms.
