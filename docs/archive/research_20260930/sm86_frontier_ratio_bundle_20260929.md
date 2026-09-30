# SM86 frontier-ratio bundle: interval analysis plus endpoint consistency

## Research motivation

This pass studied four complementary directions:

* **GPUMech interval analysis** models sustained issue rate and explicit stall
  intervals, including memory divergence and resource contention, instead of
  pretending every instruction has a known silicon latency.  See the
  [MICRO paper](https://hsienhsinlee.github.io/MARS/pub/micro14.pdf).
* **Photon** samples kernels, warps, and basic blocks online, using runtime
  phase behavior to avoid full detailed simulation.  It reports up to 24.65x
  speedup for single kernels with 6.83% average sampling error and 39.1x on a
  large ResNet case.  See the [Photon paper](https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf).
* **PipeWeave** computes analytical tensor/memory/XU/FMA pipeline features,
  then predicts a normalized efficiency ratio with a small model rather than
  regressing absolute latency directly.  Its public artifact exposes the
  feature-to-efficiency decomposition in
  [`README.md`](https://github.com/zksainx/pipeweave/blob/main/README.md).
* **Accel-Sim GPUVision/CUPTI** uses PM-sampling time series for cycle-level
  simulator correlation rather than one aggregate counter.  CUPTI's public
  PM-sampling tutorial exposes fixed GPU-sysclk sampling intervals and decoded
  sample timestamps.  See [GPUVision](https://github.com/accel-sim/accel-sim-framework)
  and the [CUPTI tutorial](https://docs.nvidia.com/cupti/tutorial/tutorial.html).

These clues suggest predicting **efficiency relative to a measured/analytical
frontier** and cross-checking the endpoint domains, not learning one absolute
NCU scalar.  The executable `tools/sm86_frontier_ratio_bundle.py` implements
that idea for the current SM86 cohort.

## Model

For each row, the model constructs only replay/trace features:

```text
GPC base      = max(sim_gpc_envelope_cycles, sim_event_cycles)
duration base = sim_event_cycles × (1,000 / SM_clock_MHz) ns
ratio target  = log(NCU_metric / base)
```

An independent-cohort ridge surface predicts each ratio.  A second independent
surface predicts `duration/GPC`; duration is geometrically blended with that
crosswalk to prevent the two endpoint predictions from drifting arbitrarily.
The residual interval widens with nearest-training frontier distance.  NCU
labels from the scored campaign are loaded only after prediction.

The model does **not** infer physical pipes, issue slots, cache owners,
read-from sources, ABI metadata, or equivalence of timing clocks.  Those remain
latent residuals.

## Fresh score-only result

Reports are under
`results/alpha/sm86_frontier_ratio_bundle_goal_20260929/`:

| holdout | phase | GPC MAPE | duration MAPE | model time |
|---|---|---:|---:|---:|
| large | cold | 8.339% | 8.752% | 0.0066 s |
| base | cold | 3.736% | 4.363% | 0.0057 s |
| large | warm | 9.566% | 9.811% | 0.0057 s |
| base | warm | 13.993% | 12.528% | 0.0057 s |

This is a **negative accuracy result** against the promoted frontier/archetype
model, not a promotion.  It is retained because its ratios and cross-endpoint
consistency terms are interpretable and can become useful when paired with
cycle-level PM time series.  The current result also falsifies a tempting
hypothesis: a simple analytic frontier normalization does not transfer across
these related SM86 families without archetype/cache conditioning.

## Four next abstractions

The negative result is informative: the next model must condition latent state,
not add more absolute regressors.

1. **PM-state semi-Markov envelope.** Partition a kernel into online phases
   using sampled `sm__cycles_elapsed`, pipe activity, scoreboard/stall, and
   DRAM/L2 emissions. A small hidden-state model can infer probabilities of
   `issue`, `memory-queue`, `barrier`, and `launch-tail` phases; each state
   contributes an interval to the frontier ratio. GPUVision/CUPTI samples are
   aggregate emissions, never per-PC physical route labels.
2. **Cache intervention signatures.** Treat public flush/invalidate controls,
   cold/warm launches, and CPU-visible reads as interventions in a causal
   graph. Predict an equivalence class of read-from/ownership worlds that agree
   on all interventions. Optimize endpoint accuracy over that class while
   retaining `owner=unknown`, rather than selecting one unobservable owner.
3. **Capability-token ABI IR.** Lower NAK/PTX/trace semantics into a small
   vendor-neutral IR whose values carry launch-ABI capability tokens. NVIDIA
   PTX/JIT and the bounded NAK bridge become two producers; metadata/QMD/
   control-word uncertainty is an interval attached to the token. This avoids
   claiming an unrestricted CUDA ELF generator while predicting timing for
   bodies with the same token class.
4. **Monotone multi-clock alignment.** Fit a monotone latent-work axis to
   simulator frontiers, CUDA events, CUPTI samples, and NCU duration/GPC
   labels. Constrain maps to preserve launch/issue/complete order and use
   endpoint residuals as a separate uncertainty term. The output is a
   calibrated crosswalk, not a declaration that tools share a clock.

These ideas directly target the four blockers using existing public
observations. None manufactures a physical pipe/owner/ABI/timing witness; the
goal is to make its effects useful and bounded at much lower runtime than full
cycle replay.

## Blocker value

| blocker | abstraction supplied | remaining gap |
|---|---|---|
| physical pipe/issue | interval efficiency and separate issue/event/launch bases | no dynamic physical route witness |
| CUDA ABI/ISA | endpoint residual absorbs compiler/template effects | no unrestricted SASS/ELF/QMD generation |
| cache ownership/read-from | phase-sensitive base/ratio can be conditioned on future PM series | no line owner/read-from/DRAM proof |
| unified timing | duration/GPC crosswalk is explicitly separate | no public clock-equivalence relation |

The next useful extension is to feed **time-binned CUPTI PM samples** into the
ratio surface as a phase trajectory, using them as aggregate observations and
not as per-instruction route labels.  That would test whether the same latent
efficiency state explains NCU duration, GPC cycles, and PM stall populations.
