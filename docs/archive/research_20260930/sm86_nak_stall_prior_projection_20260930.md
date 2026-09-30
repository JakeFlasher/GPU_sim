# SM86 NAK/CUPTI stall-prior projection (2026-09-30)

## Purpose

This is a new, non-promoted endpoint committee member.  It asks whether
aggregate stall-state observations from independently generated Mesa NAK
operation bodies can provide a useful prior when the native trace does not
expose a physical pipe or issue-slot tag.  The prior is semantic only:
`long_scoreboard`, `imc_miss`, `short_scoreboard`, wait, branch, drain, and
miscellaneous sampled states are not renamed into NVIDIA route identities.

The implementation is `tools/sm86_nak_stall_prior_projection.py`.

## Research basis

The inspected GCStack+GCScaler source keeps fine-grained interval and CPI
categories separate, including compute, memory, synchronization, control,
empty-warp, and idle terms.  The inspected GPU cache-model source uses a
Fenwick/tree-based reuse-distance calculation and explicitly models warp pools,
coalescing, associativity, and MSHRs.  The inspected Ampere memory-model
repository uses black-box microbenchmarks and piecewise latency/bandwidth laws
rather than assuming undocumented cache internals.  These patterns motivate a
small semantic mixture plus a max/partial-order frontier, not invented pipe
tags.

## Inputs and leakage boundary

* Independent CUPTI family matrix:
  `results/alpha/sm86_nak_family_cupti_pcsampling_goal_20260930/family_cupti_matrix.json`.
  Six NAK operation families were launched 128 times each; all had zero CUPTI
  drops and 17--21 decoded PCs.  Long-scoreboard and instruction-cache-miss
  buckets dominate; reciprocal adds a larger short-scoreboard/MIO component.
* Native score traces contribute only replay frontiers and static/dynamic
  semantic opcode populations.
* The fit is cohort-disjoint: NCU labels from the opposite workload family
  train a log endpoint-ratio ridge surface.  Score-row NCU values are read only
  after prediction for APE reporting.
* Rows outside the feature hull become `candidate_fallback` with widened
  residual intervals; no row is refused as OOD.

The operation-group mapping is intentionally coarse: arithmetic NAK families
form the math/tensor prior, reciprocal forms the SFU prior, and the broad
family median is used for memory/control/special because no independent body
establishes tensor routing or cache ownership.

## Live score-only results

Against the fresh per-kernel-clock campaigns:

| score cohort | rows | GPC MAPE | duration MAPE | fallback rows |
|---|---:|---:|---:|---:|
| cold / base | 8 | 13.63% | 18.04% | 3 |
| cold / large | 7 | 15.28% | 20.31% | 0 |

Model wall time was about four seconds per cohort because it walks the retained
trace metadata.  These errors are far above the promoted frontier/archetype
projection, so this path remains a negative, interpretable committee member.

## Boundary evidence

The same NVIDIA CUPTI continuous PC-sampling injection that decoded the NAK
bridge was attached to two native benchmark binaries.  The HMMA application
did not reach a bounded completion/decoded-record state; the smaller SiLU
application reported its native timing but the CUPTI worker failed to write
the data file (`error : 2`) and exceeded a 45-second wall bound.  These are
runtime/collection boundaries, not a claim that PC sampling is unavailable:
the bridge and six-family campaigns remain the successful independent
aggregate evidence.

## What this does not close

The model does not expose per-instruction physical SM pipe/sub-unit or issue
slot, unrestricted CUDA ABI/SASS generation, cache-line ownership/read-from or
complete DRAM persistence, or a unified NCU/CUPTI/CUDA-event/simulator clock.
It is an approximation mechanism for endpoint prediction only.

## Sources read

* [GCStack+GCScaler source repository](https://github.com/yonsei-hpcp/gcstack_gcscaler),
  especially `gcscaler/src/interval_model.cc` and `src/worker.cc`.
* [GPU cache model](https://github.com/tue-es/gpu-cache-model), especially
  `src/model/reusedistance.cpp` and its warp-pool/MSHR path.
* [Ampere GPU memory model](https://github.com/brucelee503jo3/gpu-memory-model),
  including its piecewise latency and bandwidth model sources.
* [Accel-Sim/GPGPU-Sim 4 performance model](https://github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/gpgpu-sim4.md),
  for configurable specialized units and sectored cache abstractions.
* [CUPTI PC Sampling tutorial](https://docs.nvidia.com/cupti/tutorial/tutorial.html)
  and [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
  for the public aggregate sampling and endpoint-control contracts.
