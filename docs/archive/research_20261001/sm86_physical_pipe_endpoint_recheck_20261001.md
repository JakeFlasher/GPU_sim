# SM86 physical-pipe endpoint recheck — 2026-10-01

**Scope:** RTX 3090 Ti / GA102 / SM86, current public Nsight Compute and
open NVIDIA kernel-module surfaces.

## Finding

No supported public endpoint was found that emits a dynamic tuple of
`(physical SM, SMSP, pipe/subunit, issue slot, cycle, PC/warp)`. The simulator
must therefore retain route identity as a candidate domain and use aggregate
pipeline observations only as bounds or comparison targets.

## Evidence

Nsight Compute's current Profiling Guide describes
`sass__inst_executed_per_opcode_pipeline` as an **estimated** opcode-pipeline
count and explicitly says that dynamic assignment is not taken into account:

<https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>

The same guide describes SM subpartitions as containing the warp scheduler,
register file, and execution units/pipelines. This identifies the logical
resource hierarchy but does not expose the dynamic physical path chosen by an
individual instruction.

The current open NVIDIA kernel-module request for globally identifiable,
time-resolved per-SM/SMSP samples remains open. Its description states that
existing aggregate `sm__*`/`smsp__*` metrics, warp-ID sampling, CUPTI, and PTX
`%smid`/`%warpid` do not provide a stable physical SM/SMSP coordinate for each
sample:

<https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1351>

## Modeling consequence

The existing `PhysicalPipelineAttribution` candidate classes, model-local
earliest/latest issue/completion intervals, operand-bank diagnostics, and
`sim-semantics::latent` unknown route factor remain the strongest admissible
software response. A new aggregate metric, balanced assignment, or scheduler
microbenchmark cannot be promoted to a per-instruction physical route.

This recheck does not alter NCU's role: local RTX 3090 Ti NCU measurements
remain the detailed silicon comparison target, while published Accel-Sim
aggregates remain contextual lower-bound references rather than ground truth.
