---
Use when: reasoning about warp scheduling, issue-slot contention, scoreboard effects, or scheduler-derived features for non-root performance prediction.
Primary raw source: references/papers/knowledge_base/active/2025__dissecting_and_modeling_the_architecture_of_modern_gpu_cores.pdf.
Supporting sources:
- references/papers/knowledge_base/active/2024__analyzing_and_improving_hardware_modeling_of_accel_sim.pdf
- references/papers/knowledge_base/active/2018__dissecting_the_nvidia_volta_gpu_architecture_via_microbenchmarking.pdf
Repo touchpoints:
- workloads/simbench_bw/
- tools/nvbit_tracer/
- .claude/knowledge/best-practices/perf-instruction-controlflow.md
Out of scope: active nvasim scheduler implementation, NCU warp-state metrics, and simulator validation.
Version family: Modern NVIDIA cores through Blackwell, with SM120 as primary target.
---

# Warp Scheduling and Issue Context

Warp scheduling matters because it shapes latency hiding, structural hazards,
and dynamic instruction timing. In this fork, scheduler behavior is used as
prior art for feature design and interpretation, not as an active simulator
component.

## Useful Concepts

- Modern SMs are divided into scheduling domains often described as SubCores.
- Resident warps hide latency by issuing while other warps wait on memory,
  scoreboard dependencies, barriers, or structural hazards.
- SASS control bits and dependency barriers can influence issue timing.
- Register pressure, shared memory, and block size determine how many warps are
  available to the scheduler.

## Allowed Evidence

- Static SASS control bits where decoded by tooling.
- Instruction class mix and dependency structure.
- NVBit dynamic traces in bounded experiments.
- CUDA-event or CUPTI Activity elapsed time under occupancy, instruction-mix,
  or branch-shape variants.
- Public microbenchmark papers.

Forbidden evidence:

- NCU `WarpStateStats`, scheduler sections, or stall-reason counters.
- CUPTI profiling/hardware-counter surfaces.
- Active nvasim scheduler output as current ground truth.

## See Also

- `.claude/knowledge/best-practices/perf-occupancy-launch.md`
- `.claude/knowledge/best-practices/perf-instruction-controlflow.md`
- `.claude/knowledge/cuda/cuda-register-file.md`
- `.claude/knowledge/measurement/nonroot-cloud-contract.md`

## Escalate to Raw When

- A claim depends on exact scheduling policy for a specific architecture.
- A new paper revises the known Blackwell or Hopper scheduling model.
