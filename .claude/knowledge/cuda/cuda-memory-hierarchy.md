---
Use when: reasoning about cache organization, sector sizes, shared memory, L2, DRAM, or memory-hierarchy features for non-root cache-behavior prediction.
Primary raw source: references/NVIDIA_Manuals/best_practices/10_memory-optimizations.md.
Supporting sources:
- references/NVIDIA_Manuals/blackwell_tuning/01_nvidia-blackwell-tuning-guide.md
- references/papers/knowledge_base/active/2018__exploring_modern_gpu_memory_system_design_challenges_through_accurate_modeling.pdf
- references/papers/knowledge_base/active/2015__dissecting_gpu_memory_hierarchy_through_microbenchmarking.pdf
Repo touchpoints:
- workloads/simbench_bw/
- tools/nvbit_tracer/
- .claude/knowledge/best-practices/perf-coalescing-memory.md
Out of scope: active simulator memory-model implementation, NCU cache metrics, and deleted parent-repo memory calibration.
Version family: Volta through Blackwell; re-check raw docs for new CUDA releases.
---

# GPU Memory Hierarchy

This card preserves architectural memory-hierarchy context for the current
prediction project. Cache hit ratio is a target to predict or bound from
allowed signals, not a hardware counter to read with NCU.

## Stable Facts

- Global-memory coalescing uses 32-byte sectors on modern NVIDIA GPUs.
- L1, texture, and shared memory share an on-SM SRAM pool whose capacity and
  carveout rules vary by architecture.
- Shared memory uses banked access; bank conflicts serialize.
- L2 is chip-wide and partitioned.
- DRAM type, L2 capacity, and shared-memory ceilings vary substantially between
  B200-class datacenter Blackwell and sm_120 workstation/consumer Blackwell.

## Current Project Feature Sources

Allowed inputs for memory/cache prediction:

- Static SASS/PTX load/store instruction mix.
- Address-space qualifiers and cache operators.
- Static resource usage and shared-memory carveout settings.
- NVBit dynamic address traces for bounded experiments.
- CUDA-event or CUPTI Activity elapsed-time response to access-pattern variants.
- Read-only clock, temperature, and throttle covariates.

Forbidden inputs:

- NCU L1/L2 hit/miss metrics.
- CUPTI hardware-counter metrics or profiling APIs.
- Simulator-generated cache-hit ratios presented as current ground truth.

## Prior-Art Use

Memory-system modeling papers are retained to explain why cache and coalescing
features matter. They should be cited as prior art; the ACTIVE simulator build
(re-founded 2026-07-02) is the primary consumer of this card when designing
its memory-hierarchy model.

## See Also

- `.claude/knowledge/best-practices/perf-coalescing-memory.md`
- `.claude/knowledge/blackwell/blackwell-tuning.md`
- `.claude/knowledge/measurement/nonroot-cloud-contract.md`
- `.claude/knowledge/measurement/nvbit-injection-playbook.md`

## Escalate to Raw When

- You need exact shared-memory carveout values for a compute capability.
- A claim depends on L2 size or memory-controller details for a specific SKU.
- A CUDA release changes persistence or cache-policy APIs.
