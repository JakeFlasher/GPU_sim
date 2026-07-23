---
Use when: reasoning about register pressure, register spills, register-file bank conflicts, or register-derived features for non-root performance prediction.
Primary raw source: references/papers/knowledge_base/active/2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks.pdf.
Supporting sources:
- references/papers/knowledge_base/active/2025__dissecting_and_modeling_the_architecture_of_modern_gpu_cores.pdf
- references/papers/knowledge_base/active/2018__dissecting_the_nvidia_volta_gpu_architecture_via_microbenchmarking.pdf
Repo touchpoints:
- workloads/simbench_bw/
- tools/nvbit_tracer/
- .claude/knowledge/best-practices/perf-occupancy-launch.md
Out of scope: active nvasim register-file implementation, NCU stall metrics, and simulator validation.
Version family: Volta through Blackwell, with SM120 as the primary target.
---

# Register File Architecture

Register pressure matters in this fork because it affects occupancy, spills,
instruction scheduling, and elapsed time. The current project should use static
build output, SASS/PTX, NVBit traces, and timing variants rather than NCU stall
metrics or simulator internals.

## Facts to Preserve

- Modern NVIDIA SMs expose a large per-SM register file.
- Registers are allocated per thread at kernel launch.
- `ptxas -v` and `cuobjdump --dump-resource-usage` can report per-kernel
  register counts without GPU counter access.
- High register counts can reduce resident blocks/warps.
- Spills appear as local-memory traffic and can be found through compiler
  output and static SASS (`LDL` / `STL`).

## Bank-Conflict Context

Microbenchmark literature reports banked register-file behavior and possible
recent-register caching effects. Those details are useful as features or prior
art, but any current quantitative claim must be supported by public sources or
non-root measurements.

Allowed feature sources:

- SASS register operands.
- Static instruction classes and operand counts.
- `ptxas -v` register and spill reports.
- NVBit traces with bounded register/address instrumentation.
- Elapsed-time response to `-maxrregcount`, `__launch_bounds__`, or block-size
  variants.

## See Also

- `.claude/knowledge/best-practices/perf-occupancy-launch.md`
- `.claude/knowledge/cuda/cuda-warp-scheduling.md`
- `.claude/knowledge/sass/sass-tooling.md`
- `.claude/knowledge/measurement/nonroot-cloud-contract.md`

## Escalate to Raw When

- A claim depends on exact register bank count, allocation granularity, or RF
  cache behavior.
- A CUDA upgrade changes resource-usage reporting.
