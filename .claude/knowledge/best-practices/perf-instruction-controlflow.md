---
Use when: tuning CUDA instruction mix, control flow, warp divergence, synchronization, or math precision under the non-root measurement contract.
Primary raw source: references/NVIDIA_Manuals/best_practices/12_instruction-optimization.md and references/NVIDIA_Manuals/best_practices/13_control-flow.md.
Supporting sources: references/NVIDIA_Manuals/cuda_guide/02-02-04-04_synchronization-functions.md.
Repo touchpoints: workloads/simbench_bw/, tools/nvbit_tracer/, .claude/knowledge/measurement/nonroot-cloud-contract.md.
Out of scope: NCU InstructionStats/WarpStateStats workflows, hardware-counter stall analysis, and simulator validation.
Version family: CUDA 13.x, CC 7.5-12.0.
---

# Instruction Optimization and Control Flow

Use static code, static SASS/PTX, NVBit traces, and elapsed-time experiments to
reason about instruction and control-flow behavior. Do not use NCU sections or
hardware-counter stall metrics as active project inputs.

## Throughput Tables

NVIDIA best-practices throughput tables are sparse. If a CC 10.0 or 12.0 cell is
blank in the raw guide, do not fill it from an older architecture. Measure with
allowed elapsed-time microbenchmarks or cite a public source.

## Math Flags

| Lever | Effect |
|---|---|
| `-ftz=true` | Flush denormals to zero. |
| `-prec-div=false` | Faster, less precise division. |
| `-prec-sqrt=false` | Faster, less precise square root. |
| `--use_fast_math` | Bundles several lower-precision choices. |
| Intrinsics such as `__expf`, `__sinf`, `__fdividef` | Faster, lower precision than full libdevice paths. |

Record math-mode choices in workload metadata when they affect numeric
correctness or timing comparability.

## Warp Divergence

- Uniform branch within a warp is effectively free.
- Per-thread divergence serializes paths.
- Short branches may be predicated by the compiler.
- Use `__syncwarp()` when Volta+ independent thread scheduling could leave
  threads diverged past a point that assumes warp convergence.

Allowed diagnostics:

- Static control-flow reconstruction from SASS.
- NVBit active-mask or branch-target traces.
- Elapsed-time response under branch-shape variants.

## Synchronization

| Primitive | Scope |
|---|---|
| `__syncwarp()` | Calling warp. |
| `__syncthreads()` | Thread block. |
| `cluster.sync()` | Thread-block cluster. |
| `__threadfence*()` | Memory ordering, not a thread-count barrier. |

Count synchronization in hot loops. If only one warp communicates through
shared memory, prefer warp-scope synchronization where correct.

## Division, Modulo, and Loop Counters

- Runtime division and modulo are expensive.
- Compile-time constants usually lower to cheaper multiply/shift sequences.
- Signed loop counters often optimize better than unsigned counters because
  signed overflow rules give the compiler more freedom.

## See Also

- `.claude/knowledge/best-practices/perf-coalescing-memory.md`
- `.claude/knowledge/best-practices/perf-occupancy-launch.md`
- `.claude/knowledge/ptx/ptx-isa-control-flow.md`
- `.claude/knowledge/ptx/ptx-isa-arithmetic.md`
- `.claude/knowledge/measurement/nvbit-injection-playbook.md`

## Escalate to Raw When

- You need exact raw-guide throughput entries.
- You need architecture-specific predication behavior for a specific compiler.
- A numerical result depends on fast-math or denormal behavior.
