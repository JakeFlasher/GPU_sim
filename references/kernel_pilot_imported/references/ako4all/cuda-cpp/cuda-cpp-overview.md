# CUDA C++ Kernel Overview (Vendored Reference)

> **SOLBENCH NOTICE.** This file was imported from kernel-pilot. solbench's
> submission surface is **CUDA C++ / CUTLASS C++ / CuTe C++ ONLY** (`CLAUDE.md` Rule 5).
> Triton / CuTe DSL Python / cuTile Python / `cudnn_frontend` Python / `torch.compile` /
> TileLang / ThunderKittens / QuACK referenced below are **algorithm references only** —
> lift the algorithm into CUDA C++ / CUTLASS C++ for the *.cu hand-port. solbench is
> also counter-free (no NCU per Rule 1); counter-free profiling lives in `scripts/baselines_harness/`.

> **Adapter note for `gpu-kernel-ako4all`**: This file was copied from
> `https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cuda-cpp-kernel`
> (`SKILL.md`). It is kept here as a **document, not an invocable skill**. The
> original frontmatter (`name`, `description`, `argument-hint`,
> `user-invocable: true`) has been removed so no skill loader treats this as a
> standalone skill.
>
> Path mapping for the bundled materials it references:
>
> | Original (cache-dit) path           | Path inside this skill                         |
> | ----------------------------------- | ---------------------------------------------- |
> | `references/ptx-docs/`              | `references/cuda-cpp/vendored-docs/ptx-docs/`  |
> | `references/ptx-simple/`            | `references/cuda-cpp/vendored-docs/ptx-simple/`|
> | `references/cuda-runtime-docs/`     | `references/cuda-cpp/vendored-docs/cuda-runtime-docs/` |
> | `references/cuda-driver-docs/`      | `references/cuda-cpp/vendored-docs/cuda-driver-docs/`  |
> | `references/cuda-guide/`            | `references/cuda-cpp/vendored-docs/cuda-guide/`        |
> | `references/best-practices-guide/`  | `references/cuda-cpp/vendored-docs/best-practices-guide/` |
> | `references/nsys-docs/`             | `references/cuda-cpp/vendored-docs/nsys-docs/` |
> | `references/debugging-tools.md`     | `references/cuda-cpp/vendored-docs/debugging-tools.md` |
> | `references/performance-traps.md`   | `references/cuda-cpp/vendored-docs/performance-traps.md` |
> | top-level `smXX-optimization-guide.md` (sm90 only; sm89 NOT imported per solbench env) | `references/architectures/smXX-optimization-guide.md` |
> | top-level `smXX-optimization-guide.md` (sm100/103/120) | `references/cuda-cpp/smXX-optimization-guide.md` (kept stack-specific) |
> | top-level `kernel-templates.md`     | `references/kernel-templates.md`               |
> | top-level `troubleshooting.md`      | `references/troubleshooting.md`                |
>
> Treat the vendored doc tree as **opt-in**: only open a file when narrowing to
> a specific PTX instruction, CUDA API, or NSYS section (NCU dropped per solbench env). Do not load the
> entire mirror into context.

# CUDA C++ and PTX Kernel Development

## Goal

Use the bundled CUDA, PTX, and profiling references in this directory to implement, debug, and optimize CUDA kernels without relying on agent-specific install paths or ad-hoc web searches.

## Bundled Reference Map (after path mapping above)

Vendored mirror, opt-in only:

- `vendored-docs/ptx-docs/` — full PTX ISA reference
- `vendored-docs/ptx-simple/` — condensed PTX quick reference
- `vendored-docs/cuda-runtime-docs/` — CUDA Runtime API reference
- `vendored-docs/cuda-driver-docs/` — CUDA Driver API reference
- `vendored-docs/cuda-guide/` — CUDA Programming Guide
- `vendored-docs/best-practices-guide/` — CUDA C++ Best Practices Guide
- `vendored-docs/nsys-docs/` — Nsight Systems docs
- `vendored-docs/debugging-tools.md` — debugging workflow notes
- `vendored-docs/performance-traps.md` — common optimization traps

Always-load architecture / template / troubleshooting files:

- ~~`../architectures/sm90-optimization-guide.md` — Hopper~~ (DROPPED per solbench trim; not in solbench hardware path)
- `sm100-optimization-guide.md` — Blackwell datacenter (CUDA-specific)
- `sm103-optimization-guide.md` — Blackwell Ultra (CUDA-specific)
- `sm120-optimization-guide.md` — Blackwell desktop (CUDA-specific)
- `../kernel-templates.md` — kernel template families
- `../troubleshooting.md` — compute-sanitizer / debugger troubleshooting

## How to Search the Bundle

Prefer narrow text search over loading large reference files into context.

1. Identify the exact instruction, API, metric, or concept.
2. Search the narrowest relevant subdirectory first.
3. Read only the matching file or a short relevant range.
4. Translate the documentation into the specific kernel or operator constraint you are implementing.

Typical search targets:

- PTX instruction syntax: `vendored-docs/ptx-docs/9-instruction-set/`
- Quick PTX lookup: `vendored-docs/ptx-simple/`
- CUDA Runtime APIs: `vendored-docs/cuda-runtime-docs/modules/`
- CUDA Driver APIs: `vendored-docs/cuda-driver-docs/modules/`
- programming-model behavior: `vendored-docs/cuda-guide/`
- timeline and launch behavior: `vendored-docs/nsys-docs/UserGuide.md`

## Architecture-Specific Profiling Workflow

Interpret `nsys` (NCU removed; solbench is counter-free) results in the context of the target architecture rather than treating all GPUs the same.

1. ~~**sm89 (Ada)**~~: NOT IMPORTED for solbench — Ada Lovelace (L40S, RTX 4090) is not solbench's target. solbench targets sm_100a (B200); see the sm100/sm103 guides below.
2. **sm90 (Hopper)**: focus on TMA overlap, warpgroup behavior, shared-memory staging, and whether the timeline shows good load/compute overlap.
3. **sm100 / sm103 (Blackwell datacenter)**: focus on `tcgen05` usage, TMEM behavior, TMA v2 overlap, cluster behavior, and whether the kernel actually benefits from datacenter features.
4. **sm120 (Blackwell desktop)**: treat closer to Ada than datacenter Blackwell — watch memory throughput, L2 hit rate, shared-memory limits, lack of TMEM or cluster, and decide explicitly between TMA or `cp.async` for staging.

Recommended order:

1. Read the matching `smXX-optimization-guide.md` file.
2. Use `nsys` to identify launch gaps, overlap, copy/compute concurrency, end-to-end bottlenecks.
4. Compare observations against the architecture guide before changing tile shapes, pipelines, or memory movement.

## Implementation Checklist

Before changing code, answer:

1. Exact shape, dtype, and layout contract?
2. Architectural assumptions: SM target, shared-memory budget, alignment, Tensor Core mode?
3. Is the bottleneck compute, memory, launch overhead, or synchronization?
4. Which CUDA Runtime, Driver, or PTX rules must be preserved?
5. What verification proves correctness and a sufficient performance gain?

## Debugging Workflow

1. Reproduce with the smallest failing input.
2. Confirm the failure mode: wrong value, launch error, illegal memory access, race, hang, or perf regression.
3. For shared memory / `cp.async` / async pipelines, treat data-sync bugs as the first hypothesis. If only some shapes or stage counts fail, inspect barriers, slot reuse, and partial-tile predicates before suspecting the math.
4. Use `compute-sanitizer` or `cuda-gdb` for correctness.
5. Use `nsys` first for end-to-end bottlenecks, then static-introspection (cuobjdump/nvdisasm/ptxas -v) for per-kernel evidence.
6. Re-run focused correctness after every change before broader benchmarks.

## Performance Workflow

Never optimize by intuition alone.

1. Establish a baseline wall-clock.
2. Use `nsys` to see where time is spent.
4. Change one dimension at a time: tile shape, memory movement, sync, vectorization, or epilogue.
5. Re-measure and compare against the previous version.

If the result differs across GPU generations, consult the matching `smXX-optimization-guide.md` before generalizing.

## Validation Requirements

Minimum:

1. Add/update unit tests for correctness.
2. Compare numerical accuracy against a PyTorch / trusted eager reference.
3. Compare performance against that baseline when the task claims a perf benefit or replaces a baseline path.
4. Record exact benchmark setup: shapes, dtypes, device, warmup, iterations, timing method.

Additional for rewrites:

1. Compare new vs pre-rewrite operator on both accuracy and performance.
2. Treat the PyTorch baseline and the pre-rewrite operator as separate comparison targets.
