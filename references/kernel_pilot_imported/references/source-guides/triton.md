# Triton Deep Reference

> **SOLBENCH NOTICE.** This file was imported from kernel-pilot. solbench's
> submission surface is **CUDA C++ / CUTLASS C++ / CuTe C++ ONLY** (`CLAUDE.md` Rule 5).
> Triton / CuTe DSL Python / cuTile Python / `cudnn_frontend` Python / `torch.compile` /
> TileLang / ThunderKittens / QuACK referenced below are **algorithm references only** —
> lift the algorithm into CUDA C++ / CUTLASS C++ for the *.cu hand-port. solbench is
> also counter-free (no NCU per Rule 1); counter-free profiling lives in `scripts/baselines_harness/`.

Repository: <https://github.com/triton-lang/triton>

Source-only policy: do not query PR notes for this repository unless the PR
corpus reaches at least 10 selected CUDA optimization PRs. Use this source
guide and current upstream source paths directly. (The kernel-source-catalog was dropped during the post-pivot trim — query GitHub directly for repo metadata.)

Use this when the user asks for Triton, the baseline is Triton or Inductor
generated code, or the target framework's hot kernel is written in Triton.

## Read Order

1. Tutorial closest to the operator.
2. Target framework's Triton kernel and wrapper.
3. Autotune key/config definitions.
4. Lowering/backend tests when SASS or scheduling is surprising.
5. Benchmark harness and p50/p90 reporting.

## Code Map

| Area | Paths to inspect |
| --- | --- |
| Tutorials | `python/tutorials/` |
| Language primitives | `python/triton/language/` |
| Runtime / autotune | `python/triton/runtime/` |
| Compiler lowering | `lib/`, `third_party/nvidia/` |
| Backend tests | `test/TritonGPU/`, `test/` |

## Search Patterns

```bash
rg -n "tl\\.dot|tl\\.load|tl\\.store|block_ptr|make_block_ptr|program_id|num_warps|num_stages|autotune|Config" python test
rg -n "TMA|warp_specialize|persistent|split_k|matmul|softmax|layer_norm|attention|moe" python test lib third_party/nvidia
```

## Candidate Use

- **Triton submissions are forbidden by Rule 5.** Triton code is an algorithm reference; lift the algorithm (tile shape, autotuning heuristic, fused-epilogue plan) into CUDA C++ / CUTLASS C++ for the *.cu hand-port when the user or baseline calls for the same algorithm.
- Record source path, commit, autotune config, compile flags, and first delta.
- Always include warmup and compile-time notes because Triton JIT effects can
  hide true kernel latency.

