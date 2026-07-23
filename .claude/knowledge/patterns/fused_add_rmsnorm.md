---
Use when: implementing the fused residual-add + RMSNorm pattern `output = rmsnorm(hidden_states + residual, weight, eps)` used twice per decoder layer in transformer models. Covers BF16 inputs with FP32 reduction, hidden-size families 2048 / 4096 / 7168, and Blackwell cluster-cooperative reductions on sm_100a.
Primary raw source: sol-execbench/examples/cuda_cpp/rmsnorm/kernel.cu
Supporting sources: references/NVIDIA_Manuals/cuda_guide/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: sol-execbench/examples/cuda_cpp/rmsnorm/kernel.cu; baselines/L1/<problem_slug>/cuda_cpp/; baselines/L2/<problem_slug>/cuda_cpp/
Patents anchored here: None
Out of scope:
  - Python / Triton / CuTe DSL Python / cuTile Python / cudnn_frontend Python / torch.compile submissions (CLAUDE.md Rule 5).
  - NCU / `nv-nsight-cu-cli` / `.ncu-rep`/`.ncu-repz` artefacts on any device, including local 5060 (CLAUDE.md Rule 1).
  - `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` / persistence / power / MIG / compute-mode mutation (CLAUDE.md Rule 2).
  - Privileged CUPTI: Events, Metrics, Range Profiling, PM Sampling, PerfWorks (CLAUDE.md Rule 3; CUPTI Activity remains allowed).
  - Host-driver work: NVIDIA driver install, DKMS, MOK, `modprobe nvidia`, kernel-module parameter toggles, reboot workflows (CLAUDE.md Rule 4).
  - User CUDA streams, background host threads, `torch.jit.fork`, null-stream side effects, CUDA Graphs without a per-problem note (CLAUDE.md Rule 7).
  - Evaluator-state exploits: `data_ptr` caching, lazy-eval `__eq__` on `FakeTensor`, monkey-patched `Event.elapsed_time` / `do_bench` (CLAUDE.md Rule 6).
  - Precision-contract violations: precision downgrade (e.g. FP16 compute upcast to FP32 output) to pass tolerance (CLAUDE.md Rule 8).
Version family: cuda-13.1; cutlass-4.5.x; sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: rmsnorm; fused-add; fused-rms; norm; residual; reduction; bf16
Latency authority: paper
---

# Fused Add + RMSNorm (Canonical CUDA C++ Recipe)

## Use when

You're starting a SOL-ExecBench problem whose operation is `output = rmsnorm(hidden_states + residual, weight, eps)` -- the canonical decoder-layer pre/post-norm pattern. The base problem of this family is `problem_001_fused_add_rmsnorm_h2048` (and `_h4096`, `_h7168` variants).

## Canonical CUDA C++ recipe

Single-file CUDA C++ kernel -- no CUTLASS templates strictly required. Reference implementation lives at `sol-execbench/examples/cuda_cpp/rmsnorm/kernel.cu` and demonstrates:

1. **Vectorised loads.** `__nv_bfloat162` (2 BF16 per load) or `nv_bfloat16` 8-element packs for 128-bit transactions.
2. **Block-level reduction.** One CTA per row (batch element). Each thread accumulates `x * x` in FP32 (NOT BF16 -- precision-contract violation per CLAUDE.md Rule 8).
3. **Warp-shuffle finalisation.** `__shfl_xor_sync` reduction tree across the warp, followed by shared-memory reduction across warps in the CTA.
4. **Reciprocal-sqrt.** `rsqrtf(mean_sq + eps)` on the FP32 partial sum.
5. **Vectorised store.** `(x * inv_rms) * weight` computed in FP32, downcast to BF16 only on store.
6. **Launch bounds.** `__launch_bounds__(1024, 1)` for max-occupancy when `hidden_size` fits in shared memory.

Host-side launch skeleton:

```cpp
void run(torch::Tensor hidden_states, torch::Tensor residual,
         torch::Tensor weight, torch::Tensor output) {
    auto N = hidden_states.size(0);
    auto H = hidden_states.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_add_rmsnorm_kernel<<<N, /* threads */, /* smem */, stream>>>(
        hidden_states.data_ptr<__nv_bfloat16>(),
        residual.data_ptr<__nv_bfloat16>(),
        weight.data_ptr<__nv_bfloat16>(),
        output.data_ptr<__nv_bfloat16>(),
        H, /* eps */ 1e-6f);
}
```

Note the launch on `at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7 (default-stream discipline).

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (RTX 5060) and `sm_100a` (B200) differences for this pattern:

- **HBM3e vs GDDR7 bandwidth** (paper authority: B200 = 8 TB/s per `references/NVIDIA_Manuals/blackwell_tuning/`). Fused-add + RMSNorm is memory-bound for typical `hidden_size`. The 5060's bandwidth ceiling limits make local wall-time a weak proxy for B200 SOL Score; sm_120 timing primarily validates correctness and launch-config sanity.
- **DSMEM (sm_100a cluster-cooperative reduction).** For very large `hidden_size` (e.g. 7168, 14336) the B200 can stage the reduction across a CTA cluster with DSMEM peer-loads, eliminating shared-memory pressure on a single CTA. sm_120 doesn't support cluster sizes large enough to matter -- per-CTA reduction is the only path.
- **L2 carveout.** B200's L2 absorbs the `weight` tensor for an entire batch; sm_120's L2 doesn't. Repeated reads of `weight` across batch rows benefit strongly on B200; on sm_120 they may hit DRAM.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Accumulate in FP32.** Computing `x * x + ...` in BF16 violates the precision contract (CLAUDE.md Rule 8). The reference impl shows FP32 accumulators are required.
- **Don't use a user-created CUDA stream** (CLAUDE.md Rule 7); use `at::cuda::getCurrentCUDAStream()`.
- **Don't cache `data_ptr` between iterations** (CLAUDE.md Rule 6) -- the harness re-shuffles input pointers.

## Linked problems

This pattern is referenced by all `problem_NNN_fused_add_rmsnorm_h*` dossiers and underpins the multi-op decoder-layer L2 problems whose internal residual-norm sub-block matches this recipe.
